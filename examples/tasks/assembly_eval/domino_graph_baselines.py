# SPDX-License-Identifier: Apache-2.0
"""Agent 5's domino graph and three baselines on the domino physics station, with the
independent evaluator scoring every episode (needs the integration branch: registered K,
``roboskillhub.tasks.graphs`` and ``roboskillhub.tasks.baselines``).

    PYTHONPATH=src python examples/tasks/assembly_eval/domino_graph_baselines.py \
        --physics mujoco --out runs/graph_mj --mode both

``baselines``: fixed_script / online_replan / simulation_search via ``baselines.compare``,
same seeds as Agent 5's fake-world report (train 1000-1005, test 0-9, 3 slots).
``negatives``: the fixed-script graph with each station fault on the last slot, seeds 0-2.
Every episode is judged by ``domino_world.independent_evaluation`` (simulator truth only);
the place skill's own result is matched to the physical release it made (by wall-clock
release time) and disagreements are listed.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.assembly_eval.domino_station import STATION_FAULTS
from roboskillhub.tasks.assembly_eval.domino_world import physics_domino_world
from roboskillhub.tasks.baselines import StationEnv, compare, simulation_search, summarize

PLACE_NODES = ("place", "place_first")


class PhysicsEnv(StationEnv):
    """``baselines.StationEnv`` over a fresh physics station per call; its ``independent``
    hook keeps the evaluator's verdict, pairs it with each place step and saves the record."""

    def __init__(self, registry, *, physics: str, slots: int, out: Path, fault: str | None = None):
        self.physics, self.slots, self.out, self.fault = physics, slots, out, fault
        super().__init__(self.make_world, registry, independent=self.record)

    def make_world(self, seed: int) -> dict:
        return physics_domino_world(seed, physics=self.physics, slots=self.slots, fault=self.fault)

    def record(self, episode: dict, ind: dict, built: dict) -> dict:
        episode["independent"] = {
            "layout": ind["layout"],
            "placements": [_brief(p) for p in ind["placements"]],
            "sim_time_s": ind["sim_time_s"],
            "events": ind["events"],
        }
        episode["skill_vs_evaluator"] = pair_results(episode, ind)
        self._save(episode, built["station"])
        return episode["independent"]

    def _save(self, episode, station) -> None:
        d = (
            self.out
            / (
                self.fault
                or episode["graph_id"]
                + json.dumps(episode.get("graph_params", {}), sort_keys=True)
                .replace(" ", "")
                .replace('"', "")
                .replace(":", "=")
                .replace(",", "_")[1:-1]
            )
            / f"seed_{episode['seed']}"
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode.json").write_text(json.dumps(episode, default=str))
        with (d / "observations.jsonl").open("w") as f:
            for s in station.observer.samples:
                f.write(json.dumps(s) + "\n")
        episode["record_dir"] = str(d)


def _brief(p: dict) -> dict:
    keys = (
        "target_id",
        "slot_index",
        "status",
        "failure_types",
        "unverified_reasons",
        "cause_hypotheses",
        "neighbours_disturbed",
        "max_tilt_deg",
        "events",
        "release_index",
        "release_wall_s",
    )
    out = {k: p.get(k) for k in keys}
    if p.get("final"):
        out["final"] = {
            k: p["final"].get(k)
            for k in ("position_error_m", "yaw_error_deg", "tilt_deg", "bottom_height_m", "touches")
        }
    return out


def pair_results(episode: dict, ind: dict) -> list[dict]:
    """Each place step's own verdict next to the independent verdict of the release it made."""
    pairs = []
    used = set()
    for step in episode["steps"]:
        if step["node"] not in PLACE_NODES:
            continue
        r = step.get("result") or {}
        ev = (r.get("evidence") or {}).get("events") or {}
        rel = ev.get("release_s")
        match = None
        if rel is not None:
            cands = [
                p
                for p in ind["placements"]
                if p["release_index"] not in used
                and p.get("release_wall_s") is not None
                and abs(p["release_wall_s"] - rel) < 2.0
            ]
            if cands:
                match = min(cands, key=lambda p: abs(p["release_wall_s"] - rel))
                used.add(match["release_index"])
        pairs.append(
            {
                "node": step["node"],
                "skill_status": r.get("status", step["status"]),
                "skill_failure_type": r.get("failure_type"),
                "skill_phase": r.get("phase"),
                "released": match is not None,
                "evaluator_status": match["status"] if match else "no_release_observed",
                "evaluator_failure_types": match["failure_types"] if match else [],
                "evaluator_unverified": match["unverified_reasons"] if match else [],
                "agree": (match["status"] == r.get("status"))
                if match
                else (r.get("status") != "succeeded"),
            }
        )
    return pairs


def independent_summary(episodes: list[dict]) -> dict:
    slot_fail = collections.Counter()
    place_fail = collections.Counter()
    disagreements = []
    pairs = 0
    for e in episodes:
        for _k, v in e["independent"]["layout"].get("slots", {}).items():
            if v["status"] != "succeeded":
                slot_fail[v["status"]] += 1
        for p in e["independent"]["placements"]:
            for f in p["failure_types"] or []:
                place_fail[f] += 1
            for u in p["unverified_reasons"] or []:
                place_fail["unverified:" + u.split(":")[0]] += 1
        for q in e["skill_vs_evaluator"]:
            pairs += 1
            if not q["agree"]:
                disagreements.append({"seed": e["seed"], "graph": e["graph_id"], **q})
    n = len(episodes)
    return {
        "independent_task_success": f"{sum(e['task_success'] for e in episodes)}/{n}",
        "independent_placement_failure_types": dict(place_fail),
        "slots_not_standing": dict(slot_fail),
        "place_steps_compared": pairs,
        "disagreements": disagreements,
        "graph_status": dict(collections.Counter(e["status"] for e in episodes)),
        "graph_failure_types": dict(
            collections.Counter(
                e.get("failure_type") or "none" for e in episodes if e["status"] != "succeeded"
            )
        ),
        "interventions": sum(e["interventions"] for e in episodes),
        "mean_sim_s": sum(e["independent"]["sim_time_s"] for e in episodes) / n if n else 0,
        "mean_wall_s": sum(e["wall_s"] for e in episodes) / n if n else 0,
    }


def partial_compare(env, names, train_seeds, test_seeds, environment, slots) -> dict:
    """``baselines.compare`` restricted to some baselines (same graphs, candidates, seeds)."""
    rows = {}
    for name in names:
        if name in ("fixed_script", "online_replan"):
            g = graphs.domino_line(
                recovery=name == "online_replan",
                attempts=1 if name == "fixed_script" else 2,
                slots=slots,
            )
            eps = [env(g, s) for s in test_seeds]
            rows[name] = {
                "params": g["params"],
                **summarize(eps),
                "planning_s": 0.0,
                "search_rollouts": 0,
                "episodes_detail": eps,
            }
        elif name == "simulation_search":
            cands = graphs.domino_candidates(slots)
            search = simulation_search(cands, env, train_seeds)
            eps = [env(search["selected"], s) for s in test_seeds]
            rows[name] = {
                "params": search["selected_params"],
                **summarize(eps),
                "planning_s": search["planning_s"],
                "search_rollouts": search["search_rollouts"],
                "candidates": len(cands),
                "train_table": search["train_table"],
                "episodes_detail": eps,
            }
        else:
            raise SystemExit(f"unknown baseline {name}")
    return {
        "task": "domino",
        "environment": environment,
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "baselines": rows,
    }


def seeds(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", choices=("mujoco", "isaac"), default="mujoco")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("baselines", "negatives", "both"), default="both")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train", default="1000-1005")
    ap.add_argument("--test", default="0-9")
    ap.add_argument("--neg-seeds", default="0-2")
    ap.add_argument("--faults", default=",".join(STATION_FAULTS))
    ap.add_argument(
        "--only",
        help="comma list of fixed_script,online_replan,simulation_search (default: all three via "
        "baselines.compare); same graphs, seeds and search candidates, so partial runs can be merged",
    )
    args = ap.parse_args()
    registry, unavailable = load_registry()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"physics": args.physics, "unavailable_modules": unavailable}
    if args.mode in ("baselines", "both"):
        env = PhysicsEnv(
            registry, physics=args.physics, slots=args.slots, out=args.out / "baselines"
        )
        probe = physics_domino_world(0, physics=args.physics, slots=args.slots)["describe"]
        if args.only:
            res = partial_compare(
                env, args.only.split(","), seeds(args.train), seeds(args.test), probe, args.slots
            )
        else:
            res = compare(
                "domino",
                env,
                train_seeds=seeds(args.train),
                test_seeds=seeds(args.test),
                environment=probe,
                slots=args.slots,
            )
        for name, row in res["baselines"].items():
            eps = row.pop("episodes_detail")
            row["independent"] = independent_summary(eps)
            row["episodes"] = [
                {
                    "seed": e["seed"],
                    "status": e["status"],
                    "failure_type": e.get("failure_type"),
                    "task_success": e["task_success"],
                    "record": e.get("record_dir"),
                    "steps": len(e["steps"]),
                }
                for e in eps
            ]
            print(
                f"{name:18s} graph {row['graph_success']}  independent {row['independent']['independent_task_success']}"
                f"  disagreements {len(row['independent']['disagreements'])}  params {row['params']}",
                flush=True,
            )
        report["baselines"] = res
        (args.out / "baselines.json").write_text(json.dumps(report, indent=1, default=str))
    if args.mode in ("negatives", "both"):
        neg = {}
        for fault in args.faults.split(","):
            env = PhysicsEnv(
                registry,
                physics=args.physics,
                slots=args.slots,
                out=args.out / "negatives",
                fault=fault,
            )
            graph = graphs.domino_line(recovery=False, attempts=1, slots=args.slots)
            eps = [env(graph, s) for s in seeds(args.neg_seeds)]
            neg[fault] = {
                **summarize(eps),
                "independent": independent_summary(eps),
                "place_pairs": [q for e in eps for q in e["skill_vs_evaluator"]],
            }
            ind = neg[fault]["independent"]
            print(
                f"neg {fault:18s} graph {neg[fault]['graph_success']} {ind['graph_failure_types']}  "
                f"independent {ind['independent_task_success']} {ind['independent_placement_failure_types']}  "
                f"disagree {len(ind['disagreements'])}",
                flush=True,
            )
        report["negatives"] = neg
        (args.out / "negatives.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
