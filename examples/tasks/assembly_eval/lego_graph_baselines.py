# SPDX-License-Identifier: Apache-2.0
"""Agent 5's LEGO graph through Agent 3's skills (and Agent 1's grasp tools) on the LEGO physics station
(Isaac Sim 5.1 + BrickSim, floating hand), with the independent evaluator scoring every episode.

Needs the integration tree (runtime, registered K, ``roboskillhub.tasks.graphs`` / ``baselines``) and the
Isaac venv:

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src python examples/tasks/assembly_eval/lego_graph_baselines.py \\
        --mode smoke --scene single_on_plate --grasps configured --out runs/lego_graph

Modes: ``smoke`` (one fixed-script episode per seed and scene), ``baselines`` (fixed_script / online_replan /
simulation_search via ``baselines.compare`` on one scene), ``negatives`` (fixed script with breakage on/off).

The runtime's SkillContext carries ``contact_description`` (contract rsh-task/1.4); the harness adds
nothing to the context.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roboskillhub.tasks.baselines import compare

from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.graph import load_graph, run_graph
from roboskillhub.runtime.task import primitive_skill_context
from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.assembly_eval.lego_world import physics_lego_world


def patch_build(g: dict) -> dict:
    g = json.loads(json.dumps(g))
    for node in ("align", "verify"):
        g["nodes"][node]["params"]["build"] = "$last.select.outputs.target.build"
    g["graph_id"] += "+build-patch-4a"
    return g


def _brief_result(step: dict) -> dict:
    r = step.get("result") or {}
    ev = r.get("evidence") or {}
    post = ev.get("postcondition") or {}
    return {
        "node": step.get("node"),
        "skill": step.get("skill"),
        "status": r.get("status"),
        "failure_type": r.get("failure_type"),
        "phase": r.get("phase"),
        "postcondition": {k: post.get(k) for k in ("name", "passed", "measured")},
        "verdict": (r.get("outputs") or {}).get("verdict"),
        "reason": ev.get("reason")
        or ev.get("reason_code")
        or (r.get("error") or {}).get("message"),
    }


class LegoPhysicsEnv:
    """``(graph, seed) -> episode`` on a fresh LEGO station per call (as ``baselines.StationEnv``), with the
    independent verdict and the skill-vs-evaluator pairing."""

    def __init__(
        self,
        registry,
        *,
        scene: str,
        out: Path,
        breakage: bool = False,
        grip_force_n: float = 2.0,
        force_limit_n: float = 6.0,
        fix_baseplate: bool = True,
        fault: str | None = None,
    ):
        self.registry, self.scene, self.out, self.breakage = registry, scene, out, breakage
        self.grip_force_n, self.force_limit_n = grip_force_n, force_limit_n
        self.fix_baseplate = fix_baseplate
        self.fault = fault
        self.episodes_run = 0

    def __call__(self, graph: dict, seed: int) -> dict:
        built = physics_lego_world(
            seed,
            scene=self.scene,
            breakage=self.breakage,
            grip_force_n=self.grip_force_n,
            force_limit_n=self.force_limit_n,
            fix_baseplate=self.fix_baseplate,
            fault=self.fault,
        )
        runtime = Runtime(built["backend"], built["limits"])
        loaded = load_graph(graph, self.registry)
        task_id = f"{graph['graph_id']}/{self.scene}/seed{seed}"
        started = time.monotonic()
        with runtime.task_lease(task_id, loaded["budgets"]["timeout_s"]) as lease:

            def make_context(node: str, request_id: str, timeout_s: float):
                # contact_description now comes from the runtime (contract 1.4); no shim.
                return primitive_skill_context(
                    runtime, lease, request_id=request_id, privileged_allowed=True
                )

            episode = run_graph(
                loaded, self.registry, built["inputs"], make_context, task_id=task_id
            )
        self.episodes_run += 1
        episode["seed"] = seed
        episode["scene"] = self.scene
        episode["environment"] = built["describe"]
        episode["graph_params"] = graph.get("params", {})
        episode["wall_s"] = time.monotonic() - started
        ind = built["independent"]()
        episode["interventions"] = built["interventions"]()
        episode["task_success"] = ind["outcome"]["status"] == "succeeded"
        episode["independent"] = {
            "outcome": ind["outcome"],
            "placements": ind["placements"],
            "assignment": ind["record"]["assignment"],
        }
        episode["skill_vs_evaluator"] = pair(episode, ind)
        self._save(graph, seed, episode, ind)
        return episode

    def _save(self, graph, seed, episode, ind) -> None:
        d = (
            self.out
            / self.scene
            / f"{graph['graph_id']}_{json.dumps(graph.get('params', {}), sort_keys=True)}".replace(
                " ", ""
            )
            .replace('"', "")
            .replace(":", "=")
            .replace(",", "_")
            .replace("{", "")
            .replace("}", "")
        )
        d = d / (
            (f"{self.fault}_" if self.fault else "")
            + f"seed_{seed}{'_breakage' if self.breakage else ''}_grip{self.grip_force_n:g}"
            f"_f{self.force_limit_n:g}{'_freeplate' if not self.fix_baseplate else ''}"
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode.json").write_text(json.dumps(episode, default=str))
        (d / "independent.json").write_text(json.dumps(ind, default=str))
        episode["record_dir"] = str(d)


def pair(episode: dict, ind: dict) -> list[dict]:
    """k-th release <-> k-th seat/release/verify node results; disagreement = the verify skill and the
    evaluator differ on success."""
    steps = episode.get("steps", [])
    by = {
        s: [_brief_result(x) for x in steps if x.get("skill") == s]
        for s in (
            "lego.align_connection",
            "lego.guarded_seat",
            "lego.release_and_retreat",
            "lego.verify_connection",
        )
    }
    out = []
    for k, p in enumerate(ind["placements"]):
        v = by["lego.verify_connection"][k] if k < len(by["lego.verify_connection"]) else None
        out.append(
            {
                "brick": p["part"],
                "evaluator": {"status": p["status"], "failure_type": p["failure_type"]},
                "align": by["lego.align_connection"][k]
                if k < len(by["lego.align_connection"])
                else None,
                "seat": by["lego.guarded_seat"][k] if k < len(by["lego.guarded_seat"]) else None,
                "verify": v,
                "disagree": v is not None
                and (v["status"] == "succeeded") != (p["status"] == "succeeded"),
            }
        )
    return out


def _summ(ep: dict) -> dict:
    return {
        "seed": ep["seed"],
        "scene": ep["scene"],
        "graph": ep.get("graph_id"),
        "graph_status": ep["status"],
        "graph_failure_type": ep.get("failure_type"),
        "independent": ep["independent"]["outcome"],
        "steps": [
            (
                s["node"],
                (s.get("result") or {}).get("status"),
                (s.get("result") or {}).get("failure_type"),
            )
            for s in ep.get("steps", [])
        ],
        "pairs": [
            {k: v for k, v in p.items() if k in ("brick", "evaluator", "disagree")}
            | {
                "verify": (p["verify"] or {}).get("status"),
                "seat": (p["seat"] or {}).get("failure_type") or (p["seat"] or {}).get("status"),
            }
            for p in ep["skill_vs_evaluator"]
        ],
        "wall_s": round(ep["wall_s"], 1),
        "record_dir": ep.get("record_dir"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "baselines", "negatives"], default="smoke")
    ap.add_argument("--scene", default="single_on_plate")
    ap.add_argument("--grasps", choices=["perception", "configured"], default="configured")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--train", type=int, nargs="+", default=[1000, 1001])
    ap.add_argument("--breakage", action="store_true")
    ap.add_argument(
        "--graph-patch",
        choices=["none", "build"],
        default="none",
        help="build: 4a-local patch giving lego.align_connection and lego.verify_connection the target's "
        "build list ($last.select.outputs.target.build); not Agent 5's graph",
    )
    ap.add_argument(
        "--grip-force", type=float, default=2.0, help="pad drive force limit (N) of the hand"
    )
    ap.add_argument(
        "--force-limit", type=float, default=6.0, help="press force_limit_N (station max 10)"
    )
    ap.add_argument(
        "--free-baseplate", action="store_true", help="plate free on the table (as RoCo)"
    )
    ap.add_argument(
        "--fault", default=None, help="lego_station.STATION_FAULTS, on the last placement"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    K, unavailable = load_registry()
    args.out.mkdir(parents=True, exist_ok=True)
    bricks = {"single_on_plate": 1, "single_on_brick": 1, "seq3_stack": 3, "seq3_spaced": 3}[
        args.scene
    ]
    env = LegoPhysicsEnv(
        K,
        scene=args.scene,
        out=args.out,
        breakage=args.breakage,
        grip_force_n=args.grip_force,
        force_limit_n=args.force_limit,
        fix_baseplate=not args.free_baseplate,
        fault=args.fault,
    )
    summary = args.out / "summary.jsonl"
    try:
        if args.mode == "smoke":
            g = graphs.lego_press(recovery=False, attempts=1, bricks=bricks, grasps=args.grasps)
            if args.graph_patch == "build":
                g = patch_build(g)
            for seed in args.seeds:
                ep = env(g, seed)
                ep["graph_id"] = g["graph_id"]
                with summary.open("a") as f:
                    f.write(json.dumps(_summ(ep), default=str) + "\n")
                print("EPISODE", json.dumps(_summ(ep), default=str), flush=True)
        elif args.mode == "baselines":
            graphs_lego = graphs.lego_press

            def lego_press(**kw):
                g = graphs_lego(**{"grasps": args.grasps, **kw})
                return patch_build(g) if args.graph_patch == "build" else g

            graphs.lego_press = lego_press
            report = compare(
                "lego",
                env,
                train_seeds=args.train,
                test_seeds=args.seeds,
                environment={
                    "scene": args.scene,
                    "station": "lego physics (Isaac+BrickSim)",
                    "grasps": args.grasps,
                },
                slots=bricks,
            )
            for name, row in report["baselines"].items():
                eps = row.pop("episodes_detail")
                row["episodes"] = [_summ(e) for e in eps]
                print(
                    "BASELINE",
                    name,
                    row["graph_success"],
                    row["independent_task_success"],
                    flush=True,
                )
            report["unavailable_modules"] = unavailable
            (args.out / f"baselines_{args.scene}.json").write_text(
                json.dumps(report, indent=1, default=str)
            )
    finally:
        os.sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
