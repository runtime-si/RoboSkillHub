# SPDX-License-Identifier: Apache-2.0
"""Agent 5's domino graph, its three baselines and Agent 4b's station negatives on the YAM arm
station, scored by the same independent evaluator, next to the floating-gripper station.

Needs the integration branch (registered K, ``roboskillhub.tasks.graphs`` / ``baselines``)
plus 4b's ``domino_graph_baselines.py`` beside this file, whose episode loop, skill/evaluator
pairing and summaries are reused unchanged; only the world factory differs.

    PYTHONPATH=src python examples/tasks/assembly_eval/yam_graph_baselines.py \
        --embodiment yam --out runs/yam --mode both

Refusal ablation (every proposed grasp tilted off top-down; the floating gripper refuses
it before motion, so pick ends ``unreachable``)::

    PYTHONPATH=src python examples/tasks/assembly_eval/yam_graph_baselines.py \
        --embodiment floating --out runs/tilt20 --mode baselines \
        --only fixed_script,online_replan --tilt-grasp-deg 20
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path

from roboskillhub.tasks import graphs, load_registry
from roboskillhub.tasks.assembly_eval.domino_geometry import quat_mul
from roboskillhub.tasks.assembly_eval.domino_station import STATION_FAULTS
from roboskillhub.tasks.assembly_eval.domino_world import physics_domino_world
from roboskillhub.tasks.assembly_eval.yam_world import yam_domino_world

_spec = importlib.util.spec_from_file_location(
    "domino_graph_baselines", Path(__file__).with_name("domino_graph_baselines.py")
)
dgb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dgb)


def with_pick_candidate(graph: dict, index: int) -> dict:
    """Diagnostic G': ``pick`` takes ``candidates.<index>`` of ``perception.propose_grasps``
    instead of ``candidates.0``. Same K; labelled in graph_id and params."""
    if index == 0:
        return graph
    g = json.loads(json.dumps(graph))
    g["nodes"]["pick"]["params"]["candidate"] = f"$last.candidate.outputs.candidates.{index}"
    g["graph_id"] += f"-cand{index}"
    g["params"] = {**g.get("params", {}), "pick_candidate": index}
    return g


def make_world(embodiment: str, seed: int, slots: int, fault: str | None) -> dict:
    if embodiment == "yam":
        return yam_domino_world(seed, slots=slots, fault=fault)
    return physics_domino_world(seed, physics="mujoco", slots=slots, fault=fault)


def tilted_grasps(registry: dict, degrees: float) -> dict:
    """Diagnostic K' for the refusal ablation: ``perception.propose_grasps`` with every
    candidate's pregrasp/grasp/lift TCP pose rotated ``degrees`` about its own x axis
    (closing axis kept horizontal, approach no longer straight down). Same ids and versions;
    the run is labelled in the report. ``degrees == 0`` returns ``registry`` unchanged."""
    if not degrees:
        return registry
    inner = registry["perception.propose_grasps"]
    h = math.radians(degrees) / 2
    rx = (math.sin(h), 0.0, 0.0, math.cos(h))

    def tilt(pose: dict) -> dict:
        return {**pose, "quaternion_xyzw": list(quat_mul(pose["quaternion_xyzw"], rx))}

    class Tilted:
        skill_id, version, kind = inner.skill_id, inner.version, inner.kind
        requires, readiness, input_schema = inner.requires, inner.readiness, inner.input_schema

        def run(self, request, ctx):
            result = inner.run(request, ctx)
            for c in (result.get("outputs") or {}).get("candidates") or []:
                for key in ("pregrasp_pose", "grasp_pose", "lift_pose"):
                    if isinstance(c.get(key), dict):
                        c[key] = tilt(c[key])
            return result

    return {**registry, inner.skill_id: Tilted()}


class EmbodimentEnv(dgb.PhysicsEnv):
    """4b's ``PhysicsEnv`` with the world chosen by embodiment; also keeps the station's
    refusals (unreachable targets) and the node/failure where each graph stopped."""

    def __init__(
        self, registry, *, embodiment: str, slots: int, out: Path, fault=None, pick_candidate=0
    ):
        self.embodiment = embodiment
        self.pick_candidate = pick_candidate
        super().__init__(registry, physics="mujoco", slots=slots, out=out, fault=fault)

    def make_world(self, seed: int) -> dict:
        return make_world(self.embodiment, seed, self.slots, self.fault)

    def __call__(self, graph: dict, seed: int) -> dict:
        return super().__call__(with_pick_candidate(graph, self.pick_candidate), seed)

    def record(self, episode: dict, ind: dict, built: dict) -> dict:
        station = built["station"]
        episode["embodiment"] = self.embodiment
        episode["station_refusals"] = getattr(station, "refusals", [])
        episode["station_motions"] = getattr(station, "motions", None)
        episode["stopped_at"] = _stopped_at(episode)
        episode["picked_candidates"] = _picked(episode, self.pick_candidate)
        return super().record(episode, ind, built)


def _stopped_at(episode: dict) -> dict | None:
    """The last non-succeeded step (node, status, failure type, message)."""
    for step in reversed(episode["steps"]):
        r = step.get("result") or {}
        st = r.get("status", step.get("status"))
        if st != "succeeded":
            return {
                "node": step["node"],
                "status": st,
                "failure_type": r.get("failure_type"),
                "phase": r.get("phase"),
                "message": ((r.get("error") or {}).get("message") or "")[:300],
            }
    return None


def _picked(episode: dict, index: int) -> list[dict]:
    """For every pick: the candidate used, its jaw yaw and score, and candidate 0's."""
    out, cands = [], None
    for step in episode["steps"]:
        r = step.get("result") or {}
        if step["node"] == "candidate":
            cands = (r.get("outputs") or {}).get("candidates")
        if step["node"] == "pick" and cands:

            def brief(c):
                q = c["grasp_pose"]["quaternion_xyzw"]
                x, y, z, w = q
                yaw = math.degrees(math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z)))
                return {
                    "score": c.get("score"),
                    "jaw_flip": (c.get("details") or {}).get("jaw_flip"),
                    "yaw_deg": round(yaw, 1),
                }

            out.append(
                {
                    "object_id": cands[0].get("object_id"),
                    "used": index,
                    "used_candidate": brief(cands[index]) if index < len(cands) else None,
                    "candidate0": brief(cands[0]),
                    "status": r.get("status"),
                }
            )
    return out


def episode_row(e: dict) -> dict:
    return {
        "seed": e["seed"],
        "status": e["status"],
        "failure_type": e.get("failure_type"),
        "task_success": e["task_success"],
        "stopped_at": e.get("stopped_at"),
        "refusals": [r["reason"] for r in e.get("station_refusals", [])],
        "motion": _motion_summary(e.get("station_motions")),
        "picked_candidates": e.get("picked_candidates"),
        "steps": len(e["steps"]),
        "sim_s": e["independent"]["sim_time_s"],
        "wall_s": e["wall_s"],
        "record": e.get("record_dir"),
    }


def _motion_summary(motions: list[dict] | None) -> dict | None:
    if not motions:
        return None
    done = [m for m in motions if m["completed"]]
    return {
        "moves": len(motions),
        "max_tracking_error_m": max(m["max_tracking_error_m"] for m in motions),
        "max_settle_s": max((m["settle_s"] or 0.0) for m in done) if done else None,
        "max_ack_error_m": max(m["ack_error_m"] for m in done) if done else None,
        "max_ack_error_rad": max(m["ack_error_rad"] for m in done) if done else None,
    }


def seeds(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", choices=("yam", "floating"), default="yam")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("baselines", "negatives", "both"), default="both")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train", default="1000-1005")
    ap.add_argument("--test", default="0-9")
    ap.add_argument("--neg-seeds", default="0-2")
    ap.add_argument("--faults", default=",".join(STATION_FAULTS))
    ap.add_argument("--only", default="fixed_script,online_replan,simulation_search")
    ap.add_argument(
        "--pick-candidate",
        type=int,
        default=0,
        help="diagnostic G': pick takes perception.propose_grasps candidates.<N> (default 0 = Agent 5's G)",
    )
    ap.add_argument(
        "--tilt-grasp-deg",
        type=float,
        default=0.0,
        help="refusal ablation (diagnostic K'): every proposed grasp pose tilted this many degrees "
        "about its closing axis, so no pick pose is top-down (default 0 = Agent 1's proposals)",
    )
    args = ap.parse_args()
    registry, unavailable = load_registry()
    registry = tilted_grasps(registry, args.tilt_grasp_deg)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "embodiment": args.embodiment,
        "pick_candidate": args.pick_candidate,
        "graph": "Agent 5's G"
        if args.pick_candidate == 0
        else f"diagnostic G' (pick uses candidates.{args.pick_candidate})",
        "tilt_grasp_deg": args.tilt_grasp_deg,
        "K": "Agent 5's K"
        if not args.tilt_grasp_deg
        else f"diagnostic K' (propose_grasps poses tilted {args.tilt_grasp_deg} deg off top-down)",
        "unavailable_modules": unavailable,
    }
    if args.mode in ("baselines", "both"):
        env = EmbodimentEnv(
            registry,
            embodiment=args.embodiment,
            slots=args.slots,
            out=args.out / "baselines",
            pick_candidate=args.pick_candidate,
        )
        probe = make_world(args.embodiment, 0, args.slots, None)["describe"]
        res = dgb.partial_compare(
            env, args.only.split(","), seeds(args.train), seeds(args.test), probe, args.slots
        )
        for name, row in res["baselines"].items():
            eps = row.pop("episodes_detail")
            row["independent"] = dgb.independent_summary(eps)
            row["episodes"] = [episode_row(e) for e in eps]
            row["stopped_at"] = dict(
                collections.Counter(
                    f"{e['stopped_at']['node']}:{e['stopped_at']['failure_type']}"
                    for e in eps
                    if e.get("stopped_at")
                )
            )
            print(
                f"{name:18s} graph {row['graph_success']}  independent {row['independent']['independent_task_success']}"
                f"  disagreements {len(row['independent']['disagreements'])}  stopped {row['stopped_at']}  params {row['params']}",
                flush=True,
            )
        report["baselines"] = res
        (args.out / "baselines.json").write_text(json.dumps(report, indent=1, default=str))
    if args.mode in ("negatives", "both"):
        neg = {}
        for fault in args.faults.split(","):
            env = EmbodimentEnv(
                registry,
                embodiment=args.embodiment,
                slots=args.slots,
                out=args.out / "negatives",
                fault=fault,
                pick_candidate=args.pick_candidate,
            )
            graph = graphs.domino_line(recovery=False, attempts=1, slots=args.slots)
            eps = [env(graph, s) for s in seeds(args.neg_seeds)]
            neg[fault] = {
                **dgb.summarize(eps),
                "independent": dgb.independent_summary(eps),
                "place_pairs": [q for e in eps for q in e["skill_vs_evaluator"]],
                "episodes": [episode_row(e) for e in eps],
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
