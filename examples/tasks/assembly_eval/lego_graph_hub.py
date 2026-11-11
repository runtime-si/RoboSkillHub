# SPDX-License-Identifier: Apache-2.0
"""The LEGO graph with the hub's verifiers in the loop (``roboskillhub.tasks.lego_hub.lego_press_hub``) on the
LEGO physics station (Isaac Sim 5.1 + BrickSim), each placement's hub verdicts paired with the independent
evaluator's.

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src python examples/tasks/assembly_eval/lego_graph_hub.py \\
        --scene seq3_stack --seeds 0 1 2 3 4 --out runs/lego_hub [--fault perception_offset] [--breakage]

Writes ``summary.jsonl`` (one line per episode) and ``pairs.jsonl`` (one line per placement: evaluator status
and failure type; lego.verify_connection, hub.pull_up_check, hub.verify_seat_depth and hub.track_build
statuses and routes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lego_graph_baselines as base  # noqa: E402

from roboskillhub.tasks import load_registry  # noqa: E402
from roboskillhub.tasks.lego_hub import lego_press_hub  # noqa: E402

HUB_NODES = {
    "lego.verify_connection": "verify",
    "hub.pull_up_check": "pullup",
    "hub.verify_seat_depth": "depth",
    "hub.track_build": "track",
    "hub.judge_place": "judge",
}


def _hub_brief(step: dict) -> dict:
    r = step.get("result") or {}
    ev = r.get("evidence") or {}
    return {
        "status": r.get("status"),
        "failure_type": r.get("failure_type"),
        "route": ev.get("hub_route") or (r.get("outputs") or {}).get("verdict"),
        "reason": (ev.get("reason") or "")[:240],
        "brick_rise_m": ev.get("brick_rise_m"),
        "truth_face": ev.get("truth_face"),
        "moved_mm": ev.get("moved_mm"),
        "hand_rise_m": ev.get("hand_rise_m"),
    }


def pair(episode: dict, ind: dict) -> list[dict]:
    steps = episode.get("steps", [])
    by = {
        node: [_hub_brief(s) for s in steps if s.get("skill") == skill and s.get("node") == node]
        for skill, node in HUB_NODES.items()
    }
    faces = [_hub_brief(s) for s in steps if s.get("node") == "face"]
    out = []
    for k, p in enumerate(ind["placements"]):
        row = {
            "brick": p["part"],
            "evaluator": {"status": p["status"], "failure_type": p["failure_type"]},
            "face": faces[k] if k < len(faces) else None,
        }
        ok = p["status"] == "succeeded"
        for node in HUB_NODES.values():
            v = by[node][k] if k < len(by[node]) else None
            row[node] = v
            row[f"{node}_agree"] = (
                None
                if v is None or v["status"] == "unverified"
                else ((v["status"] == "succeeded") == ok)
            )
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="single_on_plate")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--grasps", choices=["perception", "configured"], default="configured")
    ap.add_argument("--mode", choices=["report", "strict"], default="report")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-pullup", action="store_true")
    ap.add_argument(
        "--flip", action="store_true", help="tip wrong-face bricks studs-up before the pick"
    )
    ap.add_argument(
        "--through-seat", action="store_true", help="release after a failed seat too (diagnostic)"
    )
    ap.add_argument("--breakage", action="store_true")
    ap.add_argument("--grip-force", type=float, default=2.0)
    ap.add_argument("--force-limit", type=float, default=6.0)
    ap.add_argument(
        "--fault", default=None, help="lego_station.STATION_FAULTS, on the last placement"
    )
    ap.add_argument(
        "--judge", action="store_true", help="hub.judge_place (#162) on every placement"
    )
    ap.add_argument(
        "--recovery", action="store_true", help="online_replan (lego_press recovery=True)"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    K, unavailable = load_registry()
    args.out.mkdir(parents=True, exist_ok=True)
    from roboskillhub.tasks.assembly_eval.lego_scenes import SCENES

    sc = SCENES[args.scene]
    bricks = len(sc["goal"]) - len(sc.get("start", {}))
    base.pair = pair
    env = base.LegoPhysicsEnv(
        K,
        scene=args.scene,
        out=args.out,
        breakage=args.breakage,
        grip_force_n=args.grip_force,
        force_limit_n=args.force_limit,
        fault=args.fault,
    )
    g = lego_press_hub(
        mode=args.mode,
        face=not args.no_face,
        pull_up=not args.no_pullup,
        flip=args.flip,
        judge=args.judge,
        through_seat=args.through_seat,
        recovery=args.recovery,
        attempts=2 if args.recovery else 1,
        bricks=bricks,
        grasps=args.grasps,
    )
    g = base.patch_build(g)
    try:
        for seed in args.seeds:
            try:
                ep = env(g, seed)
            except Exception as exc:  # one broken episode must not lose the others
                print("EPISODE_ERROR", seed, repr(exc), flush=True)
                continue
            summ = {
                "seed": seed,
                "scene": args.scene,
                "fault": args.fault,
                "breakage": args.breakage,
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
                "wall_s": round(ep["wall_s"], 1),
                "face_trace": [
                    (
                        s["node"],
                        *(lambda b: (b["status"], b["route"], b["truth_face"]))(_hub_brief(s)),
                    )
                    for s in ep.get("steps", [])
                    if s.get("node") in ("face", "flip")
                ],
            }
            with (args.out / "summary.jsonl").open("a") as f:
                f.write(json.dumps(summ, default=str) + "\n")
            with (args.out / "pairs.jsonl").open("a") as f:
                for row in ep["skill_vs_evaluator"]:
                    f.write(json.dumps({"seed": seed, "scene": args.scene, "fault": args.fault,
                                        "breakage": args.breakage, **row}, default=str) + "\n")  # fmt: skip
            print("EPISODE", json.dumps(summ, default=str), flush=True)
            for row in ep["skill_vs_evaluator"]:
                print("PAIR", json.dumps(row, default=str), flush=True)
        if unavailable:
            print("UNAVAILABLE", unavailable, flush=True)
    finally:
        os.sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
