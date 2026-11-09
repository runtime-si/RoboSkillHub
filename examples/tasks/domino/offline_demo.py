"""Offline domino demo: no robot, no simulator, no station connection.

1. plan_straight_layout: five 48 x 24 x 7.5 mm tiles in a row running away from the arm
   base, 40 mm centre pitch; picks a build order that keeps the arm corridor clear.
2. plan_place (pure part of place_upright) for every slot in that order, with the tiles
   placed before it as neighbours: TCP preplace / release / retreat poses and the
   pre-motion clearance checks.
3. verify_stable_placement on a hand-written observation stream (synthetic_contract_test).

    PYTHONPATH=src python examples/tasks/domino/offline_demo.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from roboskillhub.tasks.domino import plan_place, plan_straight_layout, verify_stable_placement
from roboskillhub.tasks.domino.contract import RequestRejected

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    layout = plan_straight_layout(load("layout_request.json"))
    print(
        "layout:",
        layout["status"],
        layout["evidence"].get("order_name"),
        layout["evidence"].get("build_order"),
    )
    for name, c in layout["evidence"]["order_candidates"].items():
        print(f"  {name:9s} feasible={c['feasible']} {c['reason']}")

    base = load("place_request.json")
    now = time.time()
    slots = {s["slot_id"]: s for s in layout["evidence"]["layout_target"]["slots"]}
    placed: list[dict] = []
    for idx in layout["evidence"]["build_order"]:
        slot = slots[f"s{idx}"]
        req = {
            **base,
            "now_s": now,
            "target_id": f"d{idx}",
            "target_object_pose": slot["pose"],
            "held": {**base["held"], "object_id": f"d{idx}", "timestamp_s": now},
            "neighbors": [
                {
                    **base["neighbor_template"],
                    "object_id": p["id"],
                    "position_m": p["pose"]["position_m"],
                    "quaternion_xyzw": p["pose"]["quaternion_xyzw"],
                    "timestamp_s": now,
                }
                for p in placed
            ],
        }
        req.pop("neighbor_template")
        try:
            plan = plan_place(req)
            P = plan["poses"]
            print(
                f"slot s{idx}: preplace z={P['preplace']['position_m'][2]:.4f} release z={P['release']['position_m'][2]:.4f} "
                f"retreat z={P['retreat']['position_m'][2]:.4f}; {len(plan['descent_waypoints'])} descent waypoints; "
                f"requires {sorted(plan['requires'])}"
            )
        except RequestRejected as exc:
            print(f"slot s{idx}: rejected {exc.code}: {exc}")
        placed.append({"id": f"d{idx}", "pose": slot["pose"]})

    verdict = verify_stable_placement(load("verify_request_synthetic.json"))
    print(
        "verify (synthetic stream):",
        verdict["status"],
        verdict["failure_type"],
        verdict["evidence"]["postcondition"]["measured"],
    )


if __name__ == "__main__":
    main()
