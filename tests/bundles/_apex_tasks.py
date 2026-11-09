"""APEX-MR's LEGO assembly tasks as assigning-bricks-to-two-arms problems.

Read from tests/bundles/data/apex_mr/ (APEX-MR config/lego_tasks at commit 46a9448, MIT): the
assembly task files (cell x, y, layer z, brick type, orientation, support cell) and the brick
library (type -> studs). The plate and the two GP4 bases are APEX-MR's
(plate_calibration.json, robot_properties/r1_base.txt, r2_base.txt); each arm reaches 0.55 m
(a GP4's reach), and every brick type has a reusable station in front of each arm.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "apex_mr"
P, BH = 0.008, 0.0096
PLATE = (0.40892671825921423, 0.04573602346210516, 0.1899)
"""APEX-MR's assemble_plate calibration, taken as the 48 x 48 plate's centre: the two bases
stand 0.88 m apart on either side of it."""
HALF = 24
ARMS = [
    {"arm_id": 0, "base": [0.0, 0.0, 0.33], "reach_m": 0.55},
    {"arm_id": 1, "base": [0.88101137, -0.01304334, 0.33], "reach_m": 0.55},
]


def load(name: str) -> tuple[list[dict], list[dict]]:
    lib = json.loads((DATA / "lego_library.json").read_text())
    raw = json.loads((DATA / f"{name}.json").read_text())
    tasks, types = [], []
    for key in sorted(raw, key=int):
        t = raw[key]
        b = lib[str(t["brick_id"])]
        h, w = int(b["height"]), int(b["width"])
        sx, sy = (h, w) if t["ori"] == 0 else (w, h)
        task = {
            "id": key,
            "studs": [sx, sy],
            "position": [
                PLATE[0] + (t["x"] - HALF + sx / 2) * P,
                PLATE[1] + (t["y"] - HALF + sy / 2) * P,
                PLATE[2] + (t["z"] - 1) * BH,
            ],
            "yaw_deg": 0.0,
            "needs_support": t.get("support_x", -1) != -1,
        }
        if task["needs_support"]:
            task["support_position"] = [
                PLATE[0] + (t["support_x"] - HALF + 0.5) * P,
                PLATE[1] + (t["support_y"] - HALF + 0.5) * P,
                PLATE[2] + (t["support_z"] - 1) * BH,
            ]
        tasks.append(task)
        if tuple(sorted((sx, sy))) not in types:
            types.append(tuple(sorted((sx, sy))))
    stock = []
    for k, (a, b) in enumerate(types):
        for side, x in (("a", 0.25), ("b", 0.63)):
            stock.append(
                {
                    "id": f"station_{a}x{b}_{side}",
                    "studs": [a, b],
                    "position": [x, -0.25 + 0.05 * k, PLATE[2]],
                    "reusable": True,
                }
            )
    return tasks, stock
