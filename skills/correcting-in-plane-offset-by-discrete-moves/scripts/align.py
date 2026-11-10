# Adapted from vision-guided-rack-insertion-ot2
# (https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2, commit 3bfe3fe87b64),
# reference_implementation/ot2_rack_insertion/cnn_model_misalignment_prediction.py (predict on the
# cropped wrist frame, step a fixed dx / dy / dtheta per decision, stop after three consecutive
# "No Move, No Move, No Rotate"), MIT License, Copyright (c) 2026 Hadi Beyzaee Juneghani.
# Changes: the classifier is the move-classifier tool (world-frame sign labels, so no per-loop
# direction convention); robot.* moves read back after each step; a step and decision cap and a drift
# bound end the loop with an exit instead of running until aligned; the final lowering and release are
# left to the next skill in the graph.
"""Nudge the hand in x, y and yaw by a learned classifier on the wrist view until it says aligned."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_offsets import crop_frame  # noqa: E402
from moves import Hand, wrap  # noqa: E402

logger = logging.getLogger(__name__)

STEP_M = 0.001
"""Upstream's translation step (reference_implementation: 1.0 mm)."""
STEP_YAW_DEG = 0.5
"""Yaw step. Upstream's 0.1 deg needs 50 decisions for a 5 deg error; 0.5 deg keeps a 5 deg error
under 10 decisions, at the price of ending up to a quarter step outside the 0.2 deg label dead-band."""
STREAK = 3
"""Upstream: three consecutive aligned predictions end the loop."""
MAX_DECISIONS = 80
MAX_DRIFT_M = 0.025
MAX_DRIFT_DEG = 10.0


class Output(TypedDict):
    route: str
    decisions: int
    moved_m: list[float]
    turned_deg: float
    last_label: str
    reason: str


def run(
    ctx: NodeContext,
    model_path: str,
    camera: str,
    crop: list[int] | None = None,
    step_m: float = STEP_M,
    step_yaw_deg: float = STEP_YAW_DEG,
    streak: int = STREAK,
    max_decisions: int = MAX_DECISIONS,
    max_drift_m: float = MAX_DRIFT_M,
    max_drift_deg: float = MAX_DRIFT_DEG,
    min_confidence: float = 0.0,
    arm_id: int = 0,
) -> Output:
    hand = Hand(ctx, arm_id)
    start, start_yaw, rotation = hand.state()
    z = start[2]
    run_ok = 0
    label = ""
    for k in range(1, int(max_decisions) + 1):
        img = crop_frame(ctx.tool("robot.get_observation"), camera, crop)
        if img is None:
            return _out("no_frame", k - 1, hand, start, start_yaw, label, f"no RGB from camera {camera!r}")
        try:
            pred = ctx.tool("move-classifier.predict", image=img, model_path=model_path)
        except Exception as exc:  # noqa: BLE001
            return _out("no_frame", k - 1, hand, start, start_yaw, label, f"move-classifier.predict failed: {exc}")
        label = str(pred.get("label", ""))
        if pred.get("aligned") and float(pred.get("confidence", 1.0)) >= min_confidence:
            run_ok += 1
            if run_ok >= streak:
                return _out("aligned", k, hand, start, start_yaw, label, f"{streak} aligned predictions in a row")
            continue
        run_ok = 0
        pos, yaw, rot_now = hand.state()
        sx = {"+": 1, "-": -1}.get(label[0:1], 0)
        sy = {"+": 1, "-": -1}.get(label[1:2], 0)
        st = {"+": 1, "-": -1}.get(label[2:3], 0)
        why = ""
        if sx or sy:
            why = hand.move_xy(pos[0] + sx * step_m, pos[1] + sy * step_m, z, rot_now)
        if not why and st:
            why = hand.turn(st * step_yaw_deg)
        if why:
            return _out("diverging", k, hand, start, start_yaw, label, f"step refused: {why}")
        pos, yaw, _ = hand.state()
        drift = float(np.linalg.norm(pos[:2] - start[:2]))
        if drift > max_drift_m or abs(wrap(yaw - start_yaw)) > max_drift_deg:
            return _out("diverging", k, hand, start, start_yaw, label,
                        f"moved {drift * 1e3:.1f} mm / {wrap(yaw - start_yaw):.1f} deg from the start")
    return _out("max_steps", int(max_decisions), hand, start, start_yaw, label, f"no {streak} aligned in a row")


def _out(route: str, n: int, hand: Hand, start: np.ndarray, start_yaw: float, label: str, reason: str) -> Output:
    pos, yaw, _ = hand.state()
    d = pos - start
    logger.info("align: %s after %d decisions (%s)", route, n, reason)
    return {"route": route, "decisions": n, "moved_m": [round(float(v), 5) for v in d[:2]],
            "turned_deg": round(float(wrap(yaw - start_yaw)), 3), "last_label": label, "reason": reason}

