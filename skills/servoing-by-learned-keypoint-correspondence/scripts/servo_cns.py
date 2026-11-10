# Adapted from CNS (https://github.com/hhcaz/CNS, commit 6b42138ea8f15d872eeca916b9e1fb1f429bcac4),
# demo_real_ur5.py (set the goal frame, then per frame: camera velocity from the pipeline, stop on the
# pixel stop policy, no data, or a time limit), MIT License, Copyright (c) 2023 hhcaz.
# Changes: the robot has no velocity interface here, so each velocity is integrated over dt into one
# capped robot.go_to_pose step of the hand (camera frame -> world through the camera's pose in the
# observation); in "plane" mode only x, y, z and the turn about world +z are applied (the turn through
# the last joint, for arms whose IK leaves the wrist roll free); lost correspondences end the loop with an
# exit after `lost_limit` frames instead of stopping at once.
"""Servo the wrist camera back to a recorded goal view with CNS (learned keypoint-graph visual servo)."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moves import Hand, rot, wrap  # noqa: E402

logger = logging.getLogger(__name__)

DT_S = 0.5
"""Seconds each commanded velocity is applied for (CNS's real demo sent velocities at ~10 Hz to a UR5;
one capped position step per frame here). Tuned in the twin: SKILL.md, Measured."""
MAX_STEP_M = 0.004
MAX_STEP_DEG = 1.5
MAX_FRAMES = 120
LOST_LIMIT = 3
MIN_MATCHES = 8


class Output(TypedDict):
    route: str
    frames: int
    final_error: float
    moved_m: list[float]
    turned_deg: float
    reason: str


def _camera(obs: dict[str, Any], name: str) -> dict[str, Any] | None:
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    for c in cams:
        if str(c.get("name")) == name and c.get("rgb") is not None:
            return c
    return None


def _rgb(cam: dict[str, Any]) -> np.ndarray:
    img = np.asarray(cam["rgb"])[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _cam_rotation(cam: dict[str, Any]) -> np.ndarray | None:
    pose = cam.get("pose")
    if pose is None:
        return None
    if isinstance(pose, dict):
        return rot(pose["rotation"])
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, :3]


def _distance(cam: dict[str, Any]) -> float | None:
    depth = cam.get("depth")
    if depth is None:
        return None
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape[:2]
    c = d[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    c = c[np.isfinite(c) & (c > 0.02)]
    return float(np.median(c)) if c.size else None


def run(
    ctx: NodeContext,
    goal: dict[str, Any],
    camera: str,
    distance_m: float | None = None,
    mode: str = "plane",
    dt_s: float = DT_S,
    max_step_m: float = MAX_STEP_M,
    max_step_deg: float = MAX_STEP_DEG,
    max_frames: int = MAX_FRAMES,
    lost_limit: int = LOST_LIMIT,
    min_matches: int = MIN_MATCHES,
    detector: str = "SIFT",
    arm_id: int = 0,
) -> Output:
    """``goal``: an observation taken at the pose to come back to (its ``camera`` frame is the goal view).
    ``distance_m``: camera to the scene there (default the goal frame's median centre depth)."""
    if mode not in ("plane", "full"):
        raise ValueError(f"mode must be plane or full, got {mode!r}")
    gcam = _camera(goal, camera)
    hand = Hand(ctx, arm_id)
    start, start_yaw, _ = hand.state()
    if gcam is None or gcam.get("intrinsics") is None:
        return _out("lost_correspondence", 0, float("inf"), hand, start, start_yaw, f"goal has no frame with intrinsics from {camera!r}")
    dist = float(distance_m) if distance_m else _distance(gcam)
    if not dist:
        return _out("lost_correspondence", 0, float("inf"), hand, start, start_yaw, "no distance_m and no goal depth")
    s = ctx.tool("cns.start", goal_image=_rgb(gcam), intrinsics=np.asarray(gcam["intrinsics"]).tolist(),
                 distance_m=dist, detector=detector)
    sid = s["session_id"]
    lost = 0
    err = float("inf")
    try:
        for k in range(1, int(max_frames) + 1):
            cam = _camera(ctx.tool("robot.get_observation"), camera)
            if cam is None:
                return _out("lost_correspondence", k - 1, err, hand, start, start_yaw, f"no frame from {camera!r}")
            r = ctx.tool("cns.step", session_id=sid, image=_rgb(cam))
            err = float(r.get("error", float("inf")))
            if int(r.get("matches", 0)) < min_matches:
                lost += 1
                if lost >= lost_limit:
                    return _out("lost_correspondence", k, err, hand, start, start_yaw, f"{lost} frames under {min_matches} matches")
                continue
            lost = 0
            if r.get("converged"):
                return _out("converged", k, err, hand, start, start_yaw, f"CNS stop policy (error {err:.4f})")
            R_wc = _cam_rotation(cam)
            if R_wc is None:
                return _out("lost_correspondence", k, err, hand, start, start_yaw, "the camera has no pose in the observation")
            v = np.asarray(r["velocity"], dtype=np.float64)
            dp = R_wc @ v[:3] * dt_s
            w = R_wc @ v[3:] * dt_s
            n = np.linalg.norm(dp)
            if n > max_step_m:
                dp *= max_step_m / n
            pos, yaw, rq = hand.state()
            if mode == "plane":
                dyaw = float(np.clip(math.degrees(w[2]), -max_step_deg, max_step_deg))
                why = hand.move_xy(pos[0] + dp[0], pos[1] + dp[1], pos[2] + dp[2], rq)
                if not why and abs(dyaw) > 0.05:
                    why = hand.turn(dyaw)
            else:
                from scipy.spatial.transform import Rotation

                ang = np.linalg.norm(w)
                if ang > math.radians(max_step_deg):
                    w *= math.radians(max_step_deg) / ang
                new_r = Rotation.from_rotvec(w) * Rotation.from_quat([rq[c] for c in "xyzw"])
                q = new_r.as_quat()
                pose = {"position": {"x": pos[0] + dp[0], "y": pos[1] + dp[1], "z": pos[2] + dp[2]},
                        "rotation": dict(zip("xyzw", map(float, q), strict=True))}
                from moves import refusal

                try:
                    why = refusal(ctx.tool("robot.go_to_pose", pose=pose, **hand.arm))
                except Exception as exc:
                    why = str(exc)
            if why:
                return _out("timeout", k, err, hand, start, start_yaw, f"step refused: {why}")
        return _out("timeout", int(max_frames), err, hand, start, start_yaw, f"not converged in {max_frames} frames")
    finally:
        try:
            ctx.tool("cns.stop", session_id=sid)
        except Exception:  # noqa: BLE001
            pass


def _out(route, n, err, hand, start, start_yaw, reason) -> Output:
    pos, yaw, _ = hand.state()
    logger.info("servo_cns: %s after %d frames (%s)", route, n, reason)
    return {"route": route, "frames": n, "final_error": round(float(err), 5) if np.isfinite(err) else -1.0,
            "moved_m": [round(float(v), 5) for v in (pos - start)], "turned_deg": round(float(wrap(yaw - start_yaw)), 3),
            "reason": reason}
