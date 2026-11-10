"""Pick a flat piece by its top face with a suction cup: choose a sealing point on its mask from depth
(ARC baseline, suction_points.py), come down on it until contact, vacuum on, check the seal by lifting
a little and seeing the piece's rim rise, then carry it (optional) and release.

Needs a suction cup and a vacuum the connector switches: ``robot.set_suction(on)`` (a YAM station
has none today; see SKILL.md).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suction_points import best_point  # noqa: E402

logger = logging.getLogger(__name__)

CUP_DIAMETER_M = 0.008
"""The candidate hardware: an 8-10 mm bellows cup (jigsaw pieces are 2 mm board, ~20-25 mm wide)."""
MIN_AFFORDANCE = 0.5
APPROACH_M = 0.04
LIFT_M = 0.005
PRESS_M = 0.004
"""How far below the measured top the contact move is aimed (it stops at contact)."""
RIM_RISE_FRACTION = 0.5
"""The piece's rim (mask pixels away from the cup) must rise at least this share of the lift for the
seal to count when the connector reports no vacuum reading."""


class Output(TypedDict):
    route: str
    cup_point: list[float]
    affordance: float
    sealed: bool
    rim_rise_m: float
    reason: str


def _camera(obs: dict[str, Any], name: str) -> dict[str, Any] | None:
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    for c in cams:
        if str(c.get("name")) == name and c.get("depth") is not None:
            return c
    return None


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict):
        from scipy.spatial.transform import Rotation

        p, r = pose["position"], pose["rotation"]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
        T[:3, 3] = [p["x"], p["y"], p["z"]]
        return T
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached", "contact"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False and reply.get("outcome") != "contact":
        return str(reply.get("reason") or "the move reported failure")
    return ""


def _rim_pixels(
    mask: np.ndarray, depth: np.ndarray, pixel: list[int], away_px: float
) -> np.ndarray:
    h, w = mask.shape
    v, u = np.mgrid[0:h, 0:w]
    return (
        mask
        & ((u - pixel[0]) ** 2 + (v - pixel[1]) ** 2 > away_px**2)
        & np.isfinite(depth)
        & (depth > 0)
    )


def rim_rise(
    before: np.ndarray, after: np.ndarray, rim: np.ndarray, lift_m: float
) -> tuple[float, float]:
    """``(median rise of the piece's rim, share of rim pixels judged)``. Pixels whose depth fell by more
    than the lift plus 3 mm are something in front of the piece (the cup, the hand) and are skipped."""
    d = before[rim] - after[rim]
    seen = (d > -0.002) & (d < lift_m + 0.003) & (after[rim] > 0)
    if seen.sum() < 10:
        return 0.0, float(seen.mean()) if d.size else 0.0
    return float(np.median(d[seen])), float(seen.mean())


def run(
    ctx: NodeContext,
    target_mask: Any,
    camera: str = "top",
    observation: dict[str, Any] | None = None,
    cup_diameter_m: float = CUP_DIAMETER_M,
    cup_offset_m: list[float] | None = None,
    min_affordance: float = MIN_AFFORDANCE,
    approach_m: float = APPROACH_M,
    lift_m: float = LIFT_M,
    place_point: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    """``target_mask``: the piece's mask in ``camera``'s frame (e.g. sam3). ``cup_offset_m``: the cup lip
    centre in the tool frame (default the tool centre). ``place_point``: where the piece's top goes."""
    on_arm = {"arm_id": int(arm_id)} if int(arm_id) else {}
    out: Output = {
        "route": "no_valid_cup_point",
        "cup_point": [],
        "affordance": 0.0,
        "sealed": False,
        "rim_rise_m": 0.0,
        "reason": "",
    }
    obs = observation if observation is not None else ctx.tool("robot.get_observation")
    cam = _camera(obs, camera)
    if cam is None or cam.get("intrinsics") is None or cam.get("pose") is None:
        out["reason"] = f"no depth frame with intrinsics and pose from {camera!r}"
        return out
    mask = np.asarray(target_mask).astype(bool)
    if mask.shape != np.asarray(cam["depth"]).shape[:2]:
        out["reason"] = f"mask {mask.shape} does not match the depth frame"
        return out
    K = np.asarray(cam["intrinsics"], dtype=np.float64)
    pick = best_point(np.asarray(cam["depth"], dtype=np.float64), K, mask, cup_diameter_m / 2)
    if pick is None or pick["affordance"] < min_affordance:
        out["reason"] = (
            "no pixel a cup radius inside the mask"
            if pick is None
            else f"best affordance {pick['affordance']:.2f} < {min_affordance}"
        )
        if pick:
            out["affordance"] = round(pick["affordance"], 3)
        return out
    T_wc = _matrix(cam["pose"])
    p = T_wc[:3, :3] @ np.asarray(pick["point_camera"]) + T_wc[:3, 3]
    out["cup_point"] = [round(float(v), 5) for v in p]
    out["affordance"] = round(pick["affordance"], 3)

    tool = _matrix(ctx.tool("robot.get_ee_pose", **on_arm))
    rot = tool[:3, :3]
    offset = rot @ np.asarray(cup_offset_m or [0.0, 0.0, 0.0], dtype=np.float64)
    q = ctx.tool("robot.get_ee_pose", **on_arm)
    rotation = (q.get("pose", q) if isinstance(q, dict) else q)["rotation"]

    def pose(cup_at: np.ndarray) -> dict:
        t = cup_at - offset
        return {
            "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "rotation": rotation,
        }

    def move(tool_name: str, cup_at: np.ndarray) -> str:
        try:
            return _refusal(ctx.tool(tool_name, pose=pose(cup_at), **on_arm))
        except Exception as exc:  # routed, not raised
            return str(exc) or type(exc).__name__

    def suction(on: bool) -> dict:
        try:
            r = ctx.tool("robot.set_suction", on=bool(on), **on_arm)
            return r if isinstance(r, dict) else {}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    up = np.array([0.0, 0.0, 1.0])
    why = move("robot.go_to_pose", p + up * approach_m)
    if not why:
        why = move("robot.move_cartesian_until_contact", p - up * PRESS_M)
    if why:
        out["route"], out["reason"] = "unreachable", f"approach / descent: {why}"
        return out
    depth0 = np.asarray(cam["depth"], dtype=np.float64)
    rim = _rim_pixels(mask, depth0, pick["pixel"], pick["need_px"] * 2.0)
    reply = suction(True)
    if "error" in reply:
        out["route"], out["reason"] = "unreachable", f"robot.set_suction failed: {reply['error']}"
        return out
    here = _matrix(ctx.tool("robot.get_ee_pose", **on_arm))[:3, 3] + offset
    move("robot.go_to_pose_cartesian", here + up * lift_m)
    sealed = reply.get("sealed")
    after_cam = _camera(ctx.tool("robot.get_observation"), camera)
    rise, seen = (0.0, 0.0)
    if after_cam is not None:
        rise, seen = rim_rise(depth0, np.asarray(after_cam["depth"], dtype=np.float64), rim, lift_m)
    out["rim_rise_m"] = round(float(rise), 5)
    if sealed is None:
        sealed = rise >= RIM_RISE_FRACTION * lift_m
    out["sealed"] = bool(sealed)
    if not sealed:
        suction(False)
        move("robot.go_to_pose_cartesian", here + up * approach_m)
        out["route"], out["reason"] = (
            "no_seal",
            f"the piece's rim rose {rise * 1e3:.1f} mm of a {lift_m * 1e3:.0f} mm lift ({seen:.0%} of it visible)",
        )
        return out
    move("robot.go_to_pose_cartesian", here + up * approach_m)
    if place_point is None:
        out["route"], out["reason"] = "picked", f"sealed at affordance {pick['affordance']:.2f}"
        return out
    dst = np.asarray(place_point, dtype=np.float64)
    why = move("robot.go_to_pose", dst + up * approach_m)
    status = suction(True)  # re-assert; a connector with a vacuum sensor reports the seal here
    if status.get("sealed") is False:
        out["route"], out["reason"] = "dropped_in_transport", "the vacuum lost its seal on the way"
        suction(False)
        return out
    if not why:
        why = move("robot.move_cartesian_until_contact", dst - up * PRESS_M)
    suction(False)
    here = _matrix(ctx.tool("robot.get_ee_pose", **on_arm))[:3, 3] + offset
    move("robot.go_to_pose_cartesian", here + up * approach_m)
    out["route"] = "picked"
    out["reason"] = f"placed at {np.round(dst, 4).tolist()}" + (
        f" (place move: {why})" if why else ""
    )
    return out
