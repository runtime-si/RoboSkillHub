# Adapted from RoboTwin (https://github.com/robotwin-Platform/RoboTwin),
# envs/scan_object.py, MIT, Copyright (c) 2025 Tianxing Chen.
# Kept: the holding arm presents the object at a fixed inspection point, the
# other arm brings its sensor to a stand-off along the object's functional
# axis, and the framing test of check_success: the object's centre projected on
# the sensor's viewing ray lies 0 < d < 7 cm ahead of it and within 2.5 cm of
# the ray on every axis.
# Changed: the scanner (a held tool with a functional point) is the other arm's
# wrist camera, its pose from the arm's own reading; simulator actor poses are
# replaced by the held object's world box and the holding hand's reading; one
# arm moves at a time with an explicit arm_id; a minimum tool-centre distance
# between the arms is checked before the camera arm closes in; refused moves are
# routed, not raised.
"""Present a held object to the other arm's wrist camera along a viewing axis."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

STANDOFF = 0.05
"""Camera ahead of the object's centre along the viewing axis [m] (scan_object: pre_dis / dis 0.05)."""

MAX_RANGE = 0.07
"""The object's centre at most this far ahead of the camera [m] (check_success: dis < 0.07)."""

LATERAL_TOLERANCE = 0.025
"""Largest offset of the object's centre from the viewing ray on any axis [m] (check_success: eps 0.025)."""

MIN_TCP_DISTANCE = 0.04
"""Planned camera-arm tool centre closer than this to the holding hand's: arms_too_close [m]
(a YAM fingertip pair is ~20 mm wide either side of its tool centre)."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m] (handing-over-between-arms)."""

EMPTY_FRACTION = 0.04
"""Holding hand's jaw opening at or under this (robot.get_gripper, 1.0 = open): it holds nothing."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""


class Output(TypedDict):
    route: str
    detail: str
    range_m: float
    lateral_m: float
    camera_pose: dict[str, Any]
    object_position: dict[str, float]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rotation(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (r[j, i] + r[i, j]) / s
        v[k] = (r[k, i] + r[i, k]) / s
        q = ((r[k, j] - r[j, k]) / s, *v)
    return {"w": q[0], "x": q[1], "y": q[2], "z": q[3]}


def looking_along(view: np.ndarray) -> dict[str, float]:
    """A tool rotation whose +z (the wrist camera's optical axis) points along ``view``, x kept level."""
    z = view / np.linalg.norm(view)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:  # looking straight up or down: keep x along world x
        x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x = x / np.linalg.norm(x)
    return _quat(np.column_stack([x, np.cross(z, x), z]))


def framing(
    camera: np.ndarray, optical_axis: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """scan_object.check_success: (distance of the target ahead of the camera along its optical axis,
    largest per-axis offset of the target's projection on the ray from the camera)."""
    a = optical_axis / np.linalg.norm(optical_axis)
    d = float(a @ (target - camera))
    return d, float(np.max(np.abs(camera + d * a - target)))


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any],
    inspection_position: dict[str, float],
    view_axis: dict[str, float],
    holder_arm: int = 0,
    camera_arm: int = 1,
    standoff: float = STANDOFF,
    camera_offset: dict[str, float] | None = None,
    max_range: float = MAX_RANGE,
    lateral_tolerance: float = LATERAL_TOLERANCE,
    min_tcp_distance: float = MIN_TCP_DISTANCE,
    reach_tolerance: float = REACH_TOLERANCE,
) -> Output:
    """``view_axis``: unit world direction from the object toward where the camera should be (the face to
    see points that way). ``camera_offset``: the camera's optical centre in the tool frame [m] (default the
    tool centre); its optical axis is the tool's +z."""
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "range_m": -1.0,
        "lateral_m": -1.0,
        "camera_pose": {},
        "object_position": {},
    }

    def arm(a: int) -> dict[str, int]:
        return {"arm_id": int(a)} if int(a) else {}

    def here(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(a: int, position: np.ndarray, rotation: dict[str, float]) -> str:
        try:
            reply = ctx.tool(
                "robot.go_to_pose",
                pose={"position": _vec(position), "rotation": rotation},
                **arm(a),
            )
        except Exception as exc:  # a refused move is routed, not raised
            return str(exc)
        if isinstance(reply, dict) and reply.get("outcome") not in (None, "reached"):
            return f"{reply['outcome']}: {reply.get('reason', '')}"
        miss = float(np.linalg.norm(_xyz(here(a)) - position))
        return f"ended {miss * 1e3:.0f} mm from the target" if miss > reach_tolerance else ""

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    if int(holder_arm) == int(camera_arm):
        return finish("unreachable", "holder_arm and camera_arm are the same arm")
    holder, camera = here(holder_arm), here(camera_arm)
    if float(np.linalg.norm(_xyz(holder) - _xyz(camera))) < SAME_ARM:
        return finish(
            "unreachable",
            f"arm {camera_arm} reads the same pose as arm {holder_arm}: no second arm",
        )
    grip = float(ctx.tool("robot.get_gripper", **arm(holder_arm))["position"])
    if grip <= EMPTY_FRACTION:
        return finish(
            "target_not_in_view", f"holding hand closed to the stop ({grip:.3f}): it holds nothing"
        )

    c = object_obb["center"]
    obj0 = np.array([float(c["x"]), float(c["y"]), float(c["z"])])
    offset = obj0 - _xyz(holder)  # held rigidly: the object keeps this offset from the hand
    target = np.array([float(inspection_position[k]) for k in ("x", "y", "z")])
    v = np.array([float(view_axis[k]) for k in ("x", "y", "z")])
    if np.linalg.norm(v) < 1e-6:
        return finish("unreachable", "view_axis is zero")
    v = v / np.linalg.norm(v)
    holder_tcp = target - offset
    why = move(holder_arm, holder_tcp, holder["rotation"])
    if why:
        return finish("unreachable", f"holding arm to the inspection point: {why}")
    holder_tcp = _xyz(here(holder_arm))
    obj = holder_tcp + offset
    out["object_position"] = _vec(obj)

    rot = looking_along(-v)
    r = _rotation(rot)
    off = (
        np.zeros(3) if camera_offset is None else np.array([float(camera_offset[k]) for k in "xyz"])
    )
    cam_tcp = obj + standoff * v - r @ off
    gap = float(np.linalg.norm(cam_tcp - holder_tcp))
    if gap < min_tcp_distance:
        return finish(
            "arms_too_close",
            f"camera arm's tool centre would be {gap * 1e3:.0f} mm from the holding hand's "
            f"(min {min_tcp_distance * 1e3:.0f} mm): use a longer standoff or another view_axis",
        )
    why = move(camera_arm, cam_tcp, rot)
    if why:
        return finish("unreachable", f"camera arm to its stand-off: {why}")
    cam = here(camera_arm)
    rc = _rotation(cam["rotation"])
    cam_xyz = _xyz(cam) + rc @ off
    out["camera_pose"] = {"position": _vec(cam_xyz), "rotation": cam["rotation"]}
    d, lat = framing(cam_xyz, rc[:, 2], obj)
    out["range_m"], out["lateral_m"] = d, lat
    if 0.0 < d < max_range and lat < lateral_tolerance:
        return finish(
            "inspected",
            f"object {d * 1e3:.0f} mm ahead of the camera, {lat * 1e3:.1f} mm off its ray",
        )
    return finish(
        "target_not_in_view",
        f"object {d * 1e3:.0f} mm along the camera's axis (0-{max_range * 1e3:.0f}), "
        f"{lat * 1e3:.1f} mm off its ray (max {lateral_tolerance * 1e3:.0f})",
    )
