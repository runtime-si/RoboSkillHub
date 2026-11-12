"""Insert the tip of an object one arm holds into a hole, the other arm's wrist camera giving two side views.

The holding arm hovers the tip just over the estimated hole mouth. The other arm takes two side views of tip
and hole from ``standoff`` away, along two horizontal directions 90 deg apart on its own side. Each view gives
the tip's offset from the hole along that view's image-horizontal axis, and the two views give both lateral
axes. The holding arm corrects (bounded), the camera arm goes back to where it started, and the holding arm
descends. A hand that stops short of the commanded depth has its tip on the face.

Feasibility on a YAM bimanual station (the pens task): the station refuses tool centres closer than 0.15 m.
A funnel of half-open fingers over the hole, or a hand on the holder, puts the assisting tool centre within
~0.1 m of the inserting hand's. A camera 0.2 m off the hole does not, so this skill is the camera assist only.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

STANDOFF = 0.20
"""Camera from the hole mouth [m]: keeps the tool centres over MIN_SEPARATION apart; inside a D405's range."""

ELEVATION_DEG = 25.0
"""Camera above the mouth's level, seen from the mouth [deg]: low enough to see the gap under the tip."""

MIN_SEPARATION = 0.15
"""Tool centres closer than this are refused [m] (the YAM station's MIN_SEPARATION)."""

HOVER_GAP = 0.008
"""Tip over the mouth while the views are taken [m]."""

MAX_CORRECTION = 0.015
"""Largest total lateral correction from the views [m]: past this the hole estimate is wrong, not off."""

TOLERANCE = 0.0008
"""Offset under which no further view round is taken [m]."""

ROUNDS = 2
"""View rounds (each: two views and one correction)."""

INSERT_DEPTH = 0.020
"""Tip below the mouth at the end of the descent [m]."""

FACE_STOP = 0.004
"""The hand ended this far over the commanded depth: the tip is on the face [m]."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent before the move counts as refused [m]."""


class Output(TypedDict):
    route: str
    detail: str
    measured_offset_m: float
    correction_m: float
    min_tcp_separation_m: float
    shortfall_m: float


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


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


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pixel_on_plane(
    cam: dict[str, Any], u: float, v: float, point: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """World point where the ray through pixel (u, v) of ``cam`` (``intrinsics`` 3x3, ``pose`` world camera
    pose, optical z forward) meets the plane through ``point`` with ``normal``."""
    k = np.asarray(cam["intrinsics"], float).reshape(3, 3)
    pose = cam["pose"]
    r = _rot(pose.get("rotation") or pose.get("orientation"))
    o = _xyz(pose)
    ray = r @ np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0])
    den = float(ray @ normal)
    if abs(den) < 1e-9:
        return None
    s = float((point - o) @ normal) / den
    return o + s * ray if s > 0 else None


def offset_by_pointing(
    ctx: NodeContext,
    camera_name: str,
    tip_query: str,
    hole_query: str,
    mouth: np.ndarray,
    view: np.ndarray,
    horizontal: np.ndarray,
) -> dict[str, Any]:
    """The per-view reading from the camera arm's frame: gemini-er.point at the tip and at the hole's mouth,
    both back-projected onto the vertical plane through the mouth across the line of sight."""
    cams = ctx.tool("robot.get_observation").get("cameras") or []
    if isinstance(cams, dict):
        cams = [dict(c, name=n) for n, c in cams.items()]
    cam = next((c for c in cams if c.get("name") == camera_name), None)
    if (
        cam is None
        or cam.get("rgb") is None
        or cam.get("intrinsics") is None
        or cam.get("pose") is None
    ):
        return {"visible": False, "reason": f"no {camera_name} frame with rgb, intrinsics and pose"}
    pts = []
    for q in (tip_query, hole_query):
        r = ctx.tool(
            "gemini-er.point", image=np.asarray(cam["rgb"], dtype=np.uint8), query=q, max_points=1
        )
        if not r.get("found"):
            return {"visible": False, "reason": f"gemini-er found no {q!r}"}
        p = pixel_on_plane(cam, float(r["pixel_x"]), float(r["pixel_y"]), mouth, view)
        if p is None:
            return {"visible": False, "reason": f"{q!r} ray misses the plane through the mouth"}
        pts.append(p)
    return {"visible": True, "lateral_m": float(horizontal @ (pts[0] - pts[1]))}


def looking_along(view: np.ndarray) -> dict[str, float]:
    """Tool rotation whose +z (the wrist camera's optical axis) points along ``view``, x kept level
    (inspecting-a-held-object-with-the-other-wrist-camera)."""
    z = view / np.linalg.norm(view)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x = x / np.linalg.norm(x)
    return _quat(np.column_stack([x, np.cross(z, x), z]))


def view_directions(mouth: np.ndarray, camera_tcp: np.ndarray) -> list[np.ndarray]:
    """Two horizontal unit directions from the mouth toward the camera arm's side, 90 deg apart."""
    d = (camera_tcp - mouth)[:2]
    a = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0
    return [np.array([math.cos(a + s), math.sin(a + s), 0.0]) for s in (math.pi / 4, -math.pi / 4)]


def solve_offset(horizontals: list[np.ndarray], readings: list[float]) -> np.ndarray:
    """The tip's xy offset from the hole from its components along two views' image-horizontal axes."""
    a = np.array([[h[0], h[1]] for h in horizontals])
    e = np.linalg.solve(a, np.array(readings))
    return np.array([e[0], e[1], 0.0])


def run(
    ctx: NodeContext,
    hole_mouth: dict[str, float],
    tip_offset_m: float,
    holder_arm: int = 1,
    camera_arm: int = 0,
    assist: bool = True,
    offset_tool: str = "vision.tip_hole_offset",
    offset_source: str = "tool",
    camera_name: str = "left_wrist",
    tip_query: str = "the bottom tip of the pen held in the gripper",
    hole_query: str = "the centre of the opening of the square hole under the pen",
    standoff: float = STANDOFF,
    elevation_deg: float = ELEVATION_DEG,
    min_separation: float = MIN_SEPARATION,
    hover_gap: float = HOVER_GAP,
    max_correction: float = MAX_CORRECTION,
    tolerance: float = TOLERANCE,
    rounds: int = ROUNDS,
    insert_depth: float = INSERT_DEPTH,
    face_stop: float = FACE_STOP,
) -> Output:
    """``hole_mouth``: the hole's mouth centre (world, estimate). ``tip_offset_m``: tip below the holding hand's
    tool centre along its approach (tool z down). ``offset_tool`` (per view, called with ``arm_id`` of the
    camera arm and ``image_horizontal``, a world unit vector): ``{"visible": bool, "lateral_m": float}``, the
    tip's offset from the hole's axis along that axis, read in the camera's image. ``offset_source="point"``
    reads it instead from ``camera_name``'s frame in robot.get_observation with gemini-er.point."""
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "measured_offset_m": -1.0,
        "correction_m": 0.0,
        "min_tcp_separation_m": -1.0,
        "shortfall_m": 0.0,
    }

    def arm(a: int) -> dict[str, int]:
        return {"arm_id": int(a)} if int(a) else {}

    def here(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(a: int, position, rotation, tool: str = "robot.go_to_pose") -> str:
        try:
            reply = ctx.tool(
                tool, pose={"position": _vec(position), "rotation": rotation}, **arm(a)
            )
        except Exception as exc:  # a refused move is routed, not raised
            return str(exc)
        if isinstance(reply, dict) and reply.get("outcome") not in (None, "reached", "timeout"):
            return f"{reply['outcome']}: {reply.get('reason', '')}"
        return ""  # a servo that times out near the target is judged by the caller's reading

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    seps: list[float] = []
    mouth = np.array([float(hole_mouth[k]) for k in "xyz"])
    holder = here(holder_arm)
    rot_h = holder["rotation"]
    target = mouth + np.array([0.0, 0.0, tip_offset_m + hover_gap])
    why = move(holder_arm, target, rot_h, "robot.go_to_pose_cartesian")
    if why or np.linalg.norm(_xyz(here(holder_arm)) - target) > REACH_TOLERANCE:
        return finish("unreachable", f"holding arm over the hole: {why or 'ended off target'}")

    if assist:
        if int(holder_arm) == int(camera_arm):
            return finish("unreachable", "holder_arm and camera_arm are the same arm")
        cam0 = here(camera_arm)
        if np.linalg.norm(_xyz(cam0) - _xyz(here(holder_arm))) < 1e-3:
            return finish(
                "unreachable",
                f"arm {camera_arm} reads the same pose as arm {holder_arm}: no second arm",
            )
        el = math.radians(elevation_deg)
        dirs = view_directions(mouth, _xyz(cam0))
        aim = mouth + np.array([0.0, 0.0, hover_gap / 2])
        total = np.zeros(3)
        for _ in range(int(rounds)):
            hs, readings = [], []
            for d in dirs:
                cam = aim + standoff * (math.cos(el) * d + np.array([0.0, 0.0, math.sin(el)]))
                sep = float(np.linalg.norm(cam - _xyz(here(holder_arm))))
                seps.append(sep)
                if sep < min_separation:
                    move(camera_arm, _xyz(cam0), cam0["rotation"])
                    out["min_tcp_separation_m"] = min(seps)
                    return finish(
                        "arms_too_close",
                        f"camera pose {sep * 1e3:.0f} mm from the holding hand (min {min_separation * 1e3:.0f})",
                    )
                why = move(camera_arm, cam, looking_along(aim - cam))
                if why or np.linalg.norm(_xyz(here(camera_arm)) - cam) > REACH_TOLERANCE:
                    move(camera_arm, _xyz(cam0), cam0["rotation"])
                    out["min_tcp_separation_m"] = min(seps)
                    return finish(
                        "unreachable", f"camera arm to its view: {why or 'ended off target'}"
                    )
                h = np.cross(
                    [0.0, 0.0, 1.0], -d
                )  # image-horizontal: level, across the line of sight
                if offset_source == "point":
                    r = offset_by_pointing(ctx, camera_name, tip_query, hole_query, mouth, d, h)
                else:
                    r = ctx.tool(offset_tool, arm_id=int(camera_arm), image_horizontal=_vec(h))
                if not r.get("visible", False):
                    move(camera_arm, _xyz(cam0), cam0["rotation"])
                    out["min_tcp_separation_m"] = min(seps)
                    return finish(
                        "not_in_view",
                        "tip or hole not seen in a side view: " + str(r.get("reason", "")),
                    )
                hs.append(h)
                readings.append(float(r["lateral_m"]))
            e = solve_offset(hs, readings)
            if out["measured_offset_m"] < 0:
                out["measured_offset_m"] = float(np.linalg.norm(e))
            if np.linalg.norm(e) < tolerance:
                break
            if np.linalg.norm(total - e) > max_correction:
                move(camera_arm, _xyz(cam0), cam0["rotation"])
                out["min_tcp_separation_m"] = min(seps)
                return finish(
                    "not_in_view",
                    f"views put the tip {np.linalg.norm(e) * 1e3:.1f} mm off; total past {max_correction * 1e3:.0f} mm",
                )
            total -= e
            out["correction_m"] = float(np.linalg.norm(total))
            target = target - e
            why = move(holder_arm, target, rot_h, "robot.go_to_pose_cartesian")
            if why:
                return finish("unreachable", f"holding arm's correction: {why}")
        move(camera_arm, _xyz(cam0), cam0["rotation"])
        out["min_tcp_separation_m"] = min(seps) if seps else -1.0

    down = target - np.array([0.0, 0.0, hover_gap + insert_depth])
    move(holder_arm, down, rot_h, "robot.go_to_pose_cartesian")
    short = float(_xyz(here(holder_arm))[2] - down[2])
    out["shortfall_m"] = short
    if short > face_stop:
        return finish("tip_on_face", f"hand stopped {short * 1e3:.1f} mm over the commanded depth")
    return finish(
        "inserted",
        f"descended to {insert_depth * 1e3:.0f} mm under the mouth (short {short * 1e3:.1f} mm)",
    )
