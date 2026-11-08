# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/find_object_via_wrist/robotwin/policy.py, Apache-2.0.
# Changes: the coarse estimate is an input (a perception skill's OBB, or a point)
# instead of a head-camera pointing step inside the skill; the camera is aimed,
# not the tool (the wrist camera's extrinsic is read from the observation, and
# the hand is placed so the camera's optical axis passes through the target at
# the asked range, looking straight down when the arm reaches it); candidate
# views are previewed with robot.solve_ik and judged from a fresh pose and a
# fresh frame; the pointer is a hub tool (gemini-er.point or molmo.point_prompt)
# and of several points the one nearest the coarse estimate's projection wins;
# depth is a windowed median inside the D405's range; a refined point further
# from the coarse one than the object's own size is refused as a wrong target;
# routed exits refined / not_seen / unreachable instead of ok=False.
"""Look at a coarse target with the wrist camera from close up and re-point it to a refined world point."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

CAMERA_RANGE = 0.20
"""Distance from the wrist camera to the target along its optical axis [m]:
RoboRSI's ``hover_height_m`` default (0.20 in its code)."""

FALLBACK_RANGES = (0.20, 0.15, 0.25, 0.30)
"""Ranges tried after the asked one when a view is out of reach [m]. RoboRSI steps
down (0.20, 0.15, 0.12, 0.10) for its 18 cm finger offset; a D405 reads nothing
under ~7 cm and its depth is best inside ~0.5 m, so this stays 0.15-0.30 m."""

MIN_DEPTH = 0.07
"""Nearest depth a D405 reports [m] (Intel's minimum Z at 640x480, ~7 cm)."""

MAX_DEPTH = 0.50
"""Farthest depth trusted [m]: the top of the D405's ideal 7-50 cm range."""

MIN_TOOL_CLEARANCE = 0.05
"""Lowest tool centre over the target's top [m], so open fingertips (~10 mm below
the tool centre on a linear_4310) and the gripper body stay clear of it."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the view pose [m]. Above a real YAM station's
12 mm per-move bound and a servo's 4 mm ball; the frame is back-projected with
the camera pose the arm actually reached, so the view need not be exact."""

VIEW_MARGIN = 0.08
"""The coarse target must project at least this fraction of the image size inside
its border after the move, or the view counts as missed."""

DEPTH_WINDOW = 3
"""Half-width of the depth window around the pointed pixel [px]; its median of
valid readings is the depth."""

MIN_DEPTH_SAMPLES = 3
"""Valid depth readings the window needs before its median is trusted."""

DRIFT_MARGIN = 0.03
"""Slack on top of the target's half-diagonal before a refined point counts as a
different object [m]: an overhead estimate is off by 1-2 cm on a YAM station."""

MAX_DRIFT = 0.05
"""Allowed refined-to-coarse distance in the plane when only a point is given [m]."""

SETTLE_READS = 5
"""Most extra observations taken while waiting for the hand to stop after a move."""

SETTLE_DISTANCE = 0.0005
SETTLE_ANGLE_DEG = 0.1
"""Two consecutive tool readings closer than this count as settled [m, deg]: at a
0.2 m range on a D405 (fx ~394 px) 0.5 mm and 0.1 deg are each about 1 px."""

MAX_MOVES = 3
"""Most view moves tried before the skill gives up as unreachable."""

DEFAULT_LOCATION = "the centre of the top face"
DEFAULT_POINTER = "gemini-er.point"


class Output(TypedDict):
    route: str
    refined_point: dict[str, float]
    refined_obb: dict[str, Any]
    coarse_point: dict[str, float]
    drift: float
    pixel: dict[str, float]
    depth: float
    camera_range: float
    view_pose: dict[str, Any]
    camera_name: str
    detail: str


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = _xyz(pose)
    return m


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1.0e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1.0e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _top_point(obb: dict[str, Any]) -> np.ndarray:
    """The centre of the box's top, at the highest z any corner reaches."""
    rot = _rot(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    return np.array([centre[0], centre[1], centre[2] + float(np.abs(rot[2]) @ half)])


def _wrist_camera(obs: dict[str, Any], tool: np.ndarray, name: str | None) -> dict[str, Any] | None:
    """The named camera, else the eye-in-hand / wrist camera riding nearest the tool."""
    cams = [c for c in obs.get("cameras") or [] if c.get("pose") is not None]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    return min(wrist, key=lambda c: float(np.linalg.norm(_xyz(c["pose"]) - tool[:3, 3])))


def _project(cam: dict[str, Any], point: np.ndarray) -> tuple[float, float, float]:
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    pc = np.linalg.inv(_matrix(cam["pose"])) @ np.append(point, 1.0)
    z = float(pc[2])
    if z <= 1.0e-6:
        return -1.0, -1.0, z
    return float(k[0, 0] * pc[0] / z + k[0, 2]), float(k[1, 1] * pc[1] / z + k[1, 2]), z


def _settled(ctx: NodeContext, arm_id: int) -> dict[str, Any]:
    """An observation taken once two in a row agree on where the tool is: a frame
    and the arm reading it is posed with are read one after the other, so a hand
    still settling after its move would pose the frame wrong."""
    obs = ctx.tool("robot.get_observation")
    for _ in range(SETTLE_READS):
        nxt = ctx.tool("robot.get_observation")
        a, b = _matrix(obs["arms"][arm_id]["ee_pose"]), _matrix(nxt["arms"][arm_id]["ee_pose"])
        obs = nxt
        if (
            float(np.linalg.norm(a[:3, 3] - b[:3, 3])) <= SETTLE_DISTANCE
            and _angle_deg(a[:3, :3], b[:3, :3]) <= SETTLE_ANGLE_DEG
        ):
            break
    else:
        logger.info(
            "refining-target-from-wrist-closeup: the hand was still moving after %d reads",
            SETTLE_READS,
        )
    return obs


def _depth_at(depth: np.ndarray, u: float, v: float) -> tuple[float, int]:
    """Median of the valid readings in a window round (u, v), and how many there were."""
    h, w = depth.shape[:2]
    col, row = int(np.floor(u)), int(np.floor(v))
    window = depth[
        max(0, row - DEPTH_WINDOW) : min(h, row + DEPTH_WINDOW + 1),
        max(0, col - DEPTH_WINDOW) : min(w, col + DEPTH_WINDOW + 1),
    ].astype(np.float64)
    valid = window[np.isfinite(window) & (window >= MIN_DEPTH) & (window <= MAX_DEPTH)]
    return (float(np.median(valid)) if valid.size else 0.0), int(valid.size)


def run(
    ctx: NodeContext,
    object_name: str,
    target_obb: dict[str, Any] | None = None,
    target_point: dict[str, float] | None = None,
    location: str = DEFAULT_LOCATION,
    query: str | None = None,
    pointer: str = DEFAULT_POINTER,
    camera_range: float = CAMERA_RANGE,
    camera: str | None = None,
    rotation: dict[str, float] | None = None,
    max_drift: float | None = None,
    check_reach: bool = True,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
) -> Output:
    if target_obb is not None:
        coarse = _top_point(target_obb)
        allowed = (
            float(np.hypot(target_obb["extent"]["x"], target_obb["extent"]["y"])) + DRIFT_MARGIN
        )
    elif target_point is not None:
        coarse = np.array([float(target_point[k]) for k in ("x", "y", "z")])
        allowed = MAX_DRIFT
    else:
        raise ValueError("refining-target-from-wrist-closeup needs target_obb or target_point")
    if max_drift is not None:
        allowed = float(max_drift)
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    move: dict[str, Any] = dict(arm)
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    out: Output = {
        "route": "unreachable",
        "refined_point": _vec(coarse),
        "refined_obb": dict(target_obb) if target_obb is not None else {},
        "coarse_point": _vec(coarse),
        "drift": 0.0,
        "pixel": {"u": -1.0, "v": -1.0},
        "depth": 0.0,
        "camera_range": 0.0,
        "view_pose": {},
        "camera_name": "",
        "detail": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        logger.info("refining-target-from-wrist-closeup: %s (%s)", route, detail)
        return out

    obs = ctx.tool("robot.get_observation")
    tool_now = _matrix(obs["arms"][int(arm_id)]["ee_pose"])
    cam = _wrist_camera(obs, tool_now, camera)
    if cam is None:
        names = [c.get("name") for c in obs.get("cameras") or []]
        return finish(
            "not_seen", f"no wrist camera with a pose in the observation (cameras: {names})"
        )
    out["camera_name"] = str(cam.get("name", ""))
    # The extrinsic, from the frame the connector posed with the same tool reading.
    tool_cam = np.linalg.inv(tool_now) @ _matrix(cam["pose"])
    axis_tool, offset_tool = tool_cam[:3, 2], tool_cam[:3, 3]

    # Candidate hand rotations: each one also turned so the camera looks straight
    # down (the best depth and the least parallax), tried before it as given.
    down = np.array([0.0, 0.0, -1.0])
    bases = [_rot(rotation)] if rotation is not None else [tool_now[:3, :3]]
    if rotation is None:
        try:
            frame = ctx.tool("robot.grasp_frame", **arm)
            bases += [
                _rot(frame[k])
                for k in ("rotation", "rotation_flipped")
                if isinstance(frame, dict) and frame.get(k)
            ]
        except Exception as exc:  # optional: the hand's own rotation still serves
            logger.info("refining-target-from-wrist-closeup: no grasp frame (%s)", exc)
    rotations: list[np.ndarray] = []
    for base in bases:
        for r in (_align(base @ axis_tool, down) @ base, base) if rotation is None else (base,):
            if all(_angle_deg(r, seen) > 2.0 for seen in rotations):
                rotations.append(r)
    ranges = list(dict.fromkeys([float(camera_range), *FALLBACK_RANGES]))

    failures: list[str] = []
    moves = 0
    risen = False
    for r_tool in rotations:
        for rng in ranges:
            if moves >= MAX_MOVES:
                break
            eye = coarse - rng * (r_tool @ axis_tool)
            position = eye - r_tool @ offset_tool
            if position[2] < coarse[2] + MIN_TOOL_CLEARANCE:
                failures.append(
                    f"range {rng:.2f} m: the hand would come within {MIN_TOOL_CLEARANCE * 1e3:.0f} mm of the target"
                )
                continue
            pose = {"position": _vec(position), "rotation": _quat(r_tool)}
            if check_reach:
                try:
                    reply = ctx.tool("robot.solve_ik", pose=pose, **arm)
                except Exception as exc:
                    failures.append(f"range {rng:.2f} m: IK preview refused ({exc})")
                    continue
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    failures.append(f"range {rng:.2f} m: out of reach (IK preview)")
                    continue
            moves += 1
            why = ""
            if not risen:
                start = _pose(ctx.tool("robot.get_ee_pose", **arm))
                if float(start["position"]["z"]) < position[2] - 1.0e-3:
                    # Rise in place first, so the move across never sweeps through the target.
                    rise = {
                        "position": {**start["position"], "z": float(position[2])},
                        "rotation": start["rotation"],
                    }
                    try:
                        why = _refusal(ctx.tool("robot.go_to_pose", pose=rise, **move))
                    except Exception as exc:  # a refused move is routed, not raised
                        why = str(exc)
                    why = f"rise: {why}" if why else ""
                risen = not why
            if not why:
                try:
                    why = _refusal(ctx.tool("robot.go_to_pose", pose=pose, **move))
                except Exception as exc:
                    why = str(exc)
            if why:
                failures.append(f"range {rng:.2f} m: {why}")
                continue
            off = float(
                np.linalg.norm(_xyz(_pose(ctx.tool("robot.get_ee_pose", **arm))) - position)
            )
            if off > float(reach_tolerance):
                failures.append(
                    f"range {rng:.2f} m: the hand ended {off * 1e3:.0f} mm from the view pose"
                )
                continue
            obs = _settled(ctx, int(arm_id))
            seen = _wrist_camera(
                obs, _matrix(obs["arms"][int(arm_id)]["ee_pose"]), out["camera_name"]
            )
            if seen is None:
                return finish(
                    "not_seen", f"the wrist camera {out['camera_name']!r} left the observation"
                )
            h, w = np.asarray(seen["rgb"]).shape[:2]
            u0, v0, z0 = _project(seen, coarse)
            mx, my = VIEW_MARGIN * w, VIEW_MARGIN * h
            if z0 <= 0.0 or not (mx <= u0 <= w - mx and my <= v0 <= h - my):
                failures.append(
                    f"range {rng:.2f} m: the target projects outside the wrist image ({u0:.0f}, {v0:.0f})"
                )
                continue
            out["camera_range"], out["view_pose"] = rng, pose
            return _point_and_lift(
                ctx,
                out,
                seen,
                coarse,
                (u0, v0),
                allowed,
                target_obb,
                object_name,
                location,
                query,
                pointer,
                finish,
            )
    return finish("unreachable", "; ".join(failures) or "no view to try")


def _point_and_lift(
    ctx: NodeContext,
    out: Output,
    cam: dict[str, Any],
    coarse: np.ndarray,
    hint: tuple[float, float],
    allowed: float,
    target_obb: dict[str, Any] | None,
    object_name: str,
    location: str,
    query: str | None,
    pointer: str,
    finish: Any,
) -> Output:
    rgb = np.asarray(cam["rgb"])
    text = query or (f"{location} of the {object_name}" if location else object_name)
    kwargs: dict[str, Any] = {"image": rgb, "query": text}
    if pointer.startswith("gemini-er."):
        kwargs["max_points"] = 3
    try:
        reply = ctx.tool(pointer, **kwargs)
    except Exception as exc:
        return finish("not_seen", f"{pointer} failed: {exc}")
    if not isinstance(reply, dict) or not reply.get("found"):
        return finish("not_seen", f"{pointer} found no {text!r} in the wrist image")
    points = reply.get("points") or [reply]
    best = min(
        points, key=lambda p: np.hypot(float(p["pixel_x"]) - hint[0], float(p["pixel_y"]) - hint[1])
    )
    u, v = float(best["pixel_x"]), float(best["pixel_y"])
    out["pixel"] = {"u": u, "v": v}

    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    sy, sx = depth.shape[0] / rgb.shape[0], depth.shape[1] / rgb.shape[1]
    z, count = _depth_at(depth, u * sx, v * sy)
    if count < MIN_DEPTH_SAMPLES:
        return finish(
            "not_seen",
            f"only {count} depth readings in {MIN_DEPTH * 100:.0f}-{MAX_DEPTH * 100:.0f} cm round pixel ({u:.0f}, {v:.0f})",
        )
    out["depth"] = z
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    pc = np.array([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, 1.0])
    refined = (_matrix(cam["pose"]) @ pc)[:3]
    drift = float(np.linalg.norm(refined[:2] - coarse[:2]))
    out["refined_point"], out["drift"] = _vec(refined), drift
    if drift > allowed:
        return finish(
            "not_seen",
            f"the pointed {text!r} lies {drift * 1e3:.0f} mm from the coarse estimate in the plane "
            f"(bound {allowed * 1e3:.0f} mm): likely another object",
        )
    if target_obb is not None:
        c = target_obb["center"]
        out["refined_obb"] = {
            **target_obb,
            "center": {
                "x": float(c["x"]) + float(refined[0] - coarse[0]),
                "y": float(c["y"]) + float(refined[1] - coarse[1]),
                "z": float(c["z"]),
            },
        }
    return finish("refined", f"moved {drift * 1e3:.1f} mm in the plane from the coarse estimate")
