# Adapted from Hello Robot's stretch_ai (https://github.com/hello-robot/stretch_ai, commit
# 6272b28e3ff7be482af3b427a4551d1772ebf7d7), src/stretch/agent/operations/grasp_object.py
# (GraspObjectOperation.visual_servo_to_object, _compute_center_depth, get_target_mask,
# grasp_open_loop, _grasp) and src/stretch/utils/filters.py (MaskTemporalFilter),
# Apache-2.0, Copyright (c) Hello Robot, Inc.
# Changes: world-frame hub inputs (a coarse OBB or point) instead of the agent's instance
# memory; the pregrasp standoff is a Cartesian pose straight above the target, kept only
# where the wrist camera (its extrinsic read from the observation with
# refining-target-from-wrist-closeup's helpers, loaded from that skill) would see the
# target, else that skill's camera-aimed view, instead of a Stretch joint-space standoff;
# the target is a segmenter tool's mask (sam3.segment_text by default) instead of the
# agent's detector classes, chosen as the mask under the aim pixel, else the one nearest
# the last centroid, and gated by a jump bound; the servo error is metric (the midpoint of
# the extent of the mask's top-face points back-projected with the wrist depth, minus the
# tool centre) rather than the mask centroid's pixel offset, so it carries to any range
# and to a camera mounted behind and above the fingers; Stretch's base_x / lift / arm / wrist_pitch steps are small
# capped robot.go_to_pose_cartesian steps of the tool centre, each previewed with
# robot.solve_ik; the source's pixel alignment thresholds and eroded-mask test still gate
# the descent, and a metric tolerance gates the last part of it, past the target's top;
# close fires at a grasp height from the measured top (not a fixed 0.17 m camera depth);
# a lost frame holds the hand and re-looks instead of reusing a stale mask; the
# not-moving counter reads the tool back instead of joint deltas; a stall, a lost track
# and the iteration cap all fall back to the open-loop grasp at the last seen point (the
# source closed in place on a stall and gave up at its timeout); the jaws pre-shape to
# the target's width plus a margin; routed exits grasped / missed / lost_target /
# unreachable judged from the jaw gap instead of a bool.
"""Grasp a small part by closed-loop wrist-camera servoing: centre it under the tool, descend while centred, close."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy import ndimage

logger = logging.getLogger(__name__)

AIM_SKILL = "refining-target-from-wrist-closeup"
AIM_SCRIPT = Path("scripts") / "refine_from_wrist.py"

SOURCE_WIDTH = 320
"""Width of the Stretch D405 servo image the source's pixel constants were tuned at
[px] (its debug view tiles the 320 x 240 ``ee_rgb``); they scale with the frame."""

ALIGN_X_PX = 28
ALIGN_Y_PX = 20
"""The source's ``align_x_threshold`` / ``align_y_threshold`` [px at 320 wide]: the
target's top centre within these of the aim pixel lets the hand descend."""

CENTER_REGION_PX = 15
"""The source's ``local_region_size`` [px at 320 wide]: the half-window round the aim
pixel whose masked depths give the centre depth."""

MIN_MASK_PX = 100
"""The source's ``min_points_to_approach`` [px at 320 x 240]: smaller masks are not
trusted (scaled by the frame's area)."""

ERODE_FRAC = 0.2
"""The mask is eroded by this fraction of its bounding box's shorter side before the
source's "is the aim pixel inside the mask" test. The source eroded a fixed ~20 px
(a 4 x 4 kernel 10 times), sized for household objects; a 25 mm part at 0.15 m is
~60 px across at 640 wide and would vanish."""

SCORE_FLOOR = 0.3
"""Segmenter masks scoring under this are ignored (the sam3 bundle's advice)."""

TOP_BAND = 0.0015
"""Mask points within this of the highest (95th percentile) height count as the top
face [m], so side faces seen at 45 deg do not drag the centre toward the camera: a
4 mm band let a 25 mm cube's near face in and put the centre 4 mm short in the fake
rig. Above the D405's depth noise at 0.1-0.3 m (well under a millimetre), under a
LEGO stud's 1.7 mm (on a brick the stud tops are the top face)."""

ALIGN_TOLERANCE = 0.0015
"""Lateral error allowed before the fingertips go below the target's top, and at the
close [m]: well inside the default 6 mm pre-shape margin per side."""

DEADBAND = 0.0005
"""Lateral errors under this are not corrected [m]."""

LATERAL_STEP = 0.010
"""Largest lateral correction per step [m]."""

DESCENT_STEP = 0.015
"""Largest descent per step [m]. The source stepped 0.05 m on a Stretch (lift_arm_ratio);
small parts need the frame re-read more often near the top."""

CLOSE_TOLERANCE = 0.002
"""The tool centre within this of the grasp height counts as there [m]."""

GRASP_DEPTH = 0.012
"""How far below the target's top the tool centre goes when the height is unknown, and
the most it goes otherwise (half the height when that is less) [m]."""

TIP_REACH = 0.0102
"""Fingertips past the tool centre [m] when robot.describe_gripper states none: the
station's linear_4310 (as in grasping-flat-objects)."""

TIP_CLEARANCE = 0.005
"""Fingertips stay this far above the target's top while the error exceeds
align_tolerance [m]."""

MIN_TABLE_CLEARANCE = 0.014
"""Lowest tool centre over the support [m]: a real YAM station refuses lower."""

PREGRASP_MARGIN = 0.006
"""Jaw opening beyond the target's width on each side for the pre-shape [m]: room for
1.5 mm of servo error and the pads' compliance, narrow enough not to sweep a neighbour
a centimetre away."""

PREGRASP_HEIGHT = 0.06
"""Tool centre over the target's top at the pregrasp standoff [m]: the fingertips 50 mm
clear (refining-target-from-wrist-closeup's clearance floor). On a YAM the D405 then
sees the target ~12 deg above its optical axis at ~0.16 m. The source stood off 0.3 m
along the approach on a Stretch."""

FALLBACK_HEIGHTS = (0.06, 0.08, 0.10)
"""Standoffs tried after the asked one [m]."""

PREGRASP_RANGE = 0.20
"""Camera-to-target range of the fallback pregrasp view, the target on the optical axis
[m]: refining-target-from-wrist-closeup's default view."""

FALLBACK_RANGES = (0.20, 0.25, 0.30)
"""Fallback-view ranges tried after the asked one [m]."""

MAX_ITERATIONS = 30
"""Most servo frames before the open-loop fallback. The source ran for 120 s."""

MAX_LOST = 3
"""Consecutive frames without the target before the fallback. The source allowed 10
frames of a ~5 Hz loop while moving; here the hand holds still while it re-looks, so
more frames of the same view rarely help."""

MAX_STALLS = 3
"""Consecutive steps that moved under STALL_FRACTION of their length before the
fallback (the source's not-moving counter, 50 loop turns with joints still within
0.05)."""

STALL_FRACTION = 0.3
"""A step that moved less than this fraction of its commanded length counts as stalled."""

MAX_JUMP = 0.02
"""A top-face estimate further than this from the last one is a different object [m]."""

DRIFT_MARGIN = 0.03
"""First-sight bound over the target's half-diagonal from the coarse estimate [m]: as
refining-target-from-wrist-closeup's."""

MAX_DRIFT = 0.05
"""First-sight bound when only a point is given [m]."""

LIFT_HEIGHT = 0.05
"""Lift after the close [m] (the source lifted 0.3 m on a Stretch's lift joint)."""

EMPTY_MARGIN = 0.003
"""A jaw gap within this of fully closed closed on nothing [m] (grasping-flat-objects)."""

MAX_TILT_DEG = 20.0
"""The grasp rotation's approach axis must lie within this of straight down."""

_AIM: Any = None


def aim_module() -> Any:
    """refining-target-from-wrist-closeup's script (pose, camera and settle helpers),
    from the skill installed beside this one or under a GAP_SKILLS_PATH root."""
    global _AIM
    if _AIM is not None:
        return _AIM
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (root / AIM_SKILL / AIM_SCRIPT, root / "skills" / AIM_SKILL / AIM_SCRIPT):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("_servo_grasp_aim", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _AIM = module
                return module
    raise RuntimeError(
        f"{AIM_SKILL} is not installed beside this skill (looked under "
        f"{', '.join(str(r) for r in roots)})"
    )


class Output(TypedDict):
    route: str
    mode: str
    grasp_point: dict[str, float]
    coarse_point: dict[str, float]
    correction: float
    final_error: float
    iterations: int
    lost_frames: int
    fallback_reason: str
    grip_width: float
    held_width: float
    lifted: float
    camera_name: str
    trace: list[str]
    detail: str


def _as_mask(m: Any, shape: tuple[int, int]) -> np.ndarray | None:
    a = np.asarray(m)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] <= 4 else a[0]
    if a.ndim != 2 or a.shape != shape:
        return None
    return a > (0.5 if a.dtype.kind == "f" else 0)


def _mask_points(cam: dict[str, Any], mask: np.ndarray, near: float, far: float) -> np.ndarray:
    """The masked pixels with a depth in [near, far], back-projected to the world, [N, 3]."""
    depth = cam.get("depth")
    if depth is None:
        return np.zeros((0, 3))
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = mask.shape
    rows = (np.arange(d.shape[0]) * h // d.shape[0]).astype(int)
    cols = (np.arange(d.shape[1]) * w // d.shape[1]).astype(int)
    inside = mask[np.ix_(rows, cols)] & np.isfinite(d) & (d >= near) & (d <= far)
    r, c = np.nonzero(inside)
    if r.size == 0:
        return np.zeros((0, 3))
    z = d[r, c]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    u, v = (c + 0.5) * (w / d.shape[1]) - 0.5, (r + 0.5) * (h / d.shape[0]) - 0.5
    pc = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, np.ones_like(z)])
    t_cam = _AIM._matrix(cam["pose"])
    return (t_cam @ pc)[:3].T


def _top_face(points: np.ndarray, band: float, axes: np.ndarray | None = None) -> np.ndarray:
    """The top face's centre: the midpoint of the extent (2nd to 98th percentile) of the
    points within ``band`` of the 95th-percentile height, along the hand's two lateral
    axes (columns of ``axes``; world x, y by default), at that height. An extent is
    blind to how densely pixels sample the face (its near half covers more of them
    under perspective) and to a sliver of the near side face inside the band, both of
    which pull a pixel mean toward the camera; it is also what closing jaws centre on."""
    top = float(np.percentile(points[:, 2], 95))
    face = points[points[:, 2] >= top - band, :2]
    a = np.eye(2) if axes is None else np.asarray(axes, dtype=np.float64)[:2, :2]
    a = a / np.linalg.norm(a, axis=0)
    s = face @ a
    mid = 0.5 * (np.percentile(s, 2, axis=0) + np.percentile(s, 98, axis=0))
    xy = np.linalg.solve(a.T, mid)
    return np.array([float(xy[0]), float(xy[1]), top])


def _center_depth(cam: dict[str, Any], mask: np.ndarray, u: float, v: float, half: int) -> float:
    """The source's _compute_center_depth: the median depth of the masked pixels in a
    window round (u, v), 0 when none."""
    depth = cam.get("depth")
    if depth is None or not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
        return 0.0
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    sy, sx = d.shape[0] / mask.shape[0], d.shape[1] / mask.shape[1]
    window = np.zeros_like(mask)
    r, c = int(v), int(u)
    window[max(0, r - half) : r + half, max(0, c - half) : c + half] = True
    rr, cc = np.nonzero(window & mask)
    if rr.size == 0:
        return 0.0
    vals = d[(rr * sy).astype(int), (cc * sx).astype(int)]
    vals = vals[np.isfinite(vals) & (vals > 1e-8)]
    return float(np.median(vals)) if vals.size else 0.0


def run(
    ctx: NodeContext,
    object_name: str,
    target_obb: dict[str, Any] | None = None,
    target_point: dict[str, float] | None = None,
    query: str | None = None,
    segmenter: str = "sam3.segment_text",
    servo: bool = True,
    fallback: bool = True,
    camera: str | None = None,
    object_width: float | None = None,
    object_height: float | None = None,
    grasp_depth: float | None = None,
    pregrasp_margin: float = PREGRASP_MARGIN,
    pregrasp_height: float = PREGRASP_HEIGHT,
    pregrasp_range: float = PREGRASP_RANGE,
    align_tolerance: float = ALIGN_TOLERANCE,
    lateral_step: float = LATERAL_STEP,
    descent_step: float = DESCENT_STEP,
    max_iterations: int = MAX_ITERATIONS,
    lift_height: float = LIFT_HEIGHT,
    min_score: float = SCORE_FLOOR,
    check_reach: bool = True,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
) -> Output:
    aim = aim_module()
    if target_obb is not None:
        coarse = aim._top_point(target_obb)
        ext = target_obb["extent"]
        half = np.array([float(ext[k]) for k in "xyz"])
        r_obb = aim._rot(target_obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        height = 2.0 * float(np.abs(r_obb[2]) @ half)
        allowed = float(np.hypot(half[0], half[1])) + DRIFT_MARGIN
    elif target_point is not None:
        coarse = np.array([float(target_point[k]) for k in "xyz"])
        height = 0.0
        allowed = MAX_DRIFT
    else:
        raise ValueError("grasping-by-wrist-camera-servoing needs target_obb or target_point")
    if object_height is not None:
        height = float(object_height)
    text = query or object_name
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    move: dict[str, Any] = dict(arm)
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    out: Output = {
        "route": "unreachable",
        "mode": "servo" if servo else "open_loop",
        "grasp_point": aim._vec(coarse),
        "coarse_point": aim._vec(coarse),
        "correction": 0.0,
        "final_error": -1.0,
        "iterations": 0,
        "lost_frames": 0,
        "fallback_reason": "",
        "grip_width": -1.0,
        "held_width": -1.0,
        "lifted": 0.0,
        "camera_name": "",
        "trace": [],
        "detail": "",
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        logger.info("grasping-by-wrist-camera-servoing: %s (%s)", route, detail)
        return out

    # The hand: jaw gap reading, finger reach.
    try:
        hand = ctx.tool("robot.describe_gripper", **arm)
    except Exception as exc:  # defaults stand in
        logger.info("grasping-by-wrist-camera-servoing: no gripper description (%s)", exc)
        hand = {}
    fit = (hand or {}).get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or (hand or {}).get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    finger = (hand or {}).get("finger") or {}
    tip_reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    tip_reach = tip_reach or TIP_REACH

    def width(reply: Any = None) -> float:
        """The jaw gap [m] (robot.get_gripper's 1.0 = open), or -1.0 when unreadable."""
        try:
            fraction = reply.get("position") if isinstance(reply, dict) else None
            if fraction is None:
                fraction = ctx.tool("robot.get_gripper", **arm)["position"]
            if at_open <= 0.0:
                return -1.0
            return at_closed + max(0.0, min(1.0, float(fraction))) * (at_open - at_closed)
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.info("grasping-by-wrist-camera-servoing: could not read the gripper (%s)", exc)
            return -1.0

    def here() -> np.ndarray:
        return aim._matrix(aim._pose(ctx.tool("robot.get_ee_pose", **arm)))

    def ik_ok(position: np.ndarray, rotation: dict[str, float]) -> str:
        if not check_reach:
            return ""
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": aim._vec(position), "rotation": rotation}, **arm
            )
        except Exception as exc:
            return f"IK preview refused ({exc})"
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return "out of reach (IK preview)"
        return ""

    def step(
        position: np.ndarray, rotation: dict[str, float], tool: str = "robot.go_to_pose_cartesian"
    ) -> str:
        """Why the move did not happen, or "" (raised and routed refusals alike)."""
        try:
            kwargs = dict(arm) if tool == "robot.go_to_pose_cartesian" else dict(move)
            return aim._refusal(
                ctx.tool(
                    tool, pose={"position": aim._vec(position), "rotation": rotation}, **kwargs
                )
            )
        except Exception as exc:
            return str(exc) or type(exc).__name__

    def grasp_height(top_z: float) -> float:
        depth = (
            float(grasp_depth)
            if grasp_depth is not None
            else (min(GRASP_DEPTH, height / 2.0) if height > 0.0 else GRASP_DEPTH)
        )
        z = top_z - depth
        if height > 0.0:
            z = max(z, top_z - height + MIN_TABLE_CLEARANCE)
        return z

    def close_and_lift(rotation: dict[str, float]) -> Output:
        out["grip_width"] = width(ctx.tool("robot.close_gripper", **arm))
        reached = here()[:3, 3]
        if out["grip_width"] >= 0.0 and out["grip_width"] <= at_closed + EMPTY_MARGIN:
            ctx.tool("robot.open_gripper", **arm)
            step(reached + np.array([0.0, 0.0, float(lift_height)]), rotation)
            return finish(
                "missed",
                f"{out['mode']}: the jaws closed on nothing ({out['grip_width'] * 1e3:.1f} mm)",
            )
        note = ""
        if float(lift_height) > 0.0:
            why = step(reached + np.array([0.0, 0.0, float(lift_height)]), rotation)
            out["lifted"] = float(here()[2, 3] - reached[2])
            if out["lifted"] < 0.5 * float(lift_height):
                note = f"; the lift rose {out['lifted'] * 1e3:.0f} mm ({why or 'stalled'})"
        out["held_width"] = width()
        if out["held_width"] >= 0.0 and out["held_width"] <= at_closed + EMPTY_MARGIN:
            ctx.tool("robot.open_gripper", **arm)
            return finish(
                "missed",
                f"{out['mode']}: closed at {out['grip_width'] * 1e3:.1f} mm but the part slipped out in the lift{note}",
            )
        unread = " (the jaw gap could not be read)" if out["grip_width"] < 0.0 else ""
        return finish(
            "grasped",
            f"{out['mode']}: closed at {out['grip_width'] * 1e3:.1f} mm, "
            f"{out['held_width'] * 1e3:.1f} mm after the lift{unread}{note}",
        )

    def open_loop(point: np.ndarray, rotation: dict[str, float], why: str) -> Output:
        """The source's grasp_open_loop: straight to the point, close, lift."""
        if why:
            out["fallback_reason"] = why
            out["mode"] = "open_loop_fallback"
        out["grasp_point"] = aim._vec(point)
        start = here()[:3, 3]
        hover = max(float(start[2]), float(point[2]) + tip_reach + TIP_CLEARANCE)
        path = [np.array([start[0], start[1], hover])] if start[2] < hover - 1.0e-3 else []
        path += [
            np.array([point[0], point[1], hover]),
            np.array([point[0], point[1], grasp_height(float(point[2]))]),
        ]
        for p in path:
            refused = ik_ok(p, rotation) or step(p, rotation)
            if refused and not refused.startswith(("timeout", "contact")):
                after = f" (fell back because {why})" if why else ""
                return finish(
                    "unreachable",
                    f"{out['mode']}: the move to the grasp was refused: {refused}{after}",
                )
        return close_and_lift(rotation)

    # The grasp rotation: top-down from robot.grasp_frame, jaws across the box's short side.
    obs = ctx.tool("robot.get_observation")
    tool_now = aim._matrix(obs["arms"][int(arm_id)]["ee_pose"])
    rotations: list[np.ndarray] = []
    frame_kw: dict[str, Any] = dict(arm)
    if target_obb is not None:
        r_obb = aim._rot(target_obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        ex, ey = float(target_obb["extent"]["x"]), float(target_obb["extent"]["y"])
        if abs(ex - ey) > 0.1 * max(ex, ey):
            short = r_obb[:, 0] if ex < ey else r_obb[:, 1]
            frame_kw["close_heading_deg"] = float(np.degrees(np.arctan2(short[1], short[0])))
    try:
        try:
            frame = ctx.tool("robot.grasp_frame", **frame_kw)
        except Exception:
            if "close_heading_deg" not in frame_kw:
                raise
            frame = ctx.tool("robot.grasp_frame", **arm)
        rotations += [
            aim._rot(frame[k])
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:  # the hand's own rotation, if it is top-down
        logger.info("grasping-by-wrist-camera-servoing: no grasp frame (%s)", exc)
    rotations.append(tool_now[:3, :3])
    down = np.array([0.0, 0.0, -1.0])
    rotations = [
        r for r in rotations if float(r[:, 2] @ down) >= math.cos(math.radians(MAX_TILT_DEG))
    ]
    if not rotations:
        return finish(
            "unreachable",
            "no top-down grasp rotation: robot.grasp_frame gave none and the hand is tilted",
        )

    if object_width is not None:
        across = float(object_width)
    elif target_obb is not None:
        across = 2.0 * min(float(target_obb["extent"]["x"]), float(target_obb["extent"]["y"]))
    else:
        across = 0.0

    def preshape() -> None:
        if across > 0.0:
            w = across + 2.0 * float(pregrasp_margin)
            if at_open > 0.0:
                w = min(w, at_open)
            try:
                if not aim._refusal(ctx.tool("robot.set_grip", width_m=float(w), **arm)):
                    return
            except Exception as exc:  # a connector without a partial grip opens fully
                logger.info("grasping-by-wrist-camera-servoing: no set_grip (%s)", exc)
        ctx.tool("robot.open_gripper", **arm)

    if not servo:
        preshape()
        r = rotations[0]
        return open_loop(coarse, aim._quat(r), "")

    # Pregrasp: the source's standoff along the approach (the tool centre straight above
    # the target), kept only where the wrist camera would see the target well inside
    # its image; else refining-target-from-wrist-closeup's view with the target on the
    # camera's optical axis, the hand top-down.
    cam = aim._wrist_camera(obs, tool_now, camera)
    if cam is None:
        names = [c.get("name") for c in obs.get("cameras") or []]
        return finish(
            "lost_target", f"no wrist camera with a pose in the observation (cameras: {names})"
        )
    out["camera_name"] = str(cam.get("name", ""))
    tool_cam = np.linalg.inv(tool_now) @ aim._matrix(cam["pose"])
    axis_tool, offset_tool = tool_cam[:3, 2], tool_cam[:3, 3]
    h_img, w_img = np.asarray(cam["rgb"]).shape[:2]
    mx, my = aim.VIEW_MARGIN * w_img, aim.VIEW_MARGIN * h_img

    def in_view(position: np.ndarray, r_tool: np.ndarray) -> str:
        """ "" when the coarse target projects inside the image margin within the
        camera's depth range from this hand pose, else why not."""
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r_tool, position
        eye = pose @ tool_cam
        seen = {
            "intrinsics": cam["intrinsics"],
            "pose": {"position": aim._vec(eye[:3, 3]), "rotation": aim._quat(eye[:3, :3])},
        }
        u, v, z = aim._project(seen, coarse)
        if not (
            aim.MIN_DEPTH <= z <= aim.MAX_DEPTH and mx <= u <= w_img - mx and my <= v <= h_img - my
        ):
            return f"the target would project at ({u:.0f}, {v:.0f}), {z:.2f} m deep"
        return ""

    views: list[tuple[str, np.ndarray, np.ndarray]] = []
    for r_tool in rotations:
        for hgt in dict.fromkeys([float(pregrasp_height), *FALLBACK_HEIGHTS]):
            views.append((f"standoff {hgt:.2f} m", r_tool, coarse + np.array([0.0, 0.0, hgt])))
    for r_tool in rotations:
        for rng in dict.fromkeys([float(pregrasp_range), *FALLBACK_RANGES]):
            position = coarse - rng * (r_tool @ axis_tool) - r_tool @ offset_tool
            views.append((f"camera range {rng:.2f} m", r_tool, position))
    failures: list[str] = []
    rotation: dict[str, float] | None = None
    moves = 0
    for name, r_tool, position in views:
        if moves >= aim.MAX_MOVES or rotation is not None:
            break
        if position[2] < coarse[2] + aim.MIN_TOOL_CLEARANCE - 1.0e-9:
            failures.append(f"{name}: too close to the target")
            continue
        why = in_view(position, r_tool)
        q = aim._quat(r_tool)
        why = why or ik_ok(position, q)
        if why:
            failures.append(f"{name}: {why}")
            continue
        moves += 1
        start = here()[:3, 3]
        if start[2] < position[2] - 1.0e-3:
            why = step(
                np.array([start[0], start[1], position[2]]),
                aim._quat(here()[:3, :3]),
                "robot.go_to_pose",
            )
            why = f"rise: {why}" if why else ""
        why = why or step(position, q, "robot.go_to_pose")
        off = float(np.linalg.norm(here()[:3, 3] - position))
        if why or off > aim.REACH_TOLERANCE:
            failures.append(f"{name}: {why or f'ended {off * 1e3:.0f} mm off'}")
            continue
        rotation = q
    if rotation is None:
        return finish(
            "unreachable", "no pregrasp view: " + ("; ".join(failures) or "nothing to try")
        )
    preshape()

    # The servo loop (visual_servo_to_object).
    scale = None
    hint: tuple[float, float] | None = None
    last: np.ndarray | None = None
    lost = stalls = 0
    first = True
    while True:
        if out["iterations"] >= int(max_iterations):
            why = f"no close after {out['iterations']} frames"
            break
        out["iterations"] += 1
        obs = aim._settled(ctx, int(arm_id))
        tool = aim._matrix(obs["arms"][int(arm_id)]["ee_pose"])
        cam = aim._wrist_camera(obs, tool, out["camera_name"])
        if cam is None:
            return finish(
                "lost_target", f"the wrist camera {out['camera_name']!r} left the observation"
            )
        rgb = np.asarray(cam["rgb"])
        h, w = rgb.shape[:2]
        if scale is None:
            scale = w / float(SOURCE_WIDTH)
            if hint is None:
                u0, v0, z0 = aim._project(cam, coarse)
                hint = (u0, v0) if z0 > 0.0 else None
        tool_xyz = tool[:3, 3]
        # Where a point at the target's top under the tool centre appears: the image
        # point the target is brought onto (the source's tracked image point).
        top_guess = float(last[2]) if last is not None else float(coarse[2])
        ua, va, _ = aim._project(cam, np.array([tool_xyz[0], tool_xyz[1], top_guess]))

        seen, why_lost = None, ""
        try:
            found = ctx.tool(segmenter, image=rgb, query=text, max_results=3)
        except Exception as exc:
            found, why_lost = None, f"{segmenter} failed: {exc}"
        masks = (found or {}).get("masks") or []
        scores = list((found or {}).get("scores") or [1.0] * len(masks))
        min_px = MIN_MASK_PX * scale * scale
        cands = []
        for m, s in zip(masks, scores, strict=False):
            mk = _as_mask(m, (h, w))
            if mk is not None and float(s) >= float(min_score) and int(mk.sum()) >= min_px:
                cands.append(ndimage.binary_dilation(mk))
        if cands:
            iu, iv = int(round(ua)), int(round(va))
            under = [c for c in cands if 0 <= iv < h and 0 <= iu < w and c[iv, iu]]
            if under:
                mask = under[0]
            elif hint is not None:
                mask = min(
                    cands, key=lambda c: float(np.hypot(*(np.argwhere(c).mean(0)[::-1] - hint)))
                )
            else:
                mask = max(cands, key=lambda c: int(c.sum()))
            pts = _mask_points(cam, mask, aim.MIN_DEPTH, aim.MAX_DEPTH)
            if len(pts) < aim.MIN_DEPTH_SAMPLES:
                why_lost = f"only {len(pts)} depth readings on the mask"
            else:
                est = _top_face(pts, TOP_BAND, tool[:3, :2])
                ref, bound = (coarse, allowed) if last is None else (last, MAX_JUMP)
                jump = float(np.linalg.norm(est[:2] - ref[:2]))
                if jump > bound:
                    why_lost = (
                        f"the mask's top lies {jump * 1e3:.0f} mm from the "
                        f"{'coarse estimate' if last is None else 'last estimate'} (bound {bound * 1e3:.0f} mm)"
                    )
                else:
                    seen = est
        elif not why_lost:
            why_lost = f"{segmenter} found no {text!r} scoring {float(min_score):.2f}"
        if seen is None:
            lost += 1
            out["lost_frames"] += 1
            out["trace"].append(f"lost: {why_lost}")
            if last is None and lost >= MAX_LOST:
                return finish(
                    "lost_target", f"never saw the {text!r} from the pregrasp view: {why_lost}"
                )
            if lost >= MAX_LOST:
                why = f"lost the target for {lost} frames ({why_lost})"
                if not fallback:
                    return finish("lost_target", why + "; the hand is left where it lost it")
                break
            continue
        lost = 0
        last = seen
        if first:
            out["correction"] = float(np.linalg.norm(seen[:2] - coarse[:2]))
            first = False
        ys, xs = np.nonzero(mask)
        hint = (float(xs.mean()), float(ys.mean()))
        out["grasp_point"] = aim._vec(seen)

        err = seen[:2] - tool_xyz[:2]
        e = float(np.linalg.norm(err))
        out["final_error"] = e
        zg = grasp_height(float(seen[2]))
        z_safe = float(seen[2]) + tip_reach + TIP_CLEARANCE
        dz = float(tool_xyz[2]) - zg
        ut, vt, _ = aim._project(cam, seen)
        du, dv = ut - ua, vt - va
        aligned = abs(du) < ALIGN_X_PX * scale and abs(dv) < ALIGN_Y_PX * scale
        side = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        eroded = ndimage.binary_erosion(mask, iterations=max(1, int(ERODE_FRAC * side)))
        iu, iv = int(round(ua)), int(round(va))
        center_in_mask = bool(0 <= iv < h and 0 <= iu < w and eroded[iv, iu])
        cdepth = _center_depth(cam, mask, ua, va, int(round(CENTER_REGION_PX * scale)))
        fine = e <= float(align_tolerance)
        out["trace"].append(
            f"frame {out['iterations']}: error {e * 1e3:.1f} mm, {dz * 1e3:.0f} mm over the grasp height, "
            f"pixel offset ({du:.0f}, {dv:.0f}), aligned={aligned}, centre_in_mask={center_in_mask}, "
            f"centre depth {cdepth:.3f} m"
        )
        if fine and dz <= CLOSE_TOLERANCE and cdepth > 0.0:
            return close_and_lift(rotation)

        lateral = np.zeros(2)
        if e > DEADBAND:
            lateral = err * min(1.0, float(lateral_step) / e)
        z_now = float(tool_xyz[2])
        z_next = z_now
        if fine:
            z_next = max(zg, z_now - float(descent_step))
        elif z_now < z_safe - 1.0e-3:
            # Fingertips below the top while off: rise before moving across.
            z_next, lateral = z_safe, np.zeros(2)
        elif (aligned or center_in_mask) and z_now > z_safe:
            z_next = max(z_safe, z_now - float(descent_step))
        target = np.array([tool_xyz[0] + lateral[0], tool_xyz[1] + lateral[1], z_next])
        length = float(np.linalg.norm(target - tool_xyz))
        if length < 1.0e-4:
            # Aligned at the grasp height but the aim window is off the mask: nothing to move.
            if fine and dz <= CLOSE_TOLERANCE:
                return close_and_lift(rotation)
            stalls += 1
            if stalls >= MAX_STALLS:
                why = "no step left to take"
                break
            continue
        refused = ""
        for _ in range(2):
            refused = ik_ok(target, rotation) or step(target, rotation)
            if not refused or refused.startswith(("timeout", "contact")):
                break
            out["trace"].append(f"refused: {refused}")
            target = tool_xyz + 0.5 * (target - tool_xyz)
            length *= 0.5
        if refused and not refused.startswith(("timeout", "contact")):
            why = f"a servo step was refused at full and half length: {refused}"
            break
        moved = float(np.linalg.norm(here()[:3, 3] - tool_xyz))
        if length >= 0.002 and moved < STALL_FRACTION * length:
            stalls += 1
            out["trace"].append(f"stalled: moved {moved * 1e3:.1f} of {length * 1e3:.1f} mm")
            if stalls >= MAX_STALLS:
                why = f"the arm stalled for {stalls} steps"
                break
        else:
            stalls = 0

    if last is None:
        return finish("lost_target", f"never saw the {text!r}: {why}")
    return open_loop(last, rotation, why)
