# Adapted from MAESTRO (https://github.com/penn-pal-lab/maestro), maestro/api.py
# at e60d326233a238ddc496bf7a08acb7a278e9e66d (ImageEditApi.get_raw_images,
# world_to_image, _draw_point_with_label, _draw_pose, feedback), MIT,
# Copyright (c) 2026 The MAESTRO Authors.
# Changes: the camera pose is gap's camera-to-world Se3Pose instead of a 6-D
# [position, euler] extrinsic; every camera of the observation that sees the
# target goes into the collage (upstream: the wrist and one side camera); the
# pose is drawn as the gripper (tool centre, open jaw line with fingers at
# robot.describe_gripper's span, approach axis) instead of cv2.drawFrameAxes'
# RGB triad, with the hand's own axes read off robot.grasp_frame, and the
# labelled point is a ring, not a disc, so the object under it stays visible; segments
# behind the depth image's surface are dashed and the tool centre's distance
# to that surface and the approach's tilt from vertical go into the question
# in words (upstream drew without depth; its planner had the pose numbers); the
# collage goes to one vlm.query_yes_no with the stated intent instead of back
# into the planning LLM's context, and the answer routes approved / rejected.
"""Draw a target gripper pose onto the camera images and ask a VLM whether it fits the intent."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import pose_to_matrix
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

APPROVED = "approved"
REJECTED = "rejected"
NO_VIEW = "no_view"

#: Length of the approach line behind the tool centre [m]: MAESTRO's
#: drawFrameAxes axis length (0.10).
APPROACH_LINE_M = 0.10
#: Drawn finger length behind the jaw line [m]. A drawing length, about the
#: linear_4310's exposed finger; not used for any decision.
FINGER_DRAW_M = 0.045
#: Open span [m] when robot.describe_gripper states none: the linear_4310's
#: 94 mm (the real station's and the twin's describe_gripper).
DEFAULT_SPAN_M = 0.094
#: A drawn point counts as hidden when it lies this far behind the depth
#: image's surface [m]: the real overhead camera's calibration RMS is 5.4 mm
#: and its depth noise a few mm at 1 m; twice that keeps noise from dashing.
HIDDEN_TOL_M = 0.012
#: Window [px] of the depth median at the tool centre's pixel.
DEPTH_WINDOW_PX = 5
#: Dash length [px]: a segment is cut into pieces this long for the hidden /
#: visible split, and hidden pieces alternate drawn / blank.
DASH_PX = 5.0
MAX_SEGMENT_PIECES = 200

#: Tilt from straight down [deg] -> words for the question. Bands are for
#: wording only: 30 deg is a slanted top grasp, 60-120 deg a side approach.
APPROACH_WORDS = (
    (30.0, "from above (top-down)"),
    (60.0, "from above at a slant"),
    (120.0, "from the side (roughly horizontally)"),
    (180.0, "from below (moving upward)"),
)

# MAESTRO's colours, RGB (upstream drew on BGR images): the labelled point is
# red with a white rim on a blue label box, drawFrameAxes' y axis green and z
# axis blue.
RED = (255, 0, 0)
WHITE = (255, 255, 255)
LABEL_BLUE = (59, 130, 255)
GREEN = (0, 255, 0)
BLUE = (0, 90, 255)

# Hand frame fallback when robot.grasp_frame is not served: fingers along tool
# +z, jaws along tool +y (the real station's and the twin's grasp_frame).
TOOL_APPROACH = np.array([0.0, 0.0, 1.0])
TOOL_CLOSE = np.array([0.0, 1.0, 0.0])


class Output(TypedDict):
    route: str
    approved: bool
    answer: str
    reason: str
    views: list[str]
    pixels: dict[str, list[float]]
    surface_gap_m: dict[str, float | None]
    overlay: Any
    prompt: str


def _vec(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float64)
    return np.asarray(v, dtype=np.float64).reshape(3)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"zero-length axis {v}")
    return v / n


def _rotation(r: Any) -> np.ndarray:
    if isinstance(r, dict):
        return Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    q = np.asarray(r, dtype=np.float64).reshape(4)  # w, x, y, z as gap packs it
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _camera_to_world(pose: Any) -> np.ndarray:
    """gap's Se3Pose, a 4x4, or a packed [x, y, z, qw, qx, qy, qz]."""
    if isinstance(pose, dict):
        return pose_to_matrix(pose)
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    T = np.eye(4)
    T[:3, :3] = _rotation(arr[3:7])
    T[:3, 3] = arr[:3]
    return T


def _frames(observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    return [{"name": str(c.get("name") or f"camera{i}"), **c} for i, c in enumerate(cams)]


def _rgb(frame: dict[str, Any]) -> np.ndarray | None:
    rgb = frame.get("rgb")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _is_wrist(name: str) -> bool:
    return any(k in name.lower() for k in ("wrist", "eye_in_hand", "hand"))


def world_to_image(
    position: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray
) -> tuple[np.ndarray | None, float | None, float | None]:
    """MAESTRO's world_to_image: pixel, pixels per metre at that depth, and the camera z."""
    cam = np.linalg.inv(T_world_cam) @ np.append(np.asarray(position, dtype=np.float64), 1.0)
    if cam[2] <= 0:
        return None, None, None
    uvw = K @ cam[:3]
    uvw /= uvw[2]
    return uvw[:2], float(K[0, 0] / cam[2]), float(cam[2])


def _depth_at(depth: np.ndarray | None, uv: np.ndarray, window: int = 1) -> float | None:
    if depth is None:
        return None
    h, w = depth.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return None
    r = window // 2
    patch = depth[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1].astype(np.float64)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


def gripper_segments(
    centre: np.ndarray,
    approach: np.ndarray,
    close: np.ndarray,
    span: float,
    reach: float,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """World segments of the gripper at the target: approach line, jaw line, the two fingers.

    No palm bar: seen along the approach (an overhead camera, a top-down grasp) it lies on the
    jaw line and hides its dashes.
    """
    half = 0.5 * span * close
    tips = [centre + half + reach * approach, centre - half + reach * approach]
    roots = [centre + half - FINGER_DRAW_M * approach, centre - half - FINGER_DRAW_M * approach]
    return {
        "approach": [(centre - APPROACH_LINE_M * approach, centre)],
        "jaw": [(centre - half, centre + half)],
        "fingers": [(roots[0], tips[0]), (roots[1], tips[1])],
    }


def _draw_segment(
    image: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    depth: np.ndarray | None,
    color: tuple[int, int, int],
    thickness: int,
) -> int:
    """Draw a world segment; pieces behind the depth surface are dashed. Returns hidden pieces."""
    ends = [world_to_image(p, K, T)[0] for p in (a, b)]
    if ends[0] is not None and ends[1] is not None:
        n = int(np.clip(np.linalg.norm(ends[1] - ends[0]) / DASH_PX, 2, MAX_SEGMENT_PIECES))
    else:
        n = MAX_SEGMENT_PIECES
    pts = [a + (b - a) * t for t in np.linspace(0.0, 1.0, n + 1)]
    proj = [world_to_image(p, K, T) for p in pts]
    hidden = 0
    for i in range(n):
        (p0, _, z0), (p1, _, z1) = proj[i], proj[i + 1]
        if p0 is None or p1 is None:
            continue
        d = _depth_at(depth, (p0 + p1) / 2.0)
        behind = d is not None and (z0 + z1) / 2.0 > d + HIDDEN_TOL_M
        hidden += int(behind)
        if behind and i % 2:
            continue
        cv2.line(
            image,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            color,
            max(1, thickness - 1) if behind else thickness,
            cv2.LINE_AA,
        )
    return hidden


def _draw_point_with_label(image: np.ndarray, uv: np.ndarray, scale: float, label: str) -> None:
    """MAESTRO's _draw_point_with_label, with a ring instead of a disc so the object stays visible."""
    x, y = int(round(uv[0])), int(round(uv[1]))
    radius = max(3, int(0.01 * scale + 1))
    cv2.circle(image, (x, y), radius + 1, WHITE, thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, RED, thickness=2, lineType=cv2.LINE_AA)
    label_x, label_y = x + 8, y - 8
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    if label_x + text_w + 2 > image.shape[1]:
        label_x = x - text_w - 10
    if label_y - text_h - 2 < 0:
        label_y = y + text_h + 10
    cv2.rectangle(
        image,
        (label_x - 2, label_y - text_h - 2),
        (label_x + text_w + 2, label_y + baseline),
        LABEL_BLUE,
        -1,
    )
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def collage(parts: list[np.ndarray]) -> np.ndarray:
    """MAESTRO's get_raw_images collage: views side by side, padded to the tallest."""
    target_h = max(p.shape[0] for p in parts)
    padded = [
        np.pad(p, ((0, target_h - p.shape[0]), (0, 0), (0, 0))) if p.shape[0] < target_h else p
        for p in parts
    ]
    return np.concatenate(padded, axis=1)


def _hand_axes(
    ctx: NodeContext, rotation: np.ndarray, on_arm: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """World approach and jaw axes at the target, from the hand's own frame via robot.grasp_frame."""
    try:
        frame = ctx.tool(
            "robot.grasp_frame",
            approach={"x": 0.0, "y": 0.0, "z": -1.0},
            close_heading_deg=0.0,
            **on_arm,
        )
        R0 = _rotation(frame["rotation"])
        a_local = R0.T @ _unit(_vec(frame["approach"]))
        c_local = R0.T @ _unit(_vec(frame["close_axis"]))
        source = "robot.grasp_frame"
    except Exception as exc:  # noqa: BLE001  (connector without grasp_frame)
        logger.warning("preview_target: grasp_frame unavailable (%s); tool +z / +y assumed", exc)
        a_local, c_local, source = TOOL_APPROACH, TOOL_CLOSE, "tool +z / +y (assumed)"
    return _unit(rotation @ a_local), _unit(rotation @ c_local), source


def _gripper_size(ctx: NodeContext, on_arm: dict[str, Any]) -> tuple[float, float]:
    try:
        g = ctx.tool("robot.describe_gripper", **on_arm) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("preview_target: describe_gripper failed (%s); a linear_4310 assumed", exc)
        g = {}
    span = float(g.get("span_m") or g.get("open_footprint_m") or DEFAULT_SPAN_M)
    finger = g.get("finger") or {}
    reach = float(finger.get("reach_m") or 0.0) if finger.get("stated", True) else 0.0
    return span, reach


def build_prompt(
    intent: str,
    views: list[str],
    span: float,
    gaps: dict[str, float | None],
    geometry_cues: bool,
    approach: np.ndarray | None = None,
) -> str:
    legend = ", ".join(
        f"view {i + 1} (left to right): {'wrist' if _is_wrist(n) else 'external'} camera '{n}'"
        for i, n in enumerate(views)
    )
    lines = [
        "A robot arm is about to move its parallel-jaw gripper to a TARGET pose. "
        f"The image shows {len(views)} camera view(s) side by side ({legend}). "
        "The target is drawn on every view; the robot is NOT there yet, and the real arm "
        "in the image is wherever it is now.",
        "- Red ring labelled 'target': the tool centre, the point between the two finger "
        "pads where a grasped object would be held.",
        f"- Green line with two green fingers: the open jaws, {span * 1000:.0f} mm apart; "
        "they close along the green line. The fingertips are the finger ends away from the "
        "blue line.",
        "- Blue line: the approach direction; the hand comes in along it from its far end "
        "toward the red ring.",
    ]
    if geometry_cues and approach is not None:
        tilt = approach_tilt_deg(approach)
        lines.append(
            f"- In 3D the hand approaches {approach_words(tilt)}: its approach direction is "
            f"{tilt:.0f} degrees from straight down."
        )
    if geometry_cues:
        lines.append(
            "- Dashed parts lie behind a surface the camera sees (inside or under an object or "
            "the table)."
        )
        for name, gap in gaps.items():
            if gap is None:
                continue
            where = "in front of" if gap >= 0 else "behind"
            lines.append(
                f"- In '{name}', the tool centre is {abs(gap) * 100:.1f} cm {where} the surface "
                "this camera sees at that pixel, along its line of sight. A grasp normally puts "
                "it just inside the object, a little behind the object's visible surface; several "
                "cm behind the table means inside the table; far in front means in the air."
            )
    lines += [
        "",
        f'Intended action: "{intent}".',
        "Is this target pose right for the intended action: on the right object or part, "
        "approaching from the right side, at the right height (where the jaws can close on "
        "it, not hovering in the air above it, not inside the table or inside another object)?",
    ]
    return "\n".join(lines)


def approach_tilt_deg(approach: np.ndarray) -> float:
    """Angle [deg] between the approach and world -z (gap's world is z-up)."""
    return float(np.degrees(np.arccos(np.clip(-_unit(approach)[2], -1.0, 1.0))))


def approach_words(tilt: float) -> str:
    for limit, words in APPROACH_WORDS:
        if tilt <= limit:
            return words
    return APPROACH_WORDS[-1][1]


def _result(route: str, **fields: Any) -> Output:
    out: Output = {
        "route": route,
        "approved": route == APPROVED,
        "answer": "",
        "reason": "",
        "views": [],
        "pixels": {},
        "surface_gap_m": {},
        "overlay": None,
        "prompt": "",
    }
    out.update(fields)  # type: ignore[typeddict-item]
    return out


def run(
    ctx: NodeContext,
    target_pose: dict[str, Any],
    intent: str,
    observation: dict[str, Any] | None = None,
    cameras: list[str] | None = None,
    jaw_width: float | None = None,
    approach: Any = None,
    close_axis: Any = None,
    geometry_cues: bool = True,
    arm_id: int | None = None,
    model: str | None = None,
) -> Output:
    """Draw ``target_pose`` (a tool-centre pose) as a gripper on the cameras and ask the VLM about ``intent``.

    ``observation`` defaults to a fresh ``robot.get_observation``. ``approach`` / ``close_axis``
    (world vectors, as ``robot.grasp_frame`` returns them) override the axes read off the
    target's rotation; ``jaw_width`` overrides ``robot.describe_gripper``'s open span.
    """
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    centre = _vec(target_pose["position"])
    if approach is not None and close_axis is not None:
        a_w, c_w, axes_from = _unit(_vec(approach)), _unit(_vec(close_axis)), "given"
    else:
        a_w, c_w, axes_from = _hand_axes(ctx, _rotation(target_pose["rotation"]), on_arm)
    c_w = _unit(c_w - (c_w @ a_w) * a_w)
    span, reach = _gripper_size(ctx, on_arm)
    if jaw_width is not None:
        span = float(jaw_width)
    segments = gripper_segments(centre, a_w, c_w, span, reach)

    if observation is None:
        observation = ctx.tool("robot.get_observation")
    frames = _frames(observation)
    if cameras:
        wanted = [str(n) for n in cameras]
        frames = [f for f in frames if f["name"] in wanted]
    frames.sort(key=lambda f: not _is_wrist(f["name"]))  # MAESTRO: wrist view first

    parts: list[np.ndarray] = []
    views: list[str] = []
    pixels: dict[str, list[float]] = {}
    gaps: dict[str, float | None] = {}
    skipped: list[str] = []
    for f in frames:
        rgb = _rgb(f)
        if rgb is None or f.get("intrinsics") is None or f.get("pose") is None:
            skipped.append(f"{f['name']}: no rgb / intrinsics / pose")
            continue
        K = np.asarray(f["intrinsics"], dtype=np.float64)
        T = _camera_to_world(f["pose"])
        uv, scale, z = world_to_image(centre, K, T)
        h, w = rgb.shape[:2]
        if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
            skipped.append(f"{f['name']}: target outside the image")
            continue
        depth = f.get("depth") if geometry_cues else None
        depth = None if depth is None else np.asarray(depth, dtype=np.float64)
        image = rgb.copy()
        for name, color, thick in (
            ("approach", BLUE, 2),
            ("fingers", GREEN, 2),
            ("jaw", GREEN, 3),
        ):
            for a, b in segments[name]:
                _draw_segment(image, a, b, K, T, depth, color, thick)
        _draw_point_with_label(image, uv, float(scale), "target")
        surface = _depth_at(depth, uv, DEPTH_WINDOW_PX)
        gaps[f["name"]] = None if surface is None else float(surface - z)
        pixels[f["name"]] = [float(uv[0]), float(uv[1])]
        parts.append(image)
        views.append(f["name"])

    if not parts:
        reason = "no camera sees the target" + (f" ({'; '.join(skipped)})" if skipped else "")
        logger.info("preview_target: %s", reason)
        return _result(NO_VIEW, reason=reason)

    overlay = collage(parts)
    prompt = build_prompt(intent, views, span, gaps, geometry_cues, a_w)
    kwargs: dict[str, Any] = {"prompt": prompt, "image": overlay}
    if model:
        kwargs["model"] = model
    common = {
        "views": views,
        "pixels": pixels,
        "surface_gap_m": gaps,
        "overlay": overlay,
        "prompt": prompt,
    }
    try:
        reply = ctx.tool("vlm.query_yes_no", **kwargs)
    except Exception as exc:  # noqa: BLE001  (auth / network / unserved tool)
        logger.warning("preview_target: vlm.query_yes_no failed: %s", exc)
        return _result(NO_VIEW, reason=f"vlm.query_yes_no failed: {exc}", **common)
    answer = str(reply.get("text") or "")
    route = APPROVED if bool(reply.get("answer")) else REJECTED
    logger.info(
        "preview_target: %s (axes from %s, views %s): %s", route, axes_from, views, answer[:200]
    )
    return _result(route, answer=answer, reason=answer.strip()[:300], **common)
