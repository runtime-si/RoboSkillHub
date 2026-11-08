# Adapted from AMMistry18/aic (https://github.com/AMMistry18/aic, commit
# fd27bbe875dbd656e381f72ae2b44efdfb243924),
# flowstate/aic_perception/aic_perception/board_visibility.py (analyze_board,
# view_quality, search_progress, _continuous_direction, adaptive_action,
# decide_direction, world_delta, optical_axes_in_base) and
# flowstate/aic_perception/move_to_board_skill.py (the observe / step / re-observe
# loop), Apache-2.0.
# Changes: the target is a sam3.segment_text mask of any named object instead of
# an Otsu-thresholded dark board anchored by a magenta fiducial (the board-only
# rectangularity, long-axis and IVM-readiness tests and the "reorient" mode are
# dropped); one wrist camera instead of ranking several (combine_cameras): one arm
# carries one wrist camera; the camera's axes come from the frame's pose in the
# observation (as refining-target-from-wrist-closeup reads them, whose helpers
# this script loads) instead of a TF lookup; an optional bottom band stands in for
# the calibrated gripper mask and counts as the image's bottom edge; the base step
# halves when a correction overshoots (a clipped edge flips to the opposite one) or
# a move is refused, and a lost target sends the hand back to the last pose that
# saw it; an approach closes by the 1/range^2 area law from the mask's wrist depth;
# approach / back-off stay within the D405's depth range and keep the tool centre
# clear of the target; mask islands under 5 % of the largest are dropped; the
# skill loops to a move cap with routed exits framed / not_seen / step_limit /
# unreachable instead of one move and a message.
"""Step the wrist camera across / along its optical axis until a segmented target lies wholly inside the image with a margin."""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy import ndimage

logger = logging.getLogger(__name__)

AIM_SKILL = "refining-target-from-wrist-closeup"
AIM_SCRIPT = Path("scripts") / "refine_from_wrist.py"

MARGIN_PX = 15
"""Least clear space between the mask and every image edge [px]: analyze_board's
``margin_px`` default (15 at the source's camera resolution, comparable to a
640 x 480 D405)."""

CONTEXT_PAD_FRAC = 0.10
"""The margin grows to this fraction of the mask's larger bounding-box side when
that is more: analyze_board's ``context_pad_frac`` default, which reserves image
context round parts that protrude from the segmented body."""

MIN_AREA_FRAC = 0.003
"""Below this fraction of the image the target is too small to work with and the
camera approaches it (the source's ``min_detail_area_frac`` was 0.02 for a whole
board; 0.003 is ~30 x 30 px at 640 x 480, a 30 mm cube at ~0.4 m on a D405)."""

MIN_BLOB_PX = 50
"""Mask islands smaller than this are dropped before the edges are read: the
source's ``PURPLE_MIN_BLOB_PX``, so a speck at an image edge does not read as a
clipped target."""

ISLAND_FRAC = 0.05
"""Mask islands smaller than this fraction of the largest are dropped too. On a
real wrist D405 frame SAM3's mask of a plate seen through railing wires came in
five pieces: the wire-cut parts were 12-100 % of the largest, and a 112 px glint
on a step 20 px from the left edge (2.5 %) read as a clipped left edge."""

SCORE_FLOOR = 0.3
"""sam3.segment_text masks scoring under this are ignored (the sam3 bundle's
advice, as in executing-keypoint-motion)."""

STEP = 0.03
"""Base camera step [m]: move_to_board_skill's ``CENTER_STEP_M``. A translation
is this times the action's ``step_scale`` (0.25-1, the worst clearance deficit
over the margin); a back-off is the whole step."""

MAX_MOVES = 10
"""Most corrective moves before the skill stops with ``step_limit``. The source's
search has no count limit; a graph node needs one."""

MIN_RANGE = 0.15
"""Nearest the camera approaches the target along its axis [m]: the nearest view
refining-target-from-wrist-closeup uses (a D405 reads nothing under ~7 cm)."""

MAX_RANGE = 0.50
"""Farthest the camera backs off from the target [m]: the top of the D405's ideal
7-50 cm depth range."""

MIN_MOVE = 0.002
"""A correction shorter than this after the clearance clamp is not worth making [m]."""

APPROACH_SLACK = 0.005
"""Added to the approach that the 1/range^2 area law asks for [m], so the next
view lands past min_area_frac rather than just short of it."""


@dataclass(frozen=True)
class MaskReport:
    """Where one target mask sits in its image (analyze_board's MaskReport, the mask parts)."""

    seen: bool
    full: bool
    edges: frozenset[str] = field(default_factory=frozenset)
    area_frac: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    centroid: tuple[float, float] | None = None
    clearance_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    context_pad_px: float = 0.0
    context_ok: bool = False
    detail_ok: bool = False
    center_error: tuple[float, float] = (0.0, 0.0)
    clearance_score: float = 0.0
    detail_score: float = 0.0
    center_score: float = 0.0
    quality_score: float = 0.0


@dataclass(frozen=True)
class SearchAction:
    """The next camera action: ``translate`` in the image plane, ``backoff`` /
    ``approach`` along the optical axis, ``done`` or ``hold``."""

    mode: str
    direction_image: tuple[float, float] = (0.0, 0.0)
    backoff: bool = False
    step_scale: float = 0.0
    reason: str = ""


def analyze_mask(
    mask: np.ndarray,
    margin_px: float = MARGIN_PX,
    min_area_frac: float = MIN_AREA_FRAC,
    context_pad_frac: float = CONTEXT_PAD_FRAC,
    ignore_bottom_frac: float = 0.0,
) -> MaskReport:
    """Which image edges the target's mask comes within the margin of, and by how much."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > (0.5 if m.dtype.kind == "f" else 0)
    height, width = m.shape
    band_y = height
    if ignore_bottom_frac > 0.0:
        band_y = max(1, int(height * (1.0 - ignore_bottom_frac)))
        m[band_y:, :] = False
    labels, count = ndimage.label(m, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return MaskReport(seen=False, full=False)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= max(MIN_BLOB_PX, ISLAND_FRAC * sizes.max())) + 1
    if keep.size == 0:
        return MaskReport(seen=False, full=False)
    m = np.isin(labels, keep)
    ys, xs = np.nonzero(m)
    area = int(xs.size)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # The bottom band (the fingers) is the image's bottom edge.
    clearances = (float(x0), float(width - 1 - x1), float(y0), float(band_y - 1 - y1))
    pad = max(float(margin_px), float(context_pad_frac) * float(max(x1 - x0 + 1, y1 - y0 + 1)))
    edges = frozenset(
        e for e, c in zip(("left", "right", "top", "bottom"), clearances, strict=True) if c <= pad
    )
    area_frac = area / float(height * width)
    context_ok = not edges
    detail_ok = area_frac >= min_area_frac
    cx, cy = float(xs.mean()), float(ys.mean())
    half_w, half_h = max(1.0, 0.5 * (width - 1)), max(1.0, 0.5 * (band_y - 1))
    center_error = ((cx - half_w) / half_w, (cy - half_h) / half_h)
    clearance_score = float(np.clip(min(clearances) / pad, 0.0, 1.0)) if pad > 0.0 else 1.0
    detail_score = float(np.clip(area_frac / min_area_frac, 0.0, 1.0)) if min_area_frac > 0 else 1.0
    center_score = float(np.clip(1.0 - np.hypot(*center_error), 0.0, 1.0))
    # analyze_board's 0.40 / 0.20 / 0.15 weights, renormalised without its 0.25 shape term.
    quality = (0.40 * clearance_score + 0.20 * detail_score + 0.15 * center_score) / 0.75
    return MaskReport(
        seen=True,
        full=bool(context_ok and detail_ok),
        edges=edges,
        area_frac=area_frac,
        bbox=(x0, y0, x1, y1),
        centroid=(cx, cy),
        clearance_px=clearances,
        context_pad_px=pad,
        context_ok=context_ok,
        detail_ok=detail_ok,
        center_error=(float(center_error[0]), float(center_error[1])),
        clearance_score=clearance_score,
        detail_score=detail_score,
        center_score=center_score,
        quality_score=float(np.clip(quality, 0.0, 1.0)),
    )


def view_quality(report: MaskReport) -> float:
    return float(np.clip(report.quality_score, 0.0, 1.0)) if report.seen else 0.0


def search_progress(reports: list[MaskReport], window: int = 4, min_gain: float = 0.01) -> str:
    """complete / improving / regressing / stalled / unknown over the last ``window`` views."""
    if any(r.full for r in reports):
        return "complete"
    recent = list(reports[-window:])
    if len(recent) < 2 or any(not r.seen for r in recent):
        return "unknown"
    gain = view_quality(recent[-1]) - view_quality(recent[0])
    if gain >= min_gain:
        return "improving"
    if gain <= -min_gain:
        return "regressing"
    return "stalled"


def decide_direction(report: MaskReport) -> tuple[np.ndarray, bool]:
    """Camera-plane direction (+x image-right, +y image-down) away from the clipped
    edges, and whether opposite edges are clipped (too close: back off)."""
    if not report.seen or report.full:
        return np.zeros(2), False
    e = report.edges
    backoff = ("left" in e and "right" in e) or ("top" in e and "bottom" in e)
    d = np.array(
        [
            (-1.0 if "left" in e else 0.0) + (1.0 if "right" in e else 0.0),
            (-1.0 if "top" in e else 0.0) + (1.0 if "bottom" in e else 0.0),
        ]
    )
    n = float(np.linalg.norm(d))
    return (d / n if n > 0.0 else d), backoff


def _continuous_direction(report: MaskReport) -> np.ndarray:
    """A unit camera-plane correction from the clearance deficits."""
    pad = float(report.context_pad_px)
    d = np.zeros(2)
    if pad > 0.0:
        left, right, top, bottom = report.clearance_px
        deficits = np.maximum(0.0, pad - np.array([left, right, top, bottom])) / pad
        d = np.array([deficits[1] - deficits[0], deficits[3] - deficits[2]])
    if float(np.linalg.norm(d)) < 1e-9 and report.edges:
        d, _ = decide_direction(report)
    if float(np.linalg.norm(d)) < 1e-9:
        d = np.asarray(report.center_error, dtype=float)
    n = float(np.linalg.norm(d))
    return d / n if n > 0.0 else d


def adaptive_action(report: MaskReport, history: list[MaskReport]) -> SearchAction:
    """The next action from this view and the last three: a persistent clipped edge
    with no gain (or a growing area) means the camera is too close, so back off."""
    if not report.seen:
        return SearchAction(mode="hold", reason="target_not_seen")
    if report.full:
        return SearchAction(mode="done", reason="view_complete")
    direction = _continuous_direction(report)
    _, opposite = decide_direction(report)
    recent = [*history[-3:], report]
    progress = search_progress(recent)
    same_edges = bool(report.edges) and all(r.edges == report.edges for r in recent)
    area_growth = bool(
        len(recent) >= 3
        and recent[0].area_frac > 0.0
        and report.area_frac > 1.08 * recent[0].area_frac
    )
    if opposite or (
        len(recent) >= 3 and same_edges and (progress in {"stalled", "regressing"} or area_growth)
    ):
        reason = "opposite_context_edges" if opposite else "persistent_edge_without_clearance"
        return SearchAction(mode="backoff", backoff=True, step_scale=1.0, reason=reason)
    if not report.detail_ok and report.context_ok:
        return SearchAction(
            mode="approach",
            direction_image=(float(direction[0]), float(direction[1])),
            step_scale=float(np.clip(1.0 - report.detail_score, 0.25, 1.0)),
            reason="insufficient_detail",
        )
    pad = max(report.context_pad_px, 1.0)
    deficit = max((max(0.0, pad - c) / pad for c in report.clearance_px), default=0.0)
    return SearchAction(
        mode="translate",
        direction_image=(float(direction[0]), float(direction[1])),
        step_scale=float(np.clip(max(deficit, 0.25), 0.25, 1.0)),
        reason=f"context_correction_{progress}",
    )


def world_delta(
    direction_image: np.ndarray,
    backoff: bool,
    step_m: float,
    image_right: np.ndarray,
    image_down: np.ndarray,
    back_away: np.ndarray,
) -> np.ndarray:
    """A camera-plane direction (or a back-off) as a world-frame translation."""
    if step_m <= 0.0:
        raise ValueError("step_m must be positive")
    if backoff:
        return step_m * back_away / np.linalg.norm(back_away)
    right = image_right / np.linalg.norm(image_right)
    down = image_down / np.linalg.norm(image_down)
    return step_m * (float(direction_image[0]) * right + float(direction_image[1]) * down)


_AIM: Any = None


def aim_module() -> Any:
    """refining-target-from-wrist-closeup's script (its pose, camera and settle
    helpers), from the skill installed beside this one or under a GAP_SKILLS_PATH root."""
    global _AIM
    if _AIM is not None:
        return _AIM
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (root / AIM_SKILL / AIM_SCRIPT, root / "skills" / AIM_SKILL / AIM_SCRIPT):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("_frame_target_aim", path)
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
    edges: list[str]
    clearance_px: dict[str, float]
    area_frac: float
    center_error: dict[str, float]
    pixel: dict[str, float]
    target_point: dict[str, float]
    depth: float
    moves: int
    travel: float
    view_pose: dict[str, Any]
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


def _pick_mask(
    ctx: NodeContext,
    segmenter: str,
    rgb: np.ndarray,
    query: str,
    hint: tuple[float, float] | None,
    min_score: float,
) -> tuple[np.ndarray | None, float, str]:
    """The best-scoring mask, or the one whose centroid is nearest ``hint``."""
    try:
        found = ctx.tool(segmenter, image=rgb, query=query, max_results=3)
    except Exception as exc:
        return None, 0.0, f"{segmenter} failed: {exc}"
    masks = (found or {}).get("masks") or []
    scores = list((found or {}).get("scores") or [1.0] * len(masks))
    cands = []
    for m, s in zip(masks, scores, strict=False):
        mk = _as_mask(m, rgb.shape[:2])
        if mk is not None and float(s) >= min_score and int(mk.sum()) >= MIN_BLOB_PX:
            cands.append((mk, float(s)))
    if not cands:
        return (
            None,
            0.0,
            f"{segmenter} found no {query!r} scoring {min_score:.2f} in the wrist image",
        )
    if hint is None:
        mk, s = max(cands, key=lambda c: c[1])
    else:

        def dist(c: tuple[np.ndarray, float]) -> float:
            ys, xs = np.nonzero(c[0])
            return float(np.hypot(xs.mean() - hint[0], ys.mean() - hint[1]))

        mk, s = min(cands, key=dist)
    return mk, s, ""


def _mask_points(
    cam: dict[str, Any], mask: np.ndarray, t_cam: np.ndarray, near: float, far: float
) -> tuple[float, np.ndarray | None]:
    """Median wrist depth over the mask inside the camera's trusted range (0 if
    none), and the per-axis median of the masked pixels back-projected to the world."""
    depth = cam.get("depth")
    if depth is None:
        return 0.0, None
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = mask.shape
    rows = (np.arange(d.shape[0]) * h // d.shape[0]).astype(int)
    cols = (np.arange(d.shape[1]) * w // d.shape[1]).astype(int)
    inside = mask[np.ix_(rows, cols)] & np.isfinite(d) & (d >= near) & (d <= far)
    r, c = np.nonzero(inside)
    if r.size == 0:
        return 0.0, None
    z = d[r, c]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    u, v = (c + 0.5) * (w / d.shape[1]) - 0.5, (r + 0.5) * (h / d.shape[0]) - 0.5
    pc = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, np.ones_like(z)])
    world = (t_cam @ pc)[:3]
    return float(np.median(z)), np.median(world, axis=1)


def run(
    ctx: NodeContext,
    object_name: str,
    query: str | None = None,
    segmenter: str = "sam3.segment_text",
    target_point: dict[str, float] | None = None,
    camera: str | None = None,
    margin_px: float = MARGIN_PX,
    context_pad_frac: float = CONTEXT_PAD_FRAC,
    min_area_frac: float = MIN_AREA_FRAC,
    ignore_bottom_frac: float = 0.0,
    min_score: float = SCORE_FLOOR,
    step: float = STEP,
    max_moves: int = MAX_MOVES,
    min_range: float = MIN_RANGE,
    max_range: float = MAX_RANGE,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    reach_tolerance: float | None = None,
) -> Output:
    aim = aim_module()
    text = query or object_name
    reach_tol = float(aim.REACH_TOLERANCE if reach_tolerance is None else reach_tolerance)
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    move: dict[str, Any] = dict(arm)
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    out: Output = {
        "route": "not_seen",
        "edges": [],
        "clearance_px": {},
        "area_frac": 0.0,
        "center_error": {"x": 0.0, "y": 0.0},
        "pixel": {"u": -1.0, "v": -1.0},
        "target_point": {},
        "depth": 0.0,
        "moves": 0,
        "travel": 0.0,
        "view_pose": {},
        "camera_name": "",
        "trace": [],
        "detail": "",
    }
    start: np.ndarray | None = None

    def finish(route: str, detail: str, obs: dict[str, Any] | None = None) -> Output:
        if obs is not None:
            tool = aim._matrix(obs["arms"][int(arm_id)]["ee_pose"])
            out["view_pose"] = {
                "position": aim._vec(tool[:3, 3]),
                "rotation": aim._quat(tool[:3, :3]),
            }
            if start is not None:
                out["travel"] = float(np.linalg.norm(tool[:3, 3] - start))
        out["route"], out["detail"] = route, detail
        logger.info("framing-a-target-by-mask-edge-steering: %s (%s)", route, detail)
        return out

    def go(position: np.ndarray, rotation: dict[str, float]) -> str:
        """Why the move did not arrive, or ""."""
        try:
            why = aim._refusal(
                ctx.tool(
                    "robot.go_to_pose",
                    pose={"position": aim._vec(position), "rotation": rotation},
                    **move,
                )
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        if why:
            return why
        off = float(
            np.linalg.norm(aim._xyz(aim._pose(ctx.tool("robot.get_ee_pose", **arm))) - position)
        )
        return f"the hand ended {off * 1e3:.0f} mm from the step" if off > reach_tol else ""

    obs = aim._settled(ctx, int(arm_id))
    start = aim._matrix(obs["arms"][int(arm_id)]["ee_pose"])[:3, 3].copy()
    history: list[MaskReport] = []
    hint: tuple[float, float] | None = None
    last_good: tuple[np.ndarray, dict[str, float]] | None = None
    base = float(step)
    last_dir: np.ndarray | None = None
    lost = False
    while True:
        tool = aim._matrix(obs["arms"][int(arm_id)]["ee_pose"])
        cam = aim._wrist_camera(obs, tool, camera or out["camera_name"] or None)
        if cam is None:
            names = [c.get("name") for c in obs.get("cameras") or []]
            return finish(
                "not_seen",
                f"no wrist camera with a pose in the observation (cameras: {names})",
                obs,
            )
        out["camera_name"] = str(cam.get("name", ""))
        rgb = np.asarray(cam["rgb"])
        t_cam = aim._matrix(cam["pose"])
        if hint is None and target_point is not None:
            u, v, z = aim._project(cam, np.array([float(target_point[c]) for c in "xyz"]))
            hint = (u, v) if z > 0.0 else None
        mask, score, why = _pick_mask(ctx, segmenter, rgb, text, hint, float(min_score))
        report = (
            analyze_mask(mask, margin_px, min_area_frac, context_pad_frac, ignore_bottom_frac)
            if mask is not None
            else MaskReport(False, False)
        )
        if not report.seen:
            why = why or f"the {text!r} mask lies wholly in the ignored bottom band"
            out["trace"].append(f"lost: {why}")
            if last_good is None:
                return finish("not_seen", why, obs)
            if lost:
                return finish(
                    "not_seen",
                    f"lost the target and did not find it again back at the last view ({why})",
                    obs,
                )
            lost = True
            base *= 0.5
            back = go(*last_good)
            if back:
                return finish(
                    "unreachable", f"lost the target and the move back was refused: {back}", obs
                )
            obs = aim._settled(ctx, int(arm_id))
            continue
        lost = False
        hint = report.centroid
        depth, point = _mask_points(cam, mask, t_cam, aim.MIN_DEPTH, aim.MAX_DEPTH)
        out.update(
            edges=sorted(report.edges),
            clearance_px=dict(
                zip(("left", "right", "top", "bottom"), report.clearance_px, strict=True)
            ),
            area_frac=report.area_frac,
            center_error={"x": report.center_error[0], "y": report.center_error[1]},
            pixel={"u": report.centroid[0], "v": report.centroid[1]},
            depth=depth,
        )
        target_z: float | None = None
        if point is not None:
            out["target_point"], target_z = aim._vec(point), float(point[2])
        action = adaptive_action(report, history)
        history.append(report)
        out["trace"].append(
            f"{action.mode}: edges={','.join(sorted(report.edges)) or 'none'} area={report.area_frac:.4f} "
            f"score={score:.2f} depth={depth:.3f}"
        )
        if action.mode == "done":
            return finish(
                "framed",
                f"the {text!r} mask is inside every edge by >= {report.context_pad_px:.0f} px "
                f"(least {min(report.clearance_px):.0f} px, area {report.area_frac:.4f}) after {out['moves']} moves",
                obs,
            )
        if out["moves"] >= int(max_moves):
            return finish(
                "step_limit",
                f"still clipped at {','.join(sorted(report.edges)) or 'no edge'} (area {report.area_frac:.4f}) after {out['moves']} moves",
                obs,
            )
        right, down, away = t_cam[:3, 0], t_cam[:3, 1], -t_cam[:3, 2]
        direction = np.asarray(action.direction_image, dtype=float)
        if action.mode == "approach":
            length = base * action.step_scale
            if depth > 0.0:
                # The area grows as 1/range^2: close in to where it reaches min_area_frac.
                need = (
                    depth * (1.0 - np.sqrt(report.area_frac / float(min_area_frac)))
                    + APPROACH_SLACK
                )
                length = min(base, max(length, need), depth - float(min_range))
                if length < MIN_MOVE:
                    return finish(
                        "framed",
                        f"the {text!r} mask is inside every edge but small (area {report.area_frac:.4f} < "
                        f"{min_area_frac}); the camera is already {depth:.2f} m from it (nearest {min_range:.2f} m)",
                        obs,
                    )
            delta = -length * away / np.linalg.norm(away)
        elif action.mode == "backoff":
            if depth > 0.0 and depth + base > float(max_range):
                return finish(
                    "unreachable",
                    f"the {text!r} mask spans {','.join(sorted(report.edges))} and the camera is already "
                    f"{depth:.2f} m from it (farthest {max_range:.2f} m)",
                    obs,
                )
            delta = world_delta(direction, True, base, right, down, away)
        else:
            if last_dir is not None and float(direction @ last_dir) < 0.0:
                base *= 0.5
            last_dir = direction
            delta = world_delta(direction, False, base * action.step_scale, right, down, away)
        here = tool[:3, 3]
        rotation = aim._quat(tool[:3, :3])
        last_good = (here.copy(), rotation)
        refused = ""
        for _ in range(2):
            target = here + delta
            if target_z is not None and target[2] < target_z + aim.MIN_TOOL_CLEARANCE:
                target[2] = target_z + aim.MIN_TOOL_CLEARANCE
            if float(np.linalg.norm(target - here)) < MIN_MOVE:
                return finish(
                    "unreachable",
                    f"the {action.mode} step would bring the tool centre within "
                    f"{aim.MIN_TOOL_CLEARANCE * 1e3:.0f} mm of the target",
                    obs,
                )
            refused = go(target, rotation)
            if not refused:
                break
            out["trace"].append(f"refused: {refused}")
            delta, base = 0.5 * delta, 0.5 * base
        out["moves"] += 1
        if refused:
            obs = aim._settled(ctx, int(arm_id))
            return finish(
                "unreachable",
                f"the {action.mode} step was refused at full and half length: {refused}",
                obs,
            )
        obs = aim._settled(ctx, int(arm_id))
