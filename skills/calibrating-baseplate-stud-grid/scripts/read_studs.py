# Stud detection with OpenCV (HoughCircles, connected components; Apache-2.0,
# through cv2) and the lattice fit in stud_lattice.py; see that file's header
# for what is taken from where. New here: the brick's top plane from the masked
# depth, the footprint from the lattice indices of the inlier studs, and a ray
# test at each side of the footprint that tells a brick edge from a top that
# goes on (studs missed) or an occluder in front.
"""Read a brick's top from an RGB-D close-up: its studs, footprint (A x B) and top pose."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stud_lattice as lat  # noqa: E402

logger = logging.getLogger(__name__)

TOP_PLANE_THRESHOLD = 0.0006
"""Inlier band of the brick's top plane [m]: ~1.5 sigma of a D405 at 0.2 m,
under a LEGO stud's 1.7 mm so the studs stay out of the plane."""

MAX_TOP_TILT_DEG = 30.0
"""Largest tilt of the top face from world +z (from the camera's -z without a
camera pose) [deg]: bricks lie on a table or a plate."""

MASK_DILATE_PX = 3
"""Mask growth before the studs are looked for [px]: a segmentation edge clips
the outer studs' rims."""

EDGE_PROBE = 0.0015
"""How far past the footprint's edge (the outer stud centres + half a pitch) a
side is probed [m]: past the 0.1 mm play and a pixel or two of depth blur."""

EDGE_STEP = 0.001
"""Depth difference that decides a probe [m]: deeper than the top by more than
this = the brick ends there; within it = the top goes on; nearer = an occluder."""

ROUTES = ("lattice_found", "too_few_studs", "occluded")


class Output(TypedDict):
    route: str
    detail: str
    brick_size: list[int]
    stud_count: int
    footprint_complete: bool
    open_sides: list[str]
    top_pose: dict[str, Any]
    lattice_pose_camera: dict[str, Any]
    stud_px: list[list[float]]
    stud_points: list[list[float]]
    stud_indices: list[list[int]]
    residual_mm: float
    yaw_ambiguity_deg: float
    system: str
    stud_pitch: float
    camera_name: str


def _frame_from_obs(ctx: NodeContext, camera: str | None, arm_id: int) -> dict[str, Any] | None:
    obs = ctx.tool("robot.get_observation")
    cams = [c for c in obs.get("cameras") or [] if c.get("depth") is not None]
    if camera:
        return next((c for c in cams if c.get("name") == camera), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    arms = obs.get("arms") or []
    if (
        wrist
        and len(arms) > arm_id
        and arms[arm_id].get("ee_pose") is not None
        and wrist[0].get("pose")
    ):
        tool = lat.xyz(arms[arm_id]["ee_pose"])
        return min(wrist, key=lambda c: float(np.linalg.norm(lat.xyz(c["pose"]) - tool)))
    return wrist[0] if wrist else None


def _side_states(
    depth: np.ndarray,
    k: np.ndarray,
    t_cam_b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    pitch: float,
) -> dict[str, str]:
    """For each footprint side: 'edge' (the depth falls past it), 'continues',
    'occluded' (something nearer) or 'unseen' (outside the image / no depth)."""
    h, w = depth.shape
    out = {}
    edge = pitch / 2.0 + EDGE_PROBE
    sides = {
        "-x": ([lo[0] - edge], np.linspace(lo[1], hi[1], 7)),
        "+x": ([hi[0] + edge], np.linspace(lo[1], hi[1], 7)),
        "-y": (np.linspace(lo[0], hi[0], 7), [lo[1] - edge]),
        "+y": (np.linspace(lo[0], hi[0], 7), [hi[1] + edge]),
    }
    for name, (xs, ys) in sides.items():
        gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
        p = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], axis=1)
        pc = (p @ t_cam_b.T)[:, :3]
        votes = {"edge": 0, "continues": 0, "occluded": 0, "unseen": 0}
        for x, y, z in pc:
            if z <= 1e-6:
                votes["unseen"] += 1
                continue
            u, v = k[0, 0] * x / z + k[0, 2], k[1, 1] * y / z + k[1, 2]
            c, r = int(round(u)), int(round(v))
            if not (0 <= c < w and 0 <= r < h):
                votes["unseen"] += 1
                continue
            win = depth[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
            win = win[np.isfinite(win) & (win > 0)]
            if win.size == 0:
                votes["unseen"] += 1
                continue
            dz = float(np.median(win)) - z
            votes["edge" if dz > EDGE_STEP else "continues" if dz > -EDGE_STEP else "occluded"] += 1
        out[name] = max(votes, key=lambda key: (votes[key], key == "edge"))
    return out


def run(
    ctx: NodeContext,
    mask: Any = None,
    frame: dict[str, Any] | None = None,
    camera: str | None = None,
    system: str = "lego",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    detector: str = "both",
    max_residual: float | None = None,
    arm_id: int = 0,
    seed: int = 0,
) -> Output:
    if system not in lat.PRESETS:
        raise ValueError(f"system must be one of {sorted(lat.PRESETS)}, got {system!r}")
    if detector not in ("both", "depth", "rgb"):
        raise ValueError(f"detector must be 'both', 'depth' or 'rgb', got {detector!r}")
    preset = lat.PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_r = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"]) / 2.0
    max_res = float(max_residual if max_residual is not None else 0.125 * pitch)
    rng = np.random.default_rng(seed)
    out: Output = {
        "route": "too_few_studs",
        "detail": "",
        "brick_size": [0, 0],
        "stud_count": 0,
        "footprint_complete": False,
        "open_sides": [],
        "top_pose": {},
        "lattice_pose_camera": {},
        "stud_px": [],
        "stud_points": [],
        "stud_indices": [],
        "residual_mm": 0.0,
        "yaw_ambiguity_deg": 0.0,
        "system": system,
        "stud_pitch": pitch,
        "camera_name": "",
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("reading-brick-studs: %s (%s)", route, why)
        return out

    cam = frame if frame is not None else _frame_from_obs(ctx, camera, int(arm_id))
    if cam is None or cam.get("depth") is None:
        return finish(
            "too_few_studs", "no RGB-D frame (no wrist camera, or the named one, with depth)"
        )
    out["camera_name"] = str(cam.get("name", ""))
    depth = lat.as_depth(cam["depth"])
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    posed = cam.get("pose") is not None
    t_world_cam = lat.matrix(cam["pose"]) if posed else np.eye(4)
    m = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != depth.shape:
            m = cv2.resize(m.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
        m = m > 0
        if not m.any():
            return finish("too_few_studs", "the mask is empty")

    # The top face: the highest well-supported up-facing plane in the mask (the
    # dominant one in the whole frame without a mask).
    pts, _ = lat.backproject(depth, k, t_world_cam, mask=m)
    up = np.array([0.0, 0.0, 1.0]) if posed else -t_world_cam[:3, 2]
    if m is not None:
        fit = lat.stepped_plane(
            pts,
            TOP_PLANE_THRESHOLD,
            rng,
            drop=2.0 * stud_h,
            direction=1,
            up=up,
            max_tilt_deg=MAX_TOP_TILT_DEG,
        )
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * MASK_DILATE_PX + 1,) * 2, np.uint8)) > 0
    else:
        fit = lat.fit_plane(pts, TOP_PLANE_THRESHOLD, rng, up=up, max_tilt_deg=MAX_TOP_TILT_DEG)
    if fit is None:
        return finish(
            "too_few_studs",
            f"no flat top in the {'mask' if m is not None else 'frame'} ({len(pts)} depth points)",
        )
    n, d, inl = fit
    x_hint = np.array([1.0, 0.0, 0.0]) if posed else t_world_cam[:3, 0]
    t_w_b = lat.plane_frame(n, d, x_hint, pts[inl].mean(axis=0))
    t_b_cam = np.linalg.inv(t_w_b) @ t_world_cam
    top_b = (pts[inl] - t_w_b[:3, 3]) @ t_w_b[:3, :3]
    lo = np.percentile(top_b[:, :2], 0.5, axis=0) - pitch
    hi = np.percentile(top_b[:, :2], 99.5, axis=0) + pitch
    raster = lat.Raster(lo[0], lo[1], hi[0], hi[1], pitch / lat.RECT_PX_PER_PITCH)
    found = lat.detect_and_fit(
        cam.get("rgb"), depth, k, t_b_cam, raster, stud_h, stud_r, pitch, detector, region_cam=m
    )
    q = found["xy"]
    fitted = lat.fit_lattice(q, pitch, rng) if len(q) >= 2 else None
    if fitted is not None and fitted["n_inliers"] >= 2:
        inl_s = fitted["inliers"]
        yaw, t = fitted["yaw"], fitted["t"]
        idx = fitted["indices"][inl_s]
        out["residual_mm"] = fitted["rms"] * 1e3
    elif len(q) >= 1 and m is not None:
        # One stud: the yaw from the top face's outline.
        (_, _), (_, _), angle = cv2.minAreaRect(top_b[:, :2].astype(np.float32))
        yaw = lat.wrap(math.radians(angle), math.pi / 2.0)
        r = lat.rz(yaw)
        idx = np.zeros((1, 2), int)
        t = q[0] - r @ np.zeros(2)
        inl_s = np.zeros(len(q), bool)
        inl_s[0] = True
    else:
        return finish(
            "too_few_studs",
            f"{len(q)} stud candidates ({found['n_depth']} depth, {found['n_rgb']} RGB) on the top",
        )
    if out["residual_mm"] > max_res * 1e3:
        return finish(
            "too_few_studs",
            f"the studs fit a {pitch * 1e3:.0f} mm lattice at {out['residual_mm']:.2f} mm RMS (over {max_res * 1e3:.1f})",
        )
    imin, jmin = idx.min(axis=0)
    imax, jmax = idx.max(axis=0)
    ni, nj = int(imax - imin + 1), int(jmax - jmin + 1)
    # Lattice frame in the plane: x along the short side.
    r = lat.rz(yaw)
    lo_l = np.array([imin, jmin], float) * pitch
    hi_l = np.array([imax, jmax], float) * pitch
    if ni > nj:
        yaw += math.pi / 2.0
        r = lat.rz(yaw)
        # (i, j) -> (j, -i) in the turned frame.
        idx = np.stack([idx[:, 1], -idx[:, 0]], axis=1)
        imin, jmin = idx.min(axis=0)
        imax, jmax = idx.max(axis=0)
        ni, nj = nj, ni
        lo_l = np.array([imin, jmin], float) * pitch
        hi_l = np.array([imax, jmax], float) * pitch
    centre_b = r @ (0.5 * (lo_l + hi_l)) + t
    t_b_top = np.eye(4)
    t_b_top[:2, :2] = r
    t_b_top[:2, 3] = centre_b
    t_w_top = t_w_b @ t_b_top
    half = 0.5 * (hi_l - lo_l)
    out["brick_size"] = [ni, nj]
    out["stud_count"] = int(len(idx))
    out["top_pose"] = lat.se3(t_w_top)
    out["lattice_pose_camera"] = lat.se3(np.linalg.inv(t_world_cam) @ t_w_top)
    out["yaw_ambiguity_deg"] = 90.0 if ni == nj else 180.0
    centres_l = idx * pitch - 0.5 * (lo_l + hi_l)
    stud_w = (np.c_[centres_l, np.full(len(idx), stud_h), np.ones(len(idx))] @ t_w_top.T)[:, :3]
    pc = (np.c_[stud_w, np.ones(len(stud_w))] @ np.linalg.inv(t_world_cam).T)[:, :3]
    out["stud_points"] = stud_w.tolist()
    out["stud_px"] = np.stack(
        [k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2], k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]], 1
    ).tolist()
    out["stud_indices"] = (idx - [imin, jmin]).astype(int).tolist()
    complete = len(idx) == ni * nj
    where = f"{ni}x{nj} footprint, {len(idx)} studs ({out['residual_mm']:.2f} mm RMS)"
    if m is None:
        out["footprint_complete"] = complete
        return finish(
            "lattice_found", f"{where}; no mask, so the studs seen, not a brick's outline"
        )
    t_cam_top = np.linalg.inv(t_world_cam) @ t_w_top
    states = _side_states(depth, k, t_cam_top, -half, half, pitch)
    open_sides = sorted(s for s, v in states.items() if v != "edge")
    out["open_sides"] = open_sides
    out["footprint_complete"] = complete and not open_sides
    if not complete:
        return finish(
            "occluded", f"{where}; {ni * nj - len(idx)} studs missing inside the footprint"
        )
    if open_sides:
        desc = ", ".join(f"{s} {states[s]}" for s in open_sides)
        return finish(
            "occluded",
            f"{where}; the top does not end at side(s) {desc}: the footprint is a lower bound",
        )
    return finish("lattice_found", where)
