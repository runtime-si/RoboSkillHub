# Reimplements the method of "Slip Detection of Grasped Objects Using Optical Flow" style wrist-camera slip
# detection (Applied Sciences 13(15):8620, 2023, https://www.mdpi.com/2076-3417/13/15/8620): the held object's
# region in a camera rigid with the fingers, dense Farneback optical flow against the frame taken right after the
# grasp, and thresholds on the flow. No code was published with the paper; OpenCV's calcOpticalFlowFarneback
# (Apache-2.0) does the flow. Changes: the object region comes from the wrist depth (pixels within a near band in
# front of the camera, optionally less a finger mask) instead of background removal; the region's motion is a
# RANSAC similarity fit to the flow (the fingers, still in the image, do not drag a mean down), converted to
# millimetres with the depth; a twist about the jaws' closing axis, which an image cannot show, is read from the
# change of the region's depth-plane normal about that axis; and a shrinking region reads as a drop.
"""Watch the held part in the wrist camera and say whether it slips, twists or drops in the jaws."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

NEAR, FAR = 0.02, 0.20
"""Depth band [m] in front of the wrist camera that holds the part: a D405 sees the closed fingertips at
about 0.07-0.12 m on a YAM; parts held between them fall inside this band, the table and scene beyond it."""

SLIP_MM = 2.0
"""Translation of the part relative to the camera (= the fingers) that reads as slipping [mm]."""

TWIST_DEG = 3.0
"""Rotation of the part, about the jaws' closing axis (from the depth plane) or about the camera axis (from the
flow), that reads as twisted [deg]. Over the 0.7-2 deg a station twin's 30-100 g parts settle by when lifted."""

DROP_FRACTION = 0.4
"""The part's region shrinking under this fraction of its first frame's reads as dropped."""

FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}
"""calcOpticalFlowFarneback parameters: OpenCV's documented defaults, with a larger window and more levels so a
few millimetres of motion at 0.1 m (tens of pixels) is caught."""

MIN_PIXELS = 200
"""Fewest region pixels the fits use."""


class Output(TypedDict):
    route: str
    detail: str
    slip_mm: float
    twist_deg: float
    spin_deg: float
    area_fraction: float
    frames: int


def _camera(obs: Any, name: str | None) -> dict[str, Any] | None:
    cams = [
        c
        for c in (obs.get("cameras") if isinstance(obs, dict) else None) or []
        if isinstance(c, dict)
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    return wrist[0] if wrist else None


def _gray(cam: dict[str, Any]) -> np.ndarray:
    img = np.asarray(cam.get("rgb", cam.get("color", cam.get("image"))))
    if img.ndim == 3:
        img = img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return np.clip(img, 0, 255).astype(np.uint8)


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def region(
    depth: np.ndarray, near: float, far: float, exclude: np.ndarray | None = None
) -> np.ndarray:
    m = np.isfinite(depth) & (depth > near) & (depth < far)
    if exclude is not None:
        m &= ~exclude.astype(bool)
    return m


def plane_normal(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray | None:
    """Unit normal (camera frame, facing the camera) of the plane fitted to the region's depth points."""
    v, u = np.nonzero(mask)
    if v.size < MIN_PIXELS:
        return None
    z = depth[v, u]
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    pts = np.stack([x, y, z], axis=1)
    c = pts - pts.mean(axis=0)
    n = np.linalg.svd(c, full_matrices=False)[2][-1]
    return -n if n[2] > 0 else n


def angle_about(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle [deg] from a to b about ``axis``, both projected onto the plane normal to it."""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    pa = a - axis * float(np.dot(a, axis))
    pb = b - axis * float(np.dot(b, axis))
    if np.linalg.norm(pa) < 1e-9 or np.linalg.norm(pb) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(float(np.dot(np.cross(pa, pb), axis)), float(np.dot(pa, pb))))


def flow_motion(
    ref: np.ndarray, now: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """(translation [px], rotation about the image centre of the region [deg]) of the region's dominant motion:
    a RANSAC similarity fit to the Farneback flow sampled inside the mask."""
    import cv2

    flow = cv2.calcOpticalFlowFarneback(ref, now, None, **FARNEBACK)
    v, u = np.nonzero(mask)
    if v.size < MIN_PIXELS:
        return None
    step = max(1, v.size // 4000)
    v, u = v[::step], u[::step]
    src = np.stack([u, v], axis=1).astype(np.float32)
    dst = src + flow[v, u].astype(np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if m is None:
        return None
    centre = src.mean(axis=0)
    moved = m[:, :2] @ centre + m[:, 2]
    return moved - centre, math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))


def compare(
    ref: dict[str, Any],
    now: dict[str, Any],
    closing_axis_cam: np.ndarray,
    near: float,
    far: float,
    exclude=None,
) -> dict[str, float]:
    """Slip [mm], twist about the closing axis [deg], spin about the camera axis [deg] and the region's area
    fraction between two wrist frames (dicts with rgb, depth, intrinsics)."""
    k = np.asarray(ref["intrinsics"], dtype=float)
    d0, d1 = np.asarray(ref["depth"], float), np.asarray(now["depth"], float)
    m0, m1 = region(d0, near, far, exclude), region(d1, near, far, exclude)
    res = {
        "slip_mm": 0.0,
        "twist_deg": 0.0,
        "spin_deg": 0.0,
        "area_fraction": float(m1.sum()) / max(1, m0.sum()),
    }
    mot = flow_motion(_gray(ref), _gray(now), m0)
    if mot is not None:
        t_px, spin = mot
        z = float(np.median(d0[m0])) if m0.any() else 0.1
        res["slip_mm"] = float(np.linalg.norm(t_px)) * z / float(k[0, 0]) * 1e3
        res["spin_deg"] = spin
    n0, n1 = plane_normal(d0, m0, k), plane_normal(d1, m1, k)
    if n0 is not None and n1 is not None:
        res["twist_deg"] = angle_about(n0, n1, closing_axis_cam)
    return res


def run(
    ctx: NodeContext,
    duration_s: float = 2.0,
    period_s: float = 0.2,
    camera: str | None = None,
    near: float = NEAR,
    far: float = FAR,
    slip_mm: float = SLIP_MM,
    twist_deg: float = TWIST_DEG,
    drop_fraction: float = DROP_FRACTION,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "no_camera",
        "detail": "",
        "slip_mm": 0.0,
        "twist_deg": 0.0,
        "spin_deg": 0.0,
        "area_fraction": 1.0,
        "frames": 0,
    }

    def frame() -> dict[str, Any] | None:
        cam = _camera(ctx.tool("robot.get_observation"), camera)
        if cam is None or cam.get("depth") is None or cam.get("intrinsics") is None:
            return None
        if cam.get("rgb", cam.get("color", cam.get("image"))) is None:
            return None
        return cam

    ref = frame()
    if ref is None:
        out["detail"] = "no wrist camera with rgb, depth and intrinsics in the observation"
        return out
    # The jaws' closing axis (the tool frame's y) in the camera frame.
    axis_cam = np.array([1.0, 0.0, 0.0])
    try:
        pose = ctx.tool("robot.get_ee_pose", **on_arm)
        pose = pose.get("pose", pose)
        axis_world = _rot(pose["rotation"])[:, 1]
        if ref.get("pose") is not None:
            axis_cam = _rot(ref["pose"]["rotation"]).T @ axis_world
    except Exception as exc:  # no pose: assume the closing axis runs along the image's x
        logger.info("in-hand slip: no tool or camera pose (%s)", exc)
    m0 = region(np.asarray(ref["depth"], float), near, far)
    if m0.sum() < MIN_PIXELS:
        out["route"] = "dropped"
        out["detail"] = (
            f"nothing held in the wrist camera's {near * 1e3:.0f}-{far * 1e3:.0f} mm band at the start"
        )
        out["area_fraction"] = 0.0
        return out
    worst = {"slip_mm": 0.0, "twist_deg": 0.0, "spin_deg": 0.0, "area_fraction": 1.0}
    t0 = time.monotonic()
    n = max(1, int(round(float(duration_s) / max(float(period_s), 1e-3))))
    for i in range(n):
        if period_s > 0:
            while time.monotonic() - t0 < (i + 1) * float(period_s):
                time.sleep(0.01)
        cur = frame()
        if cur is None:
            continue
        out["frames"] += 1
        r = compare(ref, cur, axis_cam, near, far)
        worst["slip_mm"] = max(worst["slip_mm"], r["slip_mm"])
        for key in ("twist_deg", "spin_deg"):
            if abs(r[key]) > abs(worst[key]):
                worst[key] = r[key]
        worst["area_fraction"] = min(worst["area_fraction"], r["area_fraction"])
        if r["area_fraction"] < drop_fraction:
            break
    out.update(worst)
    summary = (
        f"over {out['frames']} frame(s): slip {out['slip_mm']:.1f} mm, twist {out['twist_deg']:+.1f} deg about the "
        f"closing axis, spin {out['spin_deg']:+.1f} deg, part region {out['area_fraction'] * 100:.0f} % of the first"
    )
    if out["area_fraction"] < drop_fraction:
        out["route"], out["detail"] = (
            "dropped",
            "the part left the wrist camera's near band: " + summary,
        )
    elif abs(out["twist_deg"]) > twist_deg or abs(out["spin_deg"]) > twist_deg:
        out["route"], out["detail"] = "twisted", summary
    elif out["slip_mm"] > slip_mm:
        out["route"], out["detail"] = "slipping", summary
    else:
        out["route"], out["detail"] = "stable", summary
    return out
