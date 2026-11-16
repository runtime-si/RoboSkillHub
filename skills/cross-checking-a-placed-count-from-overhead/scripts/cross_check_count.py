"""Cross-check a count of parts placed into a container against a rough overhead area estimate."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

CELL = 0.002
"""Plan-view cell [m], in the container's frame."""

MARGIN = 0.015
"""The map starts this far inside the container's outer walls [m]. tuned_on: a 238 x 196 mm
cardboard box with a 66 mm rim under one YAM station's overhead depth camera (2026-11-15)."""

FLOOR_CLEAR = 0.012
"""A cell this far over the container's support is occupied [m]. tuned_on: same; the depth's noise
there hides anything lower (a bracket lying flat is 6 mm)."""

DARK_BELOW, TINT_SLOPE, TINT_OFFSET = 0.6, 0.12, 4.0
"""dark_parts: a cell darker than 0.6 of the floor's median brightness is occupied, unless it is
floor-tinted (red over blue by more than 0.12 x brightness + 4, 8-bit): a shadow on brown cardboard
is dark but still brown. tuned_on: grey printed parts on that cardboard; before the tint test the
arm's shadow in the empty box counted as two parts. Meaningless on a floor that is not reddish."""

OPEN = 5
"""Occupied cells are opened by 5 cells (1 cm): a part is centimetres across."""

MIN_BLOB_CELLS = 300
"""A blob under 12 cm2 is not counted as a part. tuned_on: 5-9 cm parts; chosen."""

MIN_SEEN = 0.5
"""Under this share of the container's inside seen (a hand over it, a view that misses it), no
estimate is offered. Chosen here, on no robot."""


class Output(TypedDict):
    route: str
    detail: str
    estimate: int
    blobs: int
    by_area: int
    area_m2: float
    claimed_count: int
    seen_fraction: float
    floor_brightness: float
    camera: str


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    if isinstance(pose, dict):
        m = np.eye(4)
        m[:3, :3] = _rot(pose.get("rotation") or {"w": 1, "x": 0, "y": 0, "z": 0})
        p = pose["position"]
        m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
        return m
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4):
        raise ValueError("camera pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return m


def _camera(ctx: NodeContext, frame: dict | None, name: str | None) -> dict | None:
    """The named posed depth camera; unnamed, the first that is not on a wrist."""
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("depth") is not None and c.get("pose") is not None]
    if name:
        return next((c for c in cams if str(c.get("name")) == name), None)
    fixed = [
        c for c in cams if not any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))
    ]
    return fixed[0] if fixed else (cams[0] if cams else None)


def footprint(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """An upright box's (centre xy, half extents along its own x / y, yaw [rad])."""
    c = obb["center"]
    e = obb["extent"]
    r = _rot(obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0})
    return (
        np.array([float(c["x"]), float(c["y"])]),
        np.array([float(e["x"]), float(e["y"])]),
        math.atan2(r[1, 0], r[0, 0]),
    )


def plan_view(cam: dict, container_obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The container's inside as a grid in its frame: the highest point per cell (nan where
    nothing was seen) over its support, and that point's colour."""
    centre, half, yaw = footprint(container_obb)
    support = float(container_obb["center"]["z"]) - float(container_obb["extent"]["z"])
    rim = float(container_obb["center"]["z"]) + float(container_obb["extent"]["z"])
    x0, y0 = -half[0] + MARGIN, -half[1] + MARGIN
    nx = max(1, int((2 * half[0] - 2 * MARGIN) / CELL))
    ny = max(1, int((2 * half[1] - 2 * MARGIN) / CELL))
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t = _matrix(cam["pose"])
    vv, uu = np.nonzero(np.isfinite(depth) & (depth > 0))
    zc = depth[vv, uu]
    pc = np.column_stack([(uu - k[0, 2]) * zc / k[0, 0], (vv - k[1, 2]) * zc / k[1, 1], zc])
    pts = pc @ t[:3, :3].T + t[:3, 3]
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts[:, :2] - centre
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    ok = (lx > x0) & (lx < x0 + nx * CELL) & (ly > y0) & (ly < y0 + ny * CELL)
    ok &= (pts[:, 2] > support - 0.01) & (pts[:, 2] < rim + 0.12)
    r = ((ly[ok] - y0) / CELL).astype(int).clip(0, ny - 1)
    col = ((lx[ok] - x0) / CELL).astype(int).clip(0, nx - 1)
    z = pts[ok, 2] - support
    rgb_px = np.asarray(cam["rgb"])[vv[ok], uu[ok]].astype(np.float32)
    order = np.argsort(z)  # the highest written last
    height = np.full((ny, nx), np.nan, np.float32)
    rgb = np.zeros((ny, nx, 3), np.float32)
    height[r[order], col[order]] = z[order]
    rgb[r[order], col[order]] = rgb_px[order]
    return height, rgb


def estimate_count(
    height: np.ndarray, rgb: np.ndarray, part_area: float, dark_parts: bool = False
) -> dict[str, Any]:
    """A rough count of what lies in the plan view: {estimate, blobs, by_area, area_m2,
    floor_brightness}.

    Occupied: standing FLOOR_CLEAR over the floor, or (dark_parts) darker than the floor and not
    floor-tinted. Parts that touch are one blob, so the estimate is the larger of the blob count
    and the occupied area over one part's ``part_area``; with no blob it is 0. Stacked parts read
    low, touching parts and shadows read high.
    """
    import cv2

    known = np.isfinite(height)
    lum = rgb.mean(-1)
    floor = float(np.median(lum[known])) if known.sum() > 100 else 255.0
    there = known & (np.nan_to_num(height) > FLOOR_CLEAR)
    if dark_parts:
        tint = rgb[..., 0] - rgb[..., 2]
        there |= (
            known
            & (lum < DARK_BELOW * floor)
            & (tint < TINT_SLOPE * np.maximum(lum, 1.0) + TINT_OFFSET)
        )
    up = cv2.morphologyEx(there.astype(np.uint8), cv2.MORPH_OPEN, np.ones((OPEN, OPEN), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(up, connectivity=8)
    blobs = int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_CELLS))
    area = float(up.sum()) * CELL * CELL
    by_area = int(area / part_area + 0.5)
    return {
        "estimate": max(blobs, by_area) if blobs else 0,
        "blobs": blobs,
        "by_area": by_area,
        "area_m2": area,
        "floor_brightness": floor,
    }


def verdict(estimate: int, claimed: int) -> str:
    """ "supported" when the camera sees at least half of what is claimed (both empty counts too),
    "contradicted" when it sees less than half, or something where nothing is claimed,
    "uncertain" with no claim."""
    if claimed < 0:
        return "uncertain"
    if claimed == 0:
        return "supported" if estimate == 0 else "contradicted"
    return "supported" if 2 * estimate >= claimed else "contradicted"


def run(
    ctx: NodeContext,
    container_obb: dict,
    part_area: float,
    claimed_count: int = -1,
    camera: str = "",
    camera_frame: dict | None = None,
    dark_parts: bool = False,
) -> Output:
    """Does the overhead view contradict the count of parts said to be in the container?

    Args:
        container_obb: The container as this camera has it (upright; half extents of its outer
            size; its top the rim).
        part_area: The plan-view area one part covers lying in the container [m2], measured on
            this camera (count the occupied area with a known number of parts in).
        claimed_count: An independent count to check: how many times the hand let go while
            holding something over the container, an operator's count. -1: none.
        camera / camera_frame: The overhead camera's name (default: the first posed depth camera
            that is not on a wrist), or a CameraFrame {rgb, depth, intrinsics, pose}.
        dark_parts: Also count cells darker than the floor and not floor-tinted (parts flatter
            than the depth noise on reddish cardboard).

    Returns:
        ``route`` "supported", "contradicted" or "uncertain". ``estimate`` is rough in every
        case and is never the count.
    """
    out: Output = {
        "route": "uncertain",
        "detail": "",
        "estimate": -1,
        "blobs": 0,
        "by_area": 0,
        "area_m2": 0.0,
        "claimed_count": int(claimed_count),
        "seen_fraction": 0.0,
        "floor_brightness": 0.0,
        "camera": "",
    }
    if float(part_area) <= 0:
        raise ValueError("part_area must be the plan-view area of one part [m2]")
    cam = _camera(ctx, camera_frame, camera or None)
    if cam is None or cam.get("rgb") is None:
        out["detail"] = f"no posed RGB-D camera {camera or '(any fixed one)'!r} in the observation"
        return out
    out["camera"] = str(cam.get("name", ""))
    height, rgb = plan_view(cam, container_obb)
    out["seen_fraction"] = float(np.isfinite(height).mean())
    if out["seen_fraction"] < MIN_SEEN:
        out["detail"] = (
            f"{out['seen_fraction']:.0%} of the container's inside is in view (need 50 %): "
            "something is over it, or the view misses it"
        )
        return out
    found = estimate_count(height, rgb, float(part_area), bool(dark_parts))
    out["estimate"], out["blobs"], out["by_area"] = (
        found["estimate"],
        found["blobs"],
        found["by_area"],
    )
    out["area_m2"], out["floor_brightness"] = found["area_m2"], found["floor_brightness"]
    out["route"] = verdict(found["estimate"], int(claimed_count))
    seen = (
        f"the view suggests about {found['estimate']} ({found['blobs']} blob(s), "
        f"{found['area_m2'] * 1e4:.0f} cm2)"
    )
    if out["route"] == "uncertain":
        out["detail"] = f"{seen}; no independent count was given to check"
    elif out["route"] == "supported":
        out["detail"] = f"{seen}, not under half of the {int(claimed_count)} claimed"
    else:
        out["detail"] = f"{seen} against {int(claimed_count)} claimed"
    return out
