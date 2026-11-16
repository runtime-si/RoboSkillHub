"""Find an open box of known size by sliding its outline over the rim points an overhead depth frame sees."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

CELL = 0.004
"""Plan-view grid the outline is slid over [m]. tuned_on: one YAM station's overhead depth camera
over a counter (the median of 3 frames); coarser than its pixels there, so a seen rim leaves no
gaps along its length."""

BAND_BELOW, BAND_ABOVE = 0.03, 0.02
"""Rim points are those from 3 cm under to 2 cm over the stated rim height [m]. tuned_on: a 154 mm
tall cardboard box on one YAM station (real runs of 2026-11-15); the same station's 66 mm box
used (0.02, 0.03): pass band_below / band_above. Neither was swept."""

MIN_POINTS = 150
"""Fewer rim-band points than this and nothing is fitted. Chosen on the same camera, not swept."""

MIN_COVERAGE = 0.35
"""At least this share of the outline must lie on rim points. tuned_on: the same station and
boxes (walls end-on to the camera hardly show, so a whole rim is never seen). Chosen, not swept;
the fit's position was never compared with a ground truth there."""

YAW_STEP_DEG = 2.0
"""Outline turns tried, over 180 degrees."""

EXCLUDE_MARGIN = 0.02
"""Points this far outside another, already found box's footprint are still that box's [m]."""

NEAR_SQUARE = 0.20
"""A box whose sides differ by less than this share fits nearly as well a quarter turn round: its
yaw is then flagged. On the station a 238 x 196 mm box read -38 and -54 degrees in two frames of
one afternoon (2026-11-15)."""


class Output(TypedDict):
    route: str
    detail: str
    container_obb: dict[str, Any] | None
    center: dict[str, float] | None
    yaw_deg: float
    coverage: float
    rim_points: int
    yaw_ambiguous: bool
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


def world_points(cam: dict, stride: int = 1) -> np.ndarray:
    """The frame's valid depth pixels as (N, 3) world points."""
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t = _matrix(cam["pose"])
    sub = depth[::stride, ::stride]
    vv, uu = np.nonzero(np.isfinite(sub) & (sub > 0))
    z = sub[vv, uu]
    pc = np.column_stack(
        [(uu * stride - k[0, 2]) * z / k[0, 0], (vv * stride - k[1, 2]) * z / k[1, 1], z]
    )
    return pc @ t[:3, :3].T + t[:3, 3]


def fit_known_rectangle(
    xy: np.ndarray,
    size: tuple[float, float],
    cell: float = CELL,
    min_coverage: float = MIN_COVERAGE,
    yaws_deg: Any = None,
) -> dict[str, Any] | None:
    """Where a ``size`` (long, short) rectangle's outline lies along the plan-view points ``xy``.

    For each turn of the rectangle its outline (2 cells thick) is slid over the points' occupancy
    grid (grown one cell), and the place and turn with the largest share of the outline on occupied
    cells wins. A rim seen in pieces, or beside something as tall as it, is still placed: the size
    is given, not measured. Returns {center (2,), yaw_deg in [-90, 90), coverage} or None when
    there are under MIN_POINTS points or the best coverage is under ``min_coverage``.
    """
    import cv2

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < MIN_POINTS:
        return None
    long_side, short_side = float(size[0]), float(size[1])
    half = int(math.hypot(long_side, short_side) / 2 / cell) + 3
    lo = xy.min(0) - (half + 1) * cell
    n = np.ceil((xy.max(0) - lo) / cell).astype(int) + half + 2
    img = np.zeros((n[1], n[0]), np.float32)
    ij = ((xy - lo) / cell).astype(int)
    img[ij[:, 1], ij[:, 0]] = 1.0
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    yaws = np.arange(-90.0, 90.0, YAW_STEP_DEG) if yaws_deg is None else np.asarray(yaws_deg, float)
    corners0 = np.array(
        [
            [long_side / 2, short_side / 2],
            [-long_side / 2, short_side / 2],
            [-long_side / 2, -short_side / 2],
            [long_side / 2, -short_side / 2],
        ]
    )
    best: tuple[float, Any] = (0.0, None)
    for yaw in np.radians(yaws):
        c, s = math.cos(yaw), math.sin(yaw)
        kernel = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)
        corners = corners0 @ np.array([[c, s], [-s, c]])
        cv2.polylines(
            kernel, [np.round(corners / cell + half).astype(np.int32)], True, 1, thickness=2
        )
        score = cv2.filter2D(
            img, -1, kernel.astype(np.float32) / kernel.sum(), borderType=cv2.BORDER_CONSTANT
        )
        _, top, _, at = cv2.minMaxLoc(score)
        if top > best[0]:
            best = (float(top), (at, float(yaw)))
    if best[1] is None or best[0] < min_coverage:
        return None
    (ix, iy), yaw = best[1]
    yaw_deg = (math.degrees(yaw) + 90.0) % 180.0 - 90.0
    return {
        "center": lo + (np.array([ix, iy]) + 0.5) * cell,
        "yaw_deg": float(yaw_deg),
        "coverage": float(best[0]),
    }


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


def outside(xy: np.ndarray, obb: dict[str, Any], margin: float) -> np.ndarray:
    """Which plan-view points lie more than ``margin`` outside ``obb``'s footprint."""
    centre, half, yaw = footprint(obb)
    d = xy - centre
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (np.abs(lx) > half[0] + margin) | (np.abs(ly) > half[1] + margin)


def upright_obb(
    center_xy: np.ndarray, size: tuple[float, float], yaw_deg: float, rim_z: float, support_z: float
) -> dict[str, Any]:
    """The hub's OrientedBoundingBox (half extents; local x the long side) for an upright box."""
    h = math.radians(yaw_deg) / 2
    return {
        "center": {
            "x": float(center_xy[0]),
            "y": float(center_xy[1]),
            "z": (rim_z + support_z) / 2,
        },
        "extent": {"x": size[0] / 2, "y": size[1] / 2, "z": max(rim_z - support_z, 0.0) / 2},
        "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
    }


def run(
    ctx: NodeContext,
    size_long: float,
    size_short: float,
    rim_z: float,
    camera: str = "",
    camera_frame: dict | None = None,
    exclude_obbs: list | None = None,
    roi: dict | None = None,
    support_z: float = 0.0,
    band_below: float = BAND_BELOW,
    band_above: float = BAND_ABOVE,
    min_coverage: float = MIN_COVERAGE,
    stride: int = 1,
) -> Output:
    """One open box of known outer size, from one overhead depth frame.

    Args:
        size_long, size_short: The box's outer rim, long and short side [m].
        rim_z: Its rim's height [m, world].
        camera / camera_frame: The observation camera to read (default: the first posed depth
            camera that is not on a wrist), or a CameraFrame {depth, intrinsics, pose}.
        exclude_obbs: Boxes already found; their footprints (plus 2 cm) are not this box's rim.
            Find the taller box first and pass it here for the lower one beside it.
        roi: {x_min, x_max, y_min, y_max} [m, world] the box can be in; default the whole frame.
        support_z: The surface the box stands on [m, world], for the returned box's height only.
        band_below, band_above: The height band taken as rim, about rim_z [m].
        min_coverage: Share of the outline that must lie on rim points.
        stride: Read every n-th pixel.

    Returns:
        ``route`` "found" with ``container_obb`` (centre, half extents of the stated size, yaw),
        ``coverage`` and ``yaw_ambiguous``; or "not_found" with why in ``detail``.
    """
    size = (max(float(size_long), float(size_short)), min(float(size_long), float(size_short)))
    out: Output = {
        "route": "not_found",
        "detail": "",
        "container_obb": None,
        "center": None,
        "yaw_deg": 0.0,
        "coverage": 0.0,
        "rim_points": 0,
        "yaw_ambiguous": False,
        "camera": "",
    }
    if size[1] <= 0:
        out["detail"] = "size_long and size_short must be positive"
        return out
    cam = _camera(ctx, camera_frame, camera or None)
    if cam is None:
        out["detail"] = f"no posed depth camera {camera or '(any fixed one)'!r} in the observation"
        return out
    out["camera"] = str(cam.get("name", ""))
    pts = world_points(cam, max(1, int(stride)))
    keep = (pts[:, 2] > float(rim_z) - float(band_below)) & (
        pts[:, 2] < float(rim_z) + float(band_above)
    )
    if roi:
        keep &= (pts[:, 0] > float(roi["x_min"])) & (pts[:, 0] < float(roi["x_max"]))
        keep &= (pts[:, 1] > float(roi["y_min"])) & (pts[:, 1] < float(roi["y_max"]))
    xy = pts[keep, :2]
    for other in exclude_obbs or []:
        xy = xy[outside(xy, other, EXCLUDE_MARGIN)]
    out["rim_points"] = int(len(xy))
    if len(xy) < MIN_POINTS:
        out["detail"] = (
            f"{len(xy)} depth points between {rim_z - band_below:.3f} and "
            f"{rim_z + band_above:.3f} m (need {MIN_POINTS})"
        )
        return out
    fit = fit_known_rectangle(xy, size, min_coverage=0.0)
    if fit is None:
        out["detail"] = "no outline placement covered any rim point"
        return out
    out["coverage"] = fit["coverage"]
    out["yaw_deg"] = fit["yaw_deg"]
    if fit["coverage"] < float(min_coverage):
        out["detail"] = (
            f"the best placement has {fit['coverage']:.2f} of the outline on rim points "
            f"(need {float(min_coverage):.2f})"
        )
        return out
    out["route"] = "found"
    out["yaw_ambiguous"] = bool((size[0] - size[1]) < NEAR_SQUARE * size[0])
    out["center"] = {"x": float(fit["center"][0]), "y": float(fit["center"][1]), "z": float(rim_z)}
    out["container_obb"] = upright_obb(
        fit["center"], size, fit["yaw_deg"], float(rim_z), float(support_z)
    )
    out["detail"] = (
        f"{fit['coverage']:.2f} of the {size[0] * 1e3:.0f} x {size[1] * 1e3:.0f} mm outline on "
        f"{len(xy)} rim points, turned {fit['yaw_deg']:.0f} deg"
        + ("; nearly square, the turn may be a quarter out" if out["yaw_ambiguous"] else "")
    )
    return out
