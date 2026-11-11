# Adapted from OK-Robot (https://github.com/ok-robot/ok-robot, commit 174c742b6a1866a2da8f75a89e2c544bb77ab5f9,
# ok-robot-manipulation/src/anygrasp_manipulation/object_handler.py, ObjectHandler.place), MIT License,
# Copyright (c) 2024 NYU Generalizable Robotics and AI Lab (GRAIL); the license is LICENSE.ok-robot beside
# this skill. Kept: the drop point is the median of the receptacle's horizontal footprint taken over its unique
# 1 cm cells (so a densely seen part does not pull it), and the surface height is the highest receptacle point
# in a band around that point. Changed: world frame (z up) instead of the Stretch head camera's tilted frame;
# the band is a disc of `band` (3 cm) round the drop point instead of a +-10 cm strip on the far half; the
# height is a high percentile, not the maximum (one noisy depth pixel); the release is `drop_height` (2 cm) over
# the surface instead of OK-Robot's ~10 cm (a tabletop arm; chain into placing-with-contact-descent); the
# receptacle comes from sam3.segment_text and geometry.mask_to_world_points, or from given points; routed exits.
"""Find where to put an object on a receptacle: the middle of its top surface and that surface's height."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

CELL = 0.01
"""Footprint cell [m] for the median over unique cells: OK-Robot's 1 cm (x 100, astype int)."""

BAND = 0.03
"""Radius round the drop point in which the surface height is taken [m]. OK-Robot uses a 10 cm strip for a
mobile robot's furniture-sized receptacles; a 3 cm disc keeps a tabletop receptacle's rim out of the middle."""

TOP_PCT = 95.0
"""Percentile of the heights in the band read as the surface [m]: robust to a few noisy depth pixels where
OK-Robot takes the maximum."""

DROP_HEIGHT = 0.02
"""How far over the surface the held object's bottom is released [m] (the candidate's note: 1-3 cm for a
tabletop arm; OK-Robot drops from ~10 cm)."""

MIN_POINTS = 30
"""Fewest receptacle points to place on."""

MIN_IN_BAND = 5
"""Fewest points in the band round the drop point (a ring-shaped receptacle has none in its middle)."""


class Output(TypedDict):
    route: str
    detail: str
    place_pose: dict
    surface_z: float
    footprint_center: dict
    footprint_size: list
    n_points: int


def _points_from_camera(ctx: NodeContext, query: str, camera_name: str) -> tuple[np.ndarray, str]:
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else None
    if isinstance(cams, dict):
        cams = [dict(v, name=k) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("rgb") is not None and c.get("depth") is not None]
    if camera_name:
        cams = [c for c in cams if c.get("name") == camera_name]
    if not cams:
        return np.zeros((0, 3)), "no camera frame with rgb and depth"
    cam = cams[0]
    reply = ctx.tool("sam3.segment_text", image=cam["rgb"], query=query, max_results=1)
    masks = list((reply or {}).get("masks") or [])
    if not masks:
        return np.zeros((0, 3)), f"no mask for {query!r}"
    mask = np.asarray(masks[0]) > 0
    depth = np.asarray(cam["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    reply = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask.astype(np.uint8) * 255,
        depth=depth,
        intrinsics=np.asarray(cam["intrinsics"], dtype=np.float64),
        camera_pose=cam["pose"],
    )
    pts = reply.get("points", reply) if isinstance(reply, dict) else reply
    return np.asarray(pts if pts is not None else [], dtype=np.float64).reshape(-1, 3), ""


def place_point(
    points: np.ndarray, band: float = BAND, top_pct: float = TOP_PCT
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """OK-Robot's rule in the world frame: (drop xy, surface height, footprint size, points in the band)."""
    cells = np.unique(np.floor(points[:, :2] / CELL).astype(int), axis=0)
    xy = (np.median(cells, axis=0) + 0.5) * CELL
    near = points[np.linalg.norm(points[:, :2] - xy, axis=1) <= band]
    z = float(np.percentile(near[:, 2], top_pct)) if len(near) else float("nan")
    size = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
    return xy, z, size, len(near)


def run(
    ctx: NodeContext,
    query: str = "",
    points: list[Any] | None = None,
    held_bottom_offset: float = 0.0,
    drop_height: float = DROP_HEIGHT,
    band: float = BAND,
    camera_name: str = "",
    rotation: dict[str, float] | None = None,
) -> Output:
    out: Output = {
        "route": "receptacle_not_found",
        "detail": "",
        "place_pose": {},
        "surface_z": 0.0,
        "footprint_center": {},
        "footprint_size": [],
        "n_points": 0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    if points is not None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        why = ""
    elif query:
        try:
            pts, why = _points_from_camera(ctx, query, camera_name)
        except Exception as exc:  # noqa: BLE001 - a missing tool or a failed call is a not-found
            pts, why = np.zeros((0, 3)), f"perception failed ({exc})"
    else:
        return finish("invalid_input", "give the receptacle as a text query or as world points")
    pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
    out["n_points"] = int(len(pts))
    if len(pts) < MIN_POINTS:
        return finish(
            "receptacle_not_found", why or f"{len(pts)} receptacle points (fewer than {MIN_POINTS})"
        )
    xy, z, size, n_band = place_point(pts, float(band))
    out["footprint_center"] = {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "z": float(np.median(pts[:, 2])),
    }
    out["footprint_size"] = [round(float(v), 4) for v in size]
    if n_band < MIN_IN_BAND:
        return finish(
            "receptacle_not_found",
            f"only {n_band} receptacle points within {band * 1e3:.0f} mm of its footprint's median: no surface "
            "there (an open ring or a box's hollow?)",
        )
    out["surface_z"] = z
    out["place_pose"] = {
        "position": {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "z": z + float(drop_height) + float(held_bottom_offset),
        },
        "rotation": rotation or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return finish(
        "place_point",
        f"top surface at {z * 1e3:.1f} mm, drop point ({xy[0]:.3f}, {xy[1]:.3f}), footprint "
        f"{size[0] * 1e3:.0f} x {size[1] * 1e3:.0f} mm from {len(pts)} points",
    )
