# Adapted from TSR (https://github.com/personalrobotics/tsr) at
# 5279ca7baf95e6718d4628a3a26fd6ebeefedf0d, src/tsr/placement/_stable_poses.py
# (stable_poses_mesh, _group_coplanar_facets, _plane_basis, _inward_distance,
# _rotation_to_align) and src/tsr/placement/stable_placer.py (place_mesh's
# ordering and margin filter, _fits), BSD-2-Clause,
# Copyright (c) 2025, Siddhartha Srinivasa and contributors to TSR.
# Changes: the object comes as a world-frame OrientedBoundingBox or a perceived
# point cloud (optionally closed down to the support plane, since a camera above
# sees no underside) instead of a mesh in its own frame; the centre of mass
# defaults to the hull's volume centroid; each rest is reported in the world
# (the face's normal now, the turn that puts it down, the COM height and, given a
# goal point, the resting pose) instead of as a TSR template; the surface fit
# takes the support's half-extents; the object's current rest is identified.
"""Enumerate the stable resting faces of an object and how to turn it onto each."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)

NORMAL_ATOL = 1e-6
"""Upstream's coplanarity tolerance on unit face normals (radians), for exact geometry."""

CLOUD_NORMAL_ATOL = 0.05
"""Coplanarity tolerance for perceived clouds (radians, ~3 deg): depth noise tilts the
facets of one face by a few degrees; upstream's 1e-6 would split every face into
single triangles too small to hold the COM (upstream #152's failure)."""

CLOUD_PLANE_ATOL = 0.002
"""Plane-offset tolerance for perceived clouds (m): ~D405 depth noise at 0.5 m."""

CURRENT_TOL_DEG = 10.0
"""A rest is the object's current one when its face points within this of straight down."""

_NEIGHBOURS = np.array(list(itertools.product((-1, 0, 1), repeat=4)))
_NEG_Z = np.array([0.0, 0.0, -1.0])
_AXES = ("x", "y", "z")


class Rest(TypedDict):
    label: str
    tipping_deg: float
    support_margin_m: float
    com_height_m: float
    center_height_m: float
    footprint_radius_m: float
    turn_deg: float
    face_normal_world: list[float]
    rotation_to_rest: list[list[float]]
    facet_count: int
    pose: dict[str, Any] | None


class Output(TypedDict):
    route: str
    rests: list[Rest]
    n_rests: int
    best: Rest | None
    best_tipping_deg: float
    best_turn_deg: float
    best_com_height_m: float
    best_pose: dict[str, Any] | None
    current_index: int
    current_stable: bool
    com: dict[str, float]
    reason: str


def length_atol(scale: float) -> float:
    """Upstream's scale-aware length tolerance: 1e-9 + 1e-6 * scale."""
    return 1e-9 + 1e-6 * float(scale)


def rotation_to_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R @ a = b (unit vectors)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def group_coplanar_facets(
    hull: ConvexHull, atol: float, normal_atol: float = NORMAL_ATOL
) -> Iterator[tuple[np.ndarray, list[int]]]:
    """Group hull facets lying in one plane (to ``normal_atol`` / ``atol``); yield
    ``(mean plane, facet indices)``. Upstream's 4D-grid grouping against a representative."""
    cell = np.array([normal_atol, normal_atol, normal_atol, atol])
    groups: list[tuple[np.ndarray, list[int]]] = []
    index: dict[tuple[int, ...], int] = {}
    for i in range(len(hull.simplices)):
        plane = hull.equations[i, :4]
        key = np.floor(plane / cell).astype(np.int64)
        gid = None
        for candidate in (index.get(tuple(k)) for k in key + _NEIGHBOURS):
            if candidate is None:
                continue
            delta = np.abs(groups[candidate][0] - plane)
            if delta[:3].max() <= normal_atol and delta[3] <= atol:
                gid = candidate
                break
        if gid is None:
            gid = len(groups)
            groups.append((plane, []))
            index[tuple(key)] = gid
        groups[gid][1].append(i)
    for _, indices in groups:
        planes = hull.equations[indices, :4]
        n = planes[:, :3].mean(axis=0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        yield np.append(n / norm, planes[:, 3].mean()), indices


def plane_basis(n: np.ndarray) -> np.ndarray:
    """Orthonormal 3x2 basis of the plane with unit normal ``n``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, n) * n
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u)], axis=-1)


def inward_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance (m) from ``point`` to a counter-clockwise convex polygon's boundary,
    positive inside."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 0.0
    if not np.any(keep):
        return -np.inf
    inward = np.stack([-edges[keep, 1], edges[keep, 0]], axis=-1) / lengths[keep, None]
    return float(np.einsum("ij,ij->i", inward, point - polygon[keep]).min())


def stable_faces(
    vertices: np.ndarray,
    com: np.ndarray,
    normal_atol: float = NORMAL_ATOL,
    plane_atol: float | None = None,
) -> list[dict[str, Any]]:
    """Upstream's stable_poses_mesh: every hull face the COM projects inside of, by more
    than the length tolerance. Normals are outward, in the vertices' frame."""
    vertices = np.asarray(vertices, dtype=float)
    com = np.asarray(com, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError(f"need (N >= 4, 3) vertices, got {vertices.shape}")
    try:
        hull = ConvexHull(vertices)
    except QhullError as exc:
        raise ValueError("the points are coplanar or degenerate: no 3D hull") from exc
    atol = length_atol(float(np.ptp(vertices, axis=0).max()))
    faces = []
    for plane, simplices in group_coplanar_facets(hull, plane_atol or atol, normal_atol):
        n, d = plane[:3], float(plane[3])
        com_height = -(np.dot(n, com) + d)
        if com_height <= atol:
            continue
        idx = sorted({int(v) for s in simplices for v in hull.simplices[s]})
        face_verts = vertices[idx]
        basis = plane_basis(n)
        v0 = face_verts[0]
        pts = (face_verts - v0) @ basis
        p = (com - v0) @ basis
        try:
            poly = pts[ConvexHull(pts).vertices]
        except QhullError:
            continue
        margin = inward_distance(p, poly)
        if margin <= atol:
            continue
        faces.append(
            {
                "normal": n,
                "origin_height": -d,
                "com_height": float(com_height),
                "support_margin": float(margin),
                "tipping": float(np.arctan2(margin, com_height)),
                "facet_count": len(simplices),
            }
        )
    return faces


def hull_centroid(vertices: np.ndarray) -> np.ndarray:
    """Volume centroid of the convex hull (uniform density)."""
    try:
        hull = ConvexHull(vertices)
    except QhullError as exc:
        raise ValueError("the points are coplanar or degenerate: no 3D hull") from exc
    ref = vertices[hull.vertices].mean(axis=0)
    tets = vertices[hull.simplices]
    vols = (
        np.abs(
            np.einsum("ij,ij->i", tets[:, 0] - ref, np.cross(tets[:, 1] - ref, tets[:, 2] - ref))
        )
        / 6.0
    )
    cents = (tets.sum(axis=1) + ref) / 4.0
    return (cents * vols[:, None]).sum(axis=0) / vols.sum()


def quat_to_matrix(q: dict[str, float] | None) -> np.ndarray:
    q = q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
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


def matrix_to_quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.empty(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
        w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _vec(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v[k]) for k in _AXES])
    return np.asarray(v, dtype=float).reshape(3)


def _points(points: Any) -> np.ndarray:
    if isinstance(points, dict):
        points = points.get("points")
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {arr.shape}")
    return arr[np.isfinite(arr).all(axis=1)]


def close_to_support(points: np.ndarray, support_z: float, max_points: int = 4000) -> np.ndarray:
    """Add each point's drop onto the support plane: the hull of a top-down view becomes the
    prism down to the table that the camera could not see. Subsampled to ``max_points``."""
    if len(points) > max_points:
        points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
    floor = points.copy()
    floor[:, 2] = float(support_z)
    return np.vstack([points[points[:, 2] > support_z], floor])


PRISM_UPPER_SHARE = 0.5
"""Prism closing takes the footprint from points above this share of the object's
height: mixed pixels at a top-down camera's depth edges fall between the top and the
table and widened the real grey holder's footprint from 76 x 100 to 88 x 112 mm."""

PRISM_SIMPLIFY_M = 0.003
"""Douglas-Peucker tolerance on the footprint polygon (m), ~2 overhead-camera pixels at 0.6 m."""


def prism_from_top_view(points: np.ndarray, support_z: float) -> np.ndarray:
    """Vertices of the prism a top-down view implies: the footprint of the points in the
    upper part of the object, simplified, extruded from the support to the top (p98)."""
    import cv2

    above = points[points[:, 2] > support_z]
    if len(above) < 3:
        raise ValueError("no points above the support plane")
    top = float(np.percentile(above[:, 2], 98))
    upper = above[above[:, 2] >= support_z + PRISM_UPPER_SHARE * (top - support_z)]
    xy = upper[:, :2] if len(upper) >= 3 else above[:, :2]
    try:
        ring = xy[ConvexHull(xy).vertices]
    except QhullError as exc:
        raise ValueError("the footprint is degenerate (a line or a point)") from exc
    poly = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), PRISM_SIMPLIFY_M, True)
    poly = poly.reshape(-1, 2).astype(float)
    if len(poly) < 3:
        poly = ring
    n = len(poly)
    return np.vstack([np.c_[poly, np.full(n, support_z)], np.c_[poly, np.full(n, top)]])


def _label(n_local: np.ndarray, local: bool) -> str:
    axis = int(np.argmax(np.abs(n_local)))
    if local and abs(abs(n_local[axis]) - 1.0) <= 1e-6:
        return f"{'+' if n_local[axis] > 0 else '-'}{_AXES[axis]}"
    return ""


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any] | None = None,
    points: Any = None,
    com: Any = None,
    support_z: float | None = None,
    closing: str = "prism",
    min_margin_deg: float = 0.0,
    surface_half_extents: list[float] | None = None,
    goal_position: Any = None,
    current_tol_deg: float = CURRENT_TOL_DEG,
) -> Output:
    """Stable rests of the object given as ``object_obb`` (world OBB, half-extents) or
    ``points`` (world cloud; with ``support_z`` the cloud is closed down to that plane).

    ``com`` (world) defaults to the OBB centre or the hull's volume centroid. Rests with a
    tipping angle under ``min_margin_deg``, or wider (about the COM) than
    ``surface_half_extents`` [hx, hy] allow at every yaw, are dropped. With
    ``goal_position`` (a point on the support) each rest carries the world pose of the
    reference point (OBB centre, or the COM for a cloud) resting there.
    """
    if (object_obb is None) == (points is None):
        raise ValueError("give exactly one of object_obb or points")
    if object_obb is not None:
        R_obj = quat_to_matrix(object_obb.get("orientation"))
        centre = _vec(object_obb["center"])
        half = np.abs(_vec(object_obb["extent"]))
        if (half <= 0).any():
            raise ValueError(f"object_obb has a zero extent: {half.tolist()}")
        corners = np.array(list(itertools.product(*[(-h, h) for h in half])))
        com_w = centre if com is None else _vec(com)
        verts, com_local = corners, R_obj.T @ (com_w - centre)
        normal_atol, plane_atol, local = NORMAL_ATOL, None, True
    else:
        if closing not in ("prism", "drop", "none"):
            raise ValueError(f"closing {closing!r} not in ('prism', 'drop', 'none')")
        cloud = _points(points)
        exact = False
        if support_z is not None and closing == "prism":
            cloud, exact = prism_from_top_view(cloud, float(support_z)), True
        elif support_z is not None and closing == "drop":
            cloud = close_to_support(cloud, float(support_z))
        if len(cloud) < 4:
            raise ValueError(f"need at least 4 points, got {len(cloud)}")
        com_w = hull_centroid(cloud) if com is None else _vec(com)
        R_obj = np.eye(3)
        verts, com_local = cloud - com_w, np.zeros(3)
        if exact:
            normal_atol, plane_atol = NORMAL_ATOL, None
        else:
            normal_atol, plane_atol = CLOUD_NORMAL_ATOL, CLOUD_PLANE_ATOL
        local = False

    faces = stable_faces(verts, com_local, normal_atol, plane_atol)
    faces.sort(key=lambda f: -f["tipping"])
    goal = None if goal_position is None else _vec(goal_position)
    min_rad = np.radians(float(min_margin_deg))
    rests: list[Rest] = []
    dropped = {"margin": 0, "surface": 0}
    for k, f in enumerate(faces):
        if f["tipping"] < min_rad:
            dropped["margin"] += 1
            continue
        n_w = R_obj @ f["normal"]
        R_turn = rotation_to_align(n_w, _NEG_Z)
        R_rest = R_turn @ R_obj
        rel = (verts - com_local) @ R_rest.T
        footprint = float(np.linalg.norm(rel[:, :2], axis=1).max())
        if surface_half_extents is not None and footprint > min(
            float(v) for v in surface_half_extents[:2]
        ):
            dropped["surface"] += 1
            continue
        # Height of the reference point (OBB centre / cloud COM) above the support at rest.
        center_h = float(f["origin_height"])
        pose = None
        if goal is not None:
            pos = goal + np.array([0.0, 0.0, center_h])
            pose = {
                "position": {a: float(v) for a, v in zip(_AXES, pos, strict=True)},
                "rotation": matrix_to_quat(R_rest),
            }
        turn = float(np.degrees(np.arccos(np.clip(np.dot(n_w, _NEG_Z), -1.0, 1.0))))
        rests.append(
            {
                "label": _label(f["normal"], local) or f"face-{k + 1}",
                "tipping_deg": round(float(np.degrees(f["tipping"])), 3),
                "support_margin_m": round(f["support_margin"], 5),
                "com_height_m": round(f["com_height"], 5),
                "center_height_m": round(center_h, 5),
                "footprint_radius_m": round(footprint, 5),
                "turn_deg": round(turn, 2),
                "face_normal_world": [round(float(v), 6) for v in n_w],
                "rotation_to_rest": R_turn.round(6).tolist(),
                "facet_count": int(f["facet_count"]),
                "pose": pose,
            }
        )
    current = min(range(len(rests)), key=lambda i: rests[i]["turn_deg"], default=-1)
    if current >= 0 and rests[current]["turn_deg"] > float(current_tol_deg):
        current = -1
    com_out = {a: round(float(v), 5) for a, v in zip(_AXES, com_w, strict=True)}
    if not rests:
        why = (
            f"{len(faces)} stable face(s); {dropped['margin']} under {min_margin_deg} deg, "
            f"{dropped['surface']} too wide for the surface"
        )
        logger.info("compute_rests: no_stable_pose (%s)", why)
        return {
            "route": "no_stable_pose",
            "rests": [],
            "n_rests": 0,
            "best": None,
            "best_tipping_deg": 0.0,
            "best_turn_deg": 0.0,
            "best_com_height_m": 0.0,
            "best_pose": None,
            "current_index": -1,
            "current_stable": False,
            "com": com_out,
            "reason": why,
        }
    why = (
        f"{len(rests)} rest(s), most stable {rests[0]['label']} at {rests[0]['tipping_deg']:.1f} deg; "
        + (f"now on {rests[current]['label']}" if current >= 0 else "now on no stable face")
    )
    logger.info("compute_rests: %s", why)
    return {
        "route": "placements",
        "rests": rests,
        "n_rests": len(rests),
        "best": rests[0],
        "best_tipping_deg": rests[0]["tipping_deg"],
        "best_turn_deg": rests[0]["turn_deg"],
        "best_com_height_m": rests[0]["com_height_m"],
        "best_pose": rests[0]["pose"],
        "current_index": current,
        "current_stable": current >= 0,
        "com": com_out,
        "reason": why,
    }
