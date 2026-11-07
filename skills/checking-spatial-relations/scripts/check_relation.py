# Adapted from REFLECT (https://github.com/real-stanford/reflect, commit
# c2f9db993e372250321ca74e56ae27519716c934): main/scene_graph.py
# (SceneGraph.add_edge, the spatial-relation thresholds, get_iou) and
# main/utils.py (get_pcd_dist, in_hull, is_inside). MIT License, Copyright (c)
# 2023 Zeyi (Lucia) Liu; the license is skills/checking-spatial-relations/LICENSE.reflect.
# Changes: world frame z-up instead of AI2-THOR's y-up; object clouds come from
# masks + one RGB-D frame (or a caller's point cloud) instead of THOR instance
# masks; distances rescaled from a room to a tabletop (contact 0.1 -> 0.05 m,
# close 0.4 -> 0.10 m, voxel 0.01 -> 0.003 m); bounding boxes trimmed to the
# 1st-99th percentile against depth noise; up/down measured along world z and
# the blocking test taken first when the two objects line up with the camera
# ray (a downward camera's ray is world -z); upstream's "countertop / stove
# burner" rename of inside to on top of generalised to any flat reference;
# "near" holds whenever the clouds are within the close distance; scipy KD-tree
# in place of Open3D; detached mask-bleed pieces dropped (upstream denoised
# containers only); one asked relation routed holds / does_not_hold /
# not_visible.
"""Whether a spatial relation (inside, on_top_of, near, blocking) holds between two seen objects."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay, cKDTree

CONTACT_DISTANCE = 0.05
"""Clouds closer than this [m] may be inside / on top of each other. Upstream's
IN_CONTACT_DISTANCE is 0.1 m for an AI2-THOR kitchen; on a tabletop it must
still exceed the step a camera from above sees at a resting contact (a 35-40 mm
object's top over its support, its sides hidden) and stay under a held
object's lift."""

CLOSE_DISTANCE = 0.10
"""Clouds closer than this [m] are near. Upstream's CLOSE_DISTANCE is 0.4 m
between kitchen objects; a quarter of it, as for the contact distance."""

INSIDE_THRESH = 0.5
"""Fraction of the subject's points inside the reference's convex hull (upstream)."""

ON_TOP_OF_THRESH = 0.7
"""Fraction of the subject's points over the reference's footprint, and above
its top, for on top of (upstream)."""

NORM_THRESH_UP_DOWN = 0.9
NORM_THRESH_LEFT_RIGHT = 0.8
NORM_THRESH_FRONT_BACK = 0.9
"""Components of the unit vector between the box centres that name a
direction (upstream). Up/down is world z here, left/right and front/back the
camera's x and optical axis (upstream used the camera's axes for all three)."""

OCCLUDE_RATIO_THRESH = 0.5
"""Fraction of the reference's image box the subject's image box covers (upstream)."""

DEPTH_THRESH = 0.9
"""Fraction of the subject's depths no deeper than the reference's nearest one (upstream)."""

VOXEL = 0.003
"""Down-sampling voxel [m]. Upstream used 0.01 m on room-scale objects; 3 mm
keeps a 30 mm cube at ~100 points and a thin cable more than one voxel wide."""

TRIM = 1.0
"""Percent trimmed off each end of a cloud's extent before its box is taken,
so a few flying depth pixels do not grow it (upstream took the raw box)."""

FLAT_SUPPORT = 0.02
"""A reference whose trimmed height is under this [m] is a support surface
(table, board, tray): a subject in its hull is on top of it, not inside.
Upstream did this by name for countertops and stove burners only."""

LINK_VOXELS = 3.0
KEEP_CLUSTER = 0.2
"""Denoising: points within LINK_VOXELS voxels are linked, and only the pieces
holding at least KEEP_CLUSTER of the largest piece's points are kept. Upstream
removed radius outliers (30 neighbours in 3 cm) from containers only; with
real masks every object needs it, since a mask edge that crosses a depth step
lands on the occluder (measured in a YAM twin: a cube mask beside the hand
reached 0.27 m tall before this)."""

MIN_POINTS = 30
"""Fewest valid points (after down-sampling) an object needs to be judged at
all; upstream dropped THOR objects under 700 pixels, then required 30
neighbours within 3 cm when denoising containers."""

RELATIONS = ("inside", "on_top_of", "near", "blocking")
ALIASES = {
    "in": "inside",
    "inside_of": "inside",
    "on": "on_top_of",
    "on_top": "on_top_of",
    "on_top_of": "on_top_of",
    "next_to": "near",
    "beside": "near",
    "close_to": "near",
    "occluding": "blocking",
    "covering": "blocking",
}


class Output(TypedDict):
    route: str
    holds: bool
    relation: str
    relations: list[str]
    min_distance_m: float
    evidence: str
    measures: dict[str, float]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    if len(cameras) == 1:
        return cameras[0]
    names = [c.get("name") for c in cameras]
    raise ValueError(f"observation has no camera named {name!r} (has {names})")


def _pose_matrix(pose: Any) -> np.ndarray:
    if not isinstance(pose, dict) and np.asarray(pose).shape == (7,):
        flat = np.asarray(pose, dtype=np.float64)  # x y z qw qx qy qz
        pose = {
            "position": dict(zip("xyz", flat[:3], strict=True)),
            "rotation": dict(zip("wxyz", flat[3:], strict=True)),
        }
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = np.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p[k]) for k in ("x", "y", "z")]
        return m
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("camera pose is an Se3Pose or a 4x4 camera-to-world matrix")
    return m


def _mask_points(mask: Any, camera: dict[str, Any], t_cam: np.ndarray) -> np.ndarray:
    """World points of a mask's pixels with valid depth (geometry.mask_to_world_points' math)."""
    m = np.asarray(mask)
    m = m.reshape(m.shape[-2:]) > 0 if m.ndim > 2 else m > 0
    depth = np.asarray(camera["depth"], dtype=np.float64)
    if m.shape != depth.shape:
        raise ValueError(f"mask {m.shape} and depth {depth.shape} differ in size")
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    v, u = np.nonzero(m & np.isfinite(depth) & (depth >= 0.015) & (depth <= 20.0))
    z = depth[v, u]
    cam = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
    return cam @ t_cam[:3, :3].T + t_cam[:3, 3]


def _voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0.0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, points)
    return sums / np.bincount(inverse)[:, None]


def _main_clusters(points: np.ndarray, link: float, keep: float) -> np.ndarray:
    """Drop the pieces of a cloud that are not linked (within ``link``) to its body:
    a mask's edge pixels that took the depth of whatever is in front or behind."""
    if len(points) < 2 or link <= 0.0:
        return points
    pairs = cKDTree(points).query_pairs(link, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))
    )
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return points[sizes[labels] >= keep * sizes.max()]


def _box(points: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(points, trim, axis=0), np.percentile(points, 100.0 - trim, axis=0)


def _inside_fraction(src: np.ndarray, target: np.ndarray) -> float:
    """Fraction of ``src`` in the convex hull of ``target`` (upstream is_inside / in_hull)."""
    try:
        hull = ConvexHull(target)
        tri = Delaunay(target[hull.vertices])
    except Exception:  # flat or degenerate: upstream returns not inside
        return 0.0
    return float(np.mean(tri.find_simplex(src) >= 0))


def _on_top_fractions(top: np.ndarray, base: np.ndarray, trim: float) -> tuple[float, float]:
    """Fractions of ``top`` over ``base``'s footprint and above its top (upstream's
    corner_pts test, with THOR's y-up read as z-up)."""
    lo, hi = _box(base, trim)
    over = (top[:, 0] > lo[0]) & (top[:, 0] < hi[0]) & (top[:, 1] > lo[1]) & (top[:, 1] < hi[1])
    return float(np.mean(over)), float(np.mean(top[:, 2] > hi[2]))


def _image_box(cam_pts: np.ndarray, k: np.ndarray) -> np.ndarray:
    u = k[0, 0] * cam_pts[:, 0] / cam_pts[:, 2] + k[0, 2]
    v = k[1, 1] * cam_pts[:, 1] / cam_pts[:, 2] + k[1, 2]
    return np.array([u.min(), v.min(), u.max(), v.max()])


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Upstream get_iou's intersection area."""
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    return float(iw * ih)


def _normalise(relation: str) -> str:
    key = str(relation).strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in RELATIONS:
        raise ValueError(f"relation is one of {RELATIONS} (or an alias), not {relation!r}")
    return key


def _cloud(
    ctx: NodeContext,
    which: str,
    camera: dict[str, Any] | None,
    t_cam: np.ndarray | None,
    points: Any,
    mask: Any,
    description: str,
    score_min: float,
) -> tuple[np.ndarray, str]:
    """The object's world points and how they were obtained ("" when not seen)."""
    if points is not None:
        raw = points.get("points") if isinstance(points, dict) else points
        return np.asarray(raw, dtype=np.float64).reshape(-1, 3), "points"
    if camera is None:
        raise ValueError(f"{which}: a mask or a description needs an observation")
    if mask is None:
        if not description:
            raise ValueError(f"{which}: give {which}_points, {which}_mask or {which}_description")
        result = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=description, max_results=1
        )
        masks, scores = result.get("masks") or [], result.get("scores") or []
        if len(masks) == 0 or (scores and float(scores[0]) < score_min):
            best = float(scores[0]) if scores else 0.0
            return np.zeros((0, 3)), f"sam3 found no {description!r} (best score {best:.2f})"
        mask = masks[0]
    return _mask_points(mask, camera, t_cam), "mask"


def run(
    ctx: NodeContext,
    relation: str,
    observation: dict[str, Any] | None = None,
    subject_points: Any = None,
    reference_points: Any = None,
    subject_mask: Any = None,
    reference_mask: Any = None,
    subject_description: str = "",
    reference_description: str = "",
    reference_kind: str = "auto",
    camera_name: str = "overhead",
    contact_distance: float = CONTACT_DISTANCE,
    close_distance: float = CLOSE_DISTANCE,
    voxel: float = VOXEL,
    min_points: int = MIN_POINTS,
    score_min: float = 0.3,
) -> Output:
    asked = _normalise(relation)
    if reference_kind not in ("auto", "container", "surface"):
        raise ValueError("reference_kind is 'auto', 'container' or 'surface'")
    given_points = subject_points is not None and reference_points is not None
    if observation is None and not given_points:
        observation = ctx.tool("robot.get_observation")
    camera = _camera(observation, camera_name) if observation is not None else None
    t_cam = _pose_matrix(camera["pose"]) if camera is not None else None

    sub, sub_how = _cloud(
        ctx, "subject", camera, t_cam, subject_points, subject_mask, subject_description, score_min
    )
    ref, ref_how = _cloud(
        ctx,
        "reference",
        camera,
        t_cam,
        reference_points,
        reference_mask,
        reference_description,
        score_min,
    )
    link, keep = LINK_VOXELS * float(voxel), KEEP_CLUSTER
    sub = _main_clusters(
        _voxel_down(sub[np.all(np.isfinite(sub), axis=1)], float(voxel)), link, keep
    )
    ref = _main_clusters(
        _voxel_down(ref[np.all(np.isfinite(ref), axis=1)], float(voxel)), link, keep
    )
    out: Output = {
        "route": "not_visible",
        "holds": False,
        "relation": asked,
        "relations": [],
        "min_distance_m": float("inf"),
        "evidence": "",
        "measures": {"subject_points": float(len(sub)), "reference_points": float(len(ref))},
    }
    short = [
        f"{name} has {len(cloud)} points ({how or 'given'}; needs {int(min_points)})"
        for name, cloud, how in (("subject", sub, sub_how), ("reference", ref, ref_how))
        if len(cloud) < int(min_points)
    ]
    if short:
        out["evidence"] = "; ".join(short)
        return out

    m = out["measures"]
    dist = float(cKDTree(ref).query(sub, k=1)[0].min())
    out["min_distance_m"] = m["min_distance_m"] = dist
    sub_lo, sub_hi = _box(sub, TRIM)
    ref_lo, ref_hi = _box(ref, TRIM)
    m["reference_height_m"] = float(ref_hi[2] - ref_lo[2])
    surface = reference_kind == "surface" or (
        reference_kind == "auto" and m["reference_height_m"] < FLAT_SUPPORT
    )
    found: list[str] = []

    # IN CONTACT (upstream order: inside, then subject on top, then reference on top)
    if dist < float(contact_distance):
        m["inside_fraction"] = _inside_fraction(sub, ref)
        m["over_fraction"], m["above_fraction"] = _on_top_fractions(sub, ref, TRIM)
        rev_over, rev_above = _on_top_fractions(ref, sub, TRIM)
        if m["inside_fraction"] > INSIDE_THRESH:
            found.append("on_top_of" if surface else "inside")
        elif m["over_fraction"] > ON_TOP_OF_THRESH and m["above_fraction"] > ON_TOP_OF_THRESH:
            found.append("on_top_of")
        elif rev_over > ON_TOP_OF_THRESH and rev_above > ON_TOP_OF_THRESH:
            found.append("under")

    # CLOSE TO
    if dist < float(close_distance):
        found.append("near")
        if not set(found) & {"inside", "on_top_of", "under"}:
            vec = (sub_lo + sub_hi) / 2.0 - (ref_lo + ref_hi) / 2.0
            norm = vec / max(float(np.linalg.norm(vec)), 1e-9)
            along_ray = float(norm @ t_cam[:3, 2]) if t_cam is not None else 0.0
            sideways = float(norm @ t_cam[:3, 0]) if t_cam is not None else 0.0
            m["ray_alignment"] = abs(along_ray)
            if abs(along_ray) > NORM_THRESH_FRONT_BACK:
                k = np.asarray(camera["intrinsics"], dtype=np.float64)
                world_to_cam = np.linalg.inv(t_cam)
                sub_cam = sub @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
                ref_cam = ref @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
                sub_box, ref_box = _image_box(sub_cam, k), _image_box(ref_cam, k)
                ref_area = max(float((ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1])), 1e-9)
                m["occlude_ratio"] = _intersection(sub_box, ref_box) / ref_area
                m["in_front_fraction"] = float(np.mean(sub_cam[:, 2] <= ref_cam[:, 2].min()))
                if (
                    m["occlude_ratio"] > OCCLUDE_RATIO_THRESH
                    and m["in_front_fraction"] > DEPTH_THRESH
                ):
                    found.append("blocking")
            elif abs(norm[2]) > NORM_THRESH_UP_DOWN:
                found.append("above" if norm[2] > 0 else "below")
            elif abs(sideways) > NORM_THRESH_LEFT_RIGHT:
                found.append("right_of" if sideways > 0 else "left_of")

    out["relations"] = found
    holds = asked in found
    out["holds"] = holds
    out["route"] = "holds" if holds else "does_not_hold"
    shown = ", ".join(f"{k}={v:.3f}" for k, v in m.items())
    out["evidence"] = (
        f"subject {asked} reference: {'holds' if holds else 'does not hold'}; "
        f"found {found or ['nothing']} ({'surface' if surface else 'object'} reference; {shown})"
    )
    return out
