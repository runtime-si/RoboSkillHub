# SPDX-License-Identifier: MIT
"""pose6d tool bundle -- 6D pose of a known rigid part from depth.

Two ``@tool`` functions over ``_impl.py``: ``pose6d.register_mesh`` registers a
model (a box / cube / cylinder primitive, or a mesh file) to a point cloud and
``pose6d.register_mask`` does the same from a segmentation mask and a depth
frame, so the viewpoint and the world transform are exact. The maths lives in
``_impl.py``; this module is the typed boundary.

The primitive fast path needs only numpy/scipy. Open3D (FPFH + RANSAC global
registration for meshes) is imported inside ``_impl`` when that path runs, so
importing this module is always cheap.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import Mask, Se3Pose, matrix_to_pose, pose_to_matrix


class Pose6dAlternative(TypedDict):
    fitness: float
    rmse: float
    coverage: float
    score_gap: float  # winner's score minus this one's; < 0.03 makes the fit ambiguous
    rotation_deg: float  # from the winner, after folding the part's symmetry
    translation_m: float
    pose: Se3Pose


class Pose6dResult(TypedDict):
    found: bool  # False only when there were too few points to try
    status: str  # "registered" | "ambiguous_fit" | "poor_fit"
    reason: str
    pose: Se3Pose  # object -> the input points' frame (world: register_mask with camera_pose)
    matrix: list[list[float]]  # the same pose as a 4x4
    fitness: float  # share of points within inlier_threshold_m of the visible model surface
    rmse: float  # RMS distance of those inliers, metres
    coverage: float  # share of the model's camera-facing area with a point near it
    inlier_threshold_m: float
    method: str  # "primitive_box" | "primitive_cylinder" | "global_fpfh_icp" | "icp_from_init"
    symmetry_order: int  # symmetric equivalents of the pose; 0 = turns freely about z
    continuous_symmetry: bool
    yaw_ambiguity_deg: float  # smallest self-mapping turn about the object z (0 none, 360 any)
    equivalent_poses: list[Se3Pose]  # every symmetric equivalent of pose (empty when continuous)
    ambiguous_fit: bool  # a non-equivalent pose explains the data almost as well
    alternative: Pose6dAlternative | None
    n_points: int
    n_dropped_support: int
    hypotheses: int
    runtime_s: float


def _vec(v) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
    return np.asarray(v, dtype=np.float64).reshape(-1)


def _matrix(m) -> np.ndarray | None:
    if m is None:
        return None
    if isinstance(m, dict):
        return pose_to_matrix(m)
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def _to_result(r: dict, T_frame: np.ndarray | None = None) -> Pose6dResult:
    def carry(T):
        T = np.asarray(T)
        return T if T_frame is None else T_frame @ T

    T = carry(r["T"])
    alt = r.get("alternative")
    return Pose6dResult(
        found=r["found"],
        status=r["status"],
        reason=r["reason"],
        pose=matrix_to_pose(T),
        matrix=T.tolist(),
        fitness=float(r["fitness"]),
        rmse=float(r["rmse"]),
        coverage=float(r["coverage"]),
        inlier_threshold_m=float(r["inlier_threshold_m"]),
        method=r["method"],
        symmetry_order=int(r["symmetry_order"]),
        continuous_symmetry=bool(r["continuous_symmetry"]),
        yaw_ambiguity_deg=float(r["yaw_ambiguity_deg"]),
        equivalent_poses=[matrix_to_pose(carry(E)) for E in r["equivalent_T"]],
        ambiguous_fit=bool(r["ambiguous_fit"]),
        alternative=None
        if alt is None
        else Pose6dAlternative(
            fitness=float(alt["fitness"]),
            rmse=float(alt["rmse"]),
            coverage=float(alt["coverage"]),
            score_gap=float(alt["score_gap"]),
            rotation_deg=float(alt["rotation_deg"]),
            translation_m=float(alt["translation_m"]),
            pose=matrix_to_pose(carry(alt["T"])),
        ),
        n_points=int(r["n_points"]),
        n_dropped_support=int(r["n_dropped_support"]),
        hypotheses=int(r["hypotheses"]),
        runtime_s=float(r["runtime_s"]),
    )


@tool(
    name="pose6d.register_mesh",
    summary=(
        "6D pose of a known part (box / cube / cylinder primitive or mesh file) from its point "
        "cloud, with fitness, RMSE, coverage and its symmetric equivalents."
    ),
    tags=("perception",),
)
def register_mesh(
    points: np.ndarray,
    mesh: dict | str,
    init: Se3Pose | np.ndarray | None = None,
    symmetry: str | list | None = None,
    viewpoint: list[float] | None = None,
    up: list[float] | None = None,
    forward: list[float] | None = None,
    support_plane: list[float] | None = None,
    noise_m: float | None = None,
    method: str = "auto",
) -> Pose6dResult:
    """Register ``mesh`` to ``points`` ([N, 3] metres, any frame: camera or world).

    ``mesh``: ``{"box": [x, y, z]}`` (edge lengths; frame at the centre, axes
    along the edges -- a LEGO brick body, a domino), ``{"cube": s}``,
    ``{"cylinder": [radius, height]}`` (axis = z; a peg), a mesh path
    (.ply/.stl/.obj, metres) or ``{"path": p, "scale": 0.001}`` for a model in mm.
    Primitives take the analytic fast path (``method="auto"``/``"primitive"``);
    meshes, or ``method="global"``, take Open3D FPFH + RANSAC then ICP.

    ``viewpoint``: the camera centre in the points' frame (default the origin,
    right for camera-frame points; pass it for world points, or use
    ``register_mask``). Only camera-facing surfaces are matched.
    ``up``: the support normal (world +z); orders the symmetric equivalents
    (z most along it) and seeds a frame hypothesis. ``forward`` breaks the
    remaining tie (x most along it; default the frame's x).
    ``support_plane`` ``[nx, ny, nz, d]`` (``n.x + d = 0``, ``n`` up): the part
    rests on it in a stable pose; points within 3 mm of it are dropped and the
    part's resting face/side is held on it. Use it with the overhead camera:
    without it a part under ~15 mm tall seen from 0.9 m tilts by 5-25 deg.
    ``init``: a prior pose (skips the hypothesis search; the symmetric
    equivalent nearest it is returned).
    ``symmetry``: ``None`` = the primitive's own (box 4/8/24, cylinder free
    spin + end flip; none for a mesh), ``"none"``, ``"z2"``, ``"z4"``,
    ``"box"``, ``"cylinder"``, or a list of object-frame 3x3/4x4 rotations.
    It changes only which equivalent is returned and listed, never the fit.
    ``noise_m``: depth noise std at the part (default: estimated from residuals).

    ``status``: ``registered`` (fitness >= 0.6, coverage >= 0.4, no close
    non-equivalent alternative), ``ambiguous_fit`` (one exists: see
    ``alternative``) or ``poor_fit``.
    """
    from gap_skills.tools.pose6d import _impl

    r = _impl.register(
        points,
        mesh,
        init=_matrix(init),
        symmetry=symmetry,
        viewpoint=_vec(viewpoint),
        up=_vec(up),
        forward=_vec(forward),
        noise_m=noise_m,
        method=method,
        support_plane=_vec(support_plane),
    )
    return _to_result(r)


@tool(
    name="pose6d.register_mask",
    summary=(
        "6D pose of a known part from its segmentation mask and a depth frame (camera centre "
        "used as the viewpoint); world pose when a camera pose is given."
    ),
    tags=("perception",),
)
def register_mask(
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mesh: dict | str,
    camera_pose: Se3Pose | np.ndarray | None = None,
    init: Se3Pose | np.ndarray | None = None,
    symmetry: str | list | None = None,
    up: list[float] | None = None,
    forward: list[float] | None = None,
    support_plane: list[float] | None = None,
    noise_m: float | None = None,
    method: str = "auto",
    erode_px: int = 0,
) -> Pose6dResult:
    """Back-project ``mask`` (nonzero = part, [H, W]) through ``depth`` (metres)
    and pinhole ``intrinsics`` (3x3), then ``register_mesh`` with the camera
    centre as viewpoint. With ``camera_pose`` (camera-to-world) the points,
    ``init``, ``up``, ``forward``, ``support_plane`` and the result are in
    world; without it, in the camera frame. ``erode_px`` shrinks the mask
    first (a segmenter's edge pixels often carry the table's depth).
    """
    from gap_skills.tools.pose6d import _impl

    m = np.asarray(mask) > 0
    if erode_px > 0:
        from scipy.ndimage import binary_erosion

        m = binary_erosion(m, iterations=int(erode_px))
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    v, u = np.nonzero(m & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    P = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    T_wc = _matrix(camera_pose)
    viewpoint = np.zeros(3)
    if T_wc is not None:
        P = P @ T_wc[:3, :3].T + T_wc[:3, 3]
        viewpoint = T_wc[:3, 3]
    r = _impl.register(
        P,
        mesh,
        init=_matrix(init),
        symmetry=symmetry,
        viewpoint=viewpoint,
        up=_vec(up),
        forward=_vec(forward),
        noise_m=noise_m,
        method=method,
        support_plane=_vec(support_plane),
    )
    return _to_result(r)
