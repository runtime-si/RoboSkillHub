# SPDX-License-Identifier: MIT
"""6D pose of a known rigid part from a depth point cloud.

Two routes, chosen by the model spec:

* **Primitive fast path** (``{"box": [x, y, z]}``, ``{"cube": s}``,
  ``{"cylinder": [radius, height]}``): the surface is analytic, so no mesh is
  sampled and no features are matched. Rotation hypotheses come from three
  frame estimates (normal clusters, principal axes, the support/view axis with a
  minimum-area rectangle of the projected points), crossed with every way the
  box's axes can be laid on them, one per symmetry coset. Each is refined with a
  visibility-aware point-to-plane ICP against the exact surface, and the
  hypothesis with the best fitness x coverage wins. Symmetric equivalents are
  then folded to one canonical pose and listed.

* **Mesh path** (a path, or a primitive with ``method="global"``): Open3D FPFH
  features + RANSAC global registration, then point-to-plane ICP against the
  model points facing the camera. Adapted from Open3D
  (https://github.com/isl-org/Open3D, commit b6c5e19,
  ``examples/python/pipelines/registration_ransac.py`` preprocess_point_cloud
  and the feature-matching RANSAC call, ``robust_icp.py`` Tukey-loss
  point-to-plane ICP; MIT License, Copyright (c) 2018-2026 www.open3d.org,
  LICENSE.open3d). Changes: the scene is registered
  onto the model (a partial view always has a counterpart on the full model)
  and the result inverted; the voxel size follows the part's size rather than
  a fixed 5 cm; ICP is repeated against model points facing the viewpoint so
  points cannot settle on hidden faces; fitness, RMSE and coverage are
  reported in the same terms as the fast path.

Every distance is in metres, every transform a 4x4 object-to-frame matrix in
the frame of the input points.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np

#: Neighbours per PCA normal: 16 spans ~4 pixel rings, enough to average D405
#: noise (0.5 mm at 0.15 m) over its ~0.3 mm pixel footprint.
NORMAL_K = 16
#: Faces whose outward normal is up to ~6 deg past edge-on still count as
#: visible: a grazing face carries a few points and must not flip in and out.
VISIBLE_COS = -0.1
#: Inlier distance floor (1 mm = the overhead camera's depth quantisation).
INLIER_FLOOR_M = 0.001
#: Inlier distance ceiling: beyond 6 mm a point is off the part, whatever the noise.
INLIER_CEIL_M = 0.006
#: Residuals past 2x the inlier threshold (6 sigma) are background, not noise:
#: table returns a segmenter's mask bleeds in at the part's edges.
TRIM = 2.0
#: Points used to screen hypotheses (the refinement uses all of them).
SCREEN_POINTS = 600
#: ICP iterations for screening hypotheses and for the final refinement.
SCREEN_ITERS = 12
REFINE_ITERS = 40
#: Fits further apart than this (after folding symmetry) are different answers.
DISTINCT_DEG = 20.0
#: Distinct screened hypotheses carried into the full refinement.
REFINE_SEEDS = 4
#: A non-equivalent hypothesis scoring within this of the winner makes the fit ambiguous.
AMBIGUOUS_SCORE_GAP = 0.03
#: Accept thresholds behind ``status == "registered"``; measured on the synthetic
#: renders (correct fits: fitness >= 0.85, coverage >= 0.6 at 0.9 m).
MIN_FITNESS = 0.6
MIN_COVERAGE = 0.4
#: Two box dimensions within 1 mm are treated as equal (a cube, a square plate).
EQUAL_DIM_TOL_M = 0.001
#: Points kept for fitting; more adds runtime, not accuracy, on parts this size.
MAX_POINTS = 3000
#: The contact rows together weigh as much as all the data rows: at 0.25 the
#: overhead brick renders kept 2-4 deg of tilt, at 1.0 they fall to ~1 deg
#: (stronger only trades yaw accuracy for tilt that is already at the noise).
SUPPORT_WEIGHT = 1.0
#: With a support plane, points less than 3 mm above it are dropped as the
#: support's own returns (one overhead-camera sigma at 0.9 m, six D405 sigmas).
SUPPORT_CLEARANCE_M = 0.003


# ---------------------------------------------------------------------------
# Small SE(3) helpers
# ---------------------------------------------------------------------------


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _expm_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    k = _skew(w)
    if th < 1e-12:
        return np.eye(3) + k
    return np.eye(3) + math.sin(th) / th * k + (1 - math.cos(th)) / th**2 * (k @ k)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _orthonormal_frame(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    a1 = _unit(a1)
    a2 = _unit(a2 - (a2 @ a1) * a1)
    return np.column_stack([a1, a2, np.cross(a1, a2)])


def _any_perpendicular(a: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(a, ref))


def signed_permutations() -> list[np.ndarray]:
    """The 24 proper rotations that map the coordinate axes onto themselves."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                M[row, col] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------


def parse_model(mesh: Any) -> dict:
    """Normalise a model spec to ``{"kind": "box"|"cylinder"|"mesh", ...}``.

    Accepted: a mesh path (``str``/``Path``); ``{"path": p, "scale": s}``;
    ``{"box": [x, y, z]}`` (full edge lengths, object frame at the centre, axes
    along the edges); ``{"cube": s}``; ``{"cylinder": [radius, height]}`` or
    ``{"cylinder": {"radius": r, "height": h}}`` (axis = object z).
    """
    if isinstance(mesh, str) or hasattr(mesh, "__fspath__"):
        return {"kind": "mesh", "path": str(mesh), "scale": 1.0}
    if not isinstance(mesh, dict):
        raise ValueError(f"unsupported model spec {mesh!r}")
    if "path" in mesh:
        return {"kind": "mesh", "path": str(mesh["path"]), "scale": float(mesh.get("scale", 1.0))}
    if "cube" in mesh:
        s = float(mesh["cube"])
        return {"kind": "box", "dims": np.array([s, s, s])}
    if "box" in mesh:
        dims = np.asarray(mesh["box"], dtype=np.float64).reshape(-1)
        if dims.shape != (3,) or np.any(dims <= 0):
            raise ValueError(f"box needs three positive edge lengths, got {mesh['box']!r}")
        return {"kind": "box", "dims": dims}
    if "cylinder" in mesh:
        c = mesh["cylinder"]
        r, h = (c["radius"], c["height"]) if isinstance(c, dict) else c
        if r <= 0 or h <= 0:
            raise ValueError(f"cylinder needs a positive radius and height, got {c!r}")
        return {"kind": "cylinder", "radius": float(r), "height": float(h)}
    raise ValueError(f"unsupported model spec {mesh!r}")


def symmetry_group(model: dict, symmetry: Any) -> tuple[list[np.ndarray], bool]:
    """Object-frame rotations that leave the part unchanged, and whether it also
    turns freely about its z axis (a cylinder).

    ``symmetry=None`` takes the primitive's own group (a box's 4, 8 or 24
    rotations depending on equal edges; a cylinder's free spin plus end flip)
    and none for a mesh. ``"none"`` forces no symmetry (a caller that will tell
    the ends apart by colour or pips); ``"z2"`` / ``"z4"`` are 180 / 90 deg
    turns about z; ``"box"`` is a box with distinct edges; a list of 3x3 or 4x4
    matrices is taken as given (identity is added).
    """
    eye = [np.eye(3)]
    if isinstance(symmetry, str):
        name = symmetry.lower()
        if name == "none":
            return eye, False
        if name in ("z2", "z4"):
            n = 2 if name == "z2" else 4
            return [_expm_so3(np.array([0.0, 0.0, 2 * math.pi * k / n])) for k in range(n)], False
        if name == "box":
            return [M for M in signed_permutations() if np.allclose(np.abs(M), np.eye(3))], False
        if name == "cylinder":
            return [np.eye(3), np.diag([1.0, -1.0, -1.0])], True
        raise ValueError(f"unknown symmetry {symmetry!r}")
    if symmetry is not None:
        mats = [np.asarray(m, dtype=np.float64)[:3, :3] for m in symmetry]
        if not any(np.allclose(m, np.eye(3)) for m in mats):
            mats = eye + mats
        return mats, False
    if model["kind"] == "box":
        d = model["dims"]
        return [
            M for M in signed_permutations() if np.allclose(np.abs(M) @ d, d, atol=EQUAL_DIM_TOL_M)
        ], False
    if model["kind"] == "cylinder":
        return [np.eye(3), np.diag([1.0, -1.0, -1.0])], True
    return eye, False


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------


def estimate_normals(P: np.ndarray, viewpoint: np.ndarray, k: int = NORMAL_K) -> np.ndarray:
    """PCA normals over ``k`` nearest neighbours, oriented toward ``viewpoint``."""
    from scipy.spatial import cKDTree

    k = int(min(k, len(P)))
    _, idx = cKDTree(P).query(P, k=k)
    nb = P[idx] - P[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]
    flip = np.einsum("ni,ni->n", n, viewpoint[None, :] - P) < 0
    n[flip] *= -1
    return n


def point_spacing(P: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if len(P) < 2:
        return 0.0
    d, _ = cKDTree(P).query(P, k=2)
    return float(np.median(d[:, 1]))


def _dominant_direction(N: np.ndarray, cos_tol: float, exclude: np.ndarray | None = None):
    """The most supported normal direction (mean of its cluster) and its support."""
    rng = np.random.default_rng(0)
    cand = N if len(N) <= 300 else N[rng.choice(len(N), 300, replace=False)]
    if exclude is not None:
        cand = cand[np.abs(cand @ exclude) < math.sin(math.radians(20))]
        N = N[np.abs(N @ exclude) < math.sin(math.radians(30))]
    if len(cand) == 0 or len(N) == 0:
        return None, 0
    support = (N @ cand.T > cos_tol).sum(axis=0)
    best = cand[int(np.argmax(support))]
    members = N[N @ best > cos_tol]
    return _unit(members.mean(axis=0)), len(members)


def _min_area_rect_axis(P2: np.ndarray) -> np.ndarray:
    """Long-edge direction of the minimum-area rectangle around 2D points."""
    from scipy.spatial import ConvexHull

    if len(P2) < 4:
        return np.array([1.0, 0.0])
    try:
        hull = P2[ConvexHull(P2).vertices]
    except Exception:  # degenerate (collinear) points
        return _unit(P2[-1] - P2[0]) if len(P2) > 1 else np.array([1.0, 0.0])
    best, best_axis = np.inf, np.array([1.0, 0.0])
    for i in range(len(hull)):
        e = _unit(hull[(i + 1) % len(hull)] - hull[i])
        f = np.array([-e[1], e[0]])
        pe, pf = hull @ e, hull @ f
        le, lf = np.ptp(pe), np.ptp(pf)
        if le * lf < best:
            best, best_axis = le * lf, e if le >= lf else f
    return best_axis


def candidate_frames(P: np.ndarray, N: np.ndarray, viewpoint: np.ndarray, up) -> list[np.ndarray]:
    """Up to three orthonormal frames whose axes estimate the part's edges."""
    frames = []
    centroid = P.mean(axis=0)
    cos15 = math.cos(math.radians(15))
    a1, s1 = _dominant_direction(N, cos15)
    if a1 is not None and s1 >= 10:
        a2, s2 = _dominant_direction(N, cos15, exclude=a1)
        if a2 is not None and s2 >= max(8, 0.05 * len(P)):
            frames.append(_orthonormal_frame(a1, a2))
        face = P[N @ a1 > cos15]
        frames.append(_frame_from_axis_and_rect(face if len(face) >= 8 else P, a1))
    _, vecs = np.linalg.eigh(np.cov((P - centroid).T))
    frames.append(_orthonormal_frame(vecs[:, 2], vecs[:, 1]))
    axis = (
        _unit(np.asarray(up, dtype=np.float64)) if up is not None else _unit(viewpoint - centroid)
    )
    frames.append(_frame_from_axis_and_rect(P, axis))
    return frames


def _frame_from_axis_and_rect(P: np.ndarray, a1: np.ndarray) -> np.ndarray:
    e1 = _any_perpendicular(a1)
    e2 = np.cross(a1, e1)
    P2 = np.column_stack([P @ e1, P @ e2])
    d = _min_area_rect_axis(P2)
    return _orthonormal_frame(a1, d[0] * e1 + d[1] * e2)


# ---------------------------------------------------------------------------
# Analytic surfaces: closest visible point and its normal, in object coordinates
# ---------------------------------------------------------------------------


class BoxSurface:
    kind = "box"

    def __init__(self, dims: np.ndarray):
        self.h = np.asarray(dims, dtype=np.float64) / 2.0
        self.faces = [(k, s) for k in range(3) for s in (1.0, -1.0)]

    def resting_contacts(self, down_obj: np.ndarray) -> np.ndarray:
        """The four corners of the face that points most nearly down: a box at
        rest stands on a whole face."""
        k = int(np.argmax(np.abs(down_obj)))
        s = 1.0 if down_obj[k] > 0 else -1.0
        corners = np.array(list(itertools.product(*[(-h, h) for h in self.h])))
        return corners[np.isclose(corners[:, k], s * self.h[k])]

    def visible_faces(self, v_obj: np.ndarray | None) -> list[tuple[int, float]]:
        if v_obj is None:
            return self.faces
        out = []
        for k, s in self.faces:
            n = np.zeros(3)
            n[k] = s
            c = n * self.h[k]
            if n @ _unit(v_obj - c) > VISIBLE_COS:
                out.append((k, s))
        return out

    def closest(self, q: np.ndarray, v_obj: np.ndarray | None):
        faces = self.visible_faces(v_obj)
        best_d = np.full(len(q), np.inf)
        best_c = np.zeros_like(q)
        best_n = np.zeros_like(q)
        for k, s in faces:
            c = np.clip(q, -self.h, self.h)
            c[:, k] = s * self.h[k]
            d = np.linalg.norm(q - c, axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_c[better] = c[better]
            n = np.zeros(3)
            n[k] = s
            best_n[better] = n
        return best_c, best_n

    def sample_visible(self, v_obj: np.ndarray | None, spacing: float):
        """Grid samples on the visible faces with their outward normals and areas."""
        pts, nrm, area = [], [], []
        for k, s in self.visible_faces(v_obj):
            i, j = [a for a in range(3) if a != k]
            ni = max(2, int(math.ceil(2 * self.h[i] / spacing)))
            nj = max(2, int(math.ceil(2 * self.h[j] / spacing)))
            gi = (np.arange(ni) + 0.5) / ni * 2 * self.h[i] - self.h[i]
            gj = (np.arange(nj) + 0.5) / nj * 2 * self.h[j] - self.h[j]
            Gi, Gj = np.meshgrid(gi, gj, indexing="ij")
            q = np.zeros((Gi.size, 3))
            q[:, i], q[:, j], q[:, k] = Gi.ravel(), Gj.ravel(), s * self.h[k]
            n = np.zeros(3)
            n[k] = s
            pts.append(q)
            nrm.append(np.tile(n, (len(q), 1)))
            area.append(np.full(len(q), 4 * self.h[i] * self.h[j] / len(q)))
        return np.vstack(pts), np.vstack(nrm), np.concatenate(area)


class CylinderSurface:
    kind = "cylinder"

    def __init__(self, radius: float, height: float):
        self.r = float(radius)
        self.hz = float(height) / 2.0

    def resting_contacts(self, down_obj: np.ndarray) -> np.ndarray:
        """A cylinder at rest stands on a cap (8 rim points of the lower cap) or
        lies on a side line (its two ends)."""
        if abs(down_obj[2]) > math.cos(math.radians(45)):
            s = 1.0 if down_obj[2] > 0 else -1.0
            th = np.arange(8) / 8 * 2 * math.pi
            return np.column_stack(
                [self.r * np.cos(th), self.r * np.sin(th), np.full(8, s * self.hz)]
            )
        radial = _unit(np.array([down_obj[0], down_obj[1], 0.0])) * self.r
        return np.array([radial + [0.0, 0.0, self.hz], radial - [0.0, 0.0, self.hz]])

    def _cap_visible(self, s: float, v_obj: np.ndarray | None) -> bool:
        if v_obj is None:
            return True
        return s * _unit(v_obj - np.array([0.0, 0.0, s * self.hz]))[2] > VISIBLE_COS

    def closest(self, q: np.ndarray, v_obj: np.ndarray | None):
        rho = np.linalg.norm(q[:, :2], axis=1)
        radial = np.zeros((len(q), 3))
        safe = rho > 1e-9
        radial[safe, :2] = q[safe, :2] / rho[safe, None]
        radial[~safe, 0] = 1.0
        c_side = np.column_stack([self.r * radial[:, :2], np.clip(q[:, 2], -self.hz, self.hz)])
        d_side = np.linalg.norm(q - c_side, axis=1)
        if v_obj is not None:
            back = np.einsum("ni,ni->n", radial, _unit_rows(v_obj[None, :] - c_side)) <= VISIBLE_COS
            d_side[back] = np.inf
        best_c, best_n, best_d = c_side, radial, d_side
        for s in (1.0, -1.0):
            if not self._cap_visible(s, v_obj):
                continue
            scale = np.minimum(1.0, self.r / np.maximum(rho, 1e-12))
            c = np.column_stack([q[:, :2] * scale[:, None], np.full(len(q), s * self.hz)])
            d = np.linalg.norm(q - c, axis=1)
            better = d < best_d
            best_c = np.where(better[:, None], c, best_c)
            best_n = np.where(better[:, None], np.array([0.0, 0.0, s]), best_n)
            best_d = np.where(better, d, best_d)
        # A point with no visible surface (viewpoint inside the part) keeps the side.
        none = ~np.isfinite(best_d)
        best_c[none], best_n[none] = c_side[none], radial[none]
        return best_c, best_n

    def sample_visible(self, v_obj: np.ndarray | None, spacing: float):
        pts, nrm, area = [], [], []
        nt = max(8, int(math.ceil(2 * math.pi * self.r / spacing)))
        nz = max(2, int(math.ceil(2 * self.hz / spacing)))
        th = (np.arange(nt) + 0.5) / nt * 2 * math.pi
        zz = (np.arange(nz) + 0.5) / nz * 2 * self.hz - self.hz
        T, Z = np.meshgrid(th, zz, indexing="ij")
        n = np.column_stack([np.cos(T.ravel()), np.sin(T.ravel()), np.zeros(T.size)])
        q = np.column_stack([self.r * n[:, :2], Z.ravel()])
        keep = np.ones(len(q), bool)
        if v_obj is not None:
            keep = np.einsum("ni,ni->n", n, _unit_rows(v_obj[None, :] - q)) > VISIBLE_COS
        pts.append(q[keep])
        nrm.append(n[keep])
        area.append(np.full(int(keep.sum()), 2 * math.pi * self.r * 2 * self.hz / len(q)))
        for s in (1.0, -1.0):
            if not self._cap_visible(s, v_obj):
                continue
            g = np.arange(-self.r, self.r + 1e-12, spacing)
            X, Y = np.meshgrid(g, g, indexing="ij")
            inside = X.ravel() ** 2 + Y.ravel() ** 2 <= self.r**2
            qc = np.column_stack(
                [X.ravel()[inside], Y.ravel()[inside], np.full(inside.sum(), s * self.hz)]
            )
            pts.append(qc)
            nrm.append(np.tile([0.0, 0.0, s], (len(qc), 1)))
            area.append(np.full(len(qc), math.pi * self.r**2 / max(1, len(qc))))
        return np.vstack(pts), np.vstack(nrm), np.concatenate(area)


def _unit_rows(V: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(n, 1e-12)


# ---------------------------------------------------------------------------
# Visibility-aware point-to-plane ICP against an analytic surface
# ---------------------------------------------------------------------------


def icp_analytic(
    P: np.ndarray,
    surface,
    R: np.ndarray,
    t: np.ndarray,
    viewpoint: np.ndarray | None,
    delta: float,
    iters: int,
    support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on point-to-plane residuals plus an edge-overflow term.

    Each point is matched to the closest point on a surface patch facing the
    viewpoint. Its residual has a normal row (point-to-plane) and, when the
    point lies beyond the patch's edge, a tangential row along the overflow:
    point-to-plane alone leaves a single observed face free to slide. Huber
    weights at ``delta``; points beyond ``TRIM * delta`` are dropped after the
    third iteration.

    ``support`` (``[nx, ny, nz, d]``, ``n . x + d = 0``, ``n`` out of the
    support) adds rows that put the part's resting contacts (a box's lowest
    face, a cylinder's lower cap or lowest side line) on that plane.
    """
    m = P.mean(axis=0)
    Pc = P - m
    if support is not None:
        sn = _unit(np.asarray(support[:3], dtype=np.float64))
        sd = float(support[3]) / float(np.linalg.norm(support[:3]))
    for it in range(iters):
        q = (P - t) @ R
        v_obj = None if viewpoint is None else R.T @ (viewpoint - t)
        cq, nq = surface.closest(q, v_obj)
        c = cq @ R.T + t
        n = nq @ R.T
        e = P - c
        dist = np.linalg.norm(e, axis=1)
        rn = np.einsum("ni,ni->n", e, n)
        tang = e - rn[:, None] * n
        tl = np.linalg.norm(tang, axis=1)
        w = np.where(dist <= delta, 1.0, delta / np.maximum(dist, 1e-12))
        if it >= 3:
            w[dist > TRIM * delta] = 0.0
        has_t = tl > 1e-9
        u = np.zeros_like(tang)
        u[has_t] = tang[has_t] / tl[has_t, None]
        A = np.vstack(
            [
                np.hstack([np.cross(Pc, n), n]),
                np.hstack([np.cross(Pc[has_t], u[has_t]), u[has_t]]),
            ]
        )
        r = np.concatenate([rn, tl[has_t]])
        W = np.concatenate([w, w[has_t]])
        if support is not None:
            cw = surface.resting_contacts(-(R.T @ sn)) @ R.T + t
            rs = cw @ sn + sd
            p0c = cw - rs[:, None] * sn[None, :] - m
            As = np.hstack([-np.cross(p0c, sn[None, :]), -np.tile(sn, (len(rs), 1))])
            ws = np.full(len(rs), SUPPORT_WEIGHT * w.sum() / len(rs))
            A, r, W = np.vstack([A, As]), np.concatenate([r, rs]), np.concatenate([W, ws])
        H = A.T @ (A * W[:, None])
        if np.trace(H) <= 0:
            break
        H += np.eye(6) * 1e-9 * np.trace(H)
        xi = -np.linalg.solve(H, A.T @ (W * r))
        dR = _expm_so3(xi[:3])
        dT = _rt_to_T(dR, m + xi[3:] - dR @ m)
        Tb = np.linalg.inv(dT) @ _rt_to_T(R, t)
        R, t = Tb[:3, :3], Tb[:3, 3]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.norm(xi[:3]) < 1e-6 and np.linalg.norm(xi[3:]) < 1e-7:
            break
    return R, t


def surface_residuals(P, surface, R, t, viewpoint) -> np.ndarray:
    q = (P - t) @ R
    v_obj = None if viewpoint is None else R.T @ (viewpoint - t)
    cq, _ = surface.closest(q, v_obj)
    return np.linalg.norm(q - cq, axis=1)


def coverage(P_tree, surface, R, t, viewpoint, spacing: float, radius: float) -> float:
    """Share of the model's camera-facing area (weighted by how squarely it faces
    the camera) that has a data point within ``radius``."""
    v_obj = None if viewpoint is None else R.T @ (viewpoint - t)
    q, n, area = surface.sample_visible(v_obj, spacing)
    if len(q) == 0:
        return 0.0
    if v_obj is not None:
        w = area * np.clip(np.einsum("ni,ni->n", n, _unit_rows(v_obj[None, :] - q)), 0.0, 1.0)
    else:
        w = area
    if w.sum() <= 0:
        return 0.0
    d, _ = P_tree.query(q @ R.T + t, k=1)
    return float((w * (d <= radius)).sum() / w.sum())


def _initial_translation(P, surface, R, viewpoint) -> np.ndarray:
    """Centre from the extreme points on the camera-facing side of each axis."""
    centroid = P.mean(axis=0)
    q = (P - centroid) @ R
    view = R.T @ _unit(viewpoint - centroid)
    if surface.kind == "box":
        half = surface.h
    else:
        half = np.array([surface.r, surface.r, surface.hz])
    c = np.zeros(3)
    for k in range(3):
        lo, hi = np.percentile(q[:, k], [2, 98])
        if hi - lo >= 1.6 * half[k]:  # both sides seen: centre between them
            c[k] = (lo + hi) / 2
        elif view[k] >= 0:  # the +k face looks at the camera: centre behind it
            c[k] = hi - half[k]
        else:
            c[k] = lo + half[k]
    return centroid + R @ c


# ---------------------------------------------------------------------------
# Symmetry: canonical representative and the equivalent set
# ---------------------------------------------------------------------------


def canonicalize(
    R: np.ndarray,
    group: list[np.ndarray],
    continuous_z: bool,
    up: np.ndarray,
    forward: np.ndarray,
    init_R: np.ndarray | None,
) -> np.ndarray:
    """Pick one of the symmetric equivalents ``R @ S``.

    With ``init`` the one nearest it; otherwise the one whose z axis is most
    along ``up`` and then whose x axis is most along ``forward``. For a part
    that turns freely about z the x axis is set to ``forward`` projected
    perpendicular to z.
    """
    cands = [R @ S for S in group]
    if init_R is not None:
        best = max(cands, key=lambda C: np.trace(init_R.T @ C))
    else:
        best = max(cands, key=lambda C: (round(float(C[:, 2] @ up), 6), float(C[:, 0] @ forward)))
    if continuous_z:
        z = best[:, 2]
        ref = init_R[:, 0] if init_R is not None else forward
        x = ref - (ref @ z) * z
        x = _unit(x) if np.linalg.norm(x) > 1e-6 else _any_perpendicular(z)
        best = np.column_stack([x, np.cross(z, x), z])
    return best


def yaw_ambiguity_deg(group: list[np.ndarray], continuous_z: bool) -> float:
    """Smallest turn about the object's z that maps the part onto itself (0 = none,
    360 = any turn)."""
    if continuous_z:
        return 360.0
    about_z = [S for S in group if np.allclose(S[:, 2], [0.0, 0.0, 1.0])]
    return 360.0 / len(about_z) if len(about_z) > 1 else 0.0


def _same_up_to_symmetry(Ra, Rb, group, continuous_z, tol_deg=10.0) -> bool:
    if continuous_z:
        return abs(abs(Ra[:, 2] @ Rb[:, 2])) > math.cos(math.radians(tol_deg))
    return min(rotation_angle_deg(Ra @ S, Rb) for S in group) < tol_deg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _prepare(points, viewpoint, up, forward, max_points):
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) > max_points:
        P = P[np.random.default_rng(0).choice(len(P), max_points, replace=False)]
    if viewpoint is not None:
        v = np.asarray(viewpoint, dtype=np.float64).reshape(3)
    elif up is not None:
        v = (
            P.mean(axis=0) + _unit(np.asarray(up, dtype=np.float64)) * 1.0
            if len(P)
            else np.zeros(3)
        )
    else:
        v = np.zeros(3)
    up_v = _unit(np.asarray(up, dtype=np.float64)) if up is not None else None
    fwd = (
        np.asarray(forward, dtype=np.float64) if forward is not None else np.array([1.0, 0.0, 0.0])
    )
    return P, v, up_v, _unit(fwd)


def _empty_result(method: str, n: int, t0: float, reason: str) -> dict:
    return {
        "found": False,
        "status": "poor_fit",
        "reason": reason,
        "T": np.eye(4),
        "fitness": 0.0,
        "rmse": 0.0,
        "coverage": 0.0,
        "inlier_threshold_m": 0.0,
        "method": method,
        "symmetry_order": 1,
        "continuous_symmetry": False,
        "yaw_ambiguity_deg": 0.0,
        "equivalent_T": [],
        "ambiguous_fit": False,
        "alternative": None,
        "n_points": n,
        "n_dropped_support": 0,
        "hypotheses": 0,
        "runtime_s": time.perf_counter() - t0,
    }


def register(
    points,
    mesh,
    init=None,
    symmetry=None,
    viewpoint=None,
    up=None,
    forward=None,
    noise_m=None,
    method: str = "auto",
    max_points: int = MAX_POINTS,
    support_plane=None,
    support_clearance_m: float = SUPPORT_CLEARANCE_M,
) -> dict:
    """Pose of ``mesh`` in the frame of ``points``; see ``tools.register_mesh``."""
    t0 = time.perf_counter()
    model = parse_model(mesh)
    P, v, up_v, fwd = _prepare(points, viewpoint, up, forward, max_points)
    if len(P) < 12:
        return _empty_result(method, len(P), t0, f"only {len(P)} finite points")
    if method == "auto":
        method = "primitive" if model["kind"] in ("box", "cylinder") else "global"
    if method == "primitive" and model["kind"] == "mesh":
        raise ValueError("method='primitive' needs a box, cube or cylinder spec")
    init_T = None if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)
    support = (
        None if support_plane is None else np.asarray(support_plane, dtype=np.float64).reshape(4)
    )
    n_dropped = 0
    if support is not None:
        n_in = len(P)
        height = (P @ support[:3] + support[3]) / np.linalg.norm(support[:3])
        P = P[height > support_clearance_m]
        n_dropped = n_in - len(P)
        if len(P) < 12:
            return _empty_result(
                method, len(P), t0, f"only {len(P)} points above the support plane"
            )
    group, cont = symmetry_group(model, symmetry)

    spacing = point_spacing(P)
    if noise_m is not None:
        delta = float(np.clip(3.0 * noise_m, INLIER_FLOOR_M, INLIER_CEIL_M))
    else:
        delta = None  # estimated after the first fit

    if method == "primitive":
        shape_group, shape_cont = symmetry_group(model, None)
        res = _register_primitive(
            P, model, init_T, shape_group, shape_cont, v, up_v, spacing, delta, support
        )
    elif method == "global":
        res = _register_global(P, model, init_T, v, spacing, delta)
    else:
        raise ValueError(f"unknown method {method!r}")

    R, t = res["T"][:3, :3], res["T"][:3, 3]
    up_ref = up_v if up_v is not None else _unit(v - t)
    Rc = canonicalize(R, group, cont, up_ref, fwd, None if init_T is None else init_T[:3, :3])
    T = _rt_to_T(Rc, t)
    equivalent = [] if cont else [_rt_to_T(Rc @ S, t) for S in group]
    alt = res.get("alternative")
    ambiguous_fit = bool(alt is not None and alt["score_gap"] < AMBIGUOUS_SCORE_GAP)
    ok = res["fitness"] >= MIN_FITNESS and res["coverage"] >= MIN_COVERAGE
    status = "registered" if ok and not ambiguous_fit else ("ambiguous_fit" if ok else "poor_fit")
    return {
        "found": True,
        "status": status,
        "reason": "",
        "T": T,
        "fitness": res["fitness"],
        "rmse": res["rmse"],
        "coverage": res["coverage"],
        "inlier_threshold_m": res["delta"],
        "method": res["method"],
        "symmetry_order": 0 if cont else len(group),
        "continuous_symmetry": cont,
        "yaw_ambiguity_deg": yaw_ambiguity_deg(group, cont),
        "equivalent_T": equivalent,
        "ambiguous_fit": ambiguous_fit,
        "alternative": alt,
        "n_points": len(P),
        "n_dropped_support": n_dropped,
        "hypotheses": res["hypotheses"],
        "runtime_s": time.perf_counter() - t0,
    }


def _make_surface(model):
    if model["kind"] == "box":
        return BoxSurface(model["dims"])
    return CylinderSurface(model["radius"], model["height"])


def _score(P, tree, surface, R, t, v, spacing, delta):
    d = surface_residuals(P, surface, R, t, v)
    inl = d <= delta
    fit = float(inl.mean())
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 0.0
    radius = max(0.5 * delta, 2.0 * spacing, INLIER_FLOOR_M)
    cov = coverage(tree, surface, R, t, v, max(spacing, 0.0005), radius)
    f1 = 2 * fit * cov / (fit + cov) if fit + cov > 0 else 0.0
    # Truncated-quadratic data cost in [0, 1]: separates fits whose residuals are
    # all "inliers" at a loose threshold but sit a few mm apart on average.
    cost = float(np.mean(np.minimum(d, delta) ** 2)) / delta**2
    return {"fitness": fit, "rmse": rmse, "coverage": cov, "score": f1 - cost, "d": d}


def _register_primitive(P, model, init_T, group, cont, v, up_v, spacing, delta, support):
    from scipy.spatial import cKDTree

    surface = _make_surface(model)
    tree = cKDTree(P)
    size = (
        float(np.max(model["dims"]))
        if model["kind"] == "box"
        else 2 * max(model["radius"], surface.hz)
    )
    d0 = (
        delta
        if delta is not None
        else float(np.clip(max(2.0 * spacing, 0.02 * size), 0.002, INLIER_CEIL_M))
    )

    hyps: list[tuple[np.ndarray, np.ndarray]] = []
    if init_T is not None:
        hyps.append((init_T[:3, :3], init_T[:3, 3]))
    else:
        N = estimate_normals(P, v)
        frames = candidate_frames(P, N, v, up_v)
        if model["kind"] == "box":
            perms = _coset_representatives(group)
        else:  # the cylinder axis can lie on any frame axis
            perms = [
                np.eye(3),
                np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]]),
                np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]]),
            ]
            frames.append(_frame_from_axis_and_rect(P, _cylinder_axis_from_normals(N)))
        for F in frames:
            for M in perms:
                R = F @ M
                hyps.append((R, _initial_translation(P, surface, R, v)))

    Ps = (
        P
        if len(P) <= SCREEN_POINTS
        else P[np.random.default_rng(1).choice(len(P), SCREEN_POINTS, replace=False)]
    )
    screened = []
    for R, t in hyps:
        R1, t1 = icp_analytic(Ps, surface, R, t, v, d0, SCREEN_ITERS, support)
        screened.append((R1, t1, _score(P, tree, surface, R1, t1, v, spacing, d0)["score"]))
    screened.sort(key=lambda h: -h[2])

    def distinct(Ra, ta, Rb, tb):
        return np.linalg.norm(ta - tb) > 0.3 * size or not _same_up_to_symmetry(
            Ra, Rb, group, cont, DISTINCT_DEG
        )

    seeds: list[tuple[np.ndarray, np.ndarray]] = []
    for R, t, _ in screened:
        if all(distinct(R, t, Rs, ts) for Rs, ts in seeds):
            seeds.append((R, t))
        if len(seeds) == REFINE_SEEDS:
            break

    R0, t0 = icp_analytic(P, surface, *seeds[0], v, d0, REFINE_ITERS, support)
    if delta is None:  # robust noise estimate from the leading fit's residuals
        d = surface_residuals(P, surface, R0, t0, v)
        near = d[d < 3 * d0]
        sigma = 1.4826 * float(np.median(near)) if len(near) else d0 / 3
        dl = float(np.clip(3.0 * sigma, max(INLIER_FLOOR_M, spacing), INLIER_CEIL_M))
    else:
        dl = delta
    refined = []
    for i, (R, t) in enumerate(seeds):
        if i == 0:
            R, t = R0, t0
        R1, t1 = icp_analytic(P, surface, R, t, v, dl, REFINE_ITERS, support)
        refined.append((R1, t1, _score(P, tree, surface, R1, t1, v, spacing, dl)))
    refined.sort(key=lambda h: -h[2]["score"])
    R, t, best = refined[0]

    alt = None
    for R2, t2, s2 in refined[1:]:
        if not distinct(R, t, R2, t2):
            continue
        alt = {
            "fitness": s2["fitness"],
            "rmse": s2["rmse"],
            "coverage": s2["coverage"],
            "score_gap": best["score"] - s2["score"],
            "rotation_deg": min(rotation_angle_deg(R @ S, R2) for S in group),
            "translation_m": float(np.linalg.norm(t2 - t)),
            "T": _rt_to_T(R2, t2),
        }
        break
    return {
        "T": _rt_to_T(R, t),
        "fitness": best["fitness"],
        "rmse": best["rmse"],
        "coverage": best["coverage"],
        "delta": dl,
        "method": f"primitive_{model['kind']}",
        "alternative": alt,
        "hypotheses": len(hyps),
    }


def _cylinder_axis_from_normals(N: np.ndarray) -> np.ndarray:
    """Side normals are perpendicular to the axis: the scatter's weakest direction."""
    _, vecs = np.linalg.eigh(N.T @ N)
    return vecs[:, 0]


def _coset_representatives(group: list[np.ndarray]) -> list[np.ndarray]:
    """One axis assignment per class of assignments that differ by a symmetry."""
    reps, seen = [], set()
    for M in signed_permutations():
        key = min(tuple(np.round(M @ S).astype(int).ravel()) for S in group)
        if key not in seen:
            seen.add(key)
            reps.append(M)
    return reps


# ---------------------------------------------------------------------------
# Mesh path: Open3D FPFH + RANSAC, then point-to-plane ICP on facing points
# ---------------------------------------------------------------------------


def _quiet_open3d():
    import open3d as o3d

    # Open3D warns on stdout, which is the tool server's RPC channel.
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return o3d


def _model_mesh(model):
    o3d = _quiet_open3d()

    if model["kind"] == "mesh":
        m = o3d.io.read_triangle_mesh(model["path"])
        if len(m.vertices) == 0:
            raise ValueError(f"could not read a mesh from {model['path']!r}")
        if model["scale"] != 1.0:
            m.scale(model["scale"], center=np.zeros(3))
        return m
    if model["kind"] == "box":
        d = model["dims"]
        m = o3d.geometry.TriangleMesh.create_box(*d)
        m.translate(-d / 2)
        return m
    return o3d.geometry.TriangleMesh.create_cylinder(
        model["radius"], model["height"], resolution=48
    )


def _register_global(P, model, init_T, v, spacing, delta):
    o3d = _quiet_open3d()

    mesh = _model_mesh(model)
    mesh.compute_vertex_normals()
    ext = mesh.get_axis_aligned_bounding_box().get_extent()
    size = float(np.linalg.norm(ext))
    voxel = max(size / 25.0, spacing, 0.0008)
    n_model = int(np.clip(mesh.get_surface_area() / (voxel / 2) ** 2, 2000, 20000))
    model_pc = mesh.sample_points_uniformly(n_model, use_triangle_normal=True)
    model_pts = np.asarray(model_pc.points)
    model_nrm = np.asarray(model_pc.normals)
    dl = (
        delta
        if delta is not None
        else float(np.clip(max(1.5 * spacing, voxel / 2), 0.002, INLIER_CEIL_M))
    )

    scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    scene.orient_normals_towards_camera_location(v)

    if init_T is None:
        src = scene.voxel_down_sample(voxel)
        dst = model_pc.voxel_down_sample(voxel)
        for pc in (src, dst):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
        src.orient_normals_towards_camera_location(v)
        feat = o3d.pipelines.registration.compute_fpfh_feature
        prm = o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100)
        f_src, f_dst = feat(src, prm), feat(dst, prm)
        dist = 1.5 * voxel
        o3d.utility.random.seed(0)
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src,
            dst,
            f_src,
            f_dst,
            True,
            dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        T_ms = np.asarray(ransac.transformation)
    else:
        T_ms = np.linalg.inv(init_T)

    for _ in range(2):  # ICP against model points that face the camera under the current pose
        v_obj = T_ms[:3, :3] @ v + T_ms[:3, 3]
        facing = (
            np.einsum("ni,ni->n", model_nrm, _unit_rows(v_obj[None, :] - model_pts)) > VISIBLE_COS
        )
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts[facing]))
        tgt.normals = o3d.utility.Vector3dVector(model_nrm[facing])
        icp = o3d.pipelines.registration.registration_icp(
            scene,
            tgt,
            3 * dl,
            T_ms,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=dl)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=REFINE_ITERS),
        )
        T_ms = np.asarray(icp.transformation)

    T = np.linalg.inv(T_ms)
    from scipy.spatial import cKDTree

    v_obj = T_ms[:3, :3] @ v + T_ms[:3, 3]
    facing = np.einsum("ni,ni->n", model_nrm, _unit_rows(v_obj[None, :] - model_pts)) > VISIBLE_COS
    q = P @ T_ms[:3, :3].T + T_ms[:3, 3]
    d, _ = cKDTree(model_pts[facing]).query(q, k=1)
    inl = d <= dl
    fit = float(inl.mean())
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 0.0
    vis = model_pts[facing]
    w = np.clip(np.einsum("ni,ni->n", model_nrm[facing], _unit_rows(v_obj[None, :] - vis)), 0, 1)
    dm, _ = cKDTree(q).query(vis, k=1)
    cov = float((w * (dm <= max(dl, 2 * spacing))).sum() / max(w.sum(), 1e-12))
    return {
        "T": T,
        "fitness": fit,
        "rmse": rmse,
        "coverage": cov,
        "delta": dl,
        "method": "global_fpfh_icp" if init_T is None else "icp_from_init",
        "alternative": None,
        "hypotheses": 1,
    }
