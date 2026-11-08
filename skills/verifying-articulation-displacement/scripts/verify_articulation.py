# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/close_drawer/libero/policy.py (_visual_drawer_close) and
# roborsi/embodied/skills/base/pull_drawer/libero/policy.py (_visual_handle_pull,
# _resolve_drawer_handle, _refine_drawer_handle_geometry, _face_plane, _dense_face_plane,
# _fit_face_plane, _verify_drawer_handle_pixel), Apache-2.0.
# Changes: the frame is a hub Observation camera (RGB, metric depth, intrinsics, camera
# pose) deprojected here instead of the simulator's pixel_to_world; RoboRSI's pixel radii,
# set for LIBERO's 256-px agentview, are metric and turned into pixels at the handle's
# depth; the face normal is signed toward the camera instead of the robot base; the
# handle is pointed with gemini-er.point or molmo.point_prompt (RoboRSI: its own VLM
# pointer), tightened with sam3.segment_point (RoboRSI's SAM centroid refinement) and,
# optionally, confirmed with vlm.query_yes_no on a marked frame and crop (RoboRSI's
# VLM check, which it always ran); RoboRSI's drawer-layer scan and VLM pixel correction
# are dropped; a seed pixel without depth takes the median of its valid neighbours;
# handle candidates must also lie near the seed in 3D, and the handle point is the
# median of the handle's front instead of its single most protruding pixel;
# the triplet plane search is vectorised. The verdict generalises
# _visual_drawer_close: a prismatic joint is measured along any pull direction and
# either sign (open or close), a revolute joint by the handle's angle about the hinge
# with the face normal expected to turn with it, and the result is routed to named
# exits (moved, moved_short, not_moved, moved_wrong_way, inconsistent, not_seen)
# where RoboRSI returned a distance or None.
"""Re-localise a drawer's or door's handle after actuation and measure how far it moved."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from itertools import combinations
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

LIBERO_PIXEL = 0.0032
"""What one pixel of RoboRSI's camera spans at the cabinet [m]: LIBERO's 256-px agentview
(fovy 45 deg, f = 309 px) about 1 m from the drawers. RoboRSI's pixel radii below are
converted to metres with it, then to this camera's pixels at the handle's depth."""

FACE_RING_RADII = (10 * LIBERO_PIXEL, 16 * LIBERO_PIXEL, 24 * LIBERO_PIXEL)
"""Rings round the handle sampled for the face plane (RoboRSI: 10, 16 and 24 px, eight
points each)."""

HANDLE_SEARCH_RADIUS = 8 * LIBERO_PIXEL
"""Half-side of the window searched for the handle round the seed (RoboRSI: 8 px). Its
points must also lie within this plus the largest protrusion of the seed's world point (not
in RoboRSI): seen steeply from above, the window takes in the table under a door."""

DENSE_FACE_RADIUS = 16 * LIBERO_PIXEL
"""Outer half-side of the square annulus sampled when the rings fit no plane (RoboRSI:
16 px; its inner half-side is HANDLE_SEARCH_RADIUS)."""

MAX_PLANE_FIT_POINTS = 64
"""Points the triplet search draws its candidate planes from (RoboRSI)."""

PLANE_INLIER = 0.015
"""Distance from a candidate plane that counts as on it [m] (RoboRSI)."""

MIN_PLANE_INLIERS = 5
"""Least inliers for a candidate plane (RoboRSI)."""

MAX_CANDIDATE_TILT = 0.65
"""A triplet plane whose unit normal has a larger vertical component is skipped when the
face is vertical (RoboRSI)."""

MAX_FACE_TILT = 0.25
"""Largest vertical component of a vertical face's fitted normal (RoboRSI)."""

MAX_PLANARITY = 0.35
"""Largest ratio of the two smallest eigenvalues of the inliers' covariance: above it the
inliers are a line or a blob, not a face (RoboRSI)."""

MIN_NORMAL_SIGN_ALIGNMENT = 0.05
"""Least |cosine| between the face normal and the direction to the camera; a face seen
this edge-on gives no trustworthy sign (RoboRSI used the robot base)."""

PROTRUSION = (0.004, 0.06)
"""A pixel this far proud of the face plane is handle [m] (RoboRSI)."""

ADJACENCY = 2.5
"""Candidate pixels within this many grid steps are one component (RoboRSI: 2.5 px)."""

MIN_COMPONENT = 3
"""Least pixels in a handle component (RoboRSI)."""

FRONT_BAND = 0.005
"""The handle point is the median of the component's pixels within this of its most
protruding one [m]; RoboRSI took that single pixel."""

SAM_MAX_SHIFT = 24 * LIBERO_PIXEL
"""A SAM mask centroid further than this from the pointed pixel is ignored (RoboRSI's
ROBORSI_SAM_REFINE_MAX_SHIFT_PX = 24 px)."""

VLM_CROP_RADIUS = 24 * LIBERO_PIXEL
"""Half-side of the magnified crop shown to the VLM (RoboRSI: 24 px)."""

MIN_SAM_PIXELS = 10
"""A SAM mask with fewer pixels is ignored (RoboRSI)."""

NORMAL_DOT_MIN = 0.9
"""Least cosine between the face normal after and the one expected (RoboRSI's 0.9: the
same face, about 26 deg)."""

MAX_TRAVEL = 0.25
"""Largest drawer travel believed [m] (RoboRSI); more is another handle."""

MAX_TRANSVERSE = 0.07
"""Largest drift off the articulation [m] (RoboRSI's 0.07 across the pull; for a door,
the change of the handle's radius from the hinge line and of its height along it)."""

MIN_MOTION = 0.03
"""Handle travel under this is no motion [m] (RoboRSI's _MIN_VISUAL_PULL); a door's
angle threshold is this arc at the handle's radius."""

MIN_FRACTION = 0.7
"""Share of the expected opening that counts as moved (RoboRSI's close: 0.14 m measured of
a 0.20 m command)."""

MIN_RADIUS = 0.01
"""A handle closer than this to the hinge line has no arc [m] (planning-articulation-opening)."""

DEPTH_RANGE = (0.015, 20.0)
"""Valid depth [m] (geometry.mask_to_world_points' bounds)."""


class Output(TypedDict):
    route: str
    detail: str
    joint_type: str
    seen: bool
    measured: float
    expected: float
    fraction: float
    transverse: float
    normal_dot: float
    normal_angle: float
    radius: float
    handle_before: dict[str, float]
    handle_after: dict[str, float]
    face_normal_after: dict[str, float]
    handle_pixel: dict[str, float]


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _vec(v: np.ndarray | None) -> dict[str, float]:
    v = np.zeros(3) if v is None else v
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError(f"{name} must be a non-zero vector")
    return v / n


def _pose_matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p[k]) for k in ("x", "y", "z")]
        return m
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4):
        raise ValueError("camera pose is an Se3Pose or a 4x4 camera-to-world matrix")
    return m


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


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


class Frame:
    """One RGB-D frame deprojected to world points (NaN where the depth is invalid)."""

    def __init__(self, camera: dict[str, Any]):
        self.rgb = np.asarray(camera["rgb"])
        depth = np.asarray(camera["depth"], dtype=float)
        if depth.ndim == 3:
            depth = depth[..., 0]
        k = np.asarray(camera["intrinsics"], dtype=float)
        t = _pose_matrix(camera["pose"])
        self.height, self.width = depth.shape
        # Pointer and SAM pixels are the RGB image's; geometry runs on the depth image's.
        self.sx = self.width / self.rgb.shape[1]
        self.sy = self.height / self.rgb.shape[0]
        self.fx = float(k[0, 0])
        valid = np.isfinite(depth) & (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
        z = np.where(valid, depth, np.nan)
        v, u = np.mgrid[0 : self.height, 0 : self.width]
        cam = np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1)
        self.points = cam @ t[:3, :3].T + t[:3, 3]
        self.depth = z
        self.camera_position = t[:3, 3].copy()

    def sample(self, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points at the (rounded) depth pixels inside the image with valid depth,
        and those pixels as an (N, 2) array."""
        us, vs = np.rint(us).astype(int), np.rint(vs).astype(int)
        inside = (us >= 0) & (us < self.width) & (vs >= 0) & (vs < self.height)
        us, vs = us[inside], vs[inside]
        pts = self.points[vs, us]
        ok = np.isfinite(pts).all(axis=1)
        return pts[ok], np.stack((us[ok], vs[ok]), axis=1)

    def seed(self, u: float, v: float) -> tuple[np.ndarray, float] | None:
        """The world point and depth at a depth pixel; a pixel without depth takes the
        median of the valid pixels within 2 of it."""
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < self.width and 0 <= iv < self.height):
            return None
        if np.isfinite(self.depth[iv, iu]):
            return self.points[iv, iu].copy(), float(self.depth[iv, iu])
        window = (slice(max(0, iv - 2), iv + 3), slice(max(0, iu - 2), iu + 3))
        z = self.depth[window]
        ok = np.isfinite(z)
        if not ok.any():
            return None
        return np.median(self.points[window][ok], axis=0), float(np.median(z[ok]))


@lru_cache(maxsize=8)
def _triplets(n: int) -> np.ndarray:
    return np.array(list(combinations(range(n), 3)), dtype=int).reshape(-1, 3)


def fit_face_plane(
    cloud: np.ndarray, handle: np.ndarray, camera_position: np.ndarray, vertical: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """RoboRSI's face fit: the plane through three of the points with the most inliers
    (ties: the least mean residual), refined by PCA of its inliers. Returns the outward
    normal (horizontal for a vertical face, signed toward the camera), the inliers'
    centroid and the plane's own unit normal (same sign), or None."""
    cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
    if len(cloud) < 6:
        return None
    if len(cloud) > MAX_PLANE_FIT_POINTS:
        cand = cloud[np.linspace(0, len(cloud) - 1, MAX_PLANE_FIT_POINTS, dtype=int)]
    else:
        cand = cloud
    idx = _triplets(len(cand))
    a, b, c = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > 1e-8
    normals, a = normals[keep] / norms[keep, None], a[keep]
    if vertical:
        upright = np.abs(normals[:, 2]) <= MAX_CANDIDATE_TILT
        normals, a = normals[upright], a[upright]
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for start in range(0, len(normals), 2048):
        n, p = normals[start : start + 2048], a[start : start + 2048]
        d = np.abs(cloud @ n.T - np.sum(p * n, axis=1))
        inl = d <= PLANE_INLIER
        count = inl.sum(axis=0)
        residual = np.where(count > 0, (d * inl).sum(axis=0) / np.maximum(count, 1), np.inf)
        ok = count >= MIN_PLANE_INLIERS
        if not ok.any():
            continue
        order = np.lexsort((residual, -count))
        i = int(order[0])
        if best is None or (int(count[i]), -float(residual[i])) > (best[0], -best[1]):
            best = (int(count[i]), float(residual[i]), n[i], p[i])
    if best is None:
        return None
    plane = cloud[np.abs((cloud - best[3]) @ best[2]) <= PLANE_INLIER]
    centered = plane - plane.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(1, len(plane) - 1))
    if (
        float(eigenvalues[1]) <= 1e-9
        or float(eigenvalues[0]) / float(eigenvalues[1]) > MAX_PLANARITY
    ):
        return None
    plane_normal = eigenvectors[:, 0].astype(float)
    toward = np.asarray(camera_position, dtype=float) - np.asarray(handle, dtype=float)
    if vertical:
        if abs(float(plane_normal[2])) > MAX_FACE_TILT:
            return None
        normal = plane_normal.copy()
        normal[2] = 0.0
        toward[2] = 0.0
    else:
        normal = plane_normal.copy()
    if float(np.linalg.norm(normal)) <= 1e-8 or float(np.linalg.norm(toward)) <= 1e-8:
        return None
    normal = normal / np.linalg.norm(normal)
    alignment = float(normal @ (toward / np.linalg.norm(toward)))
    if abs(alignment) < MIN_NORMAL_SIGN_ALIGNMENT:
        return None
    if alignment < 0:
        normal, plane_normal = -normal, -plane_normal
    return normal, plane.mean(axis=0), plane_normal


def _ring_offsets(radius_px: float) -> np.ndarray:
    r = radius_px
    return np.array([[-r, -r], [0, -r], [r, -r], [-r, 0], [r, 0], [-r, r], [0, r], [r, r]])


def refine_handle(
    frame: Frame, u: float, v: float, vertical: bool = True, front_band: float = FRONT_BAND
) -> tuple[tuple[int, int], np.ndarray, np.ndarray] | str:
    """RoboRSI's _refine_drawer_handle_geometry at depth pixel (u, v): fit the face round
    it, find the pixels proud of the face, keep the component nearest the seed and return
    its front (the median of its pixels within ``front_band`` of the most protruding one;
    0: RoboRSI's single most protruding pixel) as a depth pixel and a world point, with
    the face's outward normal. Returns why not as a string when it fails."""
    seeded = frame.seed(u, v)
    if seeded is None:
        return f"no depth at or round pixel ({u:.0f}, {v:.0f})"
    seed, z = seeded
    ppm = frame.fx / z
    centre = np.array([u, v])
    ring = np.vstack([_ring_offsets(r * ppm) for r in FACE_RING_RADII]) + centre
    cloud, _ = frame.sample(ring[:, 0], ring[:, 1])
    face = fit_face_plane(cloud, seed, frame.camera_position, vertical)
    step = max(1, int(round(LIBERO_PIXEL * ppm)))
    if face is None:
        outer, inner = DENSE_FACE_RADIUS * ppm, HANDLE_SEARCH_RADIUS * ppm
        grid = np.arange(-outer, outer + 1e-9, step)
        du, dv = np.meshgrid(grid, grid)
        annulus = np.maximum(np.abs(du), np.abs(dv)) > inner
        cloud, _ = frame.sample(du[annulus] + u, dv[annulus] + v)
        face = fit_face_plane(cloud, seed, frame.camera_position, vertical)
    if face is None:
        return f"no {'vertical ' if vertical else ''}face plane round the handle"
    normal, plane_centre, plane_normal = face
    search = HANDLE_SEARCH_RADIUS * ppm
    grid = np.arange(-search, search + 1e-9, step)
    du, dv = np.meshgrid(grid, grid)
    pts, pixels = frame.sample(du.ravel() + u, dv.ravel() + v)
    if len(pts):
        _, unique = np.unique(pixels, axis=0, return_index=True)
        pts, pixels = pts[np.sort(unique)], pixels[np.sort(unique)]
    protrusion = (pts - plane_centre) @ plane_normal
    near = np.linalg.norm(pts - seed, axis=1) <= HANDLE_SEARCH_RADIUS + PROTRUSION[1]
    sel = (protrusion >= PROTRUSION[0]) & (protrusion <= PROTRUSION[1]) & near
    pts, pixels, protrusion = pts[sel], pixels[sel].astype(float), protrusion[sel]
    if len(pts) < MIN_COMPONENT:
        return "nothing proud of the face round the pointed pixel (no handle)"
    adjacency = np.linalg.norm(pixels[:, None] - pixels[None], axis=2) <= ADJACENCY * step
    remaining = set(range(len(pts)))
    components = []
    while remaining:
        pending = [remaining.pop()]
        component = []
        while pending:
            i = pending.pop()
            component.append(i)
            for j in np.flatnonzero(adjacency[i]):
                if int(j) in remaining:
                    remaining.remove(int(j))
                    pending.append(int(j))
        if len(component) >= MIN_COMPONENT:
            components.append(np.asarray(component, dtype=int))
    if not components:
        return "nothing proud of the face round the pointed pixel (no handle)"
    comp = max(
        components,
        key=lambda c: (
            -float(np.min(np.linalg.norm(pixels[c] - centre, axis=1))),
            len(c),
            float(np.mean(protrusion[c])),
        ),
    )
    if front_band <= 0.0:
        medoid = comp[
            np.argmin(np.linalg.norm(pixels[comp] - np.median(pixels[comp], axis=0), axis=1))
        ]
        best = max(
            comp,
            key=lambda i: (
                float(protrusion[i]),
                -float(np.linalg.norm(pixels[i] - pixels[medoid])),
            ),
        )
        return (int(pixels[best, 0]), int(pixels[best, 1])), pts[best].copy(), normal
    front = comp[protrusion[comp] >= float(np.max(protrusion[comp])) - front_band]
    point = np.median(pts[front], axis=0)
    pick = front[
        np.argmin(np.linalg.norm(pixels[front] - np.median(pixels[front], axis=0), axis=1))
    ]
    return (int(pixels[pick, 0]), int(pixels[pick, 1])), point, normal


def _marked(rgb: np.ndarray, u: int, v: int, half: int) -> np.ndarray:
    img = np.array(rgb[..., :3], dtype=np.uint8, copy=True)
    h, w = img.shape[:2]
    for d in range(-half, half + 1):
        for t in (-1, 0, 1):
            if 0 <= v + t < h and 0 <= u + d < w:
                img[v + t, u + d] = (255, 40, 40)
            if 0 <= v + d < h and 0 <= u + t < w:
                img[v + d, u + t] = (255, 40, 40)
    return img


def _crop(img: np.ndarray, u: int, v: int, radius: int, size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    rows = np.clip(np.arange(v - radius, v + radius), 0, h - 1)
    cols = np.clip(np.arange(u - radius, u + radius), 0, w - 1)
    crop = img[rows][:, cols]
    pick = np.linspace(0, crop.shape[0] - 1, size).astype(int)
    return crop[pick][:, pick]


def localize(
    ctx: NodeContext,
    frame: Frame,
    description: str,
    pixel: Any = None,
    pointer: str = "gemini-er.point",
    refine_with_sam: bool = True,
    confirm_with_vlm: bool = False,
    vertical: bool = True,
    front_band: float = FRONT_BAND,
) -> dict[str, Any]:
    """RoboRSI's localize_precise + _resolve_drawer_handle on one frame: point at the
    handle (or take ``pixel``), tighten with SAM, refine on the depth, optionally confirm
    with a VLM. Returns {"seen", "why", "pixel", "point", "normal"}."""
    out: dict[str, Any] = {"seen": False, "why": "", "pixel": None, "point": None, "normal": None}
    if pixel is not None:
        if isinstance(pixel, dict):
            u, v = float(pixel.get("u", pixel.get("x"))), float(pixel.get("v", pixel.get("y")))
        else:
            u, v = (float(x) for x in pixel)
    else:
        kwargs: dict[str, Any] = {"image": frame.rgb, "query": description}
        if pointer.startswith("gemini-er."):
            kwargs["max_points"] = 3
        try:
            reply = ctx.tool(pointer, **kwargs)
        except Exception as exc:  # a pointer failure is a handle not seen
            out["why"] = f"{pointer} failed: {exc}"
            return out
        if not isinstance(reply, dict) or not reply.get("found"):
            out["why"] = f"{pointer} found no {description!r}"
            return out
        first = (reply.get("points") or [reply])[0]
        u, v = float(first["pixel_x"]), float(first["pixel_y"])
    if refine_with_sam:
        seeded = frame.seed(u * frame.sx, v * frame.sy)
        try:
            masks = ctx.tool("sam3.segment_point", image=frame.rgb, pixel_x=u, pixel_y=v).get(
                "masks"
            )
        except Exception as exc:  # RoboRSI keeps the raw point when SAM fails
            logger.info("verifying-articulation-displacement: SAM refinement failed (%s)", exc)
            masks = None
        if masks and seeded is not None:
            m = np.asarray(masks[0])
            m = m.reshape(m.shape[-2:]) > 0
            if int(m.sum()) >= MIN_SAM_PIXELS:
                ys, xs = np.nonzero(m)
                mu, mv = float(np.median(xs)), float(np.median(ys))
                limit = SAM_MAX_SHIFT * frame.fx / frame.sx / seeded[1]
                if math.hypot(mu - u, mv - v) <= limit:
                    u, v = mu, mv
    refined = refine_handle(frame, u * frame.sx, v * frame.sy, vertical, front_band)
    if isinstance(refined, str):
        out["why"] = refined
        out["pixel"] = (u, v)
        return out
    (du, dv), point, normal = refined
    pu, pv = du / frame.sx, dv / frame.sy
    out["pixel"] = (pu, pv)
    if confirm_with_vlm:
        z = frame.depth[dv, du]
        radius = max(8, int(round(VLM_CROP_RADIUS * frame.fx / frame.sx / float(z))))
        marked = _marked(frame.rgb, int(round(pu)), int(round(pv)), max(4, radius // 3))
        prompt = (
            "The first image is the full scene and the second a magnified crop round the red "
            f"cross. Is the cross on {description}? Answer no for cabinet panels, doors, "
            "shelves, loose objects and handles on another drawer or door."
        )
        try:
            answer = ctx.tool(
                "vlm.query_yes_no",
                prompt=prompt,
                images=[marked, _crop(marked, int(round(pu)), int(round(pv)), radius)],
            )
        except Exception as exc:
            out["why"] = f"vlm.query_yes_no failed: {exc}"
            return out
        if not (isinstance(answer, dict) and answer.get("answer")):
            out["why"] = "the VLM did not confirm the refined pixel is on the handle"
            return out
    out.update(seen=True, point=point, normal=normal)
    return out


def _signed(
    normal: np.ndarray, camera_position: np.ndarray, at: np.ndarray, vertical: bool
) -> np.ndarray:
    """A given face normal, unit, horizontal for a vertical face, signed toward the camera."""
    n = np.asarray(normal, dtype=float).copy()
    toward = np.asarray(camera_position, dtype=float) - at
    if vertical:
        n[2], toward[2] = 0.0, 0.0
    n = _unit(n, "face_normal_before")
    return -n if float(n @ toward) < 0 else n


def run(
    ctx: NodeContext,
    handle_description: str,
    joint_type: str,
    expected_opening: float,
    handle_before: dict[str, Any] | None = None,
    face_normal_before: dict[str, float] | None = None,
    pull_direction: dict[str, float] | None = None,
    hinge_point: dict[str, float] | None = None,
    hinge_axis: dict[str, float] | None = None,
    observation: dict[str, Any] | None = None,
    before_observation: dict[str, Any] | None = None,
    handle_pixel: Any = None,
    handle_pixel_before: Any = None,
    camera_name: str = "overhead",
    pointer: str = "gemini-er.point",
    refine_with_sam: bool = True,
    confirm_with_vlm: bool = False,
    vertical_face: bool = True,
    front_band: float = FRONT_BAND,
    min_fraction: float = MIN_FRACTION,
    min_motion: float = MIN_MOTION,
    normal_dot_min: float = NORMAL_DOT_MIN,
    max_transverse: float = MAX_TRANSVERSE,
    max_travel: float = MAX_TRAVEL,
) -> Output:
    """Re-localise the handle in a fresh frame and judge the articulation's displacement.

    Args:
        handle_description: What the pointer is asked for ("the handle of the top drawer").
        joint_type: ``"prismatic"`` (drawer) or ``"revolute"`` (door, lid).
        expected_opening: The displacement asked for, planning-articulation-opening's
            ``opening``: metres (drawer) or radians (door); positive opens (along
            ``pull_direction`` / the door's opening sense), negative closes.
        handle_before: The handle before actuation (Vec3, or planning's ``handle_pose``).
            Not needed with ``before_observation``, which takes precedence.
        face_normal_before: The face's outward normal before actuation; default: measured
            in ``before_observation``, else (drawer) ``pull_direction``.
        pull_direction: Drawer: the opening direction (default: the face normal before).
            Door: optional, the way the handle starts moving when opening; picks the
            hinge axis's sign as planning-articulation-opening does.
        hinge_point, hinge_axis: Door: any point on the hinge line and its direction.
        observation: The frame after actuation; default: a fresh robot.get_observation.
        before_observation: A frame before actuation, localised the same way.
        handle_pixel, handle_pixel_before: [u, v] of the handle in either frame, to skip
            the pointer.
        camera_name: The observation's camera to use.
        pointer: ``"gemini-er.point"`` or ``"molmo.point_prompt"``.
        refine_with_sam: Tighten the pointed pixel to its SAM mask's centroid.
        confirm_with_vlm: Ask vlm.query_yes_no whether the refined pixel is on the handle.
        vertical_face: The face is vertical (drawers, side-hinged doors); False for lids.
        front_band: The handle point is the median of the handle's front within this of
            its most protruding pixel [m]; 0 takes that pixel alone (RoboRSI).
    """
    kind = str(joint_type).lower()
    if kind not in ("prismatic", "revolute"):
        raise ValueError(f"joint_type must be 'prismatic' or 'revolute', got {joint_type!r}")
    expected = float(expected_opening)
    if not math.isfinite(expected) or expected == 0.0:
        raise ValueError("expected_opening must be a non-zero number")
    if kind == "revolute" and (hinge_point is None or hinge_axis is None):
        raise ValueError("a revolute joint needs hinge_point and hinge_axis")
    if handle_before is None and before_observation is None:
        raise ValueError("give handle_before or before_observation")

    out: Output = {
        "route": "not_seen",
        "detail": "",
        "joint_type": kind,
        "seen": False,
        "measured": 0.0,
        "expected": expected,
        "fraction": 0.0,
        "transverse": 0.0,
        "normal_dot": 0.0,
        "normal_angle": 0.0,
        "radius": 0.0,
        "handle_before": _vec(None),
        "handle_after": _vec(None),
        "face_normal_after": _vec(None),
        "handle_pixel": {"u": -1.0, "v": -1.0},
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        return out

    look = dict(
        pointer=pointer,
        refine_with_sam=refine_with_sam,
        confirm_with_vlm=confirm_with_vlm,
        front_band=float(front_band),
    )
    after_obs = observation if observation is not None else ctx.tool("robot.get_observation")
    after_frame = Frame(_camera(after_obs, camera_name))
    after = localize(
        ctx, after_frame, handle_description, handle_pixel, vertical=vertical_face, **look
    )
    if after["pixel"] is not None:
        out["handle_pixel"] = {"u": float(after["pixel"][0]), "v": float(after["pixel"][1])}
    if not after["seen"]:
        return finish("not_seen", f"after actuation: {after['why']}")
    p_a, n_a = after["point"], after["normal"]
    out["seen"] = True
    out["handle_after"], out["face_normal_after"] = _vec(p_a), _vec(n_a)

    n_b: np.ndarray | None = None
    source = "given"
    if before_observation is not None:
        before_frame = Frame(_camera(before_observation, camera_name))
        before = localize(
            ctx,
            before_frame,
            handle_description,
            handle_pixel_before,
            vertical=vertical_face,
            **look,
        )
        if not before["seen"]:
            return finish("not_seen", f"before actuation: {before['why']}")
        p_b, n_b, source = before["point"], before["normal"], "measured"
    else:
        p_b = _xyz(handle_before)
    out["handle_before"] = _vec(p_b)
    if face_normal_before is not None:
        n_b = _signed(_xyz(face_normal_before), after_frame.camera_position, p_b, vertical_face)
    elif n_b is None and kind == "prismatic" and pull_direction is not None:
        n_b = _signed(_xyz(pull_direction), after_frame.camera_position, p_b, vertical_face)

    sign = 1.0 if expected > 0 else -1.0
    unit = "mm" if kind == "prismatic" else "deg"
    scale = 1e3 if kind == "prismatic" else math.degrees(1.0)
    delta = p_a - p_b
    if kind == "prismatic":
        if pull_direction is not None:
            direction = _unit(_xyz(pull_direction), "pull_direction")
        elif n_b is not None:
            direction = n_b
        else:
            raise ValueError(
                "a prismatic joint needs pull_direction, face_normal_before or before_observation"
            )
        measured = float(delta @ direction)
        transverse = float(np.linalg.norm(delta - measured * direction))
        normal_dot = float(n_a @ n_b) if n_b is not None else float("nan")
        threshold = float(min_motion)
        if abs(measured) > float(max_travel):
            out.update(measured=measured, transverse=transverse, normal_dot=normal_dot)
            return finish(
                "inconsistent",
                f"the handle moved {measured * 1e3:.0f} mm along the pull, over the "
                f"{float(max_travel) * 1e3:.0f} mm a drawer travels: another handle?",
            )
    else:
        c = _xyz(hinge_point)
        a = _unit(_xyz(hinge_axis), "hinge_axis")
        lever_b = (p_b - c) - a * float((p_b - c) @ a)
        radius = float(np.linalg.norm(lever_b))
        if radius < MIN_RADIUS:
            raise ValueError(f"the handle lies {radius * 1e3:.1f} mm from the hinge line")
        if pull_direction is not None:
            hint = _unit(_xyz(pull_direction), "pull_direction")
            if float(np.cross(a, lever_b) @ hint) < 0:
                a = -a
        lever_a = (p_a - c) - a * float((p_a - c) @ a)
        measured = math.atan2(float(a @ np.cross(lever_b, lever_a)), float(lever_b @ lever_a))
        transverse = max(abs(float(np.linalg.norm(lever_a)) - radius), abs(float(delta @ a)))
        out["radius"] = radius
        if n_b is not None:
            normal_dot = float(n_a @ (_axis_angle(a, measured) @ n_b))
            nb_p, na_p = n_b - a * float(n_b @ a), n_a - a * float(n_a @ a)
            out["normal_angle"] = math.atan2(float(a @ np.cross(nb_p, na_p)), float(nb_p @ na_p))
        else:
            normal_dot = float("nan")
        threshold = float(min_motion) / radius
    out.update(
        measured=measured,
        fraction=measured / expected,
        transverse=transverse,
        normal_dot=normal_dot,
    )
    moved = f"{measured * scale:+.1f} {unit} of {expected * scale:+.1f} {unit} expected"
    facts = (
        f"{moved}; drift {transverse * 1e3:.0f} mm"
        + (
            f"; face normal agrees {normal_dot:.3f}"
            if math.isfinite(normal_dot)
            else "; no face normal before"
        )
        + f"; handle before {source}"
    )
    if math.isfinite(normal_dot) and normal_dot < float(normal_dot_min):
        return finish(
            "inconsistent",
            f"the face turned away from the one expected (cosine {normal_dot:.3f} < "
            f"{float(normal_dot_min):.2f}): another face or handle; {facts}",
        )
    if transverse > float(max_transverse):
        return finish(
            "inconsistent",
            f"the handle drifted {transverse * 1e3:.0f} mm off the articulation (over "
            f"{float(max_transverse) * 1e3:.0f} mm): another handle, or the cabinet moved; {facts}",
        )
    progress = measured * sign
    if measured / expected >= float(min_fraction):
        return finish("moved", facts)
    if progress >= threshold:
        return finish("moved_short", f"short of {float(min_fraction):.0%} of the expected: {facts}")
    if progress <= -threshold:
        return finish("moved_wrong_way", f"moved against the expected sense: {facts}")
    return finish(
        "not_moved", f"under the {threshold * scale:.1f} {unit} that counts as motion: {facts}"
    )
