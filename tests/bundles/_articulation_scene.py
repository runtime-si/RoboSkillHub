"""A ray-cast RGB-D frame of a cabinet with a drawer or a side-hinged door, for the
verifying-articulation-displacement tests.

World z-up, metres. The cabinet's front is the plane x = 0.55 facing -x (toward a robot at
the origin); the camera looks at it from the front and above. Every part is a box, and a
pixel's depth is its z along the optical axis (OpenCV camera: x right, y down, z forward).
"""

from __future__ import annotations

import math

import numpy as np

FRONT_X = 0.55
DRAWER_Z = 0.30
HINGE = np.array([FRONT_X, -0.20, 0.0])
DOOR_WIDTH = 0.40
HANDLE_PROUD = (0.012, 0.032)  # the bar's back and front, proud of the face

COLOURS = {
    "table": (150, 150, 155),
    "cabinet": (185, 140, 95),
    "drawer": (200, 155, 105),
    "tray": (170, 125, 85),
    "handle": (60, 60, 65),
    "lower_drawer": (200, 155, 105),
    "lower_handle": (60, 60, 65),
    "door": (200, 155, 105),
    "wall": (225, 225, 215),
}


def rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box(name, lo, hi, rot=None, origin=None):
    """A box from its local corners ``lo``/``hi``, placed by ``rot`` about ``origin``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    rot = np.eye(3) if rot is None else rot
    origin = np.zeros(3) if origin is None else np.asarray(origin, float)
    return {
        "name": name,
        "center": origin + rot @ ((lo + hi) / 2),
        "half": (hi - lo) / 2,
        "rot": rot,
    }


def _common():
    return [
        box("table", (-0.5, -1.0, -0.02), (1.5, 1.0, 0.0)),
        box("wall", (0.95, -1.0, 0.0), (0.97, 1.0, 1.2)),
        box("cabinet", (FRONT_X, -0.25, 0.0), (0.85, 0.25, 0.45)),
    ]


def drawer_handle_front(opening: float = 0.0) -> np.ndarray:
    """The centre of the upper drawer handle's front face."""
    return np.array([FRONT_X - 0.02 - opening - HANDLE_PROUD[1], 0.0, DRAWER_Z + 0.01])


def drawer_scene(opening: float = 0.0, shift=(0.0, 0.0, 0.0)):
    """The upper drawer pulled ``opening`` m out (along -x); ``shift`` moves it bodily."""
    d = np.asarray(shift, float) + np.array([-opening, 0.0, 0.0])
    face = FRONT_X - 0.02  # the closed drawer front stands 20 mm proud of the cabinet
    parts = _common() + [
        box("drawer", (face, -0.20, DRAWER_Z - 0.09), (FRONT_X, 0.20, DRAWER_Z + 0.09)),
        box("tray", (FRONT_X, -0.19, DRAWER_Z - 0.08), (FRONT_X + 0.30, 0.19, DRAWER_Z + 0.07)),
        box(
            "handle",
            (face - HANDLE_PROUD[1], -0.06, DRAWER_Z),
            (face - HANDLE_PROUD[0], 0.06, DRAWER_Z + 0.02),
        ),
    ]
    for p in parts[3:]:
        p["center"] = p["center"] + d
    parts += [
        box("lower_drawer", (face, -0.20, 0.03), (FRONT_X, 0.20, 0.19)),
        box(
            "lower_handle",
            (face - HANDLE_PROUD[1], -0.06, 0.11),
            (face - HANDLE_PROUD[0], 0.06, 0.13),
        ),
    ]
    return parts


def door_handle_front(theta: float = 0.0) -> np.ndarray:
    local = np.array([-0.02 - HANDLE_PROUD[1], DOOR_WIDTH - 0.06, 0.25])
    return HINGE + rz(theta) @ local


def door_scene(theta: float = 0.0):
    """The door turned ``theta`` rad about the vertical hinge at HINGE (+z opens it toward -x)."""
    r = rz(theta)
    return _common() + [
        box("door", (-0.02, 0.0, 0.02), (0.0, DOOR_WIDTH, 0.44), r, HINGE),
        box(
            "handle",
            (-0.02 - HANDLE_PROUD[1], DOOR_WIDTH - 0.07, 0.18),
            (-0.02 - HANDLE_PROUD[0], DOOR_WIDTH - 0.05, 0.32),
            r,
            HINGE,
        ),
    ]


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z], axis=1)
    t[:3, 3] = eye
    return t


def pose_dict(t: np.ndarray) -> dict:
    m = t[:3, :3]
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return {
        "position": dict(zip("xyz", map(float, t[:3, 3]), strict=True)),
        "rotation": {"w": w, "x": x, "y": y, "z": z},
    }


def intrinsics(width=320, height=240, f=280.0) -> np.ndarray:
    return np.array([[f, 0.0, width / 2 - 0.5], [0.0, f, height / 2 - 0.5], [0.0, 0.0, 1.0]])


def render(parts, k, t, width=320, height=240):
    """Depth (z along the optical axis), part index per pixel (-1: nothing) and world normals."""
    v, u = np.mgrid[0:height, 0:width]
    d_cam = np.stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u, float)), -1)
    d = d_cam.reshape(-1, 3) @ t[:3, :3].T
    o = t[:3, 3]
    best = np.full(len(d), np.inf)
    ids = np.full(len(d), -1)
    normals = np.zeros((len(d), 3))
    for i, p in enumerate(parts):
        r, c, h = p["rot"], p["center"], p["half"]
        ol, dl = r.T @ (o - c), d @ r
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-h - ol) / dl, (h - ol) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmin > 1e-6) & (tmin < best)
        best[hit], ids[hit] = tmin[hit], i
        local = ol + tmin[hit, None] * dl[hit]
        axis = np.argmax(np.abs(local) / h, axis=1)
        n = np.zeros((hit.sum(), 3))
        n[np.arange(len(axis)), axis] = np.sign(local[np.arange(len(axis)), axis])
        normals[hit] = n @ r.T
    depth = np.where(np.isfinite(best), best, 0.0).reshape(height, width)
    return depth.astype(np.float32), ids.reshape(height, width), normals.reshape(height, width, 3)


def shade(parts, ids, normals, light=(-0.4, 0.3, 0.85)) -> np.ndarray:
    light = np.asarray(light, float) / np.linalg.norm(light)
    rgb = np.zeros((*ids.shape, 3))
    for i, p in enumerate(parts):
        m = ids == i
        lam = 0.35 + 0.65 * np.clip(normals[m] @ light, 0.0, 1.0)
        rgb[m] = np.outer(lam, COLOURS[p["name"]])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def frame(
    parts, eye, target, width=320, height=240, f=280.0, noise=None, rng=None, pose=None, k=None
):
    """An observation camera dict looking from ``eye`` at ``target`` (or placed by the 4x4
    camera-to-world ``pose``, with intrinsics ``k``); ``noise(depth, rng)`` perturbs the depth."""
    k = intrinsics(width, height, f) if k is None else np.asarray(k, float)
    t = look_at(eye, target) if pose is None else np.asarray(pose, float)
    depth, ids, normals = render(parts, k, t, width, height)
    if noise is not None:
        depth = noise(depth, rng or np.random.default_rng(0)).astype(np.float32)
    return {
        "name": "overhead",
        "rgb": shade(parts, ids, normals),
        "depth": depth,
        "intrinsics": k,
        "pose": pose_dict(t),
        "_ids": ids,
        "_names": [p["name"] for p in parts],
        "_t": t,
        "_k": k,
    }


def project(camera, point) -> tuple[float, float]:
    t, k = camera["_t"], camera["_k"]  # the RGB image's, whatever the depth's
    pc = t[:3, :3].T @ (np.asarray(point, float) - t[:3, 3])
    return float(k[0, 0] * pc[0] / pc[2] + k[0, 2]), float(k[1, 1] * pc[1] / pc[2] + k[1, 2])


def mask_of(camera, name) -> np.ndarray:
    idx = camera["_names"].index(name)
    return ((camera["_ids"] == idx) * 255).astype(np.uint8)
