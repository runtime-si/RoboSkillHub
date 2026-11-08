"""Synthetic top-down domino frames: an RGB-D camera looking straight down at tiles on a table.

Tiles are drawn 4x supersampled in image space (rounded corners, a dark side-face
outline, a divider groove, round pips on the double-six 3 x 3 layout with a small
random offset), then blurred, noised and optionally lit with a glare spot. Depth is
the camera's z-distance to the table (z = 0) or to the tile's top for its resting
state; the camera pose and intrinsics make every pixel a world point.
"""

from __future__ import annotations

import cv2
import numpy as np

TILE = (0.048, 0.024, 0.008)
CAM_HEIGHT = 0.20
DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

LAYOUTS = {
    0: [],
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)],
    6: [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
}
ALL_VALUES = [(a, b) for a in range(7) for b in range(a, 7)]

STYLES = {
    "white": {"tile": (238, 236, 228), "pip": (20, 20, 20), "divider": (60, 60, 60)},
    "ivory": {"tile": (232, 218, 180), "pip": (35, 30, 25), "divider": (90, 80, 60)},
    "black": {"tile": (28, 28, 30), "pip": (235, 235, 235), "divider": (150, 150, 150)},
    "colour": {"tile": (240, 240, 236), "pip": None, "divider": (30, 30, 30)},
}
# Pip colours by value, as in a coloured double-six set (yellow and light blue are
# the faint ones in luminance).
PIP_COLOURS = {
    1: (110, 180, 230),
    2: (40, 150, 60),
    3: (220, 60, 110),
    4: (140, 95, 55),
    5: (30, 60, 170),
    6: (240, 200, 30),
}


def camera_for(width_px: float, hw=(480, 640), height=CAM_HEIGHT):
    """A look-down camera whose focal length shows a flat tile ``width_px`` wide."""
    f = width_px * (height - TILE[2]) / TILE[1]
    k = np.array([[f, 0.0, hw[1] / 2.0], [0.0, f, hw[0] / 2.0], [0.0, 0.0, 1.0]])
    pose = {"position": {"x": 0.3, "y": 0.0, "z": height}, "rotation": DOWN}
    return k, pose


def pixel_to_world(k, pose, u, v, depth):
    p = pose["position"]
    return np.array(
        [
            p["x"] + (u - k[0, 2]) * depth / k[0, 0],
            p["y"] - (v - k[1, 2]) * depth / k[1, 1],
            p["z"] - depth,
        ]
    )


def world_to_pixel(k, pose, xyz):
    p = pose["position"]
    depth = p["z"] - xyz[2]
    return np.array(
        [
            k[0, 2] + (xyz[0] - p["x"]) * k[0, 0] / depth,
            k[1, 2] - (xyz[1] - p["y"]) * k[1, 1] / depth,
        ]
    )


def render(
    tiles,
    width_px=48.0,
    hw=(480, 640),
    blur=0.0,
    noise=0.0,
    glare=0.0,
    style="white",
    spinner=False,
    jitter=0.02,
    pip_scale=1.0,
    grid=0.25,
    divider=True,
    seed=0,
):
    """``tiles``: dicts with value (a, b), center_px (u, v), angle_deg, state
    ('flat' | 'upright' | 'fallen' | 'face_down'). Returns (camera frame, masks, truth)."""
    rng = np.random.default_rng(seed)
    k, pose = camera_for(width_px, hw)
    ss = 4
    h, w = hw
    st = STYLES[style]
    yy, xx = np.mgrid[0 : h * ss, 0 : w * ss]
    xs, ys = (xx + 0.5) / ss, (yy + 0.5) / ss
    base = 150 + 12 * cv2.resize(rng.normal(size=(h // 16 + 1, w // 16 + 1)), (w, h))
    img = np.repeat(base[..., None], 3, axis=2) * np.array([0.95, 0.9, 0.82])
    img = cv2.resize(img, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    depth = np.full(hw, CAM_HEIGHT, np.float32)
    masks, truth = [], []
    length_m, width_m, thick_m = TILE
    for spec in tiles:
        state = spec.get("state", "flat")
        a, b = spec.get("value", (0, 0))
        uc, vc = spec["center_px"]
        th = np.radians(spec.get("angle_deg", 0.0))
        if state in ("flat", "face_down"):
            dims, top = (length_m, width_m), thick_m
        elif state == "upright":
            dims, top = (width_m, thick_m), length_m
        else:
            dims, top = (length_m, thick_m), width_m
        f_top = k[0, 0] / (CAM_HEIGHT - top)
        L, W = dims[0] * f_top, dims[1] * f_top
        s = (xs - uc) * np.cos(th) + (ys - vc) * np.sin(th)
        t = -(xs - uc) * np.sin(th) + (ys - vc) * np.cos(th)
        rc = 0.12 * min(L, W)
        qx, qy = np.abs(s) - (L / 2 - rc), np.abs(t) - (W / 2 - rc)
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
        body = outside <= rc
        rim = (outside <= rc) & (outside > rc - 0.04 * W)
        colour = np.array(st["tile"], float)
        img[body] = colour * (0.96 + 0.04 * (s[body] / L))[:, None]
        img[rim] = colour * 0.55
        if state == "flat":
            d = st["divider"]
            if divider:
                img[body & (np.abs(s) <= 0.025 * W) & (np.abs(t) <= 0.4 * W)] = d
            for side, value in ((-1, a), (1, b)):
                for col, row in LAYOUTS[value]:
                    g = (grid, 0.5, 1.0 - grid)
                    ps = (0.0 if side > 0 else -L / 2) + g[col] * (L / 2)
                    pt = -W / 2 + g[row] * W
                    ps += rng.normal() * jitter * W
                    pt += rng.normal() * jitter * W
                    rp = 0.085 * W * pip_scale
                    pip = st["pip"] or PIP_COLOURS[value]
                    img[body & ((s - ps) ** 2 + (t - pt) ** 2 <= rp * rp)] = pip
            if spinner:
                img[body & (s**2 + t**2 <= (0.06 * W) ** 2)] = (190, 150, 60)
        small = cv2.resize(body.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
        depth[small] = CAM_HEIGHT - top
        masks.append(small)
        centre = pixel_to_world(k, pose, uc, vc, CAM_HEIGHT - top)
        if state == "upright":
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = np.array([np.cos(th), -np.sin(th), 0.0])
        truth.append(
            {
                "value": tuple(sorted((a, b))),
                "value_ab": (a, b),
                "state": "flat" if state == "face_down" else state,
                "center": centre,
                "axis": axis,
                "end_a_px": (uc - 0.5 * L * np.cos(th), vc - 0.5 * L * np.sin(th)),
                "end_b_px": (uc + 0.5 * L * np.cos(th), vc + 0.5 * L * np.sin(th)),
            }
        )
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if glare > 0:
        for spec in tiles:
            gu = spec["center_px"][0] + rng.uniform(-0.5, 0.5) * width_px
            gv = spec["center_px"][1] + rng.uniform(-0.3, 0.3) * width_px
            g = np.exp(
                -(
                    ((np.arange(w)[None] - gu) ** 2 + (np.arange(h)[:, None] - gv) ** 2)
                    / (2 * (0.2 * width_px) ** 2)
                )
            )
            img = img + glare * g[..., None]
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    frame = {"name": "left_wrist", "rgb": rgb, "depth": depth, "intrinsics": k, "pose": pose}
    return frame, masks, truth
