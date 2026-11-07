"""Procedural jigsaw pictures and pieces cut from them, for tests that must not ship real puzzle art.

The picture has the hard cases of a real one: textured regions, a smooth gradient, a large flat
white band (snow), a repetitive plaid and small repeated motifs. A piece is a cell of the picture
with a tab, a blank or a straight side on each edge (straight on the board's edge), seen by a
wrist camera: rotated, blurred, with per-channel gain and noise, on a dark mat.
"""

from __future__ import annotations

import cv2
import numpy as np

MAT = 30
"""Grey level of the mat around the piece."""


def make_reference(width=440, height=500, seed=0, band_rows=(0.0, 0.08), plaid=True):
    """RGB uint8 picture. ``band_rows`` is the white band's top and bottom as fractions of the height."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.zeros((height, width, 3), np.float32)
    for k in range(3):
        img[:, :, k] = 60 + 120 * (
            0.5 + 0.5 * np.sin(xx / width * rng.uniform(2, 6) + yy / height * rng.uniform(2, 6) + k)
        )
    noise = rng.normal(0, 1, (height // 8 + 1, width // 8 + 1, 3)).astype(np.float32)
    img += 25 * cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(int(width * height / 900)):
        color = tuple(float(v) for v in rng.uniform(0, 255, 3))
        x, y = int(rng.uniform(0, width)), int(rng.uniform(0, height))
        s = int(rng.uniform(3, 18))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), s, color, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + s, y + int(rng.uniform(3, 18))), color, -1)
        else:
            x2, y2 = x + int(rng.uniform(-30, 30)), y + int(rng.uniform(-30, 30))
            cv2.line(img, (x, y), (x2, y2), color, int(rng.uniform(1, 4)))
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    if plaid:
        x0, x1, y0, y1 = (
            int(0.55 * width),
            int(0.95 * width),
            int(0.45 * height),
            int(0.75 * height),
        )
        p = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        py, px = np.mgrid[0 : y1 - y0, 0 : x1 - x0]
        p[:] = (150, 30, 30)
        stripe = ((px % 24) < 6) | ((py % 24) < 6)
        thin = ((px % 24) == 12) | ((py % 24) == 12)
        p[stripe] = (40, 40, 40)
        p[thin] = (220, 200, 120)
        img[y0:y1, x0:x1] = p + rng.normal(0, 4, p.shape)
    y0, y1 = int(band_rows[0] * height), int(band_rows[1] * height)
    if y1 > y0:
        img[y0:y1] = 238 + rng.normal(0, 2, (y1 - y0, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def piece_mask(body_w, body_h, sides, canvas):
    """Mask of a piece body ``body_w`` x ``body_h`` centred in a ``canvas`` square. ``sides`` gives
    top, bottom, left, right: 0 straight, 1 tab, 2 blank."""
    m = np.zeros((canvas, canvas), np.uint8)
    ox, oy = (canvas - body_w) / 2, (canvas - body_h) / 2
    cv2.rectangle(
        m, (round(ox), round(oy)), (round(ox + body_w) - 1, round(oy + body_h) - 1), 255, -1
    )
    r = 0.18 * min(body_w, body_h)
    mx, my = ox + body_w / 2, oy + body_h / 2
    for kind, (x, y, dx, dy) in zip(
        sides,
        [(mx, oy, 0, -1), (mx, oy + body_h, 0, 1), (ox, my, -1, 0), (ox + body_w, my, 1, 0)],
        strict=True,
    ):
        if kind == 1:
            cv2.circle(m, (round(x + dx * r * 0.8), round(y + dy * r * 0.8)), round(r), 255, -1)
        elif kind == 2:
            cv2.circle(m, (round(x - dx * r * 0.8), round(y - dy * r * 0.8)), round(r), 0, -1)
    return m


def cut_piece(reference, cols, rows, col, row, wrist_px=96, angle=None, seed=0, degrade=True):
    """A piece as a wrist camera sees it: (rgb, mask, angle_deg, px_per_cell_width).

    ``wrist_px`` is the piece body's width in the crop; ``angle`` (CCW as displayed, degrees) is
    random when None. Sides on the board's edge are straight, others random tab or blank."""
    rng = np.random.default_rng(seed)
    h, w = reference.shape[:2]
    cw, ch = w / cols, h / rows
    s = wrist_px / cw
    bw, bh = wrist_px, ch * s
    canvas = int(np.ceil(1.7 * max(bw, bh)))
    sides = [int(rng.integers(1, 3)) for _ in range(4)]
    for i, edge in enumerate((row == 0, row == rows - 1, col == 0, col == cols - 1)):
        if edge:
            sides[i] = 0
    m = piece_mask(bw, bh, sides, canvas)
    big = cv2.resize(reference, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cx, cy = (col + 0.5) * cw * s, (row + 0.5) * ch * s
    x0, y0 = round(cx - canvas / 2), round(cy - canvas / 2)
    img = np.full((canvas, canvas, 3), MAT, np.uint8)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(big.shape[1], x0 + canvas), min(big.shape[0], y0 + canvas)
    img[ys - y0 : ye - y0, xs - x0 : xe - x0] = big[ys:ye, xs:xe]
    outside = np.ones((canvas, canvas), bool)
    outside[ys - y0 : ye - y0, xs - x0 : xe - x0] = False
    m[outside] = 0
    img[m == 0] = MAT
    ang = float(rng.uniform(0, 360)) if angle is None else float(angle)
    rot = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), ang, 1.0)
    img = cv2.warpAffine(img, rot, (canvas, canvas), borderValue=(MAT, MAT, MAT))
    m = cv2.warpAffine(m, rot, (canvas, canvas), flags=cv2.INTER_NEAREST)
    if degrade:
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        gain = rng.uniform(0.8, 1.2, 3)
        noisy = img.astype(np.float32) * gain + rng.normal(0, 4, img.shape)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img, m > 0, ang, wrist_px / cw
