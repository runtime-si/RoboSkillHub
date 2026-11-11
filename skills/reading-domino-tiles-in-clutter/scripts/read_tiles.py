# Pip counting adapted from pipcounter (github.com/jjmlovesgit/pipcounter at 79cad4071f4c, pipperjones.py,
# MIT, licence in LICENSE.pipcounter): saturated-colour mask in HSV, external contours, an area window
# for a pip. Changed: the area window scales with the tile (from its bar) instead of 200-10000 px at
# 512 x 512, and each pip is counted for the tile half it lies in. New here: tiles are found by their
# black centre bar (dark, thin, flanked by white), which separates touching tiles; the bar gives each
# tile's centre, axis and width.
"""Read every domino in a cluttered top-down photo: each tile's two halves, centre and angle, from its bar and pips."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

WORK_PX = 1200
"""Longest side the frame is resized to before reading [px]."""

BAR_FRACTION = 0.8
"""A tile's centre bar spans this share of the tile's width (double-six / double-nine sets)."""

BAR_TOL = 0.15
"""Bars within this share of the most common bar length are tiles; 0.25 let the dark seam between two
touching rows of tiles in through (a fifth tile in a 2 x 2 photo)."""

MIN_ELONGATION = 5.0
"""Bar length over thickness."""

PIP_R = (0.05, 0.16)
"""A pip's radius range as a share of the tile width."""

MAX_HALF = 9
"""Largest value of a half (double-nine sets)."""

ROUTES = ("tiles_read", "ambiguous", "no_tiles")


class Output(TypedDict):
    route: str
    detail: str
    tiles: list[dict[str, Any]]
    n_tiles: int
    total_pips: int
    camera_name: str


def _bars(im: np.ndarray, hsv: np.ndarray) -> list[tuple[float, float, float, float]]:
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    white = cv2.morphologyEx(
        ((s < 60) & (v > 150)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    near_white = cv2.dilate(white, np.ones((9, 9), np.uint8))
    dark = ((v < 80) & (s < 90)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 30 or a > 0.01 * im.shape[0] * im.shape[1]:
            continue
        comp = lab == i
        ys, xs = np.nonzero(comp)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        length, thick = max(rw, rh), max(1.0, min(rw, rh))
        if length / thick < MIN_ELONGATION or length < 15:
            continue
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~comp
        if not ring.any() or near_white[ring].mean() < 0.7:
            continue
        out.append(
            (float(cx), float(cy), float(length), math.radians(ang if rw >= rh else ang + 90.0))
        )
    if not out:
        return []
    lens = np.array([b[2] for b in out])
    ref = max((int((np.abs(lens - b) <= BAR_TOL * b).sum()), float(b)) for b in lens)[1]
    return [b for b in out if abs(b[2] - ref) <= BAR_TOL * ref]


def read(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[list[dict[str, Any]], int]:
    """Tiles (in input pixels) and the number of pip-sized blobs no tile claimed."""
    h, w = rgb.shape[:2]
    sc = WORK_PX / max(h, w)
    im = cv2.resize(rgb, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    if mask is not None:
        m = (
            cv2.resize(
                np.asarray(mask).astype(np.uint8),
                (im.shape[1], im.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        im = np.where(m[..., None], im, 0).astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    tiles = [
        {"cx": cx, "cy": cy, "W": length / BAR_FRACTION, "bar": th, "halves": [0, 0]}
        for cx, cy, length, th in _bars(im, hsv)
    ]
    sat = ((hsv[..., 1] > 80) & (hsv[..., 2] > 60)).astype(np.uint8)
    contours, _ = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stray = 0
    for c in contours:
        a = cv2.contourArea(c)
        mo = cv2.moments(c)
        if mo["m00"] == 0:
            continue
        px, py = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        best = None
        for t in tiles:
            W = t["W"]
            if not (math.pi * (PIP_R[0] * W) ** 2 <= a <= math.pi * (PIP_R[1] * W) ** 2):
                continue
            ux, uy = math.cos(t["bar"]), math.sin(t["bar"])
            dx, dy = px - t["cx"], py - t["cy"]
            across, along = dx * ux + dy * uy, -dx * uy + dy * ux
            if abs(across) <= 0.5 * W and 0.05 * W < abs(along) <= W:
                d = abs(across) + abs(along)
                if best is None or d < best[0]:
                    best = (d, t, along)
        if best:
            best[1]["halves"][0 if best[2] < 0 else 1] += 1
        elif (
            tiles
            and math.pi * (PIP_R[0] * tiles[0]["W"]) ** 2
            <= a
            <= math.pi * (PIP_R[1] * tiles[0]["W"]) ** 2
        ):
            stray += 1
    out = []
    for t in tiles:
        axis = math.degrees(t["bar"]) + 90.0
        out.append(
            {
                "left": int(t["halves"][0]),
                "right": int(t["halves"][1]),
                "value": sorted(t["halves"]),
                "center_px": [t["cx"] / sc, t["cy"] / sc],
                "angle_deg": round((axis + 90.0) % 180.0 - 90.0, 2),
                "width_px": round(t["W"] / sc, 1),
            }
        )
    return out, stray


def run(
    ctx: NodeContext,
    frame: dict[str, Any] | None = None,
    camera: str | None = None,
    mask: Any = None,
) -> Output:
    out: Output = {
        "route": "no_tiles",
        "detail": "",
        "tiles": [],
        "n_tiles": 0,
        "total_pips": 0,
        "camera_name": "",
    }
    if frame is None:
        cams = [
            c
            for c in ctx.tool("robot.get_observation").get("cameras") or []
            if c.get("rgb") is not None
        ]
        frame = (
            next((c for c in cams if c.get("name") == camera), None)
            if camera
            else (cams[0] if cams else None)
        )
    if frame is None or frame.get("rgb") is None:
        out["detail"] = "no RGB frame"
        return out
    out["camera_name"] = str(frame.get("name", ""))
    rgb = np.ascontiguousarray(np.asarray(frame["rgb"])[..., :3]).astype(np.uint8)
    tiles, stray = read(rgb, None if mask is None else np.asarray(mask))
    out["tiles"], out["n_tiles"] = tiles, len(tiles)
    out["total_pips"] = int(sum(t["left"] + t["right"] for t in tiles))
    if not tiles:
        out["detail"] = "no tile bar found"
        return out
    bad = [i for i, t in enumerate(tiles) if max(t["left"], t["right"]) > MAX_HALF]
    if bad or stray:
        out["route"] = "ambiguous"
        out["detail"] = (
            f"{len(tiles)} tiles; halves over {MAX_HALF} on tiles {bad}; {stray} pip-sized blobs on no tile"
        )
        return out
    out["route"] = "tiles_read"
    out["detail"] = f"{len(tiles)} tiles, {out['total_pips']} pips"
    return out
