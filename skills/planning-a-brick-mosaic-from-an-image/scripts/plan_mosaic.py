# Adapted from brickmos (https://github.com/merschformann/brickmos), brickmos/brickify.py at
# commit 4a7e956720d347d198f931edc38852c9f3bf4060 (read_colors, closest_color,
# replace_with_brick_colors, add_grid, write_xml and main's resize-then-replace pipeline) and
# brickmos/defaults.py (the 33-colour BrickLink palette, kept as the "brickmos" palette), MIT
# License, Copyright (c) 2020 Marius Merschformann; the licence is LICENSE.brickmos beside
# this skill.
# Changes: brickmos stretches the image to the mosaic size with bilinear resizing, maps each cell
# to the nearest colour by Euclidean distance in sRGB and counts the colours. Here the image is
# centre-cropped (or padded, or stretched as upstream) and box-averaged in linear light, matched by
# CIEDE2000 in CIELAB (the hub's colour tool's formula), optionally Floyd-Steinberg dithered (error
# diffused in Lab, serpentine), assigned under per-colour part counts (an optimal transportation
# program when the stock is short; substitutions are reported), merged into larger plates (1x2,
# 2x2, ...) where a run of one colour allows, and written out as an ordered placement list in the
# hub's stud-grid convention (cell [x, y, layer], size 'AxB', rotation) with the inputs of
# planning-brick-assembly-order and checking-design-connectivity-and-collisions. The bill of
# materials keeps brickmos' BrickLink wanted-list XML; the preview keeps its helper grid and adds
# part outlines and studs.
"""Plan a single-layer brick mosaic of an image: colours, parts, bill of materials, placement order."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as xmlET
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic_ciede2000 import deltaE_ciede2000  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

LEGO = [
    ("black", 0, 11, "1B2A34"),
    ("blue", 1, 7, "1E5AA8"),
    ("green", 2, 6, "00852B"),
    ("dark_pink", 5, 47, "D3359D"),
    ("bright_green", 10, 36, "58AB41"),
    ("yellow", 14, 3, "FAC80A"),
    ("white", 15, 1, "F4F4F4"),
    ("tan", 19, 2, "D7BA8C"),
    ("orange", 25, 4, "D67923"),
    ("magenta", 26, 71, "901F76"),
    ("lime", 27, 34, "A5CA18"),
    ("dark_tan", 28, 69, "897D62"),
    ("reddish_brown", 70, 88, "5F3109"),
    ("light_bluish_grey", 71, 86, "969696"),
    ("dark_bluish_grey", 72, 85, "646464"),
    ("medium_blue", 73, 42, "7396C8"),
    ("bright_light_orange", 191, 110, "FCAC00"),
    ("bright_light_yellow", 226, 103, "FFEC6C"),
    ("dark_green", 288, 80, "00451A"),
    ("dark_red", 320, 59, "720012"),
    ("dark_azure", 321, 153, "469BC3"),
    ("medium_azure", 322, 156, "68C3E2"),
    ("sand_green", 378, 48, "708E7C"),
    ("dark_orange", 484, 68, "91501C"),
    ("red", 4, 5, "B40000"),
]
"""(name, LDraw code, BrickLink colour id, sRGB hex): the hub colour tool's "lego" palette - the
25 current solid colours of LDraw.org's LDConfig.ldr (2026-05-29 update, CCAL 2.0 / CC BY 2.0,
attribution LDraw.org) - with BrickLink's colour ids (those of brickmos' table where it has the
colour; magenta 71 and sand green 48 from BrickLink's colour guide). LDraw's values are rendering
colours, not measured plastic."""

LEGO_BASIC_NAMES = (
    "red", "blue", "yellow", "green", "black", "white", "light_bluish_grey", "dark_bluish_grey",
    "tan", "orange", "lime", "dark_green", "reddish_brown", "medium_azure",
)  # fmt: skip
"""The colour tool's "lego_basic": the 14 commonest colours (basic-brick sets)."""

BRICKMOS = [
    ((255, 255, 255), "White", 1),
    ((175, 181, 199), "LightBluishGray", 86),
    ((89, 93, 96), "DarkBluishGray", 85),
    ((33, 33, 33), "Black", 11),
    ((106, 14, 21), "DarkRed", 59),
    ((179, 0, 6), "Red", 5),
    ((88, 42, 18), "ReddishBrown", 88),
    ((222, 198, 156), "Tan", 2),
    ((144, 116, 80), "DarkTan", 69),
    ((227, 160, 91), "MediumNougat", 150),
    ((179, 84, 8), "DarkOrange", 68),
    ((255, 126, 20), "Orange", 4),
    ((247, 186, 48), "BrightLightOrange", 110),
    ((247, 209, 23), "Yellow", 3),
    ((241, 225, 103), "BrightLightYellow", 103),
    ((223, 238, 165), "YellowishGreen", 158),
    ((166, 202, 85), "Lime", 34),
    ((127, 143, 86), "OliveGreen", 155),
    ((46, 85, 67), "DarkGreen", 80),
    ((0, 100, 46), "Green", 6),
    ((16, 203, 49), "BrightGreen", 36),
    ((0, 138, 128), "DarkTurquoise", 39),
    ((20, 48, 68), "DarkBlue", 63),
    ((0, 87, 166), "Blue", 7),
    ((73, 151, 250), "DarkAzure", 153),
    ((95, 189, 247), "MediumAzure", 156),
    ((97, 175, 255), "MediumBlue", 42),
    ((164, 194, 230), "BrightLightBlue", 105),
    ((90, 113, 132), "SandBlue", 55),
    ((95, 38, 131), "DarkPurple", 89),
    ((136, 94, 158), "MediumLavender", 157),
    ((200, 112, 128), "DarkPink", 47),
    ((255, 187, 255), "BrightPink", 104),
]
"""brickmos' defaults.py table verbatim: (sRGB, BrickLink colour name, BrickLink colour id)."""

PARTS = {
    "lego": {
        "plate": {"1x1": "3024", "1x2": "3023", "1x3": "3623", "1x4": "3710", "1x6": "3666",
                  "1x8": "3460", "2x2": "3022", "2x3": "3021", "2x4": "3020", "2x6": "3795"},
        "brick": {"1x1": "3005", "1x2": "3004", "1x3": "3622", "1x4": "3010", "1x6": "3009",
                  "1x8": "3008", "2x2": "3003", "2x3": "3002", "2x4": "3001", "2x6": "2456"},
        "tile": {"1x1": "3070b", "1x2": "3069b", "1x3": "63864", "1x4": "2431", "1x6": "6636",
                 "1x8": "4162", "2x2": "3068b", "2x4": "87079"},
    },
}  # fmt: skip
"""LEGO part numbers (BrickLink / LDraw ids) for the sizes a mosaic uses. brickmos orders 3024
(plate 1x1) only; the brick ids are checking-design-connectivity-and-collisions' LDraw table.
DUPLO gets no ids (``part`` is None; an id is only a BOM label)."""

# ---------------------------------------------------------------------------
# Constants (each with where it came from)
# ---------------------------------------------------------------------------

FIRST_LAYER = 1
"""Layer of a part standing on the baseplate: the hub convention (ICL's z,
computing-brick-pose-from-stud-grid, planning-brick-assembly-order, checking-design-...)."""

PLATE_STUDS = (32, 32)
"""Default baseplate: the common 32 x 32 LEGO baseplate (brickmos' default mosaic is 48 x 48)."""

GRID_CELL = 8
"""Helper-grid spacing in the preview, in studs: brickmos' --grid_cell default 8x8."""

PREVIEW_PX = 16
"""Preview pixels per stud (a 32 x 32 mosaic is 512 px; brickmos renders 1000 x 1000)."""

SUBSTITUTION_BIG = 1000.0
"""Bonus per covered cell in the short-stock program, larger than any CIEDE2000 between two sRGB
colours (at most ~120), so covering a cell always beats leaving it empty."""

BLUR_SIGMA = 1.0
"""Gaussian sigma in studs for the "seen from a step back" error: S-CIELAB's idea (blur both images
in Lab before the colour difference) at the scale where neighbouring studs merge - the scale at
which dithering is meant to work."""

MAX_CELLS = 128 * 128
"""Largest mosaic: four 48 x 48 baseplates are 9,216 cells; 128 x 128 (16,384) planned in a few
seconds in the timing table (SKILL.md)."""

SPARES = 0

BASEPLATE_RGB = (150, 150, 150)
"""Colour of an uncovered stud (the baseplate) in the preview and the blurred error: LDraw light
bluish grey #969696, the commonest baseplate colour."""
"""Spare parts per BOM line: brickmos' --spares default (its write_xml default is 5)."""


class Output(TypedDict):
    route: str
    detail: str
    mosaic_size: list[int]
    origin: list[int]
    plate_studs: list[int]
    n_cells: int
    n_placements: int
    n_empty: int
    n_substituted: int
    n_colours: int
    mean_delta_e: float
    median_delta_e: float
    p95_delta_e: float
    max_delta_e: float
    mean_delta_e_blurred: float
    ideal_mean_delta_e: float
    placements: list[dict[str, Any]]
    bill_of_materials: list[dict[str, Any]]
    substitutions: list[dict[str, Any]]
    empty_cells: list[list[int]]
    colour_grid: list[list[str | None]]
    placements_json: str
    steps_json: str
    bom_json: str
    bricklink_xml: str
    preview_path: str
    pixelated_path: str
    out_dir: str
    elapsed_s: float


class _Bad(Exception):
    pass


class _BadImage(Exception):
    pass


# ---------------------------------------------------------------------------
# Colour space (sRGB IEC 61966-2-1, D65; the colour tool's conversions)
# ---------------------------------------------------------------------------

_XYZ_FROM_RGB = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.clip(c, 0.0, None) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    xyz = (np.asarray(lin, dtype=np.float64) @ _XYZ_FROM_RGB.T) / _D65
    eps = (6.0 / 29.0) ** 3
    f = np.where(
        xyz > eps, np.cbrt(np.clip(xyz, eps, None)), xyz / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0
    )
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB to sRGB 0..1 (clipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    d = 6.0 / 29.0
    xyz = np.where(f > d, f**3, 3 * d * d * (f - 4.0 / 29.0)) * _D65
    return np.clip(linear_to_srgb(xyz @ _RGB_FROM_XYZ.T), 0.0, 1.0)


def srgb_to_lab(rgb_unit: np.ndarray) -> np.ndarray:
    return linear_to_lab(srgb_to_linear(rgb_unit))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _key(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("gray", "grey")


def _hex_rgb(h: str) -> list[float]:
    h = h.lstrip("#")
    return [float(int(h[i : i + 2], 16)) for i in (0, 2, 4)]


def _palette(palette: Any) -> list[dict[str, Any]]:
    """Palette entries {name, rgb (0..255), lab, ldraw, bricklink_id}."""
    if palette is None or palette == "lego":
        raw = [{"name": n, "ldraw": c, "bricklink_id": b, "hex": h} for n, c, b, h in LEGO]
    elif palette == "lego_basic":
        raw = [
            {"name": n, "ldraw": c, "bricklink_id": b, "hex": h}
            for n, c, b, h in LEGO
            if n in LEGO_BASIC_NAMES
        ]
    elif palette == "brickmos":
        raw = [{"name": n, "bricklink_id": b, "rgb": list(rgb)} for rgb, n, b in BRICKMOS]
    elif isinstance(palette, str):
        raise _Bad(f"unknown palette {palette!r} (lego, lego_basic, brickmos, or a list)")
    else:
        raw = list(palette)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in raw:
        if not isinstance(e, dict) or "name" not in e:
            raise _Bad(f"palette entries are {{name, rgb|hex|lab}} dicts, got {e!r}")
        if e.get("lab") is not None:
            lab = np.asarray(e["lab"], dtype=np.float64).reshape(3)
            rgb = lab_to_srgb(lab) * 255.0
        elif e.get("rgb") is not None:
            rgb = np.asarray(e["rgb"], dtype=np.float64).reshape(3)
            rgb = rgb * 255.0 if rgb.max() <= 1.0 else rgb
            lab = srgb_to_lab(rgb / 255.0)
        elif e.get("hex") is not None:
            rgb = np.asarray(_hex_rgb(str(e["hex"])))
            lab = srgb_to_lab(rgb / 255.0)
        else:
            raise _Bad(f"palette entry {e['name']!r} has no rgb, hex or lab")
        k = _key(e["name"])
        if k in seen:
            raise _Bad(f"palette colour {e['name']!r} appears twice")
        seen.add(k)
        out.append(
            {
                "name": str(e["name"]),
                "key": k,
                "rgb": [float(v) for v in rgb],
                "lab": lab,
                "ldraw": e.get("ldraw"),
                "bricklink_id": e.get("bricklink_id"),
            }
        )
    if not out:
        raise _Bad("empty palette")
    return out


def _size(value: Any) -> tuple[int, int]:
    """'AxB' / [A, B] -> (min, max)."""
    try:
        if isinstance(value, str):
            a, b = (int(v) for v in value.lower().split("x"))
        else:
            a, b = (int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise _Bad(f"a size is 'AxB' such as '1x2', got {value!r}") from exc
    if a < 1 or b < 1:
        raise _Bad(f"size {value!r} must be positive")
    return min(a, b), max(a, b)


def _label(s: tuple[int, int]) -> str:
    return f"{s[0]}x{s[1]}"


def _load_image(image: Any, background_rgb: list[float]) -> np.ndarray:
    """RGB float 0..1 [H, W, 3] from an array (RGB, uint8 or float 0..1) or an image file path."""
    if image is None:
        raise _BadImage("no image given")
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.isfile(path):
            raise _BadImage(f"no image file at {path}")
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise _BadImage(f"cannot decode {path} as an image")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr = np.asarray(image)
    if arr.dtype == object or arr.size == 0:
        raise _BadImage("the image is empty or not numeric")
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise _BadImage(f"an image is [H, W], [H, W, 3] or [H, W, 4], got shape {arr.shape}")
    if arr.dtype == np.uint16:
        f = arr.astype(np.float64) / 65535.0
    elif np.issubdtype(arr.dtype, np.integer):
        f = arr.astype(np.float64) / 255.0
    else:
        f = arr.astype(np.float64)
        if not np.all(np.isfinite(f)):
            raise _BadImage("the image has NaN or infinite values")
        if f.max() > 1.0 + 1e-6:
            f = f / 255.0
    f = np.clip(f, 0.0, 1.0)
    if f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    if f.shape[2] == 4:
        alpha = f[..., 3:4]
        bg = np.asarray(background_rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0
        f = srgb_to_linear(f[..., :3]) * alpha + srgb_to_linear(bg) * (1 - alpha)
        f = linear_to_srgb(f)
    return f


def _fit(img: np.ndarray, w: int, h: int, fit: str, pad_rgb: list[float]) -> np.ndarray:
    """Crop (centre, to the mosaic's aspect), pad (letterbox in pad_rgb) or leave for stretching."""
    ih, iw = img.shape[:2]
    target = w / h
    if fit == "stretch" or abs(iw / ih - target) < 1e-9:
        return img
    if fit == "crop":
        if iw / ih > target:
            cw = max(1, round(ih * target))
            x0 = (iw - cw) // 2
            return img[:, x0 : x0 + cw]
        ch = max(1, round(iw / target))
        y0 = (ih - ch) // 2
        return img[y0 : y0 + ch]
    if iw / ih > target:
        nh = round(iw / target)
        top = (nh - ih) // 2
        out = np.empty((nh, iw, 3))
        out[:] = np.asarray(pad_rgb, dtype=np.float64) / 255.0
        out[top : top + ih] = img
        return out
    nw = round(ih * target)
    left = (nw - iw) // 2
    out = np.empty((ih, nw, 3))
    out[:] = np.asarray(pad_rgb, dtype=np.float64) / 255.0
    out[:, left : left + iw] = img
    return out


def _cells(img: np.ndarray, w: int, h: int, resample: str) -> np.ndarray:
    """The image at w x h cells, sRGB 0..1. 'area' averages each cell's pixels in linear light;
    'linear' is brickmos' cv2.INTER_LINEAR on sRGB."""
    if resample == "linear":
        return np.clip(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
    lin = srgb_to_linear(img).astype(np.float32)
    ih, iw = img.shape[:2]
    if ih < h or iw < w:  # INTER_AREA upsamples bilinearly; nearest keeps a small image's pixels
        small = cv2.resize(lin, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        small = cv2.resize(lin, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(linear_to_srgb(small.astype(np.float64)), 0, 1)


def _stock(
    inventory: dict[str, Any] | None, pal: list[dict[str, Any]], sizes: list[tuple[int, int]]
) -> list[dict[tuple[int, int], int | None]]:
    """Per palette colour: size -> count (None = unlimited). An int value counts 1x1 parts."""
    if inventory is None:
        return [dict.fromkeys(sizes) for _ in pal]
    if not isinstance(inventory, dict):
        raise _Bad("inventory is {colour: count} or {colour: {'1x1': n, '1x2': m}}")
    by_key = {p["key"]: i for i, p in enumerate(pal)}
    stock: list[dict[tuple[int, int], int | None]] = [dict.fromkeys(sizes, 0) for _ in pal]
    unknown = [k for k in inventory if _key(k) not in by_key]
    if unknown:
        raise _Bad(f"inventory colours not in the palette: {unknown}")
    for name, v in inventory.items():
        i = by_key[_key(name)]
        if isinstance(v, dict):
            for s, n in v.items():
                sz = _size(s)
                if sz not in stock[i]:
                    raise _Bad(f"inventory size {s!r} of {name!r} is not in sizes")
                stock[i][sz] = _count(n, name)
        else:
            stock[i][(1, 1)] = _count(v, name)
    return stock


def _count(n: Any, name: str) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError) as exc:
        raise _Bad(f"inventory count for {name!r} must be an integer, got {n!r}") from exc
    if v < 0:
        raise _Bad(f"inventory count for {name!r} is negative")
    return v


# ---------------------------------------------------------------------------
# Colour assignment
# ---------------------------------------------------------------------------


def _transport(cost: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Min-cost assignment of cells (rows of cost) to colours with capacities (inf = unlimited),
    covering as many cells as the capacities allow. Identical cells are grouped; the program is a
    transportation problem, so the simplex vertex is integral. Returns the colour per cell, -1 =
    left empty."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n, k = cost.shape
    rounded = np.round(cost, 6)
    groups, inverse, mult = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    g = len(groups)
    usable = np.flatnonzero(caps > 0)
    ku = len(usable)
    if ku == 0:
        return np.full(n, -1, dtype=int)
    c = groups[:, usable].reshape(-1)
    short = np.isfinite(caps).all() and caps.sum() < n
    if short:
        c = c - SUBSTITUTION_BIG
    var = np.arange(g * ku)
    rows_g = np.repeat(np.arange(g), ku)
    a_group = coo_matrix((np.ones(g * ku), (rows_g, var)), shape=(g, g * ku))
    capped = [j for j in range(ku) if np.isfinite(caps[usable[j]])]
    a_cap = coo_matrix(
        (
            np.ones(g * len(capped)),
            (
                np.repeat(np.arange(len(capped)), g),
                np.concatenate([np.arange(g) * ku + j for j in capped]) if capped else [],
            ),
        ),
        shape=(len(capped), g * ku),
    )
    b_cap = caps[usable[capped]] if capped else np.zeros(0)
    if short:
        a_ub = _vstack(a_group, a_cap)
        b_ub = np.concatenate([mult, b_cap])
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs-ds")
    else:
        res = linprog(
            c,
            A_ub=a_cap if capped else None,
            b_ub=b_cap if capped else None,
            A_eq=a_group,
            b_eq=mult,
            bounds=(0, None),
            method="highs-ds",
        )
    if res.status != 0:
        raise RuntimeError(f"assignment program failed: {res.message}")
    x = np.rint(res.x).astype(int).reshape(g, ku)
    out = np.full(n, -1, dtype=int)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(mult)[:-1]])
    for gi in range(g):
        members = order[starts[gi] : starts[gi] + mult[gi]]
        # The best colour goes to the first cells; a group's members are identical in colour.
        pos = 0
        for j in np.argsort(groups[gi, usable]):
            take = min(x[gi, j], len(members) - pos)
            out[members[pos : pos + take]] = usable[j]
            pos += take
    return out


def _vstack(a: Any, b: Any) -> Any:
    from scipy.sparse import vstack

    return vstack([a, b]).tocsr()


def _dither(
    lab: np.ndarray, pal_lab: np.ndarray, caps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Floyd-Steinberg error diffusion in CIELAB, serpentine, nearest by CIEDE2000 among the colours
    with stock left. Returns the colour per cell (-1 = no stock left), the ideal colour (nearest of
    the whole palette to the diffused target) and the diffused targets."""
    h, w, _ = lab.shape
    work = lab.copy()
    out = np.full((h, w), -1, dtype=int)
    ideal = np.zeros((h, w), dtype=int)
    targets = np.zeros_like(lab)
    left = caps.astype(float).copy()
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        step = 1 if r % 2 == 0 else -1
        for c in cols:
            t = work[r, c]
            t = np.array(
                [np.clip(t[0], 0.0, 100.0), np.clip(t[1], -128, 127), np.clip(t[2], -128, 127)]
            )
            targets[r, c] = t
            d = deltaE_ciede2000(t[None, :], pal_lab)
            ideal[r, c] = int(np.argmin(d))
            d = np.where(left > 0, d, np.inf)
            j = int(np.argmin(d))
            if not np.isfinite(d[j]):
                continue
            out[r, c] = j
            left[j] -= 1
            err = t - pal_lab[j]
            for dr, dc, wgt in (
                (0, step, 7 / 16),
                (1, -step, 3 / 16),
                (1, 0, 5 / 16),
                (1, step, 1 / 16),
            ):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    work[rr, cc] += err * wgt
    return out, ideal, targets


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def _merge(
    grid: np.ndarray, sizes: list[tuple[int, int]], stock: list[dict[tuple[int, int], int | None]]
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[tuple[int, int], int | None]]]:
    """Greedy tiling: larger sizes first (both orientations), anchors in raster order (image top
    left first), a part wherever its footprint is one colour, free, and that colour has the part
    left. Returns parts {r, c, rows, cols, colour, size} in image cells, the cells whose colour has
    no 1x1 left, and the stock left."""
    h, w = grid.shape
    free = grid >= 0
    left = [dict(s) for s in stock]
    parts: list[dict[str, Any]] = []
    for sz in sorted(sizes, key=lambda s: (-s[0] * s[1], -s[1])):
        if sz == (1, 1):
            continue
        for rows, cols in {(sz[0], sz[1]), (sz[1], sz[0])}:
            if rows > h or cols > w:
                continue
            for r in range(h - rows + 1):
                for c in range(w - cols + 1):
                    if not free[r, c]:
                        continue
                    k = grid[r, c]
                    n = left[k].get(sz)
                    if n is not None and n <= 0:
                        continue
                    win = grid[r : r + rows, c : c + cols]
                    if not (free[r : r + rows, c : c + cols].all() and (win == k).all()):
                        continue
                    free[r : r + rows, c : c + cols] = False
                    if n is not None:
                        left[k][sz] = n - 1
                    parts.append(
                        {"r": r, "c": c, "rows": rows, "cols": cols, "colour": int(k), "size": sz}
                    )
    leftover = np.zeros((h, w), dtype=bool)
    for r, c in zip(*np.nonzero(free), strict=True):
        k = grid[r, c]
        n = left[k].get((1, 1))
        if n is not None and n <= 0:
            leftover[r, c] = True
            continue
        if n is not None:
            left[k][(1, 1)] = n - 1
        parts.append(
            {"r": int(r), "c": int(c), "rows": 1, "cols": 1, "colour": int(k), "size": (1, 1)}
        )
    return parts, leftover, left


# ---------------------------------------------------------------------------
# Rendering and export
# ---------------------------------------------------------------------------


def _render(
    grid: np.ndarray,
    parts: list[dict[str, Any]],
    pal: list[dict[str, Any]],
    px: int,
    grid_cell: int | None,
    studs: bool,
    baseplate: list[float],
) -> np.ndarray:
    """RGB uint8 preview: parts filled in their palette colour, outlined, with studs; empty cells
    hatched; brickmos' helper grid every grid_cell studs."""
    h, w = grid.shape
    img = np.zeros((h * px, w * px, 3), dtype=np.uint8)
    img[:] = tuple(int(v) for v in baseplate)
    for r, c in zip(*np.nonzero(grid < 0), strict=True):
        y0, x0 = r * px, c * px
        cv2.line(img, (x0, y0), (x0 + px - 1, y0 + px - 1), (90, 90, 90), 1)
        cv2.line(img, (x0 + px - 1, y0), (x0, y0 + px - 1), (90, 90, 90), 1)
    for p in parts:
        rgb = np.asarray(pal[p["colour"]]["rgb"])
        col = tuple(int(v) for v in rgb)
        edge = tuple(int(v) for v in rgb * 0.6)
        hi = tuple(int(min(255, v * 0.85 + 40)) for v in rgb)
        y0, x0 = p["r"] * px, p["c"] * px
        y1, x1 = y0 + p["rows"] * px - 1, x0 + p["cols"] * px - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), edge, 1)
        if studs and px >= 6:
            for i in range(p["rows"]):
                for j in range(p["cols"]):
                    cy, cx = y0 + i * px + px // 2, x0 + j * px + px // 2
                    cv2.circle(img, (cx, cy), max(1, int(px * 0.3)), hi, 1, cv2.LINE_AA)
    if grid_cell:
        lw = max(1, round(0.002 * min(img.shape[:2])))
        for i in range(0, w + 1, grid_cell):
            cv2.line(img, (i * px, 0), (i * px, h * px), (0, 0, 0), lw)
        for j in range(0, h + 1, grid_cell):
            cv2.line(img, (0, j * px), (w * px, j * px), (0, 0, 0), lw)
    return img


def _bricklink_xml(bom: list[dict[str, Any]], spares: int) -> str:
    """brickmos' write_xml: a BrickLink wanted list, one ITEM per part and colour."""
    root = xmlET.Element("INVENTORY")
    for line in bom:
        if not line.get("part") or line.get("bricklink_colour_id") is None:
            continue
        item = xmlET.SubElement(root, "ITEM")
        xmlET.SubElement(item, "ITEMTYPE").text = "P"
        xmlET.SubElement(item, "ITEMID").text = str(line["part"])
        xmlET.SubElement(item, "COLOR").text = str(line["bricklink_colour_id"])
        xmlET.SubElement(item, "MINQTY").text = str(line["count"] + spares)
        xmlET.SubElement(item, "CONDITION").text = "X"
    return xmlET.tostring(root, encoding="unicode")


def _empty(detail: str, route: str = "invalid_input", **kw: Any) -> Output:
    out: Output = {
        "route": route, "detail": detail, "mosaic_size": [], "origin": [], "plate_studs": [],
        "n_cells": 0, "n_placements": 0, "n_empty": 0, "n_substituted": 0, "n_colours": 0,
        "mean_delta_e": 0.0, "median_delta_e": 0.0, "p95_delta_e": 0.0, "max_delta_e": 0.0,
        "mean_delta_e_blurred": 0.0, "ideal_mean_delta_e": 0.0, "placements": [],
        "bill_of_materials": [], "substitutions": [], "empty_cells": [], "colour_grid": [],
        "placements_json": "[]", "steps_json": "[]", "bom_json": "[]",
        "bricklink_xml": "", "preview_path": "", "pixelated_path": "", "out_dir": "",
        "elapsed_s": 0.0,
    }  # fmt: skip
    out.update(kw)  # type: ignore[typeddict-item]
    return out


def _pair(v: Any, name: str) -> tuple[int, int]:
    if isinstance(v, str):
        v = v.lower().split("x")
    try:
        a, b = (int(x) for x in v)
    except (TypeError, ValueError) as exc:
        raise _Bad(f"{name} is [nx, ny] or 'NXxNY', got {v!r}") from exc
    return a, b


# ---------------------------------------------------------------------------
# The skill
# ---------------------------------------------------------------------------


def run(
    ctx: NodeContext,
    image: Any = None,
    image_path: str | None = None,
    plate_studs: Any = PLATE_STUDS,
    mosaic_size: Any = None,
    origin: Any = (0, 0),
    palette: Any = "lego",
    inventory: dict[str, Any] | None = None,
    sizes: list[Any] | None = None,
    part_kind: str = "plate",
    system: str = "lego",
    fit: str = "crop",
    resample: str = "area",
    dither: str = "none",
    order: str = "rows",
    background_rgb: list[float] | None = None,
    pad_rgb: list[float] | None = None,
    baseplate_rgb: list[float] | None = None,
    out_dir: str | None = None,
    write_files: bool = True,
    preview_px: int = PREVIEW_PX,
    grid_cell: int | None = GRID_CELL,
    preview_studs: bool = True,
    spares: int = SPARES,
) -> Output:
    """Plan a one-layer mosaic of ``image`` on a baseplate.

    Args:
        image: RGB array ([H, W], [H, W, 3] or RGBA; uint8 or float 0..1) or an image file path;
            ``image_path`` is the same as a path. Seen from above the plate: image right = plate
            +x, image up = plate +y.
        plate_studs: the baseplate [nx, ny]; mosaic_size: the mosaic [w, h] in studs (default the
            plate less origin); origin: the plate cell of the mosaic's lowest corner.
        palette: ``lego`` (25 LDraw colours), ``lego_basic`` (14), ``brickmos`` (brickmos' 33
            BrickLink colours) or a list of {name, rgb | hex | lab, ldraw?, bricklink_id?}.
        inventory: parts on hand, {colour: n 1x1 parts} or {colour: {"1x1": n, "1x2": m}};
            None = unlimited. Colours not listed have none.
        sizes: part sizes to use, e.g. ["1x1", "1x2", "2x2"] (default ["1x1"]); must include 1x1.
        part_kind: ``plate`` (default, brickmos' 3024), ``brick`` or ``tile``; system lego / duplo.
        fit: ``crop`` (centre crop to the mosaic's aspect), ``pad`` (letterbox in pad_rgb) or
            ``stretch`` (brickmos); resample: ``area`` (linear-light box average) or ``linear``
            (brickmos' bilinear); dither: ``none`` or ``floyd_steinberg``.
        order: ``rows`` (ascending plate y, then x), ``serpentine`` or ``colour`` (one colour at a
            time, most used first, rows within).
        baseplate_rgb: the colour an empty stud shows (preview, blurred error).
        out_dir: where mosaic.png, pixelated.png, bricklink.xml and plan.json go (default a new
            temp dir); write_files false writes nothing.
    """
    del ctx
    t0 = time.perf_counter()
    try:
        background = list(background_rgb or [255, 255, 255])
        pad = list(pad_rgb or [255, 255, 255])
        baseplate = list(baseplate_rgb or BASEPLATE_RGB)
        if fit not in ("crop", "pad", "stretch"):
            raise _Bad(f"unknown fit {fit!r} (crop, pad, stretch)")
        if resample not in ("area", "linear"):
            raise _Bad(f"unknown resample {resample!r} (area, linear)")
        if dither not in ("none", "floyd_steinberg"):
            raise _Bad(f"unknown dither {dither!r} (none, floyd_steinberg)")
        if order not in ("rows", "serpentine", "colour"):
            raise _Bad(f"unknown order {order!r} (rows, serpentine, colour)")
        if part_kind not in ("plate", "brick", "tile"):
            raise _Bad(f"unknown part_kind {part_kind!r} (plate, brick, tile)")
        if system not in ("lego", "duplo"):
            raise _Bad(f"unknown system {system!r} (lego, duplo)")
        nx, ny = _pair(plate_studs, "plate_studs")
        ox, oy = _pair(origin, "origin")
        if nx < 1 or ny < 1:
            raise _Bad(f"plate_studs must be positive, got {[nx, ny]}")
        if ox < 0 or oy < 0 or ox >= nx or oy >= ny:
            raise _Bad(f"origin {[ox, oy]} is off the {nx}x{ny} plate")
        w, h = _pair(mosaic_size, "mosaic_size") if mosaic_size is not None else (nx - ox, ny - oy)
        if w < 1 or h < 1:
            raise _Bad(f"mosaic_size must be positive, got {[w, h]}")
        if ox + w > nx or oy + h > ny:
            raise _Bad(f"a {w}x{h} mosaic at {[ox, oy]} runs off the {nx}x{ny} plate")
        if w * h > MAX_CELLS:
            raise _Bad(f"{w}x{h} = {w * h} cells is over the {MAX_CELLS} limit")
        size_list = [_size(s) for s in (sizes or ["1x1"])]
        size_list = sorted(set(size_list), key=lambda s: (s[0] * s[1], s))
        if (1, 1) not in size_list:
            raise _Bad("sizes must include 1x1 so that every cell can be covered")
        pal = _palette(palette)
        stock = _stock(inventory, pal, size_list)
        if int(preview_px) < 1:
            raise _Bad("preview_px must be at least 1")
    except _Bad as exc:
        return _empty(str(exc))
    try:
        img = _load_image(image if image is not None else image_path, background)
    except _BadImage as exc:
        return _empty(str(exc), route="invalid_image")
    if min(img.shape[:2]) < 1:
        return _empty("the image has no pixels", route="invalid_image")

    # Pixelate (brickmos: cv2.resize to the mosaic size).
    fitted = _fit(img, w, h, fit, pad)
    cell_rgb = _cells(fitted, w, h, resample)
    lab = srgb_to_lab(cell_rgb)
    pal_lab = np.stack([p["lab"] for p in pal])
    flat = lab.reshape(-1, 3)
    dist = deltaE_ciede2000(flat[:, None, :], pal_lab[None, :, :])  # N x K
    caps = np.array(
        [
            np.inf
            if any(v is None for v in s.values())
            else float(sum(a * b * v for (a, b), v in s.items()))
            for s in stock
        ]
    )

    # Colours (brickmos: closest_color per cell; here CIEDE2000, dithering and stock).
    if dither == "floyd_steinberg":
        grid, ideal, targets = _dither(lab, pal_lab, caps)
        ideal_flat = ideal.reshape(-1)
        target_flat = targets.reshape(-1, 3)
    else:
        ideal_flat = np.argmin(dist, axis=1)
        counts = np.bincount(ideal_flat, minlength=len(pal))
        if np.all(counts <= caps):
            assign = ideal_flat.copy()
        else:
            assign = _transport(dist, caps)
        grid = assign.reshape(h, w)
        target_flat = flat

    # Parts: merge runs of one colour, then fill single cells; a cell whose colour has no 1x1 left
    # goes to the nearest colour that has one.
    parts, leftover, left = _merge(grid, size_list, stock)
    grid = grid.copy()
    refilled: list[tuple[int, int]] = []
    for r, c in zip(*np.nonzero(leftover), strict=True):
        i = r * w + c
        choice = -1
        for k in np.argsort(dist[i]):
            n = left[k].get((1, 1))
            if n is None or n > 0:
                choice = int(k)
                if n is not None:
                    left[k][(1, 1)] = n - 1
                break
        grid[r, c] = choice
        if choice >= 0:
            parts.append(
                {"r": int(r), "c": int(c), "rows": 1, "cols": 1, "colour": choice, "size": (1, 1)}
            )
            refilled.append((int(r), int(c)))

    final = grid.reshape(-1)
    filled = final >= 0
    n_cells = w * h
    n_empty = int((~filled).sum())

    # Errors: CIEDE2000 of each covered cell to the image's cell colour, and after a 1-stud blur.
    de = np.zeros(n_cells)
    de[filled] = deltaE_ciede2000(flat[filled], pal_lab[final[filled]])
    ideal_de = dist[np.arange(n_cells), np.argmin(dist, axis=1)]
    mosaic_lab = np.repeat(
        srgb_to_lab(np.asarray(baseplate, dtype=np.float64) / 255.0)[None, :], n_cells, axis=0
    )
    mosaic_lab[filled] = pal_lab[final[filled]]
    blur_m = cv2.GaussianBlur(mosaic_lab.reshape(h, w, 3).astype(np.float32), (0, 0), BLUR_SIGMA)
    blur_t = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), BLUR_SIGMA)
    de_blur = deltaE_ciede2000(
        blur_t.reshape(-1, 3).astype(float), blur_m.reshape(-1, 3).astype(float)
    )

    # Substitutions: covered cells whose colour is not the palette's nearest (to the diffused target
    # when dithering), grouped by (wanted, used).
    subs: dict[tuple[int, int], list[float]] = {}
    for i in np.flatnonzero(filled & (final != ideal_flat)):
        t = target_flat[i]
        extra = float(
            deltaE_ciede2000(t, pal_lab[final[i]]) - deltaE_ciede2000(t, pal_lab[ideal_flat[i]])
        )
        subs.setdefault((int(ideal_flat[i]), int(final[i])), []).append(extra)
    substitutions = [
        {
            "wanted": pal[a]["name"],
            "used": pal[b]["name"],
            "cells": len(v),
            "mean_extra_delta_e": round(float(np.mean(v)), 3),
        }
        for (a, b), v in sorted(subs.items(), key=lambda kv: -len(kv[1]))
    ]
    n_sub = int(sum(s["cells"] for s in substitutions))

    # Placements in plate cells: image row r is plate y = oy + h - 1 - r.
    for p in parts:
        sx, sy = p["cols"], p["rows"]
        a, b = p["size"]
        p["x"] = ox + p["c"]
        p["y"] = oy + h - p["r"] - sy
        p["rotation"] = 0 if (sx, sy) == (a, b) else 90
    use_count = np.bincount([p["colour"] for p in parts], minlength=len(pal))
    if order == "rows":
        parts.sort(key=lambda p: (p["y"], p["x"]))
    elif order == "serpentine":
        parts.sort(key=lambda p: (p["y"], p["x"] if (p["y"] - oy) % 2 == 0 else -p["x"]))
    else:
        parts.sort(key=lambda p: (-use_count[p["colour"]], p["colour"], p["y"], p["x"]))
    ids = PARTS.get(system, {}).get(part_kind, {})
    placements: list[dict[str, Any]] = []
    for seq, p in enumerate(parts):
        col = pal[p["colour"]]
        label = _label(p["size"])
        placements.append(
            {
                "seq": seq,
                "id": seq,
                "step": seq + 1,
                "cell": [p["x"], p["y"], FIRST_LAYER],
                "cells": [
                    [p["x"] + i, p["y"] + j, FIRST_LAYER]
                    for i in range(p["cols"])
                    for j in range(p["rows"])
                ],
                "x": p["x"],
                "y": p["y"],
                "layer": FIRST_LAYER,
                "pose_cell": {
                    "x": p["x"],
                    "y": p["y"],
                    "layer": FIRST_LAYER,
                    "rotation": p["rotation"],
                },
                "size": label,
                "brick_size": label,
                "rotation": p["rotation"],
                "span_x": p["cols"],
                "span_y": p["rows"],
                "kind": part_kind,
                "system": system,
                "part": ids.get(label),
                "colour": col["name"],
                "ldraw_colour": col["ldraw"],
                "bricklink_colour_id": col["bricklink_id"],
            }
        )

    # Bill of materials (brickmos: colour stats -> BrickLink XML).
    bom_map: dict[tuple[int, tuple[int, int]], int] = {}
    for p in parts:
        bom_map[(p["colour"], p["size"])] = bom_map.get((p["colour"], p["size"]), 0) + 1
    bom = []
    for (k, sz), n in sorted(bom_map.items(), key=lambda kv: (-kv[1], kv[0])):
        stock_n = stock[k].get(sz)
        bom.append(
            {
                "colour": pal[k]["name"],
                "size": _label(sz),
                "kind": part_kind,
                "part": ids.get(_label(sz)),
                "count": n,
                "studs": n * sz[0] * sz[1],
                "in_stock": stock_n,
                "ldraw_colour": pal[k]["ldraw"],
                "bricklink_colour_id": pal[k]["bricklink_id"],
                "rgb": [round(v) for v in pal[k]["rgb"]],
            }
        )
    xml = _bricklink_xml(bom, int(spares))

    colour_grid = [[pal[k]["name"] if k >= 0 else None for k in row] for row in grid.tolist()]
    empty_cells = [
        [ox + int(c), oy + h - 1 - int(r), FIRST_LAYER]
        for r, c in zip(*np.nonzero(grid < 0), strict=True)
    ]
    # planning-brick-assembly-order reads cell as {x, y, layer} and steps as [{step, bricks}].
    order_steps = [
        {
            "step": pl["step"],
            "bricks": [
                {
                    **{k: v for k, v in pl.items() if k not in ("cell", "pose_cell")},
                    "cell": {"x": pl["x"], "y": pl["y"], "layer": pl["layer"]},
                }
            ],
        }
        for pl in placements
    ]

    if n_empty:
        route = "inventory_short"
        detail = (
            f"{n_empty} of {n_cells} cells left empty: the inventory covers "
            f"{n_cells - n_empty} studs"
        )
    elif n_sub:
        route = "planned_with_substitutions"
        top = ", ".join(f"{s['cells']} {s['wanted']}->{s['used']}" for s in substitutions[:3])
        detail = f"{n_sub} of {n_cells} cells use a substitute colour for lack of parts ({top})"
    else:
        route = "planned"
        detail = f"{len(placements)} parts cover all {n_cells} cells"
    detail += (
        f"; mean CIEDE2000 {float(de[filled].mean()) if filled.any() else 0.0:.2f}"
        f" (palette floor {float(ideal_de.mean()):.2f}), {len(placements)} placements"
    )

    paths = {"preview_path": "", "pixelated_path": "", "out_dir": ""}
    if write_files:
        d = out_dir or tempfile.mkdtemp(prefix="brick_mosaic_")
        os.makedirs(d, exist_ok=True)
        prev = _render(grid, parts, pal, int(preview_px), grid_cell, preview_studs, baseplate)
        pix = cv2.resize(
            (cell_rgb * 255).round().astype(np.uint8),
            (w * int(preview_px), h * int(preview_px)),
            interpolation=cv2.INTER_NEAREST,
        )
        paths = {
            "preview_path": os.path.join(d, "mosaic.png"),
            "pixelated_path": os.path.join(d, "pixelated.png"),
            "out_dir": d,
        }
        cv2.imwrite(paths["preview_path"], cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths["pixelated_path"], cv2.cvtColor(pix, cv2.COLOR_RGB2BGR))
        with open(os.path.join(d, "bricklink.xml"), "w", encoding="utf-8") as f:
            f.write(xml)
        with open(os.path.join(d, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "placements": placements,
                    "bill_of_materials": bom,
                    "substitutions": substitutions,
                },
                f,
            )

    fde = de[filled] if filled.any() else np.zeros(1)
    return {
        "route": route,
        "detail": detail,
        "mosaic_size": [w, h],
        "origin": [ox, oy],
        "plate_studs": [nx, ny],
        "n_cells": n_cells,
        "n_placements": len(placements),
        "n_empty": n_empty,
        "n_substituted": n_sub,
        "n_colours": len({p["colour"] for p in parts}),
        "mean_delta_e": round(float(fde.mean()), 4),
        "median_delta_e": round(float(np.median(fde)), 4),
        "p95_delta_e": round(float(np.percentile(fde, 95)), 4),
        "max_delta_e": round(float(fde.max()), 4),
        "mean_delta_e_blurred": round(float(de_blur.mean()), 4),
        "ideal_mean_delta_e": round(float(ideal_de.mean()), 4),
        "placements": placements,
        "bill_of_materials": bom,
        "substitutions": substitutions,
        "empty_cells": empty_cells,
        "colour_grid": colour_grid,
        "placements_json": json.dumps(placements),
        "steps_json": json.dumps(order_steps),
        "bom_json": json.dumps(bom),
        "bricklink_xml": xml,
        **paths,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }
