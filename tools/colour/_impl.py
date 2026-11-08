# SPDX-License-Identifier: Apache-2.0
"""Implementation of the colour bundle: robust mask colour, white balance,
palette classification by CIEDE2000, dominant colours.

``tools.py`` is the typed ``@tool`` boundary over this module and imports it
lazily. Only numpy is needed; the CIEDE2000 formula is vendored from
scikit-image in ``_ciede2000.py``.

Images are RGB (not BGR) ``[H, W, 3]``, uint8 0..255 or float 0..1. All
arithmetic on light (gains, averaging) is done in linear sRGB; distances are
CIEDE2000 in CIELAB (D65, 2 degree observer).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gap_skills.tools.colour._ciede2000 import deltaE_ciede2000

__all__ = [
    "AMBIGUOUS_MARGIN",
    "PALETTES",
    "UNKNOWN_DELTA_E",
    "classify",
    "dominant_colours",
    "lab_to_srgb",
    "measure",
    "resolve_palette",
    "srgb_to_lab",
    "white_balance",
]

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

#: LEGO / DUPLO colours: (name, LDraw code, sRGB hex) copied from the LDraw.org
#: colour table LDConfig.ldr (!LDRAW_ORG Configuration UPDATE 2026-05-29,
#: https://library.ldraw.org/library/official/LDConfig.ldr), CCAL 2.0 (CC BY
#: 2.0), LDraw.org. Solid colours in current production only: the retired
#: Light_Grey (7) and Dark_Grey (8) are left out because they sit within a few
#: CIEDE2000 of the bluish greys that replaced them in 2004 and would make
#: every grey ambiguous. LDraw's values are rendering colours, not
#: measurements of plastic under a camera: for a station, learn the palette
#: from a few labelled bricks with ``measure`` and pass it as a custom list.
LEGO = [
    ("black", 0, "1B2A34"),
    ("blue", 1, "1E5AA8"),
    ("green", 2, "00852B"),
    ("dark_pink", 5, "D3359D"),
    ("bright_green", 10, "58AB41"),
    ("yellow", 14, "FAC80A"),
    ("white", 15, "F4F4F4"),
    ("tan", 19, "D7BA8C"),
    ("orange", 25, "D67923"),
    ("magenta", 26, "901F76"),
    ("lime", 27, "A5CA18"),
    ("dark_tan", 28, "897D62"),
    ("reddish_brown", 70, "5F3109"),
    ("light_bluish_grey", 71, "969696"),
    ("dark_bluish_grey", 72, "646464"),
    ("medium_blue", 73, "7396C8"),
    ("bright_light_orange", 191, "FCAC00"),
    ("bright_light_yellow", 226, "FFEC6C"),
    ("dark_green", 288, "00451A"),
    ("dark_red", 320, "720012"),
    ("dark_azure", 321, "469BC3"),
    ("medium_azure", 322, "68C3E2"),
    ("sand_green", 378, "708E7C"),
    ("dark_orange", 484, "91501C"),
    ("red", 4, "B40000"),
]

#: The commonest LEGO/DUPLO colours (the basic-brick and DUPLO sets), a subset
#: of ``LEGO``. Fewer entries means fewer near neighbours, so larger margins.
LEGO_BASIC_NAMES = (
    "red",
    "blue",
    "yellow",
    "green",
    "black",
    "white",
    "light_bluish_grey",
    "dark_bluish_grey",
    "tan",
    "orange",
    "lime",
    "dark_green",
    "reddish_brown",
    "medium_azure",
)

#: Domino tiles and pips. White and black are LDraw's (above); ivory is the
#: sRGB of "ivory" bone/urea tiles as a nominal value (#EFE6CF, not measured --
#: measure your set); pip colours are the LDraw red/blue/green/yellow.
DOMINO = [
    ("white", 15, "F4F4F4"),
    ("ivory", -1, "EFE6CF"),
    ("black", 0, "1B2A34"),
    ("red", 4, "B40000"),
    ("blue", 1, "1E5AA8"),
    ("green", 2, "00852B"),
    ("yellow", 14, "FAC80A"),
]


def _hex_rgb(h: str) -> list[float]:
    return [float(int(h[i : i + 2], 16)) for i in (0, 2, 4)]


PALETTES: dict[str, list[dict[str, Any]]] = {
    "lego": [{"name": n, "ldraw": c, "rgb": _hex_rgb(h)} for n, c, h in LEGO],
    "lego_basic": [
        {"name": n, "ldraw": c, "rgb": _hex_rgb(h)} for n, c, h in LEGO if n in LEGO_BASIC_NAMES
    ],
    "domino": [{"name": n, "ldraw": c, "rgb": _hex_rgb(h)} for n, c, h in DOMINO],
}

# ---------------------------------------------------------------------------
# Constants (each with where it came from)
# ---------------------------------------------------------------------------

#: CIEDE2000 gap between the best and second-best palette entry below which
#: the answer is "ambiguous". On the synthetic LDraw scenes without white
#: balance (SKILL.md) a margin under 3 caught 44% of the wrong
#: answers on the 25-colour palette (62% on lego_basic) while flagging 15% (8%)
#: of the right ones; 2 caught too few (26%), 5 flagged too many right ones
#: (26%). It catches near ties only: an uncorrected illuminant shifts a colour
#: wholesale onto a neighbour with a large margin (station frames: black read
#: for the dark grey box at margin 18.8) -- that needs white_ref.
AMBIGUOUS_MARGIN = 3.0

#: CIEDE2000 to the nearest entry above which the colour is "unknown_colour":
#: nothing in the palette. Correct answers on the station frames were within
#: 0.2-17 of their entry (LDraw palette, any white balance), and LDraw's
#: nearest solid colours are 10-15 apart, so 25 is well outside both.
UNKNOWN_DELTA_E = 25.0

#: Lightness percentiles kept by the robust statistic. Above the upper one are
#: specular highlights (desaturated, bright), below the lower one cast and
#: self shadows. On the harsh synthetic patches (20% specular, 25% shadow
#: pixels, grey-reference balanced) the median's CIEDE2000 to the true colour
#: was 0.92 untrimmed, 0.79 at 5/95, 0.64 at 15/85 and 0.64 at 25/75; the
#: median itself does most of the work (a plain mean classified 50% there).
L_TRIM = (15.0, 85.0)

#: A raw channel at or above this reads as clipped (sensor saturation): such a
#: pixel says nothing about hue. Dropped unless that would drop more than half
#: the mask (then the object is simply bright, e.g. a white table).
CLIP_LEVEL = 250

#: Mask pixels peeled off the boundary before measuring: boundary pixels mix
#: the object with the background (a 2 cm brick at ~0.9 m is ~10-20 px wide,
#: so 1-2 px of mixing is 10-20% of it). Not applied if it would leave fewer
#: than a quarter of the pixels (thin cables).
ERODE_PX = 2

#: Fewer usable pixels than this and the answer is "empty".
MIN_PIXELS = 12

#: Reference colour a white-balance reference region is mapped to when the
#: caller gives none: LDraw White (#F4F4F4), i.e. white plastic or paper.
WHITE_RGB = (244.0, 244.0, 244.0)

#: Dominant colours: clusters whose centres are closer than this CIEDE2000 are
#: one colour (shading of one surface spans ~5-10 on the station frames), and
#: clusters under this share of the pixels are dropped (mask leaks, glints).
MERGE_DELTA_E = 10.0
MIN_FRACTION = 0.05

#: k-means: at most this many pixels are clustered (the rest are assigned to
#: the nearest centre); more do not move a centre by a visible amount.
KMEANS_SAMPLE = 20000
KMEANS_ITERS = 25

# ---------------------------------------------------------------------------
# Colour space (sRGB IEC 61966-2-1, D65; matrices as in scikit-image's colorconv)
# ---------------------------------------------------------------------------

_XYZ_FROM_RGB = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_D65 = np.array([0.95047, 1.0, 1.08883])
_LUMA = _XYZ_FROM_RGB[1]


def _to_unit(rgb: Any) -> np.ndarray:
    """RGB to float 0..1 (uint8 / >1 values are taken as 0..255)."""
    a = np.asarray(rgb)
    if a.dtype == np.uint8 or (a.size and np.nanmax(a) > 1.0 + 1e-6):
        return a.astype(np.float64) / 255.0
    return a.astype(np.float64)


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
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    d = 6.0 / 29.0
    xyz = np.where(f > d, f**3, 3 * d * d * (f - 4.0 / 29.0)) * _D65
    return xyz @ _RGB_FROM_XYZ.T


def srgb_to_lab(rgb: Any) -> np.ndarray:
    """sRGB (uint8 0..255 or float 0..1, last axis RGB) to CIELAB D65."""
    return linear_to_lab(srgb_to_linear(_to_unit(rgb)))


def lab_to_srgb(lab: Any) -> np.ndarray:
    """CIELAB D65 to sRGB float 0..255 (clipped)."""
    return np.clip(linear_to_srgb(lab_to_linear(lab)) * 255.0, 0.0, 255.0)


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


def resolve_palette(palette: Any) -> tuple[list[str], np.ndarray]:
    """Names and Lab of a palette: a built-in name or a list of ``{name, rgb}``
    (0..255), ``{name, lab}`` or ``{name, hex}`` entries. Repeated names are
    allowed (several measured samples of one colour)."""
    if palette is None:
        palette = "lego"
    if isinstance(palette, str):
        if palette not in PALETTES:
            raise ValueError(f"unknown palette {palette!r}; built-in: {sorted(PALETTES)}")
        palette = PALETTES[palette]
    names: list[str] = []
    labs: list[np.ndarray] = []
    for entry in palette:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"palette entries are {{name, rgb|lab|hex}} dicts, got {entry!r}")
        if entry.get("lab") is not None:
            lab = np.asarray(entry["lab"], dtype=np.float64)
        elif entry.get("rgb") is not None:
            rgb = np.asarray(entry["rgb"], dtype=np.float64)
            lab = srgb_to_lab(rgb / 255.0 if rgb.max() > 1.0 else rgb)
        elif entry.get("hex") is not None:
            lab = srgb_to_lab(np.asarray(_hex_rgb(str(entry["hex"]).lstrip("#"))) / 255.0)
        else:
            raise ValueError(f"palette entry {entry['name']!r} has no rgb, lab or hex")
        names.append(str(entry["name"]))
        labs.append(lab.reshape(3))
    if not names:
        raise ValueError("empty palette")
    return names, np.stack(labs)


# ---------------------------------------------------------------------------
# Masks and pixel selection
# ---------------------------------------------------------------------------


def _as_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} does not match image {shape}")
    return m > 0


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """4-neighbour binary erosion, ``r`` times (numpy only)."""
    m = mask.copy()
    for _ in range(max(0, int(r))):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def _select(rgb_unit: np.ndarray, mask: np.ndarray, erode_px: int) -> tuple[np.ndarray, int]:
    """The mask's pixels (unit sRGB, [N, 3]) after boundary erosion and dropping
    clipped pixels, and the mask's pixel count before either."""
    n_mask = int(mask.sum())
    if erode_px > 0:
        eroded = _erode(mask, erode_px)
        if eroded.sum() >= max(MIN_PIXELS, 0.25 * n_mask):
            mask = eroded
    px = rgb_unit[mask]
    clipped = (px >= CLIP_LEVEL / 255.0).any(axis=1)
    if clipped.any() and (~clipped).sum() >= max(MIN_PIXELS, 0.5 * len(px)):
        px = px[~clipped]
    return px, n_mask


def _trim_l(lab: np.ndarray, l_trim: tuple[float, float]) -> np.ndarray:
    if len(lab) < 2 * MIN_PIXELS:
        return lab
    lo, hi = np.percentile(lab[:, 0], l_trim)
    keep = (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    return lab[keep] if keep.sum() >= MIN_PIXELS else lab


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------


def white_balance(
    rgb: Any,
    mask: Any = None,
    method: str = "grey_world",
    reference_rgb: Any = None,
    exposure: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    """Per-channel gains in linear sRGB that neutralise the illuminant.

    ``grey_world``: the mean of ``mask`` (default the whole frame; clipped and
    near-black pixels left out) is assumed grey; the gains equalise its
    channels and keep its luminance (chromatic only -- grey world cannot know
    the exposure). ``reference``: ``mask`` covers a region of known colour
    ``reference_rgb`` (default ``WHITE_RGB``); its robust colour is mapped onto
    that, which fixes the cast and, with ``exposure`` (default), the
    brightness too -- what separates black from dark grey.
    """
    unit = _to_unit(rgb)
    if unit.ndim != 3 or unit.shape[-1] < 3:
        raise ValueError(f"rgb must be [H, W, 3], got {unit.shape}")
    unit = unit[..., :3]
    m = _as_mask(mask, unit.shape[:2])
    if method == "grey_world":
        px = unit[m]
        lin = srgb_to_linear(px)
        ok = ~(px >= CLIP_LEVEL / 255.0).any(axis=1) & (lin.sum(axis=1) > 3 * 0.002)
        lin = lin[ok] if ok.sum() >= MIN_PIXELS else lin
        if len(lin) < MIN_PIXELS:
            raise ValueError("white_balance: too few pixels for grey world")
        measured = lin.mean(axis=0)
        target = np.full(3, float(measured @ _LUMA))
        clipped = float((px >= CLIP_LEVEL / 255.0).any(axis=1).mean()) if len(px) else 0.0
    elif method == "reference":
        if mask is None:
            raise ValueError("white_balance(method='reference') needs the reference region's mask")
        clipped = float((unit[m] >= CLIP_LEVEL / 255.0).any(axis=1).mean()) if m.any() else 0.0
        px, _ = _select(unit, m, ERODE_PX)
        if len(px) < MIN_PIXELS:
            raise ValueError("white_balance: reference region has too few usable pixels")
        lab = _trim_l(srgb_to_lab(px), L_TRIM)
        measured = lab_to_linear(np.median(lab, axis=0))
        ref = np.asarray(WHITE_RGB if reference_rgb is None else reference_rgb, dtype=np.float64)
        target = srgb_to_linear(_to_unit(ref.reshape(3)))
    else:
        raise ValueError(f"unknown white-balance method {method!r}")
    measured = np.clip(measured, 1e-6, None)
    gains = target / measured
    if not exposure:
        gains = gains / float((measured * gains) @ _LUMA / (measured @ _LUMA))
    out: dict[str, Any] = {
        "gains": [float(g) for g in gains],
        "method": method,
        "clipped_fraction": clipped,
        "measured_rgb": [float(v) for v in linear_to_srgb(measured) * 255.0],
        "n_pixels": int(m.sum()),
    }
    if apply:
        bal = linear_to_srgb(srgb_to_linear(unit) * gains)
        out["image"] = np.clip(np.round(bal * 255.0), 0, 255).astype(np.uint8)
    return out


def _resolve_gains(unit: np.ndarray, white_ref: Any, reference_rgb: Any) -> np.ndarray | None:
    """``white_ref`` -> linear gains, or None for no correction."""
    if white_ref is None or white_ref is False:
        return None
    if isinstance(white_ref, str):
        if white_ref != "grey_world":
            raise ValueError(f"white_ref string must be 'grey_world', got {white_ref!r}")
        return np.asarray(white_balance(unit, None, "grey_world")["gains"])
    if isinstance(white_ref, dict):
        if "gains" not in white_ref:
            raise ValueError("a white_ref dict must carry 'gains' (a white_balance result)")
        return np.asarray(white_ref["gains"], dtype=np.float64).reshape(3)
    arr = np.asarray(white_ref)
    if arr.shape == (3,):
        return arr.astype(np.float64)
    return np.asarray(white_balance(unit, arr, "reference", reference_rgb=reference_rgb)["gains"])


# ---------------------------------------------------------------------------
# Measure / classify
# ---------------------------------------------------------------------------


def _mask_lab(
    rgb: Any,
    mask: Any,
    white_ref: Any,
    reference_rgb: Any,
    erode_px: int,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    unit = _to_unit(rgb)
    if unit.ndim != 3 or unit.shape[-1] < 3:
        raise ValueError(f"rgb must be [H, W, 3], got {unit.shape}")
    unit = unit[..., :3]
    m = _as_mask(mask, unit.shape[:2])
    gains = _resolve_gains(unit, white_ref, reference_rgb)
    px, n_mask = _select(unit, m, erode_px)
    lin = srgb_to_linear(px)
    if gains is not None:
        lin = lin * gains
    return linear_to_lab(lin), n_mask, gains


def measure(
    rgb: Any,
    mask: Any = None,
    white_ref: Any = None,
    reference_rgb: Any = None,
    erode_px: int = ERODE_PX,
    l_trim: tuple[float, float] = L_TRIM,
) -> dict[str, Any]:
    """Robust colour of a mask: boundary eroded, clipped pixels dropped, the
    lightness tails (highlights, shadows) trimmed, then the per-channel median
    in Lab."""
    lab, n_mask, gains = _mask_lab(rgb, mask, white_ref, reference_rgb, erode_px)
    out: dict[str, Any] = {
        "found": False,
        "lab": None,
        "rgb": None,
        "lab_iqr": None,
        "n_pixels": n_mask,
        "n_used": 0,
        "gains": None if gains is None else [float(g) for g in gains],
    }
    if len(lab) < MIN_PIXELS:
        return out
    kept = _trim_l(lab, tuple(l_trim))
    med = np.median(kept, axis=0)
    q1, q3 = np.percentile(kept, [25, 75], axis=0)
    out.update(
        found=True,
        lab=[float(v) for v in med],
        rgb=[float(v) for v in lab_to_srgb(med)],
        lab_iqr=[float(v) for v in q3 - q1],
        n_used=int(len(kept)),
    )
    return out


def _rank(
    lab: np.ndarray, names: list[str], labs: np.ndarray, kL: float
) -> list[tuple[str, float]]:
    """Palette names by CIEDE2000, one entry per name (its nearest sample)."""
    d = deltaE_ciede2000(np.asarray(lab).reshape(1, 3), labs, kL=kL)
    best: dict[str, float] = {}
    for n, v in zip(names, d, strict=True):
        if n not in best or v < best[n]:
            best[n] = float(v)
    return sorted(best.items(), key=lambda kv: kv[1])


def classify(
    rgb: Any,
    mask: Any = None,
    palette: Any = "lego",
    white_ref: Any = None,
    reference_rgb: Any = None,
    ambiguous_margin: float = AMBIGUOUS_MARGIN,
    unknown_delta_e: float = UNKNOWN_DELTA_E,
    kL: float = 1.0,
    erode_px: int = ERODE_PX,
    l_trim: tuple[float, float] = L_TRIM,
    top_k: int = 5,
) -> dict[str, Any]:
    """Nearest palette colour of a mask by CIEDE2000, with the margin to the
    runner-up and a status: classified / ambiguous / unknown_colour / empty."""
    names, labs = resolve_palette(palette)
    m = measure(rgb, mask, white_ref, reference_rgb, erode_px, l_trim)
    out: dict[str, Any] = {
        "status": "empty",
        "name": None,
        "delta_e": None,
        "second_name": None,
        "second_delta_e": None,
        "margin_to_second": None,
        "lab": m["lab"],
        "rgb": m["rgb"],
        "n_pixels": m["n_pixels"],
        "n_used": m["n_used"],
        "white_balanced": m["gains"] is not None,
        "gains": m["gains"],
        "ranking": [],
    }
    if not m["found"]:
        return out
    ranked = _rank(np.asarray(m["lab"]), names, labs, kL)
    name, de = ranked[0]
    second_name, second_de = ranked[1] if len(ranked) > 1 else (None, float("inf"))
    margin = second_de - de
    if de > unknown_delta_e:
        status = "unknown_colour"
    elif margin < ambiguous_margin:
        status = "ambiguous"
    else:
        status = "classified"
    out.update(
        status=status,
        name=name,
        delta_e=de,
        second_name=second_name,
        second_delta_e=None if second_name is None else second_de,
        margin_to_second=None if second_name is None else margin,
        ranking=[{"name": n, "delta_e": d} for n, d in ranked[: max(1, int(top_k))]],
    )
    return out


# ---------------------------------------------------------------------------
# Dominant colours
# ---------------------------------------------------------------------------


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ seeding (Euclidean in Lab)."""
    k = min(k, len(x))
    centres = [x[rng.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        if d2.sum() <= 0:
            break
        centres.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(axis=1))
    c = np.stack(centres)
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(KMEANS_ITERS):
        labels = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.stack(
            [x[labels == j].mean(axis=0) if (labels == j).any() else c[j] for j in range(len(c))]
        )
        if np.allclose(new, c, atol=1e-3):
            c = new
            break
        c = new
    return c, labels


def dominant_colours(
    rgb: Any,
    mask: Any = None,
    k: int | None = None,
    max_k: int = 6,
    palette: Any = None,
    white_ref: Any = None,
    reference_rgb: Any = None,
    merge_delta_e: float = MERGE_DELTA_E,
    min_fraction: float = MIN_FRACTION,
    return_labels: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster a mask's pixels into its dominant colours (k-means in Lab).

    With ``k`` given, exactly ``k`` clusters (minus empty ones). Without, start
    from ``max_k`` and merge clusters closer than ``merge_delta_e`` CIEDE2000,
    so the count adapts. Clusters under ``min_fraction`` are dropped from the
    list. With ``palette`` each colour is named (as ``classify`` would).
    """
    unit = _to_unit(rgb)
    if unit.ndim != 3 or unit.shape[-1] < 3:
        raise ValueError(f"rgb must be [H, W, 3], got {unit.shape}")
    unit = unit[..., :3]
    m = _as_mask(mask, unit.shape[:2])
    gains = _resolve_gains(unit, white_ref, reference_rgb)
    vv, uu = np.nonzero(m)
    px = unit[vv, uu]
    keep = ~(px >= CLIP_LEVEL / 255.0).any(axis=1)
    if keep.sum() < max(MIN_PIXELS, 0.5 * len(px)):
        keep[:] = True
    vv, uu, px = vv[keep], uu[keep], px[keep]
    out: dict[str, Any] = {
        "found": False,
        "colours": [],
        "n_pixels": int(m.sum()),
        "n_used": int(len(px)),
        "gains": None if gains is None else [float(g) for g in gains],
    }
    if len(px) < MIN_PIXELS:
        return out
    lin = srgb_to_linear(px)
    if gains is not None:
        lin = lin * gains
    lab = linear_to_lab(lin)
    rng = np.random.default_rng(seed)
    sample = (
        lab
        if len(lab) <= KMEANS_SAMPLE
        else lab[rng.choice(len(lab), KMEANS_SAMPLE, replace=False)]
    )
    centres, _ = _kmeans(sample, int(k) if k else int(max_k), rng)
    labels = ((lab[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
    groups = [np.flatnonzero(labels == j) for j in range(len(centres))]
    groups = [g for g in groups if len(g)]
    if not k:
        while len(groups) > 1:
            cs = np.stack([np.median(lab[g], axis=0) for g in groups])
            d = deltaE_ciede2000(cs[:, None, :], cs[None, :, :])
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= merge_delta_e:
                break
            groups[i] = np.concatenate([groups[i], groups[j]])
            groups.pop(j)
    groups.sort(key=len, reverse=True)
    names_labs = resolve_palette(palette) if palette is not None else None
    label_img = np.full(m.shape, -1, dtype=np.int16) if return_labels else None
    colours = []
    for g in groups:
        frac = len(g) / len(lab)
        if frac < min_fraction:
            continue
        kept = _trim_l(lab[g], L_TRIM)
        c = np.median(kept, axis=0)
        entry: dict[str, Any] = {
            "lab": [float(v) for v in c],
            "rgb": [float(v) for v in lab_to_srgb(c)],
            "fraction": float(frac),
            "n_pixels": int(len(g)),
        }
        if names_labs is not None:
            ranked = _rank(c, *names_labs, kL=1.0)
            entry["name"] = ranked[0][0]
            entry["delta_e"] = ranked[0][1]
            entry["margin_to_second"] = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else None
        if label_img is not None:
            label_img[vv[g], uu[g]] = len(colours)
        colours.append(entry)
    out.update(found=bool(colours), colours=colours)
    if label_img is not None:
        out["labels"] = label_img
    return out
