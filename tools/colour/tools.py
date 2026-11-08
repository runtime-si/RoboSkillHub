# SPDX-License-Identifier: Apache-2.0
"""colour tool bundle -- what colour a segmented part is, by palette.

Four ``@tool`` functions over ``_impl.py``: ``colour.classify`` names a mask's
colour from a palette (LEGO, DUPLO-basic, domino, or the caller's own) by
CIEDE2000 with the margin to the runner-up, ``colour.measure`` returns the
mask's robust colour (to build a per-camera palette from labelled parts),
``colour.white_balance`` estimates per-channel gains (grey world or a known
white/grey region), and ``colour.dominant_colours`` clusters a mask into its
main colours.

No model, no GPU -- numpy only. ``_impl`` is imported inside the functions,
so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import Mask

#: Mirrored from ``_impl.AMBIGUOUS_MARGIN`` / ``_impl.UNKNOWN_DELTA_E`` /
#: ``_impl.ERODE_PX`` (``_impl`` is imported lazily); the tests assert they agree.
_AMBIGUOUS_MARGIN = 3.0
_UNKNOWN_DELTA_E = 25.0
_ERODE_PX = 2


class RankedColour(TypedDict):
    name: str
    delta_e: float


class ColourClass(TypedDict):
    status: str  # classified | ambiguous | unknown_colour | empty
    name: str | None  # nearest palette entry (also when ambiguous / unknown)
    delta_e: float | None  # CIEDE2000 to it
    second_name: str | None
    second_delta_e: float | None
    margin_to_second: float | None  # second_delta_e - delta_e
    lab: list[float] | None  # the mask's robust CIELAB (after white balance)
    rgb: list[float] | None  # the same as sRGB 0..255
    n_pixels: int  # mask pixels
    n_used: int  # pixels left after erosion, clipping and the lightness trim
    white_balanced: bool
    gains: list[float] | None  # linear-sRGB gains applied, if any
    ranking: list[RankedColour]  # top_k palette names, nearest first


class ColourMeasure(TypedDict):
    found: bool
    lab: list[float] | None
    rgb: list[float] | None
    lab_iqr: list[float] | None  # spread of the kept pixels, per Lab channel
    n_pixels: int
    n_used: int
    gains: list[float] | None


class WhiteBalance(TypedDict):
    gains: list[float]  # per-channel, linear sRGB; pass the dict as white_ref
    method: str
    clipped_fraction: float  # reference pixels at sensor saturation; >0.5 = gains unreliable
    measured_rgb: list[float]  # the reference's (or the scene mean's) sRGB
    n_pixels: int
    image: np.ndarray  # NotRequired: uint8 [H, W, 3], only with apply=True


class DominantColour(TypedDict):
    lab: list[float]
    rgb: list[float]
    fraction: float  # share of the mask's usable pixels
    n_pixels: int
    name: str  # NotRequired: only with a palette
    delta_e: float  # NotRequired
    margin_to_second: float | None  # NotRequired


class DominantColours(TypedDict):
    found: bool
    colours: list[DominantColour]  # largest first
    n_pixels: int
    n_used: int
    gains: list[float] | None
    labels: np.ndarray  # NotRequired: int16 [H, W], index into colours, -1 elsewhere


@tool(
    name="colour.classify",
    summary=(
        "Name a segmented part's colour from a palette (LEGO, DUPLO, domino or custom) by "
        "CIEDE2000, with the margin to the runner-up and an ambiguous status."
    ),
    tags=("perception",),
)
def classify(
    rgb: np.ndarray,
    mask: Mask | None = None,
    palette: str | list[dict[str, Any]] = "lego",
    white_ref: Any = None,
    reference_rgb: list[float] | None = None,
    ambiguous_margin: float = _AMBIGUOUS_MARGIN,
    unknown_delta_e: float = _UNKNOWN_DELTA_E,
    kL: float = 1.0,
    erode_px: int = _ERODE_PX,
    top_k: int = 5,
) -> ColourClass:
    """Classify the colour inside ``mask`` (nonzero = part, [H, W]; None = the
    whole image) of ``rgb`` (RGB, uint8 or float 0..1, [H, W, 3]).

    The mask is eroded by ``erode_px``, clipped pixels are dropped, the
    lightness tails (highlights, shadows) are trimmed and the Lab median is
    compared with every ``palette`` entry by CIEDE2000. ``palette`` is
    ``"lego"`` (25 current LDraw solid colours), ``"lego_basic"`` (14),
    ``"domino"``, or a list of ``{name, rgb}`` / ``{name, lab}`` / ``{name,
    hex}`` (names may repeat: several samples of one colour).

    ``white_ref`` corrects the illuminant first: ``"grey_world"``, a mask of
    a white region (or of colour ``reference_rgb``), a ``colour.white_balance``
    result, or three gains. Status ``ambiguous`` when the runner-up is within
    ``ambiguous_margin``, ``unknown_colour`` when the nearest entry is further
    than ``unknown_delta_e``, ``empty`` when too few pixels remain.
    """
    if rgb is None:
        raise ValueError("colour.classify needs rgb")
    from gap_skills.tools.colour import _impl

    return _impl.classify(
        np.asarray(rgb),
        mask=None if mask is None else np.asarray(mask),
        palette=palette,
        white_ref=white_ref,
        reference_rgb=reference_rgb,
        ambiguous_margin=float(ambiguous_margin),
        unknown_delta_e=float(unknown_delta_e),
        kL=float(kL),
        erode_px=int(erode_px),
        top_k=int(top_k),
    )


@tool(
    name="colour.measure",
    summary=(
        "Robust colour of a mask (Lab and sRGB, highlights and shadows trimmed), to build "
        "a per-camera palette from labelled parts."
    ),
    tags=("perception",),
)
def measure(
    rgb: np.ndarray,
    mask: Mask | None = None,
    white_ref: Any = None,
    reference_rgb: list[float] | None = None,
    erode_px: int = _ERODE_PX,
) -> ColourMeasure:
    """The robust colour ``colour.classify`` compares with the palette: Lab
    median of the eroded, unclipped, lightness-trimmed mask pixels, after the
    optional ``white_ref`` correction. Feed ``{name, lab}`` entries built from
    it back to ``colour.classify(palette=[...])``."""
    if rgb is None:
        raise ValueError("colour.measure needs rgb")
    from gap_skills.tools.colour import _impl

    return _impl.measure(
        np.asarray(rgb),
        mask=None if mask is None else np.asarray(mask),
        white_ref=white_ref,
        reference_rgb=reference_rgb,
        erode_px=int(erode_px),
    )


@tool(
    name="colour.white_balance",
    summary=(
        "Per-channel white-balance gains from grey world or from a white/grey reference "
        "region; optionally the balanced image."
    ),
    tags=("perception",),
)
def white_balance(
    rgb: np.ndarray,
    mask: Mask | None = None,
    method: str = "grey_world",
    reference_rgb: list[float] | None = None,
    exposure: bool = True,
    apply: bool = False,
) -> WhiteBalance:
    """Gains in linear sRGB. ``method="grey_world"``: ``mask`` (default the
    whole frame) averages to grey; chromatic only. ``method="reference"``:
    ``mask`` covers a region whose true colour is ``reference_rgb`` (default
    white, #F4F4F4); fixes the cast and, with ``exposure``, the brightness.
    Pass the returned dict as ``white_ref`` to the other tools; ``apply``
    adds the balanced uint8 ``image``."""
    if rgb is None:
        raise ValueError("colour.white_balance needs rgb")
    from gap_skills.tools.colour import _impl

    return _impl.white_balance(
        np.asarray(rgb),
        mask=None if mask is None else np.asarray(mask),
        method=str(method),
        reference_rgb=reference_rgb,
        exposure=bool(exposure),
        apply=bool(apply),
    )


@tool(
    name="colour.dominant_colours",
    summary=(
        "Cluster a mask into its dominant colours (k-means in Lab, near clusters merged), "
        "each with its share and optionally its palette name."
    ),
    tags=("perception",),
)
def dominant_colours(
    rgb: np.ndarray,
    mask: Mask | None = None,
    k: int | None = None,
    max_k: int = 6,
    palette: str | list[dict[str, Any]] | None = None,
    white_ref: Any = None,
    reference_rgb: list[float] | None = None,
    return_labels: bool = False,
) -> DominantColours:
    """k-means in Lab over the mask's unclipped pixels. With ``k`` exactly that
    many clusters; without, ``max_k`` clusters merged while two are within 10
    CIEDE2000 (one surface's shading), clusters under 5% dropped. Largest
    first; ``palette`` names each; ``return_labels`` adds an int16 label image
    (index into ``colours``, -1 outside / dropped)."""
    if rgb is None:
        raise ValueError("colour.dominant_colours needs rgb")
    from gap_skills.tools.colour import _impl

    return _impl.dominant_colours(
        np.asarray(rgb),
        mask=None if mask is None else np.asarray(mask),
        k=None if k is None else int(k),
        max_k=int(max_k),
        palette=palette,
        white_ref=white_ref,
        reference_rgb=reference_rgb,
        return_labels=bool(return_labels),
    )
