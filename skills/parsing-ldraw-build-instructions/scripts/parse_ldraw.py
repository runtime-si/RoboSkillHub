# Adapted from ldraw-py (https://github.com/michaelgale/ldraw-py), ldrawpy/ldrmodel.py at commit
# d650ab0a9c2a774e1ba6f206aa2cd3b1925478e3 (parse_file's "0 FILE" split, parse_model's "0 STEP"
# split, get_parts_from_model's BUFEXCHG masking, recursive_parse_model's submodel transform,
# unwrap_model's submodel-steps-before-the-parent-step order), MIT License, Copyright (c) 2019
# Michael Gale; and from BrickGPT (https://github.com/AvaLovelace1/BrickGPT),
# src/brickgpt/data/brick_structure.py Brick.from_ldr at commit
# 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4 (x = x0/20 - h/2, y = z0/20 - w/2, z = -y0/24), MIT
# License, Copyright (c) 2025 Ava Pun.
# Changes: no dependencies (ldraw-py needs its toolbox, brickbom and rich packages); the LPub
# page, PLI, callout, scale and view-angle bookkeeping is dropped, steps and submodels kept; a
# submodel's colour 16 takes its parent's colour; ROTSTEP ends a step as STEP does. BrickGPT
# accepted eight brick sizes and two exact matrix strings; here any rotation about the vertical by a
# multiple of 90 deg is read (with a tolerance), plates, tiles and DUPLO are added from the official
# LDraw library's part geometry, heights are counted in plate heights so plates stack, the grid is
# anchored at a baseplate or the model's corner, and parts that do not fit are listed with a reason
# instead of raising.
"""Read an LDraw .ldr / .mpd build into steps of bricks on a stud grid."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STUD_LDU = {"lego": 20.0, "duplo": 40.0}
"""Stud pitch in LDraw units (LDU): 20 LDU per LEGO stud (LDraw spec; BrickGPT's x0 / 20), 40 per
DUPLO stud (DUPLO 2x2 brick 3437 spans -40..40)."""

UNIT_LDU = {"lego": 8.0, "duplo": 24.0}
"""The vertical unit, one plate height, in LDU: LEGO plate 8 (brick 24 = 3 plates, BrickGPT's
y0 / 24), DUPLO plate 24 (brick 48 = 2 plates), from the parts' geometry."""

UNITS_PER_BRICK = {"lego": 3, "duplo": 2}
"""Plate heights per brick height; ``layer`` = height in plates / this."""

FIRST_LAYER = 1
"""The layer of a brick standing on the ground: ICL's z (and computing-brick-pose-from-stud-grid's
``layer``) counts from 1."""

GRID_TOLERANCE_LDU = 1.0
"""How far a part's corner may sit off the stud or plate grid [LDU] (1 LDU = 0.4 mm). Editors write
coordinates such as 59.9998; a half-stud jumper offset is 10 LDU."""

MATRIX_TOLERANCE = 1e-3
"""How far a rotation matrix entry may be from 0 / +-1 to count as a 90-degree turn."""

# Basic parts: LDraw id -> (kind, system, studs along the part's own X, studs along its own Z).
# Checked against the official LDraw library (complete.zip, 2026-11-08): each part's bounding box,
# from its geometry, spans these studs along X and Z with its top face at y = 0 and its bottom at
# y = 24 (brick), 8 (plate, tile), 48 (DUPLO brick) or 24 (DUPLO plate).
_BRICKS = {
    "3005": (1, 1), "3004": (2, 1), "3622": (3, 1), "3010": (4, 1), "3009": (6, 1),
    "3008": (8, 1), "6111": (10, 1), "6112": (12, 1), "2465": (16, 1),
    "3003": (2, 2), "3002": (3, 2), "3001": (4, 2), "2456": (6, 2), "3007": (8, 2),
    "3006": (10, 2), "2356": (6, 4), "6212": (10, 4), "4202": (12, 4), "4201": (8, 8),
    "4204": (16, 8),
}  # fmt: skip
_PLATES = {
    "3024": (1, 1), "3023": (2, 1), "3023b": (2, 1), "3623": (3, 1), "3710": (4, 1),
    "78329": (5, 1), "3666": (6, 1), "3460": (8, 1), "4477": (10, 1), "60479": (12, 1),
    "3022": (2, 2), "3021": (3, 2), "3020": (4, 2), "3795": (6, 2), "3034": (8, 2),
    "3832": (10, 2), "2445": (12, 2), "4282": (16, 2), "11212": (3, 3), "3031": (4, 4),
    "3032": (6, 4), "3035": (8, 4), "3030": (10, 4), "3029": (12, 4), "3958": (6, 6),
    "3036": (8, 6), "3033": (10, 6), "3028": (12, 6), "41539": (8, 8),
}  # fmt: skip
_TILES = {
    "3070b": (1, 1), "3069b": (2, 1), "63864": (3, 1), "2431": (4, 1), "6636": (6, 1),
    "4162": (8, 1), "3068b": (2, 2), "26603": (3, 2), "87079": (4, 2), "69729": (6, 2),
    "1751": (4, 4),
}  # fmt: skip
_DUPLO_BRICKS = {
    "3437": (2, 2), "87084": (3, 2), "3011": (4, 2), "2300": (6, 2), "4199": (8, 2),
    "2291": (10, 2),
}  # fmt: skip
_DUPLO_PLATES = {
    "40666": (4, 2), "98233": (6, 2), "44524": (8, 2), "14721": (4, 4), "25549": (6, 4),
    "51262": (8, 8),
}  # fmt: skip
_BASEPLATES = {
    "3811": (32, 32), "3867": (16, 16), "4186": (48, 48), "3865": (16, 8), "4187": (32, 8),
    "3645": (40, 24), "3334": (24, 16), "3857": (32, 16),
}  # fmt: skip
_DUPLO_BASEPLATES = {
    "5930": (16, 16), "4268": (24, 24), "6851": (16, 12), "31043": (12, 8), "6475": (24, 16),
}  # fmt: skip

PARTS: dict[str, tuple[str, str, int, int]] = {
    **{k: ("brick", "lego", *v) for k, v in _BRICKS.items()},
    **{k: ("plate", "lego", *v) for k, v in _PLATES.items()},
    **{k: ("tile", "lego", *v) for k, v in _TILES.items()},
    **{k: ("brick", "duplo", *v) for k, v in _DUPLO_BRICKS.items()},
    **{k: ("plate", "duplo", *v) for k, v in _DUPLO_PLATES.items()},
}
"""Parts converted to bricks. A tile is a plate with no studs on top."""

BASEPLATES: dict[str, tuple[str, int, int]] = {
    **{k: ("lego", *v) for k, v in _BASEPLATES.items()},
    **{k: ("duplo", *v) for k, v in _DUPLO_BASEPLATES.items()},
}
"""Baseplates (top face at y = 0): the grid's ground and corner, not a step's brick."""

PATTERN_SUFFIX = re.compile(r"^(?P<base>.+?)p[0-9a-z]{2,4}$")
"""A printed or stickered variant (3001p01, 3068bp03): the base part's geometry. All 1452 such
variants of the parts above in the library have the base part's box (one sticker 1 LDU over)."""

PATTERN_EXCEPTIONS = frozenset({"3002peye"})
"""Ids that look like a variant but are not one (3002peye is a moved alias of 3003pe1, a 2x2)."""

HEIGHT_UNITS = {("brick", "lego"): 3, ("plate", "lego"): 1, ("tile", "lego"): 1}
HEIGHT_UNITS |= {("brick", "duplo"): 2, ("plate", "duplo"): 1}

# LDraw colour codes -> names, from LDraw.org's LDConfig.ldr (2026-10-06 update, CCAL 2.0 / CC BY
# 2.0, attribution LDraw.org), lowercased. The names in PALETTE_NAMES are the ones tools/colour's
# "lego" palette classifies, so a step's colour can be checked against a camera with colour.classify.
LDRAW_COLOURS = {
    0: "black", 1: "blue", 2: "green", 3: "dark_turquoise", 4: "red", 5: "dark_pink",
    6: "brown", 7: "light_grey", 8: "dark_grey", 9: "light_blue", 10: "bright_green",
    11: "light_turquoise", 12: "salmon", 13: "pink", 14: "yellow", 15: "white",
    17: "light_green", 18: "light_yellow", 19: "tan", 20: "light_violet", 22: "purple",
    23: "dark_blue_violet", 25: "orange", 26: "magenta", 27: "lime", 28: "dark_tan",
    29: "bright_pink", 30: "medium_lavender", 31: "lavender", 33: "trans_dark_blue",
    34: "trans_green", 35: "trans_bright_green", 36: "trans_red", 37: "trans_dark_pink",
    38: "trans_neon_orange", 39: "trans_very_light_blue", 40: "trans_brown",
    41: "trans_medium_blue", 42: "trans_neon_green", 43: "trans_light_blue", 45: "trans_pink",
    46: "trans_yellow", 47: "trans_clear", 52: "trans_purple", 54: "trans_neon_yellow",
    57: "trans_orange", 68: "very_light_orange", 69: "bright_reddish_lilac",
    70: "reddish_brown", 71: "light_bluish_grey", 72: "dark_bluish_grey", 73: "medium_blue",
    74: "medium_green", 77: "light_pink", 78: "light_nougat", 79: "milky_white",
    84: "medium_nougat", 85: "medium_lilac", 86: "light_brown", 89: "blue_violet",
    191: "bright_light_orange", 226: "bright_light_yellow", 288: "dark_green",
    320: "dark_red", 321: "dark_azure", 322: "medium_azure", 378: "sand_green",
    484: "dark_orange",
}  # fmt: skip
PALETTE_NAMES = frozenset(
    {
        "black", "blue", "green", "dark_pink", "bright_green", "yellow", "white", "tan",
        "orange", "magenta", "lime", "dark_tan", "reddish_brown", "light_bluish_grey",
        "dark_bluish_grey", "medium_blue", "bright_light_orange", "bright_light_yellow",
        "dark_green", "dark_red", "dark_azure", "medium_azure", "sand_green", "dark_orange",
        "red",
    }
)  # fmt: skip
MAIN_COLOUR, EDGE_COLOUR = 16, 24

STEP_COMMANDS = ("STEP", "ROTSTEP")
"""Meta commands that end a step: LDraw's ``0 STEP`` and MLCad's ``0 ROTSTEP`` (a step with a view
turn). ldraw-py splits on ``0 STEP`` only."""


class Output(TypedDict):
    route: str
    detail: str
    steps: list[dict[str, Any]]
    bricks: list[dict[str, Any]]
    unsupported: list[dict[str, Any]]
    off_grid: list[dict[str, Any]]
    baseplate: dict[str, Any]
    system: str
    n_bricks: int
    n_steps: int
    n_unsupported: int
    n_off_grid: int
    steps_json: str


class _Invalid(Exception):
    pass


def _part_key(name: str) -> str:
    key = name.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    return key[:-4] if key.endswith(".dat") else key


def _base_part(key: str) -> str:
    """The table entry a part id uses: itself, or the base of a printed variant."""
    if key in PARTS or key in BASEPLATES or key in PATTERN_EXCEPTIONS:
        return key
    m = PATTERN_SUFFIX.match(key)
    if m and (m["base"] in PARTS or m["base"] in BASEPLATES):
        return m["base"]
    return key


def _file_key(name: str) -> str:
    return name.replace("\\", "/").strip().lower()


def split_files(text: str) -> tuple[str, dict[str, list[str]]]:
    """The main model's name and every ``0 FILE`` block's lines (parse_file's split); a plain
    .ldr is one model named ``main``. ``0 NOFILE`` ends a block (embedded data follows)."""
    files: dict[str, list[str]] = {}
    current: str | None = None
    first: str | None = None
    loose: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        tok = line.split()
        if len(tok) >= 2 and tok[0] == "0" and tok[1] == "FILE":
            current = _file_key(line.split(None, 2)[2] if len(tok) > 2 else "")
            files.setdefault(current, [])
            first = first or current
            continue
        if len(tok) >= 2 and tok[0] == "0" and tok[1] == "NOFILE":
            current = None
            continue
        if current is not None:
            files[current].append(line)
        elif first is None:
            loose.append(line)
    if first is None:
        return "main", {"main": loose}
    return first, files


def _steps_of(lines: list[str]) -> list[list[str]]:
    """A model's lines cut at STEP / ROTSTEP; parts between ``BUFEXCHG STORE`` and ``RETRIEVE``
    (LPub's shown-then-removed parts) are dropped, as get_parts_from_model drops them."""
    steps: list[list[str]] = [[]]
    in_buffer = False
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "0":
            upper = [t.upper() for t in tok[1:]]
            if upper and upper[0] in STEP_COMMANDS:
                steps.append([])
            elif "BUFEXCHG" in upper and "STORE" in upper:
                in_buffer = True
            elif "BUFEXCHG" in upper and "RETRIEVE" in upper:
                in_buffer = False
            continue
        if tok[0] == "1" and not in_buffer:
            steps[-1].append(line)
    return steps


def _colour(code: Any, parent: Any) -> Any:
    if code == str(MAIN_COLOUR) or code == MAIN_COLOUR:
        return parent
    try:
        return int(code)
    except ValueError:
        return str(code)


def _colour_fields(code: Any) -> dict[str, Any]:
    if isinstance(code, int) and code in LDRAW_COLOURS:
        name = LDRAW_COLOURS[code]
        return {"colour": name, "colour_code": code, "colour_in_palette": name in PALETTE_NAMES}
    if isinstance(code, str) and code.lower().startswith("0x2") and len(code) == 9:
        return {"colour": "#" + code[3:].upper(), "colour_code": code, "colour_in_palette": False}
    label = "main_colour" if code == MAIN_COLOUR else f"ldraw_{code}"
    return {"colour": label, "colour_code": code, "colour_in_palette": False}


def _parse_type1(line: str) -> tuple[str, np.ndarray, np.ndarray, str]:
    tok = line.split()
    if len(tok) < 15:
        raise ValueError("a type-1 line needs 15 fields")
    vals = [float(v) for v in tok[2:14]]
    return tok[1], np.array(vals[0:3]), np.array(vals[3:12]).reshape(3, 3), " ".join(tok[14:])


def collect(
    main: str, files: dict[str, list[str]], submodels: str = "inline"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every part in world LDU with its step, in build order. ``inline``: a step that places a
    submodel is preceded by the submodel's own steps (unwrap_model's order); ``flatten``: the
    submodel's parts all join the placing step. Returns (steps, problems)."""
    out: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    def walk(name: str, origin: np.ndarray, rot: np.ndarray, colour: Any, chain: tuple, into):
        if name in chain:
            raise _Invalid(f"submodel {name!r} includes itself ({' > '.join((*chain, name))})")
        chain = (*chain, name)
        model = " > ".join(chain)
        for k, step_lines in enumerate(_steps_of(files[name]), start=1):
            own: list[dict[str, Any]] = []
            for line in step_lines:
                try:
                    code, loc, mat, ref = _parse_type1(line)
                except ValueError as exc:
                    problems.append({"line": line, "reason": f"unreadable line: {exc}"})
                    continue
                c = _colour(code, colour)
                w_loc = rot @ loc + origin
                w_rot = rot @ mat
                key = _file_key(ref)
                if key in files:
                    if submodels == "flatten":
                        walk(key, w_loc, w_rot, c, chain, own)
                    else:
                        walk(key, w_loc, w_rot, c, chain, None)
                    continue
                own.append(
                    {"ref": ref, "colour": c, "loc": w_loc, "rot": w_rot, "line": line,
                     "model": model, "source_step": k}
                )  # fmt: skip
            if into is not None:
                into.extend(own)
            elif own:
                out.append({"model": model, "source_step": k, "parts": own})

    walk(main, np.zeros(3), np.eye(3), MAIN_COLOUR, (), None)
    return out, problems


def _yaw_axis(rot: np.ndarray) -> str | None:
    """'x' / 'z': the world axis the part's own X lies along, for a rotation about the vertical by a
    multiple of 90 deg; None for any other matrix (tilted, upside down, mirrored, scaled)."""
    if not np.allclose(rot @ rot.T, np.eye(3), atol=MATRIX_TOLERANCE):
        return None
    if abs(rot[1, 1] - 1.0) > MATRIX_TOLERANCE or np.linalg.det(rot) < 0:
        return None
    if abs(abs(rot[0, 0]) - 1.0) <= MATRIX_TOLERANCE:
        return "x"
    if abs(abs(rot[2, 0]) - 1.0) <= MATRIX_TOLERANCE:
        return "z"
    return None


def _why_off_grid(rot: np.ndarray) -> str:
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-2):
        return "scaled or sheared matrix"
    if np.linalg.det(rot) < 0:
        return "mirrored matrix"
    if rot[1, 1] < -1.0 + 1e-2:
        return "upside down"
    if abs(rot[1, 1] - 1.0) > MATRIX_TOLERANCE:
        return "tilted off the vertical"
    yaw = math.degrees(math.atan2(rot[0, 2], rot[0, 0]))
    return f"turned {yaw:.1f} deg about the vertical, not a multiple of 90"


def _num(v: float) -> float | int:
    r = round(v)
    return int(r) if abs(v - r) < 1e-9 else round(v, 6)


def _frac(v: float) -> float:
    return v - math.floor(v)


def _near(a: float, b: float, tol: float) -> bool:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d) <= tol


def _phase(values: list[float]) -> float:
    """The fractional offset most values share, to a 20th (0 when it is within one)."""
    bins: dict[int, int] = {}
    for v in values:
        b = round(_frac(v) * 20) % 20
        bins[b] = bins.get(b, 0) + 1
    best = max(sorted(bins), key=lambda b: bins[b])
    return best / 20.0


def _aligned(lowest: float, phase: float, tol: float) -> float:
    """The grid line (``phase`` + an integer) at or just under ``lowest``."""
    return math.floor(lowest - phase + tol) + phase


def run(
    ctx: NodeContext,
    ldraw_path: str = "",
    ldraw_text: str = "",
    files: dict[str, str] | None = None,
    submodels: str = "inline",
    origin: str = "auto",
    system: str = "",
) -> Output:
    """Parse an LDraw file into stud-grid build steps.

    Args:
        ldraw_path: an .ldr / .mpd file. Sub-files it references that are not ``0 FILE`` blocks
            are looked up beside it.
        ldraw_text: the file's text instead of a path.
        files: extra sub-files, name -> text (names as the file references them).
        submodels: ``inline`` (a submodel's steps come before the step that places it, built in
            place) or ``flatten`` (the submodel goes in whole in that step).
        origin: ``auto`` (a single baseplate's corner and top if there is one, else ``min``),
            ``baseplate``, ``min`` (the lowest corner of all bricks is cell (0, 0) and its bottom
            layer 0) or ``ldraw`` (cells and layers counted from LDraw's origin, the ground y = 0).
        system: ``lego`` or ``duplo`` to override the system read from the parts.
    """
    del ctx
    empty: Output = {
        "route": "invalid_file", "detail": "", "steps": [], "bricks": [], "unsupported": [],
        "off_grid": [], "baseplate": {}, "system": "", "n_bricks": 0, "n_steps": 0,
        "n_unsupported": 0, "n_off_grid": 0, "steps_json": "[]",
    }  # fmt: skip
    if submodels not in ("inline", "flatten"):
        return {**empty, "detail": f"submodels must be 'inline' or 'flatten', not {submodels!r}"}
    if origin not in ("auto", "baseplate", "min", "ldraw"):
        return {**empty, "detail": f"unknown origin {origin!r}"}
    base_dir: Path | None = None
    if not ldraw_text:
        if not ldraw_path:
            return {**empty, "detail": "give ldraw_path or ldraw_text"}
        path = Path(ldraw_path).expanduser()
        try:
            ldraw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {**empty, "detail": f"cannot read {ldraw_path}: {exc}"}
        base_dir = path.parent

    main, blocks = split_files(ldraw_text)
    for name, text in (files or {}).items():
        blocks.setdefault(_file_key(name), [ln.strip() for ln in text.splitlines()])
    # External sub-files beside the main file (an .ldr that references another .ldr).
    if base_dir is not None:
        pending = list(blocks)
        seen = set(pending)
        while pending:
            for line in blocks[pending.pop()]:
                tok = line.split()
                if len(tok) < 15 or tok[0] != "1":
                    continue
                ref = _file_key(" ".join(tok[14:]))
                if ref in seen or not ref.endswith((".ldr", ".mpd")):
                    continue
                seen.add(ref)
                candidate = base_dir / ref
                if candidate.is_file():
                    sub_main, sub_blocks = split_files(candidate.read_text(errors="replace"))
                    blocks[ref] = sub_blocks.pop(sub_main)
                    for k, v in sub_blocks.items():
                        blocks.setdefault(k, v)
                        seen.add(k)
                    pending.append(ref)
                    pending.extend(sub_blocks)

    try:
        raw_steps, problems = collect(main, blocks, submodels)
    except _Invalid as exc:
        return {**empty, "detail": str(exc)}

    unsupported: list[dict[str, Any]] = [
        {**p, "step": None, "model": main, "part": ""} for p in problems
    ]
    off_grid: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    baseplates: list[dict[str, Any]] = []
    for step_no, step in enumerate(raw_steps, start=1):
        for p in step["parts"]:
            part = _part_key(p["ref"])
            key = _base_part(part)
            where = {"step": step_no, "model": p["model"], "source_step": p["source_step"]}
            info = {"part": part, "line": p["line"], **where}
            if key in BASEPLATES or key in PARTS:
                axis = _yaw_axis(p["rot"])
                if axis is None:
                    off_grid.append({**info, "reason": _why_off_grid(p["rot"])})
                    continue
                if key in BASEPLATES:
                    sysname, nx, nz = BASEPLATES[key]
                    baseplates.append({**p, **info, "base_part": key, "system": sysname,
                                       "nx": nx, "nz": nz, "axis": axis})  # fmt: skip
                else:
                    kind, sysname, nx, nz = PARTS[key]
                    candidates.append({**p, **info, "base_part": key, "kind": kind,
                                       "system": sysname, "nx": nx, "nz": nz,
                                       "axis": axis})  # fmt: skip
                continue
            reason = (
                "submodel file not found"
                if key.endswith((".ldr", ".mpd"))
                else "not a basic brick, plate or tile (or a part this table does not list)"
            )
            unsupported.append({**info, "reason": reason})

    systems = sorted({c["system"] for c in candidates} | {b["system"] for b in baseplates})
    if system:
        grid_system = system
    elif candidates:
        counts = {s: sum(c["system"] == s for c in candidates) for s in systems}
        grid_system = max(counts, key=lambda s: (counts[s], s == "lego"))
    elif baseplates:
        grid_system = baseplates[0]["system"]
    else:
        grid_system = "lego"
    if grid_system not in STUD_LDU:
        return {**empty, "detail": f"unknown system {grid_system!r}"}
    keep = []
    for c in candidates:
        if c["system"] != grid_system:
            unsupported.append(
                {k: c[k] for k in ("part", "line", "step", "model", "source_step")}
                | {"reason": f"{c['system']} part in a {grid_system} build"}
            )
        else:
            keep.append(c)
    candidates = keep
    plates = [b for b in baseplates if b["system"] == grid_system]

    pitch, unit = STUD_LDU[grid_system], UNIT_LDU[grid_system]
    for c in candidates + plates:
        span_x, span_z = (c["nx"], c["nz"]) if c["axis"] == "x" else (c["nz"], c["nx"])
        c["span"] = (span_x, span_z)
        c["corner"] = (c["loc"][0] / pitch - span_x / 2.0, c["loc"][2] / pitch - span_z / 2.0)
        height = HEIGHT_UNITS.get((c.get("kind", ""), grid_system), 0)
        c["height_units"] = height
        # LDraw's y points down; a part's top face is at its origin, its bottom `height` lower.
        c["bottom"] = -(c["loc"][1] + height * unit) / unit

    plate = None
    if origin == "baseplate" and len(plates) != 1:
        return {**empty, "detail": f"origin 'baseplate' needs one baseplate, found {len(plates)}"}
    if origin in ("auto", "baseplate") and len(plates) == 1:
        plate = plates[0]
    tol_xy = GRID_TOLERANCE_LDU / pitch
    tol_z = GRID_TOLERANCE_LDU / unit
    if plate is not None:
        ox, oy, oz = plate["corner"][0], plate["corner"][1], plate["bottom"]
    elif not candidates:
        ox = oy = oz = 0.0
    else:
        # The grid's phase is the one most bricks share (a brick on a jumper plate is the odd one
        # out, whichever comes first in the file).
        fx = _phase([c["corner"][0] for c in candidates])
        fy = _phase([c["corner"][1] for c in candidates])
        fz = _phase([c["bottom"] for c in candidates])
        if origin == "ldraw":
            ox, oy, oz = fx, fy, fz
        else:
            ox = _aligned(min(c["corner"][0] for c in candidates), fx, tol_xy)
            oy = _aligned(min(c["corner"][1] for c in candidates), fy, tol_xy)
            oz = _aligned(min(c["bottom"] for c in candidates), fz, tol_z)

    bricks: list[dict[str, Any]] = []
    for c in candidates:
        gx, gy, gz = c["corner"][0] - ox, c["corner"][1] - oy, c["bottom"] - oz
        info = {k: c[k] for k in ("part", "line", "step", "model", "source_step")}
        if not (_near(gx, 0.0, tol_xy) and _near(gy, 0.0, tol_xy)):
            off_grid.append({**info, "reason": f"corner {gx:.2f}, {gy:.2f} studs is off the grid"})
            continue
        if not _near(gz, 0.0, tol_z):
            off_grid.append(
                {**info, "reason": f"bottom {gz:.2f} plates up is between plate heights"}
            )
            continue
        x, y, z = round(gx), round(gy), round(gz)
        if z < 0 and plate is not None:
            off_grid.append({**info, "reason": "below the baseplate's top"})
            continue
        span_x, span_y = c["span"]
        short, long_ = sorted((span_x, span_y))
        rotation = 0 if span_x == short else 90
        per = UNITS_PER_BRICK[grid_system]
        bricks.append(
            {
                "id": len(bricks),
                "part": c["part"],
                "base_part": c["base_part"],
                "kind": c["kind"],
                "system": grid_system,
                "size": f"{short}x{long_}",
                "brick_size": f"{short}x{long_}",
                "rotation": rotation,
                "ori": rotation // 90,
                "cell": {
                    "x": x,
                    "y": y,
                    "layer": _num(FIRST_LAYER + z / per),
                    "rotation": rotation,
                },
                "span_x": span_x,
                "span_y": span_y,
                "z_units": z,
                "height_units": c["height_units"],
                "studs_on_top": c["kind"] != "tile",
                **_colour_fields(c["colour"]),
                "step": c["step"],
                "model": c["model"],
                "source_step": c["source_step"],
                "position_ldu": [_num(float(v)) for v in c["loc"]],
            }
        )

    # Renumber steps over the converted bricks only (empty steps dropped).
    steps: list[dict[str, Any]] = []
    by_step: dict[int, list[dict[str, Any]]] = {}
    for b in bricks:
        by_step.setdefault(b["step"], []).append(b)
    renumber = {old: new for new, old in enumerate(sorted(by_step), start=1)}
    for b in bricks:
        b["step"] = renumber[b["step"]]
    for old in sorted(by_step):
        raw = raw_steps[old - 1]
        steps.append({"step": renumber[old], "model": raw["model"],
                      "source_step": raw["source_step"], "bricks": by_step[old]})  # fmt: skip
    for item in unsupported + off_grid:
        item["step_in_file"] = item.pop("step", None)

    baseplate_out: dict[str, Any] = {}
    if plate is not None:
        baseplate_out = {
            "part": plate["part"], "system": grid_system,
            "size": f"{min(plate['span'])}x{max(plate['span'])}",
            "span_x": plate["span"][0], "span_y": plate["span"][1],
            "plate_studs": [plate["span"][0], plate["span"][1]],
            "position_ldu": [_num(float(v)) for v in plate["loc"]],
        }  # fmt: skip
        outside = [
            b["id"] for b in bricks
            if b["cell"]["x"] < 0 or b["cell"]["y"] < 0
            or b["cell"]["x"] + b["span_x"] > plate["span"][0]
            or b["cell"]["y"] + b["span_y"] > plate["span"][1]
        ]  # fmt: skip
        baseplate_out["bricks_outside"] = outside
    elif len(plates) > 1:
        baseplate_out = {"count": len(plates), "note": "several baseplates; grid from the bricks"}

    detail_bits = [f"{len(bricks)} bricks in {len(steps)} steps ({grid_system})"]
    if plate is not None:
        detail_bits.append(f"grid from baseplate {plate['part']}")
    if unsupported:
        parts = sorted({u["part"] or "?" for u in unsupported})
        detail_bits.append(f"{len(unsupported)} unsupported ({', '.join(parts[:8])})")
    if off_grid:
        detail_bits.append(f"{len(off_grid)} off the grid ({off_grid[0]['reason']})")
    if len(systems) > 1:
        detail_bits.append(f"systems seen: {', '.join(systems)}")
    if not bricks and not unsupported and not off_grid:
        route = "invalid_file"
        detail_bits = ["no parts found"]
    elif off_grid:
        route = "off_grid"
    elif unsupported:
        route = "unsupported_part"
    else:
        route = "steps_ready"
    logger.info("parse_ldraw: %s", "; ".join(detail_bits))
    return {
        "route": route,
        "detail": "; ".join(detail_bits),
        "steps": steps,
        "bricks": bricks,
        "unsupported": unsupported,
        "off_grid": off_grid,
        "baseplate": baseplate_out,
        "system": grid_system,
        "n_bricks": len(bricks),
        "n_steps": len(steps),
        "n_unsupported": len(unsupported),
        "n_off_grid": len(off_grid),
        "steps_json": json.dumps(steps),
    }


__all__ = ["run", "split_files", "collect", "PARTS", "BASEPLATES", "LDRAW_COLOURS"]
