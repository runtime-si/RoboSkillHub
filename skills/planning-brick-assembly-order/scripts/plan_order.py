# Adapted from BrickGPT (https://github.com/AvaLovelace1/BrickGPT),
# src/mesh2brick/src/mesh2brick/planning.py plan_robotic_operation at commit
# 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4 (layers bottom-up, within a layer descending (x, y),
# a per-type inventory), MIT License, Copyright (c) 2025 Ava Pun; and from BrickSim
# (https://github.com/intelligent-control-lab/BrickSim), python/bricksim/topology/ordering.py
# bfs_sort_connections at commit cbd5de3238e6ac12f44ef699e1507a9f16bdafc3 (every new part attaches
# to a part already in the assembly, grown from the anchor), MIT License, Copyright (c) 2026 Haowei
# Wen.
# Changes: BrickGPT sorts by layer and drops bricks past the inventory without checking that each
# brick rests on one already placed; BrickSim orders connections of a known graph. Here the two are
# one greedy loop: a brick is a candidate only when it rests on the ground or on studs of a placed
# brick (the stud-overlap graph, grown from the ground as BrickSim grows from its anchor), nothing
# placed covers its footprint, and the gripper's fingers have room on two opposite sides; among the
# candidates the earliest STEP, then the lowest layer, then BrickGPT's (x, y) order wins, skipping a
# candidate that would wall in a brick still to come. Plates, tiles, DUPLO, two-arm reach and the
# reasons a brick cannot be placed are added.
"""Order stud-grid bricks so each is supported, uncovered and graspable when it is pressed."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)

SYSTEMS = {
    "lego": {"stud_pitch": 0.008, "plate_height": 0.0032, "stud_height": 0.0017, "per_brick": 3},
    "duplo": {"stud_pitch": 0.016, "plate_height": 0.0096, "stud_height": 0.0045, "per_brick": 2},
}
"""Brick systems [m]: pitch, plate height (a third of a 9.6 mm LEGO brick, half a 19.2 mm DUPLO
brick), stud height - pressing-a-brick-onto-studs' presets; plates per brick."""

FIRST_LAYER = 1
"""Layer of a brick standing on the plate (ICL's z, computing-brick-pose-from-stud-grid's layer)."""

GRASP_DEPTH = {"lego": 0.005, "duplo": 0.013}
ONE_WIDE_EXTRA = {"lego": 0.001, "duplo": 0.0}
"""How far under a brick's top the fingertips reach [m]: BrickSim's GRASP_DEPTH for LEGO (6 mm for
1-wide); DUPLO 13 mm, the depth pressing-a-brick-onto-studs' DUPLO check grasped at. Capped at the
part's own height (a plate is held over its whole side)."""

FINGER_THICKNESS = 0.010
FINGER_WIDTH = 0.020
FINGER_MARGIN = 0.003
"""A linear_4310 fingertip's depth along the closing axis, its width across it, and the free space
it needs around it [m]: singulating-by-pushing's estimates (the gripper reports pad_thickness 0,
unmeasured)."""

JAW_OPENING = 0.094
JAW_MARGIN = 0.004
"""The YAM linear_4310's measured 94 mm span, and the clearance over the brick's width the open jaws
need (computing-brick-pose-from-stud-grid's jaw_margin)."""

EPS = 1e-6


class Output(TypedDict):
    route: str
    detail: str
    sequence: list[dict[str, Any]]
    by_arm: dict[str, list[int]]
    not_placed: list[dict[str, Any]]
    step_deviations: list[int]
    n_bricks: int
    n_placed: int
    n_not_placed: int
    system: str
    sequence_json: str


class _Bad(Exception):
    pass


def _size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        a, _, b = value.lower().partition("x")
        return int(a), int(b)
    a, b = value
    return int(a), int(b)


def _normalise(raw: dict[str, Any], index: int, default_system: str, step: int) -> dict[str, Any]:
    """A brick record (parsing-ldraw-build-instructions' or hand-written: cell, size, rotation)
    with its footprint, heights and step filled in."""
    cell = raw.get("cell") or raw
    try:
        x, y = int(cell["x"]), int(cell["y"])
        layer = float(cell.get("layer", FIRST_LAYER))
        size = raw.get("brick_size") or raw.get("size") or raw.get("brick_studs")
        a, b = _size(size)
        rotation = int(cell.get("rotation", raw.get("rotation", 0))) % 360
    except (KeyError, TypeError, ValueError) as exc:
        raise _Bad(
            f"brick {index}: needs cell x, y, layer and a size such as '2x4' ({exc})"
        ) from exc
    if rotation % 90:
        raise _Bad(f"brick {index}: rotation {rotation} is not a multiple of 90")
    system = str(raw.get("system") or default_system)
    if system not in SYSTEMS:
        raise _Bad(f"brick {index}: unknown system {system!r}")
    per = SYSTEMS[system]["per_brick"]
    kind = str(raw.get("kind", "brick"))
    span_x, span_y = (a, b) if rotation in (0, 180) else (b, a)
    span_x, span_y = int(raw.get("span_x", span_x)), int(raw.get("span_y", span_y))
    z = raw.get("z_units")
    z = int(z) if z is not None else round((layer - FIRST_LAYER) * per)
    h = raw.get("height_units")
    h = int(h) if h is not None else (per if kind == "brick" else 1)
    return {
        "id": raw.get("id", index),
        "index": index,
        "raw": raw,
        "x": x,
        "y": y,
        "z": z,
        "h": h,
        "sx": span_x,
        "sy": span_y,
        "layer": layer,
        "rotation": 0 if rotation in (0, 180) else 90,
        "size": f"{min(a, b)}x{max(a, b)}",
        "kind": kind,
        "system": system,
        "studs_on_top": bool(raw.get("studs_on_top", kind != "tile")),
        "step": int(raw.get("step", step)),
    }


def _num(v: float) -> float | int:
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 6)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 - EPS and b0 < a1 - EPS


def _stud_overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    dx = min(a["x"] + a["sx"], b["x"] + b["sx"]) - max(a["x"], b["x"])
    dy = min(a["y"] + a["sy"], b["y"] + b["sy"]) - max(a["y"], b["y"])
    return max(dx, 0) * max(dy, 0)


def _finger_boxes(b: dict[str, Any], along: str, geo: dict[str, float]) -> list[tuple]:
    """Plan boxes (x0, x1, y0, y1) [m] the two fingertips sweep beside brick b when the jaws close
    along grid axis ``along``, and the height [m] their tips come down to."""
    p = geo["pitch"]
    x0, x1, y0, y1 = b["x"] * p, (b["x"] + b["sx"]) * p, b["y"] * p, (b["y"] + b["sy"]) * p
    reach = geo["thickness"] + geo["margin"]
    half = geo["width"] / 2.0 + geo["margin"]
    if along == "x":
        cy = (y0 + y1) / 2.0
        return [(x0 - reach, x0, cy - half, cy + half), (x1, x1 + reach, cy - half, cy + half)]
    cx = (x0 + x1) / 2.0
    return [(cx - half, cx + half, y0 - reach, y0), (cx - half, cx + half, y1, y1 + reach)]


def run(
    ctx: NodeContext,
    bricks: list[dict[str, Any]] | None = None,
    steps: list[dict[str, Any]] | None = None,
    steps_json: str = "",
    system: str = "",
    step_order: str = "prefer",
    sweep: str = "descending",
    ground_supports: bool = True,
    min_overlap_studs: int = 1,
    check_fingers: bool = True,
    close_axis: str = "auto",
    grasp_depth: float | None = None,
    finger_thickness: float = FINGER_THICKNESS,
    finger_width: float = FINGER_WIDTH,
    finger_margin: float = FINGER_MARGIN,
    jaw_opening: float = JAW_OPENING,
    jaw_margin: float = JAW_MARGIN,
    arms: list[str] | None = None,
    split_x: float | None = None,
    left_side: str = "-x",
    reach: dict[str, list[float]] | None = None,
    inventory: dict[str, int] | None = None,
) -> Output:
    """Order bricks for a robot that presses each one down from above.

    Args:
        bricks: brick records (cell {x, y, layer[, rotation]}, size 'AxB', rotation; optionally
            kind, system, z_units, height_units, studs_on_top, step, id) - e.g.
            parsing-ldraw-build-instructions' ``bricks``.
        steps: or its ``steps`` ([{step, bricks}]); steps_json: or the same as JSON text.
        step_order: ``prefer`` (earlier STEPs first unless a brick cannot go yet), ``strict``
            (never start a step before the last one is done) or ``ignore`` (BrickGPT: layers only).
        sweep: within a layer, ``descending`` (x, y) - BrickGPT's order - or ``ascending``.
        close_axis: ``auto`` (jaws across the short side, else the long side), ``short``, ``long``.
        arms: e.g. ["left", "right"] to split by x; split_x: the cell x between them (default the
            middle of the build); left_side: ``-x`` or ``+x``, the side the first arm works;
            reach: per arm [x_min, x_max] in cells that its bricks' footprints must lie within.
        inventory: count per part id, per "kind size" ("plate 2x4") or per size ("2x4").
    """
    del ctx
    empty: Output = {
        "route": "invalid_input", "detail": "", "sequence": [], "by_arm": {}, "not_placed": [],
        "step_deviations": [], "n_bricks": 0, "n_placed": 0, "n_not_placed": 0, "system": "",
        "sequence_json": "[]",
    }  # fmt: skip
    if step_order not in ("prefer", "strict", "ignore"):
        return {**empty, "detail": f"unknown step_order {step_order!r}"}
    if sweep not in ("descending", "ascending"):
        return {**empty, "detail": f"unknown sweep {sweep!r}"}
    if close_axis not in ("auto", "short", "long"):
        return {**empty, "detail": f"unknown close_axis {close_axis!r}"}
    if left_side not in ("-x", "+x"):
        return {**empty, "detail": f"left_side must be '-x' or '+x', not {left_side!r}"}

    if steps is None and steps_json:
        try:
            steps = json.loads(steps_json)
        except json.JSONDecodeError as exc:
            return {**empty, "detail": f"steps_json is not JSON: {exc}"}
    records: list[tuple[dict[str, Any], int]] = []
    if bricks:
        records = [(b, int(b.get("step", 1))) for b in bricks]
    elif steps:
        for k, st in enumerate(steps, start=1):
            records.extend((b, int(st.get("step", k))) for b in st.get("bricks", []))
    if not records:
        return {**empty, "detail": "no bricks given (bricks, steps or steps_json)"}
    default_system = system or str(records[0][0].get("system") or "lego")
    try:
        items = [_normalise(r, i, default_system, s) for i, (r, s) in enumerate(records)]
    except _Bad as exc:
        return {**empty, "detail": str(exc)}
    systems = {b["system"] for b in items}
    if len(systems) > 1:
        return {**empty, "detail": f"bricks of several systems: {sorted(systems)}"}
    sysname = systems.pop()
    preset = SYSTEMS[sysname]
    n = len(items)

    # Two bricks in the same space: the design is wrong, not the order. The later one is left out
    # (an exact copy is a modelling slip; any other overlap is reported).
    excluded: dict[int, str] = {}
    cells: dict[tuple[int, int, int], int] = {}
    for b in items:
        mine = [
            (b["x"] + i, b["y"] + j, b["z"] + k)
            for i in range(b["sx"])
            for j in range(b["sy"])
            for k in range(b["h"])
        ]
        clash = next((cells[c] for c in mine if c in cells), None)
        if clash is not None:
            o = items[clash]
            same = all(o[k] == b[k] for k in ("x", "y", "z", "sx", "sy", "h"))
            what = "duplicate of" if same else "overlaps"
            excluded[b["index"]] = f"{what} brick {o['id']}"
            continue
        for c in mine:
            cells[c] = b["index"]

    # Support (stud overlap, top on bottom) and cover (above, footprints overlapping).
    supporters: list[list[int]] = [[] for _ in range(n)]
    covers: list[list[int]] = [[] for _ in range(n)]  # covers[a]: bricks a would cover
    by_cell: dict[tuple[int, int], list[int]] = {}
    for b in items:
        for i in range(b["sx"]):
            for j in range(b["sy"]):
                by_cell.setdefault((b["x"] + i, b["y"] + j), []).append(b["index"])
    for b in items:
        seen: set[int] = set()
        for i in range(b["sx"]):
            for j in range(b["sy"]):
                for o in by_cell[(b["x"] + i, b["y"] + j)]:
                    if o == b["index"] or o in seen:
                        continue
                    seen.add(o)
                    a = items[o]
                    if a["z"] + a["h"] == b["z"] and a["studs_on_top"]:
                        if _stud_overlap(a, b) >= min_overlap_studs:
                            supporters[b["index"]].append(o)
                    if a["z"] >= b["z"] + b["h"]:
                        covers[o].append(b["index"])

    # Finger clearance: blocks[a][b] = closing axes of b that a placed brick a takes away.
    geo = {"pitch": preset["stud_pitch"], "thickness": finger_thickness, "width": finger_width,
           "margin": finger_margin}  # fmt: skip
    p, unit, stud_h = preset["stud_pitch"], preset["plate_height"], preset["stud_height"]
    allowed: list[list[str]] = []
    blocks: list[dict[int, set[str]]] = [{} for _ in range(n)]
    for b in items:
        short = "x" if b["sx"] <= b["sy"] else "y"
        long_ = "y" if short == "x" else "x"
        order = {"auto": [short, long_], "short": [short], "long": [long_]}[close_axis]
        if b["sx"] == b["sy"]:
            order = ["x", "y"]
        axes = []
        for ax in dict.fromkeys(order):
            width = (b["sx"] if ax == "x" else b["sy"]) * p
            if width + jaw_margin <= jaw_opening:
                axes.append(ax)
        allowed.append(axes)
    if check_fingers:
        reach_cells = math.ceil((finger_thickness + finger_margin) / p) + 1
        for b in items:
            top = (b["z"] + b["h"]) * unit
            depth = grasp_depth
            if depth is None:
                depth = GRASP_DEPTH[sysname] + (
                    ONE_WIDE_EXTRA[sysname] if min(b["sx"], b["sy"]) == 1 else 0.0
                )
            depth = min(depth, b["h"] * unit)
            tip = top - depth
            near: set[int] = set()
            for i in range(-reach_cells - 2, b["sx"] + reach_cells + 2):
                for j in range(-reach_cells - 2, b["sy"] + reach_cells + 2):
                    near.update(by_cell.get((b["x"] + i, b["y"] + j), ()))
            near.discard(b["index"])
            for ax in allowed[b["index"]]:
                for box in _finger_boxes(b, ax, geo):
                    for o in near:
                        a = items[o]
                        a_top = (a["z"] + a["h"]) * unit + (stud_h if a["studs_on_top"] else 0.0)
                        if a_top <= tip + EPS:
                            continue
                        ax0, ax1 = a["x"] * p, (a["x"] + a["sx"]) * p
                        ay0, ay1 = a["y"] * p, (a["y"] + a["sy"]) * p
                        if _overlap(ax0, ax1, box[0], box[1]) and _overlap(
                            ay0, ay1, box[2], box[3]
                        ):
                            blocks[o].setdefault(b["index"], set()).add(ax)

    # Arms and inventory decide what may be placed at all.
    arm_of: dict[int, str] = {}
    arms = list(arms or [])
    if len(arms) > 2:
        return {**empty, "system": sysname, "detail": "at most two arms"}
    if arms:
        lo = min(b["x"] for b in items)
        hi = max(b["x"] + b["sx"] for b in items)
        split = (lo + hi) / 2.0 if split_x is None else float(split_x)
        first_low = left_side == "-x"
        for b in items:
            centre = b["x"] + b["sx"] / 2.0
            preferred = arms[0] if (centre < split) == first_low else arms[-1]
            options = [preferred] + [a for a in arms if a != preferred]
            ok = [
                a for a in options
                if reach is None or a not in reach
                or (reach[a][0] <= b["x"] and b["x"] + b["sx"] <= reach[a][1])
            ]  # fmt: skip
            if ok:
                arm_of[b["index"]] = ok[0]
            else:
                excluded[b["index"]] = "outside every arm's reach"
    if inventory:
        used: dict[str, int] = {}
        for b in sorted(items, key=lambda b: (b["step"], b["z"], b["index"])):
            if b["index"] in excluded:
                continue
            part = str(b["raw"].get("part", ""))
            key = next(
                (k for k in (part, f"{b['kind']} {b['size']}", b["size"]) if k and k in inventory),
                None,
            )
            if key is None:
                continue
            if used.get(key, 0) >= int(inventory[key]):
                excluded[b["index"]] = f"inventory: no {key} left"
            else:
                used[key] = used.get(key, 0) + 1

    # The greedy order. A brick that an earlier step's brick stands on is needed by that step.
    sign = -1 if sweep == "descending" else 1
    need = [b["step"] for b in items]
    for b in sorted(items, key=lambda b: -b["z"]):
        for s_ in supporters[b["index"]]:
            need[s_] = min(need[s_], need[b["index"]])

    def key(b: dict[str, Any]) -> tuple:
        sweep_key = (sign * b["x"], sign * b["y"])
        head = () if step_order == "ignore" else (need[b["index"]],)
        return (*head, b["z"], *sweep_key, b["index"])

    placed: list[int] = []
    is_placed = [False] * n
    covered = [False] * n
    taken: list[set[str]] = [set() for _ in range(n)]
    pending = {b["index"] for b in items if b["index"] not in excluded}

    def supported(i: int) -> bool:
        if items[i]["z"] == 0 and ground_supports:
            return True
        return any(is_placed[s] for s in supporters[i])

    def free_axes(i: int) -> list[str]:
        return [ax for ax in allowed[i] if ax not in taken[i]]

    def viable(y: int) -> bool:
        """Could y still be placed: pending, uncovered, and supported or with a support to come."""
        if y not in pending or covered[y]:
            return False
        return supported(y) or any(s in pending for s in supporters[y])

    def doomed_by(c: int) -> list[int]:
        out = []
        for y in covers[c]:
            if y != c and viable(y):
                out.append(y)
        for y, axes in blocks[c].items():
            if y != c and viable(y) and free_axes(y):
                if not [ax for ax in free_axes(y) if ax not in axes]:
                    out.append(y)
        return out

    deviations: list[int] = []
    lost: dict[int, str] = {}  # why a brick still to come can no longer be placed, when it happened
    while pending:
        cands = [i for i in pending if supported(i) and not covered[i] and free_axes(i)]
        if step_order == "strict":
            current = min(items[i]["step"] for i in pending)
            cands = [i for i in cands if items[i]["step"] == current]
        if not cands:
            break
        safe = [i for i in cands if not doomed_by(i)]
        pick = min(safe or cands, key=lambda i: key(items[i]))
        if step_order != "ignore" and any(items[i]["step"] < items[pick]["step"] for i in pending):
            deviations.append(items[pick]["id"])
        pending.discard(pick)
        is_placed[pick] = True
        placed.append(pick)
        for y in covers[pick]:
            covered[y] = True
            if y in pending:
                lost.setdefault(y, f"covered: brick {items[pick]['id']} above it was placed first")
        for y, axes in blocks[pick].items():
            taken[y] |= axes
            if y in pending and allowed[y] and not free_axes(y):
                lost.setdefault(
                    y,
                    "no room for the fingers on two opposite sides "
                    f"(brick {items[pick]['id']} took the last one)",
                )

    # The closing axis each brick had free when it was placed (later bricks may take it after).
    along_at: dict[int, str] = {}
    replay: list[set[str]] = [set() for _ in range(n)]
    for i in placed:
        axes = [ax for ax in allowed[i] if not check_fingers or ax not in replay[i]]
        along_at[i] = axes[0] if axes else ""
        for y, axs in blocks[i].items():
            replay[y] |= axs

    out_seq: list[dict[str, Any]] = []
    for seq, i in enumerate(placed, start=1):
        b = items[i]
        along = along_at[i]
        short_axis = "x" if b["sx"] <= b["sy"] else "y"
        rec = dict(b["raw"])
        rec.pop("cell", None)
        out_seq.append(
            rec
            | {
                "seq": seq,
                "id": b["id"],
                "step": b["step"],
                "arm": arm_of.get(i, ""),
                "cell": {
                    "x": b["x"],
                    "y": b["y"],
                    "layer": _num(b["layer"]),
                    "rotation": b["rotation"],
                },
                "size": b["size"],
                "brick_size": b["size"],
                "rotation": b["rotation"],
                "ori": b["rotation"] // 90,
                "span_x": b["sx"],
                "span_y": b["sy"],
                "z_units": b["z"],
                "height_units": b["h"],
                "system": sysname,
                "close_along": along,
                "close_axis": ("short" if along == short_axis or b["sx"] == b["sy"] else "long")
                if along
                else "",
                "supports": [items[s]["id"] for s in supporters[i] if is_placed[s]],
            }
        )

    not_placed = []
    for i in sorted(set(range(n)) - set(placed)):
        b = items[i]
        if i in excluded:
            reason = excluded[i]
        elif not supported(i):
            reason = (
                "its support was not placed"
                if supporters[i]
                else "nothing under it to press onto (it hangs from above, or rests on a tile or on a "
                "part not in the list)"
            )
        elif i in lost:
            reason = lost[i]
        elif covered[i]:
            reason = "covered: a brick above it was placed first"
        elif not free_axes(i):
            reason = (
                "no room for the fingers on two opposite sides"
                if allowed[i]
                else (f"wider than the jaws open on both axes ({b['sx']}x{b['sy']} studs)")
            )
        elif step_order == "strict":
            reason = "strict step order: an earlier step could not be finished"
        else:
            reason = "not placed"
        not_placed.append({"id": b["id"], "step": b["step"], "size": b["size"],
                           "cell": {"x": b["x"], "y": b["y"], "layer": _num(b["layer"])},
                           "reason": reason})  # fmt: skip

    reasons = [r["reason"] for r in not_placed]
    if any(r.startswith("inventory") for r in reasons):
        route = "inventory_short"
    elif any(r.startswith("outside") for r in reasons):
        route = "unreachable_step"
    elif any(r.startswith("no room") or r.startswith("wider") for r in reasons):
        route = "finger_blocked"
    elif any(
        r.startswith(("nothing under", "its support", "covered", "strict", "overlaps", "duplicate"))
        for r in reasons
    ):
        route = "not_buildable"
    elif reasons:
        route = "not_buildable"
    else:
        route = "sequence_ready"
    by_arm: dict[str, list[int]] = {a: [] for a in arms}
    for e in out_seq:
        if e["arm"]:
            by_arm[e["arm"]].append(e["seq"])
    bits = [f"{len(out_seq)}/{n} bricks ordered ({sysname})"]
    if deviations:
        bits.append(f"{len(deviations)} placed ahead of an unfinished step")
    if not_placed:
        counts: dict[str, int] = {}
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
        bits.append("not placed: " + "; ".join(f"{c} {r}" for r, c in counts.items()))
    if arms:
        bits.append(", ".join(f"{a} {len(v)}" for a, v in by_arm.items()))
    detail = "; ".join(bits)
    logger.info("plan_order: %s", detail)
    return {
        "route": route,
        "detail": detail,
        "sequence": out_seq,
        "by_arm": by_arm,
        "not_placed": not_placed,
        "step_deviations": deviations,
        "n_bricks": n,
        "n_placed": len(out_seq),
        "n_not_placed": len(not_placed),
        "system": sysname,
        "sequence_json": json.dumps(out_seq),
    }
