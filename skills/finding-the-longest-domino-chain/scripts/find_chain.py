# Adapted from mexican-train-domino-solver by Anton C Beza Jr
# (https://github.com/RGBz/mexican-train-domino-solver, commit
# ac758e87bbae433a5a10ba4a6907874be17d80d6), MIT License, Copyright (c) 2021 Anton C Beza Jr;
# the license is LICENSE.mexican-train-domino-solver beside this skill. Taken from it:
# - the search, src/Graph.ts Graph.findMostValuableChain: pip values are nodes, tiles are
#   edges, and every chain from the start value is walked depth first, each tile used once,
#   the next tile holding the value the last one left open (Domino.getOtherSide);
# - the ranking, src/findMostValuableChain.ts compareChains: the longer chain, then the one
#   with more pips, then the one that does not end in a double;
# - with no start value, every value in the hand is tried as the start
#   (Graph.findMostValuableChain without a side).
# Changed: TypeScript to Python; the search prunes branches that cannot beat the best chain
# found (the tiles still reachable from the open value bound the length and the pips), which
# keeps it exact, and stops at a node budget (the best chain so far is returned, marked not
# exhaustive) instead of walking every chain; Graph.generateAll's split into connected
# components is not needed with that bound; an optional pips-first objective; tiles are
# returned turned so the matching half comes first; routed exits instead of an empty list.
# New: the world placement of the chain from the train's open end (straight on, round a
# corner before the table edge or a placed tile, doubles crosswise), with the geometry of
# skills/choosing-a-domino-move-by-end-matching.
"""Find the longest (or most valuable) chain of matching dominoes from a start value, and lay it out."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

Tile = tuple[int, int]

TILE_SIZE = (0.048, 0.024)
"""Tile length x width (m), as reading-domino-pips-and-orientation and
choosing-a-domino-move-by-end-matching."""

TOUCH_GAP = 0.001
"""Gap left between a tile and the end it matches (m): touching, without pushing it
(choosing-a-domino-move-by-end-matching)."""

EDGE_MARGIN = 0.01
"""Distance kept from the table edge (m), as choosing-a-domino-move-by-end-matching."""

OVERLAP_TOLERANCE = 0.0005
"""Footprints that overlap by less than this (m) touch, they do not overlap."""

RUN_LOOKAHEAD = 4
"""Straight tiles counted past a corner to choose the turn with more room
(choosing-a-domino-move-by-end-matching)."""

LAYOUT_TRIES = 20_000
"""Tile placements tried while backtracking a layout that does not fit greedily. A full
double-six chain from one end of the YAM table needed 31, no measured layout more (see SKILL.md, Measured)."""

MAX_PIP = 12
"""Largest pip value accepted: double-twelve, the Mexican Train set (the source's
Domino.from takes 0..12)."""

MAX_NODES = 200_000
"""Search nodes (tiles tried) before the search stops and returns its best chain. The exact
search of a 15-tile double-twelve hand (a Mexican Train deal) took at most 34 515 nodes and
0.33 s in 2000 random deals (see SKILL.md, Measured)."""

OBJECTIVES = ("longest", "most_pips")


class Output(TypedDict):
    route: str
    detail: str
    start_value: int
    length: int
    chain_pips: int
    end_value: int
    ends_in_double: bool
    chain_tiles: list
    hand_indices: list
    chain: list
    leftover: list
    leftover_pips: int
    placed_count: int
    new_end: dict
    footprints_after: list
    nodes: int
    exhaustive: bool
    length_bound: int
    layout_tries: int


# ----------------------------------------------------------------------------- the search


def _rank(objective: str, n: int, pips: int, ends_double: bool) -> tuple:
    """compareChains as a sort key (bigger is better)."""
    if objective == "most_pips":
        return (pips, n, not ends_double) if n else (-1, 0, False)
    return (n, pips, not ends_double) if n else (0, -1, False)


def best_chain(
    tiles: list[Tile],
    start: int | None,
    objective: str = "longest",
    max_nodes: int = MAX_NODES,
) -> tuple[list[tuple[int, int, int]], int, bool, int | None, int]:
    """The best chain as (tile index, near value, far value) steps, the nodes searched,
    whether the search finished, the start value used (None: no tile holds it) and the
    longest chain the bound allows (a chain that long is proven longest)."""
    by_value: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(tiles):
        by_value.setdefault(a, []).append(i)
        if b != a:
            by_value.setdefault(b, []).append(i)
    starts = [start] if start is not None else sorted(by_value)
    used = [False] * len(tiles)
    best: dict[str, Any] = {"key": None, "chain": [], "start": None}
    nodes = 0
    stopped = False

    def reachable(v: int) -> tuple[int, int]:
        """At most how many tiles (and pips) a chain from v can still add: the unused tiles in
        v's component, less one non-double for every pair of odd-degree values beyond the
        two ends a chain may have (a chain is an Euler trail of the tiles it uses; one of
        its odd ends must be v, so an even v leaves one pair more)."""
        seen_v, seen_t, stack, n, p = {v}, set(), [v], 0, 0
        degree: dict[int, int] = {}
        plain: list[int] = []
        while stack:
            x = stack.pop()
            for i in by_value.get(x, ()):
                if used[i] or i in seen_t:
                    continue
                seen_t.add(i)
                a, b = tiles[i]
                n, p = n + 1, p + a + b
                if a != b:
                    degree[a] = degree.get(a, 0) + 1
                    degree[b] = degree.get(b, 0) + 1
                    plain.append(a + b)
                y = b if a == x else a
                if y not in seen_v:
                    seen_v.add(y)
                    stack.append(y)
        odd = sum(d % 2 for d in degree.values())
        drop = max(0, (odd - (2 if degree.get(v, 0) % 2 else 0)) // 2)
        if drop:
            n -= drop
            p -= sum(sorted(plain)[:drop])
        return n, p

    def beaten(n: int, pips: int, v: int) -> bool:
        if best["key"] is None:
            return False
        rn, rp = reachable(v)
        top = _rank(objective, n + rn, pips + rp, False)
        return top <= best["key"]

    def walk(v: int, path: list[tuple[int, int, int]], pips: int, s: int) -> None:
        nonlocal nodes, stopped
        key = _rank(
            objective,
            len(path),
            pips,
            bool(path) and tiles[path[-1][0]][0] == tiles[path[-1][0]][1],
        )
        if best["key"] is None or key > best["key"]:
            best.update(key=key, chain=list(path), start=s)
        if stopped or beaten(len(path), pips, v):
            return
        for i in by_value.get(v, ()):
            if used[i]:
                continue
            if nodes >= max_nodes:
                stopped = True
                return
            nodes += 1
            a, b = tiles[i]
            far = b if a == v else a
            used[i] = True
            path.append((i, v, far))
            walk(far, path, pips + a + b, s)
            path.pop()
            used[i] = False

    bound = 0
    for s in starts:
        if s not in by_value:
            continue
        bound = max(bound, reachable(s)[0])
        walk(s, [], 0, s)
    return best["chain"], nodes, not stopped, best["start"], bound


# ----------------------------------------------------------------------------- geometry
# The placement of skills/choosing-a-domino-move-by-end-matching (scripts/choose_move.py).


def _vec(v: Any, n: int = 3) -> np.ndarray:
    if isinstance(v, dict):
        v = [v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
    a = np.zeros(n)
    vals = [float(x) for x in list(v)[:n]]
    a[: len(vals)] = vals
    return a


def _unit2(v: Any) -> np.ndarray:
    a = _vec(v)[:2]
    norm = float(np.linalg.norm(a))
    if norm < 1e-9:
        raise ValueError(f"direction {v} has no length in the table plane")
    return a / norm


def _perp(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


class Rect:
    """A tile footprint in the table plane: centre, unit long axis, half length, half width."""

    def __init__(self, c: np.ndarray, u: np.ndarray, hl: float, hw: float):
        self.c, self.u, self.hl, self.hw = np.asarray(c[:2], float), np.asarray(u, float), hl, hw

    def corners(self) -> np.ndarray:
        v = _perp(self.u)
        return np.array(
            [self.c + su * self.hl * self.u + sv * self.hw * v for su in (-1, 1) for sv in (-1, 1)]
        )

    def overlaps(self, other: Rect, tol: float = OVERLAP_TOLERANCE) -> bool:
        a, b = self.corners(), other.corners()
        for ax in (self.u, _perp(self.u), other.u, _perp(other.u)):
            pa, pb = a @ ax, b @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
        return True


def _box(table: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(table, dict):
        return None
    lo = table.get("min_m", table.get("min"))
    hi = table.get("max_m", table.get("max"))
    if lo is None or hi is None:
        return None
    return _vec(lo)[:2], _vec(hi)[:2]


class Geometry:
    def __init__(self, L, W, gap, margin, box, footprints, crosswise):
        self.L, self.W, self.gap, self.margin = L, W, gap, margin
        self.box, self.rects, self.crosswise = box, list(footprints), crosswise

    def fits(self, r: Rect, extra: list[Rect] = ()) -> bool:
        if self.box is not None:
            c = r.corners()
            lo, hi = self.box
            if not (np.all(c >= lo + self.margin - 1e-9) and np.all(c <= hi - self.margin + 1e-9)):
                return False
        return not any(r.overlaps(o) for o in self.rects + list(extra))

    def candidates(self, end: dict[str, Any], double: bool) -> list[dict[str, Any]]:
        """Straight on (crosswise for a double), then round a corner, left and right."""
        p2, d, z = end["p"][:2], end["d"], float(end["p"][2])
        L, W, g = self.L, self.W, self.gap
        left = _perp(d)
        out = []
        if double and self.crosswise:
            c = p2 + d * (g + W / 2)
            out.append(dict(layout="crosswise", c=c, axis=left, new_p=p2 + d * (g + W), new_d=d))
        else:
            c = p2 + d * (g + L / 2)
            out.append(dict(layout="straight", c=c, axis=d, new_p=p2 + d * (g + L), new_d=d))
        for name, n in (("corner_left", left), ("corner_right", -left)):
            c = p2 + d * (g + W / 2) + n * (L / 2 - W / 2)
            out.append(dict(layout=name, c=c, axis=n, new_p=c + n * L / 2, new_d=n))
        for cand in out:
            cand["z"] = z
            cand["cross"] = cand["layout"] == "crosswise"
            cand["rect"] = Rect(cand["c"], cand["axis"], L / 2, W / 2)
        return out

    def room_after(self, cand: dict[str, Any]) -> bool:
        nxt = {"p": np.r_[cand["new_p"], cand["z"]], "d": cand["new_d"]}
        return any(self.fits(c["rect"], [cand["rect"]]) for c in self.candidates(nxt, False))

    def run_length(self, cand: dict[str, Any]) -> int:
        p, d = np.asarray(cand["new_p"], float), cand["new_d"]
        extra = [cand["rect"]]
        for k in range(RUN_LOOKAHEAD):
            c = p + d * (self.gap + self.L / 2 + k * (self.gap + self.L))
            r = Rect(c, d, self.L / 2, self.W / 2)
            if not self.fits(r, extra):
                return k
            extra.append(r)
        return RUN_LOOKAHEAD

    def options(self, end: dict[str, Any], double: bool, last: bool) -> list[dict[str, Any]]:
        """The placements that fit, best first: straight on when it leaves room for the next
        tile, then the corners that do, the longer straight run after them first (toward
        the table centre on a tie), then the rest. The last tile needs no room after it."""
        cands = self.candidates(end, double)
        if self.box is not None:
            centre = (self.box[0] + self.box[1]) / 2
            if float(_perp(end["d"]) @ (centre - end["p"][:2])) < 0:
                cands = [cands[0], cands[2], cands[1]]
        fitting = [c for c in cands if self.fits(c["rect"])]
        if last:
            return fitting
        roomy = [c for c in fitting if self.room_after(c)]
        first = [roomy.pop(0)] if roomy and roomy[0] is cands[0] else []
        roomy.sort(key=self.run_length, reverse=True)
        return first + roomy + [c for c in fitting if c not in first and c not in roomy]

    def lay(
        self, end: dict[str, Any], doubles: list[bool], tries: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Placements for the whole chain, searched depth first in the options' order (the
        first leaf is the greedy layout) until every tile fits or `tries` placements were
        tried; the deepest layout found otherwise. Also the placements tried."""
        best: list[dict[str, Any]] = []
        path: list[dict[str, Any]] = []
        left = [tries]

        def dfs(e: dict[str, Any]) -> bool:
            k = len(path)
            if k > len(best):
                best[:] = path
            if k == len(doubles):
                return True
            for cand in self.options(e, doubles[k], k == len(doubles) - 1):
                if left[0] <= 0:
                    return False
                left[0] -= 1
                path.append(cand)
                self.rects.append(cand["rect"])
                nxt = {"p": np.r_[cand["new_p"], cand["z"]], "d": cand["new_d"]}
                if dfs(nxt):
                    return True
                path.pop()
                self.rects.pop()
            return False

        base = len(self.rects)
        dfs(end)
        del self.rects[base:]
        return best, tries - left[0]


def _quat_yaw(yaw: float) -> dict[str, float]:
    return {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)}


def _l3(v: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in v]


# ----------------------------------------------------------------------------- inputs


def _hand(hand: Any, max_pip: int) -> list[Tile]:
    out = []
    for i, rec in enumerate(hand or []):
        ab = (
            rec.get("value_ab") or rec.get("value") or rec.get("tile")
            if isinstance(rec, dict)
            else rec
        )
        if ab is None or len(ab) != 2:
            raise ValueError(f"hand tile {i} has no two values: {rec!r}")
        a, b = int(ab[0]), int(ab[1])
        if not (0 <= a <= max_pip and 0 <= b <= max_pip):
            raise ValueError(f"hand tile {i} {list(ab)} is outside 0..{max_pip}")
        out.append((a, b))
    keys = [tuple(sorted(t)) for t in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"the hand holds a tile twice: {[list(t) for t in out]}")
    return out


def _footprints(items: Any, L: float, W: float) -> list[Rect]:
    rects = []
    for f in items or []:
        c = f.get("center", f.get("center_world", f.get("position")))
        u = f.get("axis", f.get("axis_world", f.get("direction")))
        size = f.get("size") or [f.get("length", L), f.get("width", W)]
        rects.append(Rect(_vec(c), _unit2(u), float(size[0]) / 2, float(size[1]) / 2))
    return rects


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    hand: list[Any],
    start_value: int | None = None,
    start_end: dict[str, Any] | None = None,
    objective: str = "longest",
    max_pip: int = MAX_PIP,
    max_nodes: int = MAX_NODES,
    doubles_crosswise: bool = True,
    footprints: list[Any] | None = None,
    table: dict[str, Any] | None = None,
    tile_size: list[float] | None = None,
    touch_gap: float = TOUCH_GAP,
    edge_margin: float = EDGE_MARGIN,
    layout_tries: int = LAYOUT_TRIES,
) -> Output:
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "start_value": -1,
        "length": 0,
        "chain_pips": 0,
        "end_value": -1,
        "ends_in_double": False,
        "chain_tiles": [],
        "hand_indices": [],
        "chain": [],
        "leftover": [],
        "leftover_pips": 0,
        "placed_count": 0,
        "new_end": {},
        "footprints_after": [],
        "nodes": 0,
        "exhaustive": True,
        "length_bound": 0,
        "layout_tries": 0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    try:
        tiles = _hand(hand, int(max_pip))
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, not {objective!r}")
        if start_value is None and isinstance(start_end, dict) and "value" in start_end:
            start_value = start_end["value"]
        sv = None if start_value is None else int(start_value)
        if sv is not None and not 0 <= sv <= int(max_pip):
            raise ValueError(f"start_value {sv} is outside 0..{max_pip}")
        L, W = (float(tile_size[0]), float(tile_size[1])) if tile_size else TILE_SIZE
        end0 = None
        if isinstance(start_end, dict) and (
            start_end.get("position") is not None or start_end.get("point") is not None
        ):
            end0 = {
                "p": _vec(start_end.get("position", start_end.get("point"))),
                "d": _unit2(start_end.get("direction", start_end.get("outward"))),
            }
        rects = _footprints(footprints, L, W)
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", str(exc))

    out["leftover"] = [list(t) for t in tiles]
    out["leftover_pips"] = sum(a + b for a, b in tiles)
    if not tiles:
        return finish("no_chain", "the hand is empty")
    steps, nodes, exhaustive, s, bound = best_chain(tiles, sv, objective, int(max_nodes))
    out["nodes"], out["exhaustive"], out["length_bound"] = nodes, exhaustive, bound
    if not steps:
        return finish("no_chain", f"no tile in the hand holds {sv}")

    used = {i for i, _, _ in steps}
    last = tiles[steps[-1][0]]
    out.update(
        start_value=int(s),
        length=len(steps),
        chain_pips=sum(near + far for _, near, far in steps),
        end_value=int(steps[-1][2]),
        ends_in_double=last[0] == last[1],
        chain_tiles=[[near, far] for _, near, far in steps],
        hand_indices=[i for i, _, _ in steps],
        leftover=[list(t) for k, t in enumerate(tiles) if k not in used],
    )
    out["leftover_pips"] = sum(a + b for a, b in out["leftover"])
    out["chain"] = [
        {"tile": list(tiles[i]), "hand_index": i, "near": near, "far": far}
        for i, near, far in steps
    ]
    notes = []
    if not exhaustive:
        proven = (
            "its length is proven longest" if len(steps) >= bound else f"at most {bound} possible"
        )
        notes.append(
            f"search stopped at {nodes} nodes: the best chain found ({proven}), not proven best"
        )
    summary = (
        f"{len(steps)} tile(s) from {s}, {out['chain_pips']} pips, open end {out['end_value']}"
        + (" (a double)" if out["ends_in_double"] else "")
        + f"; {len(out['leftover'])} left over"
    )

    if end0 is None:
        notes.append("no start_end position: no placement computed")
        return finish("chain_found", "; ".join([summary] + notes))

    box = _box(table)
    if box is None and table is None:
        try:
            ws = ctx.tool("robot.describe_workspace") or {}
            box = _box(ws.get("bounds"))
        except Exception as exc:  # noqa: BLE001 - a graph without a workspace tool
            notes.append(f"no workspace ({exc})")
    if box is None:
        notes.append("no table box: placements not checked against the table edge")
    geo = Geometry(L, W, float(touch_gap), float(edge_margin), box, rects, bool(doubles_crosswise))

    end = {"p": end0["p"], "d": end0["d"], "cross": bool((start_end or {}).get("crosswise", False))}
    doubles = [tiles[i][0] == tiles[i][1] for i, _, _ in steps]
    layout, out["layout_tries"] = geo.lay(end, doubles, int(layout_tries))
    for k, cand in enumerate(layout):
        i, near, _ = steps[k]
        a, b = tiles[i]
        c2, axis = cand["c"], cand["axis"]
        near_pt, far_pt = c2 - axis * L / 2, c2 + axis * L / 2
        a_end, b_end = (near_pt, far_pt) if a == near else (far_pt, near_pt)
        yaw = math.atan2(float(b_end[1] - a_end[1]), float(b_end[0] - a_end[0]))
        z = float(cand["z"])
        out["chain"][k].update(
            layout=cand["layout"],
            place_pose={
                "position": {"x": float(c2[0]), "y": float(c2[1]), "z": z},
                "rotation": _quat_yaw(yaw),
            },
            place_yaw_deg=round(math.degrees(yaw), 3),
            target_ends_world=[_l3(np.r_[a_end, z]), _l3(np.r_[b_end, z])],
        )
        out["footprints_after"].append(
            {"center": _l3(np.r_[c2, z]), "axis": _l3(np.r_[axis, 0.0]), "length": L, "width": W}
        )
        end = {"p": np.r_[cand["new_p"], z], "d": cand["new_d"], "cross": cand["cross"]}
    k = len(layout)
    out["placed_count"] = k
    out["new_end"] = {
        "value": int(steps[k - 1][2]) if k else int(steps[0][1]),
        "position": _l3(end["p"]),
        "direction": _l3(end["d"]),
        "crosswise": end["cross"],
    }
    if k < len(steps):
        a, b = tiles[steps[k][0]]
        return finish(
            "no_room",
            f"{summary}; tile {k} {[a, b]} has no placement inside the table and clear of the "
            f"placed tiles (layouts tried up to {int(layout_tries)} placements): the first {k} are placed",
        )
    return finish("chain_found", "; ".join([summary] + notes))
