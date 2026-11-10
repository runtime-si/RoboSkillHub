"""Choose the next fallen tile of a domino line to rebuild: the downstream end of the first fallen segment, where it lies."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

TILE_SIZE = (0.048, 0.024, 0.0075)
"""Tile height, width, thickness [m]: the audit's and planning-a-domino-layout's default."""

FLAT_DEG = 80.0
"""A fallen tile leaning this far or more lies flat enough to stand up from the table [deg].
Tiles fallen on each other rest at 75-80 deg in a YAM station twin; the downstream end of a
run lies at 90 (nothing ahead holds it), and the next one drops flat once it is taken."""

DOWN = ("fallen", "leaning")


class Output(TypedDict):
    route: str
    detail: str
    tile_index: int
    tile_center: dict
    long_axis: dict
    slot: dict
    segment: list
    direction: str
    remaining: int


def _xyz(p: Any) -> np.ndarray:
    if isinstance(p, dict):
        p = p.get("position", p)
        return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
    v = [float(a) for a in list(p)[:3]]
    return np.array(v + [0.0] * (3 - len(v)))


def _facing(s: Any) -> float:
    if isinstance(s, dict):
        if "facing_deg" in s:
            return float(s["facing_deg"])
        q = (s.get("pose", s)).get("rotation")
        if q is not None:
            w, x, y, z = (float(q[k]) for k in "wxyz")
            return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        raise ValueError(f"slot {s!r} has no facing")
    return float(list(s)[2])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def run(
    ctx: NodeContext,
    slots: list[Any],
    states: list[str],
    lean_deg: list[float | None],
    shift_mm: list[float | None] | None = None,
    table_z: float | None = None,
    tile_size: Any = TILE_SIZE,
    flat_deg: float = FLAT_DEG,
) -> Output:
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "tile_index": -1,
        "tile_center": {},
        "long_axis": {},
        "slot": {},
        "segment": [],
        "direction": "",
        "remaining": 0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    try:
        h, w, d = (
            float(v) for v in (tile_size.values() if isinstance(tile_size, dict) else tile_size)
        )
        if isinstance(tile_size, dict):
            h, w, d = (
                float(tile_size.get("height", h)),
                float(tile_size.get("width", w)),
                float(tile_size.get("thickness", d)),
            )
        n = len(slots or [])
        if n == 0 or len(states) != n or len(lean_deg) != n:
            raise ValueError(
                f"{n} slots, {len(states)} states, {len(lean_deg)} leans: they must match and not be empty"
            )
        centres = [_xyz(s) for s in slots]
        facings = [_facing(s) for s in slots]
        if table_z is None:
            table_z = float(np.median([c[2] for c in centres]))
        shifts = [float(v or 0.0) / 1e3 for v in (shift_mm or [0.0] * n)]
    except (ValueError, KeyError, TypeError) as exc:
        return finish("invalid_input", str(exc))

    down = [i for i, s in enumerate(states) if s in DOWN]
    out["remaining"] = len(down)
    if not down:
        missing = [i for i, s in enumerate(states) if s == "missing"]
        return finish(
            "nothing_fallen",
            "no fallen or leaning tile"
            + (f"; missing slots {missing} (place new tiles there)" if missing else ""),
        )
    # the first segment of fallen / leaning tiles in line order
    a = down[0]
    b = a
    while b + 1 < n and states[b + 1] in DOWN:
        b += 1
    out["segment"] = [a, b]
    leans = [lean_deg[i] for i in range(a, b + 1) if lean_deg[i] is not None]
    forward = (sum(1 for v in leans if v > 0) >= sum(1 for v in leans if v < 0)) if leans else True
    out["direction"] = "forward" if forward else "backward"
    k = b if forward else a  # the downstream end
    if b > a:
        # in a cascade each tile falls onto the next, so the downstream tile lies under its
        # predecessor's top end: a pinch at its centre takes that end instead (station twin:
        # the arm carried tile 3 off tile 4, and the stand-up handover then closed on nothing)
        out["tile_index"] = k
        return finish(
            "pinned",
            f"segment {a}-{b} fell {out['direction']}: tile {k} at its downstream end lies under tile "
            f"{k - 1 if forward else k + 1}'s top; lift the top of the pile ({a if forward else b}) first",
        )
    out["tile_index"] = k
    lean = lean_deg[k]
    f = math.radians(facings[k])
    u = np.array([math.cos(f), math.sin(f), 0.0])
    out["slot"] = {
        "position": _vec(np.array([centres[k][0], centres[k][1], float(table_z)])),
        "rotation": {"w": math.cos(f / 2), "x": 0.0, "y": 0.0, "z": math.sin(f / 2)},
    }
    nxt = k + 1 if forward else k - 1
    if (
        (lean is None or abs(float(lean)) < float(flat_deg))
        and 0 <= nxt < n
        and states[nxt] == "missing"
    ):
        body = lean_deg[nxt]
        if body is not None and abs(float(body)) >= float(flat_deg):
            # the audit reads the flat body across the empty next slot there, and the tile
            # itself short of flat (its window ends at that slot): it is this tile, lying
            lean = math.copysign(abs(float(body)), 1.0 if forward else -1.0)
    if lean is None or abs(float(lean)) < float(flat_deg):
        return finish(
            "not_flat",
            f"the downstream tile {k} of segment {a}-{b} leans {lean} deg, not flat: it rests on something; "
            "take that away first or lay it down",
        )
    # a tile fallen flat about its front (back) bottom edge: its centre lies h/2 beyond that edge
    sign = 1.0 if float(lean) > 0 else -1.0
    edge = np.array([centres[k][0], centres[k][1], float(table_z)]) + u * (sign * d / 2 + shifts[k])
    centre = edge + u * sign * h / 2
    centre[2] = float(table_z) + d / 2
    out["tile_center"] = _vec(centre)
    out["long_axis"] = _vec(u * sign)
    return finish(
        "rebuild_next",
        f"segment {a}-{b} fell {out['direction']}: stand tile {k} (lying {abs(float(lean)):.0f} deg) and put it back at slot {k}; "
        f"{len(down) - 1} more after it",
    )
