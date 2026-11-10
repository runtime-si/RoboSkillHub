# Reimplemented from the behaviour described in Manual2Skill++ (Tie et al., ICRA 2026, connector-aware pose
# alignment; https://github.com/crtie/Manual2SkillPP_official has no licence, so no code was copied). For LEGO
# the connector solve reduces to the stud lattice: a statement "part P's tubes on base B's studs, P's stud
# (a, b) over B's stud (i, j), turned r" fixes P's cells exactly. Statements are solved in dependency order on
# integer lattice coordinates; a part named by several statements (a bridge) must come out the same from each.
"""Turn stud-to-tube connector statements (part on base at stud (i, j), turned r) into exact plate cells and poses."""

from __future__ import annotations

import json
import math
from typing import Any, TypedDict

from gap import NodeContext

PRESETS: dict[str, dict[str, float]] = {
    "lego": {"stud_pitch": 0.008, "plate_unit": 0.0032},
    "duplo": {"stud_pitch": 0.016, "plate_unit": 0.0064},
}
"""Stud pitch and plate height [m] (a brick is 3 plate units; computing-brick-pose-from-stud-grid's presets)."""

ROTATIONS = (0, 90, 180, 270)
MIN_SUPPORT = 1
"""A part must sit on at least this many of its base's studs (one stud holds a LEGO brick)."""

ROUTES = ("poses_solved", "constraints_inconsistent")


class Output(TypedDict):
    route: str
    detail: str
    parts_json: str
    steps_json: str
    order_json: str
    problems_json: str
    n_parts: int


def _rot(r: int, u: int, v: int) -> tuple[int, int]:
    return {0: (u, v), 90: (-v, u), 180: (-u, -v), 270: (v, -u)}[r]


def _size(v: Any, name: str) -> tuple[int, int]:
    if isinstance(v, str):
        v = [int(s) for s in v.lower().split("x")]
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in v)
    ):
        raise ValueError(f"{name}: size must be two positive integers or 'AxB', got {v!r}")
    return int(v[0]), int(v[1])


class Placed:
    """A part on the plate lattice: rotation r (deg about +z), and the plate cell of its stud (0, 0)."""

    def __init__(
        self, pid: str, size: tuple[int, int], r: int, origin: tuple[int, int], z: int, h: int
    ):
        self.id, self.size, self.r, self.origin, self.z, self.h = pid, size, r, origin, z, h

    def stud(self, u: int, v: int) -> tuple[int, int]:
        du, dv = _rot(self.r, u, v)
        return self.origin[0] + du, self.origin[1] + dv

    def cells(self) -> set[tuple[int, int]]:
        return {self.stud(u, v) for u in range(self.size[0]) for v in range(self.size[1])}

    def footprint(self) -> tuple[int, int, int, int]:
        c = self.cells()
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1

    def key(self) -> tuple:
        sym = 90 if self.size[0] == self.size[1] else 180
        return (
            frozenset(self.cells()),
            self.z,
            self.r % sym if self.size[0] != self.size[1] else 0,
        )


def run(
    ctx: NodeContext,
    statements: list[dict[str, Any]] | None = None,
    statements_json: str = "",
    parts: list[dict[str, Any]] | None = None,
    plate: dict[str, Any] | None = None,
    system: str = "lego",
    min_support: int = MIN_SUPPORT,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if statements is None:
        statements = json.loads(statements_json) if statements_json else []
    out: Output = {
        "route": "constraints_inconsistent",
        "detail": "",
        "parts_json": "[]",
        "steps_json": "[]",
        "order_json": "[]",
        "problems_json": "[]",
        "n_parts": 0,
    }
    problems: list[dict[str, Any]] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        out["problems_json"] = json.dumps(problems)
        return out

    plate = dict(plate or {})
    plate_studs = _size(plate.get("studs", [32, 32]), "plate")
    plate_id = str(plate.get("id", "plate"))
    # Part sizes and heights (plate units: brick 3, plate 1).
    meta: dict[str, dict[str, Any]] = {}
    for p in parts or []:
        meta[str(p["id"])] = {
            "size": _size(p.get("size", p.get("studs", [2, 4])), f"part {p['id']}"),
            "h": int(p.get("height_units", 3)),
            "extra": {
                k: v for k, v in p.items() if k not in ("id", "size", "studs", "height_units")
            },
        }
    by_part: dict[str, list[dict[str, Any]]] = {}
    for k, s in enumerate(statements):
        try:
            pid, base = str(s["part"]), str(s["on"])
        except KeyError as exc:
            problems.append({"statement": k, "kind": "malformed", "missing": str(exc)})
            continue
        if pid == base:
            problems.append({"statement": k, "kind": "self_reference", "part": pid})
            continue
        if pid not in meta:
            meta[pid] = {
                "size": _size(s.get("size", [2, 4]), f"part {pid}"),
                "h": int(s.get("height_units", 3)),
                "extra": {},
            }
        elif "size" in s and _size(s["size"], f"part {pid}") != meta[pid]["size"]:
            problems.append({"statement": k, "kind": "size_conflict", "part": pid})
        r = int(s.get("rotation", 0)) % 360
        if r not in ROTATIONS:
            problems.append({"statement": k, "kind": "bad_rotation", "part": pid, "rotation": r})
            continue
        at = s.get("at", [0, 0])
        anchor = s.get("part_stud", [0, 0])
        by_part.setdefault(pid, []).append(
            {
                "k": k,
                "base": base,
                "at": (int(at[0]), int(at[1])),
                "anchor": (int(anchor[0]), int(anchor[1])),
                "r": r,
            }
        )
    if problems:
        return finish("constraints_inconsistent", f"{len(problems)} malformed statement(s)")

    placed: dict[str, Placed] = {plate_id: Placed(plate_id, plate_studs, 0, (0, 0), -1, 1)}
    order: list[str] = []
    pending = dict(by_part)
    while pending:
        ready = [p for p, sts in pending.items() if all(st["base"] in placed for st in sts)]
        if not ready:
            unknown = sorted(
                {
                    st["base"]
                    for sts in pending.values()
                    for st in sts
                    if st["base"] not in placed and st["base"] not in pending
                }
            )
            if unknown:
                problems.append({"kind": "unknown_base", "bases": unknown})
                return finish(
                    "constraints_inconsistent",
                    f"statements name bases nobody places: {', '.join(unknown)}",
                )
            problems.append({"kind": "cycle", "parts": sorted(pending)})
            return finish(
                "constraints_inconsistent",
                f"the statements form a cycle among {', '.join(sorted(pending))}",
            )
        for pid in sorted(ready, key=lambda p: min(st["k"] for st in pending[p])):
            m = meta[pid]
            sols: list[tuple[Placed, dict]] = []
            for st in pending.pop(pid):
                b = placed[st["base"]]
                # P's stud (a, b) goes over B's stud (i, j): P's stud (0, 0) is B-local (i, j) - R_r(a, b).
                da, db = _rot(st["r"], *st["anchor"])
                local0 = (st["at"][0] - da, st["at"][1] - db)
                origin = b.stud(*local0)
                cand = Placed(pid, m["size"], (b.r + st["r"]) % 360, origin, b.z + b.h, m["h"])
                base_cells = b.cells()
                support = len(cand.cells() & base_cells)
                if support < min_support:
                    problems.append(
                        {
                            "statement": st["k"],
                            "kind": "unsupported",
                            "part": pid,
                            "base": st["base"],
                            "studs_on_base": support,
                        }
                    )
                    return finish(
                        "constraints_inconsistent",
                        f"{pid} would sit on {support} of {st['base']}'s studs",
                    )
                sols.append((cand, st))
            first = sols[0][0]
            for cand, st in sols[1:]:
                if cand.key() != first.key():
                    problems.append({"statement": st["k"], "kind": "conflict", "part": pid,
                                     "with_statement": sols[0][1]["k"]})  # fmt: skip
                    return finish(
                        "constraints_inconsistent",
                        f"the statements for {pid} disagree on where it goes",
                    )
            # Interpenetration with parts already placed at overlapping heights.
            for other in placed.values():
                if other.id == plate_id:
                    continue
                if (
                    first.z < other.z + other.h
                    and other.z < first.z + first.h
                    and first.cells() & other.cells()
                ):
                    problems.append({"kind": "collision", "part": pid, "with": other.id})
                    return finish("constraints_inconsistent", f"{pid} would overlap {other.id}")
            x0, y0, sx, sy = first.footprint()
            if x0 < 0 or y0 < 0 or x0 + sx > plate_studs[0] or y0 + sy > plate_studs[1]:
                problems.append(
                    {"kind": "off_plate", "part": pid, "cell": [x0, y0], "span": [sx, sy]}
                )
                return finish(
                    "constraints_inconsistent", f"{pid} would hang off the plate's stud area"
                )
            placed[pid] = first
            order.append(pid)

    p_cfg = PRESETS[system]
    pitch, unit = p_cfg["stud_pitch"], p_cfg["plate_unit"]
    origin = plate.get("origin")
    yaw = math.radians(float(plate.get("yaw_deg", 0.0)))
    parts_out, steps = [], []
    for k, pid in enumerate(order):
        pl = placed[pid]
        x0, y0, sx, sy = pl.footprint()
        rec: dict[str, Any] = {
            "id": pid,
            "size": f"{pl.size[0]}x{pl.size[1]}",
            "rotation": pl.r,
            "cell": {"x": x0, "y": y0, "layer": pl.z // 3 + 1, "rotation": pl.r},
            "span_x": sx,
            "span_y": sy,
            "z_units": pl.z,
            "height_units": pl.h,
            "system": system,
            "kind": "brick" if pl.h == 3 else "plate",
            "step": k + 1,
            **meta[pid]["extra"],
        }
        lx, ly = (x0 + sx / 2.0) * pitch, (y0 + sy / 2.0) * pitch
        if origin is not None:
            o = (
                [float(origin[c]) for c in "xyz"]
                if isinstance(origin, dict)
                else [float(v) for v in origin]
            )
            rec["position"] = {
                "x": o[0] + math.cos(yaw) * lx - math.sin(yaw) * ly,
                "y": o[1] + math.sin(yaw) * lx + math.cos(yaw) * ly,
                "z": o[2] + pl.z * unit,
            }
            rec["yaw_deg"] = (math.degrees(yaw) + pl.r) % 360.0
        parts_out.append(rec)
        steps.append({"step": k + 1, "bricks": [rec]})
    out["parts_json"] = json.dumps(parts_out)
    out["steps_json"] = json.dumps(steps)
    out["order_json"] = json.dumps(order)
    out["n_parts"] = len(order)
    return finish("poses_solved", f"{len(order)} parts placed from {len(statements)} statements")
