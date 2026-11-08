"""A jigsaw board for backtracking tests: placements are checked by the shape of the procedural cut.

Pieces are named ``p{col}_{row}`` after their true cell, in the picture grid of
localizing-a-piece-in-a-reference-image (row 0 at the top). The cut comes from
``jigsaw_cut.make_puzzle`` (row 0 at the south), so picture row r is cut row rows-1-r. A
piece set down in a cell is checked against every placed neighbour: its true side facing the
neighbour and the neighbour's true side facing it are each mapped onto the shared cell edge
and compared (symmetric mean closest-point distance over the edge length). A tab against a
tab (or a knob against a flat side) cannot seat (and its seam scores CLASH_SEAM worse); other
differences only raise the seam score.
"""

from __future__ import annotations

import numpy as np

from . import jigsaw_cut as jc

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
SEAM_FAIL = 0.06
CLASH_SEAM = 0.06
"""Added to the seam score of a seam that cannot seat: the piece rides on the neighbour's knob,
so its outline seen from the wrist is displaced (a knob is 0.2-0.36 of a side high)."""


def pid(col, row):
    return f"p{col}_{row}"


def true_cell(piece):
    col, row = piece[1:].split("_")
    return int(col), int(row)


def _resample(poly, m=60):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], m)
    return np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])


def _onto(poly, a, b):
    """Similarity map of a polyline whose ends are poly[0], poly[-1] onto a -> b."""
    p0, p1 = poly[0], poly[-1]
    u, v = p1 - p0, b - a
    z = complex(*v) / complex(*u)
    q = (poly[:, 0] - p0[0]) + 1j * (poly[:, 1] - p0[1])
    q = q * z
    return np.column_stack([q.real + a[0], q.imag + a[1]])


class Board:
    def __init__(
        self,
        cols,
        rows,
        seed=0,
        style="classic",
        seam_floor=(0.005, 0.03),
        seam_noise=0.004,
        exec_fail=0.0,
    ):
        """``seam_floor``: a right seam still scores uniform(lo, hi) (mask noise, as the
        neighbour-matching skill sees it); ``exec_fail``: chance a right piece is set down
        off its seat anyway."""
        self.cols, self.rows = cols, rows
        self.cut = jc.make_puzzle(cols, rows, seed=seed, style=style)
        self.seam_floor = seam_floor
        self.seam_noise = seam_noise
        self.exec_fail = exec_fail
        self.placed: dict[tuple[int, int], str] = {}

    def _cut_ij(self, col, row):
        return col, self.rows - 1 - row

    def _edge(self, cell, d):
        """The cell's side towards d as (start, end) vertices, walked counter-clockwise."""
        i, j = self._cut_ij(*cell)
        corners = self.cut.corners(i, j)  # SW, SE, NE, NW
        k = {"south": 0, "east": 1, "north": 2, "west": 3}[d]
        return corners[k], corners[(k + 1) % 4]

    def _side(self, piece, d):
        i, j = self._cut_ij(*true_cell(piece))
        return self.cut.side(i, j, d), self.cut.side_type(i, j, d)

    def seam(self, a, cell_a, b, d):
        """Seam score and whether piece a in cell_a and piece b in the cell to its d can seat."""
        sa, ta = self._side(a, d)
        sb, tb = self._side(b, OPPOSITE[d])
        e0, e1 = self._edge(cell_a, d)
        pa = _resample(_onto(sa, e0, e1))
        pb = _resample(_onto(sb[::-1], e0, e1))
        dist = np.linalg.norm(pa[:, None] - pb[None], axis=2)
        score = 0.5 * (dist.min(axis=1).mean() + dist.min(axis=0).mean())
        score /= float(np.linalg.norm(e1 - e0))
        clash = (ta == tb and ta != "flat") or ((ta == "flat") != (tb == "flat"))
        return float(score), clash

    def check(self, piece, cell, rng):
        """``None`` if the piece seats, else a failure dict for backtracking-a-wrong-placement."""
        seams, clash = {}, False
        for d, (dc, dr) in STEP.items():
            nb = (cell[0] + dc, cell[1] + dr)
            if nb in self.placed:
                s, c = self.seam(piece, cell, self.placed[nb], d)
                s += float(rng.uniform(*self.seam_floor)) + float(rng.normal(0, self.seam_noise))
                if c:
                    s += CLASH_SEAM
                seams[self.placed[nb]] = round(max(0.0, s), 4)
                clash |= c
        if not clash and all(s <= SEAM_FAIL for s in seams.values()):
            if self.exec_fail and rng.random() < self.exec_fail:
                return {
                    "piece": piece,
                    "col": cell[0],
                    "row": cell[1],
                    "kind": "not_seated",
                    "seam_scores": {},
                }
            return None
        return {
            "piece": piece,
            "col": cell[0],
            "row": cell[1],
            "kind": "not_seated" if clash else "seam_mismatch",
            "seam_scores": seams,
        }

    def correct(self):
        return all(true_cell(p) == c for c, p in self.placed.items())


def placement_record(piece, cell, records, verified=True, shape_confidence=0.9):
    """What the board believes about a placed piece: its localizer score and margin at the
    cell, or (a cell the localizer did not list: placed by a shape match) a fixed confidence."""
    cands = records.get(piece, {}).get("candidates") or []
    scored = [c for c in cands if c.get("score") is not None]
    rec = {"piece": piece, "col": cell[0], "row": cell[1], "verified": verified}
    for k, c in enumerate(scored):
        if (c["col"], c["row"]) == tuple(cell):
            others = [o["score"] for j, o in enumerate(scored) if j != k]
            rec.update(score=c["score"], margin=c["score"] - max(others) if others else 0.15)
            return rec
    rec["confidence"] = shape_confidence
    return rec


def run_episode(script, board, records, failure, rng, max_calls=6, ledger=None, **kw):
    """Feed a failure to the skill, carry out its actions on the board (stopping at the first
    placement that fails its check), repeat. Returns (calls until the board is right with
    the watched cells filled or None, routes, physical ops)."""
    from gap.testing import FakeContext

    watch = {(failure["col"], failure["row"])} | set(kw.pop("watch", ()))
    meta = {c: placement_record(p, c, records) for c, p in board.placed.items()}
    routes, ops = [], 0
    for call in range(1, max_calls + 1):
        placements = list(meta.values()) + [
            placement_record(failure["piece"], (failure["col"], failure["row"]), records, False)
        ]
        out = script.run(
            FakeContext(),
            failure,
            placements,
            list(records.values()),
            board.cols,
            board.rows,
            ledger=ledger,
            **kw,
        )
        ledger = out["ledger"]
        routes.append(out["route"])
        failure = None
        for a in out["actions"]:
            if a["op"] == "remove":
                cell = (a["col"], a["row"])
                board.placed.pop(cell, None)
                meta.pop(cell, None)
                ops += 1
            elif a["op"] in ("place", "reseat"):
                cell = (a["col"], a["row"])
                ops += 1
                f = board.check(a["piece"], cell, rng)
                if f is not None:
                    failure = f
                    break
                board.placed[cell] = a["piece"]
                meta[cell] = placement_record(a["piece"], cell, records, verified=True)
            elif a["op"] == "defer_cell":
                return None, routes, ops
        if failure is None:
            done = board.correct() and all(c in board.placed for c in watch)
            return (call if done else None), routes, ops
    return None, routes, ops


def synthetic_localization(cols, rows, rng, confusions=(), top_k=5):
    """Localizer-like records: every piece's true cell first with a high score and margin,
    then random other cells; ``confusions`` lists (piece, cell, score, margin) to put a
    wrong cell first instead (the true cell second)."""
    recs = {}
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    for col, row in cells:
        others = [cells[k] for k in rng.choice(len(cells), top_k + 2, replace=False)]
        others = [x for x in others if x != (col, row)][: top_k - 1]
        best = float(rng.uniform(0.93, 0.99))
        second = best - float(rng.uniform(0.3, 0.5))
        cands = [{"col": col, "row": row, "score": best, "angle_deg": 0.0}]
        cands += [
            {"col": x[0], "row": x[1], "score": second - 0.02 * k, "angle_deg": 0.0}
            for k, x in enumerate(others)
        ]
        recs[pid(col, row)] = {"piece": pid(col, row), "candidates": cands}
    for piece, cell, score, margin in confusions:
        rec = recs[piece]
        true = rec["candidates"][0]
        rest = [c for c in rec["candidates"][1:] if (c["col"], c["row"]) != tuple(cell)]
        wrong = {"col": cell[0], "row": cell[1], "score": score, "angle_deg": 0.0}
        true = dict(true, score=score - margin)
        rec["candidates"] = [wrong, true] + rest[: top_k - 2]
    return recs
