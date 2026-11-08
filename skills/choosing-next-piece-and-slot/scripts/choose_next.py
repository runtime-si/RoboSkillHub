"""Choose the next jigsaw piece, cell and turn to place, and a short queue per arm.

Written for the hub (no code copied). The loose pieces are assigned to the free cells with
SciPy's ``scipy.optimize.linear_sum_assignment`` (BSD-3-Clause, used as a library) on
``-log P(piece belongs in cell)``, so no two pieces claim one cell and a piece goes where it
fits best against its rivals; the next placement is then the assigned pair that ranks first
by the stage plan (frame before interior, cells with the most placed neighbours, then
confidence), reachable by a free arm. P comes from localizing-a-piece-in-a-reference-image's
score and margin through the same calibration as backtracking-a-wrong-placement, whose ledger
(forbidden pairs, deferred cells, posteriors) is honoured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.optimize import linear_sum_assignment

CONF_BIAS, CONF_SCORE, CONF_MARGIN = -16.73, 16.01, 51.45
"""P(a piece belongs in a cell) = logistic(bias + a * score + b * min(margin, MARGIN_CAP)),
backtracking-a-wrong-placement's calibration (ridge logistic regression on 7864 (piece,
listed cell) pairs from two runs of localizing-a-piece-in-a-reference-image; its
eval/calibrate.py). Kept identical so the two skills rank the same pairs alike (a test
checks it)."""

MARGIN_CAP = 0.15
"""backtracking-a-wrong-placement's: margins above this say nothing more."""

CONF_MIN, CONF_MAX = 0.02, 0.98
"""backtracking-a-wrong-placement's clip: no localization is certain or impossible."""

FLAT_PROB_FLOOR = 0.015
"""A low-texture piece's shortlist cells (score None) share its probability equally, at least
this each (backtracking-a-wrong-placement's)."""

ATTEMPT_PROB_FACTOR = 0.5
"""Each failure of a pair that nothing explained (the ledger's ``unexplained``) halves its
probability, as in backtracking-a-wrong-placement's re-solve."""

MIN_CONFIDENCE = 0.5
"""A pair is placed on appearance alone only at P >= this. Calibrated pairs at 0.5-0.9 were
right 89 % of the time, under 0.5 at most 21 % (backtracking-a-wrong-placement's calibration
runs). The assembly benchmark: 0.3 and 0.7 changed pieces placed per failure by under 10 %
(see SKILL.md)."""

MIN_ASSIGN_PROB = 0.01
"""Pairs under this stay out of the assignment (backtracking-a-wrong-placement's); a piece
whose cells are all under it stays loose. A 66-cell flat shortlist gives 0.015 per cell."""

NEIGHBOUR_WEIGHT = 1.0
"""Nats of -log P one placed neighbour is worth when ranking pairs of the same tier: a cell
with two more neighbours wins over one whose piece is up to e^2 = 7.4 times likelier to be
wrong. Chosen on the assembly benchmark (0.5, 1, 2: see SKILL.md); more neighbours mean more
seams that catch a wrong piece and guide the seat."""

TIER_COST = 100.0
"""Ranking offset of a new island (a cell with no placed neighbour while the board has a
frontier): any frontier pair ranks first."""

FRAME_WEIGHT = 100.0
"""Nats a frame cell gains while the frame is unfinished. 100 makes the frame a strict first
stage; the assembly benchmark compared 0, 1, 3 and 100 (see SKILL.md)."""

SHAPE_MIN_NEIGHBOURS = 1
"""matching-a-piece-to-placed-neighbours needs at least one placed neighbour (or a border
side) at the slot; it is right 78-91 % top-1 with one neighbour (wavy cut, 10-50
candidates), 89-100 % with two (its SKILL.md)."""

SHAPE_MAX_CANDIDATES = 50
"""Pieces offered to the shape match for one slot, likeliest first: the most that
matching-a-piece-to-placed-neighbours was measured with (top-1 with 2-3 neighbours 91-94 %
wavy, 85-99 % classic at 50; 89-99 % at 10). A flat band's pieces all list all its cells
(51 pieces x 60 cells on the benchmark picture), so a cap of 10 often left the true piece out
(benchmark: 33 % of shape matches right with 10, see SKILL.md)."""

ARM_CLEARANCE = 0.10
"""Metres a free arm's pick, path and place keep from a busy arm's working point, and the two
arms' simultaneous jobs from each other: a YAM wrist with a D405 is about 8 cm across (an
assumption, not measured)."""

PAIR_SEPARATION_CELLS = 3
"""Two jobs run at the same time by different arms need place cells at least this many cells
apart (Chebyshev): the fingertips and the wrist camera of one arm stay out of the other's
cell and its neighbours."""

PATH_STEP = 0.01
"""Metres between checked points on a pick-to-place path."""

QUEUE_LENGTH = 3
"""Jobs per arm in the returned queue; the plan should be redone after every placement (one
call takes milliseconds)."""

POLICIES = ("plan", "frame_first", "most_neighbours", "confidence", "random")


class Output(TypedDict, total=False):
    route: str
    piece: str
    col: int
    row: int
    angle_deg: float
    prob: float
    n_neighbours: int
    cell_class: str
    tier: int
    arm: str
    queue: dict[str, list[dict[str, Any]]]
    slot: list[int]
    shape_candidates: list[str]
    relocalize: list[str]
    unreachable: list[dict[str, Any]]
    n_free: int
    n_placed: int
    n_jobs: int
    frame_done: bool
    reopened: bool
    detail: str


# --- probabilities ---------------------------------------------------------------------------


def calibrated(score: float, margin: float) -> float:
    z = CONF_BIAS + CONF_SCORE * score + CONF_MARGIN * min(margin, MARGIN_CAP)
    return min(CONF_MAX, max(CONF_MIN, 1.0 / (1.0 + math.exp(-z))))


def cell_probs(record: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """P(piece belongs in cell) and its angle for every cell its record lists. A candidate with
    ``confidence`` (a shape match's verdict, newer than the picture match) sets its cell's P;
    scored candidates are calibrated from score and margin; unscored ones (a flat shortlist)
    share the piece equally."""
    cands = [c for c in record.get("candidates") or [] if "col" in c and "row" in c]
    out: dict[tuple[int, int], dict[str, Any]] = {}

    def put(cell, p, c):
        if cell not in out or p > out[cell]["p"]:
            out[cell] = {
                "p": p,
                "angle_deg": c.get("angle_deg"),
                "score": c.get("score"),
                "shape": c.get("confidence") is not None,
            }

    scored = [c for c in cands if c.get("score") is not None and c.get("confidence") is None]
    flat = [c for c in cands if c.get("score") is None and c.get("confidence") is None]
    for c in cands:
        if c.get("confidence") is not None:
            p = min(CONF_MAX, max(CONF_MIN, float(c["confidence"])))
            put((int(c["col"]), int(c["row"])), p, c)
    shaped = set(out)
    for k, c in enumerate(scored):
        others = [o["score"] for j, o in enumerate(scored) if j != k]
        margin = c["score"] - max(others) if others else MARGIN_CAP
        cell = (int(c["col"]), int(c["row"]))
        if cell not in shaped:
            put(cell, calibrated(float(c["score"]), margin), c)
    for c in flat:
        p = max(FLAT_PROB_FLOOR, min(CONF_MAX, 1.0 / len(flat)))
        cell = (int(c["col"]), int(c["row"]))
        if cell not in out:
            put(cell, p, c)
    return out


# --- the board -------------------------------------------------------------------------------

_STEPS = ((0, -1), (0, 1), (1, 0), (-1, 0))


def cell_class(col: int, row: int, cols: int, rows: int) -> str:
    on_x = col in (0, cols - 1)
    on_y = row in (0, rows - 1)
    if on_x and on_y:
        return "corner"
    return "edge" if on_x or on_y else "interior"


def _class_ok(piece_class: str | None, cls: str, cols: int, rows: int) -> bool:
    if piece_class not in ("corner", "edge", "interior"):
        return True
    if cols == 1 or rows == 1:
        return piece_class != "interior" or cls == "interior"
    return piece_class == cls


def _xy(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, dict):
        if "position" in v:
            return _xy(v["position"])
        if "x" in v and "y" in v:
            return np.array([float(v["x"]), float(v["y"])])
        return None
    a = np.asarray(v, dtype=float).ravel()
    return a[:2] if a.size >= 2 else None


class _Geometry:
    """Cell centres in the world (table) frame. ``board_pose``: ``origin`` = world x, y of the
    picture's top-left corner, ``yaw`` = world angle (rad) of the direction the columns count
    along; rows count down the picture, at yaw - 90 degrees."""

    def __init__(self, cols, rows, board_width, board_height, board_pose):
        self.ok = bool(board_pose) and board_width and board_height
        if not self.ok:
            return
        self.cw, self.ch = float(board_width) / cols, float(board_height) / rows
        self.o = _xy(board_pose.get("origin"))
        yaw = float(board_pose.get("yaw", 0.0))
        self.ux = np.array([math.cos(yaw), math.sin(yaw)])
        self.uy = np.array([math.sin(yaw), -math.cos(yaw)])

    def centre(self, cell):
        return self.o + (cell[0] + 0.5) * self.cw * self.ux + (cell[1] + 0.5) * self.ch * self.uy


class _Arm:
    def __init__(self, spec: dict[str, Any]):
        self.id = str(spec.get("id", "arm"))
        reach = spec.get("reach") or {}
        self.poly = np.asarray(reach["polygon"], float) if reach.get("polygon") else None
        self.base = _xy(reach.get("base"))
        self.rmax = float(reach.get("max_radius", math.inf))
        self.rmin = float(reach.get("min_radius", 0.0))
        self.busy = bool(spec.get("busy", False))
        self.busy_at = _xy(spec.get("busy_at"))
        self.home = _xy(spec.get("home")) if spec.get("home") is not None else self.base

    def reaches(self, p: np.ndarray) -> bool:
        if self.poly is not None and not _in_polygon(p, self.poly):
            return False
        if self.base is not None:
            d = float(np.linalg.norm(p - self.base))
            if d > self.rmax or d < self.rmin:
                return False
        return True


def _in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y, inside = p[0], p[1], False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _path(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / PATH_STEP)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a + t * (b - a)


def _feasible(arm: _Arm, others: list[_Arm], pick, place) -> str:
    """'' if the arm can do the job, else why not. The whole pick-to-place path stays in the
    arm's own reach (so it never crosses into the zone only the other arm reaches) and clear
    of every busy arm's working point."""
    if pick is None or place is None:
        return ""
    pts = _path(pick, place)
    if not all(arm.reaches(p) for p in pts):
        return "out_of_reach"
    for o in others:
        if o.busy and o.busy_at is not None:
            if float(np.min(np.linalg.norm(pts - o.busy_at, axis=1))) < ARM_CLEARANCE:
                return f"near_busy_{o.id}"
    return ""


# --- the ledger (backtracking-a-wrong-placement's, read only) --------------------------------


def _load_ledger(ledger: dict | None, path: str | None, cols: int, rows: int) -> dict[str, Any]:
    if ledger is None and path and Path(path).exists():
        ledger = json.loads(Path(path).read_text())
    ledger = dict(ledger or {})
    if ledger and list(ledger.get("grid", [cols, rows])) != [cols, rows]:
        raise ValueError(f"ledger grid {ledger['grid']} is not {cols} x {rows}")
    ledger.setdefault("forbidden", [])
    ledger.setdefault("deferred_cells", [])
    ledger.setdefault("p_right", {})
    ledger.setdefault("unexplained", {})
    return ledger


def _key(piece: str, cell: tuple[int, int]) -> str:
    return f"{piece}@{cell[0]},{cell[1]}"


# --- the node --------------------------------------------------------------------------------


def run(
    ctx: NodeContext,
    grid_cols: int,
    grid_rows: int,
    placements: list[dict] | None = None,
    pieces: list[dict] | None = None,
    ledger: dict | None = None,
    ledger_path: str | None = None,
    arms: list[dict] | None = None,
    board_pose: dict | None = None,
    board_width: float | None = None,
    board_height: float | None = None,
    exclude_cells: list[list[int]] | None = None,
    policy: str = "plan",
    queue_length: int = QUEUE_LENGTH,
    min_confidence: float = MIN_CONFIDENCE,
    neighbour_weight: float = NEIGHBOUR_WEIGHT,
    frame_weight: float = FRAME_WEIGHT,
    max_relocalize: int = 1,
    allow_islands: bool = True,
    reopen_deferred: bool = True,
    all_pieces_known: bool | None = None,
    seed: int = 0,
) -> Output:
    """Pick the next (piece, cell, angle) to place and a short queue per free arm.

    ``placements``: the board as believed, ``[{"piece", "col", "row"}]``. ``pieces``: one record
    per known loose piece, ``{"piece", "candidates": [{"col", "row", "score", "angle_deg",
    "confidence"?}], "route"?, "piece_class"?, "position"? | "piece_pose"?, "relocalized"?}``
    (localizing-a-piece-in-a-reference-image's output; ``piece_class`` corner / edge /
    interior from matching-a-piece-to-placed-neighbours; a candidate's ``confidence`` is a
    shape match's verdict). ``ledger`` / ``ledger_path``: backtracking-a-wrong-placement's
    ledger (forbidden pairs, deferred cells and posteriors are honoured; never written).
    ``arms``: ``[{"id", "reach": {"polygon"?, "base"?, "max_radius"?, "min_radius"?},
    "busy"?, "busy_at"?}]`` with ``board_pose`` (``origin`` of the picture's top-left corner,
    ``yaw`` of the column direction, world metres) and the board size; pieces then need a
    world ``position``. ``exclude_cells``: cells to leave alone this call (a shape match found
    nothing). ``policy``: ``plan`` (default) or a baseline for comparison. Cells the ledger
    deferred are left out until nothing else can be done; then (``reopen_deferred``) they are
    offered again, for pieces not yet forbidden there. ``all_pieces_known`` (default: at least
    as many known loose pieces as free cells) lets the shape match be offered orphans."""
    cols, rows = int(grid_cols), int(grid_rows)
    if cols <= 0 or rows <= 0:
        raise ValueError("grid_cols and grid_rows must be positive")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; one of {POLICIES}")
    book = _load_ledger(ledger, ledger_path, cols, rows)

    board: dict[tuple[int, int], str] = {}
    for pl in placements or []:
        cell = (int(pl["col"]), int(pl["row"]))
        if not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
            raise ValueError(f"placement {pl['piece']} at {list(cell)} is outside the grid")
        board[cell] = str(pl["piece"])
    deferred = {tuple(int(v) for v in c) for c in book["deferred_cells"]}
    skip = {tuple(int(v) for v in c) for c in exclude_cells or []}
    all_cells = [(c, r) for r in range(rows) for c in range(cols)]
    free = [x for x in all_cells if x not in board and x not in deferred]
    border = {x for x in all_cells if cell_class(*x, cols, rows) != "interior"}
    frame_done = all(x in board or x in deferred for x in border)
    base: Output = {
        "n_free": len(free),
        "n_placed": len(board),
        "frame_done": frame_done,
        "queue": {},
    }
    waiting = [x for x in all_cells if x in deferred and x not in board]

    def reopen(why: str) -> Output | None:
        if not (reopen_deferred and waiting):
            return None
        again = dict(book, deferred_cells=[])
        out = run(
            ctx,
            cols,
            rows,
            placements=placements,
            pieces=pieces,
            ledger=again,
            arms=arms,
            board_pose=board_pose,
            board_width=board_width,
            board_height=board_height,
            exclude_cells=exclude_cells,
            policy=policy,
            queue_length=queue_length,
            min_confidence=min_confidence,
            neighbour_weight=neighbour_weight,
            frame_weight=frame_weight,
            max_relocalize=max_relocalize,
            allow_islands=allow_islands,
            reopen_deferred=False,
            all_pieces_known=all_pieces_known,
            seed=seed,
        )
        if out["route"] in ("chosen", "need_shape_match"):
            out["detail"] = f"{why}; reopened {len(waiting)} deferred cells: " + out["detail"]
            out["reopened"] = True
            return out
        return None

    if not free:
        if not waiting:
            return dict(base, route="done", detail=f"every cell is placed ({len(board)})")
        out = reopen("only deferred cells are left")
        if out is not None:
            return out
        return dict(
            base,
            route="need_more_pieces",
            detail=f"only {len(waiting)} deferred cells are left and no known piece is "
            "admissible for them: find more pieces or hand over",
        )

    def nbs(cell):
        return sum((cell[0] + dc, cell[1] + dr) in board for dc, dr in _STEPS)

    on_board = set(board.values())
    recs = [r for r in pieces or [] if str(r["piece"]) not in on_board]
    banned = {(str(f["piece"]), int(f["col"]), int(f["row"])) for f in book["forbidden"]}
    open_cells = [x for x in free if x not in skip]
    col_of = {x: j for j, x in enumerate(open_cells)}

    # Every admissible (piece, cell) pair with its probability.
    probs: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for r in recs:
        pid = str(r["piece"])
        pc = r.get("piece_class")
        ok = {}
        for cell, info in cell_probs(r).items():
            if cell not in col_of or (pid, *cell) in banned:
                continue
            if not _class_ok(pc, cell_class(*cell, cols, rows), cols, rows):
                continue
            k = _key(pid, cell)
            n = int(book["unexplained"].get(k, 0))
            # A shape match's verdict is newer than the ledger's posterior for the pair.
            p0 = info["p"] if info.get("shape") else float(book["p_right"].get(k, info["p"]))
            p = p0 * ATTEMPT_PROB_FACTOR**n
            ok[cell] = dict(info, p=p)
        probs[pid] = ok

    # One cell per piece and one piece per cell: max product of P, a piece may stay loose.
    pool = [str(r["piece"]) for r in recs]
    assignment: dict[str, tuple[int, int]] = {}
    if pool and open_cells:
        big = 1e6
        cost = np.full((len(pool), len(open_cells) + len(pool)), big)
        for i, pid in enumerate(pool):
            cost[i, len(open_cells) + i] = -math.log(MIN_ASSIGN_PROB)
            for cell, info in probs[pid].items():
                if info["p"] >= MIN_ASSIGN_PROB:
                    cost[i, col_of[cell]] = -math.log(info["p"])
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci, strict=True):
            if j < len(open_cells) and cost[i, j] < big:
                assignment[pool[i]] = open_cells[j]

    # Rank the confident assigned pairs by the policy.
    rng = np.random.default_rng(seed)
    any_frontier = any(nbs(x) > 0 for x in open_cells)
    rec_of = {str(r["piece"]): r for r in recs}
    jobs = []
    for pid, cell in assignment.items():
        info = probs[pid][cell]
        if info["p"] < min_confidence:
            continue
        n = nbs(cell)
        cls = cell_class(*cell, cols, rows)
        in_frame = cls != "interior"
        if policy == "plan":
            if n > 0 or not board or not any_frontier:
                tier = 0
            elif allow_islands:
                tier = 1
            else:
                continue
            key = TIER_COST * tier - math.log(info["p"]) - neighbour_weight * n
            if in_frame and not frame_done:
                key -= frame_weight
            if not board and cls == "corner":
                key -= neighbour_weight * 2
        elif policy == "frame_first":
            tier = 0 if (in_frame or frame_done) else 1
            key = TIER_COST * tier - math.log(info["p"])
        elif policy == "most_neighbours":
            tier = 0
            key = -TIER_COST * n - math.log(info["p"])
        elif policy == "confidence":
            tier = 0
            key = -math.log(info["p"])
        else:
            tier = 0
            key = float(rng.random())
        jobs.append(
            {
                "piece": pid,
                "col": cell[0],
                "row": cell[1],
                "angle_deg": info.get("angle_deg"),
                "prob": round(info["p"], 4),
                "n_neighbours": n,
                "cell_class": cls,
                "tier": tier,
                "key": key,
            }
        )
    jobs.sort(key=lambda j: (j["key"], j["piece"]))

    geo = _Geometry(cols, rows, board_width, board_height, board_pose)
    arm_list = [_Arm(a) for a in arms or []]
    if arm_list and not geo.ok:
        raise ValueError("arms need board_pose, board_width and board_height")
    free_arms = [a for a in arm_list if not a.busy]
    base["n_jobs"] = len(jobs)

    def strip(j, arm=None):
        out = {k: v for k, v in j.items() if k != "key"}
        if arm is not None:
            out["arm"] = arm.id
        return out

    unreachable: list[dict[str, Any]] = []
    if jobs and arm_list:
        if not free_arms:
            return dict(base, route="unreachable", unreachable=[], detail="every arm is busy")
        queues: dict[str, list[dict]] = {a.id: [] for a in free_arms}
        heads: list[dict] = []
        qlen = max(1, int(queue_length))
        for j in jobs:
            pick = _xy(rec_of[j["piece"]].get("position") or rec_of[j["piece"]].get("piece_pose"))
            place = geo.centre((j["col"], j["row"]))
            why = {}
            for a in free_arms:
                if pick is None:
                    why[a.id] = "no_position"
                    continue
                others = [o for o in arm_list if o is not a]
                why[a.id] = _feasible(a, others, pick, place)
            ok_arms = [a for a in free_arms if why[a.id] == ""]
            if not ok_arms:
                unreachable.append(dict(strip(j), why=why))
                continue
            ok_arms.sort(key=lambda a: (len(queues[a.id]), _dist(a.home, pick)))
            for a in ok_arms:
                q = queues[a.id]
                if len(q) >= qlen:
                    continue
                if not _clear_of_peers(j, len(q), queues, a.id, pick):
                    continue
                q.append(dict(strip(j, a), _pick=pick))
                if not heads:
                    heads.append(q[-1])
                break
            if all(len(q) >= qlen for q in queues.values()):
                break
        for q in queues.values():
            for x in q:
                x.pop("_pick", None)
        base["queue"] = queues
        base["unreachable"] = unreachable[:10]
        if heads:
            c = heads[0]
            return dict(
                base,
                route="chosen",
                **{k: c[k] for k in ("piece", "col", "row", "angle_deg", "prob")},
                n_neighbours=c["n_neighbours"],
                cell_class=c["cell_class"],
                tier=c["tier"],
                arm=c["arm"],
                detail=_why(c, policy, frame_done),
            )
        return dict(
            base,
            route="unreachable",
            detail=f"{len(unreachable)} confident jobs, none reachable by a free arm without "
            f"crossing the other's zone: move their pieces to a shared buffer",
        )
    if jobs:
        c = jobs[0]
        base["queue"] = {"any": [strip(j) for j in jobs[: max(1, int(queue_length))]]}
        return dict(
            base,
            route="chosen",
            **{k: c[k] for k in ("piece", "col", "row", "angle_deg", "prob")},
            n_neighbours=c["n_neighbours"],
            cell_class=c["cell_class"],
            tier=c["tier"],
            arm="",
            detail=_why(c, policy, frame_done),
        )

    # Nothing confident enough: a shape match at the best-supported slot, then re-localizing,
    # then more pieces. A slot's candidates are the pieces listing it, likeliest first, then -
    # once every piece is known - the orphans (loose pieces with no confident cell at all: a
    # repetitive region's true piece is often not in its own list), class-compatible and not
    # forbidden there.
    all_known = bool(all_pieces_known) if all_pieces_known is not None else len(recs) >= len(free)
    orphans = [
        str(r["piece"])
        for r in recs
        if not any(info["p"] >= min_confidence for info in probs[str(r["piece"])].values())
    ]
    slots = []
    for cell in open_cells:
        n = nbs(cell)
        cls = cell_class(*cell, cols, rows)
        in_frame = cls != "interior"
        if n < SHAPE_MIN_NEIGHBOURS and not (in_frame and not board):
            continue
        listed = sorted(
            (
                (info["p"], pid)
                for pid, ps in probs.items()
                for c2, info in ps.items()
                if c2 == cell
            ),
            key=lambda t: (-t[0], t[1]),
        )
        cands = [pid for _, pid in listed]
        if board and all_known:
            seen = set(cands)
            cands += [
                o
                for o in orphans
                if o not in seen
                and (o, *cell) not in banned
                and _class_ok(rec_of[o].get("piece_class"), cls, cols, rows)
            ]
        if not cands:
            continue
        best_p = listed[0][0] if listed else 0.0
        frame_rank = 0 if (in_frame or frame_done) else 1
        if policy == "plan":
            # Shape needs neighbours more than the frame needs finishing: the shape match is
            # right 78 % of the time with one neighbour and 91-94 % with 2-3 (50 candidates).
            key = (-n, frame_rank, -best_p)
        elif policy == "frame_first":
            key = (frame_rank, -best_p, 0)
        elif policy == "most_neighbours":
            key = (-n, -best_p, 0)
        elif policy == "confidence":
            key = (-best_p, 0, 0)
        else:
            key = (float(rng.random()), 0, 0)
        slots.append((key, cell, n, len(listed), cands))
    if slots:
        slots.sort(key=lambda s: (s[0], s[1][1], s[1][0]))
        _, cell, n, n_listed, cands = slots[0]
        return dict(
            base,
            route="need_shape_match",
            slot=list(cell),
            n_neighbours=n,
            cell_class=cell_class(*cell, cols, rows),
            shape_candidates=cands[:SHAPE_MAX_CANDIDATES],
            detail=f"no pair at P >= {min_confidence}; cell {list(cell)} has {n} placed "
            f"neighbours, {n_listed} loose pieces listing it and {len(cands) - n_listed} "
            f"orphans: run matching-a-piece-to-placed-neighbours on them",
        )
    out = reopen("no open cell can be filled")
    if out is not None:
        return out
    relocalize = [o for o in orphans if int(rec_of[o].get("relocalized", 0)) < max_relocalize]
    if relocalize:
        return dict(
            base,
            route="need_relocalize",
            relocalize=relocalize,
            detail=f"no confident pair for any open cell; {len(relocalize)} loose pieces have no "
            f"admissible cell at P >= {min_confidence}: photograph and localize them again",
        )
    return dict(
        base,
        route="need_more_pieces",
        detail=f"{len(recs)} known loose pieces, none fits any of {len(open_cells)} open cells "
        f"(of {len(free)} free): find and localize more pieces",
    )


def _dist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def _clear_of_peers(job, slot, queues, me, pick) -> bool:
    """A job run alongside the other arms' job at the same queue position must be
    PAIR_SEPARATION_CELLS away from it and keep ARM_CLEARANCE between the picks."""
    for aid, q in queues.items():
        if aid == me or len(q) <= slot:
            continue
        other = q[slot]
        if (
            max(abs(other["col"] - job["col"]), abs(other["row"] - job["row"]))
            < PAIR_SEPARATION_CELLS
        ):
            return False
        if _dist(other["_pick"], pick) < ARM_CLEARANCE:
            return False
    return True


def _why(c, policy, frame_done) -> str:
    phase = "interior" if frame_done else "frame"
    return (
        f"{c['piece']} to {[c['col'], c['row']]} ({c['cell_class']}, {c['n_neighbours']} placed "
        f"neighbours, P {c['prob']:.2f}, tier {c['tier']}); policy {policy}, {phase} phase"
    )
