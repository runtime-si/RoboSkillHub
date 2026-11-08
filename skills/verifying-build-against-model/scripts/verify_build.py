# Adapted from LTRON (https://github.com/aaronwalsman/ltron) at commit
# 253be3e55cef068c1842fd9a7d2e2b3c22d19d7b: ltron/matching.py (match_assemblies,
# validate_matches, match_lookup), ltron/score.py (f1, score_assemblies,
# edit_distance) and ltron/geometry/symmetry.py (pose_match_under_symmetries);
# MIT License, Copyright 2021 Aaron Walsman (LICENSE.ltron beside this bundle).
# Changes: LTRON's assemblies (LDraw shape id, colour id, 4x4 pose in LDU) become
# stud-grid bricks (sorted size 'AxB', colour name, a pose in pitches and layers
# with a yaw about the plate normal); the match radius is an ellipsoid (lateral
# pitches, vertical layers) instead of a sphere; the rendered symmetry table is
# replaced by a rectangle's 180 deg and a square's 90 deg yaw symmetry, and the
# symmetric anchor poses LTRON left commented out are tried. The offset search
# (rarest shape first, anchor pairs, skip tested pairs, restart on a better
# offset) is kept, but it starts from the identity offset and, by default
# (align='none'), stops there: on a calibrated baseplate a shifted build is wrong.
# Offsets are restricted to turns about the plate normal (and no layer change
# with align='plate'). validate_matches let two observed bricks claim one target
# brick; here the pairs are assigned one to one (scipy linear_sum_assignment,
# nearest first). An unknown colour is a wildcard (LTRON required equal non-zero
# colours). Added: the unmatched bricks are split into wrong colour (same place
# and shape), misplaced (same shape and colour elsewhere), ahead (a later step's
# brick, correctly placed), missing and extra; unseated (a matched brick high by
# about a stud, or at one end); repairs; and a depth path that casts every pixel's
# ray against the target's expected height map (stud-cell column tops and stud
# discs on the plate pose), compares measured and expected heights per stud
# cell, explains them by present / absent target bricks and cuts the rest into
# extra bricks, with colour from tools/colour.
"""Compare an observed brick build with the target model up to the current step: matches, missing, extra, misplaced, wrong-colour and unseated bricks, and an F1 score."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, float]] = {
    "lego": {
        "stud_pitch": 0.008,
        "brick_height": 0.0096,
        "stud_height": 0.0017,
        "stud_diameter": 0.0048,
        "plates_per_brick": 3,
    },
    "duplo": {
        "stud_pitch": 0.016,
        "brick_height": 0.0192,
        "stud_height": 0.0045,
        "stud_diameter": 0.0094,
        "plates_per_brick": 2,
    },
}
"""Brick systems [m], as computing-brick-pose-from-stud-grid and
pressing-a-brick-onto-studs (LEGO: ICL's 8 mm pitch, 9.6 mm brick, 1.7 mm stud;
DUPLO twice the pitch and height, 4.5 mm studs). A LEGO plate is a third of a
brick, a DUPLO plate half (parsing-ldraw-build-instructions' UNITS_PER_BRICK)."""

LATERAL_TOLERANCE = 0.3
"""Match radius along the plate [pitches]: a quarter pitch is where
computing-brick-pose-from-stud-grid calls a pose off its cell; 0.3 lets a pose
read at that limit still match. LTRON's own radius is 1 LDU (0.4 mm) on exact
simulator poses."""

LAYER_TOLERANCE = 0.45
"""Match radius along the plate normal [brick heights]: under half a brick, so
layers stay apart, and well over the unseated height (LEGO 1.7 / 9.6 = 0.18,
DUPLO 4.5 / 19.2 = 0.23) plus the overhead camera's per-brick noise on LEGO
(~0.5 mm), so an unseated brick still matches and is reported as unseated."""

YAW_TOLERANCE_DEG = 10.0
"""Yaw a matched brick may be off its target, modulo its symmetry
(computing-brick-pose-from-stud-grid's off_grid limit). LTRON: 0.08 rad."""

SEAT_TOLERANCE_FRACTION = 0.5
"""A brick is unseated when its top reads more than this fraction of a stud
height over its seated height: half way between seated (0) and resting on the
stud tops (1 stud, 1.7 mm LEGO / 4.5 mm DUPLO; pressing-a-brick-onto-studs)."""

SEAT_SIGMAS = 3.0
"""The reading must stand this many standard errors (from the depth points' own
scatter: 1.4826 MAD x 1.2533 / sqrt(n) for a median) over seated. The brick's
median is tested, and its highest stud cell (a brick seated at one end and
resting on the studs at the other). A brick whose reading cannot exclude
resting on the studs either is seat_unresolved: the overhead camera on LEGO,
~0.5 mm per brick and ~1 mm per cell against a 1.7 mm stud."""

CELL_SIGMAS = 4.0
"""The same test for the highest cell, stricter because it is the largest of
up to a few dozen cells per frame (3 sigma would be passed by chance in ~5 % of
clean frames)."""

EDGE_INSET = 0.0005
STUD_MARGIN = 0.0005
"""Pixels whose ray would hit the expected surface within 0.5 mm of a cell's
edge or of a stud's rim are dropped: walls and mixed pixels at height steps. At
the overhead camera's ~2 mm pixels a LEGO cell keeps ~8-10 points, at the wrist
D405's ~0.3-0.5 mm a few hundred."""

MEASURED_INSET = 0.25
"""Points placed by their own position (far from the expected surface) must lie
this many pitches inside their cell: a wall's points sit on a cell edge, give
or take the plate pose's error (a millimetre put a raised brick's wall into the
next cell at the wrist camera's wide angles)."""

MIN_CELL_POINTS = 3
MIN_CELL_SHARE = 0.2
"""Depth points a stud cell needs before its height is used: at least 3 and a
fifth of the frame's median cell. Cells seen only at a grazing angle past a
taller neighbour (a few points) read walls, not tops."""

MIN_VISIBLE_FRACTION = 0.5
"""Share of a brick's uncovered top cells that must be measured for a verdict
on it; otherwise it is unverified."""

LOW_CELL_FRACTION = 0.2
"""A brick is absent when at least this share (and at least one) of its
measured top cells reads half a brick height or more under its top."""

LEVEL_MIN_CELLS = 6
"""Empty plate cells needed to fit the height-offset plane (bias and tilt) that
absorbs a plate pose a millimetre off; with 3-5 only the offset is fitted."""

MIN_DEPTH = 0.05
MAX_DEPTH = 3.0
"""Depth readings outside [m] are ignored (a D405 reads from ~7 cm)."""

ALIGN_MODES = ("none", "plate", "free")
ROUTES = (
    "matches",
    "misplaced",
    "wrong_colour",
    "extra_bricks",
    "unseated",
    "missing_bricks",
    "unverifiable",
)


class Output(TypedDict):
    route: str
    detail: str
    mode: str
    score: float
    precision: float
    recall: float
    edit_distance: int
    n_target: int
    n_observed: int
    n_matched: int
    n_missing: int
    n_extra: int
    n_misplaced: int
    n_wrong_colour: int
    n_unseated: int
    n_unverified: int
    n_seat_unresolved: int
    matched: list[dict[str, Any]]
    missing: list[dict[str, Any]]
    extra: list[dict[str, Any]]
    misplaced: list[dict[str, Any]]
    wrong_colour: list[dict[str, Any]]
    unseated: list[dict[str, Any]]
    ahead: list[dict[str, Any]]
    unverified: list[dict[str, Any]]
    seat_unresolved: list[dict[str, Any]]
    repairs: list[dict[str, Any]]
    build_offset: dict[str, float]
    cells: list[dict[str, Any]]
    level_correction_mm: list[float]
    depth_sigma_mm: float
    report_json: str


# ----------------------------------------------------------------- pose helpers


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    m = np.eye(4)
    if isinstance(pose, dict):
        m[:3, :3] = _rot(pose.get("rotation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        m[:3, 3] = _xyz(pose["position"])
        return m
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError("a pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return arr


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


def _wrap(deg: float, period: float) -> float:
    return (deg + period / 2.0) % period - period / 2.0


def _colour_name(c: Any) -> str | None:
    if c is None or c == "":
        return None
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _size(v: Any) -> list[int]:
    if isinstance(v, str):
        parts = v.lower().replace(" ", "").split("x")
    else:
        parts = list(v)
    if len(parts) != 2:
        raise ValueError(f"a brick size is 'AxB' or [A, B], got {v!r}")
    a, b = (int(p) for p in parts)
    if a < 1 or b < 1:
        raise ValueError(f"a brick size must be positive stud counts, got {v!r}")
    return [a, b]


# ----------------------------------------------------------------- bricks


class _Geo:
    """The grid's metric constants and the plate frame."""

    def __init__(
        self,
        system: str,
        pitch: float,
        body_h: float,
        stud_h: float,
        stud_d: float,
        plates_per_brick: int,
        plate_studs: list[int] | None,
        t_world_plate: np.ndarray | None,
        bias: np.ndarray,
    ):
        self.system = system
        self.pitch = pitch
        self.body_h = body_h
        self.stud_h = stud_h
        self.stud_d = stud_d
        self.plate_h = body_h / plates_per_brick
        self.plate_studs = plate_studs
        self.t_world_plate = t_world_plate
        self.bias = bias


def _brick(item: Any, index: int, geo: _Geo, role: str) -> dict[str, Any]:
    """One brick in grid form: footprint corner (x, y), canonical size (a <= b),
    canonical rotation, bottom and height in layers, colour, step, and the
    pose used for matching (centre in pitches, bottom in layers, yaw in deg)."""
    if not isinstance(item, dict):
        raise ValueError(f"{role} brick {index} must be an object, got {item!r}")
    size_v = item.get("size") or item.get("brick_size") or item.get("brick_studs")
    if size_v is None:
        raise ValueError(f"{role} brick {index} has no size / brick_size")
    size = _size(size_v)
    cell = item.get("cell", item)
    rotation = item.get("rotation")
    if "cell" not in item and "x" not in item:
        cell = {"x": 0.0, "y": 0.0, "layer": 1.0}
    if isinstance(cell, (list, tuple)):
        if len(cell) < 2:
            raise ValueError(f"{role} brick {index}: a cell is [x, y, layer], got {cell!r}")
        x, y = float(cell[0]), float(cell[1])
        layer = float(cell[2]) if len(cell) > 2 else 1.0
    elif isinstance(cell, dict):
        try:
            x, y = float(cell["x"]), float(cell["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{role} brick {index} has no cell x / y: {item!r}") from exc
        layer = float(cell.get("layer", cell.get("z", item.get("layer", 1))))
        if rotation is None:
            rotation = cell.get("rotation")
    else:
        raise ValueError(f"{role} brick {index}: unreadable cell {cell!r}")
    if rotation is None:
        rotation = 90 * int(item.get("ori", 0))
    rotation = float(rotation) % 360.0
    if abs(_wrap(rotation, 90.0)) > 1e-6:
        raise ValueError(f"{role} brick {index}: rotation must be 0/90/180/270, got {rotation}")
    rotation = int(round(rotation)) % 360
    kind = str(item.get("kind") or "brick")
    if item.get("height_units") is not None:
        height = float(item["height_units"]) * geo.plate_h / geo.body_h
    else:
        height = 1.0
    bottom = (
        float(item["z_units"]) * geo.plate_h / geo.body_h
        if item.get("z_units") is not None
        else layer - 1.0
    )
    a, b = size
    canon_rot = rotation
    if a > b:
        a, b = b, a
        canon_rot = (rotation + 90) % 360
    fx, fy = (size[1], size[0]) if rotation % 180 == 90 else (size[0], size[1])

    # Continuous pose: the cell centre, or a metric pose read off the plate.
    cx, cy, cz, yaw = x + fx / 2.0, y + fy / 2.0, bottom, float(canon_rot)
    residual_m = None
    pose_v = item.get("brick_pose") or item.get("pose")
    has_cell = "cell" in item or "x" in item
    if not has_cell and isinstance(pose_v, dict) and "position" in pose_v:
        if geo.t_world_plate is None or geo.plate_studs is None:
            raise ValueError(
                f"{role} brick {index} gives a metric pose: plate_pose and plate_studs are needed"
            )
        t = np.linalg.inv(geo.t_world_plate) @ _matrix(pose_v)
        ref = str(item.get("pose_reference", "body_centre"))
        drop = {"body_centre": 0.5, "bottom": 0.0, "top": 1.0}.get(ref)
        if drop is None:
            raise ValueError(f"pose_reference must be body_centre / bottom / top, got {ref!r}")
        cx = (t[0, 3] - geo.bias[0]) / geo.pitch + geo.plate_studs[0] / 2.0
        cy = (t[1, 3] - geo.bias[1]) / geo.pitch + geo.plate_studs[1] / 2.0
        cz = t[2, 3] / geo.body_h - drop * height
        r = t[:3, :3]
        if r[2, 2] < 0.0:
            r = r @ np.diag([1.0, -1.0, -1.0])
        yaw_size_x = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
        yaw = yaw_size_x + (90.0 if size[0] > size[1] else 0.0)
        rotation = int(round(yaw_size_x / 90.0)) % 4 * 90
        canon_rot = (rotation + 90) % 360 if size[0] > size[1] else rotation
        fx, fy = (size[1], size[0]) if rotation % 180 == 90 else (size[0], size[1])
        x, y = round(cx - fx / 2.0), round(cy - fy / 2.0)
        layer = round(cz) + 1.0
        bottom = layer - 1.0
        residual_m = (cz - bottom) * geo.body_h
    else:
        res = item.get("residual")
        if item.get("height_residual_m") is not None:
            residual_m = float(item["height_residual_m"])
        elif isinstance(res, dict) and res.get("z") is not None:
            residual_m = float(res["z"])
        elif item.get("seated") is False:
            residual_m = geo.stud_h
        if residual_m is not None:
            cz += residual_m / geo.body_h

    return {
        "id": item.get("id", index),
        "index": index,
        "x": int(round(x)) if abs(x - round(x)) < 1e-6 else x,
        "y": int(round(y)) if abs(y - round(y)) < 1e-6 else y,
        "layer": int(round(layer)) if abs(layer - round(layer)) < 1e-6 else layer,
        "rotation": rotation,
        "size": f"{size[0]}x{size[1]}",
        "shape": (a, b),
        "footprint": [fx, fy],
        "bottom": bottom,
        "height": height,
        "kind": kind,
        "studs_on_top": bool(item.get("studs_on_top", kind != "tile")),
        "colour": _colour_name(item.get("colour", item.get("color"))),
        "colours": None,
        "step": item.get("step"),
        "seq": item.get("seq"),
        "pos": np.array([cx, cy, cz]),
        "yaw": yaw,
        "residual_m": residual_m,
        "inferred": bool(item.get("inferred", False)),
    }


def _record(b: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": b["id"],
        "cell": {"x": b["x"], "y": b["y"], "layer": b["layer"], "rotation": b["rotation"]},
        "size": b["size"],
        "footprint": b["footprint"],
        "colour": b["colour"],
    }
    for k in ("step", "seq"):
        if b.get(k) is not None:
            out[k] = b[k]
    if b.get("residual_m") is not None:
        out["height_residual_mm"] = round(float(b["residual_m"]) * 1e3, 2)
    if b.get("colours"):
        out["colour_candidates"] = sorted(b["colours"])
    if b.get("inferred"):
        out["inferred"] = True
    if b.get("blob_layers"):
        out["stack_layers"] = b["blob_layers"]
    return out


def _load(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        return json.loads(v) if v else []
    return v


def _target_items(target: Any) -> list[dict[str, Any]]:
    """Bricks from a list, parsing-ldraw-build-instructions' output / steps,
    planning-brick-assembly-order's sequence, or their JSON text."""
    t = _load(target)
    if isinstance(t, dict):
        if "sequence" in t:
            t = t["sequence"]
        elif "steps" in t:
            t = t["steps"]
        elif "bricks" in t:
            t = t["bricks"]
        else:
            t = [t[k] for k in sorted(t, key=lambda s: int(s) if str(s).isdigit() else 0)]
    if not isinstance(t, list):
        raise ValueError("target must be a list of bricks, steps or a sequence")
    items: list[dict[str, Any]] = []
    for i, it in enumerate(t):
        if isinstance(it, dict) and isinstance(it.get("bricks"), list):
            for b in it["bricks"]:
                items.append({**b, "step": b.get("step", it.get("step", i + 1))})
        else:
            items.append(it)
    return items


# ----------------------------------------------------------------- LTRON matching


class _Matcher:
    """LTRON's match_assemblies / validate_matches over grid bricks."""

    def __init__(self, lateral: float, layer: float, yaw_tol: float, use_colour: bool):
        self.scale = np.array([1.0 / lateral, 1.0 / lateral, 1.0 / layer])
        self.yaw_tol = yaw_tol
        self.use_colour = use_colour

    def colour_ok(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        if not self.use_colour:
            return True
        ca = a["colours"] or ({a["colour"]} if a["colour"] else None)
        cb = b["colours"] or ({b["colour"]} if b["colour"] else None)
        return ca is None or cb is None or bool(ca & cb)

    def yaw_ok(self, ya: float, yb: float, shape: tuple[int, int]) -> bool:
        period = 90.0 if shape[0] == shape[1] else 180.0
        return abs(_wrap(ya - yb, period)) <= self.yaw_tol

    @staticmethod
    def pose(b: dict[str, Any]) -> np.ndarray:
        m = _rz(b["yaw"])
        m[:3, 3] = b["pos"]
        return m

    def validate(
        self,
        obs: list[dict[str, Any]],
        tgt: list[dict[str, Any]],
        a_to_b: np.ndarray,
        colour: bool = True,
        tree: cKDTree | None = None,
    ) -> list[tuple[int, int]]:
        """Pairs (observed, target) whose shapes, colours and poses agree under
        the offset, assigned one to one, nearest first."""
        if not obs or not tgt:
            return []
        tree = tree or cKDTree(np.array([b["pos"] for b in tgt]) * self.scale)
        yaw_off = math.degrees(math.atan2(a_to_b[1, 0], a_to_b[0, 0]))
        pos = (a_to_b[:3, :3] @ np.array([b["pos"] for b in obs]).T).T + a_to_b[:3, 3]
        near = tree.query_ball_point(pos * self.scale, 1.0 + 1e-9)
        cand: list[tuple[int, int, float]] = []
        for i, js in enumerate(near):
            a = obs[i]
            for j in js:
                b = tgt[j]
                if a["shape"] != b["shape"]:
                    continue
                if colour and not self.colour_ok(a, b):
                    continue
                if not self.yaw_ok(a["yaw"] + yaw_off, b["yaw"], b["shape"]):
                    continue
                cand.append((i, j, float(np.linalg.norm((pos[i] - b["pos"]) * self.scale))))
        if not cand:
            return []
        rows = sorted({c[0] for c in cand})
        cols = sorted({c[1] for c in cand})
        ri = {r: k for k, r in enumerate(rows)}
        ci = {c: k for k, c in enumerate(cols)}
        big = 1e6
        cost = np.full((len(rows), len(cols)), big)
        for i, j, d in cand:
            cost[ri[i], ci[j]] = d
        r, c = linear_sum_assignment(cost)
        return [(rows[a], cols[b]) for a, b in zip(r, c, strict=True) if cost[a, b] < big]

    def allowed(self, off: np.ndarray, align: str) -> bool:
        yaw = math.degrees(math.atan2(off[1, 0], off[0, 0]))
        if abs(_wrap(yaw, 90.0)) > self.yaw_tol:
            return False
        if align == "plate" and abs(off[2, 3]) * self.scale[2] > 1.0:
            return False
        return True

    def match(
        self,
        obs: list[dict[str, Any]],
        tgt: list[dict[str, Any]],
        align: str,
        seed_identity: bool = True,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """LTRON's match_assemblies: the rigid offset that brings the most
        bricks into agreement (observed = LTRON's assembly_a, target = b)."""
        if not obs or not tgt:
            return [], np.eye(4)
        tree = cKDTree(np.array([b["pos"] for b in tgt]) * self.scale)
        best = self.validate(obs, tgt, np.eye(4), tree=tree) if seed_identity else []
        best_offset = np.eye(4)
        if align == "none":
            return best, best_offset
        tested: set[tuple[int, int]] = set(best)
        matched_a = {a for a, _ in best}
        matched_b = {b for _, b in best}
        shapes, counts = np.unique(
            np.array([f"{b['shape'][0]}x{b['shape'][1]}" for b in obs]), return_counts=True
        )
        shape_order = [s for s, _ in sorted(zip(shapes, counts, strict=True), key=lambda t: t[1])]
        key = [f"{b['shape'][0]}x{b['shape'][1]}" for b in obs]
        tkey = [f"{b['shape'][0]}x{b['shape'][1]}" for b in tgt]
        finished = False
        while not finished:
            finished = True
            for s in shape_order:
                idx_b = [j for j, k in enumerate(tkey) if k == s]
                if not idx_b:
                    continue
                idx_a = [i for i, k in enumerate(key) if k == s]
                for a in idx_a:
                    for b in idx_b:
                        if a in matched_a and b in matched_b:
                            continue
                        if (a, b) in tested:
                            continue
                        if not self.colour_ok(obs[a], tgt[b]):
                            continue
                        pose_a = self.pose(obs[a])
                        pose_b = self.pose(tgt[b])
                        sym = (
                            [0.0, 90.0, 180.0, 270.0]
                            if s.split("x")[0] == s.split("x")[1]
                            else [0.0, 180.0]
                        )
                        best_sym: list[tuple[int, int]] = []
                        best_sym_off = np.eye(4)
                        for turn in sym:
                            a_to_b = pose_b @ _rz(turn) @ np.linalg.inv(pose_a)
                            if not self.allowed(a_to_b, align):
                                continue
                            pos = (a_to_b[:3, :3] @ np.array([o["pos"] for o in obs]).T).T + a_to_b[
                                :3, 3
                            ]
                            hits = tree.query_ball_point(pos * self.scale, 1.0 + 1e-9)
                            if sum(1 for h in hits if h) <= len(best):
                                continue
                            valid = self.validate(obs, tgt, a_to_b, tree=tree)
                            tested.update(valid)
                            if len(valid) > len(best_sym):
                                best_sym, best_sym_off = valid, a_to_b
                        if len(best_sym) > len(best):
                            best, best_offset = best_sym, best_sym_off
                            matched_a = {x for x, _ in best}
                            matched_b = {y for _, y in best}
                            finished = False
                            break
                    if not finished:
                        break
                if not finished:
                    break
        return best, best_offset


def _same_brick(o: dict[str, Any], t: dict[str, Any]) -> bool:
    """Could observed o be target brick t put somewhere else? Same shape; a
    depth blob also when it fits inside t's footprint and covers half of it (a
    moved brick partly hidden behind a taller one, or a 1-wide brick whose far
    cells the overhead camera sees at a grazing angle)."""
    if o["shape"] == t["shape"]:
        return True
    if "_cells" not in o:
        return False
    (oa, ob), (ta, tb) = o["shape"], t["shape"]
    return oa <= ta and ob <= tb and oa * ob >= 0.5 * ta * tb


def _f1(tp: int, fp: int, fn: int) -> float:
    """LTRON's score.f1."""
    d = tp + 0.5 * (fp + fn)
    return float(tp / d) if d > 0 else 1.0


def _edit_distance(
    matcher: _Matcher,
    obs: list[dict[str, Any]],
    tgt: list[dict[str, Any]],
    first: list[tuple[int, int]],
) -> int:
    """LTRON's edit_distance: after the first matching, every further rigid
    group (a sub-assembly that one move would fix) counts 1, every brick left
    unmatched on either side counts 1."""
    left_a = [i for i in range(len(obs)) if i not in {a for a, _ in first}]
    left_b = [j for j in range(len(tgt)) if j not in {b for _, b in first}]
    d = 0
    while left_a and left_b:
        sub_a = [obs[i] for i in left_a]
        sub_b = [tgt[j] for j in left_b]
        group, _ = matcher.match(sub_a, sub_b, "free", seed_identity=False)
        if not group:
            break
        d += 1
        used_a = {left_a[a] for a, _ in group}
        used_b = {left_b[b] for _, b in group}
        left_a = [i for i in left_a if i not in used_a]
        left_b = [j for j in left_b if j not in used_b]
    return d + len(left_a) + len(left_b)


# ----------------------------------------------------------------- depth path


def _group_median(keys: np.ndarray, vals: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    med = np.full(n, np.nan)
    cnt = np.bincount(keys, minlength=n)
    if not len(keys):
        return med, cnt
    order = np.lexsort((vals, keys))
    k, v = keys[order], vals[order]
    starts = np.searchsorted(k, np.arange(n), side="left")
    for c in np.nonzero(cnt)[0]:
        s, m = starts[c], cnt[c]
        med[c] = 0.5 * (v[s + (m - 1) // 2] + v[s + m // 2])
    return med, cnt


def _largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(r0, c0, r1, c1) exclusive of the largest all-true rectangle."""
    best = (0, 0, 0, 0, 0)
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        for c0 in range(cols):
            h = heights[c0]
            for c1 in range(c0, cols):
                h = min(h, heights[c1])
                if h == 0:
                    break
                area = h * (c1 - c0 + 1)
                if area > best[0]:
                    best = (area, r - h + 1, c0, r + 1, c1 + 1)
    return best[1], best[2], best[3], best[4]


def _camera_frame(
    ctx: NodeContext, camera: dict[str, Any] | None, camera_name: str | None
) -> tuple[dict[str, Any] | None, str]:
    if camera is not None:
        return camera, ""
    try:
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        return None, f"robot.get_observation failed: {exc}"
    cams = [
        c
        for c in (obs or {}).get("cameras") or []
        if c.get("depth") is not None
        and c.get("pose") is not None
        and c.get("intrinsics") is not None
    ]
    if camera_name:
        cams = [c for c in cams if c.get("name") == camera_name]
    if not cams:
        return None, "no camera frame with depth, intrinsics and pose" + (
            f" named {camera_name!r}" if camera_name else ""
        )
    return cams[0], ""


def _observe_depth(
    ctx: NodeContext,
    cam: dict[str, Any],
    geo: _Geo,
    current: list[dict[str, Any]],
    *,
    depth_scale: float | None,
    level: bool,
    check_colour: bool,
    palette: Any,
    white_ref: Any,
    min_cell_points: int,
    min_visible_fraction: float,
    notes: list[str],
) -> dict[str, Any]:
    """Height map of the plate from a depth frame, explained by the target:
    each target brick present (seated or unseated), absent, or unverified, and
    the unexplained raised cells cut into extra bricks."""
    nx, ny = geo.plate_studs
    p, bh, sh = geo.pitch, geo.body_h, geo.stud_h
    stud_r = geo.stud_d / 2.0
    depth = np.asarray(cam["depth"])
    if depth.ndim == 3:
        depth = depth[..., 0]
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float64) * float(depth_scale if depth_scale else 0.001)
    else:
        depth = depth.astype(np.float64) * float(depth_scale if depth_scale else 1.0)
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t_pc = np.linalg.inv(geo.t_world_plate) @ _matrix(cam["pose"])
    valid = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    pv, pu = np.nonzero(valid)
    rays = np.stack([(pu - k[0, 2]) / k[0, 0], (pv - k[1, 2]) / k[1, 1], np.ones(len(pu))])
    o = t_pc[:3, 3]
    d = (t_pc[:3, :3] @ rays).T
    # Camera rays have z = 1, so the depth is the ray parameter: the measured
    # point's height over the plate is o_z + depth * d_z.
    z_meas = o[2] + depth[pv, pu] * d[:, 2]
    # The measured point's own stud cell, for surfaces far from the expected
    # one (a missing or an extra brick): there the expected ray hit lies up to
    # height x tan(view angle) away from where the ray really ended.
    mgx = (o[0] + depth[pv, pu] * d[:, 0] - geo.bias[0]) / p + nx / 2.0
    mgy = (o[1] + depth[pv, pu] * d[:, 1] - geo.bias[1]) / p + ny / 2.0
    mci, mcj = np.floor(mgx).astype(int), np.floor(mgy).astype(int)
    m_ok = (mci >= 0) & (mci < nx) & (mcj >= 0) & (mcj < ny)
    mci, mcj = np.clip(mci, 0, nx - 1), np.clip(mcj, 0, ny - 1)
    mfu, mfv = (mgx - mci - 0.5) * p, (mgy - mcj - 0.5) * p
    mrr = np.hypot(mfu, mfv)
    m_ok &= np.maximum(np.abs(mfu), np.abs(mfv)) <= p / 2.0 - max(EDGE_INSET, MEASURED_INSET * p)
    m_ok &= np.abs(mrr - stud_r) >= STUD_MARGIN
    ncell = nx * ny
    out: dict[str, Any] = {}

    def column_map(bricks: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The expected height map: per stud cell the top of its highest
        brick (0 = the plate's top), which brick that is, and whether it has studs."""
        top = np.zeros((nx, ny))
        tid = np.full((nx, ny), -1)
        studs = np.ones((nx, ny), dtype=bool)
        for b in sorted(bricks, key=lambda b: b["bottom"] + b["height"]):
            fx, fy = b["footprint"]
            x0, y0 = int(round(b["x"])), int(round(b["y"]))
            sl = (slice(max(0, x0), min(nx, x0 + fx)), slice(max(0, y0), min(ny, y0 + fy)))
            t = (b["bottom"] + b["height"]) * bh
            upd = top[sl] <= t + 1e-9
            top[sl] = np.where(upd, t, top[sl])
            tid[sl] = np.where(upd, b["index"], tid[sl])
            studs[sl] = np.where(upd, b["studs_on_top"], studs[sl])
        return top, tid, studs

    def cast(top: np.ndarray, studs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every pixel's ray against the expected height map (column tops and
        stud discs): the stud cell it would hit and that surface's height.
        Rays that would hit a wall, or land within EDGE_INSET of a cell edge or
        STUD_MARGIN of a stud rim, get cell -1. The cell comes from the expected
        surface, not from the noisy point, so depth noise cannot move points
        between cells (that biased the heights by ~0.4 mm overhead)."""
        cell = np.full(len(pu), -1)
        hz = np.zeros(len(pu))
        g = np.zeros((len(pu), 2))
        done = d[:, 2] >= 0.0
        levels = np.unique(np.concatenate([top.ravel(), (top + sh)[studs]]))[::-1]
        for lv in levels:
            todo = np.nonzero(~done)[0]
            if not len(todo):
                break
            t = (lv - o[2]) / d[todo, 2]
            gx = (o[0] + t * d[todo, 0] - geo.bias[0]) / p + nx / 2.0
            gy = (o[1] + t * d[todo, 1] - geo.bias[1]) / p + ny / 2.0
            ci, cj = np.floor(gx).astype(int), np.floor(gy).astype(int)
            inside = (t > 0) & (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
            ci, cj = np.clip(ci, 0, nx - 1), np.clip(cj, 0, ny - 1)
            fu, fv = (gx - ci - 0.5) * p, (gy - cj - 0.5) * p
            rr = np.hypot(fu, fv)
            on_stud = studs[ci, cj] & (rr <= stud_r)
            hv = np.where(inside, top[ci, cj] + np.where(on_stud, sh, 0.0), -np.inf)
            hit = hv >= lv - 1e-9
            done[todo[hit]] = True
            good = (
                hit
                & (np.abs(hv - lv) < 1e-9)
                & (np.maximum(np.abs(fu), np.abs(fv)) <= p / 2.0 - EDGE_INSET)
                & (~studs[ci, cj] | (np.abs(rr - stud_r) >= STUD_MARGIN))
            )
            idx = todo[good]
            cell[idx] = (ci * ny + cj)[good]
            hz[idx] = lv
            g[idx, 0], g[idx, 1] = gx[good], gy[good]
        return cell, hz, g

    def enough_points(cnt: np.ndarray) -> np.ndarray:
        typical = float(np.median(cnt[cnt > 0])) if np.any(cnt > 0) else 0.0
        return cnt >= max(min_cell_points, MIN_CELL_SHARE * typical)

    # Height offset plane from the empty plate cells (absorbs a plate pose a
    # millimetre high or tilted); cells under an extra brick are left out.
    top0, tid0, studs0 = column_map(current)
    cell0, hz0, g0 = cast(top0, studs0)
    if not np.any(cell0 >= 0):
        out["unverifiable"] = "no depth point falls on the plate's stud cells"
        return out
    level_c = np.zeros(3)
    if level:
        k0 = cell0 >= 0
        med, cnt = _group_median(cell0[k0], (z_meas - hz0)[k0], ncell)
        cells = np.nonzero((tid0.ravel() == -1) & enough_points(cnt) & (np.abs(med) < bh / 2.0))[0]
        if len(cells) >= 3:
            a = np.column_stack([np.ones(len(cells)), cells // ny + 0.5, cells % ny + 0.5])
            if len(cells) < LEVEL_MIN_CELLS:
                a = a[:, :1]
            m = med[cells]
            sel = np.ones(len(cells), dtype=bool)
            for _ in range(2):
                sol, *_ = np.linalg.lstsq(a[sel], m[sel], rcond=None)
                keep = np.abs(m - a @ sol) < max(2.0 * SEAT_TOLERANCE_FRACTION * sh, 0.001)
                if keep.sum() < a.shape[1]:
                    break
                sel = keep
            sol, *_ = np.linalg.lstsq(a[sel], m[sel], rcond=None)
            level_c[: len(sol)] = sol
        else:
            notes.append("too few empty plate cells seen to level the height map")
    out["level_correction_mm"] = [
        round(float(level_c[0] + level_c[1] * nx / 2.0 + level_c[2] * ny / 2.0) * 1e3, 3),
        round(float(level_c[1]) * 1e3, 4),
        round(float(level_c[2]) * 1e3, 4),
    ]

    # Explain the height map by the target: remove absent bricks until stable.
    present = {b["index"] for b in current}
    by_index = {b["index"]: b for b in current}
    while True:
        top, tid, studs = column_map([by_index[i] for i in present])
        cell, hz, g = cast(top, studs)
        r_e = z_meas - hz
        use_e = (cell >= 0) & (np.abs(r_e) <= bh / 2.0)
        hz_m = top[mci, mcj] + np.where(studs[mci, mcj] & (mrr <= stud_r), sh, 0.0)
        r_m = z_meas - hz_m
        use_m = ~use_e & m_ok & ((cell >= 0) | (np.abs(r_m) > bh / 2.0))
        cell = np.where(use_e, cell, np.where(use_m, mci * ny + mcj, -1))
        g = np.where(use_e[:, None], g, np.column_stack([mgx, mgy]))
        r_all = np.where(use_e, r_e, r_m)
        kept = cell >= 0
        cell_id = cell[kept]
        r = r_all[kept] - (level_c[0] + level_c[1] * g[kept, 0] + level_c[2] * g[kept, 1])
        med, cnt = _group_median(cell_id, r, ncell)
        enough = enough_points(cnt)
        gone = []
        for i in present:
            b = by_index[i]
            vis = np.nonzero(tid.ravel() == i)[0]
            meas = vis[enough[vis]]
            if not len(meas):
                continue
            low = int(np.sum(med[meas] < -min(b["height"] * bh, bh) / 2.0))
            if low >= max(1, math.ceil(LOW_CELL_FRACTION * len(meas))):
                gone.append(i)
        if not gone:
            break
        present -= set(gone)
    out["n_points"] = int(len(cell_id))
    # The frame's per-point depth scatter about the cell medians (robust).
    ok_pts = enough[cell_id] & np.isfinite(med[cell_id])
    sigma_pt = (
        1.4826 * float(np.median(np.abs(r[ok_pts] - med[cell_id[ok_pts]]))) if ok_pts.any() else 0.0
    )
    out["depth_sigma_mm"] = round(sigma_pt * 1e3, 3)
    pix_v, pix_u = pv[kept], pu[kept]
    observed: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    absent = [by_index[i] for i in sorted(set(by_index) - present)]
    seen_any = False
    for i in sorted(present):
        b = by_index[i]
        vis = np.nonzero(tid.ravel() == i)[0]
        if not len(vis):
            observed.append({**b, "residual_m": None, "inferred": True})
            continue
        meas = vis[enough[vis]]
        if len(meas) < min_visible_fraction * len(vis):
            unverified.append(b)
            continue
        seen_any = True
        # Cells under something the target does not have (an extra brick on
        # top) say nothing about this brick's own height.
        meas = meas[med[meas] <= (sh + bh) / 2.0]
        if not len(meas):
            observed.append({**b, "residual_m": None, "inferred": True})
            continue
        sel = np.isin(cell_id, meas)
        rb = r[sel]
        res = float(np.median(rb))
        # Standard error of the median from the points' scatter (MAD), never
        # under the frame's pooled per-point scatter (a few quantised points
        # can have a MAD of 0).
        sigma = max(1.4826 * float(np.median(np.abs(rb - res))), sigma_pt)
        se = sigma * 1.2533 / math.sqrt(len(rb))
        # The highest cell, for a brick seated at one end and resting on the
        # studs at the other: its median and that median's standard error.
        mad, _ = _group_median(cell_id[sel], np.abs(rb - med[cell_id[sel]]), ncell)
        cell_se = np.maximum(1.4826 * mad[meas], sigma_pt) * 1.2533 / np.sqrt(cnt[meas])
        top_i = int(np.argmax(med[meas] - CELL_SIGMAS * cell_se))
        ob = {
            **b,
            "residual_m": res,
            "residual_se_m": se,
            "max_cell_m": float(med[meas][top_i]),
            "max_cell_se_m": float(cell_se[top_i]),
            "max_cell": [int(meas[top_i] // ny), int(meas[top_i] % ny)],
        }
        ob["pos"] = b["pos"] + np.array([0.0, 0.0, res / bh])
        observed.append(ob)
    for b in absent:
        vis = np.nonzero(tid0.ravel() == b["index"])[0]
        if len(vis) and np.any(enough[vis]):
            seen_any = True

    # Raised cells the target does not explain: extra bricks.
    measured_top = top.ravel() + med
    high = enough & (med > (sh + bh) / 2.0)
    blobs: list[dict[str, Any]] = []
    if high.any():
        lay = np.zeros(ncell, dtype=int)
        lay[high] = np.round(measured_top[high] / bh).astype(int)
        grid_high = high.reshape(nx, ny)
        grid_lay = lay.reshape(nx, ny)
        seen = np.zeros((nx, ny), dtype=bool)
        for x0 in range(nx):
            for y0 in range(ny):
                if not grid_high[x0, y0] or seen[x0, y0]:
                    continue
                comp, stack = [], [(x0, y0)]
                seen[x0, y0] = True
                while stack:
                    cx_, cy_ = stack.pop()
                    comp.append((cx_, cy_))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        qx, qy = cx_ + dx, cy_ + dy
                        if (
                            0 <= qx < nx
                            and 0 <= qy < ny
                            and grid_high[qx, qy]
                            and not seen[qx, qy]
                            and grid_lay[qx, qy] == grid_lay[x0, y0]
                        ):
                            seen[qx, qy] = True
                            stack.append((qx, qy))
                xs = [c[0] for c in comp]
                ys = [c[1] for c in comp]
                bx0, by0 = min(xs), min(ys)
                m = np.zeros((max(xs) - bx0 + 1, max(ys) - by0 + 1), dtype=bool)
                for c in comp:
                    m[c[0] - bx0, c[1] - by0] = True
                while m.any():
                    r0_, c0_, r1_, c1_ = _largest_rectangle(m)
                    m[r0_:r1_, c0_:c1_] = False
                    fx, fy = r1_ - r0_, c1_ - c0_
                    cells = [
                        (bx0 + a_, by0 + b_) for a_ in range(r0_, r1_) for b_ in range(c0_, c1_)
                    ]
                    idx = [cx_ * ny + cy_ for cx_, cy_ in cells]
                    top_layer = int(grid_lay[cells[0]])
                    under = float(np.mean([top.ravel()[q] for q in idx]))
                    blobs.append(
                        {
                            "cells": idx,
                            "x": bx0 + r0_,
                            "y": by0 + c0_,
                            "fx": fx,
                            "fy": fy,
                            "layer": top_layer,
                            "stack": max(1, int(round((top_layer * bh - under) / bh))),
                            "comp": (x0, y0),
                        }
                    )
    out["cell_med"] = med
    out["cell_cnt"] = cnt
    out["cell_enough"] = enough
    out["top"] = top
    out["tid"] = tid
    extra_items: list[dict[str, Any]] = []
    for n, bl in enumerate(blobs):
        a, b = sorted((bl["fx"], bl["fy"]))
        item = {
            "id": f"extra_{n}",
            "cell": {
                "x": bl["x"],
                "y": bl["y"],
                "layer": bl["layer"],
                "rotation": 0 if bl["fx"] <= bl["fy"] else 90,
            },
            "size": f"{a}x{b}",
        }
        eb = _brick(item, 10_000 + n, geo, "observed")
        eb["residual_m"] = None
        eb["blob_layers"] = bl["stack"] if bl["stack"] > 1 else None
        eb["_cells"] = bl["cells"]
        eb["_comp"] = bl["comp"]
        extra_items.append(eb)
    out["observed"] = observed + extra_items
    for ob in out["observed"]:
        ob["colour"] = None
        ob["colours"] = None
    out["unverified"] = unverified
    if not seen_any and not extra_items and current:
        out["unverifiable"] = "none of the target's bricks has enough depth points on its top"

    # Colour of every seen brick from the RGB frame (tools/colour).
    rgb = cam.get("rgb")
    if check_colour and rgb is not None:
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        for ob in out["observed"]:
            if ob.get("inferred"):
                continue
            if "_cells" in ob:
                pick = np.isin(cell_id, ob["_cells"]) & (r > (sh + bh) / 2.0)
            else:
                cells_ = np.nonzero(tid.ravel() == ob["index"])[0]
                pick = np.isin(cell_id, cells_) & (np.abs(r - (ob["residual_m"] or 0.0)) < bh / 4.0)
            if pick.sum() < 4:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[pix_v[pick], pix_u[pick]] = 1
            try:
                res = ctx.tool(
                    "colour.classify", rgb=rgb, mask=mask, palette=palette, white_ref=white_ref
                )
            except Exception as exc:
                notes.append(f"colour.classify failed ({exc}); colours not checked")
                break
            status = (res or {}).get("status")
            if status == "classified" and res.get("name"):
                ob["colour"] = _colour_name(res["name"])
                ob["colours"] = {ob["colour"]}
            elif status == "ambiguous" and res.get("name"):
                ob["colour"] = _colour_name(res["name"])
                ob["colours"] = {
                    _colour_name(res["name"]),
                    _colour_name(res.get("second_name")),
                } - {None}
    return out


# ----------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    target: Any,
    observed: Any = None,
    upto_step: int | None = None,
    upto_seq: int | None = None,
    camera: dict[str, Any] | None = None,
    camera_name: str | None = None,
    plate_pose: dict[str, Any] | None = None,
    plate_studs: list[int] | None = None,
    system: str = "lego",
    stud_pitch: float | None = None,
    brick_height: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    plate_thickness: float = 0.0,
    grid_bias: list[float] | None = None,
    align: str = "none",
    check_colour: bool = True,
    palette: Any = "lego",
    white_ref: Any = None,
    lateral_tolerance: float = LATERAL_TOLERANCE,
    layer_tolerance: float = LAYER_TOLERANCE,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    seat_tolerance: float | None = None,
    level_plate: bool = True,
    depth_scale: float | None = None,
    min_cell_points: int = MIN_CELL_POINTS,
    min_visible_fraction: float = MIN_VISIBLE_FRACTION,
    require_seating: bool = False,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if align not in ALIGN_MODES:
        raise ValueError(f"align must be one of {ALIGN_MODES}, got {align!r}")
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_d = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"])
    if min(pitch, body_h, stud_h, stud_d) <= 0.0:
        raise ValueError("stud_pitch, brick_height, stud_height and stud_diameter must be positive")
    seat_tol = float(
        seat_tolerance if seat_tolerance is not None else SEAT_TOLERANCE_FRACTION * stud_h
    )
    t_world_plate = None
    if plate_pose is not None:
        t_world_plate = _matrix(plate_pose)
        t_world_plate = t_world_plate @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(plate_thickness)], [0, 0, 0, 1]]
        )
    studs = None
    if plate_studs is not None:
        studs = [int(v) for v in plate_studs]
        if len(studs) != 2 or min(studs) < 1:
            raise ValueError(f"plate_studs must be two positive stud counts, got {plate_studs!r}")
    geo = _Geo(
        system,
        pitch,
        body_h,
        stud_h,
        stud_d,
        int(preset["plates_per_brick"]),
        studs,
        t_world_plate,
        np.array([float(v) for v in (grid_bias or [0.0, 0.0])][:2]),
    )

    items = _target_items(target)
    all_t = [_brick(it, i, geo, "target") for i, it in enumerate(items)]
    for b in all_t:
        b["residual_m"] = None

    def is_current(b: dict[str, Any]) -> bool:
        if upto_step is not None and b["step"] is not None and int(b["step"]) > int(upto_step):
            return False
        if upto_seq is not None and b["seq"] is not None and int(b["seq"]) > int(upto_seq):
            return False
        return True

    current = [b for b in all_t if is_current(b)]
    future = [b for b in all_t if not is_current(b)]
    notes: list[str] = []
    out: Output = {
        "route": "unverifiable",
        "detail": "",
        "mode": "bricks" if observed is not None else "depth",
        "score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "edit_distance": 0,
        "n_target": len(current),
        "n_observed": 0,
        "n_matched": 0,
        "n_missing": 0,
        "n_extra": 0,
        "n_misplaced": 0,
        "n_wrong_colour": 0,
        "n_unseated": 0,
        "n_unverified": 0,
        "n_seat_unresolved": 0,
        "matched": [],
        "missing": [],
        "extra": [],
        "misplaced": [],
        "wrong_colour": [],
        "unseated": [],
        "ahead": [],
        "unverified": [],
        "seat_unresolved": [],
        "repairs": [],
        "build_offset": {"dx": 0.0, "dy": 0.0, "dlayer": 0.0, "yaw_deg": 0.0},
        "cells": [],
        "level_correction_mm": [],
        "depth_sigma_mm": 0.0,
        "report_json": "",
    }

    def finish(route: str, why: str) -> Output:
        out["route"] = route
        out["detail"] = "; ".join(s for s in (why, *notes) if s)
        out["report_json"] = json.dumps(
            {k: out[k] for k in out if k not in ("report_json", "cells")}, default=str
        )
        return out

    # ---- the observed bricks
    if observed is not None:
        obs_items = _load(observed)
        if isinstance(obs_items, dict):
            obs_items = obs_items.get("bricks") or obs_items.get("observed") or [obs_items]
        if not isinstance(obs_items, list):
            raise ValueError("observed must be a list of bricks")
        obs = [_brick(it, i, geo, "observed") for i, it in enumerate(obs_items)]
        for b in obs:
            if b["colour"]:
                b["colours"] = {b["colour"]}
    else:
        if t_world_plate is None or studs is None:
            raise ValueError("the depth path needs plate_pose and plate_studs (or give observed)")
        cam, why = _camera_frame(ctx, camera, camera_name)
        if cam is None:
            return finish("unverifiable", why)
        if cam.get("depth") is None or cam.get("intrinsics") is None or cam.get("pose") is None:
            return finish("unverifiable", "the camera frame lacks depth, intrinsics or pose")
        seen = _observe_depth(
            ctx,
            cam,
            geo,
            current,
            depth_scale=depth_scale,
            level=level_plate,
            check_colour=check_colour,
            palette=palette,
            white_ref=white_ref,
            min_cell_points=int(min_cell_points),
            min_visible_fraction=float(min_visible_fraction),
            notes=notes,
        )
        out["level_correction_mm"] = seen.get("level_correction_mm", [])
        out["depth_sigma_mm"] = seen.get("depth_sigma_mm", 0.0)
        if seen.get("unverifiable"):
            return finish("unverifiable", seen["unverifiable"])
        obs = seen["observed"]
        unver = seen["unverified"]
        out["unverified"] = [_record(b) for b in unver]
        skip = {b["index"] for b in unver}
        current = [b for b in current if b["index"] not in skip]
        if "cell_med" in seen:
            med, cnt, top = seen["cell_med"], seen["cell_cnt"], seen["top"]
            ny_ = studs[1]
            for c in np.nonzero(seen["cell_enough"])[0]:
                r = float(med[c])
                if abs(r) <= seat_tol:
                    continue
                out["cells"].append(
                    {
                        "x": int(c // ny_),
                        "y": int(c % ny_),
                        "expected_top_mm": round(float(top.ravel()[c]) * 1e3, 2),
                        "measured_top_mm": round(float(top.ravel()[c] + r) * 1e3, 2),
                        "n_points": int(cnt[c]),
                    }
                )

    out["n_observed"] = len(obs)
    matcher = _Matcher(
        float(lateral_tolerance),
        float(layer_tolerance),
        float(yaw_tolerance_deg),
        bool(check_colour),
    )

    # ---- LTRON: the best offset and the matching under it
    pairs, offset = matcher.match(obs, current, align)
    if align == "none" and len(pairs) < min(len(obs), len(current)):
        diag, diag_off = matcher.match(obs, current, "plate")
        if len(diag) > len(pairs):
            yaw = math.degrees(math.atan2(diag_off[1, 0], diag_off[0, 0]))
            notes.append(
                f"{len(diag)} bricks would match with the whole build moved by "
                f"({diag_off[0, 3]:+.1f}, {diag_off[1, 3]:+.1f}) studs, turned {yaw:+.0f} deg"
            )
    out["build_offset"] = {
        "dx": round(float(offset[0, 3]), 3),
        "dy": round(float(offset[1, 3]), 3),
        "dlayer": round(float(offset[2, 3]), 3),
        "yaw_deg": round(math.degrees(math.atan2(offset[1, 0], offset[0, 0])), 3),
    }
    used_o = {a for a, _ in pairs}
    used_t = {b for _, b in pairs}
    for a, b in pairs:
        o, t = obs[a], current[b]
        rec = {"target": _record(t), "observed": _record(o)}
        out["matched"].append(rec)
        res = o.get("residual_m")
        se = float(o.get("residual_se_m") or 0.0)
        top_cell = o.get("max_cell_m")
        top_se = float(o.get("max_cell_se_m") or 0.0)
        raised = res is not None and res > seat_tol and res > SEAT_SIGMAS * se
        partly = top_cell is not None and top_cell > seat_tol and top_cell > CELL_SIGMAS * top_se
        entry = {
            **_record(t),
            "height_residual_mm": round(float(res or 0.0) * 1e3, 2),
            "height_se_mm": round(se * 1e3, 2),
        }
        if top_cell is not None:
            entry["highest_cell"] = o["max_cell"]
            entry["highest_cell_mm"] = round(top_cell * 1e3, 2)
        if raised or partly:
            entry["how"] = "raised" if raised else "one end"
            out["unseated"].append(entry)
        elif res is not None and stud_h - res < SEAT_SIGMAS * se:
            out["seat_unresolved"].append(entry)

    def apply(off: np.ndarray, b: dict[str, Any]) -> np.ndarray:
        return off[:3, :3] @ b["pos"] + off[:3, 3]

    left_o = [i for i in range(len(obs)) if i not in used_o]
    left_t = [j for j in range(len(current)) if j not in used_t]

    # A later step's brick, in its place: ahead of the instructions, not extra.
    ahead_o: set[int] = set()
    if future and left_o:
        sub = [obs[i] for i in left_o]
        ahead = matcher.validate(sub, future, offset)
        for _, b in ahead:
            out["ahead"].append(_record(future[b]))
        ahead_o = {left_o[a] for a, _ in ahead}
        left_o = [i for i in left_o if i not in ahead_o]

    # Same place and shape, other colour.
    if left_o and left_t:
        sub_o = [obs[i] for i in left_o]
        sub_t = [current[j] for j in left_t]
        wc = matcher.validate(sub_o, sub_t, offset, colour=False)
        for a, b in wc:
            o, t = sub_o[a], sub_t[b]
            out["wrong_colour"].append(
                {
                    **_record(t),
                    "expected_colour": t["colour"],
                    "seen_colour": o["colour"],
                    "observed_id": o["id"],
                }
            )
        ga = {left_o[a] for a, _ in wc}
        gb = {left_t[b] for _, b in wc}
        left_o = [i for i in left_o if i not in ga]
        left_t = [j for j in left_t if j not in gb]

    # Same shape (and colour) elsewhere: misplaced, paired nearest first.
    if left_o and left_t:
        cost = np.full((len(left_o), len(left_t)), 1e6)
        for a, i in enumerate(left_o):
            for b, j in enumerate(left_t):
                o, t = obs[i], current[j]
                if _same_brick(o, t) and matcher.colour_ok(o, t):
                    cost[a, b] = float(
                        np.linalg.norm((apply(offset, o) - t["pos"]) * matcher.scale)
                    )
        r, c = linear_sum_assignment(cost)
        ga, gb = set(), set()
        for a, b in zip(r, c, strict=True):
            if cost[a, b] >= 1e6:
                continue
            o, t = obs[left_o[a]], current[left_t[b]]
            d = apply(offset, o) - t["pos"]
            yaw = _wrap(o["yaw"] - t["yaw"], 90.0 if t["shape"][0] == t["shape"][1] else 180.0)
            out["misplaced"].append(
                {
                    **_record(t),
                    "observed": _record(o),
                    "shift_studs": [round(float(d[0]), 2), round(float(d[1]), 2)],
                    "shift_layers": round(float(d[2]), 2),
                    "yaw_off_deg": round(float(yaw), 1),
                }
            )
            ga.add(left_o[a])
            gb.add(left_t[b])
        left_o = [i for i in left_o if i not in ga]
        left_t = [j for j in left_t if j not in gb]

    # A depth blob cut into rectangles (a moved brick partly hidden): the
    # rectangles of one raised region together, against a still missing brick.
    comps: dict[Any, list[int]] = {}
    for i in left_o:
        if "_comp" in obs[i]:
            comps.setdefault(obs[i]["_comp"], []).append(i)
    for members in comps.values():
        if len(members) < 2 or not left_t:
            continue
        xs = [c // studs[1] for i in members for c in obs[i]["_cells"]] if studs else []
        ys = [c % studs[1] for i in members for c in obs[i]["_cells"]] if studs else []
        if not xs:
            continue
        fx, fy = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        union = {
            "shape": tuple(sorted((fx, fy))),
            "_cells": [c for i in members for c in obs[i]["_cells"]],
            "colours": None,
            "colour": None,
        }
        best_j, best_d = None, math.inf
        for j in left_t:
            t = current[j]
            if (
                not _same_brick(union, t)
                or len(union["_cells"]) < 0.5 * t["shape"][0] * t["shape"][1]
            ):
                continue
            if not all(matcher.colour_ok(obs[i], t) for i in members):
                continue
            centre = np.array([(min(xs) + max(xs) + 1) / 2.0, (min(ys) + max(ys) + 1) / 2.0])
            dist = float(np.linalg.norm(centre - t["pos"][:2]))
            if dist < best_d:
                best_j, best_d = j, dist
        if best_j is None:
            continue
        t = current[best_j]
        first = obs[members[0]]
        rec = _record(first)
        rec["cell"] = {
            "x": min(xs),
            "y": min(ys),
            "layer": first["layer"],
            "rotation": 0 if fx <= fy else 90,
        }
        rec["footprint"] = [fx, fy]
        rec["size"] = f"{union['shape'][0]}x{union['shape'][1]}"
        out["misplaced"].append(
            {
                **_record(t),
                "observed": rec,
                "shift_studs": [
                    round(min(xs) + fx / 2.0 - float(t["pos"][0]), 2),
                    round(min(ys) + fy / 2.0 - float(t["pos"][1]), 2),
                ],
                "shift_layers": round(float(first["pos"][2] - t["pos"][2]), 2),
                "yaw_off_deg": 0.0,
                "partial_view": True,
            }
        )
        left_t = [j for j in left_t if j != best_j]
        left_o = [i for i in left_o if i not in members]

    out["missing"] = [_record(current[j]) for j in left_t]
    out["extra"] = [_record(obs[i]) for i in left_o]

    tp = len(pairs)
    fp = len(obs) - tp - len(out["ahead"])
    fn = len(current) - tp
    out["n_target"] = len(current)
    out["n_matched"] = tp
    out["score"] = round(_f1(tp, fp, fn), 4)
    out["precision"] = round(tp / (tp + fp), 4) if tp + fp else 1.0
    out["recall"] = round(tp / (tp + fn), 4) if tp + fn else 1.0
    keep = [i for i in range(len(obs)) if i not in ahead_o]
    renum = {i: n for n, i in enumerate(keep)}
    out["edit_distance"] = _edit_distance(
        matcher, [obs[i] for i in keep], current, [(renum[a], b) for a, b in pairs]
    )
    for k in (
        "missing",
        "extra",
        "misplaced",
        "wrong_colour",
        "unseated",
        "unverified",
        "seat_unresolved",
    ):
        out[f"n_{k}"] = len(out[k])  # type: ignore[literal-required]

    # Repairs, in the order a robot should do them.
    reps: list[dict[str, Any]] = []
    for e in sorted(out["extra"], key=lambda r: -float(r["cell"]["layer"])):
        reps.append({"action": "remove", "brick": e})
    for m in out["misplaced"]:
        reps.append({"action": "move", "from": m["observed"]["cell"], "to": m["cell"], "brick": m})
    for w in out["wrong_colour"]:
        reps.append({"action": "swap", "brick": w})
    for u in sorted(out["unseated"], key=lambda r: float(r["cell"]["layer"])):
        reps.append({"action": "press", "brick": u})
    for m in sorted(out["missing"], key=lambda r: (float(r["cell"]["layer"]), r.get("step") or 0)):
        reps.append({"action": "place", "brick": m})
    out["repairs"] = reps

    def few(lst: list[dict[str, Any]], key: str = "cell") -> str:
        cells = [f"({r[key]['x']}, {r[key]['y']}, L{r[key]['layer']}) {r['size']}" for r in lst[:3]]
        return ", ".join(cells) + (" ..." if len(lst) > 3 else "")

    summary = f"{tp}/{len(current)} target bricks matched, F1 {out['score']:.3f}" + (
        f", {len(out['ahead'])} ahead" if out["ahead"] else ""
    )
    if out["misplaced"]:
        return finish("misplaced", f"{summary}; misplaced: {few(out['misplaced'])}")
    if out["wrong_colour"]:
        return finish(
            "wrong_colour",
            f"{summary}; wrong colour: {few(out['wrong_colour'])} "
            f"(saw {out['wrong_colour'][0]['seen_colour']}, want {out['wrong_colour'][0]['expected_colour']})",
        )
    if out["extra"]:
        return finish("extra_bricks", f"{summary}; extra: {few(out['extra'])}")
    if out["unseated"]:
        return finish(
            "unseated",
            f"{summary}; unseated: {few(out['unseated'])} "
            f"({out['unseated'][0]['how']}: +{out['unseated'][0]['height_residual_mm']:.1f} mm, "
            f"highest cell +{out['unseated'][0].get('highest_cell_mm', 0.0):.1f} mm)",
        )
    if out["missing"]:
        return finish("missing_bricks", f"{summary}; missing: {few(out['missing'])}")
    if out["unverified"]:
        return finish(
            "unverifiable",
            f"{summary}; {len(out['unverified'])} bricks not seen well enough: {few(out['unverified'])}",
        )
    if out["seat_unresolved"]:
        why = (
            f"{len(out['seat_unresolved'])} bricks' seating is within the depth noise "
            f"(+-{max(u['height_se_mm'] for u in out['seat_unresolved']) * SEAT_SIGMAS:.1f} mm "
            f"against a {stud_h * 1e3:.1f} mm stud): {few(out['seat_unresolved'])}"
        )
        if require_seating:
            return finish("unverifiable", f"{summary}; {why}")
        notes.append(why)
    return finish("matches", summary)
