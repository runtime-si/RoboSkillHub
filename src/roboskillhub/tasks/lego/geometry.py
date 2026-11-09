"""Stud-grid geometry for standard LEGO (8 mm) and DUPLO (16 mm) vertical connections.

Pure Python (no numpy, no robot). Poses are ``{frame_id, position_m, quaternion_xyzw}``
dicts; ``T_A_B`` maps coordinates in frame B to frame A. The grid chain (plate top
centre * top-left offset * cell offset * yaw * footprint-centre offset) is the one of
skills/computing-brick-pose-from-stud-grid (itself ICL Robotic_Lego_Manipulation's
``Lego::calc_brick_loc``, MIT); the system presets are skills/pressing-a-brick-onto-studs'
(BrickSim's LEGO units, published DUPLO dimensions). See ``PROVENANCE`` in manifest.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Matrix = list[list[float]]


class GeometryError(ValueError):
    """Malformed or inconsistent geometric input (units, frames, quaternions, grid)."""


# --------------------------------------------------------------------------- systems


@dataclass(frozen=True)
class BrickSystem:
    """One stud standard. Lengths in metres."""

    name: str
    stud_pitch_m: float
    stud_height_m: float
    stud_diameter_m: float
    brick_height_m: float
    footprint_play_m: float
    version: str

    @property
    def stud_gap_m(self) -> float:
        """Clear space between neighbouring studs along a row."""
        return self.stud_pitch_m - self.stud_diameter_m


SYSTEMS: dict[str, BrickSystem] = {
    # BrickSim BRICK_UNIT_LENGTH 8 mm, 3 x PLATE_UNIT_HEIGHT 3.2 mm; 4.8 mm x 1.7 mm studs.
    "lego": BrickSystem("lego", 0.008, 0.0017, 0.0048, 0.0096, 0.0001, "lego-std-8mm/1"),
    # Twice the pitch; 9.4 mm x 4.5 mm studs; 19.2 mm bricks (published DUPLO dimensions).
    "duplo": BrickSystem("duplo", 0.016, 0.0045, 0.0094, 0.0192, 0.0001, "duplo-16mm/1"),
}

ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class BrickType:
    """A rectangular brick: ``studs`` along its own x and y."""

    system: BrickSystem
    studs: tuple[int, int]

    @property
    def type_id(self) -> str:
        return f"{self.system.name}:{self.studs[0]}x{self.studs[1]}@{self.system.version}"

    @property
    def body_m(self) -> tuple[float, float, float]:
        s = self.system
        return (
            self.studs[0] * s.stud_pitch_m - 2 * s.footprint_play_m,
            self.studs[1] * s.stud_pitch_m - 2 * s.footprint_play_m,
            s.brick_height_m,
        )

    def symmetric_yaws(self) -> tuple[int, ...]:
        """Yaws (deg) that map the stud/socket pattern onto itself."""
        return (0, 90, 180, 270) if self.studs[0] == self.studs[1] else (0, 180)


def brick_type(system: str, studs: Any) -> BrickType:
    if system not in SYSTEMS:
        raise GeometryError(f"unknown brick system {system!r}; known: {sorted(SYSTEMS)}")
    if isinstance(studs, str):
        try:
            studs = [int(v) for v in studs.lower().split("x")]
        except ValueError as exc:
            raise GeometryError(f"brick studs must look like '2x4', got {studs!r}") from exc
    if (
        not isinstance(studs, (list, tuple))
        or len(studs) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 16 for v in studs)
    ):
        raise GeometryError(f"brick studs must be two integers in [1, 16], got {studs!r}")
    return BrickType(SYSTEMS[system], (int(studs[0]), int(studs[1])))


# --------------------------------------------------------------------------- transforms


def _finite(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise GeometryError(f"{name} must be a finite number")
    return float(v)


def validate_pose(pose: Any, frame_id: str | None = None, name: str = "pose") -> dict:
    """Exactly ``frame_id, position_m, quaternion_xyzw``; unit quaternion (tol 0.01)."""
    if not isinstance(pose, dict) or set(pose) != {"frame_id", "position_m", "quaternion_xyzw"}:
        raise GeometryError(f"{name} requires exactly frame_id, position_m, quaternion_xyzw")
    if not isinstance(pose["frame_id"], str) or not pose["frame_id"]:
        raise GeometryError(f"{name}.frame_id must be a nonempty string")
    if frame_id is not None and pose["frame_id"] != frame_id:
        raise GeometryError(
            f"{name} is in frame {pose['frame_id']!r}, expected {frame_id!r}; "
            "give an explicit transform"
        )
    p, q = pose["position_m"], pose["quaternion_xyzw"]
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        raise GeometryError(f"{name}.position_m must have 3 numbers")
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise GeometryError(f"{name}.quaternion_xyzw must have 4 numbers")
    p = [_finite(v, f"{name}.position_m") for v in p]
    q = [_finite(v, f"{name}.quaternion_xyzw") for v in q]
    if any(abs(v) > 100.0 for v in p):
        raise GeometryError(f"{name}.position_m beyond 100 m: millimetres given as metres?")
    n = math.sqrt(sum(v * v for v in q))
    if abs(n - 1.0) > 0.01:
        raise GeometryError(f"{name}.quaternion_xyzw must have unit norm (got {n:.4f})")
    return {"frame_id": pose["frame_id"], "position_m": p, "quaternion_xyzw": [v / n for v in q]}


def quat_to_rot(q: list[float]) -> Matrix:
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rot_to_quat(r: Matrix) -> list[float]:
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, s / 4]
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        q = [s / 4, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s]
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        q = [(r[0][1] + r[1][0]) / s, s / 4, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s]
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        q = [(r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, s / 4, (r[1][0] - r[0][1]) / s]
    if q[3] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def matrix(pose: dict) -> Matrix:
    r = quat_to_rot(pose["quaternion_xyzw"])
    p = pose["position_m"]
    return [r[0] + [p[0]], r[1] + [p[1]], r[2] + [p[2]], [0.0, 0.0, 0.0, 1.0]]


def to_pose(m: Matrix, frame_id: str) -> dict:
    return {
        "frame_id": frame_id,
        "position_m": [m[0][3], m[1][3], m[2][3]],
        "quaternion_xyzw": rot_to_quat([row[:3] for row in m[:3]]),
    }


def mul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inv(m: Matrix) -> Matrix:
    rt = [[m[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(rt[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [rt[0] + [t[0]], rt[1] + [t[1]], rt[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def tr(x: float, y: float, z: float) -> Matrix:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def rz(deg: float) -> Matrix:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def column(m: Matrix, j: int) -> list[float]:
    return [m[0][j], m[1][j], m[2][j]]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def wrap_deg(a: float, period: float = 360.0) -> float:
    return (a + period / 2) % period - period / 2


def relative(t_world_a: Matrix, t_world_b: Matrix) -> Matrix:
    """``T_A_B`` from two poses in the same world frame."""
    return mul(inv(t_world_a), t_world_b)


def pose_delta(t_ref: Matrix, t_obs: Matrix) -> dict[str, float]:
    """Observed brick frame relative to the expected one, in the expected frame's axes:
    lateral x/y and z (m), yaw about expected z (deg), tilt between z axes (deg)."""
    d = relative(t_ref, t_obs)
    zc = max(-1.0, min(1.0, d[2][2]))
    return {
        "dx_m": d[0][3],
        "dy_m": d[1][3],
        "dz_m": d[2][3],
        "lateral_m": math.hypot(d[0][3], d[1][3]),
        "yaw_deg": math.degrees(math.atan2(d[1][0], d[0][0])),
        "tilt_deg": math.degrees(math.acos(zc)),
    }


# --------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class Cell:
    """Footprint's lowest-index corner (x, y), 0-based; layer 1 stands on the plate."""

    x: int
    y: int
    layer: int
    rotation: int

    @classmethod
    def from_dict(cls, d: Any) -> Cell:
        if not isinstance(d, dict):
            raise GeometryError(f"cell must be {{x, y, layer, rotation}}, got {d!r}")
        try:
            vals = [d["x"], d["y"], d.get("layer", 1), d.get("rotation", 0)]
        except KeyError as exc:
            raise GeometryError(f"cell needs x and y, got {d!r}") from exc
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            raise GeometryError(f"cell indices are integers (grid, not metres), got {d!r}")
        x, y, layer, rotation = vals
        if rotation % 360 not in ROTATIONS:
            raise GeometryError(f"cell rotation must be 0/90/180/270 deg, got {rotation}")
        return cls(x, y, layer, rotation % 360)

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "layer": self.layer, "rotation": self.rotation}


def footprint(bt: BrickType, rotation: int) -> tuple[int, int]:
    a, b = bt.studs
    return (b, a) if rotation % 180 == 90 else (a, b)


def cells(bt: BrickType, cell: Cell) -> list[tuple[int, int, int]]:
    fx, fy = footprint(bt, cell.rotation)
    return [(cell.x + i, cell.y + j, cell.layer) for i in range(fx) for j in range(fy)]


def top_in_plate(
    plate_studs: tuple[int, int], bt: BrickType, cell: Cell, bias_m: tuple[float, float]
) -> Matrix:
    """``T_plate_brickTop``: brick top centre (stud bases), brick axes."""
    pitch, h = bt.system.stud_pitch_m, bt.system.brick_height_m
    fx, fy = footprint(bt, cell.rotation)
    topleft = tr(-plate_studs[0] * pitch / 2 + bias_m[0], -plate_studs[1] * pitch / 2 + bias_m[1], 0)
    offset = tr(cell.x * pitch, cell.y * pitch, cell.layer * h)
    cx, cy = fx * pitch / 2, fy * pitch / 2
    c = math.cos(math.radians(cell.rotation))
    s = math.sin(math.radians(cell.rotation))
    centre = (c * cx + s * cy, -s * cx + c * cy)  # Rz(rotation)^-1 @ (cx, cy)
    return mul(mul(mul(topleft, offset), rz(cell.rotation)), tr(centre[0], centre[1], 0.0))


@dataclass(frozen=True)
class Plate:
    """A fixed baseplate: ``pose`` is the top-surface centre (stud bases), z out of the studs."""

    pose: dict
    studs: tuple[int, int]
    system: BrickSystem
    calibration_id: str
    bias_m: tuple[float, float] = (0.0, 0.0)

    @property
    def frame_id(self) -> str:
        return self.pose["frame_id"]


def seat_pose(plate: Plate, bt: BrickType, cell: Cell) -> dict:
    """World pose of the brick's bottom centre when seated (brick axes; z up the studs)."""
    t = mul(mul(matrix(plate.pose), top_in_plate(plate.studs, bt, cell, plate.bias_m)),
            tr(0.0, 0.0, -bt.system.brick_height_m))
    return to_pose(t, plate.frame_id)


def nearest_cell(plate: Plate, bt: BrickType, observed_seat: dict) -> tuple[Cell, dict[str, float]]:
    """Inverse of ``seat_pose``: nearest grid cell for an observed bottom-centre pose and the
    residual of the observation relative to that cell's seat pose."""
    t_pw = inv(matrix(plate.pose))
    t_ps = mul(t_pw, matrix(observed_seat))
    pitch, h = plate.system.stud_pitch_m, plate.system.brick_height_m
    yaw = math.degrees(math.atan2(t_ps[1][0], t_ps[0][0]))
    period = 90.0 if bt.studs[0] == bt.studs[1] else 180.0
    rotation = int(round(yaw / 90.0)) * 90 % 360
    if period == 180.0:
        rotation %= 180
    fx, fy = footprint(bt, rotation)
    x = round((t_ps[0][3] + plate.studs[0] * pitch / 2 - plate.bias_m[0]) / pitch - fx / 2)
    y = round((t_ps[1][3] + plate.studs[1] * pitch / 2 - plate.bias_m[1]) / pitch - fy / 2)
    layer = round(t_ps[2][3] / h) + 1
    cell = Cell(int(x), int(y), int(layer), rotation)
    res = pose_delta(matrix(seat_pose(plate, bt, cell)), matrix(observed_seat))
    res["yaw_deg"] = wrap_deg(res["yaw_deg"], period)
    return cell, res


# --------------------------------------------------------------------------- connections


@dataclass(frozen=True)
class PlacedBrick:
    brick_id: str
    type: BrickType
    cell: Cell


def occupied_cells(build: list[PlacedBrick]) -> dict[tuple[int, int, int], str]:
    out: dict[tuple[int, int, int], str] = {}
    for b in build:
        for c in cells(b.type, b.cell):
            out[c] = b.brick_id
    return out


def finger_cells(bt: BrickType, cell: Cell, close_axis: int, top_layer: int) -> list[tuple]:
    """Cells beside the brick's two walls the jaws go down into (closing along the brick's
    ``close_axis``), from its layer up to ``top_layer`` (computing-brick-pose's finger rows)."""
    fx, fy = footprint(bt, cell.rotation)
    # Brick axis close_axis maps to plate axis 0 or 1 depending on rotation.
    along = close_axis if cell.rotation % 180 == 0 else 1 - close_axis
    n_across = (fx, fy)[1 - along]
    span = (fx, fy)[along]
    out = []
    for z in range(cell.layer, top_layer + 1):
        for k in range(n_across):
            for off in (-1, span):
                c = [cell.x, cell.y]
                c[along] += off
                c[1 - along] += k
                out.append((c[0], c[1], z))
    return out


def connection_checks(
    plate: Plate,
    bt: BrickType,
    cell: Cell,
    build: list[PlacedBrick],
    close_axis: int,
    support_required_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    """Ordered precondition checks for placing ``bt`` at ``cell`` on a known build.

    Each entry ``{name, ok, detail}``. Mirrors computing-brick-pose's exits (out_of_plate,
    collides, unsupported, ungraspable) and checking-design's ground-connected rule, with
    the finger rows checked against every taller part of the build (damage on approach).
    """
    out: list[dict[str, Any]] = []
    if bt.system.name != plate.system.name:
        out.append(
            {"name": "interface_match", "ok": False,
             "detail": f"{bt.system.name} brick on a {plate.system.name} plate: studs do not mate"}
        )
        return out
    out.append({"name": "interface_match", "ok": True, "detail": bt.system.name})
    mine = cells(bt, cell)
    off = [c for c in mine if not (0 <= c[0] < plate.studs[0] and 0 <= c[1] < plate.studs[1])]
    ok = cell.layer >= 1 and not off
    out.append({"name": "on_plate", "ok": ok,
                "detail": "" if ok else f"layer {cell.layer} or cells off the plate {off[:2]}"})
    occ = occupied_cells(build)
    hits = sorted({occ[c] for c in mine if c in occ})
    out.append({"name": "collision_free", "ok": not hits,
                "detail": f"occupied by {hits}" if hits else ""})
    if cell.layer == 1:
        out.append({"name": "supported", "ok": True, "detail": "stands on the plate"})
    else:
        below = [c for c in mine if (c[0], c[1], c[2] - 1) in occ]
        frac = len(below) / len(mine)
        need = max(support_required_fraction, 1e-9)
        ok = frac >= need
        out.append({
            "name": "supported", "ok": ok,
            "detail": f"{len(below)}/{len(mine)} studs engage layer {cell.layer - 1}"
            + ("" if ok else f" (need >= {need:.2f})"),
        })
    top = max([c[2] for c in occ] + [cell.layer])
    blocked = sorted({occ[c] for c in finger_cells(bt, cell, close_axis, top) if c in occ})
    out.append({
        "name": "approach_clear_of_build", "ok": not blocked,
        "detail": f"jaws would strike {blocked}" if blocked else "",
    })
    return out


def tcp_target(t_world_brick: Matrix, t_tcp_object: Matrix) -> Matrix:
    """Hand pose that puts the held object at ``t_world_brick``:
    ``T_world_tcp = T_world_object @ inv(T_tcp_object)``."""
    return mul(t_world_brick, inv(t_tcp_object))


def choose_yaw(
    bt: BrickType, cell: Cell, t_world_seat_nominal: Matrix, t_world_tcp_now: Matrix,
    t_tcp_object: Matrix, ambiguity_deg: float = 10.0,
) -> tuple[int, list[dict[str, float]]]:
    """Among the pattern-symmetric yaws of the target, the one needing the least wrist turn
    from the current hand. Raises ``GeometryError`` if the two best are within
    ``ambiguity_deg`` of each other (no silent pick)."""
    cands = []
    for extra in bt.symmetric_yaws():
        t_seat = mul(t_world_seat_nominal, rz(extra))
        t_tcp = tcp_target(t_seat, t_tcp_object)
        d = relative(t_world_tcp_now, t_tcp)
        turn = abs(math.degrees(math.atan2(d[1][0], d[0][0])))
        cands.append({"extra_yaw_deg": float(extra), "wrist_turn_deg": turn})
    cands.sort(key=lambda c: c["wrist_turn_deg"])
    if len(cands) > 1 and cands[1]["wrist_turn_deg"] - cands[0]["wrist_turn_deg"] < ambiguity_deg:
        raise GeometryError(
            f"symmetric orientations {cands[0]['extra_yaw_deg']:.0f} and "
            f"{cands[1]['extra_yaw_deg']:.0f} deg need similar wrist turns "
            f"({cands[0]['wrist_turn_deg']:.1f} / {cands[1]['wrist_turn_deg']:.1f} deg); "
            "refusing to pick one silently"
        )
    return int(cands[0]["extra_yaw_deg"]), cands
