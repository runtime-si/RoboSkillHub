# Adapted from Robotic_Lego_Manipulation
# (https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation),
# src/Lego.cpp (Lego::calc_brick_loc, Lego::calc_brick_grab_pose) and
# include/Lego.hpp (P_len_, brick_height_m_, brick_len_offset_, knob_height_) at
# commit 9bc91654e708e4d8428ecfa76840103cacb2a19a, MIT License.
# Changes: calc_brick_loc's chain (plate pose * top-left offset * cell offset *
# z_90 for ori 1 * brick-centre offset) is kept, generalised to yaw 0/90/180/270
# and to any pitch / brick height (LEGO and DUPLO presets). ICL subtracts
# brick_len_offset_ (0.2 mm) three times, which leaves every brick 0.2 mm off
# the stud lattice along plate -x and -y; here the lattice is exact and that
# shift is the optional grid_bias (ICL's numbers come back with
# grid_bias=[-0.0002, -0.0002]). ICL's pose is the brick's top centre; this
# returns the body centre (pose6d's box frame), the bottom centre (the seat
# pressing-a-brick-onto-studs takes) and the top. calc_brick_grab_pose's
# y_180 * press-side edge offset (their end-of-arm tool levers a brick by one
# edge, half a pitch off centre on odd sides) becomes a top-down parallel-jaw
# grasp at the centre, jaws across the short side, fingertips grasp_depth under
# the top (BrickSim's GRASP_DEPTH), rotation from robot.grasp_frame. Added: the
# inverse (a world pose, e.g. pose6d's, to the nearest cell with its residual),
# plate bounds, an occupied-cell check (collision, support, finger room), the
# jaw-opening check and an IK preview, all routed as exits.
"""Convert between a baseplate stud-grid cell and metric brick, grasp and press poses (and back)."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, float]] = {
    "lego": {
        "stud_pitch": 0.008,
        "brick_height": 0.0096,
        "stud_height": 0.0017,
        "stud_diameter": 0.0048,
        "grasp_depth": 0.005,
    },
    "duplo": {
        "stud_pitch": 0.016,
        "brick_height": 0.0192,
        "stud_height": 0.0045,
        "stud_diameter": 0.0094,
        "grasp_depth": 0.013,
    },
}
"""Brick systems [m]. LEGO: ICL's P_len_ 8 mm, brick_height_m_ 9.6 mm,
knob_height_ 1.7 mm; 4.8 mm studs. DUPLO: twice the pitch and height, 9.4 mm
studs 4.5 mm high (the values pressing-a-brick-onto-studs uses). grasp_depth:
fingertips under the brick's top - LEGO BrickSim's GRASP_DEPTH (5 mm), DUPLO the
13 mm the press skill's DUPLO simulation check grasped at."""

FOOTPRINT_PLAY = 0.0001
"""How much shorter than n pitches a brick is on each side [m]: ICL's
brick_len_offset_ (0.2 mm) split over both sides; a 2x4 LEGO is 15.8 x 31.8 mm."""

ONE_WIDE_GRASP_DEPTH_EXTRA = 0.001
"""Deeper grasp for 1-wide bricks [m]: BrickSim's 6 mm instead of 5 mm."""

TIP_REACH = 0.0102
"""How far closed linear_4310 fingertips reach below the tool centre [m] (the
YAM station's gripper; pressing-a-brick-onto-studs' TIP_REACH)."""

APPROACH_HEIGHT = 0.04
"""Pre-grasp / press hover over the grasp pose [m]: BrickSim's
ASSEMBLY_HEIGHT_OFFSET, the press skill's approach_height."""

JAW_MARGIN = 0.004
"""Opening the jaws need beyond the brick's width [m]: 2 mm a side, about a
YAM's repeatability without visual servoing."""

LATERAL_TOLERANCE_FRACTION = 0.25
HEIGHT_TOLERANCE_FRACTION = 0.25
YAW_TOLERANCE_DEG = 10.0
TILT_TOLERANCE_DEG = 10.0
"""Inverse: a pose further than this from its nearest cell is off_grid
(lateral per axis, in pitches; height, in brick heights; yaw and tilt [deg]).
Rounding never leaves more than half a pitch, so a quarter separates "on this
cell" from "between two"."""

FLAT_TOLERANCE_DEG = 5.0
"""Largest plate tilt from level [deg] for which robot.grasp_frame's world
top-down rotation is used; a more tilted plate gets a rotation along its normal."""

ROTATIONS = (0, 90, 180, 270)
REFERENCES = ("body_centre", "bottom", "top")
ROUTES = (
    "computed",
    "off_grid",
    "out_of_plate",
    "collides",
    "unsupported",
    "ungraspable",
    "unreachable",
)


class Output(TypedDict):
    route: str
    detail: str
    direction: str
    cell: dict[str, int]
    brick_size: list[int]
    footprint: list[int]
    cells: list[list[int]]
    brick_pose: dict[str, Any]
    top_pose: dict[str, Any]
    seat_position: dict[str, float]
    seat_yaw_deg: float
    brick_studs: list[int]
    system: str
    stud_pitch: float
    stud_height: float
    stud_diameter: float
    brick_height: float
    grasp_pose: dict[str, Any]
    grasp_pose_flipped: dict[str, Any]
    approach_pose: dict[str, Any]
    press_pose: dict[str, Any]
    close_heading_deg: float
    close_axis: str
    jaw_width: float
    pose6d_mesh: dict[str, list[float]]
    support_plane: list[float]
    residual: dict[str, float]
    collisions: list[list[int]]


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


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


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    if q[0] < 0.0:
        q = tuple(-v for v in q)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


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


def _se3(m: np.ndarray) -> dict[str, Any]:
    return {"position": _vec(m[:3, 3]), "rotation": _quat(m[:3, :3])}


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


def _tr(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _wrap(deg: float, period: float) -> float:
    return (deg + period / 2.0) % period - period / 2.0


def _size(brick_size: Any) -> list[int]:
    if isinstance(brick_size, str):
        parts = brick_size.lower().replace(" ", "").split("x")
    else:
        parts = list(brick_size)
    if len(parts) != 2:
        raise ValueError(
            f"brick_size must be two stud counts like '2x4' or [2, 4], got {brick_size!r}"
        )
    a, b = (int(p) for p in parts)
    if a < 1 or b < 1:
        raise ValueError(f"brick_size must be positive stud counts, got {brick_size!r}")
    return [a, b]


def _footprint(size: list[int], rotation: int) -> list[int]:
    return [size[1], size[0]] if rotation % 180 == 90 else [size[0], size[1]]


def _brick_top_in_plate(
    plate_studs: list[int],
    x: int,
    y: int,
    layer: int,
    rotation: int,
    size: list[int],
    pitch: float,
    body_h: float,
    bias: np.ndarray,
) -> np.ndarray:
    """ICL's calc_brick_loc in the plate frame: top-left offset * cell offset *
    yaw * brick-centre offset. (x, y) is the footprint's lowest-index corner
    cell at every yaw (ICL's ori 1 keeps it); the result is the brick's top
    centre (stud bases), axes along the brick's own x (size[0] studs) and y."""
    fx, fy = _footprint(size, rotation)
    topleft = _tr(
        -plate_studs[0] * pitch / 2.0 + bias[0], -plate_studs[1] * pitch / 2.0 + bias[1], 0.0
    )
    offset = _tr(x * pitch, y * pitch, layer * body_h)
    centre = np.linalg.inv(_rz(rotation)) @ np.array([fx * pitch / 2.0, fy * pitch / 2.0, 0.0, 1.0])
    return topleft @ offset @ _rz(rotation) @ _tr(*centre[:3])


def _cells(x: int, y: int, layer: int, footprint: list[int]) -> list[list[int]]:
    return [[x + i, y + j, layer] for i in range(footprint[0]) for j in range(footprint[1])]


def _occupied_cells(occupied: list[Any] | None) -> set[tuple[int, int, int]]:
    """Stud cells [x, y, layer] from a list of cells and / or placed bricks
    ({x, y, layer, brick_size, rotation} or this skill's own output)."""
    taken: set[tuple[int, int, int]] = set()
    for item in occupied or []:
        if isinstance(item, dict):
            if item.get("cells"):
                taken.update(tuple(int(v) for v in c) for c in item["cells"])
                continue
            c = item.get("cell", item)
            fp = _footprint(_size(item.get("brick_size", [1, 1])), int(c.get("rotation", 0)) % 360)
            taken.update(tuple(v) for v in _cells(int(c["x"]), int(c["y"]), int(c["layer"]), fp))
        else:
            vals = [int(v) for v in item]
            if len(vals) != 3:
                raise ValueError(f"an occupied cell is [x, y, layer], got {item!r}")
            taken.add((vals[0], vals[1], vals[2]))
    return taken


def run(
    ctx: NodeContext,
    plate_pose: dict[str, Any],
    plate_studs: list[int],
    brick_size: Any = "2x2",
    cell: dict[str, int] | None = None,
    observed_pose: dict[str, Any] | None = None,
    observed_reference: str = "body_centre",
    system: str = "duplo",
    stud_pitch: float | None = None,
    brick_height: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    plate_thickness: float = 0.0,
    grid_bias: list[float] | None = None,
    occupied: list[Any] | None = None,
    close_axis: str = "auto",
    grasp_depth: float | None = None,
    tip_reach: float = TIP_REACH,
    approach_height: float = APPROACH_HEIGHT,
    jaw_margin: float = JAW_MARGIN,
    use_grasp_frame: bool = True,
    check_gripper: bool = True,
    check_reach: bool = False,
    lateral_tolerance: float | None = None,
    height_tolerance: float | None = None,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    tilt_tolerance_deg: float = TILT_TOLERANCE_DEG,
    arm_id: int = 0,
) -> Output:
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if close_axis not in ("auto", "short", "long"):
        raise ValueError(f"close_axis must be 'auto', 'short' or 'long', got {close_axis!r}")
    if observed_reference not in REFERENCES:
        raise ValueError(
            f"observed_reference must be one of {REFERENCES}, got {observed_reference!r}"
        )
    if (cell is None) == (observed_pose is None):
        raise ValueError("give exactly one of cell (cell -> poses) or observed_pose (pose -> cell)")
    preset = PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    body_h = float(brick_height if brick_height is not None else preset["brick_height"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_d = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"])
    if pitch <= 0.0 or body_h <= 0.0:
        raise ValueError("stud_pitch and brick_height must be positive")
    plate = [int(v) for v in plate_studs]
    if len(plate) != 2 or min(plate) < 1:
        raise ValueError(f"plate_studs must be two positive stud counts, got {plate_studs!r}")
    size = _size(brick_size)
    bias = np.array([float(v) for v in (grid_bias or [0.0, 0.0])][:2])
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}

    # The plate's top surface (stud bases): plate_pose itself, or plate_thickness
    # up its normal when plate_pose is its underside.
    t_world_plate = _matrix(plate_pose) @ _tr(0.0, 0.0, float(plate_thickness))
    up = t_world_plate[:3, 2]

    residual = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0, "tilt_deg": 0.0, "lateral": 0.0}
    notes: list[str] = []
    off_grid = ""
    if observed_pose is not None:
        direction = "to_cell"
        t_plate_obs = np.linalg.inv(t_world_plate) @ _matrix(observed_pose)
        r = t_plate_obs[:3, :3]
        if r[2, 2] < 0.0:  # a box's 180 deg twin about x: same brick, z up
            r = r @ np.diag([1.0, -1.0, -1.0])
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(r[2, 2])))))
        yaw = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
        if size[0] == size[1]:
            rotation, yaw_res = 0, _wrap(yaw, 90.0)
        else:
            rotation = int(round(yaw / 90.0)) % 2 * 90
            yaw_res = _wrap(yaw - rotation, 180.0)
        fx, fy = _footprint(size, rotation)
        c = t_plate_obs[:3, 3] - np.array([bias[0], bias[1], 0.0])
        bottom = (
            float(c[2])
            - {"body_centre": body_h / 2.0, "bottom": 0.0, "top": body_h}[observed_reference]
        )
        x = int(round((float(c[0]) + plate[0] * pitch / 2.0) / pitch - fx / 2.0))
        y = int(round((float(c[1]) + plate[1] * pitch / 2.0) / pitch - fy / 2.0))
        layer = int(round(bottom / body_h)) + 1
        res_x = float(c[0]) - (-plate[0] * pitch / 2.0 + (x + fx / 2.0) * pitch)
        res_y = float(c[1]) - (-plate[1] * pitch / 2.0 + (y + fy / 2.0) * pitch)
        res_z = bottom - (layer - 1) * body_h
        residual = {
            "x": res_x,
            "y": res_y,
            "z": res_z,
            "yaw_deg": yaw_res,
            "tilt_deg": tilt,
            "lateral": math.hypot(res_x, res_y),
        }
        lat_tol = float(
            lateral_tolerance
            if lateral_tolerance is not None
            else LATERAL_TOLERANCE_FRACTION * pitch
        )
        h_tol = float(
            height_tolerance if height_tolerance is not None else HEIGHT_TOLERANCE_FRACTION * body_h
        )
        misses = []
        if max(abs(res_x), abs(res_y)) > lat_tol:
            misses.append(
                f"{res_x * 1e3:+.1f} / {res_y * 1e3:+.1f} mm from the cell (tolerance {lat_tol * 1e3:.1f})"
            )
        if abs(res_z) > h_tol:
            misses.append(
                f"{res_z * 1e3:+.1f} mm from layer {layer}'s height (tolerance {h_tol * 1e3:.1f})"
            )
        if abs(yaw_res) > float(yaw_tolerance_deg):
            misses.append(f"yawed {yaw_res:+.1f} deg off the rows")
        if tilt > float(tilt_tolerance_deg):
            misses.append(f"tilted {tilt:.1f} deg")
        off_grid = "; ".join(misses)
    else:
        direction = "to_world"
        try:
            x, y, layer = int(cell["x"]), int(cell["y"]), int(cell.get("layer", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cell must be {{x, y, layer, rotation}}, got {cell!r}") from exc
        rotation = int(cell.get("rotation", 0)) % 360
        if rotation not in ROTATIONS:
            raise ValueError(
                f"cell rotation must be 0, 90, 180 or 270 deg, got {cell.get('rotation')!r}"
            )

    fp = _footprint(size, rotation)
    cells = _cells(x, y, layer, fp)
    t_plate_top = _brick_top_in_plate(plate, x, y, layer, rotation, size, pitch, body_h, bias)
    t_world_top = t_world_plate @ t_plate_top
    t_world_body = t_world_top @ _tr(0.0, 0.0, -body_h / 2.0)
    seat = (t_world_top @ np.array([0.0, 0.0, -body_h, 1.0]))[:3]
    brick_x = t_world_top[:3, 0]
    seat_yaw = math.degrees(math.atan2(float(brick_x[1]), float(brick_x[0])))
    lens = [n * pitch - 2.0 * FOOTPRINT_PLAY for n in size]

    out: Output = {
        "route": "computed",
        "detail": "",
        "direction": direction,
        "cell": {"x": x, "y": y, "layer": layer, "rotation": rotation},
        "brick_size": size,
        "footprint": fp,
        "cells": cells,
        "brick_pose": _se3(t_world_body),
        "top_pose": _se3(t_world_top),
        "seat_position": _vec(seat),
        "seat_yaw_deg": seat_yaw,
        "brick_studs": size,
        "system": system,
        "stud_pitch": pitch,
        "stud_height": stud_h,
        "stud_diameter": stud_d,
        "brick_height": body_h,
        "grasp_pose": {},
        "grasp_pose_flipped": {},
        "approach_pose": {},
        "press_pose": {},
        "close_heading_deg": 0.0,
        "close_axis": "",
        "jaw_width": 0.0,
        "pose6d_mesh": {"box": [lens[0], lens[1], body_h]},
        "support_plane": [float(up[0]), float(up[1]), float(up[2]), -float(up @ seat)],
        "residual": residual,
        "collisions": [],
    }
    where = (
        f"{size[0]}x{size[1]} at cell ({x}, {y}), layer {layer}, rotation {rotation} deg on a "
        f"{plate[0]}x{plate[1]} {system} plate"
    )

    def finish(route: str, why: str) -> Output:
        out["route"] = route
        out["detail"] = "; ".join(s for s in (where, why, *notes) if s)
        return out

    # The grasp: top-down, jaws across the short side (or the long one when
    # that is blocked or asked for), fingertips grasp_depth under the top.
    depth = float(grasp_depth if grasp_depth is not None else preset["grasp_depth"])
    if grasp_depth is None and min(size) == 1:
        depth += ONE_WIDE_GRASP_DEPTH_EXTRA
    if not 0.0 < depth < body_h:
        raise ValueError(f"grasp_depth must be inside the brick's height, got {depth}")
    taken = _occupied_cells(occupied)
    short_i = 0 if size[0] <= size[1] else 1
    order = {"auto": [short_i, 1 - short_i], "short": [short_i], "long": [1 - short_i]}[close_axis]
    if size[0] == size[1] and close_axis == "auto":
        order = [0, 1]

    def finger_cells(axis_i: int) -> list[tuple[int, int, int]]:
        """Cells beside the brick's walls the fingers go down into (one row on
        each side along the closing axis, at the brick's layer and above)."""
        r = np.round(_rz(rotation)[:2, :2] @ np.eye(2)[axis_i]).astype(int)
        along = 0 if abs(r[0]) else 1
        side = []
        for z in {lyr for (_, _, lyr) in taken if lyr >= layer}:
            for k in range(fp[1 - along]):
                for lo_hi in (-1, fp[along]):
                    c = [x, y]
                    c[along] += lo_hi
                    c[1 - along] += k
                    side.append((c[0], c[1], z))
        return [c for c in side if c in taken]

    t_grasp_local = _tr(0.0, 0.0, -depth + float(tip_reach))
    grasp_at = (t_world_top @ t_grasp_local)[:3, 3]
    blocked: list[str] = []
    chosen = None
    for axis_i in order:
        hit = finger_cells(axis_i)
        if hit:
            blocked.append(
                f"the fingers across the brick's {'xy'[axis_i]} would land on cells {sorted(hit)}"
            )
            continue
        chosen = axis_i
        break
    axis_i = chosen if chosen is not None else order[0]
    close_dir = t_world_top[:3, axis_i]
    close_dir = close_dir - up * float(close_dir @ up)
    close_dir /= np.linalg.norm(close_dir) or 1.0
    heading = math.degrees(math.atan2(float(close_dir[1]), float(close_dir[0])))
    out["close_heading_deg"] = heading
    if size[0] == size[1]:
        out["close_axis"] = "xy"[axis_i]
    else:
        out["close_axis"] = "short" if axis_i == short_i else "long"
    out["jaw_width"] = lens[axis_i]

    rots: list[np.ndarray] = []
    level = math.degrees(math.acos(min(1.0, max(-1.0, float(up[2]))))) <= FLAT_TOLERANCE_DEG
    if use_grasp_frame and level:
        try:
            gf = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on_arm)
            rots = [
                _rot(gf[k])
                for k in ("rotation", "rotation_flipped")
                if isinstance(gf, dict) and gf.get(k)
            ]
        except Exception as exc:  # a connector without grasp_frame
            logger.info("computing-brick-pose: robot.grasp_frame unavailable (%s)", exc)
    if not rots:
        # Tool +z down the plate normal (the fingers), jaws along tool +y.
        z = -up
        yv = close_dir
        rots = [np.column_stack([np.cross(yv, z), yv, z])]
        rots.append(rots[0] @ np.diag([-1.0, -1.0, 1.0]))
        notes.append("grasp rotation from the plate normal (tool +z down, jaws along tool +y)")

    def pose_at(p: np.ndarray, r: np.ndarray) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = r, p
        return m

    def fill(r_main: np.ndarray, r_alt: np.ndarray | None) -> None:
        grasp = pose_at(grasp_at, r_main)
        out["grasp_pose"] = _se3(grasp)
        out["press_pose"] = _se3(grasp)
        out["approach_pose"] = _se3(pose_at(grasp_at + up * float(approach_height), r_main))
        out["grasp_pose_flipped"] = _se3(pose_at(grasp_at, r_alt)) if r_alt is not None else {}

    fill(rots[0], rots[1] if len(rots) > 1 else None)

    # Exits, most fundamental first.
    outside = [c for c in cells if not (0 <= c[0] < plate[0] and 0 <= c[1] < plate[1])]
    if layer < 1 or outside:
        why = (
            f"layer {layer} is under the plate's top"
            if layer < 1
            else (f"{len(outside)} of {len(cells)} cells lie off the plate (e.g. {outside[0][:2]})")
        )
        return finish("out_of_plate", why)
    if off_grid:
        return finish("off_grid", f"nearest cell, but {off_grid}")
    hits = [c for c in cells if tuple(c) in taken]
    if hits:
        out["collisions"] = hits
        return finish("collides", f"{len(hits)} of its cells are occupied (e.g. {hits[0]})")
    if taken and layer > 1 and not any((c[0], c[1], layer - 1) in taken for c in cells):
        return finish("unsupported", f"nothing occupies layer {layer - 1} under it")
    if chosen is None:
        return finish("ungraspable", "; ".join(blocked))
    if blocked:
        notes.append(blocked[0] + ", so the jaws close across the other side")

    if check_gripper:
        try:
            g = ctx.tool("robot.describe_gripper", **on_arm)
            fit = g.get("width_fit") or {}
            opening = float(fit.get("at_open_m") or g.get("span_m") or 0.0)
            if opening and opening < lens[axis_i] + float(jaw_margin):
                return finish(
                    "ungraspable",
                    f"the jaws open {opening * 1e3:.1f} mm; the brick is {lens[axis_i] * 1e3:.1f} mm "
                    f"across its {'xy'[axis_i]} side (+{jaw_margin * 1e3:.0f} mm margin)",
                )
        except Exception as exc:
            logger.info("computing-brick-pose: gripper not described (%s)", exc)

    if check_reach:
        ok = None
        for i, r in enumerate(rots):
            if _ik_ok(ctx, pose_at(grasp_at, r), on_arm) and _ik_ok(
                ctx, pose_at(grasp_at + up * float(approach_height), r), on_arm
            ):
                ok = i
                break
        if ok is None:
            return finish(
                "unreachable",
                "neither wrist solution of the grasp / approach passes the IK preview",
            )
        if ok:
            fill(rots[ok], rots[0])
            notes.append("the flipped wrist solution is the reachable one")

    return finish("computed", "")


def _ik_ok(ctx: NodeContext, m: np.ndarray, on_arm: dict[str, Any]) -> bool:
    try:
        reply = ctx.tool("robot.solve_ik", pose=_se3(m), **on_arm)
    except Exception as exc:
        if "unreachable" in str(exc).lower():
            return False
        logger.info("computing-brick-pose: no IK preview (%s)", exc)
        return True
    return not (
        isinstance(reply, dict)
        and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
    )
