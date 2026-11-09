"""Known-brick pose against a calibrated baseplate stud lattice.

Forward (cell to pose) is ``computing-brick-pose-from-stud-grid``'s
``_brick_top_in_plate`` (ICL's calc_brick_loc chain), loaded from the hub. The
inverse snaps a measured brick pose (e.g. from pose6d, in the plate frame that
``calibrating-baseplate-stud-grid`` measures) to the nearest cell and yaw, with
its residual, using the same tolerances as that skill.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import _hub
from .frames import inv_T, make_T, rot_z
from .objects import ObjectSpec


def brick_pose_in_plate(
    spec: ObjectSpec,
    plate_studs: tuple[int, int],
    x: int,
    y: int,
    layer: int,
    rotation_deg: int,
    grid_bias_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """``T_plate_brick`` (body centre) for a brick whose lowest-index footprint cell
    is (x, y) on ``layer`` (1 = on the plate's top, the skill's convention) at yaw
    ``rotation_deg``."""
    bp = _hub.load("brick_pose")
    m = spec.meta
    top = bp._brick_top_in_plate(
        list(plate_studs),
        int(x),
        int(y),
        int(layer),
        int(rotation_deg) % 360,
        list(m["studs"]),
        m["stud_pitch_m"],
        m["body_height_m"],
        np.asarray(grid_bias_m, dtype=float),
    )
    return top @ make_T(position=(0.0, 0.0, -m["body_height_m"] / 2.0))


def snap_to_lattice(
    T_plate_brick: np.ndarray,
    spec: ObjectSpec,
    plate_studs: tuple[int, int],
    grid_bias_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Nearest lattice cell / yaw for a measured brick pose, and whether it is on it.

    ``on_grid`` uses the brick-pose skill's tolerances (a quarter pitch laterally, a
    quarter brick height vertically, 10 deg yaw and tilt). Yaws that differ by the
    brick's symmetry (180 deg, or 90 for a square brick) are the same placement; the
    smallest is reported with the rest in ``equivalent_rotations``."""
    bp = _hub.load("brick_pose")
    m = spec.meta
    pitch, h = m["stud_pitch_m"], m["body_height_m"]
    R = T_plate_brick[:3, :3]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    period = 90 if spec.symmetry == "z4" else 180
    best = None
    for rot in bp.ROTATIONS:
        if rot >= period:
            continue
        yaw_err = abs(bp._wrap(yaw - rot, period))
        fx, fy = bp._footprint(list(m["studs"]), rot)
        top = T_plate_brick @ make_T(position=(0.0, 0.0, h / 2.0))
        cx = (top[0, 3] + plate_studs[0] * pitch / 2.0 - grid_bias_m[0] - fx * pitch / 2.0) / pitch
        cy = (top[1, 3] + plate_studs[1] * pitch / 2.0 - grid_bias_m[1] - fy * pitch / 2.0) / pitch
        layer_f = top[2, 3] / h
        cell = (round(cx), round(cy), round(layer_f))
        if best is None or yaw_err < best[0]:
            best = (yaw_err, rot, cell, (cx, cy, layer_f), (fx, fy))
    assert best is not None
    yaw_err, rot, (x, y, layer), (cx, cy, lf), (fx, fy) = best
    T_exp = brick_pose_in_plate(spec, plate_studs, x, y, layer, rot, grid_bias_m)
    d = inv_T(T_exp) @ T_plate_brick
    lateral = (abs(cx - x), abs(cy - y))
    in_plate = (
        x >= 0 and y >= 0 and x + fx <= plate_studs[0] and y + fy <= plate_studs[1] and layer >= 1
    )
    on_grid = (
        max(lateral) <= bp.LATERAL_TOLERANCE_FRACTION
        and abs(lf - layer) <= bp.HEIGHT_TOLERANCE_FRACTION
        and yaw_err <= bp.YAW_TOLERANCE_DEG
        and tilt <= bp.TILT_TOLERANCE_DEG
    )
    route = "on_grid" if on_grid and in_plate else ("out_of_plate" if on_grid else "off_grid")
    return {
        "route": route,
        "cell": {"x": int(x), "y": int(y), "layer": int(layer), "rotation_deg": int(rot)},
        "equivalent_rotations": [int(r) for r in range(rot, 360, period)],
        "residual": {
            "lateral_pitches": [round(v, 4) for v in lateral],
            "height_bricks": round(abs(lf - layer), 4),
            "translation_m": [round(float(v), 5) for v in d[:3, 3]],
            "yaw_deg": round(yaw_err, 3),
            "tilt_deg": round(tilt, 3),
        },
        "T_plate_brick_snapped": T_exp,
        "source": "computing-brick-pose-from-stud-grid._brick_top_in_plate (tolerances from the same skill)",
    }


def mating_target(
    T_world_plate: np.ndarray,
    spec: ObjectSpec,
    plate_studs: tuple[int, int],
    cell: dict[str, int],
    grid_bias_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """World pose of a brick seated at ``cell``: what Agent 3's press alignment
    consumes, with the brick's yaw equivalents (a held brick may be placed in any)."""
    T = T_world_plate @ brick_pose_in_plate(
        spec, plate_studs, cell["x"], cell["y"], cell["layer"], cell["rotation_deg"], grid_bias_m
    )
    period = 90 if spec.symmetry == "z4" else 180
    eq = [T @ make_T(rot_z(math.radians(a))) for a in range(0, 360, period)]
    return {"T_world_brick": T, "equivalent_T_world_brick": eq}
