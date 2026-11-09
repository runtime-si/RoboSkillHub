"""BrickSim's clutch ("snap") model, restated so it can be read, tested and cited without the simulator.

Source: intelligent-control-lab/BrickSim@76815250d4157217807f0525ebbb6b376343e448 (MIT), the commit the
bricksim 0.1.0.dev570+g7681525 wheel pinned by RoCo_BrickAssembly@8e7ad4da was built from.

Engagement (assembly.cppm ``AssemblyChecker::detect_assembly``, called from physics_graph.cppm
``process_assembly_contacts`` inside the PhysX contact callback): for every contact pair between a stud
interface collider of one brick and a hole interface collider of another, the summed contact impulse divided
by the step dt gives a force. A connection segment is created only if, in the same step, all hold:

- vertical gap of the hole interface above the stud interface minus the stud height is in
  ``[-max_penetration, distance_tolerance]``;
- tilt between the two interface z axes <= ``z_angle_tolerance``;
- the force projected on the stud z axis pushes the hole brick down by >= ``required_force``;
- yaw error to the nearest quarter turn <= ``yaw_tolerance``;
- in-plane distance of the hole origin to the nearest stud grid point <= ``position_tolerance``;
- the snapped footprints overlap by at least one stud in both directions.

The engine then adds a PhysX weld constraint at the *snapped* grid pose (offset, quarter-turn yaw); the brick is
pulled onto the seated pose, so insertion depth is not simulated after the trigger.

Release (breakage.cppm ``BreakageChecker::solve`` + breakage_selector.cppm ``select_conns_to_break``): each step
a QP over every connected component balances the bricks' inertia change and external impulses with contact and
clutch forces. A clutch's utilisation is the used share of its friction capacity (``friction_coefficient`` x
``preloaded_force`` scaling); if any utilisation is >= 1, the minimum cut through the overloaded clutches is
removed and their welds released. A component is not re-checked for ``breakage_cooldown_time`` after it changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STUD_PITCH_M = 0.008
PLATE_HEIGHT_M = 0.0032
BRICK_HEIGHT_M = 3 * PLATE_HEIGHT_M
STUD_HEIGHT_M = 0.0017


@dataclass(frozen=True)
class AssemblyThresholds:
    distance_tolerance_m: float = 0.001
    max_penetration_m: float = 0.005
    z_angle_tolerance_rad: float = math.radians(5.0)
    required_force_n: float = 1.0
    yaw_tolerance_rad: float = math.radians(5.0)
    position_tolerance_m: float = 0.002

    @classmethod
    def from_config(cls, cfg: dict) -> AssemblyThresholds:
        a = cfg["clutch_model"]["assembly"]
        return cls(
            distance_tolerance_m=a["distance_tolerance_m"],
            max_penetration_m=a["max_penetration_m"],
            z_angle_tolerance_rad=math.radians(a["z_angle_tolerance_deg"]),
            required_force_n=a["required_force_n"],
            yaw_tolerance_rad=math.radians(a["yaw_tolerance_deg"]),
            position_tolerance_m=a["position_tolerance_m"],
        )


def nearest_quarter_turn(yaw: float) -> tuple[int, float]:
    k = round(yaw / (math.pi / 2.0))
    return k % 4, yaw - k * (math.pi / 2.0)


def footprint_overlap(
    stud_lw: tuple[int, int], hole_lw: tuple[int, int], offset: tuple[int, int], yaw_q: int
) -> tuple[int, int]:
    """Studs shared along each axis when a hole interface sits at a grid offset and quarter-turn yaw."""
    L, W = hole_lw
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][yaw_q % 4]
    corner = (offset[0] + c * L - s * W, offset[1] + s * L + c * W)
    lo = (min(offset[0], corner[0]), min(offset[1], corner[1]))
    hi = (max(offset[0], corner[0]), max(offset[1], corner[1]))
    ox = max(0, min(stud_lw[0], hi[0]) - max(0, lo[0]))
    oy = max(0, min(stud_lw[1], hi[1]) - max(0, lo[1]))
    return ox, oy


def detect_assembly(
    rel_p: tuple[float, float, float],
    rel_tilt: float,
    rel_yaw: float,
    force_along_stud_z_n: float,
    stud_lw: tuple[int, int],
    hole_lw: tuple[int, int],
    th: AssemblyThresholds | None = None,
) -> dict:
    """The trigger test for one stud/hole contact, given the hole interface pose in the stud interface frame.

    ``force_along_stud_z_n`` is the contact force on the hole brick projected on the stud z axis (negative =
    pressing down). Returns the measured terms, the per-condition verdicts and the snapped connection.
    """
    th = th or AssemblyThresholds()
    rel_dist = rel_p[2] - STUD_HEIGHT_M
    yaw_q, yaw_err = nearest_quarter_turn(rel_yaw)
    gx, gy = rel_p[0] / STUD_PITCH_M, rel_p[1] / STUD_PITCH_M
    sx, sy = round(gx), round(gy)
    p_err = math.hypot(gx - sx, gy - sy) * STUD_PITCH_M
    ov = footprint_overlap(stud_lw, hole_lw, (sx, sy), yaw_q)
    checks = {
        "gap": -th.max_penetration_m <= rel_dist <= th.distance_tolerance_m,
        "tilt": rel_tilt <= th.z_angle_tolerance_rad,
        "force": force_along_stud_z_n <= -th.required_force_n,
        "yaw": abs(yaw_err) <= th.yaw_tolerance_rad,
        "position": p_err <= th.position_tolerance_m,
        "overlap": ov[0] > 0 and ov[1] > 0,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "relative_distance_m": rel_dist,
        "position_error_m": p_err,
        "yaw_error_rad": yaw_err,
        "offset": (sx, sy),
        "yaw_q": yaw_q,
        "overlap_studs": ov,
    }


def gap_window_above_seat_m(th: AssemblyThresholds | None = None) -> tuple[float, float]:
    """Heights of the hole interface above the fully seated position at which the trigger can fire.

    Seated means the hole interface lies on the stud interface (relative z = 0, ``rel_dist = -stud height``).
    """
    th = th or AssemblyThresholds()
    return (
        -th.max_penetration_m + STUD_HEIGHT_M,
        th.distance_tolerance_m + STUD_HEIGHT_M,
    )
