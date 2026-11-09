"""Injected faults (negatives) for the LEGO evaluation.

A fault changes what the privileged baseline commands for one placement; it never touches the clutch model,
the physics or the evaluator. The evaluator does not read the fault name: the expected failure types listed
here are only compared with its verdict afterwards. Each fault becomes a ``Perturbation`` the Isaac runner
applies to the waypoint targets in the goal frame of the faulted brick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .lego_clutch import STUD_PITCH_M


@dataclass(frozen=True)
class Perturbation:
    place_offset_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # goal frame, added to place and preplace
    place_yaw_rad: float = 0.0
    press: bool = True  # False: stop above the seat instead of pressing into it
    hover_above_seat_m: float = 0.0
    open_on_release: bool = True
    # Retreat horizontally towards this part's goal position instead of straight up.
    retreat_sweep_towards: str | None = None
    retreat_sweep_distance_m: float = 0.0
    retreat_sweep_height_units: float = 0.0  # above the seat, in brick heights
    skip_parts: tuple[str, ...] = field(default_factory=tuple)


# Which placement each fault targets per scene (index into the per-part plan) and what the evaluator should
# report. Several outcomes are acceptable where the physics decides (e.g. a yawed brick may self-align and
# then must be reported as engaged, not as the fault's label).
FAULTS: dict[str, dict] = {
    "lateral_offset": {
        "scenes": {"single_on_brick": 0, "single_on_plate": 0},
        "expect": ["lego.misaligned", "lego.not_seated"],
        "doc": "place 3 mm off the stud grid in x (position tolerance 2 mm, half pitch 4 mm)",
    },
    "angle_error": {
        "scenes": {"single_on_brick": 0, "single_on_plate": 0},
        "expect": ["lego.misaligned", "lego.not_seated"],
        "doc": "place with 9 deg yaw error (yaw tolerance 5 deg)",
    },
    "partial_engagement": {
        "scenes": {"single_on_brick": 0},
        "expect": ["lego.partially_seated"],
        "doc": "place one stud along y off the goal on the 2x4 below, so 6 of 8 studs can engage",
    },
    "not_fully_pressed": {
        "scenes": {"single_on_brick": 0, "single_on_plate": 0},
        "expect": ["lego.not_seated"],
        "doc": "stop 1.2 mm above the seat and release, no press force: arrival without clutch",
    },
    "insufficient_support": {
        "scenes": {"seq3_stack": 2},
        "expect": ["lego.insufficient_support"],
        "doc": "skip the supporting brick 1, then place brick 3 at its goal pose on top of it",
    },
    "lifted_on_release": {
        "scenes": {"single_on_brick": 0, "single_on_plate": 0},
        "expect": ["lego.lifted_with_gripper", "lego.structure_damaged"],
        "doc": "keep the gripper closed at release and retreat upwards",
    },
    "damaging_build": {
        "scenes": {"seq3_stack": 1},
        "expect": ["lego.structure_damaged"],
        "doc": "after placing brick 2, retreat sideways 30 mm towards brick 1 at half a brick above the seat",
    },
}


def perturbation(
    fault: str | None, cfg: dict, faulted_part_parents: tuple[str, ...] = ()
) -> Perturbation:
    if fault is None or fault == "none":
        return Perturbation()
    f = cfg["faults"][fault]
    if fault == "lateral_offset":
        return Perturbation(place_offset_m=(f["dx_m"], f["dy_m"], 0.0))
    if fault == "angle_error":
        return Perturbation(place_yaw_rad=math.radians(f["yaw_deg"]))
    if fault == "partial_engagement":
        sx, sy = f["shift_studs"]
        return Perturbation(place_offset_m=(sx * STUD_PITCH_M, sy * STUD_PITCH_M, 0.0))
    if fault == "not_fully_pressed":
        return Perturbation(press=False, hover_above_seat_m=f["hover_above_seat_m"])
    if fault == "insufficient_support":
        return Perturbation(skip_parts=tuple(faulted_part_parents))
    if fault == "lifted_on_release":
        return Perturbation(open_on_release=not f["keep_closed"])
    if fault == "damaging_build":
        return Perturbation(
            retreat_sweep_towards=f["towards_part"],
            retreat_sweep_distance_m=f["distance_m"],
            retreat_sweep_height_units=f["height_units"],
        )
    raise KeyError(f"unknown fault {fault!r}")
