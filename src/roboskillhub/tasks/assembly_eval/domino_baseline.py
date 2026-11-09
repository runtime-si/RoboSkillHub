# SPDX-License-Identifier: Apache-2.0
"""Scripted, privileged baseline for the domino scene (label: ``scripted_privileged``).

It reads the true pose of the tile to pick from the simulator (privileged) and plans a
straight-line hand trajectory: pre-grasp above the rack, vertical descent, close, lift,
carry at transit height, vertical descent to the believed table + release gap, open a
few millimetres wider than the tile, straight-up retreat, then hold for the observation
window. The hand moves only through joint drives; the planner never touches a tile.

This is an oracle-style reference for the evaluator and the scene, not Agent 02's
``place_upright`` skill; its numbers must not be credited to that skill.

Fault injections (``fault``) change only the commands, never the physics or the
observer (observer faults are applied by the backend's observer, see
``domino_runner.Observer``):

- ``missed_grasp``: grasp 20 mm off the tile along its facing (closes on air);
- ``wrong_release_pose``: release ``fault_offset`` (default 12 mm) along the row;
- ``knocked_at_release``: come in low from the free end and open the jaws while
  still moving toward the slot (0.15 m/s);
- ``retreat_sweeps_neighbour``: after the release, close the jaws and retreat
  horizontally along the row through the previous tile at pad height;
- ``table_height_error``: believe the table ``fault_offset`` (default +60 mm) higher;
- ``no_release``: never open; hold the tile at the slot through the window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .domino_geometry import facing_yaw
from .domino_scene import DominoSceneSpec

BASELINE_VERSION = "domino_scripted_privileged/0.1.0"
COMMAND_FAULTS = (
    "missed_grasp",
    "wrong_release_pose",
    "knocked_at_release",
    "retreat_sweeps_neighbour",
    "table_height_error",
    "no_release",
)
OBSERVER_FAULTS = ("stale_observation", "lost_target")
FAULT_DEFAULT_OFFSET = {
    "wrong_release_pose": 0.012,
    "table_height_error": 0.060,
    "missed_grasp": 0.020,
}


@dataclass
class Segment:
    phase: str
    pose: tuple[float, float, float, float]  # hand TCP x, y, z, yaw (table frame)
    opening: float  # commanded inner pad gap
    duration: float
    open_from: float = 0.0  # fraction of the segment after which the opening ramps
    ease: bool = True  # cosine easing; False keeps a constant speed to the end


@dataclass
class TilePlan:
    tile_index: int
    slot_index: int
    segments: list[Segment] = field(default_factory=list)
    fault: str | None = None
    notes: list[str] = field(default_factory=list)


def _wrap_yaw(yaw: float, ref: float) -> float:
    """The hand is symmetric under a half turn: pick the equivalent yaw nearest ``ref``."""
    best = yaw
    for k in range(-2, 3):
        cand = yaw + k * math.pi
        if abs(cand - ref) < abs(best - ref):
            best = cand
    return best


def _duration(spec: DominoSceneSpec, a, b, oa: float, ob: float, speed_scale: float = 1.0) -> float:
    g = spec.gripper
    lin = math.dist(a[:3], b[:3]) / (g.linear_speed * speed_scale)
    rot = abs(b[3] - a[3]) / g.yaw_speed
    fin = abs(ob - oa) / 2 / g.finger_speed
    return max(lin, rot, fin, 0.15)


def plan_tile(
    spec: DominoSceneSpec,
    tile_index: int,
    slot_index: int,
    tile_pose: dict,
    hand_start: tuple[float, float, float, float],
    opening_start: float,
    window_s: float,
    fault: str | None = None,
    fault_offset: float | None = None,
) -> TilePlan:
    """Plan one pick-and-place. ``tile_pose`` is the privileged true pose of the tile."""
    if fault is not None and fault not in COMMAND_FAULTS + OBSERVER_FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    off = fault_offset if fault_offset is not None else FAULT_DEFAULT_OFFSET.get(fault or "", 0.0)
    t, p = spec.tile, spec.place
    plan = TilePlan(tile_index, slot_index, fault=fault)
    z_transit = spec.transit_tcp_z()
    open_w = t.width + 2 * p.grip_clearance
    close_w = t.width - p.squeeze
    fx, fy, fz = tile_pose["position"]
    fyaw = _wrap_yaw(facing_yaw(tile_pose["quat_xyzw"]), hand_start[3])
    if fault == "missed_grasp":
        fx += math.cos(fyaw) * off
        fy += math.sin(fyaw) * off
        plan.notes.append(f"grasp offset {off * 1e3:.0f} mm along the facing")
    tile_bottom = fz - t.height / 2
    slot = spec.slot_pose(slot_index)
    sx, sy, _ = slot["position"]
    syaw = _wrap_yaw(slot["yaw"], fyaw)
    row = (math.cos(slot["yaw"]), math.sin(slot["yaw"]))
    believed_table = spec.table.top_z + (off if fault == "table_height_error" else 0.0)
    if fault == "wrong_release_pose":
        sx += row[0] * off
        sy += row[1] * off
    z_release = believed_table + p.release_gap + p.grasp_height

    segs: list[Segment] = []
    cur = hand_start
    cur_open = opening_start

    def go(phase, pose, opening, speed_scale=1.0, hold=0.0, open_from=0.0):
        nonlocal cur, cur_open
        d = _duration(spec, cur, pose, cur_open, opening, speed_scale) + hold
        segs.append(Segment(phase, pose, opening, d, open_from))
        cur, cur_open = pose, opening

    go("to_pregrasp", (cur[0], cur[1], max(cur[2], z_transit), cur[3]), open_w)
    go("to_pregrasp", (fx, fy, z_transit, fyaw), open_w)
    go("descend_pick", (fx, fy, tile_bottom + p.grasp_height, fyaw), open_w, speed_scale=0.4)
    go("close", cur, close_w, hold=0.4)
    go("lift", (fx, fy, z_transit, fyaw), close_w, speed_scale=0.5)
    if fault == "knocked_at_release":
        lead = 0.03
        lx, ly = sx + row[0] * lead, sy + row[1] * lead
        go("carry", (lx, ly, z_transit, syaw), close_w)
        go("descend_place", (lx, ly, z_release, syaw), close_w, speed_scale=0.4)
        segs.append(
            Segment(
                "release", (sx, sy, z_release, syaw), open_w, lead / 0.15, open_from=0.5, ease=False
            )
        )
        cur, cur_open = (sx, sy, z_release, syaw), open_w
        plan.notes.append("jaws open while moving 0.15 m/s toward the slot")
    else:
        go("carry", (sx, sy, z_transit, syaw), close_w)
        go("descend_place", (sx, sy, z_release, syaw), close_w, speed_scale=0.3)
        if fault == "no_release":
            go("release", cur, close_w, hold=0.3)
            plan.notes.append("jaws never open")
        else:
            go("release", cur, t.width + 2 * p.grip_clearance, hold=0.3)
    if fault == "retreat_sweeps_neighbour":
        back = (cur[0] - row[0] * 0.012, cur[1] - row[1] * 0.012, cur[2], cur[3])
        go("retreat", back, cur_open, speed_scale=0.3)
        go("retreat", back, 0.004, hold=0.1)
        sweep = (back[0] - row[0] * 0.05, back[1] - row[1] * 0.05, back[2], back[3])
        go("retreat", sweep, 0.004, speed_scale=0.5)
        go("retreat", (sweep[0], sweep[1], z_transit, sweep[3]), 0.004)
        plan.notes.append("closed jaws, horizontal retreat 62 mm along the row at pad height")
    elif fault == "no_release":
        go("retreat", cur, cur_open, hold=0.2)
    else:
        go("retreat", (cur[0], cur[1], z_transit, cur[3]), cur_open, speed_scale=0.5)
    go("observe", cur, cur_open, hold=window_s + 0.25)
    plan.segments = segs
    return plan


def plan_to_dict(plan: TilePlan) -> dict:
    return {
        "baseline": BASELINE_VERSION,
        "privileged": True,
        "tile_index": plan.tile_index,
        "slot_index": plan.slot_index,
        "fault": plan.fault,
        "notes": plan.notes,
        "segments": [
            {
                "phase": s.phase,
                "pose": list(s.pose),
                "opening": s.opening,
                "duration": s.duration,
                "open_from": s.open_from,
                "ease": s.ease,
            }
            for s in plan.segments
        ],
    }


def interpolate(seg: Segment, start_pose, start_open: float, u: float) -> tuple[tuple, float]:
    """Hand pose and opening at fraction ``u`` of ``seg`` (linear, cosine-eased)."""
    u = max(0.0, min(1.0, u))
    e = 0.5 - 0.5 * math.cos(math.pi * u) if seg.ease else u
    pose = tuple(a + (b - a) * e for a, b in zip(start_pose, seg.pose, strict=True))
    if seg.open_from > 0:
        v = max(0.0, (u - seg.open_from) / (1 - seg.open_from))
        opening = start_open + (seg.opening - start_open) * v
    else:
        opening = start_open + (seg.opening - start_open) * e
    return pose, opening
