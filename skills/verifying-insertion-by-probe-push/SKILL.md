---
name: verifying-insertion-by-probe-push
description: Checks whether a held part is seated after an insertion by pushing
  it a few millimetres further along the insertion axis (optionally also a
  little sideways) with a stop-on-contact move and reading where the hand
  stopped - no force sensor. Seated when the hand did not go further (and did
  not move sideways); not seated when it did. Use when an insertion,
  a placement into a pocket or a press-fit has just finished, the hand still
  holds the part, and no camera can see whether it went home.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, verify, contact, probe, peg-in-hole]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/o2ac/o2ac-ur/blob/main/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
  required_inputs: {}
  produces_outputs:
    travel: float
    lateral_travel: float
    start_position: Vec3
    stop_position: Vec3
    returned: bool
  exit_conditions:
    seated: The push along the axis moved the hand less than seat_tolerance (default 5 mm), and every sideways push (when lateral_distance > 0) less than lateral_tolerance; the hand went back to the start (return_to_start) and still holds the part.
    not_seated: The hand went at least seat_tolerance further along the axis, or a sideways push moved it lateral_tolerance or more; `travel` / `lateral_travel` say how far. The hand went back to the start unless return_to_start is false.
    blocked: No verdict - the hand's pose could not be read, the end of a push is out of reach by the IK preview, the tool-centre floor over table_z leaves no room for a push longer than seat_tolerance, or a push or the return to the start before a sideways push was refused.
  canonical_scripts:
    - probe_push: scripts/probe_push.py
  examples:
    - title: Check that a held part went home after an insertion
      path: examples/canonical_subgraph.json
---

# verifying-insertion-by-probe-push

A part that is home cannot go further; a part that hung up on a chamfer, a
burr or the rim of a pocket can. Push it a little further and see whether the
hand moves.

## The motion

1. Read the hand's pose: the start.
2. Work out the push: `probe_distance` along `axis` (world frame, default
   straight down). With `table_z`, shorten it so the tool centre stays
   `min_tool_height` over the table; if that leaves no more than
   `seat_tolerance`, exit `blocked`.
3. Ask `robot.solve_ik` whether the end of the push can be reached
   (`check_reach`); if not, exit `blocked`. A stall at the edge of reach looks
   exactly like a seated part.
4. Push with `robot.move_cartesian_until_contact`. Whatever ends the move -
   arrival, a contact or stall outcome, a timeout, or a station's "the tool did
   not reach its target" error - read the hand's pose. A refusal
   (`unreachable`, any other error) exits `blocked`.
5. The travel is the measured displacement along the axis. Under
   `seat_tolerance`: seated so far. Otherwise `not_seated`.
6. Seated and `lateral_distance > 0`: back to the start, then push
   `lateral_distance` in each of four directions square to the axis (world +x,
   -x, +y, -y for a vertical axis), back to the start before each. Any sideways
   travel of `lateral_tolerance` or more: `not_seated`.
7. Back to the start (`return_to_start`), which takes the press off.

The verdict is what the hand did, not a measurement of the part. A part that
slipped in the fingers, or a hand that was empty, reads like a part that went
further; check the grasp first if that matters (verifying-grasps).

For a pass-through check - o2ac's cable tip that must go *on* past a ring -
wire `not_seated` (it went further) as the success.

## Inputs (all optional)

- `axis` (Vec3, default `{x: 0, y: 0, z: -1}`): the insertion direction, world frame.
- `probe_distance` (10 mm), `seat_tolerance` (5 mm): see the constants. The
  probe must be shorter than the gap a part that is not home can have, and a
  free push must get well past `seat_tolerance`.
- `lateral_distance` (0, off), `lateral_tolerance` (3 mm).
- `table_z`, `min_tool_height` (15 mm): the tool-centre floor.
- `return_to_start` (true), `check_reach` (true), `arm_id` (0, passed only
  when nonzero).

## Preconditions

- The hand holds the part, which sits where the insertion left it. The skill
  never opens or closes the hand. An open, empty hand straddles things and
  sinks around them.
- The axis is the insertion's: a sideways error in it drags the part along
  its seat's wall.

## Outputs

`travel` (m, along the axis), `lateral_travel` (m, the largest sideways
travel, 0 when not pushed sideways), `start_position`, `stop_position` (where
the axial push stopped), `returned`, and `detail`.

## The constants, and why

- **`PROBE_DISTANCE = 10 mm`**: o2ac pushes up to 30 mm with a 15 N force
  limit. A position-controlled arm with no force limit presses a seated part
  for whatever is left of the probe, so the probe is short. In the twin a free
  push ended about 2 mm short of its end (the servo's arrival ball): 6.1 mm of
  8, 7.3 of 10, 10.4 of 12; 10 mm leaves 2.3 mm over `seat_tolerance`.
- **`SEAT_TOLERANCE = 5 mm`**: o2ac's `pose_dist < 0.005`. A seated cube let
  the hand go 1.2-3.7 mm further in the twin before the stall, whatever the
  probe length: the friction grip let the cube slide up the jaws by about
  that much.
- **`LATERAL_TOLERANCE = 3 mm`**: about a loose fit's clearance.
- **`MIN_TOOL_HEIGHT = 15 mm`**: a real YAM station refuses a tool centre
  under 14 mm over its calibrated table.
- **`RETURN_TOLERANCE = 15 mm`**: no tighter than a real YAM station's own
  12 mm per-move bound.

## Measured in a YAM station twin

RoboSimStudio's YAM station twin (left arm, linear_4310 hand, MuJoCo), a
30 mm cube and the open 100 x 100 x 40 mm box. A setup script grasped the
cube top-down (`sim.get_object_obb`, `robot.grasp_frame`; its bottom hung
22.1 mm under the tool centre), carried it over the box's centre and either
lowered it with a contact move until it rested on the floor (*seated*) or
held it 15.6 mm above its resting height (*raised*, a 1 mm-tolerance
`robot.go_to_pose`). Then this skill ran with default inputs and
`table_z` = the table (0.752 m); `sim.get_object_obb` measured the cube
before and after.

| case | runs | verdict | asked push | hand travel | stop, tool centre over the box floor | cube after the return |
|---|---|---|---|---|---|---|
| seated | 3 | seated x3 | 8.3-8.4 mm (floored over the table) | 3.5, 3.7, 3.7 mm | 18.4-18.7 mm (start 22.1) | 1.1 mm off the floor, ~3.3 mm higher in the jaws |
| raised 15.6 mm | 3 | not_seated x3 | 10.0 mm | 7.3, 7.3, 7.3 mm | 30.7 mm (start 38.0) | 2.1 mm lower (the return stopped 1.7 mm short), same place in the jaws |

Six of six correct. The press is not free: in the seated runs the cube
ended about 3.3 mm higher in the jaws than before the check (the hand pressed
3.5-3.7 mm into a part that could not go), and the return to the start left
it hanging 1.1 mm off the floor. The twin is close to deterministic; the spread is in
the exploratory runs: an earlier seated run (same setup, probe 8 mm) went
1.2 mm, and a raised run with 8 / 12 mm probes went 6.1 / 10.4 mm. In the
seated case the floor over the table (15 mm) cut the push to 8.3 mm; at 10 mm
the stop would still be a stall (the twin's seated travel was 3.6-3.7 mm for
asked pushes of 8.3 mm under 8, 10 and 12 mm probes).

With `lateral_distance` = 6 mm on the seated cube the verdict was
`not_seated`: the sideways pushes went up to 4.5 mm, since a cube on the floor
of a box three times its width is not held sideways. That is the sideways
check doing its job (it asks whether the seat holds the part sideways, which
a wide pocket does not), not a misreading; it also left the cube 3.4 mm
higher in the jaws. Use the sideways pushes only for fits.

**Real YAM station.** A station reports a move reached inside 12 mm (its
sag allowance), so an 8-10 mm push into a seat may come back "reached"
without the hand moving; only the measured travel is used, never the
outcome. A stall that ends in the station's "did not reach its target"
error is read as a stop. `robot.move_cartesian_until_contact` must be
served by the connector. Not run on hardware.

## Provenance

Adapted from o2ac-ur's motor-cable insertion check
(`insert_motor_cables_without_tools_normal`, in
`catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py`, MIT, Team O2AC,
World Robot Summit 2020 assembly): after inserting, `linear_push` 15 N along
the tool's -Z for at most 30 mm, then compare the tip's pose before and after;
under 5 mm means it did not go in. Changed: the force-controlled push is a
`robot.move_cartesian_until_contact` move judged by the measured stop (the
YAM has no force-torque sensor), along a world axis; the travel is measured
along the axis rather than as a pose distance; the verdict is named by the
part's state (seated / not_seated) so either reading can be wired as success;
the sideways pushes, the IK preview, the floor over the table, the return to
the start and the `blocked` exit are new.
