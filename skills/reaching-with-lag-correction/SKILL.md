---
name: reaching-with-lag-correction
description: Moves the tool to a target pose, waits for the arm to settle, reads
  the tool pose back and, if the arm stopped short (tracking lag, gravity sag),
  commands once more aimed past the target by the shortfall, then reports the
  residual. Use when a pose must be hit closer than the arm's own move bound
  allows - a look pose, a pre-grasp or pre-insert pose, a descent near the
  table, or anywhere far out in the reach where a joint controller without an
  integral term leaves a steady error - and as a reach check whose not_reached
  exit means the arm is blocked, stuck or at the edge of its reach.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [motion, reach, tracking-lag, sag, correction, verify]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion, cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/penn-pal-lab/maestro/blob/main/maestro/api.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
  required_inputs:
    target_pose: Se3Pose
  produces_outputs:
    residual: float
    residual_rotation: float
    first_residual: float
    first_residual_rotation: float
    corrected: bool
    correction: Vec3
    reached_pose: Se3Pose
  exit_conditions:
    reached: After at most one aim-past move the settled tool centre stands within tolerance (default 5 mm) of the target and its rotation within rot_tolerance (default 0.05 rad); residual is that distance.
    not_reached: The move was refused outright (the hand did not move), or the shortfall exceeded the correction cap and the plain retry also ended off, or the aim-past move did not land; `detail` says which and residual is how far off the hand stands.
  canonical_scripts:
    - reach_with_correction: scripts/reach_with_correction.py
  examples:
    - title: Reach a pose and correct a steady shortfall once
      path: examples/canonical_subgraph.json
---

# reaching-with-lag-correction

A position-controlled arm without an integral term leaves a steady error under
a steady load: the wrist sags, the shoulder lags. A second move aimed past the
target by the remaining error lands on it. This skill is MAESTRO's
`_hold_and_wait_for_reach` on a world-frame tool pose.

## The motion

1. Read the tool pose, move to `target_pose` (`robot.go_to_pose`, or
   `robot.go_to_pose_cartesian` with `cartesian=true`).
2. Settle: read the tool pose every `settle_interval` until two readings
   differ by under 0.3 mm (at most `settle_timeout`).
3. Error = target - reading (position), and the world-frame rotation from the
   reading to the target.
4. Under `correct_above` (2 mm) and `rot_correct_above` (0.02 rad): done.
   Otherwise one more move:
   - position error within `max_correction` (25 mm): aim at target + error;
     rotation error within `max_rot_correction` (0.1 rad): aim at the target
     turned on by the error. Each part is capped on its own.
   - over a cap, that part is retried as is (MAESTRO's rule).
   With `check_roll=false` the rotation error is the approach axis's tilt
   only, and its correction is the shortest turn that removes the tilt.
5. Settle, read back, route on the residual.

A move that raises or reports it did not arrive is still read back: a real YAM
station raises when the tool stops more than 12 mm short, and that is exactly
the case to correct. A failed move that left the hand where it was (under
1 mm of travel) was refused (IK, a floor or a workspace guard): it is not
aimed past or retried, and routes `not_reached`.

## Inputs

- `target_pose` (`Se3Pose`, world frame, the tool centre).
- `tolerance` (0.005 m), `rot_tolerance` (0.05 rad): the exit bounds.
- `correct_above` (0.002 m), `rot_correct_above` (0.02 rad): when to send the
  aim-past move. Below the exit bounds on purpose: the correction is one cheap
  move, and it removes the real arm's 2-3 mm near the table.
- `max_correction` (0.025 m), `max_rot_correction` (0.1 rad): the caps.
- `cartesian` (false): straight-line moves.
- `settle_timeout` (1.0 s), `settle_interval` (0.1 s).
- `check_roll` (true): judge and correct the full rotation. Set false on a
  connector whose IK leaves the roll about the approach axis free (the
  simulated YAM's does: it landed 1.1-2.0 rad off in roll on every pose); then
  only the approach axis's direction is judged and corrected.
- `servo_tolerance`: passed to `robot.go_to_pose` as `tolerance` only when set
  (gap / simulator connectors' arrival ball; a real station's motion server does
  not take it). On a connector with an arrival ball, set it under
  `correct_above`: a correction smaller than the ball does not move the arm
  (measured below). `arm_id`: passed only when nonzero.

## Outputs

`residual` / `residual_rotation` (final), `first_residual` /
`first_residual_rotation` (after the first move, what the arm would have
delivered without this skill), `corrected` (an aim-past move was sent),
`correction` (the position offset added to the target), `reached_pose` (the
settled reading).

## Required end states

- `reached`: the settled tool pose is within `tolerance` / `rot_tolerance` of
  the target.
- `not_reached`: refused, or still off after the second move. The hand stands
  where that move left it; `detail` has the move errors and the residual. A
  residual that grew after the correction suggests contact: the aim-past move
  pressed into whatever stopped the first.

## The constants, and why

- **`MAX_CORRECTION = 25 mm`**: MAESTRO caps at 0.1 rad per joint. On the real
  left YAM (2026-11-06) the tool stopped 16 and 18 mm short at 0.505 and
  0.509 m reach (the station reported 21 mm once); 25 mm covers that. A larger
  miss is more likely an obstacle, a refusal or the edge of reach than sag, and
  aiming past it pushes the hand into whatever stopped it.
- **`CORRECT_ABOVE = 2 mm`**: the same arm stood 1.7-3.2 mm over its command on
  a descent near the table at 0.26 m reach.
- **`TOLERANCE = 5 mm`, `ROT_TOLERANCE = 0.05 rad`**: above a servo connector's
  4 mm / 3 deg arrival ball, inside a real YAM station's 12 mm / 0.04 rad
  per-move bound (on rotation just above it).
- **Settle 0.3 mm / 0.1 s / 1 s**: MAESTRO waits for blocking moves and then
  sleeps 0.3 s; the polling wait returns as soon as the arm stops creeping.

## What it cannot see

The read-back is the connector's tool pose, forward kinematics of the joint
encoders. Lag and sag in the joint controller show there and are corrected.
Bending of the links or the gripper mount below the encoders does not show in
the reading; this skill does not correct it (a camera check would). The real
YAM's shortfalls above were the station's own read-back (it reported "the tool
did not reach its target (21 mm)"), so they are the kind this corrects.

Near the table the aim-past point lies below the target. A real YAM station
refuses a tool centre under 14 mm over its table: a target at 15 mm with 3 mm
of sag gives a 12 mm command, which is refused; the hand keeps its first
landing and the exit judges that (3 mm, `reached`, the refusal in `detail`).

## Measured in a YAM station twin

RoboSimStudio's YAM station twin, left arm, top-down hand, each reach from the
same home pose, `check_roll=false` (the twin's IK leaves the roll free). The
twin tracks its commands well, so the real arm's sag was added in the test
graph: every move was sent through a wrapper that raises the command by the
real left arm's measured sag at that reach (3.3 mm, then 3.1 mm per cm past
0.455 m from the base, the station's `arm_sag` model), so the arm lands that
much over the target, as the real one does. The skill sees only its commands
and its read-back. Final tool-centre distance from the target [mm], read
independently after the skill:

| pose (reach from base, azimuth, height over table) | twin | + sag, no correction | + sag, corrected | + sag, corrected, `servo_tolerance` 1 mm |
|---|---|---|---|---|
| near 0.25 m, 0 deg, 150 mm | 0.16 | 3.32 | 3.31 | 0.13 |
| near 0.30 m, +25 deg, 100 mm | 0.23 | 3.33 | 3.32 | 0.23 |
| near 0.30 m, -25 deg, 200 mm | 0.17 | 3.31 | 3.30 | 0.21 |
| low 0.30 m, 0 deg, 20 mm | 0.25 | 3.56 | 3.43 | 0.02 |
| low 0.40 m, +15 deg, 20 mm | 0.21 | 3.46 | 3.40 | 0.25 |
| low 0.45 m, -15 deg, 25 mm | 0.21 | 3.36 | 3.35 | 0.24 |
| far 0.45 m, 0 deg, 100 mm | 0.19 | 3.37 | 3.35 | 0.29 |
| far 0.48 m, +10 deg, 60 mm | 0.33 | 11.12 | 0.36 | 0.33 |
| far 0.50 m, 0 deg, 50 mm | 0.21 | 17.29 | 0.20 | 0.20 |
| far 0.52 m, -10 deg, 80 mm | 8.47 | 20.42 | 8.28 | 8.28 |

- Without the sag the twin lands within 0.4 mm and the correction has nothing
  to do (it fired once, at 0.52 m). Run with the correction the twin column
  reads 0.15-0.26 mm on the first nine poses.
- With the sag and no correction, 7 of 10 reaches end within the 5 mm bound
  (the 3.3 mm ones) and the two far ones end 11 and 17 mm off: the real
  arm's failure at 0.50 m. Corrected, those land within 0.36 mm (`reached`).
- **The servo's arrival ball hides small corrections.** The twin's
  `robot.go_to_pose` stops within 4 mm of its command by default, so an
  aim-past move 3.3 mm from where the hand stands does not move it (third
  column). With `servo_tolerance=0.001` the 3.3 mm sag is removed too
  (0.02-0.29 mm, mean height error 0.07 mm over the ten poses). A real
  station's joint-target moves have no such ball.
- **0.52 m is past the twin's reach** for that pose: the servo timed out on
  both moves (8.5 mm off without sag, 20.4 mm with it, 8.3 mm after the
  correction), and the skill routes `not_reached` with the residual and the
  timeouts in `detail`, as it should.
- Run as a graph node (the example's shape) at 0.50 m: 0.29 mm, `reached`,
  no correction; with the sag wrapper 17.2 mm after the first move, 0.06 mm
  after the correction, `reached`.
- A first run with `check_roll=true` routed every reach `not_reached`: the
  twin lands 1.1-2.0 rad off in roll (its IK leaves roll free), which is why
  `check_roll` exists. The position corrections were the same.

## Provenance

Adapted from MAESTRO's `_hold_and_wait_for_reach` (MIT, Copyright (c) 2026 The
MAESTRO Authors, `LICENSE.maestro`): a world-frame tool pose through robot.*
tools in place of a joint vector on the robot server; position and rotation
bounds instead of one joint-space norm; a settle wait; a refused or
stopped-short move read back and routed instead of raised; a correction
threshold separate from the exit bound; the pre-correction residual reported.
