---
name: moving-both-arms-together
description: Moves both hands to their target poses as one coordinated
  motion - both straight paths are split into the same number of short
  segments and the arms take turns (the lead alternating) so neither runs
  ahead, every waypoint of both arms is previewed with IK before anything
  moves, each arm's arrival is read back per segment, and, when asked, the
  hands' relative pose is held within a bound (the second hand's waypoints
  ride on the first's). Use when a bimanual station must move both hands at
  once - a two-handed carry or lift, bringing both hands to a handover or
  pour, mirrored reaches - through connectors that drive one arm per call.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, dual-arm, synchronized, carry, cartesian]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/move_dual_arm/robotwin/policy.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.solve_ik
    - robot.get_observation
    - robot.move_to_joints
  required_inputs:
    left_target: Se3Pose
  produces_outputs:
    segments: int
    max_relative_error_m: float
    max_relative_error_deg: float
    max_transient_error_m: float
    left_residual_m: float
    right_residual_m: float
    min_hand_distance_m: float
    roll_turns: int
    left_pose: Se3Pose
    right_pose: Se3Pose
    right_target_pose: Se3Pose
  exit_conditions:
    reached: Both hands arrived at every waypoint within reach_tolerance (read back from each arm), the last being their targets; with keep_relative the hands' relative pose stayed within the bound at every segment end.
    diverged: One hand stopped more than reach_tolerance from a waypoint (refused, raised, stalled), or with keep_relative the hands' relative pose left the bound at a segment end, after at least one hand had moved; both arms stop where they are, `detail` says which and where.
    unreachable: Nothing moved - a waypoint failed the IK preview, the planned tool centres come closer than min_hand_distance, the targets break the kept relative pose, the move needs more than max_segments, the first move was refused in place, both arm ids name one arm, or the connector reports no separate second arm.
  canonical_scripts:
    - move_both_arms: scripts/move_both_arms.py
  examples:
    - title: Move arm 0 and arm 1 to their targets together
      path: examples/canonical_subgraph.json
---

# moving-both-arms-together

Move both hands to their targets as one motion. The source (RoboRSI's
`move_dual_arm`) hands both targets to one cuRobo dual-arm plan so converging
arms do not jam each other; the hub's connectors drive one arm per call, so
here the two straight paths are cut into the same number of short segments
and the arms take turns.

## The motion

1. Refuse (`unreachable`) if the two arm ids are equal, or the second arm's
   pose reads as the first's (within 1 mm) or as no pose: a connector that
   clamps or ignores `arm_id` would move the first arm twice.
2. **Targets.** A target without a rotation keeps the hand's current one.
   With `keep_relative` and no `right_target`, the right target is the left
   target composed with the hands' current relative pose (hand 2 in hand 1's
   frame); a given `right_target` must keep that relative pose within the
   bound, or nothing moves.
3. **Segments.** `n` = the larger of the longer path over `max_step`
   (20 mm; with `keep_relative`, at most half of `relative_tolerance`) and the
   larger rotation over `max_step_deg` (10 deg; with `keep_relative`, half of
   `relative_tolerance_deg`). Both arms get `n` waypoints: straight lines and
   slerped rotations, so a short path takes short steps and both hands arrive
   together. With `keep_relative` the right hand's waypoints are the left's
   composed with the relative pose (an arc about the left hand when the pair
   turns). More than `max_segments` (60) is refused.
4. **Preflight, nothing moved yet.** Planned tool centres closer than
   `min_hand_distance` (80 mm), at segment ends or between turns, are refused.
   Every waypoint of both arms goes through `robot.solve_ik`: `success` false,
   a reported `position_error_m` over 12 mm (or, without one,
   `within_servo_tolerance` false), or a raise saying "unreachable" (as a real
   station does) refuses the move. A connector without `solve_ik` skips the
   preview.
5. **Interleaved execution.** Segment `k`: one arm moves to its waypoint
   (`robot.go_to_pose_cartesian`), its pose is read back, then the other arm;
   the leading arm alternates every segment. Each arm must read back within
   `reach_tolerance` (20 mm) of its waypoint, and within 12 mm when the move
   raised or replied a refusal (a segment is shorter than 20 mm, so a stuck
   step would otherwise pass); otherwise both stop (`diverged`, or
   `unreachable` if neither hand has moved 5 mm).
6. **Wrist roll.** A hand whose roll about its approach axis (tool z) reads
   more than 3 deg from the commanded one (with `keep_relative`, half of
   `relative_tolerance_deg`) has the rest turned on its last joint
   (`robot.move_to_joints`, up to two read-back turns per segment, the first
   learning the joint's sign), as stacking-objects and
   handing-over-between-arms do: a solver that keeps only the approach axis
   leaves the roll free (PORTING lesson 10), and a full-pose arm is already
   there. Without a joint reading, `move_to_joints`, or room on the joint
   (+-120 deg less 0.05 rad; no half-turn flip, which would twist a carried
   object) it carries on and says so in `detail`. `fix_roll=false` turns this
   off.
7. **Relative-pose check per segment.** At each segment end the hands'
   measured relative pose is compared with the plan's: the world-frame offset
   between the tool centres, and the relative rotation. `segment_errors` lists
   both, the leading arm, the tool-centre distance and each hand's rotation
   off its commanded one. With `keep_relative`, an error over
   `relative_tolerance` / `relative_tolerance_deg` stops both arms
   (`diverged`); without it the error is only reported.

## Inputs

- `left_target` (Se3Pose, world), `right_target` (Se3Pose or none with
  `keep_relative`), `left_arm` (0), `right_arm` (1).
- `keep_relative` (false), `relative_tolerance` (0.02 m),
  `relative_tolerance_deg` (5 deg).
- `max_step` (0.02 m), `max_step_deg` (10 deg), `max_segments` (60),
  `reach_tolerance` (0.02 m), `min_hand_distance` (0.08 m), `preview` (true),
  `fix_roll` (true).

## Outputs

`segment_errors` (per segment: `segment`, `lead_arm`, `relative_error_m`,
`relative_error_deg`, `hand_distance_m`), their maxima, `max_transient_error_m`
(the relative-position error after the leading arm's step, before the other
followed), each hand's final residual to its target, the closest tool-centre
distance read during the move, both final poses, the right target used and
`roll_turns` (last-joint turns made).

## Required end states

- `reached`: both hands at their targets within `reach_tolerance`.
- `diverged`: both arms stand where the failing segment left them, each
  within one segment of the other's plan; whatever they hold is as the last
  readings say. Nothing is undone.
- `unreachable`: neither hand moved more than 5 mm.

## The constants, and why

- **`MAX_STEP = 0.02` m**: while one arm moves the other stands, so the
  hands' relative pose is off by up to one step between turns; 20 mm keeps a
  free move's turns short and is the default relative bound. With
  `keep_relative` the step is halved against the bound so a lead never eats
  more than half of it.
- **`REACH_TOLERANCE = 0.02` m**: no tighter than a servo's 4 mm or a real
  YAM station's 12 mm; upstream accepts 0.03 m at the end.
- **`RELATIVE_TOLERANCE = 0.02` m / 5 deg**: a default only. Two real YAMs
  each allowed 12 mm can disagree by 24 mm; a carry should pass what the held
  object tolerates.
- **`MIN_HAND_DISTANCE = 0.08` m**: a YAM's open jaws span 94 mm, so tool
  centres closer than ~80 mm put the finger bodies in each other's reach. It
  is a tool-centre proxy, not a link clearance (the connectors offer none for
  arm against arm); lower it for hands meant to meet (handover).
- **`REFUSED_TOLERANCE = 0.012` m**: a real YAM station's per-move bound;
  a refused or raised step counts as arrived only this close, and an IK
  preview miss up to this is accepted (the twin's preview missed
  edge-of-reach waypoints by 4-7 mm, outside the servo's 4 mm ball, and the
  arms then ended 6.6-11.1 mm from them).
- **`ROLL_TOLERANCE_DEG = 3`**, **`TURN_TRIES = 2`**, **`ROLL_LIMITS =
  +-2.0944` rad**: the servo's own 3 deg bound, handing-over-between-arms'
  read-back turns, the YAM joint 6 range from its MJCF.
- **`MOVED = 0.005` m**: upstream's `left_moved` / `right_moved` threshold.

## What differs from the source

- One coupled cuRobo plan becomes interleaved straight-line segments, one
  arm per call. Neither arm runs more than one segment ahead, but the hands
  do not move at the same instant; a rigidly held object sees one step of
  relative error between turns (`max_transient_error_m`).
- Upstream's preflight plans each target from a neutral start and fails open
  when the checker errors; here every waypoint is previewed by IK from the
  live joints, and a connector without `solve_ik` skips it.
- No arm-against-arm collision check exists in the connectors; the tool
  centre floor stands in, and the sim check measures the real clearance.
- Arrival is judged by position; the wrist roll is finished on the last joint
  (step 6) because the YAM twin's IK leaves it free and, uncorrected, it
  drifted 5-32 deg per hand over 0.1-0.2 m translations with a constant
  commanded rotation (measured, see below).

## Connectors and arm_id

- RoboSimStudio: every robot verb used here takes `arm_id` (0 = work arm,
  1 = support arm); an id past the last arm is clamped to it, hence step 1.
- The real YAM station bridge (2026-11-06) drives only the left arm whatever
  `arm_id` says and reports no second arm: this skill refuses there
  (`unreachable`, "no separate arm") rather than moving one arm twice.

## Simulation check

RoboSimStudio YAM station twin, both YAM arms (bases 0.61 m apart; arm 0 =
left, arm 1 = right). The harness puts both hands top-down at the start
poses (`robot.go_to_pose`, read back within 10 mm), then runs this skill with
default parameters and targets given as positions only (each hand keeps its
rotation). It reads the hands again after a 60-step settle. The twin filters
the arms' collisions against each other (`sim.clearance` answers infinity),
so the arm clearance is the harness's own: the closest pair of points on the
two arms' collision meshes, polled every 20 ms during the skill. World
positions in metres; the bench top is at z ~0.75.

| move (left start -> target; right) | route | segments | per-segment relative error, max (mm / deg) | lead step (mm) | final residual L / R (mm) | wrist turns | min arm clearance |
|---|---|---|---|---|---|---|---|
| parallel +x 0.13: (0.45, +-0.25, 0.92) -> (0.58, +-0.25, 0.92) | reached | 7 | 1.5 / 4.7 | 19.0 | 2.0 / 3.2 | 2 | 347 mm (fingertips) |
| parallel +y 0.07 +z 0.10: (0.50, 0.20 / -0.30, 0.88) -> (0.50, 0.27 / -0.23, 0.98) | reached | 7 | 1.4 / 1.1 | 18.8 | 6.6 / 7.1 | 6 | 342 mm (fingertips) |
| mirrored inward and down: (0.50, +-0.28, 0.96) -> (0.52, +-0.12, 0.88) | reached | 9 | 3.8 / 6.0 | 23.2 | 1.6 / 1.5 | 14 | 84.6 mm (elbows, link4) |
| mirrored outward, forward, up: (0.44, +-0.15, 0.88) -> (0.58, +-0.26, 0.98) | reached | 11 | 2.9 / 11.3 | 20.6 | 9.0 / 8.4 | 13 | 138.6 mm (elbows) |
| 20 cm carry, `keep_relative`: (0.45, +-0.10, 0.90), left -> (0.56, 0.05, 0.97) | **diverged** at segment 14/14 | 14 | 6.1 / 6.6 | 13.5 | 11.1 / 5.9 | 7 | 23.0 mm (elbows) |
| 20 cm carry, `keep_relative`, shorter: left -> (0.52, 0.06, 0.93) | reached | 9 | 1.3 / 3.4 | 11.3 | 3.0 / 3.1 | 5 | 22.5 mm (elbows) |

- The hands' relative *position* stayed within 3.8 mm of the plan at every
  segment end in every move; between turns the leading hand was one step
  (11-23 mm) ahead, as designed.
- The relative *rotation* is the twin's free wrist roll. Before the
  last-joint roll finish (step 6) the same moves read 11.8, 1.5, 57.6 and
  63.2 deg of relative rotation at their last segment (each hand's roll
  drifting 5-32 deg off a constant command), and the 20 cm carry diverged at
  segment 3/14 on 6.0 deg. With it, what is left is under 3 deg of roll per
  hand plus, at the edge of reach, tilt the last joint cannot turn.
- The long carry's last waypoint is at the edge of the left arm's reach (it
  ended 11 mm short, 4.9 deg tilted), which put the relative rotation over
  the 5 deg bound on the final segment: the skill stopped and routed
  `diverged`, with the hands 203.7 mm apart. The shorter carry stayed within
  1.3 mm / 3.4 deg and ended 200.1 mm apart.
- With the hands 20 cm apart top-down the elbows (link4) are only 22-23 mm
  apart; the tool-centre floor (80 mm) does not see that. Keep a carry's
  hands wider apart or check the arms' clearance another way.
- Before the IK preview read `position_error_m`, it refused three of these
  moves on waypoints missed by 4-7 mm (outside the servo's 4 mm ball); the
  arms then ended 6.6-11.1 mm from those targets.
- Wall time 14-136 s per move in the twin, most of it in the last-joint
  turns (`robot.move_to_joints` settles slowly there).

## Provenance

Adapted from RoboRSI's `move_dual_arm`
(`roborsi/embodied/skills/base/move_dual_arm/robotwin/policy.py`,
Apache-2.0): both targets checked for reach before anything moves, each arm's
arrival read back against its own target, the 0.005 m "moved" threshold.
Changed: interleaved per-arm segments instead of one coupled cuRobo plan,
IK preview of every waypoint, the optional relative-pose bound for carries,
a tool-centre separation floor, routed exits.
