---
name: turning-by-wrist-ratchet
description: Tightens or loosens a screw lid or cap by wrist ratchet - grips it
  from above, rolls only the last joint through a 218 deg stroke (joint 6
  -1.9 -> +1.9 rad on a YAM) in speed-bounded steps, opens, rolls back,
  re-grips and repeats for a number of strokes or until the turn stops, read
  from joint 6's tracking lag (no force sensor), the gripper reading and the
  tool centre's drift; after loosening it lifts to check the cap came free.
  Use when a jar lid, bottle cap or knob held on a thread (or a twist-lock)
  must be turned on or off and its box is known from perception, and the jar
  or bottle is held by the other arm or a fixture.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [screw, unscrew, lid, cap, wrist-roll, ratchet, regrasp, household]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [contact-assembly]
    provenance:
      authors: [runtime-si]
      code: https://github.com/zzigak/yam-arcade/blob/main/stations/jars/robot_demo.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.grasp_frame
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
    - robot.close_gripper
  required_inputs:
    lid_obb: OrientedBoundingBox
  produces_outputs:
    direction: str
    strokes_done: int
    total_turn: float
    seated: bool
    hand_drift: float
    lift_rise: float
    holding: bool
    rolled_with: str
  exit_conditions:
    turned: "Tighten: every stroke ran, or joint 6 stalled after the strokes had turned at least min_turn in all (seated: true); the hand opened and rose. Loosen: every stroke ran and the lift check rose lift_height with the jaws still holding (or lift_check was off); the hand keeps the cap. What the hand did; re-perceive to know the lid's state."
    stalled: Joint 6 lagged its target by more than stall_lag and stopped turning (a tightening stall before min_turn, or any loosening stall), a stroke or roll-back step was refused, the tool centre drifted past drift_tolerance, or the loosening lift check rose short (the cap still on its thread); the hand opened and rose. `detail` says which.
    slipped: The gripper reading fell under empty_grip (jaws closed on nothing) at a grip, during a stroke, or after the loosening lift; the hand opened and rose.
    unreachable: The pregrasp over the lid, the turn of joint 6 to the stroke's start, or the descent to the grip point was refused or ended off, the stroke does not fit joint 6's limits, or there is no joint reading; nothing was turned.
  canonical_scripts:
    - turn_wrist_ratchet: scripts/turn_wrist_ratchet.py
  examples:
    - title: Tighten or loosen a perceived lid
      path: examples/canonical_subgraph.json
---

# turning-by-wrist-ratchet

Screw a lid on or off with the wrist alone, a stroke at a time.

## The motion

1. Over the lid: the grip point is `lid_obb`'s centre, `grip_depth` under its
   top; the hand opens and goes `approach_height` over it, top-down
   (`rotation`, else `robot.grasp_frame`'s `rotation` then
   `rotation_flipped`).
2. Joint 6 turns, hand open, to the stroke's start: -1.9 rad to tighten,
   +1.9 to loosen (`stroke_limits`, `tighten_sign`).
3. Straight down to the grip point and close. Jaws that close on nothing are
   `slipped`.
4. A stroke: joint 6 to the other end (3.8 rad, 218 deg) in steps of at most
   `step_rad`, each taking at least `step_rad / max_speed` (0.5 rad/s), every
   other joint held (or walked to an IK solution `pitch` per turn lower when
   tightening, higher when loosening). After every step the skill reads joint
   6, the gripper and the tool centre back:
   - joint 6 more than `stall_lag` behind its target: the target is held, not
     advanced, for `stall_s`; if the joint turned less than 0.05 rad/s in that
     time the turn has stopped, and joint 6 is commanded at its reading to
     take the push off;
   - the gripper reading under `empty_grip`: the lid left the jaws;
   - the tool centre more than `drift_tolerance` sideways from the grip point:
     the jar or the hand is being dragged.
5. Open, roll joint 6 back to the start with the hand open, close again, and
   do the next stroke, `strokes` times (default 4 to tighten, 5 to loosen).
6. Tightening ends by opening and rising `retreat_height`. Loosening lifts
   `lift_height` still closed: a hand that rises with the jaws holding has the
   cap; one that rises short has a cap still on its thread; one whose jaws
   closed on nothing slipped off it.

## Inputs

- `lid_obb`: the lid's or cap's oriented box, world frame (half-extents).
- `direction` (`tighten`): `tighten` or `loosen`.
- `strokes` (0): strokes to make; 0 takes 4 to tighten, 5 to loosen.
- `stroke_limits` (`[-1.9, 1.9]`): joint 6 at the two stroke ends [rad].
- `tighten_sign` (+1): +1 when joint 6 rising tightens (clockwise seen from
  above on a YAM with a top-down hand), -1 for a left-hand thread or an arm
  whose last joint turns the other way.
- `pitch` (0 m): the thread's pitch, for the hand to follow it; 0 holds the
  height (the pads then slide along the lid by the pitch per turn).
- `grip_depth` (0.010 m), `approach_height` (0.08 m), `lift_height`
  (0.025 m), `retreat_height` (0.025 m): see the constants.
- `rotation` (none): the hand's rotation over the lid.
- `lift_check` (true): lift after loosening.
- `roll_with` (`auto`): `joints` steps with `robot.move_to_joints`; `pose`
  turns the tool about its own axis with `robot.go_to_pose` and learns the
  sign from the first step; `auto` uses joints and falls back to the pose roll
  when the first joint move is refused with the arm unmoved.
- `roll_limits` (`[-2.0944, 2.0944]`): joint 6's range, a YAM's by default.
- `step_rad` (pi/40), `max_speed` (0.5 rad/s), `stall_lag` (0.12 rad),
  `stall_s` (0.25 s), `min_turn` (0.12 rad), `track_tolerance` (0.075 rad),
  `empty_grip` (0.10), `drift_tolerance` (0.02 m), `reach_tolerance` (0.02 m).
- `arm_id` (0), `tolerance`, `max_steps`: passed to the connector only when
  set.

## Outputs

`turn_per_stroke`, `q6_trajectory`, `grip_trajectory`, `trajectory_phase`
and `grip_at_close` are returned but not listed under `produces_outputs`:
the type registry names no bare list.

- `strokes_done`, `turn_per_stroke` (joint 6's measured travel each stroke,
  + in the commanded direction [rad]), `total_turn` (their sum).
- `seated`: a tightening stall after `min_turn`.
- `q6_trajectory`, `grip_trajectory`, `trajectory_phase`: joint 6 [rad] and
  the gripper reading after every step, labelled `start`, `to_start`,
  `gripped`, `close`, `stroke<k>`, `stroke<k>_hold`, `back<k>`, `lift`.
- `grip_at_close`: the gripper reading after each close.
- `hand_drift`: the tool centre's largest sideways distance from the grip
  point over the strokes [m]. `lift_rise`: how far the loosening lift rose.
- `holding`: whether the jaws are closed on the lid at the end (a loosened
  cap stays in the hand).
- `rolled_with`: `joints` or `pose`.

## Preconditions

- The jar or bottle is held (the other arm, holding-steady-while-other-arm-acts,
  or a fixture). A free jar turns with the lid; on a joint-6 reading it looks
  like an easy turn, and only the drift check or a re-perception sees it.
- A lid is round: the grip heading does not matter, and joint 6 sets it.
- Tightening assumes the lid sits on the thread start. Upstream picked the
  hand yaw so the simulated thread landed one turn out; a real lid needs a
  short reverse turn to find the start, not part of this skill.

## No force sensor

yam-arcade capped the wrist motor's torque at 0.25 N m during a stroke (the
cap was the tightening torque) and ended the stroke on a stall (joint 6 more
than 0.12 rad behind its reference and turning under 0.05 rad/s for 0.25 s)
or when the simulated lid stopped following the wrist. A connector exposes
neither a torque cap nor the lid's yaw, and neither the gap connector nor a
YAM station publishes joint efforts, so:

- the stall rule is kept on joint 6's reading, with the target held rather
  than advanced while it lags, so the position loop's target never runs more
  than `stall_lag` and one step past the stop and is released once the stall is called;
- the lid-stopped-following test becomes the gripper reading (the jaws close
  on nothing) and the tool centre's drift.

The tightening torque is then whatever the arm's position loop exerts at a
0.12 rad lag, not a set value: on a real YAM test it on a lid that can take it.

## The constants, and why

- **`Q6_A, Q6_B = -1.9, +1.9` rad**: yam-arcade's stroke, 218 deg inside the
  YAM wrist's +-2.094 rad (RoboWaiter rolled -pi to pi on a wrist that allows
  it).
- **`STROKES` 4 / 5**: yam-arcade's `twist_on` loop and RoboWaiter's
  `untwist` loop.
- **`MAX_SPEED = 0.5` rad/s**: yam-arcade strokes at 1.6 rad/s in simulation;
  a real YAM station's supervisor trips at 0.75 rad/s measured. A stroke takes
  7.6 s.
- **`STEP_RAD = pi/40`**: as pouring-by-wrist-roll (49 steps a stroke).
- **`STALL_LAG = 0.12` rad, `STALL_S = 0.25` s, `STALL_SPEED = 0.05` rad/s**:
  yam-arcade's stall rule.
- **`MIN_TURN = 0.12` rad**: not tuned, equal to the stall lag.
- **`TRACK_TOLERANCE = 0.075` rad**: a real YAM station's settle tolerance,
  for the open-hand rolls.
- **`EMPTY_GRIP = 0.10`**: in the YAM station twin the gripper read 0.33 on
  the 30 mm cube and 0.05 closed empty.
- **`GRIP_DEPTH = 0.010` m**: closed linear_4310 fingertips reach about
  10 mm under the tool centre.
- **`APPROACH_HEIGHT = 0.08` m**: yam-arcade's pregrasp height.
- **`LIFT_HEIGHT`, `RETREAT_HEIGHT` = 0.025 m**: yam-arcade's release rise.
  **`LIFT_TOLERANCE = 0.010` m**: near a real YAM's 12 mm per-move bound.
- **`DRIFT_TOLERANCE`, `REACH_TOLERANCE` = 0.02 m**: as pouring-by-wrist-roll.

## Simulation check

The YAM station twin (left arm, linear_4310 hand) has no jar, so the 30 mm
cube on the table stood in for a lid: the skill at its defaults gripped it
from above and made 3 tightening strokes, then 3 loosening strokes with the
lift check. A wrapper around the context (in the test harness, not the skill)
read the cube's pose and `sim.get_object_obb` after every step and on both
sides of every gripper command; yaws are unwrapped across those samples. A
free cube on a table cannot stall, so the stall and seat paths are covered by
the unit tests only.

| run | stroke | joint 6 | cube turned (yaw) | cube moved (xy) | grip reading after the close |
|---|---|---|---|---|---|
| tighten | 1 | -1.900 -> +1.900 rad (+217.7 deg) | -217.7 deg | 1.3 mm | 0.338 |
| tighten | 2 | -1.900 -> +1.900 rad | -188.5 deg | 1.2 mm (1.8 max) | 0.443 |
| tighten | 3 | -1.900 -> +1.900 rad | -188.2 deg | 1.6 mm | 0.443 |
| loosen | 1 | +1.900 -> -1.900 rad (-217.7 deg) | +217.8 deg | 2.7 mm | 0.334 |
| loosen | 2 | +1.900 -> -1.900 rad | +188.8 deg | 7.7 mm (9.1 max) | 0.438 |
| loosen | 3 | +1.900 -> -1.900 rad | +216.2 deg | 10.4 mm, then lifted 23 mm | 0.337 |

- Both runs routed `turned`: tightening "3 strokes, 11.400 rad turned; no
  seat felt"; loosening "lifted 23 mm with the jaws still holding" (25 mm
  asked; the cube rose 23.2 mm with the hand and ended 5.4 mm from the tool
  centre).
- Joint 6 moved 0.0776 rad (3.8 / 49) per step on the reading in every
  stroke and every roll back, with no measurable lag; `turn_per_stroke` read
  3.800 rad each time. The tool centre stayed within 0.4 mm of the grip point
  (`hand_drift`). The open-hand roll backs left the cube where it was
  (0.0 deg, at most 0.3 mm).
- Joint 6 rising turned the cube clockwise seen from above (yaw falling), the
  tightening sense of a right-hand thread, as yam-arcade states for its YAM:
  `tighten_sign` +1 is right for this arm.
- Cube rotation over the 3 tightening strokes: -564.7 deg in all, and +577.5
  deg over the 3 loosening ones. Each close also turned the cube (13.0, 8.5,
  8.2 deg tightening; -0.3, -8.8, -36.2 deg loosening) as the jaws squared it.
- In strokes 2 and 3 the cube turned about 29 deg less than the wrist: the
  re-grip closed on a cube the jaws had not squared (reading 0.44 against
  0.33 on a squared cube), and the first three steps of the stroke turned it
  in the jaws (the reading fell 0.422 -> 0.352 -> 0.337 -> 0.333; the cube
  first turned 16 deg the other way). A cube's flats do this; a round lid
  would not change the jaw width, so the skill reports the reading in
  `grip_trajectory` and does not route on it. Without perception the skill
  cannot see a lid that turns less than the wrist.
- Speed: the 0.5 rad/s bound is enforced on the wall clock (each step waits
  out `step_rad / max_speed`), which bounds a real arm. In the twin each step
  is the simulator's own converge-and-settle move: the whole run took 36,401
  steps at 500 Hz (72.8 s simulated, about 23 min wall clock) for 516 joint
  steps plus the approaches; the peak joint speed in simulated time was not
  measured.

## On a real YAM station

The station's bridge refuses `robot.move_to_joints`, so `roll_with: auto`
turns the tool about its own axis with `robot.go_to_pose` at a fixed tool
centre, which its bounded IK turns into the same joint-6 roll. This path is
covered by the unit tests only; it has not been run on a station or in the
twin (whose servo leaves the wrist roll free, so a pose roll does not turn
joint 6 there).

## Provenance

Ported from yam-arcade's `stations/jars/robot_demo.py` (MIT): the stroke
values, the stroke / open / turn back / regrasp cycle, the hand following the
thread's pitch, and the stall rule; and from RoboWaiter's
`MujocoController.untwist` (MIT): the loosening cycle repeated N times, then
carrying the cap away (here the lift check). Changed: perception input in
place of simulator poses, stepped joint commands at 0.5 rad/s in place of a
torque-capped velocity-feed-forward stroke at 1.6 rad/s, the stall read from
the joint's lag with the target held, the gripper reading and drift in place
of the simulated lid's yaw, the pose roll for joint-less connectors, and
routed refusals.
