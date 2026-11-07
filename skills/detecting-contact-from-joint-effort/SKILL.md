---
name: detecting-contact-from-joint-effort
description: Feels for contact without a force sensor - steps the hand along a
  short straight line a few millimetres at a time and, after each step, decides
  from proprioception whether something holds it - the joint efforts against a
  baseline taken in free motion along the same line (i2rt's gripper clog test,
  generalised to the arm) where the observation carries efforts, otherwise how
  far the measured tool centre lags the commanded one beyond its free-motion lag.
  Returns where the hand stopped and which signal fired. Use when a surface's
  position is known only to a centimetre or so and must be measured by touch (a
  table or a box floor before a grasp or a place, the top of an object, a wall
  before a slide), or to check that a line is clear before a move along it.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [contact, proprioception, probe, effort, touch]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/robots/utils.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.solve_ik
    - robot.close_gripper
  required_inputs:
    end: Vec3
  produces_outputs:
    contact: bool
    signal: str
    contact_position: Vec3
    contact_point: Vec3
    travelled: float
    lag_over_baseline: float
    effort_ratio: float
    efforts_available: bool
    steps: int
    baseline_lag: float
  exit_conditions:
    contact: A step left the hand held - its lag beyond the free-motion lag over lag_threshold (3 mm), or a joint's effort residual over its threshold; contact_position is the tool centre where it stood, contact_point that pushed on by tip_reach, `signal` says which fired (lag, effort or effort+lag); the hand then backed off `backoff` (3 mm) along the line.
    free: The hand reached the line's end (or the tool-centre floor) with every step tracking as in free motion; it stays there.
    no_signal: Nothing usable to decide with - signal=effort with an observation that carries no efforts (nothing moved), the line too short for two free-motion samples, or the free-motion lag jumping more than lag_threshold from step to step; `detail` says which.
    blocked: The line is too long (over 150 mm), has no length, starts under the floor or ends out of reach (IK preview), the start was not reached, or a step was refused; `detail` says which.
  canonical_scripts:
    - detect_contact: scripts/detect_contact.py
  examples:
    - title: Probe along a line from a start to an end
      path: examples/canonical_subgraph.json
---

# detecting-contact-from-joint-effort

A YAM has no force sensor, but it feels a push two ways: its motors work
harder (joint effort), and a position-controlled joint that is held stops
following its command (tracking lag). This skill moves the hand along a short
line in small steps and asks both after every step, against what they read
while the hand moved freely along the same line just before.

## The probe

1. Read the hand. With `start`, go there first (a free move, arrival judged
   within `reach_tolerance`); the lag standing there is the first free-motion
   sample. Without it, the line starts where the hand is, and the first step's
   lag is the first sample (that step cannot itself fire).
2. Clip the line at `floor_z` (a tool-centre floor, e.g. a station's lowest
   allowed height) and refuse lines over 150 mm. Preview the end with
   `robot.solve_ik` (`check_reach`): a stall at the edge of reach reads exactly
   like contact.
3. Close the hand (`close_hand`, default true; closing a hand that holds
   something keeps holding it, and the held object becomes the probe).
4. Look once for joint efforts in `robot.get_observation`
   (`arms[arm_id].joint_state.efforts`, or `effort` / `torques`). An
   observation can render every camera, so a connector without efforts is not
   asked again.
5. Step `step` (2.5 mm) at a time with `robot.go_to_pose`, the hand's rotation
   kept. After each step wait until the hand stands still (two reads 0.3 s
   apart within 0.3 mm, at most 3 s) and judge:
   - **lag**: commanded minus measured position, along the line. Contact when
     it exceeds the median of the last three free-motion lags by more than
     `lag_threshold`.
   - **effort** (when published): the mean of `effort_samples` observations,
     dropping samples with any joint faster than `speed_threshold` (i2rt's
     speed gate). Contact when any joint's residual against the mean of the
     last three free-motion windows exceeds `max(effort_threshold, 4 sigma)`,
     sigma from the last five windows once there are three.
   A step that is not contact adds its lag and window to the baseline.
6. On contact back off `backoff` along the line so the hand stops pushing.

`signal` chooses: `auto` (efforts when published, and lag always; either
fires), `effort` (no_signal without efforts), `lag`.

## The two connectors this was written against

- **RoboSimStudio's YAM station twin** (gap connector): `robot.get_observation`
  gives each arm `joint_state: {positions}` and nothing else - no velocities,
  no efforts - so the twin decides by lag. Its `robot.go_to_pose` is a feedback
  servo with a 4 mm arrival ball that would call a 2.5 mm step reached without
  moving; `step_tolerance` (1 mm) is passed as its `tolerance`. A blocked step
  ends as a routed `timeout` after the servo's budget (`step_max_steps`, passed
  only when set).
- **The real YAM station** (`yam_real.py`'s connector, and the threading
  station's): the arm state is `joints_rad` and the tool pose; the motion
  server publishes no arm efforts (its gripper's effort stays inside the
  server), so it too decides by lag. A step is a planned move whose wait
  accepts 12 mm, so a 2.5 mm step returns at once and the settle loop (the
  station's own `still()`) decides where the hand stands; a raised "the tool
  did not reach its target" is read as a step that ended short, not as a
  refusal. The threading station's connector speaks position / quat_xyzw lists
  and a `label`; the skill follows its convention when `robot.get_ee_pose`
  answers that way, labelled `push...` so it runs at the station's base rate.
  It has no `robot.solve_ik`, `robot.close_gripper` or `robot.get_observation`;
  the preview and the close are skipped and noted.
- **A connector that publishes efforts** (an i2rt `MotorChainRobot` exposes
  `joint_state.eff` and `vel`) gets the effort test too; it is covered by unit
  tests only.

On a real station whose tool-centre floor is 8.7-14 mm over the table, closed
fingertips 10.2 mm under the tool centre touch the table with the tool centre
only 1.5 mm or less over the floor: less than `lag_threshold`, so the lag test
cannot see the table there (the floor ends the line as `free`). Probe the table
with the fingers open or through a held object, or with efforts.

## Inputs

- `end` (required), `start` (optional): the line's ends, tool centre, world
  frame.
- `signal` (`auto`), `step` (0.0025 m), `lag_threshold` (0.003 m),
  `effort_threshold` (0.5 N.m), `speed_threshold` (0.3 rad/s),
  `effort_samples` (5), `floor_z`, `tip_reach` (0; e.g. 0.0102 for closed
  linear_4310 fingertips), `backoff` (0.003 m), `close_hand` (true),
  `check_reach` (true), `reach_tolerance` (0.015 m), `step_tolerance`
  (0.001 m; 0 leaves the connector's), `step_max_steps` (0), the settle
  constants, `arm_id` (0; passed only when nonzero).

## The constants, and why

- **`STEP = 0.0025` m, `LAG_THRESHOLD = 0.003` m**: the real threading
  station's `SETTLE_STEP` and `SETTLE_CONTACT`, with which it feels for the
  table before closing on a cable (low at far reach a real YAM stood 5.4 mm
  over its aim in free air; the next 1.5 mm read as the table).
- **`EFFORT_THRESHOLD = 0.5` N.m, `SPEED_THRESHOLD = 0.3` rad/s**: i2rt's
  `clog_force_threshold` and `clog_speed_threshold` for the YAM's linear_4310
  gripper. Not tuned on arm joints, which is why the per-joint 4-sigma floor
  exists (`EFFORT_SIGMAS`, a choice).
- **`BASELINE_WINDOW = 3`**: the baseline follows a lag that grows as the arm
  reaches lower (3.6 to 6.7 mm on the real arm).
- **`BACKOFF = 0.003` m**: a lag over 3 mm means the hand was pushed back that
  far from its command; backing off as much stops the push.
- **`REACH_TOLERANCE = 0.015` m**: no tighter than a real station's 12 mm.

## Simulation check

A YAM station twin (left arm, 30 mm cube on the table, MuJoCo at 500 Hz;
simulator ground truth only to set the cases up and to measure). The hand was
turned top-down with `robot.grasp_frame`, the skill closed it and probed
straight down with its defaults, `tip_reach` 0.0102 m: from 40 mm over the
contact height to 15 mm under it on the table (6 cm from the cube, four
directions) and on the cube's top (centre, 5 mm off along x and y); through
free air over the table, 55 mm down to 90 mm (four directions) and to 40 mm
over it. Twelve rollouts, one probe each:

| case | route | fingertip stop vs true surface | cube moved |
|---|---|---|---|
| table x4 | contact x4 (lag) | +0.35 / +0.39 / +0.36 / +0.35 mm | 0 |
| cube top x3 | contact x3 (lag) | +0.01 / -0.01 / 0.00 mm | 2.8 / 2.6 / 2.7 mm along +x |
| free air x5 | free x5 | - (whole 55 mm line travelled) | 0 |

- Detection 7/7, false alarms 0/5. The twin's free-motion lag was 0.2-0.35 mm
  with 0.1-0.2 mm from step to step; a held step read +4.7 to +5.0 mm over it.
  Contact fired on the second held step (the first read about 2 mm, under the
  3 mm threshold), so the command was 5 mm past the surface when it fired.
- That press pushed the cube 2.6-2.8 mm sideways each time (the jaws' tapered
  tips on its top). With `lag_threshold` 1.5 mm (the twin's tracking allows
  it; a real YAM's does not) contact fired one step earlier: the cube moved
  1.1 mm, the stops read +0.15 mm (cube) and +0.32 mm (table), and the 40 mm
  free-air line stayed free.
- `efforts_available` was false in every run: the twin publishes no efforts,
  so the effort test ran only in the unit tests. Nothing here ran on a real
  YAM.

## What changed from i2rt, and why

i2rt's `GripperForceLimiter` runs inside the gripper motor's control loop:
it averages the motor's effort over 0.1 s and calls the gripper clogged when
the mean exceeds `clog_force_threshold` while the motor turns slower than
`clog_speed_threshold`, then caps the force. Here the same test watches every
arm joint from outside, between steps, through the connector; a free-motion
baseline replaces the gravity torque i2rt's KDL model would subtract (the
connectors expose no model torque); the speed gate drops moving samples
instead of vetoing the decision; and because neither connector this ran on
publishes efforts, tracking lag (the station's own contact rule) carries the
decision there. The force cap is not ported: this skill decides and backs off.
