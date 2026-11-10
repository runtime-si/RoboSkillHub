---
name: pressing-in-with-guarded-push
description: Pushes a held part along an axis (straight down by default) in
  small steps with a bounded push and a bounded travel, and says how it
  ended - travel or target depth reached (seated), the push collapsing as the
  hand jumps forward (snap_detected, a snap-fit or clutch giving), or the push
  limit reached short of the goal - o2ac's linear_push / do_insertion with
  tracking lag, and joint effort where published, standing in for the force
  sensor a YAM does not have. Unloads the push at the end. Use when a held
  part already sits on or in its seat (a brick on studs, a peg in a hole's
  mouth, a domino in a slot, a snap-fit lid) and must be pressed home by a
  few millimetres without crushing it, and a camera cannot tell whether it
  went in.
license: BSD-3-Clause
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [press-fit, snap-fit, insertion, contact, guarded-move, proprioception]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, assembly]
    provenance:
      authors: [runtime-si]
      code: https://github.com/o2ac/o2ac-ur/blob/65ddc5496879799038cefbad061380b7d0300d73/catkin_ws/src/o2ac_routines/src/o2ac_routines/ur_force_control.py
      upstream_license: BSD-3-Clause (file header, OMRON SINIC X); repository MIT (Team O2AC)
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.solve_ik
  produces_outputs:
    travelled: float
    contact_travel: float
    peak_lag: float
    final_lag: float
    remaining: float
    steps: int
    signal: str
    efforts_available: bool
    relieved: bool
    stop_position: Vec3
  exit_conditions:
    seated: The hand travelled max_translation (8 mm) along the axis, or reached the target's depth, before the push limit; or, with a target, hit the limit within goal_tolerance_if_lockup of it (o2ac's relaxed lock-up). contact_travel is -1 when nothing resisted on the way.
    snap_detected: After the lag had built at least snap_rise (2 mm) over its free baseline it fell by snap_drop (1.5 mm) in one step while the hand jumped as far forward - a snap-fit or clutch gave; the push stopped there (stop_on_snap).
    force_limit_no_travel: The lag over the free baseline passed max_lag (6 mm), or a joint effort residual passed effort_threshold, short of the goal; the push was unloaded. `signal` says which.
    blocked: The end of the push is out of reach (IK preview) or under floor_z, a move was refused, the floor ended the push, or signal=effort without published efforts; `detail` says which.
  canonical_scripts:
    - guarded_push: scripts/guarded_push.py
  examples:
    - title: Press a held part 5 mm down into its seat
      path: examples/canonical_subgraph.json
---

# pressing-in-with-guarded-push

The last few millimetres of an insertion are pushed, not placed: a brick's
clutch, a press-fit peg, a snap-fit cover. A force-controlled arm pushes with
a set force until the part has travelled far enough or the force is reached;
o2ac's `linear_push` and `do_insertion` do exactly that with a UR's wrist
force-torque sensor. A YAM has neither the sensor nor force control, but its
joints are position servos that act like springs: when the hand is held, the
command runs ahead of it, and how far ahead is the push.

## The push

1. Read the hand (`p0`). The goal is `max_translation` along `direction`, or,
   with `target`, the target's depth along it (a target already behind the
   hand is `seated` at once).
2. Preview the furthest command (goal + `max_lag` + one step) with
   `robot.solve_ik` (`check_reach`): a stall at the edge of reach reads exactly
   like a push. Clip at `floor_z` (a tool-centre floor) when pushing down.
3. Zero the push: command the hand where it stands and settle; the lag there
   (commanded minus measured, along the axis) is the first free-motion sample
   (o2ac zeroes its sensor before pushing). Look once for joint efforts in
   `robot.get_observation`.
4. Step the command `step` (1 mm) along the axis with `robot.go_to_pose`, the
   hand's rotation kept, wait until the hand stands still, and judge:
   - `over` = the lag minus the median of the last three free-motion lags.
     Over `contact_lag` (1.5 mm) marks resistance (`contact_travel`); until
     then each step's lag joins the baseline.
   - **snap**: the push had built `snap_rise` (2 mm), and in this step it fell
     by `snap_drop` (1.5 mm) while the hand advanced as much: `snap_detected`.
   - **goal**: the hand's travel reached the goal: `seated`.
   - **limit**: `over` passed `max_lag` (6 mm), or a joint's effort residual
     against the free-motion windows passed `effort_threshold`:
     `force_limit_no_travel`, or `seated` if within `goal_tolerance_if_lockup`
     of a target.
5. Unload (`relieve`): if the hand ended loaded, command it to where it stands
   (plus the free-motion lag) so the arm stops pushing.

## Inputs

- `direction` (Vec3, default straight down), `max_translation` (0.008 m),
  `target` (Vec3, optional; the tool-centre position to reach),
  `goal_tolerance_if_lockup` (0 m).
- `step` (0.001 m), `contact_lag` (0.0015 m), `max_lag` (0.006 m),
  `snap_rise` (0.002 m), `snap_drop` (0.0015 m), `stop_on_snap` (true).
- `signal` (`auto`: lag, plus efforts when published; `lag`; `effort`),
  `effort_threshold` (0.5 N.m), `effort_samples` (5), `speed_threshold`
  (0.3 rad/s).
- `floor_z`, `relieve` (true), `check_reach` (true), `step_tolerance`
  (0.0002 m, passed as the servo's `tolerance`; 0 leaves the connector's),
  `step_max_steps` (0), the settle constants, `arm_id` (passed only when
  nonzero).

## Outputs

`travelled` (along the axis, measured), `contact_travel` (-1 if nothing
resisted), `peak_lag` and `final_lag` (over the free baseline), `remaining`
(goal minus travel), `steps`, `signal`, `efforts_available`, `relieved`,
`stop_position` (the tool centre where the push ended). These report what the
hand did; whether the part is seated needs a check after the release (e.g.
verifying-insertion-by-probe-push, or a depth read as in
pressing-a-brick-onto-studs).

## The constants, and why

- **`MAX_LAG = 0.006` m** is the "target force". The force it stands for is
  the servo stiffness at that pose times the lag, and nothing here calibrates
  it: no force sensor was available. Lower it for fragile parts.
- **`CONTACT_LAG = 0.0015` m**: half the 3 mm a real YAM station reads as
  contact (`SETTLE_CONTACT`); the push continues past it.
- **`STEP = 0.001` m**: once in contact, each step raises the push by about
  the stiffness times the step, so the limit is overshot by at most one step.
- **`SNAP_RISE`, `SNAP_DROP`**: choices, not measurements; a station twin's
  free-motion lag jitters 0.1-0.2 mm between steps, well under 1.5 mm.
- **`EFFORT_THRESHOLD = 0.5` N.m**: i2rt's gripper clog threshold, as in
  detecting-contact-from-joint-effort; not tuned on arm joints.

## Simulation check

Setup:

- A YAM station twin (left arm, 30 mm cube on the table, MuJoCo at 500 Hz;
  simulator ground truth only to set the cases up and to measure), run with
  the sim harness's grip fix (impratio 100, solimp 0.99;
  ~/yam_gap/skill_sources/SIM_GRIP.md). The stock twin let friction creep,
  which changed the cube's numbers below.
- Defaults except as noted.
- The hand was turned top-down with `robot.grasp_frame` and closed (the
  closed fingertips, 10.2 mm under the tool centre, are the "part"), then
  stood at the case's start.
- One rollout per case:

| case | start | route | travelled | resistance from | peak lag | fingertip stop vs true surface | cube moved |
|---|---|---|---|---|---|---|---|
| free air, 8 mm | 60 mm over the table | seated | 9.0 mm | none | 0.6 mm | - | 0 |
| table | tips 3 mm over it | force_limit_no_travel | 3.2 mm | 3.2 mm | 6.8 mm | +0.36 mm | 0 |
| cube top | tips 3 mm over it | force_limit_no_travel | 3.3 mm | 3.3 mm | 6.7 mm | +0.35 mm | 0.3 mm along +x |
| cube top, target 2 mm under it, lockup tolerance 3 mm | as above | seated (lockup) | 3.3 mm | 3.3 mm | 6.7 mm | +0.35 mm | 0.4 mm along +x |
| sideways into the cube, 35 mm along +x | tips at mid-height, 20 mm off its face | seated | 35.5 mm | none | 0.9 mm | - | 25 mm along +x, tipped (rose 4.5 mm) |

- Every press stopped at the limit (10 steps, 6.7-6.8 mm over the free lag)
  with the fingertips 0.35-0.36 mm over the true surface, and was unloaded
  (`relieved`).
- The free-motion lag peaked at 0.6 mm (0.9 mm while pushing the cube
  sideways), under `contact_lag`. Neither free run read false resistance.
- The press on the cube's top moved the cube 0.3-0.4 mm. The hand ended
  0.5 mm (table) and 3.1 mm (cube) off in x after the presses: under 6-7 mm of
  lag the twin's servo gives sideways as well.
- **A part that slides is not felt.** Pushed sideways, the free 30 mm cube
  slid 25 mm and tipped, and the lag never passed 0.9 mm. The push reads as
  free travel and ends `seated`. A part that skids on its seat instead of
  going in looks the same, so verify afterwards.
- On the stock twin (before the grip fix) the same cases read the same routes
  and stops (+0.22 to +0.38 mm). There, friction creep let the cube's top
  press shove it 3.7-4.3 mm sideways, and the sideways push never passed
  0.1 mm of lag.
- A first set of runs passed a 1 mm arrival ball (`step_tolerance`) to the
  servo with these 1 mm steps, and the free lag then peaked at 1.0 mm. The
  ball is now 0.2 mm.
- No snap-fit exists in the twin, so `snap_detected` ran only in the unit
  tests. `efforts_available` was false (the twin publishes no efforts).
  Nothing here ran on a real YAM, and the force that 6-7 mm of lag stands for
  there is not known.

## What changed from o2ac, and why

o2ac runs a hybrid force/position controller at the robot's rate and stops on
`STOP_ON_TARGET_FORCE`, on a maximum translation (`linear_push`) or on a target
position along the axis relaxed by `goal_tolerance_if_lockup` once the arm has
stood still for `check_displacement_time` (`do_insertion`). Here the push is a
sequence of 1 mm position steps through the connector, the force is the
tracking lag over a free-motion baseline (and effort residuals where
published), "standing still under force" is the lag passing `max_lag`, and
the snap detection is added (o2ac has none). o2ac's spiral during insertion is
not ported: searching-insertion-spiral covers the search.
