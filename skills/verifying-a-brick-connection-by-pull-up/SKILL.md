---
name: verifying-a-brick-connection-by-pull-up
description: Checks that a brick just pressed onto studs is clutched by
  lifting it a few millimetres, still gripped, in 0.5 mm steps and watching
  the arm - a clutched brick holds the hand below its command (tracking lag,
  or joint effort where published) and the lift stops there and puts it back
  down before the clutch lets go; a loose brick comes up with the hand and is
  put back on its seat, still held, for another press. CMU MFI's PullUp skill
  (vibro_tactile_toolbox) with proprioception standing in for its wrist
  force-torque sensor. Use when a brick (or a press-fit, snap-fit or peg)
  has just been pressed and the hand still grips it - right after
  pressing-a-brick-onto-studs or pressing-in-with-guarded-push - and a wrist
  depth read cannot tell a seated brick from one resting on stud tops, or the
  hold must be confirmed before letting go.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, press-fit, verify, contact, proprioception, pull-test]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego, insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/cmu-mfi/vibro_tactile_toolbox/blob/30fb15bb05a91606dfd4725f1f37352207f0299f/src/skill/common_skills.py
      upstream_license: MIT declared in package.xml only, no LICENSE file (reference only; reimplemented, no code copied)
gap:
  requires: {connector: [robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.go_to_pose
    - robot.solve_ik
    - robot.open_gripper
  produces_outputs:
    connected: bool
    rise: float
    lag_over_baseline: float
    effort_ratio: float
    steps: int
    signal: str
    efforts_available: bool
    released: bool
    jaw_width: float
  exit_conditions:
    connected: A lift step left the hand more than hold_lag (1 mm) below its command beyond the start's lag, or a joint effort residual over effort_threshold; the hand was put back down at once, then (release, default) opened and rose `retreat`. `signal` says which fired.
    not_connected: The whole lift (lift_height, 3 mm) ran with the hand following its command and the jaws still holding something; the brick came up with the hand (or the jaws slid on it). It is put back at the start, still held (put_back), ready for another press.
    not_holding: The jaws read at their stop before the lift (nothing moved), or after a free lift (the brick slipped out).
    blocked: The top of the lift is out of reach (IK preview - a stall there would read as connected), a move was refused, or signal=effort without published efforts; `detail` says which.
  canonical_scripts:
    - pull_up_check: scripts/pull_up_check.py
  examples:
    - title: Check a just-pressed brick by pulling up on it
      path: examples/canonical_subgraph.json
---

# verifying-a-brick-connection-by-pull-up

A brick resting on the stud tops and a brick seated on them differ by a stud
height (1.7 mm LEGO), close to what a wrist depth camera resolves, and a brick
pressed on a slant can read flat. Pulling on it settles the question: a seated
brick is held by its clutch, a resting one comes up. CMU MFI's LEGO station
does exactly this after every press (PullUp: 10 mm up at 0.1 % speed, stopped
on a z reaction; connected if the force changed by more than 0.3 N).

## The pull

1. **Holding?** Read the jaws (`robot.get_gripper`, width from
   `robot.describe_gripper`'s fit): at their stop is `not_holding`, nothing
   moves (`check_holding`).
2. **Reach.** Preview the top of the lift with `robot.solve_ik`
   (`check_reach`): at the edge of reach the arm would stall, which reads as a
   brick that held.
3. **Zero.** Command the hand where it stands (this also takes off whatever
   press is left) and settle; the lag there (commanded minus measured, along
   +z) and an effort window, when efforts are published, are the start's
   reading (vibro_tactile_toolbox's baseline wrench).
4. **Lift** `step` (0.5 mm) at a time to `lift_height` (3 mm) with
   `robot.go_to_pose`, the rotation kept, settling after each step. A step that
   leaves the hand more than `hold_lag` (1 mm) further below its command than
   at the start, or a joint effort residual over `effort_threshold`, is a hold:
   stop at once and command the start again, so the pull on the clutch is never
   more than about one step past the threshold.
5. **Connected**: open (`release`) and rise `retreat` (30 mm).
   **Free lift**: read the jaws again (closed to their stop: the brick slipped
   out, `not_holding`), and put the hand back at the start (`put_back`) still
   holding, so a caller can press again.

## Inputs

- `lift_height` (0.003 m), `step` (0.0005 m), `hold_lag` (0.001 m).
- `signal` (`auto`: lag, plus efforts when published; `lag`; `effort`),
  `effort_threshold` (0.5 N.m), `effort_samples` (5), `speed_threshold`
  (0.3 rad/s).
- `check_holding` (true), `release` (true), `retreat` (0.03 m), `put_back`
  (true), `check_reach` (true), `step_tolerance` (0.0002 m, passed as the
  servo's `tolerance`; 0 leaves the connector's), `step_max_steps` (0), the
  settle constants, `arm_id` (passed only when nonzero).

## Outputs

`connected`, `rise` (how far the hand actually rose, measured), the last
`lag_over_baseline` and `effort_ratio`, `steps`, `signal`,
`efforts_available`, `released`, `jaw_width` (-1 when not read).

## Limits

- **The force is not known.** The pull at the stop is the servo stiffness
  times about `hold_lag`, uncalibrated (no force sensor). A LEGO 2x2 clutch is
  a few newtons; if the arm's stiffness lets the clutch go before 1 mm of
  lag builds, a seated brick comes up and reads `not_connected` (a false
  negative, not a false positive), and the brick is put back for a re-press.
- **Jaws that slide** on the brick also read `not_connected` (the jaw width
  does not change). Grip firmly; for a brick on a light plate hold the plate
  with the other arm.
- A real YAM station's tool-centre floor (14 mm over the table) does not
  matter here: the lift only rises.

## Simulation check

A YAM station twin (left arm, MuJoCo at 500 Hz; simulator ground truth only
to set up and to measure), run with the sim harness's grip fix (impratio 100,
solimp 0.99; ~/yam_gap/skill_sources/SIM_GRIP.md). Without it the stock twin
let held parts creep through the jaws at under 1 N. The twin has no stud
clutch, so there are two stand-ins:

- **fixed**: the box (a fixture) clamped across its 3 mm +y wall by
  grasping-container-rims (`lift_height` 0, `wall_thickness` 0.003; the jaws
  read 6.1 mm), standing in for a clutched brick.
- **loose**: the free 30 mm cube gripped by a scripted top-down grasp on the
  table, standing in for a brick resting on the stud tops.
- **empty**: the hand closed on nothing.

One rollout per case:

| case | hold_lag | route | hand rose (of commanded) | lag over start |
|---|---|---|---|---|
| fixed | 1.0 mm (default) | connected, step 3 | 0.09 of 1.5 mm | 1.41 mm |
| loose | 1.0 mm (default) | not_connected, cube put back | 2.91 of 3.0 mm | 0.09 mm |
| empty | default | not_holding (jaws 0.44 mm) | - | - |
| fixed | 0.3 mm | connected, step 1 | 0.02 of 0.5 mm | 0.48 mm |
| fixed | 1.5 mm | connected, step 4 | 0.26 of 2.0 mm | 1.74 mm |
| loose | 0.3 / 1.5 mm | not_connected | 2.91 of 3.0 mm | 0.09 mm |

- With the grip holding, the fixed wall stopped the hand within 0.3 mm at
  every threshold, and the lag grew by about 0.5 mm per step. A free lift
  stayed under 0.1 mm.
- On the stock twin (before the harness fix) the jaws slid up the fixed wall
  once the lag reached 0.5-0.76 mm, so 1.5 mm never fired. That is why this
  skill shipped with 0.3 mm. It also showed what a sliding grip does here: a
  `not_connected` with the jaw width unchanged (see Limits).
- An earlier set of runs passed a 1 mm arrival ball (`step_tolerance`) to the
  servo with 0.5 mm steps: each step was called reached without moving, and
  both stand-ins read `connected` at the first step. The ball is 0.2 mm.
- No efforts in the twin, no real clutch, no real YAM.

## What changed from vibro_tactile_toolbox, and why

vibro_tactile_toolbox runs the lift as one slow move on a UR with a wrist
force-torque sensor, stopped by its termination handler on a z reaction above
the start's, and decides from the end-minus-start force (0.3 N). Here the lift
is stepped through the connector, the reaction is the tracking lag (or joint
efforts) over the start's, the lift stops and is undone at the first hold, and
it is 3 mm instead of 10 mm. Its code is not used (no LICENSE file in the
repository; MIT is declared only in package.xml).
