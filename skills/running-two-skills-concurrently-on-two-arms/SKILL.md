---
name: running-two-skills-concurrently-on-two-arms
description: Runs two hub skills at the same time, one per arm and one thread
  each, with each skill's robot.* calls bound to its arm (arm_id injected), and
  every move of either arm passed through a shared arbiter that predicts the
  arms' capsule clearance for it (checking-dual-arm-clearance, against the
  other arm standing or on its move in flight) and makes the arm wait while the
  move would come under min_clearance_m; a failure of one skill pauses the
  other at its start pose. Arms are given or assigned by IK reach. Use when two
  independent one-arm steps (two picks, a pick and a wipe) should overlap in
  time on a bimanual station whose arms share workspace, and both skills are in
  the hub.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: coordination
  tags: [bimanual, concurrency, collision, clearance, arbiter]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [bimanual, pick-place]
    provenance:
      authors: [runtime-si]
gap:
  requires: {connector: [robot.get_observation, robot.describe_arm]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_arm
    - robot.forward_kinematics
    - robot.solve_ik
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.execute_trajectory
    - robot.open_gripper
    - robot.close_gripper
    - robot.grasp_frame
  required_inputs:
    skill_a: string
    skill_b: string
  produces_outputs:
    arm_a: int
    arm_b: int
    route_a: string
    route_b: string
    outputs_a: string
    outputs_b: string
    paused_arms: string
    waits: int
    min_predicted_clearance_m: float
  exit_conditions:
    both_done: Both skills returned one of their success exits (any exit when the skill's canonical subgraph lists none), each move having been predicted at least min_clearance_m clear of the other arm; route_a / route_b are the skills' exits and outputs_a / outputs_b their outputs (JSON text).
    one_failed: A skill raised or ended on a non-success exit; the other skill was stopped at its next move and its arm moved back to the joints it started from when that was predicted clear (paused_arms names it), otherwise left where it stood.
    deadlock_timeout: An arm waited wait_timeout_s for clearance (the other arm stood or stayed in its way); that arm was paused at its start pose if clear, and the other stopped at its next move; detail names the move and the predicted clearance.
    unsafe_refused: The clearance could not be predicted (a move the model cannot follow, an unreadable or non-YAM arm), the arms already stood under min_clearance_m, no arm assignment reaches both targets, or the skills cannot be loaded; when this happens mid-run neither arm is moved further.
  canonical_scripts:
    - run_concurrent: scripts/run_concurrent.py
  examples:
    - title: Two pick-and-place skills, one per arm, overlapping in time
      path: examples/canonical_subgraph.json
---

# running-two-skills-concurrently-on-two-arms

Two one-arm skills, two threads. Each skill gets the node's ctx wrapped for
its arm; the arms share one arbiter that sees every move before it is sent.

## How it runs

1. **Load** each skill by name: its canonical script's `run` from this hub's
   `skills/` or `GAP_SKILLS_PATH`, and the success exits of its
   `examples/canonical_subgraph.json` (`exit.success_values`).
2. **Assign arms.** `arm_a` / `arm_b` as given (default 0, 1), or with
   `arm_a = -1` by IK reach: `robot.solve_ik` at `target_a` / `target_b` for
   both arms with each arm's current tool rotation; the pairing that reaches
   both, then the smaller sum of horizontal distances from the tool centres.
   No pairing reaching both: `unsafe_refused`, nothing moved.
3. **Check the start.** The arms' clearance standing now must be at least
   `min_clearance_m`, and predictable.
4. **Bind each ctx to its arm.** Every `robot.*` call gets `arm_id` set to the
   arm (overriding the skill's own, so a one-arm skill written for arm 0 runs
   on either), except `robot.get_observation` / `robot.describe_robot`, which
   read both arms.
5. **Arbitrate every move** (`go_to_pose`, `go_to_pose_cartesian`,
   `move_to_joints`, `execute_trajectory`, grippers, ...). The move is turned
   into checking-dual-arm-clearance inputs (joint waypoints, pose waypoints,
   or a jaw opening) and checked against the other arm: standing, or, while
   the other arm's move is in flight, with that move's path in both orders
   (the two are not synchronised). Under `min_clearance_m` the arm waits; it
   is re-checked whenever the other arm's move ends. When both arms request a
   move at once, `priority_arm` is checked first; the other arm goes ahead
   only once the priority arm moved or was found blocked, so the order is
   deterministic. A move the model cannot follow (`go_home`, contact moves)
   or an `unknown` clearance stops the run: `unsafe_refused`.
6. **Stop on failure.** A skill that raises or ends off its success exits
   stops the run; the other skill's next move (or its wait) is replaced by a
   move back to its start joints, if predicted clear, and the skill is ended
   by an exception it cannot catch (`BaseException`). A wait longer than
   `wait_timeout_s` does the same: `deadlock_timeout`.

## Inputs

- `skill_a`, `skill_b`: hub skill names; `inputs_a`, `inputs_b`: their inputs.
- `arm_a` (0), `arm_b` (1); -1 for auto with `target_a`, `target_b` (Vec3).
- `priority_arm` (0), `min_clearance_m` (0.03), `wait_timeout_s` (15).

## The constants, and why

- **`MIN_CLEARANCE = 0.03` m**: the capsule model is a lower bound on the
  YAM's meshes (checking-dual-arm-clearance's own `too_close` is 5 mm); add the
  real YAM station's 12 mm arrival error and the D405 bracket, which is not in
  the model (about 15 mm off the hand housing).
- **`WAIT_TIMEOUT = 15` s**: two of the twin's longest single moves (a 0.4 m
  carry takes about 6 s at the servo's speed).

## Limits

The arbiter predicts on the YAM capsule model, so on another arm every move is
`unknown` and the run refuses. A move is checked when it starts, not tracked
while it runs: a skill that streams many small moves is checked once per move.
Unit tested on a fake two-arm robot (FakeContext); the twin harness is in
progress.
