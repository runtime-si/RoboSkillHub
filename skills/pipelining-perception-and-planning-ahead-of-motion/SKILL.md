---
name: pipelining-perception-and-planning-ahead-of-motion
description: Runs a queue of pick-and-place tasks (one arm, or two arms each with its own
  queue) with task k+1 prepared while the arm executes task k - a background thread
  perceives task k+1's object (fixed camera, or an idle arm's wrist camera, never a moving
  wrist) and runs the child pick-and-place skill up to its first move from task k's
  expected end state, keeping its planning replies (IK, grasp frames) - so task k+1 starts
  the moment task k ends; at the hand-over the plan is checked (arm where expected? object
  where it was perceived?) and re-planned live if stale. Use when a station works through
  several pick-and-places with a hub child skill and perception or planning is slow next to
  the motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: coordination
  tags: [pipelining, scheduling, pick-place, bimanual, latency]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
gap:
  requires: {connector: [robot.get_ee_pose]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_robot
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
    tasks: string
    pick_place_skill: string
  produces_outputs:
    done: int
    failed: int
    stale: int
    makespan_s: float
    idle_s: float
    cache_hits: int
    cache_misses: int
    tasks: string
  exit_conditions:
    all_done: Every task's child skill ended on one of its success exits; tasks (JSON) has each task's arm, route, start / end time and idle time before its first move.
    one_failed: At least one task's child skill raised or ended off its success exits (with a fresh or a still-valid plan); the other tasks were still run; tasks names which.
    stale_replan_failed: A task whose prepared plan was stale (the arm did not end where task k was expected to leave it, the object moved, or the look-ahead itself failed) failed after being re-planned live.
  canonical_scripts:
    - pipeline_tasks: scripts/pipeline_tasks.py
  examples:
    - title: Six pick-and-places on two arms, each arm's next task prepared during its current one
      path: examples/canonical_subgraph.json
---

# pipelining-perception-and-planning-ahead-of-motion

A pick-and-place is perceive, plan, move. Done one after the other the arm stands still
through every perception and every planning step. Here those two run for task k+1 in a
background thread while the arm executes task k.

## How

1. **Queues.** `tasks`: a list (or its JSON text) of `{object, place, arm?, object_obb?, inputs?}`; one
   queue per `arm` (default 0). One arm: tasks in order. Two arms: one thread per arm,
   every move arbitrated by running-two-skills-concurrently-on-two-arms' clearance
   arbiter (its `Arbiter` / `ArmContext`, arm_id injected, threaded semantics).
2. **Look-ahead** (when `pipelined`, from the moment task k starts):
   - **perception** of task k+1's object with `perceive_tool` (`perceive_kwargs`,
     `{object}` filled in; `perceive_output` names the reply's box). `camera` must not be
     the wrist camera (`wrist_cameras`) of an arm that is moving: the fixed camera, or
     the other arm's wrist while that arm is idle. A task with `object_obb` skips this.
   - **planning**: the child skill (`pick_place_skill`, called with `obb_input`,
     `place_input` and the task's `inputs`) runs through a context that answers the
     arm's own state reads (`robot.get_ee_pose`, `robot.get_gripper`) from task k's
     expected end state (where task k started: the child returns there), sends
     planning tools (`robot.solve_ik`, `robot.grasp_frame`, `robot.forward_kinematics`,
     `robot.describe_arm`) to the connector and keeps their replies, and stops the
     child at its first move.
   - the object is perceived again every 0.2 s until task k ends (newest view kept).
3. **Hand-over.** Stale if the arm's tool is more than `pose_tolerance_m` from the
   expected end, if the newest view moved the object more than `object_tolerance_m`, or
   if the look-ahead failed. Stale: the kept replies are dropped and the child plans
   live (with the newest box). Fresh: the child runs with the kept replies served for
   identical calls (keyed by tool and arguments, so a changed target misses and is
   asked live).
4. **Measure.** Per task: start, end, idle time from the previous task's end to its first
   move, cache hits / misses; `makespan_s`, `idle_s` in total. `pipelined: false` runs
   the same path without look-ahead, for comparison.

## The constants, and why

- **`POSE_TOLERANCE = 0.012` m**: the real YAM station's arrival tolerance (12 mm).
- **`OBJECT_TOLERANCE = 0.015` m**: half a 30 mm cube; under it the same grasp closes.

## Limits

The look-ahead's IK is solved while the arm is elsewhere; a connector that seeds IK from
the current joints can return another branch than a live solve would. Replies are reused
only by a child that asks for exactly the same thing. Two-arm mode inherits #246's
threaded mode: unsafe on a twin whose servo shares one setpoint between arms.
