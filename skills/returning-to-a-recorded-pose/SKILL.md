---
name: returning-to-a-recorded-pose
description: Opens the gripper and drives the arm back to a joint configuration
  recorded earlier in the episode, then checks from a fresh observation that it
  arrived. Use when a graph must undo a failed attempt and retry from a known
  start (a grasp that missed, an insert that jammed, a policy that drifted), or
  hand a learned policy the start pose it was trained from, and the joint state
  to return to was captured earlier with a robot.get_observation tool node.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [recovery, reset, retry, motion]
  hub:
    stage: recover
    embodiments: [any]
    version: 0.1.0
    provenance:
      authors: [runtime-si]
      code: https://github.com/graph-robots/graph-as-policy/tree/main/examples/steered_policy/graph_loop
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.open_gripper
    - robot.move_to_joints
    - robot.get_observation
  required_inputs:
    recorded_joints: JointState
  produces_outputs:
    max_joint_error: float
  exit_conditions:
    returned: The arm stands at the recorded joints, every joint within max_error (default 0.08 rad) on a fresh observation, gripper open.
    not_returned: The move was refused or failed, or a joint ended more than max_error off; `detail` says which.
  canonical_scripts:
    - return_to_pose: scripts/return_to_pose.py
  examples:
    - title: The return subgraph, fed the joints a capture node recorded
      path: examples/record_and_return.json
---

# returning-to-a-recorded-pose

Undo an attempt: open the hand, drive the joints back to where they were when
the episode (or this attempt) started, and confirm it from the arm's own
reading rather than from the move's return value.

## Recording the pose

Recording needs no skill: a tool node reads the observation and the graph
keeps the joint state.

```json
"capture_sg": {
  "skill": "generic",
  "outputs": {"recorded_joints": {"$ref": "observe.arms.0.joint_state"}},
  "nodes": {"observe": {"type": "tool", "tool": "robot.get_observation"}, "done": {"type": "noop"}},
  "edges": [["START", "observe"], ["observe", "done"], ["done", "END"]],
  "exit": {"router_field": null, "success_values": ["done"]}
}
```

Route a failed attempt's exit to this skill's subgraph
(`examples/record_and_return.json`), fed `{"$ref": "capture.recorded_joints"}`,
and its `returned` exit back to the attempt.

## Inputs

- `recorded_joints` (`JointState`, or a bare list of joint positions): the
  configuration to return to, from `robot.get_observation`'s
  `arms[arm_id].joint_state`.
- `open_first` (default true): open the gripper before moving. A reset should
  not carry a half-held object back with it; pass false to keep a grasp.
- `arm_id` (default 0): which arm's reading the arrival check compares.
  `robot.move_to_joints` itself drives arm 0.
- `tolerance`, `max_steps`: passed to `robot.move_to_joints` only when set.
  They are gap's built-in connector's convergence knobs; other connectors (a
  real station's motion server) may not take them.
- `max_error` (default 0.08 rad): the arrival bound on every joint.

## Required end states

- `returned`: every joint of a fresh `robot.get_observation` reading lies
  within `max_error` of the recorded one. `max_joint_error` reports the worst.
- `not_returned`: the move raised (refused, unreachable, timed out) or a joint
  ended further off. `detail` names the cause and `max_joint_error` is the
  worst joint's error, or -1 when no reading was taken.

## The constants, and why

- **`MAX_ERROR = 0.08` rad** (~4.6 deg): just above a real YAM station's
  0.075 rad single-arm settle tolerance, so a move that station accepts is
  accepted here. A tighter bound fails real arms that are where they should
  be: on a YAM station a return home stopped with joint 3 at 0.0517 rad,
  timed out against a 0.05 rad bound, and the supervisor switched the arm off.
  gap's built-in connector converges to 0.01 rad, well inside. Tighten it when
  a learned policy needs its start pose closer.
