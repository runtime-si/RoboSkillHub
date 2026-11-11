---
name: recovering-from-empty-grasp-by-gripper-load
description: Grasp at a given pose and, when the jaws close on nothing (they ran to the closed stop, and with a load-reporting gripper the load stayed low), escalate through a fixed ladder instead of carrying air - lift and retry, rewind the approach and retry, go home and retry, then give up. Each retry can re-measure the object with a locate tool. Use when a grasp from an imprecise pose (a policy, a coarse detection, a stale estimate) sometimes closes empty and the graph should try again on its own before failing.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasp, recovery, retry, proprioception, gripper-load, empty-grasp]
  hub:
    stage: recover
    embodiments: [any]
    tasks: [pick-place]
    version: 1.0.0
    provenance:
      code: https://github.com/HyuanTan/lerobot_far/blob/f8bf405bcd2e/src/lerobot/async_inference/smart_robot_client.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.describe_gripper, robot.go_home]}
  allowed_tools:
    - robot.describe_gripper
    - robot.get_gripper
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.go_home
  required_inputs:
    grasp_pose: Se3Pose
  produces_outputs:
    attempts: int
    recovery: string
    detail: string
    grasp_position: Vec3
  exit_conditions:
    held: The first close held something - the jaw gap ended above the closed stop by more than empty_margin_m (and the load passed load_grasp_threshold when the gripper reports one).
    recovered: A retry held something; recovery names the last ladder level used (lift, rewind or home).
    gave_up: Every close ran to the closed stop, max_total_retries + 1 of them; the jaws are left open.
    move_failed: The connector refused a move of the approach or of a retry; detail says which.
  canonical_scripts:
    - recover_empty_grasp: scripts/recover_empty_grasp.py
  streaming: false
---

# recovering-from-empty-grasp-by-gripper-load

Port of the gripper state machine in FAR's `SmartRobotClient` (`GripperStateMonitor`,
`GripperDecision`). FAR watches a streamed ACT/SmolVLA policy on an SO-101 and, when the
Feetech gripper closes on nothing, escalates `LIFT_RETRY -> REWIND_RETRY -> RECOVERY -> STOP`
instead of letting the policy carry on with an empty hand. Here the same ladder wraps one
grasp driven through `robot.*` tools.

## What it does

1. Open, go to `pre_grasp_height_m` above `grasp_pose`, descend to it, close.
2. Read the jaw opening (`robot.get_gripper`, 1.0 = open) and convert it to metres with
   `robot.describe_gripper`'s `width_fit`. Empty if it is within `empty_margin_m` of the
   closed stop (FAR's position gap: commanded closed, the jaws got there). When the reply
   carries an `effort` and `load_grasp_threshold > 0`, a held grasp also needs that load.
3. On an empty close, the next ladder level (one level per failure, as in FAR):

| failure | level | motion before the next close |
| --- | --- | --- |
| 1 | `lift` | open, rise `lift_dz_m`, (re-locate), come down on the target |
| 2 | `rewind` | open, back up to the pre-grasp height, (re-locate), full approach again |
| 3+ | `home` | open, `robot.go_home`, (re-locate), full approach again |
| after `max_total_retries` | - | open, route `gave_up` |

`locate_tool` (with `locate_args`) is called before each retry; a reply with `position`,
`center` or `obb.center` moves the target there plus `grasp_offset_m`. Without it the
retries go back to the same pose - which recovers a grasp that missed for transient
reasons (the object was still settling, the hand lagged) but not one aimed at the wrong
place.

## Inputs

| input | default | meaning |
| --- | --- | --- |
| `grasp_pose` | - | tool-centre pose to close at (`position`, `rotation`) |
| `locate_tool` / `locate_args` | `""` / `{}` | re-measures the object before each retry |
| `grasp_offset_m` | `[0,0,0]` | added to a located position to get the tool centre |
| `pre_grasp_height_m` | 0.06 | approach waypoint above the grasp |
| `lift_dz_m` | 0.05 | rise before a lift retry (FAR: FK/IK + dZ) |
| `max_total_retries` | 3 | FAR's `max_total_retries` |
| `empty_margin_m` | 0.003 | gap over the closed stop that counts as held (hub's verifying-grasps margin; FAR's `pos_gap_threshold`) |
| `load_grasp_threshold` | 0 | minimum `abs(effort)`, 0 = gap only |
| `arm_id` | 0 | arm |

## Outputs

`route`, `detail`, `attempts` (closes made), `recovery` (last ladder level, `""` on a
first-try hold), `widths_m` (jaw gap after each close), `grasp_position` (the last target).

## Twin check (YAM station, single arm, `sim_check` with the grip fix)

A 35 mm cube on the table; the first grasp pose is deliberately wrong, the
retries re-locate it with privileged `sim.get_object_obb` (standing in for a
detector) or not at all. An independent check lifts the hand 10 cm and measures
whether the cube rose. The hub's current behaviour on an empty close is a single
close and an `empty` route (verifying-grasps): 0 lifts in every missed case.

See the PR for the measured table.

## Limits

- No slip monitoring during transport: FAR's SLIP branch watches the load during
  HOLDING; a single grasp call has no HOLDING phase to watch. Use
  `verifying-grasps` after the lift for that.
- The YAM connector reports no gripper effort, so on that station the decision is the
  jaw gap alone. A thin object (under `empty_margin_m`) reads as empty.
- The rewind replays this script's own approach waypoint, not a policy's action history.
