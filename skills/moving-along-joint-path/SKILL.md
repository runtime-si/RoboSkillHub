---
name: moving-along-joint-path
description: Moves the arm through a supplied joint-space path on a backend that declares its joints and reports measured joint positions, checking limits and step size before motion and measured arrival after. Use when a validated q_path (e.g. a reorient plan checked for IK jumps) must run as planned.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: motion
  tags:
  - motion
  - joints
  - move_joints
  - workcell
  - feedback
  - yam
  - isaac
  hub:
    stage: execute
    embodiments:
    - yam_i2rt
    arms: 1
    version: 0.1.0
    tasks:
    - motion
    provenance:
      authors:
      - Runtime.si
    notes: Adapter-backed implementation; station setup and task-specific validation
      are required. No sim/real verification is claimed.
gap:
  allowed_tools:
  - workcell.invoke
  - workcell.describe
  required_inputs:
    request_json: string
  produces_outputs:
    result_json: string
    status: string
  exit_conditions:
    succeeded: The capability-specific postcondition is supported by the returned
      measurements.
    failed: The attempted operation or tracking update did not meet its required condition.
    unverified: Available evidence is insufficient to establish the requested physical
      postcondition.
    rejected: The request, station configuration or required backend capability is
      not accepted.
  canonical_scripts:
  - main: scripts/main.py
---

# moving-along-joint-path

This bundle calls the `move_joints` capability (contract `rsh-task/1.2`) through
`workcell.invoke`.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json` and `status` (`succeeded`, `failed`, `unverified`, `rejected`),
  with `failure_type`, `phase` and `recovery_hints` for routing.

## Behaviour

The backend must declare `joint_position_feedback` and `joint_motion`, publish
`joints` (names, limits, velocity limits) in its description, report measured
joints in its state and implement `move_joints`. Every point is checked against the
joint limits and every step (from the measured start) against
`limits.max_joint_step_rad` before motion; success needs fresh measured joints within
`limits.joint_tolerance_rad` of the last point. No current YAM or Isaac backend
declares these capabilities.

## Capability request

Required JSON fields: `q_path`. Optional: `joint_names`, `speed_rad_s`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1).
No sim or real verification is claimed by this bundle.
