---
name: grasping-with-feedback
description: Approaches a supplied grasp pose, closes the gripper and checks available
  grasp evidence through a configured workcell adapter. Use when an agent needs a
  local grasp with explicit feedback rather than an unchecked gripper command.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: grasping
  tags:
  - grasping
  - grasp
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
    - grasp
    - pick-place
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

# grasping-with-feedback

This bundle calls the `grasp` capability through `workcell.invoke`. The same
script can be invoked by a graph or an agent; the selected station determines
whether the implementation reaches YAM hardware or an Isaac adapter.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json`, the complete runtime result serialized as JSON, and
  `status`, one of `succeeded`, `failed`, `unverified`, `rejected`.
- `ROBOSKILLHUB_STATION` selects the station configuration. Inspect the runtime
  description with `workcell.describe` before making a capability request.
- A coordinator must route all four statuses. A successful transport call does
  not establish that the physical task succeeded; retain the returned evidence.

## Scope and verification

The robot declaration is an adapter target, not a measured success result.
The station supplies frames, calibration, observation sources and backend
capabilities. Invalid or unsupported requests must not be treated as successful.
Do not change the station, motion bounds or validation rules inside graph search.

## Source relationship

The local primitive is an independently implemented workcell capability, informed
by [ENPIRE grasping material and the existing RoboSkillHub grasp contracts](https://github.com/NVlabs/ENPIRE/tree/99ee90acf65b5b18957c8382ad580db999528be3/enpire/env/forge/cap/saved_scripts/skill_library). This bundle does not copy that project's full
expert or claim the upstream robot's results. The source-intake catalog retains
the upstream requirements and license evidence separately.

## Capability request

Required JSON fields: `pregrasp_pose`, `grasp_pose`, `lift_pose`, `target_id`, `open_width_m`, `close_width_m`.

Optional JSON fields: `speed_m_s`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#capabilities-and-required-fields)
for units, execution sequence, evidence requirements and station binding.
`workcell.describe()` returns the complete JSON Schema.
