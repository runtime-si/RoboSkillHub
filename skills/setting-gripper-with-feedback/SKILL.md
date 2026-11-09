---
name: setting-gripper-with-feedback
description: Opens, closes or releases the gripper through the configured YAM or Isaac workcell adapter and checks measured holding or width feedback. Use when a graph must let go of a placed piece or set the jaw width without moving the arm.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: grasp
  tags:
  - gripper
  - set_gripper
  - release
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
    - release
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

# setting-gripper-with-feedback

This bundle calls the `set_gripper` capability (contract `rsh-task/1`) through
`workcell.invoke`. The selected station determines whether it reaches YAM
hardware or an Isaac adapter.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json`, the complete runtime result serialized as JSON, and
  `status`, one of `succeeded`, `failed`, `unverified`, `rejected`. The result also
  carries `failure_type`, `phase` and `recovery_hints` for graph routing.
- A coordinator must route all four statuses.

## Behaviour

A `release` succeeds only when fresh gripper feedback reports no held object;
with no holding feedback the result is `unverified`. Whether a released domino
stays upright or a brick stays seated is a task postcondition checked by the
task's own verifier, not by this primitive.

## Capability request

Required JSON fields: `width_m`, `purpose` (`open`, `close` or `release`).

Optional JSON fields: `target_id` (also records the object before and after), `speed_m_s`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1)
for units, frames, evidence and station binding. `workcell.describe()` returns the
complete JSON Schema. No sim or real verification is claimed by this bundle.
