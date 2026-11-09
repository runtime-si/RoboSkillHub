---
name: moving-through-waypoints
description: Moves the measured tool pose through a bounded list of waypoints on the configured YAM or Isaac workcell adapter, verifying arrival at each, and optionally that a held object follows the TCP. Use when an agent must approach, descend or transport a held piece along an explicit path.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: motion
  tags:
  - motion
  - waypoints
  - move_waypoints
  - transport
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

# moving-through-waypoints

This bundle calls the `move_waypoints` capability (contract `rsh-task/1`) through
`workcell.invoke`. The selected station determines whether it reaches YAM
hardware or an Isaac adapter.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json`, the complete runtime result serialized as JSON, and
  `status`, one of `succeeded`, `failed`, `unverified`, `rejected`. The result also
  carries `failure_type`, `phase` and `recovery_hints` for graph routing.
- A coordinator must route all four statuses.

## Behaviour

`linear: true` requires a backend that declares `linear_cartesian_motion`;
no current backend does, so such requests are rejected rather than silently
executed as joint-space moves. With `held`, the observed object must stay within
`limits.held_object_tolerance_m` of `TCP · T_TCP_object`, otherwise the result is
`failed` with `held_object_slipped`.

## Capability request

Required JSON fields: `waypoints` (1 to 32 poses), `linear`.

Optional JSON fields: `held` (`object_id`, `T_TCP_object` in frame `tcp`), `speed_m_s`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1)
for units, frames, evidence and station binding. `workcell.describe()` returns the
complete JSON Schema. No sim or real verification is claimed by this bundle.
