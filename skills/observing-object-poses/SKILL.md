---
name: observing-object-poses
description: Reads independently observed object positions, and orientations where the backend provides them, from the configured workcell, labelling simulator truth as privileged. Use when a graph must re-observe pieces before choosing, grasping or verifying.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: perception
  tags:
  - perception
  - observe_objects
  - object-pose
  - observation
  - workcell
  - feedback
  - yam
  - isaac
  hub:
    stage: perceive
    embodiments:
    - yam_i2rt
    arms: 1
    version: 0.1.0
    tasks:
    - perception
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

# observing-object-poses

This bundle calls the `observe_objects` capability (contract `rsh-task/1`) through
`workcell.invoke`. The selected station determines whether it reaches YAM
hardware or an Isaac adapter.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json`, the complete runtime result serialized as JSON, and
  `status`, one of `succeeded`, `failed`, `unverified`, `rejected`. The result also
  carries `failure_type`, `phase` and `recovery_hints` for graph routing.
- A coordinator must route all four statuses.

## Behaviour

No motion is commanded. Every requested object must be fresh and visible
(and oriented, with `require_orientation`) for `succeeded`; otherwise the result is
`unverified` with a per-object `reason`. Observations from `simulation_truth` carry
`privileged: true` and must not be reported as vision results.

## Capability request

Required JSON fields: `object_ids` (1 to 32).

Optional JSON fields: `require_orientation`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1)
for units, frames, evidence and station binding. `workcell.describe()` returns the
complete JSON Schema. No sim or real verification is claimed by this bundle.
