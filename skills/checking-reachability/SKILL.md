---
name: checking-reachability
description: Asks a station whether TCP poses (optionally as a chained path from the current configuration) are reachable, using the backend's own IK and path rules, without moving. Use before committing to a grasp or place pose on an arm with joint limits, e.g. to choose between equal-score flipped grasp candidates.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: motion
  tags:
  - motion
  - ik
  - reachability
  - check_reachability
  - workcell
  - yam
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

# checking-reachability

This bundle calls the `check_reachability` capability (contract `rsh-task/1.3`)
through `workcell.invoke`. Nothing moves.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json` and `status`; `evidence.results` holds one `{ok, reason, q, source}`
  per pose and `evidence.all_ok` the conjunction.

## Behaviour

The backend must declare `reachability_query` and implement
`reachable(poses, chain, timeout_s)`. Poses outside the station workspace are answered
`ok: false` by the runtime without asking the backend. With `chain: true` each pose
is solved from the previous solution, starting at the current configuration, under
the same path rules the backend applies to `move_pose`. A `true` answer is a
kinematic prediction, not a guarantee against collisions or dynamics.

## Capability request

Required JSON fields: `poses`. Optional: `chain`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1).
No sim or real verification is claimed by this bundle.
