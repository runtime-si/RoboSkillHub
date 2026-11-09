---
name: pressing-with-contact-limit
description: Runs one bounded contact motion along an axis that the controller stops on a force threshold or travel limit, on a backend that declares a calibrated contact-force signal and a compliant controller. Use when a graph must press or seat a part with a measured force limit; position-only pressing is refused.
license: Apache-2.0
compatibility: requires the RoboSkillHub runtime and a configured workcell; gap>=0.1
  for graph execution
metadata:
  category: contact
  tags:
  - contact
  - guarded_press
  - force-limit
  - press
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
    - assembly
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

# pressing-with-contact-limit

This bundle calls the `guarded_press` capability (contract `rsh-task/1.1`) through
`workcell.invoke`.

## Interface

- Input: `request_json`, a JSON object serialized as a string.
- Output: `result_json`, the complete runtime result serialized as JSON, and
  `status`, one of `succeeded`, `failed`, `unverified`, `rejected`, with
  `failure_type`, `phase` and `recovery_hints` for routing.

## Behaviour

The backend must declare `contact_force_feedback` and `compliant_motion`, a
`contact_signal` (source, unit N, rate, maximum latency) and a `controller` of kind
force, admittance, impedance or hybrid, and implement `guarded_move`; the station
limits must set `max_contact_force_N`. Otherwise the request is rejected before any
motion. The stop-on-force loop runs in the controller, not in this process.
Success means only that the bounded motion ended on its force threshold or travel
limit with a confirmed stop; whether a part is seated is the task verifier's call.
No current YAM or Isaac backend declares these capabilities.

## Capability request

Required JSON fields: `axis` (`frame_id`, `vector`), `max_travel_m`, `speed_m_s`, `force_limit_N`.

Optional JSON fields: `hold_s`, `timeout_s`.

See [the workcell runtime contract](../../docs/workcell-runtime.md#task-contract-v1).
No sim or real verification is claimed by this bundle.
