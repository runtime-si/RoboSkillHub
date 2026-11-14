---
name: kinoweave
description: Time-synchronised joint trajectories for the two YAM arms going to their own goals in a shared
  workspace, with waits/yields where their link spheres would meet (KinoWeave PRM + LaCAM search, CPU), resampled
  to the yam_gap/real joint-stream `tracks` format. WORK IN PROGRESS - see Status.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [bimanual, dual-arm, multi-arm-planning, cpu]
  hub:
    stage: plan
    embodiments: [yam-pair]
    version: 0.0.1
    provenance:
      code: https://github.com/botrail/kinoweave/blob/2da9034697f150c3e0b8a1a69c807d13fca69126/src/kinoweave/planner.py
      upstream_license: Apache-2.0
gap:
  requires: {}
  tools:
    - kinoweave.plan_two_arms: Plan both arms (hardware joint convention) start->goal on one clock; returns stream tracks, makespan, waits, validation, sequential baseline.
---

# kinoweave (WIP)

Status (2026-11-13): the YAM pair problem builds and validates (start states pass KinoWeave's endpoint and
sphere checks); a crossing problem with 200 PRM samples per arm and the full YAM sphere model (~90 spheres
per arm) hit the 20 s time limit on CPU, and so did the sequential baseline. Not yet usable: needs fewer spheres
for search (coarse set) with the full set for validation, smaller time_step, and benchmark numbers.

Known gap in tools/yam-spheres (PR #247): `chain` has no `link6` body (joint6); this bundle adds it from the
robosimstudio YAM MJCF (`LINK6` in tools.py).
