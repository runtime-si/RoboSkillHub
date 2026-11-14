---
name: inserting-pauses-to-resolve-two-arm-conflicts
description: Makes two timed arm trajectories safe to run together without
  changing either path - the two arms' collision spheres (the YAM sphere
  model of tools/yam-spheres, bundled) are checked every 50 ms, and where
  they come within 10 mm the shorter trajectory holds at its last waypoint
  before the conflict, in 0.25 s steps, until the other arm has passed (the
  other arm waits instead when the shorter cannot wait clear). Use when two
  arms run independently planned joint trajectories in a shared workspace
  (both reaching into the middle of the station) and a re-plan is not
  wanted.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [bimanual, collision, scheduling, timing, spheres, cpu]
  hub:
    stage: plan
    embodiments: [yam]
    arms: 2
    version: 0.1.0
    tasks: [bimanual]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs: {}
  produces_outputs:
    delayed: string
    min_clearance_before: float
    min_clearance_after: float
    added_time: float
  exit_conditions:
    no_conflict: The arms never came within margin (10 mm between sphere surfaces); the trajectories are returned unchanged.
    pauses_inserted: One trajectory was given holds (pauses lists after which waypoint, when, how long) and the pair now stays over margin at every check; the paths are unchanged, only the timing.
    unresolved: Even max_pause (6 s) of holding on either arm leaves a conflict (the two arms' goals overlap, or one arm's resting pose is in the other's way); the last attempt's numbers are filled in.
  canonical_scripts:
    - insert_pauses: scripts/insert_pauses.py
  examples:
    - title: Two arms reaching into the middle
      path: examples/canonical_subgraph.json
---

# inserting-pauses-to-resolve-two-arm-conflicts

## What it does

`trajectory_a` / `trajectory_b`: `{times: [N], joints: [N x 6]}` (joint
angles in the model's convention, see the assets file's `joint1_note`),
linearly interpolated. Each arm's spheres come from
`assets/yam_spheres.json` (link frames from its kinematic chain; link 6 is
taken as link 5 turned by joint 6 at its origin - the chain lists no link-6
body), on the station bases (arm 0 left at y = +0.305 m, arm 1 right; or
`base_a` / `base_b` poses). The pair is checked every `check_dt` (50 ms); at
the first time the clearance (sphere surface to surface) is under `margin`
the delayed arm (`delay`: `shorter` by duration, or `a` / `b`) gets a 0.25 s
hold at its last waypoint before that time - or one waypoint earlier when
the conflict falls inside the hold - and the check repeats, up to
`max_pause`. With `shorter`, if that arm cannot wait clear, the other is
tried.

## Checks

FakeContext unit tests (`tests/bundles/test_inserting_pauses_to_resolve_two_arm_conflicts.py`):
no conflict, a clash resolved by delaying the shorter arm with the path
unchanged, a forced delay, the cap, bad inputs.

**Synthetic crossing tasks** (`python -m tests.bundles._pauses_eval`, CPU, 30
tasks): each arm goes home -> a random reach whose gripper lands in the
shared middle (|y| < 0.12 m, 0.05-0.30 m high) -> home over 4 s, the second
arm starting 0-1 s later. 18 of 30 conflict (median clearance -0.041 m, i.e.
the spheres overlap); 15 of 18 are resolved with pauses (added time median
1.0 s, max 3.0 s; clearance afterwards at least 0.011 m); 3 remain
unresolved (both goals overlap: no waiting helps). Before the fallback to the
other arm and the earlier hold, 10 of 18 were resolved.

## Limits

- Checked only at 50 ms samples of a linear joint interpolation; a fast
  swing between samples can be missed (the spheres' 15 mm fit margin and the
  10 mm margin cover about 0.5 m/s).
- Link 6 approximated (above); fingers taken closed.
- Timing only: overlapping goals need a re-plan (curobo) or a different
  order of tasks.
- No twin run.

## Provenance

The sphere model is RoboSkillHub's tools/yam-spheres (PR #247, branch
tools-yam-collision-spheres at 94bcb1c, `yam_spheres.json`), copied here
(the chain, bases and spheres) until that bundle is merged. The scheduling is
new.
