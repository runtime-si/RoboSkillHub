---
name: running-two-arm-tasks-at-once
description: Let each arm of a two-arm YAM station do its own job at the same
  time -- check the two planned motions for cross-arm clearance, run the two
  single-arm skills concurrently, and when concurrency is refused (too close,
  unknown clearance, a shared resource) run them one after the other; for a
  pair of pick-and-place jobs, fall back to the coordinated two-object
  pick-and-place. Use when a planner has two independent single-arm tasks for
  the left and the right arm (sort pens back into the black holder while the
  other arm inserts a pen; clear a tile while the other arm places one) and
  wants them done in parallel when that is safe.
license: Apache-2.0
metadata:
  hub:
    version: 0.1.0
    embodiments: [yam_station]
    tuned_on: [yam_station]
    tasks: [bimanual, pick-place, sorting, insertion, jigsaw]
    tier: camera_only
recipe:
  inputs:
    skill_a: string
    skill_b: string
    object_a_obb: OrientedBoundingBox
    object_b_obb: OrientedBoundingBox
    place_a: Vec3
    place_b: Vec3
  start: clearance
  nodes:
    clearance:
      skill: checking-dual-arm-clearance
      note: Over both arms' planned motions (each skill's plan or preview); with no plan, the current configuration.
    concurrent:
      skill: running-two-skills-concurrently-on-two-arms
      note: skill_a on arm 0 (left) and skill_b on arm 1 (right), one thread each, every move through the shared clearance arbiter; route_a / route_b carry each skill's own exit.
    sequential:
      skill: running-two-skills-concurrently-on-two-arms
      note: The same runner in its sequential mode (skill_a to its end, then skill_b; the mode is being added on the skill's branch) - neither arm moves while the other works, so a deadlock cannot recur.
    pick_place_two:
      skill: picking-and-placing-two-objects
      note: Only when both tasks are pick-and-place of one object each (object boxes and place points known); it assigns objects to arms and staggers the motions itself.
    done: {end: success}
    partial:
      end: failure
      recovery: [robot.open_gripper]
    abort:
      end: failure
      recovery: [robot.open_gripper]
  routes:
    clearance: {clear: concurrent, too_close: sequential, unknown: sequential}
    concurrent: {both_done: done, one_failed: partial, deadlock_timeout: sequential, unsafe_refused: pick_place_two}
    sequential: {both_done: done, one_failed: partial, deadlock_timeout: abort, unsafe_refused: pick_place_two}
    pick_place_two: {both_placed: done, one_placed: partial, none_placed: abort, unreachable: abort}
---

# Each arm its own job, at the same time

For a planner with two independent single-arm tasks on a two-arm YAM
station. The tasks run concurrently only when the arms provably keep apart.
Otherwise they run one after the other, and a pick-and-place pair has a
coordinated two-object skill as a last resort.

## Example task pairs

| left arm | right arm | notes |
|---|---|---|
| sort the pens back into the black holder | insert a held pen into a lattice hole | the holder and the lattice must lie on each arm's side; clearance decides |
| clear a jigsaw tile off the board | place a tile on the board | both work over the board: expect `too_close` often, so sequential |
| put object A into box A | put object B into box B | if concurrency is refused, `picking-and-placing-two-objects` takes over |

## Why this order

1. **Clearance first, not after a collision.** `checking-dual-arm-clearance`
   samples the capsule model of both arms over their planned motions.
   - `clear`: run concurrently.
   - `too_close`: run sequentially; never concurrently.
   - `unknown` (no second arm reading, FK mismatch, no IK for a waypoint): also
     sequentially, because an unverified clearance is not a clearance.
2. **The runner arbitrates every move.** `running-two-skills-concurrently-on-two-arms`
   predicts each move's clearance against the other arm and makes an arm wait
   while it would come too close.
   - `deadlock_timeout` (the arms kept blocking each other): the sequential
     mode, where one arm stands still while the other works.
   - `unsafe_refused` (clearance not predictable, the arms already too close, no
     arm assignment reaches both targets, a skill does not load): sequential
     would hit the same wall, so a pick-and-place pair goes to
     `picking-and-placing-two-objects`, which re-plans the assignment of objects
     to arms and staggers the motions itself. For other pairs that node's
     `unreachable` ends in `abort`.
3. **No retries inside the recipe.** `one_failed` ends in `partial`: the runner
   has paused the other arm at its start pose, and the finished task must not
   be redone. The planner re-plans the failed skill only (`route_a` / `route_b`
   name it). A `deadlock_timeout` in sequential mode would be a fault, so it
   aborts. Both failure ends open the grippers.

## Inputs

`skill_a`, `skill_b`: the two single-arm hub skills (with their inputs, as the
planner chose them), `skill_a` for the left arm and `skill_b` for the right. `object_a_obb` / `object_b_obb` / `place_a` / `place_b`
are read only by the pick-and-place fallback; pass them when both jobs are
pick-and-place, otherwise that branch ends in `abort` (`unreachable`).

## The concurrent runner

Exits as on branch `skills-running-two-skills-concurrently-on-two-arms`
(PR #246): `both_done`, `one_failed`, `deadlock_timeout`, `unsafe_refused`.
Its sequential / interleaved mode is still being added there. The
`sequential` node uses it through its note, because recipe nodes carry no
parameters. If the mode lands under other exit names, `roboskillhub check`
names them.
