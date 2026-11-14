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
    left_task: string
    right_task: string
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
      note: The left skill on arm 0 and the right skill on arm 1, started together; each keeps its own exits.
    sequential:
      skill: running-two-skills-concurrently-on-two-arms
      note: Same two skills in sequential mode - the left task to its end, then the right one - so neither arm moves while the other works.
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
    concurrent: {both_succeeded: done, one_failed: partial, both_failed: abort, refused: sequential}
    sequential: {both_succeeded: done, one_failed: partial, both_failed: abort, refused: pick_place_two}
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
2. **The concurrent runner can still refuse.** It may find something the
   capsule check cannot see, such as a shared camera, a shared tool or a lease
   conflict. `refused` then goes to the sequential run, which needs no
   clearance between the arms.
3. **Sequential refused means the pair cannot run as given.** Only a
   pick-and-place pair has a further fallback, `picking-and-placing-two-objects`.
   It re-plans the assignment of objects to arms and staggers the two motions
   itself. For other pairs, its `unreachable` exit ends in `abort`.
4. **No retries inside the recipe.** `one_failed` ends in `partial`: one task
   is done and the other is not. Re-running the pair would redo the finished
   task. The planner should re-plan only the failed task, named in the
   runner's outputs. `both_failed` and `none_placed` end in `abort`. Both
   failure ends open the grippers.

## Inputs

`left_task`, `right_task`: the two single-arm jobs (skill and its inputs, as
the planner chose them). `object_a_obb` / `object_b_obb` / `place_a` / `place_b`
are read only by the pick-and-place fallback; pass them when both jobs are
pick-and-place, otherwise that branch ends in `abort` (`unreachable`).

## Exits of the concurrent runner (assumed)

This recipe routes `running-two-skills-concurrently-on-two-arms` by the exits
`both_succeeded`, `one_failed`, `both_failed` and `refused`. The skill was
still being built when this recipe was written (branch
`skills-running-two-skills-concurrently-on-two-arms`, not yet on the remote).
`roboskillhub check` fails this recipe, naming any exit that does not match,
until those names match the merged skill's `exit_conditions`.
