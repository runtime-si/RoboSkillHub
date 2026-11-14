---
name: trying-arms-in-fallback-order
description: Picks an object with whichever of two arms can - the arm whose
  hand is nearer the object first, the other if the first cannot reach it
  (IK preview, a refused move) or closes on nothing - each attempt an IK
  preview, a top-down approach, descent, close, lift and a jaw-opening check;
  every attempt's failure reason is kept. Use when a two-arm station picks
  objects anywhere on a shared table and either hand may be the one that
  can, instead of routing a failed pick back into the graph by hand.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, pick, fallback, behaviour-tree, selector]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/personalrobotics/geodude/blob/80edf9d/src/geodude/robot.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_gripper
  required_inputs:
    target_position: Vec3
  produces_outputs:
    arm_used: int
  exit_conditions:
    picked_by_first: The first arm in order (the nearer one, or arms[0] with order 'given') holds the object, lifted lift_height over the grasp.
    picked_by_second: The first arm could not reach (IK preview or a refused move) or closed on nothing, and the second arm holds the object; attempts says why the first failed.
    both_failed: Neither arm picked it; attempts lists each arm's reason (unreachable / missed); both hands are open.
  canonical_scripts:
    - try_arms: scripts/try_arms.py
  examples:
    - title: Pick with the nearer arm, fall back to the other
      path: examples/canonical_subgraph.json
---

# trying-arms-in-fallback-order

A behaviour-tree Selector over two pick Sequences (geodude's two-arm pickup:
the arm on the object's side first, the other as a fallback).

## What it does

1. **Order** (`order`: `nearest` by the hands' current horizontal distance
   to the target, or `given` = `arms` as listed).
2. **Each attempt**: the arm's `robot.grasp_frame` rotation (else its
   current one); IK preview of the pre-grasp (`approach_height` 80 mm over)
   and the grasp (`target_position` + `grasp_height`) - out of reach =
   `unreachable`, nothing moves; open, pre-grasp, grasp, close, lift
   (`lift_height` 80 mm); a refused or raised move opens the hand and is
   `unreachable`; a jaw opening at or under 0.05 after closing is `missed`
   (the hand opens again).
3. The first `picked` ends the skill; else the next arm.

## Inputs / outputs

`target_position` (Vec3, the grasp centre), `arms` ([0, 1]), `order`,
`grasp_height`, `approach_height`, `lift_height`. Out: `arm_used` (-1 if
none), `order`, `attempts` (per arm: result and detail), `detail`.

## Checks

FakeContext unit tests (`tests/bundles/test_trying_arms_in_fallback_order.py`):
the nearer arm picks; out of reach -> the other arm (the first never
moves); a miss -> the other arm, the first reopens; routed and raised
refusals; both failed; given order and nearest order on the other side;
bad arguments. Not run in the twin (time).

## Limits

- The grasp is a plain top-down pick at a point; give a grasp from a grasp
  skill (selecting-reachable-grasp, proposing-short-axis-grasps) for
  anything else, or use this as the pattern around another pick skill.
- "Nearer" is by the hands' current positions, not the arm bases.
- A pick that holds the wrong thing reads as picked (jaw opening only).

## Provenance

geodude (github.com/personalrobotics/geodude, `Robot.pickup` in
`src/geodude/robot.py`, and the earlier `geodude_pickup_bimanual` Selector,
MIT): the pattern only - preferred arm first, the other as fallback, each
failure reason kept. No code is used.
