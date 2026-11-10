---
name: supporting-an-overhang-with-the-second-arm
description: Presses a LEGO / DUPLO brick onto a cantilevered or loosely
  held part with two arms - the support arm closes its hand and goes to a
  support pose under or beside the part (given as a pose, a point, or
  APEX-MR's support offset in studs from a brick), the other arm runs
  pressing-a-brick-onto-studs, and the support's read-back pose before and
  after says whether the structure moved. Use when a press would bend or
  break off an overhang, or push a loose sub-assembly away, and a second arm
  can reach a spot that takes the press force.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, assembly, bimanual, support, press]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [assembly, lego, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/APEX-MR/blob/6b5a17f1133c3f739caf50fa0fff6c478ef73b7a/src/lego_policy.cpp
      upstream_license: MIT
gap:
  requires: {connector: [robot.move_cartesian_until_contact, robot.describe_gripper, robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
    - robot.close_gripper
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    press_route: string
    press_detail: string
    support_used: bool
    support_drift_mm: float
  exit_conditions:
    seated: The press ended `seated` and the support hand drifted under moved_tolerance (3 mm) during it.
    not_seated: The press ended `not_seated` (partly pressed), the structure held still.
    misaligned: The press ended `misaligned`.
    not_holding: The press found no brick in its hand.
    unverified: Released, height not read.
    structure_moved: The support hand drifted more than 3 mm during the press - the part it holds gave way or the structure shifted.
    support_unreachable: The support arm could not reach the approach or the support pose (routed or raised refusal); nothing was pressed.
    unreachable: The press's own motion was refused.
  canonical_scripts:
    - support_and_press: scripts/support_and_press.py
  examples:
    - title: Press a brick with arm 0 while arm 1 holds the part under it
      path: examples/canonical_subgraph.json
---

# supporting-an-overhang-with-the-second-arm

APEX-MR builds LEGO with two arms; when a brick goes onto a part that hangs
out over nothing, the second arm first moves to a support pose (an offset of
`sup_dx`, `sup_dy` studs and `sup_dz` bricks from a brick of the structure,
from its task file) and stays there while the first arm presses. This skill
does that on any two-arm station through the hub's press skill.

## What it does

1. **Support pose**: `support_pose` (a full pose: from below or the side, the
   caller chooses the hand's rotation), else `support_point` with the support
   arm's top-down `robot.grasp_frame` rotation, else APEX-MR's
   `support_studs` `[dx, dy, dz]` from `support_from` (studs along the seat's
   yaw, brick heights up).
2. **Support**: the hand closes, comes to `approach_offset` from the pose
   (default 60 mm above), then to the pose plus `preload` (2 mm) along the
   approach - the servo stalls on the part.
3. **Press**: pressing-a-brick-onto-studs on `press_arm`, with `system`,
   `seat_yaw_deg` and `press_inputs`.
4. **Check**: the support's drift over 3 mm is `structure_moved`; the hand
   backs out along the approach.

## Checks

Unit tests (`tests/bundles/test_supporting_an_overhang_with_the_second_arm.py`)
with FakeContext and the press stubbed: each way to give the support, the
approach and preload, refusals, drift, route mapping, bad inputs.

**Not run in the twin yet.** The YAM twin has no overhang (a cantilevered
brick on a free sub-assembly with a clutch that can break) and the round's
time went to disassembling-a-stack-with-hold-and-twist's twin check, which
tests the same support move (a closed hand pressing down on a loose base,
reachable by the right arm with the plate at the workspace middle).

## Limits

- No twin or real check (above). APEX-MR's support analysis (which pose
  makes the structure stable, `stability_analysis_with_support.py`, Gurobi)
  is not ported: the caller gives the support.
- A top-down parallel gripper cannot get under a cantilever; supports from
  below need a `support_pose` with a sideways hand, reachable by that arm.
- Needs pressing-a-brick-onto-studs beside it in the skills root.

## Provenance

APEX-MR `LegoPolicy::support` (src/lego_policy.cpp) and its task files'
`sup_dx/sup_dy/sup_dz` (MIT): the idea and the offset convention, no code.
APEX-MR's pressdown (5 cm past contact, relying on the Yaskawa's compliance)
is deliberately not reproduced; the press is pressing-a-brick-onto-studs.
