---
name: disassembling-a-stack-with-hold-and-twist
description: Frees a clutched LEGO / DUPLO brick from a loose base with two
  arms - the support arm closes its hand and presses the base (the plate
  beside the brick, or the brick under it) down, the other arm runs
  disassembling-a-brick-by-twist-lever (grasp, lever about an edge, lift),
  and the support's read-back pose before and after says whether the base
  moved. Use when the base is not clamped (a loose plate, a sub-assembly, a
  stack on the table), so a one-arm twist would drag the base along, and a
  second arm can reach a free spot of the base.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, disassembly, bimanual, support, hold, twist]
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
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
    - robot.set_grip
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    twist_route: string
    twist_detail: string
    support_used: bool
    support_drift_mm: float
  exit_conditions:
    separated: The twist-lever skill ended `removed` and the support hand drifted less than base_moved_tolerance (3 mm) while it ran - the brick is in the twist hand, the base where it was.
    base_moved: The support hand drifted more than 3 mm during the twist (the base came along or shifted), or the twist skill read `neighbour_lifted`.
    clutch_too_strong: The twist skill ended `stuck` or `slipped`.
    missed: The twist skill found no brick to grip.
    unverified: The twist lifted free but the seat was not read.
    support_unreachable: The support arm could not get over or onto the support point (routed or raised refusal); nothing else was moved.
    unreachable: The twist skill's motion was refused.
  canonical_scripts:
    - hold_and_twist: scripts/hold_and_twist.py
  examples:
    - title: Twist a brick off a loose plate with arm 0 while arm 1 presses the plate edge
      path: examples/canonical_subgraph.json
---

# disassembling-a-stack-with-hold-and-twist

disassembling-a-brick-by-twist-lever levers a brick off studs that do not
move. On a loose plate - or a brick on a brick - the clutch is stronger than
the base's weight, and the base comes along. APEX-MR's answer for its two
arms is the support primitive: the second arm goes to a support pose on the
assembly and stays there while the first acts. Here the support is a closed
hand pressing the base down.

## What it does

1. **Support** (`support`, default true): the support arm closes its hand,
   moves over `support_point` (a free spot on the base's top, e.g. the plate
   strip beside the brick) at 60 mm, and descends to put its fingertips
   `press_depth` (4 mm) below it - the servo stalls on the base, which is
   the preload. Rotation: `robot.grasp_frame` of that arm (top-down), else its
   current one. The tool centre is kept over 14 mm above the table.
2. **Twist**: disassembling-a-brick-by-twist-lever's script, loaded from the
   same skills root (or `GAP_SKILLS_PATH`), run on `twist_arm` with
   `seat_position`, `seat_yaw_deg` and `twist_inputs` (any of its inputs).
3. **Check**: the support's tool centre read back; a drift over 3 mm is
   `base_moved`. The support hand rises 60 mm.

## Inputs

`seat_position` (required), `support_point` (required with support),
`support` (true), `seat_yaw_deg`, `twist_arm` (0), `support_arm` (1),
`press_depth` (0.004), `base_moved_tolerance` (0.003), `twist_inputs`.

## Checks

Unit tests (`tests/bundles/test_disassembling_a_stack_with_hold_and_twist.py`,
13) with FakeContext and the twist skill stubbed: the support press (and a
press leg that stalls on the base), drift, route mapping, refusals,
no-support, the 14 mm floor, bad inputs.

**YAM twin** (two arms; the twist-lever check's DUPLO scene with the plate
made free, 15 g, and the clutch applied as a pair of forces between brick and
plate, 2.5 N per stud; harness not in the hub):

| plate (bench) | support | trials | clean separation | routes |
|---|---|---|---|---|
| (-0.03, 0.17), default | none | 3 | 0 | base_moved 3 (plate dragged 33-112 mm or thrown) |
| (-0.03, 0.17) | right arm, plate's -y strip | 3 | 0 | support_unreachable 3 (IK: no pose there) |
| (-0.08, 0.0), middle | none | 3 | 0 | missed 3 (the twist skill's open jaws stopped 10-12 mm over the grasp) |
| (-0.08, 0.0) | right arm | 1 | 0 | support_unreachable: the press leg timed out on the plate - the intended stall, which version 0.1.0 now accepts (not re-run) |

| (-0.08, 0.09) | none | 2 | 0 | missed 2 (jaws stopped 27 mm over the grasp) |
| (-0.08, 0.09) | right arm, -y strip, 4 mm | 2 | 0 | support held (drift 0.0 mm), but its press pushed the loose plate 7.8 mm; missed 2 |
| (-0.08, 0.09), brick turned 90 deg | none | 2 | 0 | unverified 2: the brick lifted ~100 mm **with the plate still clutched to it** (plate up 101-102 mm) |
| (-0.08, 0.09), brick turned 90 deg | right arm on the corner stud, 2 mm | 1 | 0 | base_moved (support drift 3.9 mm, twist slipped); the plate then blew up in the solver - not a valid trial |

| (-0.08, 0.09), 90 deg, plate 60 g | none | 3 | 0 | unverified 3: brick lifted with the plate still clutched (plate up 99.3 / 100.8 / 100.6 mm) |
| (-0.08, 0.09), 90 deg, plate 60 g | right arm, corner stud, 0.5 mm | 2 | 0 | base_moved 2 (support drift 3.87 / 3.81 mm, twist slipped), then the solver blew up |

The 60 g rows stand for a plate carrying a few bricks (a 15 g plate is the
lightest case); the drift is the same at 15 g / 2 mm and 60 g / 0.5 mm, so it
is not the support's press: the lever pulls the still-clutched plate up into
the support fingers, the brick slips out of the twist hand, and the rigid
contact between a held plate and a pair-force clutch blows up in the solver.

The IK probe found all four support points reachable with the plate moved
towards the middle (bench y <= 0.09). The no-support rows show the problem
the skill is for (the base comes along). Support has **not** yet been shown
to help: the support press shoves a 15 g plate sideways in the twin, and the
twist-lever skill's grasp only works at some plate spots and brick yaws.

## Limits

- **Twin only**, and the clutch is a model (a constant pair of forces
  between brick and plate over the 4.5 mm engagement), not a measured one.
- The support point must be reachable by the support arm: on the YAM
  station a plate 0.17 m off the middle towards the twist arm was not
  (every support pose failed IK); check with `robot.solve_ik` first.
- Needs disassembling-a-brick-by-twist-lever beside it in the skills root.
- The support press is position-servoed; a real arm with a stiff servo
  presses harder than the twin's for the same 4 mm.

## Provenance

The two-arm support pattern is APEX-MR's `LegoPolicy::support`
(src/lego_policy.cpp, MIT): read for the idea, no code used. The twist is
the hub's disassembling-a-brick-by-twist-lever, called as is.
