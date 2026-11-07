---
name: sliding-to-edge-then-picking
description: Picks a flat object too thin to pinch on its support - closed
  fingers slide it in a straight line until part of it overhangs the support's
  edge, then the open jaws come down beside the overhang, below the support's
  top, close on it from its two sides and lift. Use when a slab, card, plate,
  phone or book lies flat on a table and a top-down pinch cannot reach its
  sides, a free edge of the support is within reach, the object's OBB comes
  from a perception skill, the edge from a plane or table boundary (a point,
  or a point and a direction, world frame), and the hand is empty.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [pick, thin, flat, slide, nonprehensile, edge-grasp, environment-constraint]
  hub:
    stage: execute
    embodiments: [any]
    tuned_on: [yam_i2rt]
    version: 0.1.0
    tasks: [pick-thin-flat, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/google-research/ravens/blob/master/ravens/tasks/primitives.py
      upstream_license: Apache-2.0
      paper: >-
        Eppner, Deimel, Alvarez-Ruiz, Maertens, Brock, "Exploitation of environmental
        constraints in human and robotic grasping", IJRR 2015; Hang, Morgan, Dollar,
        "Pre-grasp sliding manipulation of thin objects using soft, compliant, or
        underactuated hands", RA-L 2019
gap:
  requires: {connector: [robot.grasp_frame, robot.move_cartesian_until_contact, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.close_gripper
    - robot.open_gripper
    - robot.grasp_frame
    - robot.solve_ik
  required_inputs:
    target_obb: OrientedBoundingBox
    edge_point: Vec3
  produces_outputs:
    overhang_before: float
    overhang_target: float
    slide_planned: float
    slide_achieved: float
    overhang_expected: float
    held_width: float
  exit_conditions:
    slid: (slide only) The hand travelled the planned slide (within 5 mm) and lifted off the slab, or the slab already overhung enough; `plan` holds the pinch. The entry point of pick.
    picked: The slab was slid until it overhung the edge by about overhang_target, the open jaws came down beside the overhang to the slab's mid-height, closed, and after a 50 mm lift the jaw gap still read wider than closed by more than 4 mm.
    slid_not_picked: The slide ran but the hand travelled short of it or could not lift off, or the pinch's hover was refused, or the open hand stopped more than 6 mm above the pinch height (a blade on the slab or the support), or the jaws closed on nothing (or lost it in the lift); `detail` says which. The slab may have moved; re-perceive it.
    unreachable: Nothing moved the slab - its centre is not inside the edge, it is too short across the edge to overhang enough with its centre on the support, too wide for the jaws, more than max_slide from the edge, the IK preview refused the end of the slide or the pinch, the hover was refused for every rotation, or the descent behind it (push) / onto it (press) failed.
  canonical_scripts:
    - slide_to_edge: scripts/slide_to_edge.py
    - pick_overhang: scripts/pick_overhang.py
  examples:
    - title: Slide a perceived slab over a table edge and pick it
      path: examples/canonical_subgraph.json
---

# sliding-to-edge-then-picking

A thin slab lying flat cannot be pinched from above: the fingertips would have
to reach below the support's top to get either side of it. Slide it until part
of it overhangs the support's edge, and the space under the overhang is free
for the fingertips.

## The motion

`slide_to_edge` (Ravens' straight-line push, with the drag length set by the edge):

1. From the slab's OBB and the edge, find the outward normal `n` (horizontal,
   from the slab's side toward the drop), the slab's current overhang and the
   one to reach: `overhang_fraction` (0.4) of its depth across the edge,
   capped so its centre stays 5 mm on the support. The slide is the
   difference. A slab too short across the edge to give a pinch room
   (a blade's width past the edge plus 5 mm on the slab) is refused: turn it
   so its long side lies across the edge first.
2. Preview the end of the slide and the pinch with `robot.solve_ik`; refuse
   (`unreachable`) before touching the slab if either is out of reach.
3. Close the hand. Hover top-down with the closed fingers side by side across
   the slide (`robot.grasp_frame`, jaws closing along the edge; both wrist
   solutions, then the hand's own rotation). If the arm arrived with its
   wrist unturned (a solver that leaves the roll free), read the heading back
   and finish the turn on the last joint (`robot.move_to_joints`).
4. Come down to the slab:
   - `push` (default): `standoff` behind its back face (the closed blades'
     front, `pad_half_width` ahead of the tool centre, 3 mm off the face), to
     its mid-height, floored at 15 mm of tool centre over the support; a stop
     short within `reach_tolerance` (fingertips on the support) is accepted
     and the drag runs at that height.
   - `press`: onto its top, 12 mm inside its back edge, with
     `robot.move_cartesian_until_contact` aimed at the 15 mm floor; a stop
     more than 1.5 mm above the floor is the slab (judged by height: a servo
     calls a stop within its arrival ball "reached"); then `press_depth`
     (1.5 mm) deeper. The drag relies on the fingertips' friction beating the
     support's, which it did not in the twin (below).
5. Drag along `n` in 10 mm moves (Ravens' step), each read back; stop when
   the hand stops advancing. Lift `hover_clearance`.

`pick_overhang`:

6. Hover over the pinch: on the slab's centreline, a blade's half width plus
   3 mm past the edge, with the jaws along the edge; turn the last joint again
   if needed. Open.
7. Come down with `robot.move_cartesian_until_contact` until the tool centre
   (between the pads) is at the slab's mid-height, the fingertips below the
   support's top beside the overhang. A stop more than 6 mm high means a
   blade landed on the slab or the support: back up, `slid_not_picked`.
8. Close, lift 50 mm, and read the jaw gap (`robot.get_gripper`): wider than
   closed by more than 4 mm is `picked`.

## Inputs

- `target_obb`: the slab's oriented box, world frame (half-extents).
- `edge_point`: a point on the support's edge, world frame (z unused).
- `edge_direction` (optional): the edge's direction. Without it the edge is
  taken square to the line from the slab's centre to `edge_point`.
- `support_z` (optional): the support's top; default the slab's lowest corner.
- `mode`: `push` (default) or `press`; see step 4.
- `overhang` (optional, m) overrides `overhang_fraction`; `max_slide` (0.25 m),
  `standoff` (0.02 m), `step` (0.01 m), `hover_clearance` (0.08 m),
  `contact_fraction` (0.5), `min_tool_height` (0.015 m), `press_depth`
  (0.0015 m), `grasp_height` (default half the slab's height),
  `pad_half_width` (0.010 m), `reach_tolerance` (0.02 m), `speed_scale` (passed
  to the drag moves only when set), `roll_limits`, `check_reach` (true),
  `arm_id`.
- pick: `plan` and `slide_route` from the slide; `lift` (0.05 m),
  `grasp_tolerance` (0.006 m), `empty_margin` (0.004 m).

## Required end states

- `picked`: the slab is held in the jaws 50 mm over where they closed.
- `slid_not_picked`: the slab was pushed (or the push was tried) but not
  picked; `slide_achieved` and `overhang_expected` are the hand's travel and
  what it implies, not a measurement of the slab. Re-perceive before retrying.
- `unreachable`: the slab was not touched (or the hand never reached it).

The hand must be empty at the start: the slide closes it.

## Real-robot notes

- **The pinch's tool centre is under a real YAM station's floor.** The station
  refuses a tool centre under 14 mm over its calibrated table; the pinch puts
  it at the slab's mid-height (4 mm for an 8 mm slab), past the table's edge
  where nothing is under the hand. The station's floor has to allow that
  region, or set `grasp_height` to 0.014 m and accept a pinch on the top
  4 mm of the slab's sides.
- No force sensor: contact is the arm's own stall (`move_cartesian_until_contact`)
  and the press depth is commanded, not measured (the candidate's 1-2 mm).
- `PAD_HALF_WIDTH` (10 mm) is an estimate of the linear_4310 blade, which
  `robot.describe_gripper` does not state. The push stops the blades' front,
  this far ahead of the tool centre, at the planned overhang; in the twin the
  slab ended 0-2 mm past the estimate.
- The overhang and the pinch are judged by the hand (its travel, its stop,
  its jaw gap). Re-perceive the slab between the two scripts when a camera
  can see the edge: the slide's estimate cannot see the fingers slip.

## The constants, and why

- **`STEP = 0.01` m and `STANDOFF = 0.02` m** are Ravens' push step and
  pre-push offset.
- **`OVERHANG_FRACTION = 0.4`, `COM_INSET = 5 mm`**: the papers slide until
  "part" of the object overhangs; 0.4 of a 60 mm slab is 24 mm, leaving its
  centre 6 mm on the support.
- **`PAD_HALF_WIDTH + EDGE_CLEARANCE` = 13 mm**: where the pinch sits past
  the edge, so the blade clears it.
- **`MIN_TOOL_HEIGHT = 0.015` m, `CONTACT_FRACTION = 0.5`, `HOVER_CLEARANCE`,
  `REACH_TOLERANCE`**: pushing-objects' values.
- **`GRASP_TOLERANCE = 6 mm`**: a servo's 4 mm arrival ball plus 2 mm; a
  blade on the support stops about 6 mm high, on the slab about 14 mm.
- **`PRESS_DEPTH = 1.5 mm`**: the candidate's "fixed extra depth (1-2 mm)".
- **`EDGE_PADDING = 3 mm`**: pushing-objects' gap at the contact point.
- **`PRESS_MIN_STOP = 1.5 mm`**: a press that ends this close to the floor met
  nothing; an 8 mm slab stopped the hand 3 mm above it in the twin, so a slab
  thinner than about 6.5 mm cannot be told from the table at the 15 mm floor.
- **`YAW_TOLERANCE_DEG = 8`**, last-joint turns as stacking-objects: the twin's
  IK left the jaws 64 deg off the asked heading at the first hover, and one
  last-joint turn brought them within 0.1 deg; the push's reaction turned the
  hand 14 deg over the 10 cm drag, and the pinch's hover found it 20 deg off
  and turned it back to 0.1 deg (all three runs).

## Checked in simulation

RoboSimStudio, a YAM station twin (left arm, servo motion, an IK that leaves
the wrist roll free), `push` mode, defaults. The bench top is 0.595 x 1.30 m
(x -0.2975..+0.2975, y -0.65..+0.65 about its centre; the arm bases stand at
x = -0.3475, y = +-0.305). A 60 x 40 x 8 mm, 23 g slab (friction 0.8) lay with
its long side across the edge and its near face 80 mm from it; the overhang
and tilt between the slide and the pinch, and after the lift, are the
simulator's, measured by the harness, not by the skill.

- **Front edge** (x = +0.2975, 0.645 m ahead of the arm's base): the IK
  preview refused the end of the slide; `unreachable`, slab untouched.
- **Left side edge** (y = +0.65), three runs:

  | run | slab | hand's slide | overhang reached (expected) | tilt after slide | exit | after the 50 mm lift |
  |---|---|---|---|---|---|---|
  | 1 | nominal | 101.5 mm | 23.2 mm (21.5) | 0.006 deg | `picked` | jaws 39.9 mm, slab up 58 mm, tilted 15.6 deg in the jaws |
  | 2 | 10 mm along the edge, turned 5 deg | 101.5 mm | 22.9 mm (23.1) | 0.005 deg | `picked` | jaws 39.8 mm, up 57 mm, tilted 12.0 deg |
  | 3 | -10 mm, turned -5 deg | 101.6 mm | 23.2 mm (23.2) | 0.005 deg | `picked` | jaws 40.3 mm, up 29 mm, tilted 58 deg |

  The pinch holds the slab on the 23 mm overhang only, its centre of mass
  about 20 mm from the jaws, and the slab turns about the jaws' closing axis
  as it leaves the table: 12-16 deg twice, 58 deg (hanging) once. `picked`
  says the jaws hold it, not at what attitude.
- **Before the blade offset**: with the contact computed at the tool
  centre, the closed blades (about 10 mm ahead of it) pushed the slab 10 mm
  further than planned, past balance, and it fell off the bench; the pinch
  closed on nothing and routed `slid_not_picked`. The slide now stops the
  blades, not the tool centre, at the planned overhang; the three runs above
  put the slab within 2 mm of the hand's estimate.
- **`press` mode**, nominal slab: the fingertips met the slab's top 3 mm above
  the 15 mm floor and the drag ran its 102 mm, but the fingertips slid over the
  slab (it moved 1.7 mm): fingertip and table friction are equal in the twin.
  The hand's travel cannot see that slip; the pinch closed on nothing and
  routed `slid_not_picked`. Use `press` only where the fingertips grip
  better than the support (rubber pads on a smooth table).

## Provenance

The drag is adapted from Ravens' `push` primitive
(`ravens/tasks/primitives.py`, Apache-2.0): world-frame slab and edge in place
of two push poses, the drag length from the overhang wanted, connector moves
routed instead of a timeout flag, one 10 mm step per move (Ravens' loop sends
every step to the end point), a contact descent instead of a fixed 5 mm
height, the closed fingers across the drag instead of a spatula yaw. The
slide-to-edge-then-pinch strategy is reimplemented from Eppner et al. (IJRR
2015) and Hang et al. (RA-L 2019), which ship no code (HKUST-RML's
Scooping-Manipulation has no licence and was not used). Ideas from
pushing-objects (the hover behind, the accepted short descent) and
placing-with-contact-descent (contact moves judged by the hand's stop).
