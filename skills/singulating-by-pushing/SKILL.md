---
name: singulating-by-pushing
description: Pushes a target clear of the objects crowding it until a top-down
  grasp's open fingers fit around it - picks the push heading toward the most
  free space around the target, turned until the closed hand can come down
  behind the target without touching a neighbour, sizes each push to just free
  the fingers (at most push_step), re-checks after every push, and reports the
  gap to the nearest neighbour and the fingers' clearance before and after.
  Use when a target touches or nearly touches other objects so the open jaws
  would land on a neighbour, with OBBs of the target and its neighbours from a
  perception skill and the hand empty.
license: BSD-2-Clause
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [push, nonprehensile, singulation, clutter, pre-grasp]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/andyzeng/visual-pushing-grasping/blob/master/robot.py
      upstream_license: BSD-2-Clause
      paper: https://arxiv.org/abs/2111.11156
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper]}
  allowed_tools:
    - robot.describe_gripper
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.close_gripper
    - robot.open_gripper
    - robot.grasp_frame
  required_inputs:
    # OPTIONAL extra input (not listed: gap has no list type, so it can only
    # be wired from a script node inside the same subgraph):
    # `neighbour_obbs: list[OrientedBoundingBox]`, more neighbours.
    target_obb: OrientedBoundingBox
    neighbour_obb: OrientedBoundingBox
  produces_outputs:
    pushes: int
    gap_before: float
    gap_after: float
    clearance_before: float
    clearance_after: float
    fits_before: bool
    fits_after: bool
    required_gap: float
    expected_target_obb: OrientedBoundingBox
  exit_conditions:
    singulated: The open fingers of a top-down grasp fit around the target with FIT_MARGIN to spare - already at the start (nothing pushed) or, by the hand's travel, after `pushes` short pushes; re-perceive before grasping.
    still_cluttered: Pushes ran out (max_pushes), a push came up short (under half its length), or no usable heading would increase the fingers' clearance; the hand is open and lifted.
    unreachable: The closed hand fits behind the target in no heading before the first push, the target is wider than the opening along every grasp axis, or a hover, descent or lift was refused; the hand is open.
  canonical_scripts:
    - singulate_by_pushing: scripts/singulate_by_pushing.py
  examples:
    - title: Push a perceived target clear of its neighbours
      path: examples/canonical_subgraph.json
---

# singulating-by-pushing

Make room for the fingers before a grasp, by pushing the target away from
what crowds it.

## What "singulated" means here

A top-down grasp opens to `opening` (robot.describe_gripper's
`width_fit.at_open_m`, else `span_m`, else the YAM's 94 mm) and closes along
one of the target box's two horizontal axes. Each open fingertip is a
`finger_thickness` x `finger_width` rectangle just outside the opening. The
fingers' **clearance** is the least distance from those rectangles to any
neighbour's outline (the box's 8 corners projected on the table); the target
is singulated when it is at least `FIT_MARGIN` (3 mm). `need="both"` (default)
asks this of every axis the target fits between the jaws along, `need="any"`
of one; `grasp_heading_deg` checks only the closing heading a grasp proposer
already chose. `gap_before` / `gap_after` are the plain distance from the
target's outline to its nearest neighbour (0 when touching); `required_gap`
is the direction-free gap that frees every side,
`(opening - narrow width) / 2 + finger_thickness + FIT_MARGIN`.

## The push

1. **Heading** (Kiatos et al.'s rule, on outlines instead of a heightmap):
   in a 150 mm disc around the target, the free space at a point is its
   distance to the nearest neighbour (and to the `workspace` edge, if given);
   the push heads toward the nearest point within 90 % of the most free space,
   i.e. away from the neighbours.
2. **Insertion point**: headings are tried nearest that one first, every 10
   degrees. A heading is usable when the closed hand (a `hand_radius` disc),
   coming down `standoff` behind the push point (the target's back face along
   the heading plus 3 mm) and travelling along the push, keeps
   `HAND_MARGIN` from every neighbour, and the target's path does not run
   into a neighbour (sliding along one it touches is allowed). This is what
   turns a push straight away from a touching neighbour (the hand would have
   to come down inside it) into a sideways or slanted one.
3. **Length**: the shortest push, in 2 mm steps from 10 mm to `push_step`
   (40 mm), after which the fingers' clearance is at least 7 mm (3 mm margin
   plus 4 mm for slip); if no usable heading gets there in one push, the one
   leaving the most clearance after `push_step`.
4. **Motion** (VPG's push primitive, via pushing-objects' moves): close the
   hand, hover `hover_clearance` over the lowered point with the closed
   fingers across the push (`robot.grasp_frame`, both wrist solutions, then
   the hand's own rotation), straight down to half the target's height
   (never under 15 mm over the support), straight along the push, straight
   up.
5. **Re-check**: the target's outline is moved by the hand's measured travel
   past the push point, and the clearance is checked again; up to
   `max_pushes` (3) pushes. The hand opens at the end.

The re-check uses the hand's travel, not a new perception: an object can slip
or turn off the line. Run a perception skill afterwards and, if the fresh
OBBs still do not fit, run this skill again (it returns `singulated` at once,
without moving, when they do).

## Inputs

- `target_obb`, `neighbour_obb`: world-frame oriented boxes (half-extents)
  of the target and the neighbour crowding it. `neighbour_obbs` (optional, a
  list) adds more neighbours; gap's type registry has no list type, so a
  graph can pass it only from a script node in the same subgraph. Pass only
  things that can be pushed or must not be hit; the table is not a
  neighbour.
- `need` ("both"), `grasp_heading_deg` (none): which grasp must fit (above).
- `opening`, `finger_thickness` (0 = from describe_gripper, else the
  constants), `finger_width` (20 mm), `hand_radius` (10 mm).
- `push_step` (0.04 m), `max_pushes` (3), `standoff` (0.02 m),
  `hover_clearance` (0.08 m), `reach_tolerance` (0.02 m).
- `min_contact_height` (0.015 m): the lowest tool-centre height over the
  support during the push. The default keeps a station that refuses a tool
  centre under 14 mm over its table happy, but its fingertips then pass
  ~5 mm over the support, above any object thinner than that (a 2 mm jigsaw
  piece). Lower it only when the support height has just been measured
  (e.g. a touch-off) and the station's floor allows it.
- `workspace` (none): `{x_min, x_max, y_min, y_max}`; the target stays inside
  and its edge counts as an obstacle for the free-space heading (VPG clamps
  its push end to the workspace limits).

## The constants, and why

- **`OPENING = 0.094` m**: the linear_4310's span as a real YAM station's
  describe_gripper reports it; used only when the connector states none.
- **`FINGER_THICKNESS = 10 mm`, `FINGER_WIDTH = 20 mm`, `HAND_RADIUS = 10 mm`**:
  estimates of the tapered linear_4310 blade tips; describe_gripper reports
  `pad_thickness_m: 0` (unmeasured), so they are parameters. They decide which
  headings are usable: with a 10 mm hand radius and a 3 mm margin, the hand
  needs about 26 mm of free space to come down, so it never goes into a
  5 mm gap; it pushes from a free side instead.
- **`PUSH_STEP = 0.04` m, `MAX_PUSHES = 3`**: VPG pushes 0.10 m and Kiatos et
  al. 0.03-0.15 m; short, re-checked pushes drag a touching neighbour less.
- **`FIT_MARGIN = 3 mm`, `PLAN_SLACK = 4 mm`, `HAND_MARGIN = 3 mm`**: chosen
  for perception noise of a few millimetres.
- **`STANDOFF = 0.02` m** (pushing-objects uses 0.06 m): the hand comes down
  close behind the target so it fits between neighbours.
- **`EDGE_PADDING`, `HOVER_CLEARANCE`, `CONTACT_FRACTION`,
  `MIN_CONTACT_HEIGHT`, `REACH_TOLERANCE`** are pushing-objects' (see there).
- **150 mm free-space disc, 0.9 x max rule**: Kiatos et al.'s longest push and
  their argmax rule.

## Measured

In a YAM station twin (left arm, linear_4310, 94 mm span), two 30 mm cubes
on the table, the target's and the neighbour's boxes read from the simulator
before and after (a stand-in for perception), defaults throughout
(`need="both"`). The graph ran the skill, re-read both boxes, and ran it a
second time when the fingers did not yet fit. "Fingers" is the clearance of
the open fingertips for a grasp closing along each of the target's two axes;
3 mm or more fits.

| layout | before: gap, fingers | call 1: push planned / hand travel, then real gap, fingers | call 2 | neighbour disturbed |
|---|---|---|---|---|
| neighbour on +x, touching | 0.1 mm; 32.1 / 1.9 mm | +y (90 deg), 40 / 37.3 mm: 15.0 mm; 6.2 / 15.0 mm, **fits** | - | 11.7 mm, 13.6 deg (dragged along) |
| neighbour on +x, 5 mm apart | 5.0 mm; 33.1 / -3.0 mm | 110 deg, 34 / 31.2 mm: 20.6 mm; 26.2 / -3.2 mm | 190 deg, 30 / 27.2 mm: 52.0 mm; 53.6 / 13.2 mm, **fits** | 8.9 mm, 9.8 deg (call 1) |
| neighbour on -y, touching, turned 25 deg | 0.4 mm; -8.3 / 32.2 mm | 170 deg, 34 / 31.2 mm: 20.5 mm; 43.3 / -10.0 mm | 100 deg, 28 / 25.3 mm: 29.9 mm; 29.7 / 20.3 mm, **fits** | 11.2 mm, 33 deg (call 2) |

All three ended with both grasp axes free after at most two calls of one
push each (five pushes in all). Every call routed `singulated`
by the hand's travel, but in two of the three first calls the real boxes did
not fit: the target turned (3.9 deg, then 37.9 deg in the yawed layout) and
the neighbour was dragged. Re-perceive and re-run, as the graph did.

- The hand travelled 2.8-3.7 mm short of every planned push (servo
  tolerance), so the predicted clearance after a push (4.7-6.1 mm) is under
  the 7 mm aimed for; the 4 mm `PLAN_SLACK` absorbs it.
- Straight away from a touching or 5 mm-gap neighbour (180 deg) was never
  usable: the hand cannot come down in that gap. The skill pushed sideways
  or at a slant instead.
- The neighbour moved 9-12 mm in three of the five pushes. In the first
  layout it was dragged by the sliding target. In the yawed layout's second
  push the target moved away from it, so the hand (or the wrist) probably
  clipped it: `HAND_RADIUS = 10 mm` is likely too small for the closed
  linear_4310 with its fingers across the push.
- With `need="any"`, all three layouts already had a free grasp axis (a
  grasp closing along the contact line) and nothing would have been pushed.

## Provenance

The push primitive is adapted from visual-pushing-grasping's `Robot.push`
(BSD-2-Clause, Copyright (c) 2018 Andy Zeng): closed hand, hover a margin
above the push point, straight down, a bounded straight push kept inside the
workspace, straight up; its moves follow the hub's pushing-objects. VPG
chooses the push point and heading with a learned policy over a heightmap;
here they come from OBBs. The heading rule (push toward the nearest point of
the most free space) is reimplemented from Kiatos et al. 2021, "Improved
Reinforcement Learning Pushing Policies via Heuristic Rules"
(arXiv 2111.11156); its code (github.com/robot-clutter/improved_rl) has no
licence and was read, not copied. The insertion-point check, the
finger-clearance test and the short re-checked pushes are new.
