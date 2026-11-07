---
name: wiping-a-region
description: Wipes a flat region of a surface with back-and-forth strokes - the
  hand comes down at one end of each pass, strokes along the region's long axis
  at a contact height just over the surface, lifts, steps across and comes back
  the other way, then reports how much of the region the measured strokes swept.
  No force sensor - each descent stops on contact, so a surface higher than
  estimated is met rather than pressed into. Use when a spill, crumbs or a mark
  on a table must be wiped (with a held sponge or cloth, or the closed fingers),
  given the region's box from perception (a mask's points through
  geometry.filter_and_compute_obb), and the region is clear of objects.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [wipe, clean, surface, contact, coverage, nonprehensile]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [surface-contact]
    provenance:
      authors: [runtime-si]
      code: https://github.com/capgym/cap-x/blob/main/capx/envs/tasks/franka/franka_spill_wipe.py
      upstream_license: MIT
gap:
  requires: {connector: [robot.grasp_frame, robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
    - robot.close_gripper
    - robot.grasp_frame
  required_inputs:
    region_obb: OrientedBoundingBox
  produces_outputs:
    coverage: float
    passes_planned: int
    passes_done: int
    surface_z: float
    wipe_height: float
    min_height: float
    max_height: float
  exit_conditions:
    wiped: The measured strokes swept at least min_coverage (default 0.95) of the region at the wiping height and the hand lifted away; `coverage` says how much.
    partial: Some passes ran but the swept fraction is under min_coverage (passes out of reach, blocked, stopped part way) or the last lift failed; `detail` lists each skipped or stopped pass.
    unreachable: No pass ran - the first hover was refused with every rotation, or every pass was out of reach, blocked or refused; nothing was wiped.
  canonical_scripts:
    - wipe_region: scripts/wipe_region.py
  examples:
    - title: Wipe a region whose box a perception skill found
      path: examples/canonical_subgraph.json
---

# wiping-a-region

Cover a flat region with parallel strokes at a fixed height over the surface,
CaP-X's boustrophedon spill wipe with the height and the surface found by the
arm instead of assumed.

## The motion

1. Close the hand (`close_hand`, default true: the closed fingers wipe; a
   held sponge stays held).
2. Plan the passes in the region's own frame: strokes along its long axis,
   spaced across it by at most `min(step, tool_width)`, alternating direction,
   with a waypoint at most `step` apart along each. The tool's centre stays
   half its width inside every edge, so the swath ends at the region's edge.
3. Go to `approach_height` over the first pass's start, top-down with the
   closed fingers side by side across the strokes (`robot.grasp_frame`, both
   wrist solutions); if neither reaches, the hand's own rotation.
4. For each pass:
   - ask `robot.solve_ik` whether both ends can be held at the wiping height
     (`check_reach`) and skip the pass if not: a stall at the edge of reach
     looks like contact;
   - move straight over its start, and come down with
     `robot.move_cartesian_until_contact` toward the wiping height
     (`probe_surface`). A descent that stops up to 20 mm high met the surface
     there, and the pass is wiped at the height it stopped at; one that stops
     higher met an object (or stalled), and the pass is skipped. A connector
     without the contact move (it raised and the hand did not move) descends
     with `robot.go_to_pose_cartesian`;
   - stroke through the waypoints with `robot.go_to_pose_cartesian`, reading
     the tool centre back after every one; a refused or short waypoint ends the
     stroke there;
   - lift `approach_height` straight up (CaP-X lifts after every pass).
5. Coverage: a 2 mm raster of the region, a cell counted when a square tool
   of side `tool_width` swept along the measured stroke segments passes over
   it. Only stroke points within 6 mm of the pass's height count.

## Inputs

- `region_obb`: the region's box, world frame (half-extents). The axis nearest
  vertical is its normal; the other two span the region. A perceived spill's
  box from `geometry.filter_and_compute_obb` is upright.
- `surface_z` (default: the box's lowest corner): the surface's height under
  the region.
- `tool_width` (0.05 m): the width of the swath one stroke wipes. A closed
  linear_4310's fingers across the stroke cover about 30 mm.
- `step` (0.05 m), `approach_height` (0.10 m), `contact_height` (0.015 m),
  `min_contact_height` (0.015 m), `min_coverage` (0.95): see the constants.
- `close_hand`, `probe_surface`, `check_reach` (all true), `keep_rotation`
  (false: try the hand's own rotation before the top-down ones).

## Required end states

- `wiped`: `coverage` ≥ `min_coverage` and the hand stands `approach_height`
  over the last pass's end. `min_height` / `max_height` are the measured tool
  centre's lowest and highest over `surface_z` while stroking.
- `partial`: some passes ran; `detail` names each skipped or stopped pass and
  why. Wipe again (perhaps with a smaller region) or move the arm's base.
- `unreachable`: no pass ran; the hand is wherever the failed move left it.

Coverage is the hand's swath, not the surface's state: this skill does not
see the spill. Re-perceive (a mask's area, or `vlm.query_yes_no` "is the
table clean here?") and wipe again if something remains.

## The constants, and why

- **`STEP = 0.05` m, `APPROACH_HEIGHT = 0.10` m, the lift after every pass and
  the alternating stroke direction** are CaP-X's oracle's (its 5 cm waypoints
  keep the IK delta per move small).
- **Strokes along the long axis** (CaP-X: always along world x): fewer passes,
  fewer lifts.
- **Tool kept inside the region** (CaP-X's `np.arange(min, max + step, step)`
  wipes past the far edge by up to a step): a region beside an object must
  not sweep into it.
- **`TOOL_WIDTH = 0.05` m**: CaP-X's 5 cm stroke spacing assumes a sponge at
  least that wide.
- **`CONTACT_HEIGHT = MIN_CONTACT_HEIGHT = 0.015` m** (CaP-X wipes at z = 0
  with a sponge under its tool centre): a real YAM station refuses a tool
  centre under 14 mm over its calibrated table. With the bare closed fingers
  (they reach about 10 mm below the tool centre) that floor leaves them about
  5 mm over the table: the wipe skims rather than presses, and only a held
  tool reaching more than 15 mm below the tool centre touches. There is no
  force sensor to press with a known force; set `contact_height` to the held
  tool's reach minus the press-in you want, never below the floor.
- **`HEIGHT_TOLERANCE = 6 mm`**: above a servo's 4 mm ball and a real arm's
  2-3 mm sag; a descent stopping higher found the surface higher.
- **`MAX_RAISE = 20 mm`**: a surface that much higher than estimated is an
  object in the region, not the region.
- **`REACH_TOLERANCE = 20 mm`**: a real YAM station's 12 mm per-move bound plus
  margin.
- **`MIN_COVERAGE = 0.95`**: room for a corner or a waypoint ending short.

## Checked in simulation

RoboSimStudio, a YAM station twin, left arm, closed empty hand: a 10 x 8 cm
rectangle of bare table 25 mm beside a 30 mm cube, long side along x,
`tool_width` 0.03 m (3 passes of 2 strokes), the tool centre read back with
`robot.get_ee_pose` after every move. Heights are over the true table;
"measured coverage" is recomputed by the check from the recorded path on a
1 mm raster.

| surface estimate | exit | coverage (skill / measured) | descents stopped at | strokes at | cube moved |
|---|---|---|---|---|---|
| the region's bottom (true) | `wiped` | 0.98 / 0.97 | 16.6-16.9 mm | 15.05-15.14 mm | 0.0 mm, 0.0 deg |
| 15 mm too low | `wiped` | 0.97 / 0.97 | 10.5 mm (contact) | 10.57-10.69 mm | 0.0 mm, 0.0 deg |

With the true estimate the descent arrives at the 15 mm floor and the
fingertips skim about 5 mm over the table. With an estimate 15 mm low the
target is the table itself; the contact move stops where the fingertips meet
it (10.2 mm under the tool centre) and the passes are wiped from there, the
fingertips on the table. The tool centre came no closer than 40 mm to the
cube's face. Coverage under 1.0 is the servo ending 1-3 mm short of the
inset stroke ends.

## Provenance

Adapted from CaP-X's Franka spill-wipe oracle (MIT): a world-frame region box
in place of `get_object_pose`'s SAM3 extent, strokes along the box's long
axis kept inside it, a floored contact height over the box's bottom instead
of z = 0, each descent a contact probe (no force sensor: the arm's own stop),
an IK preview per pass, routed move refusals, and coverage measured from the
hand's read-back path. CaP-X's perception (SAM3 mask, point cloud, box) is
left to a perception skill.
