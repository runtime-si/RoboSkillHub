---
name: placing-into-container-obb
description: Puts the object the hand holds into an open bin, box or tray the
  camera sees - lifts it straight up, segments the named container with sam3,
  fits its box and its opening (rim height, opening centre, inner extents,
  floor) from depth, picks a drop point inside the opening where the held
  object's footprint keeps a margin from the walls, comes over it above the rim,
  straight down until the object is just over the container's floor, opens and
  retreats straight up. Routes no_room when the object fits nowhere and keeps
  it when no drop point can be reached. Use when a held part must be dropped
  into a fixed open container on the table (sorting into a bin, filling a box),
  with the held object's box measured after the grasp; follow with
  verifying-placement on the aperture_center and rim_z it outputs.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [place, container, bin, box, tray, drop, obb]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/place_obb/robotwin/policy.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.solve_ik
    - robot.open_gripper
    - sam3.segment_text
    - geometry.mask_to_world_points
    - geometry.filter_noise
    - geometry.compute_obb
  required_inputs:
    container_description: string
    held_obb: OrientedBoundingBox
  produces_outputs:
    container_obb: OrientedBoundingBox
    aperture_center: Vec3
    rim_z: float
    floor_z: float
    opening_obb: OrientedBoundingBox
    opening_source: string
    release_position: Vec3
    candidate: string
  exit_conditions:
    placed: The hand reached a release pose inside the container's inner opening (the held object's footprint keeping footprint_margin from the walls, its bottom release_height over the floor seen from depth), opened (its jaws read open) and retreated. What the hand did, not a sighting of the object in the container - run verifying-placement on aperture_center and rim_z.
    no_room: The held object's footprint plus footprint_margin on each side is wider than the inner opening along one of its axes; the hand lifted and still holds the object.
    unreachable: Every drop point failed - out of reach in the IK preview, the move over it refused or short, or the straight descent stopped above the release (on the rim, a wall or the contents; the hand went back up); still holding.
    container_not_found: No RGB-D camera, no sam3 mask of container_description at score_min (or under container_pixel), a mask over 40% of the frame, or fewer than 30 depth points on it; the hand lifted and still holds the object.
    not_holding: The jaws read at the stop before anything moved; nothing moved.
    not_released: At the release pose the jaws still read closed after open_gripper; the object may still be in the hand, inside the container.
  canonical_scripts:
    - place_into_container: scripts/place_into_container.py
  examples:
    - title: Drop the held object into a perceived open box
      path: examples/canonical_subgraph.json
---

# placing-into-container-obb

Drop what the hand holds into an open container it has to find first.

## The motion

1. Read the jaws (`robot.get_gripper`): at the stop they hold nothing
   (`not_holding`, nothing moves). An unreadable hand is taken as holding.
2. Lift straight up `lift_clearance` (0.10 m), keeping the rotation. A lift
   that stops short is noted and the place goes on (as in RoboRSI).
3. **Perceive.** `robot.get_observation`, the RGB-D camera `camera_name`
   (any with depth when empty), `sam3.segment_text` on
   `container_description`: the best mask at `score_min` or, with
   `container_pixel` (u, v), the mask under that pixel. A mask over 40% of the
   frame is the table. `geometry.mask_to_world_points`, then
   `geometry.filter_noise` (DBSCAN 5 mm, 10 samples) and
   `geometry.compute_obb` (upright box).
4. **Opening.** The rim is the cloud's 98th-percentile height; the points
   within 20 mm under it are the rim band. Along each of the box's two
   horizontal axes the band's 10th-90th percentile span gives the opening's
   centre (its midpoint) and the rim's half extent (half of it, never more
   than the box's), less `wall_thickness` for the inner opening
   (`opening_obb`). The floor is the 10th-percentile height of
   the points inside the inner opening (5 mm in from it): the top of anything
   already in the container. Too few rim points: the box's top and sides
   stand in; too few floor points: the box's bottom plus a wall.
5. **Room.** The held object's footprint is its box (`held_obb`) projected on
   the opening's axes. If footprint + `footprint_margin` on each side is wider
   than the inner opening along either axis: `no_room`.
6. **Drop points.** RoboRSI's: the opening centre, then half-way to each
   inner edge (-u, +u, -v, +v), each pulled in until the footprint keeps its
   margin. For each: the hand target carries the object's measured xy offset
   in the hand; preview the approach (object bottom `approach_clearance` over
   the rim) and the release (object bottom `release_height` over the floor,
   tool centre never under `min_tool_height` over it) with `robot.solve_ik`;
   go over the point, then straight down. A descent that stops more than
   `reach_tolerance` short (the rim, a wall, the contents) goes back up and
   the next point is tried. No release anywhere else (`unreachable`).
7. **Release.** `robot.open_gripper` (the jaws must read open, else
   `not_released`), then straight up `retreat_height`.

## Inputs

- `container_description`: a concrete noun phrase for sam3 ("open grey box").
- `held_obb`: the held object's box (world, half extents), measured after
  the grasp with the hand where it is now (e.g. registering-held-objects).
  Its offset from the tool centre is assumed to stay fixed.
- `camera_name` (""), `container_pixel` (none): which camera and which
  instance.
- `lift_clearance` (0.10 m), `release_height` (0.01 m), `approach_clearance`
  (0.03 m), `footprint_margin` (0.008 m), `wall_thickness` (0.003 m),
  `min_tool_height` (0.015 m), `retreat_height` (0.08 m), `reach_tolerance`
  (0.015 m), `score_min` (0.25), `empty_fraction` (0.04): see the constants.
- `check_reach` (true) turns the IK preview off; `arm_id` (0) is passed only
  when nonzero.

## Required end states

- `placed`: the hand is open, `retreat_height` over the release point.
  `release_position` is where the object's bottom centre was when the hand
  opened (from the hand, assuming the object did not move in the jaws).
  This is what the hand did. To know the object is in the container, run
  **verifying-placement** next with a fresh observation,
  `aperture_center` and `rim_z` from this skill (its `xy_tolerance_m` about
  the opening's half extent): RoboRSI's own depth containment check is not
  re-implemented here.
- `no_room`, `unreachable`, `container_not_found`: still holding, lifted, or
  (`unreachable`) back over the last drop point tried.
- `not_holding`: nothing moved.
- `not_released`: the hand is at the release pose with jaws reading closed.

## The constants, and why

- **`LIFT_CLEARANCE = 0.10` m**: RoboRSI lifts 0.20 m. In the YAM station
  twin that lift, from 0.10 m over the table, stopped 48 mm short at the edge
  of reach and the cube turned 15 deg in the jaws.
- **`RIM_PERCENTILE = 98`**: RoboRSI's `container_opening` takes the 90th.
  The twin's box has 2.3 mm walls; under 10% of its points were on the rim
  and the 90th percentile put the rim on the floor (perceived 0.754 m, true
  0.791 m). The 98th is also `geometry.compute_obb`'s top.
- **`RIM_BAND = 0.02` m, `HALF_PERCENTILE = 90`, `MIN_RIM_POINTS = 6`,
  `MIN_POINTS = 30`, `WHOLE_SCENE_FRACTION = 0.40`, DBSCAN 5 mm / 10**:
  RoboRSI's values.
- **`SCORE_MIN = 0.25`**: under sam3's usual 0.3 because the twin's "open grey
  box" scored 0.39 against 0.34 for the table; the mask-size reject catches
  the table.
- **`WALL_THICKNESS = 0.003` m**: the twin's box has 2.3 mm walls. Give the
  real value for thicker bins.
- **`FOOTPRINT_MARGIN = 0.008` m**: about one calibrated YAM station camera
  error (RMS 5.4 mm) plus a few mm of sway in the jaws.
- **`RELEASE_HEIGHT = 0.01` m**: RoboRSI releases 50 mm under the rim, which
  is under the floor of a 40 mm box. Measured from the floor instead: low
  enough that a cube does not bounce out, high enough that a floor read 5 mm
  high does not press it down.
- **`APPROACH_CLEARANCE = 0.03` m**: the object's bottom over the rim while
  coming over a drop point.
- **`MIN_TOOL_HEIGHT = 0.015` m**: a real YAM station refuses a tool centre
  under 14 mm over its table; closed fingertips reach 10 mm under it.
- **`RETREAT_HEIGHT = 0.08` m**: RoboRSI's.
- **`REACH_TOLERANCE = 0.015` m**: no tighter than a real station's 12 mm
  per move.
- **`EMPTY_FRACTION = 0.04`, `OPEN_FRACTION = 0.5`**: as in
  handing-over-between-arms.

## Not the same as

- **placing-with-contact-descent**: comes down until the arm stalls on a
  support whose height and place point are given. Here the container is found
  by the camera, the drop point is chosen inside its opening for the object's
  footprint, the hand stops at a height over the seen floor, and a stop on
  the way down (the rim) is refused rather than released on.
- **placing-into-container-held-by-other-arm**: the container is in the
  other hand and both boxes come from registration; tilts and a keep-out from
  the holder's fingers. Here the container stands on the table and is
  segmented and fitted from depth by this skill.
- **grasping-container-rims**: picks a container up by its wall (it shares
  RoboRSI's `container_opening` idea); this one drops something into it.
- **verifying-placement**: the check to run after this skill.

## Simulation check

RoboSimStudio YAM station twin (left arm, 30 mm cube, open 100 x 100 x 40 mm
box with 2.3 mm walls, overhead 640 x 480 RGB-D camera "top", sam3 and the
hub's geometry bundle; simulator truth only to grasp and to measure). A
scripted top-down grasp lifted the cube 90 mm (its box after the lift stood in
for `held_obb`); the skill ran with `container_description: "open grey box"`,
then verifying-placement on a fresh frame with this skill's `aperture_center`
and `rim_z` (`xy_tolerance_m` 0.05), then the cube was measured.

| run | route | perceived vs true | cube after | verifying-placement |
|---|---|---|---|---|
| RoboRSI's 0.20 m lift, 90th-percentile rim | unreachable (IK preview, all 5 points) | rim 0.754 vs 0.791 m (on the floor); lift stopped 48 mm short | still held, turned 15 deg | - |
| 0.10 m lift, 98th-percentile rim, band mean as centre | placed (centre point) | rim 0.792 vs 0.791 m, floor 0.754 m (true 0.754), opening centre 32 mm off, inner halves 44 / 74 mm vs 47.7 | in the box, on the floor, upright (0.0 deg), 23 mm from the centre; 4 mm from the skill's `release_position` | verified ("disappeared": sam3 did not find the cube in the box) |

The second run's off-centre opening (a tilted camera sees one wall's inner
face densely in the rim band) is why the opening is now the band's
10th-90th percentile span, capped by the box; that version's run is in the PR.
sam3 scored the box 0.56. verifying-placement passed on "the object
disappeared", the weakest of its two outcomes.

## Provenance

Adapted from RoboRSI's `place_obb` (https://github.com/nssmd/RoboRSI,
`roborsi/embodied/skills/base/place_obb/robotwin/policy.py`, with
`container_opening`, `container_drop_target` and `object_mask` from
`roborsi/embodied/skills/base/_lib/robotwin/_perception.py`), Apache-2.0.
Theirs: the holding precondition, the lift before transport, the container's
mask (whole-scene reject, instance by pixel), masked cloud, DBSCAN and box,
the opening from the cloud's top band with the box as fallback, the ordered
drop points (centre, then half-way to each edge), release only when the hand
reached the release pose, the straight retreat. Changed: sam3 and geometry
tools on a robot observation instead of Grounded-SAM on the simulator's
camera; a 0.10 m lift and the rim at the 98th percentile (measured above);
half extents along the box's axes and an inner opening; release height over
the seen floor; drop points kept inside the opening for the held object's
footprint (`no_room`); the hand keeps its own rotation (RoboRSI: a fixed
top-down quaternion); an approach over the rim then a straight descent, IK
previews, routed refusals, arrival read back from the arm; RoboRSI's
`_verify_deposit` left to verifying-placement.
