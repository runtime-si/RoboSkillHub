---
name: placing-into-container-held-by-other-arm
description: Puts the object one arm holds into an open container the other arm
  holds - the holder carries the container to a presentation point both arms
  reach without turning it (so it stays upright), the carrier goes above the
  container's opening, away from the holder's fingers (tilts that lean its
  wrist away from the holder first, then its own rotation, then RoboRSI's
  other tilts, each previewed with IK), comes straight down to a release height over the rim,
  opens, backs up and closes once to read that the object left its jaws; the
  holder's jaws are never touched. Use when a bimanual station must drop a held
  part into a cup, bowl or box that the other arm is holding (sort into a held
  bin, fill a container before carrying it away), with both held things' boxes
  from a perception skill and both arms addressable by arm_id.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, place, container, drop]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [pick-place, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/place_held_in_held_container/robotwin/policy.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
    - robot.close_gripper
    - robot.solve_ik
  required_inputs:
    container_obb: OrientedBoundingBox
    object_obb: OrientedBoundingBox
  produces_outputs:
    chosen_rotation: str
    container_tilt_deg: float
    container_position: Vec3
    release_position: Vec3
    hand_gap: float
    container_pose: Se3Pose
  exit_conditions:
    placed: The carrier released the object over the container's opening (its centre inside the opening less rim_margin, measured against where the holder's pose now puts the container), its jaws then closed to the stop (empty), and the holder's jaws still read a hold; the holder is at the presentation pose and can carry on. What the hand did, not a sighting of the object in the container.
    missed: The carrier held nothing at the start, its jaws did not open, or they closed on something after the release (the object stuck to them); or the holder's jaws ran to the stop (the container fell) or the holder ended where the release was outside the opening.
    unreachable: Nothing was released. The two arm ids are equal or the connector reports no separate second arm, the holder holds nothing, the container leans past max_tilt_deg (before or after the presentation), the holder could not reach the presentation pose, no carrier rotation fits the object in the opening, keeps its edge holder_keepout from the holder's tool centre and reaches the approach, or the straight descent stopped short (the carrier went back up still holding the object); also an unknown name in rotations.
  canonical_scripts:
    - place_in_held_container: scripts/place_in_held_container.py
  examples:
    - title: Drop arm 1's object into the container arm 0 holds
      path: examples/canonical_subgraph.json
---

# placing-into-container-held-by-other-arm

The holder (arm A, `container_arm`) presents an open container; the carrier
(arm B, `object_arm`) drops what it holds into it. One arm moves at a time,
every move names its arm (`arm_id`, sent only when nonzero).

## The motion

1. Refuse if the arm ids are equal, or if B's pose reads as A's (within 1 mm)
   or as no pose at all: a connector that clamps or ignores `arm_id` would
   move A when told to move B. Read both jaw openings: A at the stop holds no
   container (`unreachable`), B at the stop holds no object (`missed`).
2. **Upright.** The container's opening faces along its box axis nearest
   vertical; leaning more than `max_tilt_deg` (15) is refused.
3. **Presentation.** The container rides at a fixed offset from A's tool
   centre (from `container_obb` and A's pose, read together), so A goes, with
   its own rotation, to where that offset puts the container's centre on
   `presentation_position` (default: midway between the two tool centres, at
   the container's current height). A straight-line move first (the container
   stays upright on the way), a free move if that is refused. A's rotation is
   read back and the container's centre, axes and tilt follow it, so a wrist
   that came back turned is accounted for; past `max_tilt_deg` is refused.
4. **Drop point.** On the opening, away from A's hand (A's tool centre seen
   from the container's centre across the opening) as far as the object's
   footprint still keeps `rim_margin` from the container's outer edge;
   the opening's centre when A holds it from the middle.
5. **Carrier rotation.** RoboRSI's candidates: B's own rotation (the object
   keeps its pose in the hand), 30 deg about world +-x and +-y, 45 deg about
   +-x, each turning the object's measured offset from the hand with it. The
   tilts that lean B's wrist away from A (`LEAN_AWAY`) go first, 30 before 45,
   then B's own rotation, then the rest in RoboRSI's order; `rotations` names
   the order instead. A candidate is skipped when the turned object does not
   fit the opening, when its near edge would be under `holder_keepout`
   (horizontal) from A's tool centre, when `robot.solve_ik` refuses the
   approach or the release (or is not `within_servo_tolerance`), or when the
   approach move is refused.
6. **Approach, then straight down.** B goes to where the object's bottom is
   `approach_height` over the rim, reads its rotation back (a free wrist roll
   turns the object about the hand; the release is re-aimed from it), and
   comes straight down until the object's bottom is `release_height` over the
   rim. A refused descent sends B back up, still holding (`unreachable`).
7. **Release.** B opens (its jaws must read open), backs `retreat_distance`
   straight up, closes once and opens again: jaws that close to the stop say
   the object left them. A is never opened or closed.
8. **Check.** A's jaws must still read a hold, and the release point must lie
   inside the opening (less `rim_margin`) of the container where A's pose now
   puts it.

## Inputs

- `container_obb`, `object_obb`: the two held things' boxes, world frame
  (half-extents), each measured while its arm holds it (e.g.
  registering-held-objects); neither arm may move between that measurement
  and this skill.
- `presentation_position` (none): where the container's centre should be
  presented, world. Pick a point both arms reach with room above it.
- `container_arm` (0), `object_arm` (1): the connector's arm ids.
- `approach_height` (0.06 m), `release_height` (0.015 m), `rim_margin`
  (0.008 m), `holder_keepout` (0.035 m), `max_tilt_deg` (15),
  `retreat_distance` (0.10 m), `reach_tolerance` (0.02 m), `empty_fraction`
  (0.04): see the constants. `check_reach` (true) turns the IK preview off.
- `rotations` (none): carrier rotation names to try, in order (`own`,
  `tilt_30_+x`, `tilt_30_-x`, `tilt_30_+y`, `tilt_30_-y`, `tilt_45_+x`,
  `tilt_45_-x`).

## Required end states

- `placed`: B is open, `retreat_distance` over the release, empty by its jaws.
  A holds the container at the presentation pose; `container_pose` is A's
  pose, for the next skill that carries it. The object's landing is inferred
  from where the hand let go, not seen: re-perceive (verifying-placement,
  detecting-subtask-success) before relying on it.
- `missed`: see `detail`. The object may be in B's hand, on the table, or in
  the container while the container fell.
- `unreachable`: nothing was released; B still holds the object (over the
  container if the descent stopped, else where it was or at its approach), A
  holds the container (at the presentation pose if it got there).

## The constants, and why

- **`TILTS`**: RoboRSI's `_QUAT_CANDIDATES`, top-down then 30 deg about +-x
  and +-y then 45 deg about +-x, applied to the carrier's own rotation.
- **`LEAN_AWAY = 0.25`**: the order. RoboRSI tries top-down first and tilts
  "when the holding arm's elbow conflicts with the container arm". In the
  twin, with the carrier's own (top-down) rotation the two arms' forearms came
  within 0.2-0.4 mm at all six presentation points (they would have collided
  on real arms); leaning the carrier's wrist 30 deg away from the holder
  first gave 0.9-10.9 mm (table below).
- **`APPROACH_HEIGHT = 0.06` m**: RoboRSI's `drop_height_m`, there the drop
  height over the container hand; here the approach before the descent.
- **`RELEASE_HEIGHT = 0.015` m**: the object's bottom over the rim at release.
  Low enough that a cube does not bounce out of a 40 mm deep container, high
  enough that the carrier's fingertips (10 mm below its tool centre on a YAM)
  stay over the rim. Measured in the sim check below.
- **`RIM_MARGIN = 0.008` m**: a 6 mm wall plus 2 mm. A container with thicker
  walls needs a larger value.
- **`HOLDER_KEEPOUT = 0.035` m**: the object's near edge from A's tool
  centre, horizontally. A YAM linear_4310 blade is 30 mm thick along its
  slide, so a hand pinching a 6 mm wall puts its inner blade 33 mm into the
  opening. In the twin a 30 mm cube released over a 60 x 60 x 40 mm cup held
  by one wall (its edge 19 mm from A's tool centre) landed on that blade and
  bounced out onto the table; this check now refuses that cup. It stands in
  for RoboRSI's swept-sphere check between the arms (cuRobo spheres, not
  available through robot.* tools) and does not see an elbow: see LEAN_AWAY
  and the measured clearance.
- **`MAX_TILT_DEG = 15`**: a judgement, not measured; past it a drop lands
  on the low wall.
- **`RETREAT_DISTANCE = 0.10` m**, **`EMPTY_FRACTION = 0.04`**,
  **`REACH_TOLERANCE = 0.02` m**: as in handing-over-between-arms.

## Connectors and arm_id

- RoboSimStudio: every robot verb used here takes `arm_id` (0 = work arm,
  1 = support arm); an id past the last arm is clamped, which is why step 1
  compares the two arms' poses.
- The real YAM station bridge (2026-11-06) drives the left arm whatever
  `arm_id` says and reads no second arm's pose: the skill refuses there
  (`unreachable`, "no separate arm"). It needs a dual-arm bridge.

## Simulation check

RoboSimStudio YAM station twin, both YAM arms (bases 0.61 m apart; arm 0 =
left = holder, arm 1 = right = carrier). The twin has no graspable container,
so the private scene adds one: an open tray 70 x 100 x 40 mm, 6 mm walls,
60 g, one free body (a generated mesh with RoboSimStudio's `"shell"`
collider: floor and four wall boxes), plus the station's 30 mm cube moved to
the right arm's side. The harness picks the tray with the left, top-down on
its +y wall (jaws across the wall, the wrist roll finished on the last
joint), the cube with the right (top-down, tool centre 8 mm above its
centre), reads both boxes with `sim.get_object_obb` (the stand-in for
perception), runs this skill with default parameters and a presentation
point, waits 150 steps and measures. Arm clearance is the minimum over the
skill of the distance between the two arms' collision meshes, polled every
20 ms (the twin filters the arms against each other, so they can pass
through each other; an overlap reads as about 0). Presentation points are
the tray centre relative to the bench top's centre.

| point (x, y, z) m | route | rotation | cube inside the tray after | tray tilt: skill (from the hand) / measured after the wait | min arm clearance |
|---|---|---|---|---|---|
| (-0.12, 0.02, 0.09) | placed | tilt_30_+x | yes, 21 mm past the centre away from the holder, on the floor | 0.5 / 23.4 deg | 5.5 mm (forearms) |
| (-0.10, 0.04, 0.08) | placed | tilt_30_+x | yes, 23 mm, on the floor | 0.5 / 4.9 deg | 10.9 mm (forearms) |
| (-0.14, 0.00, 0.10) | missed: "the container fell" | tilt_30_+x | inside the tray, but the tray left the holder's jaws and lies on the table | 0.3 / - | 0.9 mm (forearms) |
| (-0.06, 0.02, 0.06) | placed | tilt_30_+x | yes, 22 mm, on the floor | 0.2 / 6.3 deg | 9.0 mm (forearm to right fingertip) |
| (-0.04, 0.04, 0.06) | placed | tilt_30_+x | yes, 21 mm, on the floor | 0.5 / 10.2 deg | 7.9 mm |
| (-0.08, 0.00, 0.07) | placed | tilt_30_+x | yes, 22 mm, on the floor | 0.4 / 7.4 deg | 7.8 mm (forearms) |

The same six points with the carrier's own (top-down) rotation forced first
(RoboRSI's order): six `placed`, the cube inside the tray each time, tray
tilt after 7-14 deg, but the arms' forearms (or forearm and wrist) within
0.2-0.4 mm at every point, which is why the default order now leans away.

What these numbers say and do not:

- The tray tilt the skill reports is read from the holder's wrist, and it
  stayed under 1 deg; the tray itself turned 5-23 deg in the left's pinch on
  its 6 mm wall over the run, the jaw reading opening from 0.075 to
  0.09-0.11, and in one run of six it slipped out (routed `missed` from the
  jaw reading). The twin's hold on a thin wall is weak; the skill cannot see
  a container turning in the hand without re-perceiving it. Hold containers
  by more than a thin wall, or re-measure (registering-held-objects) after
  the presentation.
- A 60 x 60 x 40 mm cup held by its wall failed in the twin (the cube landed
  on the holder's inner blade and bounced out); `HOLDER_KEEPOUT` now refuses
  it (`unreachable`). Use a container that leaves the object room beyond the
  holder's fingers.
- `sim.check_grasp` was not used; holds are judged from jaw openings and
  measured object positions.

## Provenance

Adapted from RoboRSI's `place_held_in_held_container`
(https://github.com/nssmd/RoboRSI, `roborsi/embodied/skills/base/place_held_in_held_container/robotwin/policy.py`),
Apache-2.0: repositioning the container arm under a pose the carrying arm
reaches and never opening it, the ordered list of carrier orientations taken
at the first feasible one, the drop height over the container and the
open-and-check of both grippers are theirs. Changed: world-frame boxes of the
two held things and the hands' own poses and jaw openings in place of the
simulator's EE poses and gripper values; the container keeps its rotation
(upright) on the way to the presentation and its tilt is read back; the drop
point is on the container's opening, away from the holder's hand, instead of
the container hand's xy; the object's offset in the hand is carried through
every tilt, and tilts that lean the carrier's wrist away from the holder
are tried first; cuRobo's trajectory sphere check is replaced by an IK preview
and a keep-out around the holder's hand; the carrier approaches above and comes straight down;
after the release it backs up and closes once to read its jaws empty;
refused moves are routed.
