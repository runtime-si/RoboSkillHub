---
name: carrying-over-an-obstacle-with-a-leaning-hand
description: Carries the part the hand holds from one open container over the
  rims into another and releases it, with the hand leaning sideways so the arm
  holds it higher than it can pointing straight down - searches the leans, jaw
  turns and heights for one that reaches up clear of the rims where the hand
  stands (or pulled back from under something overhead) and, separately, for
  one that reaches a drop point inside the target, rises, crosses at that
  height, lowers, opens and retreats. A carry that ends short lets go only if
  the hand is inside the container's walls by its own pose reading; otherwise
  the hand goes back, still holding. Use when a part hanging from the fingers
  must clear a tall bin wall on a small arm whose straight-down reach is too
  low, after a grasp that holds, with the target container located in the
  hand's own frame; give an arm model to search reach and clearance locally.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [place, carry, transport, container, bin, obstacle, clearance, reach, lean]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: The lean angles and heights are one YAM station's. This script is a port
      through the connector tools and has only run against a fake connector in unit
      tests; the method it ports placed parts in seven real runs.
gap:
  requires: {connector: [robot.grasp_frame]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.get_observation
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
    - arm-clearance.reachable
  required_inputs:
    container_obb: OrientedBoundingBox
  # Optional inputs: clear_z, carry_z, forward_heading_deg, margin_m, drop_over_rim, drop_margin,
  # drop_inside, arrive_tolerance, empty_fraction (float), spread_index, max_ik_queries, arm_id
  # (int). Wired from a script node (gap has no type for them): over_obbs (list of
  # OrientedBoundingBox), leans (list of [left_deg, forward_deg]), pull_back_m (list of float),
  # avoid_points (list of Vec3), arm_model, obstacles, workspace_bounds (arm-clearance's).
  produces_outputs:
    release_position: Vec3
    drop_point: Vec3
    carried_z: float
    clear_z: float
    inside_m: float
    options: int
    ik_queries: int
    clearance_checked: bool
  exit_conditions:
    placed: The hand crossed at or over clear_z, read as still holding, and opened either within arrive_tolerance (4 cm) of a drop point inside the container or, short of it, at least drop_inside (3 cm) inside the container's walls by its own pose reading; then it went back up. What the hand did - not a sighting of the part in the container.
    returned: Every carry tried ended more than arrive_tolerance from its drop point and not over the container; the hand went back to where it rose and still holds the part. fell_short_at lists the drop points to avoid next time.
    blocked: On the way across the hand ended lower than clear_z (it is pushing or resting on something); it stopped there and still holds the part.
    rise_short: After the rise the hand was under clear_z; it did not cross and still holds the part where the rise ended.
    unreachable: No lean, jaw turn and height at or over clear_z was found to rise with, or none over any drop point (within max_ik_queries previews, or by the arm model); nothing moved and the hand still holds the part.
    lost: The jaws read empty before the start, after the rise, or after the carry across; where the part is, is not known. Nothing was released on purpose.
  canonical_scripts:
    - carry_and_release: scripts/carry_and_release.py
  examples:
    - title: Carry a held part from the parts bin into the located target box
      path: examples/canonical_subgraph.json
---

# carrying-over-an-obstacle-with-a-leaning-hand

A small arm reaches highest with its wrist bent. Pointing the hand straight
down over a tall bin, the arm this comes from could lift the tool centre to
26 cm; the part hanging from the fingers at 28 cm caught the 154 mm bin wall
and dragged the bin across the counter, twice. With the fingertips leaning 55
degrees to the side the same arm held the hand at 40 to 48 cm over the bin
and 40 cm over the target. The part hangs lower than the tool centre, so the
carry height is what decides whether it clears.

## How

1. **Holding?** Jaws at or under `empty_fraction` before anything moves:
   `lost`.
2. **The hands to try.** `robot.grasp_frame` gives the hand pointing down;
   the jaws' present turn about the vertical is read from the current pose.
   Each lean in `leans` (to the left, forward; relative to
   `forward_heading_deg`) is combined with the present jaw turn, that turn
   plus 180, and 0 / 90 / 45 / 135 degrees.
3. **Where it rises.** Where the hand is, then pulled back by each
   `pull_back_m` (against forward): the first spot, hand and height (from
   `carry_z` down to `clear_z` in 3 cm steps) the arm reaches.
4. **Where it lets go.** Drop points are a 4 x 4 grid inside the container,
   `drop_margin` from its walls, the middle first, skipping `avoid_points`.
   For each, the first hand (the rising one first, then the others, then a
   25 degree forward-only lean) and height the arm reaches. Up to three
   options are kept and rotated by `spread_index`. **The hand that rises and
   the hand that lets go need not be the same**; the hand turns from one to
   the other on the way across.
5. **Rise.** Pull back first if that was chosen (straight, the hand as it
   is), then to the rise pose. Under `clear_z` afterwards: `rise_short`.
   Jaws empty: `lost`.
6. **Across.** To the first option. Ended under `clear_z`: `blocked`. Within
   `arrive_tolerance`: arrived. Otherwise the point is added to
   `fell_short_at`, and if the hand is `drop_inside` inside the container's
   walls **by its own pose reading**, that is where it lets go; if not, the
   next option. None: back to the rise pose, `returned`.
7. **Release.** Jaws empty: `lost`. Otherwise straight down to
   `drop_over_rim` over the rim where the arm reaches, open, straight back
   up: `placed`.

Nothing is released anywhere but over the container. The source task
dropped the part back into the bin it came from on every failure; here the
graph decides (route `returned`, `rise_short`, `unreachable` and `blocked`
to a release over the source, or to an operator).

## The reach search, and why it can be local

The search in steps 3 and 4 is up to a few hundred poses. Two ways:

- **`arm_model` given**: every pose goes to `arm-clearance.reachable` (local
  IK from the current joints, and clearance against `obstacles` at
  `margin_m`), the straight carry is checked in 6 cm / 20 degree steps, and
  `robot.solve_ik` is never called. `clearance_checked` is true when
  obstacles were given.
- **No model**: `robot.solve_ik`, at most `max_ik_queries` (40) times; the
  search is cut short there and `detail` says so. No clearance is checked:
  `clearance_checked` is false, and that does **not** mean the path is clear.

The cap exists because on the source station this search, sent pose by pose
to the arm's motion server as plan requests, stalled the server's control
loop until the supervisor stopped both arms with a part in the hand. Raise
it only if your connector's `solve_ik` is local. Note also that a connector
whose `solve_ik` leaves the wrist roll free will say yes to leans it then
does not hold.

## Inputs worth getting right

- **`container_obb` in the hand's frame.** Use
  `locating-a-container-from-the-wrist-camera` first. "Over the container by
  the hand's own measure" is only as good as this box.
- **`over_obbs`**: the bin the part came from, so its rim sets `clear_z`.
- **`forward_heading_deg`**: the world heading pointing away from the arm
  across the work area (90 on the source station). "Left" is a quarter turn
  anticlockwise from it. If the layout mirrors, the useful lean may be to
  the other side: pass negative left angles as further `leans`.
- **`avoid_points`** from earlier `fell_short_at`, and **`spread_index`**
  the number already placed: four parts let go over one point stacked up.

## Constants, and what they were measured on

One station: an i2rt YAM arm (linear_4310 hand) over a kitchen counter with
a cabinet above it, a 154 mm tall parts bin and a 66 mm tall target box,
5-9 cm printed parts; 2026-11-13 to -15.

- `LEANS = (55, 15), (40, 15)` degrees, `LEAN_DROP = 25`: found for that arm
  and layout. With the two boxes swapped, no single lean both cleared the
  cabinet over the bin and reached the box, which is why rise and release
  are searched separately.
- `CLEAR_OVER_RIM = 11 cm`, `CARRY_SPAN = 13.5 cm` (carries at 0.40 m over
  the counter, and 0.31-0.34 m where the cabinet was too near):
  that station's.
- `DROP_OVER_RIM = 12 cm`, `DROP_MARGIN = 5 cm`: that box.
- `DROP_INSIDE = 3 cm`, `ARRIVE_TOLERANCE = 4 cm`: added after a run in
  which four carries ended 43-46 mm short of a drop point at the edge of
  reach, over the box, and the parts were not counted.
- `EMPTY_FRACTION = 0.06`: that hand; a part pinched by an edge read 0.07.
- `MAX_IK_QUERIES = 40`: a guard, not measured.

## Evidence

**The method**, on the real station: the first real run tried to carry with
the hand pointing down, ended low and pushed the bin across the counter; the
next carried at 40 cm leaning left and placed a part, the bin moving 1.4 cm
in the whole run. On 2026-11-15, seven runs placed parts this way and four
of them put all four parts in the box, all in one layout. It is not gentle:
in the first of those, 2 of 6 held parts were lost while the hand leant and
rose, and one more was let go because the hand reached only 27 cm. With the
boxes swapped, held parts were given up in two runs because no pose cleared
both the cabinet and the rims at the margins then asked, and one part was
lost on the way across.

**This script** is a port of that method onto the connector tools, and it
differs: leans are relative to a heading input instead of fixed world axes;
"pull back" replaces the station's fixed line out from under its cabinet;
the detour round the front when the straight carry is refused is not ported
(the next drop option is tried instead); and failures keep hold of the part
instead of dropping it back. It has run only in this repository's unit
tests against a fake connector. It has not moved a robot or a simulator.
Maturity is `community`.

## Required end states

- `placed`: released over the container, hand back up.
- `returned`, `rise_short`, `blocked`, `unreachable`: still holding.
- `lost`: the hand is empty and did not open on purpose.
