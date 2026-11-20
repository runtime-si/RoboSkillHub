---
name: setting-down-a-held-part-by-measured-gap
description: Sets a part held upright in the jaws (a domino on its end, a
  block on edge) down on a support by measuring, in each wrist RGB-D frame,
  how far the part's lower end still is from the support - the part's upper
  end and the support round its foot are read in the same depth frame, the
  lower end being the part's known height under the upper - lowering in
  shrinking steps and letting go a millimetre up, or when the hand comes down
  and the gap does not (the part stands and slides in the pads). A part that
  is not seen, or a hand that cannot go lower, keeps the part and says so.
  Use when a part taken on its side must be stood up or set down gently and
  neither the grasp's geometry nor the arm's height is good to a few
  millimetres, and a stall cannot be felt because the part slips in the pads
  before the arm stops; needs the place pose, the part's held height, and the
  wrist camera seeing the part's upper end past the fingers.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [placing, wrist-camera, eye-in-hand, depth, cpu, dominoes, closed-loop]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, dominoes, stacking]
    provenance:
      authors: [runtime-si]
      code: yam_dominos skills/skills.py (_set_down, _measure_down) and skills/perception.py (held_tile), 2026-11
    notes: Draft from yam_dominos. Measured there on the real left YAM; this script has not been run through a hub connector.
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
  required_inputs:
    place_pose: Se3Pose
    part_tall: float
  produces_outputs:
    release_pose: Se3Pose
    gap: float
    lowered: float
    detail: string
  exit_conditions:
    set_down: The jaws are open and the part was let go with its lower end read within drop + 1.5 mm (default 2.5 mm) of the support, or after a step in which the hand came down more than 1.5 mm and the part's end less than a third of that (it stood on the support); the hand is where it let go.
    cannot_lower: The part is still held. Sent down twice, neither the hand nor the gap changed by more than 0.6 mm with more than 4 mm to go (a workspace floor), or eleven steps did not bring it down; gap is the last reading.
    not_seen: The part is still held. From above the place, or after a step, the held part's upper end and enough of the support round its foot were not made out in the wrist frame; nothing was lowered by the model.
    unreachable: The pose above the place failed the IK preview or the move; the part is still held.
  canonical_scripts:
    - set_down_by_gap: scripts/set_down_by_gap.py
  examples:
    - title: Stand a held tile on its end at a place pose
      path: examples/canonical_subgraph.json
---

# setting-down-a-held-part-by-measured-gap

A part taken by its edges sits where the pads took it, not where the grasp's
geometry says, and may slide in them. Lower the hand by the model and the
part is driven into the support, or let go from too high and falls over. This
skill measures the part itself: how far its lower end is over the support,
both ends of that measurement in one depth frame, so neither the camera's
calibration nor the arm's height comes into it.

## How it composes

- **Before it**: the part held, carried near its place; `place_pose` is the
  tool pose at which the model has the part's lower end on the support (only
  its x, y and rotation are relied on).
- **After it**: `set_down` leaves the jaws open round the part and the hand
  where it let go. Draw the hand back along its own approach (for a hand
  leaning over the part, straight up may catch it).
- **Not `placing-with-contact-descent`**: that comes down until the arm
  stalls on contact. A thin part held by its edges slides in the pads long
  before the arm stalls: in yam_dominos every step "reached" while tiles were
  pushed 6 and 12 mm into the counter and out of the grasp.
- **Not `measuring-a-held-piece-from-the-wrist-camera`**: that fits a flat
  part's outline to find its x, y and yaw in the grip; this reads one number,
  the height left to go, repeatedly while lowering.

## What it does

1. Moves to `place_pose` lifted by `approach` (9 cm; IK preview first).
2. Reads the gap (`wrist_depth.held_gap`): the support as a plane fitted in
   the frame; the held part as what stands over half its height above that
   plane within 4 cm of where its upper end is expected (16 mm of where it was
   last seen, after the first sighting), light pixels only when `light_part`;
   its top 4 mm are the upper end; the support's own level is taken within
   9 cm of the part. Gap = upper end's height - `part_tall`.
3. Lowers to 1.5 cm and reads again (twice at most), then creeps: 0.6 of the
   gap above 8 mm, 3 mm below, never past `drop`.
4. Lets go when the reading is within `drop` + 1.5 mm, or when the hand came
   down and the gap did not shrink by a third of that.
5. Keeps the part and routes when it is not seen, or when the hand does not
   move and the gap does not change twice running.

## Inputs

- `place_pose`, `part_tall` (required).
- `held_top_in_tool`: where the grasp has the part's upper end, tool frame
  (default the tool centre); `light_part` (true: keep only light pixels,
  CIELAB L over 60; false for dark parts, then black fingers in the search
  ball may be taken for the part); `support_z` (0.0; the top of what it is
  set on, for a part set on other parts); `approach` (0.09 m), `drop`
  (0.001 m), `camera`, `arm_id`, `check_reach` (true).

## Required end states

As in the frontmatter. `gap` is the last reading (NaN when there was none);
`lowered` how far the hand came down from above the place.

## The constants, and why

Documented where defined in `scripts/set_down_by_gap.py`. On another robot
look first at `CREEP` and `CONTACT_HAND` against the station's smallest
reliable move: on the YAM station a move within 3 mm of its pose counted as
reached, so a 3 mm step sometimes did not move the arm at all, and the stall
rule exists for that as much as for a floor.

## Measured

yam_dominos, real left YAM, tiles 49 mm tall held leaning 15-25 deg:

- Before this (lowering by the grasp's geometry): tiles pushed 6 and 12 mm
  into the counter and out of the grasp; later, tiles let go from too high
  fell.
- Run `graph-20261120-103306`: 11 tiles ended standing in a line. Ten
  logged set-downs let go at readings of -0.1 to 2.3 mm; one lost sight of
  the tile after the first lowering (the task's version let go there, and a
  tile of that run fell and was replaced). Typical sequence: 81-96 mm from
  above, 12-16 mm after the first lowering, then 2-3 creeping steps.
- In the simulator a hand stopped by the server's floor 6.5 mm up let three
  tiles go and all fell; with the stall rule the part is kept instead.

Not measured: this script through a hub connector; dark parts
(`light_part=false`); a part set on other parts (`support_z` above 0) with
this script (the task's own version did it for a tile laid across two); the
reading's accuracy against a ground truth (only the outcomes above).

## Provenance

From yam_dominos' `_set_down`, `_measure_down` and `held_tile`: robot access
through connector tools, the model fallback removed (a part never seen is
routed, not lowered blind), routed exits.
