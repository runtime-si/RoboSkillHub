---
name: regrasping-a-brick-to-turn-studs-up
description: Tips a loose LEGO or DUPLO brick a quarter turn toward studs-up
  with a hand that only serves top-down poses - closed fingertips push its
  upper part horizontally so it pivots about a bottom edge, over a fixed lip
  (a baseplate's edge) when the table is too slippery for a free push to tip
  it - one tip per call. A brick on its side needs one tip (pushed from its
  stud face), an upside-down one two (across its short side onto a long side
  first). Driven by classifying-brick-resting-face-from-depth's up_vector and
  centre, and judged by its next reading. Use when a brick in the bin lies on
  a side or on its studs and must be studs-up before a top-down pick.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, reorientation, tipping, push, top-down]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego, reorientation]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.go_to_pose
    - robot.close_gripper
  required_inputs:
    brick_position: Vec3
    up_vector: Vec3
  produces_outputs:
    push_direction: Vec3
    push_height: float
    travel: float
    tips_left: int
  exit_conditions:
    tipped: The four moves ran (above the start, down to the push height, the push, up and clear); whether the brick turned is the next face reading's to say. tips_left 1 means an upside-down brick that should now lie on a side.
    already_studs_up: up_vector points up; nothing moved.
    blocked: The fingers could not close or a move was refused; detail says which.
  canonical_scripts:
    - tip_brick: scripts/tip_brick.py
  examples:
    - title: Tip a side-lying brick studs-up
      path: examples/canonical_subgraph.json
---

# regrasping-a-brick-to-turn-studs-up

A hand that can only point straight down cannot roll a brick in its fingers.
It can push one over. This skill pushes the brick's upper part sideways with
the closed fingertips so it pivots about a bottom edge, a quarter turn at a
time, and leaves the judging to the face classifier.

## The tip

- **On a side** (up_vector horizontal, studs along u): start 6 mm off the stud
  face, at 70 % of the brick's height on that side (16 mm for a 2x4 LEGO on
  its long side), and push along -u. The top goes over the far bottom edge
  and the studs swing up.
- **Upside down** (up_vector -z): push across the short side (the brick's x
  from `yaw_deg`) at 70 % of its height on its studs. It lands on a long
  side, studs along the push; call again.
- **Over a lip** (`lip_distance`): push until the brick's far side is past a
  fixed lip, such as the baseplate's 3.2 mm edge, so its bottom catches there
  and its top rolls over.
- The tool centre never goes under 14 mm over the table (a real YAM station's
  floor); fingertips reach 10.2 mm below it. Every pose is top-down; the
  closing axis lies along the push.

## Checked

On the LEGO Isaac + BrickSim station (floating top-down hand), as `hub.flip_studs_up` in
`roboskillhub.tasks.lego_hub` between the face classifier (on Isaac-rendered depth, `LEGO_RENDER=1`) and the
pick; see the PR for the numbers. A free push slid side-lying bricks 22 mm per push without tipping
(BrickSim's table friction is under the ~0.5 a free tip needs for this brick), hence the lip.

## Limits

- Needs a lip in the push direction for a slippery table; the push direction is fixed by the brick's stud
  direction, so a brick whose studs point away from every lip cannot be tipped this way (route the flip
  elsewhere first).
- A brick tipped over the plate's edge lands on the plate's edge studs, not seated; the normal pick takes it
  from there.
- Not run on a real YAM.
