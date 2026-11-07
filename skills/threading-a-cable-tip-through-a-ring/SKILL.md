---
name: threading-a-cable-tip-through-a-ring
description: Threads the free tip of a held cable through a standing ring -
  the end turned about the vertical to point along the ring's axis, carried to
  a pre-insert pose in front of the mouth with the axis aimed between the tip
  and where the cable leaves the fingers (so a sagging end still passes),
  pushed through along the axis with an overshoot capped where the fingertips
  would reach the ring, and judged by the hand's travel past the ring's plane
  (optionally a camera point on the tip). Use when the hand already holds a
  cable a few centimetres behind its tip (a grasp skill's state, the tip
  measured or given) and the ring's centre, axis and inner radius are known
  (perception), e.g. the first step of routing a cable through a ring or loop.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, deformable, insertion, ring, threading]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/hanging_mug.py
      upstream_license: MIT
      paper: "DLOInsert / DLOPull, https://lmeee.github.io/DLOPull (no code licence; method reference only)"
gap:
  requires: {connector: [robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
    - robot.get_gripper
    - robot.describe_gripper
    - gemini-er.point
  required_inputs:
    ring_centre: Vec3
    ring_axis: Vec3
    ring_inner_radius: float
  produces_outputs:
    tip_past: float
    hand_travel: float
    push_planned: float
    tip_offset: float
    exit_offset: float
    crossing_offset: float
    sag_deg: float
    turned_deg: float
    tip_seen: bool
    seen_tip_past: float
    tip_position: Vec3
    cable_lost: bool
  exit_conditions:
    threaded: The push carried the tip, as held, at least min_past (10 mm) past the ring's plane with the hand within 6 mm of its planned travel (or already past when it stopped short); with a camera, the tip was seen past the plane and the line from the fingers to it crosses the plane inside the ring. Without a camera this is what the hand did, not what the cable did. The hand stays there, closed (opened and lifted 4 cm when release is true).
    jammed: The tip did not get through - the push stopped short with the tip, as held, under min_past past the plane, or the camera saw the tip short of the plane after the push (held back or bent on the rim; the hand backed off to the pre-insert pose, still holding) - or the jaws closed to their stop before or during the push (cable_lost true; the cable slid out of the hand, nothing more is moved).
    misaligned: Before moving, the held end cannot pass the opening (it sags so far that aimed between its tip and the fingers it stands outside the opening, or the fingertips reach the ring before the tip gets min_past through, or the opening leaves no room for the cable); or the camera saw the tip past the plane on a line crossing it outside the ring. `detail` says which.
    unreachable: The jaws are at their closed stop at the start (nothing held), the hand is more than 35 deg off straight down, the ring's axis is vertical, no wrist solution reaches the pre-insert pose or the end of the push (IK preview), or a turn, carrying move or the push was refused.
  canonical_scripts:
    - thread_tip_through_ring: scripts/thread_tip_through_ring.py
  examples:
    - title: Thread a measured held tip through a perceived ring
      path: examples/canonical_subgraph.json
---

# threading-a-cable-tip-through-a-ring

Push a held cable's tip through a ring that stands on the table, the way
RoboTwin hangs a mug on a rack: carry the held object's functional point (here
the tip) onto the target's axis a little in front of it, then go along the
axis past it.

## The motion

1. **The hold.** Where the tip is relative to the hand, from one of:
   `tip_position` (world, measured at the current pose by a camera or
   `registering-held-objects`; best), `tip_in_tool` (the tip in the tool
   frame), or `grasp_back` (how far behind the tip the hand took it) with
   `tip_direction` (the cable's way from the hand to the tip) and `sag_deg`
   (how far the free end hangs below level; 0 = straight out). Where the
   cable leaves the fingers is `exit_position`, else 10 mm ahead of the tool
   centre along the tip's way, at the held cable's height (the fingertips'
   reach less 4 mm). These two points are fixed in the hand from here on: the
   hand-to-object transform of RoboTwin's `get_place_pose`.
2. **The ring.** `ring_centre`, `ring_axis` and `ring_inner_radius` (the
   clear opening, wire excluded), e.g. from a plane and circle fit on the
   ring's points. The axis' sign is taken so the hand comes in from its own
   side (`keep_axis_sign` to keep the given one). The cable needs
   `ring_inner_radius - cable_radius - 4 mm` (the real station's
   `HOLE_MARGIN`) round the axis.
3. **Aim.** The push is a straight move along the axis, so the tip's and the
   fingers' offsets from the axis stay as they are aimed. A sagging end has
   the tip under the fingers. With the tip `t` past the plane at the end of
   the push, the stretch from the fingers to the tip crosses the plane a
   fraction `f` of the way from the tip; the point `f / 2` along it is put
   on the axis, which leaves the tip and that crossing equally far off it.
   Either beyond the opening: `misaligned`, before anything moves.
4. **Overshoot.** The tip aims `overshoot` (5 cm, RoboTwin's `dis=-0.05`)
   past the plane, capped where the fingertips' front comes `finger_clear`
   (5 mm) in front of the ring's wire. A cap that leaves the tip under
   `min_past` (10 mm) through is `misaligned` ("take the cable further behind
   its tip"). The IK preview (`robot.solve_ik`) checks the pre-insert pose and
   the end of the push.
5. **Align** (RoboTwin's `constrain="align"`, about the vertical only). The
   hand turns where it stands so the end points along the axis in plan. The
   turn is read back; an arm whose solver leaves the wrist roll free arrives
   unturned, and the rest is turned on the last joint (`robot.move_to_joints`).
   The plan is redone with the rotation the hand actually has.
6. **Carry.** A hand in front of the pre-insert pose (along the axis) goes
   level with it and then straight to it, the real station's way "in front of
   the ring, at its mouth's height"; from anywhere else it goes up, over the
   pre-insert pose (6 cm over it) and down. The pre-insert pose has the tip
   `pre_dis` (5 cm, RoboTwin's `pre_dis`) in front of the plane.
7. **Camera look** (`camera`, optional). `gemini-er.point` on that camera's
   picture finds the tip; its depth (the near points round the pixel) puts it
   in the world. More than 8 mm from where the hold says: the tip is
   re-registered, the aim redone and the hand moved, at most twice.
8. **Push** along the axis (`robot.go_to_pose_cartesian`). A routed refusal is
   `unreachable`; anything else (an arrival, a stop, a raised "did not reach")
   is judged from the hand read back: the tip, as held, is the tip's start
   along the axis plus the hand's travel. The jaw gap (`robot.get_gripper`
   against `robot.describe_gripper`'s `width_fit`) is read at the start,
   before the push and after it: jaws at their closed stop mean the cable
   slid out (`jammed`, `cable_lost`). Moves report arriving either way.
9. **Check.** With a camera, the tip is pointed at again: short of
   `min_past` past the plane is `jammed`; past it, the line from the fingers
   to the seen tip must cross the plane inside the ring (`misaligned`
   otherwise). Without one, the hand's travel decides. `jammed` backs the hand
   off to the pre-insert pose (still holding), ready for a search or another
   try. `release` opens the hand and lifts it 4 cm after `threaded`.

## Inputs

- `ring_centre`, `ring_axis` (Vec3), `ring_inner_radius` [m];
  `ring_wire_radius` (2 mm), `cable_radius` (1.75 mm).
- The hold: `tip_position` (+ `exit_position`), or `tip_in_tool`, or
  `grasp_back` + `tip_direction` (+ `sag_deg`).
- `pre_dis` (0.05), `overshoot` (0.05), `min_past` (0.01), `finger_clear`
  (0.005; negative lets the fingertips into the ring).
- `camera` ("" skips the looks), `cable_description` ("cable").
- `release` (false), `check_reach` (true), `reach_tolerance` (15 mm),
  `roll_limits` (the YAM's joint 6), `arm_id` (0; passed only when nonzero).

## Required end states

- `threaded`: the hand at the end of the push, closed on the cable. Without a
  camera, re-perceive before letting go: the cable may have bent on the rim
  and gone past outside the ring while the hand travelled.
- `jammed`: the hand at the pre-insert pose, holding. Re-perceive the tip
  (it may have bent) before another try. With `cable_lost`, the hand is
  empty where the loss was found: find the cable's end and grasp it again.
- `misaligned`: nothing moved, or (camera) the hand is at the end of the
  push with the tip past the plane outside the ring.
- `unreachable`: the hand is where the refused move left it.

## What changed from RoboTwin, and why

RoboTwin's `hanging_mug` knows the mug's functional point and the rack's from
the simulator and moves a rigid mug: `place_actor` puts the functional point
`pre_dis` before the target along its axis, then `dis` along it (negative:
5 cm through). Here the held thing is a soft cable end: its tip comes from a
measurement or the grasp, the ring from perception, and the end sags, so the
axis is aimed between the tip and the fingers instead of at the tip (the real
threading station's alignment did the same when the tip on the axis left the
cable at the fingers outside the ring, `i_align`'s "halved" shift). The
"align" constraint turns the end about the vertical only: the hand stays top-
down, as the grasp left it (the real station leaned the hand at most 35 deg
for a hanging end; this skill does not lean it). The overshoot is capped by
the fingers, and the pass is judged by the hand's travel and, when a camera
is named, a point on the tip, with routed exits; RoboTwin checks success
from the simulator. The approach-then-push-through strategy for a deformable
linear object through a ring (insert, then pull from the far side) is the one
DLOInsert / DLOPull describe; their repositories carry no licence, so nothing
is copied from them, and the pull is left to a following skill.

The YAM specifics come from a real YAM threading station's own insert
(`ring1_real.py`, `i_plan` / `i_carry` / `i_align` / `i_push` / `i_confirm`):
the 4.5-5 cm standoff, the 4 mm hole margin, the 10 mm "through" test, the
carry in front of the ring at its mouth's height, and the advice to take the
cable 5 cm rather than 6.5 cm behind its tip (a cantilever's sag grows with
the cube of its free length; in that station's twin a 6.5 cm end hung 50-70
deg and could not be aligned).

## Simulation check

A YAM station twin (left arm, linear_4310 fingertips reaching 10.2 mm, the
jaws' gains 6000 / 120) with the threading twin's 3.5 mm charging cable (a
rod of 7.2 mm capsules, 2000 kg/m^3, bending 0.08 / 0.1, pads at mu 0.5)
laid straight on the tablecloth 0.27 m in front of the arm, and the threading
twin's ring 1 (21.1 mm radius, 2 mm wire: 20 capsules on a 4 mm post) standing
on the table 7 cm ahead of the cable's tip, its centre 5 cm over the cloth,
its axis along the cable toward the tip (or turned 20 deg). The cable was
taken 4-5 cm behind its tip by `grasping-a-cable-lying-flat` (top-down,
fingertips to the table, 4 cm test lift). The simulator's capsules stood in
for perception: the ring's pose, and the held tip and where the cable leaves
the fingers after the lift (`tip_position`, `exit_position`). No camera.
Judged as the threading station's `judge_ring.py` does: the centreline
crosses the ring's disc (inside 17.4 mm: radius less wire less the cable's).
The sag is the angle of the line from where the cable leaves the fingers to
the tip, below level.

| run | taken behind tip | ring | route | crosses the disc | tip past the plane (skill's estimate) | sag after the lift -> after the push |
|---|---|---|---|---|---|---|
| r1 | 45 mm | straight ahead | threaded | yes, 6.9 mm from the centre | 15.8 mm (18.3) | -1.0 -> -1.8 deg |
| r2 | 40 mm | straight ahead | threaded | yes, 6.0 mm | 13.0 mm (14.4) | -2.7 -> 3.3 deg |
| r3 | 50 mm | straight ahead | threaded | yes, 10.0 mm | 17.7 mm (21.0) | 37.2 -> 40.5 deg |
| r4 | 45 mm | turned 20 deg | threaded (wrong) | no: the cable fell out | -53.8 mm (22.8) | 15.9 deg -> on the table |
| r5 | 45 mm | turned 20 deg | jammed, cable_lost | no: the cable fell out | -53.6 mm (not pushed) | 18.5 deg -> on the table |

- With the ring along the cable's lie, 3 of 3 went through and stayed
  through for 2 s held. The 5 cm hold sagged 37-41 deg; aimed between the tip
  and the fingers (the tip 8.6 mm under the axis, the fingers 15 mm over it),
  the cable still crossed 10 mm from the centre. The skill's tip estimate
  (the hold measured after the lift, carried rigidly) read 1.4-3.3 mm further
  than the capsules.
- With the ring turned 20 deg, the twin's solver left the wrist roll free (the
  pose move reported arrival with the end unturned) and the last-joint turn
  (18.6-19.1 deg) twisted the cable out of the pads: it lay on the table
  before the carry. r4 ran before the jaw check existed and reported
  `threaded` from the hand's travel alone; r5, with it, reported `jammed`
  with `cable_lost` and pushed nothing.
- The sag after the same 45 mm grasp came out -1, 16 and 18.5 deg in three
  runs: the twin does not repeat the hold exactly, which is why the tip is
  measured after the grasp rather than assumed.
- Before these runs, with the harness's 26 cm cable, the twin's weak pinch
  lost it twice: once rising 10 cm to come over the mouth (all of it
  hanging), once during the push (the 20 cm lying behind dragged along the
  cloth and the pads slid down the cable until the end left them). Hence the
  carry level with the mouth, and the 13 cm cable in the runs above. On the
  real station a carried end fell out only at double speed.

**What this does not show.** The hand pushed in clear air: no run met the
rim, so `jammed` from a stopped push is covered by unit tests only, as are the
camera looks. The pinch in the twin is weaker than the real pads (5.4 mm jaw
gap on a 3.5 mm cable), so cable loss is likely overstated there; a wrist turn
with the rest of the cable lying on the table is still a real risk (the real
station saw an end twisted out of the pads turning 116 deg). Nothing here ran
on a real YAM.

## Live check of the camera look

`gemini-er.point` (gemini-robotics-er-2-preview) with this skill's query and
`_deproject`, on six real watching-wrist frames at ring 1's mouth from a run
of the threading station (two looks, three frames each, depth and calibrated
pose):

- First look, 3 of 3 frames: the point sat on the plug's tip; placed in 3-D,
  39-44 mm short of the ring's plane and 25-27 mm off its axis (consistent
  to 4.4 mm across the frames; the ring there is the twin's prior, which that
  run found 20 mm off). The station's own depth fit of the same frames read
  the end 80 deg up, where the picture showed it level.
- Second look (the hand turned 15 deg), 1 of 3 found the tip; at that pixel
  the window held no cable depth, only the table behind (it read 13 cm short
  of the ring), so the tip is now not placed when fewer than three window
  points stand 2 cm in front of the rest.
- The model is served only on the API's v1beta: the hub's `gemini-er` tool
  with google-genai 3.0.0 (which defaults to v1) got 404 until the client
  was pinned to v1beta for this check.

## Provenance

Adapted from RoboTwin's `envs/hanging_mug.py` (`play_once`) and
`envs/_base_task.py` (`get_place_pose`, `place_actor`), MIT, Copyright (c)
2025 Tianxing Chen (LICENSE.robotwin), with the changes above. Method
reference (not copied, no licence): DLOInsert / DLOPull.
