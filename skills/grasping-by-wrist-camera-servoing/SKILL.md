---
name: grasping-by-wrist-camera-servoing
description: Grasps a small part by closed-loop eye-in-hand servoing - from a
  pregrasp standoff above a coarse target, it segments the part in every wrist
  RGB-D frame (sam3.segment_text), takes the centre of the mask's top-face
  points back-projected with the wrist depth,
  steps the hand sideways (capped, IK-previewed robot.go_to_pose_cartesian
  steps) to bring that point under the tool centre, descends while it stays
  centred, and closes once the tool centre reaches the grasp height with the
  part under it; a lost track, a stalled arm or the step cap fall back to an
  open-loop grasp at the last point seen. Use when an overhead estimate is a
  few millimetres off and that is too much - LEGO bricks, dominoes, jigsaw
  pieces or 20-30 mm parts picked with the jaws pre-shaped close to the part's
  width - and the arm carries a wrist camera.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasp, visual-servoing, eye-in-hand, wrist-camera, d405, small-parts, closed-loop]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/hello-robot/stretch_ai/blob/6272b28e3ff7be482af3b427a4551d1772ebf7d7/src/stretch/agent/operations/grasp_object.py
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.solve_ik
    - robot.grasp_frame
    - robot.describe_gripper
    - robot.get_gripper
    - robot.set_grip
    - robot.open_gripper
    - robot.close_gripper
    - sam3.segment_text
  required_inputs:
    object_name: str
  produces_outputs:
    grasp_point: Vec3
    coarse_point: Vec3
    mode: str
    correction: float
    final_error: float
    grip_width: float
    held_width: float
  exit_conditions:
    grasped: The jaws closed on something (the gap stayed over fully closed + 3 mm) and still held it after a lift of lift_height; `mode` says whether the servo closed (`servo`), the open-loop fallback did (`open_loop_fallback`, `fallback_reason` says why) or the skill ran open-loop (`open_loop`, servo=false). The hand stays lifted with the part.
    missed: The jaws closed on nothing, or the part slipped out in the lift; the jaws are opened and the hand lifted clear.
    lost_target: There is no wrist camera, the segmenter never found the part near the coarse estimate in MAX_LOST frames from the pregrasp view, or (fallback=false) it lost the part mid-servo; the hand stays where it was, jaws open.
    unreachable: No top-down grasp rotation, no pregrasp view passed the IK preview and the clearance floor or arrived within 15 mm, or a move of the open-loop grasp was refused; `detail` lists each attempt.
  canonical_scripts:
    - servo_grasp: scripts/servo_grasp.py
  examples:
    - title: Grasp a small part from an overhead box by wrist-camera servoing
      path: examples/canonical_subgraph.json
---

# grasping-by-wrist-camera-servoing

An overhead RGB-D camera puts a 2 cm part within a few millimetres, and a
YAM's moves are repeatable to a few millimetres more. A jaw opened a few
millimetres wider than the part does not forgive that. This skill closes the
loop on the wrist camera instead: it re-measures where the part is relative to
the hand in every frame and corrects before the fingers go past the part's
top.

## How it composes

- **Before it**: a coarse target - `perceiving-objects`' OBB, or
  `refining-target-from-wrist-closeup`'s `refined_obb` (the closer the start,
  the fewer servo frames). An OBB is better than a point: it gives the
  part's width (the jaw pre-shape), height (the grasp depth) and footprint
  (the first-sight bound).
- **After it**: the hand holds the part lifted `lift_height` above where it
  closed; `grasp_point` is the last top-face centre the camera measured.
  `verifying-grasps` can confirm it, a place skill takes over.
- **Loads `refining-target-from-wrist-closeup`'s script**: the wrist camera
  choice, the camera-to-tool transform read from the observation (no
  calibration file), the settle wait, the move-refusal test, the D405's
  7-50 cm depth range, the 50 mm clearance floor and the 15 mm arrival bound
  come from that skill, so it must be installed beside this one (it is in this
  hub). Its oblique view (the target on the optical axis, the hand
  top-down) is the fallback pregrasp.
- **Not `grasping-direct-ik` / `grasping-flat-objects`**: those trust the
  estimate and go straight down. This one uses them as its fallback shape
  (straight down to the last point seen) only when the loop cannot finish.

## On a YAM

The D405 sits ~86 mm behind and ~75 mm above the tool centre and looks ~45 deg
forward of the fingers. With the hand top-down, a point under the tool centre
crosses the camera's optical axis about 11 mm above the tool centre, so as the
hand comes down to a grasp the part moves from the top of the image to just
below its centre. The skill needs none of these numbers: each frame it
back-projects the mask with the camera pose the connector reports, so the
error it corrects is the part's position *relative to the hand*, read through
the rigid camera mount. Errors in the arm's kinematics shift the part and the
tool centre together and cancel; what is left is the mount calibration.

## What it does

1. Reads the hand (`robot.describe_gripper`: jaw-gap fit, finger reach) and
   picks a top-down grasp rotation (`robot.grasp_frame`, jaws across the OBB's
   shorter side when it has one; else the hand's own if top-down).
2. **Pregrasp** (the source's standoff along the approach): the tool centre
   straight above the coarse top at `pregrasp_height` (0.06, then 0.08,
   0.10 m), kept only if the camera-to-tool transform read from the
   observation predicts the target inside the wrist image (8 % margin) at
   7-50 cm; else refining-target-from-wrist-closeup's view, the coarse top on
   the optical axis at `pregrasp_range` (0.20, 0.25, 0.30 m). Each candidate is
   IK-previewed and kept 50 mm over the top; the hand rises in place first;
   at most three moves, each judged from a fresh pose (15 mm).
3. **Pre-shape**: `robot.set_grip` to the part's width + 2 x
   `pregrasp_margin` (6 mm a side); `robot.open_gripper` when the width is
   unknown or the connector has no partial grip.
4. **Servo loop** (at most `max_iterations` frames), per settled frame:
   - Segment `query` / `object_name` (`segmenter`, best 3, score >= 0.3,
     >= 400 px at 640 x 480). The mask under the aim pixel wins (the
     source's "centred on the right object"), else the one nearest the last
     centroid (first frame: the coarse estimate's projection).
   - Back-project the mask with the wrist depth (7-50 cm); the **top-face
     centre** is the midpoint of the extent (2nd-98th percentile) of the
     points within 1.5 mm of the 95th percentile height, along the hand's two
     lateral axes. First sight must be within the part's half-diagonal
     + 30 mm of the coarse estimate; later frames within 20 mm of the last.
   - The **aim pixel** is where a point at the part's top height under the
     tool centre projects. Lateral error = top-face centre - tool centre.
   - **Close** when the error is <= `align_tolerance` (1.5 mm), the tool
     centre is within 2 mm of the grasp height, and the masked depth in a
     30 px window round the aim pixel is valid.
   - Otherwise step: sideways by the error (capped at `lateral_step`,
     10 mm; none under 0.5 mm); down by up to `descent_step` (15 mm) - to the
     grasp height when the error is within tolerance; otherwise only while
     the source's gates hold (the pixel offset within 56 x 40 px at 640 wide,
     or the aim pixel inside the eroded mask), and only to where the
     fingertips are 5 mm above the top. Fingertips below the top while the
     error is over tolerance rise first. Every step is previewed with
     `robot.solve_ik`; a refused step is retried at half length.
   - A frame without the part holds the hand and re-looks; a step that moved
     under 30 % of its length counts as stalled.
5. **Fallback** (the source's `grasp_open_loop`) after `MAX_LOST` (3) lost
   frames, `MAX_STALLS` (3) stalls, a step refused twice, or the frame cap:
   rise straight up if the fingertips are below the top, across to the last
   seen top-face centre, down to the grasp height. `fallback=false` turns
   the lost-track case into `lost_target` instead.
6. **Close and judge** (the source's `_grasp`): `robot.close_gripper`; a gap
   within 3 mm of closed is `missed`; otherwise lift `lift_height` (5 cm) and
   read the gap again.

The grasp height is `grasp_depth` below the measured top (default half the
part's height, at most 12 mm), never under 14 mm above the support (the OBB's
bottom).

`servo=false` skips the camera: the jaws pre-shape and the hand goes straight
to the coarse estimate (the open-loop baseline, the source's
`servo_to_grasp=False`).

## Inputs

- `object_name` (required); `query` (the segmentation phrase, default
  `object_name`); `segmenter` (default `sam3.segment_text`; any tool with its
  contract: `masks`, `scores`).
- `target_obb` or `target_point`: the coarse estimate, world frame.
- `object_width`, `object_height` (override the OBB's), `grasp_depth`.
- `servo` (true), `fallback` (true), `camera`, `pregrasp_margin` (0.006 m),
  `pregrasp_height` (0.06 m), `pregrasp_range` (0.20 m), `align_tolerance` (0.0015 m), `lateral_step`
  (0.010 m), `descent_step` (0.015 m), `max_iterations` (30), `lift_height`
  (0.05 m), `min_score` (0.3), `check_reach` (true).
- `arm_id`, `tolerance`, `max_steps`: passed on only when set (`arm_id`
  when nonzero; `tolerance` / `max_steps` to `robot.go_to_pose` only).

## Outputs

`grasp_point` (last measured top-face centre; the coarse top when the camera
never saw the part), `coarse_point`, `correction` (first sight vs coarse, in
the plane), `final_error` (last measured lateral error), `mode`,
`fallback_reason`, `iterations`, `lost_frames`, `grip_width`, `held_width`,
`lifted`, `camera_name`, `trace` (one line per frame), `detail`.

## The constants, and why

- **`ALIGN_X_PX = 28`, `ALIGN_Y_PX = 20`, `CENTER_REGION_PX = 15`,
  `MIN_MASK_PX = 100`**: the source's values for Stretch's 320 x 240 D405
  servo image, scaled to the frame (x2 at 640 wide, x4 for areas).
- **`ERODE_FRAC = 0.2`**: the source eroded ~20 px, sized for household
  objects; a 25 mm part at 0.15 m is ~60 px across at 640 wide.
- **`TOP_BAND = 0.0015` m and the extent midpoint**: side faces seen at
  45 deg pull a plain centroid toward the camera, and so does perspective
  (the near half of the top face covers more pixels). In the fake rig a
  4 mm band with a pixel median put a 25 mm cube's centre 4 mm short, a
  1.5 mm band 2 mm short, a depth-squared-weighted mean 1-1.5 mm short; the
  extent midpoint is blind to both, and is what closing jaws centre on. The
  band stays above the D405's sub-millimetre depth noise at 0.1-0.3 m and
  under a LEGO stud's 1.7 mm.
- **`ALIGN_TOLERANCE = 0.0015` m**, **`PREGRASP_MARGIN = 0.006` m**: the
  servo must be within 1.5 mm before the fingertips pass the top, which
  leaves 4.5 mm of the margin for the pads' compliance and the twin's/real
  arm's step error. The margin itself keeps a neighbour a centimetre away
  clear of the open jaws.
- **`DESCENT_STEP = 0.015` m, `LATERAL_STEP = 0.010` m**: the source stepped
  5 cm on a Stretch; small parts and a YAM's few-mm repeatability want the
  frame re-read more often.
- **`MAX_LOST = 3`, `MAX_STALLS = 3`, `MAX_ITERATIONS = 30`**: the source's 10
  failed frames, 50 not-moving turns and 120 s were for a 5 Hz loop that kept
  moving; here a lost frame holds the hand, and each step waits for the arm.
- **`MAX_JUMP = 0.02` m**, first-sight bound half-diagonal + 30 mm: as in
  refining-target-from-wrist-closeup; an estimate that jumps further is
  another object.
- **`MIN_TABLE_CLEARANCE = 0.014` m**: a real YAM station refuses a lower
  tool centre. **`TIP_REACH = 0.0102` m**: the station's linear_4310
  fingertips below the tool centre, when `describe_gripper` does not say.
- **`EMPTY_MARGIN = 0.003` m**: as grasping-flat-objects.

## Measured

Simulation, YAM station twin (left arm, linear_4310 jaws), a rendered
640 x 480 wrist D405 at the real station's calibrated camera-to-tool mount,
a red cube on the table. The coarse OBB is the cube's true box shifted
3-8 mm in the plane (0 / 90 / 225 deg: along the fingers, across the jaws,
oblique). The segmenter is a harness stand-in for `sam3.segment_text`: a
colour threshold on the rendered wrist frame (fingers and all), one mask per
blob. "In hand" is the cube's centre in the tool frame after the lift;
"shove" is how far the cube moved on the table before it was lifted.

| cube | mode | trials | grasped | off-centre in hand (finger axis) | shove |
|---|---|---|---|---|---|
| 25 mm | servo | 10 (0-8 mm) | 10 | 0.03-0.66 mm | 1.2-1.5 mm |
| 25 mm | open-loop, pre-shaped | 10 (0-8 mm) | 10 | 0.7-7.7 mm (8 mm: 7.3, 7.7, 4.1) | 1.7-10.8 mm |
| 20 mm | servo | 5 (3-8 mm) | 5 | 0.01-0.41 mm | 1.0-1.2 mm |
| 20 mm | open-loop, pre-shaped | 5 (3-8 mm) | 5 | 1.1-4.5 mm | 3.4-8.6 mm |
| 20 mm | open-loop, jaws wide | 5 (3-8 mm) | 5 | 1.2-5.0 mm | 3.3-8.1 mm |

- Every servo trial closed from the servo itself (no fallback, no lost
  frame) in 6 frames from the 60 mm standoff (the camera-aimed fallback view
  was never needed), with a last measured error of 0.02-0.55 mm. The first
  frame measured the whole offset (8.4 mm for an 8 mm one) and one lateral
  step removed it; the other five were 14 mm descents at 0.1-0.6 mm. The
  measured top-face centre was 0.53-0.70 mm from the cube's true centre in
  every trial (a bias of the stand-in mask and the render, not of the
  offset).
- Across the jaws the closing jaws centre any of these offsets themselves,
  so in the twin the open-loop grasp also lifted the cube every time: at
  3-8 mm on 20-30 mm cubes the servo did not change the success count. What
  it changed is where the part sits (the open-loop error along the fingers
  stays in the grasp) and how far the jaws shove it before closing, which
  matters for a part placed next or a neighbour a centimetre away.
- Not measured: the fallback paths (lost track, stall, frame cap) in
  simulation (unit tests only), a domino or a LEGO brick, a real
  `sam3.segment_text` mask, a real YAM.

## Provenance

Adapted from Hello Robot's stretch_ai `GraspObjectOperation`
(`visual_servo_to_object`, `_compute_center_depth`, `get_target_mask`,
`grasp_open_loop`, `_grasp`), Apache-2.0; the script header lists the
changes. The source's `MaskTemporalFilter` is not ported: each frame here is
taken with the arm settled, and a frame without the part is re-looked at
rather than filled from older masks.
