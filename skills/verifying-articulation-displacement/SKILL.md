---
name: verifying-articulation-displacement
description: Checks from the camera whether a drawer or a door really moved as
  asked - re-finds its handle in a fresh RGB-D frame (pointing model, SAM, then
  the handle's front proud of the fitted face plane), checks the face normal is
  the one expected, and measures the handle's travel along the articulation
  (prismatic - metres along the pull; revolute - the angle about the hinge),
  routing moved, moved_short, not_moved, moved_wrong_way, inconsistent or
  not_seen. Use after any skill that opens or closes an articulated object (a
  drawer pull or push, a door swing, the waypoints of
  planning-articulation-opening), once the hand has let go and moved clear,
  before the graph relies on the drawer or door being open or shut.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [articulated, drawer, door, cabinet, handle, verification, rgb-d]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [articulated]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/close_drawer/libero/policy.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - gemini-er.point
    - molmo.point_prompt
    - sam3.segment_point
    - vlm.query_yes_no
  required_inputs:
    handle_description: string
    joint_type: string
    expected_opening: float
  produces_outputs:
    seen: bool
    measured: float
    fraction: float
    transverse: float
    normal_dot: float
    handle_after: Vec3
    face_normal_after: Vec3
  exit_conditions:
    moved: The handle travelled at least min_fraction (70%) of expected_opening in its sign, the face normal is the one expected and the handle stayed within 70 mm of the articulation.
    moved_short: It travelled the right way by at least 30 mm of handle travel (a door - that arc at the handle's radius), but less than min_fraction of what was expected; stuck, or the hand slipped part way.
    not_moved: It travelled less than 30 mm (that arc for a door) either way.
    moved_wrong_way: It travelled at least 30 mm against the expected sense (a close that opened it).
    inconsistent: The handle was found, but the face normal is not the one expected (cosine under 0.9), the handle drifted more than 70 mm off the articulation, or a drawer moved more than 250 mm; most likely another handle or face was found, or the cabinet itself moved.
    not_seen: The pointer found no handle, or no vertical face plane or nothing proud of it was found round the pointed pixel, in the frame after (or before) actuation; the hand may still hide it.
  canonical_scripts:
    - verify_articulation: scripts/verify_articulation.py
  examples:
    - title: Check a drawer pull planned by planning-articulation-opening
      path: examples/canonical_subgraph.json
---

# verifying-articulation-displacement

A hand that pulled a drawer only knows where the hand went. This skill looks
at the drawer: it finds the handle again in a fresh frame and measures how far
it went along the articulation - RoboRSI's `_visual_drawer_close`, made one
verify node for drawers and doors, opening and closing.

## How it measures

1. **Point.** `gemini-er.point` (or `molmo.point_prompt`, `pointer=`) is
   asked for `handle_description` in the observation's `camera_name` camera;
   `handle_pixel` skips this. `sam3.segment_point` at that pixel tightens it to
   the mask's centroid, unless the centroid lies more than 77 mm away (RoboRSI's
   SAM refinement; its failure keeps the raw point).
2. **Face.** World points on three rings round the pixel (32, 52, 77 mm) are
   fitted with RoboRSI's plane search: the plane through three of them with the
   most points within 15 mm, refined by PCA, rejected if not flat, and for a
   vertical face (`vertical_face`, the default) if its normal has a vertical
   component over 0.25. If the rings fit nothing, a square annulus (26-52 mm)
   is tried. The outward normal is horizontal and signed toward the camera.
3. **Handle.** Points within 26 mm of the pixel (and within 86 mm of its world
   point) standing 4-60 mm proud of the plane are the handle; the connected group nearest the pixel is kept and its
   front - the median of its points within 5 mm of the most protruding one -
   is the handle point.
4. **Optionally confirm** (`confirm_with_vlm`): `vlm.query_yes_no` is shown
   the frame with a red cross on the handle point and a magnified crop and
   asked whether the cross is on the handle (RoboRSI always asked).

The handle *before* actuation is found the same way in `before_observation`
(best: the same instrument both times), or taken from `handle_before`.

## The verdict

- **Prismatic** (drawer): the handle's displacement along `pull_direction`
  (default: the face normal before), and its drift across it. The face normal
  after must agree with the one before (cosine at least 0.9): the before
  frame's, else `face_normal_before`, else `pull_direction`.
- **Revolute** (door, lid): the signed angle the handle turned about the hinge
  line (`hinge_point`, `hinge_axis`; the axis is signed by `pull_direction`
  exactly as planning-articulation-opening does), and its drift - the larger of
  the change in its radius from the line and its shift along it. The face
  normal after must agree with the one before turned by that angle; without a
  normal before this check is skipped (`normal_dot` is NaN).

`expected_opening` is signed: positive opens (along the pull, or the door's
opening sense), negative closes. `measured` uses the same convention and units
(m or rad), and `fraction = measured / expected_opening`.

## Fits planning-articulation-opening

Its `joint_type` and `opening` (m or rad) go straight into `joint_type` and
`expected_opening`, and its inputs `pull_direction`, `hinge_point`,
`hinge_axis` into the same names here. Its `handle_pose` is accepted as
`handle_before`, but it is the hand's tool centre on the handle, not the
handle's front: along the pull that biases a drawer's measured travel by the
handle's depth (10-30 mm). Pass a frame from before the grasp as
`before_observation` instead whenever there is one.

## Inputs

- `handle_description`, `joint_type`, `expected_opening` (required).
- `handle_before` (Vec3 or Se3Pose) or `before_observation`;
  `face_normal_before`, `pull_direction`, `hinge_point`, `hinge_axis`.
- `observation` (default: a fresh `robot.get_observation`), `camera_name`
  (`overhead`; the only camera if there is one), `handle_pixel`,
  `handle_pixel_before`.
- `pointer`, `refine_with_sam` (true), `confirm_with_vlm` (false),
  `vertical_face` (true), `front_band` (5 mm; 0 is RoboRSI's single pixel),
  `min_fraction` (0.7), `min_motion` (30 mm), `normal_dot_min` (0.9),
  `max_transverse` (70 mm), `max_travel` (250 mm).

## Outputs

`seen`, `measured`, `expected`, `fraction`, `transverse` (m), `normal_dot`,
`normal_angle` (rad, revolute: how far the face normal itself turned),
`radius` (revolute), `handle_before`, `handle_after`, `face_normal_after`,
`handle_pixel` (`{u, v}` in the RGB image), `detail` (the numbers in words).

## Boundaries

- Let go and move the hand clear first: a hand on the handle hides it, or is
  measured instead of it.
- The handle must stand 4-60 mm proud of a flat face, with face round it
  (out to 26-77 mm). Recessed handles, knobs on narrow frames and glass fronts
  are not found (`not_seen`).
- A door swung near edge-on to the camera (face normal within ~3 deg of
  perpendicular to the view) gives no plane and routes `not_seen`; place the
  camera, or the expected angle, so the face stays in view.
- The fitted face normal takes in what lies round the face within 15 mm: next
  to a door's edge it read up to 8 deg off in the test scene, and on a drawer
  pulled far out and seen from above, 27-30 deg (routed `inconsistent`).

## Checked on synthetic frames, and with the live pointer and SAM

The YAM station twin has no articulated object, so the skill was checked on
ray-cast RGB-D frames of a cabinet with two drawers or a side-hinged door
(`tests/bundles/_articulation_scene.py`), 0.55 m in front of the left arm
base, seen by the station overhead camera at its calibrated pose (steeply
from above, 0.7-1.0 m away) with its intrinsics (f = 461 px, 640 x 480) and
its depth noise as measured on a flat table patch of a real station frame
(2.9 mm std at 1 m, quadratic in depth, 1 mm steps), 5% of pixels dropped at
random and half of those at depth edges. Before and after frames were both
localised.

**Monte Carlo** (pointer on a random pixel of the handle plus 3 px of noise,
SAM returning the handle's mask, 20 trials per case):

| case | routes | error of `measured` |
|---|---|---|
| drawer, expected +150 mm, moved 0 / 50 / 100 / 150 | not_moved 20, moved_short 40, moved 20 | mean within 0.1 mm, sd 0.2-0.3 mm, max 0.7 mm |
| drawer, expected -150 mm, moved -150 / 0 / +50 | moved 20, not_moved 18 + not_seen 1 + inconsistent 1, moved_wrong_way 19 + inconsistent 1 | mean within 0.1 mm, sd 0.2-0.3 mm, max 0.7 mm |
| door, expected +45 deg, moved 0 / 15 / 30 / 45 / 60 deg | not_moved 20, moved_short 40, moved 40 | mean within 0.3 deg, sd 0.1 deg, max 0.5 deg |

237 of 240 routed as the truth says. The misses are all on a drawer already
150 mm out, seen almost from above: the face fit took in the tray beyond the
drawer's front and its normal came out 27-30 deg off (`inconsistent`, twice),
or nothing proud of the fitted face was found round the pointed pixel
(`not_seen`, once). With RoboRSI's single most protruding pixel as the handle
point (`front_band=0`) the same trials routed 222 of 240: on the 14 cm
vertical door handle that pixel wandered along the bar with the depth noise
(15 doors `inconsistent`, 72-97 mm drift), and the drawer's error sd was
0.4-0.8 mm, with biases up to 1.5 mm. Hence the 5 mm front band. Before the
handle search was also gated to 86 mm round the seed in 3D, a door at 60 deg
read the table under it as its handle (about 276 mm drift, `inconsistent`) in
5 of 30 extra trials; with the gate that case routed `moved` 20 of 20.

**Live** (Brev L40S): the same eight before/after cases - drawer +150 mm
expected and 150 / 50 / 0 moved, -150 expected and -150 / +50, door +45 deg
expected and 45 / 20 / 0 - with `gemini-er.point`
(gemini-robotics-er-2-preview) on the rendered RGB and SAM3's point prompt
(facebook/sam3 through transformers' Sam3TrackerModel). Asked for "the drawer
handle" and "the door handle", 8 of 8 routed as the truth says, the pointer
was on the handle in 16 of 16 frames, SAM's mask matched it (IoU 0.98-1.00),
errors were within 0.3 mm and 0.1 deg; each call took 1-2 s (pointer) and
0.3-0.9 s (SAM). Asked for "the handle of the upper drawer", Gemini found
nothing in 3 of 16 frames - those where the open upper drawer hides the lower
one, so no drawer looks "upper" - and those cases routed `not_seen`; the
other 5 routed right. Word `handle_description` for what the camera will see
after the motion. `confirm_with_vlm` was not run live. No real drawer or door
has been measured yet.

## The constants, and why

RoboRSI's radii were pixels of LIBERO's 256-px agentview (f = 309 px) about
1 m from the drawers, so one pixel there is taken as **3.2 mm**; the rings
(10, 16, 24 px), the handle search (8 px), the dense annulus (16 px) and the
SAM shift limit (24 px) are that many 3.2 mm steps here, turned into this
camera's pixels at the handle's depth.

- **15 mm plane inlier band, 5 inliers, 64 triplet points, planarity 0.35,
  triplet tilt 0.65, face tilt 0.25, sign alignment 0.05, protrusion
  4-60 mm, adjacency 2.5 steps, 3-pixel groups**: RoboRSI's.
- **Normal cosine 0.9, 70 mm drift, 250 mm drawer travel**: RoboRSI's
  `_visual_drawer_close` / `_visual_handle_pull` bounds.
- **30 mm motion**: RoboRSI's `_MIN_VISUAL_PULL`.
- **70% of the expected**: RoboRSI's close counts 0.14 m measured of a 0.20 m
  command.
- **5 mm front band**: under twice the station camera's 2.9 mm noise at 1 m;
  measured above.
- **86 mm 3D gate**: the 26 mm search plus the 60 mm largest protrusion.

## Provenance

Adapted from RoboRSI (Apache-2.0), `close_drawer/libero/policy.py`
(`_visual_drawer_close`) and `pull_drawer/libero/policy.py`
(`_visual_handle_pull` and the handle resolver it calls). Changed: a hub
Observation deprojected here instead of the simulator's `pixel_to_world`;
metric radii instead of LIBERO pixels; the normal signed toward the camera,
not the robot base; hub pointing, SAM and VLM tools; RoboRSI's LIBERO
drawer-layer scan and VLM pixel correction dropped; the handle search gated
in 3D as well as in pixels; the handle's front median instead of its single
most protruding pixel; revolute joints and either sign;
named exits instead of a distance or None.
