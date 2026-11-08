---
name: correcting-insertion-offset-from-port-opening
description: Re-centres a held part over a port before or after a failed
  insertion using cameras only - finds the dark opening near its projected
  mouth in every available view (hand and part masked out by depth), meets each
  view's ray with the port plane, keeps the views that agree within 3 mm, and
  steps the hand toward that consensus in moves of at most 1 mm (6 mm total, 8
  steps) until the part's descent line is centred on the opening in at least
  two views. Use when an insertion stopped short or a camera's port estimate is
  only good to a few millimetres, the hand holds the part over the mouth, and
  two cameras see the opening (the inserting arm's wrist camera plus the other
  arm's parked wrist camera or the overhead camera); no force sensor.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, recovery, visual-servoing, multi-view, peg-in-hole]
  hub:
    stage: recover
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/AMMistry18/aic/blob/fd27bbe875dbd656e381f72ae2b44efdfb243924/aic_model/aic_model/insertion/visual_gap_recovery.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - sam3.segment_text
  required_inputs:
    port_position: Vec3
    opening_size: float
  produces_outputs:
    hole_position: Vec3
    hole_offset: float
    correction: Vec3
    steps: int
    views: int
  exit_conditions:
    centered: The held part's descent line meets the port plane within the pixel tolerance of the opening's centre in at least min_views (2) views; the hand stays there, ready for the push. hole_position is the cameras' consensus of the opening's centre on the port plane.
    opening_not_found: Fewer than min_views cameras are usable or find the opening near its projected mouth (contrast, area and centre gates); nothing moved on this pass.
    views_disagree: The views found the opening but no pair puts it within max_view_disagreement (3 mm) on the port plane; a camera's calibration or the detection is off. Nothing moved on this pass.
    offset_too_large: The consensus lies more than max_port_offset (8 mm) from the estimate, or the next step would take the commanded correction past max_total (6 mm); re-perceive the port rather than correct. An opening near the bound can be measured short until the hand has stepped toward it, so the hand may have moved before this exit.
    no_convergence: After max_steps (8) moves the part is still not centred in min_views views; the hand is where the last step left it.
    unreachable: The hand's pose could not be read, or a step was refused or reported out of reach.
  canonical_scripts:
    - correct_offset: scripts/correct_offset.py
  examples:
    - title: Re-centre a held part over a perceived port, then push it in
      path: examples/canonical_subgraph.json
---

# correcting-insertion-offset-from-port-opening

A port estimated from one camera is good to a few millimetres; a part with a
millimetre of clearance needs better, and without a force sensor the arm cannot
feel its way in. Two cameras can measure the last millimetres instead: each
sees the opening as a dark blob next to where it should be, each blob's ray
meets the port plane at a point, and where two views agree that is the hole.
The hand then steps toward it in small capped moves, re-measuring each time.

## Where it sits in a graph

```
perceiving-functional-features (aperture centre + inward axis + radius)
  -> carry the part over the mouth (inserting-a-peg-with-pose-refinement's align,
     or approaching-above-a-target)
  -> correcting-insertion-offset-from-port-opening      <- this skill
       centered        -> push in (inserting-a-peg-with-pose-refinement's push_insert,
                          or a straight stop-on-contact move)
       no_convergence  -> searching-insertion-spiral around hole_position
       offset_too_large / views_disagree / opening_not_found -> re-perceive
  -> verifying-insertion-by-probe-push
```

After a push that stopped short (`stopped_short`), lift the part back to just
above the mouth and run this skill again before retrying: the source used it
exactly there, as the first try before a full retract.

## The method, per pass (up to max_steps + 1 passes)

1. Read the hand; the part's tip is the tool centre plus `tip_in_tcp` (zero by
   default). Its descent line meets the port plane at the **intercept**:
   comparing the opening with the intercept, not the tip, keeps a tip above or
   below the plane from adding parallax that differs between cameras.
2. For each camera in the observation with rgb, intrinsics and a camera-to-world
   pose (OpenCV axes), project the mouth's four corners (a square or rectangle of
   `opening_size` x `opening_height` on the port plane, `port_x_axis` along
   its width).
3. Mask out every pixel whose depth is more than `mask_above_plane` (8 mm) nearer
   the camera than the port plane - the hand, the fingers, the held part - plus
   any `ignore_masks[camera]`.
4. Find the opening (the source's detector, unchanged): inside the quad scaled
   by 1.55 about its centre, require a grey spread (p90 - p10) of at least 24;
   threshold at min(Otsu, median - 0.1 x spread), open/close with a 3 px kernel
   when the quad's shortest edge is 9 px or more, and keep the connected
   component with area 0.16-1.35 x the quad's and centroid within 0.75
   sqrt(area) of the quad's centre that scores lowest on 2.5 x centre distance +
   0.35 x |log(area ratio / 0.55)| + 0.3 x mean darkness. Its centroid is the
   opening's centre - unless more than 15% of its outline borders masked pixels
   (an occluder cuts it): then the projected mouth quad is slid over the image to
   best cover its visible pixels, and the slid quad's centre is used. With `opening_query`
   set (e.g. "the ring's opening"), sam3 masks of that text, clipped to the
   search polygon, are the candidates instead of the dark threshold - for an
   opening that is not darker than its surround.
5. Each found centre's ray meets the port plane. Keep the closest-agreeing pair
   of views if within `max_view_disagreement` (3 mm), plus every other view
   within that of the pair's mean; their mean is the hole.
6. More than `max_port_offset` (8 mm) from the estimate: `offset_too_large`.
7. Centred when the intercept projects within max(2 px, `pixel_tol_frac` x
   sqrt(quad area)) of the found centre in at least `min_views` views: `centered`.
8. Otherwise step the hand toward hole - intercept in the port plane, at most
   `max_step` (1 mm), keeping its rotation, unless the total commanded correction
   would pass `max_total` (6 mm).

## Inputs

- `port_position` (Vec3): the mouth's estimated centre, world frame
  (perceiving-functional-features' `aperture_center`).
- `opening_size` (m): the opening's width (or diameter); `opening_height` its
  other side when rectangular (default: square). `2 x aperture_radius` from
  perceiving-functional-features.
- `port_axis` (Vec3): the inward insertion axis (`fixture_axis`); default straight
  down.
- `tip_in_tcp` (Vec3): the held part's tip in the tool-centre frame, e.g. from
  registering-held-objects; only its sideways part matters.
- `cameras`: names to use; default every camera in the observation.
- `min_views` (2): 1 skips the agreement check and trusts a single view - only
  for a well calibrated camera that sees the opening square on.
- `pixel_tol_frac` (0.12, upstream): how close counts as centred, as a fraction
  of the opening's projected size - so about that fraction of `opening_size` in
  metres. **Set it near clearance / opening_size**: upstream's 0.12 is 2.9 mm on
  a 24 mm opening, and in the simulation check it stopped 1.8-2.5 mm off centre,
  where the closed fingertips caught the rim three times in seven.
- `opening_query`: a sam3 text for the opening, used instead of the dark
  threshold. `ignore_masks`: `{camera: mask}` of pixels never to use.
  `mask_above_plane` (8 mm): the depth margin of the hand mask; 0 turns it off.
- `max_steps`, `max_step`, `max_total`, `max_port_offset`,
  `max_view_disagreement`: the bounds below.
- `move_tolerance`: passed to `robot.go_to_pose` as `tolerance` only
  when set. **Set it below 1 mm on a connector whose own arrival tolerance is
  larger than a step** (RoboSimStudio's servo: 4 mm), or it reports a 1 mm move
  done without moving.

## Required end states

- `centered`: the hand is over the opening; push next.
- `opening_not_found`: give the skill a second view (park the other arm's wrist
  camera looking at the port, e.g. with approaching-above-a-target), raise the
  hand so it hides less of the mouth, or pass `opening_query`.
- `views_disagree`: check the cameras' extrinsics; re-perceive.
- `offset_too_large`, `no_convergence`: re-perceive the port, or hand
  `hole_position` to searching-insertion-spiral.
- `unreachable`: the region is out of reach.

## The constants, and why

All the motion bounds and detector gates are the source's own values, tuned on
the AI for Industry Challenge's SFP-port insertion (simulated wrist cameras,
442x393 frames and full-resolution streams): steps of 1 mm, 6 mm total, 8
steps, 8 mm offset bound, 3 mm view agreement, pixel tolerance 0.12 sqrt(area)
with a 2 px floor, and the detector's 1.55 / 24 / 0.16-1.35 / 0.75 gates.
`MASK_ABOVE_PLANE` (8 mm) and `OCCLUSION_REFIT` (0.15 of the outline) are new
(see below).

## What changed from the source, and why

- **Gripper masks by depth.** The source loads three calibrated PNG silhouettes
  of its own gripper, one per wrist camera. A YAM station has different cameras,
  and the second view may be the overhead camera or the other arm's wrist camera,
  where the hand moves through the frame. Pixels nearer than the port plane by 8
  mm are masked instead - camera-only, any camera. The YAM's gripper is black,
  so without this mask the hand itself is the darkest blob near the mouth.
- **Each step from the last target, not the read-back pose.** The source sends
  read-back + step to a compliant controller. A YAM lags its command under
  gravity by a millimetre or more, so read-back + 1 mm can land where it started;
  integrating the targets absorbs the lag, and the 6 mm cap still bounds it.
- **Occluded openings refitted.** The wrist camera of the arm holding the part
  sees the fingertips over the far half of the mouth; the visible half's centroid
  sat 3.5 mm from the true centre in the simulation check, so a view whose
  opening is cut by masked pixels is re-centred by sliding the projected quad
  over what is visible (0.7 mm). Unoccluded views keep the source's centroid.
- **`pixel_tol_frac` exposed** (the source's constant): see Inputs.
- **Named exits instead of a fall-back.** The source falls back to its
  force-based sweep on any failure; here each failure is an exit the graph can
  route (no force sensor on a YAM).
- **The port quad from a centre, axis and size** instead of the task's CAD
  keypoints; **sam3 as an alternative detector**; no debug-image logging.

## Checked in simulation

RoboSimStudio, the YAM station twin, left arm. The closed empty hand stands in
for a held part (its fingertips 10.2 mm below the tool centre); the port is a
24 mm square opening with a black floor in a light 12 mm-walled frame, 15 mm
deep. Three cameras: the station's calibrated overhead camera, a wrist camera on
the inserting arm (60 mm beside and 90 mm behind the tool centre, looking at the
fingertips, like the YAM's D405) and a fixed camera 0.3 m from the port at 50
degrees elevation, standing in for the other arm's parked wrist camera. The
hand is put top-down 30 mm (tool centre) over an estimate a few millimetres off
the true mouth; `pixel_tol_frac` 0.04, `move_tolerance` 0.5 mm. Then a straight
stop-on-contact descent from where the hand is: the control at the estimate
before the skill, and again after it. Tip depth is how far below the mouth the
fingertips stopped (11.3 mm when centred: the tapered fingers then meet the
walls).

| estimate off by (mm) | hand off at start | control tip depth | exit, steps | hand off after | cameras' hole error | tip depth after |
|---|---|---|---|---|---|---|
| (0, 0) | 0.57 mm | 11.3 mm | centered, 0 | 0.11 mm | 0.16 mm | 11.1 mm |
| (0, -4) | 4.04 | 8.0 | centered, 4 | 0.15 | 0.20 | 11.0 |
| (3, 3) | 4.03 | 3.1 | centered, 4 | 0.22 | 0.19 | 10.9 |
| (4, 0) | 4.68 | on the rim | centered, 4 | 0.82 | 0.13 | 9.8 |
| (-2, -5) | 5.15 | 7.2 | centered, 6 | 0.48 | 0.15 | 10.8 |
| (-4, 2) | 3.57 | 11.3 | centered, 5 | 0.50 | 0.28 | 10.2 |
| (-5, -3) | 5.31 | 11.6 | centered, 6 | 0.27 | 0.24 | 10.6 |
| (10, 0) | 10.69 | on the rim | offset_too_large after 1 | 9.29 | 1.44 | on the rim |

- The overhead camera rarely found the opening (from behind the arms the hand
  hides the mouth; in these runs only with the hand 10 mm off); the wrist and the
  side camera carried every centred run. The wrist view had 16-28% of the opening's outline
  on masked fingertip pixels in every run, so it always went through the refit.
- With upstream's `pixel_tol_frac` 0.12 (an earlier build, before the refit was
  anchored on the projected centre) the six runs that needed a correction ended
  `centered` 1.8-2.6 mm off, and three descents stopped 1.6-3.5 mm shallower than
  when centred.
- Before the occlusion refit, the wrist view's centroid leaned 3.5 mm toward the
  camera: two of seven runs ended `views_disagree` (3.0 and 4.0 mm) and the
  consensus was 1.0-1.7 mm off in the rest.
- The fingers are not symmetric: from 4-5 mm toward -x the control descent went
  in anyway, so the depth column says more on the +x and y offsets.

Not checked: a real held part, a real port, real camera calibration error, the
sam3 detector against a live model, a real YAM's lag.
