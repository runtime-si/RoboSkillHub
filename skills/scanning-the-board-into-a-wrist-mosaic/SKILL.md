---
name: scanning-the-board-into-a-wrist-mosaic
description: Scans a table region (the jigsaw board, a staging area) with the
  wrist camera from straight above and blends the frames into one top-down
  image at a chosen resolution - picks the camera height for the asked pixels
  per metre (10-30 cm), tiles the region with 25 % overlap in a serpentine,
  previews each pose with IK (trying the other wrist rolls), and maps every
  frame onto the region's plane by the homography its own camera pose and
  intrinsics induce, feathered at the frame borders; checks the overlaps by
  phase correlation and also returns a height map from the depth. Gives the
  board corners in the mosaic for registering-assembled-board-to-reference.
  Use when the overhead camera sees too few pixels per piece (about 11 at
  0.9 m) and a board-wide check or a staging inventory needs 50-100 px per
  piece; moves the arm, no learned model.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, mosaic, stitching, wrist-camera, homography, scan]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/OpenStitching/stitching
      upstream_license: Apache-2.0 (idea of feathered rectified blending; no code used)
      paper: "R. Hartley, A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., 2004 (plane-induced homography, eq. 8.1)"
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.solve_ik
    - robot.go_to_pose
  required_inputs:
    region_center: Vec3
  # Optional inputs (wired from a script node where gap has no type for them): region_size
  # ([w, h] m), region_yaw_deg, px_per_m (2000), camera_name, surface_z, height_m,
  # return_height, arm_id.
  produces_outputs:
    metres_per_pixel: float
    covered_fraction: float
    seam_shift_px: float
    tiles_seen: int
  exit_conditions:
    mosaic_built: Every planned frame was reached and the frames cover at least 98 % of the region; `mosaic` (RGB), `height_map` (m over the surface), `metres_per_pixel`, `origin` and `board_corners_px` describe it.
    partial: Some frame positions were not reached (IK, refusal) or under 98 % is covered; the mosaic is returned with the gaps black and the height map NaN there.
    failed: No wrist camera with a pose in the observation, more than 64 frames needed, or no frame position reached.
  canonical_scripts:
    - scan_mosaic: scripts/scan_mosaic.py
  examples:
    - title: Scan the board at 4 px per mm for a cell-by-cell check
      path: examples/canonical_subgraph.json
---

# scanning-the-board-into-a-wrist-mosaic

The overhead camera sees a 2.25 cm piece at about 11 pixels: enough to find
pieces, not to check them. The wrist camera at 10 cm sees the same piece at
80-90 pixels, but only a hand's width of the board at a time. This skill moves
the wrist camera over the region and puts the frames together into one
straight-down image.

## How

1. **Camera.** The wrist camera (named, or the eye-in-hand camera riding
   nearest the tool) gives its intrinsics and its pose; with the tool pose they
   give the camera's mount on the tool.
2. **Height and tiles.** Height = fx / `px_per_m`, kept within 10-30 cm
   (or `height_m`). The central 90 % of a frame is its footprint; frame centres
   step by 75 % of it across the region, in a serpentine.
3. **Moves.** For each frame the camera is put straight above its centre
   looking down, its image x along the region's x; the tool pose follows from
   the camera mount. `robot.solve_ik` previews it; if the arm cannot, the camera
   is rolled 180, 90 or -90 degrees, then tilted 15 or 30 degrees (the warp
   does not care). Frames whose poses all fail are skipped.
   After the move, where the camera's optical axis actually meets the surface
   is compared with the frame's centre and the hand is shifted to close the
   gap (up to twice, to 4 mm): the station twin's IK constrains only the tool's
   approach axis, so the camera, mounted off that axis, swings away with the
   free wrist roll (without this, 4 frames covered 58 % of the region).
   Two more passes aim new frames at the largest uncovered patches.
4. **Warp.** The frame's actual camera pose (from the observation, not the
   commanded one) and intrinsics give the homography from the region's plane
   to the image: K [r1 r2 r3 z0 + t]. Composed with the plane-to-mosaic map it
   resamples the frame into the mosaic. Weights fall linearly to zero at the
   frame's border; the mosaic is the weighted mean.
5. **Height map.** Depth points are lifted to the world and binned into the
   mosaic grid (height over the surface, m; NaN where none).
6. **Seams.** Overlapping frames are compared by phase correlation (Hanning
   window); `seam_shift_px` is the median shift over the pairs: the mosaic's
   internal misregistration from pose error.

The mosaic shows the region from above with its +x to the right and +y up.
`origin` is the world point of mosaic pixel (0, 0); `board_corners_px` are the
region's corners (top-left, top-right, bottom-right, bottom-left), ready for
`registering-assembled-board-to-reference` when the region is the board.

## Inputs

- `region_center` (world; its z is the surface unless `surface_z`),
  `region_size` ([w, h] m), `region_yaw_deg` (0).
- `px_per_m` (2000, i.e. 2 px per mm), `height_m` (override), `camera_name`,
  `return_height` (true), `arm_id`.

## Outputs

`mosaic` (RGB uint8), `height_map` (float32, m), `metres_per_pixel`, `origin`,
`size_px`, `board_corners_px`, `tiles`, `tiles_seen`, `covered_fraction`,
`camera_height_m`, `seam_shift_px`, `seam_shift_max_px`, `detail`.

## Measured

**Simulation check** (YAM station twin, left arm, its wrist D405 at the real
station's calibrated pose, 640 x 480; a 2 mm jigsaw piece seated in a fixed
board of its six neighbours; harness in the pull request):

- **24 x 18 cm region at 4000 px/m, camera at 10 cm:** `failed`. No frame pose was reachable straight down or tilted 15-30 degrees at that height over this spot.
- **30 x 22 cm region at 2000 px/m** (auto height 19.7 cm): `mosaic_built`, 4 / 4 frames, 100 % covered, 600 x 440 px.
  - **Seams:** 0.24 px median, 0.32 px max (phase correlation between overlapping frames).
  - **Height map:** piece top +0.00 mm, mat -2.01 mm (true: 0 and -2.0).
  - **Geometry against the meshes at their true poses** (colour masks of the mosaic):
    - board outline 1.0 mm median from its true outline, IoU 0.79;
    - piece outline 0.7 mm median.
    - The 95th percentiles (6.5 mm and 18 mm) and the 2.5-5 mm centroid offsets come from what stands off the plane: the harness's 10 mm-tall grip block on the piece and the board's side walls smear outward by parallax, as anything not on the surface does in a planar mosaic.

Pieces at 2 px per mm are about 45 px across (22.5 mm), against 11 px from
the overhead camera; at 4 px per mm, 90 px, where the arm can reach.

## Limits

- **The mosaic is only as good as the poses.** No feature-based refinement:
  hand-eye calibration error and the arm's own error move each frame; the seam
  check measures it but does not correct it.
- **Flat regions only**: everything is mapped onto one plane; a piece 2 mm
  proud is displaced by up to 2 mm x (its distance from the image centre over
  the camera height) - under 0.5 mm at the frame edge from 10 cm.
- **Low heights may be out of reach**: in the twin no view at 10 cm over the
  board was reachable; 20 cm was. Ask for what the arm can do, or give
  `height_m`.
- **Time.** Each frame is a move and a settle; a 46 x 61 cm board at 2 px per
  mm is about 30-40 frames.
- The skill does not avoid obstacles between frames; the region must be clear
  at the camera height plus the hand's reach.

## The constants, and why

`OVERLAP` 0.25, `VIEWS` (rolls, then 15 and 30 degree tilts), `AIM_TOL_M` 4 mm and `AIM_STEPS` 2, `FILL_PASSES` 2, `MIN_HEIGHT_M` 0.10 / `MAX_HEIGHT_M` 0.30 (D405 range, hand
clearance), `USE_FRACTION` 0.9, `MAX_TILES` 64, `SETTLE_STEPS` 30,
`SEAM_MIN_OVERLAP_PX` 32.
