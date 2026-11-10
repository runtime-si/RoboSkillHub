---
name: detecting-in-hand-slip-and-twist-from-the-wrist-camera
description: Watches the held part in the wrist camera, which rides rigid
  with the fingers, and says whether it slips, twists or drops in the jaws -
  the part's region is what lies in a near depth band in front of the
  camera; dense Farneback optical flow against the frame taken at the start,
  fitted with a RANSAC similarity (so the still fingers do not drag it),
  gives the slip in millimetres and the spin about the camera axis; the
  region's depth-plane normal gives the twist about the jaws' closing axis,
  which an image alone cannot show; a shrinking region is a drop. Use when a
  part is carried or pushed with in the jaws and the graph must know, with
  no tactile or force sensing, that it moved in the hand (a domino sliding
  down the pads, a bar or peg pivoting about the grip, a piece leaving the
  hand) before relying on its pose.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [slip, twist, in-hand, optical-flow, wrist-camera, verify]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, insertion]
    provenance:
      authors: [runtime-si]
      paper: https://www.mdpi.com/2076-3417/13/15/8620
      upstream_license: paper method (no code published); OpenCV calcOpticalFlowFarneback, Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
  produces_outputs:
    slip_mm: float
    twist_deg: float
    spin_deg: float
    area_fraction: float
    frames: int
  exit_conditions:
    stable: Over duration_s, the part's region moved under slip_mm (2 mm) and turned under twist_deg (3 deg) about the closing axis and the camera axis, and kept at least drop_fraction of its area.
    slipping: The region's dominant motion moved more than slip_mm (worst frame), with no twist over twist_deg.
    twisted: The region's depth-plane normal turned more than twist_deg about the jaws' closing axis, or the flow fit turned more than twist_deg about the camera axis.
    dropped: The region shrank under drop_fraction (40 %) of its first frame, or nothing lay in the near band at the start.
    no_camera: The observation has no wrist camera with rgb, depth and intrinsics.
  canonical_scripts:
    - detect_slip: scripts/detect_slip.py
  examples:
    - title: Watch the held part for two seconds
      path: examples/canonical_subgraph.json
---

# detecting-in-hand-slip-and-twist-from-the-wrist-camera

A YAM has no tactile or force sensing in its jaws, but its wrist camera is
bolted to them. Whatever the held part does relative to the camera, it does
relative to the fingers. This skill compares wrist frames with the one taken
when it starts, inside the part's region, in the manner of the optical-flow
slip detector of Applied Sciences 13(15):8620 (2023). That paper reports
82-87 % accuracy and 87-95 % recall with a wrist D435.

## How

1. Take the wrist camera (named `camera`, else the first whose name has
   `wrist` or `eye_in_hand`). Its first frame is the reference. The part's
   region is every pixel with depth in `near`-`far` (0.02-0.20 m).
2. Every `period_s` (0.2 s) for `duration_s` (2 s), compare the new frame
   with the reference:
   - **slip and spin**: Farneback dense flow, sampled in the reference region.
     A RANSAC similarity fit (`cv2.estimateAffinePartial2D`) takes the
     dominant motion; the still fingers count as outliers when the part
     dominates the region. Its translation in pixels, times the region's
     median depth over the focal length, is the slip in mm. Its rotation is
     the spin about the camera axis.
   - **twist**: fit a plane to the region's depth points in both frames and
     measure the normal's turn about the jaws' closing axis (the tool frame's
     y axis, from `robot.get_ee_pose`, taken into the camera frame by the
     camera's pose).
   - **drop**: the region's pixel count against the reference.
3. Report the worst frame and route on it.

## Inputs

`duration_s` (2.0), `period_s` (0.2; 0 compares back to back), `camera`,
`near` / `far` (0.02 / 0.20 m), `slip_mm` (2.0), `twist_deg` (3.0),
`drop_fraction` (0.4), `arm_id`.

## Outputs

`slip_mm`, `twist_deg` (about the closing axis, signed), `spin_deg` (about
the camera axis, signed), `area_fraction`, `frames`.

## The constants, and why

- **`TWIST_DEG = 3`**: over the 0.7-2 deg a station twin's 30-100 g parts
  settle by when lifted (SIM_GRIP.md), and under the ~10 deg per push that
  inserting-mid-air-between-two-held-parts turned its parts.
- **`SLIP_MM = 2`**: about a quarter of a domino's 7 mm thickness. A domino
  lift slid 4.6 mm in the twin.
- **`NEAR` / `FAR`**: a D405 on a YAM sees the closed fingertips at about
  0.07-0.12 m. The table lies beyond 0.2 m while carrying. Lower `far`
  before reading near the table.

## Limits

- **Texture**: the flow needs texture. A plain part (a white domino back, a
  bare cube) gives weak flow, and then the depth plane carries the decision.
  A slip along a flat face of a featureless part can be missed.
- **Fingers**: they sit inside the band too. Their region does not move, so
  they show as outliers in the fit, but if they cover most of the region the
  dominant motion is theirs (zero) and slip is under-read.
- **Not validated in the station twin yet.** The unit tests use synthetic
  frames: an 8 px shift at 0.1 m and f = 200 px reads 4.0 mm, and a 6 deg
  tilt about the closing axis reads 6 deg.
