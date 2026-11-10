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
    version: 0.2.0
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
`drop_fraction` (0.4), `jaw_width_m` (0: off), `arm_id`.

With `jaw_width_m` (the jaw opening, e.g. the part's width across the jaws) and a camera pose in the observation,
the region is the jaw box instead of the band: depth points in the tool frame with |x| (the closing axis) inside
the opening less 1.5 mm, |y| under 40 mm and under 15 mm past the tool centre. Slip, twist and spin then come from
a rigid 3D fit (Kabsch, trimmed) to the flow's point matches. Use it: the plain band is not reliable (below).

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
- **The closing axis is the tool's x** (robot.get_ee_pose frame): a held domino's thickness axis lies within
  0.04 of it in the station twin. Version 0.1.0 used y.

## Station twin check (2026-11-10, privileged truth, wrist D405 at the real fit)

Domino lift (refilling-a-domino-loading-rack, 2 racks), first 3 s of each hold; DUPLO and peg mid-air insertion
(inserting-mid-air-between-two-held-parts) as stable holds before the parts meet:

| case | truth | band (0.1.0) | jaw box + rigid fit (`jaw_width_m`) |
|---|---|---|---|
| domino, full speed (4 lifts) | 3.0-3.4 mm slip | slip 0.0 mm; fired on a false 33 deg twist | slip 2.4-3.7 mm, 4/4 fire (3/4 on slip > 2 mm) |
| domino, held_speed_scale 0.25 (6 lifts) | 0.6-1.3 mm | 6/6 false alarms (23-32 deg twist) | slip 0.6-2.6 mm: 2/6 false alarms |
| mid-air, held before contact (264 frames) | 0.9-1.8 mm, < 1.7 deg | 97-134 deg twist, area < 0.4 ("dropped") | max 0.8 mm, 0.6 deg: 0 false alarms |

The band took in the fingers (the flow fit locked on their zero motion) and the table (its plane read as 100+
deg of twist when the part left it). Still open: once the mid-air parts meet, the other part enters the jaw
box and the fit reads false twist (up to 90 deg); twists about the approach axis (8-13 deg on a domino) read
3-8 deg; the jigsaw turn-fling case (#191) was not recorded.
