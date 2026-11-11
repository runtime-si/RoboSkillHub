---
name: measuring-a-held-tip-offset-by-contact-descent
description: Finds how far the tip of a held part (a pen, a peg, a tool,
  a brick) sticks out past the tool centre, with no force sensor - holds
  the hand over a free patch of a surface whose height is known, lowers it
  in 0.5 mm Cartesian steps and stops when the read-back stops following
  the command (the tip has touched), then takes the tip offset along the
  approach axis from the stop height and lifts 10 mm off. Use when a part
  was grasped at an uncertain height (slipped, regrasped, picked from a
  pile) and a later step - writing, inserting, placing on a surface - needs
  its tip position.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [contact, guarded-move, calibration, tool-tip, proprioception, pens]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.1
    tasks: [insertion, pick-place, tool-use]
    provenance:
      authors: [runtime-si]
      upstream_license: original (runtime-si); the stepped lag-thresholded descent follows pressing-in-with-guarded-push
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
  required_inputs:
    surface_z: float
  produces_outputs:
    tip_offset_m: float
    contact_tcp_z: float
    descended_m: float
    steps: int
  exit_conditions:
    measured: The hand stopped following within max_descent_m; tip_offset_m is the tip's distance from the tool centre along the approach axis, and the hand is 10 mm above the touch.
    no_contact: Nothing was touched over max_descent_m (the tip is shorter than assumed, or the surface lower); the hand is at the bottom of the descent.
    unreachable: A descent step was refused.
    invalid_input: The approach axis is more than 60 deg off straight down.
  canonical_scripts:
    - tip_offset: scripts/tip_offset.py
  examples:
    - title: Measure the held part's tip over the table
      path: examples/canonical_subgraph.json
---

# measuring-a-held-tip-offset-by-contact-descent

A YAM has no wrist force sensor, but its position servo lags when the arm is
blocked. Lowered in small steps over a free patch of a known surface, the
hand follows its command until the held part's tip touches, then it stops.
The tool centre's height there, less the surface height, over the approach
axis's downward component, is the tip offset.

## How

1. Read the hand's pose; the approach axis (tool z) must point within 60 deg
   of straight down.
2. Step down `step_m` (0.5 mm) at a time with a 0.2 mm arrival ball, and read
   back the height. The lag (read-back minus command) of the first 6 free
   steps is the baseline; later steps under a third of the threshold join it.
3. A lag `contact_lag_m` (0.6 mm) over the baseline is the touch: the
   offset is (read-back height - `surface_z`) / (downward part of the approach
   axis). Lift 10 mm.

## Inputs

`surface_z` (required, world m), `max_descent_m` (0.10), `step_m` (0.0005),
`contact_lag_m` (0.0006), `arm_id`.

## Limits

- The patch under the tip must be free: anything taller reads as the surface.
- The read-back height is taken at the step where the lag crosses the
  threshold, so the offset can be off by up to about one step plus the
  servo's give at the touch (see the twin numbers).
- A soft or springy tip (a felt pen) compresses before the hand stalls.

## Station twin check (2026-11-10)

Right arm holds a 100 mm, 10 mm peg top-down at 9 grip depths (peg-top-insert twin, SIM_GRIP on), hand over the
peg's empty start spot on the table; truth is the peg's privileged pose against the tool centre.

| grip depth [mm] | true offset [mm] | measured [mm] | error [mm] |
|---|---|---|---|
| 5 | 96.6 | 95.9 | -0.6 |
| 10 | 91.0 | 91.3 | +0.3 |
| 15 | 86.5 | 85.7 | -0.8 |
| 20 | 81.2 | 81.1 | -0.1 |
| 25 | 77.1 | 76.6 | -0.5 |
| 30 | 71.6 | 71.0 | -0.6 |
| 35 | 67.0 | 66.6 | -0.4 |
| 45 | 57.1 | 56.4 | -0.7 |

8 of 8 finished runs `measured` (the 40 mm run hit the harness's 20 min time limit before the descent): error
-0.4 mm mean, 0.8 mm worst, the touch read at 0.8-1.1 mm of lag over a 0.15 mm baseline. The offset reads a
little short (the tip presses the peg up in the jaws by a few tenths before the hand stalls).
