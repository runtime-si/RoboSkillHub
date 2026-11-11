---
name: searching-insertion-with-grid-and-plug-flip
description: Finds the hole under a held plug or peg whose position is only
  known to a few millimetres, with no force sensor - lowers it slowly until
  it touches the base (the hand stops following), then probes a small x/y
  grid at that height, nearest the start first, one short guarded descent per
  cell; a probe that sinks past the base is in; when the grid runs out (or a
  probe sinks only part way, an upside-down USB-style plug), turns the plug
  180 deg about the approach and searches again. Use when a held part must go
  into a port or hole whose camera estimate is off by more than the
  clearance, and the part may be keyed one way round.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, search, usb, connector, peg-in-hole, contact, proprioception]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, assembly]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nolann030628/ShapeForce/blob/main/third_party/utils_repo/peg_insertion/force_insert.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
  produces_outputs:
    base_z: float
    depth: float
    offset_x: float
    offset_y: float
    probes: int
    flipped: bool
  exit_conditions:
    inserted: A probe sank in_depth_m (4 mm) past the base; the hand is there, still holding (offset_x/y is the cell from the start).
    flipped_and_inserted: As inserted, after the plug was turned 180 deg about the approach.
    grid_exhausted: No probe sank far enough (both ways round when allow_flip); the hand is 7 mm over the base at the start.
    jammed: Nothing was touched within max_base_descent_m under the start.
  canonical_scripts:
    - grid_search: scripts/grid_search.py
  examples:
    - title: Search a 7 x 7 grid at 1.5 mm, flip once
      path: examples/canonical_subgraph.json
---

# searching-insertion-with-grid-and-plug-flip

Reimplements ShapeForce's USB grid search with the YAM's tracking lag in
place of the soft wrist's force reading.

## How

1. From the start (the plug over the port's estimate), step down 0.5 mm at a
   time with a 0.2 mm arrival ball. A read-back lag 0.6 mm over the free
   baseline is the touch: that height is the base.
2. For every cell of a grid within `grid_half_m` (4.5 mm) at `grid_step_m`
   (1.5 mm), nearest the start first: go 2 mm over the base there and step
   down at most `probe_depth_m` (8 mm) past it, stopping at a touch.
   - Sank `in_depth_m` (4 mm) past the base: in. Stop there.
   - `partial_depth_m` > 0 and sank that far but not in: the plug is upside
     down; stop this pass.
3. With `allow_flip`, a pass that ends without a hole turns the hand 180 deg
   about its approach axis and searches the grid again.

## Inputs

`grid_half_m` (0.0045), `grid_step_m` (0.0015), `in_depth_m` (0.004),
`partial_depth_m` (0: off), `probe_depth_m` (0.008), `max_base_descent_m`
(0.03), `allow_flip` (true), `arm_id`.

## Changes from upstream

- Contact from the hand's lag in 0.5 mm steps, not a soft-wrist Fz threshold.
- The grid is probed centre-out instead of from a corner, since the estimate
  is likeliest right.
- Depths and grid sizes are inputs, not hard-coded millimetres.

## Limits

- Each probe presses the plug onto the base: a flimsy plug or port takes the
  press (about the servo's force at 0.6 mm of lag).
- A grid step over the clearance can step over the hole: keep the step under
  twice the clearance a side.
