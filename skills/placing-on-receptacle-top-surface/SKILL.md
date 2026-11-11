---
name: placing-on-receptacle-top-surface
description: Finds where to set a held object down on a receptacle's top surface (a box
  lid, a tray, a block, a shelf) - OK-Robot's place heuristic. It takes the receptacle's
  points (segmented by text with sam3 and lifted with the depth, or given), puts the drop
  point at the median of its horizontal footprint over unique 1 cm cells, reads the surface
  height as the 95th percentile of the receptacle points within 3 cm of that point, and
  returns a release pose 2 cm over it (plus how far the held object's bottom hangs under
  the tool). No motion. Use before placing an object onto another object's top; chain into
  placing-with-contact-descent to set it down.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [place, receptacle, surface, point-cloud, plan]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/ok-robot/ok-robot/blob/174c742b6a1866a2da8f75a89e2c544bb77ab5f9/ok-robot-manipulation/src/anygrasp_manipulation/object_handler.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2401.12202
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs: {}
  # Inputs: query (text naming the receptacle) or points (world points, wired from a script node),
  # held_bottom_offset (m the held object's bottom hangs under the tool centre), drop_height, band,
  # camera_name, rotation (the release pose's rotation; default identity - pass the hand's).
  produces_outputs:
    place_pose: Se3Pose
    surface_z: float
    footprint_center: Vec3
    n_points: int
  exit_conditions:
    place_point: The receptacle was found with at least 30 points and 5 of them within `band` of its footprint's median; `place_pose` is over that point at the surface height + `drop_height` + `held_bottom_offset`.
    receptacle_not_found: No camera frame, no mask for the query, fewer than 30 points, or no surface round the footprint's median (an open ring, a box's hollow - use placing-into-container-obb for containers).
    invalid_input: Neither a query nor points were given (also the route of a raised error).
  canonical_scripts:
    - place_point: scripts/place_point.py
  examples:
    - title: Find the drop point on a receptacle's top
      path: examples/canonical_subgraph.json
---

# placing-on-receptacle-top-surface

OK-Robot's place step, for a tabletop arm.

## What it does

1. Receptacle points: `sam3.segment_text` on the first camera with depth (or `camera_name`), the best mask
   lifted by `geometry.mask_to_world_points`; or `points` given.
2. Drop xy: the median of the unique 1 cm cells of the footprint (OK-Robot's `np.unique` over cm-rounded xz),
   so a side seen densely does not pull the point toward it.
3. Surface height: the 95th percentile of the points within 3 cm of that xy (OK-Robot: the maximum in a 10 cm
   strip).
4. Release: `drop_height` (2 cm) + `held_bottom_offset` over the surface.

## Measured

Unit tests on synthetic point clouds (`tests/bundles/test_placing_on_receptacle_top_surface.py`): a box top,
a box seen with its near side dense, a tray with a rim, a ring (no surface: not found), noisy depth, the
camera path with fake tools. **No twin or live-frame check was run** (round time limit).

## Not done

- No check on rendered or real frames; the mask quality is sam3's.
- The release rotation is the caller's (identity by default).
- Containers (open boxes) are placing-into-container-obb's job; this skill reports their hollow as no surface.

## Provenance

Adapted from OK-Robot (MIT, commit 174c742): `ObjectHandler.place`. The license is `LICENSE.ok-robot` beside
this file.
