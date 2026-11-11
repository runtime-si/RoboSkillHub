---
name: aligning-a-held-tip-over-a-hole-by-visual-servoing
description: Puts the tip (foot) of a held part - a pen, a peg, a rod - over
  a hole before it is lowered in, from the wrist RGB-D frame - fits a line
  to the held part's depth points around the tool's approach axis, moves it
  back by the part's radius and takes where it crosses the face as the foot;
  finds the hole as the known-radius circle that keeps the face's points out
  and holds the deeper points, near its expected spot; moves the hand by the
  lateral offset in capped steps, checks from the read-back that the hand
  really moved, and repeats until under 1 mm. Use before inserting a held
  part into a hole when the grasp left it a few millimetres off or tilted
  and the foot lands on the face beside the hole.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, visual-servoing, wrist-camera, depth, pen, peg-in-hole]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, pick-place]
    provenance:
      authors: [runtime-si]
      upstream_license: original (runtime-si)
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
  required_inputs:
    hole_xy: Vec3
    face_z: float
    tip_offset_m: float
    part_diameter_m: float
    hole_diameter_m: float
  produces_outputs:
    offset_x: float
    offset_y: float
    iterations: int
    moved_m: float
  exit_conditions:
    aligned: The foot read within tol_m (1 mm) of the hole's centre.
    not_converging: Still over tol_m after max_iterations, a reading over 30 mm off (a misreading, not moved), or the hand made under 40 % of a commanded correction (read back).
    tip_not_seen: No wrist frame with depth, intrinsics and pose, or too few points along the tool axis.
    hole_not_seen: Too few face or deeper points around the expected hole.
  canonical_scripts:
    - align_tip: scripts/align_tip.py
  examples:
    - title: Align a held pen over its hole
      path: examples/canonical_subgraph.json
---

# aligning-a-held-tip-over-a-hole-by-visual-servoing

## How

1. Wrist frame to a base-frame point cloud (depth, intrinsics, camera pose).
2. **Foot**: points within the part's radius + 6 mm of the tool's approach
   axis, from the tool centre to `tip_offset_m` (e.g. from
   measuring-a-held-tip-offset-by-contact-descent) and above the face. A line
   fit (SVD), moved a radius away from the camera (it sees the near side), is
   the axis; where it crosses `face_z` is the foot.
3. **Hole**: within the hole's radius + 20 mm of `hole_xy`, face points (within
   3 mm of `face_z`) and deep points (5 mm under it). Start at the deep
   points' centroid; push the centre until no face point lies inside the
   hole's radius.
4. Move the hand by the offset, at most `max_step_m` (6 mm); read the hand
   back. Under 40 % of the step made is `not_converging` (the hand is not
   following small corrections). Repeat up to `max_iterations` (6).

Then lower the part with a guarded descent (pressing-in-with-guarded-push) and
check by depth that the foot went below the face.

## Real pens run (probe-insert-1, offline)

- **The hand did not follow the corrections.** The pens task's own align loop
  read the foot 10.2 mm off and asked for (+7.8, -6.5) mm; the wrist camera's
  pose (hence the tool) moved (-0.2, -1.4) mm. The next ask, (+6.5, -7.1) mm,
  moved it (-0.2, +0.6) mm; only the third moved (+4.2, -3.4) mm. This skill
  checks that from the read-back and stops (`not_converging`) instead of
  re-reading the same offset.
- On the 25 in-hole frames of repeat 1 this skill read the foot (+0.0 to +4.7,
  -1.5 to -8.5) mm from the hole, the pens task (+4.2 to +10.4, -3.7 to
  -15.2) mm: the same direction, about half the size; the hole centres it
  found sit 3-7 mm from the slot's prior, the pens task's lattice shift under
  2 mm. Which is right needs the top camera (not done).

## Limits

- Not run in the station twin yet.
- The camera sees one side of the part: a tilted part or a short visible
  length biases the axis.
