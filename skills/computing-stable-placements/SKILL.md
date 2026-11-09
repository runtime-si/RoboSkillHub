---
name: computing-stable-placements
description: Enumerates the stable resting faces of a rigid object - from its
  oriented bounding box or a perceived point cloud closed down to the table -
  by convex hull and centre-of-mass projection, and reports for each rest the
  tipping angle, the centre-of-mass height, the turn from the object's current
  orientation that puts that face down, its footprint, and (given a goal point)
  the resting pose; also says which rest the object is on now, or that it is on
  none (leaning, tilted, held). Routes placements or no_stable_pose. Use when
  a graph must choose how to set a part down (a brick, a domino on its end or
  flat, a jigsaw piece), decide whether a part lying on the table needs a
  regrasp or flip, or reject a placement too easy to tip; geometry only, no
  motion.
license: BSD-2-Clause
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [placement, stability, regrasp, reorientation, convex-hull, lego, domino, jigsaw]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, reorientation, lego, domino, jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/personalrobotics/tsr/blob/5279ca7baf95e6718d4628a3a26fd6ebeefedf0d/src/tsr/placement/_stable_poses.py
      upstream_license: BSD-2-Clause
gap:
  allowed_tools: []
  required_inputs: {}
  # Optional inputs (exactly one of object_obb / points; wired from a script node
  # when gap has no type for them): object_obb (OrientedBoundingBox, world, half
  # extents), points (PointCloud or N x 3, world), com (Vec3 or [x, y, z], world),
  # support_z (table height for closing a top-down cloud), closing (prism | drop |
  # none), min_margin_deg, surface_half_extents [hx, hy], goal_position (Vec3 on the
  # support), current_tol_deg.
  produces_outputs:
    n_rests: int
    best_tipping_deg: float
    best_turn_deg: float
    best_com_height_m: float
    best_pose: Se3Pose
    current_index: int
    current_stable: bool
    com: Vec3
    reason: string
  exit_conditions:
    placements: At least one hull face holds the centre of mass inside its support polygon with a tipping angle of at least min_margin_deg and fits the surface at every yaw; `rests` lists them most stable first, `best` is the first.
    no_stable_pose: No face holds the centre of mass (or none clears min_margin_deg, or none fits surface_half_extents); `reason` counts which filter removed them.
  canonical_scripts:
    - compute_rests: scripts/compute_rests.py
  examples:
    - title: Stable rests of a perceived box-like part
      path: examples/canonical_subgraph.json
---

# computing-stable-placements

Which faces can this part rest on, how firmly, and how far must it turn to get
there?

```text
perceive (OBB or cloud) -> rests (this skill) -> choose a rest -> place / flip / regrasp
```

## What it computes

TSR's `stable_poses_mesh`, unchanged in substance: the convex hull of the
object, its facets grouped into faces by coplanarity (a 4D grid over normal
and offset, matched against each group's first plane), and for each face the
centre of mass projected onto it. A face is a rest when the projection lies
inside the face's support polygon (the 2D hull of its contact points) by more
than the length tolerance; the **tipping angle** is
`atan(support_margin / com_height)`: how far the object must be tilted about
the nearest support edge before the COM passes over it. Rests are sorted most
stable first (upstream `place_mesh`).

Each rest in `rests`:

| field | meaning |
|---|---|
| `label` | `+x` ... `-z` (the OBB's local face) or `face-k` for a cloud |
| `tipping_deg`, `support_margin_m`, `com_height_m` | stability, as above |
| `center_height_m` | height of the reference point (OBB centre, or the COM for a cloud) over the support when resting |
| `turn_deg`, `rotation_to_rest` | the minimal world rotation from the current orientation that puts this face down (yaw is then free) |
| `face_normal_world` | the face's outward normal now |
| `footprint_radius_m` | farthest horizontal reach from the COM at rest (what a free yaw sweeps) |
| `pose` | with `goal_position`: the reference point's world pose resting there (`rotation` = `rotation_to_rest` x current) |

`current_index` is the rest whose face points within `current_tol_deg` (10)
of straight down, or -1: the object is not resting on a stable face (tilted,
leaning on something, or held).

## Inputs

- `object_obb` (world, half-extents): the exact path, a box with the COM at
  its centre unless `com` is given. Use it for bricks, dominoes, holders,
  jigsaw pieces.
- `points` (world cloud) with `support_z`: a top-down camera sees no
  underside, so the cloud is closed down to the support:
  - `closing: prism` (default): footprint of the points in the upper half of
    the object's height (mixed depth pixels at the edges fall below it),
    simplified (3 mm Douglas-Peucker) and extruded from the support to the
    top (p98). Exact geometry, so upstream's tight tolerances apply.
  - `closing: drop`: every point plus its drop onto the support; the hull of
    the raw cloud, with faces merged within 0.05 rad and 2 mm.
  - `closing: none` / no `support_z`: the cloud as given (a registered mesh's
    vertices, a multi-view scan).
- `com`: defaults to the OBB centre or the hull's volume centroid (uniform
  density). Pass it for top-heavy or compound parts.
- `min_margin_deg` (0, upstream's default): drop rests that tip more easily.
  5 degrees removes the knife-edge rests a noisy footprint produces.
- `surface_half_extents` [hx, hy]: drop rests whose footprint does not fit at
  every yaw (upstream `_fits`, about the COM here).

## Checks

Exact geometry (unit tests): a domino 48 x 24 x 7.5 mm and a LEGO 2x4 brick
31.8 x 15.8 x 11.4 mm give six rests each with tipping angles equal to the
analytic `atan(min half-side / half-height)` to 0.01 deg (domino: flat 72.65,
on its long edge 17.35, on its end 8.88; brick: 54.19 / 35.81 / 19.72). A COM
moved 10 mm toward one end of the domino gives 15.0 and 6.3 deg on the two
ends (`atan(3.75/14)`, `atan(3.75/34)`). A box cloud tilted 30 deg reports six
rests with turns 30.6 / 60 / 90 / 120 / 150 deg and no current rest.

Real station (overhead D405 frame `top_before.npz` of a gsota probe run,
points inside each holder's measured footprint + 6 mm, over the table), against
the holders' wrist-measured boxes:

| grey holder (wrist: 76.2 x 100.4 x 24.5 mm) | rests | flat rest: tipping, COM height | current |
|---|---|---|---|
| measured box, OBB path | 6 | 72.2 deg, 12.2 mm | flat |
| OBB fitted to the cloud (minAreaRect of the upper points, p98 top: 73.0 x 97.0 x 23.7 mm) | 6 | 72.0 deg, 11.8 mm; sides 18.0 / 13.7 deg vs 17.8 / 13.7 | flat |
| cloud, `prism` (7-gon 73.0 x 99.5 x 23.7 mm) | 7 | 68.8 deg, 11.8 mm | flat |
| cloud, `drop` (raw hull) | 3 | 76.1 deg, 9.3 mm | flat |

- **For box-like parts fit an OBB first.** The fitted OBB matched every rest
  of the measured box within 0.2 deg and 1.7 mm. The prism keeps the flat rest
  and the current one right, but its footprint polygon has cut corners (the
  holder's footprint fills 78 % of its rectangle), so one short side splits
  into two bent segments: rests at 4.0 deg and the side angles off by up to
  3 deg. The raw `drop` hull is a rounded frustum (mixed pixels widen its base
  to 88 x 112 mm): 3 rests of 6, COM 3 mm low.
- **Uniform density is an assumption.** The black holder with 12 markers
  standing in it (cloud top 126 mm) becomes a solid 68 x 91 x 124 mm prism
  whose most stable rest is on its side (34 deg vs 25.6 deg upright); the real
  object's mass is in the holder. Pass `com` for such parts.
- No arm motion, so no simulation check; nothing was placed to confirm a rest
  physically.

## Relation to other skills

- standing-a-lying-domino-by-reorient-and-regrasp, tipping-over-a-surface-edge:
  the motions that change the rest; this skill says which rest to aim for and
  whether the part is on one now.
- placing-with-contact-descent / placing-into-container-obb: use
  `best_pose` / `center_height_m` for the release height.

## Changes from TSR

Adapted from `src/tsr/placement/_stable_poses.py` and the `place_mesh` /
`_fits` logic of `stable_placer.py` (BSD-2-Clause, licence in
`LICENSE.tsr`):

- Input is a world OBB or a perceived cloud instead of a mesh in its own frame;
  the cloud closings (`prism`, `drop`) and the cloud tolerances are new.
- COM defaults to the hull's volume centroid.
- Output is per-rest world numbers (turn, normal now, pose at a goal) instead of
  TSR templates with sampling bounds; the current rest is identified.
- The surface fit is measured about the COM, from given half-extents.
- The analytic primitives (cylinder, sphere, torus, compound shapes) are not
  ported; a cylinder lying on its side is not a rest in upstream either.
