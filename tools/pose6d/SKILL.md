---
name: pose6d
description: 6D pose of a known rigid part from depth — registers a box, cube
  or cylinder primitive (LEGO brick body, domino, cube, peg) or a mesh file to
  the part's point cloud or to its mask in a depth frame, and returns the pose
  with fitness, RMSE, surface coverage, every symmetric equivalent and the
  best non-equivalent alternative. Boxes and cylinders take an analytic,
  visibility-aware point-to-plane ICP over a hypothesis search (numpy/scipy,
  tens to a few hundred ms); meshes take Open3D FPFH + RANSAC global
  registration then point-to-plane ICP. Use when a workflow needs the full
  pose (not just an OBB) of a known-size part from the wrist D405 or the
  overhead camera.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, pose-estimation, registration, icp, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    tasks: [pick-place, insertion, reorientation, sorting]
    version: 0.1.0
    provenance:
      code: https://github.com/isl-org/Open3D/tree/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/examples/python/pipelines
      upstream_license: MIT
gap:
  requires: {}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "pose6d"]
    protocol: stdio-msgpack
  tools:
    - pose6d.register_mesh: 6D pose of a known part (box / cube / cylinder primitive or mesh file) from its point cloud, with fitness, RMSE, coverage and its symmetric equivalents.
    - pose6d.register_mask: 6D pose of a known part from its segmentation mask and a depth frame (camera centre used as the viewpoint); world pose when a camera pose is given.
---

# pose6d

A known part's model and its depth points in, the part's 6D pose out. CPU
only, no weights.

`geometry.compute_obb` gives an upright box around whatever was segmented: its
size is measured (so it shrinks to the visible faces) and its rotation is a yaw.
When the part's size is known — a 2x4 brick, a domino, a 30 mm cube, a peg —
fitting that exact shape gives the full rotation (tilt included), the true
centre rather than the centre of the visible faces, and a residual that says
whether the part is what the caller thinks it is.

## Tools

- **`pose6d.register_mesh(points, mesh, init=None, symmetry=None, viewpoint=None,
  up=None, forward=None, support_plane=None, noise_m=None, method="auto")`** —
  `points` `[N, 3]` metres in any frame (camera or world).
- **`pose6d.register_mask(mask, depth, intrinsics, mesh, camera_pose=None, ...)`** —
  back-projects the mask, uses the camera centre as viewpoint, and returns a
  world pose when `camera_pose` (camera-to-world) is given. Prefer it: the
  viewpoint matters (below).

`mesh` is `{"box": [x, y, z]}` (edge lengths; object frame at the centre, axes
along the edges), `{"cube": s}`, `{"cylinder": [radius, height]}` (axis = z),
a mesh path (`.ply/.stl/.obj` in metres) or `{"path": p, "scale": 0.001}`.

Both return `pose` (object to frame, `Se3Pose`) and `matrix` (4x4), `fitness`
(share of points within `inlier_threshold_m` of the visible model surface),
`rmse` (of those inliers), `coverage` (share of the model's camera-facing area,
weighted by how squarely it faces the camera, with a point near it),
`symmetry_order`, `yaw_ambiguity_deg`, `equivalent_poses`, `ambiguous_fit`,
`alternative`, `status`, `runtime_s`.

`status`: `registered` (fitness >= 0.6, coverage >= 0.4 and no close
non-equivalent alternative), `ambiguous_fit` (a pose that is *not* a symmetric
twin scores within 0.03 — e.g. a brick on its side vs flat from a coarse view;
see `alternative`) or `poor_fit` (wrong size, wrong part, too few points).

## How the primitive fast path works

1. PCA normals (16 neighbours) oriented to the viewpoint.
2. Up to four frame estimates: the two dominant normal clusters; the dominant
   cluster plus the minimum-area rectangle of its face; the cloud's principal
   axes; `up` (or the view axis) plus the minimum-area rectangle of the whole
   cloud. Each is crossed with every way the box's edges can lie on its axes,
   one per symmetry coset (6 for a brick, 3 for a square plate, 1 for a cube;
   three axis choices for a cylinder plus a normal-scatter axis).
3. Every hypothesis gets 12 ICP iterations on 600 points; the best four
   distinct ones get 40 on all points (up to 3000) with the inlier threshold
   re-estimated from the leader's residuals (3 sigma, 1-6 mm).
4. ICP is Gauss-Newton on the exact surface: each point is matched to the
   closest point on a face (or cylinder patch) that faces the viewpoint, with a
   point-to-plane row plus a tangential row when it lies past the face's edge
   (point-to-plane alone lets a single seen face slide). Huber weights;
   residuals past 2x the threshold are dropped as background.
5. Hypotheses are ranked by the harmonic mean of fitness and coverage minus a
   truncated-quadratic residual cost. Coverage is what stops a box being fitted
   into a bigger face than the one observed.

**Symmetry is resolved explicitly.** A box with three different edges looks
the same after 180 deg about any axis (order 4: a brick or domino flat on the
table has a 180 deg yaw ambiguity); two equal edges give 8, a cube 24; a
cylinder spins freely and flips end for end. Depth cannot tell these apart, so
the tool returns one canonical pose — with `init`, the equivalent nearest it;
otherwise z most along `up` (default: toward the camera), then x most along
`forward` (default the frame's +x) — plus `equivalent_poses` (all of them) and
`yaw_ambiguity_deg` (180 for a brick, 90 for a cube or 2x2, 360 for a peg).
A caller that can tell the ends apart (domino pips, a printed mark, stud
colour) picks among `equivalent_poses` or passes `init`. `symmetry` only changes
which equivalents are listed (`"none"`, `"z2"`, `"z4"`, `"box"`, `"cylinder"`,
or a list of object-frame rotations); the fit always uses the shape's geometry.

## Viewpoint, up and the support plane

- Only camera-facing surfaces are matched, so the tool must know where the
  camera was: `viewpoint` defaults to the origin (right for camera-frame
  points). For world points pass the camera position, or call `register_mask`.
  Without it, a part seen only from the top could be placed under its points.
- `support_plane` `[nx, ny, nz, d]` (`n . x + d = 0`, `n` up, e.g.
  `[0, 0, 1, -table_z]`) states that the part rests on it in a stable pose
  (a box on a face, a cylinder standing or lying). Points within 3 mm of it
  are dropped (segmenter bleed) and the resting face is held on it. **Use it
  with the overhead camera**: from ~0.9 m a 2 cm part has 100-400 points at
  ~2.5 mm noise and a free fit tilts by 5-25 deg (table points bleeding in at
  the mask edge pull it over); with the plane it is 1-3 deg. Do not pass it for
  a part that leans, is stacked or is held.

## Measured

Synthetic (`eval/synthetic.py --trials 20`): ray-cast renders at known poses
with the station's depth noise — overhead Orbbec at ~0.9 m (2.9 mm std at
1 m, growing with depth^2, 1 mm steps), wrist D405 at 0.15-0.22 m oblique
(0.5 mm at 0.15 m, 0.1 mm steps) — with the object mask dilated by one pixel so
table returns bleed in. Rotation error is over the part's symmetries.
`fast` = primitive path; `support` = with the table as support plane;
`global` = Open3D FPFH + RANSAC + ICP on the same cloud. Each cell: rotation
error median (p90) / translation error median (p90) / share of 20 trials with
rotation < 5 deg and translation < 3 mm. Brick = 2x4 body 31.8 x 15.8 x 9.6 mm,
domino 48 x 24 x 7.5 mm, peg r 5 x 40 mm; "free" = random 3D rotation held
5 cm above the table.

| camera, pose | part | pts | fast: deg / mm / ok | support: deg / mm / ok | global: deg / mm / ok | ms fast / support / global |
|---|---|---|---|---|---|---|
| overhead, resting | brick_2x4 | 197 | 23.4 (34.4) / 2.0 (2.2) / 5% | 2.4 (5.1) / 0.6 (1.0) / 85% | 10.3 (48.4) / 0.8 (4.1) / 25% | 94 / 72 / 20 |
| overhead, resting | cube_30 | 404 | 1.5 (2.6) / 0.4 (0.8) / 100% | 0.7 (1.0) / 0.2 (0.4) / 100% | 2.1 (4.4) / 0.2 (0.4) / 100% | 34 / 30 / 65 |
| overhead, resting | domino_flat | 383 | 8.3 (9.5) / 1.5 (1.7) / 15% | 1.2 (2.4) / 0.7 (1.1) / 100% | 5.2 (7.4) / 0.5 (1.0) / 50% | 95 / 73 / 25 |
| overhead, resting | domino_upright | 196 | 2.0 (5.2) / 0.6 (1.1) / 70% | 1.3 (3.2) / 0.3 (0.5) / 100% | 2.8 (84.5) / 1.4 (16.4) / 55% | 78 / 77 / 20 |
| overhead, resting | peg_lying | 156 | 3.1 (5.2) / 2.5 (3.7) / 65% | 1.2 (2.6) / 1.0 (3.0) / 90% | 2.0 (3.5) / 1.0 (7.0) / 70% | 45 / 41 / 14 |
| overhead, resting | peg_standing | 88 | 1.2 (1.9) / 0.4 (0.8) / 100% | 1.3 (2.3) / 0.2 (0.4) / 100% | 1.6 (10.4) / 1.7 (13.9) / 70% | 48 / 47 / 13 |
| wrist, resting | brick_2x4 | 3127 | 0.4 (0.9) / 0.1 (0.3) / 100% | 0.2 (0.4) / 0.1 (0.4) / 100% | 0.6 (1.2) / 0.1 (3.7) / 85% | 248 / 248 / 34 |
| wrist, resting | cube_30 | 8435 | 0.2 (0.3) / 0.0 (0.5) / 100% | 0.0 (0.1) / 0.0 (0.3) / 100% | 0.2 (0.3) / 0.0 (0.8) / 95% | 127 / 125 / 66 |
| wrist, resting | domino_flat | 6408 | 0.3 (0.5) / 0.1 (0.3) / 100% | 0.1 (0.2) / 0.2 (0.4) / 100% | 0.4 (0.5) / 0.1 (0.8) / 95% | 219 / 182 / 27 |
| wrist, resting | domino_upright | 3727 | 0.1 (0.7) / 0.0 (0.2) / 100% | 0.1 (0.3) / 0.0 (0.1) / 100% | 0.2 (8.2) / 0.0 (5.2) / 85% | 250 / 239 / 26 |
| wrist, resting | peg_lying | 2401 | 0.1 (0.2) / 0.1 (0.2) / 100% | 0.1 (0.2) / 0.1 (0.4) / 100% | 0.1 (0.3) / 0.1 (0.1) / 100% | 110 / 94 / 22 |
| wrist, resting | peg_standing | 1752 | 0.1 (0.2) / 0.0 (0.1) / 100% | 0.1 (0.2) / 0.0 (0.0) / 100% | 0.1 (0.2) / 0.0 (0.1) / 95% | 134 / 104 / 19 |
| wrist, free | brick_2x4 | 2852 | 0.3 (1.1) / 0.0 (0.2) / 100% | — | 0.4 (89.1) / 0.1 (4.7) / 65% | 226 / — / 31 |
| wrist, free | cube_30 | 7704 | 0.2 (0.3) / 0.0 (0.4) / 100% | — | 0.2 (0.3) / 0.1 (0.6) / 100% | 54 / — / 34 |
| wrist, free | domino_flat | 5995 | 0.2 (0.5) / 0.1 (0.2) / 100% | — | 0.2 (9.8) / 0.1 (5.9) / 80% | 275 / — / 26 |
| wrist, free | peg_lying | 2267 | 0.1 (0.2) / 0.1 (0.5) / 100% | — | 0.1 (0.2) / 0.1 (0.6) / 100% | 152 / — / 22 |

Read-outs: from the wrist the fast path is within 5 deg / 3 mm on every trial
(medians 0.1-0.4 deg, < 0.15 mm); the global path matches it at the median but
its p90 shows symmetric flips and slides (89 deg on free bricks): FPFH cannot
rank a box's faces. From overhead the free fit of a part under ~10 mm tall is
pulled over by table returns at the mask edge (brick 23 deg, domino 8 deg);
the support plane brings that to 1-2.5 deg and < 1 mm. The fast path costs
50-280 ms (up to 24 hypotheses on up to 3000 points); `init` skips the
search.

Real frame (`eval/real_frame.py`): the station's overhead camera
(`real/snapshots/20261031-003520/obs.npz`, cube at ~0.55 m), dark-pixel mask,
table plane by RANSAC around it, `{"cube": 0.03}`:

| | fast | fast + support | global |
|---|---|---|---|
| points | 944 | 927 (17 within 3 mm of the table dropped) | 944 |
| status | registered | registered | registered |
| fitness / coverage | 0.975 / 0.83 | 0.977 / 0.80 | 0.79 / 0.74 |
| inlier RMSE | 1.36 mm | 1.58 mm | 1.16 mm |
| residual median / p90 (all points) | 1.05 / 2.28 mm | 1.21 / 2.80 mm | — |
| tilt of the cube's up axis vs the table normal | 8.2 deg | 1.0 deg | 11.2 deg |
| centre height over the table (true 15 mm) | 13.3 mm | 14.4 mm | 14.0 mm |
| bottom face to table plane | -1.5 mm | -0.6 mm | — |
| runtime | 46 ms | 41 ms | 417 ms |

The measured top surface is 29.8 mm above the table plane (95th percentile),
agreeing with the 30 mm cube. The mask is a grey-level threshold and takes in
part of the cube's shadow on the table, the likely cause of the free fits'
tilt; with the support plane the fitted box sits on the table to 0.6 mm and
level to 1 deg. A cube-size sweep (24-36 mm, with support) ranks 28 and 30 mm about
equally (RMSE 1.32 vs 1.58 mm, coverage 0.87 vs 0.80) and rejects 34-36 mm
(coverage 0.32, 0.19): at ~1 mm depth noise the size check is good to ~2 mm.

## Gotchas

- A LEGO brick is a box only up to its studs (1.8 mm tall on top, tubes
  underneath). Fit the body (`[0.0318, 0.0158, 0.0096]` for a 2x4) and expect
  the top-face residual to carry the studs; stud-level alignment for
  press-fitting needs a stud-lattice read on top of this pose.
- Size must be right to ~1 mm: a 15 mm cube model on a 30 mm cube is a
  `poor_fit`, but 28 vs 30 mm still registers (the overhead real-frame sweep
  could not separate 28-30 mm).
- `rmse` is over inliers (Open3D's convention); read it with `fitness`.
- The mesh path needs `open3d` (in this bundle's environment) and enough
  geometric detail for FPFH: on overhead clouds of a few hundred points
  Open3D finds only 3-10 mutual feature matches, and on symmetric parts its
  p90 error is a flip (table above). It does not use `support_plane`.
- No LDraw reader: export a brick to `.stl`/`.ply` first (LDraw parts are
  CC BY 2.0) or use the box primitive.
