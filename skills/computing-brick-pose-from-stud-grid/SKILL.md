---
name: computing-brick-pose-from-stud-grid
description: Converts between a cell on a LEGO or DUPLO baseplate's stud grid
  (x, y, layer, brick size such as 2x4, rotation 0/90/180/270) and metric world
  poses - ICL's calc_brick_loc chain (plate pose * top-left offset * cell
  offset * yaw * brick-centre offset) - giving the brick's body-centre pose,
  the seat pressing-a-brick-onto-studs takes, a top-down parallel-jaw grasp
  across the short side with its approach (press hover), and pose6d's box and
  support plane; and back, from a world pose (e.g. pose6d's) to the nearest
  cell with the residual. Checks the plate's bounds and, given occupied cells,
  collisions, support from the layer below and room for the fingers. Use when
  a build step names a cell on a calibrated baseplate and the arm needs where
  to grasp or press, or when a perceived brick must be read as a grid cell.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, geometry, grid]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/Lego.cpp
      upstream_license: MIT
      paper: arXiv:2309.02354
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper]}
  allowed_tools:
    - robot.grasp_frame
    - robot.describe_gripper
    - robot.solve_ik
  required_inputs:
    plate_pose: Se3Pose
  produces_outputs:
    brick_pose: Se3Pose
    top_pose: Se3Pose
    seat_position: Vec3
    seat_yaw_deg: float
    system: string
    stud_pitch: float
    stud_height: float
    stud_diameter: float
    brick_height: float
    grasp_pose: Se3Pose
    grasp_pose_flipped: Se3Pose
    approach_pose: Se3Pose
    press_pose: Se3Pose
    close_heading_deg: float
    close_axis: string
    jaw_width: float
  exit_conditions:
    computed: The cell is on the plate (and, given occupied cells, free, supported, with room for the fingers; the jaws open wide enough where the gripper is described; the grasp passes the IK preview when check_reach). For a pose input, it lies within the tolerances of its nearest cell.
    off_grid: Pose input only. The nearest cell is returned, but the pose is more than lateral_tolerance (a quarter pitch) from it along a plate axis, more than a quarter brick height from a layer, yawed more than 10 deg off the rows or tilted more than 10 deg; residual says by how much.
    out_of_plate: Part of the footprint lies off the plate's stud grid, or the layer is under 1. Poses are still filled in.
    collides: One of the brick's cells is in occupied; collisions lists them.
    unsupported: Occupied cells were given, the layer is 2 or more, and nothing occupies the layer under any of its cells.
    ungraspable: The finger rows beside the brick (along the closing axis, at its layer and above) are occupied on both sides it could be grasped across, or the described jaws open less than the brick's width + jaw_margin.
    unreachable: check_reach, and neither wrist solution of the grasp and approach passes robot.solve_ik.
    invalid: Malformed inputs (unknown system, rotation not a multiple of 90, neither or both of cell / observed_pose, a grasp_depth outside the brick).
  canonical_scripts:
    - brick_pose: scripts/brick_pose.py
  examples:
    - title: Poses for a DUPLO 2x4 at a cell of a 16x16 baseplate
      path: examples/canonical_subgraph.json
---

# computing-brick-pose-from-stud-grid

Pure geometry, no motion: a build step's grid cell becomes the poses the arm
needs, and a perceived brick becomes a grid cell.

## How it composes

- **Before it**: the baseplate's pose (a calibrated plate, e.g. read by
  registering the plate or its corner studs). For the inverse, a brick pose
  from `pose6d.register_mask` fitted with this skill's `pose6d_mesh` (pass
  `support_plane` and `brick_pose` as `init` when checking an expected cell).
- **After it**: `approach_pose` / `grasp_pose` to pick a brick at a cell;
  `seat_position`, `seat_yaw_deg`, `brick_studs`, `system`, `stud_pitch`,
  `stud_height`, `brick_height` straight into pressing-a-brick-onto-studs
  (its inputs of the same names; give it `held_obb` from the hand as well);
  `cells` appended to the `occupied` list for the next step.

## Conventions

- **Plate frame** (`plate_pose`): origin at the centre of the plate's top
  surface (the stud bases), x along its first stud count, z up out of the
  studs; ICL's plate pose. `plate_studs: [nx, ny]`. If `plate_pose` is the
  plate's underside (e.g. the table under it), give `plate_thickness` (a
  standard LEGO plate is 3.2 mm).
- **Cell** `{x, y, layer, rotation}`: (x, y) is the footprint's lowest-index
  corner cell at every rotation (ICL's anchor), 0-based; `layer` 1 is a brick
  standing on the plate (ICL's z); rotation is the brick's yaw about the plate
  normal, 90 = ICL's `ori 1`.
- **Brick** `brick_size` `"2x4"` or `[2, 4]`: studs along the brick's own x
  and y (ICL's `height` x `width`), so a 2x4 at rotation 0 covers 2 cells
  along the plate's x and 4 along y, and 4 x 2 at 90.
- **Poses**: `brick_pose` the body centre (no studs) with the brick's axes -
  pose6d's box frame; `top_pose` the top centre at the stud bases (ICL's brick
  pose); `seat_position` the bottom centre, which is the press skill's seat
  (z = the stud bases of whatever it stands on). `seat_yaw_deg` is the brick
  x axis' heading.

## The grasp

Top-down at the brick's centre, jaws closing across the short side
(`close_axis: auto` falls back to the long side if the cells beside the short
side's walls are occupied; `short` / `long` force one), fingertips
`grasp_depth` under the top (LEGO 5 mm, 6 mm for 1-wide - BrickSim's; DUPLO
13 mm - the depth the press skill's DUPLO check grasped at), so the tool
centre is `top - grasp_depth + tip_reach` (10.2 mm: closed linear_4310
fingertips). The rotation is `robot.grasp_frame(close_heading_deg=...)`'s
(`rotation`, and `rotation_flipped` as `grasp_pose_flipped`); a plate tilted
over 5 deg, or a connector without the tool, gets tool +z down the plate
normal with the jaws along tool +y. `approach_pose` is `approach_height`
(4 cm, BrickSim's) over it - the pre-grasp and the press hover. `press_pose`
is the same hand pose as `grasp_pose`: where the hand is with the brick seated
when it holds it with this grasp. ICL's own grab pose sits on one edge of the
brick (its end-of-arm tool levers the brick off), and is not ported.

With `check_gripper` (default) `robot.describe_gripper`'s open width must
exceed the brick's width across the jaws + `jaw_margin` (4 mm). With
`check_reach`, `robot.solve_ik` previews the grasp and approach, the flipped
wrist taken when only it reaches.

## The inverse

`observed_pose` (Se3Pose; `observed_reference` `body_centre` (pose6d's),
`bottom` or `top`) in the plate frame: a box twin with z down is flipped up;
the yaw snaps to 0 or 90 (a square brick to 0, its yaw folded by 90); the
centre snaps to the nearest cell for that footprint and the bottom to the
nearest layer. `residual` gives x, y (plate axes), z, yaw and tilt and the
lateral norm; over the tolerances is `off_grid`, with the nearest cell still
returned. The same checks as the forward direction follow (pass `occupied`
without the brick itself).

## Inputs

- `plate_pose`, `plate_studs` (required); `brick_size` (`2x2`); exactly one
  of `cell` or `observed_pose`.
- `system`: `duplo` (default, as pressing-a-brick-onto-studs; 16 mm pitch,
  19.2 mm bricks, 4.5 mm studs) or `lego` (8 mm, 9.6 mm, 1.7 mm). `stud_pitch`,
  `brick_height`, `stud_height`, `stud_diameter` override it.
- `plate_thickness` (0), `grid_bias` ([0, 0] m along the plate's x, y; a
  calibration shift of the lattice - ICL's code amounts to [-0.0002, -0.0002]).
- `occupied`: stud cells `[x, y, layer]`, placed bricks `{x, y, layer,
  brick_size, rotation}`, or earlier outputs of this skill (their `cells`).
- `close_axis`, `grasp_depth`, `tip_reach`, `approach_height`, `jaw_margin`,
  `use_grasp_frame`, `check_gripper`, `check_reach` (false), `arm_id` (passed
  only when nonzero).
- Inverse tolerances: `lateral_tolerance` (a quarter pitch), `height_tolerance`
  (a quarter brick height), `yaw_tolerance_deg` (10), `tilt_tolerance_deg` (10).

## Outputs

`route`, `detail`, `direction` (`to_world` / `to_cell`), `cell`, `brick_size`,
`footprint` (cells along the plate's x, y), `cells`, `brick_pose`,
`top_pose`, `seat_position`, `seat_yaw_deg`, `brick_studs`, `system`,
`stud_pitch`, `stud_height`, `stud_diameter`, `brick_height`, `grasp_pose`,
`grasp_pose_flipped`, `approach_pose`, `press_pose`, `close_heading_deg`,
`close_axis` (`short` / `long`, or `x` / `y` for a square brick), `jaw_width`
(the brick's width across the jaws), `pose6d_mesh` (`{"box": [x, y, z]}` body
without studs), `support_plane` (`[nx, ny, nz, d]` of the surface under the
brick, pose6d's form), `residual`, `collisions`.

## Checks

CPU only, no robot, no GPU (no simulation check: the skill does not move the
arm). Unit tests (`tests/bundles/test_computing_brick_pose_from_stud_grid.py`,
36): ICL's `calc_brick_loc` transcribed line by line reproduces this skill's
`top_pose` to 1e-9 m on ICL's own 48 x 48 plate pose with `grid_bias =
[-0.0002, -0.0002]` (ori 0 and 1, 1x6 to 2x4, layers 1-3); 600 random round
trips (random plate yaw and 3 deg tilt, plate sizes 6-32, seven brick sizes,
four rotations, layers 1-5, LEGO and DUPLO, from the body centre and from the
top) come back to the same cells and seat with 0 residual; every exit; the
press skill's presets and its no-box hang equal what is computed here.

After pose6d (`tests/bundles/_brick_grid_eval.py --trials 50`, seed 0): a brick
body rendered at a random cell of a randomly yawed plate with pose6d's station
depth noise, mask dilated a pixel, registered with `pose6d_mesh` and
`support_plane` (expected: `brick_pose` 2 mm off as `init`; blind: no init),
then mapped back. Shifted: the true brick half a pitch off its cell, blind.
The plate pose is exact here: its own error adds directly to the residual.

| system | camera | mode | pts | right cell, computed | off_grid | other | lateral residual med / p90 (mm) | yaw med / p90 (deg) |
|---|---|---|---|---|---|---|---|---|
| lego | wrist | expected | 2127 | 50/50 | 0 | 0 | 0.07 / 0.42 | 0.1 / 0.5 |
| lego | wrist | blind | 1827 | 50/50 | 0 | 0 | 0.06 / 0.26 | 0.1 / 0.4 |
| lego | wrist | shifted | 1993 | 0/50 | 49 | out_of_plate 1 | 3.98 / 3.99 | 0.1 / 0.4 |
| lego | overhead | expected | 141 | 49/50 | 1 | 0 | 0.66 / 1.33 | 2.2 / 4.9 |
| lego | overhead | blind | 113 | 43/50 | 7 | 0 | 0.67 / 1.40 | 2.5 / 11.0 |
| lego | overhead | shifted | 116 | 0/50 | 49 | out_of_plate 1 | 3.59 / 3.99 | 2.9 / 7.3 |
| duplo | wrist | expected | 8037 | 50/50 | 0 | 0 | 0.06 / 0.38 | 0.0 / 0.1 |
| duplo | wrist | blind | 9119 | 50/50 | 0 | 0 | 0.08 / 0.37 | 0.0 / 0.1 |
| duplo | wrist | shifted | 8441 | 0/50 | 47 | out_of_plate 3 | 7.98 / 8.00 | 0.0 / 0.1 |
| duplo | overhead | expected | 404 | 50/50 | 0 | 0 | 0.46 / 0.87 | 0.7 / 2.3 |
| duplo | overhead | blind | 427 | 50/50 | 0 | 0 | 0.43 / 0.87 | 0.5 / 1.7 |
| duplo | overhead | shifted | 439 | 0/50 | 44 | out_of_plate 6 | 7.79 / 7.95 | 0.4 / 1.9 |

No trial returned a wrong cell as `computed`. The LEGO overhead misses are
`off_grid` on yaw (a 2 cm brick is ~113 points at 0.9 m; p90 yaw 11 deg), so
read LEGO from the wrist. Every brick half a pitch off was flagged
(`out_of_plate` where the nearest cell ran off the plate's edge). Not
measured: studs (the renders are bodies; pose6d notes the stud tops load its
top-face residual), real frames (none of LEGO or DUPLO on disk), and the
plate pose's own error.

## Limits

- No motion and no simulation check: the skill does not move the arm (sim check
  not needed; no GPU host was used this round).
- The plate pose must be accurate to well under a pitch; this skill trusts it.
  The station's overhead camera alone (5.4 mm RMS) is not enough for LEGO;
  refine from the wrist.
- Finger room is checked at cell granularity: one row of cells each side, the
  brick's whole length, at its layer and above. A real fingertip is narrower
  than a 4-long brick (conservative) and may be wider than a LEGO pitch (not
  caught).
- A YAM station refuses a tool centre under 14 mm over its table: a LEGO
  layer-1 grasp is 14.8 mm over the plate's top.
- `unsupported` only knows the cells it is given; with no `occupied`, nothing
  is checked.

## The constants, and why

- **`PRESETS`**: LEGO from ICL's `Lego.hpp` (`P_len_` 8 mm, `brick_height_m_`
  9.6 mm, `knob_height_` 1.7 mm); DUPLO and stud diameters as
  pressing-a-brick-onto-studs (the values are asserted equal in the tests).
- **`FOOTPRINT_PLAY = 0.0001`**: ICL's `brick_len_offset_` (0.2 mm) split
  over both sides: a 2x4 LEGO body is 15.8 x 31.8 mm.
- **`TIP_REACH = 0.0102`**, **`APPROACH_HEIGHT = 0.04`**, grasp depths: see
  the grasp; shared with the press skill.
- **`JAW_MARGIN = 0.004`**: 2 mm a side, about a YAM's repeatability.
- **Inverse tolerances**: rounding leaves at most half a pitch, so a quarter
  separates "on this cell" from "between two"; 10 deg yaw / tilt is a brick
  clearly not seated on studs.
- **`FLAT_TOLERANCE_DEG = 5`**: past it robot.grasp_frame's world top-down is
  not the plate's.

## Provenance

Adapted from ICL's Robotic_Lego_Manipulation (MIT; `Lego::calc_brick_loc`,
`Lego::calc_brick_grab_pose`, `Lego.hpp`; Liu, Sun, Liu, arXiv:2309.02354).
The chain is kept and generalised to any yaw and to DUPLO; ICL subtracts
`brick_len_offset_` three times, which shifts every brick 0.2 mm off the
lattice along -x and -y - here that is the optional `grid_bias`, and a test
reproduces ICL's poses exactly with it. Their edge-on grab pose for their
end-of-arm tool becomes a centred parallel-jaw grasp; the inverse, the bounds,
occupancy and gripper checks are new. The script's header lists the changes.
