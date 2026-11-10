---
name: verifying-brick-seated-by-height
description: Checks from a depth camera whether a placed LEGO or DUPLO brick
  is seated on its studs - the brick's top face fitted as a plane inside its
  expected footprint (stud tops told apart from the face by their height), its
  height measured against the plate surface around the footprint (so the
  camera's extrinsic error cancels), its tilt, and how much of the footprint
  it covers - with BrickSim's connection tolerances (1 mm, 5 deg, 2 mm at
  LEGO pitch) as acceptance limits. A brick resting on the stud tops reads a
  whole stud height high (1.7 mm LEGO, 4.5 mm DUPLO), one with an end on the
  studs reads tilted, one on the wrong studs leaves part of its footprint at
  plate height. Moves the wrist camera over the seat first, or reads the
  current frame of any posed depth camera. Use after a brick is released
  (by any placing or pressing skill) to decide between going on, pressing
  again and re-seating, or as a standalone audit of a built cell.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [lego, duplo, press-fit, assembly, verification, depth, wrist-camera, d405]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/BrickSim/blob/cbd5de3238e6ac12f44ef699e1507a9f16bdafc3/python/bricksim/mdp/connection_thresholds.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_observation
    - robot.go_to_pose
    - robot.solve_ik
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    height_error: float
    tilt_deg: float
    edge_rise: float
    coverage: float
    shift_estimate: Vec3
    top_height: float
    reference: str
    reference_offset: float
    samples: int
    viewed_from: str
  exit_conditions:
    seated: The top face reads within height_tolerance of the seated height (default min(1 mm, 0.35 stud heights) - 1.0 mm DUPLO, 0.6 mm LEGO), tilted no more than tilt_tolerance_deg, and the brick covers its footprint to within position_tolerance.
    not_seated: Flat and on its footprint, but the top reads higher than height_tolerance and under 0.6 stud heights - partly pressed.
    high_by_stud: The top reads 0.6 stud heights or more over the seated height and flat - resting on the stud tops (the detail adds the offset if it also stands off the footprint).
    tilted: Tilted more than tilt_tolerance_deg (default the smaller of BrickSim's 5 deg and the tilt that lifts the long side's far end half a stud height - DUPLO 2x2 4.0 deg, LEGO 2x2 3.0 deg, LEGO 2x4 1.5 deg) - one end on the studs.
    shifted_off_grid: Part of the footprint reads at plate height, so the brick stands more than position_tolerance (default a quarter pitch - 2 mm LEGO, 4 mm DUPLO) off it - on the wrong studs; shift_estimate says which way.
    no_brick: Under a quarter of the footprint reads at brick height - no brick at the seat (it came away with the gripper, or the seat is wrong).
    unverified: Not judged - no posed depth camera, too few depth points in the footprint, reference 'local' requested but no plate surface found, or the top reads below the seated height (seat height or calibration off); detail says which.
    unreachable: view 'move' and every view of the seat was out of reach or refused.
  canonical_scripts:
    - verify_seat: scripts/verify_seat.py
  examples:
    - title: Check that a DUPLO 2x2 seated after pressing
      path: examples/canonical_subgraph.json
---

# verifying-brick-seated-by-height

A brick that is released over its studs can end up seated, partly pressed,
resting on the stud tops, with one end up, or on the wrong studs. All five
look alike from the gripper. In depth they differ by a stud height
(1.7 mm LEGO, 4.5 mm DUPLO), a slope, or a strip of the footprint at plate
height. This skill reads the brick's top in the depth image and says which.

## How it composes

- **Before it**: the brick is placed and released, by
  pressing-a-brick-onto-studs, placing-with-contact-descent or any other
  placement. The seat comes from computing-brick-pose-from-stud-grid (the
  `seat` it returns) or from perception.
- **After it**: `seated` goes on to the next brick (and to a build-state
  update). `not_seated` and `high_by_stud` call for pressing again from
  above. `tilted` and `shifted_off_grid` call for a regrasp and re-seat
  (disassembling-a-brick-by-twist-lever if the brick is clutched).
  `no_brick` means the brick left with the gripper or fell: look for it.
- pressing-a-brick-onto-studs has its own quick depth check (a dozen points
  between the studs, median height). This skill is the fuller standalone
  version: a plane, a local height reference, coverage and tilt limits, and
  it works after any placement or on a brick placed long ago.

## How it reads

1. **View** (`view: move`, the default): aim the wrist camera straight down
   at the brick's expected top from 0.20 m (then 0.25 m, then with the
   camera's current rotation), IK-preview each view and keep the tool centre
   5 cm over the brick. This uses the same views as
   pressing-a-brick-onto-studs and refining-target-from-wrist-closeup. A
   view the servo does not quite reach is still used: the depth is
   back-projected with the camera pose read back with the frame.
   `view: current` reads the frame as it is, from the wrist camera or any
   named camera that has a pose (no motion).
2. **Depth**: the median of `frames` (3) frames, back-projected to world
   points (7-60 cm range).
3. **Reference**: in a ring 0.3-1.5 pitches outside the footprint, every
   stud-covered surface: the local minima of a two-level cost (a surface and
   its stud tops a stud height up; residuals capped at the support band,
   so taller things such as neighbour bricks drop out) that have at least a
   tenth of the ring's points on them. That can be the plate between
   neighbours that fill most of the ring, or the table past a small plate's
   edge. The surface nearest the given seat z, if within half a brick height
   of it, becomes the seat's surface (`reference: local`), and the camera's
   height error cancels. Otherwise (for example, a brick stacked on one of
   the same footprint, where the plate around it is a brick lower) the
   seat's own z is used (`world`). `reference` can force either.
4. **Top face**: points inside the footprint inset by 1.5 mm, more than half
   a brick over the reference and less than two brick heights plus a stud
   over it (the wrist camera can see its own fingers). `coverage` is their share of the footprint.
   A plane is fitted to them in two layers: each point is assigned to the
   face or to a stud top one stud height up, stud points are lowered by that
   height, outliers beyond 2.5 robust sigmas are trimmed, and the fit is
   repeated six times. This works even when the depth noise is close to the
   LEGO stud height.
5. **Judge**: height error = the plane at the footprint centre minus
   (reference + brick height); tilt from the plane's normal; shift from the
   uncovered strip (its width times the uncovered share, pointing away from
   it). The order: shifted (unless also resting high), tilted, resting high,
   seated, partly pressed.

## Inputs

- `seat_position` (required): world centre of the brick's bottom face when
  seated. Its z is the top of the plate or brick under it, without studs (the
  convention of pressing-a-brick-onto-studs and
  computing-brick-pose-from-stud-grid).
- `brick_studs` ([2, 2]), `seat_yaw_deg` (0: the brick's first stud axis
  along world x).
- `system`: `duplo` (default; 16 mm pitch, 4.5 mm studs, 19.2 mm bricks) or
  `lego` (8 mm, 1.7 mm, 9.6 mm). `stud_pitch`, `stud_height` and
  `brick_height` override the preset.
- `view` (`move` | `current`), `camera` (a camera name; default the wrist /
  eye-in-hand camera nearest the tool), `frames` (3), `reference` (`auto` |
  `local` | `world`).
- `height_tolerance`, `tilt_tolerance_deg`, `position_tolerance` (defaults
  above), `check_reach` (true), `arm_id` (passed only when nonzero).

## Outputs

`route`, `detail`, `height_error` (m, top over the seated top), `tilt_deg`,
`edge_rise` (m, how much higher the long side's far end is than the near
end), `coverage` (0-1), `shift_estimate` (world xy, m; lower bound of the
offset), `top_height` (world z of the plane at the footprint centre),
`reference` (`local` / `world`), `reference_offset` (the reference surface's
reading minus the given seat z: the camera's height error plus the seat's
own), `samples` (depth points in the footprint), `viewed_from`.

## Checked

**Station twin** (RoboSimStudio YAM station, GCP pool host, 2026-11-09):
the 4 x 4 stud DUPLO baseplate and free 2 x 2 brick of pressing-a-brick-onto-studs'
check (studs colliding at their full 4.5 mm; rigid contacts, no clutch).
The real station's calibrated left wrist D405 renders depth. Each trial put
the brick at a pose (a harness tool), let it settle, ran the skill with the
true seat (view move included), then read the brick from ground truth.
Harness: `~/sim-wt/brick-camera-checks` (graph_verify).

| placed | truth after settling | route | height error mm | coverage |
|---|---|---|---|---|
| on the seat | seated | seated | +0.03 | 1.00 |
| on the seat, turned 90 deg | seated | seated | -0.02 | 1.00 |
| half a pitch off in x, dropped | on the stud tops, +4.5 mm | high_by_stud | +4.50 | 0.83 |
| half a pitch off in y, dropped | on the stud tops | high_by_stud | +4.50 | 0.86 |
| half a pitch off in x and y | on the stud tops | high_by_stud | +4.50 | 0.67 |
| one cell off in x | seated on the next studs | shifted_off_grid | -0.00 | 0.54 |
| one cell off in -y | seated on the next studs | shifted_off_grid | -0.00 | 0.55 |
| dropped tilted 12 deg, 4 mm off | settled flat on the stud tops | high_by_stud | +4.50 | 1.00 |
| dropped tilted -10 deg, 6 mm off | settled flat on the stud tops | high_by_stud | +4.50 | 0.92 |
| 9 cm away (seat empty) | no brick | no_brick | - | 0.00 |

10/10 routes match the truth. The first twin run read 6/10: the twin's plate
is 4 x 4 studs, so the reference ring reaches past its edge onto the cloth
4 mm lower. The "lowest surface" rule took the cloth, and every brick read
3.4 mm high. The reference is now the surface nearest the seat z (a unit
test on a 4 x 4 plate covers it). The rigid twin never left a brick tilted:
dropped tilted, it slid flat onto the stud tops. The tilt path is covered by
the synthetic checks below, not the twin. No partly pressed brick either:
without a clutch nothing holds one there.

**Synthetic depth** (not a simulator): `tests/bundles/_seat_eval.py` ray-casts
a 12 x 12 plate with studs and one brick in a known pose, as seen by a
640 x 480 D405 (fx 394 px) 0.2 m straight above the top, with D405-like
noise (sigma = z^2 x 0.08 / (f x 18 mm), spatially correlated, flying pixels
at depth edges, 0.5 % dropouts, 0.1 mm quantisation), 3 frames per read. It
runs the skill end to end through a fake arm: the view move, then the read.
Results are in the table below.

| brick | truth | routes (of 5 noise seeds) | height error mm (mean, sd) | tilt deg | coverage |
|---|---|---|---|---|---|
| duplo 2x2 | seated | seated 5 | -0.01 (0.01) | 0.1 | 1.00 |
| duplo 2x2 | partly pressed, 0.4 stud (1.8 mm) | not_seated 5 | +1.79 (0.01) | 0.1 | 1.00 |
| duplo 2x2 | resting on the stud tops | high_by_stud 5 | +4.50 (0.01) | 0.1 | 1.00 |
| duplo 2x2 | one end on the studs (8 deg) | tilted 5 | +2.45 (0.01) | 8.0 | 1.00 |
| duplo 2x2 | one cell off in x | shifted_off_grid 5 | +0.02 (0.04) | 0.3 | 0.55 |
| duplo 2x2 | half a cell off, on the studs | high_by_stud 5 | +4.50 (0.02) | 0.2 | 0.84 |
| duplo 2x2 | seated, camera pose 4 mm high | seated 5 | +0.01 (0.01) | 0.1 | 1.00 |
| duplo 2x2 | seated, flush neighbours on 4 sides | seated 5 | -0.01 (0.01) | 0.1 | 1.00 |
| duplo 2x2 | no brick | no_brick 5 | - | - | 0.00 |
| lego 2x4 | seated | seated 5 | +0.02 (0.04) | 0.5 | 1.00 |
| lego 2x4 | partly pressed, 0.45 stud (0.77 mm) | not_seated 5 | +0.79 (0.04) | 0.5 | 1.00 |
| lego 2x4 | resting on the stud tops | high_by_stud 5 | +1.72 (0.04) | 0.5 | 1.00 |
| lego 2x4 | one end on the studs (3.0 deg) | tilted 5 | +0.46 (0.04) | 3.0 | 1.00 |
| lego 2x4 | one cell off in y | shifted_off_grid 5 | +0.00 (0.02) | 0.4 | 0.79 |
| lego 2x2 | seated | seated 5 | +0.03 (0.03) | 0.7 | 1.00 |
| lego 2x2 | resting on the stud tops | high_by_stud 5 | +1.73 (0.02) | 0.7 | 1.00 |

80/80 verdicts match the truth. The height reads within 0.03 mm of the
truth (sd at most 0.04 mm, three frames median-combined); tilt on a level
brick 0.1-0.7 deg against limits of 1.5-4.0 deg. A one-cell shift is
estimated short (13 mm for 16 mm DUPLO, 6-7 mm for 8 mm LEGO: the inset
footprint), but always far past the quarter-pitch limit. The renderer is
ideal geometry: no specular tops, no dark plastic, no stereo shadowing beside
tall neighbours.

## Limits

- **Calibration**: the height is relative to the plate surface around the
  brick, so an extrinsic height error cancels (a 4 mm error read 4 mm in
  `reference_offset` and the verdict did not change). The shift test is
  absolute: the footprint comes from `seat_position`, so a seat taken from a
  different camera than the one checking carries both cameras' planar error.
  Prefer a seat from the same wrist camera (calibrating-baseplate-stud-grid
  or computing-brick-pose-from-stud-grid over a plate calibrated with it).
- **Yaw** is not measured. A brick on studs cannot be turned a few degrees
  and stay flat: it rides the studs (`high_by_stud` / `tilted`). A brick
  turned 90 deg on a non-square footprint shows as `shifted_off_grid`.
- **Retention force** is not tested. A brick that reads seated sits where a
  seated brick sits; a gentle probe push (verifying-insertion-by-probe-push)
  is the second check.
- **Dark or glossy tops**: D405 noise on black or specular bricks is larger
  than the model here; check `samples` and the detail, and raise `frames`.
- Not run on a real YAM. In the station twin, tilted and partly pressed
  bricks could not be produced (see Checked).

## The constants, and why

- **`DISTANCE_TOLERANCE = 0.001`**, **`Z_ANGLE_TOLERANCE_DEG = 5`**:
  BrickSim's assembly `distance_tolerance` and `z_angle_tolerance`.
- **`POSITION_FRACTION = 0.25`**: BrickSim's 2 mm `position_tolerance` at
  LEGO's 8 mm pitch, scaled with the pitch (a brick can only sit flat on
  whole cells, so anything past a quarter pitch while flat is a wrong cell).
- **`SEAT_FRACTION = 0.35`**, **`RESTING_FRACTION = 0.6`**,
  **`TILT_FRACTION = 0.5`**: pressing-a-brick-onto-studs' depth bands (its
  twin read +4.5 mm for DUPLO bricks resting on the studs).
- **LEGO stud height 1.7 mm**: ICL Robotic_Lego_Manipulation `knob_height_`;
  DUPLO 4.5 mm: published dimensions.
- **`EDGE_INSET = 1.5 mm`**, **`VIEW_RANGES`**, **`MIN_VIEW_CLEARANCE`**:
  pressing-a-brick-onto-studs / refining-target-from-wrist-closeup.
- **`RING_INNER / RING_OUTER = 0.3 / 1.5` pitch**, **`REFERENCE_WINDOW_FRACTION
  = 0.5`**, **`LOWER_SUPPORT = 0.1`**, **`SUPPORT_BAND_FRACTION / _MIN = 0.35 / 0.8 mm`**, **`TRIM_SIGMAS = 2.5`**, **`PLANE_ITERS = 6`**,
  **`NO_BRICK_COVERAGE = 0.25`**, **`MIN_POINTS = 40`**, **`FRAMES = 3`**:
  chosen on the synthetic depth above. With 0.5 mm per-pixel noise at
  0.2 m, the two-layer fit read a level LEGO 2x2 within 0.1 mm and 1.3 deg.

## Provenance

The acceptance limits are BrickSim's simulator connection thresholds
(`configure_assembly_thresholds`, MIT), and the LEGO stud height is ICL's
(`Lego.hpp`, MIT). BrickSim applies them to privileged stud and hole poses;
here they apply to a depth camera's view of the released brick. The
measurement (two-layer plane, local reference, coverage) is new. The view
code follows pressing-a-brick-onto-studs. The script's header lists the
changes; licences in LICENSE.bricksim and LICENSE.icl.

## Noisy, millimetre-stepped depth: `face_estimator="stud_mask"`

The default `two_layer` fit separates face pixels from stud-top pixels by
height; under the station's D405 (fitted noise, depth in 1 mm steps) it
misassigns them and reads seated LEGO bricks high (+1.10 mm on 12 of 16
seated placements, LEGO station, RTX wrist depth). `stud_mask` leaves out
every pixel within 3.6 mm of a stud axis (the brick's own and the plate's
lattice, from the seat pose), keeps the face pixels within +-4 mm of a first
level (a narrow window locked the quantisation offset in), trims 10 % per tail
and fits a plane; the noise dithers the 1 mm steps. Its default tilt
tolerance is 2.0 deg (`STUD_MASK_TILT_TOL_DEG`).

LEGO station (Isaac + BrickSim), RTX wrist depth + fitted D405 noise,
single_on_plate / single_on_brick x {no fault + 6 station faults}; tolerances
chosen on seeds 0-2 (38 placements), reported on seeds 3-5 (39 placements:
22 seated, 17 not):

| `stud_mask` tolerances | false done | successes accepted | unverified | failures caught |
|---|---|---|---|---|
| 0.6 mm, 1.5 deg (the LEGO defaults) | 0/17 | 7/22 | 11 | 12/17 |
| **0.6 mm, 2.0 deg (retuned)** | **0/17** | **16/22** | 11 | 12/17 |
| judging-assembly-steps-by-fused-checks with it | 0/17 | 16/22 | 6 uncertain | 17/17 |

- Seated bricks read -0.04 mm (on the plate) and +0.45 mm (on a brick);
  failed ones 1.45-2.12 mm (BrickSim: 1.7 mm high).
- **The 1.7 deg on seated plate bricks is not real**: BrickSim's settled tilt
  of those bricks is 0.00 deg, the reading is the same for every noise seed
  (deterministic), and a ray-cast scene of the same view reads 0.0 deg - an
  artifact of the rendered view with this estimator, absorbed by the 2.0 deg
  tolerance, not yet removed. Failed plate bricks read 1.2-2.1 deg but are
  caught by their height.
- The 11 unverified / 6 uncertain are lost-target and stuck-jaws placements
  (no brick to read at the seat); 7 of 84 episodes ended before a placement
  and are not counted.
