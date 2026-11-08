---
name: pressing-a-brick-onto-studs
description: Presses a held LEGO or DUPLO brick onto the studs under it and
  checks by height that it seated - BrickSim's scripted press (hover 4 cm over
  the seat, fold the lateral error in, press until the command leads the
  stopped hand by 3 mm) with tracking lag and joint effort standing in for a
  force sensor, and, as strategy 'tilt', ICL's edge-first placement (the brick
  tilted about its leading bottom edge, that edge lowered beside the stud row
  and pulled back until its wall catches the row, then rotated level and
  pressed). After the release the wrist depth camera reads the brick's top (a
  brick resting on the stud tops sits a whole stud height high (1.7 mm LEGO,
  4.5 mm DUPLO) or tilted). Use when a grasped brick must be fitted onto a
  baseplate or another brick at a seat known from perception, with stud pitch
  and heights given or taken from the 'lego' / 'duplo' presets.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, press-fit, assembly, contact, wrist-camera]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/BrickSim/blob/cbd5de3238e6ac12f44ef699e1507a9f16bdafc3/python/bricksim/envs/assemble_brick/expert.py
      upstream_license: MIT
      paper: arXiv:2309.02354
gap:
  requires: {connector: [robot.move_cartesian_until_contact, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.solve_ik
    - robot.open_gripper
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    released: bool
    seat_height_error: float
    seat_tilt: float
    contact_height: float
    lead_correction: float
    lateral_correction: float
    yaw_turn: float
    samples: int
    verified_by: str
    seated_position: Vec3
  exit_conditions:
    seated: Released, and the wrist depth reads the brick's top within 0.35 stud heights of the seated height (0.6 mm LEGO, 1.6 mm DUPLO) with a spread under half a stud height.
    not_seated: Released, and the top reads higher than that but under 0.6 stud heights and flat - partly pressed (a real clutch the press could not push through).
    misaligned: The held brick is more than yaw_tolerance_deg off the seat's heading (nothing moved); or the press, the tilted edge or the hook stopped on the stud tops / met no stud row / the rotation to level was held up (kept and lifted back, released false); or, released, the top reads 0.6 stud heights or more high, or tilted.
    not_holding: The jaws read at their stop before anything moved, or after the release the wrist depth finds no brick top at the seat.
    unverified: Released, but the height was not read (no wrist camera with a pose, no reachable view, fewer than 3 depth readings on the top, or verify false); detail says how far the press went.
    unreachable: The seated (or tilted) hand pose failed the IK preview, or a move was refused; detail says which, and whether the brick is still held.
  canonical_scripts:
    - press_brick: scripts/press_brick.py
  examples:
    - title: Press a held DUPLO brick onto a baseplate seat
      path: examples/canonical_subgraph.json
---

# pressing-a-brick-onto-studs

Fit a held brick onto studs with an arm that has no force sensor, and check
afterwards, from the wrist camera, that it went on.

## How it composes

- **Before it**: a grasp of the brick (BrickSim grasps 5 mm under the top, 6 mm
  for 1-wide bricks), its box measured in the hand (`held_obb`, e.g. from
  perceiving-objects or registering-held-objects), and the seat from perception
  (computing a brick pose from the stud grid, refined from a wrist close-up).
- **After it**: `seated` goes on to the next brick. `misaligned` with
  `released: false` still holds the brick at the hover: re-perceive the seat and
  call again. `not_seated` / `misaligned` with `released: true` leave the brick
  on the plate: press it again from above, or regrasp.

## The motion

Common start: read the hand; check the jaws are not at their stop
(`robot.get_gripper` / `robot.describe_gripper`, else `not_holding`); work out
where the hand must be for the brick's bottom centre to be on `seat_position`
(the brick's offset in the hand, measured now and carried unchanged); preview
that pose with `robot.solve_ik` (a stall at the edge of reach looks like
contact); rise straight up to the hover if under it. If `seat_yaw_deg` is given
and the held brick is more than `yaw_tolerance_deg` (1 deg) off it (modulo 90
deg for a square brick), turn the hand about the vertical where it stands, read
it back, and finish on the last joint (stacking-objects' turn: some arms'
solvers leave the wrist roll free).

**`strategy: straight`** (BrickSim's expert):

1. Hover `approach_height` (4 cm, BrickSim's ASSEMBLY_HEIGHT_OFFSET) over the
   seated pose. BrickSim descends once its lateral error is under 2 cm and
   folds the error into the press pose; here the hover is re-commanded, shifted
   by what the hand missed by, until it reads back within `align_tolerance`
   (0.5 mm), and the descent keeps that shift.
2. `robot.move_cartesian_until_contact` to `contact_tolerance` (6 mm) over the
   stud tops. A stop well above it is `misaligned` (seat higher than given or
   something in the way; kept and lifted back).
3. The press: fine `robot.go_to_pose` steps of a quarter stud height
   (1.1 mm DUPLO, 0.43 mm LEGO) with a 0.3 mm arrival ball. Contact is the
   hand lagging its command by half a step more than in free motion (or a
   joint-effort residual where the observation carries efforts,
   detecting-contact-from-joint-effort's test). If the first contact is more
   than half a stud height over the seat, the walls are on the stud tops:
   `misaligned`, kept and lifted back. Otherwise the steps go on until the
   command leads the stopped hand by `press_depth` (3 mm, BrickSim's
   PRESS_DEPTH, the depth of its press pose under the seat) or reaches 3 mm
   under the seat. On a stiff position servo that lead is the press force.
4. Hold `hold_s` (0.5 s), open, rise straight up to the hover.
5. Check (below).

**`strategy: tilt`** (ICL's edge-first placement, for a parallel gripper):

1. Level hover over the spot, then tilt about the leading bottom edge (`lead`,
   a brick axis, default `+x`) in four steps to `tilt_deg`: ICL's 14 deg,
   raised so the trailing edge clears the stud tops (`TILT_CLEAR`) and so the
   underside tube clears them while the edge hooks (`TUBE_*`), at most 27 deg
   (DUPLO 2x2: 25.9 deg; LEGO 2x2: 19.3 deg). The rest of the motion uses the
   tilt the hand actually reached.
2. The edge starts a tenth of a pitch past the seat (ICL lowers to an offset
   pose first). Fast descent to over the stud tops, then fine steps down until
   the edge meets something: on a stud top (`misaligned`, kept) or the plate
   in the gap beside the row.
3. Hook: raise the edge 0.2 stud heights, then step it back along the lead up
   to 0.4 pitch until its wall catches the seat's leading row (lag). That stop
   locates the brick along the lead: `lead_correction` is the seat's error
   along it, and `seated_position` the corrected seat. No catch is
   `misaligned` (kept).
4. Rotate level about the caught edge in four fine steps (`misaligned` if the
   hand reads more than 4 deg from level), then the straight press from step 3
   above, release and check.

ICL's release twist (its end-of-arm tool levers off the brick) is dropped: a
parallel gripper just opens.

**The check**: aim the wrist camera straight down at the seat's top from
0.20 m (then 0.25 m, then with the hand's own rotation), as
refining-target-from-wrist-closeup does, with the camera's pose read from the
observation; IK-preview and clearance-gate each view. Project points on the
brick's top that lie between its studs, take the median depth in a 3 x 3 window
at each (7-50 cm), lift them to world heights, and compare with the seated top
(`seat_position.z + brick_height`): median within 0.35 stud heights and spread
under half a stud height is `seated`; 0.6 stud heights or more, or tilted, is
`misaligned` (resting on the studs); between is `not_seated`; a whole brick
height low is `not_holding` (no brick there).

## Inputs

- `seat_position` (required): world centre of the brick's bottom face when
  seated; z = the top of the plate or brick under it, without its studs.
- `held_obb` (recommended): the held brick's box (with its studs), world
  frame. Without it, the hang is `brick_height` + 10.2 mm fingertip reach -
  BrickSim's grasp depth, and the footprint `brick_studs` x pitch.
- `strategy`: `straight` (default) or `tilt`.
- `system`: `duplo` (default; 16 mm pitch, 9.4 mm studs 4.5 mm high, 19.2 mm
  bricks) or `lego` (8 mm, 4.8 mm, 1.7 mm, 9.6 mm). `stud_pitch`,
  `stud_height`, `stud_diameter`, `brick_height` override it.
- `brick_studs` ([2, 2]), `seat_yaw_deg` (the seat's row heading; omitted:
  no yaw check), `lead` (`+x`, `-x`, `+y`, `-y`), `tilt_deg` (14).
- `approach_height`, `press_depth`, `contact_tolerance`, `lag_threshold`
  (default half a step; a real YAM's lag noise is millimetres, see Limits),
  `align_tolerance`, `yaw_tolerance_deg`, `align_yaw` (true), `roll_limits`,
  `hold_s`, `verify` (true), `camera`, `check_holding`, `check_reach`,
  `reach_tolerance`, `arm_id` (passed only when nonzero). The fine-move knobs
  (`tolerance`, `max_steps`) are passed to `robot.go_to_pose` only by
  connectors that take them.

## Outputs

`route`, `detail`, `strategy`, `released`, `seat_height_error` (the top's
median height over the seated top, from depth), `seat_tilt` (10-90 percentile
spread), `contact_height` (the brick's bottom over the seat at first contact,
read from the hand: low by any slide in the jaws), `lead_correction`,
`lateral_correction` (the hover fold), `yaw_turn`, `tilt_deg`, `samples`,
`verified_by` (`wrist_depth` or empty), `seated_position`.

## Simulation check

YAM station twin in RoboSimStudio (left arm, MuJoCo at 500 Hz) with a
DUPLO-scale scene built for this check, since the twin has no LEGO: a fixed
4 x 4 stud baseplate (9.4 mm studs, 4.5 mm high, 16 mm pitch, 0.4 mm top
chamfer) and a free 2 x 2 brick (31.8 x 31.8 x 19.2 mm, 8 g) made of convex
collision pieces - top plate, four walls whose inner faces stand 0.1 mm off
the studs, a central tube 0.3 mm off them, four top studs - and a wrist D405
at the real station's calibrated pose and intrinsics (rendered depth).
**Approximated**: the clutch. Contacts are rigid with clearance, so the brick
goes down with no interference fit and nothing holds it but gravity; the force
needed to press through a real clutch is not tested. The tube is looser than
the walls (a rigid tube that touches the studs jams any tilted entry).
Harness, not skill: each trial set the brick on the table (a harness tool),
grasped it top-down 13 mm under its top with the jaws closed to its width less
2 mm (closed to the stop, the twin's separately driven fingers let it float
2-3 mm along the closing axis), carried it 7 cm over the seat and measured its
box there as `held_obb`. The skill got the true seat shifted by (dx, dy);
simulator ground truth only set up and measured.

| strategy | seat error given | route | brick afterwards (truth) |
|---|---|---|---|
| straight | 0 | seated (depth -0.0 mm) | seated, 0.1 mm off in the plane |
| straight | +1 mm x | seated (depth -0.0 mm) | seated, 0.1 mm off (the chamfer and the servo's give took up the 1 mm) |
| straight | +2 / +3 / +4 / -2 mm x, +2 mm y | misaligned x5, kept, lifted back | in the hand; first contact read 4.0-4.1 mm over the seat (stud tops at 4.5) |
| tilt (25-26 deg) | 0, +-1, +2 mm x, +1, +2 mm y | misaligned x6, kept: "edge went 2.2 mm under the seat without the arm lagging" | in the hand |
| tilt | +3 / +4 mm x | misaligned x2, kept: the hook met no stud row | in the hand |
| tilt | -2 mm x | misaligned, kept: the edge stopped 6.1 mm over the seat | in the hand |

- Straight: 2/2 seated within 1 mm, 5/5 refused at 2-4 mm with the brick kept,
  and each verdict matched the truth. Two earlier runs agree: a straight trial
  that started with the brick 4.9 deg off the seat's heading in the hand
  turned it out at the hover and seated (truth 0.1 mm off); one at +1 mm
  seated. In earlier runs where the hand released a brick resting on the studs
  (before the press stopped on the first contact), the depth check read it
  +4.5 mm high, spread 0.0 mm, and routed `misaligned` (truth: 4.5 mm up), 3/3.
- Tilt: 0/9. Ground truth showed why: the fingertip grip does not hold a
  25 deg tilt. As the hand tilted, the brick shifted in the jaws, so its edge
  came down about 5 mm further out than a rigid grip predicts, on the next
  stud row; and once the edge touched, the brick turned back toward level in
  the jaws (23 deg to 2 deg) instead of stalling the arm, so no lag was read.
  With a 14 deg tilt (earlier runs) the underside tube landed on the leading
  row instead (the edge stopped 1.8 mm over the plate); hence the
  tube-clearing tilt. The tilt path is covered by unit tests against a rigid
  grip; it has not seated a brick in simulation, and the evidence here says
  it needs a grip that holds the brick's tilt (deeper fingers, pads, or an
  end-of-arm tool like ICL's).
- The twin's servo arrives within 4 mm / 3 deg unless asked for less, which is
  why the press uses fine steps with a 0.3 mm ball, and why the tilt is
  measured after it is reached.
- BrickSim itself (Isaac Sim) was not run.

## Limits

- Sub-millimetre placement from a YAM: the straight press captured 1 mm of
  seat error in the twin and refused 2 mm. A real YAM repeats to a few
  millimetres; give it a seat refined from a wrist close-up, and expect
  `misaligned` (kept) as the common failure, which is safe to retry.
- Lag thresholds are the twin's: its servo reaches a 0.3 mm ball, so contact
  is half a step (0.55 mm DUPLO). A real YAM stood 5.4 mm behind its aim in
  free air (detecting-contact-from-joint-effort); set `lag_threshold` from a
  free-air run on the station, and prefer DUPLO.
- Press force is bounded by the grip: a brick that slides up the jaws stops
  the arm lagging; the depth check after the release catches the result
  (`misaligned` / `not_seated`), not the cause.
- Not run on a real YAM.

## The constants, and why

- **`APPROACH_HEIGHT = 0.04`**, **`PRESS_DEPTH = 0.003`**, **`GRASP_DEPTH =
  0.005`** (+1 mm for 1-wide): BrickSim's expert.
- **`TILT_DEG = 14`**: ICL's `Twist_Deg`; raised by **`TILT_CLEAR = 1.25`**
  (trailing edge) and **`TUBE_REACH_FRACTION = 0.58`, `TUBE_CLEAR = 1.1`**
  (the first underside tube, 0.58 pitch in from the edge for 2x2 LEGO and
  DUPLO), capped at **`TILT_MAX_DEG = 27`** (the twin's arm reached 26 deg
  over the seat, not 30).
- **`HOOK_OFFSET_FRACTION = 0.1`**, **`HOOK_REACH_FRACTION = 0.3`**: the edge
  starts in the gap beside the row (3.4 mm DUPLO, 1.7 mm LEGO to the next
  row) and is pulled back past it; **`ENGAGE_FRACTION = 0.2`**: at 0.35 the
  wall's inner corner rode over the row's chamfers in the twin.
- **`FEEL_STEP_FRACTION = 0.25`**, **`LAG_FRACTION = 0.5`**,
  **`FINE_TOLERANCE = 0.0003`**, **`FINE_MAX_STEPS = 90`**: see the press.
- **`MISALIGNED_FRACTION = 0.5`** (from the hand, low by the slide: the twin
  read 4.0-4.1 mm for bricks resting on 4.5 mm studs), **`RESTING_FRACTION =
  0.6`**, **`SEAT_FRACTION = 0.35`**, **`TILT_FRACTION = 0.5`** (from depth).
- **`YAW_TOLERANCE_DEG = 1`**: a 2x2 DUPLO's corner moves 0.4 mm per degree.
- **`CONTACT_TOLERANCE = 0.006`**: about one RMS of a calibrated YAM
  station's camera error (5.4 mm).
- **`VIEW_RANGES`, `MIN_VIEW_CLEARANCE`, depth window**:
  refining-target-from-wrist-closeup's.

## Provenance

Adapted from BrickSim's `AssembleBrickExpert` (MIT; constants,
`_compute_assembly_state`, `_grasp_depth`) and from ICL's
Robotic_Lego_Manipulation place sequence (MIT; attack-angle offset, place modes
7-10, `Twist_Deg`; Liu, Sun, Liu, arXiv:2309.02354). Both read privileged
simulator or calibrated poses; here the seat and the held brick come from
perception, contact from tracking lag or joint effort, and the result from
wrist depth. The script's header lists the changes.
