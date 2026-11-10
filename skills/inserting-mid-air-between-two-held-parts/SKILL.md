---
name: inserting-mid-air-between-two-held-parts
description: Inserts the part one arm holds into (or presses it onto) the part
  the other arm holds, both in the air - ACT's scripted bimanual insertion (the
  two parts brought to a shared meet point facing each other on one axis, a
  stand-off apart, then the gap closed along that axis) with the parts' in-hand
  offsets in place of simulator poses; the moving tip is aligned across the
  axis on the holder's read-back entry before anything touches, then stepped in
  while contact is felt on both arms without a force sensor (the pushing hand's
  tracking lag beyond its free-motion lag, the holding hand pushed off its
  pose, joint efforts where published). Use when a bimanual station must join
  two parts that are each held by one arm - a peg into a socket, a pin into a
  bore, a DUPLO brick onto a held sub-assembly - with both parts' contact points
  measured in the hands (registering-held-objects or perception) and both arms
  addressable by arm_id.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, insertion, peg-in-hole, assembly, mid-air, proprioception]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [insertion, bimanual]
    provenance:
      authors: [runtime-si]
      code: https://github.com/tonyzhaozh/act/blob/main/scripted_policy.py
      upstream_license: MIT
      paper: arXiv:2304.13705
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_gripper
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.open_gripper
  required_inputs:
    moving_tip: Vec3
    receiving_entry: Vec3
  produces_outputs:
    depth: float
    lateral_error: float
    axis_angle_deg: float
    probes_resisted: int
    lag_over_baseline: float
    holder_shift: float
    contact_signal: str
    steps: int
    roll_turns: int
    efforts_available: bool
    entry_position: Vec3
    tip_position: Vec3
    mover_pose: Se3Pose
    holder_pose: Se3Pose
  exit_conditions:
    inserted: The moving tip, as the two hands' read-back poses and the in-hand offsets place it, went at least insert_depth less depth_tolerance past the holder's entry along the axis, at least one of four probes across the axis (half the clearance plus 1 mm each way) was stopped or carried the holder along (verify, default on), and both jaws still read a hold (with release=mover, the mover then opened and backed out and the holder's jaws still hold). Felt by the hands, not seen - a part turned in its jaws since it was measured can still read as inserted (simulation check).
    misaligned: Nothing went in - the tip could not be brought within half the clearance of the entry's line (after three corrections), the two parts' axes (each turned with its hand since it was measured) stood more than angle_tolerance_deg apart, or the push stopped (lag, holder pushed, effort) with the tip less than 2 mm past the entry (the rim); the mover backed off to its stand-off, both still hold their parts.
    stalled: The push stopped with the tip past the entry but short of insert_depth less depth_tolerance (a jam, a burr, a tilted part); the mover backed off 3 mm so it stops pushing, both still hold. Also routed when the mover's jaws did not open for release=mover.
    slipped: A hand's jaws read at the stop - before anything moved (nothing held), or after the push or the release (a part fell out); or the hands went the whole depth but no probe across the axis met a wall - the moving part sat on the receiving part's outside face and slid back along its jaw pads, which neither arm feels (both hands still hold, the parts are not joined).
    unreachable: The two arm ids are equal or the connector reports no separate second arm, the holder could not reach the meet point or the mover its stand-off (free moves read back beyond reach_tolerance or refused), a wrist roll could not be turned back, a step was refused, or signal=effort with no efforts published; detail says which and what each hand still holds.
  canonical_scripts:
    - insert_mid_air: scripts/insert_mid_air.py
  examples:
    - title: Insert the part arm 1 holds into the part arm 0 holds
      path: examples/canonical_subgraph.json
---

# inserting-mid-air-between-two-held-parts

Join two parts that are each in one hand, without putting either down. The
holder (`holder_arm`, 0) holds the receiving part (a socket, a bore, a brick
whose studs face out); the mover (`mover_arm`, 1) holds the moving part (a
peg, a pin, a brick whose sockets face out) and pushes it in along the
insertion axis. Every robot call names its arm (`arm_id`); one arm moves per
call.

## How it composes

- **Before it**: each arm grasps its part; the part's contact feature is
  measured in the hand - `registering-held-objects` (`feature_in_tcp`, pass
  its position as `moving_tip_in_tcp` / `receiving_entry_in_tcp`) or a
  perceived world point with that arm unmoved since (`moving_tip`,
  `receiving_entry`). Turn each hand so its part's axis lies along the
  insertion axis (both parts face each other) with `moving-both-arms-together`
  or single-arm moves; this skill only translates the hands (ACT keeps fixed
  hand quaternions throughout). Check the approach with
  `checking-dual-arm-clearance` (the connectors do not collide one arm
  against the other).
- **After it**: `inserted` with `release: none` leaves both hands on the
  assembly: hand it over (`handing-over-between-arms`), let one go, or carry
  it with both (`moving-both-arms-together`, `keep_relative`). `misaligned`
  can go back to registration (a fresh in-hand offset) or to a search
  (`searching-insertion-spiral` with the holder still, as in
  `holding-steady-while-other-arm-acts`).

## The motion

1. **Two arms, both holding.** Refuse (`unreachable`) if the arm ids are
   equal or the holder's pose reads as the mover's (within 1 mm) or as no
   pose. A jaw opening at the stop on either arm is `slipped`.
2. **In-hand offsets.** The moving tip and the receiving entry are taken into
   each hand's tool frame (from world points and the hands' poses read now,
   or directly from `*_in_tcp`); from here on both are placed by the hands'
   read-back poses, rotation included. The axis is `insertion_axis`, or from
   the tip toward the entry.
3. **Rendezvous** (ACT's meet point). With `meet_position` the holder carries
   its entry there (keeping its rotation); without it the holder stays where
   it is and the entry is the meet point. The mover goes to its stand-off: the
   tip `standoff` (50 mm) back from the entry along the axis. `mode:
   both_move` is ACT's two-sided closing: the holder first stops `standoff/2`
   beyond the meet point, the mover `standoff/2` short of it, then the holder
   brings its part in to the meet point and holds still. After each free move
   a wrist roll more than 1 deg off the hand's rotation is turned back on the
   last joint (`robot.move_to_joints`, read back; PORTING lesson 10) and the
   hand re-centred.
4. **Align on the holder's actual entry.** Both hands are read back, the tip
   and the entry placed from them, and the receiving part's axis turned with
   the holder's hand since it was measured; the tip's offset across that axis
   must be within half the diametral `clearance` (1 mm: 0.5 mm). Up to three
   corrections of the mover (`robot.go_to_pose` with `tolerance`
   `fine_tolerance` 0.3 mm, passed only to a verb that takes it), its wrist
   roll turned back on the way, else `misaligned`. The two parts' axes more
   than `angle_tolerance_deg` (2 deg) apart is `misaligned` (a hand that
   tilted since the parts were measured).
5. **Approach.** One straight move to 5 mm short of the entry, re-aligned
   there the same way.
6. **Step in** `step` (2.5 mm) at a time with `robot.go_to_pose`, the
   commanded line holding the last lateral correction, to `insert_depth`
   past the entry. After each step (the mover read until it stands still):
   - **lag**: commanded minus measured tool centre along the axis, against
     the median of the last three free steps; over `lag_threshold` (3 mm) is
     contact;
   - **holder**: the holder's tool centre more than
     `holder_shift_threshold` (2 mm) from where it stood when the steps
     began (the push reached it through the parts);
   - **effort**: when the observation carries both arms' joint efforts, a
     joint's residual against the mean of the free windows over
     `effort_threshold` (0.5 N.m) or four sigma.
7. **Seated check** (`verify`, on by default). The depth above is the
   hands' and the in-hand offsets': a part pushed along its jaw pads slides
   back in the hand without either arm lagging (measured, see below). With
   the tip at depth, the mover is moved across the axis by half the
   clearance plus 1 mm, both ways along two directions, and back each time:
   a part in its hole stops at the wall (the mover falls short, or carries
   the holder along, by more than 0.5 mm); one on the outside face does
   not. No probe stopped: `slipped`.
8. **Decide.** Tip within `depth_tolerance` (3 mm) of `insert_depth` and
   seated: `inserted` (a contact there is the bottom; the mover backs off
   3 mm).
   Stopped with the tip under 2 mm past the entry: `misaligned` (the rim),
   the mover goes back to its stand-off. Stopped further in: `stalled`, the
   mover backs off 3 mm. With `release: mover` an inserted mover opens and
   backs out `retreat` (50 mm) along its own approach, and the holder's jaws
   are read again.

## Inputs

- `moving_tip` / `moving_tip_in_tcp`: the moving part's leading point (peg
  tip centre, brick's socket-face centre), world with the mover unmoved since
  it was measured, or in the mover's tool frame.
- `receiving_entry` / `receiving_entry_in_tcp`: the receiving part's entry
  centre (hole mouth, stud-top face centre), likewise for the holder.
- `insertion_axis` (none: tip toward entry), `insert_depth` (0.02 m past the
  entry), `meet_position` (none: the holder stays), `mode` (`holder_still` |
  `both_move`), `mover_arm` (1), `holder_arm` (0).
- `clearance` (0.001 m, diametral), `standoff`, `step`, `signal` (`auto` |
  `lag` | `effort`), `lag_threshold`, `holder_shift_threshold`,
  `effort_threshold`, `depth_tolerance`, `fine_tolerance`,
  `step_max_steps` (0: the connector's), `reach_tolerance` (0.02 m),
  `angle_tolerance_deg`, `fix_roll` (true), `release` (`none` | `mover`),
  `verify` (true), `retreat`, `empty_fraction` (0.04), `settle_period` (0.3 s; 0 reads once).

## Required end states

- `inserted`: both hands hold (or the holder holds the assembly and the
  mover is backed out); the depth is the hands' and the offsets', not a
  sighting. Re-perceive (`verifying-placement`, a wrist close-up) when it
  matters.
- `misaligned`: nothing went in; the mover is at its stand-off with its part.
- `stalled`: the tip is partly in, the mover 3 mm back from where it
  stopped, both hold.
- `slipped`: see `detail`; a part may be on the table.
- `unreachable`: see `detail`.

## The constants, and why

- **`STANDOFF = 0.05` m**: ACT's InsertionPolicy approaches the meet point
  at +-0.10 m and inserts to +-0.05 m: 0.05 m of closing per arm.
- **`STEP = 0.0025` m, `LAG_THRESHOLD = 0.003` m, `EFFORT_THRESHOLD = 0.5`
  N.m, `BASELINE_WINDOW = 3`**: detecting-contact-from-joint-effort's (the
  real YAM threading station's SETTLE_STEP / SETTLE_CONTACT, i2rt's clog
  threshold).
- **`CLEARANCE = 0.001` m**: the task this was checked on (a peg 1 mm under
  its hole); the lateral bound is half of it, floored at 0.2 mm.
- **`FINE_TOLERANCE = 0.0003` m**: passed as `robot.go_to_pose`'s
  `tolerance` on alignment moves and steps (the twin's
  `go_to_pose_cartesian` takes none); a simulator servo's default 4 mm ball
  is four times the whole clearance.
- **`HOLDER_SHIFT_THRESHOLD = 0.002` m**: a choice (see the simulation
  check for what the holder read while the other arm stepped in free air).
- **`ENTRY_MARGIN = 0.002` m, `DEPTH_TOLERANCE = 0.003` m, `BACKOFF =
  0.003` m, `CONTACT_MARGIN = 0.005` m**: a stop within 2 mm of the mouth is
  the rim; a hand short by less than the lag threshold is in; backing off
  one lag stops the push; the straight approach ends beyond a servo's 4 mm
  ball from the rim.
- **`ANGLE_TOLERANCE_DEG = 2`, `ROLL_TOLERANCE_DEG = 1`**: two axes 2 deg
  apart cross by 0.7 mm over a 20 mm depth, more than a 1 mm clearance's
  half; 1 deg of free wrist roll
  swings a tip held 50 mm off the tool axis by 0.9 mm.
- **`PROBE_MARGIN = 0.001` m, `PROBE_RESIST = 0.0005` m**: the seated
  probe goes 1 mm past the half-clearance; half of that, over the 0.3 mm
  fine arrival ball, is a wall. A choice, checked in the twin below.
- **`REACH_TOLERANCE = 0.02` m, `EMPTY_FRACTION = 0.04`**: as
  handing-over-between-arms.

## Connectors and arm_id

- RoboSimStudio: every robot verb used here takes `arm_id` (0 = work arm,
  1 = support arm; an id past the last arm is clamped to it, hence step 1);
  `robot.go_to_pose` takes `tolerance` / `max_steps`, passed only when set;
  the observation carries joint positions, no efforts, so it decides by lag
  and by the holder's displacement.
- The real YAM station bridge (`yam_real.py`, 2026-11-06) drives the left arm
  whatever `arm_id` says and reads no second arm's pose: this skill refuses
  there (`unreachable`, "no separate arm"). On a dual-arm bridge the
  alignment is only as good as the two arms' base calibration against each
  other (a few mm on the YAM station today), far over a 1 mm clearance:
  register the parts against each other (both in one wrist view) before
  trusting it.

## Simulation check

RoboSimStudio YAM station twin (`yam-station-20261103-gsota`, two i2rt YAM
arms, MuJoCo-Warp at 500 Hz, RTX 4090 host), 2026-11-08. The twin has no
parts, so the harness builds them from convex pieces: a 30 x 40 x 30 mm
socket block with a blind 13 x 13 mm square hole 25 mm deep, and a peg
(30 mm cube handle, 12 x 12 x 70 mm shaft): 1 mm diametral clearance, no
chamfers, friction 0.8. The left arm (holder, 0) picks the socket, the right
arm (mover, 1) the peg, each hand's last joint turns its part's axis onto
world +y, then the skill runs with the true peg tip plus an injected in-hand
error, the true socket mouth, `meet_position` 170 mm over the bench and
`insert_depth` 20 mm. Truth from `sim.get_object_pose` after each trial; the
peg is pulled back out, still held, between trials (no re-grasp). Minimum
arm clearance measured from the two arms' collision meshes every 20 ms (the
twin does not collide one arm with the other).

| run | injected tip error | route | truth | peg moved in its hand | min arm clearance |
|---|---|---|---|---|---|
| without the probe | 0 | `inserted` | in: 21.4 mm deep, 0.56 mm off axis, 2.8 deg | 7.4 mm from the tool centre | 0.1 mm |
| without the probe | 1.0 mm | `inserted` | **not in**: on the face (-0.1 mm), 4.7 mm off | slid to 22.5 mm | 0.0 mm |
| with the probe | 0 | `slipped` (0 of 4 probes stopped) | not in: -0.4 mm, 3.8 mm off, 7 deg | slid to 28.5 mm | 0.3 mm |
| with the probe | 1.0 mm | `slipped` (0 of 4) | not in, peg turned 30 deg in the jaws | 35.4 mm, jaw 0.15 | 0.1 mm |
| with the probe | 0.5 mm (after the 30 deg turn above) | `inserted` (1 of 4) | **not in**: peg 85 deg off the socket's axis | 44.8 mm | 0.0 mm |

What this shows, honestly:

- **Success 1 of 3 clean attempts at zero injected error, 0 of 2 with
  0.5-1 mm.** The read-back alignment brought the tip within 0.5 mm of the
  entry's line before every push (by the hands and the offsets), but the
  truth at the end was 0.6-5 mm off: a peg held by its handle tilts and
  slides in the twin's jaws as the moves go on, so the offsets given at the
  start are stale by the push. Re-register the tip in the hand
  (`registering-held-objects`) just before the rendezvous.
- **Neither arm feels a part sliding back along its jaw pads**: tracking lag
  stayed within 0.13 mm of free motion and the holder moved 0.15-0.41 mm
  (under its 2 mm threshold) in every trial, jam or not. The twin's
  observation has no joint efforts, so effort was not checked.
- **The seated probe** turned both slides of the second run into `slipped`
  (the false `inserted` of the first run). It missed a peg already turned
  85 deg in its hand by the trial before (one probe stopped against the
  socket's outside): the skill compares the hands' rotations, not the parts',
  so a part turned in its jaws is invisible to it - re-measure the part's
  axis after anything that may have turned it.
- **The two hands' fingertips came within 0.0-0.1 mm of each other** at full
  depth (the twin lets them pass through): with a 70 mm shaft and grasps at
  each part's middle, the jaws meet. On a real station grip the parts
  further apart along the axis and check with `checking-dual-arm-clearance`.
- One trial took 70-870 s of simulated servo time: `robot.go_to_pose` with a
  0.3 mm `tolerance` runs to the twin's step cap on most alignment and probe
  moves.
- **DUPLO press** (`PARTS=duplo`: two 2 x 2 bricks lying on their sides,
  9.4 mm studs 4.5 mm high, 0.1 mm wall play, rigid, `clearance` 0.2 mm,
  `insert_depth` 4.5 mm, `step` 1.5 mm): 0 of 2 seated (0 and 0.5 mm
  injected). Both routed `slipped` correctly (0 of 4 probes stopped): the
  hands went 4.0 mm, the brick 0.1 and 0.8 mm, 0.45 and 0.25 mm off the studs'
  axis (over the 0.1 mm play) and 2.5-5.5 deg tilted, sliding back in its
  jaws while neither arm lagged (0.01-0.02 mm) nor the holder moved
  (0.27-0.30 mm). Fingertips again at 0.0 mm.
- Not run: `mode: both_move`, `release: mover`.

### Re-run with the sim harness's grip fix (2026-11-09/10)

The stock twin let held parts creep through and pivot in the jaws: MuJoCo friction creep on light bodies, and no
torsional friction on the blade-tip pads (~/yam_gap/skill_sources/SIM_GRIP.md). The same harness (peg 12 mm in a
13 mm blind hole, default 9 trials of injected tip error, holder still) was re-run on GCP RTX PRO 6000 hosts:

| twin | peg in | held socket's axis error over the trials | socket's offset from the left tool centre |
|---|---|---|---|
| grip fix (impratio 100, solimp 0.99), no pad torsion | 0 of 9 | 22.5 -> 78.1 deg | 14.2 -> 22.4 mm |
| + pad torsion (condim 4, 0.004 m), first 7 trials | 1 of 7 (trial 0, 0 mm injected: 17.0 mm deep, 0.75 mm off axis, 4.1 deg); trial 4 (1 mm) part-way, 2.2 mm | 4.1 -> 57.2 deg | 15.8 -> 17.3 mm |

- Pad torsion keeps the parts in place in the jaws between trials: the socket's offset from the tool centre now
  stays within 1.5 mm. The first clean insertion went 17 mm deep.
- Each push still turns the held parts about 10 deg. The skill's push through a 70 mm shaft held at its handle
  puts more torque on the grip than the pads hold. Real jaws may well turn too, so re-register the parts (or
  re-grasp) between attempts rather than reuse offsets.

## Provenance

Adapted from ACT's scripted `InsertionPolicy`
(https://github.com/tonyzhaozh/act/blob/main/scripted_policy.py, MIT): the
two-hand meet point with the parts facing each other on one axis, the
stand-off and the closing along the axis, fixed hand rotations. Changed:
in-hand offsets measured while held and the hands' read-back poses in place
of simulator part poses and ACT's tuned `lift_right`; a lateral alignment on
the holder's read-back entry; stepped closing with contact felt on both arms
(lag, holder displacement, efforts) in place of open-loop interpolation; one
arm per call with explicit `arm_id`; routed exits. ACT's pick phase is not
part of this skill (any grasp skill does it).
