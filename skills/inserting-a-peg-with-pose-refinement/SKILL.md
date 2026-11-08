---
name: inserting-a-peg-with-pose-refinement
description: Inserts a held peg into a perceived hole in three stages - carries
  it to a pre-insert pose with the tip a few centimetres in front of the mouth
  on the hole's axis, refines that pose up to three times from the residual
  between the peg as observed (hand read back times the registered
  tip-in-hand transform) and the axis, then pushes straight in along the axis
  with a stop-on-contact move and judges how far the tip went. Use when the
  hand already holds a peg, pin, plug or shaft whose tip is registered in the
  hand (registering-held-objects), the hole's mouth and inward axis are
  measured (perceiving-functional-features), and the arm does not land
  exactly where it is sent; any axis direction (side or top insertion).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, peg-in-hole, refinement, closed-loop, contact]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion]
    provenance:
      authors: [runtime-si]
      code: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.solve_ik
  required_inputs:
    hole_centre: Vec3
    hole_axis: Vec3
    peg_in_tcp: Se3Pose
  produces_outputs:
    pre_insert_pose: Se3Pose
    tcp_at_mouth: Vec3
    lateral_error: float
    axial_error: float
    angle_error_deg: float
    refinements: int
    correction: float
    depth: float
    tip_position: Vec3
    deflection: float
  exit_conditions:
    aligned: (align_pre_insert run on its own, e.g. before a re-registration) The tip is within the tolerances of the pre-insert pose on the hole's axis; in the example subgraph this route goes on to the push and never leaves.
    inserted: The push took the tip, as held, at least min_depth (10 mm) past the mouth; the hand stays at the end of the push, closed on the peg.
    stopped_short: The push stopped (contact, stall, or the hand shoved deflection_stop off its line) with the tip short of min_depth past the mouth (rim, a hole off its estimate, a jam), or the hold leaves no room to push that far with the fingers clear; the hand backed off to the pre-insert pose, still holding.
    misaligned: After the refinements the tip is still more than lateral_tolerance off the axis (or axial / angle tolerance), or the correction grew past its cap; the hand is at the last pre-insert pose and nothing was pushed.
    unreachable: The pre-insert pose or the end of the push is out of reach by the IK preview, a move was refused, or the hand's pose could not be read; nothing was pushed.
  canonical_scripts:
    - align_pre_insert: scripts/align_pre_insert.py
    - push_insert: scripts/push_insert.py
  examples:
    - title: Align a registered peg on a perceived hole's axis, refine, push in
      path: examples/canonical_subgraph.json
---

# inserting-a-peg-with-pose-refinement

ManiSkill's PegInsertionSide solution in three stages: put the peg on the
hole's axis a little in front of it, look at where the peg really is and
correct the pose until it is on the axis, then push straight in. The
refinement is the reusable part: whatever makes the peg miss its pre-insert
pose - the arm's tracking lag or sag, a grasp that seated the peg a few
millimetres off - is measured and taken out before anything touches.

## Inputs

- `hole_centre`, `hole_axis` (world, Vec3): the mouth's centre and the
  direction into the hole. `perceiving-functional-features`'
  `perceive_aperture` gives them as `aperture_center` and `fixture_axis`.
- `peg_in_tcp` (Se3Pose): the peg's tip frame in the tool frame, local z out
  through the tip along the peg's axis. `registering-held-objects` gives it as
  `feature_in_tcp` for a `tip` feature.
- Align: `standoff` (30 mm), `refinements` (3), `lateral_tolerance` (2 mm),
  `axial_tolerance` (5 mm), `angle_tolerance_deg` (3), `fine_tolerance`
  (1.5 mm; 0 leaves the connector's default), `check_reach`, `arm_id`.
- Push: `pre_insert_pose` (from align), `insert_depth` (50 mm), `min_depth`
  (10 mm), `finger_clear` (15 mm), `push_step` (5 mm), `deflection_stop`
  (1 mm; 0 turns it off), `back_off` (true), `check_reach`, `arm_id`.
  `arm_id` is passed to the robot tools only when nonzero.

## The motion

`align_pre_insert`:

1. Read the hand. The peg's tip is the hand times `peg_in_tcp`. The target
   tip frame has its z turned onto `hole_axis` by the smallest rotation (the
   peg's roll about the axis stays as the grasp left it) and its origin
   `standoff` in front of the mouth. The hand's command is that target times
   `peg_in_tcp`'s inverse (ManiSkill: `goal * peg_init^-1 * grasp`, backed off
   along the peg).
2. Preview the command with `robot.solve_ik`; out of reach is `unreachable`.
3. Carry the hand there (`robot.go_to_pose`). A refusal, or a hand more than
   15 mm off, is `unreachable`.
4. Refine, up to `refinements` times: read the hand back, compute the observed
   tip, and stop when it is within the tolerances. Otherwise the correction is
   ManiSkill's `delta = target * observed_tip^-1`, applied to the *command*
   (not to the read-back), so a steady tracking bias is compensated rather than
   chased; move there with `tolerance=fine_tolerance`. A command shifted more
   than 20 mm or turned more than 10 deg from the first is `misaligned`: the
   measurement, not the arm, is off.
5. Judge the last residual: within tolerance is `aligned` (an internal route
   that goes on to the push), otherwise `misaligned`.

`push_insert`:

1. Read the hand. The push sends the tip `insert_depth` past the mouth,
   capped so the tool centre stays `finger_clear` in front of it (the hold
   decides how deep a peg can go without the fingers meeting the rim).
2. Preview the end with `robot.solve_ik`: a push that stalls at the edge of
   reach looks exactly like a jam.
3. Push along the axis from the refined pre-insert command in `push_step`
   (5 mm) steps, each a `robot.move_cartesian_until_contact` to the start plus
   its share of the travel (a step that lands short inside a connector's
   arrival ball does not shorten the next). A contact or stall outcome, or a
   station's "did not reach its target" error, ends the push; a refusal with
   the hand unmoved is `unreachable`. So does a touch that neither stalls nor
   raises: the hand read back more than `deflection_stop` (1 mm) further
   across the axis from its command than after the first step (`deflection`
   is the largest such shift).
4. Judge from the hand read back: the tip, as held, at least `min_depth` past
   the mouth is `inserted`; otherwise `stopped_short` and the hand backs off to
   `pre_insert_pose`.

## Composing it

```
perceiving-functional-features (perceive_aperture) -> hole_centre, hole_axis
registering-held-objects (tip)                      -> peg_in_tcp
align_pre_insert ── aligned ──> push_insert ── inserted ──> verifying-insertion-by-probe-push (axis = hole_axis)
     │                              └── stopped_short ──> searching-insertion-spiral (mouth_position = tcp_at_mouth)
     └── misaligned ──> registering-held-objects again (prior_feature_in_tcp) ──> align_pre_insert
```

- **Re-registering at the pre-insert pose.** `registering-held-objects` says
  it is meant to be run again "at a pre-contact pose before a fixture
  engagement". Its fresh `feature_in_tcp` fed back into `align_pre_insert`
  re-plans from where the hand is (a short move) and refines again: a peg that
  slid in the fingers since the grasp is caught there. The script itself only
  reads the arm, so within one run it corrects the arm, not the hold.
- **A camera that is a few millimetres off.** The refinement puts the peg on
  the *estimated* axis; it cannot see the hole's error. A push that stops on
  the rim is `stopped_short`, and `searching-insertion-spiral` probes around
  the estimate. That skill works on a vertical axis and measures the tool
  centre's depth, so give it `tcp_at_mouth` (where the tool centre is when the
  tip is at the mouth) as `mouth_position`.
- **Seated?** After `inserted`, `verifying-insertion-by-probe-push` with
  `axis = hole_axis` pushes a few millimetres more and reads whether the hand
  moved.
- `computing-feature-mating-poses` / `executing-feature-mating` plan a
  collision-aware approach and mate with cuRobo; this skill is the closed-loop
  alternative that needs no planner and corrects the arm's landing error.

## What changed from ManiSkill, and why

- **No simulator poses.** ManiSkill reads `env.peg.pose` and `env.goal_pose`
  every refinement. Here the hole comes from perception and the peg is the
  hand read back times the registered tip-in-hand transform; on a real YAM
  (no force sensor, 12 mm per-move bound) that is what is observable.
- **The tip, not the peg's centre.** The registration produces a tip frame;
  the pre-insert pose puts the tip `standoff` (30 mm) in front of the mouth
  instead of ManiSkill's peg-centre offset of 10 mm plus the half-length.
- **The axis only.** The target turns the peg's axis onto the hole's and keeps
  its roll; ManiSkill matches the goal's full rotation. A YAM station twin's
  IK leaves the wrist roll free, and a round peg does not care; a square peg
  needs its roll from the grasp.
- **Refinement with a stop and a verdict.** ManiSkill makes exactly three
  corrections and pushes regardless. Here they stop inside the tolerances,
  each asks the connector for a fine arrival ball (its default ball would
  report a 2 mm correction as already arrived), the correction is capped, and a
  residual left outside tolerance is `misaligned` instead of a push.
- **The push from the refined pose, in steps, on contact.** ManiSkill pushes
  to the unrefined insert pose plus 5 cm with a velocity-limited planner. Here
  the push starts from the refined command (the arm's steady landing error is
  still there), goes in 5 mm stop-on-contact steps (one fast push wedged the
  peg in the twin, see below), ends on a sideways shove of the hand as well as
  on contact (in the twin the steps never reported contact while the peg slid
  in the fingers), is capped by the fingers, and is judged by the
  tip's depth *as held*: a peg that slid back in the fingers reads deeper
  than it is.
- The grasp (ManiSkill's off-centre grasp, so the peg protrudes) is left to a
  grasp skill: hold the peg far enough back that `insert_depth` plus
  `finger_clear` fits behind the tip.

## The constants, and why

- `STANDOFF = 30 mm`, `REFINEMENTS = 3` (ManiSkill), `LATERAL_TOLERANCE =
  2 mm` (the twin's 1.5 mm fine ball plus margin; set at or under the radial
  clearance), `AXIAL_TOLERANCE = 5 mm` (the push covers it),
  `ANGLE_TOLERANCE_DEG = 3` (the twin servo's rotation ball),
  `FINE_TOLERANCE = 1.5 mm` (the twin servo's "fine leg"), `MAX_CORRECTION =
  20 mm / 10 deg`, `REACH_TOLERANCE = 15 mm` (a real YAM station's 12 mm per
  move plus margin).
- `INSERT_DEPTH = 50 mm` (ManiSkill's final push; set it to the hole's depth
  plus a couple of millimetres for a blind hole), `MIN_DEPTH = 10 mm` (a real
  YAM threading station's "through" test), `FINGER_CLEAR = 15 mm` (closed
  linear_4310 fingertips reach 10.2 mm past the tool centre, plus 5 mm),
  `PUSH_STEP = 5 mm`, `DEFLECTION_STOP = 1 mm` (both measured below: a touch
  shoved the hand 1.4-2.6 mm across the axis, free travel stayed within
  0.6 mm).

## Checked in simulation

RoboSimStudio, a YAM station twin (left arm, linear_4310 fingertips 10.2 mm),
a 20 mm x 80 mm cylinder peg (30 g, mu 1.0) standing on the table, taken
top-down 20 mm under its top by a scripted grasp (tip 61 mm under the tool
centre) and lifted 10 cm; a fixed fixture 10 cm away with a square blind hole
30 mm deep (the table is its floor), axis straight down. The simulator stood
in for perception: the peg's pose gave `peg_in_tcp` (the role of
`registering-held-objects`), the scene layout the hole's mouth (the role of
`perceive_aperture`), plus an optional error on it. The graph registered the
peg after the grasp, aligned, registered it again at the pre-insert pose,
aligned again with the fresh registration (the re-registration loop above),
pushed (`insert_depth` 32 mm), then ran `verifying-insertion-by-probe-push`
(10 mm down) after `inserted` or `searching-insertion-spiral` after
`stopped_short`. Truth is the peg's pose from the simulator. One run per row.
First with `insert_depth` 32 mm (the hole's depth plus 2 mm), before the
deflection stop existed:

| run | hole (radial clearance) | mouth estimate | align / align again | push | true tip depth, off-axis | peg slid in the fingers |
|---|---|---|---|---|---|---|
| r30 | 30 mm (5 mm) | exact | aligned, 0 refinements, 2.0 mm / 0 refinements, 1.4 mm | inserted, 30.3 mm as held | 29.9 mm (floor), 0.2 mm | 0.5 mm |
| r24 | 24 mm (2 mm) | exact | 0 refinements, 2.0 mm / 2 refinements, 1.7 mm | inserted, 28.6 mm | 28.9 mm (1 mm off the floor), 0.8 mm | 0.6 mm |
| r24 tight | 24 mm (2 mm), `lateral_tolerance` 1 mm, `fine_tolerance` 0.7 mm | exact | 2 refinements, 0.9 mm / 2 refinements, 0.8 mm | inserted, 28.6 mm | 28.7 mm, 0.6 mm | 0.6 mm |
| r24 err | 24 mm (2 mm) | 4 mm off | 0 refinements, 1.6 mm / 2 refinements, 1.8 mm | **inserted (wrong)**, 27.9 mm | -0.7 mm (on the rim), tilted 7 deg | 27.5 mm |

- **Refinement.** The first carry landed 1.6-2.0 mm off the axis (the twin's
  servo ball), just inside the default 2 mm, so it often made no correction;
  after a re-plan with the fresh registration 1-2 corrections ran. With a
  1 mm tolerance and a 0.7 mm fine ball, two corrections brought the tip to
  0.8-0.9 mm by the read-back; after the re-registration it was 0.7 mm off
  the hole's axis in truth (before it, with the grasp-time registration,
  4.3 mm).
- **Re-registering at the pre-insert pose mattered more.** Between the grasp
  and the pre-insert pose the peg moved in the fingers: its tip by 0.8-4.8 mm
  across the axis, and up to 1.7 deg, in all four runs. Planning with the
  grasp-time registration put it that far off.
- **The push in steps.** Before `PUSH_STEP`, one stop-on-contact push of
  60 mm (it ran in 25 control steps) wedged the peg 8.5 mm into the 24 mm hole
  and, in one of two runs, 20.8 mm into the 30 mm one, the peg sliding 18 and
  10 mm up in the fingers while the hand reported 25.8 / 30.6 mm (`inserted`,
  wrong); the other 30 mm run went in 25.5 mm. Lowered
  in 3 mm fine steps the same peg went to the floor without slipping; in 5 mm
  steps (the rows above) it reached the floor or within 1 mm of it, slipping
  0.5-0.6 mm, and the as-held depth agreed with the truth to 0.1-0.4 mm.
- **What failed: a peg that slides in the fingers.** With the mouth 4 mm off
  (r24 err) the tip landed on the rim and the twin's weak pinch let the peg
  slide 27.5 mm up the fingers while the arm kept moving, so no step stalled
  and the hand's travel read `inserted`. With the default `insert_depth`
  (50 mm, capped at 44 mm by the fingers) the same happened in all three
  24 mm runs below without the deflection stop, even with the mouth exact.

Then with the defaults (`insert_depth` 50 mm, no hint of the hole's depth),
before and after `DEFLECTION_STOP` (one run per row; truth after the push):

| run | hole | mouth estimate | push without the deflection stop | push with it (1 mm) | after it |
|---|---|---|---|---|---|
| 24 | 24 mm | exact | **inserted (wrong)**, 41.0 mm as held; tip on the rim, tilted 20 deg, slid 39 mm | stopped_short on the rim (contact + 2.4 mm shove), backed off | spiral: **inserted (wrong)**, tip 0.7 mm under the mouth |
| 24 again | 24 mm | exact | - | inserted, 30.6 mm as held (1.9 mm shove at the floor); truth 30.0 mm, slid 0.8 mm | probe push: not_seated (wrong) |
| 24 err 2 | 24 mm | 2 mm off | **inserted (wrong)**, 41.1 mm; on the rim, slid 40 mm | stopped_short at 2.2 mm (1.7 mm shove), backed off | spiral: **inserted (wrong)**, on the rim |
| 24 err 4 | 24 mm | 4 mm off | **inserted (wrong)**, 40.4 mm; on the rim, slid 40 mm | stopped_short at 0.6 mm (1.7 mm shove), backed off | spiral: **inserted (wrong)**, on the rim |
| 30 | 30 mm | exact | inserted, 43.0 mm; truth ~30 mm (floor), slid 13 mm | inserted, 30.6 mm (2.6 mm shove at the floor); truth 30.0 mm, slid 0.6 mm | probe push: not_seated (wrong) |
| 22 tight | 22 mm (1 mm), tolerances 1 / 0.7 mm | exact | inserted, 40.9 mm; truth 30 mm, slid 11 mm | inserted, 30.7 mm (2.0 mm shove); truth 30.0 mm, slid 0.7 mm | probe push: not_seated (wrong) |

- **The deflection stop.** Every step in the twin reported `reached`, also
  while the peg slid tens of millimetres up the fingers; the first touch
  showed only as the hand shoved 1.4-2.6 mm across the axis (free travel
  stayed within 0.6 mm). Stopping there, the push's verdict was right in 6 of
  6 runs (3 of 5 wrong without it), and on the floor the as-held depth agreed
  with the truth to 0.6-0.8 mm instead of 11-13 mm too deep. Whether a real
  YAM's hand is shoved the same way is not measured; there a contact raises
  "did not reach its target" (read as a stop), and `deflection_stop = 0`
  turns the check off.
- **The hand-offs after it fail in this twin.** `searching-insertion-spiral`
  after `stopped_short` said `inserted` at its first probe in all three runs
  while the tip sat on the rim: it reads the tool centre's depth, and the peg
  slid about 11 mm up the fingers. The probe push after `inserted` said
  `not_seated` every time (below). Judging from the hand cannot see a slip;
  on a real YAM the pads pinch harder than the twin's (see
  threading-a-cable-tip-through-a-ring), but check the result with a camera
  (e.g. detecting-subtask-success) where a slip is possible.
- **The probe push after `inserted`** said `not_seated` in all four runs: the
  hand went 6.5-6.6 mm further, the peg sliding 5.3-6.2 mm up in the fingers
  against the floor. Right for r24 err (the peg was on the rim), wrong for the
  three that were at the floor. That skill already warns that a slip reads
  like a part that went further; in this twin the pinch is too weak for it.
- Nothing here ran on a real YAM, and no camera was in the loop.

## Provenance

Adapted from ManiSkill's
`mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py`
(`solve`: align, refine three times, insert), Apache-2.0, Copyright the
ManiSkill authors, with the changes above; each script's header lists them.
