---
name: classifying-failure-from-proprioception
description: Classifies the last moments of a manipulation step from
  proprioception alone - the gripper opening, the tool-centre pose and the arm
  joints over a short window (passed in from earlier nodes, sampled here, or
  both) - into stall, never_gripped, fumble, drop, stuck_mid_air or ok, with
  VoLoAgent's rolling-window rules re-based on the work surface and the
  gripper's stroke. Use when a graph must know cheaply, with no camera and no
  force sensor, whether a grasp, lift or carry went wrong and how, before
  spending a perception or VLM call or choosing a recovery.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [failure-detection, proprioception, grasp, drop, stall, gripper]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, failure-detection]
    provenance:
      authors: [runtime-si]
      code: https://github.com/NVlabs/VoLoAgent/blob/b4e623079ca8498a16bcd5016920d71f76c44d30/vlm_orchestrator/failure_handlers/signal_detector.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2606.07723
gap:
  requires: {connector: [robot.describe_gripper, robot.describe_workspace]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_gripper
    - robot.describe_workspace
  required_inputs: {}
  produces_outputs:
    state: string
    reason: string
    confidence: float
    samples: int
  exit_conditions:
    ok: No failure signal. `state` says which healthy state - held (closed on something and elevated, moving or still on its target), placed (opened after a descent, after a long close, or low), approaching (closed, low), open (no strong signal).
    stall: The tool centre barely moved over the window (z range under 30 mm, under 2 mm per sample) while it was more than lag_tolerance short of its commanded target, or expect_motion was set and no target was given; and it was not holding something in the air.
    never_gripped: The jaw closed to within 3 mm of its closed stop without having held anything in the window (closed on nothing), or the hand never left its open stop at all (upstream's rule, gated by chunks_elapsed).
    fumble: Repeated open / close - at least 6 grip transitions with the tool centre low, or at least 4 anywhere (gated by chunks_elapsed).
    drop: The hand opened while the tool centre was more than 60 mm over table level, or a held object's jaw closed to its stop with the hand still shut above carrying height (a slip).
    stuck_mid_air: The hand is closed on something above carrying height but the arm is frozen short of its commanded target.
    unknown: The arm could not be read, there were fewer than two samples, or no surface height was given or described; nothing was classified.
  canonical_scripts:
    - classify_failure: scripts/classify_failure.py
  examples:
    - title: Classify the hand's state from a short window it samples itself
      path: examples/canonical_subgraph.json
---

# classifying-failure-from-proprioception

What went wrong, from what the arm can feel about itself.

## How a graph uses it

The skill reads a window of samples, each one
`{"ee_position": Vec3, "gripper_fraction": float (1 = open), "joints": [...],
"target_position": Vec3 (optional), "commanded_closed": bool (optional)}`:

- `history`: samples earlier nodes recorded (one per control segment or per
  node boundary; `history` in the output is the window, so a later call can
  continue it);
- plus `n_samples` (4) it takes now from `robot.get_observation`,
  `sample_interval_s` (0.05 s) apart, stamped with `target_position` (or the
  last history sample's target) and `commanded_closed` when given.

The last `window` (16, upstream's) samples are classified. With only its own
samples, after a blocking move, the window is a still arm: the jaw signals
(never_gripped, slip-drop, held) work from one moment, but stall and
stuck_mid_air need the commanded target (`target_position`) and drop-by-opening,
fumble and placed need a history that covers the step.

Heights are over **table level**: `surface_z` (or
`robot.describe_workspace().surface_z`) plus the fingertip reach from
`robot.describe_gripper` (the tool-centre height with the fingertips on the
table). The opening in metres is `gripper_fraction x span_m`; the closed stop is
`width_fit.at_closed_m` (else `min_grasp_width_m`, else 0).

## The rules, in upstream's order

1. **stall / stuck_mid_air**: frozen (z range < 30 mm, mean travel < 2 mm per
   sample) while short of the target by more than `lag_tolerance` (15 mm), or
   `expect_motion` with no target. Holding above carrying height ->
   stuck_mid_air, else stall.
2. **drop (slip)**: a held jaw went to its stop with the hand shut, above
   carrying height.
3. **never_gripped**: the jaw ended at its stop without holding anything in the
   window; or (upstream) no transitions and never closed, once
   `chunks_elapsed >= 30`.
4. **fumble** at the table: >= 6 transitions, tool centre below placing height.
5. **ok / held**: closed above carrying height, moving (z range > 30 mm), or
   still and on target. Frozen and short of the target -> stuck_mid_air.
6. **fumble**: >= 4 transitions.
7. Open: descended > 5 mm in > 40 % of the 15 samples before the last opening
   -> ok / placed; opened above drop height -> **drop**; opened after > 20
   consecutive closed samples -> ok / placed; low -> ok / placed; else ok / open.
8. Closed and low -> ok / approaching.

`chunks_elapsed` (999 = judge everything) gates stall (10), fumble (15) and the
never-closed rule (30) as upstream, for a caller that classifies while a step
is still running.

## The constants, and why

- **`WINDOW = 16`, `STALL_Z_RANGE = 0.03` m, `STALL_SPEED = 0.002` m/sample,
  fumble 6 / 4 transitions, the 5 mm / 40 % / 15-sample placement rule, the
  20-sample open-was-closed rule, the chunk gates, the confidences**: VoLoAgent's,
  calibrated on 89 robolab episodes (failure F1 0.75). A VoLoAgent sample is one
  VLA action chunk; segments of about a centimetre keep the speed threshold
  meaningful.
- **Heights** re-based from robolab's absolute z (table level 0.22 m) to offsets
  over table level: `DROP_DZ = 0.06` (robolab drop_z - table_z),
  `PLACE_DZ = 0.03` (robolab place_z - table_z), `CARRY_DZ = 0.03` (LIBERO's
  carry_z - table_z; robolab's 0 calls a hand closed at grasp height "carrying").
  The fumble-at-table bound is placing height (upstream: table_z itself, which a
  tool centre held up by its fingertips does not go under).
- **Jaw "closed"** = more than 3.2 % of the stroke off the open stop (upstream's
  never-closed width -0.05 over its closed finger sum -pi/2), 3.0 mm on the
  linear_4310's 94 mm stroke, floored at 2 mm. Upstream thresholds a commanded
  (or measured) closure at 0.5, which on a 94 mm hand calls a 50 mm object held
  "open"; a sample's `commanded_closed` still wins when given.
- **Empty** = within `EMPTY_MARGIN = 3` mm of the closed stop
  (verifying-grasps: an empty linear_4310 close measured 1.8 mm).
- **`LAG_TOLERANCE = 0.015` m**: no tighter than a real YAM station's 12 mm
  per-move bound (a servo arrives within 4 mm).

## Changed from upstream, and why

- **Stall needs a reason to move.** VoLoAgent watches a VLA that is always
  commanding motion, so "frozen" is a stall. A graph's moves block, so the arm
  is still whenever a node runs; here frozen is a stall only with tracking lag
  (the measured tool centre short of its commanded target, the YAM's stand-in
  for a force sensor) or `expect_motion`.
- **The jaw against its stroke**: never_gripped also covers a close on nothing
  (the YAM's commonest grasp failure), and a held object's jaw collapsing to the
  stop with the hand shut is a drop even without an opening.
- **stuck_mid_air** is checked with the stall (upstream's stall rule fires first
  and makes it nearly unreachable).
- **mean_joint_speed** is the mean joint-space step (upstream computed the
  change of joint deltas and did not use it); reported, not used.
- Upstream's done / progress / failure becomes one exit per class; ok's `state`
  keeps upstream's done / progress label.

## Measured in a YAM station twin

16 staged episodes, left arm, the station cube (31 mm between the closed
jaws), every move a 1 cm Cartesian segment with one `robot.get_observation`
sample after it (the target stamped on it), then the skill with that history
plus its own 4 samples, defaults throughout (`describe_workspace` surface
0.750 m, fingertip reach 10.2 mm). Truth is what the cube did
(`sim.get_object_obb`), not what the episode was meant to do:

| truth \ predicted | ok | never_gripped | drop | stall | stuck_mid_air | fumble |
|---|---|---|---|---|---|---|
| ok (held, cube lifted 90-96 mm) | **6** | 0 | 0 | 0 | 0 | 0 |
| never_gripped (closed on air) | 0 | **4** | 0 | 0 | 0 | 0 |
| drop (opened 105 mm over table level) | 0 | 0 | **1** | 0 | 0 | 0 |
| stall (pushed 40 mm into the table) | 0 | 0 | 0 | **5** | 0 | 0 |

- ok: 2 top-down grasp-and-lifts (state held, jaw 31.4 mm) and the four
  grasps sent **25 mm** off the cube (+/- along each horizontal jaw axis): the
  twin's linear_4310 caught the cube every time (pulled it 25 mm into the jaws
  along the closing axis; along the blade width the 25 mm still left the blade
  over the cube) and lifted it 90-96 mm, so the skill's ok / held was right
  and the staged "never_gripped" label was not.
- never_gripped: grasps 35 and +/-45 mm off along the blade width and 45 mm
  along the closing axis; the jaw closed to 0.0 mm and nothing lifted.
- drop: lift 10 cm, open, lift 3 cm more (opening at 105 mm over table level).
- stall: closed hand (3 spots) and open hand (2), 10 more pushes after the
  stop; the tool centre stopped 2.8 mm over table level, z range under 0.1 mm,
  0.8-1.5 mm per sample, 51-72 mm short of the target.
- stuck_mid_air and fumble were not staged (unit tests only).

Surprises: pressing a closed hand into the table pried the jaw open by up to
3.0 mm (exactly `EMPTY_MARGIN`), which made one frame read "held" and set the
slip signal; the stall rule runs first so it did not matter. The arm also slid
up to 50 mm sideways along the table while pushing, so `lag_final_m` mixes
the vertical block with that drift. Each stalled push spent the servo's whole
step budget (one stall episode 400-650 s, about 15 000 steps).

## Provenance

Adapted from VoLoAgent's `FailureSignalDetector` / `SignalConfig`
(`vlm_orchestrator/failure_handlers/signal_detector.py`, NVIDIA, Apache-2.0).
It reports what the hand did, not what the object did: re-perceive before
relying on where an object went.
