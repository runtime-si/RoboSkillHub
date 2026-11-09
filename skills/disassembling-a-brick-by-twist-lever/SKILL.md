---
name: disassembling-a-brick-by-twist-lever
description: Removes a seated LEGO or DUPLO brick with a parallel gripper and
  no force sensor - ICL's disassembly twist (grip, seat the grip, rotate 14 deg
  about one bottom edge so the stud rows let go one at a time, lift 15 mm with
  the twist kept), the edge being the one under a finger, commanded in small
  steps and read back so tracking lag (or joint effort) tells a brick that holds
  from one that lets go; with a straight pull and a wiggle (alternate small
  levers about both edges) as strategies and fallbacks, a regrasp after a slip,
  and the wrist depth camera reading the seat and its neighbouring cells before
  and after. Use when a brick must come off a baseplate or another brick at a
  cell known from the stud grid (computing-brick-pose-from-stud-grid's seat).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [lego, duplo, disassembly, lever, contact, wrist-camera]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/ros_nodes/lego_task_planning_node.cpp
      upstream_license: MIT
      paper: arXiv:2309.02354
gap:
  requires: {connector: [robot.grasp_frame, robot.describe_gripper, robot.set_grip]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.get_gripper
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.solve_ik
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
    - robot.set_grip
  required_inputs:
    seat_position: Vec3
  produces_outputs:
    strategy: str
    attempts: int
    attempt_log: str
    pivot_side: str
    twist_deg: float
    peak_lag: float
    releases: str
    release_count: int
    seat_lag: float
    holding: bool
    top_height_change: float
    neighbour_change: float
    samples: int
    verified_by: str
  exit_conditions:
    removed: The hand lifted free, the jaws read at the brick's width, and the wrist depth reads the seat more than half a brick height under the brick's top (and less than 1.5) with no neighbouring cell moved; the brick is in the hand at the clearance height.
    stuck: Every strategy tried stalled - for three steps in a row the hand stood lag_threshold further behind its command than in free motion (or a joint effort jumped) and advanced less than half its step, or the lever's tilt stood 4 deg short; the hand went back to the grasp, opened and rose, the brick left on the seat.
    slipped: The last strategy lifted free but the jaws closed past the brick's width, or the wrist depth still reads the brick on its seat (regrasped and the next strategy tried while any were left); the jaws are open over the seat. A load that cleared during that attempt is named in detail as the grip letting go, not counted as a release.
    neighbour_lifted: The seat reads more than 1.5 brick heights low (the brick under it came along) or a neighbouring cell's stud top moved more than max(half a stud height, 2 mm) between the views before and after; still holding.
    missed: Before anything was gripped - the wrist depth finds no brick top at the seat, the open jaws stopped 3 mm or more over the grasp, or the jaws closed past the brick's width.
    unverified: Lifted free, but the seat was not read (verify false, or no wrist camera / reachable view / 3 depth readings); detail says whether the jaws hold something.
    unreachable: The grasp, or every strategy's motion, failed the IK preview, or a move was refused.
  canonical_scripts:
    - disassemble_brick: scripts/disassemble_brick.py
  examples:
    - title: Lever a DUPLO 2x2 off a baseplate, straight pull as the fallback
      path: examples/canonical_subgraph.json
---

# disassembling-a-brick-by-twist-lever

Take a seated brick off its studs with a parallel gripper and no force
sensor: pry it about one bottom edge so the far stud row lets go before the
near one, then lift. A straight pull is the fallback, a wiggle the option for
bricks that hold; the wrist depth camera decides whether the brick came off.

## How it composes

- **Before it**: the seat. `seat_position` is the world centre of the brick's
  bottom face (the top of the plate or brick under it, without studs) and
  `seat_yaw_deg` its row heading: computing-brick-pose-from-stud-grid's
  `seat_position` / `seat_yaw_deg` / `brick_studs` / `system` outputs plug in
  directly (layer 1 is a brick on the plate). Refine the seat from a wrist
  close-up first if the plate calibration is a few millimetres off.
- **After it**: `removed` holds the brick at the clearance height. In the twin
  the brick then hung turned in the fingertips (38-40 deg, 15-19 mm off
  centre): measure the held brick again (registering-held-objects /
  perceiving-objects) before pressing-a-brick-onto-studs places it.
  `stuck` / `slipped` leave the brick on the seat with the jaws open over it:
  retry with `strategy: wiggle`, a deeper `grasp_depth`, or pin the structure
  with the other arm (holding-steady-while-other-arm-acts). `neighbour_lifted`
  still holds what came off: put it back or re-perceive the build.

## The motion

1. **Look before.** Aim the wrist camera straight down at the seat from
   0.20 m (then 0.25 m, then with the hand's own rotation), IK-preview and
   clearance-gate each view, and read depth at points on the brick's top
   between its studs and at the stud centres of the cells around it. A seat
   that reads half a brick height low has no brick: `missed`, nothing gripped.
2. **Grasp** (computing-brick-pose-from-stud-grid's grasp): top-down, the jaws
   across the short side (`close_axis`), the fingertips `grasp_depth` under
   the top (DUPLO 13 mm, LEGO 5 mm, 6 mm for 1-wide), from
   `robot.grasp_frame` where served (rotation, then the flipped one) after an
   IK preview of the grasp and of its hover 4 cm up. Hover, turn the jaws onto
   the faces if they are more than 2 deg off (finishing on the last joint where
   the arm's solver leaves the wrist roll free), descend in fine steps folding
   in what the hand misses by in the plane, close (`robot.set_grip` to the
   brick's width less 2 mm, else `robot.close_gripper`), and press 1 mm down to
   seat the grip (ICL descends 2.8 mm into its grab pose with its own tool).
   The open jaws stopping 3 mm over the grasp, or closing past the brick's
   width, is `missed`.
3. **`strategy: lever`** (ICL's disassembly twist): rotate the hand about the
   brick's bottom edge under one finger (`pivot_side`; `auto` tries the edge
   whose whole motion passes the IK preview first) by `twist_deg` (14, ICL's
   `Twist_Deg`) in `twist_step_deg` (2 deg) fine steps, then lift
   `lift_height` (15 mm, ICL's mode 5) keeping the tilt, along the brick's own
   tilted axis (`lift_along: brick`; `vertical` is ICL's world-z lift, which
   stalled in the twin, see the constants).
   **`straight`**: lift `lift_height` straight up in quarter-stud-height steps.
   **`wiggle`**: lever about each edge in turn to half of `wiggle_deg` and
   back, then to all of it and back, then lift straight.
4. **Feel it** (no force sensor): every fine step is read back. A step is
   *loaded* when the hand stands `lag_threshold` further behind its command
   than in free motion (half a step by default), or where the observation
   carries joint efforts, when one jumps over max(0.5 N.m, 4 sigma)
   (detecting-contact-from-joint-effort's test). *Held* is loaded and advancing
   less than half the step, or the tilt reading 4 deg short; three held steps
   in a row stop the strategy (`stuck` if it was the last). A load that clears
   is a *release* - the studs letting go - but only once the jaws are known to
   still hold the brick: a fingertip slide clears the load the same way.
5. **After a strategy**: rise to the clearance, level the hand, read the jaws.
   Empty jaws are `slipped` (regrasp and the next strategy, unless
   `regrasp_on_slip: false`). Otherwise look again: the seat more than half a
   brick height low is `removed`; more than 1.5 low means the brick under it
   came too, and a neighbouring cell moving more than max(half a stud height,
   2 mm) between the two looks is `neighbour_lifted`; the brick still reading
   on its seat is `slipped`.
6. **Fallbacks**: `fallbacks` (default `straight` after a lever or wiggle) run
   after `stuck`, `slipped` or a refused step, from the grasp. When every
   strategy has failed the jaws open over the seat and the hand rises.

## Inputs

- `seat_position` (required), `seat_yaw_deg` (0), `brick_studs` ([2, 2]).
- `system`: `duplo` (default; 16 mm pitch, 9.4 mm studs 4.5 mm high, 19.2 mm
  bricks) or `lego` (8 mm, 4.8 mm, 1.7 mm, 9.6 mm); `stud_pitch`,
  `stud_height`, `stud_diameter`, `brick_height` override it.
- `strategy` (`lever`, `straight`, `wiggle`), `fallbacks` (a list or a
  comma-separated string; `""` for none), `pivot_side` (`auto`, `+`, `-`: the
  edge under the finger on the jaws' +y / -y side), `close_axis` (`short`,
  `long`, `x`, `y`), `twist_deg`, `twist_step_deg`, `wiggle_deg` (defaults to
  `twist_deg`), `lift_height`, `lift_along`, `approach_height`, `grasp_depth`,
  `seat_press`, `grip_squeeze`.
- `lag_threshold`: default half a step, the twin's; on a real YAM set it from a
  free-air run (its free lag is millimetres).
- `regrasp_on_slip` (true), `verify` (true), `compare_before` (true),
  `camera`, `check_reach`, `reach_tolerance`, `yaw_tolerance_deg`,
  `roll_limits`, `arm_id` (passed only when nonzero). The fine-move knobs
  (`tolerance`, `max_steps`) are passed only to connectors that take them.

## Outputs

`route`, `detail`, `strategy` (the last tried), `attempts`, `attempt_log`,
`pivot_side` (`+x`/`-x`/`+y`/`-y` in brick axes, `both` for wiggle),
`twist_deg` (the tilt read back), `peak_lag` (the largest lag over the free
baseline, m), `releases` / `release_count` (where loads cleared with the brick
still held: strategy, tilt, height of the hand over its start), `seat_lag`
(how far the 1 mm grip press stood short), `holding`, `top_height_change`
(the seat's depth reading after, minus the brick's top), `neighbour_change`,
`samples`, `verified_by` (`wrist_depth` or empty).

## Simulation check

SIM_SECTION

## Limits

- **The clutch is not a real clutch.** The twin's stud force is a constant per
  stud over its engagement (see above), not a measured interference fit, and
  real LEGO/DUPLO walls flex where the twin's are rigid with low studs. Which
  strategy wins on real bricks, and at what force, is not established here.
- **The grip is the limit.** Every failure in the twin was the fingertips
  sliding, never the arm running out of force; on a real YAM the same is
  likely (small tapered fingertips, 13 mm into a DUPLO wall). Rubber pads or a
  deeper grasp help; a 1-wide LEGO brick may not take a lever at all.
- **Lag on a real YAM** is millimetres in free air
  (detecting-contact-from-joint-effort measured 5.4 mm): the default
  threshold is the twin's 0.56 mm; pass `lag_threshold` from a free-air run,
  or rely on joint efforts where the connector reports them.
- **Releases are the hand's, not the bricks'.** A release is a load that
  cleared while the jaws still hold something; what came off is decided only
  by the wrist depth after.
- The brick turns in the jaws after a lever: re-measure before placing it.
- `neighbour_lifted` only sees cells beside the brick (and the one under it):
  a structure that rocked and settled back elsewhere is not checked.
- Not run on a real YAM.

## The constants, and why

- **`TWIST_DEG = 14`**: ICL's `Twist_Deg` (config/user_config.json).
  **`TWIST_STEP_DEG = 2`**: under a millimetre of tool travel per step on a
  DUPLO grasp, so a load is read before the grip is loaded much (ICL also
  twists in increments, `incremental_deg`).
- **`LIFT_HEIGHT = 0.015`**: ICL's mode 5. ICL lifts along world z; in the
  twin a vertical lift of a DUPLO brick levered 14 deg stalled within 3 mm
  (lag 4.3 mm, the near wall driven into the near studs), hence
  `lift_along: brick`.
- **`SEAT_PRESS = 0.001`**: a parallel grip slides along the brick under
  ICL's 2.8 mm.
- **`GRASP_DEPTH`**, **`APPROACH_HEIGHT = 0.04`**, **`GRIP_SQUEEZE =
  0.002`**: computing-brick-pose-from-stud-grid's and the press skill's.
- **`FEEL_STEP_FRACTION = 0.25`**, **`LAG_FRACTION = 0.5`**,
  **`FINE_TOLERANCE = 0.0003`**, **`FINE_MAX_STEPS = 90`**: the press skill's.
- **`STALL_STEPS = 3`, `STALL_PROGRESS = 0.5`**: a load alone is not a
  stall: in the twin a 5 N pull stood the hand 0.75-0.95 mm behind its command
  while it kept advancing a full step; an earlier version that stopped on lag
  alone called a brick that would have come off `stuck`.
- **`BASE_CREEP_FRACTION = 0.5`**: without it a load that builds a few tenths
  of a millimetre per step is folded into the free-motion baseline and never
  read (found by the unit tests' clutch).
- **`ROT_STALL_DEG = 4`**: a servo calls a rotation arrived within 3 deg.
- **`GONE_FRACTION = 0.5`**, **`BELOW_FRACTION = 1.5`**,
  **`RING_CHANGE_FRACTION = 0.5`, `MIN_RING_CHANGE = 0.002`**,
  **`INNER_FRACTION = 0.25`**: half a brick height separates "brick there"
  from "plate there"; a neighbour moving half a stud height (or 2 mm, above
  depth noise at 20 cm) is real; the seat's centre points avoid sight lines
  grazing the empty seat's rim.
- **`HOLD_MARGIN = 0.001`**: the twin's empty jaws close to the width less
  the squeeze.
- **`VIEW_RANGES`, `MIN_VIEW_CLEARANCE`, the depth window and limits**:
  refining-target-from-wrist-closeup's, via the press skill.

## Provenance

Adapted from ICL's Robotic_Lego_Manipulation (MIT; Liu, Sun, Liu,
arXiv:2309.02354): the disassembly pick of `lego_task_planning_node.cpp`
(modes 1-5: offset pre-grab, descent into the grab pose, the twist of
`Twist_Deg` about the tool's disassemble frame, the 15 mm lift with the twist
kept), `Lego::calc_brick_grab_pose`, and `Twist_Deg` from
`config/user_config.json`. ICL's custom end-of-arm tool sits on the brick and
pivots about a fixed tool frame on a compliant industrial arm, with the
brick's pose from its own bookkeeping. Here a parallel gripper grasps across
the brick, the pivot is the bottom edge under a finger, contact is tracking
lag or joint effort, the result is read from wrist depth, and the straight
pull, wiggle, regrasp and neighbour check are added. The script's header lists
the changes; LICENSE.icl is the upstream licence.
