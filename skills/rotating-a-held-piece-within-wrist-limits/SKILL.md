---
name: rotating-a-held-piece-within-wrist-limits
description: Turns a held flat part (a jigsaw piece) about the vertical by the
  angle its slot needs while the wrist's last joint stays inside its range
  (+-2.09 rad on the YAM, 240 degrees - less than a full turn) - takes the
  shorter or the longer way round when one fits, else sets the piece down,
  turns the empty hand half a turn so the jaws close on the same two contact
  points swapped, takes it again and finishes the turn (WRS-style regrasp,
  reduced to one joint and a table). Learns which way the joint turns the
  hand from a small first held step, and corrects each turn on the hand's yaw
  read back. Can also only plan, e.g. to grip with the jaws swapped at the
  pick so no regrasp is needed. Use when a jigsaw piece (or any flat part) is
  held top-down and must be turned before it is placed, after the turn is
  known from localizing-a-piece-in-a-reference-image or
  matching-a-piece-to-placed-neighbours.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [jigsaw, puzzle, reorientation, regrasp, wrist, joint-limits]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, reorientation]
    provenance:
      authors: [runtime-si]
      code: https://github.com/wanweiwei07/wrs/tree/main/0000_examples/x6wg2_regrasp
      upstream_license: MIT (idea only; no code used)
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.move_to_joints
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.open_gripper
    - robot.close_gripper
    - robot.set_grip
    - robot.get_gripper
    - robot.describe_arm
  required_inputs:
    turn_deg: float
  # Optional inputs (wired from a script node where gap has no type for them): grasp_z
  # (tool centre's world z to grip the piece on the table; needed for a regrasp),
  # set_down_xy ([x, y] free spot), grip_width_m, squeeze_m, gain (+1/-1), plan_only,
  # joint6_deg, arm_id.
  produces_outputs:
    turned_deg: float
    regrasps: int
    jaws_swapped: bool
    joint6_final: float
  exit_conditions:
    turned: The hand turned the piece by the asked angle (read back within 0.5 deg) without letting go; joint 6 stayed 3 deg inside its range.
    turned_with_regrasp: No direct turn fitted the range, so the piece was set down, the empty hand turned half a turn, took it again (jaws swapped, same contacts) and finished the turn; `jaws_swapped` is true.
    planned: plan_only was set; `plan` says direct or regrasp and the joint moves, nothing moved.
    piece_lost: After a regrasp the jaws closed past half the given grip width (nothing in them).
    unreachable: No plan fits the range, a regrasp was needed without grasp_z, the joints could not be read, joint 6 did not move the hand, or a move was refused.
  canonical_scripts:
    - rotate_piece: scripts/rotate_piece.py
  examples:
    - title: Turn a held piece to its slot's orientation before placing it
      path: examples/canonical_subgraph.json
---

# rotating-a-held-piece-within-wrist-limits

A piece comes off the staging mat at whatever angle it lay; its slot needs it
at another. Seen from above, a top-down hand turns the piece only with its last
joint, and the YAM's turns 120 degrees either way: 240 degrees, not 360. From
some joint positions a needed turn fits neither the short way nor the long way
round. Then the piece has to be put down and taken again with the hand turned.

The new grip must be the old one: the grip was chosen for the slot (clear of
the placed neighbours, `selecting-a-grip-clear-of-placed-neighbours`), so the
empty hand turns exactly half a turn. A parallel gripper is symmetric, so its
jaws then close on the same two contact points, swapped.

## The plan

`plan_turn(q6, turn, lo, hi, gain)` (joint degrees, range shrunk 3 degrees a
side):

- **direct**: the joint move `turn / gain + k * 360` (k = -1, 0, 1) with the
  smallest size that ends in range;
- **regrasp**: a held move to some q1 in range (1 degree grid), an empty
  move of +-180 degrees that stays in range, and a held move finishing the turn
  that ends in range; the cheapest by total joint travel.

With a 234-degree usable range every turn from every joint position has one
of the two (tested over a grid).

## The motion

1. Read joint 6 and the limits (`robot.describe_arm` if it reports them, else
   +-2.0944 rad). Lift to `grasp_z` + 30 mm if lower.
2. Unless `gain` is given, turn 4 degrees toward the middle of the range while
   holding the piece and read the hand's yaw back: that gives the sign of the
   joint against the world yaw (it counts toward the turn). Plan the rest.
3. Each held turn is a `robot.move_to_joints` on joint 6, corrected up to 4
   times on the yaw read back until within 0.5 degree (the station twin's wrist
   lags).
4. Regrasp: move straight across (Cartesian) to `set_down_xy` at carry height
   (default: straight below), down to `grasp_z` - re-commanded until the tool
   centre reads back within 1 mm - open, up 30 mm, turn the empty hand +-180 degrees (read
   back), down to `grasp_z`, grip (`robot.set_grip` with `grip_width_m`, else
   `robot.close_gripper`), up 30 mm; with a grip width, a hand that closed past
   half of it is `piece_lost`.

`turned_deg` is the hand's yaw change while holding, read back from
`robot.get_ee_pose`: what the hand did, not a camera measurement of the piece.

## Inputs

- `turn_deg` (required): counter-clockwise from above, the turn from how the
  piece is held to how it fits.
- `grasp_z`: the tool centre's world z for gripping the piece on the table
  (where it was picked); needed only for a regrasp.
- `set_down_xy`: a free spot for the regrasp (the slot area is crowded).
- `grip_width_m`, `squeeze_m` (3 mm), `gain` (+1 / -1), `plan_only`,
  `joint6_deg` (plan from this joint position instead of the arm's), `arm_id`.

## Outputs

`plan`, `turned_deg`, `hand_turned_deg`, `regrasps`, `jaws_swapped`,
`joint6_final` (rad), `detail`.

## Measured

**Simulation check** (YAM station twin, left arm; a 2 mm jigsaw piece gripped
by the harness's 10 mm grip block, lifted out of its slot among six fixed
neighbours; GCP pool, harness in the pull request). Truth is the piece body's
yaw before and after (simulator), against the asked turn. Joint 6 started at
+65 or -115 degrees (where squaring the jaws left it).

**Direct turns**, all runs with the final code and the earlier ones: every
one held, joint 6 always inside +-117 degrees. In the final batch, 7 / 7
turns of -150 to -60 degrees came within -4.6 to +0.5 degrees of the asked
turn. Over 22 direct turns in earlier runs (-170 to +180 degrees), the error
was -4.1 to +0.5 degrees. The hand itself turned within 0.01 degree; the rest
is the piece slipping about 3.5 degrees in the jaws on long turns.

**Regrasps** (turns that fit neither way from +65 degrees, forced through a
set-down, empty half turn and regrasp), final code:

- **Set-down spot 60 mm east of the slot:** 5 / 5 held, piece within 0.47-0.54 degrees of the asked turn, tilt at most 1.1 degrees.
- **Spot 70 mm west:** 0 / 4. The piece was not in the jaws after the regrasp; not diagnosed.
- **Spot 80 mm south:** unreachable, routed `unreachable`.

Two fixes came from earlier runs, which together held 2 / 23 regrasps:

- **Set-down height read back.** The servo reported "reached" 2.3 mm short, the piece dropped 3 mm and shifted; it is now re-commanded until within 1 mm.
- **Straight transfer.** A joint-space move to the set-down spot swung the held piece into the neighbours and knocked it out of the jaws; the transfer is now Cartesian.

## Limits

- **The piece slips in the hand as it turns**; this reports the hand's turn.
  Measure the held piece with the wrist camera before placing when it matters.
- **Regrasps depend on the set-down spot**: 5 / 5 at one free spot in the twin,
  0 / 4 at another. Check the spot is clear well beyond the piece (it hangs
  off the tool centre), and keep the jaws' grip width (`grip_width_m`).
- **The set-down spot must be free and flat**, and reachable with the hand's
  current tilt; the skill does not search for one
  (`selecting-free-placement-spots` can).
- **The regrasp assumes the piece did not move when released**: on a sloped
  or soft mat it may, and the jaws then close on different points.
- Only the last joint turns the piece; a turn a different wrist pose would
  make reachable is not explored (`robot.solve_ik` with another heading).

## The constants, and why

`WRIST_LIMIT_RAD` 2.0944 (YAM joint 6; station MJCF), `LIMIT_MARGIN_RAD` 0.05,
`REACH_TOL_M` 1 mm,
`YAW_TOL_DEG` 0.5, `MAX_CORRECTIONS` 4, `PROBE_DEG` 4, `LIFT_M` 30 mm,
`SETTLE_STEPS` 40, `STEP_DEG` 1 (planner grid).
