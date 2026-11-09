---
name: searching-insertion-with-yaw-oscillation
description: Finds the rotation at which a held keyed or polygonal part drops
  into its socket - presses it lightly onto the socket (tracking lag as the
  push, no force sensor), turns the wrist in 6 deg steps alternately either
  side of the start (0, +6, -6, +12, ... within +-45 deg), renews the press
  after every turn, and stops when the hand sinks into the socket or a turn is
  held short (ShapeForce's spin search, its torque stop read from the wrist's
  own tracking). Feels for the socket first, from 3 mm over the start. Use when a
  square, hexagonal, keyed or 2x2 part is held over or on its matching hole,
  slot or stud pattern with its position right but its rotation known only to
  tens of degrees, and the arm's last joint can turn it.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [insertion, search, contact, yaw, keyed, peg-in-hole, wrist-roll]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, assembly]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nolann030628/ShapeForce/blob/1bc2aa8c4ee66622f0a597b2112a61cfbea2a345/third_party/utils_repo/peg_insertion/force_insert.py
      upstream_license: MIT
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.move_to_joints
  produces_outputs:
    yaw_found_deg: float
    yaw_read_deg: float
    sink: float
    rest_z: float
    tries: int
  exit_conditions:
    dropped_in: Pressed at some yaw, the hand followed its command (lag over the free baseline under half contact_lag) and sank at least drop_depth (2 mm) under the resting height - the part entered its socket; yaw_found_deg is the hand's yaw from the start (read back). The press is unloaded; the hand still holds the part there.
    no_drop: Every yaw in +-yaw_range_deg was pressed without a drop; the press is unloaded and (return_on_fail) the wrist turned back to the start.
    jammed: A turn stopped more than jam_deg (3 deg) short of its command - the part is held against turning (a key wall, friction); the press is unloaded where it stands.
    blocked: No socket felt from start_lift over the start to max_descent under it, a lift or descent step, press or wrist turn was refused, or the last joint had no room for the next turn; `detail` says which.
  canonical_scripts:
    - yaw_search: scripts/yaw_search.py
  examples:
    - title: Search the rotation of a part held on its socket
      path: examples/canonical_subgraph.json
---

# searching-insertion-with-yaw-oscillation

A square peg off by 20 deg rests on its hole's rim however well it is
centred. A camera that finds the hole's centre to a millimetre often gets its
rotation only to tens of degrees (a symmetric hole has no clear heading, and
small parts give few pixels). A translation search (searching-insertion-spiral)
does not help; turning does. ShapeForce turns the held part on its socket
until its soft wrist feels a torque; here the turn is a search for the drop.

## The motion

1. **Feel for the socket.** Lift `start_lift` (3 mm), then step down
   `descent_step` (1 mm) at a time, at most `max_descent` (6 mm) under the
   start, until the hand lags its command by `contact_lag` (1.5 mm) over the
   free-motion lag of the steps before. The stop is the resting height, and
   that step's command is where the socket met the hand. This works whether
   the part already rests on the socket (after placing-with-contact-descent or
   pressing-in-with-guarded-push) or hovers over it, and on an arm that stands
   millimetres off its aim in free air (the baseline takes that out).
2. **Press.** Command the hand `press_depth` (3 mm) past that command,
   rotation kept, and settle. On the rim the hand stays held (its lag over the
   baseline is the contact lag plus the press). In the socket it follows its
   command: the lag falls under half `contact_lag` and the hand sinks at least
   `drop_depth` (2 mm) under the resting height, which is `dropped_in`.
3. **Turn and press again**, for each yaw in 0, +step, -step, +2 step, ...
   within `yaw_range_deg`: turn the wrist to that yaw from the start by the
   last joint (`robot.move_to_joints`, the yaw read back and the joint's gain
   learnt, re-commanded up to three times to within 0.3 deg; this also takes
   the press off), then press again at the resting position and check the
   sink. A turn that ends more than `jam_deg` short of its yaw is `jammed`.
4. **End.** Unload the press (command the hand where it stands). After a
   fruitless search turn back to the start (`return_on_fail`).

A connector with no joint reading gets the turn as a rotation command instead
(a station twin whose IK leaves the wrist roll free would then not turn; see
PORTING lesson 10).

## Inputs

- `yaw_range_deg` (45; 90 for a two-fold part, up to the last joint's range
  for one-fold), `yaw_step_deg` (6), `press_depth` (0.003 m), `drop_depth`
  (0.002 m), `jam_deg` (3).
- `start_lift` (0.003 m), `max_descent` (0.006 m), `descent_step`
  (0.001 m), `contact_lag` (0.0015 m).
- `return_on_fail` (true), `roll_limits` (the last joint's range, rad;
  default +-2.0944), `step_tolerance` (0.0002 m, passed as the servo's
  `tolerance`), `step_max_steps` (0), the settle constants, `arm_id` (passed
  only when nonzero).

## Outputs

`yaw_found_deg` (the hand's yaw from the start where it dropped, read back),
`yaw_read_deg` (the yaw at the end), `sink` (resting height minus the last
pressed height), `rest_z`, `tries`. These say what the hand did: whether the
part is seated needs pressing-in-with-guarded-push to push it home, or a
check after the release.

## Assumptions

- The part is held centred on the wrist's axis. An off-axis part swings
  sideways as the wrist turns (the press keeps the hand's position, not the
  part's).
- The step must be finer than the part's angular clearance, or the search
  steps over the matching yaw. A 30 mm square in a 33 mm square hole fits
  within about +-5.5 deg; 6 deg steps fall within 3 deg of any yaw.

## The constants, and why

- **`YAW_STEP_DEG = 6`**: ShapeForce's 90 deg stroke in 15 sub-steps.
- **`YAW_RANGE_DEG = 45`**: a four-fold part matches within +-45 deg.
  Alternating sides keeps the nearest yaws first and inside the YAM's
  +-120 deg last joint, where ShapeForce turns one way and regrasps.
- **`PRESS_DEPTH = 0.003` m, `DROP_DEPTH = 0.002` m**: a position servo's
  hand cannot sink further than it is commanded, so the press sets how deep a
  drop can be seen; 3 mm is a real YAM station's contact lag. A drop needs both
  the hand following its command and 2 mm of sink, so a press that only takes
  up slack does not count.
- **`JAM_DEG = 3`**: a choice, standing in for ShapeForce's torque threshold
  of 20 (in its marker-tracked soft wrist's units, which do not carry over).

## Simulation check

Ran, but the search itself could not be exercised. Setup: a YAM station twin
(left arm, MuJoCo at 500 Hz) with a square socket added as fixtures, a 33 mm
hole in 4 mm walls 15 mm high, 9 cm from the 30 mm cube. A scripted top-down
grasp (tool centre 5 mm over the cube's centre) carried the cube to hover
3 mm over the wall tops, and the skill ran from there.

| run | cube friction | contact_lag / descent_step / press_depth / drop_depth | route | what happened |
|---|---|---|---|---|
| 1 | 0.8 (twin) | defaults (1.5 / 1 / 3 / 2 mm) | blocked | "no socket felt": the hand went 9.8 mm under the start |
| 2 | 0.8 | 0.4 / 0.5 / 0.5 / 0.6 mm | blocked | the same, 9.9 mm |
| 3 | 2.0 | 0.8 / 0.5 / 1.0 / 1.0 mm | blocked | the same, 9.9 mm |

- In every run the cube ended resting on the socket's wall tops (its bottom
  at the wall-top height, 15 mm over the table), while the hand kept
  following its command for another ~5 mm. **The held cube slid up in the
  jaws** under a push too small to show as 0.4 mm of lag, so the wall tops
  were never felt and nothing was turned. Raising the cube's friction from
  0.8 to 2.0 changed nothing measurable. On the fixed box wall,
  verifying-a-brick-connection-by-pull-up saw the twin's jaws slide at
  about 0.5-0.76 mm of lag.
- The grasp and carry turned the cube by -19.8 deg in the jaws: the twin's
  wrist roll is free (PORTING lesson 10). The socket, set 20 deg from the
  cube's start, therefore needed a +39.8 deg turn.
- So `dropped_in`, `no_drop` and `jammed` ran only in the unit tests. The
  "part already in" hint in `blocked` was wrong here: the part had slid in
  the jaws. Nothing ran on a real YAM.

## Limits

- **A part that slides in the jaws is not felt.** The push the press can
  apply is bounded by the grip's friction. If the part slides first, the
  socket reads as free travel (`blocked`, "no socket felt"), and a slide
  during the press could look like a drop. The drop test needs the hand to
  follow its command back to the free-motion lag (under half
  `contact_lag`), which a friction-limited slide does not quite do, but this
  was not seen in simulation. Grip the part firmly and check it in the jaws
  (wrist camera) after a `blocked`.

## What changed from ShapeForce, and why

ShapeForce's `_phase_spin_search` turns the closed hand 90 deg in 15 steps at
a fixed height, stops early when its soft wrist's torque passes 20, and
otherwise opens, lifts 50 mm, turns back, re-grips and repeats, without
limit. Here the turn alternates around the start within a range, the part is
held down by a position-servo press renewed after each turn, the stop is a
drop measured from the hand's height or a turn held short (read from the
wrist's own tracking), and the regrasp ratchet is left out
(turning-by-wrist-ratchet covers turning past the wrist's range).
