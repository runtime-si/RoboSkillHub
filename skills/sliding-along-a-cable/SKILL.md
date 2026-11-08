---
name: sliding-along-a-cable
description: Slides the hand along a cable it holds toward one of the cable's
  ends - the jaws opened to a loose cage just wider than the cable (not a
  pinch), the hand stepped along the perceived centreline with the jaws kept
  across it, stopped a planned distance before the end (or earlier when the
  hand stalls against the end's plug or the jaws are pushed open), then closed
  firmly. Use when a cable taken somewhere along its length must be held near
  its end instead, e.g. right behind a plug before plugging it, or to run the
  hand out to the tip of a cable lying on the table without regrasping, with
  the cable's centreline from curve.fit_centerline / curve.track_centerline
  and its end from curve.endpoints.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, deformable, cage, slide, trace]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/o2ac/o2ac-ur/blob/65ddc5496879799038cefbad061380b7d0300d73/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py#L4622
      upstream_license: MIT
      paper: https://doi.org/10.1016/j.mechatronics.2023.103096
gap:
  requires: {connector: [robot.set_grip, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - robot.solve_ik
    - robot.set_grip
    - robot.get_gripper
    - robot.open_gripper
    - robot.close_gripper
    - robot.describe_gripper
  required_inputs:
    centerline: Centerline
  produces_outputs:
    stopped_by: string
    start_from_end: float
    end_from_end: float
    slid: float
    cage_width: float
    cage_reading: float
    jaw_width: float
    steps: int
    wrist_turns: int
  exit_conditions:
    reached_end: The hand stopped at the planned distance before the end (stopped_by planned_stop or already_there), was pushed open by something thicker than the cable (jaws), or stalled within end_window (60 mm) of the tip (stall, read as the end's plug against the fingertips); the jaws then closed firmly and short of their stop. The hand stands there, closed on the cable.
    lost_cable: The hand was not on the cable before it moved (more than max_offset from the centreline in plan, jaws closed on nothing, or jaws held open wider than the cage), or the firm close at the end shut to the closed stop; in the last case the hand opened and rose 4 cm.
    unreachable: The arm solver could not reach a waypoint (previewed before anything moved), a step was refused, or the hand stalled further than end_window from the tip (a snag, not the end); after a refusal or a stall the jaws closed firmly where the hand stopped. `detail` says which.
  canonical_scripts:
    - slide_along_cable: scripts/slide_along_cable.py
  examples:
    - title: Slide from a grasp to 4 cm before the cable's tip
      path: examples/canonical_subgraph.json
---

# sliding-along-a-cable

Run the hand along a cable it holds, toward an end, without letting go.

## The motion

1. **The path.** `centerline` (a fit or track; a bare points list works),
   ordered toward `end`: a `CurveEnd` from `curve.endpoints` (its `node` picks
   the end, its `position`, one radius past the last node, is the tip), or a
   point (Vec3 or [x, y, z]) naming the nearer end. The hand's tool centre is
   projected onto it in plan: more than `max_offset` (8 mm) off is
   `lost_cable` before anything moves. The hand's own offset is kept along
   the path (it holds the cable, so the offset is the fit's error there).
2. **Where it stops.** `stop_before_end` (4 cm) before the tip, along the
   centreline: behind a charging cable's 33 mm plug.
3. **Preview.** Waypoints every `step` (1 cm) from the hand to the stop,
   each with the jaws turned to keep their angle to the local tangent
   (square, or `close_heading_deg`'s angle when given), each checked with
   `robot.solve_ik` (`within_servo_tolerance`): a stall at the edge of reach
   reads like the end, so an unreachable path is `unreachable` up front.
4. **The cage.** The jaws are read: closed on the cable, the cage is that
   reading plus `cage_margin` (1.5 mm); open, it is the cable's diameter plus
   2.5 mm (6 mm for 3.5 mm); `cage_width` overrides both. Closed on nothing
   is `lost_cable`. `robot.set_grip` sets it and the jaws are read back:
   still more than 3 mm wider than the cage, something holds them open
   (the cable under a fingertip after a missed grasp), and that is
   `lost_cable` before the hand moves.
5. **The slide.** Up `slide_lift` (1 mm, the fingertips off the table but
   under the cable's centre, so the table still closes the cage's bottom),
   then one `robot.go_to_pose_cartesian` per waypoint. After each step:
   - a refused step (a routed `unreachable` or a raised error) stops it;
   - a step that advanced under half its length, or reported contact, is a
     stall (the arm's tracking lag; no force sensor);
   - jaws read 3 mm wider than the cage: something much thicker came
     between them (a 6 mm plug in a cage about 5 mm between the fingertips,
     estimated, would open them only about 1 mm, and a cable dragged round a tight curve
     opened a twin's jaws 1.6 mm: this catches only a much thicker end; the
     stall and the planned stop are the usual ones);
   - the jaws' heading is read back; more than 4 deg from where it should be
     is turned on the last joint (`robot.move_to_joints`, the joint's sense
     learned from the first turn), as an arm whose solver leaves the wrist
     roll free does not turn it.
6. **Close firmly** where the hand stopped: first commanded `press` (3 mm)
   under the height it started at, so the fingertips go back to the table and
   stay on it while they shut (a refusal, e.g. a station's tool-centre floor,
   is noted and the close goes on), then `robot.close_gripper` (once more if
   it stopped wider than twice the cable plus 4 mm). At the closed stop:
   `lost_cable`, the hand opens and rises 4 cm. Otherwise the stop decides:
   the planned stop, the jaws pushed open or a stall within `end_window`
   (60 mm) of the tip is `reached_end`; a refusal or a stall further back is
   `unreachable`.

## Inputs

- `centerline` (Centerline or points), `end` (CurveEnd, Vec3 or [x, y, z]).
- `stop_before_end` (0.04 m), `close_heading_deg` (the grasp's, e.g.
  grasping-a-cable-lying-flat's output; default square to the cable at the
  hand), `cable_radius` (else the centreline's `radius_m`, else 1.75 mm).
- `cage_width` (none: from the jaws), `cage_margin` (0.0015 m), `step`
  (0.01 m), `slide_lift` (0.001 m), `press` (0.003 m; 0 closes where the
  hand stands), `max_offset` (0.008 m), `end_window`
  (0.06 m), `check_reach` (true), `roll_limits` (the YAM's joint 6),
  `arm_id` (0; passed only when nonzero).

## Outputs

`stopped_by` (planned_stop, already_there, stall, jaws, refused), the hand's
place along the perceived centreline before and after (`start_from_end`,
`end_from_end`, `slid`, metres), the cage asked and read, the jaw width
after the firm close, the steps and the wrist turns. All are what the hand
did against the centreline it was given: the cage drags the cable a little,
so re-perceive before relying on where the cable is.

## What changed from o2ac, and why

o2ac's `insert_motor_cables_without_tools_normal` cages the motor cable with
`gripper.send_command(0.005)`, pulls the hand one straight
`move_lin_rel([-cable_straighten_distance, 0, 0])` along the tool's x at
0.03 m/s (0.127 m; 0.097-0.133 m tuned per cable), then closes. Its stop is
the planned distance, and its geometry is fixed by the assembly's frames.
Here:

- The path is the perceived centreline toward a chosen end, cut into 1 cm
  steps that follow a curve, the jaws turned with it.
- The stop is planned from the end (`stop_before_end`) rather than from the
  start, and two sensorless early stops stand in for force: a stall read
  from the hand's progress (a real YAM station's arrival bound is 12 mm per
  move, so not from the distance to the target) and the jaws pushed open.
  The joint-torque idea of reading the connector against the fingers is from
  the cited paper (Mechatronics 2023, reimplemented as tracking lag, no code
  copied).
- The cage is relative to the jaws' own reading on the cable: a YAM station
  twin's jaws read 5.4 mm closed on a 3.5 mm cable, so o2ac's absolute 5 mm
  would be a pinch there.
- Every waypoint is previewed with the arm's IK first (a stall at the edge
  of reach looks like the end), and the firm close is pressed on the table
  (the real threading station's close press) and jaw-checked.

## Simulation check

A YAM station twin (left arm, linear_4310 fingertips reaching 10.2 mm, jaws
at gains 6000 / 120, an IK that leaves the wrist roll free) with the
threading twin's 3.5 mm charging cable: a 0.26 m rod of 7.2 mm capsules on
VBD, 2000 kg/m^3, bending 0.08 / 0.1, the pads at mu 0.5, at rest on the
tablecloth straight or on a circular arc. No plug on it. Each run:
grasping-a-cable-lying-flat took the cable 15 cm from its tip (fingertips to
the table, the 4 cm test lift off, the station's floor lowered to 5 mm), then
this skill slid toward the tip with `stop_before_end` 4 cm. The simulator's
capsule centres stood in for `curve.fit_centerline` (a perfect centreline)
and measured the outcome after every arm or jaw command: the hand's place
projected on the capsules, where the cable crosses the plane between the
jaws (its offset from the tool centre across the gap, against half the jaw
reading), and how far each capsule moved in plan from before the slide to
the end (drag).

| layout | route | hand from the tip, on the capsules (start -> end) | cable between the jaws every step | max offset across the gap | drag max / mean / tip | firm close | wrist turns |
|---|---|---|---|---|---|---|---|
| straight | reached_end (planned stop) | 150.2 -> 40.4 mm | yes | 0.26 mm | 2.8 / 0.8 / 0.2 mm | 5.6 mm | 4 |
| arc, 0.20 m, turning left | reached_end (planned stop) | 146.1 -> 40.2 mm | yes | 0.30 mm | 2.1 / 0.7 / 0.7 mm | 5.0 mm | 2 |
| arc, 0.25 m, turning right | unreachable (wrist out of range, 13 deg off) | 150.1 -> 59.9 mm | yes | 0.38 mm | 2.9 / 1.4 / 0.2 mm | 5.3 mm | 5 |
| arc, 0.15 m, turning right | unreachable up front (needs 43 deg, nothing moved) | 150.1 | - | - | 0.2 mm (settling) | - | 0 |

- Straight and left: the hand ended 0.2-0.7 mm from where the skill thought
  it was on its centreline (40.8 / 40.9 mm), the cable stayed caged within
  0.3 mm of the jaws' centre (the cage read 6.9-7.0 mm on the cable), its
  centre 1.75-1.81 mm over the table (never lifted out over the fingertips,
  which ran 0.9-1.1 mm over the table), and no capsule moved 5 mm. The
  straight and left runs were made before the two wrist-range guards were
  added; neither guard would have fired (every turn converged).
- Right: the grasp left the last joint at -1.52 rad, 0.5 rad from its stop,
  and a right-turning cable turns the wrist toward it. With R = 0.15 m the
  43 deg turn is refused before moving. With R = 0.25 m the up-front check
  passed (25 deg of cable turn) but the joint still ran out 60 mm from the
  tip: the up-front check counts only the cable's turn, and in this twin the
  hand's yaw also drifts as the arm's base joint swings (4 turns on the
  straight cable). The guard stopped the slide at 13 deg off square, closed
  firmly and held the cable (5.3 mm). Grasping with the other wrist solution
  is the fix; this skill does not regrip.

What earlier rounds showed and changed (the same harness, older code):

- **Close in the air.** Closed 1 mm over the table, the tapered fingertips
  rode up from 0.6 to 2.1 mm over it and pushed the cable out: the jaws shut
  to 0.0 mm (`lost_cable`, correctly). Now the close is pressed on the table
  (`press`), and every close since held the cable (4.9-5.8 mm).
- **Jaws held open.** After a missed setup grasp (the grasp skill, judging by
  the jaw gap only, called jaws stopped at 25-26 mm on the cable `grasped`;
  in one try its pre-close `set_grip` shifted the hand 9 mm), the cage read
  30 mm instead of 6. The skill now routes that `lost_cable` before moving.
- **Jaw widening as the end.** With a 1.5 mm threshold, a 0.15 m right arc
  stopped at 79.7 mm (`jaws`) with the cable centred: dragged round the curve
  with the wrist turned only once 8 deg off square, it opened them to
  8.6 mm against a 7.0 mm cage. The threshold is now 3 mm and the turns come
  at 4 deg.
- **Wrist out of range, unguarded.** On that arc the last joint ran out
  after 3 turns; the jaws drifted to 37 deg off square, wedged to 9.3 mm and
  dragged the cable up to 12.2 mm (the hand ended 49.9 mm from the tip on the
  capsules against 43.6 mm on its own centreline). Hence both guards.

**What the twin does not show.** There is no plug on the rod, so the end by
a stall or by the jaws pushed open was not exercised in simulation (unit
tests only); the planned stop was. The twin's jaws read about 1.9 mm wider
than the cable they close on, so the cage is a twin's reading, not a real
gap. A real arm that honours the full pose turns the wrist in its own
Cartesian moves and refuses past its limit (routed as a refused step). The
centreline was perfect; a fitted one is a few millimetres off, which the
kept offset absorbs near the hand only. Nothing here ran on a real YAM.

## Provenance

Adapted from o2ac-ur's `insert_motor_cables_without_tools_normal`
(o2ac/o2ac-ur, MIT, Copyright (c) 2020 Team O2AC; the licence is
LICENSE.o2ac), with the changes above. The end-detection by joint signals
follows "Sensorless robotized cable contour following and connector
detection" (Mechatronics 2023), reimplemented from the paper.
