---
name: straightening-a-cable-toward-a-target-curve
description: Pulls a cable lying flat on a table toward a target curve, one
  correction at a time - the perceived centreline and the target are resampled
  at equal arc length in plan, matched in the cable's two directions (and,
  when their lengths differ, at every placement of the shorter along the
  longer), the point with the largest discrepancy is picked with
  grasping-a-cable-lying-flat and put down a little past its target, and the
  cable is looked at again, until every node is within a tolerance, a cap on
  corrections is reached, the error stalls or the cable is lost. Use when a
  cable on the table must lie along a line or curve, e.g. to lay its last
  10 cm straight and pointing at a ring before threading, or to undo a bend
  before routing, with an overhead RGB-D camera (sam3.segment_text and
  curve.fit_centerline) or a centreline fitted by the caller.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [cable, deformable, shape, pick-place, straighten]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/DanielTakeshi/deformable-ravens/blob/73982748bdfe756e04553e28698948facccecb41/ravens/tasks/task.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2012.03385
gap:
  requires: {connector: [robot.grasp_frame, robot.move_cartesian_until_contact, robot.set_grip, robot.describe_gripper, robot.describe_workspace]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.get_observation
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.move_to_joints
    - robot.solve_ik
    - robot.grasp_frame
    - robot.set_grip
    - robot.open_gripper
    - robot.close_gripper
    - robot.get_gripper
    - robot.describe_gripper
    - robot.describe_workspace
    - sam3.segment_text
    - geometry.mask_to_world_points
    - curve.fit_centerline
  required_inputs:
    target: Centerline
  produces_outputs:
    max_error: float
    mean_error: float
    initial_max_error: float
    corrections: int
    misses: int
    reversed: bool
    offset_m: float
    coverage: float
    pick_point: Vec3
    place_point: Vec3
    centerline: Centerline
  exit_conditions:
    shaped: The latest look put every matched node within tolerance (10 mm) of the target in plan; the hand is open and stands retreat (5 cm) over the last place, or has not moved if the cable was already there.
    corrected: No cable_description was given, so after one pick-and-place the cable was not looked at again; fit its centreline and call again. The hand is open over the place.
    not_converged: max_iters (12) corrections were made, or the best max error did not improve by 2 mm in 4 corrections, and some node is still further than the tolerance; the cable lies on the table, the hand is open.
    grasp_failed: Two grasps in a row (one without a camera) closed on nothing; the hand is open over the cable.
    lost_cable: The cable was not found by the camera, its fit was empty, or the cable seen is shorter than min_coverage (80 %) of the target in plan (part of it hidden or out of view).
    unreachable: No pick among the PICK_TRIES (3) largest discrepancies was reachable (grasping-a-cable-lying-flat routed unreachable for each), or the carry to the place was refused; after a refused carry the cable was set down where the hand stood and released. `detail` says which.
  canonical_scripts:
    - straighten_cable: scripts/straighten_cable.py
  examples:
    - title: Pull the white cable onto a target line seen by the overhead camera
      path: examples/canonical_subgraph.json
---

# straightening-a-cable-toward-a-target-curve

Make a cable lying on the table lie along a target curve, by repeatedly
moving the point that is furthest from where it should be.

## The loop

1. **Look.** The cable's centreline: `centerline` on the first round if given,
   otherwise (and on every later round) `sam3.segment_text` with
   `cable_description` on the camera named `camera` (default the overhead one)
   and `curve.fit_centerline` with `smooth` 1e-5 and 60 nodes. Each fitted
   point is slid along its camera ray onto the table plane (table plus the
   cable's radius): a flat cable's axis is there, and a point whose depth read
   high is off in plan along its ray too.
2. **Match.** Both curves are resampled at `spacing` (1 cm) of arc length *in
   plan*. The shorter is laid along the longer at every offset (`match:
   slide`, half-spacing steps) or with an end on an end (`match: ends`), in
   both directions; the placement with the least mean plan distance wins.
   This is deformable-ravens' oracle: its cable-ring branch tries every bead
   ordering and keeps the least mean distance, its cable-shape branch moves
   the bead furthest from its target first. Then each node is measured to the
   nearest target point within `along_slack` (3 cm) of its matched place
   (upstream's success metric counts a bead near *any* target), so a node
   lying on the curve a little along from its place counts as on it, and the
   node with the largest distance is the one to move.
3. **Stop?** Every node within `tolerance` (10 mm): `shaped`. Cable seen
   shorter than `min_coverage` (80 %) of the target: `lost_cable`.
   `max_iters` (12) corrections made, or no 2 mm gain in the best max error
   over 4: `not_converged`.
4. **Pull, not push.** The move is that node's correction, applied where the
   fingers can hold it (`end_margin`, 15 mm, in from an end). A move within
   30 deg of the cable's own way is kept only at an end and outward (it
   slides the cable, and is put down exactly, without overshoot); into the
   cable at an end, the other end is pulled the same way instead; along the
   cable away from its ends it is not made (it would buckle one side), and if
   no other node is left: `not_converged`.
5. **Pick.** `grasping-a-cable-lying-flat` there, its jaws across the local
   tangent, the fingertips taken to the table and a 4 cm test lift (a
   wrist-camera lift check when `lift_check` and a `cable_description` are
   given). If it routes
   `unreachable` the next-largest discrepancy at least 3 cm along the cable
   is tried, up to 3 picks. A miss counts as a correction; two in a row is
   `grasp_failed`.
6. **Place.** The hand carries the cable at the lifted height to the move's
   end pushed `overshoot` (1 cm, at most half the move) further along the
   pull (upstream's `params_no_rots`; not for a slide), comes down to
   `place_over` (3 mm) over the grasp's
   contact height, opens and rises `retreat` (5 cm); `clear_pose`, if given,
   is visited before the next look.
7. Back to 1. Without a `cable_description` there is no next look: the skill
   routes `corrected` after one pick-and-place and the caller fits the
   centreline and calls again.

## Inputs

- `target` (required): a Centerline, `{"points": ...}` or a list of
  `[x, y, z]` in world metres, in order; two points is a straight segment.
  Its height is ignored. To straighten a cable without caring where along
  the line it ends up, give a line a few centimetres longer than the cable
  (`match: slide` then lets it lie anywhere along it); a target exactly the
  cable's length also asks where its ends go.
- `centerline` (first round), `cable_description` ("" turns the camera off),
  `camera` ("overhead"), `table_z` (else `robot.describe_workspace`, else the
  cable's height less its radius), `cable_radius` (else the fit's, else
  1.75 mm).
- `tolerance`, `max_iters`, `spacing`, `match` ("slide" | "ends"),
  `along_slack`, `overshoot`, `end_margin`, `min_coverage`, `pick_tries`, `max_misses`,
  `fit_smooth`, `fit_nodes`, `place_over`, `retreat`, `clear_pose`.
- Passed to the grasp when set: `press`, `pre_close_width`, `min_tool_height`
  (the real station's 14 mm floor is too high to pinch a 3.5 mm flat cable,
  see grasping-a-cable-lying-flat), `pinch_below`, `overtravel`, and always
  `check_reach`, `lift_check` / `grasp_camera`, `arm_id` (passed to the
  connector only when nonzero).

`grasping-a-cable-lying-flat` must be installed beside this skill (same
skills directory, or under a `GAP_SKILLS_PATH` root): its script is loaded and
run for every pick, through this skill's context.

## Outputs

`max_error` / `mean_error` (the latest look, plan, m), `initial_max_error`,
`errors` (max error at every look), `corrections`, `misses`, `reversed` and
`offset_m` (the winning placement), `coverage`, the last `pick_point` and
`place_point`, and `centerline` (the latest look, on the table).

## What changed from deformable-ravens, and why

- **Perception instead of bead poses.** The oracle reads PyBullet bead
  positions. Here the cable is a fitted centreline. `curve.fit_centerline`'s
  default `smooth` is applied in metres and turns a long real cable into one
  cubic, its ends 54-58 mm off on a real overhead frame; with 1e-5 the ends
  came to 7-9 mm but depth jitter inflated the 3D length by up to 45 %
  ([#10](https://github.com/runtime-si/RoboSkillHub/issues/10)). This skill
  passes 1e-5, lays the points on the table along the camera rays and measures
  every length in plan, so the jitter in depth does not reach the matching.
  On a real 640x480 overhead frame of the YAM station (cable masked by a white
  top-hat, 0.69 m of it in view), `smooth` 1e-5 gave a 3D length of 0.720 m
  against 0.686 m at the default, while the plan length stayed 0.687 m.
- **An open cable.** The ring branch rotates the bead ordering around a closed
  loop; an open cable has two orderings, and when the target is a stretch of
  the cable (the last 10 cm straight) the target slides along it.
- **The target is a polyline,** not a set of bead targets laid by the task.
- **A gripper, not suction.** The oracle's suction picks any bead pixel;
  here the pick is grasping-a-cable-lying-flat (top-down across the
  tangent, fingertips to the table, the 4 cm test lift), kept off the ends,
  with the next discrepancy tried if the arm cannot reach the first.
- **Stopping.** Upstream judges success by every bead within 3.5 radii of any
  target and stops after 21 actions; here the max matched error against
  `tolerance`, a cap of 12, and a stall check, with routed exits.
- **Overshoot** 1 cm: upstream uses 0 for the cable ring, 2 cm for bag rings
  and 3 cm for cloth. Here it is capped at half the move, so that a small
  correction of a node that does not spring back does not land it a full
  tolerance off on the other side; and a slide of the cable along itself
  gets none.
- **Measured across, pulled not pushed.** Upstream matches bead to target by
  order only. Matched that way in the twin (below, run A), a cable already
  lying within 1.7 mm (mean) of its line but crept along it read 13.6 mm off
  at one node, and pulling that node along the cable buckled it (true max
  distance to the line 12.8 -> 30.6 mm). So each node is measured to the
  nearest target point within `along_slack` of its place, and a move along
  the cable is made only at an end, outward (or from the other end).

## Simulation check

YAM station twin on Brev (the threading twin's 3.5 mm cable: a 26 cm rod of
37 capsules on VBD beside the MuJoCo arm), bent by the arm with
grasping-a-cable-lying-flat, target the cable's settled straight line from
(0.52, 0.17) to (0.52, 0.43). The twin has no cameras, so the simulator's
cable nodes stand in for the fitted centreline; the camera loop (sam3 + curve)
is covered by the unit tests only. The check graph called the skill without a
`cable_description`, one correction per call (`corrected`), and looked again
itself, at most 9 corrections. Every pick went through
`grasping-a-cable-lying-flat`. Distances are the true plan distance of each
node to the target line, max / mean in mm, at each look:

| case | version | bend | max / mean per look (mm) | end |
|---|---|---|---|---|
| A | matched by order only, full overshoot | 4 cm at the middle | 52.3/22.3, 25.3/12.6, 43.9/18.8, 23.5/11.7, 17.7/4.8, 12.8/1.7, 30.6/9.1, 23.8/8.6, 15.2/4.1, 19.4/5.4 | 19.5 / 5.4 after 9 corrections |
| B | measured across, pulled not pushed, overshoot capped | 4 cm at the middle | 53.6/22.9, 25.4/12.7, 25.6/10.3, 25.4/10.2, 25.3/8.4, 21.5/7.9, 18.7/7.4, 17.2/6.1, 12.2/5.7, 11.2/5.4 | 11.0 / 5.4 after 9 corrections |
| C | as B, no overshoot near an end (this version) | 4 cm at the middle | 51.9/22.1, 24.9/12.5, 35.6/15.0, 29.0/13.0, 44.3/19.7, 18.2/10.2, 24.3/8.7, 14.3/8.3, 15.5/5.5, 5.8/2.3 | all nodes within 6 mm after 9 corrections |
| D | this version | S, ±3 cm at 0.3 and 0.7 | 30.7/19.0, 28.0/13.1, 28.5/8.2, 24.1/9.3, 9.6/3.7 | `shaped` (tolerance 10 mm) after 4 corrections |

All grasps succeeded and the cable never left the table (max lift 0.6 mm).
In A, by order, a cable already within 1.7 mm (mean) of its line was pulled
along itself and buckled (12.8 -> 30.6 mm). In B every move was across the
line, but the ends whipped from side to side: an end 25 mm off, pulled 15 mm
in from it with the 10 mm overshoot, swung to 17 mm off on the other side.
Convergence is still not monotone in C (a correction can push a neighbouring
part of the cable out, looks 3 and 5). Because the graph made one correction
per call, the skill's own `max_iters` and stall rules did not act in the
check; replayed on C's and B's series, the first defaults (8, and no 2 mm
gain in 3) would have stopped both with `not_converged`, hence 12 and 4.
One run per case, so these are examples, not rates. A run of 9 corrections
took about 9 minutes on the L40S (A took 50, on a crowded GPU). Not run on a
real YAM.

## Provenance

Adapted from deformable-ravens' `Task.oracle` and `Task.params_no_rots`
(DanielTakeshi/deformable-ravens, `ravens/tasks/task.py`, Apache-2.0), with
the changes above. Seita et al., "Learning to Rearrange Deformable Cables,
Fabrics, and Bags with Goal-Conditioned Transporter Networks", ICRA 2021.
