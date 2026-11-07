---
name: checking-dual-arm-clearance
description: Predicts, before a move is sent, the minimum clearance between the
  two arms along the planned path - both arms' joints and tool poses are read
  together, each planned leg (joint waypoints, or tool poses solved with the
  arm's IK every 2 cm) is sampled every 0.01 rad, and at every sample a capsule
  model of the YAM's links (plus held-object spheres) gives the closest
  cross-arm pair - and routes clear / too_close / unknown with the minimum
  distance, where along the path it falls and the closest link pair. Use when a
  bimanual station is about to move one arm near the other (a handover, a
  two-hand lift, a place beside the other hand) and its planner does not see
  the other arm, with both arms addressable by arm_id.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: safety
  tags: [bimanual, collision, clearance, verify, capsules]
  hub:
    stage: verify
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [bimanual, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/check_dual_arm_collision/robotwin
      upstream_license: Apache-2.0
gap:
  requires: {connector: [robot.describe_arm, robot.forward_kinematics]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_arm
    - robot.forward_kinematics
    - robot.solve_ik
  required_inputs: {}
  produces_outputs:
    min_clearance_m: float
    closest_link_a: string
    closest_link_b: string
    closest_point_a: Vec3
    closest_point_b: Vec3
    at_fraction: float
    start_clearance_m: float
    end_clearance_m: float
    samples: int
    model_error_m: float
  exit_conditions:
    clear: The capsule model's smallest cross-arm clearance over every sample of the planned motion is at least too_close_m (default 5 mm); min_clearance_m, closest_link_a / closest_link_b and at_fraction say where it is tightest.
    too_close: Somewhere along the planned motion (or now, with no plan given) the capsules come within too_close_m or overlap (negative min_clearance_m); at_fraction and closest_link_a / closest_link_b say where and which links.
    unknown: The check could not be made - a second arm is not reported or reads as the first, an arm has no joint reading or not 6 joints, the connector's forward kinematics disagrees with the YAM model by more than 5 mm, a pose waypoint has no IK solution within the servo's tolerance or the IK changed branch along a straight leg, or the inputs conflict; `detail` says which.
  canonical_scripts:
    - check_clearance: scripts/check_clearance.py
  examples:
    - title: Check a planned pose sequence of arm 0 against arm 1 standing still
      path: examples/canonical_subgraph.json
---

# checking-dual-arm-clearance

A per-arm planner (or a plain IK servo) does not see the other arm. Before a
move, this node asks: if arm A goes there (and then arm B goes there, or both
together), how close do the two arms come, and where? It never moves anything.

## The check

1. **Read both arms together** (`robot.get_observation`): 6 joints, tool pose
   and jaw opening each. Refuse (`unknown`) if the connector reports fewer arms
   than asked for, or if the two arms read the same tool pose and joints (a
   connector that ignores `arm_id` would answer the first arm twice).
2. **Locate each base** from its own reading: base = tool pose x (model tool
   pose at these joints)^-1, with the tool centre from `robot.describe_arm`
   (`tcp_offset`, `tcp_rotation`; the YAM's 134.7 mm grasp site otherwise). No
   base poses or spacing are configured.
3. **Check the model against the connector.** `robot.forward_kinematics` is
   asked for the tool pose at the current joints and at a test change of all
   six; the relative motion must match the model's within 5 mm, otherwise this
   is not the modelled arm (or its joints are ordered differently): `unknown`.
   A connector without that tool is trusted and `detail` says so.
4. **Plan the joint paths.** `trajectory_a` / `trajectory_b` (gap
   `Trajectory`) are taken as joint waypoints from the current joints.
   `poses_a` / `poses_b` (gap `PoseSequence`) are solved with `robot.solve_ik`
   every 2 cm / 10 deg along each straight leg (only at the end for a waypoint
   with `cartesian: false` or `mode: planned_joint`); a solve outside the
   servo's tolerance or a branch change (a joint jumping 0.8 rad between
   neighbouring solves) is `unknown`, since the arm's real path is then not
   known. An arm with neither stands still. Each path is interpolated in
   joint space so no joint moves more than 0.01 rad between samples.
5. **Combine** the two paths: `a_then_b` (default; the way the handover and
   most graphs move, one arm at a time), `b_then_a`, or `together` (both
   resampled by joint arc length, as a lock-step stream would move them).
6. **Clearance at every sample** (RoboRSI's rule, spheres replaced by
   capsules): for every capsule of arm A against every capsule of arm B,
   segment-to-segment distance minus both radii; the smallest over all pairs
   and samples is `min_clearance_m` (negative = overlap), with the two links
   (`closest_link_a`, `closest_link_b`, e.g. `arm0:tip_right`), the two
   closest points (world) and `at_fraction` (0 = now, 1 = the end of the
   combined motion). Fingertip capsules ride on the jaw opening
   read now (`grip_a` / `grip_b` override it, 1.0 = open); held objects are
   gap `AttachedObject` spheres in the tool frame (`attached_a`, `attached_b`,
   or a pose sequence's `attached_object`).

## Inputs

All optional; with none, the check is of the two arms as they stand now.

- `arm_a` (0), `arm_b` (1): the connector's arm ids.
- `trajectory_a`, `trajectory_b`: joint waypoints (gap `Trajectory`).
- `poses_a`, `poses_b`: tool-pose waypoints (gap `PoseSequence`).
- `attached_a`, `attached_b`: held-object spheres (gap `AttachedObject`).
- `grip_a`, `grip_b`: jaw opening fraction during the move (default: as read).
- `order`: `a_then_b` | `b_then_a` | `together`.
- `too_close_m` (0.005), `joint_step_rad` (0.01), `cartesian_step_m` (0.02).
- `skip_links`: comma-separated link names to leave out (e.g. `gripper,base`).

## The YAM capsule model

The i2rt YAM with the linear_4310 hand, transcribed from the i2rt MJCF (the
chain matches mujoco's own kinematics to 0.13 um over 20 random joint
vectors). Every capsule encloses all vertices of its share of the link's
collision geometry (a minimum-volume enclosing capsule per k-means cluster of
vertices, or per convex piece for the fingertips, +0.5 mm for rounding), so the
predicted clearance is a lower bound on the geometry it was fitted to. Radii
and counts, per body:

| body | capsules | radii (mm) | fitted to |
|---|---|---|---|
| base | 1 | 49 | MJCF mesh (never near the other arm at 0.61 m spacing) |
| link1 | 3 | 19-26 | collision mesh, k-means clusters |
| link2 (upper arm) | 8 | 18-33 | collision mesh, k-means clusters |
| link3 (forearm) | 8 | 16-32 | collision mesh, k-means clusters |
| link4 | 5 | 12-32 | collision mesh, k-means clusters |
| link5 | 4 | 12-29 | collision mesh, k-means clusters |
| gripper housing | 4 | 16-29 | MJCF mesh (the twin does not collide it; the real one is there) |
| tip_left, tip_right | 49 each | 2-17 (median 6-8) | one per convex piece of the fingertip's collision decomposition; they slide 0-47.5 mm with the jaw |

The constants in the script carry the end points. The base, the hand housing
and the D405 wrist camera bracket are not in the twin's arm-to-arm measure;
the bracket is not in the model at all (no geometry for it in the MJCF): on the
real arm add it as an `attached_*` sphere, or keep a margin.

## Simulation check

RoboSimStudio YAM station twin, both arms (bases 0.61 m apart, arm 0 = left).
The twin does not collide the two arms with each other, so the harness
measures their clearance itself (`arm_gap.py`): the closest pair of collision
points across link1-5 and the fingertips' convex pieces (mesh vertices; an
overlap reads as about 0, never negative; the hand housing and the base are not
in it). Ten configurations of both arms were taken from each of two
handing-over-between-arms runs recorded every 20 ms (handover point 1 with
`free_offset_fraction` 0.5, where the hands touch, and point 3 with the
default 1.0): the closest poll and the polls nearest 1, 2, 3.5, 6, 10, 20, 40,
80 and 200 mm, visited in the order the handover passed them. At each the
harness set both jaws, asked this skill for the path clearance of the move
there (left arm then right, joint targets), drove the arms there
(`robot.move_to_joints`, polling the measure every 20 ms), and asked the skill
again standing still. "full" is the skill as shipped; "matched" leaves out the
housing and the base (`skip_links="gripper,base"`) to compare like with like.

| run | measured (mm) | predicted, full | predicted, matched | route | path measured (polls) | path predicted | path route |
|---|---|---|---|---|---|---|---|
| point 1, offset 0.5 | 190.6 | 175.2 | 178.6 | clear | 37.3 (11) | 27.6 | clear |
| point 1, offset 0.5 | 76.3 | 69.0 | 69.0 | clear | 75.9 (9) | 65.4 | clear |
| point 1, offset 0.5 | 50.6 | 43.7 | 43.7 | clear | 50.6 (9) | 43.7 | clear |
| point 1, offset 0.5 | 5.2 | 0.3 | 0.3 | too_close | 5.2 (8) | 0.2 | too_close |
| point 1, offset 0.5 | 0.2 | -9.2 | -9.2 | too_close | 0.2 (9) | -13.5 | too_close |
| point 1, offset 0.5 | 9.7 | 1.5 | 1.5 | too_close | 0.6 (13) | -15.1 | too_close |
| point 1, offset 0.5 | 32.8 | 27.2 | 27.2 | clear | 9.7 (12) | 1.5 | too_close |
| point 1, offset 0.5 | 38.6 | 33.8 | 33.8 | clear | 32.8 (10) | 27.2 | clear |
| point 1, offset 0.5 | 24.7 | 19.7 | 19.7 | clear | 24.3 (17) | 16.7 | clear |
| point 1, offset 0.5 | 38.0 | 33.6 | 33.6 | clear | 24.7 (16) | 19.7 | clear |
| point 3, offset 1.0 | 226.9 | 217.0 | 217.0 | clear | 103.9 (12) | 73.7 | clear |
| point 3, offset 1.0 | 70.7 | 64.6 | 64.6 | clear | 62.6 (14) | 50.8 | clear |
| point 3, offset 1.0 | 72.7 | 66.4 | 66.4 | clear | 70.5 (10) | 64.5 | clear |
| point 3, offset 1.0 | 73.7 | 67.8 | 67.8 | clear | 72.9 (8) | 66.6 | clear |
| point 3, offset 1.0 | 23.9 | 20.0 | 20.0 | clear | 23.9 (9) | 20.0 | clear |
| point 3, offset 1.0 | 0.3 | -6.2 | -6.2 | too_close | 0.3 (8) | -6.3 | too_close |
| point 3, offset 1.0 | 59.6 | 51.4 | 51.4 | clear | 0.3 (7) | -6.2 | too_close |
| point 3, offset 1.0 | 79.9 | 71.9 | 71.9 | clear | 59.8 (6) | 51.7 | clear |
| point 3, offset 1.0 | 53.6 | 44.3 | 44.3 | clear | 53.5 (8) | 44.3 | clear |
| point 3, offset 1.0 | 57.9 | 54.8 | 54.8 | clear | 50.3 (7) | 42.1 | clear |

The table is the last pair of runs (the bundle as shipped). An earlier pair of
runs with the same capsules (the outputs then named differently) gave 20 more
configurations; the numbers below are over all 40.

- **Standing still**, the prediction was under the measure at all 40
  configurations, by 3.2-12.0 mm (mean 6.7) matched and 3.2-15.3 mm (mean 6.9)
  full (the largest are the housing, which the measure leaves out). The
  closest link pair agreed with the measure's at all 40 matched and 38 of 40
  full (at 190-218 mm the full model's closest is the housing). Every
  configuration under 200 mm was fingertip to fingertip.
- **Routes at the default 5 mm**, taking "measured under 5 mm" as truly too
  close: standing still, false clear 0 of 4, false too_close 4 of 36 (measured
  5.2-9.7 mm, predicted 0.3-3.5); along the paths, false clear 0 of 9, false
  too_close 4 of 31. At a 10 mm threshold: false clear 0 of 8 / 0 of 13,
  false too_close 1 of 32 / 0 of 27.
- **Along the path** the prediction was 3.2-32.8 mm under (mean 9.9). The
  measured path minimum comes from 3-18 polls of a move that takes well under a
  second, so it is an upper bound on the true minimum, and the twin's joint
  servo does not move exactly along the straight joint line sampled here; both
  widen this number.
- **Bases and model**: the bases derived from the arms' readings matched the
  twin's base bodies to 0.0 mm and 0.0 deg; the model-vs-connector forward
  kinematics check read under 0.001 mm on every call.
- Why it is conservative: offline, at 150 random fingertip-to-fingertip poses,
  the capsules sat 7.3 mm (mean; p90 10.9, min 1.1) under the dense surface of
  the fingertips' convex pieces, while the harness's vertex measure sat only
  0.8 mm (mean) over that surface. Capsules around plate-like fingertip pieces
  bulge; more, smaller capsules (or boxes) would tighten it. A first model
  with 6 k-means capsules per fingertip read 7.7-20.9 mm under (mean 14.1).

The handovers themselves transferred the cube in all six runs (three pairs,
the first with the 6-capsule fingertips; closest fingertip gaps 0.17-0.43 mm
at point 1 / offset 0.5, 0.32-3.12 mm at point 3). The node at the start of each graph (this skill as a subgraph, arms
at their start poses, no plan) routed clear at 347 mm (base to base).

## Connectors and arm_id

- RoboSimStudio: `robot.get_observation` lists both arms (0 = left / work,
  1 = right); `robot.describe_arm`, `robot.forward_kinematics` and
  `robot.solve_ik` take `arm_id`.
- The real YAM station bridge (`yam_real.py`, 2026-11-06) reports one arm in
  its observation and serves no `robot.forward_kinematics` or
  `robot.describe_arm`: this skill answers `unknown` ("no separate arm") there.
  With a dual-arm bridge it would run on the real joint readings; the model's
  joint zero and order would be checked by step 3 only if that bridge serves
  forward kinematics.

## Provenance

Adapted from RoboRSI's `check_dual_arm_collision` skill and its
`dual_arm_collision.py` (`check_pair_collision`, `check_trajectory_collision`),
Apache-2.0: pairwise cross-arm primitive distance minus radii, minimum with the
closest pair, attached-object primitives on the hand, the candidate-pose mode
through the arm's IK, and the trajectory mode that samples waypoints. Changed:
the robot is read only through robot.* tools (no joint state is written to
the robot to evaluate a candidate - the model's own forward kinematics is
used); YAM link capsules derived from the i2rt model replace the ALOHA sphere
YAML; the bases are located from the arms' own readings; whole planned paths of
either or both arms are sampled densely, not every 4th planner waypoint of one;
the result is routed clear / too_close / unknown instead of a boolean.
