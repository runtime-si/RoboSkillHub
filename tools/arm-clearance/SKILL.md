---
name: arm-clearance
description: Checks a planned tool pose or joint step of a serial arm against a
  scene twin made of boxes, without asking the robot - a local
  damped-least-squares inverse kinematics inside the joint limits and the
  controller's workspace box, the smallest distance from the arm's link
  capsules to the scene's boxes, and a step test that refuses a move ending
  inside a margin of the scene unless the arm is moving away. The chain, the
  capsules and the boxes are arguments, so it runs for any serial arm on CPU
  with numpy. Use when a skill searches over many candidate poses (hand
  orientations, heights, drop points, view poses) and each must be reachable
  and clear of fixed things the motion server does not know (a cabinet over
  the counter, a wall, a shelf), with a kinematic model of the arm and the
  obstacles' boxes in hand; send only the chosen pose to the robot.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [planning, ik, reachability, clearance, collision, scene-twin, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting, motion]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: A prediction from a model, not a guarantee - the source arm stood a few
      centimetres from where its joints said, so ask for a generous margin. Ships
      no arm model; the chain and capsules are arguments.
gap:
  requires: {}
  tools:
    - arm-clearance.solve_ik: Local IK for a serial chain passed as an argument; joints for a world tool pose, or why not. Asks no robot.
    - arm-clearance.clearance: Smallest distance from the arm's link capsules at given joints to a scene of boxes, and which link and box.
    - arm-clearance.check_step: Whether a joint step would end inside a margin of the scene (at its middle or end) and nearer than the arm is now.
    - arm-clearance.reachable: Whether the arm can take a tool pose clear of the scene - local IK, then the step test from the current joints.
---

# arm-clearance

Four pure functions for the inner loop of a pose search.

## Why it exists

Motion servers usually plan in free space: they know the arm and a workspace
box, not the cabinet over the counter. And a search that asks the server
"would you take this pose?" for every candidate is not free. On the station
this came from, a place step that asked the motion server for a plan per
candidate (hand orientations x heights x drop points: well over a hundred
requests in a row, each with a state request) held up the server's 20 Hz
control loop until the station's supervisor stopped both arms ("heartbeat
missed"), with a part in the hand. The search was then moved off the server
onto a local model; the next run's notes record the server healthy all run.

So: search here, send once. The server keeps the last word on the move that
is actually sent, and its refusal still has to be handled.

## The model argument

```python
model = {
    "chain": [  # parents before children; MJCF convention
        {"name": "base", "parent": None},
        {"name": "link1", "parent": "base", "position": [0, 0, 0.067],
         "quat": [0.7071, 0, 0, -0.7071], "joint": 0, "axis": [0, 0, 1]},
        # ...
    ],
    "tcp": {"body": "link6", "position": [0, 0, 0.13], "quat": [1, 0, 0, 0]},
    "limits": [[-2.6, 3.1], ...],          # rad, one pair per joint
    "base": {"position": {...}, "rotation": {...}},   # the root in the world
    "capsules": {"link2": [[[0, 0, 0], [0.26, 0, 0], 0.035]], ...},
}
obstacles = [{"name": "cabinet", "center": [0.1, 0.2, 0.7],
              "half": [0.5, 0.18, 0.2], "quat": [1, 0, 0, 0]}]
```

Each body sits at `position` / `quat` (wxyz) in its parent and, when it has
a `joint` index, turns about `axis` in its own frame at its origin, as in
MJCF. A capsule whose two ends coincide is a sphere: use one for a wrist
camera or a held object the arm model lacks. Obstacles may also be
OrientedBoundingBoxes (`center`, `extent`, `orientation`).

No arm model ships with this bundle. For the i2rt YAM, a chain and link
capsules transcribed from its MJCF are in
`skills/checking-dual-arm-clearance/scripts/check_clearance.py`.

## The four tools

- `arm-clearance.solve_ik(model, pose, q0, bounds=None, iters=80)`:
  `{ok, q, position_error_m, rotation_error_rad, reason}`. Damped least
  squares from `q0`, joints clipped to their limits each step, rotation
  weighted 0.3 against position. `ok` needs 1 mm and 0.02 rad. `bounds`
  (`{min, max}` in the chain root's frame) rejects poses outside the
  controller's workspace box before solving. It finds the solution nearest
  `q0`, not every branch: seed it with the arm's current joints.
- `arm-clearance.clearance(model, q, obstacles, limit_m=0.15)`:
  `{clearance_m, link, obstacle, nearest}`; negative inside a box.
- `arm-clearance.check_step(model, q_now, q_goal, obstacles, margin_m=0.02)`:
  `{blocked, reason, clearance_m, clearance_now_m}`. Blocked when the arm
  would be under the margin half way or at the goal **and** nearer than it is
  now by over 2 mm. An arm already inside the margin may always move away,
  otherwise it could never leave.
- `arm-clearance.reachable(model, pose, q_now, obstacles=None, margin_m=0.02,
  bounds=None)`: `{reachable, q, reason, clearance_m, nearest}`: the IK, then
  the step test from `q_now`.

## Limits

- **Two samples per step** (middle and end, in joint space). Check long moves
  as a chain of short ones; the source split moves into 6 cm / 20 degree
  steps.
- **Capsules as spheres** half a radius apart: the clearance reads up to 3 %
  of the radius short.
- **Boxes only.** Loose things, the other arm and the containers the hand
  goes into on purpose are not this tool's to vouch for.
- **The arm is not where its joints say.** On the source station the hand
  was 2 to 7 cm off across the workspace. The margin is the only allowance.
- **No self-collision, no dynamics, no path planning.**

## Constants, and what they were measured on

- `IK_ITERS = 80`, `IK_DAMPING = 1e-4`, `IK_STEP = 0.2 rad`,
  `IK_ROTATION_WEIGHT = 0.3`, `IK_ACCEPT = 1 mm / 0.02 rad`: the settings of
  the source task's local solver for a 6-joint i2rt YAM (0.6 m reach), the
  same method as that station's planners; not tuned for other arms. A pose
  it called reachable was still sometimes refused by the server on the way.
- `MOVE_AWAY_SLACK = 2 mm`, `CLEARANCE_LIMIT = 0.15 m`: the source's.
- `SPHERE_SPACING = 0.5`: chosen here.
- The margin is the caller's. The source's default was 2 cm; a run at 3 cm,
  with 15 cm asked over the container rims as well, left two held parts with
  no carry pose at all.

## Evidence

The method ran on one real YAM station on 2026-11-15, as a MuJoCo twin of a
kitchen (cabinet, doors, wall, side cabinets as boxes) with MuJoCo's geom
distance. In the one run that reports it, the twin put the arm 2.9 cm or more
from the cabinet all run where earlier runs had touched it, and no step had
to be refused. After the search moved to the local model, the next run's
notes record the motion server healthy throughout.

**This bundle is a numpy port of that method, not that code**: the chain
walk, the Jacobian, the sphere-box distance and the box format are new here
and have been exercised only by this repository's unit tests (a synthetic
6-joint arm). It has not driven a robot or been compared with MuJoCo on the
YAM model. Maturity is `community`.
