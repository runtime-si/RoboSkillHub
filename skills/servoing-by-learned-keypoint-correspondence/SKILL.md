---
name: servoing-by-learned-keypoint-correspondence
description: Brings the wrist camera back to a recorded goal view with CNS, a
  learned visual servo (MIT, weights public) - SIFT/ORB/AKAZE keypoints
  matched between the goal and the live frame become a graph, and a recurrent
  graph network outputs the camera velocity; each velocity is applied as one
  small capped hand step (x, y, z and yaw on a table by default), until CNS's
  pixel stop policy says converged. Use when a part or the hand must return
  to a taught relation with a textured target (a printed board, a stud grid,
  a jigsaw picture) and a goal frame can be recorded at the right pose.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [visual-servo, learned, wrist-camera, alignment, keypoints]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, lego, jigsaw]
    provenance:
      authors: [hhcaz]
      code: https://github.com/hhcaz/CNS/blob/6b42138ea8f15d872eeca916b9e1fb1f429bcac4/demo_real_ur5.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2309.09047
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_to_joints
    - cns.start
    - cns.step
    - cns.stop
  required_inputs:
    goal: Observation
    camera: string
  produces_outputs:
    frames: int
    final_error: float
    turned_deg: float
    reason: string
  exit_conditions:
    converged: CNS's pixel stop policy held its error near its minimum (under 0.005 normalized) for 6 frames.
    lost_correspondence: Fewer than min_matches (8) goal-live matches for lost_limit (3) frames in a row, no frame / intrinsics / camera pose, or no distance to the scene.
    timeout: max_frames (120) frames without converging, or a step was refused.
  canonical_scripts:
    - servo_cns: scripts/servo_cns.py
  examples:
    - title: Record the goal view, move away, servo back
      path: examples/canonical_subgraph.json
---

# servoing-by-learned-keypoint-correspondence

**Status: the port runs, but it diverged in the YAM twin; do not use it yet**
(see Measured).

CNS (arXiv 2309.09047, MIT, weights in the repo) servos a camera back to a
recorded goal view:
- keypoints (SIFT by default) are matched between the goal and the live frame;
- the matches become a graph;
- a recurrent graph network outputs the camera velocity.

`tools/cns` wraps CNS's own pipeline. This skill turns each velocity into one
capped hand step: the velocity goes through the observation's camera pose
into the world, times `dt_s`, capped at 4 mm / 1.5 deg. In `plane` mode
(the default) only x, y, z and the turn about world +z are applied, the turn
through the last joint; `full` sends the whole rotation to `step_tool`.

## Measured (YAM station twin, wrist D405 render, box target)

- **Goal and trials:** the goal is the wrist view with the hand 10 cm over
  the box. Trials start from the same random offsets as the discrete-move
  classifier's check (2.9-13 mm, up to 4 deg).
- **The tool works offline.** The goal view gave 71 SIFT keypoints, and
  64-65 matched after a 20 px shift.
- **The loop diverged on every trial:**
  - Before the arrival `tolerance` was passed: 8 trials ran into the 120-frame
    cap. CNS's error grew from 0.004-0.07 to 0.05-0.28, and matches fell from
    about 40 to 8-28.
  - With `tolerance` 0.3 mm: one trial went from 6.6 mm to 18 mm off, and
    another from 13 mm to 115 mm and -65 deg, with no `converged` exit.
- **Likely causes, not yet separated:**
  - `plane` mode drops CNS's tilt rates, while the network's recurrent state
    assumes its full 6-DoF velocity was executed. The wrist camera is
    tilted 45 deg.
  - Integrating at `dt_s` 0.5 s with a step cap, instead of streaming
    velocity.
  - The path tracer's noisy frames.
- **Single-axis check** (hand +4.6 mm in x, no turn, `dt_s` 0.3; 5 steps
  before the shared host ran out of time):
  - CNS commanded mostly rotation (wx up to 0.18 rad/s, wy 0.07-0.18) and
    little translation.
  - Plane mode applied only the translation, and the hand moved further
    +x by 1.3-2.6 mm per step: 4.6 mm became 14 mm off.
  - CNS's error grew from 0.010 to 0.048; matches fell from 41 to 28.
  - Either the translation sign / frame is wrong, or the translation without
    its rotation is the wrong command (a 45 deg tilted camera sees a
    sideways offset much like a tilt). Not separated yet.
- **Next:**
  - repeat the single-axis check with the translation negated;
  - `mode="full"` on an arm whose IK honours the full rotation (the twin's
    leaves the wrist roll free);
  - a smaller `dt_s`.

Compare `correcting-in-plane-offset-by-discrete-moves`, which brought the
same starts to a median 0.52 mm.

## Licence

Adapted from CNS (MIT, Copyright (c) 2023 hhcaz). See the script header,
tools/cns and NOTICE.
