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

Draft.
