---
name: cns
description: CNS learned visual servoing (MIT; arXiv 2309.09047) as tools -
  record a goal view, then for each live frame of the same camera get the
  camera velocity toward it from keypoint correspondences (SIFT / ORB /
  AKAZE) turned into a graph and a recurrent graph network, plus the pixel
  error and CNS's stop decision. Use when a skill servos a camera back to a
  recorded view without a hand-tuned IBVS gain.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, visual-servo, keypoints, learned]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: https://raw.githubusercontent.com/hhcaz/CNS/6b42138ea8f15d872eeca916b9e1fb1f429bcac4/checkpoints/cns_state_dict.pth
      sha256: 2b91c0d5b5b6d51c42456cf1d6cb798c2b66779d5e4f2c460b1963ee897eb67f
      license: MIT (shipped in the CNS repository)
    provenance:
      authors: [hhcaz]
      code: https://github.com/hhcaz/CNS/blob/6b42138ea8f15d872eeca916b9e1fb1f429bcac4/cns/benchmark/pipeline.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2309.09047
gap:
  requires: {weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "cns"]
    protocol: stdio-msgpack
    requires_gpu: false
  tools:
    - cns.start: Start a session on a goal view (RGB, intrinsics, camera-to-scene distance).
    - cns.step: Camera velocity (camera frame) toward the goal view from the current frame, with error and stop decision.
    - cns.stop: Close a session.
---

# cns

The pipeline of CNS @ 6b42138 (MIT; arXiv 2309.09047). The vendored, trimmed
modules are in `cns_core/`:
- the classic frontend (SIFT / ORB / AKAZE);
- the correspondence-graph midend;
- GraphVS and its controller;
- the pixel stop policy.

The weights (1.9 MB, MIT) download on first use from the pinned commit and
are checked against the sha256.

| tool | returns |
| --- | --- |
| `cns.start(goal_image, intrinsics, distance_m, detector="SIFT")` | `{session_id, keypoints, ok}` |
| `cns.step(session_id, image)` | `{velocity [vx, vy, vz, wx, wy, wz] camera frame, matches, error, converged, seconds}` |
| `cns.stop(session_id)` | `{closed}` |

Measured on an RTX PRO 6000: 0.03-0.05 s per step (0.28 s for the first),
71 keypoints on the twin's wrist view. For the closed-loop result (it
diverged in the twin), see `skills/servoing-by-learned-keypoint-correspondence`.

Changes:
- sessions are keyed by id;
- images are RGB in (converted to CNS's BGR);
- the stop policy counts calls instead of seconds;
- `perception.py` keeps only `CameraIntrinsic` (the rest needs open3d /
  pybullet);
- the controller keeps only GraphVS;
- progress prints are removed.
