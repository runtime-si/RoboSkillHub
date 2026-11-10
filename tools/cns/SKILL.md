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

Draft.
