---
name: edgetam
description: EdgeTAM promptable video mask tracking (SAM 2 architecture, Apache-2.0)
  -- seeded with a box, points or a mask on one frame, follows one part
  through a live camera stream or a recorded clip, returning its mask, box and
  an occlusion-head presence call per frame, and picks it up again after an
  occlusion. Live sessions hold constant GPU memory. SAM 2.1 sizes load
  through the same tools. Use when a closed loop or a verify stage must keep
  the identity of one part (a cube, a box, a plug) while the arm or a hand
  passes over or moves it; seed it from a detector or sam3.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, tracking, video-segmentation, sam2, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, insertion]
    weights:
      uri: hf://yonigozlan/EdgeTAM-hf@c266ce53b3fc00f0f495b583f6a116c4e57f53bb
      license: Apache-2.0 (model card; conversion of facebook/EdgeTAM edgetam.pt)
    provenance:
      authors: [Chong Zhou, Chenchen Zhu, Yunyang Xiong, Saksham Suri, Fanyi Xiao, Lemeng Wu, Raghuraman Krishnamoorthi, Bo Dai, Chen Change Loy, Vikas Chandra, Bilge Soran]
      code: https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/edgetam_video
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2501.07256
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "edgetam"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - edgetam.tracker_init: Open a mask-tracking session on a live frame from a box, points or a mask.
    - edgetam.tracker_update: Advance a session by one frame; mask, box, confidence, presence.
    - edgetam.tracker_close: Free a session (idempotent).
    - edgetam.track: Track one object through a recorded clip from a prompt on any frame.
---

# edgetam

(measurements pending)
