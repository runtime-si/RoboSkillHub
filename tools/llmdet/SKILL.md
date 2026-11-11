---
name: llmdet
description: LLMDet open-vocabulary detection (CVPR 2025, Apache-2.0 code and
  weights) - text labels in, labelled boxes + scores out, in grounding-dino's
  shape, so it drops into any skill or sahi slice that takes grounding-dino /
  mm-grounding-dino. Runs on transformers 5.x (MM-Grounding-DINO does not).
  Use when a graph needs parts found by name (cube, box, pen, plug, ring) on
  the hub's main transformers stack.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, open-vocabulary, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    weights:
      uri: hf://iSEE-Laboratory/llmdet_large@bec37f296f05b22f6c6b39bc05a6c611239f4e31
      sha256: d3710986876844edf0c394594f7e698165ae1affbc611596182891b21f48e56e
      license: Apache-2.0 (model card); base 3537869413e7dca01cfd86d6ab10553b72e4fe29, tiny d05199165a19320a9236396e20fca0a5e065189c with GAP_LLMDET_MODEL
    provenance:
      authors: [Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, Wei-Shi Zheng]
      code: https://github.com/iSEE-Laboratory/LLMDet/blob/53366243fba7/README.md
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2501.18954
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "llmdet"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - llmdet.detect: Open-vocabulary detection with LLMDet (labelled boxes + scores, grounding-dino's shape).
---

# llmdet

`llmdet.detect(image, prompts | query, box_threshold=0.3, text_threshold=0.2,
max_detections=0)` takes the same arguments and returns the same detection
dicts as `mm-grounding-dino.detect`. Checkpoints: large (default), base and
tiny, pinned in `MODELS`.

## Measured: the 7 labelled YAM overhead frames

Same frames, labels and protocol as `tools/mm-grounding-dino`: 12 small
parts, 3 of them cubes at ~1 m, and 4 open boxes. Raw answers were recorded
once and replayed per score floor. Reproduce with
`tests/bundles/_llmdet_station_eval.py` and `_owlv2_station_score.py text`.

Run on rshgpu2 with transformers 5.22.2 and torch 2.8, **on a GPU at 100 %
shared with other jobs**, so latencies are contention:

| backend, config | floor | small parts | cubes at ~1 m | open boxes | FP (7 frames) | s / frame |
| --- | --- | --- | --- | --- | --- | --- |
| **LLMDet large, ROI** | **0.3** | 7 / 12 | 3 / 3 | **4 / 4** | **9** | 1.9 |
| LLMDet large, ROI | 0.4 | 6 / 12 | 3 / 3 | 4 / 4 | 1 | 1.9 |
| LLMDet large, ROI, s256 | 0.3 | **8 / 12** | 3 / 3 | 4 / 4 | 25 | 8.4 |
| LLMDet large, full frame | 0.3 | 6 / 12 | 3 / 3 | 1 / 4 | 17 | 2.6 |
| MM-GDINO large, ROI (its SKILL.md) | 0.25 | 7 / 12 | 3 / 3 | 4 / 4 | 9 | 0.73 |
| MM-GDINO large, ROI, s320 | 0.3 | 8 / 12 | 3 / 3 | 4 / 4 | 8 | 3.2 |
| owlv2 base, full | 0.2 | 7 / 12 | 3 / 3 | 0 / 4 | 18 | 0.22 |

**Reading:**
- On the table ROI, LLMDet large matches the hub's best text detector
  (MM-Grounding-DINO large): same recall, same false positives.
- Its slicing configs found no more small parts and added more false
  positives than MM-GDINO's.
- **Its advantage is the stack:** it decodes correctly on transformers 5.x,
  so it runs in the hub's main environment. MM-Grounding-DINO needs
  transformers 4.x.
- Tiny was still running at the time limit and is not reported.
