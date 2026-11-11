---
name: geco
description: GeCo low-shot detection and counting (MIT code; weights licence
  unstated - research only) - every instance like 1-3 exemplar boxes, with a
  count. Use when a skill needs all copies of a boxed example part (via
  skills/detecting-all-like-exemplar).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, counting, exemplar, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    notes: "Weights GeCo.pth (2.57 GB) from the authors' Google Drive (README link); licence not stated, research use only. Set GAP_GECO_WEIGHTS."
    provenance:
      authors: [Jer Pelhan, Alan Lukezic, Vitjan Zavrtanik, Matej Kristan]
      code: https://github.com/jerpelhan/GeCo/blob/c0fad20/demo.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2409.18686
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "geco"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - geco.detect_like: Every instance like 1-3 exemplar boxes (boxes, scores, count).
---

# geco

`geco.detect_like(image, exemplar_boxes, score_ratio=0.25)` returns `{boxes,
scores, count, seconds}`.

The inference path is GeCo's `demo.py`:
- resize and pad (exemplars scaled to at most 80 px);
- normalise;
- run the model;
- keep boxes scoring above max · score_ratio;
- NMS at 0.5.

One or two exemplars are repeated to three, because GeCo was trained with
three. The model code (`geco_core/`) is vendored unchanged. Measured: see the
skill.
