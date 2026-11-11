---
name: detecting-all-like-exemplar
description: Finds every instance that looks like 1-3 example boxes in a camera
  frame and counts them - GeCo low-shot detection (geco.detect_like) or the
  hub's owlv2.detect_like - for "all the dominoes / studs / pieces like this
  one" without naming the part. Use when a graph has one example of a part
  (a box from a detector, a click, a previous step) and needs all its copies.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, counting, exemplar, few-shot]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [sorting, domino, lego, jigsaw]
    provenance:
      authors: [Jer Pelhan, Alan Lukezic, Vitjan Zavrtanik, Matej Kristan]
      code: https://github.com/jerpelhan/GeCo/blob/c0fad20/demo.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2409.18686
gap:
  allowed_tools:
    - robot.get_observation
    - geco.detect_like
    - owlv2.detect_like
  required_inputs: {}
  produces_outputs:
    count: int
    reason: string
  exit_conditions:
    found: At least min_count (1) boxes like the exemplars; `boxes`, `scores`, `count`.
    not_found: Fewer boxes, no RGB frame, or the backend failed; `reason` says which.
  canonical_scripts:
    - detect_like: scripts/detect_like.py
  examples:
    - title: Count the copies of one boxed part in the overhead frame
      path: examples/canonical_subgraph.json
---

# detecting-all-like-exemplar

Give it 1-3 boxes around examples. It returns a box for every look-alike and
the count:
- `backend="geco"` (default): GeCo, NeurIPS 2024, MIT code. Weights are
  **research-only until licensed**: the authors' Google Drive checkpoint
  states no licence.
- `backend="owlv2"`: the hub's OWLv2 image-guided search with the first box.

## Measured (real station frames, 8 pasted identical 18 x 14 px cubes + the real one)

**Setup:**
- Frames: 9 frames of the 2026-11-03 overhead recording, with the real arm
  moving and covering parts (the #131 clip).
- Exemplar: one box, pasted cube 2.
- Scoring: matched at IoU >= 0.3 against the visible pasted cubes; a hit on
  the real cube is reported apart.
- Hardware: rshgpu2.

| backend | recall | false boxes (9 frames) | real-cube hits |
| --- | --- | --- | --- |
| geco (score > max / 4, the demo's rule) | 58/60 | **65** | 3/3 |
| owlv2.detect_like (default threshold 0.87) | 58/60 | **2** | 3/3 |
| owlv2.detect_like at threshold 0.5 | 60/60 | 2927 | 3/3 |

**On these small parts GeCo finds as many as OWLv2 but has 30 times the false
boxes**: about 7 per frame on the arm, the rack and the table grain. Its
max / 4 rule is relative, so a frame always returns its best few even when
they are wrong. Prefer `owlv2` for sparse small parts. GeCo is trained for
dense crowds (FSC-147), such as a row of pips or studs; that case is not
measured here.

## Licence

GeCo code: MIT (`tools/geco/LICENSE.GeCo`). Its bundled `segment_anything`:
Apache-2.0 (`tools/geco/LICENSE.segment_anything`). The weights' licence is
unstated: research only.
