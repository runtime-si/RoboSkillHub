---
name: owlv2
description: OWLv2 open-vocabulary detection (google/owlv2-*, Apache-2.0)
  in two modes -- text prompts to labelled boxes in grounding-dino's shape,
  and image-guided one-shot search ("every brick / tile / piece like this
  one") from a box around one example, in the same frame or another, with no
  name needed. Use when a workflow must find all copies of a part it can
  point at but not name well (a 2x2 brick vs a 2x4, a domino, a jigsaw
  piece), or wants a fast detector (0.06 s a frame) for 15-25 px parts at
  ~1 m; also a backend of tools/sahi.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, exemplar, one-shot, small-objects, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    weights:
      uri: hf://google/owlv2-base-patch16-ensemble@cfd3195ba4ea9592eec887ded089f4c08eff231d
      sha256: 6b82a41489619fd6f057de88514b307b908ed0b19f6850ab2d3c0911c23af197
      license: Apache-2.0 (model card); google/owlv2-large-patch14-ensemble@95e26936e865f87db1742128404b3c035d47d89d (Apache-2.0, manifest sha256 c20b2fcc3653aa263ac8031655a87c7cc1a55f7e47a2ea4169d8406808c134ce) with GAP_OWLV2_MODEL
    provenance:
      authors: [Matthias Minderer, Alexey Gritsenko, Neil Houlsby]
      code: https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/owlv2
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2306.09683
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "owlv2"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - owlv2.detect: Open-vocabulary detection from text prompts (labelled boxes + scores, grounding-dino's shape).
    - owlv2.embed_exemplar: The OWLv2 query embedding of one object (a box in a frame, or a crop) as a string.
    - owlv2.detect_like: Every object in an image that looks like an exemplar (box in this or another frame, crop, or string).
---

# owlv2

[OWLv2](https://arxiv.org/abs/2306.09683) through transformers'
`Owlv2ForObjectDetection`, weights `google/owlv2-base-patch16-ensemble`
(default, pinned commit) or `google/owlv2-large-patch14-ensemble`
(`GAP_OWLV2_MODEL`); both model cards are Apache-2.0. Image in: RGB uint8
`[H, W, 3]`; boxes out in frame pixels.

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
# text: grounding-dino's shape, so skills can swap backends
dets = ctx.tool("owlv2.detect", image=cam["rgb"], prompts=["cube", "plug"])["detections"]
# exemplar: the operator / a VLM boxed one brick; find all its look-alikes
r = ctx.tool("owlv2.detect_like", image=cam["rgb"], query_box=one_brick, exclude_query=True)
for box, score in zip(r["boxes"], r["scores"]):
    ...  # colour.classify / a size check, sam3.segment_box for a mask
# the same exemplar on later frames, or sliced with sahi
q = ctx.tool("owlv2.embed_exemplar", query_image=cam["rgb"], query_box=one_brick)["query"]
r = ctx.tool("owlv2.detect_like", image=next_rgb, query=q)
r = ctx.tool(
    "sahi.detect_sliced", image=next_rgb, prompts=[q], backend="owlv2.detect_like", roi=table
)
```

## Tools

| call | gives |
| --- | --- |
| `owlv2.detect(image, prompts, threshold=0.2, query=None, template="a photo of a {}", nms_iou=0.3, max_detections=0)` | `{detections: [{box, label, score}]}` best-first, `label` = the prompt |
| `owlv2.embed_exemplar(query_image, query_box=None)` | `{query, exemplar: {box, iou, objectness}, dim}`; `query` is the embedding as a string |
| `owlv2.detect_like(image, query_image=None, query_box=None, threshold=0.87, query=None, nms_iou=0.3, exclude_query=False, label="exemplar", max_detections=0)` | `{detections, boxes, scores, found, count, exemplar}` |

- **Text mode** labels every box with the prompt it scores highest on
  (`post_process_object_detection`'s argmax), keeps it over `threshold` and
  suppresses same-label boxes over `nms_iou` IoU. Prompts may be a list or
  one string split on `.` (`"cube. plug."`, grounding-dino's form).
- **Exemplar mode** takes the class embedding of OWLv2's own box that best
  covers `query_box` (transformers' OWL-ViT rule: among boxes within 0.8 of
  the best IoU, the least typical one), and scores every box of the target by
  cosine similarity with it. The exemplar is `query` (a string from
  `embed_exemplar`), else `query_box` in `query_image`, else `query_box` in
  `image` itself (one forward pass), else `query_image` as a crop (its most
  object-like box). In its own frame the exemplar is found too;
  `exclude_query` drops it. `exemplar.iou` under ~0.5 means OWLv2 had no
  box fitting the query (an object too small or flat for it); check it.
- **As a sahi backend.** `sahi.detect_sliced(..., backend="owlv2.detect")`
  (text) or `backend="owlv2.detect_like"` with `prompts=[query_string]`
  (exemplar) go through sahi's generic `(image, query)` path; this bundle's
  defaults are the floors there (raise `min_score` to tighten). With the
  exemplar backend the sahi `label` is the query string.

Scores: text mode is the class head's sigmoid (as upstream). Exemplar mode
is the cosine similarity, not upstream's `post_process_image_guided_detection`
"alphas" (they rescale every image so its best box is 1.0, even when nothing
resembles the query) nor the class head's sigmoid against an image embedding
(0.99+ for most boxes of a frame on these images).

## Install

```bash
uv sync --project tools/owlv2      # torch + transformers; no torchvision needed
```

Weights download from Hugging Face on first call or `gap skills check
--download` (base 2 x 620 MB: the snapshot carries `.bin` and
`.safetensors`; large 2 x 1.75 GB). Where huggingface.co is unreachable set
`HF_ENDPOINT` to a mirror. Env knobs: `GAP_OWLV2_MODEL` (one of the two
pinned repos or a local directory), `GAP_OWLV2_DEVICE` (default `cuda`; CPU
works, slowly), `GAP_OWLV2_PREPROCESS=hf` (transformers' CPU processor
instead of the same steps on the device).

## Measured

2026-11-08 on an AutoDL RTX 4090 D (24 GB), transformers 5.19.0, torch
2.10.0, on the 7 real YAM overhead frames (Orbbec Gemini, 640 x 480) and hand
labels of the sahi check (tools/sahi): 12 small parts (4 cubes 30 mm, 16-36
px; 1 pen; 4 black rings on the black threading board; 3 cable plugs 14-17 x
8-12 px) and 4 open boxes. A hit is IoU >= 0.3 with a label of the same
name; every other box is a false positive (FP). Each backend called
in-process through `sahi.detect_sliced` (`slice_px=0` = one call on the frame
or the table ROI), the raw answers replayed per score floor. Reproduce with
`tests/bundles/_owlv2_station_eval.py` and `_owlv2_station_score.py` (labels
in `tests/bundles/data/station_small_parts_labels.json`).

**Text mode vs grounding-dino** (prompts `cube, box, pen` / `ring, plug`):

| backend, slicing | floor | small parts | cubes at ~1 m | open boxes | FP (7 frames) | s / frame |
| --- | --- | --- | --- | --- | --- | --- |
| grounding-dino-base, frame | 0.2 | 4 / 12 | 0 / 3 | 3 / 4 | 23 | 0.16 |
| grounding-dino-base, table ROI | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 | 18 | 0.14 |
| grounding-dino-base, ROI, sahi 320 px | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 | 30 | 0.65 |
| grounding-dino-base, sahi 256 px | 0.2 | 6 / 12 | 1 / 3 | 3 / 4 | 61 | 1.24 |
| **owlv2 base, frame** | 0.2 | **7 / 12** | **3 / 3** | 0 / 4 | **18** | **0.06** |
| owlv2 base, frame | 0.15 | 7 / 12 | 3 / 3 | 0 / 4 | 33 | 0.06 |
| owlv2 base, frame | 0.1 | 6 / 12 | 3 / 3 | 0 / 4 | 56 | 0.06 |
| owlv2 base, sahi 256 px | 0.2 | 8 / 12 | 3 / 3 | 0 / 4 | 26 | 0.60 |
| owlv2 base, ROI, sahi 256 px | 0.15 | 8 / 12 | 3 / 3 | 0 / 4 | 25 | 0.26 |
| owlv2 large, frame | 0.15 | 8 / 12 | 3 / 3 | 2 / 4 | 20 | 0.63 * |
| owlv2 large, ROI, sahi 256 px | 0.2 | 8 / 12 | 3 / 3 | 2 / 4 | **8** | 2.0 * |

- OWLv2 base on the whole frame finds what grounding-dino needs a table crop
  for: all 4 cubes and all 3 plugs, at 0.06 s. Slicing adds the pen (8 / 12)
  and, at 0.2, more FPs. The prompt template made no real difference
  (`"{}"`: 8, 7, 6 / 12 with 48, 29, 11 FP at 0.1, 0.15, 0.2).
- **Base calls the open box "cube"** (0.16-0.35) on all four cube frames: 4
  of its FPs and 0 / 4 boxes. Size words do not fix that; use exemplar mode
  or a size check. Large labels it "box" on 2 of 4.
- **No backend found the black rings on the black board** (0 / 4), as with
  grounding-dino, sam3 and locate-anything in the sahi check.

**Exemplar mode** (`detect_like`, cosine floor). Cross-frame: the cube (or
plug) of one frame as the exemplar, searched in each other frame of its kind
(12 cube searches, 6 plug searches; the open box of the same grey plastic is
the hard negative). Within-frame: one black ring as the exemplar, the other
ring to find (4 searches). Composite: each cube frame with crops of the
other three frames' cubes pasted on its table (3 to find per frame, the
exemplar its own cube; real pixels, synthetic layout).

| model, slicing | floor | cubes, cross-frame | FP (on the open box) | plugs, cross-frame | FP | rings, within frame | FP | pasted cubes | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base, frame | 0.85 | 12 / 12 | 1 (1) | 6 / 6 | 12 | 3 / 4 | 10 | 10 / 12 | 2 |
| **base, frame** | **0.87** | **12 / 12** | **0** | **6 / 6** | **8** | **3 / 4** | **8** | 8 / 12 | **0** |
| base, frame | 0.88 | 12 / 12 | 0 | 6 / 6 | 5 | 3 / 4 | 6 | 8 / 12 | 0 |
| base, frame | 0.90 | 10 / 12 | 0 | 5 / 6 | 2 | 2 / 4 | 4 | 6 / 12 | 0 |
| base, ROI, sahi 256 px | 0.87 | 11 / 12 | 0 | 5 / 6 | 8 | 3 / 4 | 4 | 6 / 12 | 1 |
| large, frame | 0.85 | 12 / 12 | 2 (2) | 6 / 6 | 15 | 3 / 4 | 8 | 11 / 12 | 1 |
| **large, frame** | **0.87** | **12 / 12** | **0** | 6 / 6 | 5 | 3 / 4 | 6 | **9 / 12** | **0** |
| large, frame | 0.90 | 12 / 12 | 0 | 6 / 6 | 3 | 1 / 4 | 5 | 7 / 12 | 0 |

- The exemplar finds **the black rings text mode never finds** (3 / 4 with
  either model) and every plug from another frame's plug, at the price of
  1-2 FPs per search on the board (its holes and cable ends look alike).
- **Cosine separates part from same-coloured clutter by ~0.03-0.1**: other
  frames' cubes score 0.89-0.97 (base), the open box 0.77-0.88. 0.87 is the floor
  for both models; under 0.85 the open box comes in, over 0.90 the cubes
  from the brighter / darker table start to drop.
- Pasted cubes from other lighting (8-11 / 12) are harder than the same
  cube seen in another frame: the exemplar should come from the same camera
  and light as the search.
- **Slicing does not help exemplar mode** on these frames: the exemplar's
  own embedding already finds 16-36 px parts on the whole frame, and slices
  add calls and FPs. Use sahi with it only for parts under ~15 px.
- `exemplar.iou` (OWLv2's box vs the hand label) was 0.42-0.96: the
  half-hidden ring gave 0.42 and still worked.

**Latency, VRAM.** Clean (the GPU otherwise idle), 640 x 480 frame:
`owlv2.detect` base 0.06-0.07 s (0.072 s median over the 7 frames), of which
transformers' CPU processor would add 0.27 s (`GAP_OWLV2_PREPROCESS=hf`:
0.31 s; same boxes within 1 px and scores within 0.004 on all 7 frames);
grounding-dino-base 0.14-0.16 s. Rows marked * and the following were
measured while other jobs held the shared GPU at 100 %, all models
interleaved in one process (grounding-dino's frame call took 0.95 s there,
6x its idle time): base `detect` 0.33 s, `embed_exemplar` 0.29 s,
`detect_like` 0.21-0.29 s; large 1.08-1.50 s per call (4.6x base);
grounding-dino 0.95 s. Peak torch memory: base 0.78 GB (0.87 GB reserved),
large 2.0 GB (2.2 GB), grounding-dino-base 2.2 GB (3.5 GB). First call
(weights from the local cache): base 5-7 s, large 10-20 s.

Use: base, whole frame, for text detection of small parts (floor 0.2), then
a colour / size check; `detect_like` at 0.87 from an exemplar boxed in the
same camera to find all copies of a part; large when the extra latency is
affordable and clutter of the same colour is close (pasted cubes 9 vs 8 /
12, plug FPs 5 vs 8). No LEGO bricks, dominoes or jigsaw pieces were on the
station when these frames were taken; recheck the floors on them.

## Gotchas

- Boxes are relative to the padded square the processor makes (bottom /
  right padding to the longer side): the bundle scales by `max(H, W)` and
  drops boxes centred in the padding; do not rescale them again.
- An exemplar string belongs to one model size (512 values for base, 768
  for large); the other model refuses it.
- Exemplar scores are cosines, not probabilities: compare them with each
  other and with the 0.87 floor, not with text-mode scores.
- Parts touching each other: NMS at IoU 0.3 can merge two adjacent bricks
  whose boxes overlap that much; lower `nms_iou` or raise it to 1.0 and
  dedupe downstream.
- The model is a lazy module-level singleton; the first call pays the load.
