---
name: sahi
description: Slicing-aided detection (SAHI) of small parts -- cuts the camera
  frame into overlapping slices, runs one of the hub's detectors
  (grounding-dino, sam3, locate-anything) on every slice and on the whole frame,
  and merges the boxes with SAHI's postprocess, returning boxes, labels, scores
  and (sam3) masks in frame pixels. Use when the parts are 10-30 px in the
  overhead camera (2-3 cm bricks, dominoes, jigsaw pieces, a plug at ~1 m) and a
  single detector call on the whole frame misses them.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, small-objects, slicing, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    provenance:
      authors: [obss]
      code: https://github.com/obss/sahi/blob/10061afd5fa076e00f8eda55e2ee290f9d72f3c1/sahi/predict.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2202.06934
gap:
  requires: {}
  tools:
    - sahi.detect_sliced: Detect small parts by running a detector on overlapping slices and merging the boxes (SAHI).
    - sahi.slice_windows: The overlapping slice windows SAHI cuts an image (or a region of it) into.
    - sahi.merge: Merge overlapping detections from several slices with SAHI's postprocess.
---

# sahi

[SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference, MIT)
around the hub's detectors. A detector resizes its input to a fixed size
(grounding-dino: 800 px short side; SAM3: 1008 px), so on a 640 x 480 frame a
15 px part stays ~25 px and is missed. Cut the frame into 320 px slices and
each slice is upsampled x2.5-3.2: the part is ~40-50 px for the model. SAHI
runs the detector on every slice and once on the whole frame (for parts larger
than a slice), shifts the boxes to frame pixels and merges duplicates
(greedy non-maximum merging over intersection-over-smaller, so the two halves
of a part cut by a slice edge join its whole box from the neighbouring
slice).

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
r = ctx.tool(
    "sahi.detect_sliced",
    image=cam["rgb"],
    prompts=["cube", "plug"],
    backend="grounding-dino",
    roi=table_box,
    min_score=0.3,
)
for box, label, score in zip(r["boxes"], r["labels"], r["scores"]):
    ...  # e.g. sam3.segment_box(box) for a mask, geometry.* for a 3D point
```

## Tools

| call | gives |
| --- | --- |
| `sahi.detect_sliced(image, prompts, backend="grounding-dino", slice_px=320, overlap=0.2, ...)` | `{detections, boxes, labels, scores, masks, found, count, raw_slice, raw_full, n_slices, n_backend_calls, slice_windows, backend, timing}` |
| `sahi.detect_sliced(..., slice_px=0)` | the backend once on the image / ROI through the same filters (the unsliced baseline) |
| `sahi.slice_windows(width, height, slice_px, overlap, roi)` | the windows, for a detector `detect_sliced` does not call |
| `sahi.merge(detections, postprocess, match_metric, match_threshold, class_agnostic)` | SAHI's merge over `[{box, label, score}]` in frame pixels |

`detections` has `grounding-dino.detect`'s shape (`box`, `label`, `score`)
plus `n_merged` (raw boxes merged into it) and `sources` (`slice` / `full`);
`boxes` / `labels` / `scores` are the same as parallel lists, best-first.
`label` is the prompt the backend's label matches (grounding-dino returns text
spans). Nothing found is an empty list, never an error; a backend error
propagates.

**Backends.** `grounding-dino` (one call per slice, all prompts joined
`"a. b."`; `box_threshold` / `text_threshold` passed when > 0), `sam3`
(`sam3.segment_text` once per slice **and prompt**; `return_masks=True` gives
full-frame uint8 0/255 masks, unioned across slices), `locate-anything`
(`locate-anything.detect`, categories; non-commercial weights, see that
bundle; it reports no confidence, every score is 1.0), or any tool name taking
`(image, query)` and returning `{detections}` (`locate-anything.ground`,
`gemini-er.detect`). Not ultralytics: SAHI's YOLO backend is AGPL-3.0 and is
neither used nor installed.

**Knobs** (defaults are SAHI's unless noted): `overlap` 0.2;
`postprocess` `GREEDYNMM` (or `NMM`, transitive; `NMS`, keep the best);
`match_metric` `IOS`, `match_threshold` 0.5; `class_agnostic` False (merge per
label); `full_frame` True (SAHI's `perform_standard_pred`). Added here:
`roi` (slice only a region, e.g. the table: fewer calls, nothing found on the
shelves behind it), `min_score`, `min_box_px` 3 / `max_box_px` 0 (longer side
bounds in frame pixels; 0 = none), `max_box_fraction` 0.5 (drop a box covering
more than half the slice or image it was found in: the "whole slice" box an
open-vocabulary detector gives when the slice holds no such part),
`max_detections`.

## How it runs

The bundle is in-process (no `gap.serving`, numpy only): `detect_sliced`
declares `ctx` and calls the backend with `ctx.tool(...)`, so the backend
bundle must be installed and is what runs on the GPU; each slice is one RPC
call to it. Every call counts against `GAP_MAX_PERCEPTION_CALLS` when that is
set (7 per frame at the defaults on 640 x 480; x prompts for sam3).

Ported from SAHI at commit `10061afd` (release 0.14.0), see `tools.py`'s
header: `get_slice_bboxes`, the numpy postprocess (`nms` / `greedy_nmm` /
`nmm` over a dense IoU / IoS matrix, per label), the merge of a pair (union
box, higher score, the higher-scored label, union mask) and the flow of
`get_sliced_prediction`. Not taken: the `sahi` package itself (shapely,
OpenCV, the model zoo), its sparse / numba / torchvision postprocess backends
(a frame gives hundreds of raw boxes, not thousands), and its silent switch to
NMS / IOU for model confidence thresholds under 0.1.

## Measured

Measured 2026-11-07 on the Brev L40S with real YAM overhead frames (Orbbec
Gemini, 640 x 480, fx 461 px). Each backend bundle was served the way gap
serves it (`uv run --project tools/<backend> -- python -m gap_core.rpc.server`)
and `detect_sliced` called it over the RPC; the boxes were saved unmerged and
merged afterwards with this bundle's `merge` at each score floor (the same code
path as one call). Merges are SAHI's default (greedy NMM, IOS 0.5); the port's
slicing and all three postprocesses were checked identical to sahi 0.14.0 on
300 random slicings and 9,600 random merge cases.

| frame | table | small parts (hand-labelled boxes) | larger |
| --- | --- | --- | --- |
| 2026-11-03 05:11, 06:30 | wood, ~1 m | 30 mm cube, 19 x 21 px | open box |
| 2026-11-01 07:18 | white, ~1 m | 30 mm cube 16 x 18 px; red pen 79 x 11 px | open box |
| 2026-10-31 | white cloth, 0.57 m | 30 mm cube 30 x 36 px | open box |
| 2026-11-03 11:44, 14:21 | threading board | two black rings 12-18 x 29-32 px on the black board; the cable plug 14-17 x 8 px | - |
| 2026-11-04 05:38 | threading board | the cable plug 16 x 12 px | - |

12 small parts and 4 boxes in 7 frames. Prompts: `cube`, `box`, `pen` on the
cube frames; `ring`, `plug` on the board frames. A hit is IoU >= 0.3 with a
hand label of the same name (labels are 8-36 px; 2 px of error on a 10 px box
costs ~0.3 of IoU); every other box is a false positive, wherever it lies
(the shelves and stairs behind the table give most of them). "ROI" is the
table strip (e.g. rows 255-480), the rest of the frame left out.

| backend, slicing | floor | small parts | 3 cubes at ~1 m | boxes | false pos. (7 frames) | calls / frame | s / frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grounding-dino, whole frame | 0.2 | 4 / 12 | 0 / 3 | 3 / 4 | 23 | 1 | 0.14 |
| grounding-dino, whole frame | 0.3 | 1 / 12 | 0 / 3 | 1 / 4 | 6 | 1 | 0.14 |
| grounding-dino, 320 px | 0.2 | 6 / 12 | **3 / 3** | 4 / 4 | 60 | 7 | 0.79 |
| grounding-dino, 320 px | 0.3 | 4 / 12 | 0 / 3 | 2 / 4 | 16 | 7 | 0.79 |
| grounding-dino, 320 px, `max_box_fraction=1` | 0.2 | 6 / 12 | 3 / 3 | **1 / 4** | 64 | 7 | 0.81 |
| grounding-dino, 200 px | 0.2 | 6 / 12 | 3 / 3 | 4 / 4 | 90 | 13 | 1.43 |
| grounding-dino, ROI, unsliced | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 | 19 | 1 | 0.27 * |
| grounding-dino, ROI, 320 px | 0.2 | 7 / 12 | 3 / 3 | 2 / 4 | 30 | 4.4 | 1.34 * |
| grounding-dino, ROI, 320 px | 0.3 | 4 / 12 | 0 / 3 | 1 / 4 | 8 | 4.4 | 1.34 * |
| sam3, whole frame | 0 | 2 / 12 | 0 / 3 | 0 / 4 | 934 | 2.6 | 1.16 |
| sam3, whole frame | 0.5 | 4 / 12 | 0 / 3 | 4 / 4 | 17 | 2.6 | 1.16 |
| sam3, 320 px | 0.5 | 5 / 12 | 0 / 3 | 4 / 4 | 19 | 18 | 3.81 |
| sam3, 200 px | 0.5 | **8 / 12** | **3 / 3** | 4 / 4 | 24 | 33 | 5.59 |
| sam3, ROI, 320 px | 0.5 | 5 / 12 | 0 / 3 | 4 / 4 | **6** | 11.6 | 2.55 * |
| sam3, ROI, 256 px | 0.3 | **9 / 12** | 3 / 3 | 4 / 4 | 22 | 11.6 | 2.46 * |
| locate-anything, whole frame | - | 3 / 12 | 1 / 3 | 4 / 4 | 10 | 1 | 0.83 |
| locate-anything, 320 px | - | 4 / 12 | 2 / 3 | 4 / 4 | 38 | 7 | 4.51 |
| locate-anything, 256 px | - | 7 / 12 | 1 / 3 | 4 / 4 | 40 | 10 | 11.96 |
| locate-anything, ROI, 320 px | - | 5 / 12 | 1 / 3 | 4 / 4 | 15 | 4.4 | 2.68 * |

Latency is per frame, all backend calls over the RPC plus the merge (under
10 ms), from the first run; rows marked * come from a second run while other
jobs held the GPU at 92-99 %, where the same whole-frame grounding-dino call
took 0.32 s instead of 0.14 s. Per call: grounding-dino 0.11 s on a slice,
0.14 s on the frame; sam3 0.12-0.16 s on a slice, 0.44 s on the frame (it
returns every instance, with its mask, over the RPC); locate-anything
0.45-0.6 s, some slices several seconds (long answers). Model load on the
first call: grounding-dino 6.6 s, sam3 19.4 s, locate-anything 8.2 s. GPU
memory of the backend server: grounding-dino 2.7-4.0 GB, sam3 5.3 GB,
locate-anything 9.8 GB; this bundle uses none.

- **The small cubes are what slicing buys.** The three 16-22 px cubes at ~1 m
  are found by no backend on the whole frame except one by locate-anything;
  with slices grounding-dino finds 3 / 3 at 320 px and sam3 3 / 3 at 200 px
  (and at 256 px on the table ROI with floor 0.3).
- **Grounding-dino's small-part scores are low** (0.2-0.3): at 0.3 the cubes
  are gone sliced or not. At 0.2 slicing triples its false positives (23 -> 60;
  shelves, stairs, the arms' brackets read as "box" or "cube"); the table ROI
  halves them (30) with one more small hit, though 2 of the 4 boxes drop out.
  Cropping the table strip alone, unsliced,
  found as much (a 640 x 225 strip is upsampled x2.1 instead of x1.7): on
  these frames that is the cheapest grounding-dino setting.
- **sam3 must be floored before the merge** (the 0.5 default): unfloored, its
  low-score masks merge into large boxes around the parts. Sliced at 200-256 px
  it is the best detector here (8-9 / 12 small, all three cubes), at ~2.5 s a
  frame on the table ROI.
- **`max_box_fraction` keeps the larger parts**: without it the whole-slice
  "box" boxes merge over the real boxes and 3 of the 4 are lost.
- **Black rings on the black board were found by nothing** (0 / 4 everywhere);
  slicing does not help contrast. The plug (14-17 x 8-12 px) is found in 1-3
  of its 3 frames by every grounding-dino and sam3 setting, sliced or not.
- No LEGO bricks, dominoes or jigsaw pieces were on the station when these
  frames were taken; the cubes, the pen and the plug stand in for them (the
  same 10-20 px band). Recheck `slice_px` and the floor on real bricks.

Use: grounding-dino with `roi` = the table and `slice_px=320` (or no slicing
on a short table strip), floor 0.2, then a colour / size check; sam3 with
`slice_px=256`, the table `roi` and `min_score` 0.3-0.5 when the extra ~1-2 s
is affordable or masks are wanted.

## Gotchas

- Latency scales with the number of calls: 7 backend calls per 640 x 480 frame
  at 320 px (6 slices + the whole frame), 13 at 256 px; sam3 multiplies by the
  number of prompts. Use `roi` to slice only the table.
- Open-vocabulary detectors answer every slice: a slice without the part
  often yields a look-alike or the whole slice. `max_box_fraction` removes the
  latter; for the former raise `min_score` / `box_threshold`, bound the size
  with `max_box_px`, and check colour or size downstream.
- Parts touching each other merge into one box when their boxes overlap by
  IOS >= 0.5; use `postprocess="NMS"` to keep the best box of each instead.
- Boxes only, except sam3: for masks from grounding-dino boxes call
  `sam3.segment_box` on the boxes you keep.
