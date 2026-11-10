# Which perception tool for which part

This guide covers finding and segmenting the parts the YAM tasks handle:
LEGO bricks and plates, DUPLO bricks, dominoes and jigsaw pieces. It
compares the hub's detectors (`owlv2`, `grounding-dino`, `mm-grounding-dino`,
`omdet-turbo`, through `sahi` for slicing) and `sam2` for masks on one
labelled set of these parts. All numbers were measured on 2026-11-09; the
set and protocol are described below.

## Recommendations

| part, view | find it with | measured | then |
| --- | --- | --- | --- |
| LEGO brick / plate, overhead (~1 m, 5-25 px) | `mm-grounding-dino` (large) under `sahi.detect_sliced(slice_px=320)`, `min_score=0.2`, prompt `lego brick` | 58 % found (IoU >= 0.5), 5.7 FP / frame | Confirm each candidate from the wrist (`refining-target-from-wrist-closeup`); `colour.classify` for the colour. |
| LEGO, wrist (0.18-0.35 m) | `grounding-dino`, `box_threshold=0.25` | 95 %, 1.8 FP / frame | `sam2.tracker_init(box=...)` for the mask (IoU 0.96). |
| DUPLO, overhead | `mm-grounding-dino` (large) + sahi 320, 0.25 | 79 %, 3.7 FP / frame | as above |
| DUPLO, wrist | `grounding-dino` 0.25 (or mm-grounding-dino base 0.2) | 100 %, 1.8 FP / frame | |
| domino, close view, one domino already found | `owlv2.detect_like(query_box=that_domino, threshold=0.85)` | 92 % of the others, **0 FP** | Its box goes to `sam2` (IoU 0.96). |
| domino, close view, none found yet | `mm-grounding-dino` (large) 0.25, prompt `domino` | 64 %, 0.2 FP / frame | Then exemplar mode for the rest. |
| domino, overhead | `grounding-dino` or `mm-grounding-dino` (base) + sahi 320, 0.2 | 59-62 %, ~15 FP / frame | Too many FPs to act on: confirm from the wrist. |
| domino, wrist | `grounding-dino` 0.2 | 77 %, 2.8 FP / frame | |
| jigsaw piece (photo of pieces on a mat) | Box one piece (a pointing VLM + `sam2`, or a hand-off), then `owlv2.detect_like(query_box=..., threshold=0.85)` | **100 % of 317, 0 FP** | Text prompts find 17-36 % at best. |
| any part, mask from a box | `sam2.tracker_init(image, box=...)` (or `sam2.track` on a clip) | mean mask IoU 0.82-0.99 | Thin 1 x N LEGO overhead is the worst case (0.74-0.79); take grasp masks from the wrist. |

Do not use:
- **`omdet-turbo`** for any of these parts: 0-11 % found in every view. It is
  for table-scale objects (open boxes, trays).
- **`owlv2` text prompts** for bricks: 4 % of LEGO and 16 % of DUPLO overhead
  at 0.2. Its exemplar mode is the useful one.

## Numbers

Found = share of labelled parts with a detection of the right label at IoU
>= 0.5; FP = other detections per frame (wrong label, duplicates, boxes on
the station's clutter). Text prompts are `lego brick` and `domino` on every
render (DUPLO counts as `lego brick`), and `jigsaw puzzle piece` on the
photos. Each detector ran through `sahi.detect_sliced`: `full` is one call
on the frame, `s320` is 320 px slices plus the whole frame. Floors are each
detector's own score floors.

**Overhead renders** (60 frames; 209 LEGO, 61 DUPLO, 133 dominoes)

| detector, config | floor | LEGO | DUPLO | domino | FP / frame | s / frame |
| --- | --- | --- | --- | --- | --- | --- |
| mm-grounding-dino large, full | 0.2 | 37 % | 52 % | 30 % | 3.0 | 0.34 |
| **mm-grounding-dino large, s320** | 0.2 | **58 %** | **82 %** | 50 % | 5.7 | 2.1 |
| mm-grounding-dino large, s320 | 0.25 | 44 % | 79 % | 45 % | 3.7 | 2.1 |
| mm-grounding-dino base, full | 0.2 | 59 % | 69 % | 51 % | 8.8 | 0.09 |
| mm-grounding-dino base, s320 | 0.2 | 53 % | 64 % | **62 %** | 15.4 | 0.53 |
| grounding-dino-base, full | 0.2 | 32 % | 67 % | 46 % | 5.8 | 0.12 |
| grounding-dino-base, s320 | 0.2 | 43 % | 69 % | 59 % | 14.7 | 0.76 |
| owlv2 base, full | 0.2 | 4 % | 16 % | 33 % | 2.9 | 0.05 |
| owlv2 base, s320 | 0.15 | 26 % | 51 % | 41 % | 6.0 | 0.38 |
| omdet-turbo, full | 0.15 | 2 % | 11 % | 5 % | 7.0 | 0.04 |

By size (LEGO, mm-grounding-dino large s320, 0.2): parts under 12 px
across (1 x 1, 1 x 2, plates at ~1 m; 140 of 209) 55 %, 12-24 px 64 %.
Without slicing it is 31 % and 49 %.

**Wrist renders** (36 frames; 39 LEGO, 14 DUPLO, 26 dominoes)

| detector | floor | LEGO | DUPLO | domino | FP / frame |
| --- | --- | --- | --- | --- | --- |
| **grounding-dino-base** | 0.25 | **95 %** | **100 %** | 73 % | 1.8 |
| grounding-dino-base | 0.2 | 95 % | 100 % | **77 %** | 2.8 |
| mm-grounding-dino base | 0.2 | 74 % | 100 % | 81 % (IoU >= 0.3: 85 %) | 1.8 |
| mm-grounding-dino large | 0.2 | 62 % | 86 % | 65 % | 0.9 |
| owlv2 base | 0.15 | 5 % | 7 % | 35 % | 0.9 |
| omdet-turbo | 0.15 | 5 % | 0 % | 0 % | 8.1 |

**Close domino renders** (24 frames of the odd-one-out porter's
RoboSimStudio scenes; 240 dominoes in lines and grids, white / ivory /
black)

| detector | floor | found | FP / frame |
| --- | --- | --- | --- |
| **owlv2 exemplar** (one domino boxed) | 0.85 | **92 %** of the other 216 | **0** |
| owlv2 exemplar | 0.80 | 98 % | 0.1 |
| mm-grounding-dino large | 0.25 | 64 % | 0.2 |
| grounding-dino-base | 0.2 | 54 % | 9.2 |
| owlv2 text | 0.15 | 37 % | 0 |
| omdet-turbo | 0.15 | 4 % | 11.6 |

**Jigsaw photos** (40 PuzzleMap photos with 3-35 pieces each, 357 pieces,
scaled to 640 px wide: pieces 36-100 px)

| detector | floor | found | FP / frame |
| --- | --- | --- | --- |
| **owlv2 exemplar** (one piece boxed) | 0.85 | **100 %** of the other 317 | **0** |
| mm-grounding-dino base | 0.2 | 36 % | 0.9 |
| mm-grounding-dino large | 0.25 | 31 % | 0.8 |
| owlv2 text | 0.15 | 19 % | 0.4 |
| grounding-dino-base | 0.2 | 17 % | 1.2 |
| omdet-turbo | 0.15 | 7 % | 1.4 |

**Exemplar mode across classes (overhead).** With one part as the exemplar
at 0.85, owlv2 finds 57 % of the LEGO, 95 % of the DUPLO and 55 % of the
dominoes in the same frame. It also matches parts of the other classes
(72-83 hits over 39-60 searches), because a brick exemplar matches dominoes
and the reverse. Overhead, it finds "parts like this" but does not tell
them apart; the close and photo views have no such confusion.

**sam2 box -> mask** (SAM 2.1 base-plus, `tracker_init` with the labelled
box, IoU with the rendered mask)

| view | LEGO | DUPLO | domino |
| --- | --- | --- | --- |
| overhead | 0.82 (2 x N 0.85-0.87, 1 x N 0.74-0.79) | 0.94 | 0.89 (upright 0.84) |
| wrist | 0.96 | 0.99 | 0.98 |
| close domino renders | | | 0.96 |

No mask fell under IoU 0.5. Each call took 19 ms on an idle RTX PRO 6000.

## The eval set

- **Overhead (60) and wrist (36) renders.** LEGO bricks (1 x 1 to 2 x 4) and
  plates, DUPLO bricks (2 x 2 to 2 x 4) and dominoes (48 x 24 x 7.5 mm;
  flat with pips, face down, or upright; white, ivory, black), in 12 LDraw
  colours. They are rendered in MuJoCo through the station's calibrated
  cameras: the overhead Orbbec with its ChArUco pose and intrinsics, parts
  on the measured table plane; the left wrist D405 at 0.18-0.35 m. They
  are composited with a soft contact shadow onto **real station frames**:
  the 7 overhead frames of the sahi / owlv2 checks, with their arms, box,
  board and cable as clutter, and 7 real wrist frames with the cable and
  fingers. Instance masks come from MuJoCo's segmentation; boxes are the
  visible extent.
- **Close domino renders (24).** The odd-one-out porter's RoboSimStudio
  scenes, with their instance masks.
- **Jigsaw photos (40).** PuzzleMap (`pablo-moreira/puzzle-map`, CC-BY-4.0),
  multi-piece photos from `pieces/annotations.json`, listed with their
  boxes in `tests/bundles/data/target_parts_puzzlemap_subset.json`.

Limits:
- The bricks and dominoes are rendered, not photographed: plain plastic
  shading, no specular highlights, no motion blur.
- The jigsaw photos come from another camera, over a dark mat, close to
  top-down.
- No real LEGO, domino or jigsaw frame from the station exists yet; rerun
  this when one does.
- SAM3 (`sam3.*`) was not compared: its weights are gated on Hugging Face
  and no GPU host has a token.

## Reproduce

1. Build the eval set in a folder `WORK`:
   - `_target_parts_gen_scenes.py` (seed 7, 60 overhead, 40 wrist) makes
     the renders;
   - the 24 close domino scenes are every third of the odd-one-out porter's
     72;
   - the jigsaw photos are fetched by name from the subset list.
2. Run every backend on a GPU host with `tests/bundles/_target_parts_eval.py`
   (BACKEND: `owlv2`, `owlv2-like`, `grounding-dino`, `mm-grounding-dino`,
   `omdet-turbo`, `sam2`). Each backend runs in its own environment; for
   mm-grounding-dino that is transformers 4.57.1. Measured here: transformers
   5.19.0 / torch 2.10 on rshgpu5 and rshgpu6.
3. Score with `tests/bundles/_target_parts_score.py`.
