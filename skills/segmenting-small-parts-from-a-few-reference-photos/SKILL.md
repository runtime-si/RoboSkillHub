---
name: segmenting-small-parts-from-a-few-reference-photos
description: Labels the parts in a frame - which brick, which domino side,
  which piece - from a handful of labelled reference photos, with no
  training. A memory bank of DINOv2 features pooled under the references'
  masks (tools/dinov2) is matched against the frame's part masks (given, or
  from sam3 with a generic prompt), each mask labelled by its best class
  (mean of its top-3 cosines to that class's references) when it clears a
  score and a margin gate; parts not in the bank stay unlabelled. After
  No-Time-to-Train's reference-based instance segmentation. Use when a task
  has its own parts (a kit's brick types, a tile's face) and a few photos
  of them, and a detector's text prompts or one exemplar cannot tell them
  apart.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, recognition, few-shot, segmentation, lego, domino, jigsaw, dinov2]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, lego, jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/miquel-espinosa/no-time-to-train/tree/e93a181837bd
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2507.02798
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
    - dinov2.embed_regions
  required_inputs: {}
  produces_outputs:
    n_proposals: int
    detail: string
  exit_conditions:
    instances_found: At least one proposal got a label (best class score >= min_score 0.45, margin to the second >= min_margin); instances lists them with box, score, margin and per-class scores, counts per label.
    none_found: No proposal cleared the gates, none was 20+ px, or there was no RGB frame.
    bank_empty: No usable reference (each needs label, mask or box, and an image) and no bank given.
  canonical_scripts:
    - label: scripts/label_parts.py
  examples:
    - title: Tell the brick types apart on the overhead camera from five photos each
      path: examples/canonical_subgraph.json
---

# segmenting-small-parts-from-a-few-reference-photos

Text prompts name kinds ("lego brick"), not a kit's parts (the 2x3 and not
the 2x4), and one exemplar (`owlv2.detect_like`) says "like this" without
saying "and not like that". A few labelled photos per part do both: the
bank holds every class, and each part goes to the class it is closest to.

## How it composes

- **Before it**: the references - a few photos per class with a mask or a
  box on the part (`references: [{label, mask | box, image}]`; `image` may
  be left out for references in `reference_frame`) - or a `bank` from an
  earlier run (its `bank` output), so the references are embedded once per
  task. The frame's proposals: `masks` or `boxes` from a detector or
  segmenter (sam2, sam3, sahi + owlv2, rfdetr, depth blobs), or `prompt`
  and the skill calls `sam3.segment_text`. The frame is `frame`, or
  `camera` in `robot.get_observation`.
- **After it**: `instances` (index into the proposals, `label`, `box`,
  `score`, `margin`) for the picker or the build tracker; `counts` per
  label for a kitting check.

## What it does

1. **Bank**: `dinov2.embed_regions` on each reference image's masks or
   boxes: unit vectors of the patch tokens pooled under the mask and the
   masked crop's CLS (see tools/dinov2).
2. **Proposals**: the masks (or boxes) of 20+ px, embedded the same way.
3. **Match**: per class the mean of a proposal's 3 highest cosines to that
   class's references; the best class is its label when the score is at
   least `min_score` and its lead over the second at least `min_margin`.

No-Time-to-Train also builds its bank from SAM2 masks and DINOv2 features,
but fuses SAM2's automatic proposals and scores them with a semantic-aware
soft merge; here the proposals come from whatever the graph already runs
and the match is top-k cosine (re-implemented, no upstream code).

## Inputs

`references` | `bank` (one), `masks` | `boxes` | `prompt` (one), `frame` or
`camera`, `reference_frame`, `min_score` (0.45), `min_margin` (0.0),
`top_k` (3), `keep_unknown` (false: unlabelled proposals are left out).

## Outputs

`instances` (per labelled proposal: `index`, `box`, `label`, `best_class`,
`score`, `margin`, `scores` per class), `counts`, `classes`, `bank`
(`labels`, `embeddings`), `n_proposals`, `detail`.

## Checks

CPU unit tests (`tests/bundles/test_segmenting_small_parts_from_a_few_reference_photos.py`,
8 with the tool's): cropping and mask pooling, the tool with DINOv2 faked by
patch colours, the classify rule and its gates, the skill end to end with
FakeContext (references, a reused bank, sam3 proposals, the three exits).

Live (`tests/bundles/_few_shot_parts_eval.py`): RESULTS

## Limits

- **Recognition, not detection**: a part the proposals miss is not found.
  At 1 m overhead LEGO parts are 10-25 px; proposals from SAHI tiles or the
  wrist camera, or `training-a-part-detector-from-auto-labels`, find them.
- Look-alike classes (a 2x3 against a 2x4 brick from above at 15 px) need
  references at the same scale and view as the queries.
- The bank is per model: `GAP_DINOV2_MODEL` must not change between
  building it and using it.
- The gates were set on renders, not tuned on real frames.
