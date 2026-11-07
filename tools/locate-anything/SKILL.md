---
name: locate-anything
description: NVIDIA LocateAnything-3B (non-commercial licence) run locally on a GPU -- category detection,
  referring-phrase grounding and pointing, returned in gemini-er's shapes
  (pixel boxes with label/score; pixel points). Use when a research or
  evaluation workflow wants a local, API-free alternative to gemini-er.detect /
  gemini-er.point on camera frames. NON-COMMERCIAL ONLY - the weights and the
  worker it runs are under the NVIDIA License (research or evaluation use); it
  refuses to run until GAP_LOCATE_ANYTHING_NONCOMMERCIAL=1 is set.
license: Apache-2.0 for this bundle's code; the LocateAnything-3B weights and the
  NVlabs worker it downloads at run time are under the NVIDIA License,
  non-commercial (research or evaluation) use only
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, grounding, pointing, gpu, non-commercial]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: hf://nvidia/LocateAnything-3B@c32291ca5e996f5a7a485845b4f57a233936bba0
      sha256: fdccd9f086608b8f25aeab82b59f4d30c4946220dc9e79caeba6f2c90c67714d
      license: NVIDIA License (non-commercial, research or evaluation only)
    provenance:
      authors: [NVIDIA]
      code: https://github.com/NVlabs/Eagle/blob/06edcff802dbdd167d196c7a1c5b44929e07d951/Embodied/locateanything_worker.py
      upstream_license: NVIDIA License (non-commercial, research or evaluation only)
    notes: Non-commercial research or evaluation use only (NVIDIA License); never a default for commercial deployments.
gap:
  requires: {gpu: true, weights: true, env: [GAP_LOCATE_ANYTHING_NONCOMMERCIAL]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "locate-anything"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - locate-anything.detect: Boxes for every instance of one or more categories (pixel box, label, score).
    - locate-anything.ground: Boxes for what a referring phrase names; drop-in for gemini-er.detect.
    - locate-anything.point: Pixel points for what a phrase names; drop-in for gemini-er.point.
---

# locate-anything

NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
(Qwen2.5-3B + MoonViT with parallel box decoding) as three in-process tools.
Image in: RGB uint8 `[H, W, 3]`. Out: the shapes `gemini-er` returns, so a
skill switches backend by tool name:

| this bundle | gemini-er | returns |
| --- | --- | --- |
| `locate-anything.ground(image, query, single=False)` | `gemini-er.detect(image, query)` | `{detections: [{box, label, score}]}` |
| `locate-anything.point(image, query, max_points=10)` | `gemini-er.point(image, query, max_points)` | `{pixel_x, pixel_y, found, points}` |
| `locate-anything.detect(image, categories)` | - | `{detections: [...]}`, `label` = the category |

Boxes and points are in pixels. The model reports no confidence: `score` is
1.0 (as gemini-er's when the model gives none). Nothing found is an empty
list / `found=False`, never an error.

## Licence: non-commercial only

The weights are under the [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE):
use, reproduction and modification for non-commercial research or evaluation
only (NVIDIA and its affiliates excepted). Its language model is built from
Qwen2.5-3B-Instruct (Qwen Research License), its vision encoder from
MoonViT-SO-400M (MIT). The worker the tool runs,
`Embodied/locateanything_worker.py` from NVlabs/Eagle, carries an NVIDIA
copyright header and sits under the same non-commercial `LICENSE_MODEL`.

This repository contains neither: the bundle downloads the weights from the
Hub and the worker from GitHub, both at pinned commits, the first time a tool
runs. To keep it from being picked up silently, the bundle lists
`GAP_LOCATE_ANYTHING_NONCOMMERCIAL` in `gap.requires.env` (`gap check` reports
it not ready until it is set) and every tool raises `ToolError` unless it is
`1`. Setting it is the user's statement that the use is non-commercial. For
commercial work use `gemini-er` or `grounding-dino`.

## Install

```bash
gap skills install locate-anything        # uv sync --project tools/locate-anything
export GAP_LOCATE_ANYTHING_NONCOMMERCIAL=1 # research / evaluation use only
gap skills check --download               # optional: prefetch weights + worker
```

The environment is torch 2.8 / transformers 4.57.1 plus four packages the
checkpoint's remote code imports without declaring (opencv-python-headless,
decord, lmdb, peft 0.12.0). Weights: 7.7 GB on disk, 7.8 GB of GPU memory in
bf16.

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_LOCATE_ANYTHING_NONCOMMERCIAL` | `1` = the use is non-commercial research or evaluation | unset (tools refuse) |
| `GAP_LOCATE_ANYTHING_DEVICE` | torch device | `cuda` |
| `GAP_LOCATE_ANYTHING_MODEL` | a local model directory instead of the pinned snapshot | pinned Hub commit |
| `GAP_LOCATE_ANYTHING_WORKER` | a local `locateanything_worker.py` (digest not checked) | pinned Eagle commit, sha256-checked |
| `GAP_LOCATE_ANYTHING_RUNTIME` | `batch` (the release's `batch_utils` scheduler) or `standard` (transformers generate) | `batch` |
| `GAP_LOCATE_ANYTHING_ATTN` | batch-runtime attention, `sdpa` or `la_flash` (needs flash-attn) | `sdpa` |

## Runtime, and VFlash later

The tool does not reimplement inference: it imports NVlabs' reference worker
(`LocateAnythingWorker`) at Eagle commit `06edcff8` and calls its `detect`,
`ground_multi` / `ground_single` and `point`. Decoding is greedy
(`temperature=0`; the worker's default is 0.7 with sampling) so two calls on the
same frame agree. By default the worker runs the release's `batch_utils` hybrid
scheduler (`use_batch_runtime=True`, sdpa attention), which answered 4 more of
the 23 live queries right than the worker's standard transformers path;
`GAP_LOCATE_ANYTHING_RUNTIME=standard` selects the latter. `la_flash`
attention needs `uv sync --extra la-flash` (the upstream flash-attn 2.8.3
wheel for torch 2.8 / CPython 3.12).

VFlash (Flash Primitive Decoding for visual plans: trajectories, motion, boxes,
points) lives in the same Eagle Embodied tree and targets the same stack
(torch 2.8, transformers 4.57.1, flash-attn), so this environment is meant to
host it too. Its target and drafter checkpoints were not released on
2026-11-07; there is no VFlash tool yet.

## Measured (2026-11-07, Brev L40S, real YAM station frames)

Four 640 x 480 frames from the YAM station camera: the two of the earlier
evaluation (`cube_box` = g_sota_frames_20261103-0511/frame_02, `rings_cable` =
-1421/frame_02) and two more sessions (-0630/frame_06: cube, open box, paper;
-1147/frame_06: pegboard, rings, white cable loop). 15 grounding queries
(`ground` vs `gemini-er.detect`), 8 pointing queries (`point` vs
`gemini-er.point`) and 14 category x frame items of one multi-category `detect`
call per frame (vs `gemini-er.detect` with the categories joined by commas).
Ground truth was read by eye off zoomed crops; a box counts when IoU >= 0.5 or
the two centres lie in each other's box, a point when it lies in the region;
1 = right, 0.5 = some of several targets found, 0 = wrong or nothing. Gemini
model `gemini-robotics-er-2-preview`, the server started the way gap starts it
(`uv run --project tools/locate-anything -- python -m gap_core.rpc.server`).

| | gemini-er | locate-anything (batch, default) | locate-anything (standard) |
| --- | --- | --- | --- |
| grounding, 15 | 10 | **12** | 9 |
| pointing, 8 | **7** | 6 | 5 |
| **total, 23** | 17 | **18** | 14 |
| multi-category `detect`, 14 | **12** | 7 | 7.5 |

- LocateAnything boxed the pegboard (both frames) and the white cable on -1147
  where Gemini returned nothing; Gemini found the cube on -0630 only by point,
  not by box. Neither found the rings reliably: Gemini boxed one ring on one
  frame and pointed at 2 of 4 rings on both, LocateAnything found none (its
  "centre of each black ring" points land on the gripper bases).
- The batch runtime returned nothing for "open grey box" on -0630, the
  standard runtime on both cube frames; the standard runtime also boxed the
  open box for "small dark grey cube" on both.
- One `detect` call over 3-4 categories is the weak spot (it drops categories
  and labelled the open box "cube" on -0630): ask one `ground` per object.
- Greedy decoding was deterministic: identical results on both passes of all 54
  calls and across server restarts, for both runtimes.

Latency per warm call (median / p90), on an L40S shared with other jobs (GPU at
100 % utilization, CPU load ~18 on 12 cores during most runs):

| runtime | median | p90 | note |
| --- | --- | --- | --- |
| batch, sdpa (default) | 1.8 s | 3.5 s | lighter-loaded slot |
| batch, sdpa (default) | 10.9 / 15.2 s | 19.7 / 29.7 s | two runs at CPU load ~18: its scheduler is CPU-bound |
| standard | 2.0 / 3.9 / 2.7 s | 4.0 / 10.3 / 6.9 s | three runs |
| batch, la_flash | 32.6 s | 71 s | flash-attn wheel installed in 5 s; same answers as sdpa |
| gemini-er (API) | 2.3 s | 4.5 s | |

Model load plus the first call: 12-23 s from a warm disk cache (up to 80 s under
load). GPU memory 7.8-8.7 GB. On a busy machine `GAP_LOCATE_ANYTHING_RUNTIME=standard`
keeps latency near 3 s at the cost of the accuracy above. A server started
without `GAP_LOCATE_ANYTHING_NONCOMMERCIAL` refused every call.

## Gotchas

- Not-found is `<box>None</box>` in the raw answer, an empty result here.
- The model's coordinates are x, y order normalized to 0-1000 (Gemini's are
  y, x); converted and clamped to the frame here.
- `point` asks the model with its own `Point to: <phrase>.` prompt, which has
  no count; `max_points` only truncates.
- The first call loads the model (12-23 s from a warm disk cache, plus the
  7.7 GB download the very first time); the worker stays resident in the server.
- The batch runtime sometimes garbles a `<ref>` ("gripperper"); `detect` snaps
  a ref to the closest requested category (difflib ratio >= 0.6).
- Neither runtime takes visual prompts here (the worker supports them only on
  the standard path; no tool exposes them yet).
