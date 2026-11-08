---
name: depth
description: Robbyant LingBot-Depth v0.5 run locally on a GPU -- fills the holes
  of a sensor depth image from its RGB frame (the overhead Orbbec's or the wrist
  D405's) and, on request, denoises it, returning metric depth of the same size,
  a validity mask and a mask of the filled pixels. Use before geometry, grasp or
  pose tools when raw depth has holes or speckle on dark, shiny, thin or small
  parts (a black board, a cube, a ring, a flat cable).
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, depth, depth-completion, rgbd, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: hf://robbyant/lingbot-depth-pretrain-vitl-14-v0.5@79204ed6b837f4fdd192cf563e59481fecfa0295
      sha256: 0dfe0d3f1fe2a43586fac5bca36022c6cf98e7cb0078e3b68ad35f25ba863f1c
      license: Apache-2.0
    provenance:
      authors: [Robbyant]
      code: https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/model/v2.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2601.17895
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "depth"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - depth.refine: Fill (default) or denoise an RGB-D frame's depth; same-size metric depth, validity and filled masks.
---

# depth

Robbyant's [LingBot-Depth](https://github.com/Robbyant/lingbot-depth) v0.5
(masked depth modeling, ViT-L/14, 1.28 GB; code and weights Apache-2.0) as one
tool. The model takes an RGB image and the raw depth of the same camera and
predicts dense metric depth. In: RGB uint8 `[H, W, 3]`, depth float32
`[H, W]` metres registered to it (0 / NaN / inf = hole), optionally `K`. Out:
depth of the same size in the same convention (metres, 0 = none).

| call | gives |
| --- | --- |
| `depth.refine(rgb, depth_m, K)` | **fill** (default): the sensor's depth where it measured, the model's (fitted to the sensor) in the holes |
| `depth.refine(..., keep_measured=False)` | **refine**: the model's fitted depth everywhere (denoised; see the trade-offs below) |
| `depth.refine(..., align=False)` | the model's depth as it comes (4-14 mm off the sensors on the station) |
| `depth.refine(..., return_points=True)` | also `points_m` `[H, W, 3]`, camera frame, NaN where not valid |

```python
cam = ctx.tool("robot.get_observation")["cameras"][0]
d = ctx.tool("depth.refine", rgb=cam["rgb"], depth_m=cam["depth"], K=cam["intrinsics"])
pts = ctx.tool("geometry.mask_to_world_points", mask=mask, depth=d["depth_m"], ...)
model_only = d["filled"] > 0  # pixels no sensor saw: weigh them less
```

## Output

`{depth_m, valid, filled, raw_valid_fraction, valid_fraction, filled_fraction,
aligned, align_scale, align_shift_mm, model_vs_measured_mm, points_m, backend,
model, timing}`.

- `valid` / `filled`: uint8 `[H, W]` 0 / 255 masks (`sam3`'s convention), so
  they feed `geometry.*` directly. `valid` = measured, or the model's mask head
  > 0.5 with a finite depth; `filled` = valid and a hole in the input. The mask
  head called every pixel valid on all four station frames below, so
  `filled` (where the depth is the model's alone) is the informative one.
- `align_scale`, `align_shift_mm`: the model's depth is mapped by
  `scale * depth + shift`, fitted to the pixels that both have (seeded at
  scale 1 and the median offset, then 3 rounds of least squares dropping
  residuals beyond 3 MAD; skipped with fewer than 3000 measured pixels).
- `model_vs_measured_mm`: median |fitted model - sensor| over measured pixels,
  a per-frame sign of how far to trust the fill (2.5-6.7 mm on the station).

## How it runs

- The pinned `mdm` package (lingbot-depth commit `f3a237e434ae987bc38281476d6cfb5df3e4d739`)
  provides `MDMModel`; `MDMModel.infer` at upstream's defaults (bf16 autocast,
  `resolution_level` 9 = 3600 image tokens; 0 = 1200, about 2x faster).
- The checkpoint comes from the Hub at the pinned commit (model.pt sha256
  `b60cf27d...02b3c23`; the snapshot's manifest digest is the `sha256` above).
- xFormers is not installed (its wheels pin a torch). Upstream drops the depth
  tokens of all-hole patches and hands the blocks a list of per-frame
  sequences, which it runs only through xFormers; the bundle runs a one-frame
  list through the block's tensor path (PyTorch SDPA), which is the same
  attention for one sequence. A batch of several frames would still need
  xFormers; `depth.refine` sends one.
- The model loads on the first call (module singleton); importing the module
  imports neither torch nor mdm.

## Install

```bash
gap skills install depth              # uv sync --project tools/depth
gap skills check --download           # optional: prefetch the 1.28 GB checkpoint
```

torch 2.9.1 / torchvision 0.24.1 (CUDA 12.8 wheels) and huggingface_hub. mdm
pins torch 2.6 and xFormers and lists demo dependencies (OpenCV, matplotlib,
trimesh, SciPy); `[tool.uv] override-dependencies` replaces the torch pin and
keeps the rest out, since the model path imports none of them.

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_DEPTH_DEVICE` | torch device | `cuda` |
| `GAP_DEPTH_MODEL` | a local `model.pt` instead of the pinned snapshot (digest not checked) | - |

## Measured (2026-11-07, Brev L40S, real YAM station frames)

The server was started the way gap starts it (`uv run --project tools/depth --
python -m gap_core.rpc.server --bundle depth`) and called over the RPC with four
station RGB-D frames (640 x 480, depth registered to colour by the camera
servers):

| frame | camera, range | raw depth | scene |
| --- | --- | --- | --- |
| cable 2026-11-03 14:21 | overhead Orbbec Gemini 335, table at ~1.0 m | 88.7 % | wood table, flat white cable, black perforated board with two upright rings and the plug post |
| cube 2026-10-31 | overhead Orbbec, table at ~0.57 m | 97.5 % | ~30 mm black cube and an open black box on white cloth |
| wrist cable 2026-11-04 | left wrist D405, table at ~0.15 m | 71.9 % | flat cable on wood |
| wrist ring 2026-11-03 | left wrist D405, board at ~0.28 m | 77.7 % | black perforated board, a ring on a ~100 mm post |

Every mode filled 100 % of the holes. Planes were fitted (RANSAC, then least
squares) to each depth on its own over a table / board region; heights are
above that plane. `sigma` is 1.4826 x MAD of the plane residual on pixels the
sensor measured; `bias` is the median (output - sensor) there; hole fill is
the RMS residual of filled pixels on the plane. PromptDA (ViT-L,
Apache-2.0) got the same frames with the holes filled by nearest neighbour
for its 192 x 256 prompt, as it needs.

| frame: measure | raw | **fill** (default) | refine | model unaligned | PromptDA |
| --- | --- | --- | --- | --- | --- |
| cable: table sigma / bias, mm | 3.65 / 0 | 3.64 / 0 | 1.94 / 0.0 | 1.92 / -7.6 | 0.98 / +8.7 |
| cable: flat cable height, median / p90 mm | 1.4 / 8.3 | 1.5 / 8.5 | 0.6 / 3.6 | 0.6 / 3.6 | 1.4 / 5.8 |
| cable: ring / plug post top (p90), mm | 94 / 104 | 94 / 101 | **20 / 17** | 20 / 17 | 30 / 22 |
| cube: cloth sigma / bias, mm | 1.34 / 0 | 1.34 / 0 | 1.20 / -1.8 | 1.20 / -5.9 | 0.40 / +5.4 |
| cube: cube top median / box rim p90, mm | 27.8 / 31.3 | 27.8 / 31.3 | 27.3 / 32.2 | 27.2 / 32.0 | 28.9 / 30.4 |
| wrist cable: table sigma / within 2 mm / bias | 0.49 / 100 % / 0 | 0.49 / 97 % / 0 | 2.49 / **58 %** / -1.9 | 2.48 / 58 % / **-13.7** | 0.75 / 95 % / +3.9 |
| wrist cable: cable height median, mm | 2.5 | 2.3 | 4.0 | 4.3 | 4.5 |
| wrist cable: hole fill RMS, mm | - | 6.6 | 2.6 | 2.7 | 3.6 |
| wrist ring: board sigma / within 5 mm | 4.32 / 71 % | 4.33 / 69 % | **2.0 / 98 %** | 1.89 / 98 % | 3.15 / 80 % |
| wrist ring: board hole fill RMS, mm | - | 6.0 | 4.7 | 4.9 | **23.4** |
| wrist ring: ring rim median, mm | 106 | 101 | 85 | 91 | 116 |
| wrist ring: wood table below board, mm | -10.1 | -8.1 | -12.9 | -13.4 | -4.7 |

- **Fill (default)** leaves every measured pixel as the sensor gave it (checked:
  bit-identical) and puts the fitted model only in the holes: no bias, upright
  thin parts keep their height, holes filled to 2.5-6.6 mm of the plane.
- **Refine** halves the overhead table noise and cleans the black board
  (98 % of it within 5 mm against 71 % raw), but from ~1 m it **flattens thin
  upright parts** (rings and post 94-104 mm tall come out at 17-20 mm; a flat
  cable at 0.6 mm), and at the wrist's 0.15 m it **warps a well-measured table**
  (sigma 0.49 -> 2.5 mm, a patch-grid texture). Use it for a dark, speckled,
  roughly flat surface, not for small raised features.
- **Unaligned**, the model sits 6-14 mm nearer than both sensors on their
  tables (worst at the D405's 0.15 m); the fit takes that out (refine: within
  2 mm overhead and on the wrist table; 4.3 mm on the wrist board).
- PromptDA is smoother overhead but biased +4-9 mm, also loses the rings, and
  fills the black board's holes badly (23 mm RMS).
- `resolution_level` 0 instead of 9: fill holes 5.2 / 3.6 mm (wrist cable /
  board) against 6.6 / 6.0, half the model time; overhead no different.

**Speed and memory.** The L40S was shared with other jobs at 60-96 %
utilisation, so timings are those of a busy GPU. Per 640 x 480 frame over the
RPC: level 9: 127-198 ms in the model, 200-290 ms per call; level 0: 64-97 ms /
137-177 ms. The first call loads the model in 9.5 s from a warm disk (the
1.28 GB download beforehand). GPU memory: 3.75 GB for the server process
(`nvidia-smi`). PromptDA for comparison: 119-131 ms, 1.8 GB.

## Gotchas

- `depth_m` must be registered to `rgb` (the station's camera servers align
  depth to colour). On the wrist D405 frame the raw depth of the ring 9 cm
  above the board sits 20-40 px off the ring in the colour image; refine
  redraws it where the RGB shows it, fill keeps the sensor's placement.
- Filled pixels are the model's guess, fitted to the rest of the frame: gate
  precise contact or grasp-depth decisions on `filled == 0` where you can.
- `align` fits one scale and shift per frame; a frame whose measured depth is
  mostly far background (overhead) fits the background best.
- One frame per call, 640 x 480 checked; other sizes run (the model resizes)
  but were not measured.
