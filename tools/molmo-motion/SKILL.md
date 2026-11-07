---
name: molmo-motion
description: AllenAI MolmoMotion-4B run locally on a GPU -- forecasts where chosen
  pixels (a cable tip, a cube corner) move over the next ~2 s under a language
  instruction, as pixels per step and, given depth and the camera, as camera and
  world points. Use when a propose or judge stage needs the motion a model
  expects ("the cable tip is pulled through the ring") before planning or
  checking it.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, planning, trajectory-forecasting, gpu]
  hub:
    stage: propose
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: hf://allenai/MolmoMotion-4B-H3-F30@3f5e790a511ff2cdf21c8d2a14cb4d8409c94629
      sha256: aa49c230ff0483e75cf358c266111691913d03565edb7857d6851d6a1da4a8ef
      license: Apache-2.0
    provenance:
      authors: [Allen Institute for AI]
      code: https://github.com/allenai/molmo-motion/tree/ab7c5f4e7c49359e4275e9a9b21a59659be17386
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2606.18558
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "molmo-motion"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - molmo-motion.forecast: Forecast query pixels' motion under an instruction (pixels per step; camera/world points with depth).
---

# molmo-motion

[MolmoMotion-4B](https://huggingface.co/allenai/MolmoMotion-4B-H3-F30) (Molmo2-4B
backbone, autoregressive) forecasts the 3D trajectories of query points from a
short RGB history, the points' 3D history and an action caption. This bundle
wraps it as one tool, `molmo-motion.forecast`, that takes hub-shaped inputs:

```python
out = ctx.tool(
    "molmo-motion.forecast",
    image=frame["rgb"],  # or [older, ..., current]
    query_points=[[tip_u, tip_v]],  # gemini-er.point / curve.endpoints pixel
    instruction="the cable tip is pulled through the ring",
    depth=frame["depth"],
    intrinsics=frame["intrinsics"],
    camera_pose=frame["pose"],
)
path = out["tracks"][0]["points_world"]  # [30][x, y, z] m, 1/15 s apart
```

## Output: the forecast schema

The result does not depend on the model; `backend` names it. A future
`vflash.forecast` (NVIDIA VFlash: parallel decoding of visual plans, whose
dense-motion experiment used MolmoMotion-4B as its target model; checkpoints not
released on 2026-11-07) should return exactly this structure, so a skill switches
forecaster by tool name.

| field | type | meaning |
| --- | --- | --- |
| `found` | bool | at least one step of one track came back |
| `tracks` | list | one per query, in query order |
| `tracks[i].query_pixel` | `[u, v]` | the query at t0, as given |
| `tracks[i].pixels` | `[F][u, v]` | forecast pixels at t0 + k * `dt_s`, k = 1..F, in the current frame's camera (may leave the image) |
| `tracks[i].start_camera` / `points_camera` | `[x, y, z]` / `[F][x, y, z]` | metres, camera frame at t0 (OpenCV axes); `[]` without depth |
| `tracks[i].start_world` / `points_world` | `[x, y, z]` / `[F][x, y, z]` | metres, world frame via `camera_pose`; `[]` without depth + pose |
| `tracks[i].predicted` | `[F]` bool | the model emitted this step; a missing step holds the last emitted value |
| `horizon` | int | F |
| `dt_s` | float | seconds between steps (1/15 for MolmoMotion) |
| `frame` | str | `world`, `camera` or `pixel`: the richest frame the tracks carry |
| `backend` / `model` | str | `molmo-motion` / `repo@revision` |
| `timing` | dict | `load_s` (first call only), `prepare_s`, `generate_s`, `total_s` |
| `raw` | str | the model's answer text (`<tracks coords="...">`) |

A forecaster that only predicts pixels fills `pixels` and leaves the 3D lists
empty with `frame: pixel`; one that predicts fewer steps sets `horizon` and
`dt_s` to its own.

## How the request becomes the model's input

- **8 points a call.** Both checkpoints take exactly 8 query points (their
  `num_points`), the first being the anchor the answer is relative to. Fewer
  queries are padded with points on a 4 px ring around them (training sampled 8
  points on one moving object, so the padding stays on it); only the queries'
  tracks come back. More than 8 queries run 8 at a time.
- **History.** The H3 checkpoint reads 3 frames 1/15 s apart. One frame is
  repeated three times and the points are held still (a static history);
  pass `query_history` (pixels per frame, e.g. from `sam3.tracker_update`) and
  per-frame `depth` when the points were already moving.
- **3D.** Each query is lifted with the median valid depth in a 7 x 7 window
  and K. Without depth the queries sit 1 m along their rays (K guessed as a
  focal of the image's longer side when not given) and only `pixels` are
  returned: the forecast is metric, so its 2D projection is then a rough
  guide.
- **Generation.** `predict_trajectory` decodes greedily with upstream's
  budget of 160 tokens a step; every coordinate digit is a token, so latency
  grows with `horizon`. Ask for fewer steps when ~1 s ahead is enough.

## Install

```bash
gap skills install molmo-motion          # uv sync --project tools/molmo-motion
gap skills check --download             # optional: prefetch the checkpoint
```

The environment is the `molmo_motion` package at commit `ab7c5f4` (Python 3.11,
torch 2.9.1, torchcodec 0.9, transformers 4.57.1). Only `config.yaml` and
`model.pt` (17.8 GB, fp32) are downloaded from the pinned Hub commit; the
safetensors shards in the same repo are an HF-format copy the package does not
read. The tokenizer (`Qwen/Qwen3-4B-Instruct-2507`) is fetched by the package
from the Hub at load.

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_MOLMO_MOTION_DEVICE` | torch device | `cuda` |
| `GAP_MOLMO_MOTION_REPO` | `allenai/MolmoMotion-4B-H3-F30` or `allenai/MolmoMotion-4B-H1-F32` (single keyframe, F=32), each pinned | H3-F30 |
| `GAP_MOLMO_MOTION_MODEL` | a local checkpoint dir (`config.yaml` + `model.pt`) | pinned Hub commit |

Loading differs from upstream's `MolmoMotion.from_pretrained`, which builds the
model in fp32 on the CPU and `torch.load`s the whole checkpoint next to it
(~36 GB of host RAM) before the README's cast to bf16: here the model is built
in bf16 on the device and the checkpoint is memory-mapped, giving the same bf16
weights (the model registers no buffers; RoPE tables are computed in fp32 on
use) with ~9 GB of GPU memory and little host RAM.

## Measured (2026-11-07, Brev L40S, real YAM station frames)

The GPU was shared with 10-12 simulator processes at 100 % utilization for
every run, so the latencies are those of a saturated card, not of the model.

| case | checkpoint | forecast | decode |
| --- | --- | --- | --- |
| cable free end (133, 337), "the cable tip is pulled through the ring", depth from the frame | H3-F30, frame x3 | all 30 steps = the start point (0 mm) | 796 s |
| same | H1-F32 | 1 mm in 32 steps | 1428 s |
| cube (184, 338), "the cube is pushed into the box", depth = cube-top plane through the calibrated camera (the frame has none) | H1-F32 | 3.5 mm of jitter, ends at the start | 1046 s |
| upstream's EgoDex example (hands placing a ball), 3 real history frames | H3-F30 | same lift-and-arc tracks as upstream's released prediction, by eye | (first call) |

- **On a still scene the model forecasts stillness.** Neither checkpoint moved
  the cable tip or the cube from one frame plus an instruction: the answer
  repeats the query points' 3D history (H3) or stays within a few mm (H1).
  MolmoMotion is trained on videos of motion in progress (EgoDex, HD-EPIC, ...,
  hands in view); it continues motion, it does not invent the motion an
  instruction describes when nothing in the frames is moving. Feed it a real
  history (`image` as 3 frames 1/15 s apart and `query_history` from a tracker)
  while the arm is already moving the object.
- **Loader check:** the bf16-on-device loader gives the same tracks as
  upstream's released H3 prediction for its bundled EgoDex clip (drawn side by
  side; the numeric difference was not saved).
- **Memory:** 9.1 GB after load (both checkpoints), 10.6-11.4 GB per
  `nvidia-smi` during a forecast, torch peak allocation 9.7 GB. Load 47-54 s
  from a warm disk cache (17.8 GB `model.pt`, downloaded in ~1.5 min).
- **Latency:** ~1-2 tokens/s on the saturated card; one 30-step forecast is
  ~2,400-2,600 characters of digits (~1,500+ tokens). Upstream quotes ~40 s for
  `predict_trajectory` on an idle A100. Not usable in a control loop either
  way; use it once per plan.
- One H1 attempt hit CUDA out-of-memory on its first forecast when another
  process took the free memory during the load; the tool raised `ToolError`.

## Gotchas

- The forecast is in the camera frame of the **last** frame given; world points
  need that frame's camera-to-world pose.
- A static (repeated) history tells the model nothing is moving, and on the
  station frames it then forecast no motion whatever the instruction (see
  Measured). A forecast whose `pixels` stay on `query_pixel` means "no motion
  expected", not "no answer".
- The instruction is a caption of the motion, not a command to a robot: the
  model forecasts what the scene does ("the cube is pushed into the box"), not
  what an arm should do.
- Model output is a forecast, not a plan: check it (reach, collisions with
  `curobo`) before driving an arm along it.
