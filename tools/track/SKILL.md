---
name: track
description: Google DeepMind TAPNext++ run locally on a GPU -- follows chosen
  pixels (a cable tip, a plug, a small part, a cube corner) through a video,
  returning each one's pixel position and visibility per frame and keeping its
  position estimate through occlusions until it comes back. Use when a verify or
  recover stage needs where a point went across a clip, or a closed loop needs
  it frame by frame from a live wrist or overhead camera.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, tracking, point-tracking, occlusion, gpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    weights:
      uri: https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt
      sha256: 6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f
      license: Apache-2.0
    provenance:
      authors: [Google DeepMind]
      code: https://github.com/google-deepmind/tapnet/tree/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2604.10582
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "track"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - track.points: Track query pixels through a clip; per-frame pixels, visibility and confidence (queries on any frame).
    - track.start: Open an online tracking session on a live frame and query pixels.
    - track.update: Advance a session by one frame; each query's pixel and visibility.
    - track.stop: Close a session and free its GPU state.
---

# track

Google DeepMind's [TAPNext++](https://github.com/google-deepmind/tapnet)
(TrecViT-B online point tracker; code and checkpoints Apache-2.0) as four
tools. It is causal and recurrent: given query pixels on one frame, it takes one
frame at a time and returns each query's position and a visibility call,
carrying a state between frames, so it follows a point behind an occluder
(the position stays an estimate) and picks it up again when it comes back.
Frames in: RGB uint8 `[H, W, 3]`, any size (the model sees each frame resized
to a square). Pixels in and out: OpenCV `[u, v]` of the frames as given.

| tool | use |
| --- | --- |
| `track.points(frames, query_points, query_frame=0, support_points=0, min_confidence=None)` | a recorded clip at once; queries on any frame (frames before it are tracked backwards) |
| `track.start(frame, query_points, ...)` -> `session_id` | open a live session on the current frame |
| `track.update(session_id, frame)` | one more frame of the same camera: each query's pixel, `visible`, `confidence` |
| `track.stop(session_id)` | free the session's GPU state (at most 4 stay open; a fifth evicts the least recently used) |

```python
tip = ctx.tool("gemini-er.point", image=wrist[0], query="the free end of the white cable")
out = ctx.tool("track.points", frames=wrist, query_points=[[tip["pixel_x"], tip["pixel_y"]]])
t = out["tracks"][0]
last_seen = max(k for k, v in enumerate(t["visible"]) if v)  # where the tip was last visible
```

```python
s = ctx.tool("track.start", frame=cam()["rgb"], query_points=[[u, v]])
while moving:
    f = ctx.tool("track.update", session_id=s["session_id"], frame=cam()["rgb"])
    if not f["visible"][0]:
        ...  # occluded or gone: f["pixels"][0] is the model's guess
ctx.tool("track.stop", session_id=s["session_id"])
```

## Output

`track.points` returns `{found, tracks, num_frames, query_frame, backend, model, timing}`;
each track is `{query_pixel, pixels [T][u, v], visible [T], confidence [T]}`.
`found` is False when some query is never visible on any frame but its query
frame. A session frame is `{session_id, step, pixels [Q][u, v], visible [Q],
confidence [Q], evicted, timing}`; a session gives exactly what `track.points`
gives forwards (measured: 0.0 px apart).

- `visible`: the model's visibility logit > 0, or `confidence >= min_confidence` when that is set.
- `confidence`: p(visible) x `tracker_certainty` (the coordinate distribution's
  mass within 8 model px of the estimate, as upstream's colab computes it), 0..1.
- An occluded point still has a pixel: where the model thinks it is behind the occluder.

## How it runs

- The pinned `tapnet` package (commit `ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b`)
  provides the model (`tapnextpp/votsp2026/model.py`'s `TAPNextPP.from_checkpoint`)
  and `tracker_certainty`; fp16 autocast on CUDA, as upstream.
- Frames are resized to the checkpoint's square input (bilinear, [-1, 1]), as
  upstream's `preprocess_frame` but from RGB (upstream takes BGR and flips it).
- Queries and tracks live in TAPNext's 256 x 256 grid whatever the input
  resolution. Its pixel `i` is centred at `i + 0.5` (soft-argmax + 0.5,
  `grid_sample(align_corners=False)`); the mapping shifts by half a pixel
  each way, which upstream's `display_to_model` omits (~0.5 px at 640 px).
- `support_points` (default 0; upstream's VOTSp2026 setting is 64) extra points
  on a grid within 1/16 of the frame around each query are tracked with it and
  dropped from the output (co-tracked through shared attention, upstream's
  tracker.py). They cut the p90 error a little on the whole-frame shake clip
  (0.51 vs 0.60-0.69 px) for 15-30 % more time per frame, but on a small part
  that moves against a static background (sim: 30 mm cube pushed 33 mm, then
  swept over by the hand) the grid lands on the table and drags the track:
  median 0.59 vs 0.47 px visible, 1.71 vs 0.62 px while hidden with the 512
  checkpoint, worst error during the sweep 4.6 vs 0.9 px with 256. Turn them on
  only for a query on a large surface that moves as one.
- `query_frame > 0`: the clip from the query frame on is tracked forwards and
  the clip up to it reversed, both from the queries.
- The model loads on the first call (module singleton); importing the module
  imports neither torch nor tapnet.

## Install

```bash
gap skills install track              # uv sync --project tools/track
gap skills check --download           # optional: prefetch the 2.53 GB checkpoint
```

torch 2.9.1 / torchvision 0.24.1 (CUDA 12.8 wheels) and einops; tapnet's
JAX / notebook dependencies are kept out of the environment (`[tool.uv]
override-dependencies`), the PyTorch path imports none of them. The checkpoint
is downloaded from DeepMind's bucket on first use and its SHA-256 checked:

| `GAP_TRACK_CHECKPOINT` | file | input | sha256 |
| --- | --- | --- | --- |
| `512` (default) | `gresearch/tapnextpp/tapnextpp_512.ckpt`, 2,532,283,010 B | 512 x 512 | `6cd0e793...3609f` |
| `256` | `dm-tapnet/tapnextpp/tapnextpp_ckpt.pt`, 2,532,282,370 B | 256 x 256 | `cb96a434...eff6da` |

| Env | Meaning | Default |
| --- | --- | --- |
| `GAP_TRACK_DEVICE` | torch device | `cuda` |
| `GAP_TRACK_CHECKPOINT` | `512` or `256` | `512` |
| `GAP_TRACK_CHECKPOINT_PATH` | a local checkpoint file instead (digest not checked) | - |
| `GAP_TRACK_CACHE` | download directory | `~/.cache/roboskillhub/track` |

## Measured (2026-11-07, Brev L40S, real YAM station frames)

The server was started the way gap starts it (`uv run --project tools/track --
python -m gap_core.rpc.server --bundle track`) and called over the RPC. The
L40S was shared with ~15 simulator processes (100 % utilization, 38 GB used
before the check), so **the timings are those of a saturated GPU**.

Frames: the 11 overhead-camera frames of `g_sota_frames_20261103-1421`
(640 x 480; a static scene, they differ only by sensor noise). 8 queries
picked as corners on frame 1: the cable's free end and plug end, the two
fingertips, three board holes, a board corner. Ground truth per frame comes
from warping those real frames with known motions (bilinear, reflected
borders; the warp and its ground truth agree to 0.02 px):

- static: the 11 frames cycled to 44;
- shake: 48 frames, +-40 px / +-24 px translation, +-4 deg rotation, up to
  1.1x zoom (up to 9 px a frame, 71 px range);
- big pan: +-100 px, +-10 deg, 1.25x (up to 23 px a frame; points leave the frame);
- occluded: the shake clip with a dark 61 px square over the free end, the
  left fingertip and a board hole for frames 16-27.

| clip (8 queries) | 512 median / p90 / max px | 256 median / p90 / max px |
| --- | --- | --- |
| static, 44 frames | 0.25 / 0.49 / 0.60 | 0.28 / 0.60 / 0.77 |
| static, no support points | 0.28 / 0.74 / 1.00 | 0.28 / 0.48 / 0.87 |
| shake, 48 | 0.30 / 0.51 / 0.80 | 0.27 / 0.51 / 0.80 |
| shake, no support points | 0.34 / 0.60 / 1.00 | 0.37 / 0.69 / 1.05 |
| shake, queries on frame 24 (both directions) | 0.25 / 0.49 / 0.86 | 0.28 / 0.48 / 0.85 |
| big pan, 48 | 0.29 / 0.53 / 30.2 | 0.32 / 0.61 / 32.8 |
| occluded shake | 0.31 / 0.53 / 0.84 | 0.28 / 0.57 / 1.15 |

- Every query visible in the frame was within 4 px on every frame, nearly all
  within 1 px. The big pan's outliers (2.3-3.1 % of samples) are the 11 of 384
  where the point is outside the image (only reflected border there); the
  model called most of them not visible.
- **Occlusion:** the three covered points were called not visible on all 12
  covered frames (confidence at most 0.19 with 512, 0.43 with 256, against
  0.79-0.99 before), visible again on
  the first or second uncovered frame, and their position estimate under the
  occluder stayed within 0.34 px (512) / 0.6 px (256) median of the truth.
  That occluder moved with the scene; one that follows a different motion
  than the hidden point is the harder case and was not tested.
- The two checkpoints are equally accurate here; the 512 one is 2-2.5x slower.

**Speed** (model time per frame, the frame's own transfer excluded; saturated GPU):

| model points (queries x (1 + support)) | 512 ms / fps | 256 ms / fps |
| --- | --- | --- |
| 1 | 355 / 2.8 | 152 / 6.6 |
| 65 | 374 / 2.7 | 150 / 6.7 |
| 8 | 386 / 2.6 | 153 / 6.5 |
| 520 (8 queries, default) | 454 / 2.2 | 196 / 5.1 |
| 2080 (32 queries) | 636 / 1.6 | 302 / 3.3 |

A live session (`track.update` over the RPC with a 640 x 480 frame, 8
queries) ran at 2.0 fps (510 ms) with 512, 3.7 fps (272 ms) with 256. The
first call loads the model: 29-34 s from a warm disk; the download (2.53 GB)
was done beforehand. **GPU memory** (`nvidia-smi`, the server process):
512: 3.4 GB with 8 queries, 3.9 GB with 32; 256: 2.5 GB / 2.9 GB.
TAPNext's cost is dominated by the image tokens (32 x 32 or 64 x 64 per frame),
so the query count matters little; on an idle GPU expect it to be much faster
than above, not measured here.

## Gotchas

- Feed frames in order, from one camera at a fixed size; a session refuses a
  different size. Skipping many frames between updates makes motion between
  them large (the shake clip had up to 23 px between frames and was fine).
- `visible` False with a pixel inside the image means occluded (or lost); the
  pixel is a guess. Gate decisions on `visible` / `confidence`, not on the pixel.
- On a saturated shared GPU this is a 2-7 fps tracker: a closed loop must
  budget for it (use `GAP_TRACK_CHECKPOINT=256` when speed matters more).
- Pixels only: lift to 3D with the depth at the returned pixel (`geometry.*`).
- The checks above used synthetic camera motion over real station images, not
  a recorded clip of a moving object; a deforming cable seen by a moving wrist
  camera is not yet measured.
