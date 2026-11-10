---
name: augmenting-sim-depth-with-a-fitted-d405-noise-model
description: Replaces each simulated depth image of an observation with the same
  image under Intel RealSense D405 noise fitted to real wrist frames of the
  YAM station - correlated fine noise growing with z squared, a smooth
  low-frequency error, dropouts on depth edges and the stream's 1 mm
  quantisation. Routes augmented or no_depth. Use when a depth-based check
  (seat height, flushness, resting face) is tested or tuned on rendered or
  ray-cast depth and must see what the real sensor would deliver, or to
  stress it with scale > 1; no motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [depth, noise-model, d405, sim2real, rgb-d, simulation]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [lego, jigsaw, domino, pick-place]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs: {}
  # Optional inputs: observation (default a fresh robot.get_observation), cameras (names; default
  # every camera with depth), scale (1.0), seed, params (overrides of D405_FIT).
  produces_outputs:
    scale: float
  exit_conditions:
    augmented: At least one camera's depth was replaced by the noisy version (`observation` carries it; `cameras` names them).
    no_depth: No camera of the observation carries depth; nothing changed.
  canonical_scripts:
    - augment_depth: scripts/augment_depth.py
  examples:
    - title: Add fitted D405 noise to a fresh observation's depth
      path: examples/canonical_subgraph.json
---

# augmenting-sim-depth-with-a-fitted-d405-noise-model

Simulated depth is too clean. This puts the station's real D405 back into it.

## The fitted model (`D405_FIT`)

`scripts/fit_noise.py` on **401 real wrist D405 frames** of the YAM station
(`~/yam_gap/threading/runs/**`, overhead frames and frames with a median depth
over 35 cm excluded; 13 166 flat 60 x 60 px patches, 1 337 flat 192 px windows):

| term | fitted | old `lego_hub.depth.add_noise` |
|---|---|---|
| fine noise (plane residual, 60 px patch) | sigma = 0.0228 z^2 (0.51 mm at 15 cm; patch median 0.54 mm, p10-p90 0.31-1.40 mm) | 0.0113 z^2 (0.25 mm) |
| its spatial correlation | Gaussian, 2.4 px (lag-1 0.93, lag-3 0.79) | 1 px |
| smooth error (192 px window, low-passed) | 0.0452 z^2 (1.0 mm at 15 cm; median 1.07 mm) | none |
| dropout at depth edges (> 2 cm jumps) | 41 % | 50 % flying / mixed pixels |
| quantisation of the stream | **1 mm** | 0.1 mm |

`add_fitted_noise(depth, K, rng, params, scale)` draws both noise fields as
Gaussian-blurred white noise (2.4 px, 30 px), scales them by z^2, drops edge
pixels, and rounds to 1 mm. `scale` multiplies both amplitudes.

What to know:

- **The smooth term includes the table's own relief** (wood, not a flat
  reference), so it is an upper bound on the sensor's low-frequency error.
- **The stream is quantised to 1 mm** on this station (the recorder stores
  millimetres), coarser than the 0.3 mm seat threshold of #162: a check must
  average over many pixels to resolve sub-millimetre heights.
- **Self-check**: frames synthesised with the model and measured back with
  `fit_noise.py` read sigma 0.0274 z^2, correlation 1.5 px and smooth
  0.027 z^2 against the fitted 0.0228 / 2.4 / 0.045: same order, not exact;
  the 1 mm quantisation breaks up the pixel correlation and leaks into the
  patch residual.
- The overall dropout of the real frames (26 %, mostly far background and the
  sensor's minimum range) is not modelled; only edge dropouts are.

## Re-run: depth seat and the #162 fused judge on noisy rendered depth

RESULTS_PLACEHOLDER

## In the LEGO station

`LEGO_RENDER=1 LEGO_RENDER_NOISE=fitted` (or `fitted:<scale>`) applies the
same model to the station's rendered wrist depth in the lego_hub bridge
(`lego_hub.depth.add_fitted_noise`, kept identical to this script's; a test
checks it).
