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

## Refit of the smooth term (texture-free)

The first fit's smooth term (1.0 mm at 15 cm) came from plane residuals on the
wood table, so it held the table's relief. `scripts/fit_temporal.py` measures
what changes between two captures of the same static scene instead: consecutive
wrist frames of one run with RGB mean difference < 0.5, median depth within
0.05 mm and some pixels changed (not a duplicated save) - 6 such pairs at
16 cm in the recorded data. Their depth difference / sqrt(2), low-passed at
the model's 30 px: **0.0072 z^2 = 0.16 mm at 15 cm** (p90 0.018 z^2 =
0.41 mm), against the first fit's 0.0452 z^2 = 1.0 mm: about 85 % of the old
term was not the sensor's time-varying error. Frame to frame, the fine noise
is ~0 (0.001 mm): 93-95 % of pixels repeat exactly (the stream's 1 mm steps
and its temporal filter), so the fine term (0.51 mm, spatial) stays as fitted
and is a fixed-pattern-plus-relief upper bound. Six pairs is little: the
smooth term carries that uncertainty. No flat reference surface was found in
the recorded wrist frames (the holders on the table are pocketed or carry
pens).

## The fitted model (`D405_FIT`)

`scripts/fit_noise.py` on **401 real wrist D405 frames** of the YAM station
(`~/yam_gap/threading/runs/**`, overhead frames and frames with a median depth
over 35 cm excluded; 13 166 flat 60 x 60 px patches, 1 337 flat 192 px windows):

| term | fitted | old `lego_hub.depth.add_noise` |
|---|---|---|
| fine noise (plane residual, 60 px patch) | sigma = 0.0228 z^2 (0.51 mm at 15 cm; patch median 0.54 mm, p10-p90 0.31-1.40 mm) | 0.0113 z^2 (0.25 mm) |
| its spatial correlation | Gaussian, 2.4 px (lag-1 0.93, lag-3 0.79) | 1 px |
| smooth error, frame to frame (30 px; refit, below) | **0.0072 z^2 (0.16 mm at 15 cm)** | none |
| (first fit: smooth spatial error, 192 px window - the table's relief) | (0.0452 z^2, 1.0 mm; no longer used) | |
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

LEGO physics station with the RTX cameras (#171), `LEGO_RENDER=1
LEGO_RENDER_NOISE=fitted` (and `fitted:2`), single_on_plate + single_on_brick x
{no fault + 6 station faults} x seeds 3-5 (42 placements each), the #162
thresholds unchanged:

| wrist depth | judge | false done | successes accepted | uncertain | failures caught |
|---|---|---|---|---|---|
| RTX render + old noise model (#171) | depth seat | 0/21 | 15/21 | 12 | 15/21 |
| RTX render + old noise model | fused #162 | 0/21 | 15/21 | 6 | 21/21 |
| **RTX render + fitted noise** | depth seat | 0/18 | **0/24** | 12 | 12/18 |
| **RTX render + fitted noise** | fused #162 | **0/18** | **0/24** | 24 | 18/18 |
| RTX render + fitted noise x2 | depth seat | 0/18 | 0/24 | 12 | 12/18 |
| RTX render + fitted noise x2 | fused #162 | 0/18 | 0/24 | 24 | 18/18 |

(The re-runs' physics put 24 of 42 placements in the seat this time, against
21 in #171: the outcome of the same seed is not bit-identical between runs.)

- **The verdicts do not hold.** Under the fitted noise the depth seat check
  routed **every** brick `tilted` - all 24 seated ones too - so it accepts
  nothing; the fused judge stays safe (no false done) but passes no success.
- Seat height errors under the noise: seated 0.92-1.17 mm (x2: 1.01-2.21),
  failed 2.21-2.80 mm (x2: 2.38-3.66), against 0.06-0.11 / 1.75-1.81 mm
  without it: the gap between seated and not seated shrinks from 1.6 mm to
  ~1 mm (and closes at x2).
- The likely cause is the smooth term (1.0 mm at 15 cm over a 30 px scale):
  across a 2x4 brick's top it reads as a tilt beyond the seat check's
  tolerance. That term is fitted on wood and includes the table's relief, so
  it overstates the sensor; the next step is to fit it on a flat reference (or
  on repeated frames of a static scene, which separates sensor from relief)
  and to re-tune the seat check's tilt and height tolerances on noisy depth.

## In the LEGO station

`LEGO_RENDER=1 LEGO_RENDER_NOISE=fitted` (or `fitted:<scale>`) applies the
same model to the station's rendered wrist depth in the lego_hub bridge
(`lego_hub.depth.add_fitted_noise`, kept identical to this script's; a test
checks it).
