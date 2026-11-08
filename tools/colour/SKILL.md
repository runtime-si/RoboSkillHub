---
name: colour
description: What colour a segmented part is — robust colour of a mask
  (boundary eroded, clipped highlights dropped, lightness tails trimmed, Lab
  median), nearest entry of a palette by CIEDE2000 with the margin to the
  runner-up and an ambiguous / unknown status, white balance from grey world
  or a white/grey reference region, and a mask's dominant colours. Built-in
  palettes are the LDraw LEGO/DUPLO colours and domino tile/pip colours; a
  per-camera palette learned from labelled parts is passed as a list. Use when
  a workflow must tell bricks, tiles or parts apart by colour (sorting,
  find-the-red-2x4, verifying a build), especially the pairs VLMs confuse --
  black vs dark grey, tan vs yellow.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, colour, recognition, lego, domino, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    tasks: [sorting, pick-place, recognition]
    tuned_on: [yam-station-orbbec-overhead]
    version: 0.1.0
    notes: >-
      On the YAM station's overhead camera the LDraw values alone are not
      enough (58-65% on real frames); a palette measured on that camera
      (colour.measure) with a bench reference reached 91%. LDraw colour values
      CCAL 2.0 (CC BY 2.0), LDraw.org.
    provenance:
      code: https://github.com/scikit-image/scikit-image/blob/5d91fed804ec8d9133118f02b3a785790df75934/src/skimage/color/delta_e.py
      upstream_license: BSD-3-Clause
      data: https://library.ldraw.org/library/official/LDConfig.ldr (CCAL 2.0 / CC BY 2.0)
      paper: G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference formula", Color Res. Appl. 30(1), 2005
gap:
  requires: {}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "colour"]
    protocol: stdio-msgpack
  tools:
    - colour.classify: Name a segmented part's colour from a palette (LEGO, DUPLO, domino or custom) by CIEDE2000, with the margin to the runner-up and an ambiguous status.
    - colour.measure: Robust colour of a mask (Lab and sRGB, highlights and shadows trimmed), to build a per-camera palette from labelled parts.
    - colour.white_balance: Per-channel white-balance gains from grey world or from a white/grey reference region; optionally the balanced image.
    - colour.dominant_colours: Cluster a mask into its dominant colours (k-means in Lab, near clusters merged), each with its share and optionally its palette name.
---

# colour

An RGB frame and a part's mask in; the part's colour name, how sure, and why.
Numpy only — no model weights, no GPU, ~4 ms per `classify` on a 640x480
frame, ~170 ms for `dominant_colours` over a whole frame.

## Why a tool and not the VLM

VLMs and open-vocabulary detectors name colours from appearance, and
appearance moves with the light: an underexposed dark-grey brick *is* black in
the pixels, a tan brick under warm light *is* yellow. The fix is measurement
against a reference, which a VLM cannot do: this tool converts to linear
light, divides the illuminant out against something of known colour, and
compares in CIELAB with CIEDE2000, the colour difference built to match what
people call "the same colour". Use the VLM to find and segment the part
(`sam3.segment_box`, `grounding-dino`), this tool to name its colour.

## Tools

- **`colour.classify(rgb, mask=None, palette="lego", white_ref=None,
  reference_rgb=None, ambiguous_margin=3.0, unknown_delta_e=25.0, kL=1.0,
  erode_px=2, top_k=5)`** — `rgb` is RGB (not BGR) `[H, W, 3]`, uint8 or float
  0..1; `mask` nonzero = part (None = whole image). Returns `name`,
  `delta_e` (CIEDE2000 to it), `second_name`, `margin_to_second`, `status`
  (`classified` / `ambiguous` / `unknown_colour` / `empty`), the measured
  `lab` and `rgb`, pixel counts, the `ranking` and the `gains` applied. Route
  on `status`; on `ambiguous` the two candidates are `name` and
  `second_name` — re-look closer (wrist camera), or ask about those two only.
- **`colour.measure(rgb, mask, white_ref=None, ...)`** — the robust colour
  `classify` compares: `lab`, `rgb`, `lab_iqr` (spread), `n_used`. Build a
  per-camera palette from a few labelled parts:
  `[{"name": "dark_bluish_grey", "lab": measure(...)["lab"]}, ...]` (names
  may repeat — several samples of one colour, nearest wins).
- **`colour.white_balance(rgb, mask=None, method="grey_world",
  reference_rgb=None, exposure=True, apply=False)`** — per-channel gains in
  linear sRGB. `grey_world` assumes the masked scene averages grey (fixes the
  cast only). `reference` maps the robust colour of `mask` onto
  `reference_rgb` (default white `#F4F4F4`): fixes the cast and, with
  `exposure`, the brightness — the part that separates black from dark grey.
  `clipped_fraction` over 0.5 means the reference is saturated and the gains
  are wrong: use a grey reference or a lower exposure. Pass the returned dict
  as `white_ref`.
- **`colour.dominant_colours(rgb, mask=None, k=None, max_k=6, palette=None,
  white_ref=None, return_labels=False)`** — k-means in Lab over the unclipped
  pixels; without `k`, `max_k` clusters merged while two are within 10
  CIEDE2000; clusters under 5% dropped. Each colour has `lab`, `rgb`,
  `fraction`, and with a palette `name` / `delta_e` / `margin_to_second`.
  `return_labels` adds an int16 label image. For a mask that spans two
  parts, a printed tile (domino face and pips), or a background leak.

`white_ref` (on `classify`, `measure`, `dominant_colours`) is one of:
`"grey_world"`; a mask of a white region (or of colour `reference_rgb`); a
`colour.white_balance` result; three gains.

## Palettes

- `"lego"` — 25 current solid LEGO colours with LDraw codes and values
  (black 0, blue 1, green 2, dark pink 5, bright green 10, yellow 14, white
  15, tan 19, orange 25, magenta 26, lime 27, dark tan 28, reddish brown 70,
  light / dark bluish grey 71 / 72, medium blue 73, bright light orange 191,
  bright light yellow 226, dark green 288, dark red 320, dark / medium azure
  321 / 322, sand green 378, dark orange 484, red 4). The retired old greys
  (7, 8) are left out: they sit a few CIEDE2000 from the bluish greys. Nearest
  pairs: yellow / bright light orange 10.2, dark azure / medium blue 10.3.
- `"lego_basic"` — the 14 commonest (basic bricks, DUPLO): red, blue,
  yellow, green, black, white, light / dark bluish grey, tan, orange, lime,
  dark green, reddish brown, medium azure.
- `"domino"` — white, ivory (nominal `#EFE6CF`, not measured), black and
  red / blue / green / yellow pips.
- A list of `{name, rgb}` / `{name, lab}` / `{name, hex}`.

LDraw's values are rendering colours, not measurements of plastic through
your camera; on a real station prefer a learned palette (numbers below).

## Robust statistic

Boundary pixels mix part and background (a 2 cm brick is 10-20 px wide at
0.9 m), so the mask is eroded by `erode_px` unless that would leave under a
quarter of it (thin cables keep their pixels). Clipped pixels (a channel ≥
250) say nothing about hue and are dropped unless more than half the part is
clipped (a white table). Then in Lab the lightness tails outside the 15th-85th
percentile go (highlights above, shadows below) and the per-channel median is
the colour. On harsh synthetic patches (20% specular, 25% shadow pixels) the
median's CIEDE2000 to the true colour was 0.64 (0.92 untrimmed); a plain mean
of the mask classified only 50% there against 100%.

## Measured

### Synthetic: LDraw colours under lighting changes

Every palette colour as a 32 px patch, rendered in linear light under four
casts (neutral, warm ~3200 K `(1.25, 1, 0.7)`, cool `(0.82, 1, 1.28)`,
fluorescent `(0.92, 1.1, 0.85)`) times four exposures (0.6-1.2), with a
0.75-1.0 shading ramp, 12% shadow pixels (x0.35), 8% specular pixels (+0.6),
sensor noise σ = 2/255; a white and a light-grey reference patch in the
same light. Top-1 (16 lighting conditions per colour):

| palette | none | grey world | white reference | grey reference |
|---|---|---|---|---|
| lego_basic (224) | 91% | 90% | **100%** | **100%** |
| lego (400) | 77% | 76% | **100%** | **100%** |

Harsh variant (20% specular, 25% shadow, exposures 0.4-1.5): lego 64% /
66% / 94% / **99%**. The white reference drops because at exposure 1.5 the
white patch clips (bright light orange read as yellow); the grey reference
does not clip. Without correction the confusions are the expected ones:
light/dark bluish grey, green/dark green, white/tan, tan/dark tan, dark pink /
magenta. A margin under 3 caught 44% of the wrong answers on `lego` (62% on
`lego_basic`) while flagging 15% (8%) of the right ones.

### Real station frames (overhead Orbbec, 640x480)

38 known-colour objects in 7 frames (`g_sota_frames_*` frame 1 of five
sessions, two `threading/*.npz`), masks by hand boxes plus a threshold
(Otsu on luminance for dark parts, thin whitish structure for the cable):
the 30 mm cube and the open box (dark slate grey → `dark_bluish_grey`), the
breadboard and gripper fingers (`black`), table / paper / cable (`white`),
the bench (`tan`), a pen (`red`), parts bins (`blue`), the gripper housing
(`white` or `light_bluish_grey`). Ground truth is by eye; none of these is
LEGO. "bench ref" balances on the bench with its colour calibrated once
(the 0630 bench measured against the paper sheet in that frame: sRGB
230 / 183 / 149); the reference object is left out of the score.

| palette | none | grey world | bench reference |
|---|---|---|---|
| LDraw `lego` | 22/38 (58%) | 17/38 (45%) | 19/31 (61%) |
| LDraw `lego_basic` | 22/38 (58%) | 20/38 (53%) | 20/31 (65%) |
| learned (leave one frame out) | 23/29 (79%) | 26/29 (90%) | **20/22 (91%)**, 1 wrong-but-sure |

Dark grey vs black on `lego_basic`: with no correction the dark parts read
black 6/6 (the cube at L* 11-17 next to a white table that sets the exposure)
and black parts 11/11 right; with the bench reference dark grey 3/6, black
7/11 (the matte-black board reads L* 31, between LDraw black 16 and dark
bluish grey 42). With a learned palette and the bench reference: dark grey
4/6, black 11/11. Tan: the bench reads L* 69-75, a* 2-4, b* 9-12 raw —
much less saturated than LDraw tan (75, 3, 24) — so it lands between tan and
light bluish grey (margins 0-3.6, mostly `ambiguous`); grey world makes it
worse (the bench dominates the frame, so grey world neutralises it).

What the station numbers say: this camera renders colour desaturated and
auto-exposes per frame, so absolute LDraw values are a fallback; for a real
task, measure each brick colour once on the station (`colour.measure` with
the same `white_ref`) and classify against that list. The learned frames
include four near-duplicate views of one scene, so 91% is optimistic.

## The constants

- `AMBIGUOUS_MARGIN = 3.0` CIEDE2000 — see the synthetic margin numbers
  above. It catches near ties only; a wholesale illuminant shift puts a colour
  on a neighbour with a large margin (the station box read black at margin
  18.8), which only `white_ref` fixes.
- `UNKNOWN_DELTA_E = 25.0` — right answers on the station frames were within
  0.2-17 of their entry; LDraw's nearest solid colours are 10-15 apart.
- `L_TRIM = (15, 85)` percentiles, `CLIP_LEVEL = 250`, `ERODE_PX = 2`,
  `MIN_PIXELS = 12` — robust statistic, above.
- `WHITE_RGB = #F4F4F4` (LDraw White) — default reference colour.
- `MERGE_DELTA_E = 10`, `MIN_FRACTION = 0.05` — one surface's shading spans
  ~5-10 CIEDE2000 on the station frames; under 5% is a leak or a glint.

## Quirks

- RGB order: pass RGB, not OpenCV's BGR.
- The camera's own auto white balance and auto exposure move between frames;
  a reference in the same frame (`white_ref` mask) follows them, cached gains
  do not.
- A thin mask (cable, pen) keeps its boundary pixels, so its colour is mixed
  with the background: the station's white cable read L* 75 with the bench's
  tint.
- Grey world fails when one colour fills the frame (a wooden bench, a white
  table).

## Licence

The bundle is Apache-2.0. `_ciede2000.py` is adapted from scikit-image
(`src/skimage/color/delta_e.py`, `deltaE_ciede2000`, BSD-3-Clause, copyright
the scikit-image team); its licence travels beside it as
`LICENSE.scikit-image` and its header lists the changes. The LEGO colour
names, codes and values come from the LDraw.org colour table `LDConfig.ldr`
(2026-05-29 update), licensed CCAL 2.0 (CC BY 2.0) under the LDraw
Contributor Agreement — attribution: LDraw.org. LEGO is a trademark of the
LEGO Group, which does not sponsor or endorse this.
