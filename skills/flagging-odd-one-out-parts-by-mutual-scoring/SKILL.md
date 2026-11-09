---
name: flagging-odd-one-out-parts-by-mutual-scoring
description: Finds the part that differs from its siblings in one frame -
  a fallen or turned domino in a line, a face-up tile among face-down ones,
  a standing one among lying ones, a wrong face - with no reference image,
  no before frame and no training. The parts (masks, boxes, or a sam3 text
  prompt) are cut out, turned to their long axis (the 180 deg turn chosen to
  match the others) and scored against each other three ways - appearance by
  MuSc's mutual patch scoring on DINOv2 (tools/musc), height maps by the same
  nearest-patch rule on the depth, and the boxes' orientation and footprint -
  and a robust z-score over the parts says which are odd and why. Use when
  many copies of one part were placed and one may be wrong - before a domino
  line is triggered, after a row of bricks or tiles is laid, or to pick the
  one tile that needs turning over.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [verification, anomaly, zero-shot, domino, lego, jigsaw, dinov2, depth]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [domino, lego, jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/HUST-SLOW/MuSc-V2/tree/b5e911dbc089/models/modules
      upstream_license: MIT
      paper: https://arxiv.org/abs/2401.16753
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
    - musc.score
  required_inputs: {}
  produces_outputs:
    n_parts: int
    camera_name: string
    appearance_scored: bool
    aligned_layout: bool
  exit_conditions:
    outliers_found: At least one part is odd - its appearance score over z_threshold (3.5) robust z and 1.4x the median, its height-map score over 3.5 z and 2 mm, its box turned over 20 deg from the others' (when they are laid out aligned), or a side of its box 1.25x off the median at 3.5 z; outliers lists them and parts[i].why says which test.
    all_consistent: Every part passed every test; detail gives the highest z.
    too_few_parts: Under min_parts (4) parts were given or segmented, or no RGB frame.
  canonical_scripts:
    - flag: scripts/flag_odd_one_out.py
  examples:
    - title: Check a line of dominoes from the wrist camera before triggering it
      path: examples/canonical_subgraph.json
---

# flagging-odd-one-out-parts-by-mutual-scoring

A row of copies is its own reference: if nine dominoes look one way and the
tenth another, the tenth is wrong, whatever "right" is. MuSc scores images
of one kind against each other with no labels; this skill makes the parts
of one frame into such a set and adds what the camera's depth and the parts'
boxes say.

## How it composes

- **Before it**: the parts on the table, and a way to find them: `masks` or
  `boxes` (frame pixels) from an earlier perception step, or `prompt` and
  the skill calls `sam3.segment_text` on the frame. The frame is `frame`,
  or `camera` in `robot.get_observation` (default the first non-wrist
  camera; give the wrist camera's name for a close-up, which is what
  appearance needs - see Limits).
- **After it**: `outliers` (indices into the parts) and
  `outlier_positions` (world points, when the frame has depth and a pose)
  for a recovery step - standing a domino up, turning a tile over, re-seating
  a brick; `parts[i].why` (`appearance`, `height`, `orientation`,
  `footprint`) says which kind of fix.

## What it does

1. **Parts**: masks (or boxes, or sam3's masks for `prompt`), at least 30 px.
2. **Normalise**: each part's min-area rectangle; a crop around it (25 %
   margin) turned so the long side is horizontal, the background outside its
   mask filled grey; between a crop and its 180 deg turn the one nearer the
   median 32 x 32 thumbnail of all parts (two rounds), so tiles placed
   end-for-end compare alike (`align_flip`).
3. **Appearance** (when the parts' median short side is at least 20 px):
   `musc.score` on the 448 px crops with their masks; robust z (median, 1.4826
   MAD, MAD at least 2 % of the median); odd if z > 3.5 and the score > 1.4x
   the median.
4. **Height** (`use_depth`, when the frame has depth and intrinsics): the
   support plane by RANSAC on a ring around the parts; each part's height
   map in its normalised crop at 24 x 24; each 3 x 3 patch wholly inside the
   part against its nearest patch of every other part (MuSc's rule, on
   heights in metres); the part's worst patch is its score; odd if z > 3.5
   (MAD at least 0.5 mm) and over 2 mm.
5. **Geometry**: the boxes' angles (mod 180). When their robust spread is
   under 10 deg (a line, a grid) or `check_orientation: true`, a part turned
   over 20 deg from the circular mean is odd; a box side 1.25x off the
   median at z > 3.5 is odd.

Step 2 turns every crop upright, so appearance cannot see a part turned in
place; step 5 does.

## Inputs

`prompt` | `masks` | `boxes` (one is required), `frame` or `camera`,
`use_depth` (true), `use_appearance` (true), `check_orientation` (auto),
`align_flip` (true), `max_parts` (40), `min_parts` (4), `z_threshold`
(3.5), `min_ratio` (1.4).

## Outputs

`outliers`, `outlier_positions`, `parts` (per part: `index`, `centre_px`,
`obb_px`, `area_px`, `score`, `z`, `depth_score`, `depth_z`, `turn_deg`,
`flipped`, `odd`, `why`), `scores`, `z_scores`, `depth_scores`,
`depth_z_scores`, `n_parts`, `camera_name`, `appearance_scored`,
`aligned_layout`, `detail` (says when under 8 parts or when appearance was
skipped for small parts).

## Checks

CPU unit tests (`tests/bundles/test_flagging_odd_one_out_parts_by_mutual_scoring.py`,
21 with the tool's): the mutual-scoring maths on synthetic tokens, masks, RsCon,
the tool through a fake runtime (raw pixel patches for DINOv2), and the skill
on rendered scenes - each test (appearance, height, footprint, orientation,
skipped appearance for small parts, scattered parts, the 180 deg flip),
sam3 and boxes as part sources, too few parts, bad arguments.

**Scenes** (`python -m tests.bundles._odd_one_out_scenes DIR --seeds 8`, rendered
by `tests/bundles/_domino_render.py`: a look-down camera at 0.2 m, tiles
48 x 24 mm, white / ivory / black, 0.3-1 px blur, noise, glare on a third;
masks and depth from the renderer): 10 tiles per scene, 3 layouts x (2 odd
kinds + a control with none) x 8 seeds = 72 scenes. Lying tiles are placed
end-for-end at random. The skill ran with the real tools/musc (DINOv2-base,
448 px crops; `python -m tests.bundles._odd_one_out_eval DIR`) on an RTX PRO 6000 (1.6 s per scene of 10 crops). The
baseline is the hub's current no-reference route, a VLM: Gemini Robotics-ER
2 (`gemini-robotics-er-2-preview`, temperature 0) given the frame with the
parts numbered, asked which differ ("a part turned end-for-end counts as the
same"; without that sentence it flagged the end-for-end tiles too and was
clean on 4 of 24 controls). Odd kinds: found = the odd tile flagged; exact =
nothing else flagged; controls: clean = nothing flagged.

| layout | odd kind | this skill: found / exact | Gemini-ER: found / exact |
|---|---|---|---|
| line of standing tiles | one fallen | 8 / 7 of 8 | 8 / 8 |
| | one turned 90 deg | 8 / 8 | 8 / 8 |
| | none (control): clean | **8 / 8** | 0 / 8 |
| 2 x 5 face down | one face up | 8 / 8 | 8 / 8 |
| | one standing | 8 / 8 | 8 / 7 |
| | none (control): clean | **7 / 8** | 4 / 8 |
| 2 x 5 face up, one value | one other value | 3 / 3 | 8 / 8 |
| | one face down | 4 / 3 | 8 / 8 |
| | none (control): clean | **8 / 8** | 5 / 8 |
| all | | found 47 / 56, controls clean 23 / 24 | found 56 / 56, controls clean 9 / 24 |

Seeds 0-3 were used to choose the crop size (448 against 224), `MIN_RATIO`,
the small-part rule and the geometry tests; seeds 4-7 are held out: found
19 / 24 and 20 / 24, controls clean 11 / 12 and 12 / 12. The standing line is
judged without appearance (18 px tops): fallen by height and footprint,
turned by orientation. By appearance (all 8 seeds) the odd tile had the
highest score in 30 of 32 scenes: a face-up tile among face-down ones scored
2.1-2.8x the median, a face-down one among face-up ones 1.2-2.1x, a standing
one 1.3-1.6x, a different value 1.1-1.9x; the identical controls' highest
was 1.1-1.4x. The ranking is better than the gate: a stricter `min_ratio`
trades the subtle kinds for false alarms.

So: Gemini finds more (every odd tile, including the subtle value and
face-down ones) but raises a false alarm on 15 of 24 identical rows - it
reads small angle jitter as "turned" - while this skill misses subtle
appearance differences and almost never cries wolf (1 of 24). They combine:
use the VLM to propose and this skill (or its scores) to confirm, or this
skill alone where a false alarm costs a stop.

## Limits

- **Renders, not the station.** No real frame on disk has a row of
  identical parts (the 4 black rings on the black board are the closest),
  so the numbers are rendered top-down tiles with blur, noise and glare; real
  lighting, shadows between standing tiles and segmentation errors are not
  in them.
- **Appearance needs pixels.** From the overhead camera at ~1 m a domino is
  ~10 px wide and appearance is skipped; at 0.2 m (the wrist camera) a lying
  tile is ~96 x 48 px. The top of a standing domino (18 px wide at 0.2 m)
  is judged by height and geometry only.
- **Subtle differences are under the noise**: a different pip count among
  ten identical tiles was found 3 times in 8; a face-down tile among face-up
  ones (something missing) 4 in 8. Both are "the part lacks what the others
  have", which nearest-patch scoring sees only through DINOv2's context.
- **The set must be nominally identical.** A mixed set (different values,
  colours, sizes) makes every part odd; group by colour / size first
  (routing-parts-to-sort-bins-by-id-and-colour, colour.classify).
- Thresholds were chosen on the first 4 seeds of the renders (Checks) and
  held on the other 4; with 8 or fewer parts the robust z is coarse.

## The constants, and why

See the docstrings in `scripts/flag_odd_one_out.py`: `MIN_RATIO = 1.4` (the
renders' identical controls peaked at 1.37x the median), `Z_THRESHOLD = 3.5`
(Iglewicz-Hoaglin), `MIN_APPEARANCE_PX = 20`, `TURNED_DEG = 20`,
`ALIGNED_SPREAD_DEG = 10`, `FOOTPRINT_RATIO = 1.25` (a fallen domino's box is
2-3x a standing one's), `DEPTH_MIN_SCORE = 2 mm` and `DEPTH_NOISE = 0.5 mm`
(the station cameras' depth noise at the table), `CROP_SIZE = 448`.

## Provenance

The mutual scoring is MuSc (Li et al., ICLR 2024; MuSc-V2,
github.com/HUST-SLOW/MuSc-V2 at b5e911dbc089, MIT), run by tools/musc,
which adapts its SNAMD, MSM and RsCon code (see that bundle and NOTICE). This
script's crop normalisation, the depth branch (the nearest-patch rule on
height maps in place of MuSc-V2's Point-MAE tokens), the geometry tests and
the robust-z decision are new.
