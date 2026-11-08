---
name: planning-a-brick-mosaic-from-an-image
description: Turns an image into a one-layer LEGO mosaic on a baseplate of a
  given stud size - brickmos' pipeline (pixelate to one cell per stud, nearest
  palette colour per cell, bill of materials, BrickLink wanted list) with
  CIEDE2000 matching in CIELAB, optional Floyd-Steinberg dithering, per-colour
  part counts respected (an optimal assignment when stock is short, with the
  substitutions reported), runs of one colour merged into 1x2 / 2x2 / ... plates
  to cut placements, a preview image, and an ordered placement list in the hub's
  stud-grid convention (cell [x, y, layer], layer 1 on the plate, size 'AxB',
  rotation, colour) that checking-design-connectivity-and-collisions,
  planning-brick-assembly-order and computing-brick-pose-from-stud-grid read.
  Use when a picture, logo or pattern must become a flat brick mosaic the arms
  can build from the parts in the bins, or to find out before building which
  colours are short.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, mosaic, assembly, colour, ciede2000, dithering, bill-of-materials, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/merschformann/brickmos/blob/4a7e956720d347d198f931edc38852c9f3bf4060/brickmos/brickify.py
      upstream_license: MIT
      data: https://library.ldraw.org/library/official/LDConfig.ldr (CCAL 2.0 / CC BY 2.0)
      paper: G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference formula", Color Res. Appl. 30(1), 2005; R. W. Floyd, L. Steinberg, "An adaptive algorithm for spatial greyscale", Proc. SID 17(2), 1976
gap:
  requires: {}
  allowed_tools: []
  required_inputs: {}
  produces_outputs:
    detail: string
    n_cells: int
    n_placements: int
    n_empty: int
    n_substituted: int
    n_colours: int
    mean_delta_e: float
    p95_delta_e: float
    max_delta_e: float
    mean_delta_e_blurred: float
    ideal_mean_delta_e: float
    placements_json: string
    steps_json: string
    bom_json: string
    bricklink_xml: string
    preview_path: string
    pixelated_path: string
    elapsed_s: float
  exit_conditions:
    planned: Every cell of the mosaic has a part in the colour nearest to the image there (to the dithered target with dithering) and the inventory, if given, covers every part.
    planned_with_substitutions: Every cell has a part, but some cells use another colour because their nearest colour ran out (or had no part of the size needed); `substitutions` lists wanted -> used, how many cells and the extra CIEDE2000.
    inventory_short: The inventory has fewer studs than the mosaic has cells (or no 1x1 left in any colour for a cell no larger part fits); `empty_cells` lists the uncovered cells, `placements` covers the rest.
    invalid_image: The image is missing, unreadable, empty, not [H, W], [H, W, 3] or [H, W, 4], or has NaN / infinite values.
    invalid_input: An option cannot be read - unknown palette / fit / resample / dither / order / part_kind / system, a mosaic off the plate or over 128 x 128, sizes without 1x1, an inventory colour not in the palette or a negative count; `detail` says which.
  canonical_scripts:
    - plan_mosaic: scripts/plan_mosaic.py
  examples:
    - title: A 32 x 32 plate mosaic from the bins' plates, with 1x2 / 2x2 merging
      path: examples/canonical_subgraph.json
---

# planning-a-brick-mosaic-from-an-image

A flat mosaic is the easiest complete LEGO build for an arm: one layer, every
placement one press onto a calibrated baseplate. This skill does the design
half: from an image and the parts on hand to the list of presses. No robot
tool is called and nothing moves.

## How it composes

- **Before it**: an image (a file, or an RGB array from any node) and what is
  in the bins (`inventory`, by colour name of the palette). For a palette
  measured on the station's camera, learn it with `colour.measure` and pass it
  as a list.
- **After it**:
  - `placements_json` straight into checking-design-connectivity-and-collisions
    (`bricks`, `plate_size` = `plate_studs`): a one-layer mosaic is always
    `valid` there (tested).
  - `steps_json` into planning-brick-assembly-order (`steps_json`, its
    required input): its `cell` is a `{x, y, layer}` dict and each part is its
    own STEP in this skill's order, so with `step_order: prefer` the order is
    kept unless a part cannot go yet. See the finger limit below.
  - per placement, computing-brick-pose-from-stud-grid with `cell =
    pose_cell` (`{x, y, layer, rotation}`), `brick_size`, `system`, and for
    plates `brick_height: 0.0032` and a `grasp_depth` under it (e.g. 0.002;
    its defaults are a 9.6 mm brick held 5-6 mm down, which it refuses on a
    plate); pass the placements already done as `occupied` (each carries its
    stud `cells`).
    Then pick a part of `colour` and pressing-a-brick-onto-studs.
  - `bill_of_materials` / `bricklink_xml` to restock; `substitutions` to
    decide whether a substituted colour is acceptable.

## The pipeline

1. **Load** (path or array; RGBA is composited over `background_rgb` in
   linear light; grey is expanded).
2. **Fit** to the mosaic's aspect: `crop` (centre crop, default), `pad`
   (letterbox in `pad_rgb`) or `stretch` (brickmos, distorts).
3. **Pixelate** to one cell per stud: `area` (default) averages each cell's
   pixels in linear light; `linear` is brickmos' `cv2.INTER_LINEAR` on sRGB,
   which samples ~4 pixels per cell and aliases when shrinking a photo 10x.
4. **Match** each cell's colour to the palette by CIEDE2000 in CIELAB (D65).
   brickmos takes the Euclidean distance in sRGB, which puts dark colours too
   close together and light ones too far apart.
   - `dither: floyd_steinberg` diffuses each cell's Lab error to the next
     cells (7/16, 3/16, 5/16, 1/16, serpentine rows). Cell for cell it is
     further off; seen from a step back (both images blurred by one stud) it
     is much closer (numbers below). It also breaks up runs, so fewer parts
     merge.
5. **Stock**: with an `inventory` and more cells wanting a colour than the
   colour has studs, the assignment is solved as a transportation problem
   (scipy HiGHS dual simplex; cells x colours, capacities = studs per colour;
   integral at a vertex): the total CIEDE2000 is the least possible under the
   counts. When all the stock together is short, it covers as many cells as
   the stock allows at the least total error and leaves the rest empty. With
   dithering the stock is used greedily in scan order (the first rows get
   their colours first).
6. **Merge**: for each size in `sizes`, largest first and both orientations,
   anchors in raster order from the image's top left, a part goes wherever
   its footprint is one colour, still free, and the colour has that part
   left. The remaining cells get 1x1 parts; a cell whose colour has no 1x1
   left takes the nearest colour that has one (a substitution), else stays
   empty. Merging never changes a colour, so the colour error is the same
   with or without it.
7. **Order**: `rows` (ascending plate y, then x), `serpentine`, or `colour`
   (one colour at a time, most used first, rows within: one bin at a time).

## Conventions

- **Image on the plate**: seen from above, image right = plate +x, image up =
  plate +y. The image's top-left cell is plate cell `(origin_x, origin_y + h -
  1)`. Rotate or flip the image to put it another way round.
- **Placement**: `cell [x, y, layer]` (the footprint's lowest-index corner,
  layer 1 on the plate), the same as `x`, `y`, `layer`; `size` / `brick_size`
  `'AxB'` with A <= B; `rotation` 0 = A studs along plate x, 90 = along y (so
  a horizontal 1x2 is `1x2` at 90); `span_x`, `span_y`; `cells` (stud cells
  covered); `pose_cell`; `kind` (`plate`, `brick` or `tile`), `system`,
  `part` (LEGO part number, e.g. 3024 plate 1x1, as brickmos orders),
  `colour`, `ldraw_colour`, `bricklink_colour_id`, `seq`, `id`, `step`.
- **Inventory**: `{colour: n}` is n 1x1 parts; `{colour: {"1x1": n, "1x2":
  m, "2x2": k}}` by size. Colours of the palette not listed have none; a name
  not in the palette is `invalid_input` (a typo would otherwise zero a
  colour). Names match case-, space- and grey/gray-insensitively.

## Inputs

- `image` (array or path) or `image_path`.
- `plate_studs` ([32, 32]), `mosaic_size` (default the plate less `origin`),
  `origin` ([0, 0]).
- `palette`: `lego` (25 current LDraw solid colours, the colour tool's),
  `lego_basic` (14), `brickmos` (brickmos' 33 BrickLink colours), or a list
  of `{name, rgb | hex | lab, ldraw?, bricklink_id?}`.
- `inventory` (None = unlimited), `sizes` (["1x1"]; must include 1x1),
  `part_kind` (`plate`), `system` (`lego`; `duplo` gets no part numbers).
- `fit`, `resample`, `dither`, `order`, `background_rgb`, `pad_rgb`,
  `baseplate_rgb` (an empty stud in the preview and the blurred error,
  #969696).
- `out_dir` (default a new temp dir), `write_files` (true), `preview_px`
  (16), `grid_cell` (8, brickmos' helper grid; None for none),
  `preview_studs`, `spares` (0, brickmos' CLI default).

## Outputs

`route`, `detail`, `mosaic_size`, `origin`, `plate_studs`, `n_cells`,
`n_placements`, `n_empty`, `n_substituted`, `n_colours`; the CIEDE2000 of
each covered cell to the image's cell colour: `mean_delta_e`,
`median_delta_e`, `p95_delta_e`, `max_delta_e`; `mean_delta_e_blurred` (both
blurred by a 1-stud Gaussian first, S-CIELAB's idea, empty studs as the
baseplate); `ideal_mean_delta_e` (the palette's floor: every cell its nearest
colour, no stock limit). Lists: `placements`, `bill_of_materials` (`colour`,
`size`, `kind`, `part`, `count`, `studs`, `in_stock`, colour ids, `rgb`),
`substitutions`, `empty_cells`, `colour_grid` (image rows, top first). Text:
`placements_json`, `steps_json`, `bom_json`, `bricklink_xml` (brickmos'
format). Files in `out_dir`: `mosaic.png` (parts outlined, studs, helper
grid, empty studs crossed), `pixelated.png` (the cell colours before
matching), `bricklink.xml`, `plan.json`.

## Measured

CPU only (one core of an Intel Core Ultra 7 265KF), no robot, no GPU, no
simulation (the skill does not move the arm). Run
`.venv/bin/python tests/bundles/_brick_mosaic_eval.py --out DIR`. Images:
three synthetic (`blocks`: a 4 x 4 checker of 16 exact LDraw colours;
`sweep`: hue across, lightness down; `shapes`: anti-aliased disc, triangle and
ring in off-palette colours on a gradient, blurred and noised) and three
scikit-image sample photos: astronaut (NASA, public domain), coffee (Rachel
Michetti, CC0), chelsea (Stefan van der Walt, CC0). None is committed; the
photos load from scikit-image when installed.

**Colour error against part count** (palette `lego`, crop, area; mean
CIEDE2000 per cell; placements for 1x1 only / 1x1+1x2+2x2 / 1x1 to 2x4):

| image | 16x16 | 32x32 | 48x48 | 64x64 | placements at 32x32 |
|---|---|---|---|---|---|
| blocks | 0.00 | 0.00 | 0.00 | 0.00 | 1024 / 256 / 128 |
| sweep | 11.35 | 11.33 | 11.34 | 11.33 | 1024 / 321 / 190 |
| shapes | 10.49 | 10.19 | 10.15 | 10.13 | 1024 / 338 / 218 |
| astronaut | 10.87 | 10.60 | 10.66 | 10.61 | 1024 / 512 / 413 |
| coffee | 8.86 | 8.87 | 8.75 | 8.69 | 1024 / 478 / 386 |
| chelsea | 10.72 | 10.51 | 10.48 | 10.46 | 1024 / 490 / 380 |

The per-cell error is the palette's gamut, not the resolution: it hardly
moves from 16 to 64 studs (each cell is compared with the image averaged over
that cell, so finer grids do not get easier). Merging cut the placements of
the photos to 47-50% (1x2, 2x2) and 37-40% (up to 2x4) at 32x32, and 31-32%
at 64x64 with up to 2x4, at no colour cost; flat synthetic images merge more.

**Dithering** (32x32, 1x1): per cell 13.6 vs 10.6 (astronaut), 12.2 vs 8.9
(coffee), 13.6 vs 10.5 (chelsea), 15.0 vs 10.2 (shapes); blurred by one stud
3.35 vs 7.28, 2.66 vs 5.70, 1.72 vs 6.51, 3.50 vs 8.93. On `sweep` (saturated
hues outside the palette) it barely helps (9.65 vs 9.94). It is off by
default: the FS dots (magenta specks in the astronaut's orange suit) look
worse close up and stop runs from merging.

**brickmos' pipeline against this one** (32x32, 1x1, error in CIEDE2000):
brickmos' stretch + bilinear + sRGB-Euclid gave 11.41 / 9.22 / 11.22 on
astronaut / coffee / chelsea with its own palette and 11.51 / 9.89 / 11.49
with `lego`; this skill 10.29 / 7.94 / 8.93 and 10.60 / 8.87 / 10.51. Keeping
brickmos' fit and resampling and changing only the matching to CIEDE2000:
9.80 / 7.60 / 9.86 and 10.47 / 8.61 / 10.56. The error is measured in the
metric the new matching minimises, so this shows the change is consistent,
not that people prefer it.

**Short stock** (32x32, `lego_basic`, 1x1; each colour stocked at a fraction
of what the image wants):

| image | stock | route | substituted | empty | mean dE (floor) |
|---|---|---|---|---|---|
| shapes | 100% | planned | 0 | 0 | 14.81 (14.81) |
| shapes | 70% + 25 of every colour | planned_with_substitutions | 211 | 0 | 15.82 |
| shapes | 50% | inventory_short | 12 | 513 | 13.36 |
| astronaut | 70% + 25 | planned_with_substitutions | 168 | 0 | 12.56 (11.29) |
| astronaut | 50% | inventory_short | 2 | 513 | 8.87 |
| coffee | 70% + 25 | planned_with_substitutions | 262 | 0 | 12.79 (10.27) |
| coffee | 50% | inventory_short | 28 | 514 | 7.28 |

When stock is short overall, the error of the covered cells drops under the
floor: the program leaves the hardest cells empty. Those empty cells are
scattered, not a region; plan with a smaller mosaic if that matters.

**Timing** (astronaut, `lego`, ms): 32x32 18, 48x48 32, 64x64 41, 96x96
91, 128x128 158 plain; merging up to 2x4 adds under 10 ms; dithering 87 /
183 / 316 / 699 / 1235; an inventory at 60% of every colour (the program
covering as much as it can) 60 / 163 / 393 / 1767 / 5266; generous stock
with substitutions 63 / 111 / 153 / 363 / 698 (files not written).

**Downstream** (32x32 plates, astronaut and shapes): checking-design-
connectivity-and-collisions `valid` for every mosaic; planning-brick-
assembly-order without its finger check `sequence_ready` with this skill's
order kept exactly; with the finger check (10 mm fingers) it placed 352 of
1,024 1x1 plates, 139 of 512 (astronaut, merged to 2x2) and 88 of 218 (shapes,
up to 2x4) before `finger_blocked`, whichever of the three orders went in
and with `step_order` prefer or ignore (352 / 1,024 every time for 1x1; 132-150
of 512 merged). computing-brick-pose-from-stud-grid, fed each placement's
`pose_cell` in this skill's `rows` order with the earlier ones as `occupied`
(plates: `brick_height` 3.2 mm, `grasp_depth` 2 mm), returns the same `cells`
for every part, `computed` for 63 of 1,024 1x1 plates (the first row and
column) and `ungraspable` for the rest; 40 of 512 merged. Its finger check is
one cell either side; the assembly order's is the fingertip's size in mm and
it reorders, hence the higher count there.

## Limits

- **A dense mosaic cannot be finished by a parallel gripper holding parts
  by their sides.** Every part's last-filled neighbours leave no free pair of
  opposite sides for the fingers (planning-brick-assembly-order: about a
  third of a 1x1 mosaic placed). Options not built here: grip by the studs or
  with a suction / press tool, build in strips on a separate plate, or let
  the gripper place onto a plate it then slides into place. This skill's
  output is unchanged by the choice; the downstream order will be.
- LDraw's colour values are rendering colours, not plastic under the
  station's light; to judge the result through a camera, measure the palette
  (`colour.measure`).
- The merge is greedy (larger first, raster order), not a minimum tiling; it
  never merges across colours and never staggers joints (a flat mosaic does
  not need it, a mosaic hung on a wall might).
- Substituted cells within one exact colour group are chosen in scan order,
  not spread out.
- Dithering with short stock is greedy in scan order.
- 128 x 128 cells at most (5.3 s with a 60% inventory).

## The constants, and why

- **Palettes**: `lego` / `lego_basic` are the colour tool's LDraw colours
  (LDConfig.ldr 2026-05-29, CCAL 2.0, attribution LDraw.org) plus BrickLink
  colour ids (brickmos' where it has the colour; magenta 71, sand green 48
  from BrickLink's colour guide). `brickmos` is brickmos' table verbatim.
- **`PARTS`**: LEGO part numbers; 3024 (plate 1x1) is brickmos'; the brick
  ids are checking-design-connectivity-and-collisions' table.
- **`PLATE_STUDS = 32 x 32`**: the common LEGO baseplate (brickmos' default
  mosaic is 48 x 48).
- **`GRID_CELL = 8`**, **`SPARES = 0`**: brickmos' `--grid_cell` and
  `--spares` defaults.
- **`BLUR_SIGMA = 1` stud**: where neighbouring studs merge for a viewer, the
  scale dithering works at.
- **`SUBSTITUTION_BIG = 1000`**: more than any CIEDE2000 between two sRGB
  colours (~120), so covering a cell always beats leaving it empty.
- **`BASEPLATE_RGB = #969696`**: LDraw light bluish grey.
- **`MAX_CELLS = 128 x 128`**: from the timing above.

## Provenance

Adapted from brickmos (Marius Merschformann, MIT, commit 4a7e956):
`brickify.py` (`read_colors`, `closest_color`, `replace_with_brick_colors`,
`add_grid`, `write_xml`, `main`'s resize-then-replace pipeline) and
`defaults.py` (its palette, kept as `brickmos`). Kept: the pipeline, the
palette, the colour statistics as a bill of materials, the BrickLink XML,
the helper grid. Changed: CIEDE2000 in CIELAB instead of sRGB distance
(brickmos does not use CIEDE2000 and has no dithering), linear-light box
averaging instead of bilinear resizing, crop / pad instead of stretching.
Added: dithering, stock limits with an optimal assignment and substitutions,
part merging, placements in the hub's convention, the inputs of the three
brick skills, the routes. The CIEDE2000 code is the hub colour tool's copy of
scikit-image's (`scripts/mosaic_ciede2000.py`, BSD-3-Clause,
`LICENSE.scikit-image`); brickmos' licence is `LICENSE.brickmos`. LEGO is a
trademark of the LEGO Group, which does not sponsor or endorse this.
