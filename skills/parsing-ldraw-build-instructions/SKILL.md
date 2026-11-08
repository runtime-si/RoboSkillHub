---
name: parsing-ldraw-build-instructions
description: Reads an LDraw .ldr / .mpd build file - submodels, STEP and
  ROTSTEP meta commands, part ids, colour codes, transforms in LDraw units -
  into ordered build steps of bricks on a stud grid, each brick as a cell
  (x, y, layer from 1), a size 'AxB', a rotation 0 / 90, a height in plates and a
  colour name from the LDraw colour table. Basic LEGO bricks, plates and tiles
  (1xN, 2xN, 4xN ...), DUPLO bricks and plates, and baseplates are converted;
  every other part, and any part that is turned off the 90-degree grid or
  sits between studs, is listed with a reason instead of guessed. Use when a
  build instruction comes as an LDraw file (from an editor, BrickGPT, or the
  LDraw model library) and the robot needs the bricks it describes, in order,
  in the stud-grid convention computing-brick-pose-from-stud-grid and
  planning-brick-assembly-order take.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, ldraw, instructions, parsing, assembly, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/michaelgale/ldraw-py/blob/d650ab0a9c2a774e1ba6f206aa2cd3b1925478e3/ldrawpy/ldrmodel.py
      upstream_license: MIT
      data: https://library.ldraw.org/library/updates/complete.zip (part geometry checked, colour table; CC BY 2.0 / CC BY 4.0, LDraw.org)
gap:
  requires: {}
  allowed_tools: []
  required_inputs:
    ldraw_path: str
  produces_outputs:
    system: str
    n_bricks: int
    n_steps: int
    n_unsupported: int
    n_off_grid: int
    steps_json: str
  exit_conditions:
    steps_ready: Every part became a brick on the grid; `steps` holds them in build order.
    unsupported_part: Some parts are not in the basic-brick table (slopes, Technic, minifigures, a missing sub-file, the other system's bricks); they are in `unsupported` with the line and step, and `steps` holds the rest. A plan that needs them must be changed or finished by hand.
    off_grid: Some table parts are tilted, upside down, mirrored, turned by other than a multiple of 90 deg, or sit off the stud or plate grid (a half-stud jumper offset); they are in `off_grid` with the reason. Takes precedence over unsupported_part.
    invalid_file: No file or text, an unreadable file, no parts at all, a submodel that includes itself, or a bad argument; `detail` says which.
  canonical_scripts:
    - parse_ldraw: scripts/parse_ldraw.py
  examples:
    - title: Parse an LDraw instruction file into stud-grid steps
      path: examples/canonical_subgraph.json
---

# parsing-ldraw-build-instructions

An LDraw file is how a digital LEGO instruction travels: editors (LDCad,
MLCad, Studio's export), generators (BrickGPT writes one brick per line and a
`0 STEP` after each) and the LDraw model library all write it. This skill
turns one into the robot's vocabulary: which brick, which stud cell, which
layer, which way round, what colour, in which step. No robot, no camera; pure
parsing on the CPU (1-135 ms for 60 library models of up to 9,700 parts).

## How it composes

- **Before it**: the file (a path, or the text in `ldraw_text`).
- **After it**: planning-brick-assembly-order takes `steps` (or `bricks`) and
  orders them so every brick is supported when it is pressed; each placement
  then goes through computing-brick-pose-from-stud-grid (cell, size,
  rotation to a metric pose on the calibrated baseplate) and
  pressing-a-brick-onto-studs. The colour names are tools/colour's: look for
  the next brick with `colour.classify(..., palette="lego")` and compare
  `name` with the brick's `colour`.
- `unsupported_part` and `off_grid` still return every brick that did
  convert. Whether a partial build is useful (a wall without its window) is
  the caller's call; the lists say exactly what is missing.

## The grid convention

- **x** runs along LDraw +X, **y** along LDraw +Z, **layer** up (LDraw's -Y);
  x, y, up is right-handed.
- **cell** `{x, y, layer, rotation}`: (x, y) is the brick's lowest-x,
  lowest-y stud (its corner at every rotation, ICL's `calc_brick_loc`
  anchor), 0-based. `layer` 1 is a brick standing on the baseplate (ICL's z,
  and computing-brick-pose-from-stud-grid's `layer`), or on the level of the
  lowest brick when there is no baseplate. Layers count brick heights; a
  plate is a third of a LEGO brick (half a DUPLO brick), so a brick on a
  plate is at layer `1.333333`; `z_units` (plates above the ground, from 0)
  and `height_units` give the same heights as whole plates.
- **size** `'AxB'` (also as `brick_size`) with A <= B; **rotation** 0 puts A
  along x, 90 puts B along x (ICL and BrickGPT's `ori` 0 / 1, also returned
  as `ori`). `span_x` / `span_y` repeat the footprint in studs. A 180 or 270
  degree turn of a rectangular brick is the same brick, so it reads 0 or 90.
  A brick record can go straight into the pose skill's `cell` /
  `brick_size` and `occupied`.
- **Origin** (`origin`): `auto` uses a single baseplate's corner and top
  (cell (0, 0) is the plate's corner stud, as in the pose skill's plate
  frame; `baseplate.bricks_outside` lists any brick past its edge), else
  `min` (the lowest corner and bottom of all bricks); `ldraw` counts from
  LDraw's own origin (ground y = 0).

## What it reads

1. **Files**: `0 FILE` blocks of an .mpd (the first is the model; `0 NOFILE`
   ends a block); sub-files an .ldr references are looked up beside it, or
   passed in `files`. A plain .ldr is one model.
2. **Steps**: `0 STEP` and `0 ROTSTEP` end a step; empty steps are dropped and
   steps renumbered from 1. `source_step` keeps the step's number within its
   own (sub)model.
3. **Submodels**: `inline` (default) builds a submodel in place: its own
   steps come before the step that places it (ldraw-py's unwrap order), with
   the placing line's transform applied; `flatten` puts the whole submodel in
   the placing step. Colour 16 in a submodel takes the placing line's colour.
   A submodel that includes itself is `invalid_file`.
4. **LPub metas**: parts between `BUFEXCHG ... STORE` and `RETRIEVE` (shown
   for one step, then taken away) are dropped, as ldraw-py drops them; other
   `0` lines (comments, `GHOST`, `MLCAD HIDE`, `!LPUB`) are ignored.
5. **Parts** (type-1 lines): the table below, printed variants (`3001p01`,
   `3068bp03`) as their base part, and the LDraw colour code as a name.
   Everything else goes to `unsupported` with its line and step.

## Inputs

- `ldraw_path` (required in the graph) or `ldraw_text`; `files` (name ->
  text, extra sub-files).
- `submodels`: `inline` | `flatten`. `origin`: `auto` | `baseplate` | `min` |
  `ldraw`. `system`: `lego` | `duplo` to override the system read from the
  parts (default: the one most bricks use; the other system's bricks are
  `unsupported`, since one stud grid cannot hold both).

## Outputs

`route`, `detail`, and:

- `steps`: `[{step, model, source_step, bricks: [...]}]` in build order.
- `bricks`: the same bricks flat. Each: `id`, `part`, `base_part`, `kind`
  (`brick` / `plate` / `tile`; tiles have `studs_on_top: false`), `system`,
  `size`, `brick_size`, `rotation`, `ori`, `cell`, `span_x`, `span_y`, `z_units`,
  `height_units`, `studs_on_top`, `colour`, `colour_code`,
  `colour_in_palette` (true when tools/colour's `lego` palette has the name),
  `step`, `model`, `source_step`, `position_ldu` (the part's origin in LDraw
  units, after the submodel transforms).
- `unsupported`, `off_grid`: `[{part, line, reason, model, source_step,
  step_in_file}]`.
- `baseplate`: the plate used as the grid (`part`, `size`, `span_x`,
  `span_y`, `plate_studs` - the pose skill's `[nx, ny]`, x along LDraw X -,
  `bricks_outside`), or empty.
- `system`, `n_bricks`, `n_steps`, `n_unsupported`, `n_off_grid`,
  `steps_json` (the steps as a JSON string, for graphs that pass text).

The lists are not in `produces_outputs`, which takes only gap schema types.

## The part table

Checked against the official LDraw library (complete.zip downloaded
2026-11-08): each part's box, computed from its geometry and sub-files,
spans the listed studs along its own X and Z, with its top face at y = 0 and
its bottom 24 LDU (brick), 8 (plate, tile), 48 (DUPLO brick) or 24 (DUPLO
plate) lower. All 1452 printed variants of these parts in the library have
the base part's box.

| kind | parts |
|---|---|
| LEGO bricks | 1x1 3005, 1x2 3004, 1x3 3622, 1x4 3010, 1x6 3009, 1x8 3008, 1x10 6111, 1x12 6112, 1x16 2465, 2x2 3003, 2x3 3002, 2x4 3001, 2x6 2456, 2x8 3007, 2x10 3006, 4x6 2356, 4x10 6212, 4x12 4202, 8x8 4201, 8x16 4204 |
| LEGO plates | 1x1 3024, 1x2 3023 / 3023b, 1x3 3623, 1x4 3710, 1x5 78329, 1x6 3666, 1x8 3460, 1x10 4477, 1x12 60479, 2x2 3022, 2x3 3021, 2x4 3020, 2x6 3795, 2x8 3034, 2x10 3832, 2x12 2445, 2x16 4282, 3x3 11212, 4x4 3031, 4x6 3032, 4x8 3035, 4x10 3030, 4x12 3029, 6x6 3958, 6x8 3036, 6x10 3033, 6x12 3028, 8x8 41539 |
| LEGO tiles | 1x1 3070b, 1x2 3069b, 1x3 63864, 1x4 2431, 1x6 6636, 1x8 4162, 2x2 3068b, 2x3 26603, 2x4 87079, 2x6 69729, 4x4 1751 |
| DUPLO | bricks 2x2 3437, 2x3 87084, 2x4 3011, 2x6 2300, 2x8 4199, 2x10 2291; plates 2x4 40666, 2x6 98233, 2x8 44524, 4x4 14721, 4x6 25549, 8x8 51262 |
| baseplates | LEGO 32x32 3811, 16x16 3867, 48x48 4186, 8x16 3865, 8x32 4187, 24x40 3645, 16x24 3334, 16x32 3857; DUPLO 16x16 5930, 24x24 4268, 12x16 6851, 8x12 31043, 16x24 6475 |

Not converted (listed as `unsupported`): slopes, round and modified bricks,
jumpers, hinges, Technic, minifigures, any part not above. Adding a
rectangular part is one table line once its box is checked the same way.

## Measured

- **Tests** (`tests/bundles/test_parsing_ldraw_build_instructions.py`):
  hand-written files for steps and comments, an .mpd with a submodel placed
  twice (one turned 90 deg) on a baseplate, inline vs flatten, colour 16
  inheritance, external and missing sub-files, a self-including submodel,
  LPub buffer exchange and ROTSTEP, DUPLO on a DUPLO baseplate with a LEGO
  brick mixed in, and six kinds of off-grid part; BrickGPT's own `to_ldr`
  lines round-trip to its `(h, w, x, y, z)`; the colour names agree with
  tools/colour's palette.
- **LDraw sample models** (James Jessiman, distributed in LDraw.org's
  complete library): `pyramid.ldr` - 13 bricks in 4 steps, `steps_ready`,
  an 8x8 ring of 2x4s up to a 2x2 at layer 4. `car.ldr` - 33 bricks, plates and
  tiles in 7 steps, 28 parts `unsupported` (wheels, windscreen, steering
  wheel, lights, mudguards...).
- **60 Official Model Repository files** (library.ldraw.org/omr sets 1-60,
  each `!LICENSE Redistributable under CCAL version 2.0`; mostly Technic and
  Architecture): no crash, 1-135 ms each. 5,056 parts converted, 31,244
  `unsupported`, 1,451 `off_grid` (801 tilted - SNOT sub-assemblies on their
  side, 528 off the stud grid, 70 turned, 52 between plate heights). One file
  is all basic bricks: `1869-1 South African Flag` (Merlijn Wissink, CCAL
  2.0) - 82 bricks of 8 sizes, `steps_ready`; it is in the tests.
- Not done: no robot, no simulation (nothing moves). Whether a converted
  build can actually be pressed in order is planning-brick-assembly-order's
  check.

## Limits

- One stud grid: a build mixing LEGO and DUPLO keeps the majority system.
- Sideways building (SNOT), jumper offsets and turntables are reported, not
  converted; a model built mostly that way yields little.
- The table is by part id. An unofficial or renamed id of a basic brick is
  `unsupported` until it is added.
- LDraw colours are rendering colours: on a real camera, learn the palette
  (tools/colour's `colour.measure`) and match by name.

## The constants, and why

- **`STUD_LDU` 20 / 40, `UNIT_LDU` 8 / 24, `UNITS_PER_BRICK` 3 / 2**: the
  LDraw spec's 20 LDU per stud and 24 per brick (BrickGPT's x0 / 20 and
  y0 / 24); DUPLO from its parts' geometry (2x2 brick 3437 spans 80 LDU,
  48 high; DUPLO plates 24 high).
- **`GRID_TOLERANCE_LDU = 1`** (0.4 mm): editors write 59.9998 for 60; a
  jumper's half stud is 10 LDU.
- **`MATRIX_TOLERANCE = 1e-3`**: rotation entries written with 4-6 decimals.
- **Grid phase**: the fractional offset most bricks share (to 1/20 stud),
  so a model shifted half a stud everywhere still converts, and the brick on
  a jumper is the one reported.

## Provenance

Parsing adapted from ldraw-py (Michael Gale, MIT): `parse_file`'s `0 FILE`
split, `parse_model`'s `0 STEP` split, `get_parts_from_model`'s buffer
exchange masking, `recursive_parse_model`'s transform composition, and
`unwrap_model`'s order (a submodel's steps before the step that places it).
The cell arithmetic is BrickGPT's `Brick.from_ldr` (Ava Pun et al., MIT):
x = x0 / 20 - h / 2, y = z0 / 20 - w / 2, z = -y0 / 24. Changed: no
dependencies; LPub page, PLI, callout and view bookkeeping dropped; BrickGPT's
eight sizes and two exact matrix strings became a checked part table and any
90-degree yaw; plates, tiles, DUPLO and baseplates added; heights in plates;
unsupported and off-grid parts reported instead of raising. The script's
header lists the changes; the licences are `LICENSE.ldraw-py` and
`LICENSE.brickgpt` beside this file. Colour names and part geometry: LDraw.org
library (CC BY 2.0 / CC BY 4.0, attribution LDraw.org). LEGO and DUPLO are
trademarks of the LEGO Group, which does not sponsor or endorse this.
