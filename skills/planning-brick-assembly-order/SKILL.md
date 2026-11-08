---
name: planning-brick-assembly-order
description: Orders the bricks of a stud-grid design for a robot that presses
  each one down from above - every brick rests on the baseplate or on studs
  of a brick already placed (bottom-up, connected), nothing placed covers its
  footprint, and a parallel gripper's fingers have room on two opposite sides
  of it (no flush neighbour at its height along the closing axis). Earlier
  LDraw STEPs go first unless a brick cannot go yet; ties go bottom layer
  first, then BrickGPT's (x, y) sweep, with a look-ahead that never walls in a
  brick still to come. Optionally splits the bricks between two arms by x
  (with per-arm reach) and checks an inventory. Use when a build (from
  parsing-ldraw-build-instructions or written by hand as cells, sizes and
  rotations) must become a placement sequence for
  computing-brick-pose-from-stud-grid and pressing-a-brick-onto-studs, or to
  find out before moving which bricks a gripper cannot place.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, sequencing, dual-arm, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/AvaLovelace1/BrickGPT/blob/9c53c7cb65f9de0df89562cc74b2d2e270fbcab4/src/mesh2brick/src/mesh2brick/planning.py
      upstream_license: MIT
      paper: arXiv:2505.05469
gap:
  requires: {}
  allowed_tools: []
  required_inputs:
    steps_json: str
  produces_outputs:
    system: str
    n_bricks: int
    n_placed: int
    n_not_placed: int
    sequence_json: str
  exit_conditions:
    sequence_ready: Every brick is in `sequence`, each supported, uncovered and with a free pair of finger sides when its turn comes.
    inventory_short: Some bricks are left out because the inventory has no more of their part / size; `not_placed` lists them and the bricks that then had nothing to stand on. `sequence` holds the rest.
    unreachable_step: Some bricks lie outside every arm's reach (`reach`); listed in `not_placed`, the rest ordered.
    finger_blocked: Some bricks never had two opposite sides free for the fingers (flush neighbours on both axes, e.g. the last brick of a closed ring, or wider than the jaws open both ways); `not_placed` says which brick took the last side. Bricks that would have stood on them are listed too.
    not_buildable: Some bricks have nothing under them to press onto (hanging, on a tile, on a part not in the list), were covered before they could go, overlap another brick, or wait on a step that cannot finish (strict); listed in `not_placed`.
    invalid_input: No bricks, a brick without cell or size, a rotation not a multiple of 90, bricks of two systems, more than two arms, or an unknown option; `detail` says which.
  canonical_scripts:
    - plan_order: scripts/plan_order.py
  examples:
    - title: Order a parsed LDraw build for two arms
      path: examples/canonical_subgraph.json
---

# planning-brick-assembly-order

A build instruction says which bricks go where; it does not say which order a
robot can manage. A human can tilt a brick in under a ledge or squeeze it
between two neighbours; a robot pressing straight down with a parallel gripper
needs, at each brick's turn: studs under it (or the baseplate), nothing above
it, and room beside it for two fingers on opposite sides. This skill finds an
order that keeps all three, and says which bricks no order can place.

Pure planning on the CPU: no robot calls (0-6 ms for the test models, 0.54 s
for a 1,687-brick model with the finger check).

## How it composes

- **Before it**: parsing-ldraw-build-instructions (`steps`, `bricks` or
  `steps_json`), or hand-written bricks: `{cell: {x, y, layer}, size: "2x4",
  rotation: 0}` plus optional `kind` (`brick` / `plate` / `tile`), `system`,
  `step`, `part`, `id`.
- **After it**, per entry of `sequence`: computing-brick-pose-from-stud-grid
  with the entry's `cell`, `brick_size` and `close_axis` (and the entries
  placed so far as `occupied`), then pick the brick and pressing-a-brick-
  onto-studs at the pose's seat, with the arm named in `arm`. Re-plan from
  the bricks still to place if a press fails and the brick is set aside.
- `finger_blocked` / `not_buildable` before anything moves: change the
  design, let a human place the listed bricks, or use a different end
  effector (rerun with its finger size).

## The order

Each brick is a box on the stud grid: corner cell (x, y), footprint
`span_x` x `span_y` from its size and rotation, bottom and height in plate
heights (`z_units`, `height_units`; a LEGO brick is 3 plates, a DUPLO brick
2; derived from `layer` when not given, layer 1 standing on the plate).

1. **Overlaps** first: a brick sharing space with an earlier one is left out
   (`duplicate of brick N` for an exact copy, a modelling slip seen in
   library files; else `overlaps brick N`).
2. **Support** (the stud-overlap graph, as BrickSim's BFS grows a build from
   its anchor): a brick is supported when its bottom is on the ground
   (`ground_supports`) or on the top of a placed brick with studs, over at
   least `min_overlap_studs` (1) studs. Tiles have no studs.
3. **Cover**: a brick cannot go once anything placed sits above its
   footprint (it would have to come in from the side).
4. **Fingers**: the jaws close along x or y. For each axis, the two fingertip
   boxes beside the brick's faces - `finger_thickness` + `finger_margin`
   out from the face, `finger_width` + 2 `finger_margin` across, centred -
   from the fingertips' depth (`grasp_depth`: 5 mm under the top for LEGO,
   6 mm for 1-wide, 13 mm for DUPLO; at most the part's height) upward must
   not meet a placed brick (its studs included). `close_axis: auto` tries
   the short side first (jaws across the brick's width), then the long side
   if the jaws open wide enough (`jaw_opening` 94 mm - `jaw_margin` 4 mm).
5. **Choice**: of the bricks that pass 2-4, the one with the earliest step
   - a brick that a brick of an earlier step stands on counts as that step -
   (`step_order: prefer`; `strict` never starts a step early; `ignore` is
   BrickGPT's layers only), then the lowest bottom, then BrickGPT's sweep -
   descending (x, y) within a layer (`sweep: ascending` reverses it). A
   candidate that would take the last finger side of, or cover, a brick that
   could still be placed is skipped while there is another candidate.
6. Repeat until no brick passes; the rest go to `not_placed` with the
   reason (the root cause: a brick that lost its support because the brick
   under it was blocked says so, the blocked one says which brick took its
   last side).

**Two arms** (`arms: [left, right]`): a brick goes to the arm on its side of
`split_x` (default: the middle of the build; `left_side: -x` puts the first
arm on the low-x side), or to the other arm if the first cannot reach it
(`reach`: per arm `[x_min, x_max]` cells its footprint must lie within). The
sequence is one list - the arms take turns in its order; running them at once
is checking-dual-arm-clearance's business.

**Inventory** (`inventory`): counts per part id, per `"kind size"`
(`"plate 2x4"`) or per size (`"2x4"`); bricks past the count are left out
(BrickGPT drops them the same way), the earliest steps keep theirs.

## Inputs

- One of `bricks`, `steps` or `steps_json` (the graph passes text).
- `system` (when the bricks do not say: `lego`), `step_order`, `sweep`,
  `ground_supports` (true), `min_overlap_studs` (1).
- `check_fingers` (true), `close_axis` (`auto` | `short` | `long`),
  `grasp_depth`, `finger_thickness` (10 mm), `finger_width` (20 mm),
  `finger_margin` (3 mm), `jaw_opening` (94 mm), `jaw_margin` (4 mm).
- `arms`, `split_x`, `left_side`, `reach`, `inventory`.

## Outputs

`route`, `detail`, and:

- `sequence`: per placement the input record plus `seq`, `id`, `step`,
  `arm`, `cell {x, y, layer, rotation}`, `size` / `brick_size`, `rotation`,
  `ori`, `span_x`, `span_y`, `z_units`, `height_units`, `system`,
  `close_along` (`x` / `y`, the grid axis the jaws close along),
  `close_axis` (`short` / `long`, the pose skill's input), `supports` (ids of
  the placed bricks under it).
- `by_arm`: arm -> `seq` numbers. `not_placed`: `[{id, step, size, cell,
  reason}]`. `step_deviations`: ids placed while an earlier step was still
  open.
- `n_bricks`, `n_placed`, `n_not_placed`, `system`, `sequence_json`.

The lists are not in `produces_outputs`, which takes only gap schema types.

## Measured

Tests (`tests/bundles/test_planning_brick_assembly_order.py`): a stack given
top-first, a hanging brick and a brick on a tile, plates, one-stud support,
BrickGPT's sweep, the three step orders, the L of three 2x2s (the corner must
go before its second neighbour - BrickGPT's sweep alone walls it in), a closed
square of four, a running-bond wall (every brick gripped across it), a lower
neighbour that does not block, the long-side fallback, DUPLO jaw widths, two
arms with and without reach, inventory, overlaps and duplicates, a bridge
over a gap, and three LDraw models parsed by parsing-ldraw-build-instructions:

| model | bricks | no finger check | with the finger check (10 mm fingers) |
|---|---|---|---|
| pyramid.ldr (LDraw sample) | 13 | 13 placed, in STEP order | 12: the second layer is a pinwheel of 2x4s; the last one touches a neighbour along each axis |
| 1869-1 South African Flag (OMR) | 82 | 82 | 77; 5 have flush neighbours along both axes |
| car.ldr (LDraw sample, 28 non-brick parts removed by the parser) | 33 | 11; the rest stand on removed parts (axle holders under the chassis) | 1 |

On the 23 Official Model Repository files (sets 1-60, CCAL 2.0) with 10 or
more convertible bricks after parsing - mostly Architecture sets, the flag
included - the finger check placed 655 of 4,972 bricks (15 `finger_blocked`,
8 `not_buildable`); without it 2,501 (22 `not_buildable`, the flag
`sequence_ready`). Dense plate floors and closed rings leave a parallel
gripper no free pair of sides, and everything above a blocked brick fails
with it; without the check, the misses are bricks standing on parts the parser
could not convert. The look-ahead against BrickGPT's sweep alone: flag 77 vs
70, Empire State Building (21010) 200 vs 177, Farnsworth House 65 vs 57,
pyramid 12 vs 11, one model worse (10214: 94 vs 98), the rest equal.
Duplicated parts (same part, same place) turned up in 2 library files and
overlapping ones in 4; they are left out and listed. At most 0.54 s (the
1,687-brick Empire State Building).

Not done: nothing moves, so no simulation (round-5 rule: no GPU; a sim check
would be a press sequence on the RoboSimStudio twin). The finger and jaw
sizes are estimates for the linear_4310 (its `describe_gripper` reports pad
thickness 0); measure them, and the verdicts change.

## Limits

- Stability is only "rests on at least one stud": a long cantilever placed
  before its counterweight is allowed (the clutch holds it), but pressing on
  its free end may pop it. BrickGPT's force-based stability analysis needs a
  solver and is not ported.
- Grasps are centred on the brick (as the pose skill grasps); a grasp offset
  along the face, which would free some blocked bricks, is not searched.
- The hand above the fingers and the wrist camera are not checked against
  taller neighbours.
- The look-ahead is one step deep; it does not prove an order impossible
  (the reasons say what happened in this order).

## The constants, and why

- **`SYSTEMS`**: pitch 8 / 16 mm, plate 3.2 / 9.6 mm, studs 1.7 / 4.5 mm -
  pressing-a-brick-onto-studs' presets.
- **`FIRST_LAYER = 1`**: ICL's z and the pose skill's layer.
- **`GRASP_DEPTH`** 5 mm LEGO (+1 mm 1-wide, BrickSim's), 13 mm DUPLO (the
  press skill's DUPLO check).
- **`FINGER_THICKNESS` 10 mm, `FINGER_WIDTH` 20 mm, `FINGER_MARGIN` 3 mm**:
  singulating-by-pushing's estimates of the linear_4310's tapered tips.
- **`JAW_OPENING` 94 mm, `JAW_MARGIN` 4 mm**: the measured span; the pose
  skill's margin.

## Provenance

Adapted from BrickGPT's `plan_robotic_operation` (Ava Pun et al., MIT; layers
bottom-up, descending (x, y) within a layer, per-type inventory) and BrickSim's
`bfs_sort_connections` (Haowei Wen, MIT; every new part attached to the
assembly grown from the anchor). BrickGPT does not check support at all and
BrickSim orders a given connection graph; here both become one greedy loop
over the stud-overlap graph, with cover, finger and jaw checks, a look-ahead,
plates, tiles, DUPLO, two arms and reasons added. The script's header lists the
changes; the licences are `LICENSE.brickgpt` and `LICENSE.bricksim` beside this
file. LEGO and DUPLO are trademarks of the LEGO Group, which does not sponsor or
endorse this.
