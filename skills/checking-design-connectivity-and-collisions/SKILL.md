---
name: checking-design-connectivity-and-collisions
description: Checks a brick design on a stud grid before the arms build it -
  BrickGPT's rules for bricks off the plate, overlapping bricks and bricks not
  connected to the plate through stud connections (a ground-connected component
  of the brick graph), plus a per-step check that every brick is held by what is
  already built and is not slid under a brick already placed, single-stud joints
  that carry an overhang, and a stud-clutch stability score from a linear
  program - and routes valid / out_of_plate / collision / floating /
  step_unsupported / step_blocked / weakly_connected / unstable. Use when a build
  instruction (a brick list in stud-grid cells or an LDraw step file) or a
  generated design must be accepted or rejected, or a planned build order
  checked step by step, before any brick is picked.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, design-check, connectivity, stability, ldraw]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/AvaLovelace1/BrickGPT/blob/9c53c7cb65f9de0df89562cc74b2d2e270fbcab4/src/brickgpt/data/brick_structure.py
      upstream_license: MIT
      paper: arXiv:2505.05469
gap:
  allowed_tools: []
  required_inputs: {}
  produces_outputs:
    detail: string
    n_bricks: int
    n_steps: int
    stability_max: float
    worst_brick: int
    unstable_step: int
    elapsed_s: float
  exit_conditions:
    valid: Every brick is on the plate, none overlap, all are connected to the plate through studs, each is connected and reachable from above when its step comes, no single-stud joint carries an overhang (unless allow_weak), and the stability score is under 1 (each prefix too with stability 'steps').
    invalid_input: No bricks, a size / rotation / brick_id that cannot be read, an LDraw part that is not a one-layer basic brick, an LDraw position off the stud grid or a rotation that is not an upright quarter turn; `detail` says which.
    out_of_plate: A brick leaves the plate_size grid or the layers 0 .. max_layers - 1; `out_of_plate` lists them.
    collision: Two bricks share a cell; `collisions` lists the pairs.
    floating: A brick is not connected to the plate through a chain of stud connections (above or below); `floating` lists them, `isolated` the ones touching no brick at all.
    step_unsupported: The design is connected, but some brick is not connected to the plate through the bricks before it in the step order when its turn comes; `step_unsupported` lists them.
    step_blocked: Some brick is placed under (anywhere in a higher layer over its footprint) a brick already placed, so it cannot be pressed on from above; `step_blocked` lists [brick, blocker] pairs.
    weakly_connected: A joint of a single stud carries bricks that reach past that stud (removing it would cut them off the plate); `weak_joints` lists [lower, upper] (-1 = the plate), `weak_bricks` what hangs on them.
    unstable: Holding the design (or, with stability 'steps', one of its partial builds) up needs more than the stud clutch (clutch_n, 0.98 N) at some stud; `stability_max` is the utilisation, `worst_brick` / `unstable_step` say where.
  canonical_scripts:
    - check_design: scripts/check_design.py
  examples:
    - title: Check an LDraw instruction file, fitted to the plate corner, step by step
      path: examples/canonical_subgraph.json
---

# checking-design-connectivity-and-collisions

A pure planning check: no robot tool is called and nothing moves. It answers
"can this be built, in this order, on this plate?" before the arms start, so a
bad instruction or a bad generated design is rejected for nothing.

## Input

Either `bricks` or `ldraw`.

- `bricks`: a list (or its JSON text, or BrickGPT's `{"1": {...}, ...}` dict) of
  - `{"cell": [x, y, layer], "size": "AxB", "rotation": 0 | 90 | 180 | 270,
    "step": k}`: the cell is the brick's lowest-index corner stud on the
    plate, layer 0 sits on the plate, A studs run along the plate's x at
    rotation 0 and along y at 90 / 270. `x`, `y`, `layer` (or `z`) keys work
    instead of `cell`.
  - or the ICL / BrickGPT task form `{"brick_id", "x", "y", "z", "ori"}`
    (ids 2-12 of BrickGPT's brick_library.json and ICL's lego_library.json;
    ori 1 swaps the axes; `brick_seq` is the step). ICL task files count the
    first layer as z = 1: pass `first_layer: 1`.
  - `step` defaults to the list position; bricks with the same step are placed
    in list order.
- `ldraw`: LDraw text. Each `1` line of a basic brick (3005, 3004, 3622, 3010,
  3009, 3008, 6111, 6112, 3003, 3002, 3001, 2456, 3007, 3006; more with
  `extra_parts: {"part.dat": "AxB"}`, studs along the part's own x then z) is
  a brick: x = x0/20 - h/2, y = z0/20 - w/2, layer = -y0/24 (BrickGPT's
  from_ldr / to_ldr convention, ground at y = 0). Parts between `0 STEP` lines
  share a step (no STEP lines: one brick per step). Any upright quarter turn is
  accepted; a part off the grid by more than 0.5 LDU, a tilted part, a plate or
  any other part is `invalid_input` (BrickGPT truncated to int and accepted only
  two matrices). `ldraw_offset: [dx, dy, dlayer]` shifts the model onto the
  plate, `ldraw_fit: true` puts its lowest corner at (0, 0, 0). Submodels are
  not expanded (parse them first, e.g. with parsing-ldraw-build-instructions).
- `plate_size` ("32x32"), `max_layers` (20, BrickGPT's world_dim).
- `allow_weak` (false), `stability` (`final` | `steps` | `off`), `clutch_n`
  (0.98 N), `mass_scale` (1.0).

The grid has no pitch: the same check serves LEGO and DUPLO. Only the
stability score has LEGO constants (BrickGPT's masses and clutch); for DUPLO set
`mass_scale` and `clutch_n` to your bricks (not measured here).

## The checks, in route order

1. **Out of plate** (BrickGPT `brick_in_bounds`): every cell inside
   plate_size, layer in 0 .. max_layers - 1.
2. **Collision** (BrickGPT `has_collisions`): a voxel occupancy count over 1;
   the overlapping pairs are listed.
3. **Floating** (BrickGPT `connectivity_score`): a graph with the plate as a
   node, an edge from every layer-0 brick to the plate and between any two
   bricks one layer apart whose footprints overlap (`_connected`, so a brick
   clutched under another counts); every brick outside the plate's connected
   component floats. BrickGPT's local `brick_floats` (nothing directly above or
   below) is reported as `isolated`.
4. **Per step**, bricks placed one at a time in step order:
   `step_unsupported` if a brick is not in the plate's component of the graph
   of the bricks placed so far; `step_blocked` if any already placed brick in a
   higher layer covers part of its footprint (a top-down press cannot reach it).
   `weak_at_step` (informational) lists bricks that land on a single stud with
   more than one stud of their own.
5. **Single-stud overhangs**: a joint of exactly one shared stud (a 1x1 on the
   plate counts) whose removal cuts bricks off the plate, where those bricks
   cover any cell other than that stud's column. A 1x1 column is not an
   overhang; a 2x4 on one corner stud of another is, until something else
   holds it.
6. **Stability** (added; the idea of StableLego and BrickGPT's
   `stability_analysis.py`, solved as a linear program with scipy's HiGHS, not
   Gurobi): each stud cell where a brick sits on another or on the plate is a
   joint with compression at its four corners and clutch tension at its stud.
   Every brick's vertical force and both tipping moments must balance its
   weight (BrickGPT's masses, a fit for other sizes); the program minimises the
   largest tension. `stability_max` = that tension / clutch_n: if a load path
   within the clutch exists the structure stands (the lower-bound theorem), so
   a value under 1 is stable, as BrickGPT's `max() < 1`. `stability_per_brick`
   is the largest tension at each brick's joints in that solution (indicative:
   the solution is not unique below the maximum). With `stability: steps` every
   partial build is solved too. Not modelled: horizontal stud forces (BrickGPT's
   side presses; leaving them out can only ask for more clutch), BrickGPT's 3-
   versus 4-point contact per knob, and pressing loads from the gripper.

Every list is filled whatever the route; the route is the first problem in that
order. `bricks` echoes each brick normalised (cell in the input's layer
convention, footprint along x / y, step, connected, utilisation) for the next
node (a planner or computing-brick-pose-from-stud-grid).

## Numbers (CPU, this machine, no robot)

- Unit tests: hand-built structures for every route, LDraw round trips through
  BrickGPT's own `to_ldr` formula, and 200 random structures (3-39 bricks, many
  floating) where `floating` matched BrickGPT's networkx `connectivity_score`
  (copied verbatim into the test) on all 200.
- Stability against hand statics: k 1x8 bricks cantilevered 6 studs off a 1x2
  base need clutch t = k x 3.03 g x g (0.303 of 0.98 N at k = 10); the program
  gives that to 1e-4. A beam that overhangs alone (0.0148 N) and is balanced by
  later counterweights is caught by `steps` and passes `final`.
- Timing on 100 bricks (median of 5 runs, one core of an Intel Core Ultra 7
  265KF): two 10-layer running-bond walls of 2x4s, and a random 13-layer pile
  of 1x2-2x6 bricks each resting on at least two studs.

  | model | stability off | final | steps |
  |---|---|---|---|
  | walls (100 bricks) | 6.0 ms | 48 ms | 1.07 s |
  | random (100 bricks) | 6.1 ms | 22 ms | 0.74 s |

  Both route valid; the random pile's largest clutch utilisation is 0.22.
  `steps` solves one program per partial build (100 here).

No simulation check: the skill does not move the arm (and this round has no
GPU host); there is no sim number to report.

## Provenance

Adapted from BrickGPT (Pun et al., "Generating Physically Stable and Buildable
Brick Structures from Text", arXiv:2505.05469), MIT, commit 9c53c7c:
`brick_structure.py` (`brick_in_bounds`, `has_collisions`, `brick_floats`,
`is_connected`, `Brick.from_ldr`), `connectivity_analysis.py`
(`connectivity_score`, `_connected`, `_connected_to_ground`) and the masses
of `brick_library.json`. Changed: a rectangular plate of any size instead of a
20^3 cube; breadth-first search over brick indices instead of networkx with
Brick objects as nodes (identical bricks stay distinct); LDraw with several parts
per step, four quarter turns, more parts and refusal of off-grid input. Added:
the per-step, single-stud and stability checks and the routes. The stability
score follows StableLego (Liu et al., intelligent-control-lab/StableLego, MIT)
and BrickGPT's Gurobi model in idea (static equilibrium under stud clutch limit
T = 100 g x g) but is a different, smaller linear model; its numbers will not
match BrickGPT's heat map exactly.
