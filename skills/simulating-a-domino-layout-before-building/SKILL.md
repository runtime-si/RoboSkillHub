---
name: simulating-a-domino-layout-before-building
description: Checks a planned domino line or curve in a rigid-body simulation before
  the arm builds it. It rebuilds the slots (planning-a-domino-layout's `slots` or any
  slot poses) as real-size tiles on a table, replays the trigger push
  (triggering-a-domino-chain-by-push's stroke, the fingertips the only part of the
  robot that collides, as predicators' cascade probe does) and records when each tile
  falls (10 deg held for 3 samples). It then reruns the same layout with every slot
  jittered by the placing error and reports the fraction that still reaches the target
  and the pair that breaks most often. Routes cascades, fragile (the plan works but
  placing error breaks it), stalls (with the slot where the wave stops) or push_failed.
  CPU only, about 50 ms per rollout of 12 tiles. No motion. Use after
  planning-a-domino-layout and before placing the first tile, or to choose a pitch or
  radius.
license: MIT
compatibility: requires gap>=0.1; the mujoco Python package (CPU) for the physics, else it routes sim_unavailable
metadata:
  category: planning
  tags: [domino, simulation, physics, cascade, robustness, topple, verify, plan]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/cascade_probe.py
      upstream_license: MIT (predicators); Apache-2.0 (MuJoCo model/sleep/dominos.xml)
      paper: https://arxiv.org/abs/2509.26255
gap:
  allowed_tools: []
  required_inputs: {}
  # Inputs (not listed: gap has no list or open-dict type here, so they are wired from a
  # script node in the same graph): slots (planning-a-domino-layout's `slots` or
  # `slot_poses`, Se3Pose dicts whose x axis is the facing, or [x, y, facing_deg]; slot 0
  # is the trigger tile), table_z (default: the slots' z), target_index (-1: the last
  # slot), trigger (push | kick), tile_size, tile_mass, table_friction, tile_friction,
  # contact_fraction, push_speed, overshoot, kick_factor, trials, position_sigma,
  # yaw_sigma_deg, min_success, timestep, seed.
  produces_outputs:
    n_tiles: int
    target_index: int
    reached_target: bool
    stall_at: int
    n_fallen: int
    wave_speed: float
    push_travel: float
    first_fell: bool
    trials: int
    success_rate: float
    weakest_pair: int
    sim_time: float
    wall_time: float
  exit_conditions:
    cascades: In the planned layout every tile up to the target fell, and at least `min_success` (0.9) of the `trials` rollouts with the slots jittered by the placing error did too. `topple_times` gives each tile's fall time and `wave_speed` the settled wave's speed along the path.
    fragile: The planned layout reaches the target, but fewer than `min_success` of the jittered rollouts do; `weakest_pair` is the pair (k, k + 1) where they broke most often and `trial_stalls` lists where each stopped. Tighten the pitch or widen the radius there.
    stalls: In the planned layout the wave stopped; `stall_at` is the first slot up to the target still standing (its predecessor fell), `fallen` says which fell.
    push_failed: The first tile did not fall. Either the trigger stroke would end within 6 mm of slot 1, so the trigger skill would not push a line this tight (try the `kick` trigger), or the stroke or kick did not carry the tile past 10 deg.
    sim_unavailable: The physics engine is not installed or rejected the model; nothing was judged.
    invalid_input: Fewer than two slots, a slot without a facing, two neighbouring slots overlapping, no table height, a target that is not after slot 0, a tile size that is not height > width > thickness, an unknown trigger, or a non-positive speed, timestep or mass (also the route of a raised error).
  canonical_scripts:
    - simulate_layout: scripts/simulate_layout.py
  examples:
    - title: Simulate a planned domino line before building it
      path: examples/canonical_subgraph.json
---

# simulating-a-domino-layout-before-building

The planner says where the tiles go and checks each pair with a 2D toppling model.
This skill runs the whole layout as 3D rigid bodies: curves, tiles sliding on the
table, the real trigger stroke, and the robot's placing error. It answers "will this
cascade, and how much error can it take" before any tile is placed.

## What it does

1. **Scene** (after MuJoCo's `model/sleep/dominos.xml`): a table plane at
   `table_z`, one free box per slot (`tile_size`, default 48 x 24 x 7.5 mm,
   `tile_mass` 10 g) standing on its end, broad faces normal to the slot's
   facing. Elliptic friction cones, impratio 10, a 0.5 ms step. Tile-table
   friction `table_friction` 0.5 (predicators' real-table fit), tile-tile
   `tile_friction` 0.25 (planning-a-domino-layout's measured value).
2. **Trigger** (predicators `cascade_probe.run_counterfactual_push_probe`). With
   `push`, the default, a block the size of the closed YAM fingertips (24 x 30 x
   12 mm) is the only robot part, so a knock from the arm cannot pass as a
   cascade. It moves like triggering-a-domino-chain-by-push's stroke: it starts
   8 mm behind the first tile's back face, at `contact_fraction` 0.7 of its
   height, and runs along the facing at `push_speed` 0.05 m/s. It stops
   tip travel + 8 mm past the back face and holds there while the wave runs; the
   retreat is left out, as upstream. A stroke that would end within 6 mm of slot 1
   is not run (`push_failed`), because the trigger skill refuses it too. With
   `kick`, the first tile starts upright turning about its front edge at
   `kick_factor` (1.2) times the angular speed that just tips a lone tile. That is
   planning-a-domino-layout's slow-push model, and it checks the chain alone at any
   pitch.
3. **Falls** (predicators `cascade_certificate._topple_onset`,
   `fallen_threshold`). A tile has fallen once its tilt from vertical stays at or
   past 10 deg for 3 samples 10 ms apart; its fall time is the start of that run.
   The run ends when every tile fell, 1 s after the last fall, or at
   `30 s`.
4. **Verdict**: the first slot up to `target_index` that still stands is
   `stall_at`.
5. **Robustness** (new; the idea of Roussel et al., SIGGRAPH 2019): `trials` (20)
   more rollouts with every slot moved by Gaussian noise of `position_sigma`
   (2 mm) per axis and `yaw_sigma_deg` (3 deg) in facing. A draw that would
   overlap the previous tile is redrawn (20 times, then the slot slides forward
   along its facing until clear). Under `min_success` (0.9) is `fragile`.

## Measured

All local CPU runs (`tests/bundles/_domino_layout_sim_eval.py`; the physics is
MuJoCo 3.15 on CPU, no GPU). Wall time is about 50 ms per rollout of 12 tiles and
5 s for a 22-slot arc with 20 jittered rollouts.

**Straight lines, 12 tiles, 48 x 24 x 7.5 mm** (R runs, Sk stops at slot k, P no
trigger):

| pitch (mm) | 15 | 16 | 17 | 18 | 20-48 | 55 | 56 | 57 | 58 | 60 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| push, 0.5 ms | P | P | P | P | R | R | S1 | S1 | S2 | S1 | S1 |
| push, 0.25 ms | P | P | P | P | R | R | R | S1 | S2 | S2 | S1 |
| kick, 0.5 ms | S1 | S1 | S1 | R | R | R | R | S1 | S2 | S2 | S1 |
| kick, 0.25 ms | S1 | S1 | S1 | R | R | R | R | S1 | S2 | S1 | S1 |

Against planning-a-domino-layout's 2D model (GapChain / StackedLine):

| quantity | 2D model | this simulation |
|---|---|---|
| widest pitch that runs | 55.3 mm | 55-56 mm (the step size decides 56) |
| slow push | stalls in the second tile at 16 mm, runs from 18 mm | stalls at slot 1 at 15-17 mm, runs from 18 mm |
| settled wave speed at 20 / 36.9 / 48 mm | 0.92 / 0.80 / 0.68 m/s | 0.87 / 0.755 / 0.658 m/s (tiles 20-30 of 40) |

The hub's twin runs agree as well. triggering-a-domino-chain-by-push cascaded 8 of 8
and 10 of 10 tiles at 37 mm, and this simulation runs there. The placing skill's
twin cascaded at 48 mm and not at 72 mm; here 48 mm runs and 72 mm stops at slot 1.

**Friction** (widest running pitch): table 0.3 / 0.5 / 0.8 against tile 0.15 /
0.25 / 0.4 gives 53-56 mm everywhere, and the 37 mm wave goes 0.54-0.63 m/s. The
verdicts barely depend on the two friction guesses.

**Placing error** (12 tiles, 40 rollouts, the fraction reaching the last tile):

| pitch (mm) | 20 | 25 | 30 | 37 | 42 | 48 | 52 | 55 |
|---|---|---|---|---|---|---|---|---|
| 2 mm / 3 deg (default) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.65 |
| 5.4 mm / 5 deg (camera RMS) | 0.97 | 0.97 | 0.90 | 0.65 | 0.55 | 0.17 | 0.10 | 0.00 |

The planner's auto pitch (36.9 mm) has full margin at the twin's 2 mm placing
error, but it is `fragile` if tiles land with the station camera's 5.4 mm error.
At that error, 30 mm or tighter is needed.

**Planner arcs** (180 deg sweep, 20 jittered rollouts) give the planner's verdict
against the simulation:

| radius | 40 | 50 | 60 | 80 | 100 | 150 | 250 mm |
|---|---|---|---|---|---|---|---|
| auto 36.9 mm: planner | too_tight | too_tight | planned | planned | planned | planned | planned |
| auto 36.9 mm: simulation (success) | fragile 0.80 | fragile 0.80 | cascades 0.95 | cascades 0.90 | cascades 0.95 | cascades 1.00 | cascades 1.00 |
| 48 mm: planner | too_tight | too_tight | too_tight | too_tight | too_loose | planned | planned |
| 48 mm: simulation (success) | stalls at 1 | stalls at 1 | fragile 0.40 | fragile 0.35 | fragile 0.50 | fragile 0.85 | cascades 0.90 |

Where the planner says `planned` at the auto pitch, the simulation agrees. At
48 mm (1.0 h) on a 150 mm radius, the planner accepts a curve that the simulation
finds fragile (0.85). The 2D pair check does not see sliding and placement error
adding up along a curve.

**Station twin cross-check** (GPU, rshgpu2/3; harness
`~/sim-wt/simulating-a-domino-layout-before-building`). The YAM station twin
(MuJoCo-Warp at 500 Hz, 10 g tiles, friction 0.5) had tiles standing at the slots
and the hub's triggering-a-domino-chain-by-push pushing slot 0 with the left arm.
Ground truth was each tile's tilt from `sim.get_object_obb` after the push (fallen at
10 deg or more). The prediction is this skill on the same slots, with its defaults
and with the twin's friction (0.5 / 0.5), 20 jittered rollouts:

| layout | twin (fell / tiles) | prediction, default | prediction, twin friction | nominal agrees |
|---|---|---|---|---|
| line 20 mm x 10 | all 10 | cascades 1.00 | cascades 1.00 | yes |
| line 37 mm x 10 | all 10 | cascades 1.00 | cascades 1.00 | yes |
| line 48 mm x 10 | all 10 | cascades 1.00 | cascades 1.00 | yes |
| line 55 mm x 8 | stops at slot 1 | fragile 0.75 (reaches nominally) | fragile 0.65 | no |
| line 58 mm x 8 | stops at slot 1 | stalls at 2 | stalls at 1 | yes |
| line 72 mm x 6 | stops at slot 1 | stalls at 1 | stalls at 1 | yes |
| arc r 150, 36.9 mm x 9 | all 9 | cascades 1.00 | cascades 1.00 | yes |
| arc r 100, 48 mm x 7 | all 7 | fragile 0.65 (reaches) | fragile 0.80 | yes |
| arc r 60, 36.9 mm x 6 | all 6 | cascades 0.95 | cascades 1.00 | yes |
| arc r 50, 48 mm x 4 (55 deg a tile) | all 4 | stalls at 1 | stalls at 1 | no |

The nominal verdict agrees on 8 of 10 layouts. The 55 mm line sits at the
critical gap: the twin's wave dies, the model's barely runs, and its jittered
rollouts flag it as fragile. On the 50 mm arc all four twin tiles ended flat
(90 deg). The fingertip block's swept footprint stays 20 mm short of tile 1, but the
twin's whole hand and arm collide, and the ground truth was read only after the
trigger skill's retreat, so a knock by the arm cannot be ruled out. Unresolved.

A bug caught on the way: MuJoCo reads `euler` in degrees by default, and a first
version passed radians, so every curved slot faced almost straight ahead and every
arc stalled. The scene sets `angle="radian"` now, and a test pins it: a tile on a
60 deg slot stands at 60 deg.

## The constants, and why

- **`TILE_SIZE`, `TILE_FRICTION` 0.25**: as planning-a-domino-layout.
- **`TILE_MASS` 10 g**: 8.6 cm3 of plastic or wood is 6-13 g. Identical tiles make
  mass drop out of the topple condition; it only matters against the push.
- **`TABLE_FRICTION` 0.5**: predicators' fit to its real table.
- **`FALLEN_DEG` 10, `FALLEN_SAMPLES` 3**: predicators' `fallen_threshold` and
  `_TOPPLE_MIN_STEPS`.
- **`TIMESTEP` 0.5 ms**: halving it moved only the knife-edge pitches (56 and
  60 mm, at the critical gap); 0.1 ms on the 250 mm arc gave the same fall times
  to 10 ms.
- **`CONTACT_FRACTION` 0.7, `APPROACH_CLEARANCE` 8 mm, `OVERSHOOT` 8 mm,
  `NEXT_CLEARANCE` 6 mm, `PUSH_SPEED` 0.05 m/s**: triggering-a-domino-chain-by-push
  (its measured 0.048-0.051 m/s at speed_scale 0.25).
- **`KICK_FACTOR` 1.2**: planning-a-domino-layout's `PUSH_SPEED_FACTOR`.
- **`POSITION_SIGMA` 2 mm, `YAW_SIGMA_DEG` 3**: the placing skill's twin placed
  within 2 mm; use 5.4 mm (the station camera's RMS) for a pessimistic check.
- **`TRIALS` 20, `MIN_SUCCESS` 0.9**: 20 rollouts resolve a 0.9 rate to about
  +-0.07 in about a second for 12 tiles.

## Not done

- No real clip was compared. The cross-checks are the planner's 2D model, the
  hub's twin results quoted in the other domino skills, and the station twin
  runs above (fall times were not compared: the twin's tilts were read after the
  trigger skill returned).
- The robot body is not in the scene (fingertips only, by design), so an arm that
  sweeps a standing tile while placing is placing-an-upright-domino-beside-neighbours'
  build-order check, not this one.
- The engine is not a hub dependency: the hub's test environment does not install
  it, and the physics tests skip there. The skill routes `sim_unavailable` without it.

## Provenance

Adapted from predicators (MIT, commit afb6f29): `envs/pybullet_domino/cascade_probe.py`
(`run_counterfactual_push_probe`, `_ensure_fingertips_only_collision`,
`_SETTLE_STEPS`), `cascade_certificate.py` (`_topple_onset`, `_TOPPLE_MIN_STEPS`),
`components/domino_component.py` (`fallen_threshold`) and `sim_core.py`
(`domino_friction`). Licence `LICENSE.predicators` beside this file. Also from
MuJoCo's `model/sleep/dominos.xml` (Apache-2.0, commit 70dede9): the scene and its
contact options. Changed: MuJoCo instead of PyBullet, the hub's slots and tile
instead of the env's body pool, the trigger skill's stroke (or the planner's kick)
instead of the Push option on a Fetch arm, and jittered rollouts instead of
identical retries (this engine is deterministic). Routed exits replace the
(ok, detail) verdict.
