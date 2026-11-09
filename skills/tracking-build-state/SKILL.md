---
name: tracking-build-state
description: Keeps a world model of the LEGO or DUPLO bricks placed so far
  and their stud connections - ICL Robotic_Lego_Manipulation's rules (two
  bricks connect when their footprints overlap and the upper one's bottom
  sits on the lower one's top within ICL's 0.1-1.1 brick-height window;
  moving a brick moves everything connected above it) - extended to any yaw,
  mixed plates and bricks, a grounded / floating check, refused
  interpenetrating placements, and a comparison with perceived bricks that
  reports missing, displaced and extra bricks. The state is a JSON string
  passed in and out. Use after every place, pick-off or move in a build to
  know which bricks are connected, which can be removed and which hang
  unsupported, and after a fresh observation to catch silent failures (a
  brick that came away with the gripper, one knocked off its cell).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, world-model, build-state, connectivity, cpu]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/Lego.cpp
      upstream_license: MIT
      paper: arXiv:2309.02354
gap:
  allowed_tools: []
  required_inputs:
    action: string
  # place takes brick = {id, studs, position (bottom centre), yaw_deg, height?, system?, ...};
  # move takes brick_id, new_position and/or new_yaw_deg (and carry); remove takes brick_id
  # (and carry); observe takes observations = [{position, studs, yaw_deg, id?}] (and
  # expect_visible, apply_observation). These are wired from a script node: gap has no
  # open-dict or list type here.
  produces_outputs:
    state_json: string
    n_bricks: int
    n_connections: int
    n_grounded: int
    floating_json: string
    changed_json: string
    discrepancies_json: string
    removable_json: string
  exit_conditions:
    state_updated: The action was applied (place, move, remove), the state was only read (query), or the observation agreed with every brick expected in view; state_json is the new state.
    inconsistent_with_observation: observe found missing, displaced or extra bricks (discrepancies_json); the state is unchanged unless apply_observation, which removes the missing, moves the displaced to where they were seen and adds the extras.
    rejected: The request could not be applied and the state is returned unchanged - a placement that would interpenetrate a tracked brick or reuse an id, a move that would carry bricks into others, an unknown brick id, or a malformed brick / observation.
  canonical_scripts:
    - track_build: scripts/track_build.py
  examples:
    - title: Record a placed DUPLO 2x4
      path: examples/canonical_subgraph.json
---

# tracking-build-state

A build is a stack of facts that the robot changes one brick at a time: this
brick is on those studs, that one carries three others, this one hangs
from nothing. This skill keeps those facts as a small JSON world model, so
the graph knows what is connected to what. It is also what a fresh look at
the build is checked against.

## How it composes

- **Feeds it**: a `place` after each brick the build puts down. The brick
  pose comes from computing-brick-pose-from-stud-grid's seat, and is better
  confirmed by verifying-brick-seated-by-height first: a brick resting on
  the stud tops reads a stud height high and is tracked as not connected
  (floating). A `remove` or `move` follows a pick-off
  (disassembling-a-brick-by-twist-lever, backtracking-a-wrong-placement)
  or a carried sub-assembly. An `observe` takes bricks from perception
  (verifying-build-against-model's observed list, pose6d, a stud-grid read).
- **Uses it**: `removable_json` (nothing on top) for disassembly order;
  `floating_json` (not connected to the plate through any chain) to stop a
  build before it presses onto air; `discrepancies_json` to re-place or
  re-seat. checkpointing-assembly-state-for-resume can persist `state_json`
  as part of its state.

## The model

- **A brick**: id, system (`lego` / `duplo`), studs along its own x and y,
  the world position of its bottom centre, yaw (any angle), height (a
  brick, or a 3.2 mm LEGO plate), and any extra keys (colour, part, step),
  which are kept.
- **Footprint**: studs x pitch minus 0.2 mm (ICL's `brick_len_offset_`), as
  a rotated rectangle.
- **Connected** (ICL's `is_top_connect`): the footprints overlap by more
  than a quarter pitch on every separating axis, and the upper bottom lies
  within (lower top - 0.9 h, lower top + 0.1 h), h being the smaller height.
  This is ICL's 0.1-1.1 brick-height window on z, measured against the
  lower top so that plates and bricks mix. ICL connects any strictly
  positive overlap of exact simulator poses. Here a quarter pitch keeps
  flush neighbours (0.2 mm apart) and pose noise from connecting, while one
  shared stud (a full pitch of overlap) still connects.
- **Grounded**: the bottom is within 0.1 of its height of `ground_z` (given,
  or the lowest bottom tracked), or the brick is connected on top of a
  grounded brick. Everything else is floating.
- **Interpenetration**: the footprints overlap and the height ranges overlap
  by more than 0.1 of the smaller height. A place or move that would cause
  it is `rejected`.

## Actions

- `place`: add `brick`. The detail says what it stands on (the plate, which
  bricks, or "unsupported").
- `remove`: drop `brick_id` and, with `carry` (default), every brick
  transitively on top of it; those came off with it, which is ICL's
  `update_all_top_bricks` rule for moves. With `carry: false`, the bricks
  above stay and are reported floating.
- `move`: put `brick_id` at `new_position` / `new_yaw_deg`. With `carry`,
  the bricks above get the same rigid motion (turned about the moved brick's
  bottom centre, then shifted). Refused if the moved set would enter a brick
  that is not coming along.
- `observe`: match `observations` to tracked bricks, by `id` where given,
  else greedily by nearest same-size brick within 2.5 pitches. A tracked
  brick seen beyond a quarter pitch, 0.6 stud heights in z, or 10 deg of yaw
  (modulo its symmetry) is `displaced`. One not seen is `missing`, but only
  if it is expected in view: with `expect_visible: top` (default), only
  bricks with nothing on top count, since covered bricks cannot be seen from
  above. An observation that matches nothing is `extra`.
- `query`: the counts and lists, unchanged.

## Inputs

- `action` (required); `state_json` (empty: a new build); `brick`;
  `brick_id`; `new_position`, `new_yaw_deg`; `carry` (true);
  `observations`; `expect_visible` (`top` | `all`); `apply_observation`
  (false); `system` (`duplo` default, for bricks that do not say);
  `ground_z` (the plate's top; stored in the state once given).

## Outputs

`route`, `detail`, `state_json` (schema `build-state/1`: system, ground_z,
bricks, the last 200 actions), `n_bricks`, `n_connections`, `n_grounded`,
and as JSON strings `floating_json` (ids), `changed_json` (ids added,
moved or removed by this call), `discrepancies_json` (`{id, kind, dxy_m,
dz_m, dyaw_deg}` / `{kind: extra, observation, position}`),
`removable_json` (ids with nothing on top).

## Checked

No arm motion, so no simulation check is needed: the skill calls no robot
tool. Tests (`tests/bundles/test_tracking_build_state.py`) cover stacks,
flush neighbours, bridges, one-stud and corner overlaps, rotated footprints,
a brick resting on the stud tops (not connected), refused placements and
moves, carried removals and moves (a 90 deg turn carries the brick above),
removal without carry, every observation case, LEGO plates mixed with
bricks, and three LDraw models from the hub's brick plans (from
parsing-ldraw-build-instructions / planning-brick-assembly-order), placed
brick by brick in step order:

| model | bricks | connections | floating after the build | time for all places |
|---|---|---|---|---|
| pyramid.ldr (LDraw sample) | 13 | 18 | 0 | < 0.01 s |
| 1869-1 South African Flag (OMR) | 82 | 136 | 0 | 0.20 s |
| car.ldr (LDraw sample) | 33 | 28 | 22 (they stand on axle holders and other parts the parser drops; planning-brick-assembly-order places 11 of the 33 for the same reason) | 0.02 s |

## Limits

- **Connection is geometric**: a brick tracked as connected sits where a
  connected brick sits. Whether it clutched is for
  verifying-brick-seated-by-height and a pull or probe test.
- **Stud-to-stud only**: bricks connect top to bottom. SNOT parts, hinges,
  Technic pins and tiles (no studs on top) are not modelled; a tile on top
  still lets bricks connect to it here.
- **Every call rebuilds the connection graph** (all pairs): about
  3 ms per call at 82 bricks. For thousand-brick builds, keep
  the state per sub-assembly.
- **Observation matching is greedy** (nearest first). Two same-size bricks
  swapped within 2.5 pitches can match crosswise; give ids when perception
  has them.

## The constants, and why

- **`FOOTPRINT_OFFSET = 0.2 mm`**, **`TOP_WINDOW = (0.1, 1.1)`**,
  **`GROUND_FRACTION = 0.1`**: ICL `Lego.hpp` / `is_top_connect`.
- **`MIN_OVERLAP_FRACTION = 0.25`**, **`POSITION_TOLERANCE_FRACTION =
  0.25`**: BrickSim's 2 mm position tolerance at LEGO pitch, scaled with
  the pitch (as in verifying-brick-seated-by-height).
- **`HEIGHT_TOLERANCE_FRACTION = 0.6`**: pressing-a-brick-onto-studs'
  resting band.
- **`COLLISION_FRACTION = 0.1`**, **`YAW_TOLERANCE_DEG = 10`**,
  **`MATCH_RADIUS_FRACTION = 2.5`**: chosen; a brick yawed 10 deg cannot sit
  on studs, and 2.5 pitches catches a brick knocked one or two cells over.

## Provenance

Adapted from ICL Robotic_Lego_Manipulation's `Lego` class (MIT;
`get_brick_corners`, `bricks_overlap`, `is_top_connect`,
`is_bottom_connect`, `update_brick_connection`, `update_all_top_bricks`;
Liu, Sun, Liu, arXiv:2309.02354). The C++ keeps a Gazebo-backed map of
axis-aligned, equal-height bricks and moves the simulator's models; here
the state is JSON, the bricks take any yaw and height, and the observation
check is new. The script's header lists the changes; licence in
LICENSE.icl.
