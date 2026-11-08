---
name: planning-a-domino-layout
description: Plans a domino line along a path on the table - a line, an arc, a
  polyline (corners rounded) or a sampled curve - and returns the ordered slot
  poses (tile centre and facing, the pose placing-an-upright-domino-beside-neighbours
  takes), the pitch and how far it stays under the critical gap a toppling wave
  can cross (from a 2D toppling simulation), the checks that matter on a curve
  (the inner corners, where a falling tile's top lands on the next one), safety
  gates left open while building and the order to close them, and a build order
  in which the open hand never meets a standing tile and the arm never reaches
  past one. No motion. Use before building a domino line or curve with the arm,
  to choose the pitch and the slots and to know whether the layout will topple.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [domino, layout, pitch, toppling, gates, build-order, plan]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/maxencelaunayquerre/domino-fall-simulation/blob/f6147f635b2a4a3f3f7ca0589284f45dda6aa309/scripts/spacing_threshold.py
      upstream_license: MIT
      paper: https://zmatt.net/domino-physics/
gap:
  requires: {connector: [robot.describe_workspace]}
  allowed_tools:
    - robot.describe_workspace
  required_inputs:
    # OPTIONAL extra inputs (gap has no list or union type; wire them from a
    # script node in the same subgraph): `table_z`, `pitch` (m or "auto"),
    # `n_tiles`, `length`, `tile_size`, `gates`, `gate_every`, `arms`, `table`,
    # `obstacles`, `friction`, `topple_margin_min`, `gate_margin_min`. `path`
    # may also be a line / arc / polyline dict or a plain list of points.
    path: Centerline
  produces_outputs:
    first_slot: Se3Pose
    pitch: float
    n_tiles: int
    face_gap: float
    critical_gap: float
    critical_pitch: float
    topple_margin: float
    max_face_gap: float
    gate_count: int
    gate_width_slots: int
    gate_margin: float
    min_inner_gap: float
    max_turn_deg: float
    max_contact_offset: float
    path_length: float
    used_length: float
  exit_conditions:
    planned: Every slot passes every check; `slots` / `slot_poses` (path order, slot 0 is the trigger tile), `build_steps` (placing order with each step's arm, standing neighbours and tiles at risk), `gates` (in closing order) and the margins are filled. `problems` lists what was dropped or is outside what has been seen to work (an automatic gate with no clear spot, a pitch over 1.0 h, slots placed between two standing tiles).
    too_tight: The tiles are too close for the hand or the physics - a pitch under what the open fingers need beside a neighbour (19.8 mm for a 7.5 mm tile), a pitch at which a slow trigger push stalls, footprints overlapping at an inner corner or where the path comes back on itself, a curve so sharp that a falling tile's top lands off the next tile's face, or a build step whose open fingers would touch a standing tile; `detail` names the slots.
    too_loose: A pair of slots is farther apart than the toppling wave can cross, less `topple_margin_min` (the critical gap is 47.8 mm face to face for a 48 x 24 x 7.5 mm tile, a 55.3 mm pitch); `detail` names the pair.
    too_short: The path (or `length`) holds fewer than two tiles, or not `n_tiles` at the pitch.
    out_of_reach: A slot is outside every arm's tool-centre box or reach, off the `table`, or the placing skill's lead-in leg at a slot would leave the box.
    blocked: An obstacle is where the open hand goes over a slot, a given gate cannot stop the wave or cannot be closed without the arm reaching past standing tiles, or no order places every slot without reaching past standing tiles.
    invalid_input: The path or tile size cannot be read (also the route of a raised error).
  canonical_scripts:
    - plan_layout: scripts/plan_layout.py
  examples:
    - title: Plan the slots, pitch, gates and build order of a domino line along a path
      path: examples/canonical_subgraph.json
---

# planning-a-domino-layout

Decide where every domino goes before the arm moves: the slots along the path,
a pitch that topples with room to spare, where to leave safety gates, and an
order of placements the placing skill can actually carry out.

## What it computes

1. **The path.** A `line` (`start`, `end`, or `start` + `heading_deg` +
   `length`), an `arc` (`center`, `radius`, `start_deg`, `sweep_deg` or
   `end_deg`; a negative sweep runs clockwise), a `polyline` (`points`, each
   corner rounded to an arc of `corner_radius`, default 80 mm, shortened when a
   leg is too short) or a sampled `curve` (`points`, as a curve fitter returns
   them; a gap `Centerline` is one). A plain list of points is a polyline. All
   in world metres on the table; heights are ignored. The line starts at the
   path's start: slot 0 is the trigger tile and the wave runs toward the end.
2. **The slots.** Every `pitch` of arc length, each tile's centre on the path
   and its facing (the broad faces' normal, the way it topples) along the
   path's tangent there: `position` (z = `table_z`) and `rotation` (a turn
   about z, so its x axis is the facing), which is the slot pose
   placing-an-upright-domino-beside-neighbours reads. `facing_deg` and
   `arc_length` ride along.
3. **The pitch** (`pitch`: metres or `"auto"`). The physics gives the
   critical gap: the widest face-to-face gap a wave that has run 20 tiles
   still crosses (GapChain, below). Auto picks the pitch at which the line's
   face gap stays as far under it as a one-slot gate stays over it:
   `crit - (p - d) = (2p - d) - crit`, so `p = 2 (crit + d) / 3` - **36.9 mm
   (0.77 h) for a 48 x 24 x 7.5 mm tile**, both margins 18.4 mm. It is kept
   between the hand's floor (below) and 1.0 h (the widest pitch a placed line
   has been seen to cascade at). With `n_tiles` and a path too short for it,
   auto shrinks to fit (`pitch_source` says so); a given pitch is used as is.
   Without `n_tiles` the path (or its first `length` metres) is filled.
4. **Every neighbouring pair**, on the real geometry, so curves count:
   - footprints at least 1 mm apart (`min_inner_gap`, the inner corners on a
     curve), and no slot overlapping a non-neighbour (a path that comes back);
   - where the falling tile's top centre meets the next tile's back face: the
     distance it travels from its own front face (the pair's face gap,
     `max_face_gap`; p - d on a straight line), which must stay
     `topple_margin_min` (5 mm) under the critical gap (`topple_margin`), and
     how far off the next tile's centre it lands (`max_contact_offset`), which
     must stay on its face (12 mm for a 24 mm tile). On a circle that offset is
     about p^2 / (2R): at the auto pitch an arc needs a radius of 60 mm or more
     (55 mm fails at 12.2 mm, 60 mm passes at 10.8);
   - a slow trigger push (1.2 times the speed that just tips a lone tile) runs
     a straight line at that pitch.
5. **The hand.** The placing skill opens the jaws to the tile's width plus
   4 mm a side, closes them across the tile's width with 30 mm blades beside
   it, and refuses a slot when a standing tile's footprint comes within 4 mm of
   that envelope. Its rule is mirrored here and run at every build step against
   the tiles standing then, so no emitted step is refused: on a straight line
   the floor is a 19.8 mm pitch. The placing skill's lead-in point, 30 mm along
   the row on the side away from the nearest standing tile (either side on an
   empty table), must be inside the placing arm's box too.
6. **Reach.** Each slot must be inside an arm's tool-centre box (`bounds`
   `min_m` / `max_m`, x and y) and reach annulus when given, and every tile
   footprint on the `table` rectangle when given. `arms` is a list of
   `{arm_id, base, bounds, reach}` in world; without it
   `robot.describe_workspace` gives one arm (its `bounds`, its base at
   `base_in_left_m` or the origin, the real station's left base frame). Each
   slot goes to the nearest arm that reaches it.
7. **Obstacles** (circles `{center, radius}`, boxes `{center, size | extent,
   yaw_deg | rotation}`, `{min, max}` boxes, or a standing tile's pose): none
   may come within 4 mm of the open hand over any slot.
8. **Gates.** A gate is one or more slots left empty while the line is built
   and filled last, so a tile knocked over by accident takes down only its
   run. Its width is the fewest empty slots whose face gap (tile before to tile
   after, on the real geometry) clears the critical gap by `gate_margin_min`
   (10 mm): one slot at the auto pitch (66.3 mm, 18.4 mm over), two at 24 mm.
   `gates` lists slot indices or points (nearest slot); with none given, one
   is put every `gate_every` (10) tiles, moved up to 3 slots to a spot the arm
   can close, and dropped with a note when there is none. A gate keeps 2
   standing tiles from the ends and from the next gate.
9. **The build order** (`build_steps`, `build_order`). Line slots first,
   farthest from their arm first, so the arm works toward itself; a slot at the
   free end of a standing run before one between two standing tiles (those
   happen only where two runs meet, counted in `problems`); the trigger end
   last on ties. A slot is placed only when no standing tile other than its own
   two path neighbours lies on its arm's way - nearer the base and within
   47 mm (half the linear_4310's 94 mm span) of the line from the base to the
   slot. Then the gates, smallest merged run first (a slip while closing one
   topples only the runs it joins; `at_risk`), each under the same rule with
   the tiles flanking the gate as its neighbours. Each step lists the
   `arm_id`, the standing `neighbours` (slot indices: the placing skill's
   `neighbours`), `gate`, `between_standing` and `at_risk`.

The trigger: slot 0's pose and `push_direction` (its facing).

## Inputs

- `path` (required): above.
- `table_z` (else `robot.describe_workspace`'s `surface_z`, else the path's
  height, else 0); `pitch` ("auto"); `n_tiles` (0: fill the path); `length`
  (0: the whole path); `tile_size` (`{height, width, thickness}`, 48 x 24 x
  7.5 mm).
- `gates` (None: automatic), `gate_every` (10; 0 for none); `arms`, `table`,
  `obstacles`.
- `friction` (domino on domino, 0.25), `topple_margin_min` (0.005 m),
  `gate_margin_min` (0.010 m).

## Outputs

`slots`, `slot_poses`, `first_slot`, `trigger`, `build_steps`, `build_order`,
`gates` (each: `slots`, `gap`, `margin`, `close_step`, `at_risk`, in closing
order), `gate_count`, `gate_width_slots`, `n_tiles`, `pitch`, `pitch_source`,
`face_gap` (p - d), `critical_gap`, `critical_pitch` (crit + d),
`topple_margin`, `max_face_gap`, `gate_margin`, `min_inner_gap`,
`max_turn_deg`, `max_contact_offset`, `path_length`, `used_length`,
`problems`, `detail`. Lengths in metres.

## The physics

Two 2D models of a line toppling without sliding, both CPU (scipy / numpy):

- **GapChain** (domino-fall-simulation, MIT): each tile's rotation about its
  front edge, the leading one free, the fallen ones leaning on the next,
  integrated between impacts (`solve_ivp` with events), with per-gap spacing
  and domino-domino friction; each impact gives the stack its new speeds as
  upstream's `handle_hit` does. The critical gap is found by running a wave 20
  tiles, started with upstream's push scaled to the tile (10 rad/s at h = 1,
  g = 10: 45 rad/s for a 48 mm tile), and asking whether it crosses one wider
  gap (upstream's `create_gap`), by bisection to 0.1 mm. The trigger check
  runs 12 tiles from the slow push instead.
- **StackedLine** (Matthew Chapman's domino-simulation, CC BY 4.0): the
  same no-slip, stacked-wave model with the Newton force solution and
  friction, RK4 at 0.1 ms; used here to check the port against Chapman's
  numbers and to predict the wave's speed.

What they say for a 48 x 24 x 7.5 mm tile, friction 0.25:

| | value |
|---|---|
| critical face gap (GapChain, 20-tile run-up) | 47.8 mm (0.996 h) at every pitch tried (20-48 mm) |
| critical gap vs friction 0 / 0.25 / 0.5 / 0.8 (37 mm pitch) | 47.9 / 47.8 / 47.4 / 46.8 mm |
| widest pitch that topples | 55.3 mm (1.15 h) |
| the merged placing skill's twin: 1.0 h (48 mm) cascaded, 1.5 h (72 mm) did not | agrees: 40.5 mm face gap is under 47.8; 64.5 mm is over (planned with 7.3 mm margin / `too_loose`) |
| slow push (1.2x tipping speed) | stalls in the second tile at 16 mm, runs at 18 mm and wider |
| toppling rate (StackedLine) at 20 / 36.9 / 48 mm | 45.9 / 21.8 / 14.1 tiles/s (0.92 / 0.80 / 0.68 m/s) |

Checks of the port against its sources (unit tests):

- GapChain with upstream's own parameters (h = 1, d = 0.2, friction 0.8,
  spacing 0.5, push 10 rad/s, g = 10) crosses a 0.95 gap and stops at 0.96:
  upstream's note says "Found value: s ~= 0.96".
- StackedLine at 16 / 24 / 32 mm pitch: 58.8 / 37.2 / 26.2 tiles/s; Chapman's
  simulation 59.7 / 37.2 / 26.2, his LMGC90 3D simulation 59.1 / 36.9 / 26.1,
  his experiment with 100-tile lines 64.4 / 42.3 / 28.1 (zmatt.net).

What they do not model: sliding on the table (a wide gap hits the next tile
low, where it may slide rather than tip, so wide gaps look better than they
are - hence the margin and the 1.0 h cap on auto), a tile's lean or twist,
and curves (the curve checks are geometry, not physics).

## The constants, and why

- **`TILE_SIZE` 48 x 24 x 7.5 mm**: the placing skill's, zmatt.net's.
- **`DOMINO_FRICTION = 0.25`**: measured 0.25 +- 0.05 on such tiles (zmatt.net).
- **`PUSH_SPEED_FACTOR = 1.2`**: a slow push, just over the 2.69 rad/s that tips
  a lone tile.
- **`RUN_UP = 20`**: upstream `create_gap`'s `n_wall`.
- **`TOPPLE_MARGIN_MIN = 0.005` m**: one tile 5 mm off its slot (a calibrated
  YAM station camera's RMS is 5.4 mm; the placing skill placed within 2 mm in
  its twin); it admits the 1.0 h pitch seen to cascade.
- **`GATE_MARGIN = 0.010` m**: both flanking tiles 5 mm off, apart.
- **`VALIDATED_PITCH = 1.0` h**: the placing skill's twin (above).
- **`GATE_EVERY = 10`, `GATE_MAX_SLOTS = 4`, `MIN_RUN = 2`, `GATE_SEARCH = 3`**:
  choices.
- **`FINGER_HALF_DEPTH`, `FINGER_BLADE`, `NEIGHBOUR_CLEARANCE`,
  `GRIP_CLEARANCE`, `LEAD_IN`**: the placing skill's (12 / 30 / 4 / 4 / 30 mm).
- **`OVERLAP_CLEARANCE = 0.001` m, `CONTACT_SHARE = 1.0`**: choices (the
  falling top must land on the next face).
- **`CORNER_RADIUS = 0.08` m**: over the 60 mm the contact check needs at the
  auto pitch.
- **`CORRIDOR_HALF_WIDTH = 0.047` m**: half the linear_4310's span; `REACH_TOLERANCE
  = 0.005` m.

## Preconditions and boundaries

- The arm places top-down and straight down, as the placing skill does: its
  low parts at a slot are the two blades (checked at every step). The left
  wrist D405 sits 86 mm off the tool axis and 75 mm above the tool centre
  (the real station's calibration), about 107 mm over the table at a release,
  59 mm over a standing tile's top. The corridor rule is the arm-level
  precaution on top: the arm never has to work beyond a standing tile.
- On a row that points at the arm there is no spot to close a gate without
  reaching past standing tiles (the tile two slots toward the arm is on the
  arm's way), so automatic gates are dropped there (noted in `problems`) and
  a given one is `blocked`.
- Two arms: give `arms` in world with each base; `describe_workspace` alone
  describes one arm. The dual station's per-arm boxes are in each arm's own
  base frame - transform them first.
- No robot motion and no simulation of the placed layout here. **Sim check
  pending: needs a GPU host** (this round ran on CPU only). The physics above
  is 2D and no-slip; the only 3D evidence is the placing skill's twin runs on
  straight lines at 1.0 h and 1.5 h. Curves, gates and build orders have not
  been run on a robot or in a robot simulation.

## Provenance

- `scripts/topple_physics.py` GapChain is adapted from domino-fall-simulation
  (maxencelaunayquerre/domino-fall-simulation, MIT, Copyright (c) 2026
  Maxence Launay Querre, `LICENSE.domino-fall-simulation`): `src/simulation.py`,
  `src/analysis.py` `run_until_failure`, `scripts/spacing_threshold.py`. Changed:
  SI units and g = 9.81; a leading tile that reaches the table, or stalls,
  is a failure (upstream's `run_until_failure` only caught solver errors, so a
  gap the wave never reached read as crossed once it had hit the ground); a
  bisection instead of a fixed scan; explicit solver tolerances.
- `scripts/topple_physics.py` StackedLine is adapted from domino-simulation
  by Matthew Chapman (eigenmatt/domino-simulation, `domino-simulation.py`),
  licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); article
  https://zmatt.net/domino-physics/. Changed: globals became a class, RK4
  only, a stall ends the run, it returns hit times instead of plotting.
- Slots along a path yawed to its tangent, with an overlap check, is the idea
  of Franka-Domino-Project's `pattern.py` (Zhu-Chenyu/Franka-Domino-Project,
  no licence: reference only, reimplemented with no code copied).
- Farthest-from-the-arm first is DominoRobot's field ordering rule
  (alexbaucom17/DominoRobot `FieldPlanner.py`, GPL-3.0: the rule only).
