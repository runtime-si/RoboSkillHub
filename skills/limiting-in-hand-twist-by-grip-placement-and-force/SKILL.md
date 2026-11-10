---
name: limiting-in-hand-twist-by-grip-placement-and-force
description: Ranks candidate grips on a part by the grip force each needs so
  the part neither slides nor twists about the jaws' closing axis - the
  part's weight and its torque about each grip's closing axis (from its mass
  and centre of mass), shared by the two pads and checked against the
  ellipsoidal friction limit of the Slip-Aware Parallel Gripper work (mu_c x
  fn for force, mu_c x fn x r for torque, hold margin 2) - and picks the grip
  needing least. Use when a part will be gripped off its centre of mass (a
  long bar, a domino by its end, a jigsaw piece away from its centroid) and
  the grip point is still free to choose, or to tell whether a grip can hold
  before lifting.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [grasp, torsion, slip, friction, in-hand, planning]
  hub:
    stage: propose
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/Gabrieleenx/Slip-Aware-Object-Manipulation-with-Parallel-Grippers/blob/639fe4a4bcf8/src/controlled_slippage.py
      upstream_license: MIT
gap:
  allowed_tools: []
  required_inputs:
    center_of_mass: Vec3
    candidates: GraspCandidates
  produces_outputs:
    grip_index: int
    grip_position: Vec3
    required_force: float
    available_force: float
    torque: float
    lever: float
    ranked: string
  exit_conditions:
    grip_selected: The best candidate needs at most grip_force per pad (with the hold margin); it is the one needing least, normally the one nearest the centre of mass along the part.
    no_grip_holds: Every candidate needs more than grip_force because of the torque, but a grip through the centre of mass would hold the weight; grip nearer the centre of mass or squeeze harder.
    needs_suction: Even the weight alone, gripped through the centre of mass, needs more than grip_force per pad.
  canonical_scripts:
    - limit_twist: scripts/limit_twist.py
  examples:
    - title: Rank two grips on a bar
      path: examples/canonical_subgraph.json
---

# limiting-in-hand-twist-by-grip-placement-and-force

A parallel gripper holds a part against its weight by friction in the pads'
plane, and against the weight's torque about the closing axis by the pads'
torsional friction, which is small: a pad only a few millimetres across
resists a torque of about mu x grip force x a few millimetres. Gripped 40 mm
off its centre of mass, a 100 g bar puts 39 mN.m on the pads, which is more
than plastic pads at 9.5 N can take. The bar then pivots in the jaws, which
is the failure several hub skills met in simulation.

## The model

For each candidate grip (a position and the jaws' closing axis):

- The **weight** m g, with its component along the closing axis removed (the
  squeeze bears that part), is shared by the two pads: f_t = |m g_perp| / 2.
- The **torque** about the closing axis is tau = |((COM - grip) x m g) . axis|,
  also shared: tau / 2 per pad.
- The **limit**: the Slip-Aware HoldMode's ellipsoidal limit surface.
  - gamma_ft = f_t / |(f_t, tau/r)| and gamma_tau = (tau/r) / |(f_t, tau/r)|.
  - A pad needs fn = max(f_t / (mu_c gamma_ft), tau / (mu_c r gamma_tau)).
  - The force asked for is `hold_margin` (2) x max(fn, `min_fn` = 2 N).

Candidates are ranked by that force. The first is chosen if it is within
`grip_force`.

## Inputs

- `center_of_mass` (Vec3): from the part's mask points (e.g. the centroid of
  geometry.mask_to_points), or its model.
- `candidates` (GraspCandidates, `{poses: [{position, rotation}]}`): the
  closing axis is each grasp frame's y axis (the station's TCP convention).
  A plain list of `{position: Vec3, close_axis: Vec3}` is accepted too.
- `mass` [kg], or `area` [m^2] x `thickness` [m] x `density` [kg/m^3].
- `mu_c` (0.36), `contact_radius` (0.0067 m), `hold_margin` (2), `min_fn`
  (2 N), `grip_force` (9.5 N per pad).

## Outputs

`grip_index` and `grip_position` of the best grip, its `required_force`
(per pad, with the margin), `torque`, `lever`, `available_force`, and
`ranked` (JSON: every candidate with its torque, lever, force and whether it holds).

## The constants, and why

- **`MU_C = 0.36`, `CONTACT_RADIUS = 0.0067` m**: the source's plastic-pad
  tests (mu_c 0.355 / 0.365, r 7.7 / 5.8 mm). Their wood and cardboard tests
  read 0.40-0.51. A 15 x 4 mm rubber patch averages 4.0 mm of radius
  (SIM_GRIP.md). Calibrate both on the robot (see below).
- **`HOLD_MARGIN = 2`, `MIN_FN = 2` N**: the source's `HoldMode(hold_margin=2,
  min_fn=2)`.
- **`GRIP_FORCE = 9.5` N**: what a YAM station twin's linear_4310 presses
  each pad with on a 30 mm part (SIM_GRIP.md). A real station's grip force
  depends on its gripper torque cap; measure it and pass it.

## Against the station twin

With these defaults and 9.5 N, a grip holds up to about 23 mN.m of gravity
torque: 46 mN.m at the limit, halved by the margin. In the twin's
off-centre hold test (SIM_GRIP.md) the bar's twist under 6-78 mN.m was 0.7-4
deg and did not drift. The twin's decomposed blade hulls hold more torque
than a single patch, so on the twin this skill is conservative. Nothing here
ran on a real YAM. To calibrate mu_c x r there, turn the wrist slowly with a
held test bar until detecting-in-hand-slip-and-twist-from-the-wrist-camera
reports a twist.

## What changed from the source, and why

The source evaluates HoldMode online from two fingertip force-torque sensors
and adds a slip-velocity term. Here the load is predicted before the pick
from the part's mass and centre of mass, so no sensor is needed. Its
rotational / linear slipping controllers and friction estimation are not
ported.
