---
name: picking-flat-pieces-by-suction-cup
description: Picks a flat piece (jigsaw piece, tile, card, thin plate) by its
  top face with a small suction cup - scores cup points on the piece's mask
  from one depth frame by local surface-normal variance (the ARC suction
  baseline, Apache-2.0) at least a cup radius inside the edge, comes down on
  the best one until contact, switches the vacuum on, lifts 5 mm and checks
  the piece's rim rose with it, then optionally carries it and releases. Use
  when pieces sit in tight slots or touch their neighbours so finger pads
  cannot reach beside them. NEEDS HARDWARE the YAM station does not have
  today - an 8-10 mm cup beside the fingers, a vacuum pump and valve that the
  connector switches with robot.set_suction.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [suction, grasping, flat-objects, jigsaw, depth, affordance]
  hub:
    stage: execute
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, pick-place]
    notes: "Needs hardware the YAM station lacks today: an 8-10 mm suction cup on the hand and a vacuum pump + valve switched by the connector (robot.set_suction)."
    provenance:
      authors: [Andy Zeng, Shuran Song, Kuan-Ting Yu, Elliott Donlon, Francois R. Hogan, Maria Bauza, Daolin Ma, Orion Taylor, Melody Liu, Eudald Romo, Nima Fazeli, Ferran Alet, Nikhil Chavan Dafle, Rachel Holladay, Isabella Morona, Prem Qu Nair, Druck Green, Ian Taylor, Weber Liu, Thomas Funkhouser, Alberto Rodriguez]
      code: https://github.com/andyzeng/arc-robot-vision/blob/728139e9ad0e/suction-based-grasping/baseline/predict.m
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/1710.01330
gap:
  requires: {connector: [robot.set_suction, robot.move_cartesian_until_contact]}
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.set_suction
  required_inputs:
    target_mask: Mask
  produces_outputs:
    affordance: float
    rim_rise_m: float
    sealed: bool
    reason: string
  exit_conditions:
    picked: The cup sealed (the connector's vacuum reading, or else the piece's rim rose at least half the 5 mm test lift in the camera) and the piece is held at the approach height - or, with place_point, was carried there and released.
    no_seal: Vacuum on but the piece did not come up (lip over an edge, a gap or a knob neck, a porous face); vacuum off, the hand backed off, nothing is held.
    dropped_in_transport: The connector's vacuum reading lost the seal on the way to place_point (needs a vacuum sensor; without one a drop is not seen until the next look).
    no_valid_cup_point: No mask pixel lies a cup radius plus 1 mm inside the mask, the best affordance is under min_affordance (0.5), the mask does not match the depth frame, or the camera has no depth / intrinsics / pose; nothing moved.
    unreachable: The approach or contact move was refused, or robot.set_suction failed (no vacuum hardware served); the vacuum was not left on.
  canonical_scripts:
    - pick_suction: scripts/pick_suction.py
  examples:
    - title: Pick a jigsaw piece from its mask with an 8 mm cup
      path: examples/canonical_subgraph.json
---

# picking-flat-pieces-by-suction-cup

Jigsaw pieces in a tray, or next to placed neighbours, leave no room for
finger pads: the hub's 15.5 x 3.9 mm pads made 19 direct grips in 100 on
pieces in tight slots. A suction cup needs only the top face.

## Hardware (not on the YAM today)

The YAM station has no vacuum. This skill needs:

- an 8-10 mm bellows cup on a printed holder beside the fingers (a reference
  holder: PAROL6 `vacuum_gripper_holder.STL`, CERN-OHL-S, hardware only);
- a mini pump and a valve on an output the connector switches. The
  connector tool is `robot.set_suction(on, arm_id=0)` →
  `{on, sealed?}`, where `sealed` comes from a vacuum sensor if fitted.

Give the cup's lip centre in the tool frame as `cup_offset_m`. It is
declared in `gap.requires.connector`, so a graph that uses the skill fails
its check on a connector without suction instead of at run time.

## What it does

1. **Cup point.** It projects the depth to camera points, smooths the depth
   within the mask, and takes normals pointing at the camera. The score is
   the 25 x 25 window standard deviation of the normals, as affordance = 1 -
   std / max (upstream `predict.m`). Candidates must be at least a cup
   radius plus 1 mm inside the mask (distance transform); the most interior
   of the near-best scores wins.
2. **Approach and contact.** `go_to_pose` to 40 mm above the point, keeping
   the hand's rotation, then `move_cartesian_until_contact` to 4 mm below
   the measured top.
3. **Seal check.** Vacuum on, then a 5 mm `go_to_pose_cartesian` lift. If
   the connector returns `sealed`, that decides. Otherwise the camera
   judges: the piece's rim (mask pixels more than two cup radii from the
   cup) must rise at least 2.5 mm. Rim pixels that fell by more than the
   lift plus 3 mm are the hand in front of the piece and are skipped; that
   rule fixed a false "picked" in the twin check.
4. **Carry.** Rise to the approach height. With `place_point`, it moves
   there, comes down until contact, switches the vacuum off and lifts away.

## Changes from upstream

Upstream is a MATLAB baseline; this is a numpy / scipy reimplementation
(see the script header):
- normals come from image-space gradients of the organised cloud after a
  7 x 7 masked smoothing, instead of `pcnormals` with 50 neighbours;
- the foreground is the given mask, instead of background subtraction;
- the edge-distance rule is added.

## Measured (standalone MuJoCo check, not the YAM twin)

**Scene (`~/sim-wt/suction/sim_suction.py`, MuJoCo 3, CPU):**
- A 3 x 4 tray of 22 x 22 x 2 mm pieces with **0.3 mm gaps**; the outer
  pieces carry a 7 mm knob outward.
- An overhead depth camera at 0.3 m (640 x 480, f = 600 px), built from
  `mj_multiRay`, with 0.4 mm depth noise. Its segmentation, eroded 1 px,
  stands in for sam3.
- A hand on a mocap weld carrying an 8 mm cup. Contact moves stop at 0.05 N.
- The vacuum is an adhesion actuator that engages only when the lip seals:
  9 rays from the lip ring must all hit the same piece within 0.4 mm of
  height, so a lip over a gap, an edge or a knob leaks.
- **The skill got no vacuum reading** and had to judge the seal from the
  camera. 3 depth-noise seeds per piece.

| method | held after lift | routed `picked` | false `picked` | `no_seal` (correct) |
| --- | --- | --- | --- | --- |
| this skill (ARC affordance + edge rule) | **36/36** | 36 | 0 | 0 |
| same pick, cup point random on the mask | 18/36 | 18 | 0 | 18 |
| parallel gripper, 15.5 x 3.9 mm pads | 0/12 slot positions leave room for the pads (geometry; the hub's real-gripper jigsaw run: 19/100 direct grips) | | | |

- **Point choice:** affordance at the chosen points was 0.887-0.911. They
  sat a full cup radius inside the piece, so the lip never crossed an edge
  or a knob. A random point leaked half the time, and the camera rim check
  caught every leak: the rim rose 0.0 mm, against 4.86-4.89 mm when sealed.
- **No disturbance:** neighbours moved 0.0 mm in every trial.
- **One bug found and fixed here:** before occluding rim pixels were
  skipped, the hand in front of the piece looked like a 32 mm "rise" and a
  failed pick was reported as `picked`.
- **What this does not show:**
  - real cup compliance, leakage on printed board, or flatness of warped
    pieces (the seal model is ideal);
  - the YAM twin (its gripper has no cup);
  - `place_point` transport, `dropped_in_transport` and the vacuum-sensor
    path, which are unit-tested only;
  - a real station.

## Licence

The cup-point scoring is adapted from arc-robot-vision (Apache-2.0,
Copyright (c) 2017 Andy Zeng et al.); the licence is
`LICENSE.arc-robot-vision`. See NOTICE.
