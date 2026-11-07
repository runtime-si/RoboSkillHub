---
name: checking-spatial-relations
description: Decides from one RGB-D view whether a spatial relation holds between two seen objects - inside, on_top_of, near or blocking - with REFLECT's point-cloud tests (minimum point distance, convex-hull containment, footprint-and-height, image-box occlusion with depth order), and routes holds / does_not_hold / not_visible. Use when a graph must branch on a scene fact such as "is the cube in the box", "is the plug on the board", "is the part near the fixture" or "is the hand blocking the camera's view of the target", with the two objects given as descriptions, masks or world point clouds.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, spatial-relations, scene-graph, rgb-d, point-cloud]
  hub:
    stage: verify
    embodiments: [any]
    tasks: [pick-place, insertion, cable-routing, sorting]
    version: 0.1.0
    provenance:
      code: https://github.com/real-stanford/reflect/blob/c2f9db993e372250321ca74e56ae27519716c934/main/scene_graph.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2306.15724
gap:
  allowed_tools:
    - robot.get_observation
    - sam3.segment_text
  required_inputs:
    relation: string
  produces_outputs:
    holds: bool
    relation: string
    min_distance_m: float
    evidence: string
  exit_conditions:
    holds: Both objects were seen with enough valid depth and the asked relation is among those REFLECT's tests found from the subject to the reference.
    does_not_hold: Both objects were seen with enough valid depth and the asked relation is not among those found; `relations` lists what was found instead.
    not_visible: The subject or the reference was not segmented, or kept fewer than min_points valid depth points; nothing is claimed about the relation.
  canonical_scripts:
    - check_relation: scripts/check_relation.py
  examples:
    - title: Is the cube inside the box, from the overhead camera
      path: examples/canonical_subgraph.json
  streaming: false
---

# checking-spatial-relations

Answer one question of the form *subject relation reference* from a single
RGB-D frame, the way REFLECT builds the edges of its scene graph.

## Inputs

- `relation`: `inside`, `on_top_of`, `near` or `blocking` (aliases: `in`,
  `on`, `beside`, `next_to`, `close_to`, `occluding`, `covering`).
- The subject and the reference, each as one of:
  - `*_points`: a world-frame `PointCloud` (or an `[N, 3]` array), e.g. from
    a perception skill;
  - `*_mask`: a `uint8` mask on the camera's image, back-projected with that
    frame's depth, intrinsics and camera-to-world pose (the math of
    `geometry.mask_to_world_points`);
  - `*_description`: a phrase segmented with `sam3.segment_text` (best mask,
    score at least `score_min`, 0.3).
- `observation` (optional): an `Observation`; fetched with
  `robot.get_observation` when a mask or description needs one. `blocking`
  and left/right need the camera, so they are only found with one.
- `camera_name` (`overhead`; the only camera is used if the name is absent).
- `reference_kind`: `auto` (default), `container` or `surface`. A surface
  (table, board, tray) holds things on top of it, never inside it; `auto`
  calls a reference a surface when its trimmed height is under 20 mm.
- `contact_distance` (0.05 m), `close_distance` (0.10 m), `voxel`
  (0.003 m), `min_points` (30).

World frame: z up (the hub's convention), metres.

## What is tested (REFLECT's `SceneGraph.add_edge`, in its order)

Both clouds are down-sampled on a `voxel` grid; `d` is the smallest distance
from a subject point to the reference cloud.

1. **In contact** when `d < contact_distance`:
   - `inside`: more than 50 % of the subject's points fall in the convex hull
     of the reference's (`is_inside`, `INSIDE_THRESH`). Against a surface
     this is `on_top_of` instead.
   - otherwise `on_top_of`: more than 70 % of the subject's points lie over
     the reference's xy box **and** more than 70 % above its top
     (`ON_TOP_OF_THRESH`).
   - otherwise `under`: the same test the other way round (REFLECT records
     the reverse edge).
2. **Close** when `d < close_distance`: `near`. If no contact relation was
   found, the unit vector between the two boxes' centres names a direction:
   - along the camera's ray (|component| > 0.9): `blocking` when the
     subject's image box covers more than 50 % of the reference's
     (`OCCLUDE_RATIO_THRESH`) and more than 90 % of the subject's depths are
     no deeper than the reference's nearest point (`DEPTH_THRESH`);
   - else `above` / `below` (world z, > 0.9), else `left_of` / `right_of`
     (the camera's x, > 0.8).

`relations` lists everything found from the subject to the reference;
`measures` the fractions, the distance and the point counts behind them;
`evidence` says it in one line. The script raises only on bad arguments.

## Changes from REFLECT, and why

- **Distances rescaled from a kitchen to a tabletop.** REFLECT's
  `IN_CONTACT_DISTANCE` 0.1 m, `CLOSE_DISTANCE` 0.4 m and 0.01 m voxels are
  AI2-THOR room-scale (THOR units are metres). Here: contact 0.05 m, which
  must exceed the step a downward camera sees at a resting contact (the top
  of a 35-40 mm object over its support, its sides hidden) and stay under a
  held object's lift; close 0.10 m (a quarter, like the contact distance);
  voxels 3 mm, so a 30 mm cube keeps ~100 points and a cable is more than one
  voxel wide. The ratios (0.5, 0.7, 0.9, 0.8, 0.5, 0.9) are REFLECT's.
- **z up, and a downward camera.** THOR is y-up and REFLECT took all three
  directions in the camera frame of a near-horizontal agent camera. Here
  up/down is world z, left/right and front/back the camera's x and optical
  axis, and the ray test comes first: for a camera looking down, "along the
  ray" and "above" are nearly the same vector, and the camera-specific
  relation (blocking) is the one a graph cannot get otherwise.
- **Boxes trimmed to the 1st-99th percentile.** REFLECT took the raw
  bounding box of a simulated depth cloud; a real sensor's flying pixels at
  object edges would grow it.
- **Surfaces generalised.** REFLECT renames "inside countertop / stove burner"
  to "on top of" by object name; here any reference under 20 mm tall (or
  `reference_kind: surface`) gets the rename.
- **`near` holds whenever `d < close_distance`**, also when a contact
  relation was found; REFLECT kept only the stronger edge, but a graph asking
  "is it near" about an object inside the box wants yes.
- **Mask bleed removed.** Points within 3 voxels are linked and only pieces
  with at least 20 % of the largest piece's points are kept (REFLECT removed
  radius outliers from containers only). In the twin a SAM3 cube mask beside
  the hand reached 0.27 m tall before this, its edge pixels on the hand.
- REFLECT's per-name exceptions (`BULKY_OBJECTS`, THOR fruit slices standing
  upright in bowls), its merging of clouds over time and its CLIP object
  states are not ported; Open3D is replaced by scipy's KD-tree and Qhull.

## Measured

**YAM station twin, overhead camera (640x480 RGB-D, 59 deg down), SAM3 masks
from the descriptions.** Three scenes: the cube on the table 18 cm from the
box (`start`), 30 mm outside its wall (`beside`), on its floor (`inbox`).
17 of 18 verdicts matched the truth:

| check | start | beside | inbox |
|---|---|---|---|
| cube inside box | does_not_hold ok | does_not_hold ok | holds ok |
| cube near box | does_not_hold ok (d 177 mm) | holds ok (d 25-30 mm) | holds ok |
| cube on_top_of box | does_not_hold ok | does_not_hold ok | does_not_hold ok |
| cube on_top_of table | holds ok | holds ok | **holds, truth does_not_hold** |
| box on_top_of table | holds ok | holds ok | holds ok |
| cube blocking box | does_not_hold ok | does_not_hold ok | does_not_hold ok |

The miss is REFLECT's own: on top of is footprint plus height, so a cube in a
box on a table is also "on top of" the table (its points are over the table
and above it, 38 mm away through the box wall, inside the contact distance).

**Blocking, the hand on the camera ray.** With the closed hand 8 cm in front
of the cube along the ray, SAM3 no longer finds the cube (score 0.01) and all
three questions route `not_visible` (3 scenes x 3, correct). With the hand
over the box (box masks from depth, 27-43 % of the box still visible, the
hand's mask from depth), `box blocking hand` was correctly `does_not_hold`
(9/9), but `hand blocking box` and `hand near box` came out `does_not_hold`
in all 18 runs although the hand hid most of the box and its fingertips were
45-77 mm from it: the smallest distance between the **visible** clouds was
110-162 mm (the part of the box nearest the hand is the part it hides), over
the 0.10 m close distance that gates both tests. REFLECT gates blocking on
proximity the same way; it accumulated clouds over many views in THOR, which
one fixed camera cannot.

**Real station frame** (`cable_frame_20261103-1421.npz`, Orbbec, SAM3 masks,
the 11-05 calibration levelled onto the frame's own table by 1.4 deg; the
table still spans -6..+8 mm, 5th-95th percentile): 7 of 9 matched.

| check | truth (by eye) | verdict |
|---|---|---|
| cable on_top_of board | does_not_hold (mostly on the table) | does_not_hold |
| cable on_top_of table | holds | holds |
| board on_top_of table | holds | holds |
| cable near board | holds | holds (15 mm) |
| cable blocking board | does_not_hold | does_not_hold |
| plug inside board | does_not_hold | does_not_hold |
| plug on_top_of board | holds | **does_not_hold**: SAM3's "white cable connector plug" mask was the whole cable |
| gripper near board | does_not_hold | **holds**: SAM3's "black robot gripper" (score 0.23) masked a ring post on the board |
| gripper on_top_of table | holds | holds (same wrong mask: the post is over the table) |

Both misses are the segmentation's, not the relation tests' (the verdicts are
right for the clouds they were given); check `measures` and the masks when a
verdict surprises. On the real table "on top of table" held through the hull
rule (the levelled table is 22 mm thick with depth warp), not the height test
(only 9 % of the cable's points rose above the table's 99th percentile).

## Boundaries

- One view: a subject hidden by the reference (a cube deep in an opaque box
  seen from the side) is `not_visible`, not `inside`; use
  `verifying-placement` for "it disappeared into the container".
- `inside` needs the reference's visible points to enclose the subject: an
  open box seen from above does (floor + rims), a closed lid does not.
- The relation is about the segmented clouds: a mask that bleeds onto the
  support moves the verdict. Check `measures` when a verdict surprises.
