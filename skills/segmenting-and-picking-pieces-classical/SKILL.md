---
name: segmenting-and-picking-pieces-classical
description: Finds the loose jigsaw pieces lying on a staging mat in one top-down
  RGB(-D) frame without a learned model - removes the mat by its colour (estimated
  from the frame, shadow-tolerant, or a fixed HSV band) and by height when the
  depth is fine, sorts the blobs by area and shape into pieces, touching or
  stacked pieces and foreign objects, splits touching pieces with a
  distance-transform watershed that follows the dark cut line, and gives every
  piece its mask, a pick point at the centre of its largest inscribed circle
  lifted to the world, its side angle, area, touching neighbours, a two-finger
  grip check and a face-up / face-down cue. Routes segmented, touching_pieces
  or nothing_found. Use before picking a jigsaw piece from a staging area (and
  before localizing it in the box picture), on CPU, in a fraction of a second;
  sam3 is the learned alternative when a GPU is available.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [jigsaw, puzzle, segmentation, watershed, distance-transform, grasp-point, classical]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/JPStrydom/Jigsaw-Puzzle-Building-Robot/blob/be867bda743740fac9d843f606b9211e663777f7/code/software/Functions_1.py
      upstream_license: MIT
gap:
  allowed_tools:
    - geometry.pixel_to_world_point
    - geometry.mask_to_world_points
  required_inputs:
    observation: Observation
  # Optional inputs (wired from a script node; gap has no open-dict type here): camera (a
  # CameraFrame instead of the observation), camera_name, region (mask), background_hsv
  # ({lower, upper}), piece_size_m, piece_area_px, table_z, back_rgb, finger_radius_m,
  # finger_margin_m.
  produces_outputs:
    n_pieces: int
    n_touching: int
    n_pickable: int
    piece_mask: Mask
    pick_point: Vec3
    angle_deg: float
  exit_conditions:
    segmented: At least one piece was found and none touches anything; `pieces` lists them, piece_mask / pick_point are the best pickable one (clearest of its surroundings, then the widest inscribed circle).
    touching_pieces: Pieces were found but some touch or lie on one another (split by the watershed or by height; their records say touching, neighbours, layer), or a blob did not split into piece-sized parts (`clusters`, with the source's drag point and direction); piece_mask / pick_point still name the best pickable piece when n_pickable > 0. Singulate the rest, e.g. singulating-by-pushing toward `free_spot`.
    nothing_found: No piece-sized blob on the mat (empty, or only foreign objects and specks).
  canonical_scripts:
    - segment_pieces: scripts/segment_pieces.py
  examples:
    - title: Segment the staging area and name the next piece to pick
      path: examples/canonical_subgraph.json
---

# segmenting-and-picking-pieces-classical

Loose jigsaw pieces on a plain mat are the easy case of segmentation: the mat
is one colour, every piece is about the same size, and a piece is flat and
2 mm thick. This skill uses exactly that, with no model and no GPU: take the
mat away, sort what is left by size and shape, cut apart what touches, and pick
each piece where it is widest.

It is a port of the perception half of JP Strydom's CNC jigsaw robot
([Jigsaw-Puzzle-Building-Robot](https://github.com/JPStrydom/Jigsaw-Puzzle-Building-Robot),
MIT), with the source's hard-coded pixel calibration replaced by the piece size,
and splitting, picking, lifting and grip checks added for a two-finger arm.

## The pipeline

1. **Mat removal** (source: `Clear_Background`). By default the mat's colour is
   estimated from the frame: the most populated bin of rg chromaticity and
   brightness, then every pixel within `CHROMA_TOL` (0.05) of that chromaticity
   and between 35 % and 115 % of the mat's brightness *nearby* (a blur of the
   lit mat pixels, so a lighting gradient and drop shadows stay mat) is mat. A
   mat darker than 45 (black felt) is separated by brightness alone. Or pass
   `background_hsv` (`{"lower": [h, s, v], "upper": [h, s, v]}`, OpenCV HSV):
   the source's fixed band, e.g. its `[160, 175, 125]..[175, 255, 255]` for a
   magenta mat. **With fine depth** (the mat's height noise under 0.6 mm, a
   wrist D405) every pixel more than 1 mm above the table is a piece whatever
   its colour: grey backs on a grey mat, white art on a white mat, shadows.
2. **Binarise** (source: `Binarise`): 3 x 3 Gaussian, then open / close by 6 %
   of a piece and fill holes (print in the mat's colour leaves holes).
3. **Sort the blobs** (source: `Detect_Pieces`). The piece area is
   `piece_size_m` squared through the depth (or `piece_area_px`, or the blob
   size most foreground pixels belong to), refined to the median single piece
   when three or more are seen. A blob of 0.55-1.5 pieces whose rotated box is
   at most 1.75 long is one piece; up to 3 x 1.5 pieces and 3.6 long it is
   touching or overlapping pieces; anything under 1/15 of a piece is noise;
   the rest is foreign (`fragment` or `too_large`: the arm, a tool, a heap).
   Blobs reaching the outer 1 % of the frame (or the `region`'s edge) are
   `clipped`.
4. **Stacks** (new): with fine depth, a blob of which 15-90 % is over 3 mm
   high is a piece lying on another: the high part is the `top` piece, the
   rest the `covered` one (never pickable).
5. **Split touching pieces** (new; the source only moved such blobs aside).
   Seeds: the blob's tallest distance-transform point, then the tallest at
   least 0.65 piece from every seed, as many as the area says (rounded down
   and up; whichever split - or the blob whole, if it is not too long - has
   its parts nearest the piece size wins). A priority-flood watershed on the
   inverted distance transform plus twice the dark cut line (a black-hat about
   a twelfth of a piece wide) grows the seeds. A piece-sized blob too long
   for one piece must split, or it stays a cluster.
6. **Per piece**: the **pick point** is the centre of the largest inscribed
   circle (distance-transform maximum; the source picked overlap blobs at
   `pointPolygonTest` distance >= its gripper radius, the hub's
   localizing-a-piece-in-a-reference-image anchors there too). The **side
   angle** is the mean of the largest group of Hough lines on the outline,
   modulo 90 degrees (source: `Classify_Pieces`), counter-clockwise as
   displayed. **Touching**: anything within 1 mm, with the neighbours' ids and
   the gap. **Grip**: fingertips (`finger_radius_m` 4 mm, `finger_margin_m`
   2 mm, both assumed) closing across the pick point along either side
   direction must land clear of everything else; the narrower clear axis is
   kept. **Face**: a flat face (robust grey spread under 8) is `down` when its
   colour matches `back_rgb`, `up` when it does not, `unknown` without
   `back_rgb`; a textured face is `up`.
7. **World** (hub tools): `geometry.pixel_to_world_point` lifts the pick
   point (falling back to the piece's plane when that pixel has no depth),
   `geometry.mask_to_world_points` gives the piece's surface height; the yaw,
   grip yaw and area in m^2 come from the outline on that plane. The table
   height is the commonest height in the frame (or `table_z`).
8. **Unsplit blobs** (source: `Separate_Overlap`): a Huber line through the
   blob, walked for points at least a gripper radius inside (0.22 piece,
   shrinking to 0.11); the end farther from the centroid is the `drag` point
   and the direction away from the centroid the push. **`free_spot`** is the
   mat point farthest from everything, if at least 0.8 piece clear (the
   source scanned a grid for an empty square of that size).

## Inputs

- `observation` (or `camera`, a CameraFrame; `camera_name` picks one of the
  observation's, default the first): the camera must look roughly straight
  down on the staging area. Depth and pose are optional (without them: pixels
  only).
- `region`: mask of where pieces may lie (e.g. the staging area only, not the
  board).
- `piece_size_m`: nominal piece side (e.g. 0.0225 for the 550-piece target),
  or `piece_area_px`. Give one: the median blob is a fragile scale.
- `background_hsv`, `table_z`, `back_rgb` (the colour of the pieces' backs),
  `finger_radius_m`, `finger_margin_m`.

## Outputs

- `pieces`: one record per piece - `piece` (`"p3"`), `index`, `mask_crop` /
  `rgb_crop` / `bbox_px` (crop and mask for localizing-a-piece-in-a-reference-image's
  `piece_rgb` / `piece_mask`), `area_px`, `area_m2`, `pick_pixel`,
  `centroid_pixel`, `inradius_px` / `_m`, `position` (the world pick point,
  what choosing-next-piece-and-slot reads), `height_m`, `angle_deg` (image,
  0-90), `yaw` (world, 0-pi/2), `grip` (`axis_deg`, `yaw`, `width_m`,
  `opening_m`, `clear`), `touching`, `neighbours`, `gap_px` / `_m`, `split`,
  `cluster`, `clipped`, `stacked`, `layer` (`table` / `top` / `covered`),
  `face`, `texture_std`, `pickable` (grip clear, inscribed radius at least
  0.11 piece, not clipped, not covered, not an unsplit stack).
- The best piece: `piece_index`, `piece_mask` (full frame, uint8 0/255: wire it
  with the frame into localizing-a-piece-in-a-reference-image), `pick_point`,
  `pick_pixel`, `angle_deg`. None / empty when no piece was found.
- `n_pieces`, `n_touching`, `n_pickable`, `n_clusters` (unsplit), `clusters`
  (`area_px`, `n_estimated`, `n_split`, `resolved`, `pieces`, `drag`),
  `foreign`, `labels` (pieces 1..n), `free_spot`, `piece_area_px`,
  `px_per_m`, `table_z`, `background` (the mat model), `detail`.

## Measured

**Synthetic staging frames** (`tests/bundles/_jigsaw_staging_eval.py`, held-out
seeds 5000-5059 - the constants were tuned on seeds 1000-1035). 640 x 480 top-down
RGB-D; each scene has 6 isolated pieces, 2 touching pairs (moved together
until their outlines meet), 1 still-interlocked ("flush") pair and 1 stacked
pair; classic and wavy random cuts from `jigsaw_cut.py`, printed with the
procedural picture (textures, plaid, a flat white band) or face down (15 %);
mats green, magenta, blue, grey, black, white in turn; a lighting gradient,
drop shadows, dark cut edges, blur, noise; depth noise 0.3 mm (D405-like).
Found = matched with IoU >= 0.5; split = both pieces of a pair IoU >= 0.7.

| px per piece | isolated: found / IoU | touching: found / IoU | touching pairs split | interlocked pairs split | stacked top found / covered not pickable | side angle, median |
|---|---|---|---|---|---|---|
| 24 | 100 % / 0.90 | 98.8 % / 0.88 | 97.5 % | 58 % | 100 % / 100 % | 7.4 deg |
| 32 | 100 % / 0.93 | 99.2 % / 0.92 | 97.5 % | 55 % | 100 % / 100 % | 3.1 deg |
| 48 | 100 % / 0.96 | 99.2 % / 0.94 | 97.5 % | 65 % | 100 % / 100 % | 1.5 deg |
| 64 | 99.7 % / 0.96 | 99.6 % / 0.95 | 98.3 % | 68 % | 100 % / 100 % | 1.2 deg |

- **Pick point inside the piece**: 100 % of found isolated pieces at every
  size, 99.6-100 % of touching ones, 90-95 % of interlocked ones; at least
  0.1 piece inside: 99.7-100 % isolated. On the same 360 isolated pieces at
  48 px the incenter has a median clearance of 0.37 piece against 0.29 for the
  centroid, and clears the 0.22-piece gripper radius in 99 % against 87 %.
- Touching flag right: 99-100 %. Isolated pieces `pickable`: 99.4-99.7 %.
  Stacked top pieces `pickable` 54-92 % (a fingertip lands on a neighbour).
- Face (with `back_rgb`): 100 % of face-up and 92-100 % of face-down pieces.
- 0.02-0.07 extra pieces per frame. 0.10 s per 640 x 480 frame on one core
  (median, 0.15 s max).
- **Without fine depth** (48 px): with 3 mm depth noise (overhead-like; the
  height map is then not used) isolated pieces are found 91 %, touching pairs
  split 78 %, stacks are not split (their top is found 47 %, but a stack's
  median height still marks it `stacked` and not pickable 71 % of the time);
  with no depth at all 88 %, 76 % and stacks unseen. Per mat (isolated IoU,
  misses as 0, no depth): green 0.91, magenta 0.84, blue 0.85, grey 0.85,
  black 0.80, white 0.48 - grey backs and white art vanish into a white mat.

**Real frames**: no jigsaw piece frames exist from our station yet
(`~/yam_gap/jigsaw` holds only the reference art). On the source's 8 webcam
frames (`code/software/images/*.png`, 1280 x 960, 12-14 pieces each on a
magenta mat, no depth), checked by eye: all 90 isolated pieces are segmented
with the pick point inside, in both the estimated-mat mode (0.16-0.32 s per
frame) and the source's HSV band (0.05-0.18 s); in the estimated-mat mode two
of them are not `pickable` because a fingertip spot falls on a one-pixel line
artifact that crosses the frame.
The two overlapping pairs (frames 2 and 5) each give one piece and an
unsplit remainder with a drag point; of the two heaps of three in frame 7 one
gives one piece and a remainder, the other two pieces, one of whose masks
swallows the covered middle piece. Every frame with a heap routes
`touching_pieces`.

## Against sam3

Not run side by side (sam3 needs a GPU; this skill was built and measured on
CPU only). Qualitatively:

- **Where this skill is better**: no GPU, no weights, 0.1-0.3 s on a CPU; it
  knows what a jigsaw piece is - size, flatness, thickness - so it counts
  touching pieces by area, splits them, flags stacks by height, rejects
  foreign objects by size, and returns a gripper-ready pick point, angle and
  grip check; deterministic, every threshold explainable.
- **Where sam3 is better**: any background (wood grain, a cluttered table,
  the half-built puzzle), pieces printed in the mat's colour, strong
  specular glare, and pieces still interlocked (it sees the cut line
  semantically). A text prompt ("jigsaw puzzle piece") also ignores the
  arm and tools without size rules.
- **Combine**: sam3's masks can replace steps 1-5; steps 6-8 (pick point,
  angle, grip, touching flags, world lifting) apply to any mask.

## Limits

- **A plain mat is assumed** (or fine depth). Pieces printed in the mat's
  colour break up without depth (2 of 4 synthetic magenta scenes tried with the
  HSV band lost a piece); pick a mat colour unlike the art and the backs.
- **The overhead Orbbec at 0.9 m is too coarse**: a 2.25 cm piece is about
  10 px there, below the 24 px measured here (where the side angle is already
  7 degrees off). Use the wrist D405 above the staging area (60 px at 15 cm).
- **Still-interlocked pieces** split only 55-68 % of the time; the cut-line
  weight was tuned on synthetic cut edges that are 35 % darker for one pixel.
- **Stacks need fine depth.** Without it a stack is one blob (`stacked` if its
  median height shows it) and the covered piece's visible part may be
  reported as a piece.
- **The fingertip size is assumed** (4 mm half-width, 2 mm margin), not
  measured on the YAM's tapered tips; the grip check is a 2-D clearance test,
  not a reachability check.
- **Face down** is a texture-and-colour cue: a flat-art face the colour of
  the backs reads `down`; give `back_rgb` measured under the station's light.
- **Top-down camera**: the area, angle and grip are measured in the image
  and lifted to a horizontal plane; a tilted camera foreshortens them.

## The constants, and why

Each constant in `scripts/segment_pieces.py` carries its source. The source's
values are pixel counts for ~136 px pieces (18 500 px^2) and are kept as
fractions of the piece: piece area 0.55-1.5 (source 0.81-1.19 of its one
puzzle, widened for random cuts), overlap up to 3 x the largest piece, noise
under 1/15, aspect 1.75 / 3.6, gripper radius 0.22 then 0.11 piece, open area
0.8 piece, Hough votes 0.22 piece, angle groups 0.35 rad. The new ones
(seed separation 0.65, cut-line weight 2, mat tolerances, stack heights, face
spread 8) were tuned on synthetic seeds 1000-1035; their docstrings give the
alternatives measured.
