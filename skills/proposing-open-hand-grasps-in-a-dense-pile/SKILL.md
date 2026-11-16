---
name: proposing-open-hand-grasps-in-a-dense-pile
description: Proposes top-down parallel-jaw grasps in a dense heap of
  centimetre-scale parts inside a bin, for a hand that comes down fully open -
  builds a 2 mm heightmap of the bin's inside from one or more posed depth
  frames, and for every cell and jaw direction lets the two open fingertips'
  footprints sink to the highest thing under either and takes what stands
  between them over that level as the grip, with the part's top read from what
  is between the fingers for that jaw direction, lone depth spikes ignored,
  both fingers inside the bin's walls, and optional masks for what may be
  taken and what must be kept clear of. Returns world grasp points, closing
  heading, fingertip and tool heights, ranked by grip depth and height in the
  pile. Use when parts lie heaped and touching in a box so that no single part
  can be segmented or stands clear on a support, before a top-down grasp
  skill, with the bin's box in the reading camera's frame; expect about one
  proposal in two to hold on real printed parts. CPU only; nothing moves.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: grasping
  tags: [grasping, propose, select, clutter, bin-picking, pile, depth, heightmap, cpu]
  hub:
    stage: select
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [pick-place, sorting, bin-picking]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: About 1 grasp in 2 held on real 5-9 cm printed brackets and tube sockets
      (one station), about 1 in 5 in a box-primitive simulation of the same pile. The
      ranking is geometry, not a learned success score.
gap:
  requires: {connector: [robot.describe_gripper, robot.grasp_frame]}
  allowed_tools:
    - robot.get_observation
    - robot.describe_gripper
    - robot.grasp_frame
  required_inputs:
    bin_obb: OrientedBoundingBox
  # Optional inputs: camera, rank (string), allow_mask, avoid_mask (Mask), heading_min,
  # heading_max, heading_step, half_open, pad_width, finger_thickness, min_grip, max_depth, wall,
  # prefer_high, min_range (float), top_n, arm_id (int), frame (bool). Wired from a script node
  # (gap has no type for them): camera_frame (CameraFrame), camera_frames (list of CameraFrame),
  # headings_deg (list of float), roi ({x_min, x_max, y_min, y_max}).
  produces_outputs:
    grasp_pose: Se3Pose
    position: Vec3
    close_heading_deg: float
    jaw_width_m: float
    top_z: float
    tip_z: float
    tool_z: float
    grip_depth_m: float
    score: float
    n_grasps: int
    seen_fraction: float
  exit_conditions:
    proposed: At least one place and jaw direction has something standing min_grip (15 mm) or more over the highest thing under both open fingertips, both fingers inside the walls and mostly seen; `grasps` are best first, at least 3 cm apart, and the best one's position (the tool centre), close_heading_deg, top_z, tip_z, tool_z and grasp_pose are flattened. Nothing moved. A proposal is where the hand fits, not a promise it will hold.
    nothing_fits: At least 10 % of the bin's inside was seen and no place passes - a level pile, parts wider than the open hand, everything against a wall, or only what the masks rule out. Look again from another side, or stir or spread the pile.
    not_seen: No posed depth camera, or under 10 % of the bin's inside has depth (the view misses the bin, the hand hides it, or bin_obb is in another camera's frame); `detail` says which.
  canonical_scripts:
    - propose_pile_grasps: scripts/propose_pile_grasps.py
  examples:
    - title: Propose a grasp in the parts bin from the wrist view
      path: examples/canonical_subgraph.json
---

# proposing-open-hand-grasps-in-a-dense-pile

In a heap, nothing "stands on a support" and nothing can be told apart as one
part. What can still be asked of the depth is simpler: if the open hand came
straight down here, turned this way, how far would the fingertips get before
either touched something, and is there anything between them above that
level? Where the answer is "15 mm or more", there is something to close on.

## How

1. **Heightmap.** Each frame's depth (nothing nearer than `min_range`) is
   lifted to world points and binned at 2 mm in the bin's own frame, starting
   12 mm inside its outer walls; the highest point per cell is kept, with the
   masks' values at that point. Several frames (`camera_frames`) go into one
   map. Gaps with at least 3 seen neighbours take their mean; the rest stay
   unknown.
2. **Two fixes that matter.**
   - *The part's top is what stands between the fingers.* `top` is the
     highest solid cell in a 24 x 12 mm box at the hand's centre, laid along
     this jaw direction. It is the same top the grip depth is measured from.
     Reading the top from the centre cell alone gave "depths" of -6 cm when
     the part's top was 1 cm to the side, and a hand that set off too low.
   - *Spikes are not parts.* `top` is read from the map opened by 5 cells.
     Before this, a single reading 25 cm up, 7 cm over the pile, was offered
     as the best grasp. The fingers' own floor is still read from the
     unopened map, so they keep clear of a spike that is really there.
3. **Fit.** For each jaw direction: `floor` is the highest cell under either
   fingertip footprint (`finger_thickness` x `pad_width`, just outside
   `half_open`), unseen cells counting as the bin's support height;
   `grip = top - floor`. A cell passes when `grip >= min_grip`, both
   footprints and the middle box are at least `wall` inside the map, over
   60 % of both footprints was seen, over 60 % of the 7 x 7 cells round it
   are allowed and stand over the bin's floor (12 mm), and under 15 % of the
   15 x 15 cells round it are to be avoided.
4. **Depth and rank.** The fingertips stop 2 mm over `floor`, at most
   `max_depth` (35 mm) under `top`. `rank="mixed"` scores
   `min(grip, 3 cm) + prefer_high * (top - rim)`; `rank="depth"` orders by
   how far under the top the fingertips end, in 2 mm steps, then by height.
   Proposals are thinned to 3 cm apart, `top_n` kept.
5. **Pose.** `position` is the tool centre: (x, y, `tool_z`), with
   `tool_z = tip_z + reach` (the gripper's stated fingertip reach, else
   15 mm). `grasp_pose` adds `robot.grasp_frame`'s rotation for the best one.

## How this differs from proposing-parallel-grasps-in-tight-clusters

Both slide fingertip footprints over a world heightmap, and both descend
from the same ancestor (arc-robot-vision's finger-fit baseline). They answer
different cases.

| | tight-clusters | this skill |
| --- | --- | --- |
| scene | small flat parts (2-12 mm) lying apart or packed on a table | a heap of 5-9 cm parts in a bin, touching and overlapping |
| what a part is | a mask, or a blob standing out of a fitted support plane | nothing is segmented; every cell is a candidate centre |
| reference level | the support plane; insertion below the part's median top | no support: the highest thing under the two fingertips |
| opening | grows from the part's width; the smallest that fits (`set_grip`) | always the full opening the hand comes down with |
| neighbours | no other part between the jaws; finger body against tall neighbours; edge contact, not a corner | not checked: whatever is between the fingers is "the part" |
| unseen cells | obstacles under a finger (default) | empty, but 60 % of each footprint must be seen |
| container | none (upstream's wall grasps not ported) | the bin: map inside its walls, fingers kept off them |
| search | up to 9 centres per part, 8 headings + the part's axes | every 2 mm cell, every heading asked, by morphology |
| says when it cannot | `too_crowded`, per part why | `nothing_fits` |
| measured | synthetic gap sweeps; nothing executed | executed on a real arm; hold rate below |

Run on a heap, tight-clusters has no support plane to fit and one blob to
search, so it was not the tool for this; it was not actually run on these
piles, and that claim is from its design, not a measurement. Run on a jigsaw
tray, this skill would propose full-open grasps that land on the neighbours.

**Could it be a mode of that bundle?** Partly. The heightmap, camera reading
and pose output are shared in spirit, and a `mode="pile"` there (no support
plane, full opening, dense search, bin walls) would work. It would also
double that script and give one bundle two contracts (per-part statuses in
one mode, none in the other; `too_crowded` meaningless in a pile). Kept
separate here; merging is a maintainer's call.

## Using a proposal

- **Come down fully open** (`jaw_width_m` is the full span). The fit was
  checked for that opening only.
- **`tip_z` is where the fingertips have room to, 2 mm over what is under
  them.** The source task sent the hand 8 mm lower than that ("the pile
  gives") and no lower than the arm's floor. That is the grasp skill's
  choice.
- **Try several, then look again.** The source tried up to 3 proposals per
  look and never the same spot twice in a run: a spot that failed once was
  offered again look after look.
- **A blank look is not an empty bin.** A part just dropped can hide the
  pile; the source looked again twice before concluding.
- **Check the hold after the lift** (`verifying-grasps`, or the jaw gap):
  parts pinched by an edge came up and slipped out at the next motion.
- **Restrict `headings_deg` to what the wrist can turn to.** A proposal the
  arm cannot take costs a try.

## Gotchas

- **One slanted look sees about half the pile** (34-48 % measured), and what
  is unseen counts as empty, so a grasp can look deeper than it is. Two looks
  merged were tried in simulation and did not help there (below).
- **It does not know parts.** Two parts side by side narrower than the
  opening are "one thing between the fingers". A wide flat part offers
  nothing.
- **Ties are common and are not broken towards a good grasp.** Every centre
  and heading with the same grip scores the same: the part may sit up to
  27 mm off the hand's centre along the jaw (anywhere the 24 mm middle box
  still touches it), and the heading is the first in `headings_deg` whose
  fingers clear the part, not the one square to it. The closing hand then
  has to pull the part or itself over. This is the method as it ran; a
  centring term would be the obvious next change and has not been tried.
- **Kinds of part are told apart only by the masks you pass,** and only
  round the hand's centre: `avoid_mask` keeps the centre about 1 cm clear of
  an avoided part, which does not stop that part being between the fingers
  beside the centre. The source used a colour rule (grey printed parts
  against black extrusion).
- **Parts under 12 mm over the bin's floor are never proposed.**
- **Fingertip sizes are estimates** (see Constants).

## Constants, and what they were measured on

One YAM station (i2rt YAM, linear_4310 hand, wrist D405 about 0.3 m over the
counter), a 449 x 262 x 154 mm cardboard box holding grey 3D-printed brackets
and tube sockets (5-9 cm) among black aluminium extrusion offcuts,
2026-11-14 and -15.

- `HALF_OPEN = 47 mm`: that hand's 94 mm span; read from
  `robot.describe_gripper` when it states `max_grasp_width_m`.
- `PAD_WIDTH = 28 mm`, `FINGER_THICKNESS = 14 mm`: the footprint this method
  used on that hand, generous; not caliper measurements (the hub's
  tight-clusters skill estimates the same tips at 20 x 10 mm).
- `MIN_GRIP = 15 mm`, `MAX_DEPTH = 35 mm`, `WALL = 20 mm` (the task ran
  30 mm), `MIDDLE = 24 x 12 mm`, `SPIKE_OPEN = 5 cells`, `FLOOR_CLEAR =
  12 mm`, `ALLOW_MIN / AVOID_MAX = 0.6 / 0.15`, `KNOWN_MIN = 0.6`,
  `APART = 3 cm`: chosen on that pile over two days of real runs, none swept.
- `TIP_REACH = 15 mm`: the source station's grasp site; replaced by the
  gripper's stated reach.
- `MIN_SEEN = 10 %`: chosen here, on no robot.
- `rank`: "mixed" against "depth" was compared in simulation only: 12 wins
  to 4 on 12 paired piles (p 0.04), then 17 of 32 against 17 of 32 on 8
  held-out piles (p 0.66). No difference shown; "mixed" stayed the default.

## Evidence

**Real robot** (one station, 2026-11-15, grasps counted as held when the jaws
still read open on something after the lift): 6 of 12, 5 of 10, 4 of 11, 3 of
5 and 4 of 7 in the five runs whose notes give both numbers: 22 of 45, about
one in two. Some held parts were then lost in the carry (2 in one run, 1 in
another), so "held" overstates delivered parts. Before this finder, a
proposer that asked for a part standing proud and centred found almost
nothing on the same real pile.

**Simulation** (a MuJoCo pile of box-primitive stand-ins for the parts):
about one grasp in five held. The simulated parts grasp unlike the real ones,
so the simulation was not used to tune this, and its numbers should not be
read as this skill's.

Not measured: any other part, bin, hand or camera; hold rate against
`min_grip` or any other constant; the masks' effect. Maturity is
`community`.

## Required end states

- `proposed`: `grasps` and the best one's fields.
- `nothing_fits`: the bin was seen, nothing passes.
- `not_seen`: too little depth inside the bin.
