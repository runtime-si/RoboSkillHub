# Catalog

Every bundle in this library, generated from the `SKILL.md` files by
`roboskillhub catalog`; a test fails when this page is out of date. The
overview has one line per bundle, and each name leads to its section: the
full description a planner reads, what goes in and comes out, what each exit
means, and what it calls.

16 tools, 82 skills, 2 policies, 1 recipe and 1 paper.

Under robots, `any` means the bundle reads the robot through `describe_*`
tools; *tuned on* names the robot its constants were measured on; **sim only**
means it reads simulator ground truth (`sim.*`) and never runs on a real robot.

## Tools

| Tool | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`colour`](#colour) | perceive | What colour a segmented part is. | any, *tuned on* yam-station-orbbec-overhead |
| [`curve`](#curve) | perceive | Ordered centrelines of deformable linear objects. | any |
| [`depth`](#depth) | perceive | Robbyant LingBot-Depth v0.5 run locally on a GPU. | any |
| [`gemini-er`](#gemini-er) | perceive | Open-vocabulary 2D detection and pointing via the Gemini Robotics-ER API. | any |
| [`geometry`](#geometry) | perceive | Pure-math 3D geometry toolbox. | any, *tuned on* panda |
| [`grounding-dino`](#grounding-dino) | perceive | Grounding DINO zero-shot object detection — natural-language queries to labeled 2D bounding boxes with confidence scores. | any |
| [`locate-anything`](#locate-anything) | perceive | NVIDIA LocateAnything-3B (non-commercial licence) run locally on a GPU. | any |
| [`molmo`](#molmo) | perceive | Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). | any |
| [`pose6d`](#pose6d) | perceive | 6D pose of a known rigid part from depth. | any |
| [`sam3`](#sam3) | perceive | Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker… | any |
| [`track`](#track) | perceive | Google DeepMind TAPNext++ run locally on a GPU. | any |
| [`vlm`](#vlm) | perceive | Free-form and yes/no visual question answering against a hosted vision-language model (OpenRouter API by default… | any |
| [`curobo`](#curobo) | plan | NVIDIA cuRobo motion planning. | any, *tuned on* panda |
| [`molmo-motion`](#molmo-motion) | propose | AllenAI MolmoMotion-4B run locally on a GPU. | any |
| [`robobrain`](#robobrain) | propose | BAAI RoboBrain 2.5-4B run locally on a GPU. | any |
| [`predicates`](#predicates) | judge | Geometric yes/no judgments over poses and boxes the graph already holds. | any |

## Skills

| Skill | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`detecting-contact-from-joint-effort`](#detecting-contact-from-joint-effort) | perceive | Feels for contact without a force sensor. | any |
| [`framing-a-target-by-mask-edge-steering`](#framing-a-target-by-mask-edge-steering) | perceive | Moves the wrist camera until a named target lies wholly inside its image with a margin. | any |
| [`localizing-a-piece-in-a-reference-image`](#localizing-a-piece-in-a-reference-image) | perceive | Finds which grid cell of a jigsaw's box picture one piece belongs to, and how far the piece is turned, from a close-up… | any |
| [`perceiving-deformable-linear-objects`](#perceiving-deformable-linear-objects) | perceive | Read a cable's ordered 3D centreline from one RGB-D frame and keep it current across a task. | any |
| [`perceiving-functional-features`](#perceiving-functional-features) | perceive | Locates a language-described functional part of a rigid object or fixture from calibrated RGB-D and fits a typed loop… | any |
| [`perceiving-next-item`](#perceiving-next-item) | perceive | Loop-head perception for pack-all / clean-all-items tasks. | any, *tuned on* panda |
| [`perceiving-object-parts`](#perceiving-object-parts) | perceive | Hierarchical perception for subpart targeting. | any |
| [`perceiving-objects`](#perceiving-objects) | perceive | Fast single-path 3D object perception. | any, *tuned on* panda |
| [`perceiving-objects-oneshot`](#perceiving-objects-oneshot) | perceive | Lightweight one-shot 3D object perception. | any, *tuned on* panda |
| [`perceiving-relational-correspondences`](#perceiving-relational-correspondences) | perceive | Perceives multiple source and destination instances and resolves language-conditioned one-to-one or many-to-one… | any |
| [`perceiving-routing-fixtures`](#perceiving-routing-fixtures) | perceive | Survey a cable-routing bench from one overhead RGB-D frame. | any |
| [`perceiving-sorting-pairs`](#perceiving-sorting-pairs) | perceive | Discovers a labelled four-compartment destination from RGB-D and repeatedly perceives one remaining source object with… | any |
| [`reconstructing-collision-worlds`](#reconstructing-collision-worlds) | perceive | Reconstructs a planner collision world from calibrated RGB-D views alone, excluding the grasp target, the robot, and a… | any |
| [`refining-target-from-wrist-closeup`](#refining-target-from-wrist-closeup) | perceive | Refines a coarse world-frame target (an overhead perception's OBB or point) by aiming the wrist RGB-D camera at it… | any |
| [`registering-held-objects`](#registering-held-objects) | perceive | Reobserves a grasped rigid object from the wrist cameras, estimates its functional feature in the TCP frame, and fits… | any |
| [`tracking-objects`](#tracking-objects) | perceive | Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. | any |
| [`computing-feature-mating-poses`](#computing-feature-mating-poses) | plan | Computes approach, engaged, and mate TCP poses from a held feature in the hand and a fixture feature, resolving free… | any |
| [`executing-keypoint-motion`](#executing-keypoint-motion) | plan | Plans and executes a one-step manipulation from a language instruction with MOKA's mark-based visual prompting. | any |
| [`planning-articulation-opening`](#planning-articulation-opening) | plan | Plans the hand's path that opens a drawer or a door it already holds by the handle. | any |
| [`planning-held-object-motion`](#planning-held-object-motion) | plan | Plans the carry and engagement phases for an already-held rigid object. | any |
| [`following-a-visual-plan`](#following-a-visual-plan) | propose | Follows a visual plan from robobrain.trace or molmo-motion.forecast (the shared visual-plan schema) the way those… | any |
| [`grasping-concave-contour`](#grasping-concave-contour) | propose | Grasp an awkward rigid part by the waists in its own silhouette. | any |
| [`proposing-short-axis-grasps`](#proposing-short-axis-grasps) | propose | Propose top-down grasps whose jaws close ACROSS an elongated object's short axis, derived from its oriented box. | any, **sim only** |
| [`proposing-side-grasps`](#proposing-side-grasps) | propose | Propose horizontal (side-entry) grasps swept around an object's oriented box — twelve azimuths ranked by jaw margin. | any, **sim only** |
| [`matching-a-piece-to-placed-neighbours`](#matching-a-piece-to-placed-neighbours) | select | Picks which of several loose jigsaw pieces fits an empty slot, and how far to turn it, by the shape of its sides alone. | any |
| [`sampling-labeled-points-in-mask`](#sampling-labeled-points-in-mask) | select | Picks the point on (or in) an object that a language instruction asks for. | any, *tuned on* yam-station-overhead-rgbd |
| [`selecting-free-placement-spots`](#selecting-free-placement-spots) | select | Picks world-frame spots on a named surface where a held object can be set down. | any, *tuned on* yam-station-overhead-rgbd |
| [`selecting-reachable-grasp`](#selecting-reachable-grasp) | select | Walk a ranked list of candidate grasp poses and take the first the arm can actually reach. | any, **sim only** |
| [`selecting-safe-cable-grasp-segments`](#selecting-safe-cable-grasp-segments) | select | Ranks top-down grasp points along a cable's fitted centreline that keep the fingers off its ends (a larger margin at… | any, *tuned on* yam-station-overhead-rgbd |
| [`anchoring-a-free-end`](#anchoring-a-free-end) | execute | Pins one end of a cable with the support hand before any crossing is attempted. | any, 2 arms |
| [`approaching-above-a-target`](#approaching-above-a-target) | execute | Moves the hand to hover a set clearance above a target's oriented box, rising in place first and keeping the hand's… | any |
| [`executing-feature-mating`](#executing-feature-mating) | execute | Executes a supplied feature-mating plan for an already-held rigid object. | any |
| [`executing-held-object-motion`](#executing-held-object-motion) | execute | Executes an ordered, collision-aware pose sequence for an already-grasped rigid object without releasing it. | any |
| [`grasping-a-cable-by-its-plug`](#grasping-a-cable-by-its-plug) | execute | Grasps a cable by the rigid plug at its end rather than by its thin jacket. | any |
| [`grasping-a-cable-lying-flat`](#grasping-a-cable-lying-flat) | execute | Grasps a thin cable lying flat on a table. | any |
| [`grasping-container-rims`](#grasping-container-rims) | execute | Grasps a thin-walled open container (box, bin, tray, cup, bowl) by its rim. | any |
| [`grasping-direct-ik`](#grasping-direct-ik) | execute | Direct IK align-then-descend grasping. | any, *tuned on* panda |
| [`grasping-flat-objects`](#grasping-flat-objects) | execute | Grasps an object lying flat on a surface that is too thin for a normal top-down pinch at its box centre (a card, a… | any |
| [`grasping-linear-feature`](#grasping-linear-feature) | execute | Fit the full 3D axis of an elongated segmented feature and grasp it with a perpendicular, inclination-aware… | any |
| [`grasping-short-axis`](#grasping-short-axis) | execute | Deterministic short-axis-aligned grasp with CuRobo. | any, *tuned on* panda |
| [`grasping-with-planner`](#grasping-with-planner) | execute | Top-down grasping via a fast axis-locked linear descend with a collision-aware cuRobo fallback. | any, *tuned on* panda |
| [`handing-over-between-arms`](#handing-over-between-arms) | execute | Hands the object one arm holds to the other arm. | any, 2 arms |
| [`holding-steady-while-other-arm-acts`](#holding-steady-while-other-arm-acts) | execute | One arm holds an object still. | any, 2 arms |
| [`inserting-a-peg-with-pose-refinement`](#inserting-a-peg-with-pose-refinement) | execute | Inserts a held peg into a perceived hole in three stages. | any |
| [`lifting-with-both-arms`](#lifting-with-both-arms) | execute | Lifts an elongated object (a bar, a tray, a pot by its handles) with both arms. | any, 2 arms |
| [`moving-both-arms-together`](#moving-both-arms-together) | execute | Moves both hands to their target poses as one coordinated motion. | any, 2 arms |
| [`picking-and-placing-two-objects`](#picking-and-placing-two-objects) | execute | Picks two objects with two arms, one each, and places both. | any, 2 arms |
| [`placing-held-feature`](#placing-held-feature) | execute | Move a held object so one of its declared features lands on a target point. | any, **sim only** |
| [`placing-into-container-held-by-other-arm`](#placing-into-container-held-by-other-arm) | execute | Puts the object one arm holds into an open container the other arm holds. | any, 2 arms |
| [`placing-into-container-obb`](#placing-into-container-obb) | execute | Puts the object the hand holds into an open bin, box or tray the camera sees. | any |
| [`placing-with-contact-descent`](#placing-with-contact-descent) | execute | Places the object the hand holds by approaching a fixed height above its place point, coming straight down until the… | any |
| [`plugging-a-cable-end`](#plugging-a-cable-end) | execute | Carries a cable's held free end into a terminal port after the last crossing. | any, 2 arms |
| [`pouring-by-wrist-roll`](#pouring-by-wrist-roll) | execute | Pours from a held container into a target container by standing the hand above the target's box at a set height and… | any |
| [`pushing-objects`](#pushing-objects) | execute | Pushes a loose object across its support toward a goal point with a closed hand, never grasping it. | any |
| [`reaching-with-lag-correction`](#reaching-with-lag-correction) | execute | Moves the tool to a target pose, waits for the arm to settle, reads the tool pose back and, if the arm stopped short… | any |
| [`searching-insertion-spiral`](#searching-insertion-spiral) | execute | Finds an opening the held part or the closed fingertips can enter by probing straight down over a hexagonal grid… | any |
| [`seating-a-cable-crossing`](#seating-a-cable-crossing) | execute | Streams one planned crossing leg of a cable onto a routing station, lets the rod settle, then LOOKS. | any |
| [`singulating-by-pushing`](#singulating-by-pushing) | execute | Pushes a target clear of the objects crowding it until a top-down grasp's open fingers fit around it. | any |
| [`sliding-along-a-cable`](#sliding-along-a-cable) | execute | Slides the hand along a cable it holds toward one of the cable's ends. | any |
| [`sliding-to-edge-then-picking`](#sliding-to-edge-then-picking) | execute | Picks a flat object too thin to pinch on its support. | any, *tuned on* yam_i2rt |
| [`stacking-objects`](#stacking-objects) | execute | Stacks the object the hand holds centred on top of a target object. | any |
| [`straightening-a-cable-toward-a-target-curve`](#straightening-a-cable-toward-a-target-curve) | execute | Pulls a cable lying flat on a table toward a target curve, one correction at a time. | any |
| [`threading-a-cable-tip-through-a-ring`](#threading-a-cable-tip-through-a-ring) | execute | Threads the free tip of a held cable through a standing ring. | any |
| [`tipping-over-a-surface-edge`](#tipping-over-a-surface-edge) | execute | Turn a body over on the surface it rests on instead of carrying it. | any, *tuned on* yam_i2rt, 2 arms, **sim only** |
| [`transporting-objects`](#transporting-objects) | execute | Move the currently-held object to a destination and release. | any, *tuned on* panda |
| [`turning-by-wrist-ratchet`](#turning-by-wrist-ratchet) | execute | Tightens or loosens a screw lid or cap by wrist ratchet. | any |
| [`wiping-a-region`](#wiping-a-region) | execute | Wipes a flat region of a surface with back-and-forth strokes. | any |
| [`checking-dual-arm-clearance`](#checking-dual-arm-clearance) | verify | Predicts, before a move is sent, the minimum clearance between the two arms along the planned path. | any, 2 arms |
| [`checking-spatial-relations`](#checking-spatial-relations) | verify | Decides from one RGB-D view whether a spatial relation holds between two seen objects. | any |
| [`classifying-failure-from-proprioception`](#classifying-failure-from-proprioception) | verify | Classifies the last moments of a manipulation step from proprioception alone. | any |
| [`previewing-target-overlay`](#previewing-target-overlay) | verify | Checks a target gripper pose before the arm moves. | any |
| [`verifying-a-cable-route`](#verifying-a-cable-route) | verify | Judge a finished cable route from a camera rather than from an evaluator. | any |
| [`verifying-articulation-displacement`](#verifying-articulation-displacement) | verify | Checks from the camera whether a drawer or a door really moved as asked. | any |
| [`verifying-grasps`](#verifying-grasps) | verify | Three independent checks on one grasp attempt. | any |
| [`verifying-insertion-by-probe-push`](#verifying-insertion-by-probe-push) | verify | Checks whether a held part is seated after an insertion by pushing it a few millimetres further along the insertion… | any |
| [`verifying-placement`](#verifying-placement) | verify | Confirms from one RGB-D view that a released object went through an aperture, accepting either that it vanished into… | any |
| [`centering-a-part-with-the-gripper`](#centering-a-part-with-the-gripper) | recover | Centres a loose part between the jaws without perceiving it. | any |
| [`correcting-insertion-offset-from-port-opening`](#correcting-insertion-offset-from-port-opening) | recover | Re-centres a held part over a port before or after a failed insertion using cameras only. | any |
| [`returning-to-a-recorded-pose`](#returning-to-a-recorded-pose) | recover | Opens the gripper and drives the arm back to a joint configuration recorded earlier in the episode, then checks from a… | any |
| [`classifying-failure-type-vlm`](#classifying-failure-type-vlm) | judge | Classifies why a manipulation step failed from camera frames taken before, during (optional, a short video as frames)… | any |
| [`detecting-subtask-success`](#detecting-subtask-success) | judge | Checks whether a subtask is done from the latest frame of every camera at one moment. | any |
| [`judging-step-with-before-now-vlm`](#judging-step-with-before-now-vlm) | judge | Judges the current subgoal of a task from camera frames taken before the step and now. | any |

## Policies

| Policy | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`molmoact-libero`](#molmoact-libero) | execute | Run the MolmoAct LIBERO checkpoint (allenai/MolmoAct-7B-D-LIBERO-0812) as a closed-loop VLA policy for the dexterous… | panda, **sim only** |
| [`pi05-libero`](#pi05-libero) | execute | Run the openpi π0.5 LIBERO checkpoint (pi05_libero) as a closed-loop VLA policy for the dexterous pick-and-place… | panda, **sim only** |

## Tool details

### colour

perceive · robots: any, *tuned on* yam-station-orbbec-overhead · tasks: sorting, pick-place, recognition · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/colour/SKILL.md) · [upstream](https://github.com/scikit-image/scikit-image/blob/5d91fed804ec8d9133118f02b3a785790df75934/src/skimage/color/delta_e.py)

What colour a segmented part is — robust colour of a mask (boundary eroded, clipped highlights dropped, lightness tails trimmed, Lab median), nearest entry of a palette by CIEDE2000 with the margin to the runner-up and an ambiguous / unknown status, white balance from grey world or a white/grey reference region, and a mask's dominant colours. Built-in palettes are the LDraw LEGO/DUPLO colours and domino tile/pip colours; a per-camera palette learned from labelled parts is passed as a list. Use when a workflow must tell bricks, tiles or parts apart by colour (sorting, find-the-red-2x4, verifying a build), especially the pairs VLMs confuse -- black vs dark grey, tan vs yellow.

- **Runs**: in its own environment over `stdio-msgpack`
- **Note**: On the YAM station's overhead camera the LDraw values alone are not enough (58-65% on real frames); a palette measured on that camera (colour.measure) with a bench reference reached 91%. LDraw colour values CCAL 2.0 (CC BY 2.0), LDraw.org.
- **Functions**:
  - `colour.classify`: Name a segmented part's colour from a palette (LEGO, DUPLO, domino or custom) by CIEDE2000, with the margin to the runner-up and an ambiguous status.
  - `colour.measure`: Robust colour of a mask (Lab and sRGB, highlights and shadows trimmed), to build a per-camera palette from labelled parts.
  - `colour.white_balance`: Per-channel white-balance gains from grey world or from a white/grey reference region; optionally the balanced image.
  - `colour.dominant_colours`: Cluster a mask into its dominant colours (k-means in Lab, near clusters merged), each with its share and optionally its palette name.

### curve

perceive · robots: any · tasks: cable-routing · Apache-2.0 · v0.2.0 · [`SKILL.md`](tools/curve/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/curve)

Ordered centrelines of deformable linear objects — skeletonises a cable, rope or hose mask, recovers one traversal order through occlusion breaks and self-crossings (TrackDLO's chain merge), back-projects the ordered pixels through depth onto the object's axis as an arc-length-parameterised polyline, and carries a known centreline onto the next frame with motion imputed for hidden stretches, and reads off its two ends (tip position, outward direction, pixel). Use when a workflow needs an ordered centreline of a cable, rope or hose from a mask and a depth frame, must carry one through occlusion between frames, or needs where a cable's tip is and which way it points.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `curve.fit_centerline`: Fit an ordered, arc-length-parameterised 3D centreline to a deformable linear object from its mask and a depth frame.
  - `curve.track_centerline`: Advance a known centreline onto a new frame, imputing motion for occluded stretches from their visible neighbours.
  - `curve.endpoints`: Both ends of a centreline -- each end's position, outward direction and pixel -- and which end a pixel or point hint names.

### depth

perceive · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/depth/SKILL.md) · [upstream](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/model/v2.py)

Robbyant LingBot-Depth v0.5 run locally on a GPU -- fills the holes of a sensor depth image from its RGB frame (the overhead Orbbec's or the wrist D405's) and, on request, denoises it, returning metric depth of the same size, a validity mask and a mask of the filled pixels. Use before geometry, grasp or pose tools when raw depth has holes or speckle on dark, shiny, thin or small parts (a black board, a cube, a ring, a flat cable).

- **Weights**: `hf://robbyant/lingbot-depth-pretrain-vitl-14-v0.5@79204ed6b837f4fdd192cf563e59481fecfa0295`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `depth.refine`: Fill (default) or denoise an RGB-D frame's depth; same-size metric depth, validity and filled masks.

### gemini-er

perceive · robots: any · MIT · v0.2.0 · [`SKILL.md`](tools/gemini-er/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/gemini-er)

Open-vocabulary 2D detection and pointing via the Gemini Robotics-ER API — one call returns pixel-space bounding boxes with labels and scores, or pixel points, for a text query. Use when a workflow needs a detection box or a point to seed segmentation (e.g. sam3.segment_box) or coarse localization without any local GPU model, including parts a box bounds poorly such as a cable's tip.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `gemini-er.detect`: Detect pixel-space 2D bounding boxes for a text query via Gemini Robotics-ER.
  - `gemini-er.point`: Point at what a text query names via Gemini Robotics-ER; returns pixel points.

### geometry

perceive · robots: any, *tuned on* panda · MIT · v0.1.0 · [`SKILL.md`](tools/geometry/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/geometry)

Pure-math 3D geometry toolbox — back-project masks and depth to point clouds, DBSCAN-filter noise, fit oriented bounding boxes, derive top-down/front grasp poses, and reconstruct collision worlds from RGB-D frames. Use when a workflow needs perception geometry or planner inputs computed on CPU with no model weights.

- **Runs**: in its own environment over `stdio-msgpack`
- **Note**: exclude_robot_points models the Franka Panda's links; on another arm its points are not masked.
- **Functions**:
  - `geometry.depth_to_point_cloud`: Convert a metric depth image to a camera-frame point cloud.
  - `geometry.mask_to_world_points`: Back-project a 2D mask to 3D world points via depth + calibration.
  - `geometry.pixel_to_world_point`: Back-project a single pixel to a 3D world point.
  - `geometry.transform_points`: Apply a rigid SE(3) transform to 3D points.
  - `geometry.fit_planar_feature`: Fit a 6D frame, normal, median radius, and robust inner/outer radii to a ring, loop, rim, or planar opening cloud.
  - `geometry.fit_linear_feature`: Fit an axis and robust endpoints to a rod, peg, shaft, or hook cloud.
  - `geometry.cloud_to_attachment`: Convert an observed object cloud into an attached collision model and object-to-TCP transform.
  - `geometry.compute_feature_mate`: Compute metric approach, engagement, and seating poses for compatible held and fixture features.
  - `geometry.exclude_robot_points`: FK-sphere removal of robot-body points from a cloud (7-DOF Franka; other arms pass through).
  - `geometry.filter_noise`: DBSCAN noise filtering (keeps all non-noise points).
  - `geometry.compute_obb`: Fit an upright oriented bounding box to 3D points (HyRL fit).
  - `geometry.filter_and_compute_obb`: DBSCAN filter then OBB fit in one call.
  - `geometry.top_down_grasp_from_obb`: One world-aligned top-down grasp pose from an OBB.
  - `geometry.top_down_grasp_candidates`: Fan of top-down grasp candidates (canonical primary+alt first).
  - `geometry.select_top_down_grasp`: Pick the most top-down grasp from candidates.
  - `geometry.front_grasp_from_obb`: Front-approach grasp/pre-grasp for handles (drawers, doors).
  - `geometry.build_world_config`: Reconstruct a multi-surface collision world from RGB-D; touching planes are separated into closed slabs, residual non-planar clusters are meshed independently, and optional live robot spheres remove self-returns before meshing.
  - `geometry.rotate_quat_z90`: Rotate a wxyz quaternion 90 degrees around world Z.
  - `geometry.compute_drop_position`: Drop position above a container OBB.
  - `geometry.compute_xy_distance`: XY-plane distance between two 3D points.
  - `geometry.iou`: IoU of two axis-aligned 2D boxes.
  - `geometry.pose_distance`: Euclidean distance between two 3D positions.

### grounding-dino

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/grounding-dino/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/grounding-dino)

Grounding DINO zero-shot object detection — natural-language queries to labeled 2D bounding boxes with confidence scores. Use when a workflow needs to locate named objects in an RGB image before segmenting or grasping them.

- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `grounding-dino.detect`: Zero-shot detection from a text prompt (labeled boxes + scores).

### locate-anything

perceive · robots: any · Apache-2.0 for this bundle's code; the LocateAnything-3B weights and the NVlabs worker it downloads at run time are under the NVIDIA License, non-commercial (research or evaluation) use only · v0.1.0 · [`SKILL.md`](tools/locate-anything/SKILL.md) · [upstream](https://github.com/NVlabs/Eagle/blob/06edcff802dbdd167d196c7a1c5b44929e07d951/Embodied/locateanything_worker.py)

NVIDIA LocateAnything-3B (non-commercial licence) run locally on a GPU -- category detection, referring-phrase grounding and pointing, returned in gemini-er's shapes (pixel boxes with label/score; pixel points). Use when a research or evaluation workflow wants a local, API-free alternative to gemini-er.detect / gemini-er.point on camera frames. NON-COMMERCIAL ONLY - the weights and the worker it runs are under the NVIDIA License (research or evaluation use); it refuses to run until GAP_LOCATE_ANYTHING_NONCOMMERCIAL=1 is set.

- **Weights**: `hf://nvidia/LocateAnything-3B@c32291ca5e996f5a7a485845b4f57a233936bba0`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Note**: Non-commercial research or evaluation use only (NVIDIA License); never a default for commercial deployments.
- **Functions**:
  - `locate-anything.detect`: Boxes for every instance of one or more categories (pixel box, label, score).
  - `locate-anything.ground`: Boxes for what a referring phrase names; drop-in for gemini-er.detect.
  - `locate-anything.point`: Pixel points for what a phrase names; drop-in for gemini-er.point.

### molmo

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/molmo/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/molmo)

Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). Use when a workflow needs a single pixel coordinate for a named object (point_prompt) or Molmo-grade visual question answering and a vLLM server is available; for an API-only zero-GPU alternative use the gemini-er bundle.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `molmo.point_prompt`: Point at a named object in an image; returns pixel coordinates.
  - `molmo.query`: Free-form visual question answering via Molmo.
  - `molmo.query_yes_no`: Yes/no visual question answering via Molmo; coerces the reply to a bool.

### pose6d

perceive · robots: any · tasks: pick-place, insertion, reorientation, sorting · MIT · v0.1.0 · [`SKILL.md`](tools/pose6d/SKILL.md) · [upstream](https://github.com/isl-org/Open3D/tree/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/examples/python/pipelines)

6D pose of a known rigid part from depth — registers a box, cube or cylinder primitive (LEGO brick body, domino, cube, peg) or a mesh file to the part's point cloud or to its mask in a depth frame, and returns the pose with fitness, RMSE, surface coverage, every symmetric equivalent and the best non-equivalent alternative. Boxes and cylinders take an analytic, visibility-aware point-to-plane ICP over a hypothesis search (numpy/scipy, tens to a few hundred ms); meshes take Open3D FPFH + RANSAC global registration then point-to-plane ICP. Use when a workflow needs the full pose (not just an OBB) of a known-size part from the wrist D405 or the overhead camera.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `pose6d.register_mesh`: 6D pose of a known part (box / cube / cylinder primitive or mesh file) from its point cloud, with fitness, RMSE, coverage and its symmetric equivalents.
  - `pose6d.register_mask`: 6D pose of a known part from its segmentation mask and a depth frame (camera centre used as the viewpoint); world pose when a camera pose is given.

### sam3

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/sam3/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/sam3)

Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker that carries object identity through SAM3's memory bank. Use when a workflow needs open-vocabulary masks from an RGB image or needs to follow one object across frames.

- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `sam3.segment_text`: Segment all instances matching a text description (masks, scores, boxes best-first).
  - `sam3.segment_point`: Segment the object at a pixel coordinate (multimask, best-first).
  - `sam3.segment_box`: Segment within a bounding box, optionally refined by a foreground point.
  - `sam3.tracker_init`: Open a tracker session seeded with one frame + a text/box/point prompt.
  - `sam3.tracker_update`: Advance a tracker session by one frame (mask, box, confidence).
  - `sam3.tracker_close`: Free a tracker session (idempotent).

### track

perceive · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/track/SKILL.md) · [upstream](https://github.com/google-deepmind/tapnet/tree/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp)

Google DeepMind TAPNext++ run locally on a GPU -- follows chosen pixels (a cable tip, a plug, a small part, a cube corner) through a video, returning each one's pixel position and visibility per frame and keeping its position estimate through occlusions until it comes back. Use when a verify or recover stage needs where a point went across a clip, or a closed loop needs it frame by frame from a live wrist or overhead camera.

- **Weights**: `https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `track.points`: Track query pixels through a clip; per-frame pixels, visibility and confidence (queries on any frame).
  - `track.start`: Open an online tracking session on a live frame and query pixels.
  - `track.update`: Advance a session by one frame; each query's pixel and visibility.
  - `track.stop`: Close a session and free its GPU state.

### vlm

perceive · robots: any · MIT · v0.1.1 · [`SKILL.md`](tools/vlm/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/vlm)

Free-form and yes/no visual question answering against a hosted vision-language model (OpenRouter API by default; Vertex AI Gemini selectable by config). Use when a workflow needs scene descriptions, semantic checks ("is the drawer open?"), or LLM-judged verification of a camera frame — no GPU required.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `vlm.query`: Free-form visual question answering via a hosted VLM.
  - `vlm.query_yes_no`: Yes/no visual question answering; coerces the model reply to a bool.

### curobo

plan · robots: any, *tuned on* panda · tasks: motion-planning · MIT · v0.1.0 · [`SKILL.md`](tools/curobo/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/curobo)

NVIDIA cuRobo motion planning — collision-free trajectories to grasp goalsets, transport with an attached object, constrained linear moves, single-pose planning, geometric IK, batch grasp feasibility, and joint-trajectory collision validation. Use when a workflow needs GPU-accelerated, collision-aware arm motion plans.

- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Note**: Plans for curobo's built-in franka.yml. It knows yam.yml is 6-DoF but does not ship that config; register its path in _PROJECT_ROBOT_CONFIGS before planning for yam.
- **Functions**:
  - `curobo.plan_to_grasp_poses`: Collision-free trajectory to one of several grasp poses (goalset).
  - `curobo.plan_with_grasped_object`: Transport plan with the grasped object attached to the gripper.
  - `curobo.plan_linear`: Straight Cartesian trajectory between two EE poses.
  - `curobo.plan_directed_linear`: Constrained linear motion with axis/orientation holds.
  - `curobo.plan_grasp_motion`: Approach + grasp + lift sequence (three trajectories).
  - `curobo.plan_to_pose`: Collision-aware plan to a single TCP pose.
  - `curobo.solve_ik`: Geometric IK for a single TCP pose (no world collision).
  - `curobo.batch_grasp_feasibility`: Per-pose grasp/approach IK + corridor feasibility for a grasp batch.
  - `curobo.validate_joint_trajectory_robot`: Collision-validate joint waypoints (robot vs world + self).
  - `curobo.validate_joint_trajectory_grasped`: Same, with a grasped object attached at waypoint 0.
  - `curobo.cloud_to_attachment`: MORPHIT collision spheres for a held object's cloud, in the TCP frame.

### molmo-motion

propose · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/molmo-motion/SKILL.md) · [upstream](https://github.com/allenai/molmo-motion/tree/ab7c5f4e7c49359e4275e9a9b21a59659be17386)

AllenAI MolmoMotion-4B run locally on a GPU -- forecasts where chosen pixels (a cable tip, a cube corner) move over the next ~2 s under a language instruction, as pixels per step and, given depth and the camera, as camera and world points. Use when a propose or judge stage needs the motion a model expects ("the cable tip is pulled through the ring") before planning or checking it.

- **Weights**: `hf://allenai/MolmoMotion-4B-H3-F30@3f5e790a511ff2cdf21c8d2a14cb4d8409c94629`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `molmo-motion.forecast`: Forecast query pixels' motion under an instruction (pixels per step; camera/world points with depth).

### robobrain

propose · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/robobrain/SKILL.md) · [upstream](https://github.com/FlagOpen/RoboBrain2.5/blob/af98c932aac9ff715d70da177088d7bb95573ff7/inference.py)

BAAI RoboBrain 2.5-4B run locally on a GPU -- an ordered keypoint trajectory for a manipulation from one frame and an instruction (pixels, and camera / world points with the camera), pixel points for a phrase in gemini-er.point's shape, and task progress 0..1 against the task's start frame. Use when a propose stage wants a candidate path ("move the cable's free end through the ring") or a judge stage a progress score, without an API.

- **Weights**: `hf://BAAI/RoboBrain2.5-4B@ec4b3bf5e969b48ad17df432b479c10402a0079b`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `robobrain.trace`: Ordered keypoint trajectory for an instruction from one frame (visual-plan schema; pixels, camera/world points with intrinsics).
  - `robobrain.point`: Pixel points for what a phrase names; drop-in for gemini-er.point.
  - `robobrain.progress`: Task progress 0..1 of the current frame against the task's start frame.

### predicates

judge · robots: any · tasks: pick-place, insertion, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/predicates/SKILL.md)

Geometric yes/no judgments over poses and boxes the graph already holds -- is a point inside an oriented box, does one box rest on another, are two points within a distance, is a body upright -- each returning a boolean, a verdict string a router can branch on, and the margin it was decided by. Use when a graph must choose a branch from geometry it already has (a perceived OBB, an end-effector pose), on CPU, with no model and no simulator state.

- **Functions**:
  - `predicates.point_in_obb`: Whether a world point lies inside an oriented box, and by how much.
  - `predicates.resting_on`: Whether one box sits on top of another, from footprint overlap and the vertical gap.
  - `predicates.within_distance`: Whether two points are within a threshold, in 3D or in the table plane.
  - `predicates.upright`: Whether a body's local axis points up, within a tilt bound.

## Skill details

### detecting-contact-from-joint-effort

perceive · robots: any · tasks: pick-place, insertion · MIT · v0.1.0 · [`SKILL.md`](skills/detecting-contact-from-joint-effort/SKILL.md) · [upstream](https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/robots/utils.py)

Feels for contact without a force sensor - steps the hand along a short straight line a few millimetres at a time and, after each step, decides from proprioception whether something holds it - the joint efforts against a baseline taken in free motion along the same line (i2rt's gripper clog test, generalised to the arm) where the observation carries efforts, otherwise how far the measured tool centre lags the commanded one beyond its free-motion lag. Returns where the hand stopped and which signal fired. Use when a surface's position is known only to a centimetre or so and must be measured by touch (a table or a box floor before a grasp or a place, the top of an object, a wall before a slide), or to check that a line is clear before a move along it.

- **In**: `end` Vec3
- **Out**: `contact` bool, `signal` str, `contact_position` Vec3, `contact_point` Vec3, `travelled` float, `lag_over_baseline` float, `effort_ratio` float, `efforts_available` bool, `steps` int, `baseline_lag` float
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.solve_ik`
- **Scripts**: [`detect_contact`](skills/detecting-contact-from-joint-effort/scripts/detect_contact.py)
- **Exits**:
  - `contact`: A step left the hand held - its lag beyond the free-motion lag over lag_threshold (3 mm), or a joint's effort residual over its threshold; contact_position is the tool centre where it stood, contact_point that pushed on by tip_reach, `signal` says which fired (lag, effort or effort+lag); the hand then backed off `backoff` (3 mm) along the line.
  - `free`: The hand reached the line's end (or the tool-centre floor) with every step tracking as in free motion; it stays there.
  - `no_signal`: Nothing usable to decide with - signal=effort with an observation that carries no efforts (nothing moved), the line too short for two free-motion samples, or the free-motion lag jumping more than lag_threshold from step to step; `detail` says which.
  - `blocked`: The line is too long (over 150 mm), has no length, starts under the floor or ends out of reach (IK preview), the start was not reached, or a step was refused; `detail` says which.

### framing-a-target-by-mask-edge-steering

perceive · robots: any · tasks: pick-place, insertion, cable-routing, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/framing-a-target-by-mask-edge-steering/SKILL.md) · [upstream](https://github.com/AMMistry18/aic/blob/fd27bbe875dbd656e381f72ae2b44efdfb243924/flowstate/aic_perception/aic_perception/board_visibility.py)

Moves the wrist camera until a named target lies wholly inside its image with a margin - segments the target in the wrist frame (sam3.segment_text), reads which image edges its mask comes within the margin of and by how much, and steps the camera away from those edges in its image plane (or backs it off along its optical axis when opposite edges are clipped or an edge persists, or approaches when the target is too small), re-segmenting after each step, until the mask is inside every edge. Use when a part, ring, fixture or cable end is only partly in the wrist camera's view (after a coarse approach, a grasp, or a refined close-up that left it at the image border) and the next step needs all of it in one wrist frame, with no world estimate required.

- **In**: `object_name` str
- **Out**: `target_point` Vec3, `view_pose` Se3Pose, `area_frac` float, `depth` float, `moves` int, `travel` float
- **Calls**: [`sam3`](#sam3); connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`
- **Scripts**: [`frame_target`](skills/framing-a-target-by-mask-edge-steering/scripts/frame_target.py)
- **Exits**:
  - `framed`: A fresh wrist frame's mask of the target is inside every image edge by the margin (max(margin_px, 0.1 x its larger side)) and covers at least min_area_frac of the image - or covers less, but the camera is already min_range from it. The hand stays at that view; target_point is the per-axis median of the masked pixels back-projected with the wrist depth (a rough point on the visible surface).
  - `not_seen`: There is no wrist camera, the segmenter failed or found no mask scoring min_score in the first frame, or the target was lost after a step and not found again back at the last view that saw it (the hand is left there); `detail` says which.
  - `step_limit`: max_moves steps were taken and the mask still comes within the margin of an edge; `edges` names them. The hand stays at the last view.
  - `unreachable`: A step was refused (raised or routed) or ended more than reach_tolerance off at full and at half length, or the next step would bring the tool centre within 50 mm of the target or back the camera past max_range; `detail` says which.

### localizing-a-piece-in-a-reference-image

perceive · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/localizing-a-piece-in-a-reference-image/SKILL.md)

Finds which grid cell of a jigsaw's box picture one piece belongs to, and how far the piece is turned, from a close-up wrist-camera crop and its mask - masked normalised cross-correlation searched over rotation at the one scale the grid and board size fix, coarse over the whole picture then refined round the best cells, with border pieces matched along their straight side. Returns the cell, the angle, the position in picture pixels and board metres, the score, the margin over the next-best cell and a ranked candidate list; flat pieces (white snow, black cloth) get a shortlist of flat cells of their colour instead. Use when a jigsaw-assembly step needs to know where a picked or inspected piece goes and the box picture is available; under a second per piece on CPU.

- **In**: `camera` CameraFrame, `piece_mask` Mask, `reference` string, `grid_cols` int, `grid_rows` int, `board_width` float, `board_height` float
- **Out**: `col` int, `row` int, `angle_deg` float, `score` float, `margin` float
- **Scripts**: [`locate_piece`](skills/localizing-a-piece-in-a-reference-image/scripts/locate_piece.py)
- **Exits**:
  - `located`: The best cell's score is at least min_score (0.45) and beats every other cell by min_margin (0.04; 0.10 when the scale was guessed from the mask area); col, row and angle_deg are the answer.
  - `ambiguous`: The best cell does not stand out (repetitive or flat art); `candidates` ranks the cells for a shape check. A low-texture piece always lands here, with the flat cells of its colour as candidates.
  - `not_found`: The best score is under min_score, a flat piece matches no flat cell, or the input could not be used (empty mask, no scale, no placement inside the picture).

### perceiving-deformable-linear-objects

perceive · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/perceiving-deformable-linear-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-deformable-linear-objects)

Read a cable's ordered 3D centreline from one RGB-D frame and keep it current across a task — seed it from an unobstructed survey, re-fit it cold whenever the whole rod is in view, and fall back to tracking the carried prior only when the fresh fit comes back short. Use when a manipulation loop needs the current shape of a deformable linear object (a cable, rope, or hose) once per step rather than every frame, from a fixed third-person camera.

- **In**: `prior` string, `scene` string
- **Out**: `rod` string, `arclength_m` float, `source` string, `camera` string, `visible` int
- **Calls**: [`curve`](#curve), [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`perceive_rod`](skills/perceiving-deformable-linear-objects/scripts/perceive_rod.py)
- **Exits**:
  - `found`: A centreline is bound in `rod` — seeded, initialised, re-fitted, tracked, or the prior held because this frame could not improve on it.
  - `lost`: No centreline could be born — no depth camera by the given name, no mask for the query, or no candidate fitted as a curve — and there was no prior to hand back.

### perceiving-functional-features

perceive · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/perceiving-functional-features/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-functional-features)

Locates a language-described functional part of a rigid object or fixture from calibrated RGB-D and fits a typed loop, shaft, tip, aperture, surface, or region feature with a metric centre, axis, and radius, either inside an already-segmented parent or end to end from a described object, protruding shaft, directed tip, or lidded aperture. Use when a manipulation step depends on where a specific part is rather than on the whole object, such as a tool loop to hang, a shaft to hang it on, an insertion tip, or an opening to insert into.

- **In**: -
- **Out**: -
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3)
- **Scripts**: [`perceive_feature`](skills/perceiving-functional-features/scripts/perceive_feature.py), [`perceive_object_feature`](skills/perceiving-functional-features/scripts/perceive_object_feature.py), [`perceive_protruding_shaft`](skills/perceiving-functional-features/scripts/perceive_protruding_shaft.py), [`perceive_directed_tip`](skills/perceiving-functional-features/scripts/perceive_directed_tip.py), [`perceive_aperture`](skills/perceiving-functional-features/scripts/perceive_aperture.py), [`perceive_fixture_feature`](skills/perceiving-functional-features/scripts/perceive_fixture_feature.py)
- **Exits**:
  - `found`: One geometrically valid feature was fitted and its centre, axis, and radius are bound.
  - `not_found`: No described object or feature passed the segmentation and geometry gates (raised by every script, or returned as `route` by `perceive_protruding_shaft` when `required` is false).

### perceiving-next-item

perceive · robots: any, *tuned on* panda · tasks: pick-place, sorting · MIT · v0.1.0 · [`SKILL.md`](skills/perceiving-next-item/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-next-item)

Loop-head perception for pack-all / clean-all-items tasks. From ONE observation it localizes BOTH the destination container (basket, bin, box) AND the next remaining target item, using the pairwise VLM crop tournament with a container-excluding description so the target is never confused with the basket, then makes a clean found / none decision: an item was found (grasp it) or only the container remains (all items packed — exit the loop to done). Use when a workflow must pick up EVERY object and place each into a container in a loop (pack-all / clean-all-items) and each pass must reliably answer "is there still a graspable item, or are we done?" while also exposing a fresh container OBB for the downstream place. This subgraph is self-contained and takes NO inputs (inputs: {}); the container and item phrases are literal strings written inside the perception nodes, never subgraph parameters — do not declare item_name / container_name / item_description as subgraph inputs.

- **In**: -
- **Out**: `target_obb` OrientedBoundingBox, `target_mask` Mask, `container_obb` OrientedBoundingBox, `container_mask` Mask, `container_cloud` PointCloud
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`perceive_dino_vlm`](skills/perceiving-next-item/scripts/perceive_dino_vlm.py), [`exterior_view`](skills/perceiving-next-item/scripts/exterior_view.py), [`decide_next_item`](skills/perceiving-next-item/scripts/decide_next_item.py)
- **Exits**:
  - `found`: A graspable target item distinct from the container was localized; target_obb/mask and container_obb/mask/cloud bound in the subgraph outputs. Route to a grasp skill.
  - `none`: No target item remains — only the container is left, so every item is packed. This is the clean loop terminator; route to done (NOT abort — it is a success).
  - `perception_failed`: Perception raised an actual error (empty detection is NOT this — that is the normal `none` path). Route to abort. Named so it is never confused with "nothing left"; do NOT route it to done.

### perceiving-object-parts

perceive · robots: any · tasks: pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/perceiving-object-parts/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-object-parts)

Hierarchical perception for subpart targeting. Detects a parent object first (DINO+VLM), crops the camera image to the parent's bounding box, then detects and segments the named subpart inside the crop (DINO + SAM3), and uncrops + fuses depth to a world-frame OBB/mask/cloud — plus the parent object's OBB and cloud for downstream placement/collision reasoning. Use when the graspable affordance is a subpart of a larger object — pan handle, drawer pull, mug rim, moka-pot handle, stove burner — where detecting the subpart at full image resolution is unreliable because it occupies few pixels.

- **In**: -
- **Out**: `<name>_obb` OrientedBoundingBox, `<name>_mask` Mask, `<name>_cloud` PointCloud, `<name>_parent_obb` OrientedBoundingBox, `<name>_parent_cloud` PointCloud
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`perceive_subpart`](skills/perceiving-object-parts/scripts/perceive_subpart.py)
- **Exits**:
  - `found`: Subpart detected; OBB/mask/cloud bound in subgraph outputs.
  - `not_found`: Parent or subpart not visible in any view. Coordinator routes to abort.

### perceiving-objects

perceive · robots: any, *tuned on* panda · tasks: pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/perceiving-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-objects)

Fast single-path 3D object perception. Runs Grounding-DINO broad detection, a pairwise VLM crop tournament to identify the target box, SAM3 box segmentation, and depth back-projection to a world-frame point cloud, finished by geometry.filter_and_compute_obb for a clean oriented bounding box, mask, and cloud. Multi-camera rigs get a verified wrist-camera fallback gate. Use when a manipulation workflow needs to localize one named object quickly — uncluttered scenes with visually distinct targets, latency-bound loops, or platforms deploying only DINO + VLM + SAM3 + geometry.

- **In**: -
- **Out**: `<name>_obb` OrientedBoundingBox, `<name>_mask` Mask, `<name>_cloud` PointCloud
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`perceive_dino_vlm`](skills/perceiving-objects/scripts/perceive_dino_vlm.py)
- **Exits**:
  - `found`: Target detected; OBB and mask bound in subgraph outputs.
  - `not_found`: Target not visible in any view. In clean-all-items loops route to done; in normal pick-and-place, route to abort.

### perceiving-objects-oneshot

perceive · robots: any, *tuned on* panda · tasks: pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/perceiving-objects-oneshot/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-objects-oneshot)

Lightweight one-shot 3D object perception. Runs Grounding-DINO broad detection, a SINGLE VLM set-of-marks letter pick over the labeled boxes, SAM3 box segmentation, depth back-projection, and geometry.filter_and_compute_obb. No pairwise tournament, no multi-view safe-gate. Returns a clean not_found output (no exception) when the VLM answers "none" or DINO emits no detections — making this the right skill for clean-all-items loops whose natural termination signal is "no more matching objects in view". Use when a multi-item loop needs a clean no-match exit, or for generic target descriptions on uncluttered scenes with reasonably sized targets.

- **In**: -
- **Out**: `<name>_obb` OrientedBoundingBox, `<name>_mask` Mask, `<name>_cloud` PointCloud
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`perceive_simple`](skills/perceiving-objects-oneshot/scripts/perceive_simple.py)
- **Exits**:
  - `found`: Target detected; OBB and mask bound in subgraph outputs.
  - `not_found`: VLM replied "none" or DINO emitted no detections. In clean-all-items loops route to done; in normal pick-and-place route to abort.

### perceiving-relational-correspondences

perceive · robots: any · tasks: sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/perceiving-relational-correspondences/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-relational-correspondences)

Perceives multiple source and destination instances and resolves language-conditioned one-to-one or many-to-one correspondences. Use when an instruction relates instances by spatial order, proximity, appearance, labels, or a shared destination and downstream manipulation needs stable candidate IDs, masks, point clouds, and OBBs. Do not use for single-instance localization or for functional-feature fitting.

- **In**: `instruction` string
- **Out**: `status` string, `source_id` string, `target_kind` string, `target_mask` Mask, `correspondence_anchor` float, `arm_id` int
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3)
- **Scripts**: [`select_next_source`](skills/perceiving-relational-correspondences/scripts/select_next_source.py), [`match_candidates`](skills/perceiving-relational-correspondences/scripts/match_candidates.py)
- **Exits**:
  - `found`: One eligible source instance and its correspondence anchor were selected (select_next_source).
  - `finished`: No eligible uncompleted source remains, or no source was given (both scripts).
  - `paired`: Every source was assigned a destination under the declared relation (match_candidates).
  - `not_found`: No destination candidate exists (match_candidates).
  - `cardinality_mismatch`: The sources and destinations cannot be assigned under the relation (match_candidates).
  - `ambiguous`: The instruction maps to multiple declared categories, or the declared pairs do not resolve to exactly one assignment.

### perceiving-routing-fixtures

perceive · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/perceiving-routing-fixtures/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-routing-fixtures)

Survey a cable-routing bench from one overhead RGB-D frame — find every station (spool, cleat) by colour, roundness and height off the work surface, fit the cable's ordered centreline through a text-prompted segmenter, derive the side each crossing owes from the instruction's alternation, and place the physical seat beside each post. Use when a routing plan needs the fixture layout and the starting shape of a deformable linear object, once, before the first move.

- **In**: -
- **Out**: `scene` string, `stations` int, `rod` string, `body` string
- **Calls**: [`curve`](#curve), [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`perceive_weave_vision`](skills/perceiving-routing-fixtures/scripts/perceive_weave_vision.py)
- **Exits**:
  - `found`: At least one station and a cable-shaped centreline were measured; `scene` carries stations, sides, seats and the rod.
  - `not_found`: No station passed the colour and height gates and the text fallback, or no rod mask fitted as a curve; `scene` is empty.

### perceiving-sorting-pairs

perceive · robots: any · tasks: sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/perceiving-sorting-pairs/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-sorting-pairs)

Discovers a labelled four-compartment destination from RGB-D and repeatedly perceives one remaining source object with its matching metric destination region, using an identity table the graph supplies for what each label looks like and what its graspable part is called. Use when a workflow must sort several visible objects into compartments identified by printed category labels.

- **In**: `instruction` string, `layout_description` string, `source_description` string
- **Out**: `target_obb` OrientedBoundingBox, `target_mask` Mask, `target_cloud` PointCloud, `destination_obb` OrientedBoundingBox, `layout_json` string
- **Calls**: [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`discover_regions`](skills/perceiving-sorting-pairs/scripts/discover_regions.py), [`select_pair`](skills/perceiving-sorting-pairs/scripts/select_pair.py)
- **Exits**:
  - `found`: An object and matching region were selected.
  - `ambiguous`: Identity or association needs another view.
  - `finished`: No unsorted object remains.

### reconstructing-collision-worlds

perceive · robots: any · tasks: pick-place, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/reconstructing-collision-worlds/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/reconstructing-collision-worlds)

Reconstructs a planner collision world from calibrated RGB-D views alone, excluding the grasp target, the robot, and a perceived fixture mask, dropping depth-edge slivers and invalid-depth sheets, and carving the free space an intentional-contact goal needs as an approach tube along a fixture axis or a corridor through a lid aperture. Use when a held-object motion must be planned against obstacles no simulator or CAD scene supplies, before registering the object and planning the carry.

- **In**: `observation` Observation, `target_mask` Mask
- **Out**: `world_config` WorldConfig, `strategy` string
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `motion.get_robot_collision_spheres`
- **Scripts**: [`build_collision_world`](skills/reconstructing-collision-worlds/scripts/build_collision_world.py), [`filter_mesh_components`](skills/reconstructing-collision-worlds/scripts/filter_mesh_components.py)
- **Exits**:
  - `built`: A scene mesh set with the goal's free space carved was produced.
  - `failed`: No camera was selected or the reconstruction produced no scene mesh (raised).

### refining-target-from-wrist-closeup

perceive · robots: any · tasks: pick-place, insertion, sorting, jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/refining-target-from-wrist-closeup/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/find_object_via_wrist/robotwin)

Refines a coarse world-frame target (an overhead perception's OBB or point) by aiming the wrist RGB-D camera at it from 0.15-0.3 m - the hand is placed so the camera's optical axis passes through the target, looking straight down when the arm reaches that - then re-pointing the object in the close-up frame with a pointing model (gemini-er.point or molmo.point_prompt) and back-projecting the pointed pixel with the wrist depth and the camera's pose to a refined world point and a shifted OBB. Use when an overhead detection is too coarse for the next step (a small part, a grasp that missed by 1-3 cm, a target the overhead camera sees at a few pixels, such as a jigsaw piece to be matched from a wrist close-up) and the arm carrying the wrist camera is free.

- **In**: `object_name` str
- **Out**: `refined_point` Vec3, `refined_obb` OrientedBoundingBox, `coarse_point` Vec3, `drift` float, `depth` float, `camera_range` float, `view_pose` Se3Pose
- **Calls**: [`gemini-er`](#gemini-er), [`molmo`](#molmo); connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`refine_from_wrist`](skills/refining-target-from-wrist-closeup/scripts/refine_from_wrist.py)
- **Exits**:
  - `refined`: The wrist camera saw the coarse target from camera_range (0.15-0.30 m), the pointer found the object, the wrist depth round the pointed pixel was valid, and the back-projected point lies within the object's half-diagonal + 30 mm of the coarse estimate in the plane; refined_point and refined_obb carry it. The hand stays at the view pose.
  - `not_seen`: There is no wrist camera in the observation, or the pointer found nothing (or failed), or fewer than 3 depth readings in 7-50 cm lay round the pointed pixel, or the pointed point lies further from the coarse estimate than allowed (another object); `detail` says which. The hand stays at the view pose if it moved.
  - `unreachable`: No view pose passed the IK preview and the clearance floor, or three view moves were refused, ended more than reach_tolerance off, or left the target outside the wrist image; `detail` lists each attempt.

### registering-held-objects

perceive · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/registering-held-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/registering-held-objects)

Reobserves a grasped rigid object from the wrist cameras, estimates its functional feature in the TCP frame, and fits attached collision spheres, retaining the grasp-time transform at low confidence when the object is not seen. Use when a held object must be localized in the hand right after grasping or re-checked at a pre-contact pose before a fixture engagement.

- **In**: `reference_cloud` PointCloud, `functional_feature` FunctionalFeature, `object_description` string
- **Out**: `feature_in_tcp` Se3Pose, `object_in_tcp` Se3Pose, `attached_object` AttachedObject, `registration_confidence` float, `registration_method` string, `fallback_used` bool, `translation_uncertainty_m` float
- **Calls**: [`curobo`](#curobo), [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.get_ee_pose`
- **Scripts**: [`register_held`](skills/registering-held-objects/scripts/register_held.py)
- **Exits**:
  - `registered`: A feature-in-TCP transform and attachment were produced; registration_confidence is the accepted mask score, or 0.25 when the grasp-time or prior transform was retained.
  - `lost`: Registration raised — no cloud could be fitted into an attachment or a geometry fit failed, so the object is no longer localized.

### tracking-objects

perceive · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/tracking-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/tracking-objects)

Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. Seeds the tracker via text prompt on the first frame, then polls the stream at update_hz and advances via sam3.tracker_update until the workflow signals termination, publishing a tracker snapshot per tick. Use when a workflow needs the live mask + box of an object across many frames — e.g. a supervisor branch that monitors a target's location while a policy manipulates it.

- **In**: -
- **Out**: -
- **Calls**: [`sam3`](#sam3)
- **Functions**:
  - `tracking-objects.track`: Run the SAM3 tracker loop over the observation stream; returns the final mask/box/confidence.

### computing-feature-mating-poses

plan · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/computing-feature-mating-poses/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/computing-feature-mating-poses)

Computes approach, engaged, and mate TCP poses from a held feature in the hand and a fixture feature, resolving free mate symmetries from the current robot pose without executing motion. Use when a held loop must go over a shaft or hook, or a held shaft or tip must enter an aperture, and explicit approach, engaged, and mate poses are needed for planning.

- **In**: `held_feature_in_tcp` Se3Pose, `fixture_feature` FunctionalFeature, `relation` string, `attached_object` AttachedObject
- **Out**: `mate_pose` Se3Pose, `approach_pose` Se3Pose, `engaged_pose` Se3Pose, `approach_axis` Vec3, `seating_distance` float, `minimum_clearance` float
- **Calls**: [`geometry`](#geometry); connector `robot.get_ee_pose`
- **Scripts**: [`compute_mate`](skills/computing-feature-mating-poses/scripts/compute_mate.py)
- **Exits**:
  - `computed`: A valid mate was computed.
  - `infeasible`: The fixture feature has no directed axis or the feature dimensions cannot satisfy the relation.

### executing-keypoint-motion

plan · robots: any · tasks: tool-use, pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/executing-keypoint-motion/SKILL.md) · [upstream](https://github.com/moka-manipulation/moka/blob/main/moka/policies/franka_policy.py)

Plans and executes a one-step manipulation from a language instruction with MOKA's mark-based visual prompting - candidate keypoints on the named objects and a lettered waypoint grid are drawn on the overhead image, a VLM picks the grasp point, the function point, the target point and the pre/post-contact waypoint tiles and heights, depth makes them world points, and the arm grasps the object at the grasp point (or closes its hand) and carries the function point through pre-contact, target and post-contact. Use when a short instruction such as "push the cube to the right", "press the button", "sweep the cable aside with the stick" or "put the plug on the board" must become one motion without a task-specific skill, with a calibrated overhead RGB-D camera and a hand that is empty.

- **In**: `observation` Observation, `instruction` str
- **Out**: `final_function_point` Vec3, `hand_offset` Vec3, `held` bool
- **Calls**: [`gemini-er`](#gemini-er), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`
- **Scripts**: [`plan_keypoint_motion`](skills/executing-keypoint-motion/scripts/plan_keypoint_motion.py), [`execute_keypoint_motion`](skills/executing-keypoint-motion/scripts/execute_keypoint_motion.py)
- **Exits**:
  - `planned`: (plan only) The VLM's reply named offered marks and tiles, every chosen keypoint has depth, and `plan` holds the world points, heights, mode and hand heading; execute follows.
  - `no_plan`: (plan only) The instruction names no object, an object was not found (sam3 / gemini-er returned nothing), no candidate had depth, or every VLM reply (up to `attempts`) was unparsable or picked marks that were not offered; `detail` says which. Nothing moved.
  - `done`: (execute only) Every phase reached its point within reach_tolerance (read back from the arm), including the grasp (jaws above empty_fraction) when there is one; the hand then released and lifted. This is what the hand did, not what the object did - re-perceive before relying on it.
  - `grasp_failed`: (execute only) The jaws closed to empty_fraction or less at the grasp keypoint; the hand opened and rose back to the safe height.
  - `unreachable`: (execute only) A free move (the first lift, a hover above the grasp or the pre-contact point) or the descent to the grasp was refused or ended off with every rotation tried (nothing touched), or a contact move (pre-contact, target, post-contact) stopped short; `detail` names the phase. After a contact-phase shortfall the hand still released and lifted.

### planning-articulation-opening

plan · robots: any · tasks: articulated · MIT · v0.1.0 · [`SKILL.md`](skills/planning-articulation-opening/SKILL.md) · [upstream](https://github.com/mosart-system/code/blob/main/modules/motion_planning.py)

Plans the hand's path that opens a drawer or a door it already holds by the handle - a straight pull for a prismatic joint, or an arc about the hinge that turns the hand with the door so its grip on the handle never changes - and checks every waypoint with the arm's IK, chained from the previous solution, returning the reachable poses and joint trajectory. Use when a handle is grasped (or about to be) and its articulation is known (pull direction and distance, or hinge point, axis and opening angle), before an execution skill moves the arm along the path.

- **In**: `handle_pose` Se3Pose, `joint_type` string
- **Out**: `waypoints` PoseSequence, `all_poses` PoseSequence, `joint_trajectory` Trajectory, `reachable_fraction` float, `opening` float, `planned_opening` float, `radius` float
- **Calls**: connector `robot.solve_ik`
- **Scripts**: [`plan_articulation_opening`](skills/planning-articulation-opening/scripts/plan_articulation_opening.py)
- **Exits**:
  - `planned`: Every waypoint from the grasp to the full opening solved (or, with min_fraction under 1, the reachable prefix covers at least that share of the steps); `waypoints` and `joint_trajectory` hold the executable prefix.
  - `unreachable`: The grasp pose itself did not solve, or the reachable prefix falls short of min_fraction; `waypoints` holds whatever prefix did solve (possibly none), `reachable` marks every waypoint, and `detail` names the first failure.

### planning-held-object-motion

plan · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/planning-held-object-motion/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/planning-held-object-motion)

Plans the carry and engagement phases for an already-held rigid object: a clearance-first lift, the smallest feasible symmetry-equivalent reorientation, an orientation-locked transit above the fixture, or a direct plan to the approach pose; then a typed engagement or a straight linear insertion, optionally re-observing the held tip from the wrist first. Use when a held object must be brought to a fixture for insertion, hanging, packing, racking, or constrained sorting and an execution skill will run the resulting plans.

- **In**: `held_feature_in_tcp` Se3Pose, `fixture_feature` FunctionalFeature, `relation` string, `world_config` WorldConfig, `attached_object` AttachedObject
- **Out**: `reorientation_plan` PoseSequence, `placement_plan` PoseSequence
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `motion.plan_joint`, `robot.get_ee_pose`
- **Scripts**: [`plan_clearance_motion`](skills/planning-held-object-motion/scripts/plan_clearance_motion.py), [`plan_feature_engagement`](skills/planning-held-object-motion/scripts/plan_feature_engagement.py), [`plan_linear_engagement`](skills/planning-held-object-motion/scripts/plan_linear_engagement.py)
- **Exits**:
  - `planned`: A carry plan or an engagement plan was produced.
  - `blocked`: No feasible symmetry-equivalent held-object orientation was found, or the direct strategy was requested without an approach pose.

### following-a-visual-plan

propose · robots: any · tasks: pick-place, cable-routing, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/following-a-visual-plan/SKILL.md)

Follows a visual plan from robobrain.trace or molmo-motion.forecast (the shared visual-plan schema) the way those models can be trusted - the plan's endpoint pixel becomes a target lifted with the measured depth and the calibrated camera (not the model's depth), checked (on the table, in reach by IK preview, optionally against a pointing model for the target phrase), and geometry decides the motion - rise, travel at a height that clears the measured scene (optionally through the plan's path), hover above the target, descend to the caller's goal height or a clearance over the object's measured top. Optionally asks the planner once more from a fresh frame when a check fails. Use when a model proposes where something should go ("place the cube in the box", "move the cable end to the ring") and the hand should be brought there, above or into the target, before a placing, inserting or releasing step.

- **In**: `instruction` str
- **Out**: `target_point` Vec3, `goal_pose` Se3Pose, `hover_pose` Se3Pose, `approach_direction` Vec3, `target_obb` OrientedBoundingBox, `final_position` Vec3, `position_error` float
- **Calls**: [`gemini-er`](#gemini-er), [`locate-anything`](#locate-anything), [`molmo-motion`](#molmo-motion), [`robobrain`](#robobrain); connector `robot.describe_workspace`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`propose_target_from_plan`](skills/following-a-visual-plan/scripts/propose_target_from_plan.py), [`follow_proposal`](skills/following-a-visual-plan/scripts/follow_proposal.py)
- **Exits**:
  - `proposed`: (propose only) The plan's endpoint lifted to a target on the table, in reach (IK preview of the hover and the goal), and, with target_phrase, within cross_check_tolerance of the pointing model's point; `proposal` holds the goal, hover, travel height and waypoints. Nothing moved.
  - `no_plan`: (propose only) No plan was given and no instruction to ask for one, the planner raised or returned no track, or the track's endpoint is within 12 px of its start (no motion proposed - what MolmoMotion answers for a still scene); with replan, both plans. Nothing moved.
  - `plan_rejected`: (propose only) The endpoint is outside the image or has no depth, is off the table (its surface under the table plane, past workspace_bounds) or implausibly high over it, the route needs a travel height over the ceiling, no rotation reaches the hover and the goal (IK preview), or the cross-check found no target or disagrees; `detail` says which. With replan, both plans failed. Nothing moved.
  - `arrived`: (execute only) Every leg was accepted and a fresh pose reading puts the tool centre within reach_tolerance (15 mm) of the goal above the target. What the hand did; re-perceive before relying on what is under it.
  - `unreachable`: (execute only) A leg (rise, waypoint, hover, descend) raised or reported it did not arrive (a servo timeout counts only when the hand is further than reach_tolerance from the leg's end), or the hand ended further than reach_tolerance from the goal; `detail` names the leg.

### grasping-concave-contour

propose · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-concave-contour/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/grasping-concave-contour)

Grasp an awkward rigid part by the waists in its own silhouette — take the alpha shape of its point cloud from above, find where that boundary turns inward, and close the jaws across a pair of those inward runs. Use when a part has no graspable long axis and no pair of parallel walls an OBB would find — pliers and scissors gripped at the joint, a spool at its neck, a gear between two teeth, a bulb socket at its skirt. Use grasping-linear-feature instead for handles, rods and shafts, and grasping-short-axis when an upright box is enough.

- **In**: `target_obb` OrientedBoundingBox, `target_cloud` PointCloud
- **Out**: `grasp_pose` Se3Pose, `pregrasp_pose` Se3Pose, `grip_width` float, `confidence` float
- **Calls**: [`geometry`](#geometry); connector `robot.describe_gripper`, `robot.get_observation`, `robot.go_home`, `robot.go_to_pose`, `robot.open_gripper`
- **Scripts**: [`compute_concave_grasp`](skills/grasping-concave-contour/scripts/contour_concave_grasp.py)
- **Exits**:
  - `posed`: The wrist reached the computed pose with the jaws open, ready for the close a verifying skill will make.
  - `failed`: No pair of concave runs fits the jaw span, or the pose could not be reached.

### proposing-short-axis-grasps

propose · robots: any, **sim only** · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/proposing-short-axis-grasps/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/proposing-short-axis-grasps)

Propose top-down grasps whose jaws close ACROSS an elongated object's short axis, derived from its oriented box. Use when the object is longer than it is wide, as the first grasp-proposal rung — a handle, a spanner, a screwdriver, a bar, a laid-down bottle — where closing along the length would put the fingers down the object and meet them on each other. Declines when the short axis is wider than the hand opens, which is the signal to try `proposing-side-grasps` rather than to re-tune this one.

- **In**: `target_obb` OrientedBoundingBox
- **Out**: -
- **Calls**: connector `robot.describe_gripper`, `robot.grasp_frame`, `sim.get_object_obb`
- **Scripts**: [`propose_short_axis`](skills/proposing-short-axis-grasps/scripts/propose_short_axis.py)
- **Exits**:
  - `proposed`: A closing heading and one or more stations along the long axis.
  - `declined`: The short axis exceeds the jaw span. Route to another propose rung.

### proposing-side-grasps

propose · robots: any, **sim only** · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/proposing-side-grasps/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/proposing-side-grasps)

Propose horizontal (side-entry) grasps swept around an object's oriented box — twelve azimuths ranked by jaw margin. Use when top-down access is blocked by an overhead shelf, a lid or a bin wall, or when the object's vertical dimension is the only one the hand can span. Declines when the object is wider than the jaws from every azimuth.

- **In**: `target_obb` OrientedBoundingBox
- **Out**: -
- **Calls**: connector `robot.describe_gripper`, `robot.grasp_frame`, `sim.get_object_obb`
- **Scripts**: [`propose_side_grasps`](skills/proposing-side-grasps/scripts/propose_side_grasps.py)
- **Exits**:
  - `proposed`: One or more horizontal approaches, widest jaw margin first.
  - `declined`: No azimuth leaves a width the hand can span.

### matching-a-piece-to-placed-neighbours

select · robots: any · tasks: jigsaw · MIT · v0.1.0 · [`SKILL.md`](skills/matching-a-piece-to-placed-neighbours/SKILL.md) · [upstream](https://github.com/Kawaboongawa/Zolver/blob/master/Img/filters.py)

Picks which of several loose jigsaw pieces fits an empty slot, and how far to turn it, by the shape of its sides alone - each piece's mask is split at its 4 corners into sides typed flat / tab / blank (Zolver's turning-angle signature), then every piece in each of its 4 rotations is fitted rigidly to the sides the already-placed neighbours leave at the slot and ranked by how closely the outlines meet. Also returns each piece's class (corner / edge / interior) and its grip point (the inner point farthest from the outline), in board metres for the winner. Use for low-texture regions where the box picture cannot place a piece (white snow, black cloth, blurred background), or to pick between a picture match's candidates; needs top-down piece masks (e.g. wrist-camera crops) and at least one placed neighbour or a border side at the slot.

- **In**: -
- **Out**: `best_id` string, `best_rotation_deg` float, `best_score` float, `margin` float, `piece_class` string
- **Scripts**: [`match_piece`](skills/matching-a-piece-to-placed-neighbours/scripts/match_piece.py)
- **Exits**:
  - `fits`: One candidate fits clearly - its side types agree with the slot, its score is under max_fit (0.06) and the next candidate scores at least min_margin times worse (1.25; 1.6 with a single neighbour). best_id, best_rotation_deg and grip_point_board are the answer. With no neighbour, the only candidate whose side types fit the border sides.
  - `ambiguous`: The best fit does not stand out from the next candidate (or, with no neighbour, several candidates' side types fit); `ranked` lists them best first for another cue (the box picture, a test placement).
  - `no_fit`: No candidate's side types agree with the slot, the best score is over max_fit, or no candidate mask could be split into 4 sides (`pieces` says why per candidate); bad slot sides also land here.

### sampling-labeled-points-in-mask

select · robots: any, *tuned on* yam-station-overhead-rgbd · tasks: pick-place, cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/sampling-labeled-points-in-mask/SKILL.md) · [upstream](https://github.com/penn-pal-lab/maestro/blob/e60d326233a238ddc496bf7a08acb7a278e9e66d/maestro/api.py)

Picks the point on (or in) an object that a language instruction asks for - lays an evenly spaced grid of numbered markers inside the object's mask (the largest stride that leaves 35-45 inside, plus markers along its outline), draws them on an upscaled crop, asks a VLM which number(s) fit, and back-projects the chosen marker through depth to a world point. Use when a grasp, contact or placement point is a named part or region of one object ("where to grasp the mug handle", "the plug end of the cable", "a free spot on the tray") and a perception step already has, or sam3 can make, the object's mask in a calibrated RGB-D frame.

- **In**: `observation` Observation, `instruction` str
- **Out**: `point` Vec3, `marker_count` int
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3), [`vlm`](#vlm)
- **Scripts**: [`sample_labeled_points`](skills/sampling-labeled-points-in-mask/scripts/sample_labeled_points.py)
- **Exits**:
  - `selected`: The VLM named offered marker(s); `point` is the first one's world point (near-side depth inside the mask), `points` / `selected` all of them in the VLM's order with index, label and pixel.
  - `none_suitable`: The markers were drawn but the VLM answered [] (no marker fits), its replies (up to `attempts`) held no usable list or named numbers not offered, or no marker had depth; `detail` says which. `marked_image` shows what it was offered.
  - `not_seen`: No mask - the given one is empty or under 20 px, or sam3.segment_text found no instance of `object_name` scoring 0.3 or more.

### selecting-free-placement-spots

select · robots: any, *tuned on* yam-station-overhead-rgbd · tasks: pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/selecting-free-placement-spots/SKILL.md) · [upstream](https://github.com/moka-manipulation/moka/blob/f6b642514e5eb5f11f45d9da22af7b4b97514f6d/moka/planners/visual_prompt_utils.py)

Picks world-frame spots on a named surface where a held object can be set down - asks a pointing model for points in the vacant area (RoboPoint's question to gemini-er.point) or a VLM for free tiles of a grid drawn over the image (MOKA), back-projects each through calibrated depth, keeps those whose footprint is level, flat, on the surface and clear of anything standing on it, and ranks them by free room. Use when a place step needs a target and the task says only where in general ("put it somewhere free on the table", "on the board", "on the empty part of the tray"), from one overhead RGB-D frame.

- **In**: `observation` Observation, `region` string
- **Out**: `place_position` Vec3, `spot_count` int, `footprint_radius` float
- **Calls**: [`gemini-er`](#gemini-er), [`geometry`](#geometry), [`vlm`](#vlm)
- **Scripts**: [`select_free_spots`](skills/selecting-free-placement-spots/scripts/select_free_spots.py)
- **Exits**:
  - `found`: At least one model-proposed spot passed the depth check; place_position is the best, place_positions all of them, ranked.
  - `none_free`: The model named no spot or tile, or every candidate failed the depth check (tilted, occupied, edge, rough, no_depth); `detail` says which.

### selecting-reachable-grasp

select · robots: any, **sim only** · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/selecting-reachable-grasp/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/selecting-reachable-grasp)

Walk a ranked list of candidate grasp poses and take the first the arm can actually reach — IK solves it, the solution lands where it was asked, and (where `sim.clearance` is available) the arm is clear of the scene there. Use after any propose rung. This is the fix for taking candidate 0 on faith and closing the jaws on nothing.

- **In**: -
- **Out**: -
- **Calls**: connector `motion.plan_joint`, `robot.describe_arm`, `robot.solve_ik`, `sim.clearance`
- **Scripts**: [`select_reachable`](skills/selecting-reachable-grasp/scripts/select_reachable.py)
- **Exits**:
  - `selected`: A pose the arm reaches, with the joint configuration that reaches it.
  - `none_reachable`: No candidate solved. Route to another propose rung.

### selecting-safe-cable-grasp-segments

select · robots: any, *tuned on* yam-station-overhead-rgbd · tasks: cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/selecting-safe-cable-grasp-segments/SKILL.md) · [upstream](https://github.com/HAORAN808/dual-arm-cable-manipulation/blob/11b4201032074792fa04979eec918b3fbc536c27/perception/deliverable_perception_min/grasp_detection/core/safe_segments.py)

Ranks top-down grasp points along a cable's fitted centreline that keep the fingers off its ends (a larger margin at the plug end), off hidden stretches and crossings, off tight bends, and clear of obstacle boxes, other cables and the cable's own other stretches, then checks each with robot.grasp_frame (jaws across the local tangent) and robot.solve_ik and returns the reachable ones as grasp poses, best first. Use when a cable must be picked up somewhere along its length (to pull it, lay it, hand it over or lift it clear) and the centreline comes from curve.fit_centerline / curve.track_centerline, optionally with curve.endpoints naming the plug end.

- **In**: `centerline` Centerline
- **Out**: `pose` Se3Pose, `candidate_count` int, `checked` int, `detail` string
- **Calls**: connector `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`select_cable_grasps`](skills/selecting-safe-cable-grasp-segments/scripts/select_cable_grasps.py)
- **Exits**:
  - `found`: At least one safe window solved IK; `pose` is the best grasp, `poses` all of them (at most max_poses, at least min_spacing_m apart along the cable), `grasps` their metrics.
  - `none_safe`: No fingertip-wide stretch of the cable passed the geometry (ends, hidden, crossings, bends, clearance), or the centreline was missing or unordered; `rejected` gives the share of the cable each rule ruled out.
  - `none_reachable`: Safe stretches exist but robot.solve_ik reached none of the best max_checks with either wrist solution; re-perceive after moving the cable or use the other arm.

### anchoring-a-free-end

execute · robots: any, 2 arms · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/anchoring-a-free-end/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/anchoring-a-free-end)

Pins one end of a cable with the support hand before any crossing is attempted — asks the support-hand planner for the anchor grasp as joint-space legs, refuses it when its worst IK error is past the limit, streams the legs, closes the jaws to the plan's own width and dwells for the fingers' travel, and carries the solved chain and grasp orientation out on the exit so later pay-outs seed from them. Use when a deformable linear object must be held at one end so a sideways pull bends it instead of dragging the whole thing.

- **In**: -
- **Out**: `segment` int, `worst_error_m` float, `q_support` string, `quat` string
- **Calls**: connector `cable.plan_support`, `robot.execute_trajectory`, `robot.set_grip`, `robot.wait_steps`
- **Scripts**: [`anchor_far_end`](skills/anchoring-a-free-end/scripts/anchor_far_end.py)
- **Exits**:
  - `anchored`: The support hand holds the far end; `segment`, `q_support` and `quat` are bound for the pay-outs that follow.
  - `unanchored`: The planner returned no legs or a grasp past the IK error limit; nothing moved. Abort the weave rather than proceed one-handed.

### approaching-above-a-target

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/approaching-above-a-target/SKILL.md) · [upstream](https://github.com/graph-robots/graph-as-policy/tree/main/examples/steered_policy/graph_loop/scripts/approach)

Moves the hand to hover a set clearance above a target's oriented box, rising in place first and keeping the hand's current rotation, then checks from a fresh pose reading that it arrived. Use when a learned policy is about to take over and should start near the object from the rotation it was trained with, or whenever a graph needs the hand parked above an object before a step of its own; needs the target's OBB from a perception skill.

- **In**: `target_obb` OrientedBoundingBox
- **Out**: `hover_pose` Se3Pose, `position_error` float, `rotation_error_deg` float
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose`
- **Scripts**: [`approach_above_target`](skills/approaching-above-a-target/scripts/approach_above_target.py)
- **Exits**:
  - `above`: The tool centre stands within reach_tolerance (default 10 mm) of the point clearance (default 0.12 m) above the target's top, its rotation within rot_tolerance_deg (default 4 deg) of the one asked for.
  - `unreachable`: A move raised or reported that it did not arrive, or the hand ended further off in position or rotation; `detail` says which.

### executing-feature-mating

execute · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/executing-feature-mating/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/executing-feature-mating)

Executes a supplied feature-mating plan for an already-held rigid object - collision-aware planner legs tracked to millimetre precision, Cartesian servo for short corrections and fixture crossings, contact-controlled seating - then releases and retreats along the fixture axis or straight up. Use when the goal is a constrained mate such as a loop over a shaft, a shaft into an aperture, or a handle seated on a support; use transporting-objects for ordinary container drops.

- **In**: `placement_plan` PoseSequence
- **Out**: `final_pose` Se3Pose, `waypoint_count` int, `fallback_count` int, `registration_uncertainty_m` float, `released` bool, `retreat_pose` Se3Pose
- **Calls**: connector `motion.plan_linear`, `motion.plan_to_pose`, `robot.describe_arm`, `robot.execute_trajectory`, `robot.get_ee_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.open_gripper`, `robot.wait_steps`
- **Scripts**: [`execute_placement_plan`](skills/executing-feature-mating/scripts/execute_placement_plan.py), [`release_and_retract`](skills/executing-feature-mating/scripts/release_and_retract.py)
- **Exits**:
  - `seated`: The feature-mating waypoint sequence completed; the object is engaged with the fixture and still held.
  - `released`: The gripper opened at the mate and retreated clear of the released object.
  - `blocked`: A planner refused an engagement waypoint, or a waypoint or release motion failed before completion.

### executing-held-object-motion

execute · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/executing-held-object-motion/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/executing-held-object-motion)

Executes an ordered, collision-aware pose sequence for an already-grasped rigid object without releasing it - the escape from support, clear-space rotation, fixture transit and orientation-locked local legs of a carry plan, or only the free-space approach waypoint of a placement plan. Use when a held object must be carried or reoriented before insertion, mating, packing, or sorting, and use executing-feature-mating afterwards for the engagement and release.

- **In**: `reorientation_plan` PoseSequence
- **Out**: `final_pose` Se3Pose, `approach_pose` Se3Pose, `fallback_count` int, `registration_uncertainty_m` float
- **Calls**: connector `motion.plan_linear`, `motion.plan_to_pose`, `robot.execute_trajectory`, `robot.get_ee_pose`, `robot.go_to_pose_cartesian`
- **Scripts**: [`execute_reorientation`](skills/executing-held-object-motion/scripts/execute_reorientation.py), [`execute_approach_only`](skills/executing-held-object-motion/scripts/execute_approach_only.py)
- **Exits**:
  - `reoriented`: The held-object waypoint sequence completed while the gripper remained closed.
  - `reached`: Only the first (approach) waypoint of a placement plan was executed; the object is held at the approach pose.
  - `blocked`: A planner refused a waypoint, or a waypoint motion failed before the requested pose was reached.

### grasping-a-cable-by-its-plug

execute · robots: any · tasks: cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/grasping-a-cable-by-its-plug/SKILL.md) · [upstream](https://github.com/zzigak/yam-arcade/blob/2804ddeb49ca1f00cb61b8a73c65b4a58e0190e7/stations/cables/run.py)

Grasps a cable by the rigid plug at its end rather than by its thin jacket - finds the plug body behind the cable's tip (curve.endpoints, optionally refined by gemini-er.point on the plug and the depth under it), comes down top-down with the jaws closing across the plug's short axis and the fingertips just over the plug's bottom (never over its mid-height, never under the station's tool-centre floor), closes, lifts 4 cm and checks the jaw gap still matches the plug's width (and, given a description, the plug seen to rise in the hand). Use when a charging, USB or audio cable must be picked up to be plugged in or routed and its end has a moulded plug lying flat on the table.

- **In**: -
- **Out**: `located_by` string, `grasp_point` Vec3, `close_heading_deg` float, `plug_height` float, `plug_width` float, `planned_tool_z` float, `fingertips_over_table` float, `jaw_width` float, `jaw_width_lifted` float, `plug_rise` float, `lift_checked` bool
- **Calls**: [`gemini-er`](#gemini-er), [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`grasp_plug`](skills/grasping-a-cable-by-its-plug/scripts/grasp_plug.py)
- **Exits**:
  - `grasped`: The jaws closed on the plug (the gap no more than width_slack, 4 mm, under the plug's width) and still read so after the 4 cm test lift; with a plug_description the plug was also seen at least min_rise (2 cm) up in the hand (lift_checked true). The hand is left lifted, closed on the plug.
  - `missed`: The descent stopped on the plug or the cable, the jaws closed to their stop or under the plug's width (the jacket or the plug's edge taken), closed further with the lift (the plug slipped out), or the camera saw the plug stay down; the hand opened and stands over the plug. Re-perceive before another try.
  - `unreachable`: The plug is not resting flat on the table, the fingertips at the tool-centre floor would stand over the plug's mid-height, the plug is wider than the hand opens, no wrist solution reaches it (IK preview), or a rise, the approach, the wrist turn or the descent was refused; `detail` says which.

### grasping-a-cable-lying-flat

execute · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-a-cable-lying-flat/SKILL.md) · [upstream](https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_franka_cable_ik_pick_place.py)

Grasps a thin cable lying flat on a table - top-down over a point on its centreline with the jaws closing across its local tangent, the fingertips brought down until they meet the table (the arm's stall, no force sensor), optionally narrowed first and held pressed on the table while they shut, then a test lift that checks the cable came up with the hand. Use when a cable, wire or cord a few millimetres thick lies on its support (from curve.fit_centerline / curve.endpoints, or a given point and direction) and must be picked up, e.g. to route or plug it.

- **In**: -
- **Out**: `grasp_point` Vec3, `close_heading_deg` float, `planned_tool_z` float, `contact_z` float, `fingertips_over_table` float, `touched_table` bool, `pressed` bool, `pre_closed` bool, `jaw_width` float, `cable_rise` float, `lift_checked` bool
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_cartesian_until_contact`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`grasp_flat_cable`](skills/grasping-a-cable-lying-flat/scripts/grasp_flat_cable.py)
- **Exits**:
  - `grasped`: The jaws closed short of their closed stop and, with a cable_description, the cable beside the fingers was seen to rise at least min_rise (7 mm) with the test lift (lift_checked true); without one it is judged by the jaw gap alone (lift_checked false), which on a real YAM does not tell a cable caught on the fingertips from a held one. The hand is left lifted, closed.
  - `missed`: The jaws closed to their stop, or the cable beside the fingers did not rise with the hand; the hand opened and stands over the cable.
  - `unreachable`: The cable does not lie flat at the grasp point, the depth the fingertips need is under the tool-centre floor (min_tool_height), no wrist solution reaches the grasp pose (IK preview), or a rise, the approach, the wrist turn or the descent was refused; `detail` says which.

### grasping-container-rims

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-container-rims/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/grasp_rim/robotwin)

Grasps a thin-walled open container (box, bin, tray, cup, bowl) by its rim - picks a wall midpoint the arm can reach, turns the open jaws across the wall (one finger inside the mouth, one outside), hovers above, comes straight down until the pads straddle the wall below the rim top, closes, checks the jaw gap reads about the wall's thickness (not empty, not wider), and lifts a little; tries the other walls when one misses. Use when an open container must be picked or held and a body grasp would close on the empty inside (or the container is wider than the open hand), with its oriented box from a perception skill and the hand empty.

- **In**: `container_obb` OrientedBoundingBox
- **Out**: `side` string, `grasp_position` Vec3, `close_heading_deg` float, `jaw_error_deg` float, `wrist_turn_deg` float, `grip_width` float, `held_width` float, `wall_thickness` float, `lifted` float, `attempts` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`grasp_rim`](skills/grasping-container-rims/scripts/grasp_rim.py)
- **Exits**:
  - `grasped`: The jaws closed across a wall with the gap above empty_margin and at most wall_thickness + full_margin (max_wall when the thickness is unknown), and still did after the lift; the hand holds the rim. `lifted` says how far the hand rose (short when the container is heavy or fixed; `detail` says so).
  - `missed`: At least one wall was tried down to the grasp height but every close read empty or too wide, the wall slipped out during the lift, or the descent stopped on the rim; the hand is open.
  - `unreachable`: No wall could be tried - the container is too shallow or too narrow for the open hand, every grasp pose failed the IK preview, the hover was refused, or the jaws could not be turned across the wall; nothing was closed.

### grasping-direct-ik

execute · robots: any, *tuned on* panda · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-direct-ik/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/grasping-direct-ik)

Direct IK align-then-descend grasping. The gripper pre-rotates to the grasp orientation at a safe height ABOVE the target before descending straight down, avoiding the twist-while-closing failure mode of a blended rotate+descend. The grasp rotation is rebuilt so the hand's declared closing axis (robot.describe_gripper, composed by robot.grasp_frame) closes across the target's short horizontal axis, and the hover clearance is the hand's own (robot.describe_workspace) unless the workflow pins it. Use when no trajectory planner (curobo) is deployed or the scene is uncluttered enough that a straight-line approach is safe.

- **In**: `target_obb` OrientedBoundingBox
- **Out**: -
- **Calls**: [`curobo`](#curobo), [`geometry`](#geometry); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`
- **Scripts**: [`compute_align_pose`](skills/grasping-direct-ik/scripts/compute_align_pose.py), [`refine_top_down_grasp`](skills/grasping-direct-ik/scripts/refine_top_down_grasp.py), [`execute_grasp_align`](skills/grasping-direct-ik/scripts/execute_grasp_align.py), [`plan_to_pose`](skills/grasping-direct-ik/scripts/plan_to_pose.py)
- **Exits**:
  - `grasped`: Object held in the gripper after `close`.
  - `failed`: Any failure during the grasp attempt — planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort.

### grasping-flat-objects

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-flat-objects/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/grasp_flat/robotwin)

Grasps an object lying flat on a surface that is too thin for a normal top-down pinch at its box centre (a card, a phone, a thin plate or lid, a coin if wide enough) - turns the jaws across its short side, brings the fingertips down to just above the support (as low as the tool-centre floor allows), closes, checks the jaw gap reads about the slab's width, lifts and checks again; retries once across the long side. An optional tilt brings one fingertip in along the support at the slab's edge (it did not hold in a station twin). Use when a flat object must be picked from a table and a grasp at its box centre would close on air above it, with its oriented box from a perception skill and the hand empty.

- **In**: `object_obb` OrientedBoundingBox
- **Out**: `axis` string, `close_heading_deg` float, `tilt_deg` float, `grasp_position` Vec3, `tip_height` float, `object_width` float, `thickness` float, `jaw_error_deg` float, `wrist_turn_deg` float, `grip_width` float, `held_width` float, `lifted` float, `attempts` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`grasp_flat`](skills/grasping-flat-objects/scripts/grasp_flat.py)
- **Exits**:
  - `grasped`: The jaws closed across the slab, the gap within narrow_margin under and full_margin over its width across the jaws (and above empty_margin), and still did after the lift; the hand holds it. `lifted` says how far the hand rose (short when the object is heavy or caught; `detail` says so).
  - `missed`: At least one axis was tried down to the grasp height but every close read empty, too wide or too narrow (not across the slab), the slab slipped out during the lift, a tilted grasp's closing steps were refused, or the descent stopped on the slab's top; the hand is open.
  - `unreachable`: No axis could be tried - the slab is too thin for the fingertips at this tilt (they cannot come below its top), wider than the open hand both ways, every grasp pose failed the IK preview, the hover was refused, or the jaws could not be turned; nothing was closed.

### grasping-linear-feature

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-linear-feature/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/grasping-linear-feature)

Fit the full 3D axis of an elongated segmented feature and grasp it with a perpendicular, inclination-aware parallel-jaw pose. Use for handles, rods, shafts, tools, utensils, and other linear parts whose pitch or roll matters; use grasping-short-axis instead when an upright OBB and vertical approach are sufficient.

- **In**: `target_cloud` PointCloud
- **Out**: `grasp_pose` Se3Pose, `pregrasp_pose` Se3Pose, `approach_axis` Vec3, `ee_pose_at_grasp` Se3Pose
- **Calls**: [`geometry`](#geometry); connector `robot.close_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.open_gripper`
- **Scripts**: [`compute_grasp`](skills/grasping-linear-feature/scripts/axis_aware_grasp_pose.py), [`compute_grasp_top_down`](skills/grasping-linear-feature/scripts/axis_grasp_top_down.py)
- **Exits**:
  - `grasped`: The gripper closed after reaching the inclination-aware grasp pose.
  - `failed`: Feature fitting or full-orientation motion failed.

### grasping-short-axis

execute · robots: any, *tuned on* panda · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-short-axis/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/grasping-short-axis)

Deterministic short-axis-aligned grasp with CuRobo. The grasp pose is computed geometrically (NOT sampled/scored) — the gripper approaches straight down with its finger-opening axis locked to the OBB's SHORTER horizontal axis, so the jaws close ACROSS the narrow dimension of an elongated target. Use for pan/pot handles, bottles, tools, utensils — anywhere the OBB centroid is graspable but orientation is the thing that matters. An optional offset_from_base node slides the grasp outward along the handle, clear of the heavier object body, when a second OBB of that body is wired in. A scored/sampled pose often holds at close but shears out under the lift; the short-axis pose grips the bar squarely so it survives lift + transport.

- **In**: `target_obb` OrientedBoundingBox, `target_mask` Mask
- **Out**: `ee_pose_at_grasp` Se3Pose, `grasp_pose` Se3Pose
- **Calls**: [`curobo`](#curobo), [`geometry`](#geometry); connector `robot.close_gripper`, `robot.execute_trajectory`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`compute_grasp`](skills/grasping-short-axis/scripts/short_axis_grasp_pose.py), [`offset_from_base`](skills/grasping-short-axis/scripts/offset_grasp_from_base.py), [`approach_above`](skills/grasping-short-axis/scripts/approach_above.py), [`build_world`](skills/grasping-short-axis/scripts/build_world.py), [`plan_grasp`](skills/grasping-short-axis/scripts/plan_grasp.py), [`plan_grasp_motion`](skills/grasping-short-axis/scripts/plan_grasp_motion.py), [`finalize_trajectory`](skills/grasping-short-axis/scripts/finalize_trajectory.py)
- **Exits**:
  - `grasped`: Object held in the gripper after `close`.
  - `failed`: Any failure during the grasp attempt — collision-aware planning failure or trajectory execution error (a raise propagating to `on_error`). Coordinator routes to abort.

### grasping-with-planner

execute · robots: any, *tuned on* panda · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-with-planner/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/grasping-with-planner)

Top-down grasping via a fast axis-locked linear descend with a collision-aware cuRobo fallback. A single `grasp_descend_linear` node rises, translates over the target, and descends straight down (Z-locked, orientation held) onto the object; if the straight-line solve is infeasible (far-edge reach, no IK for the fixed wrist) it falls back to the cuRobo planner over the candidate fan. Use when the curobo tool bundle is installed and a perceived object (OBB) must be grasped — the default grasping skill whenever cuRobo is deployed, in clean and cluttered scenes alike.

- **In**: `target_obb` OrientedBoundingBox
- **Out**: `ee_pose_at_grasp` Se3Pose, `grasp_pose` Se3Pose
- **Calls**: [`curobo`](#curobo), [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.close_gripper`, `robot.execute_trajectory`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose_cartesian`, `robot.open_gripper`
- **Scripts**: [`grasp_descend_linear`](skills/grasping-with-planner/scripts/grasp_descend_linear.py), [`select_short_axis`](skills/grasping-with-planner/scripts/select_short_axis.py)
- **Exits**:
  - `grasped`: Object held in the gripper after `close`.
  - `failed`: Any failure during the grasp attempt — the cartesian descend AND the collision-aware planner fallback both failed (a raise propagating to `on_error`). Coordinator routes to abort.

### handing-over-between-arms

execute · robots: any, 2 arms · tasks: pick-place, bimanual · MIT · v0.1.0 · [`SKILL.md`](skills/handing-over-between-arms/SKILL.md) · [upstream](https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/handover_block.py)

Hands the object one arm holds to the other arm - the donor carries it to a handover point both arms reach (keeping its rotation), the receiver pre-poses back along a horizontal approach, moves in on the side of the object away from the donor's hand and closes; only when the receiver's jaw opening says it holds something does the donor open, and the hold is rechecked before and after the donor backs out. Use when a bimanual station must pass a held object between its arms (regrasp from the other side, reach a place only the other arm reaches, free the first arm), with the held object's box from a perception skill, both arms addressable by arm_id and the receiver empty.

- **In**: `object_obb` OrientedBoundingBox, `handover_position` Vec3
- **Out**: `receiver_grip` float, `receiver_roll_error_deg` float, `donor_handover_pose` Se3Pose, `receiver_grasp_pose` Se3Pose, `object_position` Vec3
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`hand_over`](skills/handing-over-between-arms/scripts/hand_over.py)
- **Exits**:
  - `transferred`: The receiver closed on something (jaw opening above the stop), the donor opened, the receiver still held after the donor backed out retreat_distance and after it drew back receiver_withdraw; object_position is where the receiver's draw-back should have left the object's centre.
  - `receiver_missed`: The receiver could not reach its pre-grasp pose with either wrist solution, stopped short moving in, or closed to the stop; the receiver opened and backed off, the donor never opened and still holds the object at the handover pose.
  - `dropped`: The donor's jaws were at the stop at the start (nothing held), or the receiver's jaws ran to the stop after the donor opened, while it backed out, or as the receiver drew back; `detail` says when.
  - `donor_stuck`: The donor could not reach the handover pose (it still holds the object), its jaws did not open (both hands hold it), or it opened but could not back out; also routed when donor_arm equals receiver_arm or the connector reports no separate receiver arm.

### holding-steady-while-other-arm-acts

execute · robots: any, 2 arms · tasks: pick-place, bimanual · MIT · v0.1.0 · [`SKILL.md`](skills/holding-steady-while-other-arm-acts/SKILL.md) · [upstream](https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/put_object_cabinet.py)

One arm holds an object still - a top-down grasp, or its closed hand pressed down on the object's top - while the other arm does a step, then the holder checks it still holds (its jaw opening and its own pose unchanged since the hold, and optionally a re-perceived box of the object) and lets go. Three scripts with clean entry and exit, hold, act and release, so a graph can put any skill between hold and release; act is a simple step for the other arm (a straight-line move, or a press whose last leg may end on contact). Use when a bimanual station must fix a part, fixture or container with one hand while the other pushes, inserts, wipes or places on it, with the object's box from a perception skill and both arms addressable by arm_id.

- **In**: `object_obb` OrientedBoundingBox, `actor_target` Vec3
- **Out**: `holder_grip` float, `stopped_short` float, `holder_shift` float, `grip_change` float
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`hold`](skills/holding-steady-while-other-arm-acts/scripts/hold.py), [`act`](skills/holding-steady-while-other-arm-acts/scripts/act.py), [`release`](skills/holding-steady-while-other-arm-acts/scripts/release.py)
- **Exits**:
  - `holding`: (hold only) The holder grips the object (jaws above the stop) or presses on it (stopped short of the press target by at least half the press depth); `hold` records its read-back pose and jaw opening. The entry point for the step between hold and release.
  - `acted`: (act only) The other arm reached its start, every via point and the target (move), or stopped on contact on the last leg (press, `contact` true) or reached it; then it backed out to its start unless `detail` says it could not. The step's exit; release follows.
  - `held_and_released`: The holder held (jaws above the stop, or a press that stopped on the object), the other arm's step ended `acted`, and at release the holder's tool centre was within pose_tolerance and angle_tolerance_deg of the hold pose and (grip) its jaw opening within grip_tolerance of the one at the hold; it then opened and backed out to its stand-off.
  - `hold_lost`: The holder's jaws closed to the stop at the hold (missed), its press went down to the target (nothing under it), or at release its pose or jaw opening had changed past the tolerances (or a re-perceived object box had moved); the holder has let go and backed out.
  - `unreachable`: The holder could not reach above the object or down onto it, the connector does not report a separate second arm, the other arm could not reach its start or a waypoint (move mode, or the press target failed the IK preview), or the holder could not back out after letting go; `detail` says which. After a failed step the hold is still checked and released.

### inserting-a-peg-with-pose-refinement

execute · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/inserting-a-peg-with-pose-refinement/SKILL.md) · [upstream](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py)

Inserts a held peg into a perceived hole in three stages - carries it to a pre-insert pose with the tip a few centimetres in front of the mouth on the hole's axis, refines that pose up to three times from the residual between the peg as observed (hand read back times the registered tip-in-hand transform) and the axis, then pushes straight in along the axis with a stop-on-contact move and judges how far the tip went. Use when the hand already holds a peg, pin, plug or shaft whose tip is registered in the hand (registering-held-objects), the hole's mouth and inward axis are measured (perceiving-functional-features), and the arm does not land exactly where it is sent; any axis direction (side or top insertion).

- **In**: `hole_centre` Vec3, `hole_axis` Vec3, `peg_in_tcp` Se3Pose
- **Out**: `pre_insert_pose` Se3Pose, `tcp_at_mouth` Vec3, `lateral_error` float, `axial_error` float, `angle_error_deg` float, `refinements` int, `correction` float, `depth` float, `tip_position` Vec3, `deflection` float
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.solve_ik`
- **Scripts**: [`align_pre_insert`](skills/inserting-a-peg-with-pose-refinement/scripts/align_pre_insert.py), [`push_insert`](skills/inserting-a-peg-with-pose-refinement/scripts/push_insert.py)
- **Exits**:
  - `aligned`: (align_pre_insert run on its own, e.g. before a re-registration) The tip is within the tolerances of the pre-insert pose on the hole's axis; in the example subgraph this route goes on to the push and never leaves.
  - `inserted`: The push took the tip, as held, at least min_depth (10 mm) past the mouth; the hand stays at the end of the push, closed on the peg.
  - `stopped_short`: The push stopped (contact, stall, or the hand shoved deflection_stop off its line) with the tip short of min_depth past the mouth (rim, a hole off its estimate, a jam), or the hold leaves no room to push that far with the fingers clear; the hand backed off to the pre-insert pose, still holding.
  - `misaligned`: After the refinements the tip is still more than lateral_tolerance off the axis (or axial / angle tolerance), or the correction grew past its cap; the hand is at the last pre-insert pose and nothing was pushed.
  - `unreachable`: The pre-insert pose or the end of the push is out of reach by the IK preview, a move was refused, or the hand's pose could not be read; nothing was pushed.

### lifting-with-both-arms

execute · robots: any, 2 arms · tasks: pick-place, bimanual · MIT · v0.1.0 · [`SKILL.md`](skills/lifting-with-both-arms/SKILL.md) · [upstream](https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/lift_pot.py)

Lifts an elongated object (a bar, a tray, a pot by its handles) with both arms - one near-top-down grasp near each end of the object's long axis (or at two given handle points), each arm taking the end nearer its hand, its wrist leaned 15 deg away from the other hand so the two wrists keep clear, then both hands rise together in small interleaved steps that keep their separation and height difference bounded, with both jaw openings and the separation checked after every step. Use when an object is too long, heavy or floppy for one hand and a bimanual station has both arms addressable by arm_id, the object's box comes from a perception skill and both hands are empty.

- **In**: `object_obb` OrientedBoundingBox
- **Out**: `rise_m` float, `hand_tilt_deg` float, `separation_change_m` float, `first_grip` float, `second_grip` float, `first_grasp_pose` Se3Pose, `second_grasp_pose` Se3Pose, `object_obb` OrientedBoundingBox
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`lift_with_both_arms`](skills/lifting-with-both-arms/scripts/lift_with_both_arms.py)
- **Exits**:
  - `lifted`: Both hands rose lift_height in steps, every step arrived within level_tolerance of its height, both jaw openings stayed above the stop and the hands' separation within separation_tolerance of its value at the grasp; object_obb is the input box raised by the hands' mean rise (the hands' motion, not a re-perception).
  - `slipped`: A jaw closed to the stop at its grasp point (both hands then opened and backed off to their hovers), or during the lift a jaw ran to the stop or the hands' separation drifted past separation_tolerance; mid-lift the hands stay where they stopped, closed, and `detail` says which arm and at what height.
  - `unreachable`: No hover, wrist solution or descent for one arm (hands opened and backed off), or an arm could not follow a lift step (the other was brought back level with it, both still closed on the object); also routed when the two arm ids are equal, the connector reports no separate second arm, or the box's long axis is not horizontal.

### moving-both-arms-together

execute · robots: any, 2 arms · tasks: pick-place, bimanual · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/moving-both-arms-together/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/move_dual_arm/robotwin/policy.py)

Moves both hands to their target poses as one coordinated motion - both straight paths are split into the same number of short segments and the arms take turns (the lead alternating) so neither runs ahead, every waypoint of both arms is previewed with IK before anything moves, each arm's arrival is read back per segment, and, when asked, the hands' relative pose is held within a bound (the second hand's waypoints ride on the first's). Use when a bimanual station must move both hands at once - a two-handed carry or lift, bringing both hands to a handover or pour, mirrored reaches - through connectors that drive one arm per call.

- **In**: `left_target` Se3Pose
- **Out**: `segments` int, `max_relative_error_m` float, `max_relative_error_deg` float, `max_transient_error_m` float, `left_residual_m` float, `right_residual_m` float, `min_hand_distance_m` float, `roll_turns` int, `left_pose` Se3Pose, `right_pose` Se3Pose, `right_target_pose` Se3Pose
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.solve_ik`
- **Scripts**: [`move_both_arms`](skills/moving-both-arms-together/scripts/move_both_arms.py)
- **Exits**:
  - `reached`: Both hands arrived at every waypoint within reach_tolerance (read back from each arm), the last being their targets; with keep_relative the hands' relative pose stayed within the bound at every segment end.
  - `diverged`: One hand stopped more than reach_tolerance from a waypoint (refused, raised, stalled), or with keep_relative the hands' relative pose left the bound at a segment end, after at least one hand had moved; both arms stop where they are, `detail` says which and where.
  - `unreachable`: Nothing moved - a waypoint failed the IK preview, the planned tool centres come closer than min_hand_distance, the targets break the kept relative pose, the move needs more than max_segments, the first move was refused in place, both arm ids name one arm, or the connector reports no separate second arm.

### picking-and-placing-two-objects

execute · robots: any, 2 arms · tasks: pick-place, bimanual · MIT · v0.1.0 · [`SKILL.md`](skills/picking-and-placing-two-objects/SKILL.md) · [upstream](https://github.com/robotwin-Platform/RoboTwin/blob/main/envs/place_dual_shoes.py)

Picks two objects with two arms, one each, and places both - each object goes to the arm whose IK preview reaches its whole pick-and-place plan (the nearer arm when both do), both are grasped top-down with the two arms' hover / descend / close / lift steps interleaved, carried to their place points, lowered, released, and the arms go back to where they started; every move is gated on the two tool centres staying min_hand_gap apart, so where the arms would meet one finishes while the other waits. Use when a bimanual station must move two objects at once (two parts into a box or a tray, two items to their marks), with both objects' boxes from a perception skill, both arms addressable by arm_id and both hands empty.

- **In**: `object_a_obb` OrientedBoundingBox, `object_b_obb` OrientedBoundingBox, `place_a` Vec3, `place_b` Vec3
- **Out**: `arm_a` int, `arm_b` int, `outcome_a` string, `outcome_b` string, `failed_arms` string, `min_hand_gap_m` float, `staggered` bool, `release_pose_a` Se3Pose, `release_pose_b` Se3Pose
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`pick_place_two`](skills/picking-and-placing-two-objects/scripts/pick_place_two.py)
- **Exits**:
  - `both_placed`: Both arms held their object by the jaw opening after the lift and after the lowering, opened at the release height over their place points; failed_arms is empty. What the hands did, not where the objects ended - re-perceive (verifying-placement) before relying on it.
  - `one_placed`: One arm placed its object, the other did not; failed_arms names it and outcome_a / outcome_b say why (grasp_missed, dropped, move_refused, clearance).
  - `none_placed`: Neither arm placed its object; failed_arms names both, outcome_a / outcome_b say why.
  - `unreachable`: No assignment of the two objects to the two arms has an IK preview reaching every pose of both plans, the arm ids are equal, or the connector reports no separate second arm; failed_arms names the arm(s) without a plan and nothing moved.

### placing-held-feature

execute · robots: any, **sim only** · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/placing-held-feature/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/placing-held-feature)

Move a held object so one of its declared features lands on a target point — a ring onto a hook, a tip into a hole, a plug at a bore. Composes the hand pose from the live object pose and the feature's centre in the object's own frame, recomputed each round so in-jaw slip cannot compound. Use for any goal written as "this object's feature at that place"; do NOT drive the hand to the target directly, which is wrong by the length of the held object.

- **In**: `target` Vec3, `object_name` str, `feature_local` Vec3
- **Out**: -
- **Calls**: connector `robot.get_ee_pose`, `sim.get_object_pose`
- **Scripts**: [`mate_feature`](skills/placing-held-feature/scripts/mate_feature.py)
- **Exits**:
  - `ready`: A hand pose that puts the feature on the target, and the move it applies.
  - `unknown`: A pose or a point could not be read; nothing was moved.

### placing-into-container-held-by-other-arm

execute · robots: any, 2 arms · tasks: pick-place, bimanual · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/placing-into-container-held-by-other-arm/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/place_held_in_held_container/robotwin/policy.py)

Puts the object one arm holds into an open container the other arm holds - the holder carries the container to a presentation point both arms reach without turning it (so it stays upright), the carrier goes above the container's opening, away from the holder's fingers (tilts that lean its wrist away from the holder first, then its own rotation, then RoboRSI's other tilts, each previewed with IK), comes straight down to a release height over the rim, opens, backs up and closes once to read that the object left its jaws; the holder's jaws are never touched. Use when a bimanual station must drop a held part into a cup, bowl or box that the other arm is holding (sort into a held bin, fill a container before carrying it away), with both held things' boxes from a perception skill and both arms addressable by arm_id.

- **In**: `container_obb` OrientedBoundingBox, `object_obb` OrientedBoundingBox
- **Out**: `chosen_rotation` str, `container_tilt_deg` float, `container_position` Vec3, `release_position` Vec3, `hand_gap` float, `container_pose` Se3Pose
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`place_in_held_container`](skills/placing-into-container-held-by-other-arm/scripts/place_in_held_container.py)
- **Exits**:
  - `placed`: The carrier released the object over the container's opening (its centre inside the opening less rim_margin, measured against where the holder's pose now puts the container), its jaws then closed to the stop (empty), and the holder's jaws still read a hold; the holder is at the presentation pose and can carry on. What the hand did, not a sighting of the object in the container.
  - `missed`: The carrier held nothing at the start, its jaws did not open, or they closed on something after the release (the object stuck to them); or the holder's jaws ran to the stop (the container fell) or the holder ended where the release was outside the opening.
  - `unreachable`: Nothing was released. The two arm ids are equal or the connector reports no separate second arm, the holder holds nothing, the container leans past max_tilt_deg (before or after the presentation), the holder could not reach the presentation pose, no carrier rotation fits the object in the opening, keeps its edge holder_keepout from the holder's tool centre and reaches the approach, or the straight descent stopped short (the carrier went back up still holding the object); also an unknown name in rotations.

### placing-into-container-obb

execute · robots: any · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/placing-into-container-obb/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/place_obb/robotwin/policy.py)

Puts the object the hand holds into an open bin, box or tray the camera sees - lifts it straight up, segments the named container with sam3, fits its box and its opening (rim height, opening centre, inner extents, floor) from depth, picks a drop point inside the opening where the held object's footprint keeps a margin from the walls, comes over it above the rim, straight down until the object is just over the container's floor, opens and retreats straight up. Routes no_room when the object fits nowhere and keeps it when no drop point can be reached. Use when a held part must be dropped into a fixed open container on the table (sorting into a bin, filling a box), with the held object's box measured after the grasp; follow with verifying-placement on the aperture_center and rim_z it outputs.

- **In**: `container_description` string, `held_obb` OrientedBoundingBox
- **Out**: `container_obb` OrientedBoundingBox, `aperture_center` Vec3, `rim_z` float, `floor_z` float, `opening_obb` OrientedBoundingBox, `opening_source` string, `release_position` Vec3, `candidate` string
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`place_into_container`](skills/placing-into-container-obb/scripts/place_into_container.py)
- **Exits**:
  - `placed`: The hand reached a release pose inside the container's inner opening (the held object's footprint keeping footprint_margin from the walls, its bottom release_height over the floor seen from depth), opened (its jaws read open) and retreated. What the hand did, not a sighting of the object in the container - run verifying-placement on aperture_center and rim_z.
  - `no_room`: The held object's footprint plus footprint_margin on each side is wider than the inner opening along one of its axes; the hand lifted and still holds the object.
  - `unreachable`: Every drop point failed - out of reach in the IK preview, the move over it refused or short, or the straight descent stopped above the release (on the rim, a wall or the contents; the hand went back up); still holding.
  - `container_not_found`: No RGB-D camera, no sam3 mask of container_description at score_min (or under container_pixel), a mask over 40% of the frame, or fewer than 30 depth points on it; the hand lifted and still holds the object.
  - `not_holding`: The jaws read at the stop before anything moved; nothing moved.
  - `not_released`: At the release pose the jaws still read closed after open_gripper; the object may still be in the hand, inside the container.

### placing-with-contact-descent

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/placing-with-contact-descent/SKILL.md) · [upstream](https://github.com/google-research/ravens/blob/master/ravens/tasks/primitives.py)

Places the object the hand holds by approaching a fixed height above its place point, coming straight down until the arm stops on contact, and releasing there before retreating straight up - Ravens' pick-place discipline, with the arm's stall detection standing in for a contact sensor. Contact well above the expected height (stacked on the wrong thing, a rim) or no contact at all is routed and the object kept. Use when a held object must be set down on a support or into an open container whose height is known only to a centimetre or so (a camera's estimate of a table, a box floor, the top of a stack), with its box measured after the grasp.

- **In**: `place_position` Vec3
- **Out**: `released` bool, `expected_contact_z` float, `contact_z` float, `contact_offset` float, `placed_position` Vec3
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`place_with_contact_descent`](skills/placing-with-contact-descent/scripts/place_with_contact_descent.py)
- **Exits**:
  - `placed`: The descent stopped on contact inside the contact window (or, with table_z, ended at the tool-centre floor with the object's bottom at or under the table), the hand opened and retreated straight up; placed_position is where the hand left the object's bottom centre.
  - `contact_too_early`: The descent stopped more than contact_tolerance (12 mm) above the expected contact; the hand did not release and lifted the object back to the approach height.
  - `no_contact`: The descent reached its bottom without stopping (no support there, or the object slid up the jaws); the hand did not release and lifted back.
  - `blocked`: The expected contact is out of reach (IK preview), the approach or a descent move was refused, or the retreat after the release failed (`released` says which side of the release); `detail` says which.

### plugging-a-cable-end

execute · robots: any, 2 arms · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/plugging-a-cable-end/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/plugging-a-cable-end)

Carries a cable's held free end into a terminal port after the last crossing — finds the port by vision as the terminal block furthest from the run's anchored end, takes the delivering hand and its working height from the support-hand planner, stages the carry three sides of a rectangle around the seated crossings at a hand's depth above the bench, drops into the mouth, opens the jaws and retreats along the bench. Use when a routed deformable linear object must terminate in a socket and the hand that anchored its free end is the one that can reach it.

- **In**: `scene` string, `prior` string, `arm_id` int, `grip_open_m` float
- **Out**: `port` string
- **Calls**: [`sam3`](#sam3); connector `cable.plan_support`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.grasp_frame`, `robot.set_grip`, `robot.wait_steps`
- **Scripts**: [`plug_port`](skills/plugging-a-cable-end/scripts/plug_port.py)
- **Exits**:
  - `plugged`: The delivering hand reached the port mouth and opened; `detail` says whether it got there routed or on the diagonal.
  - `unplugged`: No centreline or stations, no port in view, no support hand identified, the mouth unreachable at every height, or a tool raised on the carry; `detail` says which.

### pouring-by-wrist-roll

execute · robots: any · tasks: pouring · MIT · v0.1.0 · [`SKILL.md`](skills/pouring-by-wrist-roll/SKILL.md) · [upstream](https://github.com/OpenMOSS/VLABench/blob/main/VLABench/utils/skill_lib.py)

Pours from a held container into a target container by standing the hand above the target's box at a set height and rolling only the last joint (the wrist roll) to the pour angle in small speed-bounded steps, holding, and rolling back, checking every step from fresh joint readings. Use when a cup, bottle, scoop or bin is already in the hand and its contents must go into a bowl, box or other receiver whose box a perception skill found, and the grasp left the wrist-roll axis near horizontal (the roll then tips the container rather than spinning it).

- **In**: `target_obb` OrientedBoundingBox
- **Out**: `roll_reached` float, `hand_drift` float, `final_joint_error` float, `roll_axis_elevation_deg` float, `rolled_with` str
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`
- **Scripts**: [`pour_wrist_roll`](skills/pouring-by-wrist-roll/scripts/pour_wrist_roll.py)
- **Exits**:
  - `poured`: The hand reached the pour pose, the last joint followed every step to the full pour angle within track_tolerance with the tool centre within drift_tolerance of the pour pose, held hold_s, and rolled back with every joint within 0.08 rad of its start. This is what the hand did; re-perceive the receiver to know what landed in it.
  - `unreachable`: The move to the pour pose above the target was refused or ended more than reach_tolerance from it; nothing was rolled.
  - `joint_limit`: The asked roll does not fit between the last joint's reading and its limits (less a 0.05 rad margin) in the chosen direction; the hand stands at the pour pose, unrolled.
  - `roll_refused`: A roll step was refused, the last joint lagged its step's target by more than track_tolerance, the tool centre drifted past drift_tolerance, or the roll back failed or ended off; the skill rolled back as far as it could, and `detail` says which.

### pushing-objects

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/pushing-objects/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/push_object/libero)

Pushes a loose object across its support toward a goal point with a closed hand, never grasping it - hovers behind the object, comes straight down to a contact height on its side, pushes along a straight line, lifts away and opens. Use when an object must be slid rather than lifted (out of a corner, away from a wall or a neighbour so it can be grasped, too flat or wide for the gripper, or onto a mark), with its OBB from a perception skill and the hand empty.

- **In**: `target_obb` OrientedBoundingBox, `goal_position` Vec3
- **Out**: `push_distance` float, `achieved` float, `expected_center` Vec3
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`
- **Scripts**: [`push_object`](skills/pushing-objects/scripts/push_object.py)
- **Exits**:
  - `pushed`: The hand travelled past the contact point toward the goal by at least min(80 mm, 0.75 x the asked push), lifted away and opened; the object should now sit near expected_center.
  - `not_pushed`: The goal was under 30 mm from the object's centre, or the push ran but the hand travelled short (blocked, or the move refused part way), or the retract failed; `detail` says which.
  - `unreachable`: The hover above and behind the object was refused or ended off with every rotation tried, or the descent stopped more than reach_tolerance from the contact height; nothing was pushed.

### reaching-with-lag-correction

execute · robots: any · tasks: pick-place, insertion, cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/reaching-with-lag-correction/SKILL.md) · [upstream](https://github.com/penn-pal-lab/maestro/blob/main/maestro/api.py)

Moves the tool to a target pose, waits for the arm to settle, reads the tool pose back and, if the arm stopped short (tracking lag, gravity sag), commands once more aimed past the target by the shortfall, then reports the residual. Use when a pose must be hit closer than the arm's own move bound allows - a look pose, a pre-grasp or pre-insert pose, a descent near the table, or anywhere far out in the reach where a joint controller without an integral term leaves a steady error - and as a reach check whose not_reached exit means the arm is blocked, stuck or at the edge of its reach.

- **In**: `target_pose` Se3Pose
- **Out**: `residual` float, `residual_rotation` float, `first_residual` float, `first_residual_rotation` float, `corrected` bool, `correction` Vec3, `reached_pose` Se3Pose
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`
- **Scripts**: [`reach_with_correction`](skills/reaching-with-lag-correction/scripts/reach_with_correction.py)
- **Exits**:
  - `reached`: After at most one aim-past move the settled tool centre stands within tolerance (default 5 mm) of the target and its rotation within rot_tolerance (default 0.05 rad); residual is that distance.
  - `not_reached`: The move was refused outright (the hand did not move), or the shortfall exceeded the correction cap and the plain retry also ended off, or the aim-past move did not land; `detail` says which and residual is how far off the hand stands.

### searching-insertion-spiral

execute · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/searching-insertion-spiral/SKILL.md)

Finds an opening the held part or the closed fingertips can enter by probing straight down over a hexagonal grid around the opening's estimated mouth, nearest points first, stopping on contact and judging each probe by how far below the mouth the hand got - no force sensor. Use when a vertical insertion's target is known only to a few millimetres (a camera's estimate of a hole, a slot, a ring's opening, a socket), the hand already holds the part over it or is empty with the fingers closed, and the arm can stop on contact.

- **In**: `mouth_position` Vec3
- **Out**: `entered_at` Vec3, `depth` float, `probes` int
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.solve_ik`
- **Scripts**: [`spiral_search`](skills/searching-insertion-spiral/scripts/spiral_search.py)
- **Exits**:
  - `inserted`: A probe's tool centre got at least enter_depth (default 5 mm) below the mouth; the hand stays there unless stay_in is false.
  - `search_exhausted`: Every reachable probe inside search_radius stopped short of enter_depth; `detail` gives the count and the deepest probe.
  - `unreachable`: No probe position could be reached, or the hand's pose could not be read.

### seating-a-cable-crossing

execute · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/seating-a-cable-crossing/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/seating-a-cable-crossing)

Streams one planned crossing leg of a cable onto a routing station, lets the rod settle, then LOOKS — segments the rod, fits its centreline and counts how much of the curve lies inside the station's seat box on the side the route requires — and reports the fraction honestly rather than assuming the crossing took. It measures and does not correct. Use when a deformable-linear-object route is laid one crossing at a time and each crossing must be verified by vision before the next is planned.

- **In**: `plan` string, `scene` string, `index` int, `arm_id` int
- **Out**: `seated` bool, `fraction` float, `detail` string
- **Calls**: [`curve`](#curve), [`sam3`](#sam3); connector `robot.execute_trajectory`, `robot.get_observation`, `robot.wait_steps`
- **Scripts**: [`seat_station`](skills/seating-a-cable-crossing/scripts/seat_station.py)
- **Exits**:
  - `done`: The place leg was streamed and the crossing measured; `seated` and `fraction` say whether it took.
  - `blocked`: A tool raised during the place or the look (the subgraph's on_error).

### singulating-by-pushing

execute · robots: any · tasks: pick-place, sorting · BSD-2-Clause · v0.1.0 · [`SKILL.md`](skills/singulating-by-pushing/SKILL.md) · [upstream](https://github.com/andyzeng/visual-pushing-grasping/blob/master/robot.py)

Pushes a target clear of the objects crowding it until a top-down grasp's open fingers fit around it - picks the push heading toward the most free space around the target, turned until the closed hand can come down behind the target without touching a neighbour, sizes each push to just free the fingers (at most push_step), re-checks after every push, and reports the gap to the nearest neighbour and the fingers' clearance before and after. Use when a target touches or nearly touches other objects so the open jaws would land on a neighbour, with OBBs of the target and its neighbours from a perception skill and the hand empty.

- **In**: `target_obb` OrientedBoundingBox, `neighbour_obb` OrientedBoundingBox
- **Out**: `pushes` int, `gap_before` float, `gap_after` float, `clearance_before` float, `clearance_after` float, `fits_before` bool, `fits_after` bool, `required_gap` float, `expected_target_obb` OrientedBoundingBox
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`
- **Scripts**: [`singulate_by_pushing`](skills/singulating-by-pushing/scripts/singulate_by_pushing.py)
- **Exits**:
  - `singulated`: The open fingers of a top-down grasp fit around the target with FIT_MARGIN to spare - already at the start (nothing pushed) or, by the hand's travel, after `pushes` short pushes; re-perceive before grasping.
  - `still_cluttered`: Pushes ran out (max_pushes), a push came up short (under half its length), or no usable heading would increase the fingers' clearance; the hand is open and lifted.
  - `unreachable`: The closed hand fits behind the target in no heading before the first push, the target is wider than the opening along every grasp axis, or a hover, descent or lift was refused; the hand is open.

### sliding-along-a-cable

execute · robots: any · tasks: cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/sliding-along-a-cable/SKILL.md) · [upstream](https://github.com/o2ac/o2ac-ur/blob/65ddc5496879799038cefbad061380b7d0300d73/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py#L4622)

Slides the hand along a cable it holds toward one of the cable's ends - the jaws opened to a loose cage just wider than the cable (not a pinch), the hand stepped along the perceived centreline with the jaws kept across it, stopped a planned distance before the end (or earlier when the hand stalls against the end's plug or the jaws are pushed open), then closed firmly. Use when a cable taken somewhere along its length must be held near its end instead, e.g. right behind a plug before plugging it, or to run the hand out to the tip of a cable lying on the table without regrasping, with the cable's centreline from curve.fit_centerline / curve.track_centerline and its end from curve.endpoints.

- **In**: `centerline` Centerline
- **Out**: `stopped_by` string, `start_from_end` float, `end_from_end` float, `slid` float, `cage_width` float, `cage_reading` float, `jaw_width` float, `steps` int, `wrist_turns` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`slide_along_cable`](skills/sliding-along-a-cable/scripts/slide_along_cable.py)
- **Exits**:
  - `reached_end`: The hand stopped at the planned distance before the end (stopped_by planned_stop or already_there), was pushed open by something thicker than the cable (jaws), or stalled within end_window (60 mm) of the tip (stall, read as the end's plug against the fingertips); the jaws then closed firmly and short of their stop. The hand stands there, closed on the cable.
  - `lost_cable`: The hand was not on the cable before it moved (more than max_offset from the centreline in plan, jaws closed on nothing, or jaws held open wider than the cage), or the firm close at the end shut to the closed stop; in the last case the hand opened and rose 4 cm.
  - `unreachable`: The arm solver could not reach a waypoint (previewed before anything moved), a step was refused, or the hand stalled further than end_window from the tip (a snag, not the end); after a refusal or a stall the jaws closed firmly where the hand stopped. `detail` says which.

### sliding-to-edge-then-picking

execute · robots: any, *tuned on* yam_i2rt · tasks: pick-thin-flat, pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/sliding-to-edge-then-picking/SKILL.md) · [upstream](https://github.com/google-research/ravens/blob/master/ravens/tasks/primitives.py)

Picks a flat object too thin to pinch on its support - closed fingers slide it in a straight line until part of it overhangs the support's edge, then the open jaws come down beside the overhang, below the support's top, close on it from its two sides and lift. Use when a slab, card, plate, phone or book lies flat on a table and a top-down pinch cannot reach its sides, a free edge of the support is within reach, the object's OBB comes from a perception skill, the edge from a plane or table boundary (a point, or a point and a direction, world frame), and the hand is empty.

- **In**: `target_obb` OrientedBoundingBox, `edge_point` Vec3
- **Out**: `overhang_before` float, `overhang_target` float, `slide_planned` float, `slide_achieved` float, `overhang_expected` float, `held_width` float
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_cartesian_until_contact`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`slide_to_edge`](skills/sliding-to-edge-then-picking/scripts/slide_to_edge.py), [`pick_overhang`](skills/sliding-to-edge-then-picking/scripts/pick_overhang.py)
- **Exits**:
  - `slid`: (slide only) The hand travelled the planned slide (within 5 mm) and lifted off the slab, or the slab already overhung enough; `plan` holds the pinch. The entry point of pick.
  - `picked`: The slab was slid until it overhung the edge by about overhang_target, the open jaws came down beside the overhang to the slab's mid-height, closed, and after a 50 mm lift the jaw gap still read wider than closed by more than 4 mm.
  - `slid_not_picked`: The slide ran but the hand travelled short of it or could not lift off, or the pinch's hover was refused, or the open hand stopped more than 6 mm above the pinch height (a blade on the slab or the support), or the jaws closed on nothing (or lost it in the lift); `detail` says which. The slab may have moved; re-perceive it.
  - `unreachable`: Nothing moved the slab - its centre is not inside the edge, it is too short across the edge to overhang enough with its centre on the support, too wide for the jaws, more than max_slide from the edge, the IK preview refused the end of the slide or the pinch, the hover was refused for every rotation, or the descent behind it (push) / onto it (press) failed.

### stacking-objects

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/stacking-objects/SKILL.md) · [upstream](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/stack_cube.py)

Stacks the object the hand holds centred on top of a target object - turns the hand about the vertical so the held box's faces line up with the target's, approaches above the target's top, comes straight down until the arm stops on contact, releases and retreats straight up (ManiSkill's stack-cube solution and RoboTwin's block stacking, with placing-with-contact-descent's descent in place of a move to a known height). A stop far from the target's top, a descent that meets nothing or a stop that pushed the hand sideways is routed and the object kept. Use when a held block or box must go on top of another one whose oriented box a camera has measured (towers, pyramids, stacking boxes).

- **In**: `target_obb` OrientedBoundingBox, `held_obb` OrientedBoundingBox
- **Out**: `released` bool, `yaw_turn_deg` float, `yaw_error_deg` float, `expected_contact_z` float, `contact_z` float, `contact_offset` float, `stacked_position` Vec3
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`stack_on_object`](skills/stacking-objects/scripts/stack_on_object.py)
- **Exits**:
  - `stacked`: The descent over the target stopped on contact inside the contact window and within 8 mm of the planned point, the hand opened and retreated straight up; stacked_position is where the hand left the held object's bottom centre. What the hand did, not a check that the stack stands.
  - `misaligned`: The descent did not meet the target's top where expected - it stopped more than contact_tolerance (12 mm) above it, reached its bottom without stopping (the held object missed the top, or slid up the jaws), or stopped pushed sideways more than lateral_tolerance; the hand did not release and lifted the object back to the approach height.
  - `blocked`: The target's top is tilted more than 10 deg, no aligned turn can reach the contact pose (IK preview), the rise, the approach, the re-centring after a wrist turn or a descent move was refused, or the retreat after the release failed (`released` says which side of the release); `detail` says which.

### straightening-a-cable-toward-a-target-curve

execute · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/straightening-a-cable-toward-a-target-curve/SKILL.md) · [upstream](https://github.com/DanielTakeshi/deformable-ravens/blob/73982748bdfe756e04553e28698948facccecb41/ravens/tasks/task.py)

Pulls a cable lying flat on a table toward a target curve, one correction at a time - the perceived centreline and the target are resampled at equal arc length in plan, matched in the cable's two directions (and, when their lengths differ, at every placement of the shorter along the longer), the point with the largest discrepancy is picked with grasping-a-cable-lying-flat and put down a little past its target, and the cable is looked at again, until every node is within a tolerance, a cap on corrections is reached, the error stalls or the cable is lost. Use when a cable on the table must lie along a line or curve, e.g. to lay its last 10 cm straight and pointing at a ring before threading, or to undo a bend before routing, with an overhead RGB-D camera (sam3.segment_text and curve.fit_centerline) or a centreline fitted by the caller.

- **In**: `target` Centerline
- **Out**: `max_error` float, `mean_error` float, `initial_max_error` float, `corrections` int, `misses` int, `reversed` bool, `offset_m` float, `coverage` float, `pick_point` Vec3, `place_point` Vec3, `centerline` Centerline
- **Calls**: [`curve`](#curve), [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_cartesian_until_contact`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`straighten_cable`](skills/straightening-a-cable-toward-a-target-curve/scripts/straighten_cable.py)
- **Exits**:
  - `shaped`: The latest look put every matched node within tolerance (10 mm) of the target in plan; the hand is open and stands retreat (5 cm) over the last place, or has not moved if the cable was already there.
  - `corrected`: No cable_description was given, so after one pick-and-place the cable was not looked at again; fit its centreline and call again. The hand is open over the place.
  - `not_converged`: max_iters (12) corrections were made, or the best max error did not improve by 2 mm in 4 corrections, and some node is still further than the tolerance; the cable lies on the table, the hand is open.
  - `grasp_failed`: Two grasps in a row (one without a camera) closed on nothing; the hand is open over the cable.
  - `lost_cable`: The cable was not found by the camera, its fit was empty, or the cable seen is shorter than min_coverage (80 %) of the target in plan (part of it hidden or out of view).
  - `unreachable`: No pick among the PICK_TRIES (3) largest discrepancies was reachable (grasping-a-cable-lying-flat routed unreachable for each), or the carry to the place was refused; after a refused carry the cable was set down where the hand stood and released. `detail` says which.

### threading-a-cable-tip-through-a-ring

execute · robots: any · tasks: cable-routing · MIT · v0.1.0 · [`SKILL.md`](skills/threading-a-cable-tip-through-a-ring/SKILL.md) · [upstream](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/hanging_mug.py)

Threads the free tip of a held cable through a standing ring - the end turned about the vertical to point along the ring's axis, carried to a pre-insert pose in front of the mouth with the axis aimed between the tip and where the cable leaves the fingers (so a sagging end still passes), pushed through along the axis with an overshoot capped where the fingertips would reach the ring, and judged by the hand's travel past the ring's plane (optionally a camera point on the tip). Use when the hand already holds a cable a few centimetres behind its tip (a grasp skill's state, the tip measured or given) and the ring's centre, axis and inner radius are known (perception), e.g. the first step of routing a cable through a ring or loop.

- **In**: `ring_centre` Vec3, `ring_axis` Vec3, `ring_inner_radius` float
- **Out**: `tip_past` float, `hand_travel` float, `push_planned` float, `tip_offset` float, `exit_offset` float, `crossing_offset` float, `sag_deg` float, `turned_deg` float, `tip_seen` bool, `seen_tip_past` float, `tip_position` Vec3, `cable_lost` bool
- **Calls**: [`gemini-er`](#gemini-er); connector `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`thread_tip_through_ring`](skills/threading-a-cable-tip-through-a-ring/scripts/thread_tip_through_ring.py)
- **Exits**:
  - `threaded`: The push carried the tip, as held, at least min_past (10 mm) past the ring's plane with the hand within 6 mm of its planned travel (or already past when it stopped short); with a camera, the tip was seen past the plane and the line from the fingers to it crosses the plane inside the ring. Without a camera this is what the hand did, not what the cable did. The hand stays there, closed (opened and lifted 4 cm when release is true).
  - `jammed`: The tip did not get through - the push stopped short with the tip, as held, under min_past past the plane, or the camera saw the tip short of the plane after the push (held back or bent on the rim; the hand backed off to the pre-insert pose, still holding) - or the jaws closed to their stop before or during the push (cable_lost true; the cable slid out of the hand, nothing more is moved).
  - `misaligned`: Before moving, the held end cannot pass the opening (it sags so far that aimed between its tip and the fingers it stands outside the opening, or the fingertips reach the ring before the tip gets min_past through, or the opening leaves no room for the cable); or the camera saw the tip past the plane on a line crossing it outside the ring. `detail` says which.
  - `unreachable`: The jaws are at their closed stop at the start (nothing held), the hand is more than 35 deg off straight down, the ring's axis is vertical, no wrist solution reaches the pre-insert pose or the end of the push (IK preview), or a turn, carrying move or the push was refused.

### tipping-over-a-surface-edge

execute · robots: any, *tuned on* yam_i2rt, 2 arms, **sim only** · tasks: reorientation · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/tipping-over-a-surface-edge/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/tipping-over-a-surface-edge)

Turn a body over on the surface it rests on instead of carrying it — pinch two opposite walls LOOSELY and off centre so the pads act as a pin joint, lift so the surface and gravity stand the body up on its own edge, then press it onto the surface and pull the pins toward you about whichever edge is grounded until it is just past balance, and let go. Use when a rigid grasp cannot transmit the torque a full roll needs (a parallel jaw rolling about its own closing axis), when the body is too big to turn over in hand, or when the arm's envelope cannot follow a held body through the roll.

- **In**: -
- **Out**: -
- **Calls**: connector `motion.plan_joint`, `motion.plan_linear`, `robot.describe_arm`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.grasp_frame`, `robot.set_grip`, `robot.stream_dual`, `robot.wait_steps`, `sim.get_object_pose`, `sim.query`
- **Scripts**: [`read_scene`](skills/tipping-over-a-surface-edge/scripts/read_scene.py), [`approach`](skills/tipping-over-a-surface-edge/scripts/approach.py), [`pinch`](skills/tipping-over-a-surface-edge/scripts/pinch.py), [`lift`](skills/tipping-over-a-surface-edge/scripts/lift.py), [`press_pull`](skills/tipping-over-a-surface-edge/scripts/press_pull.py), [`release`](skills/tipping-over-a-surface-edge/scripts/release.py), [`check`](skills/tipping-over-a-surface-edge/scripts/check.py)
- **Exits**:
  - `inverted`: The body came to rest turned over on the surface after the release.
  - `standing`: The body stands on its edge or wall; it rocked back at the release. Re-grasp through the mouth and pull again, a few degrees further.
  - `other`: The body is neither turned over nor standing — it slipped out of the pads during the lift, or the release threw it.

### transporting-objects

execute · robots: any, *tuned on* panda · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/transporting-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/transporting-objects)

Move the currently-held object to a destination and release. The gripper enters this subgraph holding the object; on exit the object has been placed. For a walled container (basket / bin / box / tote) the default is a straight-Z descent to inside the walls (transport_descend_linear → release); for placement ONTO a surface or into a described sub-region it computes an explicit drop pose (compute_drop → move_above → release). Use after a successful grasp when the destination has a known OBB. When the destination is a sub-region described in natural language (e.g. "the left compartment of the caddy", "to the left of the plate", "the inside of the top drawer"), an optional VLM-grounded perceive_zone state localizes the zone before the drop pose is computed. Heights derive from the perceived container and the live wrist pose rather than a resting tool height, and the top-down orientation from the hand actually on the arm (robot.grasp_frame).

- **In**: `container_obb` OrientedBoundingBox, `container_mask` Mask, `target_obb` OrientedBoundingBox, `target_mask` Mask, `ee_pose_at_grasp` Se3Pose
- **Out**: -
- **Calls**: [`curobo`](#curobo), [`geometry`](#geometry), [`grounding-dino`](#grounding-dino), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.describe_arm`, `robot.describe_gripper`, `robot.describe_workspace`, `robot.execute_trajectory`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_home`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`, `robot.wait_steps`
- **Scripts**: [`transport_descend_linear`](skills/transporting-objects/scripts/transport_descend_linear.py), [`place_release`](skills/transporting-objects/scripts/place_release.py), [`compute_drop_pose`](skills/transporting-objects/scripts/compute_drop_pose.py), [`drop_offset_pose`](skills/transporting-objects/scripts/drop_offset_pose.py), [`approach_above`](skills/transporting-objects/scripts/approach_above.py), [`descend_release`](skills/transporting-objects/scripts/descend_release.py), [`descend_release_linear`](skills/transporting-objects/scripts/descend_release_linear.py), [`lift_grasped`](skills/transporting-objects/scripts/lift_grasped.py), [`waypoint_move`](skills/transporting-objects/scripts/waypoint_move.py), [`waypoint_move_carve`](skills/transporting-objects/scripts/waypoint_move_carve.py), [`perceive_placement_zone`](skills/transporting-objects/scripts/perceive_placement_zone.py)
- **Exits**:
  - `placed`: Object released at the destination.
  - `blocked`: Path blocked or motion failed; aborted before release. Coordinator routes to abort.

### turning-by-wrist-ratchet

execute · robots: any · tasks: contact-assembly · MIT · v0.1.0 · [`SKILL.md`](skills/turning-by-wrist-ratchet/SKILL.md) · [upstream](https://github.com/zzigak/yam-arcade/blob/main/stations/jars/robot_demo.py)

Tightens or loosens a screw lid or cap by wrist ratchet - grips it from above, rolls only the last joint through a 218 deg stroke (joint 6 -1.9 -&gt; +1.9 rad on a YAM) in speed-bounded steps, opens, rolls back, re-grips and repeats for a number of strokes or until the turn stops, read from joint 6's tracking lag (no force sensor), the gripper reading and the tool centre's drift; after loosening it lifts to check the cap came free. Use when a jar lid, bottle cap or knob held on a thread (or a twist-lock) must be turned on or off and its box is known from perception, and the jar or bottle is held by the other arm or a fixture.

- **In**: `lid_obb` OrientedBoundingBox
- **Out**: `direction` str, `strokes_done` int, `total_turn` float, `seated` bool, `hand_drift` float, `lift_rise` float, `holding` bool, `rolled_with` str
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`turn_wrist_ratchet`](skills/turning-by-wrist-ratchet/scripts/turn_wrist_ratchet.py)
- **Exits**:
  - `turned`: Tighten: every stroke ran, or joint 6 stalled after the strokes had turned at least min_turn in all (seated: true); the hand opened and rose. Loosen: every stroke ran and the lift check rose lift_height with the jaws still holding (or lift_check was off); the hand keeps the cap. What the hand did; re-perceive to know the lid's state.
  - `stalled`: Joint 6 lagged its target by more than stall_lag and stopped turning (a tightening stall before min_turn, or any loosening stall), a stroke or roll-back step was refused, the tool centre drifted past drift_tolerance, or the loosening lift check rose short (the cap still on its thread); the hand opened and rose. `detail` says which.
  - `slipped`: The gripper reading fell under empty_grip (jaws closed on nothing) at a grip, during a stroke, or after the loosening lift; the hand opened and rose.
  - `unreachable`: The pregrasp over the lid, the turn of joint 6 to the stroke's start, or the descent to the grip point was refused or ended off, the stroke does not fit joint 6's limits, or there is no joint reading; nothing was turned.

### wiping-a-region

execute · robots: any · tasks: surface-contact · MIT · v0.1.0 · [`SKILL.md`](skills/wiping-a-region/SKILL.md) · [upstream](https://github.com/capgym/cap-x/blob/main/capx/envs/tasks/franka/franka_spill_wipe.py)

Wipes a flat region of a surface with back-and-forth strokes - the hand comes down at one end of each pass, strokes along the region's long axis at a contact height just over the surface, lifts, steps across and comes back the other way, then reports how much of the region the measured strokes swept. No force sensor - each descent stops on contact, so a surface higher than estimated is met rather than pressed into. Use when a spill, crumbs or a mark on a table must be wiped (with a held sponge or cloth, or the closed fingers), given the region's box from perception (a mask's points through geometry.filter_and_compute_obb), and the region is clear of objects.

- **In**: `region_obb` OrientedBoundingBox
- **Out**: `coverage` float, `passes_planned` int, `passes_done` int, `surface_z` float, `wipe_height` float, `min_height` float, `max_height` float
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_cartesian_until_contact`, `robot.solve_ik`
- **Scripts**: [`wipe_region`](skills/wiping-a-region/scripts/wipe_region.py)
- **Exits**:
  - `wiped`: The measured strokes swept at least min_coverage (default 0.95) of the region at the wiping height and the hand lifted away; `coverage` says how much.
  - `partial`: Some passes ran but the swept fraction is under min_coverage (passes out of reach, blocked, stopped part way) or the last lift failed; `detail` lists each skipped or stopped pass.
  - `unreachable`: No pass ran - the first hover was refused with every rotation, or every pass was out of reach, blocked or refused; nothing was wiped.

### checking-dual-arm-clearance

verify · robots: any, 2 arms · tasks: bimanual, pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/checking-dual-arm-clearance/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/check_dual_arm_collision/robotwin)

Predicts, before a move is sent, the minimum clearance between the two arms along the planned path - both arms' joints and tool poses are read together, each planned leg (joint waypoints, or tool poses solved with the arm's IK every 2 cm) is sampled every 0.01 rad, and at every sample a capsule model of the YAM's links (plus held-object spheres) gives the closest cross-arm pair - and routes clear / too_close / unknown with the minimum distance, where along the path it falls and the closest link pair. Use when a bimanual station is about to move one arm near the other (a handover, a two-hand lift, a place beside the other hand) and its planner does not see the other arm, with both arms addressable by arm_id.

- **In**: -
- **Out**: `min_clearance_m` float, `closest_link_a` string, `closest_link_b` string, `closest_point_a` Vec3, `closest_point_b` Vec3, `at_fraction` float, `start_clearance_m` float, `end_clearance_m` float, `samples` int, `model_error_m` float
- **Calls**: connector `robot.describe_arm`, `robot.forward_kinematics`, `robot.get_observation`, `robot.solve_ik`
- **Scripts**: [`check_clearance`](skills/checking-dual-arm-clearance/scripts/check_clearance.py)
- **Exits**:
  - `clear`: The capsule model's smallest cross-arm clearance over every sample of the planned motion is at least too_close_m (default 5 mm); min_clearance_m, closest_link_a / closest_link_b and at_fraction say where it is tightest.
  - `too_close`: Somewhere along the planned motion (or now, with no plan given) the capsules come within too_close_m or overlap (negative min_clearance_m); at_fraction and closest_link_a / closest_link_b say where and which links.
  - `unknown`: The check could not be made - a second arm is not reported or reads as the first, an arm has no joint reading or not 6 joints, the connector's forward kinematics disagrees with the YAM model by more than 5 mm, a pose waypoint has no IK solution within the servo's tolerance or the IK changed branch along a straight leg, or the inputs conflict; `detail` says which.

### checking-spatial-relations

verify · robots: any · tasks: pick-place, insertion, cable-routing, sorting · MIT · v0.1.0 · [`SKILL.md`](skills/checking-spatial-relations/SKILL.md) · [upstream](https://github.com/real-stanford/reflect/blob/c2f9db993e372250321ca74e56ae27519716c934/main/scene_graph.py)

Decides from one RGB-D view whether a spatial relation holds between two seen objects - inside, on_top_of, near or blocking - with REFLECT's point-cloud tests (minimum point distance, convex-hull containment, footprint-and-height, image-box occlusion with depth order), and routes holds / does_not_hold / not_visible. Use when a graph must branch on a scene fact such as "is the cube in the box", "is the plug on the board", "is the part near the fixture" or "is the hand blocking the camera's view of the target", with the two objects given as descriptions, masks or world point clouds.

- **In**: `relation` string
- **Out**: `holds` bool, `relation` string, `min_distance_m` float, `evidence` string
- **Calls**: [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`check_relation`](skills/checking-spatial-relations/scripts/check_relation.py)
- **Exits**:
  - `holds`: Both objects were seen with enough valid depth and the asked relation is among those REFLECT's tests found from the subject to the reference.
  - `does_not_hold`: Both objects were seen with enough valid depth and the asked relation is not among those found; `relations` lists what was found instead.
  - `not_visible`: The subject or the reference was not segmented, or kept fewer than min_points valid depth points; nothing is claimed about the relation.

### classifying-failure-from-proprioception

verify · robots: any · tasks: pick-place, failure-detection · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/classifying-failure-from-proprioception/SKILL.md) · [upstream](https://github.com/NVlabs/VoLoAgent/blob/b4e623079ca8498a16bcd5016920d71f76c44d30/vlm_orchestrator/failure_handlers/signal_detector.py)

Classifies the last moments of a manipulation step from proprioception alone - the gripper opening, the tool-centre pose and the arm joints over a short window (passed in from earlier nodes, sampled here, or both) - into stall, never_gripped, fumble, drop, stuck_mid_air or ok, with VoLoAgent's rolling-window rules re-based on the work surface and the gripper's stroke. Use when a graph must know cheaply, with no camera and no force sensor, whether a grasp, lift or carry went wrong and how, before spending a perception or VLM call or choosing a recovery.

- **In**: -
- **Out**: `state` string, `reason` string, `confidence` float, `samples` int
- **Calls**: connector `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_observation`
- **Scripts**: [`classify_failure`](skills/classifying-failure-from-proprioception/scripts/classify_failure.py)
- **Exits**:
  - `ok`: No failure signal. `state` says which healthy state - held (closed on something and elevated, moving or still on its target), placed (opened after a descent, after a long close, or low), approaching (closed, low), open (no strong signal).
  - `stall`: The tool centre barely moved over the window (z range under 30 mm, under 2 mm per sample) while it was more than lag_tolerance short of its commanded target, or expect_motion was set and no target was given; and it was not holding something in the air.
  - `never_gripped`: The jaw closed to within 3 mm of its closed stop without having held anything in the window (closed on nothing), or the hand never left its open stop at all (upstream's rule, gated by chunks_elapsed).
  - `fumble`: Repeated open / close - at least 6 grip transitions with the tool centre low, or at least 4 anywhere (gated by chunks_elapsed).
  - `drop`: The hand opened while the tool centre was more than 60 mm over table level, or a held object's jaw closed to its stop with the hand still shut above carrying height (a slip).
  - `stuck_mid_air`: The hand is closed on something above carrying height but the arm is frozen short of its commanded target.
  - `unknown`: The arm could not be read, there were fewer than two samples, or no surface height was given or described; nothing was classified.

### previewing-target-overlay

verify · robots: any · tasks: pick-place, cable-routing, insertion · MIT · v0.1.0 · [`SKILL.md`](skills/previewing-target-overlay/SKILL.md) · [upstream](https://github.com/penn-pal-lab/maestro/blob/e60d326233a238ddc496bf7a08acb7a278e9e66d/maestro/api.py)

Checks a target gripper pose before the arm moves - projects the pose into every camera that sees it (tool centre, open jaw line with its fingers at the hand's span, approach axis), draws it as MAESTRO's image editor does, dashes what lies behind the depth image's surface, and asks the VLM a yes/no question - is this target right for the stated intent (e.g. "grasp the cube from above")? Use when a perceived or computed grasp, place or approach pose should be sanity-checked against the scene (wrong object, wrong side, mid-air, inside the table) before committing a move, with only cameras and the hand's own description.

- **In**: `target_pose` Se3Pose, `intent` string
- **Out**: `approved` bool, `answer` string, `reason` string
- **Calls**: [`vlm`](#vlm); connector `robot.describe_gripper`, `robot.get_observation`, `robot.grasp_frame`
- **Scripts**: [`preview_target`](skills/previewing-target-overlay/scripts/preview_target.py)
- **Exits**:
  - `approved`: The VLM answered yes - on the overlay the target fits the intent (right object or part, right side, right height); `answer` is its one-line justification.
  - `rejected`: The VLM answered no; `answer` says why (wrong object, approach, height ...). Nothing moved.
  - `no_view`: No camera with RGB, intrinsics and pose has the tool centre inside its image (or in front of it), or vlm.query_yes_no raised; nothing was judged and `reason` says which.

### verifying-a-cable-route

verify · robots: any · tasks: cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-a-cable-route/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/verifying-a-cable-route)

Judge a finished cable route from a camera rather than from an evaluator — re-read the cable's centreline from the clear-view camera (tracking the carried prior when the support hand still hides one end), test that rod material lies on the owed side of every station within the seat's reach, and that the hand has left the rod. Use when a routing policy must decide for itself whether every crossing is seated and it is safe to stop, and report margins so a disagreement with a scorer is diagnosable.

- **In**: `scene` string, `prior` string
- **Out**: `success` bool, `worst_margin_m` float, `tcp_clear_m` float, `detail` string
- **Calls**: [`curve`](#curve), [`sam3`](#sam3); connector `robot.get_ee_pose`, `robot.get_observation`
- **Scripts**: [`verify_weave_vision`](skills/verifying-a-cable-route/scripts/verify_weave_vision.py)
- **Exits**:
  - `woven`: Every station has rod on its owed side within the seat's reach, and the hand is clear of the rod.
  - `short`: At least one station is unseated (no rod abreast of it, on the wrong side, or outside the box), the hand is still on the rod, or no rod could be read at all; `detail` says which.

### verifying-articulation-displacement

verify · robots: any · tasks: articulated · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-articulation-displacement/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/blob/main/roborsi/embodied/skills/base/close_drawer/libero/policy.py)

Checks from the camera whether a drawer or a door really moved as asked - re-finds its handle in a fresh RGB-D frame (pointing model, SAM, then the handle's front proud of the fitted face plane), checks the face normal is the one expected, and measures the handle's travel along the articulation (prismatic - metres along the pull; revolute - the angle about the hinge), routing moved, moved_short, not_moved, moved_wrong_way, inconsistent or not_seen. Use after any skill that opens or closes an articulated object (a drawer pull or push, a door swing, the waypoints of planning-articulation-opening), once the hand has let go and moved clear, before the graph relies on the drawer or door being open or shut.

- **In**: `handle_description` string, `joint_type` string, `expected_opening` float
- **Out**: `seen` bool, `measured` float, `fraction` float, `transverse` float, `normal_dot` float, `handle_after` Vec3, `face_normal_after` Vec3
- **Calls**: [`gemini-er`](#gemini-er), [`molmo`](#molmo), [`sam3`](#sam3), [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`verify_articulation`](skills/verifying-articulation-displacement/scripts/verify_articulation.py)
- **Exits**:
  - `moved`: The handle travelled at least min_fraction (70%) of expected_opening in its sign, the face normal is the one expected and the handle stayed within 70 mm of the articulation.
  - `moved_short`: It travelled the right way by at least 30 mm of handle travel (a door - that arc at the handle's radius), but less than min_fraction of what was expected; stuck, or the hand slipped part way.
  - `not_moved`: It travelled less than 30 mm (that arc for a door) either way.
  - `moved_wrong_way`: It travelled at least 30 mm against the expected sense (a close that opened it).
  - `inconsistent`: The handle was found, but the face normal is not the one expected (cosine under 0.9), the handle drifted more than 70 mm off the articulation, or a drawer moved more than 250 mm; most likely another handle or face was found, or the cabinet itself moved.
  - `not_seen`: The pointer found no handle, or no vertical face plane or nothing proud of it was found round the pointed pixel, in the frame after (or before) actuation; the hand may still hide it.

### verifying-grasps

verify · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-grasps/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/verifying-grasps)

Three independent checks on one grasp attempt -- that the hand arrived where it was sent before the jaws close, that the jaw gap after the close is not the mechanical stop, and that the lifted object is visible above the surface and near the hand. Use when a grasp closed on a small or thin object and the graph must know, before carrying or inserting, whether anything is actually in the gripper.

- **In**: `object_description` string
- **Out**: `verified` bool, `observed_center` Vec3
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.describe_gripper`, `robot.describe_workspace`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose_cartesian`
- **Scripts**: [`verify_grasp`](skills/verifying-grasps/scripts/verify_grasp.py), [`verify_reach`](skills/verifying-grasps/scripts/verify_reach.py), [`verify_grip`](skills/verifying-grasps/scripts/verify_grip.py)
- **Exits**:
  - `verified`: The object was seen after the lift, above the surface and near the hand.
  - `not_held`: Nothing matching was seen, or what was seen stayed on the surface or far from the hand.
  - `reached`: The hand is close enough to the commanded grasp pose to have the object between the pads.
  - `missed`: The hand stopped further from the commanded pose than the object's own smallest half-extent allows.
  - `held`: The jaws closed on something -- the gap is above the mechanical stop.
  - `empty`: The jaws ran to the stop, so they closed on nothing.
  - `unknown`: The hand could not be read, which is a gap in the instrument rather than a failed grasp.

### verifying-insertion-by-probe-push

verify · robots: any · tasks: insertion, pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/verifying-insertion-by-probe-push/SKILL.md) · [upstream](https://github.com/o2ac/o2ac-ur/blob/main/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py)

Checks whether a held part is seated after an insertion by pushing it a few millimetres further along the insertion axis (optionally also a little sideways) with a stop-on-contact move and reading where the hand stopped - no force sensor. Seated when the hand did not go further (and did not move sideways); not seated when it did. Use when an insertion, a placement into a pocket or a press-fit has just finished, the hand still holds the part, and no camera can see whether it went home.

- **In**: -
- **Out**: `travel` float, `lateral_travel` float, `start_position` Vec3, `stop_position` Vec3, `returned` bool
- **Calls**: connector `robot.get_ee_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.solve_ik`
- **Scripts**: [`probe_push`](skills/verifying-insertion-by-probe-push/scripts/probe_push.py)
- **Exits**:
  - `seated`: The push along the axis moved the hand less than seat_tolerance (default 5 mm), and every sideways push (when lateral_distance &gt; 0) less than lateral_tolerance; the hand went back to the start (return_to_start) and still holds the part.
  - `not_seated`: The hand went at least seat_tolerance further along the axis, or a sideways push moved it lateral_tolerance or more; `travel` / `lateral_travel` say how far. The hand went back to the start unless return_to_start is false.
  - `blocked`: No verdict - the hand's pose could not be read, the end of a push is out of reach by the IK preview, the tool-centre floor over table_z leaves no room for a push longer than seat_tolerance, or a push or the return to the start before a sideways push was refused.

### verifying-placement

verify · robots: any · tasks: pick-place, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-placement/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/verifying-placement)

Confirms from one RGB-D view that a released object went through an aperture, accepting either that it vanished into the container or that its visible centre lies below the rim within the opening's footprint, and routing not_placed when it is still visible above or beside the aperture. Use when an insertion-and-release must be checked without simulator state, after the release has completed.

- **In**: `observation` Observation, `object_description` string, `aperture_center` Vec3, `rim_z` float
- **Out**: `verified` bool, `evidence` string
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3)
- **Scripts**: [`verify_placement`](skills/verifying-placement/scripts/verify_placement.py)
- **Exits**:
  - `verified`: The object is no longer visible, or its visible centre is below the rim within the aperture footprint.
  - `not_placed`: The object is still visible above the rim or outside the aperture footprint, or too little depth was valid to confirm.

### centering-a-part-with-the-gripper

recover · robots: any · tasks: pick-place, insertion · MIT · v0.1.0 · [`SKILL.md`](skills/centering-a-part-with-the-gripper/SKILL.md) · [upstream](https://github.com/o2ac/o2ac-ur/blob/main/catkin_ws/src/o2ac_routines/src/o2ac_routines/common.py)

Centres a loose part between the jaws without perceiving it - closes on it, opens, turns the wrist a quarter turn about the hand's approach axis, closes again (each close squares the part up along one axis), opens and turns back; fails when the jaw gap after a close says nothing is between the pads. Use when the hand already stands around a small part whose estimated position is a few millimetres off (before a pick that must be centred, an insertion of a plug or peg, or threading a ring held in the fingers), the part is free to slide on its support, and the hand is open or empty.

- **In**: -
- **Out**: `first_width` float, `second_width` float, `turned_deg` float, `turned_by` string, `returned` bool
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`center_with_gripper`](skills/centering-a-part-with-the-gripper/scripts/center_with_gripper.py)
- **Exits**:
  - `centered`: Both closes ended on something (the jaw gap above the closed stop by more than min_held_width, or unread), the wrist turn between them arrived, and the hand opened again (kept closed with release=false); `returned` says whether it turned back.
  - `nothing_caught`: A close ended within min_held_width of the closed stop, so nothing was between the pads; the hand is open and, after the turn, turned back when return_to_start.
  - `unreachable`: Neither quarter turn arrived, by pose or by the last joint (refused, or the read-back rotation more than rotation_tolerance_deg or the tool centre more than position_tolerance off); the hand is open and was sent back to its start rotation.

### correcting-insertion-offset-from-port-opening

recover · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/correcting-insertion-offset-from-port-opening/SKILL.md) · [upstream](https://github.com/AMMistry18/aic/blob/fd27bbe875dbd656e381f72ae2b44efdfb243924/aic_model/aic_model/insertion/visual_gap_recovery.py)

Re-centres a held part over a port before or after a failed insertion using cameras only - finds the dark opening near its projected mouth in every available view (hand and part masked out by depth), meets each view's ray with the port plane, keeps the views that agree within 3 mm, and steps the hand toward that consensus in moves of at most 1 mm (6 mm total, 8 steps) until the part's descent line is centred on the opening in at least two views. Use when an insertion stopped short or a camera's port estimate is only good to a few millimetres, the hand holds the part over the mouth, and two cameras see the opening (the inserting arm's wrist camera plus the other arm's parked wrist camera or the overhead camera); no force sensor.

- **In**: `port_position` Vec3, `opening_size` float
- **Out**: `hole_position` Vec3, `hole_offset` float, `correction` Vec3, `steps` int, `views` int
- **Calls**: [`sam3`](#sam3); connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`
- **Scripts**: [`correct_offset`](skills/correcting-insertion-offset-from-port-opening/scripts/correct_offset.py)
- **Exits**:
  - `centered`: The held part's descent line meets the port plane within the pixel tolerance of the opening's centre in at least min_views (2) views; the hand stays there, ready for the push. hole_position is the cameras' consensus of the opening's centre on the port plane.
  - `opening_not_found`: Fewer than min_views cameras are usable or find the opening near its projected mouth (contrast, area and centre gates); nothing moved on this pass.
  - `views_disagree`: The views found the opening but no pair puts it within max_view_disagreement (3 mm) on the port plane; a camera's calibration or the detection is off. Nothing moved on this pass.
  - `offset_too_large`: The consensus lies more than max_port_offset (8 mm) from the estimate, or the next step would take the commanded correction past max_total (6 mm); re-perceive the port rather than correct. An opening near the bound can be measured short until the hand has stepped toward it, so the hand may have moved before this exit.
  - `no_convergence`: After max_steps (8) moves the part is still not centred in min_views views; the hand is where the last step left it.
  - `unreachable`: The hand's pose could not be read, or a step was refused or reported out of reach.

### returning-to-a-recorded-pose

recover · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/returning-to-a-recorded-pose/SKILL.md) · [upstream](https://github.com/graph-robots/graph-as-policy/tree/main/examples/steered_policy/graph_loop)

Opens the gripper and drives the arm back to a joint configuration recorded earlier in the episode, then checks from a fresh observation that it arrived. Use when a graph must undo a failed attempt and retry from a known start (a grasp that missed, an insert that jammed, a policy that drifted), or hand a learned policy the start pose it was trained from, and the joint state to return to was captured earlier with a robot.get_observation tool node.

- **In**: `recorded_joints` JointState
- **Out**: `max_joint_error` float
- **Calls**: connector `robot.get_observation`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`return_to_pose`](skills/returning-to-a-recorded-pose/scripts/return_to_pose.py)
- **Exits**:
  - `returned`: The arm stands at the recorded joints, every joint within max_error (default 0.08 rad) on a fresh observation, gripper open.
  - `not_returned`: The move was refused or failed, or a joint ended more than max_error off; `detail` says which.

### classifying-failure-type-vlm

judge · robots: any · tasks: pick-place, failure-detection · MIT · v0.1.0 · [`SKILL.md`](skills/classifying-failure-type-vlm/SKILL.md) · [upstream](https://github.com/MINT-SJTU/RoboFAC/blob/638eb5320d409e4072710a0db26f1c0d68cef3de/evaluation/eval.py)

Classifies why a manipulation step failed from camera frames taken before, during (optional, a short video as frames) and after it - asks a VLM RoboFAC's failure questions (detection, its six failure types, explanation, high- and low-level correction) together with a routing class and exits on the class - missed_grasp, slipped, wrong_object, misaligned, collision_or_stuck, incomplete, or no_failure - with the VLM's explanation and correction. Use when a graph knows (or suspects) a step went wrong and must pick the recovery that fits the cause, with only cameras, after capturing a robot.get_observation before the step.

- **In**: `before` Observation, `task` string, `step` string
- **Out**: `failed` bool, `failure_class` string, `robofac_type` string, `misalignment` string, `explanation` string, `correction` string, `low_level_correction` string
- **Calls**: [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`classify_failure_vlm`](skills/classifying-failure-type-vlm/scripts/classify_failure_vlm.py)
- **Exits**:
  - `no_failure`: The VLM said the step succeeded (AFTER shows its intended result); the class is ignored.
  - `missed_grasp`: The gripper closed but the object is not in it - closed beside it or on air, or pushed it away.
  - `slipped`: The object was held at some point and is no longer in the hand - slipped or fell while lifting or carrying.
  - `wrong_object`: The robot grasped, moved or holds an object other than the one the task names.
  - `misaligned`: The right object, the step carried out, but the object or hand ended offset or turned from its target (placed beside the box, tilted); `misalignment` is translation / rotation when RoboFAC's type says which.
  - `collision_or_stuck`: The hand or the object ran into something (a wall, a fixture, another object, the table) and stopped or jammed.
  - `incomplete`: The step stopped short or part of it was never done, with no collision and no wrong object.
  - `no_verdict`: No camera has RGB in both BEFORE and AFTER, vlm.query raised, the reply had no JSON, no success verdict, or a failure with neither a class nor a RoboFAC type that maps to one.

### detecting-subtask-success

judge · robots: any · tasks: pick-place, sorting, insertion, cable-routing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/detecting-subtask-success/SKILL.md) · [upstream](https://github.com/google-deepmind/gemini-robotics-sdk/blob/26b49f1a15b96f1e65b05160f5b991d6167aba74/safari_sdk/agent/framework/tools/success_detection.py)

Checks whether a subtask is done from the latest frame of every camera at one moment - sends them to a VLM with the Gemini Robotics SDK's four guided questions (object states, relations, required end state, differences) and reads its "FINAL ANSWER yes / no", once or polling fresh frames until yes or a timeout. Use when a graph needs a yes/no "is &lt;subtask&gt; done now?" gate after a step or while a policy runs, with only cameras and no frame from before the step; use judging-step-with-before-now-vlm instead when the decision needs a BEFORE frame or a recovery choice.

- **In**: `subtask` string
- **Out**: `success` bool, `answer` string, `checks` int, `follow_ups` int, `errors` int, `elapsed_s` float, `reason` string, `raw` string
- **Calls**: [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`detect_success`](skills/detecting-subtask-success/scripts/detect_success.py)
- **Exits**:
  - `succeeded`: The VLM's reply ended in FINAL ANSWER yes for the latest frames (on the first check, or on any check before timeout_s).
  - `not_yet`: Asked once (timeout_s 0) and the reply was FINAL ANSWER no, or had no FINAL ANSWER even after the follow-up (read as no, as upstream did).
  - `timed_out`: timeout_s &gt; 0 passed without a yes; `checks` replies were all no or unparsed.
  - `no_view`: Nothing was judged - no camera had RGB in any observation, or every vlm.query call raised; `reason` says which.

### judging-step-with-before-now-vlm

judge · robots: any · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/judging-step-with-before-now-vlm/SKILL.md) · [upstream](https://github.com/NVlabs/VoLoAgent/blob/b4e623079ca8498a16bcd5016920d71f76c44d30/vlm_orchestrator/failure_handlers/vlm.py)

Judges the current subgoal of a task from camera frames taken before the step and now - sends the BEFORE and NOW frames of each camera, the task, the subgoal and the subgoals after it to a VLM that returns a status (complete / in_progress / failure) and an action (next / continue / replan / grasp_tool / place_tool), and routes on the action. Use when a graph must decide after (or during) a step whether to advance, keep going, replan, or recover a wrong object or a wrong destination, with only cameras, after capturing a robot.get_observation before the step.

- **In**: `before` Observation, `task` string, `subgoal` string
- **Out**: `status` string, `action` string, `reason` string, `grasp_target` string, `place_destination` string, `place_held_object` string
- **Calls**: [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`judge_step`](skills/judging-step-with-before-now-vlm/scripts/judge_step.py)
- **Exits**:
  - `complete`: The VLM chose "next" - clear visual evidence in NOW that the subgoal is done (its object at its destination).
  - `in_progress`: The VLM chose "continue" - not done yet but still achievable, including not started; also any action outside recovery_mode's vocabulary.
  - `failed`: The VLM chose "replan" - dropped, stuck, no visible progress, or the plan no longer fits the NOW scene; `reason` says which.
  - `wrong_object`: The VLM chose "grasp_tool" - the hand holds or is closing on an object other than the subgoal's; `grasp_target` names the right one.
  - `wrong_destination`: The VLM chose "place_tool" - the held object is heading to or was released at another destination; `place_destination` is a freeform phrase for the right one.
  - `no_verdict`: No camera has RGB in both BEFORE and NOW, vlm.query raised, or its reply had no JSON; nothing was judged (VoLoAgent continued in this case).

## Policy details

### molmoact-libero

execute · robots: panda, **sim only** · tasks: libero · Apache-2.0 · v0.1.0 · [`SKILL.md`](policies/molmoact-libero/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/policies/molmoact-libero)

Run the MolmoAct LIBERO checkpoint (allenai/MolmoAct-7B-D-LIBERO-0812) as a closed-loop VLA policy for the dexterous pick-and-place segment of a task. Drives a Franka Panda in the LIBERO/robosuite OSC_POSE action space from agentview + wrist cameras, served behind a vLLM-style script speaking the openpi websocket protocol; the policy server is the bundle's own preset (no policy_id). Reads the graph-scoped observation_stream each window and terminates on a gripper open→close→open cycle, a VLM yes/no check, or max_windows. Use when a pick/place (or pick-and-drop-in-container) segment on tabletop rigid LIBERO objects is delegated to a learned policy — best steered (perceive + hover above the target) first; this is the MolmoAct alternative to pi05-libero for the same task family. NOT for deformables/cloth folding, articulated objects, non-Franka embodiments, or tasks outside the LIBERO pick-place distribution.

- **Calls**: [`vlm`](#vlm); connector `robot.execute_trajectory`, `robot.get_observation`, `sim.apply_policy_action`
- **Trained on**: action `osc_pose_delta` ×7 at 20 Hz; state 8 (eef_pos, eef_axis_angle, gripper_qpos); cameras `agentview`, `wrist`
- **Weights**: `hf://allenai/MolmoAct-7B-D-LIBERO-0812`
- **Runs**: in its own environment over `websocket`, needs a GPU
- **Note**: LIBERO observation and OSC_POSE action conventions; drives the arm through sim.apply_policy_action.
- **Exits**:
  - `gripper_cycle`: One grasp/release (open→close→open) cycle completed — one item picked and released.
  - `completed_by_vlm`: The VLM termination prompt answered yes.
  - `max_windows`: The hard window cap was reached without a completion signal (the policy ran but did not declare itself done).
  - `failed`: The policy loop errored (server/inference/execution failure). This is the subgraph's on_error exit.
- **Functions**:
  - `molmoact-libero.run`: Run the MolmoAct LIBERO checkpoint in closed loop (replan/execute/terminate) against its own preset server.

### pi05-libero

execute · robots: panda, **sim only** · tasks: libero · Apache-2.0 · v0.1.0 · [`SKILL.md`](policies/pi05-libero/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/policies/pi05-libero)

Run the openpi π0.5 LIBERO checkpoint (pi05_libero) as a closed-loop VLA policy for the dexterous pick-and-place segment of a task. Drives a Franka Panda in the LIBERO/robosuite OSC_POSE action space from agentview + wrist cameras; the policy server is the bundle's own preset (no policy_id). Reads the graph-scoped observation_stream each window and terminates on a gripper open→close→open cycle, a VLM yes/no check, or max_windows. Use when a pick/place (or pick-and-drop-in-container) segment on tabletop rigid LIBERO objects is delegated to a learned policy — best steered (perceive + hover above the target) first. NOT for deformables/cloth folding, articulated objects, non-Franka embodiments, or tasks outside the LIBERO pick-place distribution.

- **Calls**: [`vlm`](#vlm); connector `robot.execute_trajectory`, `robot.get_observation`, `sim.apply_policy_action`
- **Trained on**: action `osc_pose_delta` ×7 at 20 Hz; state 8 (eef_pos, eef_axis_angle, gripper_qpos); cameras `agentview`, `wrist`
- **Weights**: `s3://openpi-assets/checkpoints/pi05_libero`
- **Runs**: in its own environment over `websocket`, needs a GPU
- **Note**: LIBERO observation and OSC_POSE action conventions; drives the arm through sim.apply_policy_action.
- **Exits**:
  - `gripper_cycle`: One grasp/release (open→close→open) cycle completed — one item picked and released.
  - `completed_by_vlm`: The VLM termination prompt answered yes.
  - `max_windows`: The hard window cap was reached without a completion signal (the policy ran but did not declare itself done).
  - `failed`: The policy loop errored (server/inference/execution failure). This is the subgraph's on_error exit.
- **Functions**:
  - `pi05-libero.run`: Run the π0.5 LIBERO checkpoint in closed loop (replan/execute/terminate) against its own preset server.

## Recipes

- [`pick-place-camera-only`](recipes/pick-place-camera-only/RECIPE.md): Put one named rigid object into a named container from the cameras alone. Uses [`grasping-short-axis`](#grasping-short-axis), [`grasping-with-planner`](#grasping-with-planner), [`perceiving-objects`](#perceiving-objects), [`transporting-objects`](#transporting-objects), [`verifying-grasps`](#verifying-grasps).

## Papers

- [`curobo-2023`](papers/curobo-2023/PAPER.md): cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation (2023)
