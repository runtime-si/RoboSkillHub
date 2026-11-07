# Catalog

Every bundle in this library, generated from the `SKILL.md` files by
`roboskillhub catalog`; a test fails when this page is out of date. The
overview has one line per bundle, and each name leads to its section: the
full description a planner reads, what goes in and comes out, what each exit
means, and what it calls.

9 tools, 57 skills, 2 policies, 1 recipe and 1 paper.

Under robots, `any` means the bundle reads the robot through `describe_*`
tools; *tuned on* names the robot its constants were measured on; **sim only**
means it reads simulator ground truth (`sim.*`) and never runs on a real robot.

## Tools

| Tool | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`curve`](#curve) | perceive | Ordered centrelines of deformable linear objects. | any |
| [`gemini-er`](#gemini-er) | perceive | Open-vocabulary 2D detection and pointing via the Gemini Robotics-ER API. | any |
| [`geometry`](#geometry) | perceive | Pure-math 3D geometry toolbox. | any, *tuned on* panda |
| [`grounding-dino`](#grounding-dino) | perceive | Grounding DINO zero-shot object detection — natural-language queries to labeled 2D bounding boxes with confidence scores. | any |
| [`molmo`](#molmo) | perceive | Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). | any |
| [`sam3`](#sam3) | perceive | Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker… | any |
| [`vlm`](#vlm) | perceive | Free-form and yes/no visual question answering against a hosted vision-language model (OpenRouter API by default… | any |
| [`curobo`](#curobo) | plan | NVIDIA cuRobo motion planning. | any, *tuned on* panda |
| [`predicates`](#predicates) | judge | Geometric yes/no judgments over poses and boxes the graph already holds. | any |

## Skills

| Skill | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`detecting-contact-from-joint-effort`](#detecting-contact-from-joint-effort) | perceive | Feels for contact without a force sensor. | any |
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
| [`registering-held-objects`](#registering-held-objects) | perceive | Reobserves a grasped rigid object from the wrist cameras, estimates its functional feature in the TCP frame, and fits… | any |
| [`tracking-objects`](#tracking-objects) | perceive | Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. | any |
| [`computing-feature-mating-poses`](#computing-feature-mating-poses) | plan | Computes approach, engaged, and mate TCP poses from a held feature in the hand and a fixture feature, resolving free… | any |
| [`planning-articulation-opening`](#planning-articulation-opening) | plan | Plans the hand's path that opens a drawer or a door it already holds by the handle. | any |
| [`planning-held-object-motion`](#planning-held-object-motion) | plan | Plans the carry and engagement phases for an already-held rigid object. | any |
| [`grasping-concave-contour`](#grasping-concave-contour) | propose | Grasp an awkward rigid part by the waists in its own silhouette. | any |
| [`proposing-short-axis-grasps`](#proposing-short-axis-grasps) | propose | Propose top-down grasps whose jaws close ACROSS an elongated object's short axis, derived from its oriented box. | any, **sim only** |
| [`proposing-side-grasps`](#proposing-side-grasps) | propose | Propose horizontal (side-entry) grasps swept around an object's oriented box — twelve azimuths ranked by jaw margin. | any, **sim only** |
| [`selecting-free-placement-spots`](#selecting-free-placement-spots) | select | Picks world-frame spots on a named surface where a held object can be set down. | any, *tuned on* yam-station-overhead-rgbd |
| [`selecting-reachable-grasp`](#selecting-reachable-grasp) | select | Walk a ranked list of candidate grasp poses and take the first the arm can actually reach. | any, **sim only** |
| [`selecting-safe-cable-grasp-segments`](#selecting-safe-cable-grasp-segments) | select | Ranks top-down grasp points along a cable's fitted centreline that keep the fingers off its ends (a larger margin at… | any, *tuned on* yam-station-overhead-rgbd |
| [`anchoring-a-free-end`](#anchoring-a-free-end) | execute | Pins one end of a cable with the support hand before any crossing is attempted. | any, 2 arms |
| [`approaching-above-a-target`](#approaching-above-a-target) | execute | Moves the hand to hover a set clearance above a target's oriented box, rising in place first and keeping the hand's… | any |
| [`executing-feature-mating`](#executing-feature-mating) | execute | Executes a supplied feature-mating plan for an already-held rigid object. | any |
| [`executing-held-object-motion`](#executing-held-object-motion) | execute | Executes an ordered, collision-aware pose sequence for an already-grasped rigid object without releasing it. | any |
| [`grasping-container-rims`](#grasping-container-rims) | execute | Grasps a thin-walled open container (box, bin, tray, cup, bowl) by its rim. | any |
| [`grasping-direct-ik`](#grasping-direct-ik) | execute | Direct IK align-then-descend grasping. | any, *tuned on* panda |
| [`grasping-linear-feature`](#grasping-linear-feature) | execute | Fit the full 3D axis of an elongated segmented feature and grasp it with a perpendicular, inclination-aware… | any |
| [`grasping-short-axis`](#grasping-short-axis) | execute | Deterministic short-axis-aligned grasp with CuRobo. | any, *tuned on* panda |
| [`grasping-with-planner`](#grasping-with-planner) | execute | Top-down grasping via a fast axis-locked linear descend with a collision-aware cuRobo fallback. | any, *tuned on* panda |
| [`handing-over-between-arms`](#handing-over-between-arms) | execute | Hands the object one arm holds to the other arm. | any, 2 arms |
| [`holding-steady-while-other-arm-acts`](#holding-steady-while-other-arm-acts) | execute | One arm holds an object still. | any, 2 arms |
| [`picking-and-placing-two-objects`](#picking-and-placing-two-objects) | execute | Picks two objects with two arms, one each, and places both. | any, 2 arms |
| [`placing-held-feature`](#placing-held-feature) | execute | Move a held object so one of its declared features lands on a target point. | any, **sim only** |
| [`placing-with-contact-descent`](#placing-with-contact-descent) | execute | Places the object the hand holds by approaching a fixed height above its place point, coming straight down until the… | any |
| [`plugging-a-cable-end`](#plugging-a-cable-end) | execute | Carries a cable's held free end into a terminal port after the last crossing. | any, 2 arms |
| [`pouring-by-wrist-roll`](#pouring-by-wrist-roll) | execute | Pours from a held container into a target container by standing the hand above the target's box at a set height and… | any |
| [`pushing-objects`](#pushing-objects) | execute | Pushes a loose object across its support toward a goal point with a closed hand, never grasping it. | any |
| [`reaching-with-lag-correction`](#reaching-with-lag-correction) | execute | Moves the tool to a target pose, waits for the arm to settle, reads the tool pose back and, if the arm stopped short… | any |
| [`searching-insertion-spiral`](#searching-insertion-spiral) | execute | Finds an opening the held part or the closed fingertips can enter by probing straight down over a hexagonal grid… | any |
| [`seating-a-cable-crossing`](#seating-a-cable-crossing) | execute | Streams one planned crossing leg of a cable onto a routing station, lets the rod settle, then LOOKS. | any |
| [`stacking-objects`](#stacking-objects) | execute | Stacks the object the hand holds centred on top of a target object. | any |
| [`tipping-over-a-surface-edge`](#tipping-over-a-surface-edge) | execute | Turn a body over on the surface it rests on instead of carrying it. | any, *tuned on* yam_i2rt, 2 arms, **sim only** |
| [`transporting-objects`](#transporting-objects) | execute | Move the currently-held object to a destination and release. | any, *tuned on* panda |
| [`wiping-a-region`](#wiping-a-region) | execute | Wipes a flat region of a surface with back-and-forth strokes. | any |
| [`checking-spatial-relations`](#checking-spatial-relations) | verify | Decides from one RGB-D view whether a spatial relation holds between two seen objects. | any |
| [`classifying-failure-from-proprioception`](#classifying-failure-from-proprioception) | verify | Classifies the last moments of a manipulation step from proprioception alone. | any |
| [`previewing-target-overlay`](#previewing-target-overlay) | verify | Checks a target gripper pose before the arm moves. | any |
| [`verifying-a-cable-route`](#verifying-a-cable-route) | verify | Judge a finished cable route from a camera rather than from an evaluator. | any |
| [`verifying-grasps`](#verifying-grasps) | verify | Three independent checks on one grasp attempt. | any |
| [`verifying-insertion-by-probe-push`](#verifying-insertion-by-probe-push) | verify | Checks whether a held part is seated after an insertion by pushing it a few millimetres further along the insertion… | any |
| [`verifying-placement`](#verifying-placement) | verify | Confirms from one RGB-D view that a released object went through an aperture, accepting either that it vanished into… | any |
| [`centering-a-part-with-the-gripper`](#centering-a-part-with-the-gripper) | recover | Centres a loose part between the jaws without perceiving it. | any |
| [`returning-to-a-recorded-pose`](#returning-to-a-recorded-pose) | recover | Opens the gripper and drives the arm back to a joint configuration recorded earlier in the episode, then checks from a… | any |
| [`detecting-subtask-success`](#detecting-subtask-success) | judge | Checks whether a subtask is done from the latest frame of every camera at one moment. | any |
| [`judging-step-with-before-now-vlm`](#judging-step-with-before-now-vlm) | judge | Judges the current subgoal of a task from camera frames taken before the step and now. | any |

## Policies

| Policy | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`molmoact-libero`](#molmoact-libero) | execute | Run the MolmoAct LIBERO checkpoint (allenai/MolmoAct-7B-D-LIBERO-0812) as a closed-loop VLA policy for the dexterous… | panda, **sim only** |
| [`pi05-libero`](#pi05-libero) | execute | Run the openpi π0.5 LIBERO checkpoint (pi05_libero) as a closed-loop VLA policy for the dexterous pick-and-place… | panda, **sim only** |

## Tool details

### curve

perceive · robots: any · tasks: cable-routing · Apache-2.0 · v0.2.0 · [`SKILL.md`](tools/curve/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/curve)

Ordered centrelines of deformable linear objects — skeletonises a cable, rope or hose mask, recovers one traversal order through occlusion breaks and self-crossings (TrackDLO's chain merge), back-projects the ordered pixels through depth onto the object's axis as an arc-length-parameterised polyline, and carries a known centreline onto the next frame with motion imputed for hidden stretches, and reads off its two ends (tip position, outward direction, pixel). Use when a workflow needs an ordered centreline of a cable, rope or hose from a mask and a depth frame, must carry one through occlusion between frames, or needs where a cable's tip is and which way it points.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `curve.fit_centerline`: Fit an ordered, arc-length-parameterised 3D centreline to a deformable linear object from its mask and a depth frame.
  - `curve.track_centerline`: Advance a known centreline onto a new frame, imputing motion for occluded stretches from their visible neighbours.
  - `curve.endpoints`: Both ends of a centreline -- each end's position, outward direction and pixel -- and which end a pixel or point hint names.

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

### molmo

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/molmo/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/molmo)

Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). Use when a workflow needs a single pixel coordinate for a named object (point_prompt) or Molmo-grade visual question answering and a vLLM server is available; for an API-only zero-GPU alternative use the gemini-er bundle.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `molmo.point_prompt`: Point at a named object in an image; returns pixel coordinates.
  - `molmo.query`: Free-form visual question answering via Molmo.
  - `molmo.query_yes_no`: Yes/no visual question answering via Molmo; coerces the reply to a bool.

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
