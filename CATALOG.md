# Catalog

Every bundle in this library, generated from the `SKILL.md` files by
`roboskillhub catalog`; a test fails when this page is out of date. The
overview has one line per bundle, and each name leads to its section: the
full description a planner reads, what goes in and comes out, what each exit
means, and what it calls.

29 tools, 153 skills, 2 policies, 1 recipe and 1 paper.

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
| [`mm-grounding-dino`](#mm-grounding-dino) | perceive | MM-Grounding-DINO open-vocabulary detection (OpenMMLab's open retraining of Grounding DINO, Apache-2.0 weights) with… | any, *tuned on* yam-station-orbbec-overhead |
| [`molmo`](#molmo) | perceive | Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). | any |
| [`move-classifier`](#move-classifier) | perceive | Trains and runs the 27-class "which way to nudge" classifier of vision-guided rack insertion (ResNet18, ImageNet… | any |
| [`omdet-turbo`](#omdet-turbo) | perceive | OmDet-Turbo fast open-vocabulary detection (omlab/omdet-turbo-swin-tiny-hf, Apache-2.0). | any, *tuned on* yam-station-orbbec-overhead |
| [`owlv2`](#owlv2) | perceive | OWLv2 open-vocabulary detection (google/owlv2-*, Apache-2.0) in two modes. | any |
| [`pose6d`](#pose6d) | perceive | 6D pose of a known rigid part from depth. | any |
| [`rfdetr`](#rfdetr) | perceive | RF-DETR (Roboflow, Apache-2.0) train-and-run tools for a small station-specific detector. | any |
| [`sahi`](#sahi) | perceive | Slicing-aided detection (SAHI) of small parts. | any |
| [`sam2`](#sam2) | perceive | SAM 2.1 promptable video mask tracking (facebook/sam2.1-hiera-*, Apache-2.0; EdgeTAM as the fast option). | any, *tuned on* yam-station-orbbec-overhead, yam-left-wrist-d405 |
| [`sam3`](#sam3) | perceive | Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker… | any |
| [`track`](#track) | perceive | Google DeepMind TAPNext++ run locally on a GPU. | any |
| [`vlm`](#vlm) | perceive | Free-form and yes/no visual question answering against a hosted vision-language model (OpenRouter API by default… | any |
| [`curobo`](#curobo) | plan | NVIDIA cuRobo motion planning. | any, *tuned on* panda |
| [`grconvnet`](#grconvnet) | propose | Proposes ranked top-down parallel-jaw grasps from one RGB-D frame with GR-ConvNet (skumra/robotic-grasping), on CPU… | any |
| [`molmo-motion`](#molmo-motion) | propose | AllenAI MolmoMotion-4B run locally on a GPU. | any |
| [`robobrain`](#robobrain) | propose | BAAI RoboBrain 2.5-4B run locally on a GPU. | any |
| [`workcell`](#workcell) | execute | Typed JSON boundary for RoboSkillHub workcell capabilities and station descriptions. | yam_i2rt |
| [`musc`](#musc) | verify | MuSc zero-shot mutual anomaly scoring (MuSc / MuSc-V2, MIT) on DINOv2 (facebook/dinov2-base, Apache-2.0). | any |
| [`densereward`](#densereward) | judge | DenseReward reward models (Qwen3-VL-4B fine-tunes, Apache-2.0) run locally on a GPU. | any |
| [`predicates`](#predicates) | judge | Geometric yes/no judgments over poses and boxes the graph already holds. | any |
| [`robojev`](#robojev) | judge | RoboJev-27B success judge (josefchen/RoboJev-27B, Apache-2.0). | any |
| [`rynnvalue`](#rynnvalue) | judge | RynnValue value model (RynnBrain / Qwen3-VL, 8B or 4B, Apache-2.0) run locally on a GPU. | any |

## Skills

| Skill | Stage | What it does | Robots |
| --- | --- | --- | --- |
| [`calibrating-arm-to-arm-base-transform`](#calibrating-arm-to-arm-base-transform) | perceive | Measures where one arm's base sits in the other arm's base frame. | any |
| [`calibrating-baseplate-stud-grid`](#calibrating-baseplate-stud-grid) | perceive | Estimates a LEGO or DUPLO baseplate's pose (the plate frame computing-brick-pose-from-stud-grid takes, and APEX-MR's… | any |
| [`classifying-brick-resting-face-from-depth`](#classifying-brick-resting-face-from-depth) | perceive | Tells from depth whether a loose LEGO or DUPLO brick of known size lies studs up, upside down (resting on its stud… | any |
| [`classifying-piece-sides-for-border-first`](#classifying-piece-sides-for-border-first) | perceive | Types a jigsaw piece's four sides flat / tab / blank from a top-down mask and calls it a corner, edge or interior piece. | any |
| [`detecting-contact-from-joint-effort`](#detecting-contact-from-joint-effort) | perceive | Feels for contact without a force sensor. | any |
| [`estimating-cut-grid-from-pieces`](#estimating-cut-grid-from-pieces) | perceive | Estimates a jigsaw's real cut grid (cols x rows) and cell size from the piece count printed on the box, the board's… | any |
| [`estimating-relative-rotation-and-symmetry`](#estimating-relative-rotation-and-symmetry) | perceive | How far a part is turned in the image plane between a reference view and the current view (both from the wrist camera… | any, *tuned on* yam-left-wrist-d405-renders |
| [`framing-a-target-by-mask-edge-steering`](#framing-a-target-by-mask-edge-steering) | perceive | Moves the wrist camera until a named target lies wholly inside its image with a margin. | any |
| [`localizing-a-piece-in-a-reference-image`](#localizing-a-piece-in-a-reference-image) | perceive | Finds which grid cell of a jigsaw's box picture one piece belongs to, and how far the piece is turned, from a close-up… | any |
| [`observing-object-poses`](#observing-object-poses) | perceive | Reads independently observed object positions, and orientations where the backend provides them, from the configured… | yam_i2rt |
| [`perceiving-deformable-linear-objects`](#perceiving-deformable-linear-objects) | perceive | Read a cable's ordered 3D centreline from one RGB-D frame and keep it current across a task. | any |
| [`perceiving-functional-features`](#perceiving-functional-features) | perceive | Locates a language-described functional part of a rigid object or fixture from calibrated RGB-D and fits a typed loop… | any |
| [`perceiving-next-item`](#perceiving-next-item) | perceive | Loop-head perception for pack-all / clean-all-items tasks. | any, *tuned on* panda |
| [`perceiving-object-parts`](#perceiving-object-parts) | perceive | Hierarchical perception for subpart targeting. | any |
| [`perceiving-objects`](#perceiving-objects) | perceive | Fast single-path 3D object perception. | any, *tuned on* panda |
| [`perceiving-objects-oneshot`](#perceiving-objects-oneshot) | perceive | Lightweight one-shot 3D object perception. | any, *tuned on* panda |
| [`perceiving-relational-correspondences`](#perceiving-relational-correspondences) | perceive | Perceives multiple source and destination instances and resolves language-conditioned one-to-one or many-to-one… | any |
| [`perceiving-routing-fixtures`](#perceiving-routing-fixtures) | perceive | Survey a cable-routing bench from one overhead RGB-D frame. | any |
| [`perceiving-sorting-pairs`](#perceiving-sorting-pairs) | perceive | Discovers a labelled four-compartment destination from RGB-D and repeatedly perceives one remaining source object with… | any |
| [`reading-domino-pips-and-orientation`](#reading-domino-pips-and-orientation) | perceive | Reads each domino's two pip values, which end is which, its long axis and whether it lies flat, stands or has fallen… | any |
| [`reconstructing-collision-worlds`](#reconstructing-collision-worlds) | perceive | Reconstructs a planner collision world from calibrated RGB-D views alone, excluding the grasp target, the robot, and a… | any |
| [`refining-target-from-wrist-closeup`](#refining-target-from-wrist-closeup) | perceive | Refines a coarse world-frame target (an overhead perception's OBB or point) by aiming the wrist RGB-D camera at it… | any |
| [`registering-held-objects`](#registering-held-objects) | perceive | Reobserves a grasped rigid object from the wrist cameras, estimates its functional feature in the TCP frame, and fits… | any |
| [`scanning-the-board-into-a-wrist-mosaic`](#scanning-the-board-into-a-wrist-mosaic) | perceive | Scans a table region (the jigsaw board, a staging area) with the wrist camera from straight above and blends the… | any |
| [`segmenting-and-picking-pieces-classical`](#segmenting-and-picking-pieces-classical) | perceive | Finds the loose jigsaw pieces lying on a staging mat in one top-down RGB(-D) frame without a learned model. | any |
| [`tracking-build-state`](#tracking-build-state) | perceive | Keeps a world model of the LEGO or DUPLO bricks placed so far and their stud connections. | any |
| [`tracking-identical-parts-by-id`](#tracking-identical-parts-by-id) | perceive | Keeps a stable id for each of many look-alike parts (dominoes, same-colour bricks, jigsaw pieces, cubes) across… | any |
| [`tracking-objects`](#tracking-objects) | perceive | Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. | any |
| [`tracking-points`](#tracking-points) | perceive | Tracks selected image points through the configured workcell tracking backend and reports visibility evidence. | yam_i2rt |
| [`training-a-part-detector-from-auto-labels`](#training-a-part-detector-from-auto-labels) | perceive | Gets a fast station-specific detector for a fixed set of parts (bricks, dominoes, jigsaw pieces) without hand labelling. | any, *tuned on* yam-station-orbbec-overhead, yam-left-wrist-d405 |
| [`aligning-parts-by-connector-constraints`](#aligning-parts-by-connector-constraints) | plan | Turns connector statements about a LEGO or DUPLO build. | any |
| [`assigning-bricks-to-two-arms`](#assigning-bricks-to-two-arms) | plan | Assigns every brick of a LEGO or DUPLO build sequence to an arm. | any |
| [`checking-design-connectivity-and-collisions`](#checking-design-connectivity-and-collisions) | plan | Checks a brick design on a stud grid before the arms build it. | any |
| [`choosing-a-domino-move-by-end-matching`](#choosing-a-domino-move-by-end-matching) | plan | Chooses the robot's move in a game of dominoes and where the tile goes on the table. | any |
| [`computing-brick-pose-from-stud-grid`](#computing-brick-pose-from-stud-grid) | plan | Converts between a cell on a LEGO or DUPLO baseplate's stud grid (x, y, layer, brick size such as 2x4, rotation… | any |
| [`computing-feature-mating-poses`](#computing-feature-mating-poses) | plan | Computes approach, engaged, and mate TCP poses from a held feature in the hand and a fixture feature, resolving free… | any |
| [`computing-stable-placements`](#computing-stable-placements) | plan | Enumerates the stable resting faces of a rigid object. | any |
| [`executing-keypoint-motion`](#executing-keypoint-motion) | plan | Plans and executes a one-step manipulation from a language instruction with MOKA's mark-based visual prompting. | any |
| [`finding-the-longest-domino-chain`](#finding-the-longest-domino-chain) | plan | Finds the longest chain of matching dominoes a hand can lay from a start value. | any |
| [`parsing-ldraw-build-instructions`](#parsing-ldraw-build-instructions) | plan | Reads an LDraw .ldr / .mpd build file. | any |
| [`planning-a-brick-mosaic-from-an-image`](#planning-a-brick-mosaic-from-an-image) | plan | Turns an image into a one-layer LEGO mosaic on a baseplate of a given stud size. | any |
| [`planning-a-domino-layout`](#planning-a-domino-layout) | plan | Plans a domino line along a path on the table. | any |
| [`planning-a-flat-domino-mosaic-from-an-image`](#planning-a-flat-domino-mosaic-from-an-image) | plan | Plans a domino mosaic of a picture, every tile lying flat with its pips up. | any |
| [`planning-articulation-opening`](#planning-articulation-opening) | plan | Plans the hand's path that opens a drawer or a door it already holds by the handle. | any |
| [`planning-brick-assembly-order`](#planning-brick-assembly-order) | plan | Orders the bricks of a stud-grid design for a robot that presses each one down from above. | any |
| [`planning-held-object-motion`](#planning-held-object-motion) | plan | Plans the carry and engagement phases for an already-held rigid object. | any |
| [`routing-parts-to-sort-bins-by-id-and-colour`](#routing-parts-to-sort-bins-by-id-and-colour) | plan | Sorts parts dropped one at a time on a staging pad into bins. | any |
| [`following-a-visual-plan`](#following-a-visual-plan) | propose | Follows a visual plan from robobrain.trace or molmo-motion.forecast (the shared visual-plan schema) the way those… | any |
| [`grasping-concave-contour`](#grasping-concave-contour) | propose | Grasp an awkward rigid part by the waists in its own silhouette. | any |
| [`proposing-short-axis-grasps`](#proposing-short-axis-grasps) | propose | Propose top-down grasps whose jaws close ACROSS an elongated object's short axis, derived from its oriented box. | any, **sim only** |
| [`proposing-side-grasps`](#proposing-side-grasps) | propose | Propose horizontal (side-entry) grasps swept around an object's oriented box — twelve azimuths ranked by jaw margin. | any, **sim only** |
| [`choosing-next-piece-and-slot`](#choosing-next-piece-and-slot) | select | Chooses the next jigsaw placement (which loose piece, which grid cell, which turn) from the board as believed, the… | any |
| [`matching-a-piece-to-placed-neighbours`](#matching-a-piece-to-placed-neighbours) | select | Picks which of several loose jigsaw pieces fits an empty slot, and how far to turn it, by the shape of its sides alone. | any |
| [`proposing-parallel-grasps-in-tight-clusters`](#proposing-parallel-grasps-in-tight-clusters) | select | Proposes top-down parallel-jaw grasps on small flat parts packed close together (jigsaw pieces, LEGO bricks, dominoes… | any |
| [`sampling-labeled-points-in-mask`](#sampling-labeled-points-in-mask) | select | Picks the point on (or in) an object that a language instruction asks for. | any, *tuned on* yam-station-overhead-rgbd |
| [`selecting-a-grip-clear-of-placed-neighbours`](#selecting-a-grip-clear-of-placed-neighbours) | select | Chooses where a two-finger gripper should hold a jigsaw piece, and along which direction it should close, so that at… | any |
| [`selecting-free-placement-spots`](#selecting-free-placement-spots) | select | Picks world-frame spots on a named surface where a held object can be set down. | any, *tuned on* yam-station-overhead-rgbd |
| [`selecting-reachable-grasp`](#selecting-reachable-grasp) | select | Walk a ranked list of candidate grasp poses and take the first the arm can actually reach. | any, **sim only** |
| [`selecting-safe-cable-grasp-segments`](#selecting-safe-cable-grasp-segments) | select | Ranks top-down grasp points along a cable's fitted centreline that keep the fingers off its ends (a larger margin at… | any, *tuned on* yam-station-overhead-rgbd |
| [`anchoring-a-free-end`](#anchoring-a-free-end) | execute | Pins one end of a cable with the support hand before any crossing is attempted. | any, 2 arms |
| [`approaching-above-a-target`](#approaching-above-a-target) | execute | Moves the hand to hover a set clearance above a target's oriented box, rising in place first and keeping the hand's… | any |
| [`checking-reachability`](#checking-reachability) | execute | Asks a station whether TCP poses (optionally as a chained path from the current configuration) are reachable, using… | yam_i2rt |
| [`correcting-brick-offset-from-stud-circles`](#correcting-brick-offset-from-stud-circles) | execute | Closes the last millimetre over a LEGO or DUPLO brick before a pick or a press. | any |
| [`correcting-in-plane-offset-by-discrete-moves`](#correcting-in-plane-offset-by-discrete-moves) | execute | Final x, y and yaw alignment of a held part (or the hand) over its target from the wrist camera, by a small classifier… | any |
| [`disassembling-a-brick-by-twist-lever`](#disassembling-a-brick-by-twist-lever) | execute | Removes a seated LEGO or DUPLO brick with a parallel gripper and no force sensor. | any |
| [`executing-feature-mating`](#executing-feature-mating) | execute | Executes a supplied feature-mating plan for an already-held rigid object. | any |
| [`executing-held-object-motion`](#executing-held-object-motion) | execute | Executes an ordered, collision-aware pose sequence for an already-grasped rigid object without releasing it. | any |
| [`grasping-a-cable-by-its-plug`](#grasping-a-cable-by-its-plug) | execute | Grasps a cable by the rigid plug at its end rather than by its thin jacket. | any |
| [`grasping-a-cable-lying-flat`](#grasping-a-cable-lying-flat) | execute | Grasps a thin cable lying flat on a table. | any |
| [`grasping-by-wrist-camera-servoing`](#grasping-by-wrist-camera-servoing) | execute | Grasps a small part by closed-loop eye-in-hand servoing. | any |
| [`grasping-container-rims`](#grasping-container-rims) | execute | Grasps a thin-walled open container (box, bin, tray, cup, bowl) by its rim. | any |
| [`grasping-direct-ik`](#grasping-direct-ik) | execute | Direct IK align-then-descend grasping. | any, *tuned on* panda |
| [`grasping-flat-objects`](#grasping-flat-objects) | execute | Grasps an object lying flat on a surface that is too thin for a normal top-down pinch at its box centre (a card, a… | any |
| [`grasping-linear-feature`](#grasping-linear-feature) | execute | Fit the full 3D axis of an elongated segmented feature and grasp it with a perpendicular, inclination-aware… | any |
| [`grasping-short-axis`](#grasping-short-axis) | execute | Deterministic short-axis-aligned grasp with CuRobo. | any, *tuned on* panda |
| [`grasping-with-feedback`](#grasping-with-feedback) | execute | Approaches a supplied grasp pose, closes the gripper and checks available grasp evidence through a configured workcell… | yam_i2rt |
| [`grasping-with-planner`](#grasping-with-planner) | execute | Top-down grasping via a fast axis-locked linear descend with a collision-aware cuRobo fallback. | any, *tuned on* panda |
| [`handing-over-between-arms`](#handing-over-between-arms) | execute | Hands the object one arm holds to the other arm. | any, 2 arms |
| [`holding-steady-while-other-arm-acts`](#holding-steady-while-other-arm-acts) | execute | One arm holds an object still. | any, 2 arms |
| [`inserting-a-held-peg-top-down`](#inserting-a-held-peg-top-down) | execute | Inserts a peg that one arm holds top-down (tip below the fingers) into a perceived hole straight down. | any |
| [`inserting-a-peg-with-pose-refinement`](#inserting-a-peg-with-pose-refinement) | execute | Inserts a held peg into a perceived hole in three stages. | any |
| [`inserting-mid-air-between-two-held-parts`](#inserting-mid-air-between-two-held-parts) | execute | Inserts the part one arm holds into (or presses it onto) the part the other arm holds, both in the air. | any, 2 arms |
| [`lifting-with-both-arms`](#lifting-with-both-arms) | execute | Lifts an elongated object (a bar, a tray, a pot by its handles) with both arms. | any, 2 arms |
| [`moving-along-joint-path`](#moving-along-joint-path) | execute | Moves the arm through a supplied joint-space path on a backend that declares its joints and reports measured joint… | yam_i2rt |
| [`moving-both-arms-together`](#moving-both-arms-together) | execute | Moves both hands to their target poses as one coordinated motion. | any, 2 arms |
| [`moving-through-waypoints`](#moving-through-waypoints) | execute | Moves the measured tool pose through a bounded list of waypoints on the configured YAM or Isaac workcell adapter… | yam_i2rt |
| [`moving-to-pose`](#moving-to-pose) | execute | Moves the measured tool pose through the configured YAM or Isaac workcell adapter. | yam_i2rt |
| [`picking-and-placing-two-objects`](#picking-and-placing-two-objects) | execute | Picks two objects with two arms, one each, and places both. | any, 2 arms |
| [`picking-upright-dominoes-from-a-loading-rack`](#picking-upright-dominoes-from-a-loading-rack) | execute | Takes the next domino out of a slotted loading rack, where tiles stand one per slot at a known pitch, instead of… | any |
| [`placing-an-upright-domino-beside-neighbours`](#placing-an-upright-domino-beside-neighbours) | execute | Picks a standing domino top-down and sets it upright at a slot in a row without touching the tiles already standing… | any |
| [`placing-held-feature`](#placing-held-feature) | execute | Move a held object so one of its declared features lands on a target point. | any, **sim only** |
| [`placing-into-container-held-by-other-arm`](#placing-into-container-held-by-other-arm) | execute | Puts the object one arm holds into an open container the other arm holds. | any, 2 arms |
| [`placing-into-container-obb`](#placing-into-container-obb) | execute | Puts the object the hand holds into an open bin, box or tray the camera sees. | any |
| [`placing-with-contact-descent`](#placing-with-contact-descent) | execute | Places the object the hand holds by approaching a fixed height above its place point, coming straight down until the… | any |
| [`plugging-a-cable-end`](#plugging-a-cable-end) | execute | Carries a cable's held free end into a terminal port after the last crossing. | any, 2 arms |
| [`pouring-by-wrist-roll`](#pouring-by-wrist-roll) | execute | Pours from a held container into a target container by standing the hand above the target's box at a set height and… | any |
| [`pressing-a-brick-onto-studs`](#pressing-a-brick-onto-studs) | execute | Presses a held LEGO or DUPLO brick onto the studs under it and checks by height that it seated. | any |
| [`pressing-in-with-guarded-push`](#pressing-in-with-guarded-push) | execute | Pushes a held part along an axis (straight down by default) in small steps with a bounded push and a bounded travel… | any |
| [`pressing-with-contact-limit`](#pressing-with-contact-limit) | execute | Runs one bounded contact motion along an axis that the controller stops on a force threshold or travel limit, on a… | yam_i2rt |
| [`pulling-drawer`](#pulling-drawer) | execute | Executes a bounded pull along a drawer axis using handle-grasp and opening evidence. | yam_i2rt |
| [`pushing-objects`](#pushing-objects) | execute | Pushes a loose object across its support toward a goal point with a closed hand, never grasping it. | any |
| [`pushing-on-plane`](#pushing-on-plane) | execute | Executes a bounded planar pushing segment and checks measured target displacement. | yam_i2rt |
| [`reaching-with-lag-correction`](#reaching-with-lag-correction) | execute | Moves the tool to a target pose, waits for the arm to settle, reads the tool pose back and, if the arm stopped short… | any |
| [`refilling-a-domino-loading-rack`](#refilling-a-domino-loading-rack) | execute | Puts a standing domino back into an empty slot of a slotted loading rack, so… | any |
| [`rotating-a-held-piece-within-wrist-limits`](#rotating-a-held-piece-within-wrist-limits) | execute | Turns a held flat part (a jigsaw piece) about the vertical by the angle its slot needs while the wrist's last joint… | any |
| [`searching-insertion-spiral`](#searching-insertion-spiral) | execute | Finds an opening the held part or the closed fingertips can enter by probing straight down over a hexagonal grid… | any |
| [`seating-a-cable-crossing`](#seating-a-cable-crossing) | execute | Streams one planned crossing leg of a cable onto a routing station, lets the rod settle, then LOOKS. | any |
| [`seating-a-piece-by-wiggle-search`](#seating-a-piece-by-wiggle-search) | execute | Seats a held jigsaw piece in its slot among already-placed neighbours when the arm is less accurate than the slot's… | any |
| [`setting-gripper-with-feedback`](#setting-gripper-with-feedback) | execute | Opens, closes or releases the gripper through the configured YAM or Isaac workcell adapter and checks measured holding… | yam_i2rt |
| [`singulating-by-pushing`](#singulating-by-pushing) | execute | Pushes a target clear of the objects crowding it until a top-down grasp's open fingers fit around it. | any |
| [`sliding-along-a-cable`](#sliding-along-a-cable) | execute | Slides the hand along a cable it holds toward one of the cable's ends. | any |
| [`sliding-to-edge-then-picking`](#sliding-to-edge-then-picking) | execute | Picks a flat object too thin to pinch on its support. | any, *tuned on* yam_i2rt |
| [`stacking-objects`](#stacking-objects) | execute | Stacks the object the hand holds centred on top of a target object. | any |
| [`standing-a-lying-domino-by-reorient-and-regrasp`](#standing-a-lying-domino-by-reorient-and-regrasp) | execute | Stands a flat-lying domino upright so a top-down pick can take it. | any, 2 arms |
| [`straightening-a-cable-toward-a-target-curve`](#straightening-a-cable-toward-a-target-curve) | execute | Pulls a cable lying flat on a table toward a target curve, one correction at a time. | any |
| [`threading-a-cable-tip-through-a-ring`](#threading-a-cable-tip-through-a-ring) | execute | Threads the free tip of a held cable through a standing ring. | any |
| [`tipping-over-a-surface-edge`](#tipping-over-a-surface-edge) | execute | Turn a body over on the surface it rests on instead of carrying it. | any, *tuned on* yam_i2rt, 2 arms, **sim only** |
| [`transporting-objects`](#transporting-objects) | execute | Move the currently-held object to a destination and release. | any, *tuned on* panda |
| [`triggering-a-domino-chain-by-push`](#triggering-a-domino-chain-by-push) | execute | Topples the first domino of a standing line so the chain runs. | any |
| [`turning-by-wrist-ratchet`](#turning-by-wrist-ratchet) | execute | Tightens or loosens a screw lid or cap by wrist ratchet. | any |
| [`wiping-a-region`](#wiping-a-region) | execute | Wipes a flat region of a surface with back-and-forth strokes. | any |
| [`auditing-a-domino-line-for-fallen-or-tilted-tiles`](#auditing-a-domino-line-for-fallen-or-tilted-tiles) | verify | Checks a standing domino line or curve from one overhead depth frame, with no reference image, against its planned slots. | any |
| [`checking-dual-arm-clearance`](#checking-dual-arm-clearance) | verify | Predicts, before a move is sent, the minimum clearance between the two arms along the planned path. | any, 2 arms |
| [`checking-pick-by-scene-diff`](#checking-pick-by-scene-diff) | verify | Checks whether a pick (or a place, a push, any intended change) happened by showing a VLM one panel. | any |
| [`checking-spatial-relations`](#checking-spatial-relations) | verify | Decides from one RGB-D view whether a spatial relation holds between two seen objects. | any |
| [`classifying-failure-from-proprioception`](#classifying-failure-from-proprioception) | verify | Classifies the last moments of a manipulation step from proprioception alone. | any |
| [`detecting-disturbed-pieces-by-snapshot-diff`](#detecting-disturbed-pieces-by-snapshot-diff) | verify | Checks whether anything already placed was knocked since the last verified step. | any |
| [`flagging-odd-one-out-parts-by-mutual-scoring`](#flagging-odd-one-out-parts-by-mutual-scoring) | verify | Finds the part that differs from its siblings in one frame. | any |
| [`previewing-target-overlay`](#previewing-target-overlay) | verify | Checks a target gripper pose before the arm moves. | any |
| [`registering-assembled-board-to-reference`](#registering-assembled-board-to-reference) | verify | Aligns a top-down view of the part-built jigsaw (overhead frame, rectified view or wrist mosaic) to the box picture… | any |
| [`simulating-a-domino-layout-before-building`](#simulating-a-domino-layout-before-building) | verify | Checks a planned domino line or curve in a rigid-body simulation before the arm builds it. | any |
| [`tracking-gripper-opening-stall`](#tracking-gripper-opening-stall) | verify | Drives the jaws to a target gap (or fully closed) and watches the measured gap over a sliding window, reporting… | any |
| [`verifying-a-brick-connection-by-pull-up`](#verifying-a-brick-connection-by-pull-up) | verify | Checks that a brick just pressed onto studs is clutched by lifting it a few millimetres, still gripped, in 0.5 mm… | any |
| [`verifying-a-cable-route`](#verifying-a-cable-route) | verify | Judge a finished cable route from a camera rather than from an evaluator. | any |
| [`verifying-a-domino-cascade-from-video`](#verifying-a-domino-cascade-from-video) | verify | Decides from a short RGB(-D) clip of the overhead or a wrist camera which tiles of a planned domino line fell and… | any |
| [`verifying-articulation-displacement`](#verifying-articulation-displacement) | verify | Checks from the camera whether a drawer or a door really moved as asked. | any |
| [`verifying-brick-seated-by-height`](#verifying-brick-seated-by-height) | verify | Checks from a depth camera whether a placed LEGO or DUPLO brick is seated on its studs. | any |
| [`verifying-build-against-model`](#verifying-build-against-model) | verify | Compares a brick build with its target model up to the current step. | any |
| [`verifying-grasps`](#verifying-grasps) | verify | Three independent checks on one grasp attempt. | any |
| [`verifying-insertion-by-probe-push`](#verifying-insertion-by-probe-push) | verify | Checks whether a held part is seated after an insertion by pushing it a few millimetres further along the insertion… | any |
| [`verifying-piece-flushness-from-wrist-depth`](#verifying-piece-flushness-from-wrist-depth) | verify | Checks from one close wrist depth frame (D405 at 8-16 cm) whether a just-placed jigsaw piece lies flush. | any |
| [`verifying-placement`](#verifying-placement) | verify | Confirms from one RGB-D view that a released object went through an aperture, accepting either that it vanished into… | any |
| [`backtracking-a-wrong-placement`](#backtracking-a-wrong-placement) | recover | Decides what to undo after a jigsaw piece fails its check at a cell (it does not seat, a seam with a placed neighbour… | any |
| [`centering-a-part-with-the-gripper`](#centering-a-part-with-the-gripper) | recover | Centres a loose part between the jaws without perceiving it. | any |
| [`checkpointing-assembly-state-for-resume`](#checkpointing-assembly-state-for-resume) | recover | Saves and restores the state of a long assembly run (a 550-piece jigsaw, a LEGO build, a domino layout) so a run… | any |
| [`correcting-insertion-offset-from-port-opening`](#correcting-insertion-offset-from-port-opening) | recover | Re-centres a held part over a port before or after a failed insertion using cameras only. | any |
| [`returning-to-a-recorded-pose`](#returning-to-a-recorded-pose) | recover | Opens the gripper and drives the arm back to a joint configuration recorded earlier in the episode, then checks from a… | any |
| [`classifying-failure-type-vlm`](#classifying-failure-type-vlm) | judge | Classifies why a manipulation step failed from camera frames taken before, during (optional, a short video as frames)… | any |
| [`detecting-subtask-success`](#detecting-subtask-success) | judge | Checks whether a subtask is done from the latest frame of every camera at one moment. | any |
| [`judging-assembly-steps-by-fused-checks`](#judging-assembly-steps-by-fused-checks) | judge | Turns the routes and numbers of the hub's per-step checks into one verdict for an assembly step. | any |
| [`judging-attempt-success-from-frames-robojev`](#judging-attempt-success-from-frames-robojev) | judge | Decides whether a step succeeded from cameras alone with one calibrated number. | any |
| [`judging-step-with-before-now-vlm`](#judging-step-with-before-now-vlm) | judge | Judges the current subgoal of a task from camera frames taken before the step and now. | any |
| [`scoring-step-progress-from-one-frame`](#scoring-step-progress-from-one-frame) | judge | Judges whether a step is done from the current camera frame alone. | any |

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

### mm-grounding-dino

perceive · robots: any, *tuned on* yam-station-orbbec-overhead · tasks: pick-place, sorting · MIT · v0.1.0 · [`SKILL.md`](tools/mm-grounding-dino/SKILL.md) · [upstream](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/mm_grounding_dino)

MM-Grounding-DINO open-vocabulary detection (OpenMMLab's open retraining of Grounding DINO, Apache-2.0 weights) with grounding-dino's call and answer -- text phrases to labelled boxes. On the station's overhead frames the large checkpoint was the most precise detector the hub has measured, finding every 30 mm cube at ~1 m and the open boxes that owlv2 misses, at 2-9 false positives over 7 frames instead of 18-19. It finds fewer plugs than owlv2. Use when a workflow needs table parts and containers named by text with few false positives, as a drop-in for grounding-dino.detect or as a sahi backend. Needs transformers 4.x.

- **Weights**: `hf://openmmlab-community/mm_grounding_dino_large_all@0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `mm-grounding-dino.detect`: Open-vocabulary detection with MM-Grounding-DINO (labelled boxes + scores, grounding-dino's shape).

### molmo

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/molmo/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/molmo)

Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). Use when a workflow needs a single pixel coordinate for a named object (point_prompt) or Molmo-grade visual question answering and a vLLM server is available; for an API-only zero-GPU alternative use the gemini-er bundle.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `molmo.point_prompt`: Point at a named object in an image; returns pixel coordinates.
  - `molmo.query`: Free-form visual question answering via Molmo.
  - `molmo.query_yes_no`: Yes/no visual question answering via Molmo; coerces the reply to a bool.

### move-classifier

perceive · robots: any · MIT · v0.1.0 · [`SKILL.md`](tools/move-classifier/SKILL.md) · [upstream](https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2/blob/3bfe3fe87b64/reference_implementation/vlm_cnn_train/train_cnn.py)

Trains and runs the 27-class "which way to nudge" classifier of vision-guided rack insertion (ResNet18, ImageNet weights, MIT) - from one cropped wrist image, the sign of the x, y (world) and yaw correction that brings a held part over its target. Training data is self-labelled by skills/correcting-in-plane-offset-by-discrete-moves (collect_offsets). Use when a per-target learned alignment cue is needed and a GPU is available for training (inference runs on CPU too).

- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `move-classifier.train`: Train the classifier on a folder of self-labelled crops (index.jsonl of measured offsets).
  - `move-classifier.predict`: Which way to nudge (x, y, yaw signs, world frame) from one cropped wrist image.

### omdet-turbo

perceive · robots: any, *tuned on* yam-station-orbbec-overhead · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/omdet-turbo/SKILL.md) · [upstream](https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/omdet_turbo)

OmDet-Turbo fast open-vocabulary detection (omlab/omdet-turbo-swin-tiny-hf, Apache-2.0) -- text prompts to labelled boxes in grounding-dino's shape at ~0.02 s a 640 x 480 frame and 0.6 GB of GPU memory, 10x faster than owlv2 and 17-23x than grounding-dino on a shared GPU. Use for video-rate re-detection of table-scale objects (an open box, a tray, a bin, a board, a held part) in closed loops or as a cheap sahi backend; on 15-30 px parts at ~1 m (cubes, plugs, bricks) it finds fewer than owlv2, so use owlv2 there.

- **Weights**: `hf://omlab/omdet-turbo-swin-tiny-hf@7fe93cecfb770c4d76cf71163956221249cab566`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `omdet-turbo.detect`: Fast open-vocabulary detection from text prompts (labelled boxes + scores, grounding-dino's shape).

### owlv2

perceive · robots: any · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/owlv2/SKILL.md) · [upstream](https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/owlv2)

OWLv2 open-vocabulary detection (google/owlv2-*, Apache-2.0) in two modes -- text prompts to labelled boxes in grounding-dino's shape, and image-guided one-shot search ("every brick / tile / piece like this one") from a box around one example, in the same frame or another, with no name needed. Use when a workflow must find all copies of a part it can point at but not name well (a 2x2 brick vs a 2x4, a domino, a jigsaw piece), or wants a fast detector (0.06 s a frame) for 15-25 px parts at ~1 m; also a backend of tools/sahi.

- **Weights**: `hf://google/owlv2-base-patch16-ensemble@cfd3195ba4ea9592eec887ded089f4c08eff231d`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `owlv2.detect`: Open-vocabulary detection from text prompts (labelled boxes + scores, grounding-dino's shape).
  - `owlv2.embed_exemplar`: The OWLv2 query embedding of one object (a box in a frame, or a crop) as a string.
  - `owlv2.detect_like`: Every object in an image that looks like an exemplar (box in this or another frame, crop, or string).

### pose6d

perceive · robots: any · tasks: pick-place, insertion, reorientation, sorting · MIT · v0.1.0 · [`SKILL.md`](tools/pose6d/SKILL.md) · [upstream](https://github.com/isl-org/Open3D/tree/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/examples/python/pipelines)

6D pose of a known rigid part from depth — registers a box, cube or cylinder primitive (LEGO brick body, domino, cube, peg) or a mesh file to the part's point cloud or to its mask in a depth frame, and returns the pose with fitness, RMSE, surface coverage, every symmetric equivalent and the best non-equivalent alternative. Boxes and cylinders take an analytic, visibility-aware point-to-plane ICP over a hypothesis search (numpy/scipy, tens to a few hundred ms); meshes take Open3D FPFH + RANSAC global registration then point-to-plane ICP. Use when a workflow needs the full pose (not just an OBB) of a known-size part from the wrist D405 or the overhead camera.

- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `pose6d.register_mesh`: 6D pose of a known part (box / cube / cylinder primitive or mesh file) from its point cloud, with fitness, RMSE, coverage and its symmetric equivalents.
  - `pose6d.register_mask`: 6D pose of a known part from its segmentation mask and a depth frame (camera centre used as the viewpoint); world pose when a camera pose is given.

### rfdetr

perceive · robots: any · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/rfdetr/SKILL.md) · [upstream](https://github.com/roboflow/rf-detr/tree/1.12.0)

RF-DETR (Roboflow, Apache-2.0) train-and-run tools for a small station-specific detector -- fine-tune Nano / Small / Medium / Large from its COCO checkpoint on a COCO-format folder (e.g. one written by training-a-part-detector-from-auto-labels), then detect at 7 ms a 640 x 480 frame with labelled boxes in grounding-dino's shape. Use when an open-vocabulary detector is too slow or too noisy for a fixed set of parts (bricks, dominoes, pieces) seen by the station's own cameras.

- **Weights**: `https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `rfdetr.train`: Fine-tune RF-DETR (Apache-2.0 sizes) on a COCO-format dataset folder; returns the model directory.
  - `rfdetr.detect`: Run a trained RF-DETR model on an image (labelled boxes + scores, grounding-dino's shape).

### sahi

perceive · robots: any · tasks: pick-place, sorting · MIT · v0.1.0 · [`SKILL.md`](tools/sahi/SKILL.md) · [upstream](https://github.com/obss/sahi/blob/10061afd5fa076e00f8eda55e2ee290f9d72f3c1/sahi/predict.py)

Slicing-aided detection (SAHI) of small parts -- cuts the camera frame into overlapping slices, runs one of the hub's detectors (grounding-dino, sam3, locate-anything) on every slice and on the whole frame, and merges the boxes with SAHI's postprocess, returning boxes, labels, scores and (sam3) masks in frame pixels. Use when the parts are 10-30 px in the overhead camera (2-3 cm bricks, dominoes, jigsaw pieces, a plug at ~1 m) and a single detector call on the whole frame misses them.

- **Functions**:
  - `sahi.detect_sliced`: Detect small parts by running a detector on overlapping slices and merging the boxes (SAHI).
  - `sahi.slice_windows`: The overlapping slice windows SAHI cuts an image (or a region of it) into.
  - `sahi.merge`: Merge overlapping detections from several slices with SAHI's postprocess.

### sam2

perceive · robots: any, *tuned on* yam-station-orbbec-overhead, yam-left-wrist-d405 · tasks: pick-place, sorting, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/sam2/SKILL.md) · [upstream](https://github.com/huggingface/transformers/tree/b39d97c836e8d9f0ee6035a87c7a3ce5eaea43a8/src/transformers/models/sam2_video)

SAM 2.1 promptable video mask tracking (facebook/sam2.1-hiera-*, Apache-2.0; EdgeTAM as the fast option) -- seeded with a box, points or a mask on one frame, follows one part through a live camera stream or a recorded clip, returning its mask, box and an occlusion-head presence call per frame, and picks it up again after an arm or hand has covered or moved it. Live sessions hold constant GPU memory. Use when a closed loop or a verify stage must keep the identity of one part (a cube, a box, a plug) while the arm or a hand passes over or moves it; seed it from a detector (owlv2, grounding-dino) or a sam3 mask. Not open-vocabulary.

- **Weights**: `hf://facebook/sam2.1-hiera-base-plus@b7320756a13354e7530a63935656d35b2f91a290`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `sam2.tracker_init`: Open a mask-tracking session on a live frame from a box, points or a mask.
  - `sam2.tracker_update`: Advance a session by one frame; mask, box, confidence, presence.
  - `sam2.tracker_close`: Free a session (idempotent).
  - `sam2.track`: Track one object through a recorded clip from a prompt on any frame.

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

plan · robots: any, *tuned on* panda · tasks: motion-planning · MIT for this bundle's code (as open-robot-skills declared it); the NVIDIA cuRobo it installs and calls (NVlabs/curobo@4ea7736, nvidia-curobo) is Apache-2.0 · v0.2.0 · [`SKILL.md`](tools/curobo/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/curobo)

NVIDIA cuRobo motion planning — collision-free trajectories to grasp goalsets, transport with an attached object, constrained linear moves, single-pose planning, geometric IK, batch grasp feasibility, and joint-trajectory collision validation. Use when a workflow needs GPU-accelerated, collision-aware arm motion plans.

- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Note**: Plans for curobo's built-in franka.yml. It knows yam.yml is 6-DoF but does not ship that config; plan_with_grasped_object loads one registered in _PROJECT_ROBOT_CONFIGS, the other v0.8 planners only resolve curobo's built-in configs.
- **Functions**:
  - `curobo.plan_to_grasp_poses`: Collision-free trajectory to one of several grasp poses (goalset).
  - `curobo.plan_with_grasped_object`: Transport plan with the held object attached to the gripper (failure_reason on failure).
  - `curobo.plan_linear`: Straight Cartesian trajectory between two EE poses.
  - `curobo.plan_directed_linear`: Constrained linear motion with axis/orientation holds.
  - `curobo.plan_grasp_motion`: Approach + grasp + lift sequence (three trajectories).
  - `curobo.plan_to_pose`: Collision-aware plan to a single TCP pose.
  - `curobo.solve_ik`: Geometric IK for a single TCP pose (no world collision).
  - `curobo.batch_grasp_feasibility`: Per-pose grasp/approach IK + corridor feasibility for a grasp batch.
  - `curobo.validate_joint_trajectory_robot`: Collision-validate joint waypoints (robot vs world + self).
  - `curobo.validate_joint_trajectory_grasped`: Same, with a grasped object attached at waypoint 0.
  - `curobo.cloud_to_attachment`: MORPHIT collision spheres for a held object's cloud, in the TCP frame.

### grconvnet

propose · robots: any · tasks: pick-place · BSD-3-Clause · v0.1.0 · [`SKILL.md`](tools/grconvnet/SKILL.md) · [upstream](https://github.com/skumra/robotic-grasping/tree/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d/inference)

Proposes ranked top-down parallel-jaw grasps from one RGB-D frame with GR-ConvNet (skumra/robotic-grasping), on CPU (75 ms a call on 8 desktop cores) or a GPU -- pixel centre, jaw angle, opening in px and metres and a quality, and with a camera pose the world grasp point, the heading the jaws close along (for robot.grasp_frame) and an oriented box measured from the depth along the jaw line (for OBB-driven grasp skills). Use when a pick needs a learned propose rung beside proposing-short-axis-grasps because the object's box does not say where to grasp it -- a container wider than the hand (grasp its wall), a curved cable, an irregular part -- with a mask from a perception skill to focus it on one target.

- **Weights**: `https://raw.githubusercontent.com/skumra/robotic-grasping/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d/trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94`
- **Runs**: in its own environment over `stdio-msgpack`
- **Functions**:
  - `grconvnet.propose`: Ranked top-down parallel-jaw grasps from an RGB-D frame; pixel centre, angle, width (px, m), quality, and world grasp point / closing heading / box with a camera pose.

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

### workcell

execute · robots: yam_i2rt · tasks: motion, grasp, point-tracking, pushing, drawer · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/workcell/SKILL.md)

Typed JSON boundary for RoboSkillHub workcell capabilities and station descriptions. Use when an agent or graph must call an adapted YAM or Isaac capability through one interface and retain its measured result.

- **Note**: A station-bound adapter boundary; source availability and interface tests do not imply sim/real verification.
- **Functions**:
  - `workcell.describe`: Describe the configured station and capability interface without issuing motion.
  - `workcell.invoke`: Invoke one named capability with a JSON object request and return its JSON result.

### musc

verify · robots: any · tasks: domino, lego, jigsaw · MIT · v0.1.0 · [`SKILL.md`](tools/musc/SKILL.md) · [upstream](https://github.com/HUST-SLOW/MuSc-V2/tree/b5e911dbc089/models/modules)

MuSc zero-shot mutual anomaly scoring (MuSc / MuSc-V2, MIT) on DINOv2 (facebook/dinov2-base, Apache-2.0) - given N &gt;= 3 images of things that should look the same (crops of parts in one orientation and scale), each image's patches are scored by their distance to the nearest patch of each other image, so the one that differs scores high, with no reference image and no training. Per-image scores, the re-scored ones (RsCon) and patch maps. Use when a workflow must find the odd part among many copies (a face-up tile among face-down ones, a fallen domino, a wrong brick) in one frame; skills/flagging-odd-one-out-parts-by-mutual-scoring cuts and normalises the crops.

- **Weights**: `hf://facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `musc.score`: Mutual anomaly scores of N &gt;= 3 nominally identical images (per-image scores, RsCon re-scores, patch maps).

### densereward

judge · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/densereward/SKILL.md) · [upstream](https://huggingface.co/densereward/densereward-1frame/blob/9796cb2a7be55b339670d8f95fb07841a61487a8/README.md)

DenseReward reward models (Qwen3-VL-4B fine-tunes, Apache-2.0) run locally on a GPU -- task progress 0..1 from ONE RGB frame and the task text (0 not started or failed, 1 complete), or at the last of three chronological frames with a one-word reason for a drop (miss / collision / fall / smooth / failure). No start frame, no video. Use when a judge stage wants a cheap progress score right after a step, as a second opinion beside a geometric check.

- **Weights**: `hf://densereward/densereward-1frame@9796cb2a7be55b339670d8f95fb07841a61487a8`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `densereward.progress`: Task progress 0..1 from one RGB frame and the task text (DenseReward-1frame).
  - `densereward.progress_with_history`: Progress at the last of three chronological frames plus a one-word reason (DenseReward-3frame-thinking).

### predicates

judge · robots: any · tasks: pick-place, insertion, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/predicates/SKILL.md)

Geometric yes/no judgments over poses and boxes the graph already holds -- is a point inside an oriented box, does one box rest on another, are two points within a distance, is a body upright -- each returning a boolean, a verdict string a router can branch on, and the margin it was decided by. Use when a graph must choose a branch from geometry it already has (a perceived OBB, an end-effector pose), on CPU, with no model and no simulator state.

- **Functions**:
  - `predicates.point_in_obb`: Whether a world point lies inside an oriented box, and by how much.
  - `predicates.resting_on`: Whether one box sits on top of another, from footprint overlap and the vertical gap.
  - `predicates.within_distance`: Whether two points are within a threshold, in 3D or in the table plane.
  - `predicates.upright`: Whether a body's local axis points up, within a tilt bound.

### robojev

judge · robots: any · tasks: pick-place, assembly, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/robojev/SKILL.md) · [upstream](https://github.com/josefchen/robojev/blob/79a5e2f/robojev/robojev.py)

RoboJev-27B success judge (josefchen/RoboJev-27B, Apache-2.0) - given the task sentence, frames of one attempt from a main camera (start to end) and the final frame of other cameras, one forward pass reads the false / true answer and the released calibration returns P(task done). Use when a workflow must decide after a step whether it succeeded from cameras alone, or grade recorded attempts; needs a GPU with ~56 GB.

- **Weights**: `hf://josefchen/RoboJev-27B@e089a9b69a365744b80765057d86583647af95a9`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `robojev.judge`: Calibrated P(task completed) for one attempt from its main-camera frames (start to end) and final frames of other cameras.

### rynnvalue

judge · robots: any · tasks: pick-place, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](tools/rynnvalue/SKILL.md) · [upstream](https://github.com/alibaba-damo-academy/RynnValue/blob/cd29294f1ac56bc817074bd665ba92b630936ad7/rynn_infer/inference.py)

RynnValue value model (RynnBrain / Qwen3-VL, 8B or 4B, Apache-2.0) run locally on a GPU -- reads a task instruction and up to 8 frames of a clip (oldest first) and returns, for every frame, the predicted remaining time in seconds until the task is complete, plus a generated analysis with Match (clip fits the instruction) and Success verdicts. One forward pass, ~0.2 s for 8 frames on the 8B. Use when a judge stage has the frames of a step (start to now) and wants a ranking-quality "how close to done" signal -- an alternative backend to robobrain.progress / densereward.progress for progress and stall scoring; threshold it on your own frames before routing on it.

- **Weights**: `hf://Alibaba-DAMO-Academy/RynnValue-8B-Quantile@0fb3ac12c73844ea1abaf4f767c895aae248a9ec`
- **Runs**: in its own environment over `stdio-msgpack`, needs a GPU
- **Functions**:
  - `rynnvalue.remaining_time`: Remaining seconds to completion for each frame of a clip (oldest first, at most 8 used) plus Match / Success from the model's analysis.

## Skill details

### calibrating-arm-to-arm-base-transform

perceive · robots: any · tasks: bimanual, pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/calibrating-arm-to-arm-base-transform/SKILL.md) · [upstream](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_handeye.cpp)

Measures where one arm's base sits in the other arm's base frame - each arm in turn visits a list of joint configurations, its wrist camera poses a ChArUco board (or ArUco marker) fixed in the cell, or one fixed camera poses a marker each gripper carries; each arm's views are solved as robot-world / hand-eye (AX = ZB, Shah's closed form plus least squares) and the two "board in my base" results compose into base0 -&gt; base1, with per-view residuals and a leave-one-view-out spread. Use when a two-arm task (handover, holding a part while the other arm works, one shared grid for both arms) needs the arms' relative placement measured instead of the nominal layout, or to check it after a base was moved.

- **In**: -
- **Out**: `yaw_deg` float, `tilt_deg` float, `jackknife_mm` float, `jackknife_deg` float, `reason` string
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.move_to_joints`
- **Scripts**: [`calibrate_arms`](skills/calibrating-arm-to-arm-base-transform/scripts/calibrate_arms.py)
- **Exits**:
  - `calibrated`: Both arms posed the target in at least min_views (6) views, each arm's views agree on it within max_rms_mm (6 mm) / max_rms_deg (1.5 deg) RMS, and leaving any one view out moves base0 -&gt; base1 by at most max_jackknife_mm (5 mm); T_base0_base1 is arm 1's base in arm 0's base frame.
  - `residual_too_high`: Solved, but an arm's views disagree (RMS over the limits) or one view decides the result (leave-one-out over 5 mm); T_base0_base1 and the numbers are filled in - add views from more varied angles, or check the kinematics / camera intrinsics.
  - `tag_not_seen`: An arm posed the target in fewer than min_views views (moves refused, no frame, target not found); nothing is solved and `reason` lists the misses.

### calibrating-baseplate-stud-grid

perceive · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.1 · [`SKILL.md`](skills/calibrating-baseplate-stud-grid/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/APEX-MR/blob/6b5a17f1133c3f739caf50fa0fff6c478ef73b7a/config/lego_tasks/plate_calibration.json)

Estimates a LEGO or DUPLO baseplate's pose (the plate frame computing-brick-pose-from-stud-grid takes, and APEX-MR's x, y, z, roll, pitch, yaw, width, height) in two steps - a coarse pose from ArUco / AprilTag markers at known places on or beside the plate (OpenCV aruco + solvePnP), a given pose, or a plate mask and its depth plane; then a sub-millimetre refinement from a wrist RGB-D close-up, where the stud tops are found in depth and as circles in the plate-rectified image and an 8 mm (16 mm DUPLO) lattice is fitted by RANSAC (yaw + offset, the coarse pose picking the period) - with the residual and the number of studs used. A second script reads a brick's own studs - footprint (A x B), top pose and stud pixels from a close-up and a mask. Use when bricks will be placed or picked by grid cell and the plate's pose must be known to well under a pitch, or a brick's size and pose must be read from its studs.

- **In**: -
- **Out**: `plate_pose` Se3Pose, `coarse_plate_pose` Se3Pose, `coarse_source` string, `lattice_status` string, `residual_mm` float, `max_residual_mm` float, `studs_used` int, `studs_detected` int, `snap_margin` float, `marker_reprojection_px` float, `yaw_ambiguity_deg` float, `system` string, `stud_pitch` float, `camera_name` string
- **Calls**: [`sam3`](#sam3); connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`calibrate_plate`](skills/calibrating-baseplate-stud-grid/scripts/calibrate_plate.py), [`read_studs`](skills/calibrating-baseplate-stud-grid/scripts/read_studs.py)
- **Exits**:
  - `calibrated`: A coarse pose was found, a wrist close-up was taken, at least min_studs (12) stud tops lie on the lattice with an RMS residual under max_residual (1/8 pitch - 1.0 mm LEGO, 2.0 mm DUPLO), the yaw correction is under 10 deg and the coarse pose is within snap_tolerance (0.35 pitch) of the nearest stud row along both axes; plate_pose is the refined pose.
  - `coarse_only`: A coarse pose was found but no close-up was used - refine off, no wrist camera with a pose (and depth), or no view pose reached (IK preview, refused or short moves); plate_pose is the coarse pose and lattice_status says which.
  - `no_lattice`: A close-up was taken but the lattice is not trusted - no plate plane near the coarse one (no_plane), too few studs (too_few_studs), residual over max_residual (high_residual), yaw correction over 10 deg (yaw_off) or the coarse pose about half a pitch off (ambiguous); plate_pose stays the coarse pose, the fit's numbers are filled in.
  - `no_markers`: No coarse pose - none of the listed markers seen, no coarse_plate_pose, no usable plate mask; plate_pose is empty.

### classifying-brick-resting-face-from-depth

perceive · robots: any · tasks: assembly, lego, reorientation · MIT · v0.1.0 · [`SKILL.md`](skills/classifying-brick-resting-face-from-depth/SKILL.md) · [upstream](https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation/blob/596c496867c11ba63f47127833e0f49b7480ffd3/catkin_ws/src/vision/scripts/lego-vision.py)

Tells from depth whether a loose LEGO or DUPLO brick of known size lies studs up, upside down (resting on its stud tops, hollow underside up) or on a side wall, and which of its axes is vertical, with its yaw, its stud direction and its body centre. The brick is the connected cluster of depth points over the table nearest a rough position. Every resting face of the brick is scored on its height over the table and its footprint (the height-versus-dimensions match of a UR5 LEGO pick-and-place pipeline). Up is told from down by the hollow underside (most of the inner footprint far under the rim) instead of the source's YOLO model. Moves the wrist camera over the brick first, or reads the current frame of any posed depth camera. Use before picking a loose brick, to grasp it directly when it is studs up or to route it to a flip / regrasp otherwise, and to pick the grasp axis and yaw.

- **In**: `brick_position` Vec3
- **Out**: `vertical_axis` str, `up_vector` Vec3, `stud_direction_known` bool, `yaw_deg` float, `centre` Vec3, `height` float, `footprint_long` float, `footprint_short` float, `table_height` float, `deep_share` float, `samples` int
- **Calls**: connector `robot.get_observation`, `robot.go_to_pose`, `robot.solve_ik`
- **Scripts**: [`classify_face`](skills/classifying-brick-resting-face-from-depth/scripts/classify_face.py)
- **Exits**:
  - `studs_up`: The brick's z axis is vertical (its height and footprint fit that face best by a margin) and at most 10 % of its inner footprint lies far under its top - a face with studs on it; up_vector (0, 0, 1).
  - `upside_down`: The z axis is vertical and at least 25 % of the inner footprint lies far under the rim - the hollow underside, the brick resting on its stud tops; up_vector (0, 0, -1).
  - `on_side`: A side wall is down - the brick's x or y axis is vertical (vertical_axis); up_vector is the horizontal stud direction when the silhouette resolves which side carries the studs (stud_direction_known), else its axis only.
  - `unknown`: No resting face of the given brick fits the reading (wrong size, two bricks touching, a part that is not that brick), two faces fit about equally, the up / down share is between 10 % and 25 % or the part is too thin to tell (a plate lying flat), or the depth could not be read (no posed camera, too few points around the brick); detail says which.
  - `no_brick`: Fewer than 40 depth points stand over the table near brick_position.
  - `unreachable`: view 'move' and every view of the brick was out of reach or refused.

### classifying-piece-sides-for-border-first

perceive · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/classifying-piece-sides-for-border-first/SKILL.md)

Types a jigsaw piece's four sides flat / tab / blank from a top-down mask and calls it a corner, edge or interior piece - the side lines are found as the direction (1 degree steps) whose projections pile the outline's points into the sharpest peak near each extreme (straight shoulders and flat sides), then each side's middle stretch is a tab where it reaches past its line by 12 % of the side, a blank where it falls behind it, flat otherwise. Returns the outward direction of each flat side (where the board's edge must lie), sides near the threshold as uncertain, and, for many pieces, the ids sorted into corners, edges and interiors for a border-first build. Use when sorting loose pieces for a frame-first strategy, or to orient a border piece before placing it, from wrist-camera masks of pieces lying flat; CPU only, about 8 ms a piece.

- **In**: -
- **Out**: `piece_class` string, `uncertain_sides` int
- **Scripts**: [`classify_sides`](skills/classifying-piece-sides-for-border-first/scripts/classify_sides.py)
- **Exits**:
  - `classified`: Every side was typed with no more than max_uncertain (0) of them within 0.35 x 12 % of the threshold; for a list of pieces, at least one mask was usable and the ids are sorted into corners / edges / interiors / uncertain.
  - `uncertain`: One piece, but a side's deviation lies near the threshold or the piece has three or four flat sides; look closer, or decide by shape matching.
  - `mismatch`: expected_sides was given and no turn of the piece makes at least min_agree (4) of its sides read as expected, or two turns agree equally (a symmetric piece); `yaw_deg` and `sides_agree` say the best; look again (re-scan, closer) rather than place it 90 or 180 degrees off.
  - `no_piece`: No usable mask (empty, under 400 px, or no square body found).

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

### estimating-cut-grid-from-pieces

perceive · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/estimating-cut-grid-from-pieces/SKILL.md)

Estimates a jigsaw's real cut grid (cols x rows) and cell size from the piece count printed on the box, the board's size or the box picture's aspect, and the corner-to-corner side lengths of a few measured pieces (top- down masks with depth scale, corners found as in matching-a-piece-to-placed-neighbours). Every grid with c x r within 10 % of the count is scored against the pieces' sides (with the side pairs mixed both ways unless each piece's turn is known), under one shared scale factor, a prior on how manufacturers round the count and, if given, a full count of edge pieces (2(c + r) - 4). Returns the grid, its posterior confidence and the runner-up. Use before localizing-a-piece-in-a-reference-image or matching-a-piece-to-placed-neighbours, which both take the grid as input, when the real grid has not been counted; CPU only, under 0.1 s plus a few ms per piece.

- **In**: `piece_count` int
- **Out**: `grid_cols` int, `grid_rows` int, `cell_width` float, `cell_height` float, `confidence` float, `runner_up_cols` int, `runner_up_rows` int
- **Calls**: [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`estimate_grid`](skills/estimating-cut-grid-from-pieces/scripts/estimate_grid.py)
- **Exits**:
  - `estimated`: At least min_pieces (6) pieces were measured, the best grid's posterior probability is at least min_confidence (0.9) and a given edge-piece count agrees with it (within 3 sigma). grid_cols, grid_rows and the cell size are the answer.
  - `ambiguous`: Pieces were measured but the best grid is under min_confidence, or a given edge-piece count disagrees with it; grid_cols / grid_rows are the best guess and `candidates` ranks the alternatives. Give each piece's turn (angle_deg), measure more pieces, or count the edge pieces.
  - `insufficient_pieces`: Fewer than min_pieces usable pieces (the prior-only best guess is still returned), no board size / aspect / reference, a piece count under 4, no camera for measure_view, or the corner skill is not installed; `detail` says which.

### estimating-relative-rotation-and-symmetry

perceive · robots: any, *tuned on* yam-left-wrist-d405-renders · tasks: pick-place, reorientation, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/estimating-relative-rotation-and-symmetry/SKILL.md) · [upstream](own code (OpenCV, numpy))

How far a part is turned in the image plane between a reference view and the current view (both from the wrist camera at a similar height), and modulo what -- from the two masks and the pixels inside them. The outline gives the turn (principal axis for elongated parts, minimum-area rectangle for square ones), known modulo the part's symmetry (180 deg for a 2 x 4 brick, 90 for a 2 x 2). For a part whose look breaks its outline's symmetry (a face-up domino's pips, a printed piece), the rotated reference pixels pick the full turn, or the skill says the turns look alike. Routes rotation_found, symmetric_ambiguous or low_confidence. Use before a wrist turn to match a reference, to choose the shorter of equivalent wrist turns, or to check a placed part's yaw; camera-only.

- **In**: `reference_frame` CameraFrame, `reference_mask` Mask, `frame` CameraFrame, `mask` Mask
- **Out**: `rotation_deg` float, `period_deg` float, `symmetry_order` int
- **Scripts**: [`relative_rotation`](skills/estimating-relative-rotation-and-symmetry/scripts/relative_rotation.py)
- **Exits**:
  - `rotation_found`: The turn from the reference to the current view, in (-period/2, period/2], known modulo period_deg (360 when the part's look resolved every turn; 180 / 90 for a part that looks the same after a half / quarter turn).
  - `symmetric_ambiguous`: The caller said the part is less symmetric than its outline, but its outline turns look alike in these pixels (within 0.05 correlation); the turn is reported modulo the outline's period only.
  - `low_confidence`: A mask has under 60 px, or the best appearance match is under 0.3 correlation (another part, occluded, relit); rotation_deg is the outline's turn, unconfirmed.

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

### observing-object-poses

perceive · robots: yam_i2rt · tasks: perception · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/observing-object-poses/SKILL.md)

Reads independently observed object positions, and orientations where the backend provides them, from the configured workcell, labelling simulator truth as privileged. Use when a graph must re-observe pieces before choosing, grasping or verifying.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/observing-object-poses/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### reading-domino-pips-and-orientation

perceive · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/reading-domino-pips-and-orientation/SKILL.md) · [upstream](https://github.com/SponTanious/Domino-Arm/blob/master/Functions/Domino_Detection.m)

Reads each domino's two pip values, which end is which, its long axis and whether it lies flat, stands or has fallen, from above. Per tile mask (sam3.segment_text, or upstream masks) it finds the centre divider (an elongated dark MSER region, as Domino-Arm did), scores the pips, splits them into the halves and decodes each against the seven double-six layouts, and lifts the tile with the depth (geometry.mask_to_world_points) to its world centre, ends, long axis and resting state (flat / upright / fallen, from its height over the table). Returns (a, b) with a at end a, end and half-centre pixels and world points, pip scores and layout margins. Pips are ~2 px from the station's overhead camera, so a tile under 30 px wide routes needs_closeup with its world centre - read it from a wrist D405 close-up (refining-target-from-wrist-closeup / framing-a-target-by-mask-edge-steering first). Use when a domino task needs a tile's value, ends, pose or whether it still stands (game play, sorting, a toppled chain).

- **In**: -
- **Out**: `state` str, `center_world` Vec3, `axis_world` Vec3, `camera_name` str
- **Calls**: [`geometry`](#geometry), [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`read_dominoes`](skills/reading-domino-pips-and-orientation/scripts/read_dominoes.py)
- **Exits**:
  - `read`: Every judged tile (the one nearest target_point if given, else all) lies flat face up, its mask is one 2:1 tile, its divider was found and both halves decoded with a layout margin of at least 0.3 (glare-covered positions left out) and no strong pip left unexplained; value / value_ab / ends carry the (first) tile.
  - `ambiguous`: Tiles were segmented but at least one judged tile was not read, its status says why - `not_flat` (standing or fallen; state, axis and ends are still given), `face_down` (no divider and no pips), or `ambiguous` (a small layout margin, glare over a deciding position, an unexplained pip, no divider, or a mask that is not one tile).
  - `needs_closeup`: A judged flat tile is narrower than min_tile_width_px (30 px) in this frame and none failed otherwise; `closeup_targets` lists their world centres for refining-target-from-wrist-closeup, then run this skill again on the wrist frame.
  - `not_seen`: No camera frame, the segmenter failed or returned no mask scoring min_score and 40 px.

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

### scanning-the-board-into-a-wrist-mosaic

perceive · robots: any · tasks: jigsaw, pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/scanning-the-board-into-a-wrist-mosaic/SKILL.md) · [upstream](https://github.com/OpenStitching/stitching)

Scans a table region (the jigsaw board, a staging area) with the wrist camera from straight above and blends the frames into one top-down image at a chosen resolution - picks the camera height for the asked pixels per metre (10-30 cm), tiles the region with 25 % overlap in a serpentine, previews each pose with IK (trying the other wrist rolls), and maps every frame onto the region's plane by the homography its own camera pose and intrinsics induce, feathered at the frame borders; checks the overlaps by phase correlation and also returns a height map from the depth. Gives the board corners in the mosaic for registering-assembled-board-to-reference. Use when the overhead camera sees too few pixels per piece (about 11 at 0.9 m) and a board-wide check or a staging inventory needs 50-100 px per piece; moves the arm, no learned model.

- **In**: `region_center` Vec3
- **Out**: `metres_per_pixel` float, `covered_fraction` float, `seam_shift_px` float, `tiles_seen` int
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.solve_ik`
- **Scripts**: [`scan_mosaic`](skills/scanning-the-board-into-a-wrist-mosaic/scripts/scan_mosaic.py)
- **Exits**:
  - `mosaic_built`: Every planned frame was reached and the frames cover at least 98 % of the region; `mosaic` (RGB), `height_map` (m over the surface), `metres_per_pixel`, `origin` and `board_corners_px` describe it.
  - `partial`: Some frame positions were not reached (IK, refusal) or under 98 % is covered; the mosaic is returned with the gaps black and the height map NaN there.
  - `failed`: No wrist camera with a pose in the observation, more than 64 frames needed, or no frame position reached.

### segmenting-and-picking-pieces-classical

perceive · robots: any · tasks: jigsaw · MIT · v0.1.0 · [`SKILL.md`](skills/segmenting-and-picking-pieces-classical/SKILL.md) · [upstream](https://github.com/JPStrydom/Jigsaw-Puzzle-Building-Robot/blob/be867bda743740fac9d843f606b9211e663777f7/code/software/Functions_1.py)

Finds the loose jigsaw pieces lying on a staging mat in one top-down RGB(-D) frame without a learned model - removes the mat by its colour (estimated from the frame, shadow-tolerant, or a fixed HSV band) and by height when the depth is fine, sorts the blobs by area and shape into pieces, touching or stacked pieces and foreign objects, splits touching pieces with a distance-transform watershed that follows the dark cut line, and gives every piece its mask, a pick point at the centre of its largest inscribed circle lifted to the world, its side angle, area, touching neighbours, a two-finger grip check and a face-up / face-down cue. Routes segmented, touching_pieces or nothing_found. Use before picking a jigsaw piece from a staging area (and before localizing it in the box picture), on CPU, in a fraction of a second; sam3 is the learned alternative when a GPU is available.

- **In**: `observation` Observation
- **Out**: `n_pieces` int, `n_touching` int, `n_pickable` int, `piece_mask` Mask, `pick_point` Vec3, `angle_deg` float
- **Calls**: [`geometry`](#geometry)
- **Scripts**: [`segment_pieces`](skills/segmenting-and-picking-pieces-classical/scripts/segment_pieces.py)
- **Exits**:
  - `segmented`: At least one piece was found and none touches anything; `pieces` lists them, piece_mask / pick_point are the best pickable one (clearest of its surroundings, then the widest inscribed circle).
  - `touching_pieces`: Pieces were found but some touch or lie on one another (split by the watershed or by height; their records say touching, neighbours, layer), or a blob did not split into piece-sized parts (`clusters`, with the source's drag point and direction); piece_mask / pick_point still name the best pickable piece when n_pickable &gt; 0. Singulate the rest, e.g. singulating-by-pushing toward `free_spot`.
  - `nothing_found`: No piece-sized blob on the mat (empty, or only foreign objects and specks).

### tracking-build-state

perceive · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/tracking-build-state/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/Lego.cpp)

Keeps a world model of the LEGO or DUPLO bricks placed so far and their stud connections - ICL Robotic_Lego_Manipulation's rules (two bricks connect when their footprints overlap and the upper one's bottom sits on the lower one's top within ICL's 0.1-1.1 brick-height window; moving a brick moves everything connected above it) - extended to any yaw, mixed plates and bricks, a grounded / floating check, refused interpenetrating placements, and a comparison with perceived bricks that reports missing, displaced and extra bricks. The state is a JSON string passed in and out. Use after every place, pick-off or move in a build to know which bricks are connected, which can be removed and which hang unsupported, and after a fresh observation to catch silent failures (a brick that came away with the gripper, one knocked off its cell).

- **In**: `action` string
- **Out**: `state_json` string, `n_bricks` int, `n_connections` int, `n_grounded` int, `floating_json` string, `changed_json` string, `discrepancies_json` string, `removable_json` string
- **Scripts**: [`track_build`](skills/tracking-build-state/scripts/track_build.py)
- **Exits**:
  - `state_updated`: The action was applied (place, move, remove), the state was only read (query), or the observation agreed with every brick expected in view; state_json is the new state.
  - `inconsistent_with_observation`: observe found missing, displaced or extra bricks (discrepancies_json); the state is unchanged unless apply_observation, which removes the missing, moves the displaced to where they were seen and adds the extras.
  - `rejected`: The request could not be applied and the state is returned unchanged - a placement that would interpenetrate a tracked brick or reuse an id, a move that would carry bricks into others, an unknown brick id, or a malformed brick / observation.

### tracking-identical-parts-by-id

perceive · robots: any · tasks: sorting, pick-place, domino, jigsaw, lego · MIT · v0.1.0 · [`SKILL.md`](skills/tracking-identical-parts-by-id/SKILL.md) · [upstream](https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff/yolox/tracker/byte_tracker.py)

Keeps a stable id for each of many look-alike parts (dominoes, same-colour bricks, jigsaw pieces, cubes) across overhead frames while the arm covers and uncovers them - ByteTrack (Kalman box prediction, two-stage IoU matching that also uses low-score boxes) over a detector's boxes, one frame per call with the tracker state carried by the graph; reports parts that went out of sight and new ids born where a lost part was (a likely id switch). CPU only. Use when a graph must know which of several identical parts is which over time (checkpointing, backtracking, disturbance checks) and a per-frame detector alone renumbers them.

- **In**: `prompts` string
- **Out**: `n_tracked` int, `n_lost` int, `n_detections` int, `reason` string
- **Calls**: [`grounding-dino`](#grounding-dino), [`owlv2`](#owlv2); connector `robot.get_observation`
- **Scripts**: [`track_parts`](skills/tracking-identical-parts-by-id/scripts/track_parts.py)
- **Exits**:
  - `tracks_updated`: Every track that was in view is matched (or new parts started tracks); `tracks` lists id, box, label, state (tracked / lost) and frames lost for every live part.
  - `track_lost`: At least one tracked part had no box this frame (covered by the arm, picked, or missed by the detector); its id stays in `tracks` as lost at its last box for track_buffer frames and `lost_ids` names it.
  - `id_switch_suspected`: A new track started within switch_radius (2.5) part sizes of where a lost part of the same label and size was last seen - probably that part, moved while covered; `suspected_switches` pairs new_id with lost_id. Decide which (re-identify, or accept the new id).
  - `no_frame`: The camera had no RGB or the detector raised; nothing was tracked and `state` is returned unchanged.

### tracking-objects

perceive · robots: any · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/tracking-objects/SKILL.md) · [upstream](https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/skills/tracking-objects)

Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. Seeds the tracker via text prompt on the first frame, then polls the stream at update_hz and advances via sam3.tracker_update until the workflow signals termination, publishing a tracker snapshot per tick. Use when a workflow needs the live mask + box of an object across many frames — e.g. a supervisor branch that monitors a target's location while a policy manipulates it.

- **In**: -
- **Out**: -
- **Calls**: [`sam3`](#sam3)
- **Functions**:
  - `tracking-objects.track`: Run the SAM3 tracker loop over the observation stream; returns the final mask/box/confidence.

### tracking-points

perceive · robots: yam_i2rt · tasks: point-tracking · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/tracking-points/SKILL.md)

Tracks selected image points through the configured workcell tracking backend and reports visibility evidence. Use when an agent needs point correspondences for motion observation without moving the robot.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/tracking-points/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

### training-a-part-detector-from-auto-labels

perceive · robots: any, *tuned on* yam-station-orbbec-overhead, yam-left-wrist-d405 · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/training-a-part-detector-from-auto-labels/SKILL.md) · [upstream](https://github.com/autodistill/autodistill/tree/838e1c39f9fd)

Gets a fast station-specific detector for a fixed set of parts (bricks, dominoes, jigsaw pieces) without hand labelling. An open-vocabulary teacher (mm-grounding-dino by default, through sahi.detect_sliced for 10-30 px parts) boxes a folder of station frames from a caption ontology ("lego brick" -&gt; brick, "domino" -&gt; domino). The boxes become a COCO dataset with a train / valid split, the labels are checked (per-class counts, empty frames), and a small RF-DETR student is fine-tuned on them with tools/rfdetr. Routes detector_trained, too_few_labels, label_quality_low or training_failed. Use when a closed loop needs a detector for a fixed set of parts and the open-vocabulary detectors are too slow (0.3-2 s a frame) - once per station and part set, again when new parts arrive; the student runs in 7 ms.

- **In**: `frames_dir` string, `ontology` string, `out_dir` string
- **Out**: `model_path` string, `n_images` int, `n_empty` int, `median_score` float, `dataset_dir` string
- **Calls**: [`mm-grounding-dino`](#mm-grounding-dino), [`rfdetr`](#rfdetr), [`sahi`](#sahi)
- **Scripts**: [`train_from_auto_labels`](skills/training-a-part-detector-from-auto-labels/scripts/train_from_auto_labels.py)
- **Exits**:
  - `detector_trained`: The teacher's labels passed the checks, the dataset was written to out_dir/dataset and rfdetr.train wrote the model directory model_path (use it with rfdetr.detect).
  - `too_few_labels`: Some class has fewer than min_labels (50) teacher boxes over all frames; counts says which; add frames that show it (or lower min_score).
  - `label_quality_low`: The teacher found nothing in more than max_empty (50 %) of the frames - wrong prompts, wrong scale or frames without the parts; nothing was trained.
  - `training_failed`: rfdetr.train raised (no GPU, out of memory, a broken dataset) - the on_error route; the dataset stays in out_dir/dataset.

### aligning-parts-by-connector-constraints

plan · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/aligning-parts-by-connector-constraints/SKILL.md)

Turns connector statements about a LEGO or DUPLO build - "part P's tubes on base B's studs, P's stud (a, b) over B's stud (i, j), turned r" - into exact plate cells, layers and world poses, solving them in dependency order on the stud lattice (Manual2Skill++'s connector-aware pose alignment, which for studs and tubes reduces to integer lattice arithmetic). A part named by two statements (a bridge over two bases) must come out the same from both. Unsupported parts (on none of the base's studs), overlaps, cycles, unknown bases and parts off the plate are reported. Outputs the steps planning-brick-assembly-order reads and per-part cells for computing-brick-pose-from-stud-grid. Use when a build is described at the connector level (by a VLM reading a manual page, or a person saying "put the 1x2 on the left two studs of the red 2x4") rather than as an LDraw file, and needs exact stud-grid poses.

- **In**: `statements_json` string
- **Out**: `parts_json` string, `steps_json` string, `order_json` string, `problems_json` string, `n_parts` int
- **Scripts**: [`align_connectors`](skills/aligning-parts-by-connector-constraints/scripts/align_connectors.py)
- **Exits**:
  - `poses_solved`: Every part got cells, a layer and a rotation (and a world pose when the plate's origin is given); steps_json lists them in dependency order.
  - `constraints_inconsistent`: A statement is malformed, names an unknown base, forms a cycle, puts a part on fewer than min_support of its base's studs, overlaps another part, hangs off the plate, or two statements for one part disagree; problems_json names the first one and the statement index.

### assigning-bricks-to-two-arms

plan · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/assigning-bricks-to-two-arms/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/APEX-MR/blob/46a9448d9eaac5bd5973e0cd064623c5e7f5254e/src/python/task_assignment.py)

Assigns every brick of a LEGO or DUPLO build sequence to an arm - which arm places it, from which stock brick, with which grasp (jaws across the brick's x or y, offset along it), and which other arm holds the structure while it is pressed - with APEX-MR's task-assignment MILP (each task one arm, one brick of its type and one grasp; a support arm that is never the placing arm and takes the same grasp side; stock bricks used once unless they are reusable stations; stock precedence; the per-arm load balanced in every window of n consecutive tasks, weight 1000, so the arms alternate), solved with SciPy's HiGHS. Costs are travel distances from each arm's base to the stock and on to the target, and grasps off the brick's centre cost more. Arm reach (a radius, and optionally an IK preview) and the fingers' room beside the target decide what is feasible. Use after planning-brick-assembly-order, when two arms build together and the graph needs per-brick arm, grasp and support duties before computing poses.

- **In**: `sequence_json` string, `plate_origin` Vec3
- **Out**: `assignment_json` string, `by_arm_json` string, `n_tasks` int, `n_support` int, `objective` float, `imbalance` float, `travel` float, `solve_s` float
- **Calls**: connector `robot.solve_ik`
- **Scripts**: [`assign_arms`](skills/assigning-bricks-to-two-arms/scripts/assign_arms.py)
- **Exits**:
  - `assignment_ready`: Every task has an arm, a stock brick (when stock is given) and a grasp, and every task that needs support has a second arm; assignment_json lists them in task order.
  - `infeasible`: Some task has no feasible choice (out of every arm's reach, no stock brick of its size, the jaws too narrow or placed neighbours blocking the fingers on both grasp sides), more arms than tasks, or the MILP has no solution; detail names the tasks.
  - `needs_support_arm`: The bricks can all be placed, but a task that needs support has no second arm that reaches its support point (or the support duties cannot be fitted with the rest); detail names the tasks.

### checking-design-connectivity-and-collisions

plan · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/checking-design-connectivity-and-collisions/SKILL.md) · [upstream](https://github.com/AvaLovelace1/BrickGPT/blob/9c53c7cb65f9de0df89562cc74b2d2e270fbcab4/src/brickgpt/data/brick_structure.py)

Checks a brick design on a stud grid before the arms build it - BrickGPT's rules for bricks off the plate, overlapping bricks and bricks not connected to the plate through stud connections (a ground-connected component of the brick graph), plus a per-step check that every brick is held by what is already built and is not slid under a brick already placed, single-stud joints that carry an overhang, and a stud-clutch stability score from a linear program - and routes valid / out_of_plate / collision / floating / step_unsupported / step_blocked / weakly_connected / unstable. Use when a build instruction (a brick list in stud-grid cells or an LDraw step file) or a generated design must be accepted or rejected, or a planned build order checked step by step, before any brick is picked.

- **In**: -
- **Out**: `detail` string, `n_bricks` int, `n_steps` int, `stability_max` float, `worst_brick` int, `unstable_step` int, `elapsed_s` float
- **Scripts**: [`check_design`](skills/checking-design-connectivity-and-collisions/scripts/check_design.py)
- **Exits**:
  - `valid`: Every brick is on the plate, none overlap, all are connected to the plate through studs, each is connected and reachable from above when its step comes, no single-stud joint carries an overhang (unless allow_weak), and the stability score is under 1 (each prefix too with stability 'steps').
  - `invalid_input`: No bricks, a size / rotation / brick_id that cannot be read, an LDraw part that is not a one-layer basic brick, an LDraw position off the stud grid or a rotation that is not an upright quarter turn; `detail` says which.
  - `out_of_plate`: A brick leaves the plate_size grid or the layers 0 .. max_layers - 1; `out_of_plate` lists them.
  - `collision`: Two bricks share a cell; `collisions` lists the pairs.
  - `floating`: A brick is not connected to the plate through a chain of stud connections (above or below); `floating` lists them, `isolated` the ones touching no brick at all.
  - `step_unsupported`: The design is connected, but some brick is not connected to the plate through the bricks before it in the step order when its turn comes; `step_unsupported` lists them.
  - `step_blocked`: Some brick is placed under (anywhere in a higher layer over its footprint) a brick already placed, so it cannot be pressed on from above; `step_blocked` lists [brick, blocker] pairs.
  - `weakly_connected`: A joint of a single stud carries bricks that reach past that stud (removing it would cut them off the plate); `weak_joints` lists [lower, upper] (-1 = the plate), `weak_bricks` what hangs on them.
  - `unstable`: Holding the design (or, with stability 'steps', one of its partial builds) up needs more than the stud clutch (clutch_n, 0.98 N) at some stud; `stability_max` is the utilisation, `worst_brick` / `unstable_step` say where.

### choosing-a-domino-move-by-end-matching

plan · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/choosing-a-domino-move-by-end-matching/SKILL.md) · [upstream](https://github.com/abw333/dominoes/blob/ea9f532c9b834117a5c07d214711515872f7537e/dominoes/players.py)

Chooses the robot's move in a game of dominoes and where the tile goes on the table. From the hand (values as reading-domino-pips-and-orientation reads them), the board's two open ends (value, world edge point, outward direction), the other players' tile counts and the boneyard, it lists the legal moves (a tile holding an end's value goes on that end, as abw333/dominoes does), ranks them with a player - random, greedy (biggest pip sum), doubles first, or search (the source's probabilistic alpha-beta over deals of the hidden tiles) - and returns the tile, the end and the tile's world pose with its matching half touching that end - straight on, round a corner before the table edge or a placed tile, doubles crosswise. With no legal move it routes draw, pass or game_over. Block or draw rules, 2-4 players, double-six or bigger sets. No motion. Use on the robot's turn in a domino game, after reading the hand and the board ends, before placing the tile.

- **In**: -
- **Out**: `hand_index` int, `end` int, `end_value` int, `layout` string, `place_pose` Se3Pose, `place_yaw_deg` float, `my_pips` int, `game_over_reason` string
- **Calls**: connector `robot.describe_workspace`
- **Scripts**: [`choose_move`](skills/choosing-a-domino-move-by-end-matching/scripts/choose_move.py)
- **Exits**:
  - `play`: A legal move was chosen and fits; `tile` (the hand's [a, b], a at the tile's end a), `hand_index`, `end` (0 / 1, -1 for the opening), `layout` (straight / crosswise / corner_left / corner_right), `place_pose` (tile centre, yaw of the a-to-b axis), `target_ends_world` (where end a and end b go), `new_ends` and `new_footprint` for the next turn. Ends given without positions give the move with an empty pose.
  - `draw`: No tile holds either end value, the rules are `draw` and the boneyard has tiles; draw one, read it, and run this skill again.
  - `pass`: No tile holds either end value and the boneyard is empty (or the block game); also when the opening tile `must_play` is not in this hand.
  - `game_over`: The hand is empty (`out`), an opponent has no tiles (`opponent_out`), or the game is blocked (`blocked`) - every other player passed in a row and this hand cannot play either, or every tile holding either end value is in `played`.
  - `no_room`: Legal moves exist but none has a placement inside the table (less `edge_margin`) and clear of the placed footprints, straight or round a corner; `tile` / `end` carry the player's choice. Make room (slide the line) or play it by hand.
  - `invalid_input`: A hand tile or end value outside 0..max_pip, a tile twice, not two ends, an end direction with no length, an unknown player or rules, search without opponent_counts, or fewer unseen tiles than the hidden places (also the route of a raised error).

### computing-brick-pose-from-stud-grid

plan · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/computing-brick-pose-from-stud-grid/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/Lego.cpp)

Converts between a cell on a LEGO or DUPLO baseplate's stud grid (x, y, layer, brick size such as 2x4, rotation 0/90/180/270) and metric world poses - ICL's calc_brick_loc chain (plate pose * top-left offset * cell offset * yaw * brick-centre offset) - giving the brick's body-centre pose, the seat pressing-a-brick-onto-studs takes, a top-down parallel-jaw grasp across the short side with its approach (press hover), and pose6d's box and support plane; and back, from a world pose (e.g. pose6d's) to the nearest cell with the residual. Checks the plate's bounds and, given occupied cells, collisions, support from the layer below and room for the fingers. Use when a build step names a cell on a calibrated baseplate and the arm needs where to grasp or press, or when a perceived brick must be read as a grid cell.

- **In**: `plate_pose` Se3Pose
- **Out**: `brick_pose` Se3Pose, `top_pose` Se3Pose, `seat_position` Vec3, `seat_yaw_deg` float, `system` string, `stud_pitch` float, `stud_height` float, `stud_diameter` float, `brick_height` float, `grasp_pose` Se3Pose, `grasp_pose_flipped` Se3Pose, `approach_pose` Se3Pose, `press_pose` Se3Pose, `close_heading_deg` float, `close_axis` string, `jaw_width` float
- **Calls**: connector `robot.describe_gripper`, `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`brick_pose`](skills/computing-brick-pose-from-stud-grid/scripts/brick_pose.py)
- **Exits**:
  - `computed`: The cell is on the plate (and, given occupied cells, free, supported, with room for the fingers; the jaws open wide enough where the gripper is described; the grasp passes the IK preview when check_reach). For a pose input, it lies within the tolerances of its nearest cell.
  - `off_grid`: Pose input only. The nearest cell is returned, but the pose is more than lateral_tolerance (a quarter pitch) from it along a plate axis, more than a quarter brick height from a layer, yawed more than 10 deg off the rows or tilted more than 10 deg; residual says by how much.
  - `out_of_plate`: Part of the footprint lies off the plate's stud grid, or the layer is under 1. Poses are still filled in.
  - `collides`: One of the brick's cells is in occupied; collisions lists them.
  - `unsupported`: Occupied cells were given, the layer is 2 or more, and nothing occupies the layer under any of its cells.
  - `ungraspable`: The finger rows beside the brick (along the closing axis, at its layer and above) are occupied on both sides it could be grasped across, or the described jaws open less than the brick's width + jaw_margin.
  - `unreachable`: check_reach, and neither wrist solution of the grasp and approach passes robot.solve_ik.
  - `invalid`: Malformed inputs (unknown system, rotation not a multiple of 90, neither or both of cell / observed_pose, a grasp_depth outside the brick).

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

### computing-stable-placements

plan · robots: any · tasks: pick-place, reorientation, lego, domino, jigsaw · BSD-2-Clause · v0.1.0 · [`SKILL.md`](skills/computing-stable-placements/SKILL.md) · [upstream](https://github.com/personalrobotics/tsr/blob/5279ca7baf95e6718d4628a3a26fd6ebeefedf0d/src/tsr/placement/_stable_poses.py)

Enumerates the stable resting faces of a rigid object - from its oriented bounding box or a perceived point cloud closed down to the table - by convex hull and centre-of-mass projection, and reports for each rest the tipping angle, the centre-of-mass height, the turn from the object's current orientation that puts that face down, its footprint, and (given a goal point) the resting pose; also says which rest the object is on now, or that it is on none (leaning, tilted, held). Routes placements or no_stable_pose. Use when a graph must choose how to set a part down (a brick, a domino on its end or flat, a jigsaw piece), decide whether a part lying on the table needs a regrasp or flip, or reject a placement too easy to tip; geometry only, no motion.

- **In**: -
- **Out**: `n_rests` int, `best_tipping_deg` float, `best_turn_deg` float, `best_com_height_m` float, `best_pose` Se3Pose, `current_index` int, `current_stable` bool, `com` Vec3, `reason` string
- **Scripts**: [`compute_rests`](skills/computing-stable-placements/scripts/compute_rests.py)
- **Exits**:
  - `placements`: At least one hull face holds the centre of mass inside its support polygon with a tipping angle of at least min_margin_deg and fits the surface at every yaw; `rests` lists them most stable first, `best` is the first.
  - `no_stable_pose`: No face holds the centre of mass (or none clears min_margin_deg, or none fits surface_half_extents); `reason` counts which filter removed them.

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

### finding-the-longest-domino-chain

plan · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/finding-the-longest-domino-chain/SKILL.md) · [upstream](https://github.com/RGBz/mexican-train-domino-solver/blob/ac758e87bbae433a5a10ba4a6907874be17d80d6/src/Graph.ts)

Finds the longest chain of matching dominoes a hand can lay from a start value - the Mexican Train move of building your own train from the engine double or your train's open end, or a solitaire end-to-end line - and lays it out on the table. Tiles are edges between pip values; every chain is walked depth first (as RGBz/mexican-train-domino-solver does) with an exact bound that cuts branches which cannot win, ranked longest, then most pips, then not ending on a double (a double left open must be covered). Returns the chain with each tile turned so its matching half comes first, the leftover tiles, the open value at the end and, from the train end's world point and direction, every tile's place pose and end points - straight on, round a corner before the table edge or a placed tile, doubles crosswise. No motion. Use on a Mexican Train turn or to plan a matching-ends line for the arm to build, after reading the tiles' values.

- **In**: -
- **Out**: `start_value` int, `length` int, `chain_pips` int, `end_value` int, `ends_in_double` bool, `leftover_pips` int, `placed_count` int, `nodes` int, `exhaustive` bool, `length_bound` int, `layout_tries` int
- **Calls**: connector `robot.describe_workspace`
- **Scripts**: [`find_chain`](skills/finding-the-longest-domino-chain/scripts/find_chain.py)
- **Exits**:
  - `chain_found`: A chain from the start value was found and, when start_end has a position, every tile has a placement inside the table and clear of the placed tiles; `chain` (per tile - `tile` as held, `hand_index`, `near` / `far` value, `layout`, `place_pose`, `place_yaw_deg` of the a-to-b axis, `target_ends_world`), `chain_tiles` ([near, far]), `leftover`, `new_end` (the train's end after the chain) and `footprints_after` are filled. `exhaustive` false means the node budget stopped the search; `length_bound` tells whether the length is still proven longest.
  - `no_chain`: The hand is empty or no tile holds the start value; `leftover` is the hand.
  - `no_room`: The chain was found but tile `placed_count` has no placement inside the table (less `edge_margin`) and clear of `footprints` and the tiles before it, straight or round a corner; the first `placed_count` tiles are placed and `new_end` is the end they leave.
  - `invalid_input`: A tile outside 0..max_pip or with not two values, a tile twice, an unknown objective, a start value outside 0..max_pip, a start direction with no length in the table plane, or a footprint without an axis (also the route of a raised error).

### parsing-ldraw-build-instructions

plan · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/parsing-ldraw-build-instructions/SKILL.md) · [upstream](https://github.com/michaelgale/ldraw-py/blob/d650ab0a9c2a774e1ba6f206aa2cd3b1925478e3/ldrawpy/ldrmodel.py)

Reads an LDraw .ldr / .mpd build file - submodels, STEP and ROTSTEP meta commands, part ids, colour codes, transforms in LDraw units - into ordered build steps of bricks on a stud grid, each brick as a cell (x, y, layer from 1), a size 'AxB', a rotation 0 / 90, a height in plates and a colour name from the LDraw colour table. Basic LEGO bricks, plates and tiles (1xN, 2xN, 4xN ...), DUPLO bricks and plates, and baseplates are converted; every other part, and any part that is turned off the 90-degree grid or sits between studs, is listed with a reason instead of guessed. Use when a build instruction comes as an LDraw file (from an editor, BrickGPT, or the LDraw model library) and the robot needs the bricks it describes, in order, in the stud-grid convention computing-brick-pose-from-stud-grid and planning-brick-assembly-order take.

- **In**: `ldraw_path` str
- **Out**: `system` str, `n_bricks` int, `n_steps` int, `n_unsupported` int, `n_off_grid` int, `steps_json` str
- **Scripts**: [`parse_ldraw`](skills/parsing-ldraw-build-instructions/scripts/parse_ldraw.py)
- **Exits**:
  - `steps_ready`: Every part became a brick on the grid; `steps` holds them in build order.
  - `unsupported_part`: Some parts are not in the basic-brick table (slopes, Technic, minifigures, a missing sub-file, the other system's bricks); they are in `unsupported` with the line and step, and `steps` holds the rest. A plan that needs them must be changed or finished by hand.
  - `off_grid`: Some table parts are tilted, upside down, mirrored, turned by other than a multiple of 90 deg, or sit off the stud or plate grid (a half-stud jumper offset); they are in `off_grid` with the reason. Takes precedence over unsupported_part.
  - `invalid_file`: No file or text, an unreadable file, no parts at all, a submodel that includes itself, or a bad argument; `detail` says which.

### planning-a-brick-mosaic-from-an-image

plan · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/planning-a-brick-mosaic-from-an-image/SKILL.md) · [upstream](https://github.com/merschformann/brickmos/blob/4a7e956720d347d198f931edc38852c9f3bf4060/brickmos/brickify.py)

Turns an image into a one-layer LEGO mosaic on a baseplate of a given stud size - brickmos' pipeline (pixelate to one cell per stud, nearest palette colour per cell, bill of materials, BrickLink wanted list) with CIEDE2000 matching in CIELAB, optional Floyd-Steinberg dithering, per-colour part counts respected (an optimal assignment when stock is short, with the substitutions reported), runs of one colour merged into 1x2 / 2x2 / ... plates to cut placements, a preview image, and an ordered placement list in the hub's stud-grid convention (cell [x, y, layer], layer 1 on the plate, size 'AxB', rotation, colour) that checking-design-connectivity-and-collisions, planning-brick-assembly-order and computing-brick-pose-from-stud-grid read. Use when a picture, logo or pattern must become a flat brick mosaic the arms can build from the parts in the bins, or to find out before building which colours are short.

- **In**: -
- **Out**: `detail` string, `n_cells` int, `n_placements` int, `n_empty` int, `n_substituted` int, `n_colours` int, `mean_delta_e` float, `p95_delta_e` float, `max_delta_e` float, `mean_delta_e_blurred` float, `ideal_mean_delta_e` float, `placements_json` string, `steps_json` string, `bom_json` string, `bricklink_xml` string, `preview_path` string, `pixelated_path` string, `elapsed_s` float
- **Scripts**: [`plan_mosaic`](skills/planning-a-brick-mosaic-from-an-image/scripts/plan_mosaic.py)
- **Exits**:
  - `planned`: Every cell of the mosaic has a part in the colour nearest to the image there (to the dithered target with dithering) and the inventory, if given, covers every part.
  - `planned_with_substitutions`: Every cell has a part, but some cells use another colour because their nearest colour ran out (or had no part of the size needed); `substitutions` lists wanted -&gt; used, how many cells and the extra CIEDE2000.
  - `inventory_short`: The inventory has fewer studs than the mosaic has cells (or no 1x1 left in any colour for a cell no larger part fits); `empty_cells` lists the uncovered cells, `placements` covers the rest.
  - `invalid_image`: The image is missing, unreadable, empty, not [H, W], [H, W, 3] or [H, W, 4], or has NaN / infinite values.
  - `invalid_input`: An option cannot be read - unknown palette / fit / resample / dither / order / part_kind / system, a mosaic off the plate or over 128 x 128, sizes without 1x1, an inventory colour not in the palette or a negative count; `detail` says which.

### planning-a-domino-layout

plan · robots: any · tasks: domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/planning-a-domino-layout/SKILL.md) · [upstream](https://github.com/maxencelaunayquerre/domino-fall-simulation/blob/f6147f635b2a4a3f3f7ca0589284f45dda6aa309/scripts/spacing_threshold.py)

Plans a domino line along a path on the table - a line, an arc, a polyline (corners rounded) or a sampled curve - and returns the ordered slot poses (tile centre and facing, the pose placing-an-upright-domino-beside-neighbours takes), the pitch and how far it stays under the critical gap a toppling wave can cross (from a 2D toppling simulation), the checks that matter on a curve (the inner corners, where a falling tile's top lands on the next one), safety gates left open while building and the order to close them, and a build order in which the open hand never meets a standing tile and the arm never reaches past one. No motion. Use before building a domino line or curve with the arm, to choose the pitch and the slots and to know whether the layout will topple.

- **In**: `path` Centerline
- **Out**: `first_slot` Se3Pose, `pitch` float, `n_tiles` int, `face_gap` float, `critical_gap` float, `critical_pitch` float, `topple_margin` float, `max_face_gap` float, `gate_count` int, `gate_width_slots` int, `gate_margin` float, `min_inner_gap` float, `max_turn_deg` float, `max_contact_offset` float, `path_length` float, `used_length` float
- **Calls**: connector `robot.describe_workspace`
- **Scripts**: [`plan_layout`](skills/planning-a-domino-layout/scripts/plan_layout.py)
- **Exits**:
  - `planned`: Every slot passes every check; `slots` / `slot_poses` (path order, slot 0 is the trigger tile), `build_steps` (placing order with each step's arm, standing neighbours and tiles at risk), `gates` (in closing order) and the margins are filled. `problems` lists what was dropped or is outside what has been seen to work (an automatic gate with no clear spot, a pitch over 1.0 h, slots placed between two standing tiles).
  - `too_tight`: The tiles are too close for the hand or the physics - a pitch under what the open fingers need beside a neighbour (19.8 mm for a 7.5 mm tile), a pitch at which a slow trigger push stalls, footprints overlapping at an inner corner or where the path comes back on itself, a curve so sharp that a falling tile's top lands off the next tile's face, or a build step whose open fingers would touch a standing tile; `detail` names the slots.
  - `too_loose`: A pair of slots is farther apart than the toppling wave can cross, less `topple_margin_min` (the critical gap is 47.8 mm face to face for a 48 x 24 x 7.5 mm tile, a 55.3 mm pitch); `detail` names the pair.
  - `too_short`: The path (or `length`) holds fewer than two tiles, or not `n_tiles` at the pitch.
  - `out_of_reach`: A slot is outside every arm's tool-centre box or reach, off the `table`, or the placing skill's lead-in leg at a slot would leave the box.
  - `blocked`: An obstacle is where the open hand goes over a slot, a given gate cannot stop the wave or cannot be closed without the arm reaching past standing tiles, or no order places every slot without reaching past standing tiles.
  - `invalid_input`: The path or tile size cannot be read (also the route of a raised error).

### planning-a-flat-domino-mosaic-from-an-image

plan · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/planning-a-flat-domino-mosaic-from-an-image/SKILL.md) · [upstream](https://github.com/ArnoldF/DominoArtSolver/blob/3f9e260ea97a6ad659784fc6931bbb64d9196eba/src/domino_model.py)

Plans a domino mosaic of a picture, every tile lying flat with its pips up - DominoArtSolver's integer program chooses, for a number of complete double-six sets, which tile covers which two neighbouring grid cells and which way round, so the pips' darkness reproduces the image (all tiles used once per set; a smaller grid picks from spare tiles). Returns an ordered placement list in world (grid origin, cell pitch and yaw given; 48 x 24 mm tiles by default) with each tile's value, which end carries which value, its pose, its arm, the copy of the set it comes from and the free sides for the fingers; the order places the tiles farthest from each arm first and never has an arm reach across a placed tile (one or two arms). Also a preview render, per-set usage and the image error. CPU only, no motion. Use before laying a domino mosaic with the arm, to know which tile to fetch from the supply (reading-domino-pips-and-orientation identifies it there), where to put it and which way it faces.

- **In**: `image_path` str, `sets` int
- **Out**: `placements_json` str, `pose_sequence` PoseSequence, `first_pose` Se3Pose, `n_tiles` int, `rows` int, `cols` int, `sets` int, `max_pips` int, `objective` float, `cell_rmse` float, `floor_rmse` float, `image_rmse` float, `correlation` float, `solver` str, `solver_status` str, `solve_time_s` float, `reach_over` int, `steps_without_finger_room` int, `preview_path` str
- **Scripts**: [`plan_mosaic`](skills/planning-a-flat-domino-mosaic-from-an-image/scripts/plan_mosaic.py)
- **Exits**:
  - `planned`: Every cell is covered once and every tile used `sets` times (or at most `sets` times on a given smaller grid); `placements` (in placing order), `pose_sequence`, `by_arm`, `per_set`, `supply`, `values`, `preview` and the error measures are filled. `problems` lists what a robot should know - poses in the grid's own frame (no `grid_origin`), a grid that stretches the image, steps without finger room at this pitch, a plan found by the time limit but not proven optimal, steps that reach across a placed tile.
  - `not_enough_sets`: `sets` is under 1, or the given `rows` x `cols` grid needs more tiles than the sets hold (`detail` says how many sets it needs).
  - `solver_timeout`: The solver found no tiling within `time_limit_s`, or (with `accept_suboptimal` false) found one but could not prove it optimal.
  - `invalid_image`: No image, an unreadable file, an array that is not H x W (x 3), an image smaller than the grid in pixels, or no contrast between the grid's fields.
  - `out_of_reach`: Some cells are outside every arm's `reach_m` (or inside its `min_reach_m`); `out_of_reach` lists them as [row, col]. Checked before solving, so nothing is planned.
  - `solver_unavailable`: Neither PuLP with a CBC binary nor SciPy's `milp` (HiGHS) could be loaded.
  - `solver_failed`: The solver reported the model infeasible or returned values that are not a tiling (not seen; the model is always feasible).
  - `invalid_input`: A grid with an odd cell count or only one of rows / cols, a pitch smaller than the tile, an unknown option, a bad origin or arm (also the route of a raised error).

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

### planning-brick-assembly-order

plan · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/planning-brick-assembly-order/SKILL.md) · [upstream](https://github.com/AvaLovelace1/BrickGPT/blob/9c53c7cb65f9de0df89562cc74b2d2e270fbcab4/src/mesh2brick/src/mesh2brick/planning.py)

Orders the bricks of a stud-grid design for a robot that presses each one down from above - every brick rests on the baseplate or on studs of a brick already placed (bottom-up, connected), nothing placed covers its footprint, and a parallel gripper's fingers have room on two opposite sides of it (no flush neighbour at its height along the closing axis). Earlier LDraw STEPs go first unless a brick cannot go yet; ties go bottom layer first, then BrickGPT's (x, y) sweep, with a look-ahead that never walls in a brick still to come. Optionally splits the bricks between two arms by x (with per-arm reach) and checks an inventory. Use when a build (from parsing-ldraw-build-instructions or written by hand as cells, sizes and rotations) must become a placement sequence for computing-brick-pose-from-stud-grid and pressing-a-brick-onto-studs, or to find out before moving which bricks a gripper cannot place.

- **In**: `steps_json` str
- **Out**: `system` str, `n_bricks` int, `n_placed` int, `n_not_placed` int, `sequence_json` str
- **Scripts**: [`plan_order`](skills/planning-brick-assembly-order/scripts/plan_order.py)
- **Exits**:
  - `sequence_ready`: Every brick is in `sequence`, each supported, uncovered and with a free pair of finger sides when its turn comes.
  - `inventory_short`: Some bricks are left out because the inventory has no more of their part / size; `not_placed` lists them and the bricks that then had nothing to stand on. `sequence` holds the rest.
  - `unreachable_step`: Some bricks lie outside every arm's reach (`reach`); listed in `not_placed`, the rest ordered.
  - `finger_blocked`: Some bricks never had two opposite sides free for the fingers (flush neighbours on both axes, e.g. the last brick of a closed ring, or wider than the jaws open both ways); `not_placed` says which brick took the last side. Bricks that would have stood on them are listed too.
  - `not_buildable`: Some bricks have nothing under them to press onto (hanging, on a tile, on a part not in the list), were covered before they could go, overlap another brick, or wait on a step that cannot finish (strict); listed in `not_placed`.
  - `invalid_input`: No bricks, a brick without cell or size, a rotation not a multiple of 90, bricks of two systems, more than two arms, or an unknown option; `detail` says which.

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

### routing-parts-to-sort-bins-by-id-and-colour

plan · robots: any · tasks: sorting, pick-place · MIT · v0.1.0 · [`SKILL.md`](skills/routing-parts-to-sort-bins-by-id-and-colour/SKILL.md) · [upstream](https://github.com/mbric/lego-sorter-public/blob/525683d3da4ec03017d9355afff8df2f5218b179/src/orchestrator/sort_cycle.py)

Sorts parts dropped one at a time on a staging pad into bins - watches the pad against an empty reference frame (grey and depth change, largest blob in percent of the zone), waits until a part has landed and the zone stops moving, rejects a capture while two parts or a part on the zone's edge are in it, classifies the part's colour (colour.classify), measures its footprint from depth and optionally its part id, chooses the bin by a rule list (colour, then part id, then category, then an "unknown" bin) with per-bin capacities and an overflow bin, and outputs the bin's drop position; the execute step picks the part (grasping-flat-objects) and drops it into the bin (placing-into-container-obb's drop points over a taught bin, or its perceived place for a bin given by description). Use when parts arrive on a known pad one by one (a feeder, the other arm, a person) and must go to bins by colour or size - LEGO bricks before a build, dominoes, jigsaw pieces into edge / colour trays.

- **In**: `reference` Observation
- **Out**: `bin_name` string, `drop_position` Vec3, `part_obb` OrientedBoundingBox, `part_colour` string, `category` string, `part_id` string, `label` string, `long_mm` float, `short_mm` float, `height_mm` float, `occupancy` float, `pose6d_model` string, `release_position` Vec3
- **Calls**: [`colour`](#colour), [`geometry`](#geometry), [`pose6d`](#pose6d), [`sam3`](#sam3); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`plan_sort_route`](skills/routing-parts-to-sort-bins-by-id-and-colour/scripts/plan_sort_route.py), [`sort_into_bin`](skills/routing-parts-to-sort-bins-by-id-and-colour/scripts/sort_into_bin.py)
- **Exits**:
  - `sorted`: A part was captured and routed, the hand picked it (grasping-flat-objects read its jaw gap across it after the lift) and opened over the chosen bin with the part's bottom release_over_rim over the rim (taught bin), or placing-into-container-obb placed it (described bin); the bin's count went up by one. What the hand did, not a sighting of the part in the bin.
  - `unknown_part`: The part was captured but no rule matched and no bin accepts its colour, part id, category or "unknown" (or a rule names a bin not in bins); nothing moved.
  - `bin_full`: The part's bin holds capacity parts (by bin_counts) and no overflow bin has room; nothing moved.
  - `staging_empty`: No part landed (largest changed blob under occupied_threshold % of the zone for occupied_frames frames) within max_wait_s, the change was an empty-crop ghost, or there is no reference / camera frame or the zone projects outside it; nothing moved.
  - `staging_not_clear`: Something landed but the capture was rejected 4 times (two parts in the zone), or it never settled (zone motion over motion_threshold) within max_wait_s; nothing moved - clear or singulate the pad.
  - `grasp_failed`: The pick did not hold the part (every close read empty, too wide or too narrow, it slipped in the lift, or no grasp could be tried), there was no part box (no depth), or a perceived place found the hand empty; the hand is open, the part is on the pad or near it.
  - `drop_failed`: The part was held but no drop point over the bin was reached (IK preview, a refused or short move), it does not fit the bin's opening, the jaws did not open, or a perceived place routed no_room / unreachable / container_not_found / not_released; still holding unless not_released.

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

### choosing-next-piece-and-slot

select · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/choosing-next-piece-and-slot/SKILL.md) · [upstream](https://github.com/scipy/scipy/blob/main/scipy/optimize/_lsap.py)

Chooses the next jigsaw placement (which loose piece, which grid cell, which turn) from the board as believed, the pieces' localization candidates and backtracking-a-wrong-placement's ledger. Loose pieces are assigned to free cells with SciPy's linear_sum_assignment on -log P(piece belongs in cell), and the confident pairs are ranked by a stage plan - frame before interior, cells with the most placed neighbours, then confidence - skipping forbidden pairs and deferred cells and respecting corner / edge / interior classes. With two arms each job stays inside one free arm's reach and clear of the other's work, and each arm gets a short separated queue. When nothing is confident it names the best-supported slot and candidates for a shape match, the pieces to photograph again, or asks for more pieces. Use when a jigsaw-assembly loop needs its next pick, after localizing-a- piece-in-a-reference-image and after any backtrack; planning only, milliseconds on CPU.

- **In**: `grid_cols` int, `grid_rows` int
- **Out**: `piece` string, `col` int, `row` int, `angle_deg` float, `prob` float, `n_neighbours` int, `arm` string
- **Scripts**: [`choose_next`](skills/choosing-next-piece-and-slot/scripts/choose_next.py)
- **Exits**:
  - `chosen`: A confident pair (P &gt;= min_confidence, 0.5) was assigned and is reachable; piece, col, row, angle_deg (the localizer's, so turn the piece by -angle_deg) and arm are the next placement, `queue` the next few per free arm.
  - `need_shape_match`: No pair is confident; `slot` is the open cell with the most placed neighbours (frame first on ties) and `shape_candidates` the pieces listing it, likeliest first, then the orphans (loose pieces with no confident cell). Run matching-a-piece-to-placed-neighbours on them and place its `fits` answer.
  - `need_relocalize`: No confident pair and no slot a shape match could use; `relocalize` lists the loose pieces with no confident cell that were not re-photographed max_relocalize (1) times yet.
  - `need_more_pieces`: The known loose pieces fit none of the open cells (or there are none); find and localize more pieces, or hand over.
  - `unreachable`: Confident jobs exist but none can be done by a free arm with the whole pick-to-place path in its reach and clear of a busy arm (or every arm is busy); `unreachable` lists them with each arm's reason. Move those pieces to a buffer both arms reach.
  - `done`: Every cell is placed.

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

### proposing-parallel-grasps-in-tight-clusters

select · robots: any · tasks: jigsaw, pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/proposing-parallel-grasps-in-tight-clusters/SKILL.md) · [upstream](https://github.com/andyzeng/arc-robot-vision/blob/728139e9ad0e003106c71a7c0d0a2fc9d8430644/parallel-jaw-grasping/baseline/predict.m)

Proposes top-down parallel-jaw grasps on small flat parts packed close together (jigsaw pieces, LEGO bricks, dominoes lying flat) from one posed depth frame (wrist D405 or overhead) and optional part masks - builds a world heightmap, slides the two fingertip footprints over centres, 8 jaw headings plus the part's axes and growing openings (arc-robot-vision's finger fit), keeps grasps whose fingertips reach half the part's height into free gaps at the opening the hand comes down with, with no other part between the jaws, the finger body clear of tall neighbours and the plates on an edge rather than a corner, and ranks them by finger clearance, centring and coverage. Returns world grasp points, the closing heading and jaw opening for robot.grasp_frame / robot.set_grip, and the fraction of parts with a collision-free grasp. Use when a part to pick has neighbours within a few fingertip widths, before a grasp skill, to find a grasp that fits or learn that the cluster must be spread first; CPU only.

- **In**: -
- **Out**: `grasp_pose` Se3Pose, `position` Vec3, `close_heading_deg` float, `jaw_width_m` float, `object_width_m` float, `tip_z` float, `tool_z` float, `clearance_m` float, `score` float, `n_pieces` int, `n_graspable` int, `fraction_graspable` float, `support_z` float
- **Calls**: connector `robot.describe_gripper`, `robot.get_observation`, `robot.grasp_frame`
- **Scripts**: [`propose_cluster_grasps`](skills/proposing-parallel-grasps-in-tight-clusters/scripts/propose_cluster_grasps.py)
- **Exits**:
  - `proposed`: At least one grasp fits (the target's, or any part's without a target mask); `grasps` are best first, and the best one's position, close_heading_deg, jaw_width_m, tip_z / tool_z and grasp_pose (robot.grasp_frame's rotation at the tool centre) are flattened. Nothing moved.
  - `too_crowded`: Parts were seen but no fingertip footprint fits beside any of them down to half their height, or every fit has another part between the jaws; `pieces` says why per part. Spread or singulate them (singulating-by-pushing), then call again.
  - `nothing_graspable`: No part stands out of the support (too thin for the depth noise, or none in view), every part is wider than the hand or offers the plates only corners, the target mask covers no depth, or the frame has too few depth pixels; `reason` says which.

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

### selecting-a-grip-clear-of-placed-neighbours

select · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/selecting-a-grip-clear-of-placed-neighbours/SKILL.md)

Chooses where a two-finger gripper should hold a jigsaw piece, and along which direction it should close, so that at the piece's slot neither fingertip (nor its outward travel on release) lands on an already-placed neighbour - the piece and its neighbours are rasterised at the slot, every closing direction (5 degree steps) and jaw offset (1 mm) is closed onto the piece's extreme points within the finger's width, and each fingertip footprint is checked against a distance transform of the neighbours; the clear grip nearest the piece's centre with the most contact and clearance wins. With no clear grip at the slot it searches the smallest offset at which the piece can be set down clear and pushed in. Also returns the same grip on the piece as it lies (pixel and closing yaw) for the pick. Use when a jigsaw piece is about to be picked for a known slot with a parallel gripper (suction robots never meet this), after the slot and turn are known; planning only, under 0.1 s on CPU (up to 1.5 s with the offset search).

- **In**: -
- **Out**: `closing_yaw_deg` float, `grip_width_m` float, `clearance_m` float, `n_clear` int
- **Scripts**: [`select_grip`](skills/selecting-a-grip-clear-of-placed-neighbours/scripts/select_grip.py)
- **Exits**:
  - `grip_found`: A grip at the slot keeps both fingertip footprints (plus release travel) at least min_clearance (0.5 mm) from every placed neighbour; grip_point, closing_yaw_deg, grip_width_m and contacts are in board metres at the slot.
  - `only_offset_place`: No grip is clear at the slot, but the piece can be set down offset_m away (up to 0.6 cell) clear of the neighbours with a clear grip there, then pushed into the slot (seating-a-piece-by-wiggle-search's offset-push); the grip is given at the offset pose.
  - `no_clear_grip`: No grip clears at the slot and no offset placement clears (typically three or four sides already placed), or no grip spans the piece within the gripper's opening, or the input could not be used; the least-bad grip is still returned for a release-above-and-press strategy.

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

### checking-reachability

execute · robots: yam_i2rt · tasks: motion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/checking-reachability/SKILL.md)

Asks a station whether TCP poses (optionally as a chained path from the current configuration) are reachable, using the backend's own IK and path rules, without moving. Use before committing to a grasp or place pose on an arm with joint limits, e.g. to choose between equal-score flipped grasp candidates.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/checking-reachability/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

### correcting-brick-offset-from-stud-circles

execute · robots: any · tasks: assembly, lego · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/correcting-brick-offset-from-stud-circles/SKILL.md) · [upstream](https://github.com/kevinpoi/legoscope/blob/2abdc205ef36/detect_offset.py)

Closes the last millimetre over a LEGO or DUPLO brick before a pick or a press. The hand peeks at a fixed height over the expected stud pair; the wrist RGB-D close-up gives the brick top's plane and a circle at the known stud radius for each stud, fitted only to the rim that is seen (occlusion-aware, as Eye-in-Finger forces its circle tangent to the visible mask edges); the two studs nearest the expected pair (with every other stud on their lattice) give the pair's midpoint and line, compared with the tool axis (or a calibrated tool-centre pixel); the hand steps by the difference in x, y and yaw and repeats until under 0.2 mm and 1 deg. Use when a brick's pose is known to a few millimetres (an overhead estimate, read_studs, computing-brick-pose-from-stud-grid) and a grasp or a press-fit needs it to a few tenths, on an arm whose moves land a millimetre off.

- **In**: `pair_pose` Se3Pose
- **Out**: `pair_pose` Se3Pose, `tool_pose` Se3Pose, `yaw_offset_deg` float, `iterations` int, `moves` int, `pair_range_m` float, `tool_axis_source` string, `camera_name` string
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`
- **Scripts**: [`correct_offset`](skills/correcting-brick-offset-from-stud-circles/scripts/correct_offset.py)
- **Exits**:
  - `aligned`: The tool axis (or the tool_center_px ray) was measured within tolerance_mm (0.2) of the stud pair's midpoint and within tolerance_deg (1.0) of the wanted yaw (and, with tool_center_range, the camera's range within tolerance_mm of it); the hand is at the peek height over the pair and pair_pose is the last measurement.
  - `ambiguous_stud_pair`: The nearest stud pair is over 0.4 pitch from the expected one, or two pairs are almost as near - the expected pose is about half a stud off, so which pair is meant cannot be told; nothing more is moved.
  - `lost_studs`: No posed wrist RGB-D frame, no brick-top plane within half a brick of the expected pair, or under two stud circles / no pair one pitch apart along the expected line.
  - `max_iterations`: max_iterations measurements were taken and the last was still over tolerance (with max_iterations 1 the skill only measures).
  - `unreachable`: The rise, the peek pose or a correction move was refused (routed or raised); detail says which.

### correcting-in-plane-offset-by-discrete-moves

execute · robots: any · tasks: insertion, lego, domino, jigsaw · MIT · v0.1.0 · [`SKILL.md`](skills/correcting-in-plane-offset-by-discrete-moves/SKILL.md) · [upstream](https://github.com/cooper-group-uol-robotics/vision-guided-rack-insertion-ot2/blob/3bfe3fe87b64/reference_implementation/ot2_rack_insertion/cnn_model_misalignment_prediction.py)

Final x, y and yaw alignment of a held part (or the hand) over its target from the wrist camera, by a small classifier that looks at one cropped wrist frame and says which way to nudge - +/-x, +/-y, +/-yaw or stay per axis (27 classes); the hand steps 1 mm / 0.5 deg and repeats until three "aligned" in a row. The classifier is trained per target from images the arm labels itself (collect_offsets jogs from a taught aligned pose to random offsets and records each offset). Use when a part must sit over a slot, stud grid or rack within a millimetre and the target has no reference image or keypoints to servo on (textureless bricks, domino backs, slots).

- **In**: `model_path` string, `camera` string
- **Out**: `decisions` int, `turned_deg` float, `last_label` string, `reason` string
- **Calls**: [`move-classifier`](#move-classifier); connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`
- **Scripts**: [`align`](skills/correcting-in-plane-offset-by-discrete-moves/scripts/align.py), [`collect_offsets`](skills/correcting-in-plane-offset-by-discrete-moves/scripts/collect_offsets.py)
- **Exits**:
  - `aligned`: The classifier said "000" (no move on any axis) streak (3) times in a row; the hand stays there for the next skill to lower / place.
  - `max_steps`: max_decisions (80) decisions without three aligned in a row (dithering, or a target the model was not trained on).
  - `diverging`: The hand drifted more than max_drift (25 mm / 10 deg) from where the loop started, or a step was refused.
  - `no_frame`: The camera gave no RGB or move-classifier.predict raised; nothing was moved by this decision.

### disassembling-a-brick-by-twist-lever

execute · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/disassembling-a-brick-by-twist-lever/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation/blob/9bc91654e708e4d8428ecfa76840103cacb2a19a/src/ros_nodes/lego_task_planning_node.cpp)

Removes a seated LEGO or DUPLO brick with a parallel gripper and no force sensor - ICL's disassembly twist (grip, seat the grip, rotate 14 deg about one bottom edge so the stud rows let go one at a time, lift 15 mm with the twist kept), the edge being the one under a finger, commanded in small steps and read back so tracking lag (or joint effort) tells a brick that holds from one that lets go; with a straight pull and a wiggle (alternate small levers about both edges) as strategies and fallbacks, a regrasp after a slip, and the wrist depth camera reading the seat and its neighbouring cells before and after. Use when a brick must come off a baseplate or another brick at a cell known from the stud grid (computing-brick-pose-from-stud-grid's seat).

- **In**: `seat_position` Vec3
- **Out**: `strategy` str, `attempts` int, `attempt_log` str, `pivot_side` str, `twist_deg` float, `peak_lag` float, `releases` str, `release_count` int, `seat_lag` float, `holding` bool, `top_height_change` float, `neighbour_change` float, `samples` int, `verified_by` str
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`disassemble_brick`](skills/disassembling-a-brick-by-twist-lever/scripts/disassemble_brick.py)
- **Exits**:
  - `removed`: The hand lifted free, the jaws read at the brick's width, and the wrist depth reads the seat more than half a brick height under the brick's top (and less than 1.5) with no neighbouring cell moved; the brick is in the hand at the clearance height.
  - `stuck`: Every strategy tried stalled - for three steps in a row the hand stood lag_threshold further behind its command than in free motion (or a joint effort jumped) and advanced less than half its step, or the lever's tilt stood 4 deg short; the hand went back to the grasp, opened and rose, the brick left on the seat.
  - `slipped`: The last strategy lifted free but the jaws closed past the brick's width, or the wrist depth still reads the brick on its seat (regrasped and the next strategy tried while any were left); the jaws are open over the seat. A load that cleared during that attempt is named in detail as the grip letting go, not counted as a release.
  - `neighbour_lifted`: The seat reads more than 1.5 brick heights low (the brick under it came along) or a neighbouring cell's stud top moved more than max(half a stud height, 2 mm) between the views before and after; still holding.
  - `missed`: Before anything was gripped - the wrist depth finds no brick top at the seat, the open jaws stopped 3 mm or more over the grasp, or the jaws closed past the brick's width.
  - `unverified`: Lifted free, but the seat was not read (verify false, or no wrist camera / reachable view / 3 depth readings); detail says whether the jaws hold something.
  - `unreachable`: The grasp, or every strategy's motion, failed the IK preview, or a move was refused.

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

### grasping-by-wrist-camera-servoing

execute · robots: any · tasks: pick-place, sorting, jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-by-wrist-camera-servoing/SKILL.md) · [upstream](https://github.com/hello-robot/stretch_ai/blob/6272b28e3ff7be482af3b427a4551d1772ebf7d7/src/stretch/agent/operations/grasp_object.py)

Grasps a small part by closed-loop eye-in-hand servoing - from a pregrasp standoff above a coarse target, it segments the part in every wrist RGB-D frame (sam3.segment_text), takes the centre of the mask's top-face points back-projected with the wrist depth, steps the hand sideways (capped, IK-previewed robot.go_to_pose_cartesian steps) to bring that point under the tool centre, descends while it stays centred, and closes once the tool centre reaches the grasp height with the part under it; a lost track, a stalled arm or the step cap fall back to an open-loop grasp at the last point seen. Use when an overhead estimate is a few millimetres off and that is too much - LEGO bricks, dominoes, jigsaw pieces or 20-30 mm parts picked with the jaws pre-shaped close to the part's width - and the arm carries a wrist camera.

- **In**: `object_name` str
- **Out**: `grasp_point` Vec3, `coarse_point` Vec3, `mode` str, `correction` float, `final_error` float, `grip_width` float, `held_width` float
- **Calls**: [`sam3`](#sam3); connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`servo_grasp`](skills/grasping-by-wrist-camera-servoing/scripts/servo_grasp.py)
- **Exits**:
  - `grasped`: The jaws closed on something (the gap stayed over fully closed + 3 mm) and still held it after a lift of lift_height; `mode` says whether the servo closed (`servo`), the open-loop fallback did (`open_loop_fallback`, `fallback_reason` says why) or the skill ran open-loop (`open_loop`, servo=false). The hand stays lifted with the part.
  - `missed`: The jaws closed on nothing, or the part slipped out in the lift; the jaws are opened and the hand lifted clear.
  - `lost_target`: There is no wrist camera, the segmenter never found the part near the coarse estimate in MAX_LOST frames from the pregrasp view, or (fallback=false) it lost the part mid-servo; the hand stays where it was, jaws open.
  - `unreachable`: No top-down grasp rotation, no pregrasp view passed the IK preview and the clearance floor or arrived within 15 mm, or a move of the open-loop grasp was refused; `detail` lists each attempt.

### grasping-container-rims

execute · robots: any · tasks: pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-container-rims/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/tree/main/roborsi/embodied/skills/base/grasp_rim/robotwin)

Grasps a thin-walled open container (box, bin, tray, cup, bowl) by its rim - picks a wall midpoint the arm can reach, turns the open jaws across the wall (one finger inside the mouth, one outside), hovers above, comes straight down until the pads straddle the wall below the rim top, closes, checks the jaw gap reads about the wall's thickness (not empty, not wider), and lifts a little; tries the other walls when one misses. Use when an open container must be picked or held and a body grasp would close on the empty inside (or the container is wider than the open hand), with its oriented box from a perception skill and the hand empty.

- **In**: `container_obb` OrientedBoundingBox
- **Out**: `side` string, `grasp_position` Vec3, `close_heading_deg` float, `jaw_error_deg` float, `wrist_turn_deg` float, `grip_width` float, `held_width` float, `wall_thickness` float, `lifted` float, `attempts` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`grasp_rim`](skills/grasping-container-rims/scripts/grasp_rim.py)
- **Exits**:
  - `grasped`: The jaws closed across a wall with the gap above empty_margin (and above half the wall thickness when it is given) and at most wall_thickness + full_margin (max_wall when the thickness is unknown), and still did after the lift; the hand holds the rim. `lifted` says how far the hand rose (short when the container is heavy or fixed; `detail` says so).
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

### grasping-with-feedback

execute · robots: yam_i2rt · tasks: grasp, pick-place · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/grasping-with-feedback/SKILL.md)

Approaches a supplied grasp pose, closes the gripper and checks available grasp evidence through a configured workcell adapter. Use when an agent needs a local grasp with explicit feedback rather than an unchecked gripper command.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/grasping-with-feedback/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### inserting-a-held-peg-top-down

execute · robots: any · tasks: insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/inserting-a-held-peg-top-down/SKILL.md) · [upstream](composed from hub skills inserting-a-peg-with-pose-refinement (ManiSkill, Apache-2.0), searching-insertion-spiral, pressing-in-with-guarded-push (o2ac, BSD-3-Clause) and correcting-insertion-offset-from-port-opening (Apache-2.0); new code is this repository's)

Inserts a peg that one arm holds top-down (tip below the fingers) into a perceived hole straight down - aligns the tip over the mouth from the hand read back, refines from the wrist camera with the tip and the mouth in the same depth view until the lateral error is under half the clearance, descends with stop-on-contact, runs a bounded spiral (and, with a chamfer, a tilt-then-upright try) if blocked at the mouth, pushes to the target depth with a tracking-lag limit, releases, backs out and probes the peg's top with the closed fingers to tell a tip at depth from a peg that slid up in the grip. Use after a top-down regrasp (handing-over-a-peg-to-a-top-down-grasp gives `held`) when the hole's mouth and depth are perceived (`hole`), on an arm with no force sensor (YAM).

- **In**: -
- **Out**: `stage` string, `refined_by` string, `lateral_error` float, `camera_correction` float, `search` string, `probes` int, `contact_depth` float, `hand_depth` float, `tip_depth` float, `target_depth` float, `slip` float, `peak_lag` float, `released` bool, `tip_in_tcp` Se3Pose, `hole_position` Vec3, `elapsed_s` float
- **Calls**: connector `robot.close_gripper`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`insert_held_peg`](skills/inserting-a-held-peg-top-down/scripts/insert_held_peg.py)
- **Exits**:
  - `inserted`: The hand pushed the tip, as held, to the target depth, released and backed out, and the closed fingers found the peg's top within slip_tolerance (2 mm) of where a tip at that depth puts it; tip_depth is from the probe.
  - `jammed`: The tip entered the hole (at least enter_depth, 4 mm) but the lag-limited push stopped short of the target depth; the hand released the peg where it stuck and backed out.
  - `missed_hole`: The first contact was above the mouth plane (something over the hole, or the hole estimate is high), or the tip was blocked at the mouth and the spiral (and tilt-then-upright with a chamfer) did not enter; the hand is back over the mouth, still holding.
  - `slipped_in_grip`: The hand reached the target depth but the probe found the peg's top more than slip_tolerance higher than expected - the peg slid up the fingers and the tip is short (tip_depth); the hand is open and backed out.
  - `unverified`: The hand reached the depth but the top probe could not tell - no peg length, release or probe turned off, the probe start out of reach, or no top within 12 mm of where it should be (the peg fell over, or went further than the hand); open and backed out unless release is false.
  - `unreachable`: A move was refused or out of reach by the IK preview, the hand's pose could not be read, the hole's axis is more than 30 deg off straight down, or a sibling skill is not installed (raised; on_error).

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

### inserting-mid-air-between-two-held-parts

execute · robots: any, 2 arms · tasks: insertion, bimanual · MIT · v0.1.0 · [`SKILL.md`](skills/inserting-mid-air-between-two-held-parts/SKILL.md) · [upstream](https://github.com/tonyzhaozh/act/blob/main/scripted_policy.py)

Inserts the part one arm holds into (or presses it onto) the part the other arm holds, both in the air - ACT's scripted bimanual insertion (the two parts brought to a shared meet point facing each other on one axis, a stand-off apart, then the gap closed along that axis) with the parts' in-hand offsets in place of simulator poses; the moving tip is aligned across the axis on the holder's read-back entry before anything touches, then stepped in while contact is felt on both arms without a force sensor (the pushing hand's tracking lag beyond its free-motion lag, the holding hand pushed off its pose, joint efforts where published). Use when a bimanual station must join two parts that are each held by one arm - a peg into a socket, a pin into a bore, a DUPLO brick onto a held sub-assembly - with both parts' contact points measured in the hands (registering-held-objects or perception) and both arms addressable by arm_id.

- **In**: `moving_tip` Vec3, `receiving_entry` Vec3
- **Out**: `depth` float, `lateral_error` float, `axis_angle_deg` float, `probes_resisted` int, `lag_over_baseline` float, `holder_shift` float, `contact_signal` str, `steps` int, `roll_turns` int, `efforts_available` bool, `entry_position` Vec3, `tip_position` Vec3, `mover_pose` Se3Pose, `holder_pose` Se3Pose
- **Calls**: connector `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`
- **Scripts**: [`insert_mid_air`](skills/inserting-mid-air-between-two-held-parts/scripts/insert_mid_air.py)
- **Exits**:
  - `inserted`: The moving tip, as the two hands' read-back poses and the in-hand offsets place it, went at least insert_depth less depth_tolerance past the holder's entry along the axis, at least one of four probes across the axis (half the clearance plus 1 mm each way) was stopped or carried the holder along (verify, default on), and both jaws still read a hold (with release=mover, the mover then opened and backed out and the holder's jaws still hold). Felt by the hands, not seen - a part turned in its jaws since it was measured can still read as inserted (simulation check).
  - `misaligned`: Nothing went in - the tip could not be brought within half the clearance of the entry's line (after three corrections), the two parts' axes (each turned with its hand since it was measured) stood more than angle_tolerance_deg apart, or the push stopped (lag, holder pushed, effort) with the tip less than 2 mm past the entry (the rim); the mover backed off to its stand-off, both still hold their parts.
  - `stalled`: The push stopped with the tip past the entry but short of insert_depth less depth_tolerance (a jam, a burr, a tilted part); the mover backed off 3 mm so it stops pushing, both still hold. Also routed when the mover's jaws did not open for release=mover.
  - `slipped`: A hand's jaws read at the stop - before anything moved (nothing held), or after the push or the release (a part fell out); or the hands went the whole depth but no probe across the axis met a wall - the moving part sat on the receiving part's outside face and slid back along its jaw pads, which neither arm feels (both hands still hold, the parts are not joined).
  - `unreachable`: The two arm ids are equal or the connector reports no separate second arm, the holder could not reach the meet point or the mover its stand-off (free moves read back beyond reach_tolerance or refused), a wrist roll could not be turned back, a step was refused, or signal=effort with no efforts published; detail says which and what each hand still holds.

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

### moving-along-joint-path

execute · robots: yam_i2rt · tasks: motion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/moving-along-joint-path/SKILL.md)

Moves the arm through a supplied joint-space path on a backend that declares its joints and reports measured joint positions, checking limits and step size before motion and measured arrival after. Use when a validated q_path (e.g. a reorient plan checked for IK jumps) must run as planned.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/moving-along-joint-path/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### moving-through-waypoints

execute · robots: yam_i2rt · tasks: motion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/moving-through-waypoints/SKILL.md)

Moves the measured tool pose through a bounded list of waypoints on the configured YAM or Isaac workcell adapter, verifying arrival at each, and optionally that a held object follows the TCP. Use when an agent must approach, descend or transport a held piece along an explicit path.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/moving-through-waypoints/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

### moving-to-pose

execute · robots: yam_i2rt · tasks: motion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/moving-to-pose/SKILL.md)

Moves the measured tool pose through the configured YAM or Isaac workcell adapter. Use when an agent has a target pose and needs bounded motion with measured arrival verification.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/moving-to-pose/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### picking-upright-dominoes-from-a-loading-rack

execute · robots: any · tasks: pick-place, domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/picking-upright-dominoes-from-a-loading-rack/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/pick.py)

Takes the next domino out of a slotted loading rack, where tiles stand one per slot at a known pitch, instead of standing up each tile from lying. It reads every slot from the overhead depth as full, empty (only the walls' tops) or jammed (a top that is neither) and chooses the full slot nearest the hand. It picks top-down with the jaws across the tile's width at a height that keeps the fingertips over the rack's walls, lifts straight up until the tile clears them, and reads the jaw gap; a lift held short with the tile in the jaws is a jammed tile. Returns the held tile's grasp height for placing-an-upright-domino-beside-neighbours (tile None). Use when a domino build is fed from a rack filled by hand or by the other arm, as its supply step.

- **In**: `table_z` float
- **Out**: `slot_index` int, `picked` bool, `grasp_height` float, `grip_width` float, `jaw_error_deg` float, `wrist_turn_deg` float, `tile` Se3Pose, `full_count` int, `remaining` int, `lift_shortfall` float
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`pick_from_rack`](skills/picking-upright-dominoes-from-a-loading-rack/scripts/pick_from_rack.py)
- **Exits**:
  - `picked`: A tile was taken from `slot_index`, lifted straight out and carried to transit height with the jaws reading its width; `grasp_height` is the measured tool-centre height over its bottom (pass it to placing-an-upright-domino-beside-neighbours with tile None), `tile` where it stood, `slot_states` every slot as read, `remaining` the full slots left.
  - `slot_empty`: The requested `slot_index` is not full (empty or jammed); nothing moved.
  - `rack_empty`: No slot reads full; `detail` lists the jammed and unseen ones (clear a jam by hand or with another skill).
  - `grasp_failed`: The descent into the slot stopped high (a fingertip on the tile's top or a wall), the jaws closed on nothing or not across the tile, or the tile slipped out during the lift or the rise; the hand is open over the rack.
  - `stuck`: The straight lift stopped more than 6 mm short with the tile still in the jaws (it is jammed in its slot); the jaws reopened and the hand rose.
  - `hand_full`: The jaws already read a tile's width before anything moved (a tile from an earlier step still in the hand); nothing moved - place it or set it down first.
  - `not_seen`: No full slot was seen and some slots were not (no camera saw them whole and unoccluded, also after `look_pose`), and no `occupied` list covers them.
  - `unreachable`: Nothing was picked - every full slot has a standing neighbour where an open finger goes (a rack pitch under 19.8 mm for a 7.5 mm tile), no grasp height clears the walls and keeps 10 mm of tile above the fingertips, the grasp failed the IK preview for both wrist solutions, a move was refused, or the jaws stayed turned off the tile; also a refused rise after the pick (still holding).
  - `invalid_input`: A rack without slots, a pitch or slot count that is not positive, a wall as high as the tile, an unknown order, an `occupied` list of the wrong length or a `slot_index` past the rack.

### placing-an-upright-domino-beside-neighbours

execute · robots: any · tasks: pick-place, domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/placing-an-upright-domino-beside-neighbours/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/domino/options.py)

Picks a standing domino top-down and sets it upright at a slot in a row without touching the tiles already standing there - the jaws close across the tile's width so the fingers stay beside the row, open only a few millimetres wider than the tile, the hand crosses above every standing tile, comes in along the row from its free end, descends vertically, opens just enough and retreats straight up; the jaw heading is read back and finished on the last joint. Then checks from depth that the tile stands at the slot and that every neighbour still stands. Use when building a domino line or curve one tile at a time, with the slot poses planned and the tile to pick standing upright.

- **In**: `slot` Se3Pose, `tile` Se3Pose, `table_z` float
- **Out**: `released` bool, `picked` bool, `grasp_height` float, `grip_width` float, `held_width` float, `jaw_error_deg` float, `wrist_turn_deg` float, `release_position` Vec3, `verified` bool, `tile_top` float, `placement_offset` float, `neighbours_standing` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`place_domino`](skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py)
- **Exits**:
  - `placed`: Released at the slot and retreated, and the depth check sees a tile standing there (its top within 10 mm of the tile's height and within 8 mm of the slot) with every seen neighbour standing; with verify=false, what the hand did only.
  - `knocked_neighbour`: Released, but a neighbour that stood before the place (or was not seen then) no longer stands at its pose; `neighbours_knocked` lists which, `detail` says whether the placed tile stands.
  - `fell`: Released, the neighbours stand, but the placed tile does not stand at the slot - its top is low (lying, leaning hard), more than 8 mm off the slot (leaning, misplaced), or nothing is there.
  - `unverified`: Released and retreated, but no camera saw the slot whole and unoccluded (the hand in the way, no depth, outside every image).
  - `unreachable`: Nothing was released - the slot is so close to a neighbour that an open finger would touch it, the grasp height is outside what the fingertips can do, a grasp or release pose failed the IK preview, a move was refused, the jaws stayed turned off the tile, or the descent to the slot stopped high (lifted back); `detail` says whether the tile is still held. Also a refused retreat after the release (`released` true).
  - `not_holding`: The grasp closed on nothing or not across the tile, the descent to the tile stopped on it, or the jaws read empty over the slot (dropped in transit); nothing was placed.

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

### pressing-a-brick-onto-studs

execute · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/pressing-a-brick-onto-studs/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/BrickSim/blob/cbd5de3238e6ac12f44ef699e1507a9f16bdafc3/python/bricksim/envs/assemble_brick/expert.py)

Presses a held LEGO or DUPLO brick onto the studs under it and checks by height that it seated - BrickSim's scripted press (hover 4 cm over the seat, fold the lateral error in, press until the command leads the stopped hand by 3 mm) with tracking lag and joint effort standing in for a force sensor, and, as strategy 'tilt', ICL's edge-first placement (the brick tilted about its leading bottom edge, that edge lowered beside the stud row and pulled back until its wall catches the row, then rotated level and pressed). After the release the wrist depth camera reads the brick's top (a brick resting on the stud tops sits a whole stud height high (1.7 mm LEGO, 4.5 mm DUPLO) or tilted). Use when a grasped brick must be fitted onto a baseplate or another brick at a seat known from perception, with stud pitch and heights given or taken from the 'lego' / 'duplo' presets.

- **In**: `seat_position` Vec3
- **Out**: `released` bool, `seat_height_error` float, `seat_tilt` float, `contact_height` float, `lead_correction` float, `lateral_correction` float, `yaw_turn` float, `samples` int, `verified_by` str, `seated_position` Vec3
- **Calls**: connector `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_cartesian_until_contact`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`press_brick`](skills/pressing-a-brick-onto-studs/scripts/press_brick.py)
- **Exits**:
  - `seated`: Released, and the wrist depth reads the brick's top within 0.35 stud heights of the seated height (0.6 mm LEGO, 1.6 mm DUPLO) with a spread under half a stud height.
  - `not_seated`: Released, and the top reads higher than that but under 0.6 stud heights and flat - partly pressed (a real clutch the press could not push through).
  - `misaligned`: The held brick is more than yaw_tolerance_deg off the seat's heading (nothing moved); or the press, the tilted edge or the hook stopped on the stud tops / met no stud row / the rotation to level was held up (kept and lifted back, released false); or, released, the top reads 0.6 stud heights or more high, or tilted.
  - `not_holding`: The jaws read at their stop before anything moved, or after the release the wrist depth finds no brick top at the seat.
  - `unverified`: Released, but the height was not read (no wrist camera with a pose, no reachable view, fewer than 3 depth readings on the top, or verify false); detail says how far the press went.
  - `unreachable`: The seated (or tilted) hand pose failed the IK preview, or a move was refused; detail says which, and whether the brick is still held.

### pressing-in-with-guarded-push

execute · robots: any · tasks: insertion, assembly · BSD-3-Clause · v0.1.0 · [`SKILL.md`](skills/pressing-in-with-guarded-push/SKILL.md) · [upstream](https://github.com/o2ac/o2ac-ur/blob/65ddc5496879799038cefbad061380b7d0300d73/catkin_ws/src/o2ac_routines/src/o2ac_routines/ur_force_control.py)

Pushes a held part along an axis (straight down by default) in small steps with a bounded push and a bounded travel, and says how it ended - travel or target depth reached (seated), the push collapsing as the hand jumps forward (snap_detected, a snap-fit or clutch giving), or the push limit reached short of the goal - o2ac's linear_push / do_insertion with tracking lag, and joint effort where published, standing in for the force sensor a YAM does not have. Unloads the push at the end. Use when a held part already sits on or in its seat (a brick on studs, a peg in a hole's mouth, a domino in a slot, a snap-fit lid) and must be pressed home by a few millimetres without crushing it, and a camera cannot tell whether it went in.

- **In**: -
- **Out**: `travelled` float, `contact_travel` float, `peak_lag` float, `final_lag` float, `remaining` float, `steps` int, `signal` str, `efforts_available` bool, `relieved` bool, `stop_position` Vec3
- **Calls**: connector `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.solve_ik`
- **Scripts**: [`guarded_push`](skills/pressing-in-with-guarded-push/scripts/guarded_push.py)
- **Exits**:
  - `seated`: The hand travelled max_translation (8 mm) along the axis, or reached the target's depth, before the push limit; or, with a target, hit the limit within goal_tolerance_if_lockup of it (o2ac's relaxed lock-up). contact_travel is -1 when nothing resisted on the way.
  - `snap_detected`: After the lag had built at least snap_rise (2 mm) over its free baseline it fell by snap_drop (1.5 mm) in one step while the hand jumped as far forward - a snap-fit or clutch gave; the push stopped there (stop_on_snap).
  - `force_limit_no_travel`: The lag over the free baseline passed max_lag (6 mm), or a joint effort residual passed effort_threshold, short of the goal; the push was unloaded. `signal` says which.
  - `blocked`: The end of the push is out of reach (IK preview) or under floor_z, a move was refused, the floor ended the push, or signal=effort without published efforts; `detail` says which.

### pressing-with-contact-limit

execute · robots: yam_i2rt · tasks: assembly · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/pressing-with-contact-limit/SKILL.md)

Runs one bounded contact motion along an axis that the controller stops on a force threshold or travel limit, on a backend that declares a calibrated contact-force signal and a compliant controller. Use when a graph must press or seat a part with a measured force limit; position-only pressing is refused.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/pressing-with-contact-limit/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

### pulling-drawer

execute · robots: yam_i2rt · tasks: drawer · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/pulling-drawer/SKILL.md)

Executes a bounded pull along a drawer axis using handle-grasp and opening evidence. Use when an agent has a drawer handle pose and a prismatic opening direction in a configured workcell.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/pulling-drawer/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### pushing-on-plane

execute · robots: yam_i2rt · tasks: pushing · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/pushing-on-plane/SKILL.md)

Executes a bounded planar pushing segment and checks measured target displacement. Use when an agent has a contact pose and a push direction in a configured single-arm workcell.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/pushing-on-plane/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### refilling-a-domino-loading-rack

execute · robots: any · tasks: pick-place, domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/refilling-a-domino-loading-rack/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/place.py)

Puts a standing domino back into an empty slot of a slotted loading rack, so picking-upright-dominoes-from-a-loading-rack always has tiles to take. It reads every slot from the overhead depth (full, empty, jammed) and chooses an empty one whose neighbours leave room for the open fingers. It picks the tile top-down across its width (or takes the tile already in the hand) at a height that keeps the fingertips over the rack's walls, and squares the jaws to the slot. It lowers the tile from just over the walls into the slot, trying small offsets along the row when the descent stops on a wall, releases it, and reads the slot again. Use when one arm, or a person, keeps the rack filled while the other arm builds.

- **In**: `table_z` float
- **Out**: `slot_index` int, `picked` bool, `released` bool, `grasp_height` float, `grip_width` float, `jaw_error_deg` float, `wrist_turn_deg` float, `insert_offset_mm` float, `align_error_mm` float, `tile_tilt_deg` float, `attempts` int, `slot_state_after` str, `empty_count` int
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`refill_rack`](skills/refilling-a-domino-loading-rack/scripts/refill_rack.py)
- **Exits**:
  - `refilled`: The tile went into slot `slot_index` (at `insert_offset_mm` along the row, after `attempts` descents), was released and the hand rose, and the slot reads full from depth (with verify false, what the hand did only).
  - `rack_full`: No slot reads empty (or the requested `slot_index` is not empty); `detail` lists jammed slots; nothing moved.
  - `jammed`: Every offset's descent into the slot stopped more than 3 mm high on a wall; the hand lifted back to transit, still holding the tile.
  - `fell`: Released, but the slot reads empty or jammed after (the tile tipped out, or lies across the slot).
  - `unverified`: Released, but no camera saw the slot whole after the release (give look_pose).
  - `not_holding`: The pick closed on nothing or not across the tile, its descent stopped on the tile, the tile was dropped in transit, or (tile None) the hand holds nothing.
  - `hand_full`: A tile to pick was given but the jaws already read a tile's width; nothing moved.
  - `not_seen`: No camera saw the rack, or no empty slot among those seen while some were not.
  - `unreachable`: Nothing was put in - every empty slot has a standing neighbour where an open finger goes, no grasp height clears the walls, an IK preview or a move failed, or the jaws stayed turned; `detail` says whether the tile is still held.
  - `invalid_input`: A rack without slots, a wall as high as the tile, an unknown order, a slot_index past the rack, or no tile and no grasp_height (also the route of a raised error).

### rotating-a-held-piece-within-wrist-limits

execute · robots: any · tasks: jigsaw, reorientation · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/rotating-a-held-piece-within-wrist-limits/SKILL.md) · [upstream](https://github.com/wanweiwei07/wrs/tree/main/0000_examples/x6wg2_regrasp)

Turns a held flat part (a jigsaw piece) about the vertical by the angle its slot needs while the wrist's last joint stays inside its range (+-2.09 rad on the YAM, 240 degrees - less than a full turn) - takes the shorter or the longer way round when one fits, else sets the piece down at a clear spot it finds in the cameras' depth, turns the empty hand half a turn so the jaws close on the same two contact points swapped, takes it again and finishes the turn (WRS-style regrasp, reduced to one joint and a table). Learns which way the joint turns the hand from a small first held step, and corrects each turn on the hand's yaw read back. Can also only plan, e.g. to grip with the jaws swapped at the pick so no regrasp is needed. Use when a jigsaw piece (or any flat part) is held top-down and must be turned before it is placed, after the turn is known from localizing-a-piece-in-a-reference-image or matching-a-piece-to-placed-neighbours.

- **In**: `turn_deg` float
- **Out**: `turned_deg` float, `regrasps` int, `jaws_swapped` bool, `joint6_final` float
- **Calls**: connector `robot.close_gripper`, `robot.describe_arm`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`
- **Scripts**: [`rotate_piece`](skills/rotating-a-held-piece-within-wrist-limits/scripts/rotate_piece.py)
- **Exits**:
  - `turned`: The hand turned the piece by the asked angle (read back within 0.5 deg) without letting go; joint 6 stayed 3 deg inside its range.
  - `turned_with_regrasp`: No direct turn fitted the range, so the piece was set down, the empty hand turned half a turn, took it again (jaws swapped, same contacts) and finished the turn; `jaws_swapped` is true.
  - `planned`: plan_only was set; `plan` says direct or regrasp and the joint moves, nothing moved.
  - `dropped`: After a held turn the gripper reads under 0.78 of its reading with the piece in the jaws (closed on nothing), so the piece fell during the turn; turned_deg says how far the hand had turned.
  - `piece_lost`: After a regrasp the jaws closed past half the given grip width (nothing in them).
  - `unreachable`: No plan fits the range, a regrasp was needed without grasp_z, the joints could not be read, joint 6 did not move the hand, or a move was refused.

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

### seating-a-piece-by-wiggle-search

execute · robots: any · tasks: jigsaw, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/seating-a-piece-by-wiggle-search/SKILL.md) · [upstream](https://github.com/markroberyoutube/puzzle_bot/blob/main/src/application.py)

Seats a held jigsaw piece in its slot among already-placed neighbours when the arm is less accurate than the slot's clearance - lowers the piece over the slot until it stops (tracking lag, or joint effort where published), reads from the stop height whether it dropped onto the mat or rests on the neighbours (a piece thickness higher), and if it rests, lifts and re-lowers it over small x / y / yaw offsets taken most likely first, finishing with a scoot-and-wiggle at the best spot (Mark Rober's puzzle robot's routine, reimplemented without its spring and encoder). Releases a dropped piece and reads its top with the wrist depth camera to confirm it is flush. Options - tilted tab-first, or set down short and pushed in. Use when a jigsaw piece (or any thin flat part with a matching cut-out) is held over its slot with the slot pose known to a millimetre or two and a few degrees, the neighbours fixed around it, and fingers clear of the neighbours.

- **In**: `slot_position` Vec3
- **Out**: `released` bool, `probes` int, `moves` int, `elapsed_s` float, `stop_height` float, `best_height` float, `resting_height` float, `offset_found` Vec3, `yaw_found_deg` float, `seated_position` Vec3, `flush_error` float, `flush_spread` float, `samples` int, `verified_by` str
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`seat_piece`](skills/seating-a-piece-by-wiggle-search/scripts/seat_piece.py)
- **Exits**:
  - `seated`: A probe stopped with the piece's bottom within a quarter of its thickness of the mat (it dropped), the jaws opened, and the wrist depth reads the piece's top within a quarter of the thickness of flush with a spread under the same.
  - `not_seated`: No probe or scoot dropped the piece (kept and lifted back, released false); or, released, the wrist depth reads its top high or tilted (resting on a neighbour).
  - `jammed`: The piece went partly in (best stop between a quarter and 0.6 of its thickness over the mat) and the scoot did not finish it (kept, lifted back); or the hand could not lift the piece back out.
  - `piece_lost`: The jaws read at their stop before anything moved, a descent went a whole thickness under the mat without meeting anything, or after the release the wrist depth finds no piece top at the slot.
  - `unverified`: Released after a drop, but the flush check could not run (verify false, no wrist camera, no view reached).
  - `unreachable`: The seated hand pose fails the IK preview, the hover or a turn was not reached, or a move was refused.

### setting-gripper-with-feedback

execute · robots: yam_i2rt · tasks: grasp, release · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/setting-gripper-with-feedback/SKILL.md)

Opens, closes or releases the gripper through the configured YAM or Isaac workcell adapter and checks measured holding or width feedback. Use when a graph must let go of a placed piece or set the jaw width without moving the arm.

- **In**: `request_json` string
- **Out**: `result_json` string, `status` string
- **Calls**: [`workcell`](#workcell)
- **Scripts**: [`main`](skills/setting-gripper-with-feedback/scripts/main.py)
- **Note**: Adapter-backed implementation; station setup and task-specific validation are required. No sim/real verification is claimed.
- **Exits**:
  - `succeeded`: The capability-specific postcondition is supported by the returned measurements.
  - `failed`: The attempted operation or tracking update did not meet its required condition.
  - `unverified`: Available evidence is insufficient to establish the requested physical postcondition.
  - `rejected`: The request, station configuration or required backend capability is not accepted.

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

### standing-a-lying-domino-by-reorient-and-regrasp

execute · robots: any, 2 arms · tasks: domino, reorientation · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/standing-a-lying-domino-by-reorient-and-regrasp/SKILL.md) · [upstream](https://github.com/Zhu-Chenyu/Franka-Domino-Project/blob/main/place_dominoes/place_dominoes/place_dominoes.py)

Stands a flat-lying domino upright so a top-down pick can take it. Default strategy (handover, two arms) - this arm pinches the lying tile across its width, carries it flat to a point in the air between the arms, the helper arm pinches its other end across the thickness with a level approach and rolls a quarter turn on its last joint so the tile hangs upright, this arm takes the top end top-down across the broad faces, the helper lets go, and this arm sets the tile on its end at the staging spot; a depth check then confirms a tile stands there. Single-arm strategies (table_pivot, pose_pitch, hinge_tip) are kept but stood no tile in a YAM station twin. Use when a domino lies flat (from reading-domino-pips-and-orientation or a 6D pose) and must stand before placing-an-upright-domino-beside-neighbours picks it into a row.

- **In**: `table_z` float, `tile_center` Vec3, `long_axis` Vec3
- **Out**: `strategy` str, `pitch_joint` int, `picked` bool, `released` bool, `grasp_height` float, `grip_width` float, `facing_deg` float, `yaw_turn_deg` float, `pitch_turn_deg` float, `roll_turn_deg` float, `tilt_at_release_deg` float, `release_position` Vec3, `standing_tile` Se3Pose, `verified` bool, `tile_top` float, `standing_offset` float
- **Calls**: connector `robot.close_gripper`, `robot.describe_arm`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.move_to_joints`, `robot.open_gripper`, `robot.set_grip`, `robot.solve_ik`
- **Scripts**: [`stand_domino`](skills/standing-a-lying-domino-by-reorient-and-regrasp/scripts/stand_domino.py)
- **Exits**:
  - `standing`: Released and retreated, and the depth check sees a tile's top at standing height (within 10 mm of the tile's height) at or near the spot; with verify=false, what the hands did only. `standing_tile` is the pose to hand to placing-an-upright-domino-beside-neighbours; when the top was seen more than 8 mm from the spot, `standing_tile` is moved to where it was seen and `detail` says so.
  - `fell`: Released, but the depth check sees no tile's top at standing height in the window around the spot (lying, on its side, or gone).
  - `unverified`: Released and retreated, but no camera saw the spot whole and unoccluded.
  - `ungraspable`: Nothing was lifted - the descent to the tile stopped high (a fingertip on it, or it is not where it was seen), the jaws closed on nothing or not across its width, or the grasp height leaves the fingertips over the tile.
  - `dropped`: The tile was picked, then the jaws read empty or not across it (after the lift, during a pitch, at the handover point, after the helper took it).
  - `unreachable`: A preview, move or turn was refused before the release - no handover point both arms reach, the helper's jaws not across the thickness, no room on a last joint, a chained IK step that would jump a joint by more than pitch_joint_jump, no spot where a single-arm strategy reaches the upright hand - or the retreat after the release. `detail` says which hand still holds the tile.

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

### triggering-a-domino-chain-by-push

execute · robots: any · tasks: domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/triggering-a-domino-chain-by-push/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/ground_truth_models/skill_factories/push.py)

Topples the first domino of a standing line so the chain runs - closes the hand with the blades across the tile, comes down behind the tile (against its facing) to a contact height on its upper half, pushes slowly along the facing a short computed stroke that carries the tile past its tipping angle without reaching the next tile, backs off along the stroke and rises, then checks from depth that the first slot no longer stands. Use after a domino line or curve is built (planning-a-domino-layout's slot 0 is the trigger tile), to start the cascade; hand the clip to verifying-a-domino-cascade-from-video to judge the whole chain.

- **In**: `slot` Se3Pose, `table_z` float
- **Out**: `pushed` bool, `contact_height` float, `tip_travel` float, `push_travel` float, `stroke_length` float, `stroke_time` float, `stroke_speed` float, `stroke_steps` int, `trigger_time` float, `push_direction` Vec3, `verified` bool, `first_standing` bool, `next_standing` bool
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_observation`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.grasp_frame`, `robot.solve_ik`
- **Scripts**: [`trigger_domino`](skills/triggering-a-domino-chain-by-push/scripts/trigger_domino.py)
- **Exits**:
  - `triggered`: The stroke ran, the hand backed off, rose and parked, and the depth check sees no tile standing at the slot (`next_standing` says whether the next tile still stands); with verify=false, the stroke carried the blade front at least the tipping travel past the tile's back face (what the hand did only).
  - `no_contact`: The tile still stands at the slot after the stroke - the stroke ended inside the servo's arrival tolerance but short of the tipping travel and the tile rocked back (`pushed` false), or the hand passed over or beside it, or it is not where the slot says (`pushed` true) - or nothing stood at the slot before the push (nothing moved then).
  - `blocked`: The descent behind the tile stopped high (something stands where the hand comes down; lifted back), or the stroke was held - it timed out or ended more than 5 mm short of its end - before the blade reached the tipping travel (a tile that does not give, or the arm at a limit); the hand backed off and rose.
  - `unverified`: The stroke ran, but no camera saw the slot whole and unoccluded after the push; `pushed` says whether the blade reached the tipping travel.
  - `unreachable`: Nothing was pushed - the contact height is outside its band, the stroke would reach the next tile (the line is too tight to push the first tile alone) or the hand would come down on the tile behind, the stroke's start or end failed the IK preview for both wrist solutions, or a move before the stroke was refused.

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

### auditing-a-domino-line-for-fallen-or-tilted-tiles

verify · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/auditing-a-domino-line-for-fallen-or-tilted-tiles/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/components/domino_component.py)

Checks a standing domino line or curve from one overhead depth frame, with no reference image, against its planned slots. For every slot it says whether the tile stands upright, is displaced (off its slot or turned), tilted, leaning on its neighbour, fallen or missing. It fits each tile's lean angle (forward or backward) jointly with its shift along the row, to the depth profile along its fall corridor (the cascade-video skill's rotated-box model), and its turn from its top face. A fallen or leaning neighbour's body is taken out of a tile's window, and neighbours are checked against each other physically. It also lists the runs of fallen tiles and the gaps a wave cannot cross. Routes line_ok, fix_needed or fallen. No motion. Use before triggering a line, after building or bumping it, or after a stopped cascade, to know which tiles to re-place.

- **In**: `table_z` float
- **Out**: `n_tiles` int, `n_upright` int, `first_fault` int, `camera` str
- **Calls**: connector `robot.get_observation`
- **Scripts**: [`audit_line`](skills/auditing-a-domino-line-for-fallen-or-tilted-tiles/scripts/audit_line.py)
- **Exits**:
  - `line_ok`: Every tile stands upright on its slot (lean under 5 deg, within 4 mm and 10 deg of it) and no gap between neighbours is wider than a wave crosses; trigger the line.
  - `fix_needed`: No tile has fallen or is missing, but some are displaced, tilted, leaning on a neighbour, or a gap between two standing tiles is over the critical gap (`wide_gaps`); `faults` lists them in line order with lean, shift and direction.
  - `fallen`: At least one tile has fallen (lean 60 deg or more, or past where its top meets the neighbour) or is missing; `fallen_runs` gives the runs of fallen and leaning tiles (a stopped or premature cascade), `faults` every faulty tile.
  - `not_seen`: No camera frame with depth saw the slots (none in the observation, out of the image, or a hand over them), or some slots stayed unjudged while the rest were fine; `states` says `unseen` / `unclear` for those.
  - `invalid_input`: No slots, a slot without a facing, or a tile size that is not height &gt; width &gt; thickness (also the route of a raised error).

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

### checking-pick-by-scene-diff

verify · robots: any · tasks: pick-place, sorting, jigsaw, lego, domino · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/checking-pick-by-scene-diff/SKILL.md) · [upstream](https://github.com/nssmd/RoboRSI/blob/90c8569705bf4004a4f79e34e650c87ab7f075d4/roborsi/embodied/skills/base/visual_diff/robotwin/policy.py)

Checks whether a pick (or a place, a push, any intended change) happened by showing a VLM one panel - the camera frame from before the step, the frame now, and a hot pixel-difference heatmap side by side - and asking whether anything changed and whether the expected change happened; when the graph knows where the object was (or should go), the pixels there must agree (changed, and emptied for a pick or filled for a place), else the verdict is uncertain. Routes changed_as_expected, changed_otherwise, unchanged, uncertain or no_verdict. Use when a pick (or place) has just run and the gripper width alone cannot tell held from empty (thin or small parts), so a source-side check is wanted, with a robot.get_observation captured before the step; camera-only, no motion.

- **In**: `before` Observation, `expected_change` string
- **Out**: `changed` bool, `matches_expectation` bool, `reason` string, `changed_fraction` float, `source_changed_fraction` float, `source_contrast_ratio` float
- **Calls**: [`vlm`](#vlm); connector `robot.get_observation`
- **Scripts**: [`check_scene_diff`](skills/checking-pick-by-scene-diff/scripts/check_scene_diff.py)
- **Exits**:
  - `changed_as_expected`: The VLM answered CHANGED yes and MATCH yes, and (when a source region is given) at least 15 % of its pixels changed and it emptied (vacated, contrast ratio &lt;= 0.85) or filled (filled, ratio &gt;= 1.5) as region_expect asks.
  - `changed_otherwise`: The VLM saw a change (CHANGED yes) but not the expected one (MATCH no) - e.g. the hand moved over the object, or another object moved.
  - `unchanged`: The VLM saw no change (CHANGED no, MATCH no) and the source region, if given, did not change either.
  - `uncertain`: The VLM and the pixels disagree - MATCH yes but the region did not change, did not empty or did not fill as expected, MATCH yes with CHANGED no, or CHANGED no while the region changed; re-observe (move the hand clear) before trusting either.
  - `no_verdict`: Nothing was judged - no camera has RGB in both observations, the frames differ in size, vlm.query raised, or the reply lacked a CHANGED or MATCH line.

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

### detecting-disturbed-pieces-by-snapshot-diff

verify · robots: any · tasks: jigsaw, pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/detecting-disturbed-pieces-by-snapshot-diff/SKILL.md) · [upstream](https://github.com/scikit-image/scikit-image/blob/e098ebbb684788df60b42e3bde112c014d0d9517/src/skimage/metrics/_structural_similarity.py)

Checks whether anything already placed was knocked since the last verified step - registers a new RGB(-D) snapshot onto the reference taken after that step (ORB + RANSAC homography refined by masked ECC on the static background; the camera may have moved a little), divides out lighting drift, maps colour/SSIM and depth change, and re-finds every known item (jigsaw pieces, placed bricks, standing dominoes - given as masks, pixel polygons, world boxes or grid cells) by a local rigid registration, reporting which moved, by how many mm and degrees, and how much their depth changed; changes no item explains become occluders (a hand or the arm, masked from its pose) or unexpected objects. Routes unchanged, disturbed, unexpected_object, occluded or unregistrable. Use when a place, press or push has just happened near items already placed and the next step will build on them; camera-only, no motion.

- **In**: `reference` Observation
- **Out**: `n_disturbed` int, `n_unexpected` int, `max_shift_mm` float, `camera_shift_px` float, `changed_fraction` float
- **Calls**: connector `robot.get_observation`
- **Scripts**: [`diff_snapshots`](skills/detecting-disturbed-pieces-by-snapshot-diff/scripts/diff_snapshots.py)
- **Exits**:
  - `disturbed`: At least one item not listed as expected moved (translated at least max(1 mm, 0.6 px at its scale), turned at least 2 degrees and 0.8 px at its rim, or changed depth by 5 mm, each standing out of the noise for its texture) or is lost (not found as it looked within 25 mm); `disturbed` lists them and `items` says by how much.
  - `unexpected_object`: No item moved, but a change inside the region that no item explains is at least 80 mm^2 (30 px without a scale) and does not enter from the frame edge or the arm; `unexpected` locates it (kind appeared / vacated with depth).
  - `occluded`: Nothing moved and nothing unexpected, but at least one item is less than 60 % visible (under the arm, a hand, an ignore mask, or outside the registered view); it could not be checked.
  - `unchanged`: Every judged item is where it was and nothing unexplained changed in the region.
  - `unregistrable`: The background does not correlate (median tile correlation under 0.8 after registration) or the camera moved more than 8 % of the frame diagonal; the views cannot be compared pixel by pixel - re-take the reference.

### flagging-odd-one-out-parts-by-mutual-scoring

verify · robots: any · tasks: domino, lego, jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/flagging-odd-one-out-parts-by-mutual-scoring/SKILL.md) · [upstream](https://github.com/HUST-SLOW/MuSc-V2/tree/b5e911dbc089/models/modules)

Finds the part that differs from its siblings in one frame - a fallen or turned domino in a line, a face-up tile among face-down ones, a standing one among lying ones, a wrong face - with no reference image, no before frame and no training. The parts (masks, boxes, or a sam3 text prompt) are cut out, turned to their long axis (the 180 deg turn chosen to match the others) and scored against each other three ways - appearance by MuSc's mutual patch scoring on DINOv2 (tools/musc), height maps by the same nearest-patch rule on the depth, and the boxes' orientation and footprint - and a robust z-score over the parts says which are odd and why. Use when many copies of one part were placed and one may be wrong - before a domino line is triggered, after a row of bricks or tiles is laid, or to pick the one tile that needs turning over.

- **In**: -
- **Out**: `n_parts` int, `camera_name` string, `appearance_scored` bool, `aligned_layout` bool
- **Calls**: [`musc`](#musc), [`sam3`](#sam3); connector `robot.get_observation`
- **Scripts**: [`flag`](skills/flagging-odd-one-out-parts-by-mutual-scoring/scripts/flag_odd_one_out.py)
- **Exits**:
  - `outliers_found`: At least one part is odd - its appearance score over z_threshold (3.5) robust z and 1.4x the median, its height-map score over 3.5 z and 2 mm, its box turned over 20 deg from the others' (when they are laid out aligned), or a side of its box 1.25x off the median at 3.5 z; outliers lists them and parts[i].why says which test.
  - `all_consistent`: Every part passed every test; detail gives the highest z.
  - `too_few_parts`: Under min_parts (4) parts were given or segmented, or no RGB frame.

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

### registering-assembled-board-to-reference

verify · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/registering-assembled-board-to-reference/SKILL.md) · [upstream](https://github.com/opencv/opencv/blob/4.x/modules/video/src/ecc.cpp)

Aligns a top-down view of the part-built jigsaw (overhead frame, rectified view or wrist mosaic) to the box picture and judges every placed cell - an initial homography from the board's corners (or SIFT + RANSAC) is refined by masked ECC coarse-to-fine (translation, affine, then homography) on the placed cells only, dropping cells that do not match and fitting again; then each placed cell is compared with its own cell of the picture by NCC over a small shift and over the turns its shape allows, and judged match, mismatch (with where it belongs instead), rotated, empty (mat colour), low_texture_match or unseen. Returns the homography, how far the content sits from the given corners (a slid assembly), and the lists of cells to fix. Use after a batch of placements, before building on a region, or before declaring the puzzle done, when the box picture is available; camera-only, a tenth of a second to a second on CPU.

- **In**: `camera` CameraFrame, `reference` string, `grid_cols` int, `grid_rows` int
- **Out**: `ecc_correlation` float, `correction_cells` float, `n_placed` int, `n_match` int, `n_mismatch` int, `n_rotated` int, `n_empty` int
- **Scripts**: [`register_board`](skills/registering-assembled-board-to-reference/scripts/register_board.py)
- **Exits**:
  - `registered`: The view aligned to the picture (final ECC correlation at least 0.7 and at least half the textured placed cells match their own place); `cells` judges every placed cell and `mismatched_cells`, `rotated_cells` and `empty_cells` list the ones to fix (all empty when the board is right).
  - `did_not_converge`: ECC diverged or ended under the gates from the given corners or homography (or moved them more than a cell) and from SIFT features, or there was no usable image; nothing is judged. Re-take the view, give better corners, or wait for more placed pieces.
  - `too_little_placed`: Fewer than 4 placed cells, or fewer than 3 of them textured; there is nothing to align on.

### simulating-a-domino-layout-before-building

verify · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/simulating-a-domino-layout-before-building/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/cascade_probe.py)

Checks a planned domino line or curve in a rigid-body simulation before the arm builds it. It rebuilds the slots (planning-a-domino-layout's `slots` or any slot poses) as real-size tiles on a table, replays the trigger push (triggering-a-domino-chain-by-push's stroke, the fingertips the only part of the robot that collides, as predicators' cascade probe does) and records when each tile falls (10 deg held for 3 samples). It then reruns the same layout with every slot jittered by the placing error and reports the fraction that still reaches the target and the pair that breaks most often. Routes cascades, fragile (the plan works but placing error breaks it), stalls (with the slot where the wave stops) or push_failed. A stop decided by a graze routes knife_edge. CPU only, about 50 ms per rollout of 12 tiles. No motion. Use after planning-a-domino-layout and before placing the first tile, or to choose a pitch or radius.

- **In**: -
- **Out**: `n_tiles` int, `target_index` int, `reached_target` bool, `stall_at` int, `n_fallen` int, `wave_speed` float, `push_travel` float, `first_fell` bool, `trials` int, `success_rate` float, `weakest_pair` int, `sim_time` float, `wall_time` float
- **Scripts**: [`simulate_layout`](skills/simulating-a-domino-layout-before-building/scripts/simulate_layout.py)
- **Exits**:
  - `cascades`: In the planned layout every tile up to the target fell, and at least `min_success` (0.9) of the `trials` rollouts with the slots jittered by the placing error did too. `topple_times` gives each tile's fall time and `wave_speed` the settled wave's speed along the path.
  - `fragile`: The planned layout reaches the target, but fewer than `min_success` of the jittered rollouts do; `weakest_pair` is the pair (k, k + 1) where they broke most often and `trial_stalls` lists where each stopped. Tighten the pitch or widen the radius there.
  - `stalls`: In the planned layout the wave stopped; `stall_at` is the first slot up to the target still standing (its predecessor fell), `fallen` says which fell, and the falling tile came no closer to it than 2 mm short (`pair_clearance`), a clear miss.
  - `knife_edge`: The wave stopped at `stall_at`, but the falling tile's closest approach to it was a graze (from 1 mm into it to 2 mm short of it, `pair_clearance`). A millimetre of placement or of contact model decides the pair, so the stop is not a confident verdict either way; widen the margin (tighter pitch, wider radius) rather than trust it.
  - `push_failed`: The first tile did not fall. Either the trigger stroke would end within 6 mm of slot 1, so the trigger skill would not push a line this tight (try the `kick` trigger), or the stroke or kick did not carry the tile past 10 deg.
  - `sim_unavailable`: The physics engine is not installed or rejected the model; nothing was judged.
  - `invalid_input`: Fewer than two slots, a slot without a facing, two neighbouring slots overlapping, no table height, a target that is not after slot 0, a tile size that is not height &gt; width &gt; thickness, an unknown trigger, or a non-positive speed, timestep or mass (also the route of a raised error).

### tracking-gripper-opening-stall

verify · robots: any · tasks: pick-place, lego, domino, jigsaw, sorting · MIT · v0.1.0 · [`SKILL.md`](skills/tracking-gripper-opening-stall/SKILL.md) · [upstream](https://github.com/Open-RUA/uni-robot-agent/blob/a98934c6a1a16317821fed41a3b816b1d300b427/uni_robot_agent/tools/gripper/opening.py)

Drives the jaws to a target gap (or fully closed) and watches the measured gap over a sliding window, reporting whether it reached the target, stopped wide of it while closing (something is between the fingers - with its width in millimetres, checked against the expected part width), jammed (an opening blocked, or no travel at all), or is still moving. Routes reached, stalled_on_object, unexpected_width, stalled_jammed, still_moving, refused or unknown. Use when a graph closes on a part and must know at once whether it closed on something of the right width (a brick across its 15.8 mm side, not its 31.8 mm one; one domino, not two), or must confirm a partial opening before reaching between tightly packed parts; reads only the gripper.

- **In**: -
- **Out**: `target_m` float, `start_m` float, `actual_m` float, `error_m` float, `object_width_m` float, `width_error_m` float, `samples` int, `reason` string
- **Calls**: connector `robot.close_gripper`, `robot.describe_gripper`, `robot.get_gripper`, `robot.open_gripper`, `robot.set_grip`, `robot.wait_steps`
- **Scripts**: [`track_opening`](skills/tracking-gripper-opening-stall/scripts/track_opening.py)
- **Exits**:
  - `reached`: The gap came within tolerance_m (2 mm) of the target. Closing fully and reaching it means the jaws closed on nothing.
  - `stalled_on_object`: While closing, the gap stopped gaining (under 0.5 mm over 8 samples) wide of the target after moving at least 1 mm; object_width_m is the gap there, within width_tolerance_m of expected_width_m if one was given.
  - `unexpected_width`: As stalled_on_object, but the gap is more than width_tolerance_m from expected_width_m (wrong part, two parts, an edge or corner grasp).
  - `stalled_jammed`: The gap stopped short while opening, settled past the target, or never moved 1 mm from where it started.
  - `still_moving`: max_samples readings without reaching or stalling.
  - `refused`: The gripper command raised or reported failure.
  - `unknown`: The gripper could not be read (no position); nothing was judged.

### verifying-a-brick-connection-by-pull-up

verify · robots: any · tasks: assembly, lego, insertion · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-a-brick-connection-by-pull-up/SKILL.md) · [upstream](https://github.com/cmu-mfi/vibro_tactile_toolbox/blob/30fb15bb05a91606dfd4725f1f37352207f0299f/src/skill/common_skills.py)

Checks that a brick just pressed onto studs is clutched by lifting it a few millimetres, still gripped, in 0.5 mm steps and watching the arm - a clutched brick holds the hand below its command (tracking lag, or joint effort where published) and the lift stops there and puts it back down before the clutch lets go; a loose brick comes up with the hand and is put back on its seat, still held, for another press. CMU MFI's PullUp skill (vibro_tactile_toolbox) with proprioception standing in for its wrist force-torque sensor. Use when a brick (or a press-fit, snap-fit or peg) has just been pressed and the hand still grips it - right after pressing-a-brick-onto-studs or pressing-in-with-guarded-push - and a wrist depth read cannot tell a seated brick from one resting on stud tops, or the hold must be confirmed before letting go.

- **In**: -
- **Out**: `connected` bool, `rise` float, `lag_over_baseline` float, `effort_ratio` float, `steps` int, `signal` str, `efforts_available` bool, `released` bool, `jaw_width` float
- **Calls**: connector `robot.describe_gripper`, `robot.get_ee_pose`, `robot.get_gripper`, `robot.get_observation`, `robot.go_to_pose`, `robot.open_gripper`, `robot.solve_ik`
- **Scripts**: [`pull_up_check`](skills/verifying-a-brick-connection-by-pull-up/scripts/pull_up_check.py)
- **Exits**:
  - `connected`: A lift step left the hand more than hold_lag (1 mm) below its command beyond the start's lag, or a joint effort residual over effort_threshold; the hand was put back down at once, then (release, default) opened and rose `retreat`. `signal` says which fired.
  - `not_connected`: The whole lift (lift_height, 3 mm) ran with the hand following its command and the jaws still holding something; the brick came up with the hand (or the jaws slid on it). It is put back at the start, still held (put_back), ready for another press.
  - `not_holding`: The jaws read at their stop before the lift (nothing moved), or after a free lift (the brick slipped out).
  - `blocked`: The top of the lift is out of reach (IK preview - a stall there would read as connected), a move was refused, or signal=effort without published efforts; `detail` says which.

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

### verifying-a-domino-cascade-from-video

verify · robots: any · tasks: domino · MIT · v0.1.0 · [`SKILL.md`](skills/verifying-a-domino-cascade-from-video/SKILL.md) · [upstream](https://github.com/BasisResearch/predicators/blob/afb6f2918091f5ca372a75b9a335aea317a8a209/predicators/envs/pybullet_domino/cascade_certificate.py)

Decides from a short RGB(-D) clip of the overhead or a wrist camera which tiles of a planned domino line fell and when, how fast the wave ran and where it stopped. Per planned slot it measures a fall angle each frame (depth - a rotated-box height profile fitted along the tile's fall corridor; RGB only - how much of the gap in front of the tile its body covers; optionally a tracked point (track) or tracked mask (sam3) fitted to the rotated box's image), drops frames where a hand or the arm hides the tile, confirms a fall after 3 frames past the fallen threshold and dates it back to when the tile left the upright band (predicators' cascade certificate), then routes all_fell, stopped_at (with the first standing tile), not_triggered, fell_before_trigger, irregular or unverifiable. No motion. Use after triggering a domino line or curve to check the chain, before re-triggering from where it stopped or rebuilding.

- **In**: `first_slot` Se3Pose, `pitch` float, `n_tiles` int
- **Out**: `n_tiles` int, `n_fell` int, `stop_index` int, `last_fallen_index` int, `stop_point` Vec3, `wave_speed` float, `wave_rate` float, `first_onset_time` float, `last_onset_time` float, `signal` str
- **Calls**: [`sam3`](#sam3), [`track`](#track); connector `robot.get_observation`
- **Scripts**: [`verify_cascade`](skills/verifying-a-domino-cascade-from-video/scripts/verify_cascade.py)
- **Exits**:
  - `all_fell`: Every tile fell, one after another from slot 0 (onsets never run backwards by more than a frame plus their uncertainty) and none before the trigger; `wave_speed` (m/s along the path) and `wave_rate` (tiles/s) are robust fits over the onsets, `intervals` the successive propagation intervals.
  - `stopped_at`: Tiles 0 to k-1 fell and tile k was seen standing after the wave reached it, with every tile after it standing or unseen; `stop_index` is k, `stop_point` its centre (where to re-trigger or fix the gap).
  - `not_triggered`: No tile fell and the first tile was seen standing.
  - `fell_before_trigger`: A tile started falling before `trigger_frame` / `trigger_time` (the frame the push began) - the line was knocked, not triggered.
  - `irregular`: The falls are not one wave from slot 0 - a tile fell while the first stood, a tile fell beyond a standing one, or a later tile started falling clearly before an earlier one.
  - `unverifiable`: No clip, no slots, a tile already down in its first frames, too few usable frames to tell whether the first tile or the tile where the wave stopped fell (a hand over it, out of view), or a raised error.

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

### verifying-brick-seated-by-height

verify · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/verifying-brick-seated-by-height/SKILL.md) · [upstream](https://github.com/intelligent-control-lab/BrickSim/blob/cbd5de3238e6ac12f44ef699e1507a9f16bdafc3/python/bricksim/mdp/connection_thresholds.py)

Checks from a depth camera whether a placed LEGO or DUPLO brick is seated on its studs - the brick's top face fitted as a plane inside its expected footprint (stud tops told apart from the face by their height), its height measured against the plate surface around the footprint (so the camera's extrinsic error cancels), its tilt, and how much of the footprint it covers - with BrickSim's connection tolerances (1 mm, 5 deg, 2 mm at LEGO pitch) as acceptance limits. A brick resting on the stud tops reads a whole stud height high (1.7 mm LEGO, 4.5 mm DUPLO), one with an end on the studs reads tilted, one on the wrong studs leaves part of its footprint at plate height. Moves the wrist camera over the seat first, or reads the current frame of any posed depth camera. Use after a brick is released (by any placing or pressing skill) to decide between going on, pressing again and re-seating, or as a standalone audit of a built cell.

- **In**: `seat_position` Vec3
- **Out**: `height_error` float, `tilt_deg` float, `edge_rise` float, `coverage` float, `shift_estimate` Vec3, `top_height` float, `reference` str, `reference_offset` float, `samples` int, `viewed_from` str
- **Calls**: connector `robot.get_observation`, `robot.go_to_pose`, `robot.solve_ik`
- **Scripts**: [`verify_seat`](skills/verifying-brick-seated-by-height/scripts/verify_seat.py)
- **Exits**:
  - `seated`: The top face reads within height_tolerance of the seated height (default min(1 mm, 0.35 stud heights) - 1.0 mm DUPLO, 0.6 mm LEGO), tilted no more than tilt_tolerance_deg, and the brick covers its footprint to within position_tolerance.
  - `not_seated`: Flat and on its footprint, but the top reads higher than height_tolerance and under 0.6 stud heights - partly pressed.
  - `high_by_stud`: The top reads 0.6 stud heights or more over the seated height and flat - resting on the stud tops (the detail adds the offset if it also stands off the footprint).
  - `tilted`: Tilted more than tilt_tolerance_deg (default the smaller of BrickSim's 5 deg and the tilt that lifts the long side's far end half a stud height - DUPLO 2x2 4.0 deg, LEGO 2x2 3.0 deg, LEGO 2x4 1.5 deg) - one end on the studs.
  - `shifted_off_grid`: Part of the footprint reads at plate height, so the brick stands more than position_tolerance (default a quarter pitch - 2 mm LEGO, 4 mm DUPLO) off it - on the wrong studs; shift_estimate says which way.
  - `no_brick`: Under a quarter of the footprint reads at brick height - no brick at the seat (it came away with the gripper, or the seat is wrong).
  - `unverified`: Not judged - no posed depth camera, too few depth points in the footprint, reference 'local' requested but no plate surface found, or the top reads below the seated height (seat height or calibration off); detail says which.
  - `unreachable`: view 'move' and every view of the seat was out of reach or refused.

### verifying-build-against-model

verify · robots: any · tasks: assembly, lego · MIT · v0.1.0 · [`SKILL.md`](skills/verifying-build-against-model/SKILL.md) · [upstream](https://github.com/aaronwalsman/ltron/blob/253be3e55cef068c1842fd9a7d2e2b3c22d19d7b/ltron/matching.py)

Compares a brick build with its target model up to the current step - LTRON's assembly matching (the best assignment of observed to target bricks by shape, colour and pose within a tolerance, F1 score and edit distance) - and reports the bricks that are missing, extra, misplaced (same brick elsewhere), the wrong colour, unseated (top a stud height high, 1.7 mm LEGO / 4.5 mm DUPLO, or at one end) or ahead of the instructions, with an ordered repair list. The observed bricks come either as stud-grid cells from upstream perception, or from a depth frame and the plate pose - every pixel's ray is cast against the target's expected height map and the measured height compared per stud cell; colours by tools/colour when the frame has RGB. Use after each placement or at the end of a build, to decide whether to go on, re-press, remove, move or place a brick.

- **In**: `target` string
- **Out**: `score` float, `edit_distance` int, `n_target` int, `n_matched` int, `n_missing` int, `n_extra` int, `n_misplaced` int, `n_wrong_colour` int, `n_unseated` int, `n_unverified` int, `report_json` string
- **Calls**: [`colour`](#colour); connector `robot.get_observation`
- **Scripts**: [`verify_build`](skills/verifying-build-against-model/scripts/verify_build.py)
- **Exits**:
  - `matches`: Every target brick up to the step is matched (seated where its seating could be measured), nothing extra; bricks of later steps already in place are listed in `ahead`, bricks whose seating is within the depth noise in `seat_unresolved` (a note, unless require_seating).
  - `misplaced`: A target brick is missing and a brick of the same shape and colour that matches no target brick is elsewhere (shifted, turned or a layer off); `misplaced` pairs them with the shift.
  - `wrong_colour`: A brick of the right shape sits at a target brick's place but its colour is another; `wrong_colour` lists expected and seen.
  - `extra_bricks`: A brick (or depth blob) matches no target brick of this or a later step and pairs with no missing brick; `extra` lists it.
  - `unseated`: A matched brick's top reads more than half a stud over its seated height (`how` raised), or its highest stud cell does (`how` one end - seated at one end, on the studs at the other), significant at 3 (brick) / 4 (cell) standard errors of the depth noise; `unseated` lists the heights.
  - `missing_bricks`: Some target bricks up to the step are not there, and nothing above explains them; `missing` lists them.
  - `unverifiable`: No camera frame / depth, no depth point on the plate, or target bricks seen too little to judge (`unverified`); with require_seating, also a seating within the noise.

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

### verifying-piece-flushness-from-wrist-depth

verify · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/verifying-piece-flushness-from-wrist-depth/SKILL.md) · [upstream](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudSegmentation.cpp)

Checks from one close wrist depth frame (D405 at 8-16 cm) whether a just-placed jigsaw piece lies flush - fits a RANSAC plane (Open3D's segment_plane, reimplemented) to the placed neighbours' tops, or to a ring round the piece holding the table and neighbours as two parallel levels, then measures the piece's own points against it - median height (raised or sunk), a plane through the piece's top (tilted, which side is high) and height profiles across the seam facing each neighbour (open seam where the table shows between two tops). Routes flush, raised, tilted, gap or unverifiable. Use when a piece has just been placed or pressed, before releasing the region to the next placement; camera-only, about 30 ms on CPU.

- **In**: `camera` CameraFrame
- **Out**: `level_m` float, `tilt_rise_m` float, `tilt_deg` float, `seam_gap_m` float
- **Scripts**: [`verify_flush`](skills/verifying-piece-flushness-from-wrist-depth/scripts/verify_flush.py)
- **Exits**:
  - `flush`: The piece's median height is within 0.5 mm above / 0.6 mm below its expected top, its top rises under 1 mm across it, and no open seam of 3 mm or more faces a neighbour.
  - `tilted`: The plane through the piece's top rises 1 mm or more across the piece (one side or a tab is up); `high_side` is the world x, y direction it rises toward.
  - `raised`: Not tilted, but the piece's median height is 0.5 mm or more above its expected top (riding on something).
  - `gap`: Not tilted or raised, but the piece sits 0.6 mm or more below its neighbours (`gap_kind` low) or the table shows between it and a neighbour over 3 mm or more of the outline (`gap_kind` seam, `seam_gap_points` in world x, y).
  - `unverifiable`: Too little depth on the piece (under 40 % valid, or under 150 points) or round it, no depth or intrinsics, no piece given, or most of the slot shows the surface under the piece.

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

### backtracking-a-wrong-placement

recover · robots: any · tasks: jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/backtracking-a-wrong-placement/SKILL.md) · [upstream](https://github.com/scipy/scipy/blob/main/scipy/optimize/_lsap.py)

Decides what to undo after a jigsaw piece fails its check at a cell (it does not seat, a seam with a placed neighbour scores badly, or a re-check puts it elsewhere) - blames the piece or one of the cell's placed neighbours by a small Bayesian vote over this failure and every earlier one in a persistent JSON ledger (priors from the pieces' localization scores and margins), then re-solves the assignment of the known loose pieces and the lifted ones to the free cells with SciPy's linear_sum_assignment on -log P(piece belongs in cell). Returns the corrective actions (seat again, remove piece X from cell C, place piece Y at C' with its angle and attempt number, set aside, re-localize, search for a cell, defer a cell) under a retry ladder with limits, and saves the ledger so a run can resume. Use when a jigsaw placement has just failed verification and the pieces have records from localizing-a-piece-in-a-reference-image (seam scores from matching-a-piece-to-placed-neighbours help); planning only, no robot motion.

- **In**: `grid_cols` int, `grid_rows` int
- **Out**: `culprit_piece` string, `culprit_p_wrong` float, `p_false_alarm` float, `n_actions` int
- **Scripts**: [`backtrack`](skills/backtracking-a-wrong-placement/scripts/backtrack.py)
- **Exits**:
  - `retry_seat`: The piece did not seat but no seam is bad and the check being wrong is likelier than any placement being wrong; seat the same piece again (once by default, max_seat_retries).
  - `reassigned`: The failed piece itself is the likeliest wrong placement; it is lifted and the best admissible known piece for its cell is placed there.
  - `remove_neighbour`: A placed neighbour is the likeliest wrong placement (with neighbour removals left for this cell); it and the failed piece are lifted, its cell gets its best known piece and the failed piece is placed again.
  - `no_alternative`: A culprit was lifted but no known piece is admissible for a cell it emptied (every candidate forbidden or under min_assign_prob); `actions` asks to re-localize the lifted pieces and to search the loose pieces for that cell.
  - `give_up`: The cell has failed with more than max_pieces_per_cell different pieces and its neighbours cannot be blamed, or max_backtracks is spent; the piece is lifted and the cell deferred (left out of every later re-solve).

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

### checkpointing-assembly-state-for-resume

recover · robots: any · tasks: jigsaw, lego, assembly · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/checkpointing-assembly-state-for-resume/SKILL.md) · [upstream](https://github.com/splintered-reality/py_trees/blob/49bcffd069afa2b7bd9ca7b0e7312f5cbb3640bd/py_trees/blackboard.py)

Saves and restores the state of a long assembly run (a 550-piece jigsaw, a LEGO build, a domino layout) so a run stopped after hours resumes where it was. save_checkpoint writes one versioned JSON checkpoint atomically (temp file, fsync, rename) with a SHA-256 over its payload - task, step, placed pieces / bricks / items with cells and poses, the pending queue, backtracking-a-wrong-placement's ledger, the reference snapshot and items for detecting-disturbed-pieces-by-snapshot-diff, arms and what they hold, timestamps and an activity stream of what changed - keeping the newest N. resume_from_checkpoint loads the newest checkpoint that passes its hash (migrating older schema versions), compares it with a fresh observation through the snapshot-diff skill and, for bricks, verifying-build-against-model, and lists the items to re-verify before going on. Use after every few verified placements of a long run (save), and when a stopped run starts up again (resume); files and CPU only, no motion.

- **In**: `checkpoint_dir` string
- **Out**: `checkpoint_path` string, `task_id` string, `step_index` int, `seq` int, `sha256` string, `n_recheck` int, `n_skipped` int, `age_s` float, `state_json` string
- **Calls**: [`colour`](#colour); connector `robot.get_observation`
- **Scripts**: [`save_checkpoint`](skills/checkpointing-assembly-state-for-resume/scripts/save_checkpoint.py), [`resume_from_checkpoint`](skills/checkpointing-assembly-state-for-resume/scripts/resume_from_checkpoint.py)
- **Exits**:
  - `saved`: save_checkpoint wrote the checkpoint (whole or not at all) and removed the ones past `keep`; `checkpoint_path`, `seq`, `sha256`.
  - `save_failed`: The write raised (disk full, no permission, an unstorable value); the earlier checkpoints are untouched.
  - `resumed`: resume_from_checkpoint loaded a valid checkpoint and every placed item was cleared by a check against the fresh view (unchanged in the snapshot diff, matched by the build check); nothing held or half done.
  - `resumed_with_rechecks`: A valid checkpoint was loaded but some items must be re-verified first; `recheck` lists them, surest problems first, each with its reasons (lost, held_when_stopped, missing, misplaced, wrong_colour, moved with its shift, appearance_changed, in_progress_when_stopped, ledger_after_checkpoint, unseated, extra, unexpected_object, occluded, unverified_at_save, placed_after_reference, not_checked).
  - `no_checkpoint`: The directory has no checkpoint for the task, or none passes its hash and schema checks (`skipped` says why for each); start the task fresh.
  - `state_mismatch`: The fresh view does not register onto the reference snapshot, at least a quarter (and 3) of the checked items moved, vanished or are wrong, or the grid differs from expect_grid; re-perceive the board instead of trusting the checkpoint.

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

### judging-assembly-steps-by-fused-checks

judge · robots: any · tasks: lego, pick-place, domino, jigsaw · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/judging-assembly-steps-by-fused-checks/SKILL.md)

Turns the routes and numbers of the hub's per-step checks into one verdict for an assembly step - done, failed with the reason, or uncertain - trusting per step type the checks that see its failure modes. Pick - the gripper-opening stall (and a scene-diff / VLM check only to contradict). Place / press - depth seat height and the brick's rise under the pull-up must both pass; the jaws must have opened at the release; a camera VLM only breaks a tie towards failure; RynnValue's remaining time can hold back a done. Routes done, failed or uncertain. Use when a LEGO, domino or jigsaw graph must decide after a pick or a place whether to go on, retry or ask, with whichever of those checks ran; logic only, no motion, no camera.

- **In**: `step_type` string
- **Out**: `reason` string, `step_type` string
- **Scripts**: [`fuse_checks`](skills/judging-assembly-steps-by-fused-checks/scripts/fuse_checks.py)
- **Exits**:
  - `done`: Pick - the jaws stopped on the part (tracking-gripper-opening-stall stalled_on_object, or the hand reports it held) and no camera check contradicts. Place - the jaws opened at the release, the depth seat reads seated within max_height_error_m (0.3 mm) and the brick rose at most max_pull_rise_m (0.6 mm) under the pull-up; and RynnValue, if given, expects at most rynn_veto_s (2.5 s) more.
  - `failed`: A check that sees this failure mode failed - the jaws closed on nothing or on the wrong width (pick), did not open at the release, both geometric checks failed, the only geometric check failed, or they disagree and the camera judge says not done; `reason` names it.
  - `uncertain`: The evidence cannot pass the step - no geometric check ran (a camera judge alone passes unseated bricks), only one of the two ran and passed, they disagree without a camera "not done", the jaws hold something the camera says is still at the source, or RynnValue vetoed a done.

### judging-attempt-success-from-frames-robojev

judge · robots: any · tasks: pick-place, assembly, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/judging-attempt-success-from-frames-robojev/SKILL.md) · [upstream](https://github.com/josefchen/robojev/blob/79a5e2f/robojev/robojev.py)

Decides whether a step succeeded from cameras alone with one calibrated number - the task sentence, the attempt's frames from a main camera (a recorded clip, or the observation captured before the step and the one now) and the final frame of the other cameras go to RoboJev-27B (tools/robojev), which returns P(done); at 0.7 or more the route is success, at 0.3 or less failure, between uncertain. Use when a graph must verify a whole step (a cube in a box, a brick placed, a piece seated) before moving on, without a task-specific check, or to grade recorded attempts.

- **In**: `task` string
- **Out**: `p_done` float, `main_camera` string, `n_frames` int
- **Calls**: [`robojev`](#robojev); connector `robot.get_observation`
- **Scripts**: [`judge_attempt`](skills/judging-attempt-success-from-frames-robojev/scripts/judge_attempt.py)
- **Exits**:
  - `success`: P(done) &gt;= success_at (0.7).
  - `failure`: P(done) &lt;= failure_at (0.3).
  - `uncertain`: P(done) between failure_at and success_at - RoboJev's calibration was fitted on SO-101 runs; treat as unknown (look again, ask a second judge, or a human).
  - `no_frames`: Fewer than two frames of the main camera (no `frames`, and no `before` with that camera), or no camera with RGB.

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

### scoring-step-progress-from-one-frame

judge · robots: any · tasks: pick-place, sorting · Apache-2.0 · v0.1.0 · [`SKILL.md`](skills/scoring-step-progress-from-one-frame/SKILL.md) · [upstream](https://huggingface.co/densereward/densereward-1frame/blob/9796cb2a7be55b339670d8f95fb07841a61487a8/README.md)

Judges whether a step is done from the current camera frame alone - scores the task's progress 0..1 with the DenseReward single-frame reward model (densereward.progress) on a few fresh frames, takes the median, and routes done / in_progress / regressed, or low_confidence when the frames disagree or nothing parses. No start frame, no video, ~0.1 s a frame. Use when a graph needs a cheap "is this step done?" right after a place, press or push, with only the current camera view, as a second opinion beside a geometric check (measured useful for whole pick-place and a finished domino row; no signal on a single LEGO brick placement).

- **In**: `task` string
- **Out**: `progress` float, `reason` string, `camera` string, `model` string
- **Calls**: [`densereward`](#densereward); connector `robot.get_observation`
- **Scripts**: [`score_progress`](skills/scoring-step-progress-from-one-frame/scripts/score_progress.py)
- **Exits**:
  - `done`: The median progress of the sampled frames is at least done_threshold (0.85) and they agree within max_spread.
  - `in_progress`: The median is below done_threshold and the frames agree; also "not started" (the model scores 0 for both not started and failed).
  - `regressed`: The progress fell more than regress_drop below previous_progress, or the 3-frame model (history given) named a failure (miss / collision / fall / smooth / failure).
  - `low_confidence`: The sampled frames disagree by more than max_spread, the camera has no RGB, densereward raised, or no float parsed; nothing was judged.

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
