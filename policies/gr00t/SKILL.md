---
name: gr00t
description: NVIDIA GR00T as the contact-rich step of a GaP graph on the bimanual
  YAM station - the graph decides what and where (perception, planning) and
  marks grasp / release points on the policy's top camera view; GR00T does the
  how. gr00t.act returns one chunk of 14-D absolute joint + jaw targets (left 7,
  right 7) from the top and two wrist views, the joint state, an instruction and
  the marks; gr00t.run_skill executes chunks closed-loop through the robot
  connector until a success check passes or the step budget runs out. Backend
  GR00T N1.7 now (robocurve/gr00t-n1.7-yam-molmoact2, or base N1.7 on its XDOF
  tag), with a GR00T 2 slot behind the same protocol. Use when a graph has already chosen the object and the place and
  wants a learned bimanual YAM policy to grasp, place or press it, optionally
  steered by pixel marks. Not for embodiments other than the YAM, for tasks far
  from the checkpoint's data (block, box, charging) without fine-tuning, or as
  a success signal - attach a verify skill after it.
license: Apache-2.0
compatibility: requires gap>=0.1; the server needs a CUDA GPU with ~7 GB free
metadata:
  category: policy
  tags: [policy, vla, gr00t, yam, bimanual, visual-steering, gpu, long-running]
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    tuned_on: [yam_i2rt]
    arms: 2
    tasks: [pick-place, assembly]
    io:
      action: {space: joint_position_absolute, dim: 14, hz: 30}
      state:
        dim: 14
        order: [left_joints_6, left_opening, right_joints_6, right_opening]
      cameras: [top, left_wrist, right_wrist]
    notes: >-
      Opening 0 closed, 1 open. Joint frame = the MolmoAct2-YAM data's; the real
      station adds pi/2 to each base joint (GAP_GR00T_JOINT_OFFSET). Chunk 16 rows
      (YAM fine-tune) or 40 (base, XDOF tag) at 30 Hz. N1.7 never saw marks: they
      are drawn into the image (steering emulated-overlay); GR00T 2 takes them as input.
    version: 0.1.0
    weights:
      uri: hf://robocurve/gr00t-n1.7-yam-molmoact2@f04fe258e6a5d84c939547cd49043fbc3b1e8985
      sha256: cd7949227a6d198fd676d97ac1605b1da272483678faf8d4801f6613f7d5346b
      license: NVIDIA Open Model License (base nvidia/GR00T-N1.7-3B; card tag nvidia-license); the VLM backbone nvidia/Cosmos-Reason2-2B is gated (NVIDIA Open Model License, accept on Hugging Face)
    provenance:
      authors: [NVIDIA GEAR (GR00T N1.7), Robocurve (YAM fine-tune)]
      code: https://github.com/NVIDIA/Isaac-GR00T/tree/7aa0bca50f1097f8b17f5c274c98264b72a1419f/gr00t/policy
      upstream_license: Apache-2.0
      paper: https://research.nvidia.com/labs/gear/gr00t-2/
      data: allenai/MolmoAct2-BimanualYAM-Dataset (~5,145 episodes, block / box / charging) for the fine-tune
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "server.py", "--backend", "n1.7", "--profile", "yam-molmoact2",
              "--port", "{port}"]
    protocol: in-process
    requires_gpu: true
    weights_uri: hf://robocurve/gr00t-n1.7-yam-molmoact2@f04fe258e6a5d84c939547cd49043fbc3b1e8985
  allowed_tools:
    - robot.get_observation
    - robot.execute_trajectory
    - robot.move_to_joints
    - robot.open_gripper
    - robot.close_gripper
  exit_conditions:
    succeeded: The success check passed (checked after every check_every chunks and at the end).
    max_steps: max_steps action rows ran without the success check passing (or no check was given).
    unsafe_action: A chunk asked a joint to move further than max_joint_step_rad in one row (from the measured arm or between rows); nothing of that chunk was executed.
    failed: The server did not answer, refused the request, or the connector refused a move. The subgraph's on_error exit.
  tools:
    - gr00t.act: One GR00T chunk [T, 14] for the YAM station from three views, the joint state, an instruction and optional grasp/release marks.
    - gr00t.run_skill: Closed-loop GR00T until a success check passes or max_steps rows ran.
    - gr00t.marks_from_points: World points (grasp, release) to pixel marks on a camera view.
    - gr00t.info: What the server runs (backend, checkpoint, horizon, rate, steering mode, GPU memory).
---

# gr00t

GR00T 2's agent interface - a reasoning agent decides *what* and *where*, marks
the grasp and release points on the policy's camera view, and the
world-action model does the *how* - is the shape of a GaP policy node. This
bundle gives a graph that node now, with GR00T N1.7, and keeps the graph side
unchanged for GR00T 2:

```python
# World-point marks are re-projected onto the top view every chunk.
marks = [
    {"kind": "grasp", "point": brick_obb["center"]},
    {"kind": "release", "point": seat_above},
]
r = ctx.tool(
    "gr00t.run_skill",
    instruction="Pick up the red 2x2 DUPLO brick and press it onto the plate at the blue dot.",
    marks=marks,
    max_steps=600,  # rows at 30 Hz
    success_check="gripper_cycle",  # or {"tool": ..., "field": ..., "equals": ...}
)
r["route"]  # succeeded | max_steps | unsafe_action
```

### The GaP node example

`examples/lego_place_with_gr00t/make_graph.py OUT` writes a GaP workflow
(and copies the skills' canonical scripts beside it) in which the graph does
the what and where and GR00T the how:

| subgraph | skill | does |
|---|---|---|
| plate_sg | calibrating-baseplate-stud-grid | the plate's pose (markers + wrist close-up) |
| plan_sg | planning-brick-assembly-order | the build as a placement order |
| seat_sg | next_brick.py + computing-brick-pose-from-stud-grid | the next brick (cell, size, name) and its seat |
| find_sg | perceiving-objects | the loose brick on the table (DINO + VLM + SAM3 + depth) |
| gr00t_sg | gr00t (place_with_gr00t.py) | grasp mark on the brick, release mark 3 cm over the seat; `gr00t.run_skill` until one grasp-and-release cycle |
| verify_sg | verifying-brick-seated-by-height | seated, or not_seated / high_by_stud / tilted / ... |

Every route each subgraph can return is mapped (tested), and the workflow
passes gap's validator with the hub's registry
(`tests/bundles/test_gr00t_policy.py`). A domino version swaps the plan,
seat and verify skills (planning-a-domino-layout,
auditing-a-domino-line-for-fallen-or-tilted-tiles) and keeps gr00t_sg.

## Tools

- `gr00t.act(observation, instruction, marks=None, cameras=None,
  joint_offset_rad=None, mark_text=False)` -> `{actions [T, 14], dt, horizon,
  steering, latency_ms, server_ms, gpu_mem_mb, backend, model, marks_sent}`.
  `observation` is `robot.get_observation()`'s: cameras by name (top / base /
  scene / overhead; a name with left or a lone "wrist"; one with right), arms
  0 = left, 1 = right. A marks entry is `{"kind": "grasp" | "release", "uv":
  [u, v]}` (pixels on that camera's image) or `{"kind", "point": [x, y, z]}`
  (world; projected each call), with `"camera"` a role (default top). One arm
  only: the other arm's columns hold its measured joints.
- `gr00t.run_skill(instruction, marks=None, max_steps=480, success_check=None,
  arms="both", execute_steps=8, ...)`: query, execute the first
  `execute_steps` rows (cut at a jaw switch so the jaws act when predicted),
  re-observe, repeat. Arm rows go through `robot.execute_trajectory` per arm
  (`executor="stream_dual"` moves both in lockstep where the connector has
  `robot.stream_dual`); the predicted opening drives `robot.open_gripper` /
  `robot.close_gripper` with hysteresis (close under 0.35, open over 0.65).
  `success_check` is `{"tool", "args", "field", "equals"}` - any tool the
  graph may call (a camera-only judge on a real station, a simulator query in
  a twin) - `"gripper_cycle"` (a jaw closed, then opened: one grasp and
  release), or a callable from Python. `control_hz` interpolates the 30 Hz rows
  to the connector's waypoint rate. A guard refuses a chunk whose joints jump
  more than `max_joint_step_rad` (0.35) in one row.
- `gr00t.marks_from_points(observation, points, camera="top")`: OpenCV pinhole
  projection with the camera's intrinsics and camera-to-world pose (the
  convention `geometry.pixel_to_world_point` inverts); off-view points are
  returned in `dropped`.

## Protocol (gr00t-wam/1) and backends

`server.py` serves `info`, `act`, `reset` over HTTP (JSON; images as JPEG /
PNG / raw npy) or ZMQ (Isaac-GR00T's msgpack framing, `{"endpoint", "data"}`).
The request speaks YAM, not checkpoint keys: images by role (top, left_wrist,
right_wrist), `joints[14]`, optional `ee` poses, `instruction`, `marks`. A
*profile* maps that onto a checkpoint's modality keys:

| profile | checkpoint | tag | views | state | chunk |
|---|---|---|---|---|---|
| `yam-molmoact2` (default) | robocurve/gr00t-n1.7-yam-molmoact2 | new_embodiment | base_view, left/right_wrist_view | left/right arm (6) + gripper (1), absolute | 16 x 14 absolute |
| `xdof` | nvidia/GR00T-N1.7-3B | xdof_relative_eef_relative_joint | 3 x 320x240 at t-1 s and t | + wrist EEF xyz+rot6d | 40 x 14 (relative, made absolute) |
| `yam-roles` / `yam-left-roles` | a fine-tune given by `--checkpoint DIR` | new_embodiment | top, left_wrist(, right_wrist) | left_arm, left_gripper(, right_*) | as trained |

`yam-roles` fits a fine-tune on a LeRobot dataset that names its views and
joint groups after the roles (the station's recorder and the hub's sim demos
do), so a new YAM checkpoint is served with
`--profile yam-roles --checkpoint <dir>` and nothing else changes.

Backends: `n1.7` (Isaac-GR00T `Gr00tPolicy`, bf16), `gr00t2` (placeholder:
raises, and its docstring lists what it must return - the same `actions`
plus optional `predicted_frames`, with marks taken as input), `fake` (no
model; tests and graph dry runs).

The client never sees which backend runs. Swapping in GR00T 2 means a new
backend class and profile in `server.py`; `gr00t.act`, `gr00t.run_skill`,
the marks and every graph stay as they are (see "GR00T 2").

## Install and serve

On a GPU host (measured on the GCP pool's RTX PRO 6000):

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T
git checkout 7aa0bca50f1097f8b17f5c274c98264b72a1419f
uv venv --python 3.12 && uv pip install -e .       # torch 2.9.0+cu128, transformers 4.57.3
hf auth login                                       # accept nvidia/Cosmos-Reason2-2B's licence first
cd <hub>/policies/gr00t
<Isaac-GR00T>/.venv/bin/python server.py --backend n1.7 --profile yam-molmoact2 --port 5555
```

then point the graph's process at it: `GAP_GR00T_URL=http://<host>:5555`
(or `zmq://<host>:5555` with `--transport zmq`). The tools run in the graph's
process (`protocol: in-process`); `gap.serving.command` is how the server they
talk to is started, normally on another machine with the GPU. `uv sync` in a plain clone
fails on an aarch64 torchcodec wheel kept in git LFS; `uv pip install -e .`
does not read it.

**The backbone is gated.** Every N1.7 checkpoint builds its VLM from
`nvidia/Cosmos-Reason2-2B` (gated, auto-approve, NVIDIA Open Model License)
and then loads its own weights over it. Without a token on the host,
`--backbone qwen3-vl-2b` builds the same Qwen3-VL-2B architecture from the
ungated `Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0) instead: all 493 backbone
tensors are then the checkpoint's own (max difference 0, `eval/smoke_server.py`),
but the tokenizer, chat template and image-processor configs are Qwen's, which
Cosmos-Reason2 may have changed. The canonical path is the token.

## Measured

2026-11-09/10 on GCP RTX PRO 6000 Blackwell (96 GB) pool hosts, Isaac-GR00T
7aa0bca, torch 2.9.0+cu128, transformers 4.57.3, bf16, flash-attn 2, backbone
built from Qwen/Qwen3-VL-2B-Instruct (no Hugging Face token on the pool hosts,
so the gated Cosmos-Reason2-2B could not be fetched: `GatedRepoError 401`).

**Server** (`eval/smoke_server.py`, 3 x 640x480 views, in-process, idle GPU):

| checkpoint (profile) | load | chunk | per chunk mean / p95 | GPU memory (torch alloc / reserved / peak) |
|---|---|---|---|---|
| robocurve YAM fine-tune (`yam-molmoact2`) | 10.5 s | 16 x 14 | 64 / 66 ms | 6.02 / 6.24 / 6.17 GB |
| base N1.7 (`xdof`) | 8.9 s | 40 x 14 | 82 / 83 ms | 6.02 / 6.41 / 6.29 GB |

Backbone parity with the stand-in: all 493 backbone tensors of either
checkpoint equal the checkpoint's own after loading (max |diff| 0; only
`lm_head`, unused after the layer cut, is not in the model). On a GPU shared
with the twin and other agents' Isaac jobs the same call took 200-1070 ms
(table below); the HTTP round trip with three JPEG views adds 7-16 ms.

**Twin**: the YAM station twin (calibrated top camera, a D405-like camera on
each wrist, grip fix on), "Pick up the grey block and put it in the blue
box." - a 30 mm cube at a seeded offset (+-30 mm, yaw +-45 deg) from the
station's cube, the box in the middle. Both arms start home with the jaws
open (where every MolmoAct2-YAM episode starts); `gr00t.run_skill` with both
arms, 8 rows a chunk interpolated to the twin's 60 Hz, `robot.stream_dual`,
600 rows (20 s) per episode; success = the cube resting inside the box with
no hand on it. Marks: a grasp mark on the cube and a release mark on the box
centre, projected on the top view each chunk (`marks_text` also appends
"Grasp at the green dot and release at the blue dot."). Seeds 0-9 per
condition, 50 episodes:

| condition | success | jaw closed | cube lifted | nearest TCP to cube, median (min) | chunk latency mean (worst p95) |
|---|---|---|---|---|---|
| YAM fine-tune, no marks | 0/10 | 0 | 0 | 343 (299) mm | 380 (3132) ms |
| YAM fine-tune, marks | 0/10 | 0 | 0 | 359 (318) mm | 382 (1286) ms |
| YAM fine-tune, marks + text | 0/10 | 0 | 0 | 357 (341) mm | 315 (1035) ms |
| base N1.7 XDOF, no marks | 0/10 | 3 | 1 by 9 mm | 109 (5) mm | 300 (714) ms |
| base N1.7 XDOF, marks | 0/10 | 1 | 1 by 9 mm | 190 (8) mm | 317 (691) ms |

- The plumbing works end to end: 50 episodes, 3,663 chunks, every episode
  ran its 600 rows with no protocol or guard failure (one episode was re-run:
  the twin's own bus timed out while its host was overloaded); GPU memory per
  server 6.0 GB allocated, 6.6 GB reserved.
- The YAM fine-tune lifts both arms and pulls them back in every episode (left
  tool centre from 0.94 m to a median peak of 1.32 m) and never closes a jaw.
  It predicts absolute joints for AllenAI's MolmoAct2 rig; the twin's arm,
  table and camera geometry differ, and no task in its data is "cube into a
  box" - a fine-tune on station data is the fix, not a protocol change.
- The base model on its relative-joint XDOF head reaches for the cube (tool
  centre within 5-8 mm of it in 2 of 20 episodes, jaws closing 4 times, the
  cube lifted 9 mm twice) but never holds it.
- Marks: no measurable benefit on N1.7 (base median 190 mm with vs 109 mm
  without; the fine-tune unchanged), as expected for a model never trained on
  them. They are GR00T 2's input; the overlay is only an emulation.
- Latency on the twin's host was 3-5x the idle number: the GPU was shared with
  the twin and with other agents' Isaac jobs at 93-99 % utilisation.

The harness (station twin, camera rig, seeded placement, the trials graph)
is outside the hub, as for the other twin checks; the PR lists it.

**A station fine-tune drops in.** The YAM demo-dataset fine-tune being trained
on the pool (LeRobot views and joint groups named after the roles) had only a
60-step smoke checkpoint by the end of this work; served with
`--profile yam-roles --checkpoint <dir>` it loaded unchanged (493 / 493
backbone tensors equal, 16 x 14 chunks, 6.0 GB). Its final checkpoint was not
there to evaluate (the run had restarted at step 37 of 6000).

## π0.5 backend (`--backend pi05`)

`pi05_backend.py` serves openpi's π0.5 through LeRobot's `PI05Policy` behind the
same wire; `profiles/yam-pi05-molmoact2.yaml` is the VLA_SKILL_SPEC profile
(checkpoint + revision + sha256, tokenizer, roles -> image keys, 14-D state,
joint offset, gripper units, 30 x 14 abs joints at 30 Hz). Same data, joint
frame and gripper units as `yam-molmoact2`, so `gr00t.act` / `run_skill`
work unchanged (`--wire-frame station` applies the profile's offset in the
server instead, for clients that send the station frame; the top level is the
twin's offset 0, `--station real` adds pi/2 on each base joint from
`stations.real`, as in molmoact2-yam's profile). `roboskillhub.vla.VlaYamClient`
maps frames on the client and talks to either wire.

```bash
uv venv --python 3.12 && uv pip install "lerobot[pi,dataset]==0.7.0"   # torch 2.15+cu132, transformers 5.22.2
.venv/bin/python server.py --backend pi05 --profile yam-pi05-molmoact2 --port 5560
```

The checkpoint's processor names the gated `google/paligemma-3b-pt-224`
tokenizer; the profile substitutes the ungated `leo009/paligemma-3b-pt-224`,
whose `tokenizer.model` is byte-identical (sha256 8986bb4f...) to openpi's
public `gs://big_vision/paligemma_tokenizer.model`. `torch.compile`
(max-autotune, in the checkpoint's config) is off unless the profile's
`extra.compile` is set: it costs ~7 min at startup.

### YAM π0 / π0.5 checkpoints checked (2026-11-10)

| checkpoint | format | data | io | licence |
|---|---|---|---|---|
| **helen9975/pi05-molmoact-yam** @6b50a13 (served) | LeRobot PyTorch, 200k steps (tag step-100000) | MolmoAct2-BimanualYAM (5,145 ep) | top/left/right 360x640, state 14, 30 x 14 abs joints @30 Hz, gripper 0..1 | card Apache-2.0; base pi05_base (PaliGemma, Gemma terms) |
| NUSMAGIC/pi05-MolmoAct2-YAM | LeRobot, 100k | same (card copied from Jiafei1224) | same | card Apache-2.0 |
| robocurve/pi0.5-yam @ee17bb3 | openpi JAX/Orbax, 12 GB | same | 16 x 14 abs, needs the private `yam_pi05` config (robocurve/pi05-yam-replication) | Gemma |
| jeqcho/pi05-yam-bimanual | openpi JAX, 45 GB | same | 16 x 14, `yam_pi05` config not published | card Apache-2.0 |
| agents2agents/Pi0.5-BimanualYAM-Jetson-Thor | TensorRT for Thor only | robocurve's | 16 x 14 | Gemma |
| openpi pi05_base / pi05_droid / pi0(5)_aloha | openpi | base / DROID Franka / ALOHA (Trossen) | not YAM | Apache-2.0 code, Gemma terms for weights |
| i2rt-robotics/yam-abc-reproduce | code only (openpi `pi0_yam` / `pi05_yam` (+ LoRA) configs, `yam_policy.py`) | your own LeRobot data | 14-D, top + 2 wrists, delta joints for arms | Apache-2.0 |

No YAM-pretrained π0/π0.5 weights from i2rt or TRI were found (the i2rt org
publishes no models on Hugging Face; `yam-abc-reproduce` trains from
`gs://openpi-assets/checkpoints/pi05_base`). Many other community YAM π0.5
fine-tunes (single tasks: vials, towels, pouring) exist; none is on our scene.

### Measured (π0.5)

2026-11-10, rshgpu2 (RTX PRO 6000 Blackwell), lerobot 0.7.0, bf16, 10 flow
steps, no compile, `eval/smoke_pi05.py`, in-process:

| load | chunk | per chunk mean / p50 / p95 | GPU memory (alloc / reserved / peak) |
|---|---|---|---|
| 69.5 s | 30 x 14 | 149 / 140 / 234 ms | 8.9 / 9.4 / 9.1 GB |

Open-loop on the held-out `allenai/19012026-block-13` episode 0 ("spell out
NEURIPS", 6,139 frames, a chunk every 60 frames, 102 chunks): MSE 0.0037,
MAE 0.030 rad in raw joint units, against 0.0086 for holding the current
state - the wiring (image keys, state, normalisation, tokenizer) is right.
Twin episodes: not run yet (expect the same rig gap as the GR00T fine-tune).

## GR00T 2

GR00T 2 (announced 2026-11-09; research.nvidia.com/labs/gear/gr00t-2/) is a
world-action model (one DiT denoising future video and actions, per-robot
heads, YAM in its pretraining) with a WAM API and visual steering. No
weights, card or code yet. What changes when they appear:

- `server.py`: implement `Gr00t2Backend` (its docstring is the contract) and
  a profile for its YAM head; pass the marks to the model natively
  (`steering: native`) instead of drawing them; return `predicted_frames` if
  the API exposes the imagined future.
- `SKILL.md`: weights, licence, io (chunk length, rate), the measured
  latency and memory; the serving command's `--backend gr00t2`.
- Joint frame: check the YAM head's against the station's offset as for N1.7.
- Nothing in `gr00t.act` / `gr00t.run_skill`, the marks or the graphs.

## Limits

- The YAM fine-tune is open-loop only on its card (MSE 0.00279 on a held-out
  repo); the Rho report puts GR00T N1.7 fine-tuned on YAM BusyBox at 53 %.
  Expect low success zero-shot on a new scene and on the station twin.
- Marks are an N1.7 *emulation*: a coloured disc in the image the model was
  never trained on. Measure with and without before relying on them.
- The base model's XDOF tag ships empty absolute statistics for its wrist-EEF
  groups; the server takes their width from the relative statistics so the
  chunk decodes (`repaired_action_widths` in `info`). The XDOF EEF frame is
  not published; the client sends world-frame poses.
- Not a success signal: `succeeded` means the graph's check passed. Put a
  verify skill after the step.

## Licences

Code: this bundle and Isaac-GR00T are Apache-2.0 (the server imports
Isaac-GR00T; nothing is copied - NOTICE). Weights: nvidia/GR00T-N1.7-3B and
the robocurve fine-tune are under the NVIDIA Open Model License (commercial
use allowed by the card, with its terms); nvidia/Cosmos-Reason2-2B likewise,
gated. Qwen/Qwen3-VL-2B-Instruct is Apache-2.0. GR00T 2's licence is not
published.
