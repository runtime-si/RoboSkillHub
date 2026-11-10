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
    protocol: http
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
marks = ctx.tool("gr00t.marks_from_points", observation=obs, points=[
    {"kind": "grasp", "point": brick_center},
    {"kind": "release", "point": seat_position},
])["marks"]
r = ctx.tool("gr00t.run_skill", instruction="Put the red brick on the plate.",
             marks=marks, max_steps=480,
             success_check={"tool": "robot.get_gripper", "field": "position", "equals": 1.0})
r["route"]  # succeeded | max_steps | unsafe_action
```

`examples/` has a LEGO GaP graph that plans the brick and its seat with
existing hub skills, hands the place-and-press step to `gr00t.run_skill` with
marks, and checks it with `verifying-brick-seated-by-height`.

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
  a twin) - or a callable from Python. `control_hz` interpolates the 30 Hz rows
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
(or `zmq://<host>:5555` with `--transport zmq`). `uv sync` in a plain clone
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

PENDING

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
