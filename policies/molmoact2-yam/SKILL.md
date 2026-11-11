---
name: molmoact2-yam
description: AllenAI MolmoAct2-BimanualYAM as a vla-yam/1 policy server for the bimanual
  YAM station - one 30-row chunk of absolute joint + jaw targets (left 7, right 7, station
  frame and units) at 30 Hz from the top and two wrist views, the 14-D joint state and an
  instruction. The checkpoint-specific parts (camera order, joint offset, gripper scale,
  norm tag, solver steps, dtype, pinned revision + sha256) live in a profile YAML, so the
  same request and reply serve the twin and the real station. Speaks gr00t-wam/1 too, so
  gr00t.act / gr00t.run_skill drive it unchanged with GAP_GR00T_URL. Use when a graph or a
  pick_object implementation wants a learned bimanual YAM policy trained on AllenAI's YAM
  mixture (tabletop pick-place, dishwasher, box packing) and runs its own verifier
  afterwards. Not for other embodiments, not as a success signal, and not on the real arms
  before the profile's joint offset and gripper mapping are confirmed in the twin.
license: Apache-2.0
compatibility: requires gap>=0.1; the server needs a CUDA GPU with ~12 GB free (bf16) or ~26 GB (fp32)
metadata:
  category: policy
  tags: [policy, vla, molmoact2, yam, bimanual, gpu, long-running]
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    tuned_on: [yam_i2rt]
    arms: 2
    tasks: [pick-place]
    io:
      action: {space: joint_position_absolute, dim: 14, hz: 30}
      state:
        dim: 14
        order: [left_joints_6, left_opening, right_joints_6, right_opening]
      cameras: [top, left_wrist, right_wrist]
    notes: >-
      Station frame and units on the wire; profiles/yam-molmoact2-native.yaml maps them to the
      checkpoint's (model joint = station joint - joint_offset_rad; linear gripper map). Twin
      offset 0 and gripper identity (model 1 open, 0 closed) measured in the twin; real station
      +pi/2 on each base joint (--station real) still PROVISIONAL. Twin cube-in-box 0/10.
      Chunk 30 rows at 30 Hz, execute 25. All three views required; marks ignored.
    version: 0.1.0
    weights:
      uri: hf://allenai/MolmoAct2-BimanualYAM@8dcbed66f2380e4393189c303ea72488eb9e63c2
      sha256: b5a0a14038adb2d6a9ab2ac336ebae0b51d87ea9c4973ccaa10b67383a65c0d3
      license: Apache-2.0 (github.com/allenai/molmoact2 README section 7, "This model is licensed under Apache 2.0"; intended for research and educational use under Ai2's Responsible Use Guidelines). The Hugging Face card carries no license field.
    provenance:
      authors: [Allen Institute for AI (MolmoAct2)]
      code: https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2605.02881
      data: AllenAI bimanual YAM mixture (MolmoAct2-BimanualYAM datasets), absolute joint-pose control
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "server.py", "--profile", "profiles/yam-molmoact2-native.yaml",
              "--port", "{port}"]
    protocol: in-process
    requires_gpu: true
    weights_uri: hf://allenai/MolmoAct2-BimanualYAM@8dcbed66f2380e4393189c303ea72488eb9e63c2
  allowed_tools:
    - gr00t.act
    - gr00t.run_skill
    - gr00t.info
  exit_conditions:
    chunk_ready: The server returned a [30, 14] chunk in the station frame (execute the first execute_steps rows, then ask again).
    refused: The request was malformed or incomplete (a view missing, joints not 14 finite numbers, empty instruction, bad frame or num_steps); HTTP 400, nothing to execute.
    failed: The server did not answer or the model raised; HTTP 500 or a transport error. The subgraph's on_error exit.
---

# molmoact2-yam

[MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) (Molmo2-ER
VLM + a flow-matching action expert, fine-tuned by AllenAI on its bimanual YAM mixture with
absolute joint-pose control) behind the hub's YAM policy protocol. The server takes YAM
terms - three camera roles, 14 joint/jaw values in the station's frame, an instruction - and
returns a chunk in the same terms; everything checkpoint-specific is in the profile.

## Why a sibling bundle (and not a backend inside policies/gr00t)

- **Environment.** gr00t's server runs in Isaac-GR00T's pinned environment (its own torch,
  transformers 4.57.3, flash-attn). MolmoAct2's remote code wants plain transformers 4.57 +
  torch, nothing from Isaac-GR00T; one venv for both would pin each to the other's versions.
- **Reuse, not a fork.** `server.py` loads `../gr00t/server.py` (or `$GR00T_BUNDLE`) and uses its
  request parsing (`ActRequest`), image decoding, `Service`, HTTP and ZMQ transports. It adds
  only the profile, the frame mapping and the `molmoact2` backend (~250 lines).
- **Catalogue.** Its own weights licence, numbers and acceptance records per implementation
  (`policy:molmoact2-yam` in the `pick_object` contract).

## Protocol: vla-yam/1 (= gr00t-wam/1 + station frame)

`GET /info`, `POST /act`, `POST /reset` over HTTP (JSON), or Isaac-GR00T's msgpack framing over
ZMQ (`--transport zmq`); `/info` says `protocol: vla-yam/1, compatible: [gr00t-wam/1]`.

```python
POST /act {
  "instruction": "pick up the red cup",
  "images": {"top": img, "left_wrist": img, "right_wrist": img},   # {"jpeg"|"png"|"npy": b64}
  "joints": [l_j1..l_j6, l_grip, r_j1..r_j6, r_grip],             # station frame and units
  "frame": "station",          # default; "model" skips the profile mapping (gr00t-wam/1 clients
                               # that apply GAP_GR00T_JOINT_OFFSET themselves)
  "num_steps": 10              # optional, flow-matching solver steps
}
-> {"actions": float[30][14], "dt": 0.0333, "dt_ms": 516.0, "execute_steps": 25,
    "profile": "yam-molmoact2-native", "station": "twin", "frame": "station",
    "timing_ms": {...}, "gpu_mem_mb": {...}, "model": ..., "revision": ..., "steering": "none"}
```

A gr00t client (`gr00t.act`, `gr00t.run_skill`) works against it with
`GAP_GR00T_URL=http://host:port`; leave `GAP_GR00T_JOINT_OFFSET` unset when the profile carries
the offset (`--station real`), or the offset is applied twice.

## Profile (`profiles/yam-molmoact2-native.yaml`)

| field | value | source |
|---|---|---|
| checkpoint | allenai/MolmoAct2-BimanualYAM @ 8dcbed66, 5 shard sha256 | HF API (LFS oids); checked on load (`--verify-sha256 cache` = HF blob names, `full` = hash) |
| cameras | top, left_wrist, right_wrist | norm_stats.json camera_keys top / left / right; model card |
| state | left j1-6, left gripper, right j1-6, right gripper | norm_stats names `left_joint_0.pos` .. `right_gripper.pos` |
| joint_offset_rad | twin 0; `stations.real` +pi/2 on both base joints | twin 0 measured (rest pose held within 0.05 rad; +pi/2 is pulled back 0.46-0.63 rad in the first chunk); real +pi/2 from #173 (molmo-yam-joint-mapping.json), **still provisional** |
| gripper | station 1 open / 0 closed -> model 1 open / 0 closed | measured: data actions 0.007..0.993, 0.99 at rest; twin open jaws fed as 1 are held at 0.99, fed as 0 are driven back to 0.96; a grasp attempt commands 0.02 |
| action | abs_joint, 14, horizon 30, execute 25, 30 Hz | norm_stats action_horizon 30, control_mode absolute joint pose |
| extra | norm_tag yam_dual_molmoact2, num_steps 10, bfloat16, continuous | model card |

## Run

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv/bin/python -e .        # transformers 4.57, pyyaml, pyzmq, msgpack-numpy
.venv/bin/python server.py --profile profiles/yam-molmoact2-native.yaml --port 5560            # twin
.venv/bin/python server.py --profile profiles/yam-molmoact2-native.yaml --station real --port 5560
.venv/bin/python server.py --backend fake --port 5561                                           # CPU
```

`server.py` needs `../gr00t/server.py` beside it (the repo layout) or `GR00T_BUNDLE=<dir>`.
Options: `--dtype float32` (the authors' reported setting, ~26 GB), `--cuda-graph`,
`--warmup N` (random-view calls before binding, default 3).

## Measured (rshgpu6, RTX PRO 6000 Blackwell 96 GB, torch 2.11 cu128, transformers 4.57.6, bf16)

`eval/smoke_server.py` on the model card's sample (assets/sample_{top,left,right}_rgb.png
640x360, its 14-D state and task); raw numbers in `eval/out/`.

| | value |
|---|---|
| load (snapshot cached, full sha256 of 21.8 GB included) | 20.9 s |
| GPU memory after load / peak / reserved | 10.5 / 11.3 / 12.1 GB |
| warm-up calls | 1026, 856, 512 ms |
| chunk latency over HTTP, 20 calls (client wall) | mean 562 ms, p50 528, p90 708 |
| model time per chunk (inside the server) | mean 551 ms, p50 516 |
| AllenAI host_server_yam.py, same GPU, mean of 5 | 503 ms |
| actions vs AllenAI's server, same seed, no autocast | identical (max abs diff 0.0) |
| actions vs AllenAI's server, same seed, with autocast (default) | max 0.0088 rad joints, 0.0039 gripper |
| spread from the flow noise alone (seed 0 vs 1) | max 0.33 rad joints |
| `--cuda-graph`: chunk over HTTP, 20 calls | mean 425 ms, p50 359, p90 557 (AllenAI's: 375 ms); peak 11.4 GB |

The chunk takes 0.35-0.55 s against 25 executed rows = 0.83 s at 30 Hz, so a synchronous loop
keeps up. Repeated calls with the same seed are bit-identical; the parity holds with
`--cuda-graph` too (`eval/out/smoke_bf16_cudagraph.json`).

## Open

- Real-station joint offset (+pi/2 base) unverified; the twin cannot check the real arm's zero.
- Twin closed loop (A2, official server, cube-in-box, 10 seeds, 600 rows, top view cropped to the
  640x360 table band): 0/10, no jaw closes, nearest TCP to cube median 74 mm (min 17). Raw 640x480
  top view, left/right swap, +pi/2 base, wrist views turned 180 deg, zoomed top view, wrist camera
  tilted +-35 deg: all 0 (one grasp attempt, missed). π0.5 (policies/gr00t pi05) on the same block: 0/10 raw and cropped
  (nearest TCP median 250 / 276 mm).
  Open loop on AllenAI's own frames (19012026-block-13 ep0, 102 chunks) the model tracks the demo:
  MSE 0.0045 vs hold-still 0.0086 (π0.5 0.0048). So the gap is the twin's cameras (wrist mount sees
  the fingertips at the top and the floor; AllenAI's sees them at the bottom over the table) and scene
  look - fix the twin wrist mount, then fine-tune on station data.
- The real station executor (`robot.stream_joints`) is E's; nothing here moves an arm.

## License

Code: Apache-2.0 (this bundle; the backend follows AllenAI's Apache-2.0
`examples/yam/host_server_yam.py`, which `eval/smoke_server.py` downloads for the parity check -
no code copied). Weights: Apache-2.0 per the molmoact2 README, for research and educational use
under Ai2's Responsible Use Guidelines; the Hugging Face card has no license field.
