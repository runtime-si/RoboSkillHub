---
name: rynnvalue
description: RynnValue value model (RynnBrain / Qwen3-VL, 8B or 4B, Apache-2.0)
  run locally on a GPU -- reads a task instruction and up to 8 frames of a
  clip (oldest first) and returns, for every frame, the predicted remaining
  time in seconds until the task is complete, plus a generated analysis with
  Match (clip fits the instruction) and Success verdicts. One forward pass,
  ~0.2 s for 8 frames on the 8B. Use when a judge stage has the frames of a
  step (start to now) and wants a ranking-quality "how close to done" signal
  -- an alternative backend to robobrain.progress / densereward.progress for
  progress and stall scoring; threshold it on your own frames before routing
  on it.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, judge, progress, value-model, reward-model, success-detection, gpu]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, lego]
    weights:
      uri: hf://Alibaba-DAMO-Academy/RynnValue-8B-Quantile@0fb3ac12c73844ea1abaf4f767c895aae248a9ec
      sha256: 38aaf511983dfbfd6c8f26f5e299e0321e1180aa3738c72def3086a7991fa288
      license: Apache-2.0 (model card); GAP_RYNNVALUE_MODEL=4b loads hf://Alibaba-DAMO-Academy/RynnValue-4B-Quantile@d48ffefcefaf8b5381ae1e412e9a62faf1456bcd (Apache-2.0, manifest sha256 8afd41c9d5ec9546a68003407a848adaf9383e8bd0450bd92a497e9a347be6de)
    provenance:
      authors: [Dongchi Huang, Hongyin Zhang, Bohan Hou, Siteng Huang, Zhian Su, Hang Guo, Tong Lu, Jiahao Tang, Jianfei Yang, Donglin Wang, Peixi Peng, Mingxiu Chen, Deli Zhao, Xin Li]
      code: https://github.com/alibaba-damo-academy/RynnValue/blob/cd29294f1ac56bc817074bd665ba92b630936ad7/rynn_infer/inference.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2608.09853
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "rynnvalue"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - rynnvalue.remaining_time: Remaining seconds to completion for each frame of a clip (oldest first, at most 8 used) plus Match / Success from the model's analysis.
---

# rynnvalue

[RynnValue](https://github.com/alibaba-damo-academy/RynnValue) (Alibaba DAMO
Academy, arXiv 2608.09853) predicts **temporal distance**: for each frame of a
clip, the seconds left until the instruction is done (a 256-bin quantile head,
decoded as the expected bin centre). Trained on ~3M clips (OXE, AgiBot,
RoboTwin, EgoDex, ...) without progress or preference labels; RBM-EVAL-OOD
Kendall tau 0.704 (8B) / 0.679 (4B) on the card, above Robometer (0.655).

| tool | returns |
| --- | --- |
| `rynnvalue.remaining_time(images, instruction, robot_description=None, camera_description=None, analysis=True)` | `{remaining_s, final_remaining_s, delta_s, entropy, match, success, description, analysis, frames_used, model, timing}` |

```python
r = ctx.tool(
    "rynnvalue.remaining_time",
    images=frames_since_step_start,
    instruction="put the blue brick on the red brick",
)
if r["final_remaining_s"] <= 2.0:  # calibrate on your own frames (below)
    ...
```

- `images`: RGB `[H, W, 3]`, oldest first. More than 8 are subsampled
  uniformly with the last always kept (upstream: `np.linspace` over the
  prefix); frames are downscaled to 640 px on the long side (upstream default).
- The released checkpoints refuse a prompt without the embodiment / camera
  meta block; when neither description is given the tool sends
  "a robot arm with a parallel-jaw gripper" / "a fixed third-person camera
  looking down at the table".
- Inference attention is the causal mask (`sdpa`), which the card's headline
  numbers use; upstream's inference.py forces the training-time isolation
  mask, which the card reports ranking worse (`GAP_RYNNVALUE_ATTN` to change).
- bf16, ~20 GB of GPU memory for the 8B with 8 frames (4B ~10 GB). Load 4 s
  warm; value pass 0.2 s, analysis generation +0.8 s (`analysis=False` skips
  it).

## Measured: LEGO placements in Isaac Sim (BrickSim), 2026-11-09

GCP pool (RTX PRO 6000), torch 2.8.0+cu128, transformers 4.57.6. Frames from
the hub's LEGO evaluation (`src/roboskillhub/tasks/assembly_eval`, RoCo /
BrickSim on Isaac Sim 5.1, the privileged goal_fsm baseline with its injected
faults), with a 640 x 480 RTX camera added ~0.24 m from the base plate
(`examples/tasks/assembly_eval/judge_compare/`). One unit per brick placement:
the 8 recorded snapshots (pick start, grasped, before place, released,
retreated, settled, after the pull-up probe), the step's instruction, and
BrickSim's own verdict (clutch engaged, seated pose, holds under the probe) as
truth. 75 units in three sets:

- **main**: all 16 cases (5 scenes, 7 fault types), default camera, the
  runner's gripper force: 25 placements, 11 succeeded.
- **other view**: the same episodes (deterministic physics, same verdicts)
  seen from the opposite side: 25, 11 succeeded.
- **force 1.0**: the same cases with gripper force 1.0, which loses or misses
  most grasps: 25, 3 succeeded.

Failures: not seated 20, structure damaged 8, grasp lost 8, insufficient
support 7, grasp missed 4, partially seated 3. Baselines on the same units:
`densereward.progress` (#124's tool) on the last frame, and the hub's two VLM
judges through `vlm.query` on Gemini: `detecting-subtask-success` (last frame)
and `judging-step-with-before-now-vlm` (pick-start frame -> last frame).

GEMINI_TABLE

What to know:

- **RynnValue-8B on the whole clip ranks best of the scores** (AUROC
  0.75 / 0.92 / 0.94 vs DenseReward 0.45 / 0.55 / 0.14). Fed only the first
  and last frame it drops to 0.49 / 0.64 / 0.73: give it the step's frames.
- **Its Success verdict is useless here: "No" on all 300 calls** (8B and 4B,
  successes included), and Match was "Yes" on 71/75 for the 8B. Route on the
  remaining time, not on the generated text.
- **The scale is seconds of a task it imagines.** Successful placements
  ended at 0.9-3.0 s remaining, failed ones at 1.9-5.1 s (8B); the 4B ended
  everything at 4.7-10.1 s. A threshold of 2.0 s gave 0 false "done" in the
  main and other-view sets (1 of 22 in the force-1.0 set) but passed only 3/11
  and 5/11 successes; 2.5 s passed 6/11 and 10/11 with 4 and 3 false "done".
  These thresholds were read off the same 75 units (in-sample).
- The 4B is not a cheaper substitute on these frames (AUROC 0.46 / 0.54 /
  0.89).
- Mm-level failures (a brick 1.7 mm high, a 9 degree yaw) are where every
  judge struggles; see the per-failure table.

## Licence

Weights Apache-2.0 (both model cards); the modelling code is the checkpoints'
own `trust_remote_code` files at the pinned revisions. The tool's inference
follows upstream `rynn_infer/inference.py` (Apache-2.0); see NOTICE.
