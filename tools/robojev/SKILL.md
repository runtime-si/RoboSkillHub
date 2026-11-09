---
name: robojev
description: RoboJev-27B success judge (josefchen/RoboJev-27B, Apache-2.0) -
  given the task sentence, frames of one attempt from a main camera (start to
  end) and the final frame of other cameras, one forward pass reads the
  false / true answer and the released calibration returns P(task done). Use
  when a workflow must decide after a step whether it succeeded from cameras
  alone, or grade recorded attempts; needs a GPU with ~56 GB.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [verification, success-detection, reward-model, vlm, judge, gpu]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, assembly, sorting]
    weights:
      uri: hf://josefchen/RoboJev-27B@e089a9b69a365744b80765057d86583647af95a9
      sha256: WEIGHTS_SHA
      license: Apache-2.0 (model card; base autotrust/JEV-27B-VL, Apache-2.0, built on Qwen); the card notes the training-data licence chain is not fully reconstructed
    provenance:
      authors: [Josef Chen]
      code: https://github.com/josefchen/robojev/blob/79a5e2f/robojev/robojev.py
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2610.33718
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "robojev"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - robojev.judge: Calibrated P(task completed) for one attempt from its main-camera frames (start to end) and final frames of other cameras.
---

# robojev

[RoboJev](https://arxiv.org/abs/2610.33718) is a Qwen3.5-architecture VLM
(JEV-27B-VL with a LoRA) trained to answer one question - did the robot
complete the task? - from ten frames of the attempt plus the final views of
other cameras. It answers with two tokens; their log-probabilities, a bias
and a temperature fitted on held-out runs give P(done). The model card
reports 0.993 AUROC / 95.2 % balanced accuracy on 268 held-out SO-101 runs
and 0.890 AUROC on unseen robots (RBM-1M OOD).

```python
r = ctx.tool(
    "robojev.judge",
    task="Put the cube in the box.",
    frames=clip_from_top_camera,
    views={"left_wrist": wrist_now},
    camera="top",
)
r["p_done"]  # calibrated P(task done)
```

## Tool

`robojev.judge(task, frames, views=None, camera="main")` returns
`{p_done, logit_false, logit_true, frames_used, views_used, prompt_state, model}`.

- `frames`: RGB uint8 frames of the attempt from one camera, start first,
  end last, any number - ten are taken evenly (first and last included,
  repeated when there are fewer, as the RoboJev client does).
- `views`: camera name -> that camera's last frame.
- `camera`: the main camera's name; names containing `top`, `front` or
  `wrist` get the model's own wording ("overhead camera"...), others
  "<name> camera".
- Every image goes to a 288 px short side through a JPEG round trip, as the
  client sends them.

## How it is run

The released protocol (the client's `DENSE_QUESTION`, the `[kind] noul`
frame, `decision.json`'s tokens 3721 / 1802, bias and temperature 0.9),
evaluated in this process with transformers: the tokenizer, the PIL image
processor (AutoProcessor would build the video processor, which needs
torchvision), the image placeholders expanded per image as the processor
does, `mm_token_type_ids` for the multimodal RoPE, one forward pass with
`logits_to_keep=1`. Upstream serves the model with vLLM and restricts the
answer to the two tokens; that renormalises the pair, which leaves their
difference - and P(done) - unchanged. `GAP_ROBOJEV_URL` sends the same
request to such a vLLM server instead.

## Install

```bash
uv sync --project tools/robojev      # torch + transformers >= 5.19
```

Weights: 12 safetensors shards, 51 GB, from Hugging Face on first call or
`gap skills check --download` (pinned to the evaluated revision e089a9b).
bf16 on one GPU: 55.6 GB allocated. Env knobs: `GAP_ROBOJEV_MODEL`,
`GAP_ROBOJEV_DEVICE`, `GAP_ROBOJEV_URL` / `GAP_ROBOJEV_SERVED`.

## Measured

MEASURED

## Limits

- **Calibrated on SO-101, not YAM.** The card's ECE is 0.032 on SO-101 and
  0.148 on unseen robots; refit the threshold on labelled YAM steps.
- **Ten frames can miss brief events**; a clip that stops before the outcome
  is "not done".
- 56 GB of GPU memory; transformers' pure-PyTorch fallbacks for the linear
  attention layers (no `causal_conv1d` / `flash-linear-attention`) are slower
  than upstream's vLLM (0.26 s on an H100).
- The card notes that part of the training-data licence chain is unresolved;
  the weights themselves are declared Apache-2.0.

## Provenance

`tools.py` adapts the RoboJev client (github.com/josefchen/robojev at
79a5e2f, `robojev/robojev.py`, Apache-2.0, Copyright Josef Chen;
LICENSE.robojev): the prompt strings and state text, `camera_name`,
`pick_frames`, the 288 px JPEG round trip, the vLLM request and the
calibrated read-out with `decision.json`'s values. The header of `tools.py`
lists the changes.
