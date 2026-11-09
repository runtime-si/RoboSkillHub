---
name: densereward
description: DenseReward reward models (Qwen3-VL-4B fine-tunes, Apache-2.0) run
  locally on a GPU -- task progress 0..1 from ONE RGB frame and the task text
  (0 not started or failed, 1 complete), or at the last of three chronological
  frames with a one-word reason for a drop (miss / collision / fall / smooth /
  failure). No start frame, no video. Use when a judge stage wants a cheap
  progress score right after a step, as a second opinion beside a geometric
  check.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, judge, progress, reward-model, success-detection, gpu]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place]
    weights:
      uri: hf://densereward/densereward-1frame@9796cb2a7be55b339670d8f95fb07841a61487a8
      sha256: 4af2a003692dab911a2d5395cbfa01b735a2449ca49f3b7bc33d83df4da243a6
      license: Apache-2.0 (model card + LICENSE); densereward.progress_with_history loads hf://densereward/densereward-3frame-thinking@e9c4ed9c7662c53ba742db71a52b4ccd5589757d (Apache-2.0, manifest sha256 3f8096c12d0f1185ad0abe6d4b8649eebe377910811f793746925e68e57b0817)
    provenance:
      authors: [Yu Fang, Wanxi Dong, Jiaqi Liu, Yue Yang, Mingxiao Huo, Yao Mu, Huaxiu Yao, Li Erran Li, Daniel Szafir, Mingyu Ding]
      code: https://huggingface.co/densereward/densereward-1frame/blob/9796cb2a7be55b339670d8f95fb07841a61487a8/README.md
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2607.13033
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "densereward"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - densereward.progress: Task progress 0..1 from one RGB frame and the task text (DenseReward-1frame).
    - densereward.progress_with_history: Progress at the last of three chronological frames plus a one-word reason (DenseReward-3frame-thinking).
---

# densereward

The two reward models of [DenseReward](https://dense-reward.github.io/)
(arXiv 2607.13033, UNC Chapel Hill), Qwen3-VL-4B fine-tuned on DROID, Isaac,
robosuite and LIBERO pick-and-place scenes with synthesized failures (miss,
collision, fall, stall). bf16, about 9 GB of weights each, ~10 GB of GPU
memory per resident model. Image in: RGB uint8 `[H, W, 3]`.

| tool | returns |
| --- | --- |
| `densereward.progress(image, task)` | `{progress, found, reason: "", raw, model, timing}` |
| `densereward.progress_with_history(images, task)` | `{progress, found, reason, raw, model, timing}`; `images` oldest first, the current frame last (fewer than three: the oldest is repeated; more: the last three) |

```python
r = ctx.tool("densereward.progress", image=obs_top_rgb, task="put the cube in the black box")
if r["found"] and r["progress"] >= 0.85:
    ...
```

The skill `scoring-step-progress-from-one-frame` wraps it with frame choice,
several samples and exits; use that unless you need the raw score.

## What is sent

Exactly the model cards' quickstart:

- System prompt: the `system_prompt.txt` each checkpoint ships, verbatim
  (the 6-phase reach / grasp / up / move / place / return rubric; "output a
  single float with three decimals").
- User turn: the image(s) then the text; the 1-frame card writes
  `Task: <task>.`, the 3-frame card the bare task. `task_text` adds the
  `Task:` prefix and the full stop when missing.
- Images: qwen_vl_utils 0.0.14's resize (to multiples of 28 px, bicubic) is
  re-implemented here (`card_size`), then the processor's own resize.
  Without it the scores were not the card's: on 20 station frames the plain
  processor path differed from the card by more than 0.1 on 8 (up to 0.57);
  with it they matched the card 20/20 to the printed digit.
- Greedy decoding, 8 / 32 new tokens. Parse: the cards' regexes; the value
  is clipped to 0..1, no float gives `found=False`, progress 0.

## Measured (real YAM station overhead frames, 2026-11-09)

RTX PRO 6000 (rshgpu1), torch 2.8 cu128, transformers 5.2.0, the cards'
tested stack. 98 overhead frames (640 x 480), task text "put the cube in the
black box" (or "thread the white cable through the black ring" for two cable
scenes):

- E1, a real pick-and-place of a 30 mm cube into a box by the left arm
  (station run 2026-11-03), 33 frames 2 s apart from the overhead recording:
  7 before the motion, 17 in reach / grasp / carry / release, 7 with the
  cube in the box (arm above it or parked), 2 after a person took the cube
  back out (not done).
- E2, a missed grasp (cube left beside the box), 6 step frames; E3, a reach
  the station refused, 4 step frames.
- 55 frames of five static station scenes (cube beside the box, cable not
  threaded): nothing done.

Against the hub's `robobrain.progress` (RoboBrain 2.5 / Robo-Dopamine, given
the episode's or scene's first frame as its start frame):

| | densereward.progress | same, median of 3 consecutive frames | progress_with_history | robobrain.progress |
| --- | --- | --- | --- | --- |
| done frames (7): mean, range | 0.74, 0.38-0.93 | 0.75, 0.41-0.88 | 0.65, 0.33-0.98 | 0.20, 0.17-0.25 |
| not-done frames (91): mean, max | 0.19, 0.94 | 0.18, 0.94 | 0.40, 0.98 | 0.05, 1.00 |
| AUROC done vs not done | 0.95 | 0.98 | 0.76 | 0.96 |
| >= 0.85: done passed / not-done passed | 5/7, 3/91 | 5/7, 1/91 | 3/7, 18/91 | 0/7, 1/91 |
| Spearman with time over E1 up to release | 0.31 | 0.36 | 0.07 | 0.93 |
| latency per call (warm) | 0.12 s | 3 calls | 0.24 s | 0.42 s |

(The median column has neighbours for E1 only; elsewhere it equals the single
frame.) What to know:

- **It says "done" on real frames, robobrain.progress cannot.** Robobrain
  ranks the episode well (it has the start frame) but never scored a
  finished frame above 0.25, under in-progress frames' 0.25, so no threshold
  separates done from carrying. DenseReward scored 5 of the 7 cube-in-box
  frames 0.87-0.93, every not-started frame (64) at most 0.34, and dropped
  to 0.53 / 0.00 once the cube was taken back out.
- **It is unstable on this out-of-distribution scene.** Most frames read
  0.33 or 0.37 whatever the arm does (hence the low Spearman); consecutive
  camera frames 1/15 s apart moved the score by 0.40-0.61 on 8 of 33 frames
  (the rest by at most 0.10). Score a few frames, take the median, and treat
  a large spread as "unknown". The three single-frame false "done" scores
  (0.94) were mid-reach and mid-grasp; the median removed two.
- **Two of seven done frames were missed** (0.38-0.53 single frame). It is a
  second opinion, not a success detector: pair it with a geometric check.
- **The 3-frame model added nothing here**: 97 of 98 reasons were "correct"
  (one "collision"), on the missed grasp too, and it scored the static cable
  scenes 0.98. Frames 2 s apart may be off its training spacing (not
  documented); it is kept for trials, not recommended.
- The frames are the station's recorder frames, which carry a date overlay
  (E1) or had a 22 px caption bar cropped off (E2, E3); live
  `robot.get_observation` frames have neither.
- Weights download once (~9 GB each); a warm load took 2-4 s, cold 70-100 s
  including the download.

## Licence

Weights and the copied prompts are Apache-2.0 (model card and LICENSE of both
repos; base model Qwen/Qwen3-VL-4B-Instruct, Apache-2.0). See NOTICE.
