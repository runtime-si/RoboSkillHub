---
id: vflash-2026
title: "VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding"
authors:
  - Yu-Cheng Chou
  - Shihao Wang
  - Jin Huang
  - Chi-Pin Huang
  - Chiao-An Yang
  - Xianzhe Fan
  - Jing Wang
  - Zhiqi Li
  - Yuanguo Kuang
  - Ruijie Zheng
  - Johan Bjorck
  - Sifei Liu
  - Hang Su
  - Abhishek Badki
  - Lei Zhang
  - Alan Yuille
  - Linxi "Jim" Fan
  - Jan Kautz
  - Zhiding Yu
year: 2026
arxiv: "2611.08449"
url: https://arxiv.org/abs/2611.08449
code: https://github.com/NVlabs/Eagle/tree/06edcff802dbdd167d196c7a1c5b44929e07d951/Embodied
license: Apache-2.0
tasks: [pick-place, cable-routing, bimanual]
derived: [skills/requesting-a-typed-visual-plan]
---

# VFlash

## The method

A vision-language model that plans for a robot answers with visual plans:
points, boxes, trajectories. Written one coordinate token at a time, a
trajectory is the slow part of every step. VFlash uses the fact that such an
answer's structure is known before any coordinate is: the model first writes a
tool call that declares the primitive (its type, 2D or 3D or both, one arm or
two, and its length, which fixes the number of waypoints), and a five-layer
drafter reading the frozen model's features then fills the whole block in one
forward pass. Three parts make up the planning procedure:

- **The primitive is declared, then predicted.** A robot plan is a short
  sub-task text and, per gripper, ten waypoints: image coordinates in 0..1000
  and metric points in the camera frame on a 1 cm grid, resampled by arc
  length over a nominal five seconds, "path geometry rather than fixed-time
  control samples". A dual-arm block is the left arm's layout followed by the
  right arm's, in the same answer.
- **A structurally valid block is accepted as it is.** Every delimiter in its
  declared position, a number at every coordinate position, the waypoint count
  the tool call declared: no agreement with the autoregressive model's own
  tokens is asked for. A block that fails is produced again by the
  autoregressive model under the same constraints. The evaluation scorer
  "requires the expected arm identities and exactly ten points per arm, without
  interpolation or repair".
- **The plan is renewed on a period.** The action model (a world action model
  reading the planner's hidden states at the question, the sub-task and the
  trajectory) executes 20-action chunks and gets a new plan every three, 60
  control steps.

The paper reports, for open-loop 3D robot trajectories on InternA1, AgiBot and
MolmoBot, a mean waypoint error (ADE) of 0.060-0.065 m, 0.036-0.042 m and
0.052-0.055 m over its three training variants (the autoregressive target:
0.101-0.153 m, 0.083-0.105 m, 0.066-0.076 m), with every reported
configuration returning a valid trajectory for every record. Plan decoding on
MolmoSpaces takes 258 ms on a Jetson Thor and 38 ms on an H100 (3,108 ms and
412 ms autoregressively). A world action model conditioned on the plans reaches
70.2 % average success on four MolmoSpaces suites in simulation, against 62.6 %
for the same action model with its own text encoder and no plan.

## What the hub uses

`requesting-a-typed-visual-plan` keeps the procedure and none of the model.
The request declares the arms and the waypoint count (ten by default) before
anything is asked, both arms of a bimanual plan are one request, coordinates
are 0..1000 across the image, the answer carries a sub-task text, an answer is
taken on its structure and refused whole when the structure is wrong, a
refused answer is produced again by the next planner of a ladder, and an
earlier plan is handed out again only for a bounded time.

What it changes, because the planners are the hub's general ones (`vlm.query`,
`robobrain.trace`) and not a model trained on annotated trajectories:

- **Structure is not enough.** An answer must also agree with what is
  measured: its own start on the arm's tool centre in the image, a depth under
  its endpoint, its metric part (when it has one) within 5 cm of the measured
  depth, and two arms' tool centres at least 0.15 m apart along the plan. On
  real station frames a small model answered in perfect structure and ended
  104-223 px from its target on four of five tasks (the skill's SKILL.md,
  Measured).
- **The plan is 2D unless its 3D agrees with the depth image.** VFlash's
  2D+3D primitive is one prediction; here the metric part of a planner's
  answer is passed on only when it matches the measured depth, and
  `following-a-visual-plan` lifts the pixels otherwise.
- **A waypoint outside the image is a refusal.** VFlash keeps such a
  waypoint's metric point and marks its pixel out of frame.
- **Renewal is by age and by what moved**, not by control steps: the hub's
  planners took 0.5-15 s a call, so a plan is reused while it is under 30 s
  old and the hands and the surfaces under its endpoints are where they were.

The plan's reader is `following-a-visual-plan`, by geometry. VFlash's reader
is an action model trained to read the planner's hidden states; nothing in the
hub does that.

When the checkpoints are released, the model belongs in the hub as a tool
that returns the visual-plan schema (`tools/molmo-motion/SKILL.md`), named in
this skill's `planners`.

## When it is the wrong tool

- **The weights are not public.** As of 2026-11-13 the repository says the
  target and drafter checkpoints "will be released later" (code Apache-2.0,
  models under an NVIDIA licence), and the Eagle Embodied base checkpoint its
  training starts from was not found on the Hugging Face Hub. Nothing here
  runs VFlash.
- **Its speed is the drafter's.** A plan every 60 control steps needs a plan
  in tens of milliseconds. Asked of a general VLM, the same protocol takes
  seconds: it fits a step boundary, not a control loop.
- **Its accuracy is for where to go, not for contact.** 3.6-6.5 cm mean
  waypoint error on its own benchmarks, on a 1 cm grid, on robots it was
  trained on (simulated Franka for the downstream result; no YAM in the listed
  data). An insertion needs its last centimetres from another skill.
- **The downstream gain needs a trained reader.** The 62.6 to 70.2 % is a
  world action model post-trained to read the plans, in simulation; a
  geometric follower does not inherit it.
