---
name: detecting-subtask-success
description: Checks whether a subtask is done from the latest frame of every
  camera at one moment - sends them to a VLM with the Gemini Robotics SDK's
  four guided questions (object states, relations, required end state,
  differences) and reads its "FINAL ANSWER yes / no", once or polling fresh
  frames until yes or a timeout. Use when a graph needs a yes/no "is <subtask>
  done now?" gate after a step or while a policy runs, with only cameras and
  no frame from before the step; use judging-step-with-before-now-vlm instead
  when the decision needs a BEFORE frame or a recovery choice.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, success-detection, vlm, guided-questions, multi-camera]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, insertion, cable-routing]
    provenance:
      authors: [runtime-si]
      code: https://github.com/google-deepmind/gemini-robotics-sdk/blob/26b49f1a15b96f1e65b05160f5b991d6167aba74/safari_sdk/agent/framework/tools/success_detection.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query
  required_inputs:
    subtask: string
  produces_outputs:
    success: bool
    answer: string
    checks: int
    follow_ups: int
    errors: int
    elapsed_s: float
    reason: string
    raw: string
  exit_conditions:
    succeeded: The VLM's reply ended in FINAL ANSWER yes for the latest frames (on the first check, or on any check before timeout_s).
    not_yet: Asked once (timeout_s 0) and the reply was FINAL ANSWER no, or had no FINAL ANSWER even after the follow-up (read as no, as upstream did).
    timed_out: timeout_s > 0 passed without a yes; `checks` replies were all no or unparsed.
    no_view: Nothing was judged - no camera had RGB in any observation, or every vlm.query call raised; `reason` says which.
  canonical_scripts:
    - detect_success: scripts/detect_success.py
  examples:
    - title: Ask once whether a subtask is done
      path: examples/canonical_subgraph.json
---

# detecting-subtask-success

A yes/no success gate: is the subtask done in what the cameras see now?

## This skill or judging-step-with-before-now-vlm

| | detecting-subtask-success (this) | judging-step-with-before-now-vlm |
|---|---|---|
| frames | the latest frame of each camera, at one moment (start frames optional) | a BEFORE observation captured before the step, and NOW |
| question | "is `<subtask>` done?" | "where is this subgoal in a plan, and what next?" |
| answer | yes / no | complete / in progress / failed / wrong object / wrong destination |
| repeats | built in: poll until yes or `timeout_s` | one judgment per call |

Use this one when the end state is visible on its own ("the cube is in the
box", "the cable end is held"), when nothing was captured before the step,
or when you need to wait for success while a policy or a slow motion runs.
Use the before/now judge when success is a change (moved 8 cm, pushed
aside) that a single frame cannot show, or when the graph has recovery
edges for a wrong object or a wrong destination.

## How a graph uses it

```text
<step: skills / policy> -> detect (this skill) -> succeeded: next step
                                               -> not_yet / timed_out: retry or recover
                                               -> no_view: check the cameras / the VLM
```

With `timeout_s` 0 (the default) it asks once about `now` (or a fresh
`robot.get_observation`) and routes `succeeded` or `not_yet`. With
`timeout_s` > 0 it reads a fresh observation, asks, and on "no" waits
`interval_s` (upstream's 0.2 s) and asks again about the next one, until a
yes (`succeeded`) or the deadline (`timed_out`). The deadline is checked
between checks, so the last check can end up to one reply (2-13 s, see below)
after it. Frames identical to the last judged ones (same RGB bytes and the same
observation `timestamp` / `stamp` / `time`, if there is one) are not asked about
again. This is upstream's stale-image guard, in a synchronous loop.

## What it sends

One `vlm.query` per check. Upstream's prompt, in its order: the optional start
images, the current image of every camera, `The robot's task is: <subtask>`,
"answer the following questions first", the four guided questions with their
worked spoon-in-the-compost-bin example, the leniency line, the FINAL ANSWER
format, and "limit your response to 50 words". The hub's `vlm.query` attaches
images after the text, so each image is announced by number and camera name
where upstream put it inline.

- `cameras` picks and orders frames by name. By default every camera with RGB
  is sent, in the observation's order.
- `start`: an observation from the start of the subtask. Its frames go first
  under upstream's "what the robot saw at the start of the task". This is
  upstream's default (`sd_use_start_images`); here it is off unless given, so
  the check is single-moment.
- `lenient` (true, upstream): "partially completed or it may have completed
  (cannot be determined) counts as completed". `lenient=False` swaps in a
  strict line (only a clearly visible end state counts). On the frames below
  both readings gave no false yes. Choose strict when a false yes costs more
  than one more check.
- `word_limit` (50, upstream's default; `None` drops the line), `model`
  (per-call model, otherwise the tool's configured one).
- `follow_up` (true): when a reply has no FINAL ANSWER, ask once more with
  the same images and the cut-off reply, for the final line only. See below
  for why. With it off, or when the follow-up has no FINAL ANSWER either, the
  reply reads as no, as upstream read it.

Parsing: the last `FINAL ANSWER: yes|no`, case-insensitive, with `**` or other
punctuation between them, or upstream's `\boxed{yes}`. `answer` is `yes`, `no`
or empty (no FINAL ANSWER).

## Live check

The skill's `run()` was called through the hub's `vlm.query`, using its OpenAI-compatible provider
pointed at Gemini's endpoint. It ran on 14 questions with known answers:

- **10 on twin frames, two cameras each.** The YAM station twin's calibrated
  overhead camera (`top`, 640 x 480) and a second view across the table
  (`front`). A scripted grasp staged the 30 mm cube through six states:
  on the table, the open hand 8 cm over it, held 15 cm up, held over the blue
  box, dropped in the box with the hand still over it, and the hand moved
  away. Ground truth came from the cube's simulated pose.
- **4 on real station overhead frames** (`g_sota_frames_20261103-1421`: a
  white cable on a pegboard, both grippers empty at rest).

| frames | subtask | expected |
|---|---|---|
| sim: cube on the table | put the small grey cube in the blue box | no |
| sim: cube on the table | pick up the small grey cube | no |
| sim: open hand over the cube | pick up the small grey cube | no |
| sim: cube held 15 cm up | pick up the small grey cube | yes |
| sim: cube held 15 cm up | put the small grey cube in the blue box | no |
| sim: cube held over the box | put the small grey cube in the blue box | no |
| sim: cube held over the box | pick up the small grey cube | yes |
| sim: cube in the box, hand over it | put the small grey cube in the blue box | yes |
| sim: cube in the box, hand away | put the small grey cube in the blue box | yes |
| sim: cube in the box, hand away | pick up the small grey cube | no |
| real: frame_01 | pick up the white cable | no |
| real: frame_08 | pick up the white cable | no |
| real: frame_04 | put the white cable on the floor | no |
| real: frame_11 | place the black pegboard on the table | yes |

Each question was asked twice per model and per reading:

| model, reading | correct | false yes | false no | follow-ups | latency median / max |
|---|---|---|---|---|---|
| gemini-robotics-er-2-preview, lenient | 28/28 | 0 | 0 | 2 | 2.2 s / 10.2 s |
| gemini-robotics-er-2-preview, strict | 28/28 | 0 | 0 | 1 | 2.1 s / 8.9 s |
| gemini-3-flash-preview, lenient | 28/28 | 0 | 0 | 3 | 3.7 s / 12.8 s |
| gemini-3-flash-preview, strict | 27/28 | 0 | 1 | 4 | 3.4 s / 11.5 s |

A check with no follow-up took a median 2.1 s (max 3.0 s) on ER-2 and 3.4 s
(max 5.3 s) on Flash. A follow-up made it 7-13 s.

Polling (`timeout_s`, strict), with a fresh observation cycling through the frames:

| frames cycled | subtask | timeout | ER-2 | Flash |
|---|---|---|---|---|
| table, open hand over cube, held | pick up the small grey cube | 90 s | succeeded on check 3, 7.8 s | succeeded on check 3, 16.2 s |
| table, open hand over cube | put the small grey cube in the blue box | 20 s | timed_out, 9 checks, 21.3 s | timed_out, 5 checks, 22.6 s |

What to know:

- **Without the follow-up, the replies got cut off.** In a first pass (one
  repeat, no follow-up), 5 of 56 replies stopped mid-sentence ("Q3: The cube
  should be") before any FINAL ANSWER and read as no. ER-2 went 14/14 lenient
  and 13/14 strict; Flash went 13/14 in both readings. Flash's 90 s poll
  until held timed out after 21 checks without a yes (the replies were not
  kept; the single held check was cut off the same way). The cause is
  that the hub's `vlm.query` caps output at 1024 tokens, thinking included.
  Upstream set no output cap. The follow-up fixed every cut-off but one:
  there, Flash continued its answers instead of giving only the line and was
  cut again (the one false no above).
- **No false yes** in 112 single checks and 4 polls, in either reading. That
  includes the cube held right over the box, which upstream's lenient reading
  could have counted as partial success for "put in the box".
- Cut-off replies came mostly on the held cube. There the two views disagree:
  the overhead camera sees the hand from above, hiding the cube, while the
  front view shows it in the fingers. The models spend their words on that.
  More cameras help the answer but lengthen the reply.
- The real frames are static scenes (no real arm moved for this check). The
  twin has one camera on the real station. `front` is a render-only second
  view, standing in for a wrist or side camera.

## Changes from the Gemini Robotics SDK

Adapted from `SubtaskSuccessDetectorV4` and `_GUIDED_THINKING_SD_QUESTIONS`
in `safari_sdk/agent/framework/tools/success_detection.py`, Apache-2.0:

- The event-bus image buffer and the asyncio background queries every 0.2 s
  become a synchronous loop in `run()`: one observation, one query, wait,
  repeat. The stale-image guard becomes "skip frames identical to the last
  judged ones".
- Default is one check (`timeout_s` 0). Upstream polled for 60 s
  (`sd_timeout_seconds`). The timeout is now a graph exit (`timed_out`)
  instead of a `timeout` flag in the tool response.
- Start images are optional and off by default (upstream: on). History frames
  (`sd_num_history_frames`, 0 upstream) are not ported.
- Images are announced by number in the text, because the hub tool appends
  them after the prompt.
- A strict reading is added beside upstream's lenient one.
- The FINAL ANSWER parse is case-insensitive and tolerant of markdown.
  Upstream matched four literal strings.
- The follow-up for a reply with no FINAL ANSWER is new. Upstream read such a
  reply as no.
- A failed `vlm.query` (or no RGB at all) routes `no_view` instead of being
  read as no. Upstream logged it and kept polling, as this loop does until
  the deadline.
- Dropped: the external success-signal and cancellation events (a graph stops
  a node by not running it), the dry-run flag, and the evergreen client.
  Temperature (0) and the output cap come from the hub's `vlm` tool, not from
  upstream's `sd_*` config.
