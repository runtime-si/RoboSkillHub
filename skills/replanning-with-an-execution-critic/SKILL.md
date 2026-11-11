---
name: replanning-with-an-execution-critic
description: CLEA's critic - before the graph executes its next action, a VLM
  reads the task, the subgoal, what was done so far and the current camera
  frames and says whether the proposed action is suitable now (YAML "Action
  Suitability" True/False plus feedback); a rejection routes a re-plan of the
  subgoal, repeated rejections a re-plan of the task. Routes continue,
  replan_subtask, replan_task or no_verdict. Use when a graph is about to
  commit to an action whose precondition only the camera can confirm (move on
  to the next brick, release, push) and should re-plan instead when the scene
  says otherwise; one VLM call, ~6 s.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [critic, replanning, recovery, vlm, precondition]
  hub:
    stage: recover
    embodiments: [any]
    version: 0.1.0
    tasks: [lego, pick-place]
    provenance:
      authors: [runtime-si]
      code: https://github.com/sp4595/clea-closed-loop-embodied-agent/blob/eea73b60cf09/CLEA/templates/prompts/Critic.prompt
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2503.00729
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query
  required_inputs:
    task: string
    subgoal: string
    proposed_action: string
  # Optional: history, knowledge, observation (default a fresh robot.get_observation),
  # rejections (earlier rejections for this subgoal), max_rejections (2), model.
  produces_outputs:
    feedback: string
    rejections: int
  exit_conditions:
    continue: The critic found the proposed action suitable now.
    replan_subtask: The critic rejected it (feedback says why and how to improve); fewer than max_rejections rejections so far for this subgoal.
    replan_task: The critic rejected it and the subgoal has now been rejected max_rejections (2) times.
    no_verdict: No camera frame with RGB, vlm.query failed, or the reply had no "Action Suitability".
  canonical_scripts:
    - critique_action: scripts/critique_action.py
  examples:
    - title: Critique the next action before executing it
      path: examples/canonical_subgraph.json
---

# replanning-with-an-execution-critic

CLEA (arXiv 2503.00729) separates an actor that proposes the next action from a
critic that checks it against the current view before execution. This skill is
the critic and its escalation, not the agent.

## Live check: as a "step done, move on?" gate on #155's 75 LEGO placements

Proposed action "the current subgoal is complete: move on to the next brick",
the subgoal's instruction, a one-line history, the last frame (after the
pull-up probe); gemini-3-flash-preview through vlm.query, BrickSim's verdict as
truth; median 6.1 s per call:

| set | `continue` on successes | `continue` on failures (false done) | no_verdict |
|---|---|---|---|
| main view (11 ok / 14 failed) | 7/11 | 5/14 | 1 |
| other view (11 / 14) | 7/11 | 8/14 | 1 |
| gripper force 1.0 (3 / 22) | 2/3 | 0/22 | 4 |

Failures passed: not seated 6 of 20, structure damaged 5 of 8, partially seated
2 of 3; all lost / missed grasps rejected. Against #155 / #162 on the same
units: the Gemini step judges passed 10-11 of 14 failures (more), RynnValue-8B
at 2 s passed 0-1 (fewer, with 3-5 of 11 successes), the #162 fused judge with
geometry passes none. The critic is a precondition check, not a seating
detector: gate seating with geometry.

## Changes from CLEA

Adapted from `CLEA/templates/prompts/Critic.prompt` and
`CLEA/Modules/Critic.py` (Apache-2.0, `LICENSE.clea`): one hub vlm.query call
with the camera frames; the household-simulator lines (two-robot hand-off,
navigation points, API list, "give less False") are dropped; optional
sections only when given; the verdict and a rejection counter become exits; a
reply without a verdict routes no_verdict (upstream raised).
