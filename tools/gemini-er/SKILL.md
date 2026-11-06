---
name: gemini-er
description: Open-vocabulary 2D detection and pointing via the Gemini
  Robotics-ER API — one call returns pixel-space bounding boxes with labels and
  scores, or pixel points, for a text query. Use when a workflow needs a
  detection box or a point to seed segmentation (e.g. sam3.segment_box) or
  coarse localization without any local GPU model, including parts a box bounds
  poorly such as a cable's tip.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, detection, gemini, api]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.2.0
    provenance:
      code: https://github.com/graph-robots/open-robot-skills/tree/752dfbd96968d4d08379f8111765efbf6457800b/tools/gemini-er
      upstream_license: MIT
gap:
  requires: {env_any: [GOOGLE_API_KEY, GEMINI_API_KEY]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "gemini-er"]
    protocol: stdio-msgpack
  tools:
    - gemini-er.detect: Detect pixel-space 2D bounding boxes for a text query via Gemini Robotics-ER.
    - gemini-er.point: Point at what a text query names via Gemini Robotics-ER; returns pixel points.
---

# gemini-er

API-backed 2D detection on Gemini Robotics-ER. Zero GPU. The canonical
perception recipe (from the dev tree's `perceive_gemini_er` workflow script)
is: `gemini-er.detect` → best box by `score` → `sam3.segment_box` on the full
frame at that box → depth projection → OBB fit.

## Install

```bash
uv sync --extra gemini-er    # google-genai  (pip: pip install -e ".[gemini-er]")
export GOOGLE_API_KEY=...    # or GEMINI_API_KEY
```

## Config

| Env                   | Meaning                                  | Default                          |
|-----------------------|------------------------------------------|----------------------------------|
| `GAP_GEMINI_ER_MODEL` | Gemini model name                        | `gemini-robotics-er-2-preview`   |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | API key (SDK default resolution) | —                  |

## Contract

`gemini-er.detect(image, query)` returns
`{"detections": [{"box": BoundingBox2D, "label": str, "score": float}]}`:

- `box` is pixel-space `{x1, y1, x2, y2}` (top-left → bottom-right), clamped
  to the image bounds. The model emits the Gemini `box_2d` convention
  (`[ymin, xmin, ymax, xmax]` normalized 0–1000); conversion happens here.
- `score` defaults to 1.0 when the model reports none — callers select the
  best detection with `max(..., key=score)`.
- No match (or unparseable model output) → empty `detections`, never an
  error. Treat empty as "object not visible".

`gemini-er.point(image, query, max_points=10)` returns
`{"pixel_x": float, "pixel_y": float, "found": bool, "points": [{"pixel_x", "pixel_y", "label"}]}`:

- The first point sits at the top level, the shape `molmo.point_prompt`
  returns, so either backend can seed the same next step.
- The model emits the Gemini `point` convention (`[y, x]` normalized
  0–1000); conversion and clamping happen here.
- No match → `found: false`, zeros, empty `points`, never an error.

ER 1.5 (`gemini-robotics-er-1.5-preview`), the default before 0.2.0, was no
longer listed by the Gemini API on 2026-11-06; `gemini-robotics-er-2-preview`
and `gemini-robotics-er-2-streaming-preview` were.

## When to use

- Detection boxes for open-vocabulary prompts, no local weights.
- Points (`gemini-er.point`) for objects a few pixels across and for parts
  of an object: a cable's tip, a port's mouth, a handle.
- `molmo.point_prompt` is the self-hosted alternative for one point, and
  `vlm.query_yes_no` serves semantic checks without localization.
