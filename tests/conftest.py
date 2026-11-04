from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMIT = "0" * 40


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def hub(tmp_path: Path, monkeypatch) -> Path:
    """A small hub: one native tool, two upstream skills, an overlay, a recipe, a paper."""
    root = tmp_path / "hub"
    monkeypatch.setenv("MOTIF_CACHE", str(root / ".motif"))
    write(
        root / "sources.yaml",
        f"""
        engine: {{name: graph-as-policy, repo: https://example.invalid/gap, commit: "{COMMIT}"}}
        sources:
          - name: upstream
            repo: https://example.invalid/up
            commit: "{COMMIT}"
            overlay: overlays/upstream.yaml
        """,
    )
    write(
        root / "tools/judge/SKILL.md",
        """
        ---
        name: judge
        description: Decides things from geometry. Use when a graph must branch.
        license: Apache-2.0
        metadata:
          hub: {stage: judge, embodiments: [any], version: 0.1.0, tasks: [pick-place]}
        gap:
          requires: {}
          tools:
            - judge.check: Check a thing.
        ---
        body
        """,
    )
    write(root / "tools/judge/tools.py", "import numpy as np\n")
    up = root / ".motif/sources/upstream"
    write(
        up / "skills/grabbing-things/SKILL.md",
        """
        ---
        name: grabbing-things
        description: Grabs a thing and lifts it. Use when an object must be picked up.
        gap:
          allowed_tools: [robot.close_gripper, judge.check]
          required_inputs: {target_obb: OrientedBoundingBox}
          produces_outputs: {grasp_pose: Se3Pose}
          exit_conditions: {grasped: held, failed: not held}
        ---
        """,
    )
    write(
        up / "skills/peeking-at-sim/SKILL.md",
        """
        ---
        name: peeking-at-sim
        description: Reads object poses from the simulator. Use when the pose of an object is needed.
        gap:
          allowed_tools: [sim.get_object_pose]
          produces_outputs: {"<name>_obb": OrientedBoundingBox}
          exit_conditions: {found: seen, not_found: unseen}
        ---
        """,
    )
    write(
        up / "skills/routing-cables/SKILL.md",
        """
        ---
        name: routing-cables
        description: Lays a cable through clips with two hands. Use when a cable must follow a route.
        gap:
          allowed_tools: [robot.get_observation]
          exit_conditions: {routed: done, failed: not done}
        ---
        """,
    )
    write(
        root / "overlays/upstream.yaml",
        """
        grabbing-things: {stage: execute, embodiments: [any], tasks: [pick-place]}
        peeking-at-sim: {stage: perceive, embodiments: [any], tasks: [pick-place]}
        routing-cables: {stage: execute, embodiments: [yam_i2rt], arms: 2, tasks: [cable-routing]}
        """,
    )
    write(
        root / "recipes/peek-and-grab/RECIPE.md",
        """
        ---
        name: peek-and-grab
        description: Find a thing, then pick it up. Use when an object must be picked.
        license: Apache-2.0
        metadata:
          hub: {version: 0.1.0, embodiments: [any], tasks: [pick-place]}
        recipe:
          start: peek
          nodes:
            peek: {skill: peeking-at-sim, bind: {name: target}}
            grab: {skill: grabbing-things}
            done: {end: success}
            abort: {end: failure, recovery: [robot.open_gripper]}
          routes:
            peek: {found: grab, not_found: abort}
            grab: {grasped: done, failed: abort}
        ---
        """,
    )
    write(
        root / "papers/grab-2026/PAPER.md",
        """
        ---
        id: grab-2026
        title: Grabbing Things
        year: 2026
        url: https://example.invalid/paper
        derived: [skills/grabbing-things, recipes/peek-and-grab]
        ---
        """,
    )
    return root
