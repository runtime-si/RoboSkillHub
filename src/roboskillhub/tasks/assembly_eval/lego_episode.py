"""Episode records for the LEGO evaluation: hashes, versions of K and G, and the per-episode summary row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config") / "lego_eval_v1.json"

# K: the skills the scripted graph may call; G: the fixed-order graph that sequences them. The baseline is one
# privileged skill (exact brick poses from the simulator), so it is never reported as a vision-policy result.
K_VERSION = {"lego.privileged_naive_fsm": "0.1.0"}
G_VERSION = {"graph_id": "lego-fixed-order-scripted", "version": "1.0.0"}
OBSERVATION_MODE = "privileged_sim_state"

REQUIRED_KEYS = (
    "episode_id",
    "seed",
    "scene",
    "fault",
    "policy",
    "privileged",
    "observation_mode",
    "hashes",
    "versions",
    "K",
    "G",
    "goal",
    "outcome",
    "placements",
    "sim_steps",
    "planning_s",
    "wall_s",
    "interventions",
    "recoveries",
    "trajectory_ref",
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary_row(ep: dict) -> dict:
    """One line per episode for the results table."""
    o = ep["outcome"]
    return {
        "episode_id": ep["episode_id"],
        "seed": ep["seed"],
        "scene": ep["scene"],
        "fault": ep["fault"],
        "policy": ep["policy"],
        "status": o["status"],
        "failure_type": o["failure_type"],
        "placements": f"{o['placements_succeeded']}/{o['placements']}",
        "eval_config_sha256": ep["hashes"]["eval_config"][:12],
        "physics_sha256": ep["hashes"]["physics"][:12],
        "K": ep["K"],
        "G": f"{ep['G']['graph_id']}@{ep['G']['version']}",
    }


def missing_keys(ep: dict) -> list[str]:
    return [k for k in REQUIRED_KEYS if k not in ep]
