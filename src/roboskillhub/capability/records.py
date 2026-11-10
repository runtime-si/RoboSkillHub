"""Eval records for capability acceptance runs, and the gate that reads them.

A capability record is a SPEC.md eval record (``evals/<bundle>/...json``) with the extra
keys ``capability``, ``implementation``, ``verifier``, ``exits``, ``false_picked`` and
``truth``. ``successes`` counts verifier ``picked`` exits that the truth label (when given)
confirms; ``false_picked`` counts ``picked`` exits the truth label contradicts.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .spec import CapabilitySpec, Gate
from .verify import EXITS, VERIFIER_VERSION, PickVerdict


@dataclass
class Episode:
    exit: str
    claim: str | None = None
    truth_held: bool | None = None
    """Ground truth at the end of the episode (simulator state, or an operator's label)."""
    seed: int | None = None

    @classmethod
    def from_verdict(
        cls, verdict: PickVerdict, truth_held: bool | None = None, seed: int | None = None
    ) -> Episode:
        return cls(exit=verdict.exit, claim=verdict.claim, truth_held=truth_held, seed=seed)


@dataclass
class Tally:
    episodes: int
    successes: int
    false_picked: int
    exits: dict[str, int]
    claim_disagreements: int
    labelled: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def false_picked_rate(self) -> float:
        return self.false_picked / self.episodes if self.episodes else 0.0


def tally(episodes: Iterable[Episode | dict[str, Any]]) -> Tally:
    eps = [e if isinstance(e, Episode) else Episode(**e) for e in episodes]
    exits = {k: 0 for k in EXITS}
    successes = false_picked = disagree = labelled = 0
    for e in eps:
        if e.exit not in exits:
            raise ValueError(f"unknown exit {e.exit!r}")
        exits[e.exit] += 1
        if e.truth_held is not None:
            labelled += 1
        if e.exit == "picked":
            if e.truth_held is False:
                false_picked += 1
            else:
                successes += 1
        if e.claim is not None and e.claim != e.exit:
            disagree += 1
    return Tally(len(eps), successes, false_picked, exits, disagree, labelled)


def gate_failures(
    episodes: int, successes: int, false_picked: int | None, labelled: int | None, gate: Gate
) -> list[str]:
    """Why these numbers miss *gate*; empty when they pass."""
    out = []
    if episodes < gate.min_episodes:
        out.append(f"{episodes} episodes < {gate.min_episodes}")
    rate = successes / episodes if episodes else 0.0
    if rate < gate.min_success_rate:
        out.append(f"success {rate:.2f} < {gate.min_success_rate}")
    if false_picked is None or labelled is None or labelled < episodes:
        out.append("false picked unknown: not every episode has a truth label")
    elif episodes and false_picked / episodes > gate.max_false_picked_rate:
        out.append(f"false picked {false_picked / episodes:.3f} > {gate.max_false_picked_rate}")
    return out


def build_record(
    spec: CapabilitySpec,
    implementation: str,
    *,
    embodiment: str,
    setting: str,
    episodes: Iterable[Episode | dict[str, Any]],
    suite: str,
    evidence: str,
    version: str = "0.1.0",
    tier: str = "camera_only",
    truth: str = "sim_truth",
    date: str | None = None,
) -> dict[str, Any]:
    impl = spec.implementation(implementation)
    if setting not in spec.acceptance:
        raise ValueError(f"{spec.name} has no acceptance gate for setting {setting!r}")
    t = tally(episodes)
    gate = spec.acceptance[setting]
    failures = gate_failures(t.episodes, t.successes, t.false_picked, t.labelled, gate)
    return {
        "bundle": impl.bundle,
        "version": version,
        "embodiment": embodiment,
        "setting": setting,
        "tier": tier,
        "suite": suite,
        "episodes": t.episodes,
        "successes": t.successes,
        "passed": not failures,
        "date": date or _dt.date.today().isoformat(),
        "evidence": evidence,
        "capability": spec.name,
        "capability_version": spec.version,
        "implementation": impl.id,
        "verifier": VERIFIER_VERSION,
        "truth": truth,
        "labelled": t.labelled,
        "false_picked": t.false_picked,
        "claim_disagreements": t.claim_disagreements,
        "exits": t.exits,
        "gate_failures": failures,
    }


def write_record(root: Path, record: dict[str, Any]) -> Path:
    """File *record* under ``evals/<bundle>/<embodiment>-<setting>-<capability>-<date>[-n].json``."""
    folder = root / "evals" / record["bundle"]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{record['embodiment']}-{record['setting']}-{record['capability']}-{record['date']}"
    path, n = folder / f"{stem}.json", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}-{n}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_records(root: Path, bundle: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted((root / "evals" / bundle).glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append({**data, "_file": p.relative_to(root).as_posix()})
    return out
