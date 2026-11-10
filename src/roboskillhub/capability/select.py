"""Pick the implementations of a capability that passed its acceptance gate on a robot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import gate_failures, read_records
from .spec import CapabilitySpec, Implementation


@dataclass
class Choice:
    implementation: Implementation
    eligible: bool
    why: list[str] = field(default_factory=list)
    record: dict[str, Any] | None = None

    @property
    def success_rate(self) -> float:
        r = self.record or {}
        n = r.get("episodes") or 0
        return (r.get("successes") or 0) / n if n else 0.0

    def to_dict(self) -> dict[str, Any]:
        r = self.record or {}
        return {
            "implementation": self.implementation.id,
            "kind": self.implementation.kind,
            "bundle": self.implementation.bundle,
            "eligible": self.eligible,
            "why": self.why,
            "record": r.get("_file"),
            "episodes": r.get("episodes"),
            "successes": r.get("successes"),
            "false_picked": r.get("false_picked"),
        }


def _matching(records, spec: CapabilitySpec, impl: Implementation, embodiment, setting):
    return [
        r
        for r in records
        if r.get("capability") == spec.name
        and r.get("implementation", impl.id) == impl.id
        and r.get("embodiment") == embodiment
        and r.get("setting") == setting
    ]


def judge(
    spec: CapabilitySpec, impl: Implementation, root: Path, embodiment: str, setting: str
) -> Choice:
    gate = spec.acceptance.get(setting)
    if gate is None:
        return Choice(impl, False, [f"{spec.name} has no {setting} acceptance gate"])
    records = _matching(read_records(root, impl.bundle), spec, impl, embodiment, setting)
    if not records:
        return Choice(impl, False, [f"no {setting} eval record for {spec.name} on {embodiment}"])
    newest = max(records, key=lambda r: (str(r.get("date", "")), r.get("_file", "")))
    episodes, successes = newest.get("episodes"), newest.get("successes")
    if not isinstance(episodes, int) or not isinstance(successes, int):
        return Choice(impl, False, [f"{newest['_file']}: episodes/successes missing"], newest)
    fp = newest.get("false_picked")
    labelled = newest.get("labelled", episodes if fp is not None else None)
    failures = gate_failures(
        episodes, successes, fp if isinstance(fp, int) else None, labelled, gate
    )
    rate = successes / episodes if episodes else 0.0
    summary = (
        f"{newest['_file']}: {setting} {successes}/{episodes} ({rate:.2f}), "
        f"false picked {fp if fp is not None else '?'}"
    )
    if failures:
        return Choice(impl, False, [summary, *failures], newest)
    return Choice(impl, True, [summary, "passes the gate"], newest)


def select(spec: CapabilitySpec, root: Path, embodiment: str, setting: str = "sim") -> list[Choice]:
    """Every implementation judged, eligible ones first, best success rate first."""
    choices = [judge(spec, impl, root, embodiment, setting) for impl in spec.implementations]
    return sorted(choices, key=lambda c: (not c.eligible, -c.success_rate))
