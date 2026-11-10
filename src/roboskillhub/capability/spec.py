"""Capability definitions: one YAML per capability under ``capabilities/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SETTINGS = ("sim", "real")
GATE_KEYS = {"min_episodes": int, "min_success_rate": float, "max_false_picked_rate": float}
REQUIRED = (
    "capability",
    "inputs",
    "outputs",
    "exits",
    "preconditions",
    "timeout_s",
    "implementations",
    "acceptance",
)
IMPL_KINDS = ("program", "skill", "policy")


@dataclass
class Gate:
    min_episodes: int
    min_success_rate: float
    max_false_picked_rate: float


@dataclass
class Implementation:
    id: str
    kind: str
    bundle: str
    """The directory its eval records are filed under: ``evals/<bundle>/``."""


@dataclass
class CapabilitySpec:
    name: str
    version: str
    description: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    exits: dict[str, str]
    preconditions: list[str]
    timeout_s: float
    implementations: list[Implementation]
    acceptance: dict[str, Gate]
    verifier: str | None = None
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def implementation(self, ident: str) -> Implementation:
        for impl in self.implementations:
            if impl.id == ident:
                return impl
        raise KeyError(f"{self.name} has no implementation {ident!r}")


def validate(data: Any, where: str = "capability") -> list[str]:
    """Problems with a parsed capability YAML; empty when it is well-formed."""
    if not isinstance(data, dict):
        return [f"{where}: a capability is a mapping"]
    errors = [f"{where}: missing {k}" for k in REQUIRED if k not in data]
    if data.get("capability_version", 1) != 1:
        errors.append(f"{where}: capability_version must be 1")
    for key in ("inputs", "outputs", "exits"):
        if key in data and not isinstance(data[key], dict):
            errors.append(f"{where}: {key} must be a mapping")
    if not isinstance(data.get("preconditions", []), list):
        errors.append(f"{where}: preconditions must be a list")
    timeout = data.get("timeout_s")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        errors.append(f"{where}: timeout_s must be a positive number")
    seen: set[str] = set()
    for i, impl in enumerate(data.get("implementations") or []):
        if not isinstance(impl, dict) or not isinstance(impl.get("id"), str):
            errors.append(f"{where}: implementations[{i}] needs an id")
            continue
        if impl["id"] in seen:
            errors.append(f"{where}: implementation {impl['id']} listed twice")
        seen.add(impl["id"])
        if impl.get("kind") not in IMPL_KINDS:
            errors.append(f"{where}: {impl['id']} kind must be one of {IMPL_KINDS}")
    gates = data.get("acceptance")
    if isinstance(gates, dict):
        for setting, gate in gates.items():
            if setting not in SETTINGS:
                errors.append(f"{where}: acceptance setting {setting!r} not in {SETTINGS}")
                continue
            if not isinstance(gate, dict):
                errors.append(f"{where}: acceptance.{setting} must be a mapping")
                continue
            for key, kind in GATE_KEYS.items():
                value = gate.get(key)
                if not isinstance(value, (int, float)) or (
                    kind is int and not isinstance(value, int)
                ):
                    errors.append(f"{where}: acceptance.{setting}.{key} must be a {kind.__name__}")
    elif "acceptance" in data:
        errors.append(f"{where}: acceptance must be a mapping of setting -> gate")
    return errors


def parse(data: dict[str, Any], path: Path | None = None) -> CapabilitySpec:
    errors = validate(data, str(path or data.get("capability", "capability")))
    if errors:
        raise ValueError("; ".join(errors))
    impls = [
        Implementation(
            id=i["id"], kind=i["kind"], bundle=str(i.get("bundle") or i["id"].split(":")[-1])
        )
        for i in data["implementations"]
    ]
    gates = {
        s: Gate(
            min_episodes=int(g["min_episodes"]),
            min_success_rate=float(g["min_success_rate"]),
            max_false_picked_rate=float(g["max_false_picked_rate"]),
        )
        for s, g in data["acceptance"].items()
    }
    return CapabilitySpec(
        name=data["capability"],
        version=str(data.get("version", "0.1.0")),
        description=str(data.get("description", "")),
        inputs=data["inputs"],
        outputs=data["outputs"],
        exits=data["exits"],
        preconditions=list(data["preconditions"]),
        timeout_s=float(data["timeout_s"]),
        implementations=impls,
        acceptance=gates,
        verifier=data.get("verifier"),
        path=path,
        raw=data,
    )


def capabilities_dir(root: Path) -> Path:
    return root / "capabilities"


def list_capabilities(root: Path) -> list[str]:
    return sorted(p.stem for p in capabilities_dir(root).glob("*.yaml"))


def load_capability(name: str, root: Path) -> CapabilitySpec:
    path = capabilities_dir(root) / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(list_capabilities(root)) or "none"
        raise FileNotFoundError(f"no capability {name!r} (known: {known})")
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")), path)
