"""A closed loop any backend can run in: observe, ask the model for a chunk, execute it safely, repeat.

    result = run_policy(client, profile, "pick up the red cube", max_steps=300,
                        stop_check=lambda obs, info: ..., robot=ctx)

``client.act(model_input, instruction)`` is the only thing a backend provides (gr00t-wam/1, molmoact2,
pi05: whatever the wire is). ``model_input`` comes from :func:`observation.model_input` (images in the
profile's camera order, the state in model frame and units). The reply is ``{"actions": [T, dim], "frame":
"model" | "station" (default "model"), ...}``; any other keys (``dt_ms``, ``timing_ms``, ``gpu_mem_mb``)
are kept per chunk in ``result["chunks"]``.

``stop_check(obs, info)`` runs on a fresh observation after every chunk (and before the first): truthy
ends the loop with ``route = "stopped"``; a dict may name the exit (``{"stop": True, "exit": "picked"}``).
The loop decides nothing about success - the capability's shared verifier does.

Routes: ``stopped`` | ``max_steps`` | ``refused`` (the executor refused a chunk: ``code`` / ``reason``) |
``bad_observation`` (stale or missing frames) | ``client_error``.

policies/gr00t's ``gr00t.run_skill`` (#173) still runs its own loop: the station twin harness loads that
bundle's tools.py into an environment without roboskillhub installed, and its executors (per-arm
``execute_trajectory``, ``move_to_joints``) and success-check specs have no counterpart here yet. The
``gr00t-wam/1`` / ``vla-yam/1`` client is :class:`~roboskillhub.vla.client.VlaYamClient`; moving
``run_skill`` onto ``run_policy`` still needs a twin re-run, not a drop-in swap.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from .executor import SafetyLimits, execute_chunk
from .observation import ObservationError, get_observation, model_input
from .profile import Profile


class PolicyClient(Protocol):
    def act(self, model_input: dict[str, Any], instruction: str) -> dict[str, Any]: ...


def _stop(stop_check: Any, obs: dict[str, Any], info: dict[str, Any]) -> tuple[bool, Any]:
    if stop_check is None:
        return False, None
    r = stop_check(obs, info)
    if isinstance(r, dict):
        return bool(r.get("stop", r.get("success", False))), r
    return bool(r), r


def run_policy(
    client: PolicyClient,
    profile: Profile,
    instruction: str,
    max_steps: int = 300,
    stop_check: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    *,
    robot: Any,
    arms: tuple[str, ...] | None = None,
    execute_steps: int | None = None,
    cameras: dict[str, str] | None = None,
    limits: SafetyLimits | None = None,
    station_hz: float | None = None,
    transport: str = "auto",
    transport_args: dict[str, Any] | None = None,
    requery_after_switch: bool = True,
    estop: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.time,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run ``client`` on ``robot`` (a gap ``ctx`` with ``robot.*`` tools) until ``stop_check`` fires or
    ``max_steps`` chunk rows ran. Never raises for refusals or stale frames; the route says what happened."""
    lim = limits or SafetyLimits.from_profile(profile)
    steps = chunks = 0
    jaw: dict[str, bool | None] = {}
    log: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_check: Any = None

    def finish(route: str, detail: str, **extra: Any) -> dict[str, Any]:
        return {
            "route": route,
            "steps": steps,
            "chunks": chunks,
            "detail": detail,
            "check": last_check,
            "jaw_events": events,
            "chunk_log": log,
            "profile": profile.name,
            **extra,
        }

    def observe() -> dict[str, Any]:
        return get_observation(robot, cameras=cameras, clock=clock)

    try:
        obs = observe()
    except ObservationError as e:
        return finish("bad_observation", str(e))
    stop, last_check = _stop(stop_check, obs, {"steps": 0, "chunks": 0})
    if stop:
        return finish("stopped", "stop_check passed before the first chunk", exit=_exit(last_check))

    while steps < max_steps:
        try:
            inp = model_input(obs, profile, now=clock())
        except ObservationError as e:
            return finish("bad_observation", str(e))
        t0 = time.perf_counter()
        try:
            reply = client.act(inp, instruction)
        except Exception as e:
            return finish("client_error", f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1e3
        chunks += 1
        n = min(execute_steps or profile.action.execute_steps, max_steps - steps)
        out = execute_chunk(
            robot,
            reply.get("actions"),
            profile,
            obs,
            t_obs=inp["t"],
            frame=str(reply.get("frame", "model")),
            arms=arms,
            jaw_closed=jaw,
            execute_steps=max(1, n),
            station_hz=station_hz,
            limits=lim,
            estop=estop,
            clock=clock,
            transport=transport,
            transport_args=transport_args,
            requery_after_switch=requery_after_switch,
        )
        entry = {
            "chunk": chunks,
            "latency_ms": round(latency, 2),
            **{k: v for k, v in reply.items() if k not in ("actions",)},
            **{k: out.get(k) for k in ("ok", "code", "rows", "segments_sent", "notes")},
        }
        log.append(entry)
        if on_chunk is not None:
            on_chunk({**entry, "actions": reply.get("actions")})
        if not out.get("ok"):
            return finish("refused", out.get("reason", ""), code=out.get("code"))
        steps += int(out.get("rows") or 0)
        for ev in out.get("jaw_events") or []:
            jaw[ev["arm"]] = ev["jaw"] == "close"
            events.append({"chunk": chunks, "step": steps, **ev})
        try:
            obs = observe()
        except ObservationError as e:
            return finish("bad_observation", str(e))
        stop, last_check = _stop(stop_check, obs, {"steps": steps, "chunks": chunks})
        if stop:
            return finish(
                "stopped",
                f"stop_check passed after {steps} rows ({chunks} chunks)",
                exit=_exit(last_check),
            )
    return finish("max_steps", f"{steps} rows ({chunks} chunks) without stop_check passing")


def _exit(check: Any) -> str | None:
    return str(check["exit"]) if isinstance(check, dict) and check.get("exit") else None
