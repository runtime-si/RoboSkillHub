# Adapted from two sources:
# - predicators (https://github.com/BasisResearch/predicators, commit
#   afb6f2918091f5ca372a75b9a335aea317a8a209), MIT License, Copyright (c) 2022 Rohan Chitnis
#   and Tom Silver; the license is LICENSE.predicators beside this skill. Taken from
#   predicators/envs/pybullet_domino/cascade_probe.py run_counterfactual_push_probe: rebuild
#   the layout, replay the trigger push with the fingertips as the only thing of the robot
#   that collides, stop at the end of the stroke (the retreat is left out, so a knock on the
#   way back cannot pass as a cascade) and hold the stroke's last pose while the wave runs
#   (_SETTLE_STEPS), then read which tiles fell; from cascade_certificate.py _topple_onset
#   and components/domino_component.py: a tile has fallen when its tilt stays at or past
#   10 deg (fallen_threshold) for 3 samples running (_TOPPLE_MIN_STEPS); from sim_core.py the
#   domino-table friction 0.5 fitted to their real table.
# - MuJoCo's model/sleep/dominos.xml (https://github.com/google-deepmind/mujoco, commit
#   70dede9373277de3a781428e4ce3a428a34abc8b), Apache License 2.0, Copyright 2025 DeepMind
#   Technologies Limited: the scene (free box dominoes on a plane, elliptic friction cones,
#   impratio 10).
# Changed: MuJoCo (CPU) instead of PyBullet; the scene is built from the hub's slot poses
# and the real tile size instead of the env's body pool; the push is
# triggering-a-domino-chain-by-push's stroke (a fingertip block driven straight along the
# first tile's facing at its contact height) instead of the Push option on a Fetch arm, or
# planning-a-domino-layout's slow-push model (the first tile set turning) instead of
# dominos.xml's tile started atilt; no
# identical retries (this engine is deterministic) - repeated rollouts with the slots
# jittered by the placing error instead, after the robustness idea of Roussel et al.,
# "Designing Chain Reaction Contraptions from Causal Graphs" (SIGGRAPH 2019); routed exits
# instead of a (bool, detail) verdict.
"""Rebuild a planned domino layout in a rigid-body simulation, push the first tile and see how far the wave runs."""

from __future__ import annotations

import math
import random
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

TILE_SIZE = {"height": 0.048, "width": 0.024, "thickness": 0.0075}
"""A standard domino standing on its end [m], as planning-a-domino-layout."""

TILE_MASS = 0.010
"""Tile mass [kg]: 48 x 24 x 7.5 mm (8.6 cm3) of plastic or wood, 6-13 g. Identical tiles
make it drop out of the toppling condition (predicators' mass note); it matters only
against the push."""

TABLE_FRICTION = 0.5
"""Tile-table friction: predicators' value fitted to its real table (sim_core.py
domino_friction; their real-table fit moved planning friction 0.1 -> ~0.5)."""

TILE_FRICTION = 0.25
"""Tile-tile friction: 0.25 +- 0.05 measured on 48 x 24 x 7.5 mm tiles
(zmatt.net/domino-physics), planning-a-domino-layout's DOMINO_FRICTION."""

FALLEN_DEG = 10.0
"""A tile tilted at least this far from vertical has fallen (predicators'
fallen_threshold)."""

FALLEN_SAMPLES = 3
"""Samples running a tilt must hold to count as a fall (predicators' _TOPPLE_MIN_STEPS)."""

SAMPLE_DT = 0.01
"""Seconds between tilt samples."""

TIMESTEP = 0.0005
"""Physics step [s]. Halving it changed no verdict in the measured sweeps (SKILL.md)."""

CONTACT_FRACTION = 0.7
"""Fingertip height over the table at the stroke, as a fraction of the tile height
(triggering-a-domino-chain-by-push)."""

APPROACH_CLEARANCE = 0.008
"""Gap between the blade front and the tile's back face where the stroke starts [m]
(triggering-a-domino-chain-by-push)."""

OVERSHOOT = 0.008
"""How far past the tipping travel the blade front pushes [m]
(triggering-a-domino-chain-by-push)."""

PUSH_SPEED = 0.05
"""Stroke speed [m/s]: the trigger skill's measured 0.048-0.051 m/s at speed_scale 0.25."""

NEXT_CLEARANCE = 0.006
"""Least gap between the stroke's end and the next tile's back face [m]; closer, the
trigger skill does not push (triggering-a-domino-chain-by-push's NEXT_CLEARANCE)."""

KICK_FACTOR = 1.2
"""With trigger `kick`, the first tile starts upright turning about its front edge at this
multiple of the angular speed that just tips a lone tile (2.69 rad/s for 48 x 7.5 mm):
planning-a-domino-layout's PUSH_SPEED_FACTOR, the slow push its 2D model checks."""

BLADE_HALF = (0.012, 0.015, 0.006)
"""The closed fingertips as a block: half depth along the stroke (the trigger skill's
FINGER_HALF_DEPTH), half span across it (both blades across a 24 mm tile), half height."""

QUIET_TIME = 1.0
"""Seconds after the stroke with no new fall before the run is over."""

MAX_TIME = 30.0
"""Simulated seconds at most, whatever the length of the line."""

POSITION_SIGMA = 0.002
"""Placing error per axis [m], one standard deviation: the placing skill's twin placed
within 2 mm. A calibrated station camera's 5.4 mm RMS is the pessimistic case."""

YAW_SIGMA_DEG = 3.0
"""Placing error in facing [deg], one standard deviation."""

TRIALS = 20
"""Jittered rollouts after the nominal one."""

MIN_SUCCESS = 0.9
"""Fraction of jittered rollouts that must reach the target for `cascades`."""

RESAMPLE = 20
"""Draws per tile before a jittered tile that overlaps its neighbour is taken from its slot
slid forward until clear."""


class Output(TypedDict):
    route: str
    detail: str
    n_tiles: int
    target_index: int
    reached_target: bool
    stall_at: int
    n_fallen: int
    fallen: list
    topple_times: list
    wave_speed: float
    push_travel: float
    first_fell: bool
    trials: int
    success_rate: float
    trial_stalls: list
    weakest_pair: int
    sim_time: float
    wall_time: float


# ----------------------------------------------------------------------------- inputs


def _yaw_of(rot: Any) -> float:
    if isinstance(rot, dict):
        w, x, y, z = (float(rot.get(k, 0.0)) for k in ("w", "x", "y", "z"))
    else:
        w, x, y, z = (float(v) for v in rot)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _slot(s: Any) -> tuple[float, float, float, float | None]:
    """(x, y, yaw rad, z or None) from a slot dict, an Se3Pose or [x, y, yaw_deg]."""
    if isinstance(s, dict):
        pose = s.get("pose", s)
        p = pose.get("position", pose)
        if isinstance(p, dict):
            x, y, z = float(p["x"]), float(p["y"]), p.get("z")
        else:
            x, y, z = float(p[0]), float(p[1]), (p[2] if len(p) > 2 else None)
        if s.get("facing_deg") is not None:
            yaw = math.radians(float(s["facing_deg"]))
        elif pose.get("rotation") is not None:
            yaw = _yaw_of(pose["rotation"])
        else:
            raise ValueError(f"slot {s!r} has no facing")
        return x, y, yaw, None if z is None else float(z)
    v = [float(a) for a in s]
    if len(v) < 3:
        raise ValueError(f"slot {s!r} is not [x, y, facing_deg]")
    return v[0], v[1], math.radians(v[2]), None


def _footprint(x: float, y: float, yaw: float, t: float, w: float) -> np.ndarray:
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    c = np.array([x, y])
    return np.array([c + su * t / 2 * u + sv * w / 2 * v for su in (-1, 1) for sv in (-1, 1)])


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# ----------------------------------------------------------------------------- the rollout


def tip_travel(height: float, thickness: float, contact: float) -> float:
    """How far the back face moves at the contact height when the tile reaches its tipping
    angle atan(t / h) (triggering-a-domino-chain-by-push)."""
    a = math.atan2(thickness, height)
    return thickness * (1 - math.cos(a)) + contact * math.sin(a)


def tip_speed(height: float, thickness: float, g: float = 9.81) -> float:
    """Angular speed about the front edge that just carries a lone tile over it [rad/s]:
    the centre of mass rises from h/2 to half the diagonal."""
    r = math.hypot(height, thickness) / 2
    inertia = (height**2 + thickness**2) / 3  # about the edge, per unit mass
    return math.sqrt(2 * g * (r - height / 2) / inertia)


def build_xml(
    slots: list[tuple[float, float, float]],
    table_z: float,
    size: dict[str, float],
    mass: float,
    table_mu: float,
    tile_mu: float,
    timestep: float,
) -> str:
    h, w, t = size["height"], size["width"], size["thickness"]
    bodies = []
    for k, (x, y, yaw) in enumerate(slots):
        bodies.append(
            f'<body name="d{k}" pos="{x:.6f} {y:.6f} {table_z + h / 2 + 1e-5:.6f}" '
            f'euler="0 0 {yaw:.6f}"><freejoint/>'
            f'<geom type="box" size="{t / 2:.6f} {w / 2:.6f} {h / 2:.6f}" mass="{mass:.6f}" '
            f'friction="{tile_mu} 0.005 0.0001"/></body>'
        )
    bx, by, bz = BLADE_HALF
    return (
        '<mujoco model="domino layout"><compiler angle="radian"/>'
        f'<option timestep="{timestep}" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>'
        "<worldbody>"
        f'<geom name="table" type="plane" size="0 0 0.01" pos="0 0 {table_z:.6f}" '
        f'friction="{table_mu} 0.005 0.0001" priority="1"/>'
        '<body name="blade" mocap="true" pos="0 0 -1">'
        f'<geom type="box" size="{bx} {by} {bz}" friction="{table_mu} 0.005 0.0001" priority="1"/>'
        "</body>" + "".join(bodies) + "</worldbody></mujoco>"
    )


def rollout(
    mujoco: Any,
    slots: list[tuple[float, float, float]],
    table_z: float,
    size: dict[str, float],
    mass: float = TILE_MASS,
    table_mu: float = TABLE_FRICTION,
    tile_mu: float = TILE_FRICTION,
    contact_fraction: float = CONTACT_FRACTION,
    push_speed: float = PUSH_SPEED,
    overshoot: float = OVERSHOOT,
    timestep: float = TIMESTEP,
    max_time: float = MAX_TIME,
    trigger: str = "push",
    kick_factor: float = KICK_FACTOR,
) -> dict[str, Any]:
    """One run: push slot 0 along its facing (or start it turning about its front edge),
    hold, and record when each tile fell."""
    h, t = size["height"], size["thickness"]
    model = mujoco.MjModel.from_xml_string(
        build_xml(slots, table_z, size, mass, table_mu, tile_mu, timestep)
    )
    data = mujoco.MjData(model)
    n = len(slots)
    x0, y0, yaw0 = slots[0]
    u = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    if trigger == "kick":
        # body frame: x the facing, y the width, z the height; a free joint's angular
        # velocity is in the body frame, its linear velocity in the world
        w = kick_factor * tip_speed(h, t)
        data.qvel[0:3] = w * (u * h / 2 + np.array([0.0, 0.0, t / 2]))
        data.qvel[3:6] = [0.0, w, 0.0]
    mujoco.mj_forward(model, data)
    contact = contact_fraction * h
    back = np.array([x0, y0, 0.0]) - u * t / 2
    bx, _, bz = BLADE_HALF
    z_blade = table_z + contact + bz
    travel = APPROACH_CLEARANCE + tip_travel(h, t, contact) + overshoot
    start = back - u * (APPROACH_CLEARANCE + bx)
    start[2] = z_blade
    stroke_time = travel / push_speed if trigger == "push" else 0.0
    quat = np.array([math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)])
    data.mocap_pos[0] = start if trigger == "push" else np.array([0.0, 0.0, table_z - 1.0])
    data.mocap_quat[0] = quat

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"d{k}") for k in range(n)]
    cos_fallen = math.cos(math.radians(FALLEN_DEG))
    run = [0] * n
    run_start = [-1.0] * n
    onset = [-1.0] * n
    last_fall = 0.0
    every = max(1, round(SAMPLE_DT / timestep))
    limit = min(max_time, stroke_time + QUIET_TIME + 0.4 * n + 1.0)
    steps = 0
    while data.time < limit:
        if stroke_time > 0:
            data.mocap_pos[0] = start + u * travel * min(data.time / stroke_time, 1.0)
        mujoco.mj_step(model, data)
        steps += 1
        if steps % every:
            continue
        for k, b in enumerate(body_ids):
            if onset[k] >= 0:
                continue
            up = data.xmat[b].reshape(3, 3)[2, 2]  # body z axis . world z
            if up <= cos_fallen:
                if run[k] == 0:
                    run_start[k] = data.time
                run[k] += 1
                if run[k] >= FALLEN_SAMPLES:
                    onset[k] = run_start[k]
                    last_fall = data.time
            else:
                run[k] = 0
        if all(o >= 0 for o in onset):
            break
        if data.time > stroke_time and data.time - max(last_fall, stroke_time) > QUIET_TIME:
            break
    return {
        "onset": onset,
        "sim_time": float(data.time),
        "travel": travel if trigger == "push" else 0.0,
    }


def _stall(onset: list[float], target: int) -> int:
    """First tile up to the target that did not fall (-1: the wave reached the target)."""
    for k in range(target + 1):
        if onset[k] < 0:
            return k
    return -1


def _jitter(
    slots: list[tuple[float, float, float]],
    rng: random.Random,
    sigma: float,
    yaw_sigma: float,
    t: float,
    w: float,
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for k, (x, y, yaw) in enumerate(slots):
        pick = None
        for _ in range(RESAMPLE):
            cand = (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), yaw + rng.gauss(0, yaw_sigma))
            if k == 0 or not _overlap(_footprint(*cand, t, w), _footprint(*out[-1], t, w)):
                pick = cand
                break
        if pick is None:
            # every draw overlapped: the slot itself, slid along its facing until clear
            step = 0.0
            while _overlap(
                _footprint(x + step * math.cos(yaw), y + step * math.sin(yaw), yaw, t, w),
                _footprint(*out[-1], t, w),
            ):
                step += 0.0005
            pick = (x + step * math.cos(yaw), y + step * math.sin(yaw), yaw)
        out.append(pick)
    return out


# ----------------------------------------------------------------------------- the skill


def run(
    ctx: NodeContext,
    slots: list[Any],
    table_z: float | None = None,
    target_index: int = -1,
    tile_size: dict[str, float] | None = None,
    tile_mass: float = TILE_MASS,
    table_friction: float = TABLE_FRICTION,
    tile_friction: float = TILE_FRICTION,
    trigger: str = "push",
    kick_factor: float = KICK_FACTOR,
    contact_fraction: float = CONTACT_FRACTION,
    push_speed: float = PUSH_SPEED,
    overshoot: float = OVERSHOOT,
    trials: int = TRIALS,
    position_sigma: float = POSITION_SIGMA,
    yaw_sigma_deg: float = YAW_SIGMA_DEG,
    min_success: float = MIN_SUCCESS,
    timestep: float = TIMESTEP,
    seed: int = 0,
) -> Output:
    out: Output = {
        "route": "invalid_input",
        "detail": "",
        "n_tiles": 0,
        "target_index": -1,
        "reached_target": False,
        "stall_at": -1,
        "n_fallen": 0,
        "fallen": [],
        "topple_times": [],
        "wave_speed": 0.0,
        "push_travel": 0.0,
        "first_fell": False,
        "trials": 0,
        "success_rate": 0.0,
        "trial_stalls": [],
        "weakest_pair": -1,
        "sim_time": 0.0,
        "wall_time": 0.0,
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        return out

    t_wall = time.perf_counter()
    try:
        parsed = [_slot(s) for s in slots or []]
        if len(parsed) < 2:
            raise ValueError(f"need at least two slots, got {len(parsed)}")
        size = {**TILE_SIZE, **{k: float(v) for k, v in (tile_size or {}).items()}}
        if not size["height"] > size["width"] > size["thickness"] > 0:
            raise ValueError(f"tile_size {size} is not height > width > thickness > 0")
        if table_z is None:
            zs = [z for *_, z in parsed if z is not None]
            if not zs:
                raise ValueError("no table_z and the slots carry no z")
            table_z = float(np.median(zs))
        n = len(parsed)
        target = int(target_index) % n if int(target_index) < 0 else int(target_index)
        if not 0 < target < n:
            raise ValueError(f"target_index {target_index} is not a slot after the first")
        if not 0.2 <= float(contact_fraction) <= 1.0:
            raise ValueError(f"contact_fraction {contact_fraction} is outside 0.2..1")
        if trigger not in ("push", "kick"):
            raise ValueError(f"trigger must be 'push' or 'kick', not {trigger!r}")
        if float(push_speed) <= 0 or float(timestep) <= 0 or float(tile_mass) <= 0:
            raise ValueError("push_speed, timestep and tile_mass must be positive")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return finish("invalid_input", str(exc))
    xyyaw = [(x, y, yaw) for x, y, yaw, _ in parsed]
    feet = [_footprint(*s, size["thickness"], size["width"]) for s in xyyaw]
    for k in range(n - 1):
        if _overlap(feet[k], feet[k + 1]):
            return finish("invalid_input", f"slots {k} and {k + 1} overlap")
    out.update(n_tiles=n, target_index=target)
    if trigger == "push":
        h, t = size["height"], size["thickness"]
        x0, y0, yaw0 = xyyaw[0]
        u = np.array([math.cos(yaw0), math.sin(yaw0)])
        end = (
            np.array([x0, y0])
            - u * t / 2
            + u * (tip_travel(h, t, contact_fraction * h) + overshoot)
        )
        room = float(min((feet[1] - end) @ u))
        if room < NEXT_CLEARANCE:
            return finish(
                "push_failed",
                f"the trigger stroke would end {room * 1e3:.1f} mm from slot 1 (under "
                f"{NEXT_CLEARANCE * 1e3:.0f} mm): triggering-a-domino-chain-by-push does not push "
                "a line this tight; trigger 'kick' checks the chain alone",
            )

    try:
        import mujoco
    except ImportError as exc:
        return finish("sim_unavailable", f"the physics engine is not installed ({exc})")

    kw = dict(
        mass=float(tile_mass),
        table_mu=float(table_friction),
        tile_mu=float(tile_friction),
        contact_fraction=float(contact_fraction),
        push_speed=float(push_speed),
        overshoot=float(overshoot),
        timestep=float(timestep),
        trigger=trigger,
        kick_factor=float(kick_factor),
    )
    try:
        nominal = rollout(mujoco, xyyaw, float(table_z), size, **kw)
    except Exception as exc:  # noqa: BLE001 - a model the engine rejects
        return finish("sim_unavailable", f"the simulation failed: {exc}")
    onset = nominal["onset"]
    stall = _stall(onset, target)
    fell = [o >= 0 for o in onset]
    out.update(
        reached_target=stall < 0,
        stall_at=stall,
        n_fallen=int(sum(fell)),
        fallen=fell,
        topple_times=[round(o, 4) for o in onset],
        push_travel=round(nominal["travel"], 5),
        first_fell=fell[0],
        sim_time=round(nominal["sim_time"], 3),
    )
    # the settled wave: the second half of the run when it is 6 tiles or more (the first
    # tiles go at the trigger's pace)
    run_len = target + 1 if stall < 0 else stall
    if run_len >= 2:
        a = run_len // 2 if run_len >= 6 else 0
        b = run_len - 1
        arc = sum(math.dist(xyyaw[k][:2], xyyaw[k + 1][:2]) for k in range(a, b))
        if onset[b] > onset[a]:
            out["wave_speed"] = round(arc / (onset[b] - onset[a]), 4)

    if not fell[0]:
        out["wall_time"] = round(time.perf_counter() - t_wall, 3)
        how = (
            f"the stroke ({nominal['travel'] * 1e3:.1f} mm at {contact_fraction:.2f} h, "
            f"{push_speed:.3f} m/s) did not tip it"
            if trigger == "push"
            else f"the kick ({kick_factor:.2f} x the tipping speed) did not carry it past 10 deg"
        )
        return finish("push_failed", f"the first tile did not fall: {how}")

    rng = random.Random(seed)
    stalls = []
    for _ in range(max(0, int(trials))):
        jit = _jitter(
            xyyaw,
            rng,
            float(position_sigma),
            math.radians(float(yaw_sigma_deg)),
            size["thickness"],
            size["width"],
        )
        stalls.append(_stall(rollout(mujoco, jit, float(table_z), size, **kw)["onset"], target))
    out["trials"] = len(stalls)
    out["trial_stalls"] = stalls
    if stalls:
        out["success_rate"] = round(sum(s < 0 for s in stalls) / len(stalls), 4)
        failed = [s for s in stalls if s >= 0]
        if failed:
            out["weakest_pair"] = max(set(failed), key=failed.count) - 1
    out["wall_time"] = round(time.perf_counter() - t_wall, 3)

    jitter_txt = (
        f"; {out['success_rate']:.0%} of {len(stalls)} rollouts with the slots jittered "
        f"({position_sigma * 1e3:.1f} mm, {yaw_sigma_deg:.1f} deg) reached it"
        if stalls
        else ""
    )
    if stall >= 0:
        return finish(
            "stalls",
            f"the wave stopped at slot {stall}: tile {stall - 1} fell, {stall} still stands "
            f"({out['n_fallen']} of {n} fell){jitter_txt}",
        )
    if stalls and out["success_rate"] < float(min_success):
        return finish(
            "fragile",
            f"the planned layout reaches slot {target}, but{jitter_txt.lstrip(';')}; the pair "
            f"most often broken is {out['weakest_pair']}-{out['weakest_pair'] + 1}",
        )
    return finish(
        "cascades",
        f"the wave reaches slot {target} {onset[target] - onset[0]:.2f} s after the first "
        f"tile fell, settled at {out['wave_speed']:.2f} m/s{jitter_txt}",
    )
