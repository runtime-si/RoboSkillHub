# Adapted from Robotic_Lego_Manipulation (https://github.com/intelligent-control-lab/Robotic_Lego_Manipulation),
# src/Lego.cpp (get_brick_corners, bricks_overlap, is_top_connect, is_bottom_connect,
# update_brick_connection, update_all_top_bricks) and include/Lego.hpp (P_len_ 8 mm,
# brick_len_offset_ 0.2 mm, brick_height_m_ 9.6 mm) at commit
# 9bc91654e708e4d8428ecfa76840103cacb2a19a, MIT License.
# Changes: the C++ class keeps a Gazebo-backed brick map with axis-aligned (0 / 90 deg) bricks of
# one height and moves the simulator's models; here the state is a JSON string passed in and out
# (so a graph can keep it between nodes or persist it with checkpointing-assembly-state-for-resume),
# bricks take any yaw (separating-axis overlap of rotated footprints, with a minimum overlap so
# that pose noise and flush neighbours do not connect) and their own heights (plates and bricks:
# ICL's 0.1-1.1 brick-height window is taken against the lower brick's top and the smaller of
# the two heights), a brick is grounded when it stands on the plate or on a grounded brick, a
# removal carries the bricks above it (update_all_top_bricks' rule) or leaves them floating,
# placements that would interpenetrate a tracked brick are refused, and the state can be
# compared with perceived bricks to catch silent failures (a brick that came away with the
# gripper, one knocked off its cell).
"""Keep a world model of placed LEGO / DUPLO bricks and their stud connections, and check it against perception."""

from __future__ import annotations

import json
import math
from typing import Any, TypedDict

from gap import NodeContext

PRESETS: dict[str, dict[str, float]] = {
    "lego": {"stud_pitch": 0.008, "stud_height": 0.0017, "brick_height": 0.0096},
    "duplo": {"stud_pitch": 0.016, "stud_height": 0.0045, "brick_height": 0.0192},
}
"""Brick systems [m]: ICL's P_len_ 8 mm, knob 1.7 mm and brick 9.6 mm; DUPLO twice the pitch,
4.5 mm studs, 19.2 mm bricks (published dimensions)."""

FOOTPRINT_OFFSET = 0.0002
"""ICL's brick_len_offset_: a footprint is n x pitch minus this [m]."""

TOP_WINDOW = (0.1, 1.1)
"""ICL's is_top_connect: the upper brick's z is 0.1-1.1 brick heights over the lower's.
Here, with bricks of their own heights, the upper bottom must lie within
(top - 0.9 h, top + 0.1 h) of the lower top, h the smaller height: the same window."""

MIN_OVERLAP_FRACTION = 0.25
"""Footprints connect when they overlap by more than this fraction of the pitch along
every separating axis (ICL: any strictly positive overlap of exact simulator poses;
perceived poses are a few millimetres off, flush neighbours stand 0.2 mm apart)."""

GROUND_FRACTION = 0.1
"""A brick whose bottom is within this fraction of its height of ground_z stands on the
plate (ICL's 0.1 brick-height margin)."""

COLLISION_FRACTION = 0.1
"""Two bricks whose footprints overlap and whose height ranges overlap by more than this
fraction of the smaller height interpenetrate."""

POSITION_TOLERANCE_FRACTION = 0.25
"""observe: a brick seen within this fraction of the pitch of its tracked place is
present (BrickSim's 2 mm at LEGO pitch, as in verifying-brick-seated-by-height)."""

HEIGHT_TOLERANCE_FRACTION = 0.6
"""... and within this fraction of the stud height in z (resting on the studs reads
a whole stud height high: pressing-a-brick-onto-studs' RESTING_FRACTION)."""

YAW_TOLERANCE_DEG = 10.0
"""... and within this of its yaw (modulo the brick's symmetry)."""

MATCH_RADIUS_FRACTION = 2.5
"""Unlabelled observations are matched to tracked bricks of the same size within this
many pitches; a match beyond the position tolerance is 'displaced'."""

ACTIONS = ("place", "move", "remove", "observe", "query")

ROUTES = ("state_updated", "inconsistent_with_observation", "rejected")


class Output(TypedDict):
    route: str
    detail: str
    state_json: str
    n_bricks: int
    n_connections: int
    n_grounded: int
    floating_json: str
    changed_json: str
    discrepancies_json: str
    removable_json: str


def _f(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _pos(v: Any, name: str) -> list[float]:
    if isinstance(v, dict) and "position" in v:
        v = v["position"]
    if isinstance(v, dict):
        return [_f(v[k], f"{name}.{k}") for k in ("x", "y", "z")]
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return [_f(x, name) for x in v]
    raise ValueError(f"{name} must be a Vec3 (x, y, z), got {v!r}")


def _studs(v: Any) -> list[int]:
    if isinstance(v, str):
        v = [int(s) for s in v.lower().split("x")]
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in v)
    ):
        raise ValueError(f"studs must be two positive integers or 'AxB', got {v!r}")
    return [int(v[0]), int(v[1])]


class Brick:
    __slots__ = ("id", "system", "studs", "p", "yaw", "h", "extra")

    def __init__(self, d: dict[str, Any], defaults: dict[str, Any]) -> None:
        self.id = str(d["id"])
        self.system = str(d.get("system", defaults["system"]))
        if self.system not in PRESETS:
            raise ValueError(f"brick {self.id}: system must be one of {sorted(PRESETS)}")
        self.studs = _studs(d.get("studs", [2, 2]))
        self.p = _pos(d["position"], f"brick {self.id}.position")
        self.yaw = _f(d.get("yaw_deg", 0.0), f"brick {self.id}.yaw_deg")
        self.h = _f(
            d.get("height", PRESETS[self.system]["brick_height"]), f"brick {self.id}.height"
        )
        keep = {"id", "system", "studs", "position", "yaw_deg", "height"}
        self.extra = {k: v for k, v in d.items() if k not in keep}

    @property
    def pitch(self) -> float:
        return PRESETS[self.system]["stud_pitch"]

    @property
    def top(self) -> float:
        return self.p[2] + self.h

    def half(self) -> tuple[float, float]:
        return (
            (self.studs[0] * self.pitch - FOOTPRINT_OFFSET) / 2.0,
            (self.studs[1] * self.pitch - FOOTPRINT_OFFSET) / 2.0,
        )

    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        a = math.radians(self.yaw)
        return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "studs": list(self.studs),
            "position": dict(zip("xyz", self.p, strict=True)),
            "yaw_deg": self.yaw,
            "height": self.h,
            **self.extra,
        }


def footprint_overlap(a: Brick, b: Brick) -> float:
    """The smallest overlap of two footprints over the separating axes [m]; <= 0 if apart."""
    ha, hb = a.half(), b.half()
    ax, bx = a.axes(), b.axes()
    d = (b.p[0] - a.p[0], b.p[1] - a.p[1])
    least = math.inf
    for axis in (*ax, *bx):
        ra = sum(ha[i] * abs(axis[0] * ax[i][0] + axis[1] * ax[i][1]) for i in range(2))
        rb = sum(hb[i] * abs(axis[0] * bx[i][0] + axis[1] * bx[i][1]) for i in range(2))
        dist = abs(d[0] * axis[0] + d[1] * axis[1])
        least = min(least, ra + rb - dist)
    return least


def on_top(lower: Brick, upper: Brick) -> bool:
    """ICL's is_top_connect: ``upper`` sits on ``lower``'s studs."""
    h = min(lower.h, upper.h)
    dz = upper.p[2] - lower.top
    if not (TOP_WINDOW[0] - 1.0) * h < dz < (TOP_WINDOW[1] - 1.0) * h:
        return False
    return footprint_overlap(lower, upper) > MIN_OVERLAP_FRACTION * min(lower.pitch, upper.pitch)


def collides(a: Brick, b: Brick) -> bool:
    h = min(a.h, b.h)
    z_overlap = min(a.top, b.top) - max(a.p[2], b.p[2])
    if z_overlap <= COLLISION_FRACTION * h:
        return False
    return footprint_overlap(a, b) > MIN_OVERLAP_FRACTION * min(a.pitch, b.pitch)


class State:
    def __init__(self, raw: str | dict | None, system: str, ground_z: float | None) -> None:
        data: dict[str, Any] = {}
        if isinstance(raw, dict):
            data = raw
        elif raw:
            data = json.loads(raw)
        self.system = str(data.get("system", system))
        g = data.get("ground_z", ground_z)
        self.ground_z = None if g is None else float(g)
        if ground_z is not None:
            self.ground_z = float(ground_z)
        self.history = list(data.get("history", []))
        self.bricks: dict[str, Brick] = {}
        for d in data.get("bricks", []):
            b = Brick(d, {"system": self.system})
            self.bricks[b.id] = b

    def ground(self) -> float:
        if self.ground_z is not None:
            return self.ground_z
        return min((b.p[2] for b in self.bricks.values()), default=0.0)

    def graph(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """(top_connect, bottom_connect): ICL's update_brick_connection."""
        up: dict[str, set[str]] = {k: set() for k in self.bricks}
        down: dict[str, set[str]] = {k: set() for k in self.bricks}
        items = list(self.bricks.values())
        for a in items:
            for b in items:
                if a is not b and on_top(a, b):
                    up[a.id].add(b.id)
                    down[b.id].add(a.id)
        return up, down

    def grounded(self, down: dict[str, set[str]], up: dict[str, set[str]]) -> set[str]:
        g = self.ground()
        seen = {k for k, b in self.bricks.items() if abs(b.p[2] - g) <= GROUND_FRACTION * b.h}
        stack = list(seen)
        while stack:
            k = stack.pop()
            for j in up[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen

    def above(self, k: str, up: dict[str, set[str]]) -> list[str]:
        """Every brick transitively on top of ``k`` (update_all_top_bricks' recursion)."""
        out, stack, seen = [], [k], {k}
        while stack:
            for j in sorted(up[stack.pop()]):
                if j not in seen:
                    seen.add(j)
                    out.append(j)
                    stack.append(j)
        return out

    def dump(self) -> str:
        return json.dumps(
            {
                "schema": "build-state/1",
                "system": self.system,
                "ground_z": self.ground_z,
                "bricks": [b.to_dict() for b in self.bricks.values()],
                "history": self.history[-200:],
            },
            sort_keys=True,
        )


def _turn(p: list[float], about: list[float], deg: float) -> list[float]:
    a = math.radians(deg)
    dx, dy = p[0] - about[0], p[1] - about[1]
    return [
        about[0] + math.cos(a) * dx - math.sin(a) * dy,
        about[1] + math.sin(a) * dx + math.cos(a) * dy,
        p[2],
    ]


def _yaw_diff(a: float, b: float, studs: list[int]) -> float:
    period = 90.0 if studs[0] == studs[1] else 180.0
    return abs((a - b + period / 2.0) % period - period / 2.0)


def run(
    ctx: NodeContext,
    action: str,
    state_json: str = "",
    brick: dict[str, Any] | None = None,
    brick_id: str | None = None,
    new_position: dict[str, float] | None = None,
    new_yaw_deg: float | None = None,
    carry: bool = True,
    observations: list[dict[str, Any]] | None = None,
    expect_visible: str = "top",
    apply_observation: bool = False,
    system: str = "duplo",
    ground_z: float | None = None,
) -> Output:
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
    if system not in PRESETS:
        raise ValueError(f"system must be one of {sorted(PRESETS)}, got {system!r}")
    if expect_visible not in ("top", "all"):
        raise ValueError(f"expect_visible must be 'top' or 'all', got {expect_visible!r}")
    st = State(state_json, system, ground_z)
    out: Output = {
        "route": "state_updated",
        "detail": "",
        "state_json": "",
        "n_bricks": 0,
        "n_connections": 0,
        "n_grounded": 0,
        "floating_json": "[]",
        "changed_json": "[]",
        "discrepancies_json": "[]",
        "removable_json": "[]",
    }
    changed: list[str] = []
    discrepancies: list[dict[str, Any]] = []

    def finish(route: str, detail: str) -> Output:
        up, down = st.graph()
        grounded = st.grounded(down, up)
        out["route"], out["detail"] = route, detail
        out["state_json"] = st.dump()
        out["n_bricks"] = len(st.bricks)
        out["n_connections"] = sum(len(v) for v in up.values())
        out["n_grounded"] = len(grounded)
        out["floating_json"] = json.dumps(sorted(set(st.bricks) - grounded))
        out["changed_json"] = json.dumps(changed)
        out["discrepancies_json"] = json.dumps(discrepancies, sort_keys=True)
        out["removable_json"] = json.dumps(sorted(k for k in st.bricks if not up[k]))
        return out

    def target_id() -> str | None:
        k = brick_id if brick_id is not None else (brick or {}).get("id")
        return None if k is None else str(k)

    if action == "query":
        return finish("state_updated", f"{len(st.bricks)} bricks tracked")

    if action == "place":
        if not isinstance(brick, dict) or "id" not in brick or "position" not in brick:
            return finish("rejected", "place needs brick = {id, position, studs, yaw_deg, ...}")
        try:
            b = Brick(brick, {"system": st.system})
        except (ValueError, KeyError) as exc:
            return finish("rejected", f"bad brick: {exc}")
        if b.id in st.bricks:
            return finish("rejected", f"brick {b.id} is already placed; move or remove it")
        hit = [k for k, o in st.bricks.items() if collides(b, o)]
        if hit:
            return finish("rejected", f"brick {b.id} would interpenetrate {', '.join(hit)}")
        st.bricks[b.id] = b
        changed.append(b.id)
        up, down = st.graph()
        grounded = st.grounded(down, up)
        st.history.append({"action": "place", "id": b.id})
        where = (
            "on the plate"
            if b.id in grounded and not down[b.id]
            else (f"on {', '.join(sorted(down[b.id]))}" if down[b.id] else "unsupported")
        )
        return finish("state_updated", f"placed {b.id} {where}")

    if action in ("move", "remove"):
        k = target_id()
        if k is None or k not in st.bricks:
            return finish("rejected", f"no tracked brick {k!r}")
        up, _ = st.graph()
        riders = st.above(k, up) if carry else []
        if action == "remove":
            for j in [k, *riders]:
                del st.bricks[j]
            changed.extend([k, *riders])
            st.history.append({"action": "remove", "id": k, "carried": riders})
            return finish(
                "state_updated",
                f"removed {k}"
                + (f" with {len(riders)} brick(s) on it: {', '.join(riders)}" if riders else ""),
            )
        if new_position is None and new_yaw_deg is None:
            return finish("rejected", "move needs new_position and/or new_yaw_deg")
        b = st.bricks[k]
        old_p, old_yaw = list(b.p), b.yaw
        to_p = _pos(new_position, "new_position") if new_position is not None else old_p
        turn = (float(new_yaw_deg) - old_yaw) if new_yaw_deg is not None else 0.0
        shift = [to_p[i] - old_p[i] for i in range(3)]
        moved = {}
        for j in [k, *riders]:
            o = st.bricks[j]
            p = _turn(o.p, old_p, turn)
            moved[j] = ([p[i] + shift[i] for i in range(3)], o.yaw + turn)
        # Refuse a move into a brick that is not coming along.
        others = {j: o for j, o in st.bricks.items() if j not in moved}
        for j, (p, yaw) in moved.items():
            o = st.bricks[j]
            probe = Brick({**o.to_dict(), "position": p, "yaw_deg": yaw}, {"system": st.system})
            hit = [i for i, x in others.items() if collides(probe, x)]
            if hit:
                return finish("rejected", f"moving {k} would put {j} into {', '.join(hit)}")
        for j, (p, yaw) in moved.items():
            st.bricks[j].p, st.bricks[j].yaw = p, yaw
        changed.extend(moved)
        st.history.append({"action": "move", "id": k, "carried": riders})
        return finish(
            "state_updated",
            f"moved {k}" + (f" with {', '.join(riders)}" if riders else ""),
        )

    # observe
    if not isinstance(observations, list):
        return finish("rejected", "observe needs observations = [{position, studs, yaw_deg, id?}]")
    try:
        obs = [
            {
                "id": None if o.get("id") is None else str(o["id"]),
                "p": _pos(o["position"], "observation.position"),
                "studs": _studs(o.get("studs", [2, 2])),
                "yaw": _f(o.get("yaw_deg", 0.0), "observation.yaw_deg"),
            }
            for o in observations
        ]
    except (ValueError, KeyError) as exc:
        return finish("rejected", f"bad observation: {exc}")
    up, _ = st.graph()
    expected = [k for k in st.bricks if expect_visible == "all" or not up[k]]
    used: set[int] = set()
    matched: dict[str, int] = {}
    # Labelled observations first, then nearest same-size ones.
    for i, o in enumerate(obs):
        if o["id"] is not None and o["id"] in st.bricks:
            matched[o["id"]] = i
            used.add(i)
    pairs = []
    for k in st.bricks:
        if k in matched:
            continue
        b = st.bricks[k]
        for i, o in enumerate(obs):
            if i in used or o["id"] is not None or sorted(o["studs"]) != sorted(b.studs):
                continue
            d = math.dist(o["p"], b.p)
            if d <= MATCH_RADIUS_FRACTION * b.pitch:
                pairs.append((d, k, i))
    for _d, k, i in sorted(pairs):
        if k not in matched and i not in used:
            matched[k] = i
            used.add(i)
    stud_h = PRESETS[st.system]["stud_height"]
    for k in st.bricks:
        b = st.bricks[k]
        if k not in matched:
            if k in expected:
                discrepancies.append({"id": k, "kind": "missing"})
            continue
        o = obs[matched[k]]
        dxy = math.hypot(o["p"][0] - b.p[0], o["p"][1] - b.p[1])
        dz = o["p"][2] - b.p[2]
        dyaw = _yaw_diff(o["yaw"], b.yaw, b.studs)
        tol = POSITION_TOLERANCE_FRACTION * b.pitch
        if dxy > tol or abs(dz) > HEIGHT_TOLERANCE_FRACTION * stud_h or dyaw > YAW_TOLERANCE_DEG:
            discrepancies.append(
                {
                    "id": k,
                    "kind": "displaced",
                    "dxy_m": round(dxy, 5),
                    "dz_m": round(dz, 5),
                    "dyaw_deg": round(dyaw, 2),
                }
            )
    extras = [i for i in range(len(obs)) if i not in used]
    for i in extras:
        discrepancies.append(
            {
                "kind": "extra",
                "observation": i,
                "position": dict(zip("xyz", obs[i]["p"], strict=True)),
            }
        )
    if not discrepancies:
        return finish("state_updated", f"{len(matched)} tracked bricks seen where expected")
    summary = ", ".join(
        f"{n} {kind}"
        for kind in ("missing", "displaced", "extra")
        if (n := sum(1 for d in discrepancies if d["kind"] == kind))
    )
    if apply_observation:
        for d in discrepancies:
            if d["kind"] == "missing":
                del st.bricks[d["id"]]
                changed.append(d["id"])
            elif d["kind"] == "displaced":
                o, b = obs[matched[d["id"]]], st.bricks[d["id"]]
                b.p, b.yaw = list(o["p"]), o["yaw"]
                changed.append(d["id"])
        n0 = len(st.bricks)
        for i in extras:
            new_id = f"seen_{len(st.history)}_{i}"
            st.bricks[new_id] = Brick(
                {
                    "id": new_id,
                    "studs": obs[i]["studs"],
                    "position": obs[i]["p"],
                    "yaw_deg": obs[i]["yaw"],
                },
                {"system": st.system},
            )
            changed.append(new_id)
        st.history.append({"action": "observe", "applied": changed, "added": len(st.bricks) - n0})
        return finish(
            "inconsistent_with_observation", f"{summary}; the state now follows the observation"
        )
    return finish("inconsistent_with_observation", f"{summary}; the state is unchanged")
