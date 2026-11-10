# SPDX-License-Identifier: Apache-2.0
"""Hub skill bundles as ``rsh-task/1`` TaskSkills for the LEGO graph (``graph.lego_press_hub``).

- ``hub.pull_up_check`` - skills/verifying-a-brick-connection-by-pull-up, right after ``lego.guarded_seat``
  while the hand still grips the brick (lifts 3 mm in 0.5 mm steps, clutch read from tracking lag; the
  brick is put back on its seat and kept gripped: the graph's release node opens the jaws);
- ``hub.verify_seat_depth`` - skills/verifying-brick-seated-by-height, after release and
  ``lego.verify_connection``: wrist depth over the seat;
- ``hub.classify_resting_face`` - skills/classifying-brick-resting-face-from-depth, before a piece is
  picked from the feeder;
- ``hub.track_build`` - skills/tracking-build-state: the build's world model, seeded from the pre-built
  bricks and updated with every placed brick's observed pose; fails when the new brick is not connected
  (ICL's rule: not on the studs of the plate or a brick) or a tracked brick has moved.

Statuses: the hub route maps to ``succeeded`` / ``failed`` (``lego.*`` failure types) / ``unverified``;
the route, the hub output and the gap tool calls are kept in ``evidence``.
"""

from __future__ import annotations

import json
import math
from typing import Any

from roboskillhub.runtime.task import SkillContext

from .bridge import GapTools, load_script

VERSION = "0.1.0"
READINESS = "unit-tested"
_OBJ, _STR, _ARR = {"type": "object"}, {"type": "string"}, {"type": "array"}


def _schema(required: list[str], **props: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def _yaw_deg(q_xyzw) -> float:
    x, y, z, w = q_xyzw
    # Heading of the brick's +x (the 2-stud side) in the plane.
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    return math.degrees(math.atan2(r10, r00))


def _ids(request: dict) -> list[str]:
    ids = list(request.get("pieces") or [])
    for m in request.get("prebuilt") or []:
        if m.get("target_id") not in ids:
            ids.append(m["target_id"])
    mating = request.get("mating") or {}
    plate = (mating.get("baseplate") or {}).get("object_id") or request.get("plate_id")
    if plate and plate not in ids:
        ids.append(plate)
    return ids


def _support_z(p, request: dict) -> float:
    """The surface the brick lies on: the baseplate's top when the brick stands over the plate (a brick
    tipped over its edge), else the table."""
    table = float(request.get("table_z", 0.0))
    lat = request.get("lattice")
    if not lat:
        return table
    pl = lat["T_W_baseplate"]["position_m"]
    half = 0.008 * lat.get("plate_studs", [32, 32])[0] / 2.0
    if abs(p[0] - pl[0]) <= half and abs(p[1] - pl[1]) <= half:
        return float(pl[2])
    return table


class _HubSkill:
    kind = "skill"
    version = VERSION
    readiness = READINESS
    bundle = script = ""
    routes: dict[str, tuple[str, str | None]] = {}
    measured: tuple[str, ...] = ()

    def _run_hub(self, ctx: SkillContext, request: dict, **params: Any) -> tuple[dict, GapTools]:
        tools = GapTools(ctx, _ids(request), seed=int(request.get("seed", 0)))
        out = load_script(self.bundle, self.script).run(tools, **params)
        return out, tools

    def _finish(
        self,
        ctx: SkillContext,
        out: dict,
        tools: GapTools,
        outputs: dict | None = None,
        observe: list[str] | None = None,
    ):
        status, failure = self.routes.get(out["route"], ("unverified", "evidence_missing"))
        hub = {k: v for k, v in out.items() if isinstance(v, (str, int, float, bool, dict, list))}
        observations = dict(tools.observations)
        if observe:
            r = ctx.observe(observe, require_orientation=True)
            for o in (r.get("evidence") or {}).get("observations", []):
                if o.get("position_m") is not None:
                    observations[o["object_id"]] = o
        return ctx.result(
            self,
            status,
            phase="verify",
            failure_type=failure,
            observations=list(observations.values()),
            evidence={
                "postcondition": {
                    "name": f"{self.bundle}:{out['route']}",
                    "passed": status == "succeeded",
                    "measured": {k: hub[k] for k in self.measured if k in hub},
                },
                "hub_bundle": self.bundle,
                "hub_route": out["route"],
                "reason": out.get("detail", ""),
                "hub_outputs": hub,
                "gap_tools": sorted(set(tools.calls)),
                "depth_renders": tools.renders,
            },
            outputs={"route": out["route"], **(outputs or {})},
        )


class PullUpCheck(_HubSkill):
    skill_id = "hub.pull_up_check"
    bundle, script = "verifying-a-brick-connection-by-pull-up", "pull_up_check.py"
    requires = frozenset({"pose_feedback", "cartesian_motion", "gripper_feedback", "stop"})
    input_schema = _schema(
        [], object_id=_STR, lift_height_m={"type": "number"}, hold_lag_m={"type": "number"}
    )
    measured = ("rise", "lag_over_baseline", "steps", "signal")
    routes = {
        "connected": ("succeeded", None),
        "not_connected": ("failed", "lego.not_seated"),
        "not_holding": ("failed", "grasp_lost"),
        "blocked": ("unverified", "evidence_missing"),
    }

    def run(self, request: dict, ctx: SkillContext) -> dict:
        kw: dict[str, Any] = {"release": False, "put_back": True, "signal": "lag"}
        if "lift_height_m" in request:
            kw["lift_height"] = float(request["lift_height_m"])
        if "hold_lag_m" in request:
            kw["hold_lag"] = float(request["hold_lag_m"])
        # Station time follows wall time; the settle loop's own sleeps let it catch up.
        tools = GapTools(ctx, [], seed=int(request.get("seed", 0)))
        tools.watch = request.get("object_id")
        out = load_script(self.bundle, self.script).run(tools, **kw)
        log = tools.watch_log
        zs = [e["obj_z"] for e in log if e["obj_z"] is not None]
        brick_rise = (max(zs) - zs[0]) if len(zs) > 1 else None
        hand = [e["tcp_z"] for e in log]
        hand_rise = (max(hand) - hand[0]) if len(hand) > 1 else None
        held = [request["object_id"]] if request.get("object_id") else None
        result = self._finish(
            ctx,
            out,
            tools,
            {"connected": bool(out.get("connected")), "brick_rise_m": brick_rise},
            observe=held,
        )
        result["evidence"]["pull_log"] = log
        result["evidence"]["brick_rise_m"] = brick_rise
        result["evidence"]["hand_rise_m"] = hand_rise
        return result


class VerifySeatDepth(_HubSkill):
    skill_id = "hub.verify_seat_depth"
    bundle, script = "verifying-brick-seated-by-height", "verify_seat.py"
    requires = frozenset({"pose_feedback", "cartesian_motion", "object_pose_feedback"})
    input_schema = _schema(
        ["mating"], mating=_OBJ, pieces=_ARR, prebuilt=_ARR, seed={"type": "integer"}
    )
    measured = ("height_error", "tilt_deg", "coverage", "reference", "samples")
    routes = {
        "seated": ("succeeded", None),
        "not_seated": ("failed", "lego.not_seated"),
        "high_by_stud": ("failed", "lego.not_seated"),
        "tilted": ("failed", "lego.partially_seated"),
        "shifted_off_grid": ("failed", "lego.misaligned"),
        "no_brick": ("failed", "lego.lifted_with_gripper"),
        "unverified": ("unverified", "evidence_missing"),
        "unreachable": ("unverified", "unreachable"),
    }

    def run(self, request: dict, ctx: SkillContext) -> dict:
        m = request["mating"]
        pose = m["mating_pose"]
        seat = dict(zip("xyz", map(float, pose["position_m"]), strict=True))
        out, tools = self._run_hub(
            ctx,
            request,
            seat_position=seat,
            seat_yaw_deg=_yaw_deg(pose["quaternion_xyzw"]),
            brick_studs=list(m.get("brick", {}).get("studs", [2, 4])),
            system=m.get("brick", {}).get("system", "lego"),
        )
        return self._finish(
            ctx,
            out,
            tools,
            {
                "seated": out["route"] == "seated",
                "height_error_m": float(out.get("height_error", 0.0)),
            },
        )


class ClassifyRestingFace(_HubSkill):
    skill_id = "hub.classify_resting_face"
    bundle, script = "classifying-brick-resting-face-from-depth", "classify_face.py"
    requires = frozenset({"pose_feedback", "cartesian_motion", "object_pose_feedback"})
    input_schema = _schema(
        ["object_id"],
        object_id=_STR,
        pieces=_ARR,
        studs=_ARR,
        system=_STR,
        table_z={"type": "number"},
        lattice=_OBJ,
        seed={"type": "integer"},
    )
    measured = ("vertical_axis", "height", "deep_share", "yaw_deg", "samples")
    routes = {
        "studs_up": ("succeeded", None),
        "upside_down": ("failed", "lego.wrong_face_up"),
        "on_side": ("failed", "lego.wrong_face_up"),
        "unknown": ("unverified", "evidence_missing"),
        "no_brick": ("failed", "object_not_observed"),
        "unreachable": ("unverified", "unreachable"),
    }

    def run(self, request: dict, ctx: SkillContext) -> dict:
        oid = request["object_id"]
        seen = (ctx.observe([oid]).get("evidence") or {}).get("observations") or [{}]
        p = seen[0].get("position_m")
        if p is None:
            return ctx.result(
                self, "failed", phase="verify", failure_type="object_not_observed",
                evidence={"reason": f"{oid} not observed"},
            )  # fmt: skip
        pieces = list(request.get("pieces") or [oid])
        support = _support_z(p, request)
        out, tools = self._run_hub(
            ctx,
            {**request, "pieces": pieces},
            brick_position={"x": float(p[0]), "y": float(p[1]), "z": support},
            brick_studs=list(request.get("studs", [2, 4])),
            system=request.get("system", "lego"),
            table_height=support,
        )
        result = self._finish(
            ctx,
            out,
            tools,
            {
                "face": out["route"],
                "vertical_axis": out.get("vertical_axis", ""),
                "up_vector": out.get("up_vector"),
                "centre": out.get("centre"),
                "yaw_deg": float(out.get("yaw_deg", 0.0)),
            },
        )
        # For scoring only (not used by any decision): the face the privileged pose says.
        q = seen[0].get("quaternion_xyzw")
        if q is not None:
            x, y, z, w = q
            up_z = 1 - 2 * (x * x + y * y)
            result["evidence"]["truth_face"] = (
                "studs_up" if up_z > 0.9 else "upside_down" if up_z < -0.9 else "on_side"
            )
        return result


def _ray_entry(p: tuple, d: tuple, rect: dict) -> float | None:
    """Distance along the ray p + t d (t > 0) at which it enters an axis-aligned rect from outside."""
    lo, hi = (rect["x"][0], rect["y"][0]), (rect["x"][1], rect["y"][1])
    if all(lo[i] <= p[i] <= hi[i] for i in (0, 1)):
        return None
    t0, t1 = 0.0, math.inf
    for i in (0, 1):
        if abs(d[i]) < 1e-9:
            if not lo[i] <= p[i] <= hi[i]:
                return None
            continue
        a, b = (lo[i] - p[i]) / d[i], (hi[i] - p[i]) / d[i]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return t0 if t0 <= t1 and t0 > 0 else None


def choose_lip(c: dict, lips: list[dict] | None, reach: float = 0.12) -> tuple[float, float] | None:
    """(lip distance, yaw_deg to pass) for the tip: along -u for a brick on a side, along +-x of the brick
    for an upside-down one; the nearest fence, else the baseplate (a brick tipped onto the plate's studs is
    snapped on by the clutch and can no longer be picked: fences first)."""
    if not lips:
        return None
    u, yaw = c["up_vector"], float(c.get("yaw_deg", 0.0))
    if u["z"] <= -0.9:
        options = [(yaw, (math.cos(math.radians(yaw)), math.sin(math.radians(yaw))))]
        options.append((yaw + 180.0, (-options[0][1][0], -options[0][1][1])))
    else:
        n = math.hypot(u["x"], u["y"]) or 1.0
        options = [(yaw, (-u["x"] / n, -u["y"] / n))]
    p = (c["centre"]["x"], c["centre"]["y"])
    best = None
    for rank, kind in enumerate(("fence", "baseplate")):
        for y_deg, d in options:
            for r in lips:
                if r.get("kind", "fence") != kind:
                    continue
                t = _ray_entry(p, d, r)
                if t is not None and t <= reach and (best is None or (rank, t) < best[0]):
                    best = ((rank, t), (t, y_deg))
        if best is not None:
            break
    return None if best is None else best[1]


class FlipStudsUp(_HubSkill):
    """skills/regrasping-a-brick-to-turn-studs-up: one quarter-turn tip from the face classifier's reading."""

    skill_id = "hub.flip_studs_up"
    bundle, script = "regrasping-a-brick-to-turn-studs-up", "tip_brick.py"
    requires = frozenset({"pose_feedback", "cartesian_motion", "gripper", "stop"})
    input_schema = _schema(
        ["object_id", "classified"],
        object_id=_STR,
        classified=_OBJ,
        lips=_ARR,
        table_z={"type": "number"},
        studs=_ARR,
        system=_STR,
    )
    routes = {
        "tipped": ("succeeded", None),
        "already_studs_up": ("succeeded", None),
        "blocked": ("failed", "unreachable"),
    }
    measured = ("push_height", "travel", "tips_left")

    def run(self, request: dict, ctx: SkillContext) -> dict:
        c = request["classified"]
        if not c.get("up_vector") or not c.get("centre"):
            return ctx.result(
                self, "rejected", phase="validate", failure_type="precondition_unmet",
                evidence={"reason": "no up_vector / centre from the face classifier"},
            )  # fmt: skip
        tools = GapTools(ctx, [], seed=0)
        chosen = choose_lip(c, request.get("lips"))
        if request.get("lips") is not None and chosen is None:
            return ctx.result(
                self, "failed", phase="validate", failure_type="unreachable",
                evidence={"reason": "no fence or plate edge along the push direction to tip the brick over"},
            )  # fmt: skip
        lip, yaw_used = chosen if chosen else (None, float(c.get("yaw_deg", 0.0)))
        out = load_script(self.bundle, self.script).run(
            tools,
            brick_position=c["centre"],
            up_vector=c["up_vector"],
            yaw_deg=yaw_used,
            table_height=float(request.get("table_z", 0.0)),
            brick_studs=list(request.get("studs", [2, 4])),
            system=request.get("system", "lego"),
            lip_distance=lip,
        )
        # The tip is judged by the next face reading, not here: observe the brick for the record.
        return self._finish(
            ctx, out, tools, {"route": out["route"]}, observe=[request["object_id"]]
        )


class TrackBuild(_HubSkill):
    """``action: init`` seeds the world model from the pre-built bricks' observed poses; ``action: place``
    adds ``object_id`` at its observed pose and checks every tracked brick against a fresh observation."""

    skill_id = "hub.track_build"
    kind = "tool"
    bundle, script = "tracking-build-state", "track_build.py"
    requires = frozenset({"object_pose_feedback"})
    input_schema = _schema(
        ["action"],
        action={"enum": ["init", "place"]},
        state_json=_STR,
        object_id=_STR,
        prebuilt=_ARR,
        plate_id=_STR,
        mating=_OBJ,
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        track = load_script(self.bundle, self.script).run
        state = request.get("state_json") or ""
        tracked = [b["id"] for b in json.loads(state)["bricks"]] if state else []
        plate = request.get("plate_id") or (
            (request.get("mating") or {}).get("baseplate") or {}
        ).get("object_id")
        new = (
            [m["target_id"] for m in request.get("prebuilt") or []]
            if request["action"] == "init"
            else [request["object_id"]]
        )
        want = [*dict.fromkeys([*tracked, *new, *([plate] if plate else [])])]
        r = ctx.observe(want, require_orientation=True)
        obs = {o["object_id"]: o for o in (r.get("evidence") or {}).get("observations", [])}
        ground = None
        if plate and (obs.get(plate) or {}).get("position_m") is not None:
            ground = float(obs[plate]["position_m"][2])

        def brick(oid: str) -> dict | None:
            o = obs.get(oid) or {}
            if o.get("position_m") is None or o.get("quaternion_xyzw") is None:
                return None
            return {
                "id": oid,
                "studs": [2, 4],
                "position": list(o["position_m"]),
                "yaw_deg": _yaw_deg(o["quaternion_xyzw"]),
                "height": 0.0096,
            }

        last: dict = {"route": "state_updated", "detail": "nothing to add", "state_json": state}
        missing = []
        for oid in new:
            b = brick(oid)
            if b is None:
                missing.append(oid)
                continue
            last = track(
                ctx, action="place", state_json=last["state_json"], brick=b, system="lego",
                ground_z=ground,
            )  # fmt: skip
            if last["route"] != "state_updated":
                break
        evidence = {"hub_bundle": self.bundle, "hub_route": last["route"], "reason": last["detail"]}
        if missing:
            return ctx.result(
                self, "failed", phase="verify", failure_type="object_not_observed",
                evidence={**evidence, "missing": missing}, outputs={"state_json": state},
            )  # fmt: skip
        if last["route"] == "rejected":
            return ctx.result(
                self, "failed", phase="verify", failure_type="lego.structure_damaged",
                evidence=evidence, outputs={"state_json": state},
            )  # fmt: skip
        state_now = last["state_json"]
        if not state_now:
            state_now = track(ctx, action="query", state_json="", system="lego", ground_z=ground)[
                "state_json"
            ]
        seen = [b for oid in json.loads(state_now)["bricks"] if (b := brick(oid["id"]))]
        check = track(ctx, action="observe", state_json=state_now, observations=[
            {"id": b["id"], "position": b["position"], "studs": b["studs"], "yaw_deg": b["yaw_deg"]}
            for b in seen
        ], expect_visible="all", system="lego")  # fmt: skip
        floating = json.loads(check["floating_json"])
        discrepancies = json.loads(check["discrepancies_json"])
        evidence.update(
            floating=floating,
            discrepancies=discrepancies,
            n_bricks=check["n_bricks"],
            n_connections=check["n_connections"],
        )
        outputs = {"state_json": state_now, "floating": floating, "n_bricks": check["n_bricks"]}
        if request["action"] == "place" and request["object_id"] in floating:
            return ctx.result(
                self, "failed", phase="verify", failure_type="lego.not_seated",
                evidence={**evidence, "reason": f"{request['object_id']} is not on any studs"},
                outputs=outputs,
            )  # fmt: skip
        if discrepancies:
            return ctx.result(
                self, "failed", phase="verify", failure_type="lego.structure_damaged",
                evidence=evidence, outputs=outputs,
            )  # fmt: skip
        return ctx.result(self, "succeeded", phase="verify", evidence=evidence, outputs=outputs)


SKILLS = (PullUpCheck(), VerifySeatDepth(), ClassifyRestingFace(), FlipStudsUp(), TrackBuild())
