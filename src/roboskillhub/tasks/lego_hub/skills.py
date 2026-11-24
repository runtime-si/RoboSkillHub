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
        tools = GapTools(ctx, _ids(request), seed=_noise_seed(ctx, request))
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


MAX_PULL_RISE_M = 0.0006
"""judging-assembly-steps-by-fused-checks' max_pull_rise_m: a clutched brick rises at most this much."""
SLIP_MARGIN_M = 0.001
"""The hand rose at least this much more than the brick: the jaws slid on it."""


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
        tools = GapTools(ctx, [], seed=_noise_seed(ctx, request))
        tools.watch = request.get("object_id")
        out = load_script(self.bundle, self.script).run(tools, **kw)
        log = tools.watch_log
        zs = [e["obj_z"] for e in log if e["obj_z"] is not None]
        brick_rise = (max(zs) - zs[0]) if len(zs) > 1 else None
        hand = [e["tcp_z"] for e in log]
        hand_rise = (max(hand) - hand[0]) if len(hand) > 1 else None
        held = [request["object_id"]] if request.get("object_id") else None
        slid = (
            out["route"] == "not_connected"
            and brick_rise is not None
            and hand_rise is not None
            and brick_rise <= MAX_PULL_RISE_M
            and hand_rise - brick_rise >= SLIP_MARGIN_M
        )
        if slid:
            # The hand rose and the brick did not: the jaws slid on a clutched brick (2 N pads on the
            # Isaac station), which the hand's tracking lag alone reads as "nothing held the lift".
            out = {
                **out,
                "route": "connected",
                "detail": out.get("detail", "") + "; the brick stayed: jaws slid",
            }
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


def _noise_seed(ctx: SkillContext, request: dict) -> int:
    """The depth-noise seed of one call: the request's ``seed`` if given, else one derived from the
    request id (episode, node, attempt). A constant 0 gave every episode the same noise draw, so episodes
    of different seeds were not independent under depth noise."""
    import zlib

    if request.get("seed") is not None:
        return int(request["seed"])
    return zlib.crc32(str(getattr(ctx, "request_id", "") or getattr(ctx, "task_id", "")).encode())


def _seat_options() -> dict:
    """verifying-brick-seated-by-height options from the environment (LEGO_SEAT_ESTIMATOR,
    LEGO_SEAT_HEIGHT_TOL_M, LEGO_SEAT_TILT_TOL_DEG); none set: the skill's defaults."""
    import os

    opts: dict = {}
    if os.environ.get("LEGO_SEAT_ESTIMATOR"):
        opts["face_estimator"] = os.environ["LEGO_SEAT_ESTIMATOR"]
    if os.environ.get("LEGO_SEAT_HEIGHT_TOL_M"):
        opts["height_tolerance"] = float(os.environ["LEGO_SEAT_HEIGHT_TOL_M"])
    if os.environ.get("LEGO_SEAT_TILT_TOL_DEG"):
        opts["tilt_tolerance_deg"] = float(os.environ["LEGO_SEAT_TILT_TOL_DEG"])
    return opts


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
            **_seat_options(),
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


class JudgePlace(_HubSkill):
    """skills/judging-assembly-steps-by-fused-checks on one placement: the pull-up (before the release) and
    the depth seat (after it) fused into done / failed / uncertain. No motion."""

    skill_id = "hub.judge_place"
    kind = "tool"
    bundle, script = "judging-assembly-steps-by-fused-checks", "fuse_checks.py"
    requires = frozenset()
    input_schema = _schema(["pull", "depth"], pull=_OBJ, depth=_OBJ, release_status=_STR)

    def run(self, request: dict, ctx: SkillContext) -> dict:
        pull, depth = request["pull"], request["depth"]
        out = load_script(self.bundle, self.script).run(
            ctx,
            step_type="place",
            release_grip_route="opened" if request.get("release_status") == "succeeded" else "",
            seat_route=str(depth.get("route") or ""),
            height_error_m=depth.get("height_error_m"),
            pull_route=str(pull.get("route") or ""),
            brick_rise_m=pull.get("brick_rise_m"),
        )
        status = {"done": "succeeded", "failed": "failed"}.get(out["route"], "unverified")
        return ctx.result(
            self,
            status,
            phase="verify",
            failure_type={"succeeded": None, "failed": "lego.not_seated"}.get(
                status, "evidence_missing"
            ),
            evidence={
                "hub_bundle": self.bundle,
                "hub_route": out["route"],
                "reason": out.get("reason", ""),
            },
            outputs={"verdict": out["route"]},
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
        # How far every brick already tracked moved since it was tracked (a press tilting or pushing the layer
        # below shows here first).
        prev = {b["id"]: b for b in json.loads(state)["bricks"]} if state else {}
        moved = {}
        for oid, b0 in prev.items():
            b1 = brick(oid)
            if b1 is not None:
                moved[oid] = (
                    round(
                        1e3
                        * max(
                            abs(
                                b1["position"][i]
                                - b0["position"][
                                    i if isinstance(b0["position"], list) else "xyz"[i]
                                ]
                            )
                            for i in range(3)
                        ),
                        3,
                    )
                    if isinstance(b0["position"], list)
                    else round(
                        1e3
                        * max(abs(b1["position"][i] - b0["position"]["xyz"[i]]) for i in range(3)),
                        3,
                    )
                )
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
            moved_mm=moved,
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


TIP_REACH_M = 0.0102
"""Closed fingertips below the TCP [m] (the LEGO station's hand; regrasping-a-brick-to-turn-studs-up)."""
STUD_TOP_M = 0.0113
"""A LEGO brick's stud tops over its bottom face [m] (9.6 mm body + 1.7 mm stud)."""
TIP_OVERTRAVEL_M = 0.0015
"""Push the fingertips this far under the seated stud tops: seats the brick whether they land on studs
or between them (body top 1.7 mm lower)."""
TIP_SEATED_M = 0.0006
"""Seated after the push: bottom face at most this far over the seat (lego.guarded_seat's depth band)."""


class FingertipPress(_HubSkill):
    """Seat a brick the gripped press left on the stud tops: at a light grip the pads slide up the brick
    instead of pushing it down. Open, lift, close the jaws and push the brick's top down with the closed
    fingertips over its centre, then lift with the jaws open to ``retreat_height_m`` over the press end, the
    pose ``lego.release`` retreats to (it then opens and finds itself retreated)."""

    skill_id = "hub.fingertip_press"
    requires = frozenset({"pose_feedback", "cartesian_motion", "gripper"})
    input_schema = _schema(
        ["object_id", "alignment", "press", "open_width_m", "retreat_height_m"],
        object_id=_STR,
        alignment=_OBJ,
        press=_OBJ,
        open_width_m={"type": "number"},
        retreat_height_m={"type": "number"},
        speed_m_s={"type": "number"},
    )

    def run(self, request: dict, ctx: SkillContext) -> dict:
        oid = request["object_id"]
        seat = request["alignment"]["evidence"]["plan"]["seat_pose"]
        end = request["press"].get("press_end_tcp_pose")
        if not end:
            return ctx.result(
                self, "rejected", phase="validate", failure_type="precondition_unmet",
                evidence={"reason": "no press end pose from lego.guarded_seat"},
            )  # fmt: skip
        speed = float(request.get("speed_m_s", 0.005))
        seat_z = float(seat["position_m"][2])
        trace: list[dict] = []

        def brick() -> dict | None:
            r = ctx.observe([oid], require_orientation=True)
            for o in (r.get("evidence") or {}).get("observations", []):
                if o.get("object_id") == oid and o.get("position_m") is not None:
                    return o
            return None

        def move(label: str, xyz: list[float]) -> dict:
            r = ctx.invoke(
                "move_pose", {"target_pose": {**end, "position_m": list(xyz)}, "speed_m_s": speed}
            )
            trace.append(
                {"step": label, "status": r.get("status"), "failure": r.get("failure_type")}
            )
            return r

        def grip(label: str, width: float) -> dict:
            r = ctx.invoke(
                "set_gripper", {"width_m": width, "purpose": "release", "target_id": oid}
            )
            trace.append(
                {"step": label, "status": r.get("status"), "failure": r.get("failure_type")}
            )
            return r

        before = brick()
        if before is None:
            return ctx.result(
                self, "unverified", phase="validate", failure_type="evidence_missing",
                evidence={"reason": f"{oid} not observed"},
            )  # fmt: skip
        dz0 = float(before["position_m"][2]) - seat_z
        ex, ey, ez = end["position_m"]
        bx, by = float(before["position_m"][0]), float(before["position_m"][1])
        retreat_z = ez + float(request["retreat_height_m"])
        tip_z = seat_z + STUD_TOP_M - TIP_OVERTRAVEL_M + TIP_REACH_M
        hover_z = max(seat_z + dz0 + STUD_TOP_M + TIP_REACH_M + 0.005, retreat_z)
        failed = None
        if grip("open", float(request["open_width_m"])).get("status") != "succeeded":
            failed = "could not open the jaws"
        elif move("lift", [ex, ey, hover_z]).get("status") != "succeeded":
            failed = "lift refused"
        elif grip("close", 0.0).get("status") not in ("succeeded", "unverified"):
            failed = "could not close the jaws"
        elif move("over the brick", [bx, by, hover_z]).get("status") != "succeeded":
            failed = "move over the brick refused"
        else:
            # The push ends on the brick: the hand may stop short of the commanded depth.
            move("push", [bx, by, tip_z])
            move("up", [bx, by, hover_z])
            grip("open again", float(request["open_width_m"]))
            move("to the retreat pose", [ex, ey, retreat_z])
        after = brick()
        dz1 = None if after is None else float(after["position_m"][2]) - seat_z
        ok = failed is None and dz1 is not None and dz1 <= TIP_SEATED_M
        return ctx.result(
            self,
            "succeeded" if ok else "failed",
            phase="verify",
            failure_type=None if ok else "lego.misaligned",
            observations=[o for o in (before, after) if o],
            evidence={
                "postcondition": {
                    "name": "hub.fingertip_press:seated",
                    "passed": ok,
                    "measured": {"dz_before_m": dz0, "dz_after_m": dz1, "tip_z_m": tip_z},
                },
                "reason": failed
                or f"brick {dz0 * 1e3:+.2f} -> {(dz1 or 0) * 1e3:+.2f} mm over its seat",
                "trace": trace,
            },
            outputs={"dz_before_m": dz0, "dz_after_m": dz1},
        )


SKILLS = (
    PullUpCheck(),
    VerifySeatDepth(),
    ClassifyRestingFace(),
    FlipStudsUp(),
    JudgePlace(),
    TrackBuild(),
    FingertipPress(),
)
