"""Two observation providers behind one interface.

* :class:`SimTruthProvider` reads the simulator's object states. Every result is
  ``privileged=True`` / ``provider="sim_truth"``; it exists for wiring and for
  scoring the other provider, never as a substitute for it.
* :class:`RgbdProvider` fits each object's known box to its masked depth points
  with the hub's ``tools/pose6d`` primitive path (symmetry folding, fitness /
  coverage / ambiguity), optionally with the table as support plane, then adds the
  domino state or the LEGO lattice cell. Lying dominoes get their pip face from
  ``reading-domino-pips-and-orientation``'s ``read_face`` when an RGB image is given.

Segmentation is an input (masks with object ids): this module does not decide
what is in the image; the hub's segmenters (sam3, owlv2, colour) do that upstream.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import _hub
from .contract import ObjectObservation, Uncertainty
from .domino import domino_state
from .frames import inv_T, make_T
from .lego import snap_to_lattice
from .objects import ObjectSpec

#: Coverage under which a registered fit is reported ``partially_occluded``. The
#: fast path's correct fits had coverage >= 0.6 on unoccluded synthetic renders
#: (pose6d SKILL.md); its own accept floor is 0.4.
FULL_COVERAGE = 0.6
#: Fewest masked depth points to attempt a fit (pose6d refuses under 12).
MIN_POINTS = 30
#: Position / rotation std floors for the residual heuristic [m, rad].
POS_STD_FLOOR_M = 0.0005
ROT_STD_FLOOR_RAD = math.radians(0.5)


class ObservationProvider(Protocol):
    name: str
    privileged: bool

    def observe(self, object_ids: Sequence[str] | None = None) -> list[ObjectObservation]: ...


@dataclass
class SimTruthProvider:
    """``read_truth() -> {"timestamp_s": t, "frame_id": f, "objects": {id: {"T_frame_object": 4x4 | pose dict, "object_class": c}}}``."""

    read_truth: Callable[[], dict[str, Any]]
    specs: dict[str, ObjectSpec]
    table_z: float | None = None
    name: str = "sim_truth"
    privileged: bool = True

    def observe(self, object_ids: Sequence[str] | None = None) -> list[ObjectObservation]:
        truth = self.read_truth()
        frame, t = truth["frame_id"], float(truth["timestamp_s"])
        out = []
        ids = list(object_ids) if object_ids is not None else list(truth["objects"])
        for oid in ids:
            spec = self.specs.get(oid)
            st = truth["objects"].get(oid)
            if spec is None:
                raise KeyError(f"no ObjectSpec for {oid!r}")
            if st is None:
                out.append(
                    _unseen(oid, spec, frame, t, self.name, True, "not in the simulator state")
                )
                continue
            T = st["T_frame_object"]
            if isinstance(T, dict):
                from .frames import pose_to_T

                T = pose_to_T(T, frame)
            T = np.asarray(T, dtype=float)
            obs = ObjectObservation(
                object_id=oid,
                object_class=spec.object_class,
                size_m=spec.size_m,
                frame_id=frame,
                timestamp_s=t,
                provider=self.name,
                privileged=True,
                visibility="visible",
                T_frame_object=T,
                uncertainty=Uncertainty(0.0, 0.0, "simulator_truth"),
                evidence={"source": "simulator object state"},
            )
            _annotate(obs, spec, self.table_z, pip_face_known=True)
            out.append(obs)
        return out


@dataclass
class RgbdFrame:
    depth_m: np.ndarray
    K: np.ndarray
    T_frame_camera: np.ndarray
    timestamp_s: float
    frame_id: str
    camera_name: str = ""
    rgb: np.ndarray | None = None


@dataclass
class MaskedObject:
    object_id: str
    mask: np.ndarray
    score: float = 1.0
    source: str = ""


@dataclass
class RgbdProvider:
    """``get_frame() -> RgbdFrame``; ``segment(frame) -> list[MaskedObject]``.

    ``table_z`` (frame z of the support, up = +z) enables pose6d's support-plane
    constraint for parts resting on the table; ``resting`` limits it to the ids
    known to rest (a held or stacked part must not get it)."""

    get_frame: Callable[[], RgbdFrame]
    segment: Callable[[RgbdFrame], list[MaskedObject]]
    specs: dict[str, ObjectSpec]
    table_z: float | None = None
    resting: set[str] | None = None
    lattice: dict[str, Any] | None = None
    """``{"T_frame_plate": 4x4, "plate_studs": (nx, ny), "grid_bias_m": (bx, by)}``
    from calibrating-baseplate-stud-grid; LEGO bricks then get a lattice cell."""
    read_pips: bool = True
    name: str = "rgbd_pose6d"
    privileged: bool = False
    last_frame_evidence: dict[str, Any] = field(default_factory=dict)

    def observe(self, object_ids: Sequence[str] | None = None) -> list[ObjectObservation]:
        frame = self.get_frame()
        masks = {m.object_id: m for m in self.segment(frame)}
        ids = list(object_ids) if object_ids is not None else list(masks)
        out = []
        for oid in ids:
            spec = self.specs.get(oid)
            if spec is None:
                raise KeyError(f"no ObjectSpec for {oid!r}")
            m = masks.get(oid)
            if m is None:
                out.append(
                    _unseen(
                        oid,
                        spec,
                        frame.frame_id,
                        frame.timestamp_s,
                        self.name,
                        False,
                        "no mask for this id",
                    )
                )
                continue
            out.append(self._fit(frame, m, spec))
        return out

    def _fit(self, frame: RgbdFrame, m: MaskedObject, spec: ObjectSpec) -> ObjectObservation:
        p6 = _hub.load("pose6d")
        synth = _hub.load("pose6d_synth")
        pts_cam = synth.backproject(frame.depth_m, m.mask, frame.K)
        ev = {
            "camera": frame.camera_name,
            "mask_px": int(np.count_nonzero(m.mask)),
            "mask_score": float(m.score),
            "mask_source": m.source,
            "points": int(len(pts_cam)),
            "tool": "tools/pose6d primitive fast path",
        }
        if len(pts_cam) < MIN_POINTS:
            return _unseen(
                m.object_id,
                spec,
                frame.frame_id,
                frame.timestamp_s,
                self.name,
                False,
                f"{len(pts_cam)} depth points in the mask, need {MIN_POINTS}",
                ev,
                "occluded",
            )
        Tfc = np.asarray(frame.T_frame_camera, dtype=float)
        pts = pts_cam @ Tfc[:3, :3].T + Tfc[:3, 3]
        use_support = self.table_z is not None and (
            self.resting is None or m.object_id in self.resting
        )
        r = p6.register(
            pts,
            {"box": list(spec.size_m)},
            symmetry=spec.symmetry if spec.symmetry != "box" else None,
            viewpoint=Tfc[:3, 3],
            up=[0.0, 0.0, 1.0],
            support_plane=[0.0, 0.0, 1.0, -float(self.table_z)] if use_support else None,
        )
        ev.update(
            status=r["status"],
            fitness=r["fitness"],
            rmse_m=r["rmse"],
            coverage=r["coverage"],
            support_plane=use_support,
            runtime_s=r["runtime_s"],
            alternative=None
            if r["alternative"] is None
            else {k: r["alternative"][k] for k in ("score_gap", "rotation_deg", "translation_m")},
        )
        if use_support and r["status"] != "poor_fit" and not _rests_on(np.asarray(r["T"]), spec):
            # The support constraint pins a face to the table; a part that is not
            # resting (leaning, stacked) is refitted without it (pose6d's advice).
            ev["support_refit"] = {
                "tilt_with_support_deg": round(_rest_tilt_deg(np.asarray(r["T"])), 2)
            }
            r = p6.register(
                pts,
                {"box": list(spec.size_m)},
                symmetry=spec.symmetry if spec.symmetry != "box" else None,
                viewpoint=Tfc[:3, 3],
                up=[0.0, 0.0, 1.0],
            )
            use_support = False
            ev.update(
                status=r["status"],
                fitness=r["fitness"],
                rmse_m=r["rmse"],
                coverage=r["coverage"],
                support_plane=False,
            )
        if r["status"] == "poor_fit":
            return _unseen(
                m.object_id,
                spec,
                frame.frame_id,
                frame.timestamp_s,
                self.name,
                False,
                f"pose6d poor_fit (fitness {r['fitness']:.2f}, coverage {r['coverage']:.2f}): "
                "wrong size, wrong part or too little of it seen",
                ev,
                "occluded",
            )
        rmse = float(r["rmse"])
        visible_extent = float(sorted(spec.size_m)[1])
        unc = Uncertainty(
            position_std_m=max(rmse, POS_STD_FLOOR_M),
            rotation_std_rad=max(math.atan2(rmse, visible_extent / 2.0), ROT_STD_FLOOR_RAD),
            kind="fit_residual",
            ambiguous=r["status"] == "ambiguous_fit",
            symmetry_order=int(r["symmetry_order"]),
            yaw_ambiguity_deg=float(r["yaw_ambiguity_deg"]),
            fit={"fitness": float(r["fitness"]), "rmse_m": rmse, "coverage": float(r["coverage"])},
        )
        vis = "visible" if r["coverage"] >= FULL_COVERAGE else "partially_occluded"
        obs = ObjectObservation(
            object_id=m.object_id,
            object_class=spec.object_class,
            size_m=spec.size_m,
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            provider=self.name,
            privileged=False,
            visibility=vis,
            T_frame_object=np.asarray(r["T"], dtype=float),
            uncertainty=unc,
            reason=""
            if not unc.ambiguous
            else "pose6d ambiguous_fit: a non-equivalent pose scores within 0.03",
            evidence=ev,
        )
        pip_known = False
        if spec.object_class == "domino" and self.read_pips and frame.rgb is not None:
            pip_known = self._orient_by_pips(obs, frame, m, spec)
        _annotate(
            obs,
            spec,
            self.table_z if use_support else None,
            pip_face_known=pip_known,
            lattice=self.lattice,
        )
        return obs

    def _orient_by_pips(
        self, obs: ObjectObservation, frame: RgbdFrame, m: MaskedObject, spec: ObjectSpec
    ) -> bool:
        """For a lying tile: +z is set to the pip face when the reader finds pips or
        a divider facing the camera, and flipped when it reads ``face_down``."""
        T = obs.T_frame_object
        assert T is not None
        cam_dir = frame.T_frame_camera[:3, 3] - T[:3, 3]
        if abs(T[:3, 2] @ np.array([0, 0, 1.0])) < math.cos(math.radians(30)):
            return False  # not lying: the face is not toward the overhead camera
        try:
            face = _hub.load("domino_reader").read_face(frame.rgb, m.mask.astype(bool))
        except Exception as exc:  # the reader needs cv2; a failure leaves the face unknown
            obs.evidence["pip_read"] = f"failed: {exc}"
            return False
        seen = face["divider_found"] or face["strong_pips"] > 0
        obs.evidence["pip_read"] = {
            "divider_found": bool(face["divider_found"]),
            "strong_pips": int(face["strong_pips"]),
            "values": [h["value"] for h in face["halves"]],
            "source": "reading-domino-pips-and-orientation.read_face",
        }
        facing_cam = float(T[:3, 2] @ cam_dir) > 0
        if seen != facing_cam:
            obs.T_frame_object = T @ make_T(
                np.diag([1.0, -1.0, -1.0])
            )  # half-turn about x: +z flips
        return True


def _rest_tilt_deg(T: np.ndarray) -> float:
    return math.degrees(
        math.acos(min(1.0, float(np.max(np.abs(T[:3, :3].T @ np.array([0.0, 0.0, 1.0]))))))
    )


def _rests_on(T: np.ndarray, spec: ObjectSpec, tilt_tol_deg: float = 10.0) -> bool:
    """A face within ``tilt_tol_deg`` of level (the domino state's tolerance)."""
    return _rest_tilt_deg(T) <= tilt_tol_deg


def _annotate(
    obs: ObjectObservation,
    spec: ObjectSpec,
    table_z: float | None,
    pip_face_known: bool,
    lattice: dict[str, Any] | None = None,
) -> None:
    if obs.T_frame_object is None:
        return
    if spec.object_class == "domino":
        obs.attributes["domino"] = domino_state(
            obs.T_frame_object, spec, table_z=table_z, pip_face_known=pip_face_known
        )
    elif spec.object_class in ("lego_brick", "lego_plate") and lattice is not None:
        T_plate = inv_T(np.asarray(lattice["T_frame_plate"], dtype=float)) @ obs.T_frame_object
        snap = snap_to_lattice(
            T_plate,
            spec,
            tuple(lattice["plate_studs"]),
            tuple(lattice.get("grid_bias_m", (0.0, 0.0))),
        )
        snap.pop("T_plate_brick_snapped")
        obs.attributes["lattice"] = snap


def _unseen(
    oid: str,
    spec: ObjectSpec,
    frame_id: str,
    t: float,
    provider: str,
    privileged: bool,
    reason: str,
    evidence: dict[str, Any] | None = None,
    visibility: str = "not_seen",
) -> ObjectObservation:
    return ObjectObservation(
        object_id=oid,
        object_class=spec.object_class,
        size_m=spec.size_m,
        frame_id=frame_id,
        timestamp_s=t,
        provider=provider,
        privileged=privileged,
        visibility=visibility,
        T_frame_object=None,
        uncertainty=Uncertainty(None, None, "simulator_truth" if privileged else "fit_residual"),
        reason=reason,
        evidence=evidence or {},
    )


def pose_error(T_est: np.ndarray, T_ref: np.ndarray, spec: ObjectSpec) -> dict[str, float]:
    """Translation and symmetry-folded rotation error of an estimate against an
    independent reference (simulator truth or a labelled pose)."""
    from .frames import rotation_angle
    from .objects import symmetry_rotations

    rot = min(rotation_angle(T_est[:3, :3], T_ref[:3, :3] @ S) for S in symmetry_rotations(spec))
    return {
        "translation_m": float(np.linalg.norm(T_est[:3, 3] - T_ref[:3, 3])),
        "rotation_deg": math.degrees(rot),
    }
