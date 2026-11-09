"""Parallel-jaw grasp candidates on a known box for the YAM linear_4310 gripper.

Each candidate states its TCP target, approach direction, jaw widths,
``T_TCP_object``, contact patch, the checks it passed or failed (jaw span, contact
area, table clearance of fingers / housing / wrist at the open width, the
station's lowest TCP height, neighbour overlap) and the score terms. The current
``grasp`` primitive executes supplied poses; this module supplies them, it does
not execute anything.

Hand geometry (TCP frame: +z approach out of the fingers, +y the jaw closing
axis): fingertips 10.2 mm past the TCP (the station measurement every hub YAM skill
uses, ``TIP_REACH``); the finger, housing and wrist boxes are the extents of the
i2rt linear_4310 / link5 meshes in the TCP frame at six openings (the station
twin's ``gripper_envelope``), rounded outward. The jaw span limits are the
station's (94 mm span, ``max_grasp_width_m`` 86 mm, as in
proposing-parallel-grasps-in-tight-clusters). The finger thickness along the jaw
motion (10 mm) is that skill's estimate, unmeasured.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frames import T_to_pose, inv_T, make_T
from .objects import ObjectSpec, to_contract_frame


@dataclass(frozen=True)
class GripperSpec:
    name: str = "yam_linear_4310"
    max_opening_m: float = 0.094
    max_grasp_width_m: float = 0.086
    min_grasp_width_m: float = 0.002
    tip_reach_m: float = 0.0102
    finger_thickness_m: float = 0.010
    finger_half_width_tip_m: float = 0.008
    """Half the fingertip's extent across the jaw motion over its last 30 mm."""
    finger_half_width_root_m: float = 0.020
    finger_tip_section_m: float = 0.030
    finger_length_m: float = 0.055
    """Fingertip to housing along the approach."""
    housing_half_m: tuple[float, float] = (0.034, 0.071)
    housing_length_m: float = 0.055
    wrist_half_m: tuple[float, float] = (0.041, 0.029)
    wrist_length_m: float = 0.105
    open_margin_m: float = 0.006
    """Opening beyond the part on each side when the hand comes down
    (grasping-by-wrist-camera-servoing's PREGRASP_MARGIN)."""
    squeeze_m: float = 0.002
    """Commanded close width below the part's width (pads' compliance)."""

    def hand_boxes(self, open_width_m: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(name, centre, half extents) boxes in the TCP frame."""
        tip = self.tip_reach_m
        th = self.finger_thickness_m
        w2 = open_width_m / 2.0
        boxes = []
        for side, s in (("finger_pos", 1.0), ("finger_neg", -1.0)):
            yc = s * (w2 + th / 2.0)
            z0, z1 = tip - self.finger_tip_section_m, tip
            boxes.append(
                (
                    side + "_tip",
                    np.array([0.0, yc, (z0 + z1) / 2]),
                    np.array([self.finger_half_width_tip_m, th / 2, (z1 - z0) / 2]),
                )
            )
            z0b = tip - self.finger_length_m
            boxes.append(
                (
                    side + "_root",
                    np.array([0.0, yc, (z0b + z0) / 2]),
                    np.array([self.finger_half_width_root_m, th / 2, (z0 - z0b) / 2]),
                )
            )
        hz1 = tip - self.finger_length_m
        hz0 = hz1 - self.housing_length_m
        boxes.append(
            (
                "housing",
                np.array([0.0, 0.0, (hz0 + hz1) / 2]),
                np.array([*self.housing_half_m, (hz1 - hz0) / 2]),
            )
        )
        wz0 = hz0 - self.wrist_length_m
        boxes.append(
            (
                "wrist",
                np.array([0.0, 0.0, (wz0 + hz0) / 2]),
                np.array([*self.wrist_half_m, (hz0 - wz0) / 2]),
            )
        )
        return boxes

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


LINEAR_4310 = GripperSpec()


@dataclass(frozen=True)
class GraspLimits:
    table_clearance_m: float = 0.002
    """Fingers' lowest corner over the table at the grasp [m]."""
    min_tcp_height_m: float = 0.014
    """Lowest TCP over the table a real YAM station accepts (servo-grasp skill's MIN_TABLE_CLEARANCE)."""
    min_contact_area_m2: float = 2.5e-5
    """Pad-on-part overlap per finger: 25 mm^2, e.g. 3.5 x 7 mm."""
    max_approach_up: float = math.sin(math.radians(10.0))
    """Approach direction may point at most 10 deg upward (no grasps from under the table)."""
    neighbour_margin_m: float = 0.002
    pregrasp_standoff_m: float = 0.04
    lift_m: float = 0.05


DEFAULT_LIMITS = GraspLimits()


@dataclass
class GraspCandidate:
    object_id: str
    T_object_tcp: np.ndarray
    T_frame_tcp: np.ndarray
    frame_id: str
    closing_axis: str
    approach_face: str
    tilt_deg: float
    jaw_flip: bool
    offset_m: float
    grasp_width_m: float
    open_width_m: float
    close_width_m: float
    contact_area_m2: float
    insertion_m: float
    feasible: bool
    contact_centre_obj: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Middle of the pad-on-part patch, object frame (the pivot when the part turns in the pinch)."""
    checks: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    object_class: str = ""
    score_terms: dict[str, float] = field(default_factory=dict)

    @property
    def T_tcp_object(self) -> np.ndarray:
        return inv_T(self.T_object_tcp)

    @property
    def approach_dir(self) -> np.ndarray:
        return self.T_frame_tcp[:3, 2].copy()

    def to_v1(self, limits: GraspLimits | None = None) -> dict[str, Any]:
        """The contract's ``GraspCandidate`` (directly usable as a ``grasp`` request
        with ``target_id``), plus this module's details."""
        req = to_grasp_request(self, limits or DEFAULT_LIMITS)
        return {
            "object_id": self.object_id,
            "pregrasp_pose": req["pregrasp_pose"],
            "grasp_pose": req["grasp_pose"],
            "lift_pose": req["lift_pose"],
            "open_width_m": req["open_width_m"],
            "close_width_m": req["close_width_m"],
            "T_TCP_object": T_to_pose(
                to_contract_frame(self.T_tcp_object, self.object_class), "tcp"
            ),
            "score": float(self.score),
            "checks": {"feasible": bool(self.feasible), **self.checks},
            "details": self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "tcp_pose": T_to_pose(self.T_frame_tcp, self.frame_id),
            "approach_dir": [float(v) for v in self.approach_dir],
            "T_TCP_object_module_frame": T_to_pose(self.T_tcp_object, "tcp"),
            "closing_axis": self.closing_axis,
            "approach_face": self.approach_face,
            "tilt_deg": self.tilt_deg,
            "jaw_flip": self.jaw_flip,
            "offset_m": self.offset_m,
            "grasp_width_m": self.grasp_width_m,
            "open_width_m": self.open_width_m,
            "close_width_m": self.close_width_m,
            "contact_area_m2": self.contact_area_m2,
            "insertion_m": self.insertion_m,
            "contact_centre_obj_m": [float(v) for v in self.contact_centre_obj],
            "feasible": self.feasible,
            "checks": self.checks,
            "score": self.score,
            "score_terms": self.score_terms,
        }


_AX = ("x", "y", "z")


def candidate_frames(
    spec: ObjectSpec,
    gripper: GripperSpec = LINEAR_4310,
    tilts_deg: tuple[float, ...] = (0.0,),
    offsets_frac: tuple[float, ...] = (0.0, 0.6, -0.6),
    closing_axes: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """``T_object_tcp`` seeds before any world check: every closing axis whose
    width fits the jaws, every approach face perpendicular to it, both jaw
    orientations, tilts about the closing axis, and offsets of the contact along
    the remaining axis (as a fraction of the room left beside the fingertip)."""
    d = np.asarray(spec.size_m, dtype=float)
    out = []
    for c in closing_axes if closing_axes is not None else (0, 1, 2):
        if not gripper.min_grasp_width_m <= d[c] <= gripper.max_grasp_width_m:
            continue
        for a in (k for k in range(3) if k != c):
            o = 3 - a - c
            for s, tilt, flip, frac in itertools.product(
                (1.0, -1.0), tilts_deg, (False, True), offsets_frac
            ):
                e_a, e_c, e_o = np.eye(3)[a], np.eye(3)[c], np.eye(3)[o]
                z0 = -s * e_a
                th = math.radians(tilt)
                z = math.cos(th) * z0 + math.sin(th) * e_o
                y = e_c * (-1.0 if flip else 1.0)
                x = np.cross(y, z)
                R = np.column_stack([x, y, z])
                room = max(0.0, d[o] / 2.0 - gripper.finger_half_width_tip_m)
                u = frac * room
                out.append(
                    {
                        "R": R,
                        "a": a,
                        "c": c,
                        "o": o,
                        "s": s,
                        "tilt": float(tilt),
                        "flip": flip,
                        "u": float(u),
                        "closing_axis": _AX[c],
                        "approach_face": ("+" if s > 0 else "-") + _AX[a],
                    }
                )
    return out


def _section_overlap(
    spec: ObjectSpec, seed: dict[str, Any], f_obj: np.ndarray, gripper: GripperSpec
) -> float:
    """Area where one fingertip pad (in the TCP x-z plane) overlaps the part's
    cross-section perpendicular to the closing axis [m^2]."""
    a, o = seed["a"], seed["o"]
    R = seed["R"]
    d = np.asarray(spec.size_m)
    # Pad rectangle corners in the object frame, projected on (o, a).
    x, z = R[:, 0], R[:, 2]
    hw = gripper.finger_half_width_tip_m
    corners = [
        f_obj + sx * hw * x - sz * gripper.finger_tip_section_m * z
        for sx, sz in ((-1, 0), (1, 0), (1, 1), (-1, 1))
    ]
    pad = [(p[o], p[a]) for p in corners]
    box = [
        (-d[o] / 2, -d[a] / 2),
        (d[o] / 2, -d[a] / 2),
        (d[o] / 2, d[a] / 2),
        (-d[o] / 2, d[a] / 2),
    ]
    return _polygon_area(_clip(pad, box))


def _clip(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: subject polygon clipped by a convex counter-clockwise polygon."""

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-15

    def cross(p1, p2, a, b):
        x1, y1, x2, y2 = *p1, *p2
        x3, y3, x4, y4 = *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-18:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    out.append(cross(prev, cur, a, b))
                out.append(cur)
            elif inside(prev, a, b):
                out.append(cross(prev, cur, a, b))
            prev = cur
    return out


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1], strict=True):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def obb_overlap(
    Ta: np.ndarray, ha: np.ndarray, Tb: np.ndarray, hb: np.ndarray, margin: float = 0.0
) -> bool:
    """Separating-axis test for two oriented boxes (centre/rotation in T, half extents h)."""
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    t = Tb[:3, 3] - Ta[:3, 3]
    ha = np.asarray(ha) + margin / 2
    hb = np.asarray(hb) + margin / 2
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for L in axes:
        n = np.linalg.norm(L)
        if n < 1e-9:
            continue
        L = L / n
        ra = float(np.sum(ha * np.abs(Ra.T @ L)))
        rb = float(np.sum(hb * np.abs(Rb.T @ L)))
        if abs(float(t @ L)) > ra + rb:
            return False
    return True


def hand_lowest(
    T_frame_tcp: np.ndarray, gripper: GripperSpec, open_width: float, up: np.ndarray
) -> dict[str, float]:
    """Lowest point (along ``up``) of each hand part at this TCP pose and opening."""
    out = {}
    for name, c, h in gripper.hand_boxes(open_width):
        corners = c + np.array(list(itertools.product(*[(-v, v) for v in h])))
        w = corners @ T_frame_tcp[:3, :3].T + T_frame_tcp[:3, 3]
        out[name] = float((w @ up).min())
    return out


def propose_grasps(
    object_id: str,
    T_frame_object: np.ndarray,
    spec: ObjectSpec,
    frame_id: str,
    table_z: float,
    gripper: GripperSpec = LINEAR_4310,
    limits: GraspLimits = DEFAULT_LIMITS,
    neighbours: list[tuple[str, np.ndarray, ObjectSpec]] | None = None,
    tilts_deg: tuple[float, ...] = (0.0,),
    offsets_frac: tuple[float, ...] = (0.0, 0.6, -0.6),
    closing_axes: tuple[int, ...] | None = None,
    up: np.ndarray | None = None,
) -> list[GraspCandidate]:
    """All candidates, feasible first, best score first. For each seed the TCP is
    put as deep along the approach as the table, the station's lowest TCP height
    and the part allow (fingertips at most ``finger_tip_section_m`` into it), so a
    thin lying tile is taken as low as the station permits."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
    d = np.asarray(spec.size_m, dtype=float)
    out: list[GraspCandidate] = []
    for seed in candidate_frames(spec, gripper, tilts_deg, offsets_frac, closing_axes):
        R, a, c, o, s = seed["R"], seed["a"], seed["c"], seed["o"], seed["s"]
        z = R[:, 2]
        width = float(d[c])
        open_w = min(width + 2 * gripper.open_margin_m, gripper.max_opening_m)
        close_w = max(width - gripper.squeeze_m, 0.0)
        z_world = T_frame_object[:3, :3] @ z
        checks: dict[str, Any] = {}
        checks["jaw_width"] = bool(
            width <= gripper.max_grasp_width_m and width + 2 * 0.001 <= gripper.max_opening_m
        )
        checks["approach_direction"] = bool(float(z_world @ up) <= limits.max_approach_up)
        # Fingertip on the entry face, then pushed in along the approach as far as allowed.
        entry = s * d[a] / 2.0
        base = seed["u"] * np.eye(3)[o] + entry * np.eye(3)[a]
        best = None
        for depth in np.linspace(min(gripper.finger_tip_section_m, float(d[a])), 0.0, 25):
            f_obj = base + depth * z  # fingertip centre
            tcp_obj = f_obj - gripper.tip_reach_m * z
            T_obj_tcp = make_T(R, tcp_obj)
            T_w = T_frame_object @ T_obj_tcp
            low = hand_lowest(T_w, gripper, open_w, up)
            tcp_h = float(T_w[:3, 3] @ up) - table_z
            ok_table = min(low.values()) - table_z >= limits.table_clearance_m
            ok_tcp = tcp_h >= limits.min_tcp_height_m
            if ok_table and ok_tcp:
                best = (depth, f_obj, T_obj_tcp, T_w, low, tcp_h)
                break
        if best is None:
            depth, f_obj = 0.0, base
            T_obj_tcp = make_T(R, f_obj - gripper.tip_reach_m * z)
            T_w = T_frame_object @ T_obj_tcp
            low = hand_lowest(T_w, gripper, open_w, up)
            tcp_h = float(T_w[:3, 3] @ up) - table_z
        else:
            depth, f_obj, T_obj_tcp, T_w, low, tcp_h = best
        area = _section_overlap(spec, seed, f_obj, gripper)
        checks["table_clearance"] = {
            "ok": bool(min(low.values()) - table_z >= limits.table_clearance_m),
            "lowest_m": round(min(low.values()) - table_z, 5),
            "lowest_part": min(low, key=low.get),
        }
        checks["tcp_height"] = {
            "ok": bool(tcp_h >= limits.min_tcp_height_m),
            "tcp_over_table_m": round(tcp_h, 5),
        }
        checks["contact_area"] = {
            "ok": bool(area >= limits.min_contact_area_m2),
            "area_m2": float(area),
        }
        # Housing / wrist must stay out of the part itself.
        T_obj_w = T_frame_object
        hand_hit = []
        nbr_hit = []
        for name, cen, half in gripper.hand_boxes(open_w):
            Tb = T_w @ make_T(position=cen)
            if not name.startswith("finger") and obb_overlap(T_obj_w, spec.half, Tb, half):
                hand_hit.append(name)
            for nid, Tn, nspec in neighbours or []:
                if obb_overlap(Tn, nspec.half, Tb, half, limits.neighbour_margin_m):
                    nbr_hit.append(f"{name}:{nid}")
        checks["hand_clear_of_part"] = {"ok": not hand_hit, "hits": hand_hit}
        checks["neighbours"] = {"ok": not nbr_hit, "hits": nbr_hit}
        pre = T_w @ make_T(position=(0.0, 0.0, -limits.pregrasp_standoff_m))
        pre_low = hand_lowest(pre, gripper, open_w, up)
        checks["pregrasp_clearance"] = {
            "ok": bool(min(pre_low.values()) - table_z >= limits.table_clearance_m)
        }
        feasible = all(v if isinstance(v, bool) else v["ok"] for v in checks.values())
        insertion = max(0.0, depth)
        contact_centre = f_obj - 0.5 * min(insertion, gripper.finger_tip_section_m) * z
        terms = {
            "contact": min(1.0, area / (4 * limits.min_contact_area_m2)),
            "centred": 1.0 - abs(seed["u"]) / max(d[o] / 2.0, 1e-9),
            "vertical": float(-(z_world @ up)),
            "width_margin": 1.0 - width / gripper.max_grasp_width_m,
        }
        score = (
            0.4 * terms["contact"]
            + 0.2 * terms["centred"]
            + 0.3 * max(0.0, terms["vertical"])
            + 0.1 * terms["width_margin"]
        )
        out.append(
            GraspCandidate(
                object_id=object_id,
                T_object_tcp=T_obj_tcp,
                T_frame_tcp=T_w,
                frame_id=frame_id,
                closing_axis=seed["closing_axis"],
                approach_face=seed["approach_face"],
                tilt_deg=seed["tilt"],
                jaw_flip=seed["flip"],
                offset_m=seed["u"],
                grasp_width_m=width,
                open_width_m=open_w,
                close_width_m=close_w,
                contact_area_m2=float(area),
                insertion_m=insertion,
                feasible=feasible,
                contact_centre_obj=contact_centre,
                object_class=spec.object_class,
                checks=checks,
                score=round(score, 4),
                score_terms={k: round(v, 4) for k, v in terms.items()},
            )
        )
    out.sort(key=lambda g: (not g.feasible, -g.score))
    return out


def to_grasp_request(
    cand: GraspCandidate,
    limits: GraspLimits = DEFAULT_LIMITS,
    up: np.ndarray | None = None,
) -> dict[str, Any]:
    """The kit runtime's ``grasp`` request (pregrasp / grasp / lift, widths, target id)."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
    pre = cand.T_frame_tcp @ make_T(position=(0.0, 0.0, -limits.pregrasp_standoff_m))
    lift = cand.T_frame_tcp.copy()
    lift[:3, 3] += limits.lift_m * up
    return {
        "pregrasp_pose": T_to_pose(pre, cand.frame_id),
        "grasp_pose": T_to_pose(cand.T_frame_tcp, cand.frame_id),
        "lift_pose": T_to_pose(lift, cand.frame_id),
        "target_id": cand.object_id,
        "open_width_m": round(cand.open_width_m, 5),
        "close_width_m": round(cand.close_width_m, 5),
    }
