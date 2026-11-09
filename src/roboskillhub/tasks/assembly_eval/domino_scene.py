# SPDX-License-Identifier: Apache-2.0
"""Domino first scene for the assembly evaluation: spec, derived poses and CPU checks.

Flat table, 3-5 identical 48 x 24 x 7.5 mm tiles standing in an upright feeder rack,
a straight target line, and explicit collision geometry for the table, the targets,
the rack and the gripper. The same spec builds the Isaac (PhysX) scene and the MuJoCo
comparison scene; nothing here imports a simulator.

Every number below is an evaluation-scene choice unless its ``source`` says otherwise;
none is a calibration of a real station.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .domino_geometry import polygon_gap, quat_from_yaw, rect_corners

SCENE_SCHEMA = "roboskillhub.assembly_eval.domino_scene/v1"


@dataclass(frozen=True)
class TileSpec:
    thickness: float = 0.0075  # local x, the facing
    width: float = 0.024  # local y
    height: float = 0.048  # local z, the up axis
    mass: float = 0.010
    friction: float = 0.5
    source: str = (
        "48 x 24 x 7.5 mm standard domino, 10 g, friction 0.5: the values the "
        "placing-an-upright-domino-beside-neighbours MuJoCo check used (Franka-Domino-Project "
        "tile size; predicators' table fit)"
    )

    @property
    def half(self) -> tuple[float, float, float]:
        return (self.thickness / 2, self.width / 2, self.height / 2)


@dataclass(frozen=True)
class TableSpec:
    top_z: float = 0.10
    center_xy: tuple[float, float] = (0.0, 0.0)
    size_xy: tuple[float, float] = (0.60, 0.60)
    thickness: float = 0.05
    tilt_deg: float = 0.0  # flat: the evaluator assumes normal +z
    friction: float = 0.5
    calibration: str = "scene ground truth (simulation); no real-station calibration"


@dataclass(frozen=True)
class FeederSpec:
    """Upright feeder: tiles stand on the table between thin static separators."""

    origin_xy: tuple[float, float] = (-0.08, -0.15)
    yaw: float = 0.0  # tiles face +x of the rack
    pitch: float = 0.040
    separator_size: tuple[float, float, float] = (0.003, 0.012, 0.015)


@dataclass(frozen=True)
class TargetLine:
    start_xy: tuple[float, float] = (-0.07, 0.05)
    yaw: float = 0.0  # row direction == the tiles' facing
    pitch: float = 0.036  # centre-to-centre along the row (0.75 tile heights)


@dataclass(frozen=True)
class GripperSpec:
    """Parallel-jaw hand without an arm: x/y/z prismatic + yaw revolute mount, two
    prismatic fingers closing along the hand's local y."""

    pad_size: tuple[float, float, float] = (0.012, 0.006, 0.040)  # x, y (closing), z
    palm_size: tuple[float, float, float] = (0.030, 0.080, 0.015)
    max_opening: float = 0.070
    pad_friction: float = 1.0
    grip_force: float = 3.0  # N per finger drive limit (30x the tile's weight with friction 1)
    finger_speed: float = 0.05  # m/s per finger
    linear_speed: float = 0.12  # m/s
    yaw_speed: float = 1.5  # rad/s
    source: str = (
        "scene choice loosely after the YAM linear_4310 hand (30 mm blades); no YAM asset or "
        "measured dynamics"
    )


@dataclass(frozen=True)
class PlaceParams:
    grasp_height: float = 0.030  # pad centre over the tile's bottom
    grip_clearance: float = 0.004  # per side when open at pick / release
    squeeze: float = 0.004  # close target = tile width - squeeze
    release_gap: float = 0.0015  # tile bottom over the believed table at release
    transit_clearance: float = 0.020
    neighbour_clearance: float = 0.004
    table_margin: float = 0.02
    source: str = (
        "placing-an-upright-domino-beside-neighbours defaults (grasp 30 mm, 4 mm, 1.5 mm, 20 mm)"
    )


@dataclass(frozen=True)
class PhysicsSpec:
    gravity: float = -9.81
    isaac_dt: float = 1.0 / 240.0
    isaac_solver_position_iterations: int = 32
    isaac_solver_velocity_iterations: int = 4
    isaac_contact_offset: float = 0.002
    isaac_rest_offset: float = 0.0
    mujoco_timestep: float = 0.0005
    control_hz: float = 60.0
    observe_hz: float = 30.0


@dataclass
class DominoSceneSpec:
    n_tiles: int = 4
    tile: TileSpec = field(default_factory=TileSpec)
    table: TableSpec = field(default_factory=TableSpec)
    feeder: FeederSpec = field(default_factory=FeederSpec)
    line: TargetLine = field(default_factory=TargetLine)
    gripper: GripperSpec = field(default_factory=GripperSpec)
    place: PlaceParams = field(default_factory=PlaceParams)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    version: str = "domino_scene_v1"

    # ---- derived poses -------------------------------------------------
    def tile_center_z(self) -> float:
        return self.table.top_z + self.tile.height / 2

    def slot_pose(self, k: int) -> dict:
        c, s = math.cos(self.line.yaw), math.sin(self.line.yaw)
        d = k * self.line.pitch
        x = self.line.start_xy[0] + c * d
        y = self.line.start_xy[1] + s * d
        return {
            "frame": "table",
            "position": [x, y, self.tile_center_z()],
            "quat_xyzw": list(quat_from_yaw(self.line.yaw)),
            "yaw": self.line.yaw,
        }

    def feeder_pose(self, k: int) -> dict:
        c, s = math.cos(self.feeder.yaw), math.sin(self.feeder.yaw)
        d = k * self.feeder.pitch
        x = self.feeder.origin_xy[0] + c * d
        y = self.feeder.origin_xy[1] + s * d
        return {
            "frame": "table",
            "position": [x, y, self.tile_center_z()],
            "quat_xyzw": list(quat_from_yaw(self.feeder.yaw)),
            "yaw": self.feeder.yaw,
        }

    def separator_poses(self) -> list[dict]:
        c, s = math.cos(self.feeder.yaw), math.sin(self.feeder.yaw)
        out = []
        for k in range(self.n_tiles + 1):
            d = (k - 0.5) * self.feeder.pitch
            out.append(
                {
                    "frame": "table",
                    "position": [
                        self.feeder.origin_xy[0] + c * d,
                        self.feeder.origin_xy[1] + s * d,
                        self.table.top_z + self.feeder.separator_size[2] / 2,
                    ],
                    "yaw": self.feeder.yaw,
                }
            )
        return out

    def tile_ids(self) -> list[str]:
        return [f"tile_{k}" for k in range(self.n_tiles)]

    def collision_geometry(self) -> list[dict]:
        """Every collider of the scene, explicitly (boxes, full extents, table frame)."""
        t = self.table
        geo = [
            {
                "body": "table",
                "kind": "static",
                "shape": "box",
                "size": [t.size_xy[0], t.size_xy[1], t.thickness],
                "position": [t.center_xy[0], t.center_xy[1], t.top_z - t.thickness / 2],
                "yaw": 0.0,
                "friction": t.friction,
            }
        ]
        size = [self.tile.thickness, self.tile.width, self.tile.height]
        for k in range(self.n_tiles):
            p = self.feeder_pose(k)
            geo.append(
                {
                    "body": f"tile_{k}",
                    "kind": "dynamic",
                    "shape": "box",
                    "size": size,
                    "position": p["position"],
                    "yaw": p["yaw"],
                    "mass": self.tile.mass,
                    "friction": self.tile.friction,
                }
            )
        for k in range(self.n_tiles):
            p = self.slot_pose(k)
            geo.append(
                {
                    "body": f"target_{k}",
                    "kind": "marker_no_collision",
                    "shape": "box",
                    "size": size,
                    "position": p["position"],
                    "yaw": p["yaw"],
                    "note": "target footprint; checked geometrically against the hand, not a collider",
                }
            )
        for k, p in enumerate(self.separator_poses()):
            geo.append(
                {
                    "body": f"feeder_sep_{k}",
                    "kind": "static",
                    "shape": "box",
                    "size": list(self.feeder.separator_size),
                    "position": p["position"],
                    "yaw": p["yaw"],
                    "friction": t.friction,
                }
            )
        g = self.gripper
        geo += [
            {
                "body": "gripper_palm",
                "kind": "articulated",
                "shape": "box",
                "size": list(g.palm_size),
            },
            {
                "body": "gripper_pad_l",
                "kind": "articulated",
                "shape": "box",
                "size": list(g.pad_size),
                "friction": g.pad_friction,
            },
            {
                "body": "gripper_pad_r",
                "kind": "articulated",
                "shape": "box",
                "size": list(g.pad_size),
                "friction": g.pad_friction,
            },
        ]
        return geo

    # ---- (de)serialisation and hashing ---------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = SCENE_SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DominoSceneSpec:
        d = copy.deepcopy(d)
        schema = d.pop("schema", SCENE_SCHEMA)
        if schema != SCENE_SCHEMA:
            raise ValueError(f"unsupported scene schema {schema!r}")

        def mk(kind, key):
            v = d.pop(key, None) or {}
            return kind(**{k: tuple(x) if isinstance(x, list) else x for k, x in v.items()})

        return cls(
            n_tiles=d.pop("n_tiles", 4),
            tile=mk(TileSpec, "tile"),
            table=mk(TableSpec, "table"),
            feeder=mk(FeederSpec, "feeder"),
            line=mk(TargetLine, "line"),
            gripper=mk(GripperSpec, "gripper"),
            place=mk(PlaceParams, "place"),
            physics=mk(PhysicsSpec, "physics"),
            version=d.pop("version", "domino_scene_v1"),
        )

    def config_hash(self) -> str:
        return config_hash(self.to_dict())

    def transit_tcp_z(self) -> float:
        """Hand height at which a held tile's bottom clears every standing tile and the
        rack by ``transit_clearance``."""
        tallest = max(self.tile.height, self.feeder.separator_size[2])
        return self.table.top_z + tallest + self.place.transit_clearance + self.place.grasp_height


def config_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=list).encode()
    return hashlib.sha256(blob).hexdigest()


def _issue(code: str, detail: str, **extra) -> dict:
    return {"code": code, "detail": detail, **extra}


def pad_rects(spec: DominoSceneSpec, center_xy, yaw: float, opening: float) -> list[list[tuple]]:
    """The two pads' footprints for an inner opening ``opening`` about ``center_xy``."""
    g = spec.gripper
    off = opening / 2 + g.pad_size[1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sign in (1, -1):
        px = center_xy[0] - s * sign * off
        py = center_xy[1] + c * sign * off
        out.append(rect_corners(px, py, yaw, g.pad_size[0] / 2, g.pad_size[1] / 2))
    return out


def check_scene(spec: DominoSceneSpec, order: list[int] | None = None) -> list[dict]:
    """CPU checks of the spec before any simulator is started. An empty list means the
    scene is geometrically consistent for the given placement order (default 0..n-1)."""
    issues: list[dict] = []
    t, tile, g, p = spec.table, spec.tile, spec.gripper, spec.place
    if not 3 <= spec.n_tiles <= 5:
        issues.append(_issue("tile_count", f"first scene uses 3-5 tiles, got {spec.n_tiles}"))
    if abs(t.tilt_deg) > 0.1:
        issues.append(
            _issue("table_not_flat", f"table tilt {t.tilt_deg} deg; scene assumes a flat table")
        )
    for name, v in (
        ("thickness", tile.thickness),
        ("width", tile.width),
        ("height", tile.height),
        ("mass", tile.mass),
    ):
        if not v > 0:
            issues.append(_issue("tile_size", f"tile {name} must be positive"))
    order = list(range(spec.n_tiles)) if order is None else list(order)
    if sorted(order) != list(range(spec.n_tiles)):
        issues.append(_issue("order", f"order {order} is not a permutation of the tiles"))
        return issues
    hx, hy = tile.thickness / 2, tile.width / 2
    slots = [spec.slot_pose(k) for k in range(spec.n_tiles)]
    feeds = [spec.feeder_pose(k) for k in range(spec.n_tiles)]
    seps = spec.separator_poses()

    def foot(pose):
        return rect_corners(pose["position"][0], pose["position"][1], pose["yaw"], hx, hy)

    def sep_foot(pose):
        sx, sy, _ = spec.feeder.separator_size
        return rect_corners(pose["position"][0], pose["position"][1], pose["yaw"], sx / 2, sy / 2)

    tx0 = t.center_xy[0] - t.size_xy[0] / 2 + p.table_margin
    tx1 = t.center_xy[0] + t.size_xy[0] / 2 - p.table_margin
    ty0 = t.center_xy[1] - t.size_xy[1] / 2 + p.table_margin
    ty1 = t.center_xy[1] + t.size_xy[1] / 2 - p.table_margin
    for label, poses in (("slot", slots), ("feeder", feeds)):
        for k, pose in enumerate(poses):
            for x, y in foot(pose):
                if not (tx0 <= x <= tx1 and ty0 <= y <= ty1):
                    issues.append(
                        _issue("off_table", f"{label} {k} footprint leaves the table margin")
                    )
                    break
    for i in range(spec.n_tiles):
        for j in range(i + 1, spec.n_tiles):
            gap = polygon_gap(foot(slots[i]), foot(slots[j]))
            if gap <= 0:
                issues.append(
                    _issue("slot_overlap", f"slots {i} and {j} overlap ({gap * 1e3:.1f} mm)")
                )
    for k, f in enumerate(feeds):
        for m, sp in enumerate(seps):
            if polygon_gap(foot(f), sep_foot(sp)) <= 0:
                issues.append(_issue("feeder_overlap", f"feeder tile {k} overlaps separator {m}"))
        for s_k, s in enumerate(slots):
            if polygon_gap(foot(f), foot(s)) < g.max_opening:
                issues.append(
                    _issue(
                        "feeder_near_line", f"feeder tile {k} within a hand opening of slot {s_k}"
                    )
                )
    if p.grasp_height - g.pad_size[2] / 2 < 0.003:
        issues.append(
            _issue("pad_hits_table", "pad bottom under 3 mm above the table at the grasp height")
        )
    if p.grasp_height + g.pad_size[2] / 2 <= tile.height:
        issues.append(
            _issue("palm_hits_tile", "palm (above the pads) would rest on the tile's top")
        )
    opening = tile.width + 2 * p.grip_clearance
    if opening > g.max_opening:
        issues.append(
            _issue(
                "opening",
                f"needs {opening * 1e3:.1f} mm opening; hand opens {g.max_opening * 1e3:.0f}",
            )
        )
    if tile.width - p.squeeze <= 0:
        issues.append(_issue("squeeze", "close target must stay positive"))
    # Open hand at each slot against the tiles already standing (placement order) and at
    # each feeder tile against the tiles and separators still in the rack.
    for n, k in enumerate(order):
        rects = pad_rects(spec, slots[k]["position"], slots[k]["yaw"], opening)
        for j in order[:n]:
            gap = min(polygon_gap(r, foot(slots[j])) for r in rects)
            if gap < p.neighbour_clearance:
                issues.append(
                    _issue(
                        "hand_near_neighbour",
                        f"open hand at slot {k} is {gap * 1e3:.1f} mm from standing tile {j}",
                        slot=k,
                        neighbour=j,
                        gap_m=gap,
                    )
                )
    for k in range(spec.n_tiles):
        rects = pad_rects(spec, feeds[k]["position"], feeds[k]["yaw"], opening)
        others = [foot(feeds[j]) for j in range(spec.n_tiles) if j != k] + [
            sep_foot(s) for s in seps
        ]
        gap = min(polygon_gap(r, o) for r in rects for o in others)
        if gap < p.neighbour_clearance:
            issues.append(
                _issue(
                    "hand_near_rack",
                    f"open hand at feeder tile {k} is {gap * 1e3:.1f} mm from the rack",
                    feeder=k,
                    gap_m=gap,
                )
            )
    return issues


def scene_summary(spec: DominoSceneSpec) -> dict:
    return {
        "version": spec.version,
        "config_sha256": spec.config_hash(),
        "n_tiles": spec.n_tiles,
        "pitch_center_m": spec.line.pitch,
        "face_gap_m": spec.line.pitch - spec.tile.thickness,
        "pitch_over_height": spec.line.pitch / spec.tile.height,
        "transit_tcp_z": spec.transit_tcp_z(),
        "slots": [spec.slot_pose(k) for k in range(spec.n_tiles)],
        "feeder": [spec.feeder_pose(k) for k in range(spec.n_tiles)],
        "collision_geometry": spec.collision_geometry(),
    }
