# SPDX-License-Identifier: Apache-2.0
"""MuJoCo state -> domino evaluator truth, for any MuJoCo model (this package's scene or
the YAM station twin in RoboSimStudio).

A twin harness calls :func:`mujoco_truth` after each control step with its own body and
geom names mapped onto the evaluator's ids (``tile_k``, ``table``, ``gripper_pad_l``,
``gripper_pad_r``, ``gripper_palm``) and the table frame in world coordinates; the result
feeds :class:`domino_runner.Observer`-style samples and ``evaluate_placement`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .domino_geometry import quat_mul, quat_rotate, wxyz_to_xyzw

TOUCH_DIST = 0.0005


def _inverse(pos: Sequence[float], quat_xyzw: Sequence[float]):
    x, y, z, w = quat_xyzw
    qi = (-x, -y, -z, w)
    p = quat_rotate(qi, pos)
    return (-p[0], -p[1], -p[2]), qi


def mujoco_truth(
    model,
    data,
    tile_bodies: Mapping[str, str],
    geom_alias: Mapping[str, str] | None = None,
    body_alias: Mapping[str, str] | None = None,
    T_world_table: tuple[Sequence[float], Sequence[float]] = ((0, 0, 0), (0, 0, 0, 1)),
) -> dict:
    """Tile poses in the table frame and the contacts of each tile.

    ``tile_bodies`` maps evaluator ids to MuJoCo body names. Contact partners are named by
    ``geom_alias`` (geom name -> id), else ``body_alias`` (body name -> id), else the
    geom's own name. ``T_world_table`` is the table frame's pose in world (position,
    quaternion xyzw); poses are returned as ``T_table_tile``.
    """
    geom_alias = dict(geom_alias or {})
    body_alias = dict(body_alias or {})
    p_tw, q_tw = _inverse(*T_world_table)
    objects = {}
    body_to_tile = {}
    for tid, bname in tile_bodies.items():
        b = data.body(bname)
        pw = [float(v) for v in b.xpos]
        qw = wxyz_to_xyzw([float(v) for v in b.xquat])
        r = quat_rotate(q_tw, pw)
        objects[tid] = {
            "position": [r[0] + p_tw[0], r[1] + p_tw[1], r[2] + p_tw[2]],
            "quat_xyzw": list(quat_mul(q_tw, qw)),
        }
        body_to_tile[model.body(bname).id] = tid

    def name_of(geom_id: int) -> str:
        g = model.geom(geom_id)
        if g.name in geom_alias:
            return geom_alias[g.name]
        bid = int(model.geom_bodyid[geom_id])
        if bid in body_to_tile:
            return body_to_tile[bid]
        bname = model.body(bid).name
        if bname in body_alias:
            return body_alias[bname]
        return g.name or f"geom_{geom_id}"

    contacts: dict[str, set] = {tid: set() for tid in tile_bodies}
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > TOUCH_DIST:
            continue
        a, b = name_of(int(c.geom1)), name_of(int(c.geom2))
        for u, v in ((a, b), (b, a)):
            if u in contacts and u != v:
                contacts[u].add("feeder" if v.startswith("feeder_sep") else v)
    return {
        "t": float(data.time),
        "objects": objects,
        "contacts": {k: sorted(v) for k, v in contacts.items()},
    }
