"""tools/yam-spheres: the YAM chain has 6 revolute joints matching the URDF (and the MJCF when available)."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOC = json.loads((ROOT / "tools" / "yam-spheres" / "yam_spheres.json").read_text())
CUROBO = ROOT / "tools" / "curobo" / "robots" / "yam"
MJCF = Path.home() / "robosimstudio/assets/embodiments/yam_i2rt/yam_i2rt_bimanual.xml"
LINKS = ["link1", "link2", "link3", "link4", "link5", "link6"]


def quat_wxyz_from_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (r, p, y) for f in (math.cos, math.sin))
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    return q


def same_rot(q1, q2, tol=1e-5):
    return abs(abs(float(np.dot(q1, q2))) - 1.0) < tol


def revolute(chain):
    return [(b, e) for b, e in chain.items() if e.get("joint", {}).get("type") == "revolute"]


def test_chain_has_six_revolute_joints_link1_to_link6():
    rev = revolute(DOC["chain"])
    assert [b for b, _ in rev] == LINKS
    assert [e["joint"]["name"] for _, e in rev] == [f"joint{i}" for i in range(1, 7)]
    assert [e["parent"] for _, e in rev] == ["base", *LINKS[:-1]]
    assert DOC["tcp"]["parent"] == "link6"
    assert DOC["link_names_urdf"]["link6"] == "link_6"
    assert DOC["spheres"]["link6"]


def test_chain_matches_the_urdf():
    joints = ET.parse(CUROBO / "yam_i2rt_arm.urdf").getroot().findall("joint")
    assert len(joints) == 6
    for (b, e), j in zip(revolute(DOC["chain"]), joints, strict=True):
        o = j.find("origin")
        assert np.allclose(e["pos"], [float(v) for v in o.get("xyz").split()], atol=1e-6), b
        assert same_rot(e["quat_wxyz"], quat_wxyz_from_rpy(*map(float, o.get("rpy").split()))), b
        assert np.allclose(
            e["joint"]["axis"], [float(v) for v in j.find("axis").get("xyz").split()]
        ), b
        lim = j.find("limit")
        assert np.allclose(
            e["joint"]["range"], [float(lim.get("lower")), float(lim.get("upper"))], atol=1e-4
        ), b


def test_chain_matches_the_mjcf():
    mujoco = pytest.importorskip("mujoco")
    if not MJCF.exists():
        pytest.skip(f"no robosimstudio MJCF at {MJCF}")
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    for b, e in revolute(DOC["chain"]):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"left_{b}")
        j = m.body_jntadr[bid]
        assert m.body_jntnum[bid] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, b
        assert np.allclose(e["pos"], m.body_pos[bid], atol=1e-6), b
        assert same_rot(e["quat_wxyz"], m.body_quat[bid]), b
        assert np.allclose(e["joint"]["axis"], m.jnt_axis[j]), b
        assert np.allclose(e["joint"]["range"], m.jnt_range[j], atol=1e-4), b


def test_fk_tcp_matches_mujoco():
    v = DOC["validation_fk_tcp"]
    assert v["n"] >= 100 and v["max_pos_err_m"] < 1e-5 and v["max_rot_err_rad"] < 1e-4


def test_curobo_yml_spheres_and_tcp():
    k = yaml.safe_load((CUROBO / "yam.yml").read_text())["robot_cfg"]["kinematics"]
    assert (CUROBO / k["urdf_path"]).exists()
    assert set(k["collision_spheres"]) == {"base_link", *(f"link_{i}" for i in range(1, 7))}
    assert len(k["collision_spheres"]["link_6"]) == len(DOC["spheres_link6_fingers_open"])
    assert k["extra_links"]["tcp"]["fixed_transform"] == DOC["tcp"]["pos"] + DOC["tcp"]["quat_wxyz"]
