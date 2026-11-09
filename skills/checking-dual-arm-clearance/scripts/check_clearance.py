# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/check_dual_arm_collision/robotwin/policy.py and
# roborsi/embodied/sim/robotwin/dual_arm_collision.py (check_pair_collision,
# check_trajectory_collision), Apache-2.0.
# Changes: the arms' joints, tool poses and IK come from the connector's robot.*
# tools instead of the simulator's articulations (no qpos is set on the robot);
# link capsules derived from the i2rt YAM model replace the ALOHA sphere YAML;
# each arm's base is located from its own joints and tool pose; the whole planned
# path of either arm (or both, one after the other or together) is sampled, not
# only its end configuration; attached objects are TCP-frame spheres (gap's
# AttachedObject); the answer is routed clear / too_close / unknown.
# The chain and capsule constants are derived from i2rt's YAM model and meshes
# (https://github.com/i2rt-robotics/i2rt), MIT, Copyright (c) I2RT Robotics; see
# LICENSE.i2rt beside this bundle.
"""Predict the two arms' minimum clearance along a planned move from their joints and a capsule model."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
try:
    from gap import NodeContext
except ImportError:  # annotation only; task modules import these functions without gap
    NodeContext = Any

logger = logging.getLogger(__name__)

TOO_CLOSE = 0.005
"""Predicted clearance under which the move is routed too_close [m]. The capsules
enclose the link geometry, so the prediction is already a lower bound (in the twin
check at 40 configurations it read 3.2-12.0 mm under the measured mesh
clearance); this margin is for joint-reading and path-tracking error. RoboRSI's
threshold was -5 mm (penetration allowed) because its spheres were inflated."""

JOINT_STEP = 0.01
"""Largest joint change between two path samples [rad]: about 6 mm of motion at
the YAM's 0.6 m reach, so a pass between samples cannot hide much more than that.
RoboRSI sampled every 4th planner waypoint."""

CARTESIAN_STEP = 0.02
"""Spacing of the IK solves along a straight pose leg [m]; joint interpolation in
between."""

ROTATION_STEP = math.radians(10.0)
"""Spacing of the IK solves along a pose leg that mostly turns the hand [rad]."""

IK_JUMP = 0.8
"""A joint changing more than this between two neighbouring IK solves [rad]
means the solver changed branch: the arm's real path between them is unknown."""

MODEL_TOLERANCE = 0.005
"""How far the connector's own forward kinematics may disagree with the YAM
model, for the same joint change, before the model is not trusted [m]."""

SAME_ARM = 0.001
"""Two arm readings this close (tool position) are one arm read twice [m]."""

FINGER_TRAVEL = 0.0475
"""Each linear_4310 finger's slide at full open [m]; 0 is closed (i2rt MJCF
joint7/joint8 range)."""

DEFAULT_TCP = ((0.0, 0.0, -0.1347), (0.0, 1.0, 0.0, 0.0))
"""Tool centre in the ``gripper`` body frame: 134.7 mm along -z, turned 180 deg
about x (wxyz), the i2rt MJCF's grasp_site. Used when robot.describe_arm states
no tcp_offset."""

# The i2rt YAM chain (one arm, base frame), transcribed from the i2rt MJCF
# (yam_i2rt_arms.xml, left arm; both arms are identical). Each entry: body,
# parent, origin in the parent frame, rotation in the parent frame (wxyz), and its
# joint: index into the 6 arm joints (or "finger" for the two slides) and axis.
_CHAIN: tuple[tuple[str, str | None, tuple, tuple, Any, tuple], ...] = (
    ("base", None, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None, ()),
    ("link1", "base", (0.0, 0.0, 0.067), (0.7071053, 0.0, 0.0, -0.7071083), 0, (0.0, 0.0, 1.0)),
    (
        "link2",
        "link1",
        (-0.0329, 0.02, 0.0455),
        (0.499998, 0.5, -0.500002, -0.5),
        1,
        (0.0, 0.0, 1.0),
    ),
    (
        "link3",
        "link2",
        (0.264, 4e-07, -0.06375),
        (9e-07, -0.7071053, -0.7071083, 9e-07),
        2,
        (0.0, 0.0, 1.0),
    ),
    (
        "link4",
        "link3",
        (0.0600003, -0.244999, -0.00205),
        (1.3e-06, 0.0, 0.0, -1.0),
        3,
        (0.0, 0.0, 1.0),
    ),
    (
        "link5",
        "link4",
        (-0.0403003, 0.0703851, -0.0323887),
        (9e-07, -0.7071053, -9e-07, -0.7071083),
        4,
        (0.0, 0.0, 1.0),
    ),
    (
        "link6",
        "link5",
        (2e-07, -0.0419481, 0.0404996),
        (0.499998, -0.5, -0.5, -0.500002),
        5,
        (0.0, 0.0, -1.0),
    ),
    ("gripper", "link6", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None, ()),
    (
        "tip_left",
        "gripper",
        (-0.0238981, 0.0450619, -0.0545599),
        (0.499998, -0.5, -0.5, -0.500002),
        "finger",
        (0.0, 0.0, -1.0),
    ),
    (
        "tip_right",
        "gripper",
        (0.0238981, -0.0450619, -0.0545599),
        (0.7071053, 0.7071083, 0.0, 0.0),
        "finger",
        (0.0, 0.0, -1.0),
    ),
)

# Link capsules in each body's frame: (end a, end b, radius) [m], each enclosing every
# vertex of its share of the link's collision geometry, radius +0.5 mm for the
# rounding of the end points. base, gripper: the i2rt MJCF meshes; link1-5: k-means
# clusters of the collision-mesh vertices, one minimum-volume enclosing capsule per
# cluster; tip_left / tip_right: one capsule per convex piece of the fingertip's
# collision decomposition (49 each). Body frames as in _CHAIN.
_CAPSULES: dict[
    str, tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...]
] = {
    "base": (((0.0001, 0.0795, 0.0310), (0.0000, -0.0853, 0.0330), 0.0489),),
    "link1": (
        ((0.0072, -0.0000, 0.0072), (0.0398, 0.0012, 0.0103), 0.0212),
        ((-0.0235, -0.0012, 0.0098), (-0.0342, 0.0171, 0.0404), 0.0258),
        ((0.0357, 0.0200, 0.0455), (0.0357, 0.0216, 0.0436), 0.0187),
    ),
    "link2": (
        ((0.1802, 0.0138, -0.0335), (0.1875, 0.0162, -0.0413), 0.0331),
        ((-0.0250, 0.0131, 0.0027), (0.0904, 0.0168, -0.0110), 0.0176),
        ((-0.0266, -0.0132, -0.0698), (0.0902, -0.0166, -0.0517), 0.0175),
        ((0.2639, 0.0000, -0.0586), (0.2177, 0.0000, -0.0579), 0.0324),
        ((-0.0161, 0.0146, -0.0642), (0.1009, 0.0167, -0.0506), 0.0177),
        ((0.2638, -0.0000, -0.0054), (0.2164, 0.0000, -0.0068), 0.0322),
        ((0.2077, -0.0194, -0.0245), (0.1714, -0.0083, -0.0314), 0.0283),
        ((-0.0160, -0.0142, -0.0012), (0.1010, -0.0161, -0.0119), 0.0175),
    ),
    "link3": (
        ((0.0642, -0.0221, -0.0025), (0.0608, -0.0827, -0.0188), 0.0312),
        ((0.0747, -0.1320, -0.0489), (0.0782, -0.2622, -0.0619), 0.0166),
        ((0.0643, -0.0229, -0.0653), (0.0604, -0.0836, -0.0558), 0.0318),
        ((-0.0006, -0.0000, 0.0050), (0.0104, 0.0022, -0.0061), 0.0324),
        ((0.0739, -0.1433, -0.0180), (0.0791, -0.2640, -0.0084), 0.0163),
        ((0.0137, 0.0011, -0.0662), (-0.0000, -0.0000, -0.0748), 0.0324),
        ((0.0438, -0.1348, -0.0168), (0.0490, -0.2699, 0.0011), 0.0171),
        ((0.0445, -0.1544, -0.0525), (0.0489, -0.2700, -0.0705), 0.0170),
    ),
    "link4": (
        ((-0.0156, -0.0088, -0.0727), (0.0130, -0.0080, -0.0722), 0.0115),
        ((-0.0040, 0.0345, -0.0003), (-0.0004, 0.0051, 0.0033), 0.0234),
        ((0.0114, 0.0291, -0.0721), (-0.0179, 0.0290, -0.0709), 0.0127),
        ((-0.0316, 0.0367, -0.0337), (-0.0402, 0.0711, -0.0316), 0.0318),
        ((-0.0940, 0.0527, -0.0324), (-0.0980, 0.0744, -0.0324), 0.0288),
    ),
    "link5": (
        ((-0.0224, -0.0362, 0.0337), (-0.0094, -0.0344, 0.0634), 0.0121),
        ((0.0000, 0.0183, 0.0212), (-0.0000, 0.0167, 0.0405), 0.0290),
        ((-0.0000, -0.0129, 0.0060), (-0.0009, -0.0316, 0.0221), 0.0215),
        ((0.0106, -0.0355, 0.0623), (0.0232, -0.0363, 0.0322), 0.0121),
    ),
    "gripper": (
        ((0.0000, -0.0000, -0.0450), (0.0000, -0.0000, -0.0450), 0.0290),
        ((-0.0251, -0.0375, -0.0509), (0.0249, -0.0372, -0.0514), 0.0158),
        ((0.0244, 0.0375, -0.0511), (-0.0249, 0.0372, -0.0514), 0.0159),
        ((0.0000, -0.0000, 0.0042), (-0.0002, 0.0005, 0.0044), 0.0293),
    ),
    "tip_left": (
        ((-0.0329, 0.0481, -0.0586), (-0.0574, 0.0361, -0.0558), 0.0083),
        ((-0.0165, -0.0065, -0.0589), (-0.0166, 0.0208, -0.0594), 0.0117),
        ((-0.0764, 0.0144, -0.0535), (-0.0321, 0.0001, -0.0566), 0.0098),
        ((-0.0270, 0.0186, -0.0577), (-0.0274, 0.0194, -0.0489), 0.0062),
        ((-0.0144, 0.0314, -0.0539), (-0.0308, 0.0323, -0.0538), 0.0113),
        ((-0.0370, 0.0467, -0.0484), (-0.0628, 0.0361, -0.0474), 0.0064),
        ((-0.0618, 0.0180, -0.0492), (-0.0871, 0.0202, -0.0463), 0.0052),
        ((-0.0651, 0.0267, -0.0506), (-0.0857, 0.0261, -0.0494), 0.0046),
        ((-0.0629, 0.0351, -0.0478), (-0.0853, 0.0255, -0.0467), 0.0080),
        ((-0.0345, 0.0474, -0.0513), (-0.0808, 0.0292, -0.0512), 0.0064),
        ((-0.0237, 0.0236, -0.0484), (-0.0235, 0.0538, -0.0487), 0.0083),
        ((-0.0154, 0.0034, -0.0011), (0.0021, 0.0038, -0.0011), 0.0068),
        ((-0.0277, 0.0108, -0.0574), (-0.0280, -0.0097, -0.0574), 0.0071),
        ((-0.0130, 0.0224, -0.0629), (-0.0130, 0.0365, -0.0629), 0.0036),
        ((-0.0144, 0.0088, -0.0010), (0.0034, 0.0098, -0.0011), 0.0058),
        ((-0.0131, 0.0298, -0.0600), (-0.0292, 0.0304, -0.0595), 0.0086),
        ((-0.0062, -0.0044, -0.0443), (-0.0062, -0.0044, -0.0010), 0.0121),
        ((-0.0280, 0.0113, -0.0648), (-0.0272, -0.0062, -0.0639), 0.0065),
        ((-0.0067, 0.0033, -0.0187), (-0.0067, 0.0033, -0.0443), 0.0104),
        ((-0.0262, 0.0458, -0.0513), (-0.0242, 0.0486, -0.0513), 0.0124),
        ((-0.0016, 0.0131, -0.0440), (-0.0016, 0.0133, -0.0199), 0.0066),
        ((-0.0250, 0.0475, -0.0558), (-0.0233, 0.0498, -0.0558), 0.0111),
        ((-0.0135, 0.0229, -0.0666), (-0.0137, 0.0388, -0.0662), 0.0044),
        ((0.0018, 0.0032, -0.0456), (-0.0056, 0.0030, -0.0457), 0.0029),
        ((-0.0299, 0.0556, -0.0628), (-0.0300, 0.0360, -0.0629), 0.0040),
        ((-0.0055, 0.0133, -0.0012), (0.0036, 0.0135, -0.0009), 0.0058),
        ((-0.0296, 0.0093, -0.0479), (-0.0299, -0.0087, -0.0481), 0.0076),
        ((-0.0166, 0.0042, -0.0472), (-0.0162, -0.0097, -0.0479), 0.0077),
        ((-0.0133, 0.0355, -0.0488), (-0.0132, 0.0228, -0.0484), 0.0035),
        ((-0.0364, 0.0013, -0.0490), (-0.0851, 0.0203, -0.0465), 0.0064),
        ((-0.0027, -0.0041, -0.0451), (-0.0011, -0.0065, -0.0450), 0.0068),
        ((-0.0248, 0.0459, -0.0599), (-0.0227, 0.0481, -0.0601), 0.0121),
        ((-0.0147, 0.0079, -0.0410), (0.0029, 0.0084, -0.0410), 0.0052),
        ((-0.0016, 0.0141, -0.0065), (-0.0016, 0.0141, -0.0179), 0.0063),
        ((0.0019, 0.0107, -0.0455), (-0.0068, 0.0104, -0.0457), 0.0062),
        ((-0.0210, 0.0241, -0.0629), (-0.0217, 0.0550, -0.0629), 0.0063),
        ((-0.0194, 0.0231, -0.0666), (-0.0198, 0.0564, -0.0664), 0.0042),
        ((-0.0111, 0.0101, -0.0470), (-0.0200, 0.0101, -0.0487), 0.0065),
        ((-0.0112, 0.0054, -0.0122), (-0.0000, 0.0079, -0.0122), 0.0099),
        ((-0.0249, 0.0137, -0.0652), (-0.0250, 0.0138, -0.0532), 0.0029),
        ((-0.0115, 0.0187, -0.0492), (-0.0229, 0.0187, -0.0487), 0.0045),
        ((-0.0141, 0.0080, -0.0242), (0.0029, 0.0087, -0.0242), 0.0070),
        ((-0.0292, 0.0138, -0.0533), (-0.0290, 0.0138, -0.0577), 0.0032),
        ((-0.0279, -0.0097, -0.0513), (-0.0278, 0.0151, -0.0513), 0.0058),
        ((-0.0269, 0.0318, -0.0651), (-0.0275, 0.0570, -0.0662), 0.0060),
        ((-0.0237, 0.0138, -0.0483), (-0.0313, 0.0138, -0.0481), 0.0033),
        ((-0.0146, 0.0087, -0.0336), (0.0029, 0.0091, -0.0336), 0.0060),
        ((-0.0081, -0.0101, -0.0459), (-0.0080, 0.0049, -0.0460), 0.0031),
        ((-0.0336, 0.0387, -0.0477), (-0.0338, 0.0537, -0.0481), 0.0037),
    ),
    "tip_right": (
        ((-0.0036, -0.0063, -0.0431), (-0.0034, -0.0062, 0.0027), 0.0109),
        ((0.0045, -0.0063, 0.0009), (0.0045, -0.0061, -0.0439), 0.0121),
        ((-0.0016, -0.0355, -0.0491), (-0.0221, -0.0853, -0.0473), 0.0079),
        ((-0.0109, -0.0033, -0.0141), (-0.0106, -0.0033, -0.0420), 0.0094),
        ((0.0019, -0.0111, -0.0519), (0.0002, -0.0312, -0.0519), 0.0099),
        ((-0.0081, -0.0129, -0.0145), (-0.0081, -0.0130, -0.0430), 0.0047),
        ((-0.0375, -0.0135, -0.0562), (-0.0190, -0.0134, -0.0562), 0.0038),
        ((-0.0267, -0.0858, -0.0471), (-0.0435, -0.0454, -0.0491), 0.0070),
        ((-0.0136, -0.0064, -0.0452), (-0.0137, 0.0038, -0.0454), 0.0030),
        ((-0.0081, -0.0129, -0.0057), (-0.0080, -0.0128, 0.0027), 0.0047),
        ((-0.0468, -0.0348, -0.0496), (-0.0435, -0.0385, -0.0491), 0.0091),
        ((-0.0089, 0.0031, -0.0101), (-0.0082, -0.0149, -0.0104), 0.0056),
        ((-0.0196, -0.0135, -0.0494), (-0.0365, -0.0139, -0.0499), 0.0043),
        ((-0.0137, 0.0030, -0.0101), (-0.0138, -0.0073, -0.0102), 0.0051),
        ((-0.0566, -0.0241, -0.0489), (-0.0204, -0.0241, -0.0494), 0.0082),
        ((-0.0149, -0.0300, -0.0492), (-0.0149, -0.0218, -0.0488), 0.0054),
        ((-0.0088, -0.0032, -0.0027), (-0.0107, -0.0015, -0.0032), 0.0084),
        ((0.0090, -0.0220, -0.0574), (-0.0318, -0.0210, -0.0585), 0.0122),
        ((-0.0096, -0.0059, 0.0017), (-0.0130, -0.0010, 0.0016), 0.0062),
        ((-0.0177, -0.0159, -0.0487), (0.0093, -0.0156, -0.0493), 0.0070),
        ((-0.0002, -0.0331, -0.0566), (-0.0103, -0.0556, -0.0554), 0.0073),
        ((-0.0091, -0.0211, -0.0519), (-0.0357, -0.0212, -0.0526), 0.0115),
        ((-0.0531, -0.0251, -0.0547), (-0.0317, -0.0234, -0.0546), 0.0100),
        ((-0.0474, -0.0225, -0.0596), (-0.0457, -0.0241, -0.0595), 0.0128),
        ((-0.0381, -0.0510, -0.0540), (-0.0472, -0.0311, -0.0580), 0.0085),
        ((-0.0011, -0.0358, -0.0526), (-0.0156, -0.0722, -0.0523), 0.0062),
        ((-0.0324, -0.0711, -0.0526), (-0.0478, -0.0322, -0.0532), 0.0067),
        ((-0.0564, -0.0244, -0.0562), (-0.0206, -0.0241, -0.0562), 0.0082),
        ((-0.0392, -0.0296, -0.0649), (-0.0376, -0.0121, -0.0656), 0.0050),
        ((-0.0467, -0.0304, -0.0634), (-0.0496, -0.0193, -0.0651), 0.0099),
        ((-0.0046, -0.0368, -0.0591), (0.0004, -0.0259, -0.0616), 0.0087),
        ((-0.0584, -0.0267, -0.0533), (-0.0296, -0.0265, -0.0533), 0.0065),
        ((-0.0055, 0.0016, -0.0450), (-0.0047, -0.0138, -0.0459), 0.0072),
        ((-0.0010, -0.0220, -0.0526), (-0.0010, -0.0220, -0.0526), 0.0165),
        ((-0.0352, -0.0182, -0.0654), (-0.0212, -0.0157, -0.0656), 0.0086),
        ((-0.0010, -0.0214, -0.0586), (-0.0010, -0.0214, -0.0586), 0.0161),
        ((-0.0352, -0.0179, -0.0533), (-0.0322, -0.0155, -0.0533), 0.0072),
        ((-0.0368, -0.0203, -0.0610), (-0.0366, -0.0108, -0.0610), 0.0021),
        ((-0.0247, -0.0208, -0.0547), (0.0090, -0.0217, -0.0547), 0.0119),
        ((-0.0059, -0.0295, -0.0493), (0.0020, -0.0259, -0.0497), 0.0099),
        ((-0.0030, -0.0148, -0.0657), (-0.0020, -0.0219, -0.0659), 0.0144),
        ((-0.0302, -0.0245, -0.0617), (-0.0301, -0.0103, -0.0618), 0.0043),
        ((-0.0244, -0.0180, -0.0617), (0.0089, -0.0184, -0.0617), 0.0088),
        ((-0.0065, -0.0213, -0.0562), (-0.0044, -0.0189, -0.0562), 0.0172),
        ((-0.0582, -0.0188, -0.0533), (-0.0411, -0.0187, -0.0533), 0.0026),
        ((-0.0365, -0.0133, -0.0547), (-0.0294, -0.0132, -0.0545), 0.0036),
        ((0.0057, -0.0153, -0.0455), (0.0060, 0.0013, -0.0452), 0.0055),
        ((-0.0466, -0.0252, -0.0520), (-0.0490, -0.0232, -0.0520), 0.0114),
        ((-0.0345, -0.0103, -0.0618), (-0.0345, -0.0306, -0.0617), 0.0026),
    ),
}


class Output(TypedDict):
    route: str
    detail: str
    min_clearance_m: float
    closest_link_a: str
    closest_link_b: str
    closest_point_a: dict[str, float]
    closest_point_b: dict[str, float]
    at_fraction: float
    start_clearance_m: float
    end_clearance_m: float
    samples: int
    model_error_m: float


# ---------------------------------------------------------------- geometry
def _quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_quat(r: np.ndarray) -> np.ndarray:
    """wxyz from a rotation matrix (Shepperd)."""
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def _frame(position, rotation_wxyz) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _quat_mat(rotation_wxyz)
    t[:3, 3] = np.asarray(position, dtype=float)
    return t


_FIXED = {name: _frame(pos, quat) for name, _, pos, quat, _, _ in _CHAIN}


def _axis_angle(axis, angle: float) -> np.ndarray:
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def forward_kinematics(q, finger: float = FINGER_TRAVEL) -> dict[str, np.ndarray]:
    """Every body's frame in the arm's base frame (4x4) for 6 arm joints and one finger slide [m]."""
    out: dict[str, np.ndarray] = {}
    for name, parent, _, _, joint, axis in _CHAIN:
        t = _FIXED[name] if parent is None else out[parent] @ _FIXED[name]
        if joint == "finger":
            m = np.eye(4)
            m[:3, 3] = np.asarray(axis) * float(finger)
            t = t @ m
        elif joint is not None:
            m = np.eye(4)
            m[:3, :3] = _axis_angle(axis, float(q[joint]))
            t = t @ m
        out[name] = t
    return out


def _segment_distance(p1, q1, p2, q2):
    """Closest distances between segment sets [p1, q1] (..., 3) and [p2, q2], broadcast; also the two points."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("...i,...i", d1, d1)
    e = np.einsum("...i,...i", d2, d2)
    f = np.einsum("...i,...i", d2, r)
    c = np.einsum("...i,...i", d1, r)
    b = np.einsum("...i,...i", d1, d2)
    eps = 1e-12
    a_ok, e_ok = a > eps, e > eps
    denom = a * e - b * b
    safe = np.where(np.abs(denom) > eps, denom, 1.0)
    s = np.where(np.abs(denom) > eps, np.clip((b * f - c * e) / safe, 0.0, 1.0), 0.0)
    e_safe = np.where(e_ok, e, 1.0)
    a_safe = np.where(a_ok, a, 1.0)
    t = (b * s + f) / e_safe
    s = np.where(
        t < 0.0,
        np.clip(-c / a_safe, 0.0, 1.0),
        np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s),
    )
    t = np.clip(t, 0.0, 1.0)
    # degenerate segments (points)
    s = np.where(~a_ok, 0.0, np.where(~e_ok, np.clip(-c / a_safe, 0.0, 1.0), s))
    t = np.where(~e_ok, 0.0, np.where(~a_ok, np.clip(f / e_safe, 0.0, 1.0), t))
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1), c1, c2


# ---------------------------------------------------------------- arm model
class _Arm:
    """One arm: its base in the world, its capsules, what it carries."""

    def __init__(
        self,
        arm_id: int,
        base: np.ndarray,
        attached: list[tuple[np.ndarray, float]],
        skip: set[str],
    ):
        self.arm_id = arm_id
        self.base = base
        self.attached = attached
        self.bodies = [
            (
                body,
                np.array([c[0] for c in caps]),
                np.array([c[1] for c in caps]),
                np.array([c[2] for c in caps]),
            )
            for body, caps in _CAPSULES.items()
            if body not in skip
        ]

    def segments(self, q, finger: float, tcp: np.ndarray):
        """World capsule ends (n, 3), (n, 3), radii (n,), labels for one configuration."""
        frames = forward_kinematics(q, finger)
        a, b, r, labels = [], [], [], []
        for body, ca, cb, cr in self.bodies:
            t = self.base @ frames[body]
            a.append(ca @ t[:3, :3].T + t[:3, 3])
            b.append(cb @ t[:3, :3].T + t[:3, 3])
            r.append(cr)
            labels.extend([body] * len(cr))
        if self.attached:
            t = self.base @ frames["gripper"] @ tcp
            centers = np.array([c for c, _ in self.attached]) @ t[:3, :3].T + t[:3, 3]
            a.append(centers)
            b.append(centers)
            r.append(np.array([radius for _, radius in self.attached]))
            labels.extend(["attached"] * len(self.attached))
        return np.vstack(a), np.vstack(b), np.concatenate(r), labels


def _vec(v) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _quat(q) -> np.ndarray:
    if isinstance(q, dict):
        return np.array([float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])])
    return np.asarray(q, dtype=float).reshape(4)


def _pose_matrix(pose: dict) -> np.ndarray:
    pose = pose.get("pose", pose)
    return _frame(_vec(pose["position"]), _quat(pose["rotation"]))


def _pose_dict(t: np.ndarray) -> dict:
    q = _mat_quat(t[:3, :3])
    return {
        "position": {"x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3])},
        "rotation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])},
    }


def _positions(js) -> np.ndarray:
    if isinstance(js, dict):
        js = js.get("positions", js.get("joint_config", js))
        if isinstance(js, dict):
            js = js.get("positions")
    return np.asarray(js, dtype=float).reshape(-1)


def _tcp(ctx: NodeContext, arm_id: int) -> np.ndarray:
    try:
        info = ctx.tool("robot.describe_arm", **_on(arm_id))
    except Exception as exc:  # a connector may not describe its arm
        logger.info("checking-dual-arm-clearance: no robot.describe_arm (%s)", exc)
        info = None
    if isinstance(info, dict) and info.get("tcp_offset") is not None:
        rot = info.get("tcp_rotation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        return _frame(_vec(info["tcp_offset"]), _quat(rot))
    return _frame(*DEFAULT_TCP)


def _on(arm_id: int) -> dict:
    return {"arm_id": int(arm_id)} if int(arm_id) else {}


def _vec_out(p) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _attached(obj: dict | None) -> list[tuple[np.ndarray, float]]:
    if not obj:
        return []
    return [
        (np.asarray(s["center"], dtype=float).reshape(3), float(s["radius"]))
        for s in obj.get("spheres", [])
    ]


def _densify(waypoints: list[np.ndarray], step: float) -> list[np.ndarray]:
    out = [waypoints[0]]
    for prev, nxt in zip(waypoints, waypoints[1:], strict=False):
        n = max(1, int(math.ceil(float(np.max(np.abs(nxt - prev))) / step)))
        out.extend(prev + (nxt - prev) * (i / n) for i in range(1, n + 1))
    return out


def _slerp_pose(t0: np.ndarray, t1: np.ndarray, u: float) -> np.ndarray:
    q0, q1 = _mat_quat(t0[:3, :3]), _mat_quat(t1[:3, :3])
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(min(1.0, dot))
        q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    t = np.eye(4)
    t[:3, :3] = _quat_mat(q)
    t[:3, 3] = t0[:3, 3] + u * (t1[:3, 3] - t0[:3, 3])
    return t


def _angle(t0: np.ndarray, t1: np.ndarray) -> float:
    c = (float(np.trace(t0[:3, :3].T @ t1[:3, :3])) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


class _Unknown(Exception):
    pass


def _solve(ctx: NodeContext, arm_id: int, t: np.ndarray, label: str) -> np.ndarray:
    try:
        reply = ctx.tool("robot.solve_ik", pose=_pose_dict(t), **_on(arm_id))
    except Exception as exc:  # a real station raises an unreachable pose
        raise _Unknown(f"arm {arm_id}: IK refused {label} ({exc})") from exc
    if (
        not isinstance(reply, dict)
        or reply.get("success") is False
        or reply.get("within_servo_tolerance") is False
    ):
        raise _Unknown(f"arm {arm_id}: IK found no solution within tolerance for {label}")
    q = _positions(reply.get("joint_config", reply))
    if q.size < 6:
        raise _Unknown(f"arm {arm_id}: IK returned {q.size} joints, the YAM model needs 6")
    return q[:6]


def _pose_path(
    ctx: NodeContext, arm_id: int, start_q, start_tcp, seq: dict, cart_step: float
) -> list[np.ndarray]:
    joints = [start_q]
    prev_t = start_tcp
    for k, wp in enumerate(seq.get("waypoints", [])):
        target = _pose_matrix(wp["pose"] if "pose" in wp else wp)
        straight = wp.get("cartesian", True) and wp.get("mode") != "planned_joint"
        n = 1
        if straight:
            n = max(
                1,
                int(math.ceil(float(np.linalg.norm(target[:3, 3] - prev_t[:3, 3])) / cart_step)),
                int(math.ceil(_angle(prev_t, target) / ROTATION_STEP)),
            )
        for i in range(1, n + 1):
            q = _solve(
                ctx, arm_id, _slerp_pose(prev_t, target, i / n), f"pose waypoint {k} ({i}/{n})"
            )
            jump = float(np.max(np.abs(q - joints[-1])))
            if straight and jump > IK_JUMP:
                raise _Unknown(
                    f"arm {arm_id}: IK changed branch along pose waypoint {k} (a joint jumps {jump:.2f} rad)"
                )
            joints.append(q)
        prev_t = target
    return joints


def _resample(path: list[np.ndarray], n: int) -> list[np.ndarray]:
    """n samples of a dense joint path by arc length."""
    if len(path) == 1:
        return [path[0]] * n
    arr = np.array(path)
    seg = np.max(np.abs(np.diff(arr, axis=0)), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return [path[0]] * n
    u = np.linspace(0.0, s[-1], n)
    return [np.array([np.interp(x, s, arr[:, j]) for j in range(arr.shape[1])]) for x in u]


# ---------------------------------------------------------------- the skill
def run(
    ctx: NodeContext,
    arm_a: int = 0,
    arm_b: int = 1,
    trajectory_a: dict | None = None,
    trajectory_b: dict | None = None,
    poses_a: dict | None = None,
    poses_b: dict | None = None,
    attached_a: dict | None = None,
    attached_b: dict | None = None,
    grip_a: float = -1.0,
    grip_b: float = -1.0,
    order: str = "a_then_b",
    too_close_m: float = TOO_CLOSE,
    joint_step_rad: float = JOINT_STEP,
    cartesian_step_m: float = CARTESIAN_STEP,
    skip_links: str = "",
) -> Output:
    nan = float("nan")
    out: Output = {
        "route": "unknown",
        "detail": "",
        "min_clearance_m": nan,
        "closest_link_a": "",
        "closest_link_b": "",
        "closest_point_a": {},
        "closest_point_b": {},
        "at_fraction": nan,
        "start_clearance_m": nan,
        "end_clearance_m": nan,
        "samples": 0,
        "model_error_m": nan,
    }
    notes: list[str] = []

    def unknown(why: str) -> Output:
        out["detail"] = why
        return out

    arm_a, arm_b = int(arm_a), int(arm_b)
    if arm_a == arm_b:
        return unknown(f"arm_a and arm_b are both {arm_a}")
    if order not in ("a_then_b", "b_then_a", "together"):
        return unknown(f"order must be a_then_b, b_then_a or together, not {order!r}")

    # 1. Both arms' joints, tool poses and jaw openings, read together.
    try:
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        return unknown(f"no observation ({exc})")
    arms = (obs or {}).get("arms") or []
    if max(arm_a, arm_b) >= len(arms):
        return unknown(
            f"the connector reports {len(arms)} arm(s); no separate arm {max(arm_a, arm_b)} to check against"
        )
    state = {}
    for arm_id in (arm_a, arm_b):
        reading = arms[arm_id]
        try:
            q = _positions(reading["joint_state"])
            tcp_world = _pose_matrix(reading["ee_pose"])
        except Exception as exc:
            return unknown(f"arm {arm_id}: no joint or tool-pose reading ({exc})")
        if q.size != 6 or not np.all(np.isfinite(q)):
            return unknown(
                f"arm {arm_id}: {q.size} joint readings; the YAM capsule model has 6 joints"
            )
        grip = reading.get("gripper_fraction")
        grip = 1.0 if grip is None or not math.isfinite(float(grip)) else float(grip)
        state[arm_id] = {"q": q, "tcp_world": tcp_world, "grip": grip}
    pa, pb = state[arm_a]["tcp_world"][:3, 3], state[arm_b]["tcp_world"][:3, 3]
    if float(np.linalg.norm(pa - pb)) < SAME_ARM and np.allclose(
        state[arm_a]["q"], state[arm_b]["q"]
    ):
        return unknown(
            f"arms {arm_a} and {arm_b} read the same pose: the connector does not address a separate arm"
        )

    # 2. Each arm's base from its joints and its tool pose; the model checked
    #    against the connector's own forward kinematics where it has one.
    skip = {s.strip() for s in str(skip_links).split(",") if s.strip()}
    model: dict[int, _Arm] = {}
    worst_model = 0.0
    checked = True
    for arm_id, attached in ((arm_a, attached_a), (arm_b, attached_b)):
        st = state[arm_id]
        tcp = _tcp(ctx, arm_id)
        st["tcp"] = tcp
        own = forward_kinematics(st["q"])["gripper"] @ tcp
        base = st["tcp_world"] @ np.linalg.inv(own)
        q_test = st["q"] + np.array([0.25, 0.2, -0.2, 0.3, -0.3, 0.4])
        try:
            fk = ctx.tool(
                "robot.forward_kinematics",
                joint_configs=[st["q"].tolist(), q_test.tolist()],
                **_on(arm_id),
            )
            p0, p1 = (_pose_matrix(p) for p in fk["poses"][:2])
            theirs = np.linalg.inv(p0) @ p1
            ours = np.linalg.inv(own) @ forward_kinematics(q_test)["gripper"] @ tcp
            err = float(np.linalg.norm(theirs[:3, 3] - ours[:3, 3]))
            worst_model = max(worst_model, err)
        except Exception as exc:  # not every connector exposes its FK
            logger.info("checking-dual-arm-clearance: no robot.forward_kinematics (%s)", exc)
            checked = False
        model[arm_id] = _Arm(arm_id, base, _attached(attached), skip)
    out["model_error_m"] = worst_model if checked else nan
    if checked and worst_model > MODEL_TOLERANCE:
        return unknown(
            f"the connector's forward kinematics differs from the YAM model by {worst_model * 1e3:.1f} mm "
            "for the same joint change: this is not the modelled arm (or its joints are ordered differently)"
        )
    if not checked:
        notes.append("model not cross-checked (no robot.forward_kinematics)")
    if not all(model[i].bodies or model[i].attached for i in model):
        return unknown(f"skip_links={skip_links!r} leaves an arm with no capsules")

    # 3. Each arm's planned joint path.
    paths: dict[int, list[np.ndarray]] = {}
    try:
        for arm_id, traj, seq in ((arm_a, trajectory_a, poses_a), (arm_b, trajectory_b, poses_b)):
            st = state[arm_id]
            if traj and seq:
                return unknown(f"arm {arm_id}: give a trajectory or a pose sequence, not both")
            if traj:
                wps = [_positions(w)[:6] for w in traj.get("waypoints", [])]
                if any(w.size != 6 for w in wps):
                    return unknown(f"arm {arm_id}: a trajectory waypoint does not have 6 joints")
                joints = [st["q"], *wps]
            elif seq:
                joints = _pose_path(
                    ctx, arm_id, st["q"], st["tcp_world"], seq, float(cartesian_step_m)
                )
                if seq.get("attached_object") and not model[arm_id].attached:
                    model[arm_id].attached = _attached(seq["attached_object"])
            else:
                joints = [st["q"]]
            paths[arm_id] = _densify(joints, float(joint_step_rad))
    except _Unknown as exc:
        return unknown(str(exc))

    # 4. The combined motion: one arm after the other, or both together.
    pa_, pb_ = paths[arm_a], paths[arm_b]
    if order == "together":
        n = max(len(pa_), len(pb_))
        pairs = list(zip(_resample(pa_, n), _resample(pb_, n), strict=True))
    else:
        first, second = (arm_a, arm_b) if order == "a_then_b" else (arm_b, arm_a)
        f_path, s_path = paths[first], paths[second]
        seq_pairs = [(q, s_path[0]) for q in f_path] + [(f_path[-1], q) for q in s_path[1:]]
        pairs = seq_pairs if first == arm_a else [(b, a) for a, b in seq_pairs]

    fingers = {}
    for arm_id, override in ((arm_a, grip_a), (arm_b, grip_b)):
        frac = (
            float(override)
            if override is not None and float(override) >= 0
            else state[arm_id]["grip"]
        )
        fingers[arm_id] = FINGER_TRAVEL * min(1.0, max(0.0, frac))

    # 5. Capsule clearance at every sample (RoboRSI: d - (r1 + r2) over all
    #    cross-arm pairs, minimum kept with the pair that made it).
    best = (math.inf, -1, 0, 0, None, None)
    per_sample = []
    for k, (qa, qb) in enumerate(pairs):
        a0, a1, ar, al = model[arm_a].segments(qa, fingers[arm_a], state[arm_a]["tcp"])
        b0, b1, br, bl = model[arm_b].segments(qb, fingers[arm_b], state[arm_b]["tcp"])
        d, ca, cb = _segment_distance(
            a0[:, None, :], a1[:, None, :], b0[None, :, :], b1[None, :, :]
        )
        gap = d - ar[:, None] - br[None, :]
        i, j = np.unravel_index(int(np.argmin(gap)), gap.shape)
        g = float(gap[i, j])
        per_sample.append(g)
        if g < best[0]:
            best = (g, k, i, j, (al[i], bl[j]), (ca[i, j], cb[i, j]))
    g, k, _, _, labels, points = best
    out.update(
        min_clearance_m=g,
        closest_link_a=f"arm{arm_a}:{labels[0]}",
        closest_link_b=f"arm{arm_b}:{labels[1]}",
        closest_point_a=_vec_out(points[0]),
        closest_point_b=_vec_out(points[1]),
        at_fraction=k / max(1, len(pairs) - 1),
        start_clearance_m=per_sample[0],
        end_clearance_m=per_sample[-1],
        samples=len(pairs),
    )
    where = "now" if len(pairs) == 1 else f"at {out['at_fraction']:.0%} of the path"
    text = f"capsule clearance {g * 1e3:.1f} mm {where}, between {out['closest_link_a']} and {out['closest_link_b']}"
    out["route"] = "too_close" if g < float(too_close_m) else "clear"
    out["detail"] = "; ".join([text, *notes])
    return out
