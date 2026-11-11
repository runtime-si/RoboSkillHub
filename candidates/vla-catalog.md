# VLA checkpoint and dataset catalog

Generated from `vla-checkpoints.json` and `vla-datasets.json` by `roboskillhub vla page --write`; do not edit by hand.
Index entries are not executable bundles: a checkpoint runs only through a policy bundle's server with a checked-in profile.

Snapshot 2026-11-11: 7 checkpoints, 2 datasets.

**YAM compatibility:** as_is 7, adapter 0, finetune_base 0, no 0

**Family:** pi05 5, gr00t-n1.7 1, molmoact2 1

**Embodiment:** yam_i2rt 7

## Checkpoints

| id | rev | family | embodiment | arms | action | YAM | our server | licence (verified) | evals |
|---|---|---|---|---|---|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM | 8dcbed66 | molmoact2 | yam_i2rt | 2 | abs_joint/14/30/30 | as_is | policies/molmoact2-yam | Apache-2.0 (yes) | twin closed loop, top view cropped to 640x360 0/10 |
| helen9975/pi05-molmoact-yam | 6b50a13e | pi05 | yam_i2rt | 2 | abs_joint/14/30/30 | as_is | policies/gr00t:pi05 | Apache-2.0 (card) (yes) | twin closed loop, raw 640x480 top view 0/10; twin closed loop, top view cropped to 640x360 0/10 |
| NUSMAGIC/pi05-MolmoAct2-YAM | 2f28d00a | pi05 | yam_i2rt | 2 | abs_joint/14/30/30 | as_is | policies/gr00t:pi05 | Apache-2.0 (card) (yes) | - |
| robocurve/gr00t-n1.7-yam-molmoact2 | f04fe258 | gr00t-n1.7 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | policies/gr00t:n1.7 | NVIDIA Open Model License (card: license other, nvidia-license) (yes) | twin closed loop, no marks 0/10; twin closed loop, grasp/release marks 0/10; twin closed loop, marks + text 0/10 |
| robocurve/pi0.5-yam | ee17bb36 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Gemma (card) (yes) | - |
| jeqcho/pi05-yam-bimanual | 36fa3017 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Apache-2.0 (card) (yes) | - |
| agents2agents/Pi0.5-BimanualYAM-Jetson-Thor | 5962a4b4 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Gemma (card); Apache-2.0 runtime code; CUTLASS licence for plugins (yes) | - |

## Datasets

| id | rev | embodiment | episodes | format | licence | size GB |
|---|---|---|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM-Dataset | e9f21ae1 | yam_i2rt | 5145 | LeRobot (124 per-session repos; top + left/right wrist 640x360, 14-D abs joints, 30 Hz) | Apache-2.0 |  |
| allenai/19012026-block-13 | dc6a332d | yam_i2rt |  | LeRobot | Apache-2.0 |  |
