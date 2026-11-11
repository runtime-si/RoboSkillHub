# VLA checkpoint and dataset catalog

Generated from `vla-checkpoints.json` and `vla-datasets.json` by `roboskillhub vla page --write`; do not edit by hand.
Index entries are not executable bundles: a checkpoint runs only through a policy bundle's server with a checked-in profile.

Snapshot 2026-11-11: 381 checkpoints, 48 datasets.

**YAM compatibility:** as_is 56, adapter 256, finetune_base 25, no 44

**Family:** pi05 118, molmoact2 91, act 23, gr00t 23, diffusion-policy 20, smolvla 20, unknown 11, lingbot-vla 10, xvla 9, galaxea-g0 8, wam 7, cosmos-policy 4, flow-matching-policy 4, pi0 4, rdt 4, vla-jepa 3, wall-oss 3, dm0.5 2, flexpi 2, go-1 2, deva 1, dit-flow-matching-policy 1, dit-meanflow 1, eva 1, internvla-a1.5 1, nanovla 1, pelican-vla-0.5 1, pi0.6-style 1, rho 1, unifolm-vla 1, unifolm-wma 1, univla 1, wla 1

**Embodiment:** yam_i2rt 284, other 30, aloha 21, franka 14, so100 13, cross_embodiment 7, unitree 6, galaxea 3, piper 2, ur 1

## Checkpoints

| id | rev | family | embodiment | arms | action | YAM | our server | licence (verified) | evals |
|---|---|---|---|---|---|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM | 8dcbed66 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30/30 | as_is | policies/molmoact2-yam | Apache-2.0 (molmoact2 GitHub README; HF card has no license field) (yes) | twin closed loop, top view cropped to 640x360 0/10 |
| helen9975/pi05-molmoact-yam | 6b50a13e | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (card) (yes) | twin closed loop, raw 640x480 top view 0/10; twin closed loop, top view cropped to 640x360 0/10 |
| NUSMAGIC/pi05-MolmoAct2-YAM | 2f28d00a | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| robocurve/gr00t-n1.7-yam-molmoact2 | f04fe258 | gr00t | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | policies/gr00t:n1.7 | NVIDIA Open Model License (card: license other, nvidia-license) (yes) | twin closed loop, no marks 0/10; twin closed loop, grasp/release marks 0/10; twin closed loop, marks + text 0/10 |
| RoboColosseum/yam-cups-molmoact2 | 1eb569f7 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| RoboColosseum/yam-drawer-molmoact2 | 8946de60 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| RoboColosseum/yam-dustpan-molmoact2 | 864799c4 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| RoboColosseum/yam-microwave-molmoact2 | 08bd2654 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| RoboColosseum/yam-molmoact2 | db2c2c8b | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| RoboColosseum/yam-pi0.5 | ffa15ff8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| RoboColosseum/yam-tray-molmoact2 | 6d18a71c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| helen9975/molmoact2_telearms_v4 | e72ed998 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-003600 | e1c34cfb | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-007200 | b3615745 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-010800 | d1481d9c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-continuous-20261010-211513-step-003600 | dac0f100 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-continuous-20261010-211513-step-007200 | a57489cb | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-004000 | d9fa7a48 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-006000 | aaa9afcc | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-008000 | 3cd12007 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-002000 | 4c0b6c56 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-004000 | cc4798c7 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-006000 | 4c921f4d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-008000 | 9fbcd52d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-010000 | 56377a5c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-003320 | 18b25e2c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-006640 | 70df425d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-009960 | 039fc733 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| mikisafr/pi05-yam-towel-fold-100ep-step6000 | d0204e80 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | as_is | policies/gr00t:pi05 | None (no) | - |
| npow/pi05-yam-gello-red-cap-wsp-v15 | 3d887b21 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | as_is | policies/gr00t:pi05 | gemma (yes) | - |
| npow/servo-groot-yam-v3 | 63999202 | gr00t | yam_i2rt | 2 | absolute joint + gripper (inferred)/14/40 | as_is | policies/gr00t:n1.7 | apache-2.0 (yes) | - |
| nuffnuff/pi05-yam-finetuned | 40ef13e6 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/robocoin-yam-v3-208x30-pi05-m1 | 2bf312d8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-master-v1-strong-aug-pi05-m1 | a3dddbc1 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-intervene-i2-pi05-m1 | 6114f6f0 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-intervene-pi05-m1 | c6b4240e | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-pi05-m1 | 839da28c | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-teleop-combined-pi05-m1 | eaac5803 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| pztang/yam-abc-pickplace-safe-pi05-8gpu-m1 | 44cc2cd8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | None (no) | - |
| sra-vjti/molmoact2-yam-pi05-fold-pink-towel-vr | 78caa809 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | None (no) | - |
| sra-vjti/molmoact2-yam-pi05-fold-pink-towel-vr-expert | 09001a41 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | None (no) | - |
| vovw/MolmoAct2-Pink-Towel-DAgger | b0dcb251 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30/30 | as_is | policies/molmoact2-yam | None (no) | - |
| xuxie1031/MolmoAct2-YAM-StationeryBench-Sim | ab6b8b69 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | apache-2.0 (yes) | - |
| nvidia/GR00T-N1.7-3B | 373d34e2 | gr00t | franka | 2 | relative joint + relative EEF (XDOF tag), made absolute by server/14/40/30 | as_is | policies/gr00t:n1.7 | NVIDIA Open Model License (yes) | - |
| robocurve/pi0.5-yam | ee17bb36 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Gemma (card) (yes) | - |
| jeqcho/pi05-yam-bimanual | 36fa3017 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Apache-2.0 (card) (yes) | - |
| agents2agents/Pi0.5-BimanualYAM-Jetson-Thor | 5962a4b4 | pi05 | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | - | Gemma (card); Apache-2.0 runtime code; CUTLASS licence for plugins (yes) | - |
| ASethi04/pi05-BimanualYAM-oranges | 2c8132cf | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | - | apache-2.0 (yes) | - |
| ASethi04/pi05-BimanualYAM-oranges-uw-notop | 2986e7ea | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | - | apache-2.0 (yes) | - |
| ASethi04/pi05-BimanualYAM-oranges-uw-top | 48135ac8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | - | apache-2.0 (yes) | - |
| NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM | 5ba9c403 | gr00t | yam_i2rt | 2 | absolute joint/14/16/30 | as_is | - | apache-2.0 (card) (yes) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft | 0a81a531 | molmoact2 | yam_i2rt | 2 | absolute joint position/14/30/20 | as_is | - | not stated on card (yes) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert | a7488c87 | molmoact2 | yam_i2rt | 2 | absolute joint position/14/30/20 | as_is | - | not stated on card (yes) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft | 530fb6cf | molmoact2 | yam_i2rt | 2 | absolute joint position/14/30/20 | as_is | - | not stated on card (yes) | - |
| dddddsds31341/Long-WAM-Sorting-Dolls-20k-Step-20000 | 6ba38420 | wam | yam_i2rt | 2 | absolute joint position + gripper (original gripper units)/14/32/30 | as_is | - | other (yes) | - |
| fanqi-robo/pi05_insert_gear_in_gripper_expert_only | f2962292 | pi05 | yam_i2rt | 2 | absolute joint position/14/50/20 | as_is | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-dual-lidar-umi-12k | d83e40ee | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-long-gripper182-joint14 | d3bce515 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/24/30 | adapter | policies/molmoact2-yam | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges | 611ac1b5 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k | 501f9b0f | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-fixed | 0f0746e4 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-fixed-nodrop | b0fc2ae0 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-noaffine | 59d015d8 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-v2 | 9bde9a6c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-v2-nodropout | 272a168d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-20k | b9e6257e | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-ultrawide-12k | bc10c76f | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-ultrawide-wristonly-12k | 53bccd2b | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| Jiafei1224/2 | 9bb27605 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/10 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Jiafei1224/pi-05-flipplate2 | 2c93840c | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/15 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Jiafei1224/pi-05-microwave | 8d4e7eeb | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/15 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Jiafei1224/pi-05-pnp | c41a44f8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/10 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Jiafei1224/pi-05-pnp2 | e3881d21 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16/10 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Jiafei1224/pi05-liftpot | 48a8bbe1 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/10 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| RoboColosseum/yam-cups-gr00t-n1.7 | 683d984e | gr00t | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/40 | adapter | policies/gr00t:n1.7 | None (no) | - |
| RoboColosseum/yam-drawer-gr00t-n1.7 | 4d23c6ad | gr00t | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/40 | adapter | policies/gr00t:n1.7 | None (no) | - |
| RoboColosseum/yam-dustpan-gr00t-n1.7 | dc6e3859 | gr00t | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/40 | adapter | policies/gr00t:n1.7 | None (no) | - |
| RoboColosseum/yam-microwave-gr00t-n1.7 | a856e39e | gr00t | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/40 | adapter | policies/gr00t:n1.7 | None (no) | - |
| RoboColosseum/yam-tray-gr00t-n1.7 | 4d5dd48c | gr00t | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/40 | adapter | policies/gr00t:n1.7 | None (no) | - |
| Zasha01/yam_wire_pi | 42201f4e | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | None (no) | - |
| Zasha01/yam_wire_pi2 | 600531f6 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Zasha01/yam_wire_pi5 | 3f1cdd40 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | None (no) | - |
| Zasha01/yam_wire_pi_rename | 44c70e28 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Zasha01/yam_wire_policy_pi | 2d99114b | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| asawareeb/GR00T-YAM-box-to-bin-119demos | b188c8f5 | gr00t | yam_i2rt | None | unknown (no readable config) | adapter | policies/gr00t:n1.7 | None (no) | - |
| asawareeb/GR00T-YAM-box-to-bin-219demos | b18624fe | gr00t | yam_i2rt | None | unknown (no readable config) | adapter | policies/gr00t:n1.7 | None (no) | - |
| brandonyang/MolmoAct2-BimanualYAM-vive-teleop | 89b363d4 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert | e455c6f1 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step12000 | 4e550554 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step16000 | 4f92be2c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step20000 | 2cb6fe07 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step4000 | d23a7bf9 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step8000 | 8d8d2a43 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step12000 | 52d2a963 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step16000 | e318d45d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step20000 | e3c6c778 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step4000 | e46c662d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step8000 | 758efb3b | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step12000 | 867f09c2 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step16000 | 3b085858 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step20000 | 64f9bbdf | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step4000 | 27d5008a | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step8000 | de2f9a33 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step12000 | 2dc2aebf | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step16000 | 8f79c6ad | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step20000 | 7287d23d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step4000 | 564dde26 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step8000 | e8b54fad | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_zeroshot | 8886e0c6 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | policies/molmoact2-yam | None (no) | - |
| helen9975/pi05_telearms_v5_exp_deminf_1500 | 2715b285 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_elo_1500 | 32a5ce6f | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_random_100 | fca850c4 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_random_1000 | 31ab80bb | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_random_1500 | 49b6da3b | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_random_50 | 4d65a8d9 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_random_500 | 058b2898 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_exp_score_1500 | 91cd9758 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_high_elo | 971bd5e6 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_high_score | 5ea258c1 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v5_mixed | 61be0850 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_v6_lora | d9c66b85 | pi05 | yam_i2rt | 2 | delta/relative joint (gripper absolute)/14/50 | adapter | policies/gr00t:pi05 | None (no) | - |
| helen9975/pi05_telearms_yam_clean_30hz_recovered | 7914733b | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/gr00t:pi05 | None (no) | - |
| hqfang/yam_fold_100 | 4e1f980c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | None (no) | - |
| hqfang/yam_fold_70 | 6d67cbf1 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_3gpu | 54133dd2 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_ft_both_brightness_1ep_3gpu | c109e109 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_ft_continuous_brightness_1ep_3gpu | a9762b07 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | policies/molmoact2-yam | None (no) | - |
| lerobot/MolmoAct2-BimanualYAM-LeRobot | 743af86d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | policies/molmoact2-yam | None (no) | - |
| lilkm/MolmoAct2-BimanualYAM-LeRobot | 794bfe13 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | policies/molmoact2-yam | None (no) | - |
| monishluffy/yam-groot-n17-public | d6521544 | gr00t | yam_i2rt | None | unknown (no readable config) | adapter | policies/gr00t:n1.7 | None (no) | - |
| nuffnuff/yam-pour-direct-molmoact2-m1 | fbdf4bd0 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | policies/molmoact2-yam | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-m1 | 6943e497 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/15 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-v5-molmoact2-m1 | aa08235c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/15 | adapter | policies/molmoact2-yam | None (no) | - |
| sengi/rabc_pi05_matcha_yam_adv | 7c4c0fe6 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| TrossenRoboticsCommunity/pi05-block-transfer-lerobot | ef188f51 | pi05 | aloha | 2 | delta joints (12) + absolute grippers (2)/14/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (card) (yes) | - |
| TrossenRobotics/groot-n17-block-transfer | 690c3bed | gr00t | aloha | 2 | joint/14/16/30 | adapter | policies/gr00t:n1.7 | apache-2.0 (card) (yes) | - |
| TrossenRoboticsCommunity/pi0-handover-cube-widowxai | 0c8d2027 | pi0 | aloha | 2 | joint/14/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (card) (yes) | - |
| spikefly/pi05-arx-fold-towel-step30k | 25386a77 | pi05 | other | 2 | absolute joint/14/70/20 | adapter | policies/gr00t:pi05 | gemma (yes) | - |
| karisora/openarm-pi05-beaker100-v2-rand-slew004-qpos-torque-noise003-a100x4-25k-20261028 | efd978a3 | pi05 | other | 2 | absolute joint/16/50/30 | adapter | policies/gr00t:pi05 | none on card (no) | - |
| pepijn223/bimanual-so100-handover-cube_pi05 | 1b3bc873 | pi05 | so100 | 2 | absolute joint/12/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-canonical-umi75-teleop25-ee20-12k | 8a17b029 | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-dual-lidar-umi-currentrel-r6d-onset-v3-12k | 4e1a2c8d | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-freshbase-raw-umi75-teleop25-ee20 | c5bc8222 | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-freshbase-rotcontract-umi75-teleop25-ee20 | 6989d89e | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-long-gripper182-ee20 | 4b8761be | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24/30 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-yam-ultrawide-teleop-ee20-measured-fk-12k | 856928c1 | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24/30 | adapter | - | None (no) | - |
| ASethi04/pi05-BimanualYAM-yam-ultrawide-teleop-ee20-measured-fk-12k | 498131af | pi05 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24/30 | adapter | - | gemma (yes) | - |
| AaronHuangWei/Fast-WAM-YAM-Stack-Bowls-Deploy | e8df5e3f | wam | yam_i2rt | None | unknown (no readable config) | adapter | - | other (yes) | - |
| AaronHuangWei/Long-WAM-YAM-Stack-Bowls-Deploy | c90048dc | wam | yam_i2rt | None | unknown (no readable config) | adapter | - | other (yes) | - |
| Dimios45/yam-duster-bspline-dp | 3ee8b29e | diffusion-policy | yam_i2rt | None | unknown (no readable config)/30 | adapter | - | mit (yes) | - |
| Dimios45/yam-duster-dp | b9c7e286 | diffusion-policy | yam_i2rt | None | unknown (no readable config)/30 | adapter | - | mit (yes) | - |
| Dimios45/yam-pick-duster-200-bspline-ee | ff1b07f7 | diffusion-policy | yam_i2rt | None | EEF | adapter | - | mit (yes) | - |
| Dimios45/yam-pick-duster-200-bspline-joint | 04920660 | diffusion-policy | yam_i2rt | None | unknown (no readable config)/25 | adapter | - | mit (yes) | - |
| Dimios45/yam-pick-duster-bspline-ee | 9d537590 | diffusion-policy | yam_i2rt | None | EEF | adapter | - | mit (yes) | - |
| Dimios45/yam-pick-duster-bspline-joint | be9e487e | diffusion-policy | yam_i2rt | None | unknown (no readable config)/25 | adapter | - | mit (yes) | - |
| Efficient-Large-Model/Long-WAM-YAM | e104e72d | wam | yam_i2rt | 2 | absolute joint (z-scored)/14/32/30 | adapter | - | other (unspecified on card) (no) | - |
| HGLLL/act-bimanual-yam-production-0320 | 21f803e9 | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | apache-2.0 (yes) | - |
| HGLLL/act-bimanual-yam-test | e911788e | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | apache-2.0 (yes) | - |
| HGLLL/diffusion-bimanual-yam-production-0321 | 5dc814f0 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16/30 | adapter | - | apache-2.0 (yes) | - |
| Jiafei1224/VLA-JEPA-MolmoAct2YAMData | 8bd5446e | vla-jepa | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| Lebruhbruh/dp_yam_cube_pnplace_r1 | d1f2bac7 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/48/30 | adapter | - | None (no) | - |
| MikhailBird/MolmoAct2-ABC130K-70k | c9f200a6 | molmoact2 | yam_i2rt | 2 | unknown (no readable config) | adapter | - | None (no) | - |
| Noietch/dual-yam-delta-eef-v2-full-30k-checkpoint-20000 | 2aafc6b0 | eva | yam_i2rt | 2 | EEF | adapter | - | None (no) | - |
| RoboColosseum/yam-cups-g0.5 | 1e43799a | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| RoboColosseum/yam-cups-lingbot-vla-v2 | 6defeab7 | lingbot-vla | yam_i2rt | None | unknown (no readable config)/50 | adapter | - | None (no) | - |
| RoboColosseum/yam-cups-pi0.5 | a94aadc2 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| RoboColosseum/yam-drawer-g0.5 | 9f71851c | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| RoboColosseum/yam-drawer-lingbot-vla-v2 | 57848053 | lingbot-vla | yam_i2rt | None | unknown (no readable config)/50 | adapter | - | None (no) | - |
| RoboColosseum/yam-drawer-pi0.5 | 707319e0 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| RoboColosseum/yam-dustpan-g0.5 | b5151c04 | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| RoboColosseum/yam-dustpan-lingbot-vla-v2 | 32c9bfd9 | lingbot-vla | yam_i2rt | None | unknown (no readable config)/50 | adapter | - | None (no) | - |
| RoboColosseum/yam-dustpan-pi0.5 | 369c07f8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| RoboColosseum/yam-g0.5 | 52d438f2 | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | apache-2.0 (yes) | - |
| RoboColosseum/yam-gr00t-n1.7 | 59bd2a3e | gr00t | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16/30 | adapter | - | apache-2.0 (yes) | - |
| RoboColosseum/yam-lingbot-vla-v2 | 70dfa1bb | lingbot-vla | yam_i2rt | 2 | unknown (no readable config)/30 | adapter | - | apache-2.0 (yes) | - |
| RoboColosseum/yam-microwave-g0.5 | 7686c511 | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| RoboColosseum/yam-microwave-lingbot-vla-v2 | 3a9d07cc | lingbot-vla | yam_i2rt | None | unknown (no readable config)/50 | adapter | - | None (no) | - |
| RoboColosseum/yam-microwave-pi0.5 | 255a943b | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| RoboColosseum/yam-tray-g0.5 | 448f096f | galaxea-g0 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| RoboColosseum/yam-tray-lingbot-vla-v2 | 4a8b4186 | lingbot-vla | yam_i2rt | None | unknown (no readable config)/50 | adapter | - | None (no) | - |
| RoboColosseum/yam-tray-pi0.5 | a9b75a2f | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| SeerEnd1pity/YAM-deploy | 1be86bf5 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | apache-2.0 (yes) | - |
| Sichang0621/YAM-ABC-earbuds-pi05-v1-15k | d009fee5 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/YAM-ABC-earbuds-pi05-v1-18k | 02b345f9 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/YAM-ABC-fold-box-pi05-v1-15k | cc16f36f | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/YAM-ABC-fold-box-pi05-v1-20k | b0a5b885 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/YAM-ABC-fold-box-pi05-v1-23k | 8b430dc2 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/pi05_yam_abc_sort_legos_15k | 3ab86a69 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/pi05_yam_abc_sort_legos_18k | 29e8903c | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/pi05_yam_earbuds_teleop_10k | b5640f03 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/pi05_yam_earbuds_teleop_15k | 0ef8699c | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/pi05_yam_vial_4_30fps_recovery_10k | 86c38d8e | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-4-pi05-v1 | 1f027908 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-4-pi05-v2-10k | 7309d2a8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-4-pi05-v2-12k | 94f7631f | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-4-pi05-v2-7k | 3eb20d9f | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-aug-pi05-v1-10k | caae42fc | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-aug-pi05-v1-15k | c4848bfe | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| Sichang0621/yam-vial-pi06-v1-3k | b251ac83 | pi0.6-style | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| YzyLmc/pi05_pick_block_red_yam | 926459c7 | pi05 | yam_i2rt | None | unknown (no readable config)/32/50/30 | adapter | - | apache-2.0 (yes) | - |
| Zasha01/yam_wire_act_final | 6b4ef8d5 | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100 | adapter | - | None (no) | - |
| Zenith6666/yam-backbone | e8755fcc | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| akuls/cosmos3-edge-yam-yellow-box-policy | 4b5f2fab | cosmos-policy | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ashiq24/molmoact2-yam-lora | 869c4e07 | molmoact2 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ashiq24/pi05-yam-lora | 5b571803 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| atharva-pantheon/MolmoAct2-BimanualYAM-stackcube | 27526b19 | molmoact2 | yam_i2rt | 2 | unknown (no readable config) | adapter | - | apache-2.0 (yes) | - |
| atharva-pantheon/act-pantheon-yam-screwdriver-vn | 0a3ce10d | act | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/30 | adapter | - | None (no) | - |
| collected-ai/diffusion_yam_pick_up_box | bc4c302f | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/64/30 | adapter | - | None (no) | - |
| crilling/act_box_filling_yam_may31 | 2ea91589 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| crilling/smolvla_box_filling_yam_may31 | 408e2845 | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| dentronic/smolvla_policy | a082a902 | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/6/50/15 | adapter | - | apache-2.0 (yes) | - |
| ewinchell07/anvil-yam-pi05-all18-short-20261010 | c6799a4c | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-duration-sweep-20261010 | ca995e30 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-full-20k-20261011 | 7cffee29 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-legacy-20k-20261010 | 8e811ee9 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-preflight-20261010 | 56c2cd11 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| flex-pi/flexpi-pretrain-preview | c27e885a | flexpi | yam_i2rt | 2 | unknown (no readable config)/30 | adapter | - | mit (yes) | - |
| flex-pi/flexpi-pretrain-rgb-preview | 64fedf9f | flexpi | yam_i2rt | 2 | unknown (no readable config)/30 | adapter | - | mit (yes) | - |
| hcisbmm/smolvla_yam-loosen-bottle | b2d5922e | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/6/50/30 | adapter | - | None (no) | - |
| hcisbmm/smolvla_yam_pink | 66241215 | smolvla | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | adapter | - | apache-2.0 (yes) | - |
| hcisbmm/yam-loosen-bottle_policy_act | 9b378a9c | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | apache-2.0 (yes) | - |
| helen9975/groot-n1.7-molmoact-yam | 384ed60e | gr00t | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16/30 | adapter | - | apache-2.0 (yes) | - |
| helen9975/lingbot-vla-v2-6b-molmoact2-yam | af7529f2 | molmoact2 | yam_i2rt | 2 | unknown (no readable config)/30 | adapter | - | apache-2.0 (yes) | - |
| helen9975/pi05_telearms_frontback_lora | a7fb537a | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| helen9975/vla-jepa-molmoact-yam-3cam-unfrozen | ad3e07db | vla-jepa | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| helen9975/vla-jepa-molmoact-yam-blocks-v2 | 5e9c3526 | vla-jepa | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| hqfang/1203-yam-cortex_block-ditx-vit-clip-flow_matching | 43c896dc | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| hqfang/1203-yam-cortex_block-ditx-vit-clip-flow_matching-10k | 153e6679 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| hqfang/nov21-yam-fold_towel-dit-vit-clip | 67dfcd85 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| hqfang/nov22-yam-fold_towel-ditx-vit-clip | 38eae184 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| hqfang/nov22-yam-fold_towel-ditx-vit-clip-flow_matching | 47c182e7 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| hqfang/nov26-yam-ma_v2_fold-ditx-vit-clip-flow_matching | ed6dd262 | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| huzheyuan/pi05_yam_raiden5g_histnow | 0b7dc176 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/acrft-yam-extraction | 3786c16d | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | apache-2.0 (yes) | - |
| jellyho/pi05_yam_lego_taxi_alphaflow | 0d651a7b | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | apache-2.0 (yes) | - |
| jellyho/pi05_yam_lego_taxi_bc_s300_h30 | 399d7de4 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/pi05_yam_lego_taxi_bc_s300_h50 | 5c93adfc | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/pi05_yam_lego_taxi_rlt_s300 | b371d1b9 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep | 64bb00e9 | molmoact2 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| jlee24/molmoact2-yam-touch-grab-lora-20260602 | b91e352b | molmoact2 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/10 | adapter | - | None (no) | - |
| jlee24/pi05-yam-touch-grab-lora-20260602 | aafd18ab | pi05 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50 | adapter | - | None (no) | - |
| johannesmichalke/yam-pick-silver-pi05-v1 | 2ad7beaf | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| kabilanKB/yam_put_pot_act_m3 | 7511d136 | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | apache-2.0 (yes) | - |
| lastgeyan/1010_yam11_selfrep_pen_rel34_pm_caug_ft_abc4pq_bs9_2n_10ep_step180000 | 819980b3 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| lastgeyan/1012_yam11_selfrep_pen_rel34_nopm_caug_ft_abc4pq_bs16_2n_15ep_step046000 | 73480db3 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| lin881105/act_yam-toy-tape-3 | 9a24641f | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | None (no) | - |
| mengqz9/DeVA | 4f321201 | deva | yam_i2rt | None | unknown (no readable config) | adapter | - | mit (yes) | - |
| michaelwaves/lerobot_yam_green_can_pi | 612c1d3a | pi05 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50 | adapter | - | None (no) | - |
| michaelwaves/lerobot_yam_redbull_pi | 701d3653 | pi05 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| microsoft/rho-yam-box | 2dbbfdcf | rho | yam_i2rt | None | unknown (no readable config) | adapter | - | mit (yes) | - |
| miniFranka/wla_pretrain_diff_codefront_ViT_yam_egozed_3k_11k | 980c0a95 | wla | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| monishluffy/pi05-yam-4cam | 1b003395 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| monishluffy/pi05-yam-4cam-20k | d13dea57 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| npow/nanovla-bimanual-yam-3cam | 5949bfac | nanovla | yam_i2rt | 2 | unknown (no readable config)/30 | adapter | - | mit (yes) | - |
| npow/pi05-yam-red-cap-full-7500 | a0a41110 | pi05 | yam_i2rt | 1 | absolute joint + gripper (inferred)/7 | adapter | - | apache-2.0 (yes) | - |
| orchestrallabs/pi05-yam-red-cap-full-7500 | ececaedf | pi05 | yam_i2rt | 1 | absolute joint + gripper (inferred)/7 | adapter | - | apache-2.0 (yes) | - |
| pztang/yam-abc-pickplace-safe-molmoact2-yam-m1 | 5e4fbcc4 | molmoact2 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| qualia-robotics/act-yam-so101-teleop-0597d5d9 | 3666cae8 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| qualia-robotics/act-yam-so101-teleop-19cc84e9 | b3237b81 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| qualia-robotics/act-yam-so101-teleop-2d148523 | 964c2f85 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| qualia-robotics/act-yam-so101-teleop-63bf3b63 | 75726b37 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| qualia-robotics/act-yam-so101-teleop-a8b33bbc | 9724b97d | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| qualia-robotics/smolvla-yam-so101-teleop-ada6217a | e3d3385e | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| robocurve/dreamzero-yam-molmoact2 | f9b72b8d | wam | yam_i2rt | 2 | absolute joint (padded 32)/14/24/30 | adapter | - | nvidia-open-model-license (tag) (no) | - |
| robocurve/dreamzero-yam-molmoact2-checkpoints | a4fa70fa | wam | yam_i2rt | None | unknown (no readable config) | adapter | - | other (yes) | - |
| shaohuan1/miniworld-f2f-yam-public | 42718d61 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| shrg7/pi05-yam-toybox-5000 | 6891a636 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| shrg7/pi05-yam-toybox-9999 | f43be259 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| shrg7/pi05-yam-toybox-pglora-visl2sp-5000 | aadfb18d | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14/50 | adapter | - | apache-2.0 (yes) | - |
| sra-vjti/molmoact2-yam-pi05-jax | 27597dc0 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
| swang23/dreamzero_yam_20260413_2000 | 58024220 | wam | yam_i2rt | None | unknown (no readable config)/24 | adapter | - | None (no) | - |
| ttotmoon/yam-place-vial-dp-v0 | 0d8ed4ae | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-place-vial-fm-v0 | 088d9b1e | flow-matching-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-place-vial-pi05-v0 | 9b59cbee | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| ttotmoon/yam-place-vial-xvla-v0 | f4d1364b | xvla | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-vial-place-fm-v1 | b8c5808d | flow-matching-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-vial-place-fm-v2-h32 | c0ae8792 | flow-matching-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/32 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-vial-place-fm-v2-h32-train90 | f1b55595 | flow-matching-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/32 | adapter | - | apache-2.0 (yes) | - |
| ttotmoon/yam-vial-place-pi05-v1 | 89a33e80 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| ttotmoon/yam-vial-place-pi05-v1-train90 | e138e938 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| ttotmoon/yam-vial-place-xvla-v1 | c3740c68 | xvla | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | apache-2.0 (yes) | - |
| williamtsai726/fold_towel_0128 | 8bb635df | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/16/10 | adapter | - | apache-2.0 (yes) | - |
| RLinf/RLinf-Pi05-RealWorld-DualYAM-Clean_Desk | 15a037b4 | pi05 | yam_i2rt | 2 | delta joints + absolute normalized gripper/14/30 | adapter | - | none on card (no) | - |
| kiteml/dual-openyam-close-box-pi0_fast | 185c8187 | pi0 | other | 2 | absolute joint (FAST tokens)/14/50/30 | adapter | - | gemma (yes) | - |
| TrossenRoboticsCommunity/pi05-block-transfer-trossen-ai-openpi | 40aee785 | pi05 | aloha | 2 | joint (openpi trossen config)/14/50 | adapter | - | none on card (no) | - |
| TrossenRoboticsCommunity/smolvla-handover-cube-widowxai | fac7c568 | smolvla | aloha | 2 | joint/14/50/30 | adapter | - | apache-2.0 (yes) | - |
| ZUO66/pi05-r1lite-microwave | b7319da1 | pi05 | galaxea | 2 | delta joint flow (past 16 actions in)/14/16/15 | adapter | - | none (LICENSE.openpi retained) (no) | - |
| Dexmal/DM05-Table30v2-ALOHA | a852a10f | dm0.5 | aloha | 2 | joint/14/25 | adapter | - | gemma (yes) | - |
| robbyant/lingbot-vla-4b-posttrain-robotwin | fb71a2c9 | lingbot-vla | aloha | 2 | joint/14/50 | adapter | - | Apache-2.0 (README) (no) | - |
| liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex | 39300a2b | lingbot-vla | aloha | 2 | joint/14/50 | adapter | - | apache-2.0 (yes) | - |
| GM717/cosmos3-edge-robotwin-aloha-agilex | 6cfa2141 | cosmos-policy | aloha | 2 | absolute joint, raw units/14/16/16 | adapter | - | openmdw-1.1 (yes) | - |
| lerobot/smolvla_robotwin | eb55d3a7 | smolvla | aloha | 2 | joint/14/50 | adapter | - | apache-2.0 (yes) | - |
| DaMiTian/smolvla-aloha-bimanual | 55fdc769 | smolvla | other | 2 | joint/14/50 | adapter | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-SoftFold | a67dc68f | xvla | other | 2 | ee6d/20/30 | adapter | - | apache-2.0 (yes) | - |
| lerobot/xvla-agibot-world | 85764348 | xvla | other | 2 | ee6d/20/30 | adapter | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-RoboTwin2 | a157c580 | xvla | aloha | 2 | ee6d/20/30 | adapter | - | apache-2.0 (yes) | - |
| robotics-diffusion-transformer/RDT2-VQ | ec7e7f3a | rdt | franka | 2 | relative EEF chunk, RVQ tokens/20/24 | adapter | - | apache-2.0 (yes) | - |
| robotics-diffusion-transformer/RDT2-FM | f803808e | rdt | other | 2 | relative EEF/20/24 | adapter | - | apache-2.0 (yes) | - |
| nvidia/Cosmos-Policy-ALOHA-Predict2-2B | 68e3f927 | cosmos-policy | aloha | 2 | absolute joint (6+grip per arm)/14/50/25 | adapter | - | NVIDIA One-Way Noncommercial (NSCLv1) (yes) | - |
| JerryYuu/BrickCraft | 174fa340 | diffusion-policy | other | 1 | absolute EE pose: xyz + 6D rotation + gripper/10/16 | adapter | - | unspecified (policy) / AGPL-3.0 (yolo) (yes) | - |
| DreamMachines/arena2036_insertion_pi05_2op | 34642d77 | pi05 | other | 2 | joint position/14/50/50 | adapter | - | apache-2.0 (yes) | - |
| hud-evals/pi05-AssembleBench-cgdagger-r3 | c7e6a77a | pi05 | franka | 1 | absolute joint position (7) + gripper/8/15/15 | adapter | - | apache-2.0 (yes) | - |
| RoboticsCollectiveAachen/smolvla_piper_hex_bolt_insertion_80 | 39fc2a7c | smolvla | piper | 1 | absolute joint position/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| kelvinzhaozg/pi05_gim_arm_pin_insertion | 4b0bf39f | pi05 | other | 1 | joint position/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| ataabdulrhman/pi05_ur_markers_v2_37ep | 5989f7fc | pi05 | other | 2 | joint position/14/50/30 | adapter | - | apache-2.0 (yes) | - |
| ethanCSL/openarm_plugbattery_lerobot_real_smolvla_v00 | ba97a525 | smolvla | other | 2 | absolute joint position/16/50/30 | adapter | - | apache-2.0 (yes) | - |
| funXedu/so101_smolvla_lego_brick | 346e4658 | smolvla | so100 | 1 | absolute joint position/6/50/30 | adapter | - | apache-2.0 (yes) | - |
| arpitg1304/smolvla_s101_stack_lego | 0a05b5ad | smolvla | so100 | 1 | absolute joint position/6/50/30 | adapter | - | apache-2.0 (yes) | - |
| RoboticsCollectiveAachen/act_piper_hexagonal_bolt_insertion_zero_home | ec025c0c | act | piper | 1 | absolute joint position/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| kelvinzhaozg/pi05_gim_arm_industrial_assembly | b28da6de | pi05 | other | 1 | joint position/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| kelvinzhaozg/pi05_gim_arm_socket_plug | ec0c1066 | pi05 | other | 1 | joint position/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| thomas0829/policy-bimanual-smolvla-stacking | 9b5240c0 | smolvla | so100 | 2 | absolute joint position/12/50 | adapter | - | apache-2.0 (yes) | - |
| lerobot/pi05_base | 22c30e5b | pi05 | aloha | 2 | padded 32-D, flow matching/32/50 | finetune_base | policies/gr00t:pi05 | Gemma Terms (card tag gemma) (yes) | - |
| allenai/MolmoAct2 | e432d85f | molmoact2 | cross_embodiment | 2 | continuous flow, per norm_tag | finetune_base | policies/molmoact2-yam | Apache-2.0 (GitHub README; card has no field) (no) | - |
| lerobot/pi0_base | 6e245399 | pi0 | cross_embodiment | 2 | padded/32/50 | finetune_base | policies/gr00t:pi05 | gemma (yes) | - |
| Dexmal/DM05 | 5cd18734 | dm0.5 | aloha | 2 | per robot_type/50 | finetune_base | - | gemma (yes) | - |
| robbyant/lingbot-vla-v2-6b | 11c703bf | lingbot-vla | cross_embodiment | 2 | unified action space/55/50 | finetune_base | - | Apache-2.0 (README) (no) | - |
| robbyant/lingbot-vla-4b | 87712223 | lingbot-vla | other | 2 | unified, max 75-D/75/50 | finetune_base | - | Apache-2.0 (README) (no) | - |
| 2toINF/X-VLA-Pt | c1c4a64a | xvla | cross_embodiment | 2 | ee6d (EEF pos + rot6d + grip), 20-D per arm-pair/20/30 | finetune_base | - | apache-2.0 (yes) | - |
| lerobot/xvla-base | cdb7964e | xvla | cross_embodiment | 2 | ee6d/20/30 | finetune_base | - | apache-2.0 (yes) | - |
| robotics-diffusion-transformer/rdt-1b | eb09036c | rdt | aloha | 2 | unified 128-D space (joint/EEF pos/vel)/128/64 | finetune_base | - | MIT (yes) | - |
| robotics-diffusion-transformer/rdt-170m | 8aa386ca | rdt | other | 2 | unified/128/64 | finetune_base | - | MIT (yes) | - |
| nvidia/GR00T-N1.6-3B | d0814e7e | gr00t | franka | 2 | per tag/50 | finetune_base | - | NVIDIA One-Way Noncommercial (card link) (yes) | - |
| nvidia/GR00T-N1.5-3B | 869830fc | gr00t | franka | 2 | per tag/32/16 | finetune_base | - | NVIDIA One-Way Noncommercial (card link) (yes) | - |
| x-square-robot/wall-oss-flow | 44e82768 | wall-oss | other | 2 | flow, per dataset/20/32 | finetune_base | - | not on card (no) | - |
| lerobot/wall-oss-0.5 | 03ab102b | wall-oss | other | 2 | canonical 26-D/26/32 | finetune_base | - | none on card (no) | - |
| x-square-robot/wall-oss-1.0-lite-robodojo-base | 9d473873 | wall-oss | other | 2 | per norm | finetune_base | - | apache-2.0 (yes) | - |
| agibot-world/GO-1 | 0bc37be9 | go-1 | other | 2 | joint+grip+body/16/30/30 | finetune_base | - | CC BY-NC-SA 4.0 (yes) | - |
| agibot-world/GO-1-Air | 1cd2ac03 | go-1 | other | 2 | joint/16/30/30 | finetune_base | - | CC BY-NC-SA 4.0 (yes) | - |
| OpenGalaxea/G05 | e312be81 | galaxea-g0 | so100 | 2 | AR action tokens | finetune_base | - | G0.5 Community License (non-commercial) (yes) | - |
| OpenGalaxea/G0-VLA | d24d4d15 | galaxea-g0 | galaxea | 2 | joint | finetune_base | - | cc-by-nc-sa-4.0 (card); G0 PLUS license for newer commits (yes) | - |
| qwbu/univla-7b | 968e2896 | univla | other | 1 | latent -> per-robot decoder | finetune_base | - | apache-2.0 (yes) | - |
| unitreerobotics/UnifoLM-VLA-Base | 06fee501 | unifolm-vla | unitree | 2 | joint/7/16 | finetune_base | - | CC BY-NC-SA 4.0 (README) (yes) | - |
| unitreerobotics/UnifoLM-WMA-0-Dual | 88e73c19 | unifolm-wma | unitree | 2 | joint | finetune_base | - | card tag apache-2.0 but README says CC BY-NC-SA 4.0 (no) | - |
| X-Humanoid/Pelican-VLA05 | 5623cf16 | pelican-vla-0.5 | cross_embodiment | 2 | padded/32/50 | finetune_base | - | apache-2.0 (yes) | - |
| lerobot/pi0fast-base | 308eb844 | pi0 | cross_embodiment | 2 | FAST tokens/32/50 | finetune_base | - | gemma (yes) | - |
| lerobot/smolvla_base | 5e8d12a6 | smolvla | so100 | 1 | joint/6/50/30 | finetune_base | - | apache-2.0 (yes) | - |
| nvidia/G1-Factory-GR00T-1.7 | e0f5a056 | gr00t | unitree | 2 | whole-body/35/40 | no | policies/gr00t:n1.7 | NVIDIA Open Model License (yes) | - |
| xiaopeng-wu/pi05_unitree_g1 | 58023837 | pi05 | unitree | 2 | joint/18/50 | no | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| Dimios45/yam-world-model | 052b94bf | unknown | yam_i2rt | None | unknown (no readable config)/25 | no | - | apache-2.0 (yes) | - |
| Physis-AI/DreamDojo-YAM-14B-Diffusers | 5cc5e349 | unknown | yam_i2rt | None | unknown (no readable config) | no | - | None (no) | - |
| Physis-AI/DreamDojo-YAM-2B-Diffusers | 6de29da6 | unknown | yam_i2rt | None | unknown (no readable config) | no | - | other (yes) | - |
| agents2agents/MolmoAct2-BimanualYAM-Dynamic-Jetson-Thor | 5acc4333 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | no | - | apache-2.0 (yes) | - |
| agents2agents/MolmoAct2-Jetson-Thor | d7e73406 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | no | - | apache-2.0 (yes) | - |
| dgeorgiou/robometer-dual-yam-vision-exp6 | 168f218c | unknown | yam_i2rt | 2 | unknown (no readable config) | no | - | apache-2.0 (yes) | - |
| jellyho/acrft-yam-critics | 8469f751 | unknown | yam_i2rt | None | unknown (no readable config) | no | - | apache-2.0 (yes) | - |
| TrossenRoboticsCommunity/act-peg-insertion-aloha-stationary | 673d12c1 | act | aloha | 2 | absolute joint/14/100/50 | no | - | none on card (no) | - |
| TrossenRoboticsCommunity/act-fold-tshirt-aloha | f0691cb4 | act | aloha | 2 | joint/18/100/30 | no | - | none on card (no) | - |
| lerobot/act_aloha_sim_insertion_human | 33259aa8 | act | aloha | 2 | absolute joint position/14/100/50 | no | - | apache-2.0 (yes) | - |
| lerobot/act_aloha_sim_transfer_cube_human | ba73b276 | act | aloha | 2 | absolute joint/14/100/50 | no | - | apache-2.0 (yes) | - |
| nvidia/GR00T-N1.6-G1-PnPAppleToPlate | 897d0313 | gr00t | unitree | 2 | whole-body/50 | no | - | NVIDIA One-Way Noncommercial (no) | - |
| nvidia/GN1x-Tuned-Arena-GR1-Manipulation | 236cc79b | gr00t | galaxea | 2 | arms+hands/32/16 | no | - | NVIDIA License (card link) (no) | - |
| crislmfroes/groot-openarm-bimanual-open-microwave-sim | 39716589 | gr00t | other | 2 | joint/7/50 | no | - | apache-2.0 (yes) | - |
| nvidia/Cosmos3-Nano-Sim-Bimanual | da4d2c09 | cosmos-policy | yam_i2rt | 2 | action-conditioned video/64/24 | no | - | OpenMDW 1.1 (yes) | - |
| danielsanjosepro/lego_stack_f10_v1 | 1bd168ba | dit-flow-matching-policy | franka | 1 | EE pose xyz+rpy + gripper (absolute target)/7/16/10 | no | - | apache-2.0 (yes) | - |
| rangers-intrinsic/SC-only-connector-insertion-72successes-simplified | 6975de4f | act | other | 1 | absolute Cartesian TCP pose (xyz + quaternion)/7/50 | no | - | apache-2.0 (yes) | - |
| lucazanett/pi05-plug_adaptive_3 | 8f660100 | pi05 | other | 1 | absolute EE pose xyz + 6D rotation + motion stiffness + gripper/11/50/50 | no | - | apache-2.0 (yes) | - |
| Artefacts/smolvla-shape-sorting-30fps | 1e20b7b7 | smolvla | so100 | 1 | absolute joint position/6/50/30 | no | - | apache-2.0 (yes) | - |
| heeraku/smolvla_puzzle_cat_30k_v5 | aaa72bb4 | smolvla | other | 1 | absolute joint position/6/50 | no | - | apache-2.0 (yes) | - |
| hud-evals/pi05-AssembleBench-12k | 9b5843a8 | pi05 | franka | 1 | absolute joint position + gripper/8/15/15 | no | - | apache-2.0 (yes) | - |
| EvanMath/sort-lego-bricks | d06819a1 | act | other | 1 | absolute joint position/6/100 | no | - | apache-2.0 (yes) | - |
| RoboColosseum/droid-put-marker-pen-molmoact2 | fc01c9db | molmoact2 | franka | 1 | absolute joint position (7) + gripper/8/30/15 | no | - | apache-2.0 (yes) | - |
| Syqnal/act_kitting_connector | d06a1e3d | act | other | 1 | absolute joint position/6/100/15 | no | - | apache-2.0 (yes) | - |
| TomasAnderegg/pill_sorting_full_ft_a100_v3_dagger60 | f5c9df7b | smolvla | so100 | 1 | absolute joint position/6/50/30 | no | - | apache-2.0 (yes) | - |
| chomeed/board_insertion_ablation_head_pi05_mem_delta_recomputed_stats | 60aded89 | pi05 | other | 2 | delta (per name) - 8-D action/8/30/30 | no | - | apache-2.0 (yes) | - |
| danielsanjosepro/ditmeanflow_stacking_ft_v2 | 1033553e | dit-meanflow | franka | 1 | EE pose xyz+rpy + gripper/7/16/15 | no | - | apache-2.0 (yes) | - |
| lukasskellijs/pi05_assembly_bench_3_full_vlm_12k | 6a447e34 | pi05 | franka | 1 | absolute joint position (7) + gripper/8/15/15 | no | - | apache-2.0 (yes) | - |
| moonjongsul/manufacturing-kitting-smolvla | cd377a91 | smolvla | franka | 1 | joint position/8/50/30 | no | - | apache-2.0 (yes) | - |
| pravsels/molmoact2_cable_clip_20k | 97d6abab | molmoact2 | so100 | 1 | absolute joint position/6/30/20 | no | - | not stated (yes) | - |
| Grigorij/Domino_sum_xvla | 84ab657d | xvla | so100 | 1 | absolute joint position/6/30/30 | no | - | apache-2.0 (yes) | - |
| Jlbrockett/pi05_sponge_marker_cup_cube | 3e0166cb | pi05 | other | 1 | absolute joint position/6/50/30 | no | - | apache-2.0 (yes) | - |
| TrossenRoboticsCommunity/act-lego-aloha-solo | 3a38538b | act | aloha | 1 | absolute joint position/100 | no | - | none on card (yes) | - |
| bkuster/jsi_franka_peg_in_hole_v12 | 734bd8d0 | smolvla | franka | 1 | tool-frame relative Cartesian command (xyz + rot) + gripper/7/50/15 | no | - | apache-2.0 (yes) | - |
| castanetnicolas/diffusion_UR5e_BS_64_H_16_TASK_three_piece_assembly_ID_149425 | 091a0f3d | diffusion-policy | ur | 1 | EE delta pose + gripper/7/16/20 | no | - | apache-2.0 (yes) | - |
| jyang-ca/so101_test_domino_from_colab | 411fb23e | act | so100 | 1 | absolute joint position/6/100/30 | no | - | apache-2.0 (yes) | - |
| mulligan/real-marker-d2-r05-mulligan-dp | cb461d1b | diffusion-policy | franka | 1 | joint position (7) + gripper; also cartesian heads/7/12 | no | - | mit (yes) | - |
| nvidia/GR00T-N1.5-3B_Assemble_Trocar | d52a68c9 | gr00t | unitree | 2 | joint position (arm 7 per side + hands)/16 | no | - | other (yes) | - |
| pravsels/groot1.7_cable_clip_20k | b646e6a6 | gr00t | so100 | 1 | joint position/6/40/20 | no | - | not stated (yes) | - |
| InternRobotics/InternVLA-A1.5-DOMINO | 5c13efbf | internvla-a1.5 | aloha | 2 | padded 32-D (DOMINO EE/joint)/32/50 | no | - | cc-by-nc-sa-4.0 (yes) | - |
| VivekSai07/xvla-franka-stacking-fixed-layout | 65d1d565 | xvla | franka | 1 | 8-D/8/30 | no | - | other (yes) | - |
| dhruv1672/smolvla-so101-peg-insertion | b5ed1006 | smolvla | so100 | 1 | absolute joint position/6/50 | no | - | not stated (yes) | - |

## Datasets

| id | rev | embodiment | episodes | format | licence | size GB |
|---|---|---|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM-Dataset | e9f21ae1 | yam_i2rt | 5145 | LeRobot (124 per-session repos; top + left/right wrist 640x360, 14-D abs joints, 30 Hz) | Apache-2.0 |  |
| allenai/19012026-block-13 | dc6a332d | yam_i2rt |  | LeRobot | Apache-2.0 |  |
| fanqi-robo/insert_gear_in_gripper | acdc9ac8 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.52 |
| XDOF/ABC-130k | d4c79f5a | yam_i2rt |  | see card | apache-2.0 | 23333.9 |
| hud-evals/AssembleBench | 016fefea | franka | 1355 | LeRobot v3.0 | apache-2.0 | 3.6 |
| robot-lev/reassemble | 37d242d3 | franka | 149 | LeRobot v3.0 | cc-by-4.0 | 86.33 |
| Samisaliveagain/piper-hexagonal-bolt-insertion-100-demos-v1 | 1ec623d8 | piper | 100 | LeRobot v3.0 | not stated | 1.27 |
| RDLwicked/gim_arm_pin_insertion_teleop_3cam_joint | 0f29cb73 | other | 175 | LeRobot v3.0 | apache-2.0 | 3.32 |
| danielsanjosepro/lego_stack_f10_v4 | 447ec020 | franka | 30 | LeRobot v2.1 | apache-2.0 | 0.06 |
| ZebTaylor/panthera-four-gear-assembly-real-20hz | 0c27a752 | other | 59 | LeRobot v3.0 | not stated | 27.96 |
| DreamMachines/arena2036_insertion_2op_224x224 | 279ba9d9 | other | 178 | LeRobot v3.0 | not stated | 2.27 |
| isaiahbjork/rehearse-nist-round8-insert-v1 | fbb6b2cc | franka | 1050 | LeRobot v3.0 | cc-by-4.0 | 1.56 |
| RDLwicked/gim_arm_industrial_assembly_teleop_3cam_joint | 0ab60d82 | other | 272 | LeRobot v3.0 | apache-2.0 | 2.9 |
| danielsanjosepro/lego-stacking-ft | 53bc5069 | franka | 19 | LeRobot v2.1 | apache-2.0 | 0.08 |
| funXedu/so101_lego_brick | 8353c054 | so100 | 86 | LeRobot v3.0 | apache-2.0 | 2.37 |
| arpitg1304/stack_lego | 67ca1898 | so100 | 52 | LeRobot v2.1 | not stated | 0.07 |
| dddddsds31341/sorting-dolls-lerobot | ffca92ae | yam_i2rt | 100 | LeRobot v2.1 | not stated | 1.65 |
| rocochallenge2025/rocochallenge2026_Industrial_Assembly | dc03b003 | other | 200 | LeRobot v3.0 | mit | 1.2 |
| rpwang/bimanual_assembly_insert_force_lerobot_mix854 | a8bd60ec | other | 854 | LeRobot v3.0 | apache-2.0 | 7.95 |
| ethanCSL/openarm_plugbattery_lerobot_real_v00 | 24eb9016 | other | 50 | LeRobot v3.0 | apache-2.0 | 1.06 |
| lucazanett/plug_adaptive_3 | ba45205b | other | 80 | LeRobot v3.0 | apache-2.0 | 1.11 |
| pravsels/cable_clip_remote_v2 | d152ff52 | so100 | 50 | LeRobot v3.0 | not stated | 1.23 |
| IPEC-COMMUNITY/furniture_bench_dataset_lerobot | 00ce2405 | franka | 5100 | LeRobot v2.0 | apache-2.0 | 19.64 |
| TrossenRoboticsCommunity/aloha_static_peg_insertion | 246790bf | aloha |  | see card | not stated | 1.52 |
| physical-intelligence/aloha_pen_uncap_diverse | e82d8b40 | aloha | 123 | LeRobot v2.0 | apache-2.0 | 144.22 |
| SalZa2004/so101_insert_gear_in_gripper | 632f32a8 | so100 | 60 | LeRobot v3.0 | apache-2.0 | 4.86 |
| Artefacts/shape-sorting-so101-30fps | 3738b5e6 | so100 | 1389 | LeRobot v3.0 | apache-2.0 | 19.11 |
| TomasAnderegg/so101_pill_sorting_v3 | 1d5758b2 | so100 | 300 | LeRobot v3.0 | apache-2.0 | 1.11 |
| ataabdulrhman/ur_markers_v2_train | b8c1e870 | other | 37 | LeRobot v3.0 | apache-2.0 | 0.59 |
| ZebTaylor/panthera-single-bolt-insertion-real-20hz | b217be18 | other | 105 | LeRobot v3.0 | not stated | 6.12 |
| TrossenRoboticsCommunity/trossen_ai_stationary_peg_insertion | e546a349 | aloha | 51 | LeRobot v2.1 | apache-2.0 | 1.17 |
| bkuster/jsi_franka_peg_in_hole_v12 | 5a49c733 | franka | 2308 | LeRobot v3.0 | apache-2.0 | 1.53 |
| Syqnal/kitting_connector | 92e90050 | other | 175 | LeRobot v3.0 | apache-2.0 | 0.96 |
| moonjongsul/manufacturing_kitting_dataset | 6fc4df49 | franka | 183 | LeRobot v3.0 | not stated | 3.16 |
| lerobot/koch_pick_place_5_lego | af1f4ff7 | so100 | 50 | LeRobot v3.0 | apache-2.0 | 4.23 |
| Voxel51/sharpa-assemble-gears-410ep | 9f2a1753 | other | 410 | LeRobot v3.0 | cc-by-4.0 | 19.43 |
| IPEC-COMMUNITY/berkeley_cable_routing_lerobot | 2ce23db5 | franka | 1482 | LeRobot v2.1 | apache-2.0 | 0.3 |
| lerobot/aloha_sim_insertion_human | cc571a3c | aloha | 50 | LeRobot v3.0 | mit | 4.32 |
| lerobot/svla_so100_sorting | 13870ca9 | so100 | 52 | LeRobot v3.0 | apache-2.0 | 2.5 |
| lerobot/svla_so100_stacking | 933bcc61 | so100 | 56 | LeRobot v3.0 | apache-2.0 | 1.77 |
| lerobot/svla_so101_pickplace | f641879e | so100 | 50 | LeRobot v3.0 | apache-2.0 | 0.26 |
| jyang-ca/so101_test_domino | 70394564 | so100 | 50 | LeRobot v2.1 | apache-2.0 | 0.45 |
| Grigorij/Domino_sum | 54d0d0c4 | so100 | 49 | LeRobot v3.0 | apache-2.0 | 0.92 |
| chomeed/peg_insertion_full_unprivileged_lerobot | 3808ad59 | other | 100 | LeRobot v3.0 | apache-2.0 | 0.01 |
| Jlbrockett/sponge_marker_cup_cube | a40d8f5a | other | 250 | LeRobot v3.0 | not stated | 1.7 |
| LightwheelAI/iros2026-ikea-assembly | fdd09a66 | other |  | see card | cc-by-4.0 | 175.7 |
| H-EmbodVis/DOMINO | 8da54aa2 | aloha |  | see card | apache-2.0 | 770.73 |
| armnet/busybox_push_green_button | aceecaaf | so100 | 39 | LeRobot v3.0 | apache-2.0 | 0.59 |
