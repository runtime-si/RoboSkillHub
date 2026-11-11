# VLA checkpoint and dataset catalog

Generated from `vla-checkpoints.json` and `vla-datasets.json` by `roboskillhub vla page --write`; do not edit by hand.
Index entries are not executable bundles: a checkpoint runs only through a policy bundle's server with a checked-in profile.

Snapshot 2026-11-11: 527 checkpoints, 233 datasets.

**YAM compatibility:** as_is 57, adapter 316, finetune_base 66, no 88

**Family:** pi05 138, molmoact2 93, gr00t 38, act 31, smolvla 25, xvla 23, diffusion-policy 20, openvla 13, unknown 11, lingbot-vla 10, pi0 10, galaxea-g0 9, wam 7, wall-oss 6, nora 5, cogact 4, cosmos-policy 4, dm0 4, flow-matching-policy 4, hpt 4, octo 4, rdt 4, spatialvla 4, being-h0 3, being-h05 3, dm0.5 3, internvla-a1.5 3, twinvla 3, vla-adapter 3, vla-jepa 3, eo-1 2, flexpi 2, gigabrain 2, go-1 2, internvla-a1 2, internvla-m1 2, tracevla 2, crossformer 1, deva 1, dit-flow-matching-policy 1, dit-meanflow 1, eva 1, hy-vla 1, internw0 1, magma 1, minivla 1, nanovla 1, pelican-vla-0.5 1, pi0.6-style 1, rho 1, tau0-vla 1, uniact 1, unifolm-vla 1, unifolm-wma 1, univla 1, vitra 1, vla-foundry 1, wla 1

**Embodiment:** yam_i2rt 289, other 58, aloha 47, cross_embodiment 44, franka 43, so100 21, piper 7, unitree 7, galaxea 5, widowx 5, ur 1

## Checkpoints

| id | rev | family | embodiment | arms | action | YAM | our server | licence (verified) | evals |
|---|---|---|---|---|---|---|---|---|---|
| allenai/MolmoAct2-BimanualYAM | 8dcbed66 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30/30 | as_is | policies/molmoact2-yam | Apache-2.0 (molmoact2 GitHub README; HF card has no license field) (yes) | twin closed loop, top view cropped to 640x360 0/10; twin closed loop zero-shot screen, A2 2026-11-10 (mma2-eval, crop, official server) 0/10 |
| helen9975/pi05-molmoact-yam | 6b50a13e | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (card) (yes) | twin closed loop, raw 640x480 top view 0/10; twin closed loop, top view cropped to 640x360 0/10; twin closed loop zero-shot screen, A2 2026-11-10 (mma2-eval, crop) 0/10 |
| NUSMAGIC/pi05-MolmoAct2-YAM | 2f28d00a | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| robocurve/gr00t-n1.7-yam-molmoact2 | f04fe258 | gr00t | yam_i2rt | 2 | abs_joint/14/16/30 | as_is | policies/gr00t:n1.7 | NVIDIA Open Model License (card: license other, nvidia-license) (yes) | twin closed loop, no marks 0/10; twin closed loop, grasp/release marks 0/10; twin closed loop, marks + text 0/10; twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| RoboColosseum/yam-cups-molmoact2 | 1eb569f7 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| RoboColosseum/yam-drawer-molmoact2 | 8946de60 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| RoboColosseum/yam-dustpan-molmoact2 | 864799c4 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| RoboColosseum/yam-microwave-molmoact2 | 08bd2654 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| RoboColosseum/yam-molmoact2 | db2c2c8b | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| RoboColosseum/yam-pi0.5 | ffa15ff8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| RoboColosseum/yam-tray-molmoact2 | 6d18a71c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/6 |
| helen9975/molmoact2_telearms_v4 | e72ed998 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-003600 | e1c34cfb | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-007200 | b3615745 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-easy-brightness-1ep-both-20261010-211513-step-010800 | d1481d9c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/4 |
| jinkmn/molmoact2-yam-easy-brightness-1ep-continuous-20261010-211513-step-003600 | dac0f100 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| jinkmn/molmoact2-yam-easy-brightness-1ep-continuous-20261010-211513-step-007200 | a57489cb | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-06 0/4 |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-004000 | d9fa7a48 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-006000 | aaa9afcc | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-easy-ft-continuous-gpu2-20261008-232437-step-008000 | 3cd12007 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-002000 | 4c0b6c56 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-004000 | cc4798c7 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-006000 | 4c921f4d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-008000 | 9fbcd52d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-ft-continuous-2gpu-20261009-083506-step-010000 | 56377a5c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-003320 | 18b25e2c | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-006640 | 70df425d | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep-step-009960 | 039fc733 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | None (no) | - |
| mikisafr/pi05-yam-towel-fold-100ep-step6000 | d0204e80 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | as_is | policies/gr00t:pi05 | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| npow/pi05-yam-gello-red-cap-wsp-v15 | 3d887b21 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | as_is | policies/gr00t:pi05 | gemma (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| npow/servo-groot-yam-v3 | 63999202 | gr00t | yam_i2rt | 2 | absolute joint + gripper (inferred)/14/40 | as_is | policies/gr00t:n1.7 | apache-2.0 (yes) | - |
| nuffnuff/pi05-yam-finetuned | 40ef13e6 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| nuffnuff/robocoin-yam-v3-208x30-pi05-m1 | 2bf312d8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| nuffnuff/yam-master-v1-strong-aug-pi05-m1 | a3dddbc1 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| nuffnuff/yam-pour-direct-intervene-i2-pi05-m1 | 6114f6f0 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-intervene-pi05-m1 | c6b4240e | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-direct-pi05-m1 | 839da28c | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| nuffnuff/yam-pour-direct-teleop-combined-pi05-m1 | eaac5803 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| pztang/yam-abc-pickplace-safe-pi05-8gpu-m1 | 44cc2cd8 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/30 | as_is | policies/gr00t:pi05 | None (no) | - |
| sra-vjti/molmoact2-yam-pi05-fold-pink-towel-vr | 78caa809 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| sra-vjti/molmoact2-yam-pi05-fold-pink-towel-vr-expert | 09001a41 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | as_is | policies/gr00t:pi05 | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-04 0/6 |
| vovw/MolmoAct2-Pink-Towel-DAgger | b0dcb251 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30/30 | as_is | policies/molmoact2-yam | None (no) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
| xuxie1031/MolmoAct2-YAM-StationeryBench-Sim | ab6b8b69 | molmoact2 | yam_i2rt | 2 | absolute joint pose (norm_stats)/14/30 | as_is | policies/molmoact2-yam | apache-2.0 (yes) | twin closed loop zero-shot screen, twin cube-in-box, top view cropped 640x360, 600 rows @30 Hz, execute 25; task1 seeds 0-2 + task2 'Pick up the grey block and put it in the blue box.' seeds 3-5; host rsh-gpu-02 0/6 |
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
| ASethi04/pi05-BimanualYAM-long-gripper182-joint14 | fe70f115 | pi05 | yam_i2rt | 2 | absolute joint (14)/14/24 | as_is | - | gemma (yes) | - |
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
| monishluffy/yam-groot-n17-public | d6521544 | gr00t | yam_i2rt | None | unknown (no readable config) | adapter | policies/gr00t:n1.7 | None (no) | - |
| nuffnuff/yam-pour-m1 | 6943e497 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50/15 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| sengi/rabc_pi05_matcha_yam_adv | 7c4c0fe6 | pi05 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/50 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| TrossenRoboticsCommunity/pi05-block-transfer-lerobot | ef188f51 | pi05 | aloha | 2 | delta joints (12) + absolute grippers (2)/14/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (card) (yes) | - |
| TrossenRobotics/groot-n17-block-transfer | 690c3bed | gr00t | aloha | 2 | joint/14/16/30 | adapter | policies/gr00t:n1.7 | apache-2.0 (card) (yes) | - |
| TrossenRoboticsCommunity/pi0-handover-cube-widowxai | 0c8d2027 | pi0 | aloha | 2 | joint/14/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (card) (yes) | - |
| spikefly/pi05-arx-fold-towel-step30k | 25386a77 | pi05 | other | 2 | absolute joint/14/70/20 | adapter | policies/gr00t:pi05 | gemma (yes) | - |
| karisora/openarm-pi05-beaker100-v2-rand-slew004-qpos-torque-noise003-a100x4-25k-20261028 | efd978a3 | pi05 | other | 2 | absolute joint/16/50/30 | adapter | policies/gr00t:pi05 | none on card (no) | - |
| pepijn223/bimanual-so100-handover-cube_pi05 | 1b3bc873 | pi05 | so100 | 2 | absolute joint/12/50/30 | adapter | policies/gr00t:pi05 | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-canonical-umi75-teleop25-ee20-12k | 8a17b029 | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-dual-lidar-umi-12k | d83e40ee | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-dual-lidar-umi-currentrel-r6d-onset-v3-12k | 4e1a2c8d | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-freshbase-raw-umi75-teleop25-ee20 | c5bc8222 | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-freshbase-rotcontract-umi75-teleop25-ee20 | 6989d89e | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-long-gripper182-ee20 | 4b8761be | molmoact2 | yam_i2rt | 2 | EEF (2x xyz+rot6d+gripper, 20-D)/20/24/30 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-long-gripper182-joint14 | d3bce515 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/24/30 | adapter | - | None (no) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges | 611ac1b5 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k | 501f9b0f | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-fixed | 0f0746e4 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-fixed-nodrop | b0fc2ae0 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-noaffine | 59d015d8 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-v2 | 9bde9a6c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-12k-v2-nodropout | 272a168d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-20k | b9e6257e | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-ultrawide-12k | bc10c76f | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| ASethi04/MolmoAct2-BimanualYAM-oranges-ultrawide-wristonly-12k | 53bccd2b | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
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
| brandonyang/MolmoAct2-BimanualYAM-vive-teleop | 89b363d4 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | None (no) | - |
| collected-ai/diffusion_yam_pick_up_box | bc4c302f | diffusion-policy | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/64/30 | adapter | - | None (no) | - |
| crilling/act_box_filling_yam_may31 | 2ea91589 | act | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/100/30 | adapter | - | apache-2.0 (yes) | - |
| crilling/smolvla_box_filling_yam_may31 | 408e2845 | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50/30 | adapter | - | apache-2.0 (yes) | - |
| dentronic/smolvla_policy | a082a902 | smolvla | yam_i2rt | 1 | absolute joint + gripper (verified: config)/6/50/15 | adapter | - | apache-2.0 (yes) | - |
| ewinchell07/anvil-yam-pi05-all18-short-20261010 | c6799a4c | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-duration-sweep-20261010 | ca995e30 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-full-20k-20261011 | 7cffee29 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-legacy-20k-20261010 | 8e811ee9 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| ewinchell07/anvil-yam-pi05-preflight-20261010 | 56c2cd11 | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert | e455c6f1 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step12000 | 4e550554 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step16000 | 4f92be2c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step20000 | 2cb6fe07 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step4000 | d23a7bf9 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_expert_step8000 | 8d8d2a43 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step12000 | 52d2a963 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step16000 | e318d45d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step20000 | e3c6c778 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step4000 | e46c662d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_base_fft_step8000 | 758efb3b | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step12000 | 867f09c2 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step16000 | 3b085858 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step20000 | 64f9bbdf | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step4000 | 27d5008a | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_expert_step8000 | de2f9a33 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step12000 | 2dc2aebf | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step16000 | 8f79c6ad | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step20000 | 7287d23d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step4000 | 564dde26 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_fft_step8000 | e8b54fad | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
| fanqi-robo/molmoact2_insert_gear_in_gripper_yam_zeroshot | 8886e0c6 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/20 | adapter | - | None (no) | - |
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
| hqfang/yam_fold_100 | 4e1f980c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | None (no) | - |
| hqfang/yam_fold_70 | 6d67cbf1 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | None (no) | - |
| huzheyuan/pi05_yam_raiden5g_histnow | 0b7dc176 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/acrft-yam-extraction | 3786c16d | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | apache-2.0 (yes) | - |
| jellyho/pi05_yam_lego_taxi_alphaflow | 0d651a7b | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | apache-2.0 (yes) | - |
| jellyho/pi05_yam_lego_taxi_bc_s300_h30 | 399d7de4 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/pi05_yam_lego_taxi_bc_s300_h50 | 5c93adfc | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jellyho/pi05_yam_lego_taxi_rlt_s300 | b371d1b9 | pi05 | yam_i2rt | 2 | absolute joint + gripper (inferred)/14 | adapter | - | None (no) | - |
| jinkmn/molmoact2-yam-spell-ceatec-makuhari-both-brightness-1ep | 64bb00e9 | molmoact2 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_3gpu | 54133dd2 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_ft_both_brightness_1ep_3gpu | c109e109 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
| jinkmn/molmoact2_yam_spell_ceatec_ft_continuous_brightness_1ep_3gpu | a9762b07 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
| jlee24/molmoact2-yam-touch-grab-lora-20260602 | b91e352b | molmoact2 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/10 | adapter | - | None (no) | - |
| jlee24/pi05-yam-touch-grab-lora-20260602 | aafd18ab | pi05 | yam_i2rt | 1 | absolute joint + gripper (verified: config)/7/50 | adapter | - | None (no) | - |
| johannesmichalke/yam-pick-silver-pi05-v1 | 2ad7beaf | pi05 | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| kabilanKB/yam_put_pot_act_m3 | 7511d136 | act | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/100/30 | adapter | - | apache-2.0 (yes) | - |
| lastgeyan/1010_yam11_selfrep_pen_rel34_pm_caug_ft_abc4pq_bs9_2n_10ep_step180000 | 819980b3 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| lastgeyan/1012_yam11_selfrep_pen_rel34_nopm_caug_ft_abc4pq_bs16_2n_15ep_step046000 | 73480db3 | unknown | yam_i2rt | None | unknown (no readable config) | adapter | - | None (no) | - |
| lerobot/MolmoAct2-BimanualYAM-LeRobot | 743af86d | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
| lilkm/MolmoAct2-BimanualYAM-LeRobot | 794bfe13 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30 | adapter | - | None (no) | - |
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
| nuffnuff/yam-pour-direct-molmoact2-m1 | fbdf4bd0 | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/30 | adapter | - | apache-2.0 (yes) | - |
| nuffnuff/yam-pour-v5-molmoact2-m1 | aa08235c | molmoact2 | yam_i2rt | 2 | absolute joint + gripper (verified: config)/14/30/15 | adapter | - | None (no) | - |
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
| 2toINF/X-VLA-AgiWorld-Challenge | 0305b747 | xvla | unitree | 2 | ee6d (xyz + 6D rot + gripper per arm)/20/30 | adapter | - | apache-2.0 (yes) | - |
| lerobot/xvla-folding | c87d6e8f | xvla | aloha | 2 | ee6d/20/30 | adapter | - | apache-2.0 (yes) | - |
| InternRobotics/InternVLA-A1-3B-RoboTwin | ecfef92d | internvla-a1 | aloha | 2 | joint (14 used, padded 32)/32/50 | adapter | - | cc-by-nc-sa-4.0 (yes) | - |
| InternRobotics/InternVLA-A1.5-RoboTwin | af0d0e26 | internvla-a1.5 | aloha | 2 | joint padded 32/32/50 | adapter | - | cc-by-nc-sa-4.0 (yes) | - |
| InternRobotics/InternW0-RoboTwin-clean | e9db4c1a | internw0 | aloha | 2 | joint/14 | adapter | - | mit (yes) | - |
| Dexmal/DM05-robotwin2 | bef09062 | dm0.5 | aloha | 2 | abs joint (14 used of 32)/14/50 | adapter | - | gemma (yes) | - |
| Dexmal/DM0-table30_generalist_aloha | 3f018be1 | dm0 | aloha | 2 | abs joint/32/50 | adapter | - | cc (card tag) (yes) | - |
| Dexmal/DM0-table30_plug_in_network_cable | 6f71fe5e | dm0 | aloha | 2 | abs joint/32/50 | adapter | - | cc (card tag) (yes) | - |
| Dexmal/DM0-table30_stack_color_blocks | 60bd346f | dm0 | aloha | 2 | abs joint/32/50 | adapter | - | cc (card tag) (yes) | - |
| open-gigaai/GigaBrain-0.7-3.5B-RoboTwin2.0-Clean | 3de7c7e6 | gigabrain | aloha | 2 | abs joint/14/50 | adapter | - | apache-2.0 (yes) | - |
| lerobot/g05_robotwin20 | b3b9695c | galaxea-g0 | aloha | 2 | abs joint (flow head)/14/32 | adapter | - | other (yes) | - |
| lerobot/hy_vla_robotwin | b9136475 | hy-vla | aloha | 2 | abs joint/16/40 | adapter | - | apache-2.0 (yes) | - |
| jellyho/TwinVLA-aloha_handover_box | d9c2c30d | twinvla | aloha | 2 | abs joint/14 | adapter | - | none on card (no) | - |
| jellyho/TwinVLA-aloha_dish_drainer | 9605014e | twinvla | aloha | 2 | abs joint/14 | adapter | - | none on card (no) | - |
| jellyho/TwinVLA-aloha_lift_box | 7459ccc3 | twinvla | aloha | 2 | abs joint/14 | adapter | - | none on card (no) | - |
| ASethi04/pi05-BimanualYAM-long-gripper182-ee20 | 8a42009f | pi05 | yam_i2rt | 2 | EE20 (2x xyz+rot6d+gripper)/20/24 | adapter | - | gemma (yes) | - |
| ASethi04/pi05-BimanualYAM-canonical-umi75-teleop25-ee20-12k | 055bfde4 | pi05 | yam_i2rt | 2 | EE20 (2x xyz+rot6d+gripper)/20/24 | adapter | - | none declared on card (no) | - |
| ASethi04/pi05-BimanualYAM-freshbase-raw-umi100-ee20 | 4e28ece6 | pi05 | yam_i2rt | 2 | EE20 (2x xyz+rot6d+gripper)/20/24 | adapter | - | none declared on card (no) | - |
| kiteml/dual-openyam-close-box-xvla | f4686bbd | xvla | yam_i2rt | 2 | joint (auto action mode)/14/30 | adapter | - | apache-2.0 (yes) | - |
| HITdongdong/pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc_step29999_pytorch | a27af148 | pi05 | piper | 2 | abs joint/14/50 | adapter | - | none on card (no) | - |
| HITdongdong/pi05_piper_dual_weigh_apple_place_only_joint_step30000_pytorch | 1cfd2aa9 | pi05 | piper | 2 | abs joint/14/50 | adapter | - | none on card (no) | - |
| LightwheelAI/smolvla-double-piper-pnp | 2bc739e7 | smolvla | piper | 2 | abs joint/12/50 | adapter | - | none on card (no) | - |
| NONHUMAN-RESEARCH/handumi-openarm-562-640-pi05 | 86abb66a | pi05 | other | 2 | abs joint/16/50 | adapter | - | apache-2.0 (yes) | - |
| NONHUMAN-RESEARCH/handumi-openarm-562-640-act | 596bc382 | act | other | 2 | abs joint/16/100 | adapter | - | apache-2.0 (yes) | - |
| ethanCSL/openarm_insertbattery_lerobot_real_gr00t_v02 | da42182b | gr00t | other | 2 | abs joint/16/16 | adapter | - | apache-2.0 (yes) | - |
| ethanCSL/openarm_insertbattery_lerobot_real_pi05_v02 | aac172a6 | pi05 | other | 2 | abs joint/16/50 | adapter | - | apache-2.0 (yes) | - |
| ethanCSL/openarm_plugbattery_lerobot_real_gr00t_v00 | 1f172feb | gr00t | other | 2 | abs joint/16/16 | adapter | - | apache-2.0 (yes) | - |
| enactic/GR00T-N1.7-openarm-2-cell-pillow | ece62e5c | gr00t | other | 2 | abs joint/16/32 | adapter | - | apache-2.0 (yes) | - |
| Cybernetic-Labs/openarm-packingbench-v2-groot-n17-90k-390ep | cf3b3dce | gr00t | other | 2 | abs joint/16/40 | adapter | - | cc-by-4.0 (yes) | - |
| hebbianrobotics/pi05_openarm_folding_final | 307cdd6b | pi05 | other | 2 | abs joint/16/30 | adapter | - | apache-2.0 (yes) | - |
| polarisai-robots/bento_v2_openarm_pi05 | 9b4d4590 | pi05 | other | 2 | abs joint/24/30 | adapter | - | apache-2.0 (yes) | - |
| saybeok03/xvla_folding_openarm16_224 | 9d06d2ea | xvla | other | 2 | joint (auto)/16/30 | adapter | - | apache-2.0 (yes) | - |
| aractingi/bimanual-handover-groot-10k | e1f2c4fb | gr00t | so100 | 2 | abs joint/12/50 | adapter | - | none on card (no) | - |
| YieumYoon/groot-bimanual-so100-handover-cube | d726e2f5 | gr00t | so100 | 2 | abs joint/12/50 | adapter | - | apache-2.0 (yes) | - |
| pepijn223/bimanual-so100-handover-cube_pi0 | 9cfe6969 | pi0 | so100 | 2 | abs joint/12/50 | adapter | - | apache-2.0 (yes) | - |
| delvingdeep/pi0fast-so101-bimanual | 03723078 | pi0 | so100 | 2 | FAST tokens -> abs joint/12/50 | adapter | - | apache-2.0 (yes) | - |
| anuj0798/SO101_HexPeg_Bimanual_ACT_20K_20episodes | ea69c3e5 | act | so100 | 2 | abs joint/12/30 | adapter | - | apache-2.0 (yes) | - |
| cadene/act_koch_bimanual_folding | 16d0f5ee | act | so100 | 2 | abs joint/12/100 | adapter | - | none on card (no) | - |
| bilimili/xlerobot-dice-4cam-v2-act | 11584a88 | act | other | 2 | abs joint/15/100 | adapter | - | none on card (no) | - |
| bilimili/xlerobot-dice-4cam-v2-smolvla | 02f96db6 | smolvla | other | 2 | abs joint/15/50 | adapter | - | none on card (no) | - |
| zonglin11/xvla-xlerobot-fold | 2ab469d6 | xvla | other | 2 | joint (auto)/16/30 | adapter | - | apache-2.0 (yes) | - |
| wego-mink/dualarm_bolt_and_nut_act | 4bf8abfc | act | other | 2 | abs joint/14/100 | adapter | - | apache-2.0 (yes) | - |
| wego-mink/2601026_dualarm_bolt_and_nut_smolvla | 19b181bf | smolvla | other | 2 | abs joint/14/50 | adapter | - | apache-2.0 (yes) | - |
| zsoo0o/pi0fast-color-sort-dual-arm | 630195b1 | pi0 | other | 2 | FAST -> abs joint/12/10 | adapter | - | none on card (no) | - |
| zsoo0o/groot-color-sort-dual-arm | cc799e9c | gr00t | other | 2 | abs joint/12/16 | adapter | - | none on card (no) | - |
| rainbowrobotics/pi05_dualArm_singleObj_24-26_40k | ba53eda7 | pi05 | other | 2 | abs joint/16/50 | adapter | - | apache-2.0 (yes) | - |
| andnetdeboer/molmoact2-agilex-test_tube | faf60e52 | molmoact2 | aloha | 2 | abs joint/14/10 | adapter | - | apache-2.0 (yes) | - |
| locht131/gr00t_n17_motion_robot_agilex_stack_three_cups_rel_bs64 | 621444c5 | gr00t | aloha | 2 | relative joint/14 | adapter | - | none on card (no) | - |
| Xianzhe-Fan/pi05_plug_agilex_49999 | 01642b4c | pi05 | aloha | 2 | abs joint/14/50 | adapter | - | mit (yes) | - |
| Xianzhe-Fan/pi05_screw_agilex_49999 | a461b64f | pi05 | aloha | 2 | abs joint/14/50 | adapter | - | mit (yes) | - |
| TrossenRoboticsCommunity/bimanual-widowxai-pour-items-red-cup-pi05 | e4281ebd | pi05 | aloha | 2 | abs joint/14/50 | adapter | - | none on card (no) | - |
| TrossenRoboticsCommunity/pi0-fold-tshirt-aloha | 9c137870 | pi0 | aloha | 2 | abs joint/18/50 | adapter | - | none on card (no) | - |
| TrossenRoboticsCommunity/smolvla-block-stacking-trossen-ai | 00a62cd8 | smolvla | aloha | 2 | abs joint/14/50 | adapter | - | none on card (no) | - |
| TrossenRoboticsCommunity/act-peg-insertion-trossen-ai-stationary | 0bad8cd2 | act | aloha | 2 | abs joint/14/100 | adapter | - | none on card (no) | - |
| TrossenRoboticsCommunity/pi05-organize-tools-trossen-ai | f6931f65 | pi05 | aloha | 2 | abs joint/14/50 | adapter | - | none on card (no) | - |
| shashuo0104/260723_pi05_aloha_pipe_insert_unify_image_aug_v3 | 9be10c82 | pi05 | aloha | 2 | abs joint/50 | adapter | - | mit (yes) | - |
| prachigarg23/pi05_FAST_aloha_gear_insertion_ES | d91f21a2 | pi05 | aloha | 2 | FAST tokens -> abs joint/14 | adapter | - | apache-2.0 (yes) | - |
| ngo275/galaxea_act_20260514_50k_v2 | c69a91ce | act | galaxea | 2 | abs joint/20/100 | adapter | - | none on card (no) | - |
| Cache-SCA/Gr00t-Teleop-DualArmFoldTowel-20ep | 4c9e4c8c | gr00t | so100 | 2 | abs joint/12/16 | adapter | - | apache-2.0 (yes) | - |
| THU98/pi05_franka_dual_arm_uev_full_20k | da240440 | pi05 | franka | 2 | abs joint/16/20 | adapter | - | none on card (no) | - |
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
| openvla/openvla-7b | 47a0ec7f | openvla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | finetune_base | - | mit (yes) | - |
| rail-berkeley/octo-small | 03d88976 | octo | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion head)/7/4 | finetune_base | - | mit (yes) | - |
| rail-berkeley/octo-base | 39d6c88f | octo | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion head)/7/4 | finetune_base | - | mit (yes) | - |
| rail-berkeley/octo-small-1.5 | dc9aa301 | octo | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion head)/7/4 | finetune_base | - | mit (yes) | - |
| rail-berkeley/octo-base-1.5 | ee3c10e8 | octo | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion head)/7/4 | finetune_base | - | mit (yes) | - |
| rail-berkeley/crossformer | c7dea269 | crossformer | aloha | 2 | per-head: bimanual joint pos (ALOHA head) / EE delta (single-arm) / nav / quadruped/14/100/50 | finetune_base | - | mit (yes) | - |
| liruiw/hpt-small | fc12980e | hpt | cross_embodiment | 1 | embodiment-specific head | finetune_base | - | MIT (GitHub repo) (no) | - |
| liruiw/hpt-base | 4586c8c3 | hpt | cross_embodiment | 1 | embodiment-specific head | finetune_base | - | MIT (GitHub repo) (no) | - |
| liruiw/hpt-large | e0d9a371 | hpt | cross_embodiment | 1 | embodiment-specific head | finetune_base | - | MIT (GitHub repo) (no) | - |
| liruiw/hpt-xlarge | 0adad50d | hpt | cross_embodiment | 1 | embodiment-specific head | finetune_base | - | MIT (GitHub repo) (no) | - |
| IPEC-COMMUNITY/spatialvla-4b-224-pt | 886c4557 | spatialvla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), adaptive spatial action grids/7/4 | finetune_base | - | mit (yes) | - |
| IPEC-COMMUNITY/spatialvla-4b-mix-224-pt | cb7538f0 | spatialvla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), adaptive spatial action grids/7/4 | finetune_base | - | mit (yes) | - |
| CogACT/CogACT-Small | ca302a30 | cogact | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion transformer)/7/16 | finetune_base | - | mit (yes) | - |
| CogACT/CogACT-Base | 6550bf09 | cogact | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion transformer)/7/16 | finetune_base | - | mit (yes) | - |
| CogACT/CogACT-Large | 0a7e03ce | cogact | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (diffusion transformer)/7/16 | finetune_base | - | mit (yes) | - |
| Dexmal/Dexbotic-CogACT-HArm | ee6b2da7 | cogact | other | 2 | abs/delta per dexbotic config/16/16 | finetune_base | - | cc (card) (yes) | - |
| microsoft/Magma-8B | 4159b1de | magma | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) tokens/7/1 | finetune_base | - | mit (yes) | - |
| microsoft/VITRA-VLA-3B | 4bd47d5b | vitra | other | 1 | hand/EE action chunks | finetune_base | - | mit (yes) | - |
| 2toINF/UniAct | a59a6626 | uniact | cross_embodiment | 1 | universal action codebook -> embodiment head/7 | finetune_base | - | apache-2.0 (yes) | - |
| BeingBeyond/Being-H0-1B-2508 | 37d50d61 | being-h0 | other | 2 | hand motion tokens (MANO)/13 | finetune_base | - | mit (yes) | - |
| BeingBeyond/Being-H0-8B-2508 | e91be7e6 | being-h0 | other | 2 | hand motion tokens (MANO)/13 | finetune_base | - | mit (yes) | - |
| BeingBeyond/Being-H0-14B-2508 | f68ca50a | being-h0 | other | 2 | hand motion tokens (MANO)/13 | finetune_base | - | mit (yes) | - |
| BeingBeyond/Being-H05-2B | bb31ffcf | being-h05 | cross_embodiment | 2 | unified state-action space (padded)/32 | finetune_base | - | apache-2.0 (yes) | - |
| lerobot/being_h05_base | 44da81d1 | being-h05 | cross_embodiment | 2 | unified padded/12/16 | finetune_base | - | apache-2.0 (yes) | - |
| declare-lab/nora | 3f373eff | nora | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) FAST tokens/7/1 | finetune_base | - | none declared on card (no) | - |
| declare-lab/nora-long | 1fd12607 | nora | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) FAST tokens/7/5 | finetune_base | - | none declared on card (no) | - |
| declare-lab/nora-1.5 | 90c06ba4 | nora | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) flow-matching/7/5 | finetune_base | - | MIT (yes) | - |
| 2toINF/X-VLA-Pt-Auto-action-space | 89f2b56a | xvla | cross_embodiment | 2 | auto (joint or ee6d per embodiment)/20/30 | finetune_base | - | apache-2.0 (yes) | - |
| IPEC-COMMUNITY/EO-1-3B | fe0a015d | eo-1 | cross_embodiment | 2 | flow-matching continuous (padded 32)/32/50 | finetune_base | - | mit (yes) | - |
| lerobot/eo1-base | bba306bb | eo-1 | cross_embodiment | 2 | flow-matching padded/32/50 | finetune_base | - | mit (yes) | - |
| InternRobotics/InternVLA-M1 | ce8a3424 | internvla-m1 | cross_embodiment | 1 | EE delta (xyz, rpy, gripper) (spatial-prompted DiT)/7/16 | finetune_base | - | cc-by-nc-sa-4.0 (yes) | - |
| InternRobotics/InternVLA-A1-3B | 08cf716d | internvla-a1 | cross_embodiment | 2 | delta/abs joint (padded 32)/32/50 | finetune_base | - | cc-by-nc-sa-4.0 (yes) | - |
| InternRobotics/InternVLA-A1.5-base | 325331b5 | internvla-a1.5 | cross_embodiment | 2 | joint/EE padded 32/32/50 | finetune_base | - | cc-by-nc-sa-4.0 (yes) | - |
| nvidia/GR00T-N1-2B | fc879581 | gr00t | cross_embodiment | 2 | flow-matching (padded 32)/32/16 | finetune_base | - | NVIDIA open model license (repo LICENSE) (no) | - |
| x-square-robot/wall-oss-fast | 9af477e7 | wall-oss | cross_embodiment | 2 | FAST tokens/32 | finetune_base | - | Apache-2.0 (wall-x GitHub) (no) | - |
| x-square-robot/wall-oss-flow-0.1 | 325f0307 | wall-oss | cross_embodiment | 2 | flow-matching/32 | finetune_base | - | Apache-2.0 (wall-x GitHub) (no) | - |
| x-square-robot/wall-oss-0.5 | b7dc23a8 | wall-oss | cross_embodiment | 2 | flow-matching | finetune_base | - | Apache-2.0 (wall-x GitHub) (no) | - |
| Dexmal/DM0-base | e891caaa | dm0 | cross_embodiment | 2 | padded 32/32/50 | finetune_base | - | cc (card tag) (yes) | - |
| open-gigaai/GigaBrain-0.7-3.5B-Base | 2936d6f2 | gigabrain | cross_embodiment | 2 | padded 32/32/50 | finetune_base | - | apache-2.0 (yes) | - |
| TRI-ML/Foundry-VLA-1.7B-real | 459144b1 | vla-foundry | franka | 2 | bimanual EE relative xyz + rot6d + gripper/20 | finetune_base | - | apache-2.0 (yes) | - |
| sii-research/tau-0-vla | 671ea261 | tau0-vla | cross_embodiment | 2 | padded 40/40/30 | finetune_base | - | apache-2.0 (yes) | - |
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
| openvla/openvla-v01-7b | da55c886 | openvla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| openvla/openvla-7b-finetuned-libero-spatial | 962318ce | openvla | franka | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| openvla/openvla-7b-finetuned-libero-object | 287d6cfd | openvla | franka | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| openvla/openvla-7b-finetuned-libero-goal | fa5ae1e7 | openvla | franka | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| openvla/openvla-7b-finetuned-libero-10 | 80970322 | openvla | franka | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| moojink/openvla-7b-oft-finetuned-libero-spatial | 6d0231af | openvla | franka | 1 | EE delta (xyz, rpy, gripper), continuous L1 regression, parallel decoding/7/8 | no | - | mit (yes) | - |
| moojink/openvla-7b-oft-finetuned-libero-object | 4c89574e | openvla | franka | 1 | EE delta (xyz, rpy, gripper), continuous L1 regression, parallel decoding/7/8 | no | - | mit (yes) | - |
| moojink/openvla-7b-oft-finetuned-libero-goal | c2d0f9fb | openvla | franka | 1 | EE delta (xyz, rpy, gripper), continuous L1 regression, parallel decoding/7/8 | no | - | mit (yes) | - |
| moojink/openvla-7b-oft-finetuned-libero-10 | 95220f9a | openvla | franka | 1 | EE delta (xyz, rpy, gripper), continuous L1 regression, parallel decoding/7/8 | no | - | mit (yes) | - |
| moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 | 638918f3 | openvla | franka | 1 | EE delta (xyz, rpy, gripper), continuous L1 regression, parallel decoding/7/8 | no | - | mit (yes) | - |
| Haozhan72/Openvla-oft-SFT-libero10-trajall | 97d6e58b | openvla | franka | 1 | EE delta (xyz, rpy, gripper) (OFT tokens)/7/8 | no | - | mit (yes) | - |
| gen-robot/openvla-7b-rlvla-rl | 93c54b03 | openvla | widowx | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| Stanford-ILIAD/minivla-vq-libero90-prismatic | b08980cd | minivla | franka | 1 | EE delta (xyz, rpy, gripper), VQ action-chunk tokens/7/8 | no | - | mit (yes) | - |
| furonghuang-lab/tracevla_7b | 23ac599e | tracevla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| furonghuang-lab/tracevla_phi3v | b879e426 | tracevla | cross_embodiment | 1 | EE delta (xyz, rpy, gripper), 256-bin tokens/7/1 | no | - | mit (yes) | - |
| VLA-Adapter/LIBERO-Spatial-Pro | cb023c67 | vla-adapter | franka | 1 | EE delta (xyz, rpy, gripper) continuous/7/8 | no | - | mit (yes) | - |
| VLA-Adapter/LIBERO-Long-Pro | 9e21d603 | vla-adapter | franka | 1 | EE delta (xyz, rpy, gripper) continuous/7/8 | no | - | mit (yes) | - |
| VLA-Adapter/CALVIN-ABC-Pro | 1caaab71 | vla-adapter | franka | 1 | EE delta (xyz, rpy, gripper) continuous/7/8 | no | - | mit (yes) | - |
| IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge | cfb5ff76 | spatialvla | widowx | 1 | EE delta (xyz, rpy, gripper), adaptive spatial action grids/7/4 | no | - | mit (yes) | - |
| IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal | 34e24251 | spatialvla | other | 1 | EE delta (xyz, rpy, gripper), adaptive spatial action grids/7/4 | no | - | mit (yes) | - |
| BeingBeyond/Being-H05-2B_libero_robocasa | 7ad7f91c | being-h05 | franka | 1 | EE/joint per benchmark | no | - | apache-2.0 (yes) | - |
| declare-lab/nora-1.5-libero | 422ccdc7 | nora | franka | 1 | EE delta (xyz, rpy, gripper)/7 | no | - | none declared on card (no) | - |
| declare-lab/nora-long-finetuned-libero-spatial | ef6b673d | nora | franka | 1 | EE delta (xyz, rpy, gripper) FAST tokens/7/5 | no | - | none declared on card (no) | - |
| lerobot/pi0fast-libero | 840f4b50 | pi0 | franka | 1 | FAST tokens -> EE delta/7/10 | no | - | gemma (pi0fast family) (no) | - |
| lerobot/pi0fast-libero-v044 | 2bb8d559 | pi0 | franka | 1 | FAST tokens -> EE delta/7/10 | no | - | gemma (yes) | - |
| 2toINF/X-VLA-Libero | 129e7146 | xvla | franka | 1 | ee6d (xyz + 6D rot + gripper per arm)/10/30 | no | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-WidowX | 8d7ea1aa | xvla | widowx | 1 | ee6d (xyz + 6D rot + gripper per arm)/10/30 | no | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-Google-Robot | afaad7ac | xvla | other | 1 | ee6d (xyz + 6D rot + gripper per arm)/10/30 | no | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-Calvin-ABC_D | d76710ee | xvla | franka | 1 | ee6d (xyz + 6D rot + gripper per arm)/10/30 | no | - | apache-2.0 (yes) | - |
| 2toINF/X-VLA-VLABench | 0995f2f5 | xvla | franka | 1 | ee6d (xyz + 6D rot + gripper per arm)/10/30 | no | - | apache-2.0 (yes) | - |
| lerobot/xvla-libero | 8e6aeef0 | xvla | franka | 1 | ee6d/20/30 | no | - | apache-2.0 (yes) | - |
| lerobot/xvla-widowx | 529ecd54 | xvla | widowx | 1 | ee6d/20/30 | no | - | apache-2.0 (yes) | - |
| lerobot/xvla-google-robot | f60504c9 | xvla | other | 1 | ee6d/20/30 | no | - | apache-2.0 (yes) | - |
| InternRobotics/InternVLA-M1-LIBERO-Long | 67a325f3 | internvla-m1 | franka | 1 | EE delta (xyz, rpy, gripper)/7/16 | no | - | cc-by-nc-sa-4.0 (yes) | - |
| nvidia/GR00T-N1.6-DROID | ae3ebe8d | gr00t | franka | 1 | joint/EE per embodiment/16 | no | - | NVIDIA open model license (see LICENSE) (no) | - |
| nvidia/GR00T-N1.6-bridge | e76ae688 | gr00t | widowx | 1 | joint/EE per embodiment/16 | no | - | NVIDIA open model license (see LICENSE) (no) | - |
| nvidia/GR00T-N1.7-DROID | 113eb584 | gr00t | franka | 1 | joint/EE per embodiment/40 | no | - | NVIDIA open model license (see LICENSE) (yes) | - |
| nvidia/GR00T-N1.7-LIBERO | 65854032 | gr00t | franka | 1 | joint/EE per embodiment/40 | no | - | NVIDIA open model license (see LICENSE) (yes) | - |
| nvidia/SO_ARM_Starter_Gr00tN17 | 93a5a88f | gr00t | so100 | 1 | joint abs/6/40 | no | - | NVIDIA open model license (no) | - |
| anvil-robotics/pi05-openarm-pick-and-place-ikea-red-block | b35d6d41 | pi05 | other | 1 | abs joint/8/50 | no | - | apache-2.0 (yes) | - |
| aHiroakiIshikawa/molmoact2-openarm-dual-cam-sponge-v4-last | 271a7915 | molmoact2 | other | 1 | abs joint/8/30 | no | - | none on card (no) | - |
| Samisaliveagain/smolvla-agilex-piper-hex-bolt-insertion-batch64-step20000 | fac834e1 | smolvla | piper | 1 | abs joint/7/50 | no | - | none on card (no) | - |
| RoboticsCollectiveAachen/act_piper_hexagonal_bolt_insertion | f40cf505 | act | piper | 1 | abs joint/7/100 | no | - | apache-2.0 (yes) | - |
| pengyue-polaron/pi05-galaxea-a1-fruit-placement-eef | 1dc5eecd | pi05 | galaxea | 1 | episode-relative EEF pose/8 | no | - | gemma (yes) | - |

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
| Sichang0621/abc_sort_legos_v21 | bb2dab42 | yam_i2rt | 500 | LeRobot v2.1 | apache-2.0 | 34.08 |
| jellyho/yam_lego_taxi | d9d977d1 | yam_i2rt | 347 | LeRobot v3.0 | apache-2.0 | 4.83 |
| robots123/insert_pin | 4e72d156 | yam_i2rt | 100 | LeRobot v3.0 | apache-2.0 | 3.04 |
| villekuosmanen/insert_actuator_into_camera_mount | 59ec5cd9 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.72 |
| villekuosmanen/assemble_rail | 9f834332 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.99 |
| villekuosmanen/assemble_enpire_gripper | b1256865 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.94 |
| huzheyuan/yam3lego_whole_round1_baseline | ebd803fa | yam_i2rt | 103 | LeRobot v2.1 | apache-2.0 | 34.39 |
| YOLO2431/plug_white_usb_cable_v2 | ddf95648 | yam_i2rt | 500 | LeRobot v2.1 | apache-2.0 | 94.87 |
| YOLO2431/plug-usb-v3.2 | b7276620 | yam_i2rt | 300 | LeRobot v2.1 | apache-2.0 | 36.37 |
| Jiafei1224/22042026-pegboard-tools-combined | a88a3a32 | yam_i2rt | 200 | LeRobot v3.0 | apache-2.0 | 11.19 |
| pyrrosk/yam_three_pipes | 5583cf34 | aloha | 120 | LeRobot v3.0 | apache-2.0 | 40.94 |
| villekuosmanen/insert_gear_in_gripper | 8356f785 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.52 |
| fanqi-robo/insert_actuator_into_camera_mount | b1ede60c | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 1.74 |
| villekuosmanen/assemble_enpire_gripper_new_3 | 16168f50 | yam_i2rt | 20 | LeRobot v3.0 | apache-2.0 | 1.11 |
| YOLO2431/put_pen_into_bag_V3 | 57153cd9 | yam_i2rt | 534 | LeRobot v2.1 | apache-2.0 | 118.3 |
| johannesmichalke/place-pencil-v1 | 2296fd1e | yam_i2rt | 219 | LeRobot v3.0 | mit | 6.02 |
| johannesmichalke/pick-pencil-v8-merged | a613c8a1 | yam_i2rt | 181 | LeRobot v3.0 | apache-2.0 | 1.27 |
| YOLO2431/put_pen_into_bag_V4_new_gripper | c3b1be8e | yam_i2rt | 101 | LeRobot v2.1 | apache-2.0 | 12.07 |
| ETHRC/yams-carton-box-closing-sat-michael-mat-varing-fan-position-25-04-2025 | a0cb7321 | yam_i2rt | 133 | LeRobot v3.0 | apache-2.0 | 0.38 |
| andreaskoepf/dk1_domino_sorting_001 | fabf22d2 | other | 63 | LeRobot v3.0 | apache-2.0 | 4.38 |
| Zasha01/lego_cube_final | e8dd7870 | other | 251 | LeRobot v3.0 | apache-2.0 | 7.21 |
| kkipngenokoech/abc-molmoact2-yam-mix | dadfea01 | yam_i2rt | 21767 | LeRobot v3.0 | apache-2.0 | 1355.98 |
| IPEC-COMMUNITY/fmb_dataset_lerobot | e2c90a49 | franka | 8612 | LeRobot v2.1 | apache-2.0 | 13.34 |
| pyrrosk/longer_insertions_side_grasp_oracle_randomized | a77dcad9 | aloha | 1106 | LeRobot v3.0 | apache-2.0 | 17.53 |
| pyrrosk/assembly_benchmark | d1cc37ce | aloha | 821 | LeRobot v3.0 | apache-2.0 | 18.3 |
| AdityaShah/self_repair_gripper_screwdriver_bc | ed47f81c | yam_i2rt | 800 | LeRobot v2.1 | cc-by-4.0 | 4.45 |
| Travor278/evorl-piperx-copper-screw-attempt-aware-clean-v2 | d8c2c0f0 | piper | 482 | LeRobot v3.0 | other | 14.63 |
| thewisp/cylinder_socket_merged_head | 4d818b10 | other | 471 | LeRobot v3.0 | apache-2.0 | 13.57 |
| ZebTaylor/panthera-gear-carrier-pin-retimed-restyled-20hz | 83162f95 | other | 400 | LeRobot v3.0 | none on card | 7.75 |
| Sichang0621/yam-vital-left-hand | fd019de7 | yam_i2rt | 350 | LeRobot v3.0 | apache-2.0 | 0.23 |
| ZebTaylor/panthera-gear-carrier-pin-real-20hz | ff6c6852 | other | 317 | LeRobot v3.0 | none on card | 104.48 |
| lyl472324464/insert_random_shaft_random_medium_gear | af3e272f | aloha | 300 | LeRobot v3.0 | apache-2.0 | 3.51 |
| lyl472324464/insert_random_shaft_random_small_gear | d040bfbc | aloha | 300 | LeRobot v3.0 | apache-2.0 | 3.54 |
| ZebTaylor/panthera-gear-carrier-pin-retimed-real-20hz | cae3a648 | other | 255 | LeRobot v3.0 | none on card | 9.39 |
| Jiafei1224/pegboard-tools-combined | d89d3764 | yam_i2rt | 200 | LeRobot v3.0 | apache-2.0 | 11.19 |
| J-joon/espada_aloha_insertion | a9332789 | other | 200 | LeRobot v3.0 | apache-2.0 | 0.18 |
| pyrrosk/yam_three_pipes_granular | 8578d86b | aloha | 120 | LeRobot v3.0 | apache-2.0 | 47.8 |
| knpt/yam_bimanual_tool_rack | f54bf2cc | yam_i2rt | 66 | LeRobot v3.0 | mit | 3.13 |
| allenai/eval_molmoact_pegboard_in-distribution | 1817233f | yam_i2rt | 52 | LeRobot v3.0 | apache-2.0 | 2.71 |
| fanqi-robo/eval_insert_gear_in_gripper_molmoact2_yam_zeroshot | 87a10be4 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 3.01 |
| fanqi-robo/eval_insert_actuator_into_camera_mount_molmoact2_yam_fft_step20000 | 8867791f | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 4.19 |
| armnet/eval_assemble_enpire_gripper_molmoact2_fft_step20000_ep1to50 | f30f6b89 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 3.75 |
| armnet/eval_insert_gear_in_gripper_pi05_b0_step20000_ep1to50 | cb528c96 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 3.1 |
| armnet/eval_assemble_enpire_gripper_pi05_fft_step20000_50eps | 6327ae77 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 4.83 |
| armnet/eval_assemble_rail_molmoact2_fft_step20000_ep1to38 | a731054e | yam_i2rt | 38 | LeRobot v3.0 | apache-2.0 | 2.87 |
| armnet/eval_assemble_rail_pi05_fft_step20000_ep1to35 | 98a89805 | yam_i2rt | 35 | LeRobot v3.0 | apache-2.0 | 2.54 |
| fanqi-robo/eval_insert_gear_in_gripper_pi0_lr1x_s1000_step20000 | a4bc4090 | yam_i2rt | 33 | LeRobot v3.0 | apache-2.0 | 2.52 |
| armnet/eval_insert_gear_in_gripper_molmoact2_yam_fft_step20000_ep21to50 | f5905056 | yam_i2rt | 30 | LeRobot v3.0 | apache-2.0 | 2.51 |
| fanqi-robo/eval_insert_actuator_into_camera_mount_pi05_fft_step20000_ep20to50 | 3b0f628d | yam_i2rt | 30 | LeRobot v3.0 | apache-2.0 | 2.34 |
| ml-resources/Agilex_Split_Aloha_take_out_a_pen_from_the_pen_holder | dd81f3a1 | other | 488 | LeRobot v2.1 | apache-2.0 | 7.0 |
| imumtozee/aloha_mobile_uncap_pen_all | 196d0db8 | aloha | 276 | LeRobot v2.0 | apache-2.0 | 56.0 |
| vo2yager/pen_uncap_all | d41ce6cd | aloha | 250 | LeRobot v2.0 | apache-2.0 | 48.67 |
| johannesmichalke/pick-pencil-v5 | 9461ed18 | yam_i2rt | 192 | LeRobot v3.0 | mit | 2.05 |
| shobbs/aloha_pen_uncap_diversePI | 5e1faa5f | aloha | 123 | LeRobot v2.0 | apache-2.0 | 72.11 |
| sarayu-li/grip_pen0711_lerobot | c13fc91a | aloha | 120 | LeRobot v2.1 | apache-2.0 | 30.18 |
| aksako/Task_30_place_marker_in_bowl_v30 | 514f0d74 | aloha | 103 | LeRobot v3.0 | apache-2.0 | 1.12 |
| vo2yager/pen_uncap_all_filtered | 77a11641 | aloha | 101 | LeRobot v2.0 | apache-2.0 | 21.01 |
| lyl472324464/place_soldering_iron_and_solder_on_marker | 2ba20da0 | aloha | 100 | LeRobot v3.0 | apache-2.0 | 3.36 |
| lyl472324464/place_soldering_iron_on_marker | dd9d38d5 | aloha | 100 | LeRobot v3.0 | apache-2.0 | 3.44 |
| vo2yager/pen_incontext | d50a7ffe | aloha | 100 | LeRobot v2.0 | apache-2.0 | 22.98 |
| reece-omahoney/remove-pen-lid-2 | 6ab5b163 | piper | 100 | LeRobot v3.0 | apache-2.0 | 0.31 |
| johannesmichalke/sharpen-pencil | 84f781ce | yam_i2rt | 84 | LeRobot v3.0 | mit | 1.35 |
| YOLO2431/put_pen_v2.1 | 6adcece0 | yam_i2rt | 80 | LeRobot v2.1 | apache-2.0 | 66.74 |
| mliu48/telearms-yam-cube-v5-realistic | 052e5550 | yam_i2rt | 865180 | LeRobot v3.0 | apache-2.0 | 2159.99 |
| mliu48/telearms-yam-cube-v4-dr | 71e209ea | yam_i2rt | 213990 | LeRobot v3.0 | apache-2.0 | 749.07 |
| mliu48/telearms-yam-cube-v5-front-to-back-cube-line | b8207cef | yam_i2rt | 79248 | LeRobot v3.0 | apache-2.0 | 159.78 |
| mliu48/telearms-yam-cube-v4 | fa8f65ed | yam_i2rt | 62121 | LeRobot v3.0 | apache-2.0 | 317.85 |
| mliu48/telearms-yam-cube-v5-ablation-mixed | 20dca673 | yam_i2rt | 60093 | LeRobot v3.0 | apache-2.0 | 148.34 |
| nuffnuff/robocoin-yam-v3-269x500 | 0c2f1885 | yam_i2rt | 46173 | LeRobot v3.0 | apache-2.0 | 596.08 |
| albagon/telearms-yam-red-left-blue-rig-dr-nolip | cfc0cde8 | yam_i2rt | 12130 | LeRobot v3.0 | apache-2.0 | 11.88 |
| albagon/telearms-yam-red-left-blue-v6-rig-dr-v7 | 1345dcc8 | yam_i2rt | 9842 | LeRobot v3.0 | apache-2.0 | 7.94 |
| nuffnuff/yam-master-v1-strong-aug | 647337c1 | yam_i2rt | 5995 | LeRobot v3.0 | apache-2.0 | 129.94 |
| andlyu/Public-YAM-runs | ef5e9230 | yam_i2rt | 607 | LeRobot v2.1 | none on card | 183.74 |
| axiboai/piper_stacking_abc_mix_balanced | 534a8e17 | piper | 480 | LeRobot v2.1 | apache-2.0 | 22.94 |
| Dimios45/yam-pick-duster-200 | c5605732 | yam_i2rt | 200 | LeRobot v3.0 | apache-2.0 | 1.01 |
| brandonyang/yam-vive-teleop | 38fc0d95 | yam_i2rt | 80 | LeRobot v3.0 | apache-2.0 | 0.24 |
| brandonyang/yam-ultrawide-teleop | e0fae691 | yam_i2rt | 75 | LeRobot v3.0 | apache-2.0 | 0.65 |
| ETHRC/yams-carton-box-closing-combined | 9c65ebac | yam_i2rt | 72 | LeRobot v3.0 | apache-2.0 | 0.21 |
| HCIS-Lab/bi-yam-toy-tape | 902fe932 | yam_i2rt | 60 | LeRobot v3.0 | apache-2.0 | 1.56 |
| ETHRC/yams-carton-box-closing-wed-tom-elias | 0ff4253e | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 0.2 |
| hcisbmm/yam-pink-utensil-into-cup_20260707_180338 | b408d333 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 0.64 |
| Archaive16/yam-two-cube-place-01-clean | f08b5018 | yam_i2rt | 48 | LeRobot v3.0 | none on card | 0.89 |
| zekaili/yam-cube-layer-turn | e0afe40d | yam_i2rt | 46 | LeRobot v3.0 | apache-2.0 | 5.48 |
| atharva-pantheon/yam-stack-cube | 3ac38351 | yam_i2rt | 44 | LeRobot v3.0 | apache-2.0 | 0.69 |
| HGLLL/bimanual-yam-0318 | b84670ad | yam_i2rt | 41 | LeRobot v3.0 | apache-2.0 | 1.0 |
| ETHRC/yams-carton-box-closing-noe-exploring-30-04-2026 | db7d4d24 | yam_i2rt | 22 | LeRobot v3.0 | apache-2.0 | 0.04 |
| ETHRC/yams-carton-box-closing-tue-tom-mat-2 | 3c266fb8 | yam_i2rt | 20 | LeRobot v3.0 | apache-2.0 | 0.08 |
| HGLLL/bimanual-yam-0318-fix | 06b6d8ec | yam_i2rt | 20 | LeRobot v3.0 | apache-2.0 | 1.0 |
| knpt/yam_bimanual_tomato | e9561e42 | yam_i2rt | 20 | LeRobot v3.0 | mit | 0.31 |
| kabilanKB/yam_put_pot | cf88bb96 | yam_i2rt | 20 | LeRobot v2.0 | apache-2.0 | 0.02 |
| gsethia08/part1-arrange_topple_dominoes | 2c505fb7 | other | 646 | LeRobot v3.0 | apache-2.0 | 8.27 |
| AnonymousMouse404/DominoPNP | 33af68d6 | so100 | 102 | LeRobot v3.0 | apache-2.0 | 0.88 |
| AnonymousMouse404/DominoRGB | ecbc8db5 | so100 | 101 | LeRobot v3.0 | apache-2.0 | 0.85 |
| AnonymousMouse404/Green_domino_20260629_144632 | cb31af6a | so100 | 35 | LeRobot v3.0 | apache-2.0 | 0.29 |
| AnonymousMouse404/Red_domino2_20260629_140019 | f63c7c4e | so100 | 25 | LeRobot v3.0 | apache-2.0 | 0.22 |
| aivanni/so101-puzzle-v8 | 9fe84958 | so100 | 164 | LeRobot v3.0 | apache-2.0 | 5.51 |
| ni25y/puzzlemate-pick-flip-7d-105epi | 7f4fc3ee | so100 | 105 | LeRobot v3.0 | apache-2.0 | 0.13 |
| ni25y/puzzlemate_6d | daed50f5 | so100 | 105 | LeRobot v3.0 | apache-2.0 |  |
| ni25y/puzzlemate-pick-flip | fe1d50e4 | so100 | 50 | LeRobot v3.0 | apache-2.0 | 0.2 |
| ni25y/puzzlemate-picknflip | c3fdf6d1 | so100 | 50 | LeRobot v3.0 | apache-2.0 | 0.04 |
| ni25y/puzzlemate-pick-flip-7d-2 | f80030e8 | so100 | 50 | LeRobot v3.0 | apache-2.0 | 0.06 |
| gsethia08/part1-assemble_disassemble_legos | b91d6c47 | other | 3653 | LeRobot v3.0 | apache-2.0 | 9.93 |
| windfromthenorth/extreme_randomization_6_brick_03 | e9ca20c0 | other | 2108 | LeRobot v2.1 | apache-2.0 | 1.88 |
| mrkschtr/imperio_lego | 25dbed88 | so100 | 1854 | LeRobot v3.0 | other | 0.18 |
| gsethia08/part1-build_unstack_lego | e1df77e4 | other | 1206 | LeRobot v3.0 | apache-2.0 | 7.06 |
| ajaysri/lego_stack_openpi_oracle_success_dot_goalimage_lerobot_v3_1000 | 2a7852e1 | other | 1000 | LeRobot v3.0 | apache-2.0 | 2.39 |
| ajaysri/lego_mimicgen_5_6block_dot_subgoal_lerobot_v3_1000 | 1bb6e760 | other | 1000 | LeRobot v3.0 | apache-2.0 | 3.59 |
| JerryYuu/brickcraft | a3668d5c | other | 865 | LeRobot v3.0 | cc-by-4.0 | 0.59 |
| gsethia08/part1-assemble_disassemble_soft_legos | 54d665ef | other | 704 | LeRobot v3.0 | apache-2.0 | 7.12 |
| ssdunit/abc_sim_lego_blocks_sorting | d09e52dc | other | 389 | LeRobot v3.0 | apache-2.0 | 9.46 |
| airshop/Lego_task_complex | fcf75eab | so100 | 70 | LeRobot v3.0 | apache-2.0 | 4.11 |
| zhanghit/merged_bottle_lego_language | 8896336a | franka | 63 | LeRobot v3.0 | apache-2.0 | 0.4 |
| love3165303/so100_pick_lego1 | 9ad43561 | so100 | 54 | LeRobot v2.1 | apache-2.0 | 0.35 |
| LeRobot-worldwide-hackathon/160-LegoBot-lego_pixelart_stage_2_v1 | 71e67b67 | so100 | 53 | LeRobot v2.1 | apache-2.0 | 1.22 |
| Zasha01/lego_cube | 5e2fcc80 | other | 25 | LeRobot v3.0 | apache-2.0 | 1.27 |
| villekuosmanen/agilex_stack_letter_brick | 6548c089 | other | 20 | LeRobot v2.1 | apache-2.0 | 0.13 |
| pravsels/BimanualYAM-PackFoldedClothes | c401a98f | yam_i2rt | 469 | LeRobot v3.0 | apache-2.0 | 36.25 |
| nuffnuff/yam-pour-combined | 466523c5 | yam_i2rt | 217 | LeRobot v3.0 | none on card | 3.07 |
| jellyho/yam_cable_tie | 9dc7a2d9 | yam_i2rt | 160 | LeRobot v3.0 | apache-2.0 | 1.24 |
| vikram-avea/yam-physical-ai-hack-mega-640x480 | e104f40c | yam_i2rt | 108 | LeRobot v3.0 | none on card | 1.23 |
| robots123/yam-espresso | b2210866 | yam_i2rt | 91 | LeRobot v3.0 | apache-2.0 | 3.18 |
| allenai/28112025-yam-01 | 7a889d4f | yam_i2rt | 53 | LeRobot v3.0 | apache-2.0 | 5.67 |
| allenai/24112025-yam-01 | a5a42115 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 2.98 |
| allenai/25112025-yam-02 | e941ab81 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 5.49 |
| allenai/26112025-yam-02 | af9d23ca | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 4.95 |
| allenai/27112025-yam-03 | 13f9c267 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 4.39 |
| HCIS-Lab/bi-yam-loosen-bottle | 25be0043 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 2.47 |
| leokswang/YAM_lerobot_format | 990006b3 | yam_i2rt | 50 | LeRobot v2.1 | apache-2.0 | 75.26 |
| Dimios45/pi05-molmoact2-yam-autonmous-buffer | 4745f920 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 12.62 |
| dreamscale-labs/yam-fold-cloth | 79e4c98e | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 2.98 |
| vikram-avea/yam-physical-ai-hack-640x480 | 23c21ac3 | yam_i2rt | 47 | LeRobot v3.0 | none on card | 0.49 |
| Dimios45/pi05-molmoact2-yam-autonomous | f3668c5a | yam_i2rt | 25 | LeRobot v3.0 | apache-2.0 | 6.15 |
| ozayash/yam_so101_teleop | 2263e640 | yam_i2rt | 22 | LeRobot v3.0 | apache-2.0 | 0.28 |
| lerobot/stanford_kuka_multimodal_dataset | 93927d10 | other | 3000 | LeRobot v3.0 | mit | 0.59 |
| lerobot/fmb | 76c86f11 | other | 1804 | LeRobot v3.0 | apache-2.0 | 9.1 |
| lerobot/iamlab_cmu_pickup_insert | 5806e104 | other | 631 | LeRobot v3.0 | apache-2.0 | 3.88 |
| ajaysri/long_cable_insertion_2_subtasks_3cam_combined | 6f68ff7d | other | 501 | LeRobot v3.0 | apache-2.0 | 0.81 |
| Takizz/evorl-piperx-copper-screw-3h-demo-1h-intervention-clean | 80a64c41 | piper | 410 | LeRobot v3.0 | none on card | 7.46 |
| YOLO2431/plug-usb-dagger-rt-v1.1 | ccc13346 | yam_i2rt | 390 | LeRobot v2.1 | apache-2.0 | 11.18 |
| noisyduck/alohasim_insertion_downsampled_ours_2_4_final | d87a864f | aloha | 200 | LeRobot v3.0 | mit | 0.14 |
| kd-forge/aloha_sim_transfer_cube_insertion | 81c92b01 | aloha | 200 | LeRobot v3.0 | apache-2.0 | 0.3 |
| YOLO2431/plug-usb-dagger-rt-maniflow-v1 | eaf52c87 | yam_i2rt | 194 | LeRobot v2.1 | apache-2.0 | 6.58 |
| ajaysri/long_cable_insertion_2_center | 0194b97f | other | 177 | LeRobot v3.0 | apache-2.0 | 0.28 |
| Libra-TeV/Libra_bimanual_fairino_plug_in_container_merged | 44f7e4ba | other | 165 | LeRobot v3.0 | apache-2.0 | 10.53 |
| YOLO2431/plug-usb-new-gripper-r1-dagger-with-policy | 713e485a | yam_i2rt | 164 | LeRobot v2.1 | apache-2.0 | 20.8 |
| YOLO2431/plug-usb-dagger-v2 | d71f1a7b | yam_i2rt | 137 | LeRobot v2.1 | apache-2.0 | 3.94 |
| YOLO2431/plug-white-usb-cable-v1.2-dagger-r1-with-policy | 0ee6fcbc | yam_i2rt | 100 | LeRobot v2.1 | apache-2.0 | 12.76 |
| YOLO2431/plug-white-usb-cable-v2-pi05-r1-dagger-with-policy | 22dfb6a3 | yam_i2rt | 79 | LeRobot v2.1 | apache-2.0 | 10.79 |
| lerobot/aloha_sim_insertion_scripted | 8ab66091 | aloha | 50 | LeRobot v3.0 | mit | 3.11 |
| lerobot/aloha_static_battery | 06dc3da8 | other | 49 | LeRobot v3.0 | apache-2.0 | 3.05 |
| Codingpae/openarm_sim_pen_dr1000 | ec264c88 | other | 1000 | LeRobot v3.0 | apache-2.0 | 23.82 |
| lerobot/berkeley_autolab_ur5 | c4e26a69 | other | 1000 | LeRobot v3.0 | cc-by-4.0 | 31.0 |
| s-choi/biplay_aloha_pen_uncap_diverse_dataset_merge_ready_v3 | b13d2846 | aloha | 199 | LeRobot v3.0 | apache-2.0 | 0.48 |
| YOLO2431/put-pen-into-bag-v2-pi05-r1-dagger-with-policy | 0a3adafa | yam_i2rt | 129 | LeRobot v2.1 | apache-2.0 | 33.43 |
| YOLO2431/put-pen-into-bag-v4.2-maniflow-r1-dagger-with-policy | 808204a6 | yam_i2rt | 102 | LeRobot v2.1 | apache-2.0 | 14.91 |
| s-choi/biplay_aloha_pick_place_dataset_merge_ready_v3 | a2b56197 | aloha | 2589 | LeRobot v3.0 | apache-2.0 | 2.8 |
| Autobrik/bi_piper_pick_and_place_fruits_mantra | 7e3f5337 | piper | 2301 | LeRobot v3.0 | apache-2.0 | 12.57 |
| puheliang/dual_arm_collaborative_express_parcel_sorting_20260525_20260528_all_clean_merged_748episodes | 3ffa8fa4 | other | 748 | LeRobot v3.0 | apache-2.0 | 5.05 |
| puheliang/dual_arm_collaborative_express_parcel_sorting_20260525_20260528_all_clean_merged_747episodes | f18c7eb6 | other | 747 | LeRobot v3.0 | apache-2.0 | 5.12 |
| axiboai/piper_stacking_folding_v1 | 6ed960df | piper | 500 | LeRobot v2.1 | apache-2.0 | 1.03 |
| csjmacmi/piperx_pick_and_place_v1_4 | 15c8bb6e | piper | 301 | LeRobot v3.0 | apache-2.0 | 8.14 |
| axiboai/piper_stacking_abc_mix | 4a1f2cb3 | piper | 300 | LeRobot v2.1 | apache-2.0 | 22.93 |
| wego-mink/dualarm_sort_apple_stage | 549767d6 | piper | 300 | LeRobot v3.0 | none on card | 0.15 |
| Shiki42/piperx-sortletter-20260921-140ep-ctr-2x | 9519c57b | piper | 280 | LeRobot v3.0 | none on card | 1.43 |
| unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2 | 8db6251e | other | 254 | LeRobot v3.0 | apache-2.0 | 9.22 |
| J-joon/espada_aloha_cube | 5d1bc0af | other | 200 | LeRobot v3.0 | apache-2.0 | 0.12 |
| Anlorla/push_block_dual_lerobot30 | 6cf17c25 | other | 200 | LeRobot v3.0 | apache-2.0 | 0.15 |
| puheliang/dual_arm_collaborative_express_parcel_sorting_20260525_all_merged | f60bfb6f | other | 197 | LeRobot v3.0 | apache-2.0 | 1.57 |
| murobotics/tblock-all-piper-clean-piper-joints | 34945ab5 | piper | 194 | LeRobot v3.0 | other | 0.22 |
| kewalk123/yam_bean_scoop_0905_v5 | a5cea20e | yam_i2rt | 89 | LeRobot v2.1 | cc-by-4.0 | 56.19 |
| dentronic/grasp_minigold_yam1 | 3dcf418c | yam_i2rt | 63 | LeRobot v3.0 | apache-2.0 | 1.29 |
| Rtop-YAM/pick-and-place-combined-dagger-v2 | 5bf40d5a | yam_i2rt | 62 | LeRobot v3.0 | apache-2.0 | 0.85 |
| vikram-avea/yam-block-sort-1 | edcf9b57 | yam_i2rt | 56 | LeRobot v3.0 | none on card | 1.16 |
| Dimios45/yam_duster_in_box | 61fbee0f | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 0.55 |
| armnet/eval_take_actuator_from_box_molmoact2_fft_step20000_ep1to50 | fc103441 | yam_i2rt | 50 | LeRobot v3.0 | apache-2.0 | 3.51 |
| Rtop-YAM/rollout_pick_and_place_dagger_20260722_155012 | 46b3cb53 | yam_i2rt | 20 | LeRobot v3.0 | apache-2.0 | 3.22 |
| Marco10pez/Build_lego_simple | 12c0c93e | so100 | 25 | LeRobot v3.0 | apache-2.0 | 1.17 |
| lerobot/droid_1.0.1 | 71da9dcb | franka | 95658 | LeRobot v3.0 | apache-2.0 | 824.93 |
| IPEC-COMMUNITY/droid_lerobot | 96dd57f3 | franka | 92233 | LeRobot v2.0 | apache-2.0 | 392.06 |
| nuffnuff/yam-pour-v5 | 162df99c | yam_i2rt | 150 | LeRobot v3.0 | apache-2.0 | 0.83 |
| pengyue-polaron/nyush-galaxea-a1-relation | a34e1df4 | other | 210 | LeRobot v2.1 | none on card | 1.83 |
| sortingroboticsinc/individual_grasp_collect_ibvs_axial174 | b9c6eb3e | other | 207 | LeRobot v3.0 | apache-2.0 | 0.49 |
| pengyue-polaron/nyush-galaxea-a1-fruit-placement | fe7dfced | other | 130 | LeRobot v2.1 | none on card | 5.63 |
| pengyue-polaron/nyush-galaxea-a1-fruit-and-blocks | 2917ded3 | other | 100 | LeRobot v2.1 | none on card | 0.31 |
| pengyue-polaron/nyush-galaxea-a1-kitchen | 31035cec | other | 100 | LeRobot v2.1 | none on card | 1.91 |
