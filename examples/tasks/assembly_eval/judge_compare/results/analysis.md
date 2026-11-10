
### default view, gripper force 0.1 (main set): 25 units, 11 succeeded

| judge | n | accuracy | success recall | failure recall | false done | AUROC |
|---|---|---|---|---|---|---|
| RynnValue-4b first_last: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.34 |
| RynnValue-4b first_last: remaining <= 2 s | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.34 |
| RynnValue-4b full: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.46 |
| RynnValue-4b full: remaining <= 2 s | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.46 |
| RynnValue-8b first_last: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.49 |
| RynnValue-8b first_last: remaining <= 2 s | 25 | 0.60 | 0.09 | 1.00 | 0 | 0.49 |
| RynnValue-8b full: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.75 |
| RynnValue-8b full: remaining <= 2 s | 25 | 0.68 | 0.27 | 1.00 | 0 | 0.75 |
| DenseReward last frame >= 0.85 (#124 done) | 25 | 0.48 | 0.09 | 0.79 | 3 | 0.45 |
| detecting-subtask-success (flash) | 25 | 0.52 | 0.91 | 0.21 | 11 |  |
| judging-step-with-before-now-vlm (flash) | 25 | 0.56 | 1.00 | 0.21 | 11 |  |
| detecting-subtask-success (er2) | 25 | 0.52 | 0.82 | 0.29 | 10 |  |
| judging-step-with-before-now-vlm (er2) | 25 | 0.60 | 1.00 | 0.29 | 10 |  |

### same episodes, opposite camera: 25 units, 11 succeeded

| judge | n | accuracy | success recall | failure recall | false done | AUROC |
|---|---|---|---|---|---|---|
| RynnValue-4b first_last: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.32 |
| RynnValue-4b first_last: remaining <= 2 s | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.32 |
| RynnValue-4b full: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.54 |
| RynnValue-4b full: remaining <= 2 s | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.54 |
| RynnValue-8b first_last: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.64 |
| RynnValue-8b first_last: remaining <= 2 s | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.64 |
| RynnValue-8b full: Success=Yes | 25 | 0.56 | 0.00 | 1.00 | 0 | 0.92 |
| RynnValue-8b full: remaining <= 2 s | 25 | 0.76 | 0.45 | 1.00 | 0 | 0.92 |
| DenseReward last frame >= 0.85 (#124 done) | 25 | 0.56 | 0.18 | 0.86 | 2 | 0.55 |
| detecting-subtask-success (flash) | 25 | 0.56 | 1.00 | 0.21 | 11 |  |
| judging-step-with-before-now-vlm (flash) | 25 | 0.56 | 1.00 | 0.21 | 11 |  |
| detecting-subtask-success (er2) | 25 | 0.64 | 1.00 | 0.36 | 9 |  |
| judging-step-with-before-now-vlm (er2) | 25 | 0.60 | 1.00 | 0.29 | 10 |  |

### gripper force 1.0 (more grasp failures): 25 units, 3 succeeded

| judge | n | accuracy | success recall | failure recall | false done | AUROC |
|---|---|---|---|---|---|---|
| RynnValue-4b first_last: Success=Yes | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.59 |
| RynnValue-4b first_last: remaining <= 2 s | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.59 |
| RynnValue-4b full: Success=Yes | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.89 |
| RynnValue-4b full: remaining <= 2 s | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.89 |
| RynnValue-8b first_last: Success=Yes | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.73 |
| RynnValue-8b first_last: remaining <= 2 s | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.73 |
| RynnValue-8b full: Success=Yes | 25 | 0.88 | 0.00 | 1.00 | 0 | 0.94 |
| RynnValue-8b full: remaining <= 2 s | 25 | 0.84 | 0.00 | 0.95 | 1 | 0.94 |
| DenseReward last frame >= 0.85 (#124 done) | 25 | 0.36 | 0.00 | 0.41 | 13 | 0.14 |
| detecting-subtask-success (flash) | 25 | 0.80 | 1.00 | 0.77 | 5 |  |
| judging-step-with-before-now-vlm (flash) | 25 | 0.68 | 1.00 | 0.64 | 8 |  |
| detecting-subtask-success (er2) | 25 | 0.84 | 0.67 | 0.86 | 3 |  |
| judging-step-with-before-now-vlm (er2) | 25 | 0.80 | 1.00 | 0.77 | 5 |  |

### failures by type (all 75 units): judges that called them done

| failure | n | RynnValue-4b first_last: Success=Yes | RynnValue-4b first_last: remaining <= 2 s | RynnValue-4b full: Success=Yes | RynnValue-4b full: remaining <= 2 s | RynnValue-8b first_last: Success=Yes | RynnValue-8b first_last: remaining <= 2 s | RynnValue-8b full: Success=Yes | RynnValue-8b full: remaining <= 2 s | DenseReward last frame >= 0.85 (#124 done) | detecting-subtask-success (flash) | judging-step-with-before-now-vlm (flash) | detecting-subtask-success (er2) | judging-step-with-before-now-vlm (er2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grasp_lost | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| grasp_missed | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| lego.insufficient_support | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 3 | 0 | 2 |
| lego.not_seated | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 | 19 | 19 | 13 | 14 |
| lego.partially_seated | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 3 | 3 | 3 |
| lego.structure_damaged | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 6 | 6 |
