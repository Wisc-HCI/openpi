# Spatial-legibility robustness checks

| Run | Action dims | Samples x seeds | Flow-time range | Train acc. | Test acc. | Test AUC | Test/train energy | Outcome agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | 8 | 8x3 | [0.3, 0.7] | 0.969 | 0.375 | 0.507 | 16.9x | 1.000 |
| joint_only | 7 | 8x3 | [0.3, 0.7] | 0.969 | 0.375 | 0.505 | 15.7x | 1.000 |
| wide_tau | 8 | 24x1 | [0.1, 0.9] | 0.969 | 0.500 | 0.565 | 20.9x | 0.875 |

`joint_only` excludes the gripper action from the residual. `wide_tau` uses the same total of 24 noise draws
per chunk as the primary run but spreads them across flow time [0.1, 0.9]. Outcome agreement compares each
held-out trajectory's final correct/incorrect classification with the primary run.

The principal conclusion is stable: all runs retain 31/32 accuracy on the fine-tuning trajectories, while
held-out performance remains 3/8 to 4/8 and absolute residual energy is at least 15.7x the fine-tuning level.
The wider flow-time range recovers one held-out trajectory, but does not establish reliable generalization.
