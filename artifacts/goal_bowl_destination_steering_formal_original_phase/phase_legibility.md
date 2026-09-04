# Bowl-destination phase-aligned legibility

Lift/transport onset is the first three consecutive states after the close command with bowl height at least 5.0 mm above its initial value.

Evidence is measured from bowl-lift onset. Positive values indicate relative progress toward the true destination; negative values indicate motion toward the competing destination.

| Method | Target | Success | W close | W grasp | W lift | Pre-lift path (m) | Pre-lift TV | AUC20 | Wrong first 15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | cabinet | 100/100 | 0.000 | 0.000 | 0.000 | 0.4359 | 7.235 | 0.2816 | 0.000 |
| base | plate | 99/100 | 0.000 | 0.000 | 0.000 | 0.4750 | 8.197 | 0.0054 | 0.639 |
| time_decay | cabinet | 96/100 | 0.579 | 0.392 | 0.360 | 0.4475 | 9.164 | 0.3130 | 0.000 |
| time_decay | plate | 99/100 | 0.462 | 0.350 | 0.297 | 0.5106 | 9.094 | 0.0537 | 0.456 |
| belief | cabinet | 97/100 | 0.027 | 0.002 | 0.000 | 0.4406 | 7.822 | 0.2821 | 0.000 |
| belief | plate | 99/100 | 0.018 | 0.001 | 0.000 | 0.4826 | 8.185 | -0.0026 | 0.670 |

## Belief phase profile

| Target | b+ close | b+ grasp | b+ lift | W grasp+1 | W grasp+2 | W grasp+3 |
|---|---:|---:|---:|---:|---:|---:|
| cabinet | 0.973 | 0.998 | 1.000 | 0.001 | 0.000 | 0.000 |
| plate | 0.982 | 0.999 | 1.000 | 0.000 | 0.000 | 0.000 |

## Paired deltas on jointly successful episodes

Deltas are method minus reference. Intervals resample the 50 init states as clusters.

| Comparison | Target | Metric | Pairs | Delta [95% CI] |
|---|---|---|---:|---:|
| time_decay - base | cabinet | pre_lift_eef_path_m | 96 | +0.0120 [-0.0042, +0.0308] |
| time_decay - base | cabinet | pre_lift_action_tv_l2 | 96 | +1.8896 [+0.8461, +3.1032] |
| time_decay - base | cabinet | post_lift_early_auc20 | 96 | +0.0301 [+0.0215, +0.0390] |
| time_decay - base | cabinet | post_lift_wrong_fraction15 | 96 | +0.0000 [+0.0000, +0.0000] |
| time_decay - base | plate | pre_lift_eef_path_m | 94 | +0.0360 [+0.0184, +0.0530] |
| time_decay - base | plate | pre_lift_action_tv_l2 | 94 | +0.8247 [+0.0132, +1.4752] |
| time_decay - base | plate | post_lift_early_auc20 | 94 | +0.0485 [+0.0376, +0.0588] |
| time_decay - base | plate | post_lift_wrong_fraction15 | 94 | -0.1850 [-0.2259, -0.1449] |
| belief - base | cabinet | pre_lift_eef_path_m | 97 | +0.0054 [-0.0050, +0.0182] |
| belief - base | cabinet | pre_lift_action_tv_l2 | 97 | +0.6503 [+0.0875, +1.3221] |
| belief - base | cabinet | post_lift_early_auc20 | 97 | +0.0003 [-0.0042, +0.0050] |
| belief - base | cabinet | post_lift_wrong_fraction15 | 97 | +0.0000 [+0.0000, +0.0000] |
| belief - base | plate | pre_lift_eef_path_m | 98 | +0.0048 [-0.0078, +0.0122] |
| belief - base | plate | pre_lift_action_tv_l2 | 98 | -0.1540 [-0.7805, +0.1882] |
| belief - base | plate | post_lift_early_auc20 | 98 | -0.0088 [-0.0182, +0.0005] |
| belief - base | plate | post_lift_wrong_fraction15 | 98 | +0.0333 [-0.0000, +0.0667] |
| belief - time_decay | cabinet | pre_lift_eef_path_m | 93 | -0.0063 [-0.0291, +0.0142] |
| belief - time_decay | cabinet | pre_lift_action_tv_l2 | 93 | -1.2835 [-2.4237, -0.2642] |
| belief - time_decay | cabinet | post_lift_early_auc20 | 93 | -0.0298 [-0.0366, -0.0227] |
| belief - time_decay | cabinet | post_lift_wrong_fraction15 | 93 | +0.0000 [+0.0000, +0.0000] |
| belief - time_decay | plate | pre_lift_eef_path_m | 94 | -0.0311 [-0.0451, -0.0220] |
| belief - time_decay | plate | pre_lift_action_tv_l2 | 94 | -0.9837 [-1.4344, -0.6968] |
| belief - time_decay | plate | post_lift_early_auc20 | 94 | -0.0563 [-0.0686, -0.0436] |
| belief - time_decay | plate | post_lift_wrong_fraction15 | 94 | +0.2136 [+0.1701, +0.2558] |
