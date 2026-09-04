# Bowl-destination phase-aligned legibility

Lift/transport onset is the first three consecutive states after the close command with bowl height at least 5.0 mm above its initial value.

Evidence is measured from bowl-lift onset. Positive values indicate relative progress toward the true destination; negative values indicate motion toward the competing destination.

| Method | Target | Success | W close | W grasp | W lift | Pre-lift path (m) | Pre-lift TV | AUC20 | Wrong first 15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | cabinet | 100/100 | 0.000 | 0.000 | 0.000 | 0.4359 | 7.235 | 0.2816 | 0.000 |
| base | plate | 99/100 | 0.000 | 0.000 | 0.000 | 0.4750 | 8.197 | 0.0054 | 0.639 |
| time_decay | cabinet | 96/100 | 0.579 | 0.392 | 0.360 | 0.4475 | 9.164 | 0.3130 | 0.000 |
| time_decay | plate | 99/100 | 0.462 | 0.350 | 0.297 | 0.5106 | 9.094 | 0.0537 | 0.456 |
| belief | cabinet | 94/100 | 0.653 | 0.344 | 0.290 | 0.4603 | 9.883 | 0.3016 | 0.000 |
| belief | plate | 100/100 | 0.549 | 0.373 | 0.230 | 0.5144 | 9.498 | 0.0518 | 0.466 |

## Belief phase profile

| Target | b+ close | b+ grasp | b+ lift | W grasp+1 | W grasp+2 | W grasp+3 |
|---|---:|---:|---:|---:|---:|---:|
| cabinet | 0.674 | 0.828 | 0.855 | 0.271 | 0.184 | 0.106 |
| plate | 0.726 | 0.813 | 0.885 | 0.315 | 0.216 | 0.114 |

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
| belief - base | cabinet | pre_lift_eef_path_m | 94 | +0.0262 [+0.0084, +0.0467] |
| belief - base | cabinet | pre_lift_action_tv_l2 | 94 | +2.7267 [+1.5256, +4.1456] |
| belief - base | cabinet | post_lift_early_auc20 | 94 | +0.0191 [+0.0126, +0.0258] |
| belief - base | cabinet | post_lift_wrong_fraction15 | 94 | +0.0000 [+0.0000, +0.0000] |
| belief - base | plate | pre_lift_eef_path_m | 98 | +0.0365 [+0.0198, +0.0503] |
| belief - base | plate | pre_lift_action_tv_l2 | 98 | +1.1451 [+0.3222, +1.7975] |
| belief - base | plate | post_lift_early_auc20 | 98 | +0.0464 [+0.0348, +0.0582] |
| belief - base | plate | post_lift_wrong_fraction15 | 98 | -0.1733 [-0.2153, -0.1313] |
| belief - time_decay | cabinet | pre_lift_eef_path_m | 91 | +0.0233 [+0.0074, +0.0421] |
| belief - time_decay | cabinet | pre_lift_action_tv_l2 | 91 | +1.2335 [+0.4164, +2.1990] |
| belief - time_decay | cabinet | post_lift_early_auc20 | 91 | -0.0128 [-0.0187, -0.0071] |
| belief - time_decay | cabinet | post_lift_wrong_fraction15 | 91 | +0.0000 [+0.0000, +0.0000] |
| belief - time_decay | plate | pre_lift_eef_path_m | 94 | -0.0034 [-0.0173, +0.0075] |
| belief - time_decay | plate | pre_lift_action_tv_l2 | 94 | +0.1557 [-0.3771, +0.6173] |
| belief - time_decay | plate | post_lift_early_auc20 | 94 | -0.0016 [-0.0111, +0.0078] |
| belief - time_decay | plate | post_lift_wrong_fraction15 | 94 | +0.0082 [-0.0272, +0.0442] |
