# pi05-LIBERO spatial-legibility report

Frozen geometry digest: `456da6d01af8ec7a03e0ec7d694e9b96d0c4da7d35fbca7d0715852033cb92c9`
Episodes: 40
Scripted task success: 100.0%

## Early target inference

- Main L0-L3 balanced accuracy at 20%: 50.0%
- Main L0-L3 balanced accuracy at 30% (primary): 62.5%
- Main L0-L3 balanced accuracy at 40%: 59.4%
- Including anti-legible A1 at 30% (diagnostic): 62.5%

## Monotonicity

- Mean matched Spearman rho(level, C30): 0.225
- Fraction L3 > L0: 50.0%
- Adjacent-order success: 62.5%
- Mean L3 seed sign agreement: 90.3%
- Minimum per-condition L3 seed sign agreement: 54.0%

## Preregistered criteria

- PASS — all_executed_trajectories_collision_free
- PASS — geometry_L0_to_L3_deviation_increasing_in_every_matched_group
- PASS — geometry_L0_to_L3_confidence_increasing_in_every_matched_group
- PASS — geometry_A1_confidence_below_L0_in_every_matched_group
- PASS — geometry_L0_not_near_certain_at_10pct_in_every_matched_group
- PASS — geometry_L0_to_L3_path_cost_increasing_in_every_matched_group
- PASS — scripted_task_success_at_least_95pct
- FAIL — main_L0_L3_balanced_accuracy_20_at_least_70pct
- FAIL — main_L0_L3_balanced_accuracy_30_at_least_80pct
- FAIL — main_L0_L3_balanced_accuracy_40_at_least_80pct
- FAIL — mean_spearman_rho_at_least_0_6
- FAIL — L3_above_L0_at_least_75pct
- FAIL — adjacent_order_success_at_least_75pct
- FAIL — every_L3_condition_seed_sign_agreement_at_least_80pct

No trajectory parameter or weighting function is selected from these pi05 results. Any failed criterion remains reported as a failure.
