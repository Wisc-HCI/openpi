# pi0.5-base versus pi0.5-DROID: fixed-temperature comparison

This analysis reads saved residual energies; it does not rescore trajectories or modify prior artifacts.
Temperature 1 is the pre-specified primary probability mapping. The remaining temperatures are a sensitivity
analysis, not candidates selected by held-out performance. Final accuracy and raw-sign AUC are temperature-free.

## Fixed temperature 1

| Run | Split | N | Final acc. | Raw-sign AUC | Probability AUC | Final belief | Prefix NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05_base_h16 | test | 8 | 0.500000 | 0.582884 | 0.500679 | 0.500643 | 0.691624 |
| pi05_base_h16 | train | 32 | 0.593750 | 0.615493 | 0.500528 | 0.500776 | 0.691899 |
| pi05_droid_h16 | test | 8 | 0.875000 | 0.751931 | 0.516623 | 0.522284 | 0.658102 |
| pi05_droid_h16 | train | 32 | 0.593750 | 0.541877 | 0.500001 | 0.501631 | 0.693222 |

## Held-out conditions at temperature 1

| Run | Condition | Final acc. | Raw-sign AUC | Probability AUC | Final belief |
|---:|---:|---:|---:|---:|---:|
| pi05_base_h16 | C0_natural_direct | 0.500000 | 0.501338 | 0.499490 | 0.499418 |
| pi05_base_h16 | C1_mild_exaggeration | 1.000000 | 0.624804 | 0.500161 | 0.500568 |
| pi05_base_h16 | C2_strong_exaggeration | 0.500000 | 0.708887 | 0.503136 | 0.504001 |
| pi05_base_h16 | C3_extreme_exaggeration | 0.000000 | 0.496506 | 0.499928 | 0.498583 |
| pi05_droid_h16 | C0_natural_direct | 0.500000 | 0.403424 | 0.493169 | 0.492943 |
| pi05_droid_h16 | C1_mild_exaggeration | 1.000000 | 0.957614 | 0.510929 | 0.510517 |
| pi05_droid_h16 | C2_strong_exaggeration | 1.000000 | 0.673173 | 0.528821 | 0.545207 |
| pi05_droid_h16 | C3_extreme_exaggeration | 1.000000 | 0.973514 | 0.533574 | 0.540468 |

## Held-out temperature sensitivity

| Run | Temperature | Test probability AUC | Test final belief | Test prefix NLL |
|---:|---:|---:|---:|---:|
| pi05_base_h16 | 0.010000 | 0.533478 | 0.508786 | 0.674106 |
| pi05_base_h16 | 0.030000 | 0.520112 | 0.517042 | 0.661171 |
| pi05_base_h16 | 0.100000 | 0.506711 | 0.506292 | 0.679560 |
| pi05_base_h16 | 0.300000 | 0.502260 | 0.502137 | 0.688213 |
| pi05_base_h16 | 1.000000 | 0.500679 | 0.500643 | 0.691624 |
| pi05_base_h16 | 3.000000 | 0.500226 | 0.500214 | 0.692635 |
| pi05_base_h16 | 10.000000 | 0.500068 | 0.500064 | 0.692993 |
| pi05_droid_h16 | 0.010000 | 0.738316 | 0.840511 | 1.125330 |
| pi05_droid_h16 | 0.030000 | 0.697748 | 0.748492 | 0.541726 |
| pi05_droid_h16 | 0.100000 | 0.623750 | 0.656282 | 0.501369 |
| pi05_droid_h16 | 0.300000 | 0.553065 | 0.570350 | 0.592985 |
| pi05_droid_h16 | 1.000000 | 0.516623 | 0.522284 | 0.658102 |
| pi05_droid_h16 | 3.000000 | 0.505564 | 0.507466 | 0.680970 |
| pi05_droid_h16 | 10.000000 | 0.501670 | 0.502241 | 0.689442 |

No temperature is declared "best" here. Selecting one requires a separate, representative calibration set
and a criterion fixed before evaluating the eight held-out trajectories.
