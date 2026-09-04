# Two-object pickup fine-tune steering pilot

Checkpoint: `/workspace/checkpoints/checkpoints/pi05_libero_legibility_finetune/libero_legibility_v1/19999`

All 40 episodes passed paired initial-observation and common sampling-seed checks.

| Condition | Target | Success | Successful steps | Successful EEF path (m) | Evidence Δ@20 / @50 | W q1 / q2 / q5 / at lift |
|---|---|---:|---:|---:|---:|---:|
| base | cream_cheese | 5/5 | 188.4 | 0.452 | -0.003 / -0.012 | 0.000 / 0.000 / 0.000 / 0.000 |
| base | tomato_sauce | 3/5 | 218.0 | 0.479 | 0.009 / 0.076 | 0.000 / 0.000 / 0.000 / 0.000 |
| time_decay_g0p9 | cream_cheese | 4/5 | 185.2 | 0.460 | 0.001 / 0.021 | 1.000 / 0.900 / 0.656 / 0.028 |
| time_decay_g0p9 | tomato_sauce | 2/5 | 233.5 | 0.505 | 0.018 / 0.127 | 1.000 / 0.900 / 0.656 / 0.010 |
| time_decay_g0p5 | cream_cheese | 5/5 | 188.2 | 0.457 | -0.001 / -0.008 | 1.000 / 0.500 / 0.062 / 0.000 |
| time_decay_g0p5 | tomato_sauce | 3/5 | 227.3 | 0.483 | 0.013 / 0.086 | 1.000 / 0.500 / 0.062 / 0.000 |
| belief | cream_cheese | 5/5 | 194.2 | 0.510 | 0.001 / 0.050 | 0.000 / 1.000 / 0.999 / 0.784 |
| belief | tomato_sauce | 1/5 | 287.0 | 0.538 | 0.019 / 0.162 | 0.000 / 1.000 / 0.998 / 0.681 |

The geometric evidence is the change in normalized EEF closeness to the instructed object versus the competing object; positive values favor the instructed target.

This is a screening pilot (n=5/group), so differences are descriptive rather than statistically conclusive.
