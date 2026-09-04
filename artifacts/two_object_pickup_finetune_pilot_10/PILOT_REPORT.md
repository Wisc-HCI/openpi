# Two-object pickup fine-tune steering pilot

Checkpoint: `/workspace/checkpoints/checkpoints/pi05_libero_legibility_finetune/libero_legibility_v1/19999`

All 80 episodes passed paired initial-observation and common sampling-seed checks.

| Condition | Target | Success | Successful steps | Successful EEF path (m) | Evidence Δ@20 / @50 | W q1 / q2 / q5 / at lift |
|---|---|---:|---:|---:|---:|---:|
| base | cream_cheese | 8/10 | 181.8 | 0.446 | -0.003 / -0.011 | 0.000 / 0.000 / 0.000 / 0.000 |
| base | tomato_sauce | 3/10 | 195.3 | 0.454 | 0.009 / 0.082 | 0.000 / 0.000 / 0.000 / 0.000 |
| time_decay_g0p9 | cream_cheese | 8/10 | 197.1 | 0.464 | 0.001 / 0.020 | 1.000 / 0.900 / 0.656 / 0.026 |
| time_decay_g0p9 | tomato_sauce | 2/10 | 217.0 | 0.493 | 0.018 / 0.135 | 1.000 / 0.900 / 0.656 / 0.013 |
| time_decay_g0p5 | cream_cheese | 8/10 | 187.2 | 0.456 | -0.001 / -0.008 | 1.000 / 0.500 / 0.062 / 0.000 |
| time_decay_g0p5 | tomato_sauce | 3/10 | 195.7 | 0.451 | 0.013 / 0.091 | 1.000 / 0.500 / 0.062 / 0.000 |
| belief | cream_cheese | 8/10 | 178.1 | 0.473 | 0.001 / 0.049 | 0.000 / 1.000 / 0.999 / 0.797 |
| belief | tomato_sauce | 0/10 | nan | nan | 0.018 / 0.169 | 0.000 / 1.000 / 0.998 / nan |

The geometric evidence is the change in normalized EEF closeness to the instructed object versus the competing object; positive values favor the instructed target.

This is a screening pilot (n=10/group), so differences are descriptive rather than statistically conclusive.
