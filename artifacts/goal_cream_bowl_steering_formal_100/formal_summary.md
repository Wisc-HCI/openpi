# Goal cream-cheese / bowl-stove steering formal evaluation

| Method | Target | Success | 95% Wilson CI | EEF path (m) | Action TV (L2) | Policy steps |
|---|---|---:|---:|---:|---:|---:|
| base | cream_cheese | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8210 | 15.2186 | 92.4 |
| base | bowl_stove | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8190 | 11.1727 | 87.7 |
| time_decay | cream_cheese | 95/100 (95.0%) | [88.8%, 97.8%] | 0.9314 | 20.4984 | 111.7 |
| time_decay | bowl_stove | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8230 | 11.7988 | 88.0 |
| belief | cream_cheese | 99/100 (99.0%) | [94.6%, 99.8%] | 0.8242 | 15.4310 | 94.4 |
| belief | bowl_stove | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8141 | 10.7474 | 86.6 |

## Executed guidance weights

| Method | Target | First segment | Mean max after first | Mean over later segments |
|---|---|---:|---:|---:|
| time_decay | cream_cheese | 1.0000 | 0.9000 | 0.3930 |
| time_decay | bowl_stove | 1.0000 | 0.9000 | 0.4428 |
| belief | cream_cheese | 0.0000 | 0.3617 | 0.0326 |
| belief | bowl_stove | 0.0000 | 0.3668 | 0.0343 |

## Paired comparisons

The confidence interval resamples the 50 initial states as clusters; the exact McNemar p-value is episode-level.

- time_decay vs base, cream_cheese: success-rate difference -5.0% (state-cluster bootstrap 95% CI [-11.0%, -1.0%]); episode-level McNemar exact p=0.0625; discordant wins/losses=0/5.
- time_decay vs base, bowl_stove: success-rate difference +0.0% (state-cluster bootstrap 95% CI [+0.0%, +0.0%]); episode-level McNemar exact p=1; discordant wins/losses=0/0.
- belief vs base, cream_cheese: success-rate difference -1.0% (state-cluster bootstrap 95% CI [-3.0%, +0.0%]); episode-level McNemar exact p=1; discordant wins/losses=0/1.
- belief vs base, bowl_stove: success-rate difference +0.0% (state-cluster bootstrap 95% CI [+0.0%, +0.0%]); episode-level McNemar exact p=1; discordant wins/losses=0/0.
- belief vs time_decay, cream_cheese: success-rate difference +4.0% (state-cluster bootstrap 95% CI [-1.0%, +10.0%]); episode-level McNemar exact p=0.2188; discordant wins/losses=5/1.
- belief vs time_decay, bowl_stove: success-rate difference +0.0% (state-cluster bootstrap 95% CI [+0.0%, +0.0%]); episode-level McNemar exact p=1; discordant wins/losses=0/0.

## Motion deltas on jointly successful pairs

Negative deltas mean the first method used less motion than the reference. Confidence intervals resample init-state clusters.

| Comparison | Target | Pairs | EEF path delta (m) | Action-TV delta |
|---|---|---:|---:|---:|
| time_decay - base | cream_cheese | 95 | +0.0812 [+0.0441, +0.1201] | +2.433 [+0.278, +4.567] |
| time_decay - base | bowl_stove | 100 | +0.0040 [-0.0110, +0.0181] | +0.626 [-0.053, +1.323] |
| belief - base | cream_cheese | 99 | -0.0005 [-0.0219, +0.0164] | -0.007 [-1.677, +1.926] |
| belief - base | bowl_stove | 100 | -0.0049 [-0.0156, +0.0043] | -0.425 [-0.907, -0.023] |
| belief - time_decay | cream_cheese | 94 | -0.0816 [-0.1187, -0.0465] | -2.367 [-4.705, +0.191] |
| belief - time_decay | bowl_stove | 100 | -0.0090 [-0.0209, +0.0026] | -1.051 [-1.609, -0.583] |
