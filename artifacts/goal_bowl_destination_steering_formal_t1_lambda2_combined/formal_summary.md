# Goal bowl-destination steering T=1 lambda=2 combined evaluation

| Method | Target | Success | 95% Wilson CI | EEF path (m) | Action TV (L2) | Policy steps |
|---|---|---:|---:|---:|---:|---:|
| base | cabinet | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8709 | 11.0092 | 85.0 |
| base | plate | 99/100 (99.0%) | [94.6%, 99.8%] | 0.7001 | 11.6300 | 78.0 |
| time_decay | cabinet | 96/100 (96.0%) | [90.2%, 98.4%] | 0.8921 | 14.2718 | 94.7 |
| time_decay | plate | 99/100 (99.0%) | [94.6%, 99.8%] | 0.7002 | 12.3473 | 79.6 |
| belief | cabinet | 94/100 (94.0%) | [87.5%, 97.2%] | 0.9145 | 15.9711 | 102.7 |
| belief | plate | 100/100 (100.0%) | [96.3%, 100.0%] | 0.6981 | 12.4455 | 78.0 |

## Executed guidance weights

| Method | Target | First segment | Mean max after first | Mean over later segments |
|---|---|---:|---:|---:|
| time_decay | cabinet | 1.0000 | 0.9000 | 0.4399 |
| time_decay | plate | 1.0000 | 0.9000 | 0.4764 |
| belief | cabinet | 0.0000 | 0.9846 | 0.3649 |
| belief | plate | 0.0000 | 0.9866 | 0.5057 |

## Paired comparisons

The confidence interval resamples the 50 initial states as clusters; the exact McNemar p-value is episode-level.

- time_decay vs base, cabinet: success-rate difference -4.0% (state-cluster bootstrap 95% CI [-8.0%, -1.0%]); episode-level McNemar exact p=0.125; discordant wins/losses=0/4.
- time_decay vs base, plate: success-rate difference +0.0% (state-cluster bootstrap 95% CI [-3.0%, +3.0%]); episode-level McNemar exact p=1; discordant wins/losses=1/1.
- belief vs base, cabinet: success-rate difference -6.0% (state-cluster bootstrap 95% CI [-11.0%, -2.0%]); episode-level McNemar exact p=0.03125; discordant wins/losses=0/6.
- belief vs base, plate: success-rate difference +1.0% (state-cluster bootstrap 95% CI [+0.0%, +3.0%]); episode-level McNemar exact p=1; discordant wins/losses=1/0.
- belief vs time_decay, cabinet: success-rate difference -2.0% (state-cluster bootstrap 95% CI [-8.0%, +3.0%]); episode-level McNemar exact p=0.7266; discordant wins/losses=3/5.
- belief vs time_decay, plate: success-rate difference +1.0% (state-cluster bootstrap 95% CI [+0.0%, +3.0%]); episode-level McNemar exact p=1; discordant wins/losses=1/0.

## Motion deltas on jointly successful pairs

Negative deltas mean the first method used less motion than the reference. Confidence intervals resample init-state clusters.

| Comparison | Target | Pairs | EEF path delta (m) | Action-TV delta |
|---|---|---:|---:|---:|
| time_decay - base | cabinet | 96 | +0.0089 [-0.0139, +0.0334] | +1.985 [+0.864, +3.262] |
| time_decay - base | plate | 98 | -0.0044 [-0.0182, +0.0073] | +0.532 [-0.242, +1.062] |
| belief - base | cabinet | 94 | +0.0235 [-0.0035, +0.0518] | +2.936 [+1.712, +4.318] |
| belief - base | plate | 99 | +0.0015 [-0.0134, +0.0150] | +1.011 [+0.172, +1.686] |
| belief - time_decay | cabinet | 91 | +0.0212 [+0.0022, +0.0444] | +1.294 [+0.384, +2.330] |
| belief - time_decay | plate | 99 | +0.0057 [-0.0035, +0.0164] | +0.481 [+0.185, +0.865] |
