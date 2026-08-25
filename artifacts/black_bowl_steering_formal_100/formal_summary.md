# Black-bowl steering formal evaluation

| Method | Target | Success | 95% Wilson CI | EEF path (m) | Action TV (L2) | Policy steps |
|---|---|---:|---:|---:|---:|---:|
| base | cookie | 100/100 (100.0%) | [96.3%, 100.0%] | 0.8472 | 12.0578 | 84.8 |
| base | cabinet | 82/100 (82.0%) | [73.3%, 88.3%] | 1.2782 | 19.2443 | 137.1 |
| time_decay | cookie | 97/100 (97.0%) | [91.5%, 99.0%] | 0.8797 | 13.0273 | 89.8 |
| time_decay | cabinet | 99/100 (99.0%) | [94.6%, 99.8%] | 1.3065 | 17.6103 | 121.1 |
| belief | cookie | 99/100 (99.0%) | [94.6%, 99.8%] | 0.8558 | 12.4524 | 86.3 |
| belief | cabinet | 80/100 (80.0%) | [71.1%, 86.7%] | 1.2863 | 19.6870 | 138.8 |

## Executed guidance weights

| Method | Target | First segment | Mean max after first | Mean over later segments |
|---|---|---:|---:|---:|
| time_decay | cookie | 1.0000 | 0.9000 | 0.4462 |
| time_decay | cabinet | 1.0000 | 0.9000 | 0.3533 |
| belief | cookie | 0.0000 | 0.0480 | 0.0057 |
| belief | cabinet | 0.0000 | 0.0456 | 0.0019 |

## Paired comparisons

The confidence interval resamples the 50 initial states as clusters; the exact McNemar p-value is episode-level.

- time_decay vs base, cookie: success-rate difference -3.0% (state-cluster bootstrap 95% CI [-7.0%, +0.0%]); episode-level McNemar exact p=0.25; discordant wins/losses=0/3.
- time_decay vs base, cabinet: success-rate difference +17.0% (state-cluster bootstrap 95% CI [+7.0%, +28.0%]); episode-level McNemar exact p=7.629e-05; discordant wins/losses=18/1.
- belief vs base, cookie: success-rate difference -1.0% (state-cluster bootstrap 95% CI [-3.0%, +0.0%]); episode-level McNemar exact p=1; discordant wins/losses=0/1.
- belief vs base, cabinet: success-rate difference -2.0% (state-cluster bootstrap 95% CI [-7.0%, +2.0%]); episode-level McNemar exact p=0.6875; discordant wins/losses=2/4.
- belief vs time_decay, cookie: success-rate difference +2.0% (state-cluster bootstrap 95% CI [-2.0%, +6.0%]); episode-level McNemar exact p=0.625; discordant wins/losses=3/1.
- belief vs time_decay, cabinet: success-rate difference -19.0% (state-cluster bootstrap 95% CI [-29.0%, -10.0%]); episode-level McNemar exact p=3.815e-06; discordant wins/losses=0/19.

## Motion deltas on jointly successful pairs

Negative deltas mean the first method used less motion than the reference. Confidence intervals resample init-state clusters.

| Comparison | Target | Pairs | EEF path delta (m) | Action-TV delta |
|---|---|---:|---:|---:|
| time_decay - base | cookie | 97 | +0.0185 [-0.0066, +0.0436] | +0.454 [-0.463, +1.445] |
| time_decay - base | cabinet | 81 | +0.0398 [+0.0058, +0.0741] | +0.720 [-1.848, +2.934] |
| belief - base | cookie | 99 | +0.0039 [-0.0006, +0.0103] | +0.208 [-0.062, +0.510] |
| belief - base | cabinet | 78 | -0.0151 [-0.0509, +0.0064] | -1.370 [-3.986, +0.187] |
| belief - time_decay | cookie | 96 | -0.0243 [-0.0462, -0.0045] | -0.597 [-1.506, +0.197] |
| belief - time_decay | cabinet | 80 | -0.0496 [-0.0874, -0.0118] | -1.916 [-3.686, -0.205] |
