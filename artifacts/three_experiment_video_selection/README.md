# Three-experiment qualitative video selection

这 6 条视频用于定性展示。每一对都复用相同的 initial state；视频选择不替代完整组统计。

## Experiment 1: early-distinguishable intent

- `01_early_clear_belief_normal.mp4`
  - Belief-weighted，目标为从 cookie box 上拿起 black bowl 并放到 plate。
  - `init=2, repeat=1`，success。
  - EEF path `0.7802 m`，Action TV `10.7876`，policy steps `76`。
  - 同一配对的 Base 为 `0.7844 m / 10.9198 / 77`，因此这条 Belief 轨迹可作为接近 Base 的“正常”示例。
- `02_early_clear_time_decay_oversteered.mp4`
  - Time-decay，和上一条完全相同的 target、initial state 与 repeat，success。
  - EEF path `1.2736 m`，Action TV `28.6509`，policy steps `176`。
  - 轨迹出现明显绕行和反复修正，适合作为 oversteering 示例。

这一对是为了让视觉差异清楚而挑出的 extreme illustrative episode，不应被描述为组均值。

## Experiment 2: shared pickup prefix, cabinet destination

- `03_shared_prefix_base_cabinet.mp4`
- `04_shared_prefix_belief_cabinet.mp4`

两条均为 `init=46, repeat=0`、`put the bowl on top of the cabinet`，且均成功。选择这一对是因为 shared pickup 部分接近，同时 lift 后的目的地表达有明显差异：

| Metric | Base | Belief (`T=1`, `lambda=2`) |
|---|---:|---:|
| Pre-lift EEF path | 0.4297 m | 0.4217 m |
| Pre-lift Action TV | 6.9251 | 7.7471 |
| Post-lift AUC20 | 0.2875 | 0.3831 |
| Full EEF path | 0.8437 m | 0.8161 m |
| Policy steps | 81 | 78 |

该 Belief episode 的实际 guidance weight 在 close / grasp / lift 时分别为 `0.696 / 0.459 / 0.321`。

## Experiment 3: SpaceMouse two-object pickup

- `05_two_object_base_tomato_sauce.mp4`
- `06_two_object_belief_tomato_sauce.mp4`

两条均来自正式 `H=400` pilot，使用 `init=1, repeat=0` 和较远的 `pick up the tomato sauce` 目标，且均成功。Step 70 时，Base 的 EEF `y=0.0121 m`，Belief 的 EEF `y=0.0478 m`；tomato-sauce 目标位于 `y=0.0603 m`。在前向位置接近时，Belief 已额外向罐头方向横移约 `3.57 cm`。

| Metric | Base | Belief |
|---|---:|---:|
| EEF path | 0.4630 m | 0.5378 m |
| Action TV | 5.8914 | 11.6890 |
| Evidence @ 50 | 0.1002 | 0.1932 |
| Policy steps | 200 | 287 |

正式运行保存了完整 simulator states、但当时关闭了 MP4 写出；这里的视频是在相同 BDDL 场景中逐状态回放并重新渲染的 20 fps 双视角视频。当前仓库记录的微调集为两个指令各 100 条、总计 200 条 demo。

## Metric wording

- Post-lift geometric evidence、AUC20 和 Wrong-first-15 是 Experiment 2 的直接 geometric-legibility proxies。
- EEF path、Action TV 和 policy steps 是运动代价/效率指标，不应直接称为 legibility。
