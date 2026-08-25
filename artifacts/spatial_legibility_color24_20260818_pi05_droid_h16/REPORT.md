# 2026-08-18 颜色×左右×layout 平衡实验：24 条轨迹分析

## 核心结论

这 24 条轨迹给出了比之前更清楚的结论：

1. **真实的 blue/red 指令在早期不能可靠识别目标。** 10%、20%、30%、40% accuracy 分别只有 `41.7% / 37.5% / 29.2% / 29.2%`。但最后一个 local chunk 是 `24/24` 正确，完整 Flat accumulation 回升到 `17/24 = 70.8%`。这批数据的有效颜色证据主要出现在轨迹后段。
2. **因此 early weighting 对颜色任务适得其反。** Flat 为 `70.8%`，Linear 只有 `33.3%`，Exponential (`alpha=3`) 只有 `29.2%`。早期 evidence 明显偏 Red，并混有 base-right preference。
3. **换成 left/right prompts 后，最早阶段确实能读到 robot-base-compatible 的方向信号。** 10% accuracy 为 `19/24 = 79.2%`，其中 base-left 为 `12/12`，base-right 为 `7/12`。由于当前 exterior camera 中 base-left 位于 screen-right，反事实 camera-frame accuracy 只有 `20.8%`；这支持早期 cue 不只是 exterior screen-left/right。
4. **但在原图条件下，left/right observer 从 30% 起完全塌缩成 Left。** 30%、40%、100% 以及 Flat、Linear、Exponential 的轨迹均值全部预测 Left：Left accuracy `12/12`，Right accuracy `0/12`。Early weighting 没有保住最初的 Right signal。
5. **交换蓝红位置、改变 L1/L3、改变 C0/C1/C2 都没有消除最终 Left bias。** 它也不是 evaluator、batch order 或 noise 导致：两套 prompt 各自的 216 个 sanity comparisons 全部精确通过，最大 absolute delta 为 `0.0`。

因此这批数据支持一个分层解释：

> observer 在最早阶段包含与机械臂 base frame 相容的方向 cue；原图的中后段 `left/right` residual 则受到 Left asymmetry 与 exterior-camera reference-frame conflict 的共同影响。另一方面，颜色 residual 在早期受 Red 与空间 prior 支配，到接近目标时才形成可靠的对象颜色 evidence。

实际实验指令仍然是颜色，所以 **blue/red 应作为主任务 observer，left/right 只能作为参考系诊断**。不建议用 side prompt 的最终 50% 结果替代颜色结果。

### Exterior-camera 水平反转补充实验

随后对同一批 24 条轨迹只水平反转 exterior image，保持 wrist、state、action、chunk、seed、noise 和 flow timestep 不变。结果显著修正了对原图 all-Left 的解释：

- Side Flat 从 `50.0%` 提升到 `75.0%`，Linear/Exponential 提升到 `79.2%`；
- Right accuracy 从 `0/12` 提升到 Flat `6/12`、Linear/Exponential `7/12`；
- C0 完全不改善，C1 变为 `8/8`，C2 为 Flat `6/8`；
- 10% accuracy 反而从 `79.2%` 降到 `54.2%`，所以不是简单的 screen-frame 标签交换；
- Color Flat 仍为 `70.8%`，最后一个 color local chunk 仍为 `24/24` 正确。

因此 camera layout 确实解释了相当一部分 curved Right trajectory 的 late Left bias，但 fixed Left prior、state/action/wrist cue 和 mirror OOD effect 仍没有被完全分离。详见 [24 条轨迹 exterior-flip 配对报告](EXTERIOR_HFLIP_REPORT.md)。

## 数据设计与审计

纳入范围从：

`2026_08_18_00_40_18_199635_L1_left_C0`

开始，正好覆盖完整的：

`2 layouts × 2 blue sides × 2 target colors × 3 curvatures = 24 trajectories`

每个 factorial cell 恰好一条：

- layout：L1 12 条、L3 12 条；
- blue side：base-left 12 条、base-right 12 条；
- target color：blue 12 条、red 12 条；
- target side：base-left 12 条、base-right 12 条；
- curvature：C0、C1、C2 各 8 条。

所有 episode 的以下内容均互相一致：

- `language_instruction = pick up the {target_color} block`；
- `target_side` 等于目标颜色当前所在的 base side；
- `blue_block_side` 与 `red_block_side` 相反；
- 文件名中的 layout、target side、condition 与 metadata 一致；
- `trial_qc.json: passed=true`。

排除了 failure 目录中的：

`2026_08_18_00_58_53_653392_L3_left_C2`

纳入随后成功重采的：

`2026_08_18_00_59_48_029579_L3_left_C2`

一个 collection metadata caveat：24 个 success-directory episode 的 `metadata_openpi.json` 都仍保留 recorder 初始化时的 `success=false, failure=true`。Collector 源码显示这个字段在成功归档后没有重写；本分析以 final success directory 和 `trial_qc.passed=true` 为纳入依据。

## 几何验证

| Factor | n | Mean grasp x | Mean grasp y | Mean path length | Mean path ratio | Mean max line deviation |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 12 | `0.6416 m` | `-0.0006 m` | `0.4832 m` | `1.189` | `0.0946 m` |
| L3 | 12 | `0.6670 m` | `-0.0006 m` | `0.4993 m` | `1.176` | `0.0947 m` |
| Base-left | 12 | `0.6544 m` | `+0.0427 m` | `0.4914 m` | `1.183` | `0.0947 m` |
| Base-right | 12 | `0.6541 m` | `-0.0439 m` | `0.4911 m` | `1.182` | `0.0947 m` |
| C0 | 8 | `0.6546 m` | `-0.0013 m` | `0.4199 m` | `1.010` | `0.0098 m` |
| C1 | 8 | `0.6541 m` | `-0.0002 m` | `0.4683 m` | `1.127` | `0.0892 m` |
| C2 | 8 | `0.6541 m` | `-0.0003 m` | `0.5856 m` | `1.410` | `0.1851 m` |

L3 相比 L1 的 grasp `x` 平均增加约 `2.54 cm`。左右 endpoint 的 `y` 间距约 `8.66 cm`。C0/C1/C2 的 path deviation 逐级增加，确认轨迹弯曲设计真实进入了记录数据，而不只是 metadata 标签。

[查看 L1/L3 与 blue-side 的 exterior-camera 首帧审计](color_layout_visual_audit.png)。画面确认：

- base-left 对应 exterior screen-right；
- base-right 对应 exterior screen-left；
- metadata 中叫 `red` 的方块视觉上更接近 pink/coral。这可能带来 `red` prompt 与物体外观的额外语义校准问题。

## 评分设置

- 官方 `pi05_droid` checkpoint；
- `pi05_droid_finetune` matched config，action horizon 16；
- seeds `0,...,9`；
- 每个 seed 八个 flow samples；
- flow timestep `[0.3, 0.7]`；
- action dimensions 8；
- 两个候选使用完全相同的 action、noise、noisy action 和 flow timestep；
- 评分文件包含 post-grasp 以便审计，正式分析只用 pre-grasp；
- evidence 记在 chunk stop，避免 future-action leakage；
- prefix 使用累计 margin 在真实 chunk-stop progress 上线性插值。

两套 prompt 各产生 `2,610` 个 chunk×seed score。

颜色 margin：

`m_color = E_blue - E_red`，正值支持 Red，负值支持 Blue。

方向 margin：

`m_side = E_left - E_right`，正值支持 Right，负值支持 Left。

## Blue/red prompts

### Prefix accuracy

| Prefix | Overall | Blue target | Red target | Base-left target | Base-right target | Seed-pooled |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 41.7% | 16.7% | 66.7% | 25.0% | 58.3% | 41.7% |
| 20% | 37.5% | 8.3% | 66.7% | 16.7% | 58.3% | 36.3% |
| 30% | 29.2% | 0.0% | 58.3% | 8.3% | 50.0% | 29.2% |
| 40% | 29.2% | 0.0% | 58.3% | 16.7% | 41.7% | 26.3% |
| 100% | 70.8% | 50.0% | 91.7% | 58.3% | 83.3% | 66.7% |

最强的反例是 30% 和 40%：12 条 Blue-target trajectory 的均值判断为 `0/12` 正确。此时不是“早期已经知道颜色，后期噪声覆盖”，而是早期 residual 本身对 Blue 目标给出了系统性错误的 Red evidence。

### Flat、Linear、Exponential

| Method | Overall | Blue | Red | Base-left | Base-right | L1 | L3 | C0 | C1 | C2 | Seed-pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat | 70.8% | 50.0% | 91.7% | 58.3% | 83.3% | 75.0% | 66.7% | 62.5% | 62.5% | 87.5% | 66.7% |
| Linear | 33.3% | 8.3% | 58.3% | 16.7% | 50.0% | 25.0% | 41.7% | 37.5% | 25.0% | 37.5% | 40.0% |
| Exponential | 29.2% | 0.0% | 58.3% | 8.3% | 50.0% | 33.3% | 25.0% | 37.5% | 25.0% | 25.0% | 34.2% |

其他关键诊断：

- Flat 预测 Blue `7/24`、Red `17/24`；Exponential 预测 Blue `5/24`、Red `19/24`。
- Exponential 的颜色预测映射回当前 block layout 后，有 `17/24 = 70.8%` 指向 base-right，说明早期 bias 是 Red prior 与 spatial cue 的混合，而不是单一 color prior。
- 第一 local chunk 只有 `10/24` 正确；最后一个 local chunk 是 `24/24` 正确。
- `early-correct@20--40%` 为 `10/24`；late reversal 只有 `1/24`。这批颜色任务更常见的是 **early-wrong / late-recovery**。
- Flat mean prediction 在 22/24 条轨迹上有至少 80% seed agreement；两个不稳定例子是 L1-left-C0 Blue（5/10）和 L3-left-C1 Red（6/10）。
- C2 的 Flat `87.5%` 高于 C0/C1 的 `62.5%`，说明强弯曲轨迹最终更容易区分颜色目标；但这个优势主要在后段出现，early weighting 没有捕捉到单调的 C0<C1<C2 改善。

在固定 `layout + target side + curvature`、只交换两种颜色的 12 个 matched pairs 中：

- 30% 和 40% 没有任何一对做到 Blue 与 Red 两条都正确；
- 100%/Flat 有 6/12 pairs 两条都正确；
- matched mean `Red-target score - Blue-target score` 从 10% 的 `-0.0207`、40% 的 `-0.1106`，到 100% 变成 `+0.1141`。正值才是正确方向。

这进一步确认真正的 color-target contrast 是后期才出现的。

[颜色总体 accuracy 与 bias 图](aggregate_accuracy_and_bias.png)

## Left/right prompts

这些不是实际 recorded instruction，而是同一 observation/state/action 上的 counterfactual diagnostic prompts。

### Prefix accuracy

| Prefix | Base-frame overall | Base-left | Base-right | Camera-frame counterfactual | Seed-pooled |
|---|---:|---:|---:|---:|---:|
| 10% | 79.2% | 100.0% | 58.3% | 20.8% | 71.3% |
| 20% | 66.7% | 100.0% | 33.3% | 33.3% | 64.2% |
| 30% | 50.0% | 100.0% | 0.0% | 50.0% | 57.5% |
| 40% | 50.0% | 100.0% | 0.0% | 50.0% | 53.8% |
| 100% | 50.0% | 100.0% | 0.0% | 50.0% | 48.8% |

10% 是最有信息量的结果：

- 所有 12 条 base-left trajectory 都预测 Left；
- 12 条 base-right 中有 7 条预测 Right；
- Blue target 为 83.3%，Red target 为 75.0%；
- L1 为 83.3%，L3 为 75.0%；
- blue-side-left 为 75.0%，blue-side-right 为 83.3%。

因此 10% signal 并不依赖某一个 target color、blue placement 或 layout。由于外部画面左右与 base frame 相反，base-frame `79.2%` 对 camera-frame `20.8%` 是当前最强的 robot-centric-compatible evidence。不过 evaluator 还输入 wrist image、state 和 action，所以它不能单独证明 cue 只来自 proprioceptive state。

### Flat、Linear、Exponential

| Method | Overall | Left accuracy | Right accuracy | Left seed-pooled | Right seed-pooled | Predicted Left |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 50.0% | 100.0% | 0.0% | 95.8% | 1.7% | 24/24 |
| Linear | 50.0% | 100.0% | 0.0% | 100.0% | 2.5% | 24/24 |
| Exponential | 50.0% | 100.0% | 0.0% | 100.0% | 9.2% | 24/24 |

Flat、Linear、Exponential 在 L1/L3、Blue/Red target、blue-side-left/right、C0/C1/C2 的每一个 balanced marginal 中都保持相同的最终 all-Left 结构。因此这个现象不能用某一种颜色摆放或某一个 layout 单独解释。

- 第一 local chunk：19/24 正确；
- 最后一个 local chunk：17/24 正确；
- 完整 accumulation：12/24 正确，24/24 均值预测 Left；
- `early-correct@20--40%`：16/24；
- late reversal：4/24，全部是 Right trajectory，prefix pattern 都是 `R/R/L/L/L`；
- Flat final mean prediction：23/24 具有至少 80% seed agreement。

四条明确的 side-prompt late reversal 是：

- `2026_08_18_00_46_25_193744_L1_right_C2`；
- `2026_08_18_00_50_22_202653_L1_right_C0`；
- `2026_08_18_00_53_23_339528_L3_right_C1`；
- `2026_08_18_00_56_48_148319_L3_right_C1`。

Exponential (`alpha=3`) 虽然提高了 Right trajectory 的 seed-pooled accuracy 到 9.2%，但 12 条 Right 的均值仍全部小于零。因此当前 early weighting 强度不足以把 10% 的 transient Right signal 保留到最终 score；更根本的问题是 20% 以后 local evidence 很快转为 Left。

## Blue/red 与 left/right 直接比较

| Decision | Color accuracy | Side accuracy | Both correct | Color only | Side only | Neither |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 41.7% | 79.2% | 8 | 2 | 11 | 3 |
| 20% | 37.5% | 66.7% | 5 | 4 | 11 | 4 |
| 30% | 29.2% | 50.0% | 1 | 6 | 11 | 6 |
| 40% | 29.2% | 50.0% | 2 | 5 | 10 | 7 |
| 100% / Flat | 70.8% | 50.0% | 7 | 10 | 5 | 2 |
| Linear | 33.3% | 50.0% | 2 | 6 | 10 | 6 |
| Exponential | 29.2% | 50.0% | 1 | 6 | 11 | 6 |

这不是“left/right 比 color 更好”或反过来的单一排名：

- 最早 10%--20%，side prompts 更能读出实际 movement side；
- 完整 pre-grasp，color Flat 更能识别实际目标对象；
- side 的最终 50% 不是有用的 balanced classifier，而是 all-Left 恰好命中一半；
- color 的 Linear/Exponential 低于 50%，是因为早期 color evidence 系统性错误，不是随机猜测。

[查看两套 prompt 的直接对比图](color_vs_side_prompt_comparison.png)

## 每条轨迹

`Color prefix` 与 `Side prefix` 的五个字符依次为 10/20/30/40/100%；`F/L/E` 为 Flat/Linear/Exponential。Seed 列是 Flat 下十个 seed 支持各自 GT 的比例。

| Episode suffix | Layout | C | Blue side | GT color/side | Color prefix | Color F/L/E | Side prefix | Side F/L/E | Flat seed color/side |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| `00_40_18...L1_left_C0` | L1 | C0 | L | B/L | RRRRB | BRR | LLLLL | LLL | 50% / 100% |
| `00_40_47...L1_left_C1` | L1 | C1 | L | B/L | RRRRR | RRR | LLLLL | LLL | 0% / 100% |
| `00_41_27...L1_left_C2` | L1 | C2 | L | B/L | RRRRB | BRR | LLLLL | LLL | 100% / 50% |
| `00_45_28...L1_right_C0` | L1 | C0 | L | R/R | RRRBR | RBR | LLLLL | LLL | 100% / 0% |
| `00_45_56...L1_right_C1` | L1 | C1 | L | R/R | RRRRR | RRR | RLLLL | LLL | 100% / 0% |
| `00_46_25...L1_right_C2` | L1 | C2 | L | R/R | RRRRR | RRR | RRLLL | LLL | 100% / 0% |
| `00_48_13...L1_left_C0` | L1 | C0 | R | R/L | BRRRR | RRR | LLLLL | LLL | 100% / 100% |
| `00_49_05...L1_left_C1` | L1 | C1 | R | R/L | BBBBB | BBB | LLLLL | LLL | 0% / 100% |
| `00_49_36...L1_left_C2` | L1 | C2 | R | R/L | RBBBR | RBB | LLLLL | LLL | 90% / 100% |
| `00_50_22...L1_right_C0` | L1 | C0 | R | B/R | BBRRR | RRR | RRLLL | LLL | 0% / 0% |
| `00_50_52...L1_right_C1` | L1 | C1 | R | B/R | BRRRB | BRR | RLLLL | LLL | 100% / 0% |
| `00_51_25...L1_right_C2` | L1 | C2 | R | B/R | RRRRB | BRR | LLLLL | LLL | 100% / 0% |
| `00_52_54...L3_right_C0` | L3 | C0 | R | B/R | RRRRR | RRR | LLLLL | LLL | 0% / 0% |
| `00_53_23...L3_right_C1` | L3 | C1 | R | B/R | RRRRB | BRR | RRLLL | LLL | 100% / 0% |
| `00_53_52...L3_right_C2` | L3 | C2 | R | B/R | RRRRB | BBR | RLLLL | LLL | 100% / 0% |
| `00_54_32...L3_left_C0` | L3 | C0 | R | R/L | BBBBR | RRB | LLLLL | LLL | 100% / 100% |
| `00_55_01...L3_left_C1` | L3 | C1 | R | R/L | RRBBR | RBB | LLLLL | LLL | 60% / 100% |
| `00_55_32...L3_left_C2` | L3 | C2 | R | R/L | RBBRR | RBB | LLLLL | LLL | 100% / 100% |
| `00_56_19...L3_right_C0` | L3 | C0 | L | R/R | BRRRR | RRR | LLLLL | LLL | 100% / 20% |
| `00_56_48...L3_right_C1` | L3 | C1 | L | R/R | RRRRR | RRR | RRLLL | LLL | 100% / 0% |
| `00_57_15...L3_right_C2` | L3 | C2 | L | R/R | RRRRR | RRR | LLLLL | LLL | 100% / 0% |
| `00_57_54...L3_left_C0` | L3 | C0 | L | B/L | RRRRR | RRR | LLLLL | LLL | 0% / 100% |
| `00_58_22...L3_left_C1` | L3 | C1 | L | B/L | RRRRR | RRR | LLLLL | LLL | 0% / 100% |
| `00_59_48...L3_left_C2` | L3 | C2 | L | B/L | RRRRR | RRR | LLLLL | LLL | 0% / 100% |

每条 trajectory 的 local margin、seed SD、cumulative margin、Linear curve 和 Exponential curve 分别位于：

- [Blue/red trajectory curves](trajectory_curves/)
- [Left/right trajectory curves](side_prompts/trajectory_curves/)

## Sanity checks

两套 prompt 都在全部 24 条 trajectory 上取 early、约 40%、final-pregrasp 三个代表 chunks，并使用 seeds 0、1、2：每套 `24 × 3 × 3 = 216` comparisons。

| Check | Blue/red | Left/right |
|---|---:|---:|
| Same first prompt max delta | `0.0` | `0.0` |
| Same second prompt max delta | `0.0` | `0.0` |
| Batch-swap first energy max delta | `0.0` | `0.0` |
| Batch-swap second energy max delta | `0.0` | `0.0` |
| Batch-swap mapped margin max delta | `0.0` | `0.0` |
| Prediction agreement | 100% | 100% |
| Paired noise | Yes | Yes |

因此：

- `[blue,blue]`、`[red,red]` 完全一致；
- `[left,left]`、`[right,right]` 完全一致；
- candidate batch order 不影响结果；
- 10 seeds 的正式结果均使用 paired noise；
- bias 不是由 evaluator position 或候选间随机 noise 差异引起。

## 对三个核心问题的回答

### 1. Observer 是否在轨迹早期已经正确区分目标？

**对实际颜色目标：没有。** 10%--40% 均低于 50%，30%/40% 对 Blue 为 0/12。颜色身份主要在后段才清楚。

**对 movement side：只在非常早的 10%--20% 部分成立。** Side prompt 在 10% 为 79.2%、20% 为 66.7%，而且明显更符合 base frame 而不是 exterior camera frame；30% 后则全部变成 Left。

### 2. Flat 是否被后期 chunk 覆盖了正确的 early legibility？

**颜色 prompt：相反。** 后期 evidence 修复了错误 early signal；early weighting 严重降低 accuracy。

**Side prompt：Right trajectory 中确实存在 early-correct/late-Left，但 Linear/Exponential 仍不能恢复它。** 10% 后 Left evidence 增长太快，`alpha=3` 的 Exponential 最终仍为 all-Left。

### 3. Bias 来自 evaluator、noise、checkpoint prior 还是参考系？

- evaluator / batch order：证据明确否定；
- unpaired random noise：证据明确否定；
- exterior-camera-only frame：原图 10% 的 base accuracy 79.2% 对 camera accuracy 20.8%，不支持纯 screen-coordinate 解释；但只翻 exterior 后 Side Flat 从 50.0% 升到 75.0%，证明 camera layout 确实实质参与判断；
- color prior：存在明显 Red preference；
- side prior：原图存在极稳定的 Left preference；flip 后仍有 Left asymmetry，但 final all-Left 被打破，因此它不是唯一来源；
- layout / block side / curvature：不能单独解释，因为最终 all-Left 横跨所有 balanced cells；
- checkpoint prompt calibration、mirror OOD effect 与随时间变化的多模态 cue conflict：目前最符合证据，但这些来源尚未被单独分离。

## 建议

1. 正式任务结果保留 `pick up the blue/red block`，因为这才是收集时定义的目标；主结果使用 Flat，并把 prefix 与 final-local 分开报告。
2. `pick up the left/right block` 只作为 10%/20% 的 robot-frame diagnostic，不把它的最终 50% 当作有效 accuracy。
3. 下一步优先测试 no-motion baseline calibration，例如对每条轨迹使用 `m_t - m_start`，检查能否去除 Red/Left prompt prior；必须作为固定规则在新数据上验证，不能针对这 24 条调参。
4. 因为物体实际看起来更像 pink，可额外比较 `pick up the pink block` 与 `pick up the red block`，但应预先固定 prompt，不择优汇报。
5. 如果要直接证明 robot-state grounding，继续做 image 固定、state/action swap，以及 state-only 与 action-only swap。
6. 后续 instruction 应显式写参考系，例如 `on the robot-base left/right`，避免跨实验室 camera layout 带来的歧义。
7. 若要验证 camera-frame conflict，优先自然移动相机使 base-left=screen-left 后重采，而不是把镜像图像当作正式输入；本次 exterior flip 只应作为 OOD causal probe。

## 产物

主要表格：

- [Exterior-camera 水平反转配对报告](EXTERIOR_HFLIP_REPORT.md)
- [Exterior-flip 总体图](exterior_hflip/exterior_hflip_aggregate.png)
- [Exterior-flip 输入图像审计](exterior_hflip/exterior_hflip_visual_audit.png)
- [Blue/red per-episode summary](color24_episode_summary.csv)
- [Blue/red per-chunk curves and seed statistics](color24_chunk_curves.csv)
- [Blue/red method summary](color24_method_summary.csv)
- [Blue/red prefix summary](color24_prefix_summary.csv)
- [Blue/red factorial strata](color24_stratified_summary.csv)
- [Matched-pair summaries](color24_matched_pair_summary.csv)
- [Color-vs-side per-episode comparison](color_vs_side_episode_comparison.csv)
- [Color-vs-side summary](color_vs_side_summary.csv)
- [Side-prompt factorial strata](side_prompt_stratified_summary.csv)
- [Side-prompt matched-pair summary](side_prompt_matched_pair_summary.csv)
- [Geometry summary](color24_geometry_summary.csv)

Raw scores与 sanity：

- [Blue/red 10-seed raw scores](ten_seed/chunk_scores.csv)
- [Blue/red sanity summary](ten_seed/sanity_summary.json)
- [Left/right 10-seed raw scores](side_prompts/ten_seed/chunk_scores.csv)
- [Left/right sanity summary](side_prompts/ten_seed/sanity_summary.json)

可复现脚本：

- [Blue/red evaluator](../../examples/droid/evaluate_color_legibility_24.py)
- [Blue/red analyzer](../../examples/droid/analyze_color_legibility_24.py)
- [Left/right evaluator](../../examples/droid/evaluate_side_legibility_color24.py)
- [Left/right analyzer](../../examples/droid/analyze_early_weighted_legibility.py)
- [Color-vs-side comparator](../../examples/droid/compare_color_and_side_prompts_24.py)
- [Blue/red sanity evaluator](../../examples/droid/evaluate_color_legibility_sanity_24.py)
- [Left/right sanity evaluator](../../examples/droid/evaluate_side_legibility_sanity_color24.py)
- [Exterior-flip evaluator](../../examples/droid/evaluate_exterior_flip_color24.py)
- [Exterior-flip analyzer](../../examples/droid/analyze_exterior_flip_color24.py)
