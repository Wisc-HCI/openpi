# 24 条轨迹：仅反转外部相机的配对消融

## 核心结论

这次结果说明：**外部相机的左右布局确实是原先 left/right 不一致的重要来源，但不是唯一来源，也不能简化成“模型完全按画面左右”或“模型完全按机械臂自身状态”。**

最直接的证据来自同一批 24 条轨迹上的 `left/right` counterfactual prompts：

- 原图 Flat、Linear、Exponential 都是 `12/24 = 50.0%`，并且 `24/24` 全预测 Left；
- 只水平反转 exterior image 后，Flat 变为 `18/24 = 75.0%`，Linear 和 Exponential 都变为 `19/24 = 79.2%`；
- 12 条 base-left 仍全部正确；12 条 base-right 从 `0/12` 提升到 Flat `6/12`、Linear/Exponential `7/12`；
- Flat 的 seed-pooled accuracy 从 `48.8%` 提升到 `75.8%`，不是只靠少数接近零的均值碰巧过界。

因此，原图中的 final all-Left 不能再解释为纯粹、固定的 Left prompt/checkpoint prior。外部视觉布局对它有显著贡献。

但效果并不是一个简单的坐标系开关：

- 10% side accuracy 反而从 `79.2%` 降到 `54.2%`，反转后第一 local chunk 有 `23/24` 预测 Left；
- 改善从 30% 后才明显，在 40% 达到 `83.3%`，到 100% 又降至 `75.0%`；
- C0 仍为 `4/8 = 50%`，C1 变成 `8/8 = 100%`，C2 Flat 为 `6/8 = 75%`；
- `66.5%` 的逐 chunk×seed margin 在反转后保持原符号，与原 margin 仍正相关，而不是整体反号；
- 反转后最后一个 local chunk 仅 `6/24` 正确，尽管完整累积分数为 `18/24`。

最符合全部结果的表述是：

> π0.5-DROID observer 同时使用 state/action、外部视觉动态、腕部视觉和 prompt prior。原相机布局造成的参考系冲突解释了相当一部分中后段 Right 轨迹的 Left evidence；但各线索的权重随轨迹进度和弯曲程度变化，水平镜像本身又是 OOD 干预，所以不能把结果归结为单一 base-frame 或 screen-frame grounding。

对于实际记录的 `blue/red` 指令，主结论基本不变：Flat accuracy 反转前后都是 `17/24 = 70.8%`，最后一个 local chunk 都是 `24/24` 正确。外部翻转会显著改变单条轨迹的 residual，但在平衡设计中改善与恶化互相抵消。

## 干预设计

对全部 24 条轨迹，仅执行：

```python
exterior_image = np.ascontiguousarray(exterior_image[:, ::-1])
```

保持不变：

- wrist image；
- robot joint/gripper state；
- ground-truth action chunk；
- chunk 边界及 `executed_steps`；
- seed 和 `rng_index = episode_index * 10000 + chunk_index`；
- 每组候选 prompt 内的 noise tensor、noisy action 和 flow timestep。

颜色与方向 prompt 分别为：

- `[pick up the blue block, pick up the red block]`；
- `[pick up the left block, pick up the right block]`。

使用官方 `pi05_droid`、`pi05_droid_finetune`、horizon 16、seeds `0,...,9`、每个 seed 八个 flow samples。反转条件共得到 `5,220` 行 score；两组 prompt 各有 `2,610` 行，其中各 `2,130` 行属于 pre-grasp。原图与反转图的 `(episode, phase, chunk, seed)` key 全部逐一匹配。

原图与反转图在两个独立进程中评分，但使用完全相同的 RNG key。此前同 checkpoint 的独立 original rerun 测得 mean absolute numerical drift 为 `0.0001435`、max 为 `0.0004641`；本次翻转的 mean absolute margin change 为 side `0.02311`、color `0.03257`，约为该 mean drift 的 `161×` 和 `227×`。因此效果不可能由独立运行抖动解释。

Margin 定义：

- color：`E_blue - E_red`，正值支持 Red；
- side：`E_left - E_right`，正值支持 Right。

正式准确率始终按用户定义的 robot-base left/right 计算。水平翻转是 causal probe，不是自然部署图像。

[查看四种 layout × blue-side 组合的原图/反转输入审计](exterior_hflip/exterior_hflip_visual_audit.png)。该图直接确认：原图中的 base-left 位于 exterior screen-right，镜像后才位于 screen-left；base-right 则相反。

## Left/right：prefix 结果

| Prefix | 原图 accuracy | Exterior flip accuracy | 原图 seed-pooled | Flip seed-pooled | 原图预测 Left | Flip 预测 Left |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 79.2% | 54.2% | 71.3% | 54.2% | 17/24 | 23/24 |
| 20% | 66.7% | 62.5% | 64.2% | 62.1% | 20/24 | 21/24 |
| 30% | 50.0% | 70.8% | 57.5% | 71.3% | 24/24 | 19/24 |
| 40% | 50.0% | 83.3% | 53.8% | 80.4% | 24/24 | 16/24 |
| 100% | 50.0% | 75.0% | 48.8% | 75.8% | 24/24 | 18/24 |

反转没有改善最早期判断，反而把 10% 几乎变成 all-Left。真正的 Right recovery 出现在 30%--40%。这说明原图中被相机布局干扰的主要是**运动展开后的视觉动态线索**，而不是一个从首帧开始就稳定存在的 screen-coordinate lookup。

40% 是反转条件下的最好 prefix：

- base-left `12/12`；
- base-right `8/12`；
- L1、L3 都是 `10/12`；
- C1、C2 都是 `8/8`，C0 仍为 `4/8`。

到 100% 时有两条 Right trajectory 再次发生 late reversal，所以从 `20/24` 降到 `18/24`。

## Left/right：最终 weighted score

| Method | 原图 overall | Flip overall | Flip Left | Flip Right | 原图 seed-pooled | Flip seed-pooled | Flip 预测 Left |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 50.0% | 75.0% | 100.0% | 50.0% | 48.8% | 75.8% | 18/24 |
| Linear | 50.0% | 79.2% | 100.0% | 58.3% | 51.3% | 80.4% | 17/24 |
| Exponential (`alpha=3`) | 50.0% | 79.2% | 100.0% | 58.3% | 54.6% | 74.6% | 17/24 |

Linear 在本次反转条件下最好：相比 Flat 多恢复一条 Right trajectory，同时 seed-pooled accuracy 也是最高的 `80.4%`。这与原图下三种方法全部 all-Left 不同，说明 early/mid weighting 只有在视觉参考系冲突部分缓解后才开始有用。

### 按 trajectory curvature

| Condition | 原图 Flat | Flip Flat | Flip Linear | Flip Exponential |
|---|---:|---:|---:|---:|
| C0 | 4/8 | 4/8 | 4/8 | 4/8 |
| C1 | 4/8 | 8/8 | 8/8 | 8/8 |
| C2 | 4/8 | 6/8 | 7/8 | 7/8 |

C0 完全没有改善；所有提升都来自 C1/C2。静态 target screen position 并不足以解释这一结构，因为它在 C0/C1/C2 中相同。更可能的原因是弯曲运动在 exterior image 中产生了可被镜像改变的动态方向 cue。

L1 与 L3 的 Flat 都从 `6/12` 提升到 `9/12`，所以效果不是由某一个 x 位置单独驱动。

## 每条轨迹的 side 判断

`Prefix` 五个字符依次为 10/20/30/40/100%；`F/L/E` 为 Flat/Linear/Exponential；seed 为十个 seed 中 Flat 支持 base-frame GT 的比例。

| Episode suffix | GT color/side | C | 原图 prefix | Flip prefix | 原图 F/L/E | Flip F/L/E | Flat seed 原/Flip |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| `00_40_18...L1_left_C0` | B/L | C0 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_40_47...L1_left_C1` | B/L | C1 | LLLLL | LLLLL | LLL | LLL | 100% / 60% |
| `00_41_27...L1_left_C2` | B/L | C2 | LLLLL | LLLLL | LLL | LLL | 50% / 100% |
| `00_45_28...L1_right_C0` | R/R | C0 | LLLLL | LLLLL | LLL | LLL | 0% / 0% |
| `00_45_56...L1_right_C1` | R/R | C1 | RLLLL | LRRRR | LLL | RRR | 0% / 100% |
| `00_46_25...L1_right_C2` | R/R | C2 | RRLLL | LRRRR | LLL | RRR | 0% / 100% |
| `00_48_13...L1_left_C0` | R/L | C0 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_49_05...L1_left_C1` | R/L | C1 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_49_36...L1_left_C2` | R/L | C2 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_50_22...L1_right_C0` | B/R | C0 | RRLLL | LLLLL | LLL | LLL | 0% / 0% |
| `00_50_52...L1_right_C1` | B/R | C1 | RLLLL | RRRRR | LLL | RRR | 0% / 90% |
| `00_51_25...L1_right_C2` | B/R | C2 | LLLLL | LLRRL | LLL | LRR | 0% / 60% |
| `00_52_54...L3_right_C0` | B/R | C0 | LLLLL | LLLLL | LLL | LLL | 0% / 0% |
| `00_53_23...L3_right_C1` | B/R | C1 | RRLLL | LLLRR | LLL | RRR | 0% / 100% |
| `00_53_52...L3_right_C2` | B/R | C2 | RLLLL | LLLRL | LLL | LLL | 0% / 30% |
| `00_54_32...L3_left_C0` | R/L | C0 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_55_01...L3_left_C1` | R/L | C1 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_55_32...L3_left_C2` | R/L | C2 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_56_19...L3_right_C0` | R/R | C0 | LLLLL | LLLLL | LLL | LLL | 20% / 10% |
| `00_56_48...L3_right_C1` | R/R | C1 | RRLLL | LLLRR | LLL | RRR | 0% / 70% |
| `00_57_15...L3_right_C2` | R/R | C2 | LLLLL | LLRRR | LLL | RRR | 0% / 100% |
| `00_57_54...L3_left_C0` | B/L | C0 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_58_22...L3_left_C1` | B/L | C1 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |
| `00_59_48...L3_left_C2` | B/L | C2 | LLLLL | LLLLL | LLL | LLL | 100% / 100% |

反转后的 Flat 有 20/24 条达到至少 80% seed prediction agreement。四条未达到的是：

- L1-left-C1 Blue：Left，6/10；
- L1-right-C2 Blue：均值仍 Left，但 6/10 seed 支持 Right；
- L3-right-C2 Blue：Left，3/10 支持 Right；
- L3-right-C1 Red：Right，7/10。

因此最稳健的改善是四条 C1 Right 中的三条以及两条 seed-stable C2 Right；接近零的边界轨迹不能过度解释。

## 逐 margin 的 paired effect

统计单位为 pre-grasp `chunk × seed`，两组 prompt 各 `2,130` 个严格同 key 配对。

| Prompt family | Mean signed delta | Mean absolute delta | 保持原符号 | 反号 | Mean episode corr(original) |
|---|---:|---:|---:|---:|---:|
| Blue/red | `+0.00214` | `0.03257` | 73.1% | 26.9% | `+0.450` |
| Left/right | `+0.01313` | `0.02311` | 66.5% | 33.5% | `+0.419` |

Side margin 整体向 Right 移动，但并非简单反号：

- base-left 的 mean local delta 也是 `+0.00804`，即 Left evidence 平均变弱，但最终没有翻成 Right；
- base-right 的 mean local delta 更大，为 `+0.01827`，足以让部分 C1/C2 轨迹翻成 Right；
- base-right 的同符号率只有 55.8%，而 base-left 为 77.1%，说明视觉干预主要重构了 Right trajectory 的 evidence。

如果模型只按 exterior screen coordinate，水平镜像应让大部分 margin 近似反号；实际并没有。如果模型只按 state/action，图像反转应几乎无效；实际变化又远大于数值噪声。两种纯假设都不成立。

## Blue/red 主任务

| Decision | 原图 accuracy | Flip accuracy | 原图 seed-pooled | Flip seed-pooled | 改善条数 | 恶化条数 |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 41.7% | 45.8% | 41.7% | 42.1% | 7 | 6 |
| 20% | 37.5% | 33.3% | 36.3% | 37.1% | 4 | 5 |
| 30% | 29.2% | 33.3% | 29.2% | 35.4% | 4 | 3 |
| 40% | 29.2% | 41.7% | 26.3% | 39.2% | 6 | 3 |
| 100% / Flat | 70.8% | 70.8% | 66.7% | 74.6% | 2 | 2 |
| Linear | 33.3% | 50.0% | 40.0% | 54.6% | 6 | 2 |
| Exponential | 29.2% | 41.7% | 34.2% | 45.4% | 5 | 2 |

Flat 的四个 prediction changes 恰好互相抵消：

- 两条原本错误的 base-right/C0/Blue 变正确；
- 两条原本正确的 base-left/C2 变错误。

颜色 Flat overall 不变并不代表模型忽略 exterior image：逐 margin 的 mean absolute change 是 `0.03257`，Flat 有 4/24 改变类别，10% 有 13/24 改变类别。只是平衡设计把方向相反的 effect 抵消了。

另一方面，反转前后最后一个 color local chunk 都是 `24/24` 正确。这再次说明实际 blue/red 任务最可靠的 object evidence 出现在接近目标的后段，不能因为 side prompt 的改善就把正式任务替换成 left/right。

## 与此前两条轨迹实验的关系

8 月 17 日最后两条 centered-camera 轨迹中，只翻 exterior：

- base-left 的 Left evidence 增强；
- base-right 从 Left 翻成 Right。

24 条实验对第二点给出了**部分推广**：两个 layout 中的多条 C1/C2 base-right 都恢复成 Right，说明 camera/reference-frame conflict 确实可重复。

但第一点没有总体推广：24 条中 base-left margin 平均反而向 Right 移动，只是仍保持 Left 分类；而且 C0 base-right 完全没有恢复。这迫使我们把原先较强的说法收窄为：

> exterior camera layout 会实质影响 left/right residual，且能解释一部分 curved Right trajectory 的 late Left bias；它不是一个对所有位置、轨迹阶段和曲率都一致的坐标变换。

## 对核心论据的判断

### “左右判断是否根据机械臂自身状态规定？”

结果仍支持 observer 中存在 robot-centric-compatible cue：原图最早 10% 的 base-frame accuracy 为 79.2%，尽管 exterior screen frame 与 base frame 相反。

但结果不能证明左右由 state 单独规定：只改 exterior image 就把 side Flat 改变 25 个百分点，并重构大量逐 chunk margin。Evaluator 还同时看到 state、action、wrist image 和 exterior image，剩余 evidence 不能全部归因于 proprioception。

### “相机布局是否解释左右不一致？”

**解释相当一部分，但不是全部。** 最强证据是 final all-Left 被打破，C1/C2 Right 得到系统恢复；反证是 C0 不改善、10% 变差、末 local evidence 变差以及仍有 5--6 条 Right 最终错误。

### “是否仍有 fixed Left prior？”

**有，但不再能视为唯一解释。** 反转后：

- 10% 仍有 23/24 Left；
- Flat 仍有 18/24 Left；
- 所有 C0 Right 仍错误；
- 12 条 Left 全正确，而 Right 只有一半左右正确。

因此 residual 中仍存在明显 Left asymmetry。它可能来自 prompt/checkpoint calibration、OOD mirror、wrist/state/action cue，或这些因素的组合。

## 结论边界

- 水平镜像会反转文字、纹理、机械臂视觉几何和光照方向，属于 checkpoint 训练分布之外的干预；不能把 79.2% 当作自然相机部署 accuracy。
- 只有 24 条轨迹，每个 factorial cell 一条。C1/C2 interaction 是很清楚的描述性结构，但还不是跨采集批次的总体统计结论。
- 原图与 flip 共用 seed/RNG key，但不是同一进程内同时前向；已用独立 rerun jitter 定量确认 effect size 远大于该误差。
- 本实验隔离了 exterior image，却没有在 state、action、wrist image 之间继续拆分。

## 建议

1. 正式任务继续使用真实采集的 `pick up the blue/red block` 与原始图像；镜像只作为诊断，不作为性能增强。
2. left/right prompt 必须显式写参考系，例如 `robot-base left/right` 或 `camera-image left/right`。当前结果已经证明未指定参考系会实质改变结论。
3. 下一步若要直接验证 robot-centric grounding，应固定图像后分别交换 state、action，并做 wrist/exterior 的独立 swap；这比继续增加 mirror 变体更有辨别力。
4. 若重新采集，用自然方式把相机放到 base-left=screen-left 的位置，再重复 C0/C1/C2。这样可以检验本次 mirror 结果是否来自真实参考系对齐，而不是 OOD 镜像伪影。

## 产物

- [原图/Flip 总体对比图](exterior_hflip/exterior_hflip_aggregate.png)
- [原图/Flip 输入图像审计](exterior_hflip/exterior_hflip_visual_audit.png)
- [24 条轨迹的颜色与方向配对曲线](exterior_hflip/paired_trajectory_curves/)
- [逐 episode × decision 对比](exterior_hflip/episode_decision_comparison.csv)
- [Prefix 与 method 总表](exterior_hflip/decision_summary.csv)
- [Layout、target、color、condition 分层表](exterior_hflip/stratified_summary.csv)
- [逐 episode paired effect](exterior_hflip/episode_paired_effects.csv)
- [逐 chunk paired effect](exterior_hflip/chunk_paired_effects.csv)
- [Paired effect 分层总表](exterior_hflip/paired_effect_summary.csv)
- [5,220 行反转 raw scores](exterior_hflip/ten_seed/chunk_scores.csv)
- [配对与运行配置](exterior_hflip/ten_seed/scoring_config.json)
- [分析诊断与 key 校验](exterior_hflip/analysis_diagnostics.json)
- [Evaluator](../../examples/droid/evaluate_exterior_flip_color24.py)
- [Analyzer](../../examples/droid/analyze_exterior_flip_color24.py)
