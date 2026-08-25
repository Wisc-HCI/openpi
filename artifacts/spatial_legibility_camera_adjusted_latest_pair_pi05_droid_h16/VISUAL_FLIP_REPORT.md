# 8 月 17 日最后两条轨迹：水平翻转图像消融

## 核心结论

翻转结果**部分支持“模型使用机械臂自身参考系”这个论据，但不支持“left/right 完全由机械臂 state 单独决定”**。数据最符合一个更具体的解释：

> observer 很可能同时使用了 robot-centric 的 state/action cue 和相机画面中的 spatial cue；在原图中两者的左右定义互相冲突，外部相机水平翻转后两种 cue 恰好一致，因此分类明显改善。

最关键的结果来自只翻转 exterior camera：

- Base-left 原本位于外部画面右侧；翻转后位于画面左侧，与 base-frame 的 Left 标签一致。Flat margin 从 `-0.01980` 变成 `-0.03078`，Left evidence 更强。
- Base-right 原本位于外部画面左侧；翻转后位于画面右侧，与 base-frame 的 Right 标签一致。Flat margin 从 `-0.01610 (L)` 变成 `+0.00219 (R)`，十个 seed 中支持 base-right 的比例从 `0/10` 变成 `7/10`。
- exterior-flip 条件下，Flat、Linear、Exponential 三种方法的轨迹均值都达到 base-frame `2/2`；seed-pooled accuracy 分别为 `85%`、`90%`、`95%`。

这不是“翻一下以后所有判断都机械地反号”。每个 chunk × seed 的 margin 在翻转 exterior 后仍有 `65%--79%` 保持原符号，而且与原 margin 正相关。因此，**纯 screen-left/right 模型**和**纯 state-only 模型**都不符合观测；多模态 cue conflict 是目前最有解释力、但仍属描述性的结论。

## 实验设计

只使用 8 月 17 日最后收集、外部相机已移到中间的两条轨迹：

| Trajectory | Base-frame GT | 原 exterior 画面位置 | exterior-flip 后画面位置 |
|---|:---:|:---:|:---:|
| `2026_08_17_02_29_03_938361_L1_left_C2` | Left | Right | Left |
| `2026_08_17_02_30_47_786891_L1_right_C2` | Right | Left | Right |

对每个 action chunk 比较四种 visual condition：

1. `original`：两路图像均不变；
2. `exterior_hflip`：只水平翻转外部相机图像；
3. `wrist_hflip`：只水平翻转腕部相机图像；
4. `both_hflip`：两路图像都水平翻转。

每个配对严格保持以下内容相同：

- robot state；
- ground-truth action chunk；
- random seed 和 noise tensor；
- flow timestep 和 noisy action。

唯一变化是对应图像沿宽度方向翻转。每条轨迹包含 10 个 pre-grasp chunk，使用 seeds `0,...,9`；每个 seed、chunk、condition 使用八个 paired-noise flow samples。总计 `2 × 10 × 10 × 4 = 800` 个 condition-level score。

margin 定义为 `m = E_left - E_right`：正值支持 Right，负值支持 Left。

## 两个假设原本应当预测什么

| Hypothesis | 翻转图像后的预期 |
|---|---|
| 纯 exterior screen-coordinate grounding | 只翻 exterior 后，两条轨迹的 margin 应大体反号，Left/Right 判断应一起交换 |
| 纯 robot-state/action grounding | 图像翻转不应显著改变 margin |
| 多模态 cue conflict | 翻转会显著改变 margin，但不必整体反号；当 screen cue 与 base cue 对齐时，base-frame 判断应增强 |

实际结果最符合第三项。

## 逐轨迹最终评分

括号内为均值预测；`seed` 列表示十个 seed 中支持 base-frame GT 的数量。

### Base-left trajectory

| Visual condition | Prefix 10/20/30/40/100% | Flat | Flat seed | Linear | Linear seed | Exp (`alpha=3`) | Exp seed |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Original | L/L/L/L/L | `-0.01980 (L)` | 10/10 | `-0.03449 (L)` | 10/10 | `-0.04485 (L)` | 10/10 |
| Exterior flip | L/L/L/L/L | `-0.03078 (L)` | 10/10 | `-0.04497 (L)` | 10/10 | `-0.04994 (L)` | 10/10 |
| Wrist flip | L/L/L/L/L | `-0.01495 (L)` | 10/10 | `-0.02263 (L)` | 10/10 | `-0.02770 (L)` | 10/10 |
| Both flip | L/L/L/L/L | `-0.03638 (L)` | 10/10 | `-0.05132 (L)` | 10/10 | `-0.05453 (L)` | 10/10 |

Base-left 在所有条件、方法和 seed 下都保持 Left。不过，只翻 exterior 后三种分数都更负，这与“翻转使画面位置也成为 Left，于是 visual cue 与 base label 一致”相符。

[查看 Base-left 的 local、cumulative 与 weighted-score 曲线](visual_flip/2026_08_17_02_29_03_938361_L1_left_C2_visual_flip.png)

### Base-right trajectory

| Visual condition | Prefix 10/20/30/40/100% | Flat | Flat seed | Linear | Linear seed | Exp (`alpha=3`) | Exp seed |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Original | R/R/R/R/L | `-0.01610 (L)` | 0/10 | `-0.00901 (L)` | 2/10 | `+0.00205 (R)` | 5/10 |
| Exterior flip | R/R/R/R/R | `+0.00219 (R)` | 7/10 | `+0.00769 (R)` | 8/10 | `+0.01208 (R)` | 9/10 |
| Wrist flip | R/R/R/L/L | `-0.00678 (L)` | 1/10 | `-0.00758 (L)` | 2/10 | `-0.00355 (L)` | 3/10 |
| Both flip | R/R/R/R/L | `-0.00476 (L)` | 0/10 | `+0.00213 (R)` | 8/10 | `+0.00865 (R)` | 9/10 |

Base-right 是最有区分度的轨迹。原图中它在前 40% 正确支持 Right，后半程却累积反转到 Left；只翻 exterior 后，10%--100% 均保持 Right，三种最终评分也全部转为 Right。也就是说，**外部画面的原始 screen-left cue 很可能参与了后期 Left reversal；翻转以后 screen-right 与 base-right 不再冲突，反转消失。**

[查看 Base-right 的 local、cumulative 与 weighted-score 曲线](visual_flip/2026_08_17_02_30_47_786891_L1_right_C2_visual_flip.png)

## 整体准确率

这里的 GT 始终按机械臂 base frame 计算。`Mean` 是两条轨迹的均值分类，`seed-pooled` 是两个轨迹 × 十个 seed 的分类率。

| Visual condition | Method | Base-frame mean accuracy | Base-frame seed-pooled accuracy |
|---|---|---:|---:|
| Original | Flat | 50% | 50% |
| Original | Linear | 50% | 60% |
| Original | Exponential | 100% | 75% |
| Exterior flip | Flat | 100% | 85% |
| Exterior flip | Linear | 100% | 90% |
| Exterior flip | Exponential | 100% | 95% |
| Wrist flip | Flat | 50% | 55% |
| Wrist flip | Linear | 50% | 60% |
| Wrist flip | Exponential | 50% | 65% |
| Both flip | Flat | 50% | 50% |
| Both flip | Linear | 100% | 90% |
| Both flip | Exponential | 100% | 95% |

只有 exterior flip 给出最干净的结果：三种方法均为 2/2，并且 base-right 的 late reversal 消失。Wrist 画面的 camera frame 会随手腕姿态改变，简单水平翻转并不对应稳定的 base-frame 反射，所以 wrist-only 和 both-flip 不宜作相同的几何解释。

## 翻转是否直接让 margin 反号

以下统计在每条轨迹的 `10 seeds × 10 chunks = 100` 个严格配对 margin 上计算。

| Trajectory | Flip | Mean delta vs original | Mean absolute delta | 保持原符号 | 反号 | Corr(original) |
|---|---|---:|---:|---:|---:|---:|
| Base-left | Exterior | `-0.01098` | `0.02064` | 79% | 21% | `+0.836` |
| Base-left | Wrist | `+0.00485` | `0.02172` | 74% | 26% | `+0.743` |
| Base-left | Both | `-0.01658` | `0.03345` | 75% | 25% | `+0.603` |
| Base-right | Exterior | `+0.01829` | `0.02669` | 65% | 35% | `+0.674` |
| Base-right | Wrist | `+0.00932` | `0.02349` | 70% | 30% | `+0.621` |
| Base-right | Both | `+0.01134` | `0.02393` | 69% | 31% | `+0.608` |

这组结果同时说明两件事：

- 图像确实影响 residual：mean absolute change 为 `0.0206--0.0335`，而 original 条件相对于之前独立运行的数值漂移只有 mean `0.00014`、max `0.00046`，不是运行抖动造成的假差异。
- 图像不是唯一依据：所有翻转条件下，大部分 paired margins 仍保持原符号，且 flipped margin 与 original margin 始终为正相关，而不是与 `-original` 正相关。

## 对论据的最终判断

### 支持到什么程度

结果支持以下较弱但更准确的论据：

> π0.5-DROID observer 中存在与机械臂 base-frame 标签相容、又不由 exterior screen coordinate 单独决定的 cue。Proprioceptive state、被评分的 action trajectory，或二者组合是很合理的来源；这个 cue 能在当前外部画面左右与 base 标签相反时，让两条轨迹在早期仍按 base frame 正确区分。

尤其是 base-right 的原始前 40% 已经支持 Right，尽管目标在外部画面左侧。这直接排除了“observer 在整段轨迹中只按外部画面左右判断”。

### 不能证明什么

这次实验不能证明 left/right **完全**由机械臂 state 规定：

- 只翻图像就会显著改变 margin，并把 base-right Flat 从稳定错误的 `0/10` 改成 `7/10` 正确，说明 visual cue 有实质作用。
- evaluator 同时输入 state 和 action；本实验没有把这两者彼此分离。
- Wrist image 和其他翻转后仍保留的视觉内容也可能提供轨迹方向信息，因此不能仅凭 exterior flip 把剩余 evidence 全部归因于 state/action。
- prompt/checkpoint 自身的 Left prior 仍可能与 state/action cue 混合；只翻图像无法单独排除它。
- 水平翻转会产生 checkpoint 训练分布之外的镜像视觉，结果应作为 causal probe，而不是自然场景 accuracy。
- 样本只有一左一右两条轨迹，结论是强描述性证据，不是总体统计结论。

因此最合适的表述不是“模型根据机械臂自身状态规定左右”，而是：

> **当前证据支持 observer 使用 robot-centric state/action 信息来解释左右，同时也使用相机画面中的左右信息；两个参考系冲突很可能正是原始 base-right late reversal 和最终 all-Left 现象的一部分来源。**

## 如果要进一步区分 state 与 action

下一步最直接的纯离线因果检查是做 cross-modal pairing：

1. 图像保持不变，在两条轨迹之间交换或镜像 state/action；
2. state/action 保持不变，交换两条轨迹的 exterior image；
3. 分别做 state-only swap 与 action-only swap。

如果 margin 跟着 state/action 的 base 方向移动，就能比本次图像翻转更直接地证明 robot-centric grounding。正式实验的 instruction 最好也显式写成 `robot-base left/right` 或 `camera-image left/right`，避免让不同实验室和 camera layout 共享一个未定义参考系的 `left/right`。

## 产物

- [800 条 paired raw scores](visual_flip/raw_scores.csv)
- [逐轨迹、条件、prefix 与 method 汇总](visual_flip/episode_summary.csv)
- [逐 chunk 汇总](visual_flip/chunk_summary.csv)
- [整体 method 汇总](visual_flip/method_summary.csv)
- [整体 prefix 汇总](visual_flip/prefix_summary.csv)
- [paired flip effect 汇总](visual_flip/paired_effect_summary.csv)
- [运行配置](visual_flip/config.json)
- [Evaluator](../../examples/droid/evaluate_spatial_legibility_visual_flip.py)
- [Analyzer](../../examples/droid/analyze_spatial_legibility_visual_flip.py)
