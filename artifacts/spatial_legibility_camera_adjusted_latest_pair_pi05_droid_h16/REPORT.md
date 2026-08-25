# 8 月 17 日最后两条居中相机轨迹：left/right 参考系审计

## 结论

这两条轨迹的结果需要分成两层看：

1. **如果仍使用机械臂 base-frame 标签，Flat 最终结果和之前一样：两条都预测 Left，准确率 1/2。** Linear 也一样。
2. **时间过程不一样。** 最新的 base-right 轨迹在 10%、20%、30%、40% 都被正确预测为 Right，到后半程才反转成 Left。此前 8 条中的对应 L1/C2 right 轨迹在 20% 就已经反转。
3. **Exponential early weighting (`alpha=3`) 的十-seed 均值在 base-frame 下达到 2/2**：base-left 为 Left，base-right 为 Right。但 base-right 的最终指数分数只有 `+0.00203`，且十个 seed 恰好 5 个支持 Right、5 个支持 Left，因此不能把 2/2 当成稳定恢复。
4. **把标签简单改成外部相机画面左右并不能解释结果。** 画面标签下，10%--40% 的均值准确率反而从 base-frame 的 2/2 变成 0/2；Flat 最终仍是 1/2，只是正确的是另一条轨迹；Exponential 均值从 base-frame 的 2/2 变成 0/2。

因此，你指出的语言歧义是真实且重要的实验设计问题，但这两条数据不支持“模型其实一直按外部相机画面左右理解，所以之前才不一致”这个简单解释。更准确的结论是：**camera layout 会改变 residual evidence 的时间结构，但 observer 的 left/right 语义并不等于外部画面的 screen-left/screen-right。**

## 两套参考系

metadata 中的标签由机械臂 base 前方轴定义：

| 轨迹 | Base-frame target | 目标 base `y` | 外部相机画面中的位置 |
|---|:---:|---:|:---:|
| `2026_08_17_02_29_03_938361_L1_left_C2` | Left | `+0.0424 m` | Right |
| `2026_08_17_02_30_47_786891_L1_right_C2` | Right | `-0.0424 m` | Left |

外部相机画面确认了这个翻转：

- [Base-left 轨迹：首帧、中段、近抓取帧](camera_view_base_left.png)——机械臂最终到达画面右侧方块。
- [Base-right 轨迹：首帧、中段、近抓取帧](camera_view_base_right.png)——机械臂最终到达画面左侧方块。

但 π0.5-DROID observer 并不只看这一个画面。当前 evaluator 同时输入：

- exterior camera；
- wrist camera；
- joint/gripper state；
- 被评分的 action chunk。

而且 [wrist-base-left](wrist_view_base_left.png) 与 [wrist-base-right](wrist_view_base_right.png) 的画面会随手腕运动旋转，screen-left/right 不是固定参考系。因此，即使外部相机中左右与 base-frame 相反，也不能直接推断模型 prompt 应当按外部画面重标。

## 评分设置

- 只使用上述最后两条轨迹，不包含此前 8 条。
- 官方 `pi05_droid` checkpoint，matched 16-action config。
- pre-grasp 共 10 个 chunk；post-grasp 不参与分析。
- seeds `0,...,9`，每 seed 八个 paired-noise flow samples；两条候选 prompt 使用相同 noise、flow timestep、action 和 noisy action。
- margin 为 `m = E_left - E_right`：正值支持 Right，负值支持 Left。
- Prefix 使用 chunk-stop 累积 margin 的线性插值。首个实际端点为 10.26%，所以 10% 只有很小的插值误差。

## Prefix 结果：base-frame 与 exterior-camera frame

“Camera frame” 是把两条轨迹的 GT 左右互换后的反事实计分，不重新运行模型。

| Prefix | Base-frame mean accuracy | Camera-frame mean accuracy | Base seed-pooled | Camera seed-pooled |
|---|---:|---:|---:|---:|
| 10% | 100% | 0% | 100% | 0% |
| 20% | 100% | 0% | 95% | 5% |
| 30% | 100% | 0% | 95% | 5% |
| 40% | 100% | 0% | 80% | 20% |
| 100% | 50% | 50% | 50% | 50% |

最关键的是 10%--40%：模型在这段时间的 evidence 明确更符合 **base-frame 标签**，而不是外部画面标签。到完整轨迹时两条都累积成 Left，因此无论采用哪套参考系都只有 1/2。

## Flat、Linear、Exponential

| Method | Base-frame mean accuracy | Camera-frame mean accuracy | Base seed-pooled | Camera seed-pooled |
|---|---:|---:|---:|---:|
| Flat | 50% | 50% | 50% | 50% |
| Linear early-weighted | 50% | 50% | 60% | 40% |
| Exponential (`alpha=3`) | 100% | 0% | 75% | 25% |

逐轨迹结果：

| Trajectory | Base GT | Camera GT | 10% | 20% | 30% | 40% | 100% | Flat | Linear | Exp (`alpha=3`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| `...02_29_03...L1_left_C2` | L | R | L | L | L | L | L | `-0.01982 (L)` | `-0.03453 (L)` | `-0.04489 (L)` |
| `...02_30_47...L1_right_C2` | R | L | R | R | R | R | L | `-0.01613 (L)` | `-0.00905 (L)` | `+0.00203 (R)` |

Seed 稳定性：

- Base-left：Flat、Linear、Exponential 均为 10/10 seeds 支持 Left。
- Base-right：Flat 为 10/10 支持 Left；Linear 为 8/10 支持 Left；Exponential 为 5/10 Left、5/10 Right。

所以 Exponential 的 base-frame 2/2 只是在均值上越过零点，不是稳定分类结果。

## Local margin 与反转位置

- Base-left local margins：`[-0.04831, -0.10630, -0.08836, -0.01096, +0.03315, +0.02071, +0.00854, -0.00010, -0.00551, -0.00105]`。[完整曲线](trajectory_curves/2026_08_17_02_29_03_938361_L1_left_C2.png)
- Base-right local margins：`[+0.06128, +0.00619, -0.01211, -0.03374, -0.05606, -0.03619, -0.05522, -0.02323, -0.01297, +0.00073]`。[完整曲线](trajectory_curves/2026_08_17_02_30_47_786891_L1_right_C2.png)

Base-right 的第一段 Right evidence 很强且 10/10 seeds 同号；第二段均值仍略微支持 Right，但只有 5/10 seeds 同号。第三段以后开始连续支持 Left。由于首段足够强，Flat 累积值一直到 40% 仍为正，约在 40%--50% 之间才穿过零点。这是清楚的 **early-correct / late-reversal**。

## 与此前 L1/C2 pair 的比较

此前 8 条数据中的 L1/C2 pair 与最新 pair 使用相同 base-frame target 定义和同一 checkpoint，但外部相机位置不同：

| Pair | Base-right 10% | 20% | 30% | 40% | 100% | Exp final |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| Earlier L1/C2 | R | L | L | L | L | `-0.02063 (L)` |
| Latest centered-camera L1/C2 | R | R | R | R | L | `+0.00203 (R)` |

所以答案不是“完全一样”：

- **一样的部分**：Flat 最终仍然全部偏 Left，base-right 最终仍错。
- **不同的部分**：居中相机这对轨迹在 10%--40% 可以正确区分两个 base-frame 目标；Exponential 均值还能保留这个信号到最终评分。

这说明 camera layout 很可能影响 observer，但由于轨迹也重新采集了，当前 pair 不能把变化严格归因于相机位置；motion、照明、起始状态和视频内容也同时有细小变化。

## 对“不同实验室、不同 camera layout 导致语言歧义”的判断

这个担心是合理的，尤其是 DROID 式数据同时包含多实验室、多视角和自然语言。如果数据生产者没有统一规定 `left/right` 是：

- robot-base frame；
- operator frame；
- exterior-camera frame；
- wrist-camera frame；
- 或场景中某个固定标记的 frame；

那么同一句 prompt 的监督确实可能相互冲突，最终表现为弱 margin、校准偏差或对 camera layout 敏感。

不过，本 pair 给出的证据更具体：

- 它**确认了当前 metadata 的 base-left/base-right 与外部画面左右相反**；
- 它**确认了换 camera layout 后 early evidence 明显变化**；
- 但它**否定了“模型只是按当前 exterior screen-left/right 理解”的简单版本**，因为 10%--40% 的模型预测恰好 2/2 符合 base-frame、0/2 符合 exterior-camera frame。

最稳妥的结论是：参考系歧义可能是跨实验室 checkpoint 的噪声来源，但当前 all-Left residual bias 不能仅靠交换这两个标签解释。

## 下一步最有区分度的纯离线检查

如果继续追查而不采新数据，最有信息量的不是再换一次 GT，而是对这两条做输入消融：

1. 只水平翻转 exterior image，保持 wrist/state/action/noise 完全不变；
2. 分别遮蔽 exterior 与 wrist image；
3. 比较完整输入、image-only 近似和 state/action 主导条件下的 margin 变化。

如果水平翻转 exterior 会稳定交换 left/right residual，才支持 exterior-camera-centric grounding；如果结果基本不变，则 evidence 更可能来自 wrist、proprioception/action dynamics 或 prompt prior。

这项检查现已完成。结果既不是稳定反号，也不是基本不变，而是：翻转 exterior 使 screen frame 与 base frame 对齐后，base-left evidence 增强、base-right 的 late reversal 消失。详见 [水平翻转图像消融报告](VISUAL_FLIP_REPORT.md)。

## 产物

- [Reference-frame per-episode summary](reference_frame_episode_summary.csv)
- [Reference-frame method summary](reference_frame_method_summary.csv)
- [Reference-frame prefix summary](reference_frame_prefix_summary.csv)
- [10-seed raw chunk scores](ten_seed/chunk_scores.csv)
- [All chunk statistics and cumulative curves](early_weighted_chunk_scores.csv)
- [Base-frame episode summary](early_weighted_episode_summary.csv)
- [Visual-flip causal ablation](VISUAL_FLIP_REPORT.md)

样本量只有一对，因此 2/2 只应作为描述性结果，尤其不能把 Exponential 的 2/2 当作 checkpoint 已经稳定解决左右语义。
