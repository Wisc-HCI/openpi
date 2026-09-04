# pi0.5-LIBERO Legibility 微调 checkpoint：双物体 Pick-up Steering Pilot

## 1. 结论摘要

本次使用 checkpoint：

`/workspace/checkpoints/checkpoints/pi05_libero_legibility_finetune/libero_legibility_v1/19999`

主结果采用 400 policy steps 的 horizon、每个“方法 × 指令”5 条轨迹，共 40 条。所有方法均使用完全相同的 simulator 初始状态；同一 target/init-state 下，各方法每个 policy query 的 sampling seed 前缀也完全相同。

主要观察如下：

1. `time_decay, gamma=0.9` 与固定 `belief, T=1, lambda=2` 的整体结果确实非常接近：两者均为 9/10 成功；只统计成功轨迹时，平均长度分别为 246.4 和 247.0 steps，EEF 路径分别为 0.528 m 和 0.530 m。
2. `time_decay, gamma=0.5` 明显更适合这个任务的时间尺度：10/10 成功，平均 218.1 steps、0.486 m，接近或略优于 Base 的 10/10、227.7 steps、0.504 m。
3. Belief observer 在共享前缀阶段没有过早变得自信。第 2 个 query 的平均正目标 belief 约为 0.500；前 10 个 queries 的平均值仅为 0.503（cream cheese）和 0.505（tomato sauce）。因此 `T=1` 达到了预期的“微小几何差异不足以立即形成强证据”。
4. Belief steering 在产生目标方向差异时给出了最强的几何 intent evidence，但同时明显降低了效率，并在 tomato-sauce 的一个 seed 上直到 400 steps 仍未成功。因此，本 pilot 支持“belief 能按状态保留 steering”，但尚不支持“固定 lambda=2 在成功率和效率上优于调好的 time decay”。
5. 220/300-step 的初始结果受到严重 horizon censoring，不能作为论文主结果。训练 demo 的 P95 长度为 cream cheese 375.2 帧、tomato sauce 364.4 帧，因此主 pilot 改用 400 steps。

## 2. 数据与实验设置

微调数据集为 `Wisc-HCI/libero_legibility_v1`：

- 总计 200 条 demo；两个指令各 100 条。
- `pick up the cream cheese`：平均 260.24 帧，P95 375.2 帧，最大 526 帧。
- `pick up the tomato sauce`：平均 251.67 帧，P95 364.4 帧，最大 560 帧。
- 两个 BDDL 共享同一场景、物体布局和初始状态定义，只改变 `Up` goal 与语言指令。

四种方法：

| 条件 | 配置 |
|---|---|
| Base | 不使用 negative prompt，W=0 |
| Time decay 0.9 | W(k)=1.0×0.9^k，首 chunk 立即 steering |
| Time decay 0.5 | W(k)=1.0×0.5^k，首 chunk 立即 steering |
| Belief | T=1，lambda=2，8 residual samples，tau∈[0.3,0.7]，首 chunk W=0；之后 W=2×b_negative |

每个 target 使用另一个指令作为 negative prompt。Policy 每 5 个动作重新规划一次，action horizon 为 10。

## 3. 主结果（H=400，n=5/target/method）

| 方法 | Cream | Tomato | 总成功率 | 成功轨迹 steps | 成功轨迹 EEF path | Action TV（全部） |
|---|---:|---:|---:|---:|---:|---:|
| Base | 5/5 | 5/5 | 10/10 | 227.7 | 0.504 m | 9.212 |
| Time decay 0.9 | 5/5 | 4/5 | 9/10 | 246.4 | 0.528 m | 14.349 |
| Time decay 0.5 | 5/5 | 5/5 | 10/10 | 218.1 | 0.486 m | 9.145 |
| Belief | 5/5 | 4/5 | 9/10 | 247.0 | 0.530 m | 11.514 |

样本量只有每组5条，上表只能用于筛选参数和发现 failure mode，不能用于统计显著性结论。

配对结果中，Base 成功而 `time_decay 0.9` 失败的是 tomato init-state 3；Base 成功而 Belief 失败的是 tomato init-state 0。`time_decay 0.5` 与 Base 的所有成败结果一致。

## 4. 几何 Legibility 指标

对每条轨迹定义：

`e_t = [(d_alt(t)-d_true(t))-(d_alt(0)-d_true(0))] / ||p_true-p_alt||`

其中 `d_true` 和 `d_alt` 是 EEF 到真实目标与竞争目标初始位置的距离。正值表示 EEF 的运动相对更支持真实指令；0 附近表示尚未从几何运动中显露目标。

| 方法 | Cream e@20 | Cream e@50 | Tomato e@20 | Tomato e@50 |
|---|---:|---:|---:|---:|
| Base | -0.003 | -0.012 | 0.009 | 0.076 |
| Time decay 0.9 | 0.001 | 0.021 | 0.018 | 0.128 |
| Time decay 0.5 | -0.001 | -0.008 | 0.013 | 0.086 |
| Belief | 0.001 | 0.050 | 0.019 | 0.162 |

第 20 步时四种方法的 evidence 都接近 0，支持“初始部分是共享且难以区分的”这一实验设定。到第 50 步，Belief 对两个目标都产生了最大的正 evidence：相对 Base，cream 增加 0.062，tomato 增加 0.086。这是当前结果中最直接支持 belief-based steering 的几何证据。

不过，e@50 更高没有自动转化为更短或更稳的完整任务轨迹。Belief 后续轨迹更长，说明几何 legibility 与 task efficiency/success 必须分别报告，不能用一个指标替代另一个。

## 5. 实际 Belief 与 Steering 权重

Belief 条件：

| Target | b_positive@query2 | 前10 queries平均 b_positive | 最终 b_positive | 平均 W | 抬起3 cm时 W |
|---|---:|---:|---:|---:|---:|
| Cream cheese | 0.500 | 0.503 | 0.611 | 0.862 | 0.784 |
| Tomato sauce | 0.500 | 0.505 | 0.682 | 0.786 | 0.643 |

这说明 observer 的 softmax/recursive belief 行为是正确的：两个 candidate belief 始终归一化为1，并且共享前缀没有被微小差异迅速放大。`T=1` 比此前较低温度更符合该任务。

同时，lambda=2 意味着当 belief 仍为 0.5/0.5 时 W=1。即使到真正抬起物体时，negative belief 仍约为 0.392（cream）和 0.322（tomato），所以 W 仍约为 0.784 和 0.643。

对比之下：

| Target | Time decay 0.9 抬起时 W | Time decay 0.5 抬起时 W | Belief 抬起时 W |
|---|---:|---:|---:|
| Cream cheese | 0.027 | 约0 | 0.784 |
| Tomato sauce | 0.005 | 约0 | 0.643 |

因此，本实验非常清楚地验证了 scheduling 差异：time decay 在真正抓取/抬起阶段几乎不再 steering；belief 仍按当前不确定性保留较强 steering。当前未验证的是“这种较强 steering 一定改善最终任务表现”；事实上，它提高了早期 intent evidence，但也增加了轨迹长度，并产生一个 tomato failure。

## 6. 为什么必须使用 H=400

初始 H=220 的 n=10 pilot 和 H=300 的 n=5 pilot 都低估了成功率。典型 belief/tomato 轨迹在第220步仍朝正确目标靠近，但尚未闭合 gripper，并非走向错误目标。

成功率随 horizon 的变化：

| Horizon | Base | Time decay 0.9 | Time decay 0.5 | Belief |
|---|---:|---:|---:|---:|
| 220（n=10/target） | 11/20 | 10/20 | 11/20 | 8/20 |
| 300（n=5/target） | 8/10 | 6/10 | 8/10 | 6/10 |
| 400（n=5/target，主结果） | 10/10 | 9/10 | 10/10 | 9/10 |

前两行不能与主结果直接做统计比较，因为 n 和 horizon 不同；它们只用于证明评估存在右删失。正式实验应固定 H=400 或更长，并同时报告 success-conditioned efficiency 与 capped failure 数量。

## 7. 当前能支持和不能支持的论文结论

可以支持：

- 这组数据确实存在较长的共享前缀；T=1 的 observer 在该阶段保持接近 0.5/0.5。
- Belief weighting 不依赖绝对时间，而是在共享前缀后仍保留 steering。
- Time decay 对 gamma 很敏感：0.9 在这个任务上衰减过慢，0.5 明显更合适。
- 固定 belief 配置与未调好的 time-decay 0.9 在成功率和效率上非常接近，同时 belief 产生更强的第50步几何 intent evidence。

尚不能支持：

- Belief 在任务成功率或效率上优于经过验证集调参的 time decay；当前最佳是 gamma=0.5。
- lambda=2 完全不需要任何验证。它是通过“初始 b_negative=0.5 时令 W=1”得到的原则性设定，但当前结果显示持续的高 W 有效率代价。
- 仅凭 n=5 就对成功率差异作显著性结论。
- 两种物体在控制难度上完全等价。数据条数相同只能控制语言/task prior，不能消除 cream-cheese 与 tomato-sauce 的几何抓取差异。

## 8. 正式实验建议

1. 不要现在直接扩展到600条。先固定 H=400，并用额外 held-out seeds 将 pilot 扩至至少20条/组，确认 tomato init-state 0 的 failure 是否稳定复现。
2. Time-decay 主 baseline 应在独立 validation seeds 上选择 gamma，而不是在 test 上挑最好结果。建议验证 `{0.3, 0.5, 0.7, 0.9}`，然后冻结一个 gamma 用于所有正式测试任务；0.9 可作为 sensitivity ablation 保留。
3. Belief 主方法可固定 `T=1, lambda=2`，明确说明 lambda=2 来自初始尺度匹配而非 test tuning。同时加入 `lambda=1` 或“只 steering arm dimensions、不 steering gripper”作为诊断性 ablation，以确认长期高 W 的效率代价来自空间运动还是 gripper timing。
4. 论文中同时报告三类指标：任务成功率；成功轨迹 steps/path/action-TV；分阶段几何 evidence（例如 e@20、e@50、e@100 及达到目标附近时的 W/belief）。
5. 若论文核心要证明“共享 manipulation prefix 后才分歧”，比两个不同物体更干净的正式场景仍是“同一物体、两个不同 destination”，因为它能控制抓取几何差异。本任务可以作为补充的 object-choice ambiguity 实验。

## 9. 输出文件

- `PILOT_REPORT.md`：自动生成的简表。
- `pilot_report.json`：完整 group summary、配对成败结果和权重/belief统计。
- `pilot_episode_metrics.csv`：逐 episode 的 success、长度、平滑性、几何 evidence、belief 和 W。
- 每条 trajectory 的 `.npz`：完整 simulator state、body/site geometry、query observations、actions 和 diagnostics。

H=400 主 pilot 未保存 MP4，以避免额外空间开销。早先 H=220 的诊断目录保留了视频，可用于浏览共享前缀和 failure behavior，但不应使用其中的成功率作为论文结果。
