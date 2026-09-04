# 基于 Belief 的 Policy Steering：两组 LIBERO 实验完整阶段性报告

> 用途：导师汇报与后续论文实验设计讨论  
> Checkpoint：OpenPI `pi0.5-LIBERO`  
> 报告状态：阶段性结果，不是最终论文结论  
> 统计单位：正式实验每个 method-target 组合 100 条轨迹，由 50 个初始状态各重复 2 次构成

## 1. 执行摘要

目前完成了两类互补实验：

1. **早期即可区分意图的任务**：
   - `put the cream cheese in the bowl`
   - `put the bowl on the stove`
   - 两条指令操作的物体和目的地都不同，因此机器人从最初的 reaching motion 就可能暴露意图。

2. **共享前缀、后期目的地分叉的任务**：
   - `put the bowl on top of the cabinet`
   - `put the bowl on the plate`
   - 两条指令先操作同一个 bowl，理论上 reach-and-grasp 是 shared prefix，抓起以后才需要向不同目的地运动。

两个实验共同研究三种方法：

- **Base policy**：不做 CFG steering。
- **Time-decay**：Legibility-Diffuser-style baseline，初始权重为 1，每次重新生成 action chunk 后乘以 `0.9`。
- **Belief weighting**：根据已执行轨迹对正、负候选指令的 posterior belief 调整 steering 权重；第一个 chunk 强制 `W=0`，之后使用 `W=lambda * b_negative`。

当前结果可以概括为：

- 在早期即可区分意图的实验中，Belief 方法在第一个 chunk 后迅速降低 steering，整体行为接近 Base；Time-decay 在 cream-cheese 任务中造成更低的成功率、更长的路径和更多动作变化。这一结果支持“当 base motion 已经足够表达意图时，持续 steering 可能是不必要且有代价的”。
- 在共享 bowl 的实验中，低温度配置 `T=0.1, lambda=1` 过早将 belief 推到接近 1，导致真正抬碗时 `W` 接近 0，因此几乎没有改善抓取后的 geometric legibility。
- 提高到 `T=1.0, lambda=2` 后，Belief steering 在抬碗时得到恢复，并且相较旧 Belief 和 Base 明显改善了几何 legibility；但它没有优于当前 `gamma=0.9` 的 Time-decay，并在 cabinet 任务中增加了动作代价、降低了成功率。
- 因而，当前结果**支持 belief calibration 的必要性和早期明确任务中的自适应关闭能力**，但**还不能支持 Belief weighting 在 delayed-intent 任务中优于调好的 Time-decay**。

---

## 2. 方法与统一实验配置

### 2.1 Steering 形式

用正指令对应的 flow/noise prediction 记为 `s_positive`，竞争指令对应的 prediction 记为 `s_negative`。实验使用的 steering 形式可概括为：

```text
s_steered = s_positive + W * (s_positive - s_negative)
```

三种方法的区别仅在于 `W`：

```text
Base:        W_c = 0
Time-decay:  W_c = W_0 * gamma^c
Belief:      W_c = lambda * b_c(negative)
```

其中 `c` 是 action-chunk/replanning index。两个候选指令经过 softmax 后满足：

```text
b_c(positive) + b_c(negative) = 1
```

因此两候选情况下：

```text
W_c = lambda * (1 - b_c(positive))
```

### 2.2 三个正式实验组

| 组别 | 配置 | 解释 |
|---|---|---|
| Base | 无 negative prompt，`W=0` | 任务成功率和运动效率参考 |
| Time-decay | `W0=1.0`，`gamma=0.9`，首 chunk 立即 steering | 时间是唯一调度变量，不计算 belief |
| Belief | 8 次 belief samples，flow tau 范围 `[0.3, 0.7]`，7 个物理 action dimensions，首 chunk `W=0` | 使用上一个实际执行 chunk 的 residual/energy 更新 belief |

Belief 的两个正式参数版本为：

| 版本 | Temperature | Lambda | 状态 |
|---|---:|---:|---|
| 原始版本 | 0.1 | 1.0 | 两个正式实验均完成，每目标 100 条 |
| 增强版本 | 1.0 | 2.0 | 仅在第二个 bowl-destination 实验完成，每目标 100 条 |

此外还运行了下面的单初始状态参数预览：

| Temperature | Lambda | Cabinet | Plate | 统计地位 |
|---:|---:|---:|---:|---|
| 0.1 | 2.0 | 1 条 | 1 条 | Smoke test，不可用于成功率结论 |
| 0.3 | 2.0 | 1 条 | 1 条 | Smoke test，不可用于成功率结论 |
| 1.0 | 2.0 | 1 条 | 1 条 | Smoke test；随后已扩展为正式 200 条 |

**未运行的组合**包括 `T=0.3, lambda=1`、`T=1.0, lambda=1`，以及 `T=0.1, lambda=2` 的 100 条/目标正式评估。本报告不会为这些未运行组合推断结果。

### 2.3 配对与可复现设置

- Simulator seed：`7`。
- Sampling seed base：`20250822`。
- 每次 policy query 后执行 5 个 action steps，再重新规划。
- 正式实验：50 个 initialization，每个 initialization 重复 2 次，共 100 条/组。
- 同一 target 下，各方法复用相同初始状态和相同 sampling-seed prefix。
- 第一个实验严格审计通过：六组均为 100 条，初始 observation hash、sampling-seed prefix 和全部 paired units 一致。
- 第二个实验严格审计通过的核心条件包括：初始 simulator state、一致的首个实际 policy observation、受控 policy 输入、sampling-seed prefix 和 paired units。三组 raw reset render hash 存在不一致，但这些 reset render 不会作为 policy 输入；对应的首个实际 policy observations 完全一致。

---

## 3. 评价指标

### 3.1 两个实验共用的任务与运动指标

#### Success rate

LIBERO task predicate 是否在 episode 内完成。成功率是最重要的 feasibility 指标。

#### EEF path length

末端执行器在 policy-controlled rollout 中的累计三维位移：

```text
L_EEF = sum_t ||x_(t+1) - x_t||_2
```

单位为米。相同任务中，路径越长通常表示更大的运动代价或更多不必要偏移；但它本身不是 legibility 指标，因为有意的 legible exaggeration 也可能增加路径长度。

#### Action total variation（Action TV）

连续 action vector 之间变化量的累计值：

```text
TV = sum_t ||a_(t+1) - a_t||_2
```

数值越大表示动作变化、纠正或振荡越多。它是 smoothness/控制代价 proxy，不等价于人类主观自然度。

#### Policy steps

episode 实际执行的 model-generated action 数量。成功任务中 steps 越多，通常表示完成速度更慢。

#### Executed guidance weight

记录首 chunk、首 chunk 后最大 `W`，以及后续 chunks 的平均 `W`，用于验证调度器实际在何时施加 steering。

### 3.2 第二个实验新增的 phase-aligned 指标

第二个实验需要区分“共同抓取阶段”和“目的地 transport 阶段”。事件定义如下：

- **Close**：第一次执行 gripper close command。
- **Grasp**：LIBERO `_check_grasp` 第一次确认 bowl 已被抓住。
- **Lift onset**：close 之后，bowl 连续 3 个状态高于初始高度至少 5 mm 的第一个状态。

#### Pre-lift path / Pre-lift Action TV

只统计 policy 开始至 lift onset 之间的 EEF path 和 Action TV，用于衡量 steering 是否干扰本应共享的 reach-and-grasp 阶段。

#### Post-lift geometric intent evidence

从 lift onset 开始，对每个后续状态比较末端执行器相对于真实目标和竞争目标的相对进展，并用两个目标的距离归一化：

```text
e_k = [(d_true(0) - d_true(k)) - (d_alt(0) - d_alt(k))]
      / ||g_true - g_alt||
```

- `e_k > 0`：相对而言更朝真实目的地运动。
- `e_k < 0`：相对而言更朝竞争目的地运动。

#### AUC20

对 lift 后前 20 步的 `e_k` 做 early-weighted average，越早的动作权重越大。数值越大，表示机器人越早表现出朝真实目的地移动的几何趋势。

#### Wrong first 15

lift 后前 15 步中 `e_k < 0` 的比例。越低越好。

这些指标是 simulation 中的 geometric proxy，不能直接替代 human intent-recognition study。

---

## 4. 实验一：从运动开始就容易区分的两个任务

### 4.1 实验动机

两条实际执行的指令为：

1. `put the cream cheese in the bowl`
2. `put the bowl on the stove`

两条指令需要 reach 不同物体，并且最终 destination 也不同。因此它们不是 shared-object/shared-prefix 对照。从机器人开始接近物体时，base trajectory 本身就很可能提供充分意图证据。

该实验的主要问题不是“谁能制造最大的 legible deviation”，而是：

> 当 base policy 已经通过正常任务动作表达了目标时，调度器能否及时停止不必要的 steering，并维持 base policy 的成功率、路径效率和动作平滑性？

因此本实验主要使用 success、EEF path、Action TV、policy steps 和实际 `W` 来衡量。它没有使用第二个实验中的 post-lift destination AUC，因为两个任务甚至不操作同一个物体，无法定义同一个可比的 destination branch point。

### 4.2 正式配置

| 方法 | 参数 |
|---|---|
| Base | `W=0` |
| Time-decay | `W0=1.0, gamma=0.9`，首 chunk steering |
| Belief | `T=0.1, lambda=1.0`，首 chunk `W=0` |

每个 method-target 100 条，共 600 条。

### 4.3 正式结果：按任务报告

| Method | Target | Success | EEF path (m) | Action TV | Policy steps |
|---|---|---:|---:|---:|---:|
| Base | cream cheese -> bowl | 100/100 | 0.8210 | 15.2186 | 92.4 |
| Time-decay | cream cheese -> bowl | 95/100 | 0.9314 | 20.4984 | 111.7 |
| Belief | cream cheese -> bowl | 99/100 | 0.8242 | 15.4310 | 94.4 |
| Base | bowl -> stove | 100/100 | 0.8190 | 11.1727 | 87.7 |
| Time-decay | bowl -> stove | 100/100 | 0.8230 | 11.7988 | 88.0 |
| Belief | bowl -> stove | 100/100 | 0.8141 | 10.7474 | 86.6 |

### 4.4 两个目标的宏平均

由于两个 target 各有相同数量的 episodes，下面是简单宏平均：

| Method | Success | Mean EEF path (m) | Mean Action TV | Mean policy steps | Mean later W |
|---|---:|---:|---:|---:|---:|
| Base | 100.0% | 0.8200 | 13.1956 | 90.0 | 0.0000 |
| Time-decay | 97.5% | 0.8772 | 16.1486 | 99.8 | 0.4179 |
| Belief | 99.5% | 0.8192 | 13.0892 | 90.5 | 0.0335 |

### 4.5 Guidance 调度实际行为

| Method | Target | First chunk W | Max W after first | Mean later W |
|---|---|---:|---:|---:|
| Time-decay | cream cheese | 1.0000 | 0.9000 | 0.3930 |
| Time-decay | bowl/stove | 1.0000 | 0.9000 | 0.4428 |
| Belief | cream cheese | 0.0000 | 0.3617 | 0.0326 |
| Belief | bowl/stove | 0.0000 | 0.3668 | 0.0343 |

Belief 的 `W` 在后续 episode 中平均只有约 `0.033`，说明 observer 很快把 positive instruction 识别为高概率目标。Time-decay 与任务是否已经明确无关，仍按照预定曲线保持平均约 `0.42` 的后续 steering。

### 4.6 配对统计

在 cream-cheese 任务中：

- Time-decay 相比 Base 成功率下降 5 个百分点；按 init-state cluster bootstrap 的 95% CI 为 `[-11%, -1%]`，episode-level McNemar exact `p=0.0625`。
- 在双方都成功的 paired episodes 上，Time-decay 相比 Base：
  - EEF path 增加 `0.0812 m`，95% CI `[0.0441, 0.1201]`。
  - Action TV 增加 `2.433`，95% CI `[0.278, 4.567]`。
- Belief 相比 Base：
  - 成功率只下降 1 个百分点，区间包括 0。
  - EEF path 差异 `-0.0005 m`，区间包括 0。
  - Action TV 差异 `-0.007`，区间包括 0。
- Belief 相比 Time-decay 的成功率高 4 个百分点，但 95% CI `[-1%, 10%]` 且 McNemar `p=0.2188`，不能称为显著成功率优势。
- Belief 相比 Time-decay 的 paired EEF path 减少 `0.0816 m`，95% CI `[-0.1187, -0.0465]`。

在 bowl-to-stove 任务中，三组成功率都是 100%。运动指标差异较小；Belief 相比 Time-decay 的 Action TV 减少 `1.051`，95% CI `[-1.609, -0.583]`。

### 4.7 实验一支持的结论

该实验支持以下阶段性结论：

1. **Belief 调度能够在意图已经被 base motion 明确表达后主动降低 steering。** 其后续平均 `W` 约为 `0.033`，而 Time-decay 约为 `0.418`。
2. **不考虑当前意图歧义、只按时间施加较强 early steering 可能有实际代价。** cream-cheese 任务中出现了更长路径、更大 Action TV、更多 policy steps 和 5 个百分点的成功率下降。
3. **Belief 方法在该场景更接近 Base policy。** 宏平均路径、Action TV 和 steps 都基本回到 Base 水平。

### 4.8 实验一不能支持的结论

1. 结果不能证明 Belief 在所有“早期明确”任务中都优于 Time-decay；明显代价主要集中在 cream-cheese 任务，bowl-to-stove 的差异很小。
2. 结果不能证明 Belief 提高了 human-perceived legibility，因为该实验没有独立 human observer 或 goal-inference metric。
3. 结果不能证明任意 CFG 都会降低成功率；它只评价了当前 `W0=1, gamma=0.9` 的配置。
4. 结果不能排除 candidate instructions 的有效训练频率或 difficulty 不完全一致。候选先验相等是理论假设，但还需要独立的数据覆盖审计。

---

## 5. 实验二：共同抓 bowl，之后选择 cabinet 或 plate

### 5.1 实验动机

两条指令为：

1. `put the bowl on top of the cabinet`
2. `put the bowl on the plate`

它们操作同一个 bowl，正常任务流程为：

```text
shared reach -> shared grasp -> lift -> destination-specific transport -> place
```

理论假设是：如果 shared reach-and-grasp 的动作分布在两个候选指令下足够接近，那么 observer 在抓取前应保持较高的不确定性；Belief steering 应在这一阶段保留较大 `W`，并在抓起后帮助轨迹尽早向真实 destination 分离。Time-decay 与 state 无关，因此可能在真正的 destination split 发生前已经衰减。

### 5.2 固定 Base 与 Time-decay 的正式结果

Base 与 Time-decay 只正式运行一次；新 Belief 参数实验复用了完全相同的 400 条 controls。

| Method | Target | Success | EEF path (m) | Action TV | Policy steps |
|---|---|---:|---:|---:|---:|
| Base | cabinet | 100/100 | 0.8709 | 11.0092 | 85.0 |
| Base | plate | 99/100 | 0.7001 | 11.6300 | 78.0 |
| Time-decay | cabinet | 96/100 | 0.8921 | 14.2718 | 94.7 |
| Time-decay | plate | 99/100 | 0.7002 | 12.3473 | 79.6 |

Time-decay 的配置仍为 `W0=1.0, gamma=0.9`。其 phase-aligned 权重为：

| Target | W at close | W at grasp | W at lift |
|---|---:|---:|---:|
| Cabinet | 0.579 | 0.392 | 0.360 |
| Plate | 0.462 | 0.350 | 0.297 |

这里的数值是 steering weight，不是 belief；Time-decay 不运行 belief observer。

### 5.3 原始 Belief：`T=0.1, lambda=1`

#### 完整 rollout 指标

| Method | Target | Success | EEF path (m) | Action TV | Policy steps |
|---|---|---:|---:|---:|---:|
| Belief T0.1 L1 | cabinet | 97/100 | 0.9010 | 13.3341 | 94.2 |
| Belief T0.1 L1 | plate | 99/100 | 0.7161 | 12.0574 | 79.4 |

#### Phase-aligned belief 和 steering

| Target | b+ close | b+ grasp | b+ lift | W close | W grasp | W lift |
|---|---:|---:|---:|---:|---:|---:|
| Cabinet | 0.973 | 0.998 | 1.000 | 0.027 | 0.002 | 0.000 |
| Plate | 0.982 | 0.999 | 1.000 | 0.018 | 0.001 | 0.000 |

因为 `lambda=1`，`W=1-b_positive`。低温度导致多次微小 residual difference 快速累积，observer 在真正 lift 前已经饱和到接近 1。

#### Geometric legibility

| Method | Target | AUC20 | Wrong first 15 |
|---|---|---:|---:|
| Base | cabinet | 0.2816 | 0.000 |
| Time-decay | cabinet | 0.3130 | 0.000 |
| Belief T0.1 L1 | cabinet | 0.2821 | 0.000 |
| Base | plate | 0.0054 | 0.639 |
| Time-decay | plate | 0.0537 | 0.456 |
| Belief T0.1 L1 | plate | -0.0026 | 0.670 |

旧 Belief 在 lift 时已经没有 steering，因此 cabinet AUC 基本等于 Base，plate 甚至没有改善。Time-decay 在两个 target 上都提高了 AUC20。

### 5.4 参数敏感性 smoke tests

为判断问题来自低温度还是 lambda 幅度，使用同一个 initialization `0`、每目标 1 条轨迹测试了三种配置。以下平均值是两个 targets 的简单平均，但总样本只有 2，因此只能用于观察调度趋势。

| T | Lambda | Success | b+ close | b+ grasp | b+ lift | W close | W grasp | W lift | AUC20 | Wrong15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 2 | 2/2 | 0.9988 | 1.0000 | 1.0000 | 0.0025 | 0.0001 | 0.0000 | 0.1362 | 0.367 |
| 0.3 | 2 | 2/2 | 0.9829 | 0.9907 | 0.9968 | 0.0343 | 0.0186 | 0.0064 | 0.0948 | 0.467 |
| 1.0 | 2 | 2/2 | 0.8341 | 0.8817 | 0.9093 | 0.3319 | 0.2366 | 0.1815 | 0.1449 | 0.300 |

单例趋势非常清楚：

- 单独把 lambda 从 1 提到 2 不能解决 posterior saturation；`T=0.1` 时 lift 权重仍近似 0。
- `T=0.3` 稍微减缓饱和，但 lift 时权重仍只有约 `0.006`。
- `T=1.0` 才在 close、grasp 和 lift 阶段保留可观的 steering。

但 smoke tests 不提供可靠成功率或均值估计，因此最终对 `T=1.0, lambda=2` 进行了 100 条/目标的正式评估。

### 5.5 增强 Belief：`T=1.0, lambda=2`

#### 完整 rollout 指标

| Method | Target | Success | EEF path (m) | Action TV | Policy steps |
|---|---|---:|---:|---:|---:|
| Belief T1 L2 | cabinet | 94/100 | 0.9145 | 15.9711 | 102.7 |
| Belief T1 L2 | plate | 100/100 | 0.6981 | 12.4455 | 78.0 |

#### 两目标宏平均

| Method/config | Success | Mean EEF path (m) | Mean Action TV | Mean steps | Mean later W |
|---|---:|---:|---:|---:|---:|
| Base | 99.5% | 0.7855 | 11.3196 | 81.5 | 0.0000 |
| Time-decay, W0=1 gamma=0.9 | 97.5% | 0.7961 | 13.3095 | 87.2 | 0.4582 |
| Belief, T=0.1 lambda=1 | 98.0% | 0.8086 | 12.6958 | 86.8 | 0.0561 |
| Belief, T=1.0 lambda=2 | 97.0% | 0.8063 | 14.2083 | 90.3 | 0.4353 |

#### 正式 phase-aligned belief 与 steering

| Target | b+ close | b+ grasp | b+ lift | W close | W grasp | W lift |
|---|---:|---:|---:|---:|---:|---:|
| Cabinet | 0.674 | 0.828 | 0.855 | 0.653 | 0.344 | 0.290 |
| Plate | 0.726 | 0.813 | 0.885 | 0.549 | 0.373 | 0.230 |

公式核对：

```text
Cabinet lift: W = 2 * (1 - 0.855) = 0.290
Plate lift:   W = 2 * (1 - 0.885) = 0.230
```

与 Time-decay 对比：

| Target | Phase | Time-decay W | Belief T1 L2 W | Belief - Time |
|---|---|---:|---:|---:|
| Cabinet | Close | 0.579 | 0.653 | +0.074 |
| Cabinet | Grasp | 0.392 | 0.344 | -0.048 |
| Cabinet | Lift | 0.360 | 0.290 | -0.070 |
| Plate | Close | 0.462 | 0.549 | +0.087 |
| Plate | Grasp | 0.350 | 0.373 | +0.023 |
| Plate | Lift | 0.297 | 0.230 | -0.067 |

Belief 在 gripper close 时比 Time-decay 更强，但到真正 lift 时反而更弱。形式化 paired analysis 中，Belief-Time 的 lift-weight difference 在两个 targets 上都为负，且置信区间不跨 0。

#### Geometric legibility

| Method | Target | AUC20 | Wrong first 15 |
|---|---|---:|---:|
| Base | cabinet | 0.2816 | 0.000 |
| Time-decay | cabinet | 0.3130 | 0.000 |
| Belief T1 L2 | cabinet | 0.3016 | 0.000 |
| Base | plate | 0.0054 | 0.639 |
| Time-decay | plate | 0.0537 | 0.456 |
| Belief T1 L2 | plate | 0.0518 | 0.466 |

在双方都成功的 paired episodes 上：

- Cabinet：
  - Time-decay 相比 Base，AUC20 `+0.0301`，95% CI `[+0.0215, +0.0390]`。
  - Belief T1 L2 相比 Base，AUC20 `+0.0191`，95% CI `[+0.0126, +0.0258]`。
  - Belief 相比 Time-decay，AUC20 `-0.0128`，95% CI `[-0.0187, -0.0071]`。
- Plate：
  - Time-decay 相比 Base，AUC20 `+0.0485`，95% CI `[+0.0376, +0.0588]`。
  - Belief T1 L2 相比 Base，AUC20 `+0.0464`，95% CI `[+0.0348, +0.0582]`。
  - Belief 相比 Time-decay，AUC20 `-0.0016`，95% CI `[-0.0111, +0.0078]`，没有可靠差异。

因此增强 Belief 在两个目标上均比 Base 更早表达真实目的地方向，但 cabinet 上弱于 Time-decay，plate 上与 Time-decay 统计上相当。

#### 成功率与运动代价

- Cabinet：
  - Base：100%。
  - Time-decay：96%。
  - Belief T1 L2：94%。
  - Belief 相比 Base 下降 6 个百分点，state-cluster bootstrap 95% CI `[-11%, -2%]`，McNemar exact `p=0.03125`。
  - Belief 相比 Time-decay 下降 2 个百分点，差异不显著。
  - 在双方成功 episodes 上，Belief 相比 Time-decay：EEF path `+0.0212 m`，95% CI `[+0.0022, +0.0444]`；Action TV `+1.294`，95% CI `[+0.384, +2.330]`。
- Plate：
  - Base 和 Time-decay 都是 99%，Belief T1 L2 是 100%；1 个百分点差异没有统计证据。
  - Belief 与 Time-decay 的 path 差异不可靠，但 Action TV 增加 `0.481`，95% CI `[+0.185, +0.865]`。

### 5.6 新旧 Belief 的直接比较

`T=1, lambda=2` 相比 `T=0.1, lambda=1`：

| Target | Metric | Old | New | Change |
|---|---|---:|---:|---:|
| Cabinet | Success | 97% | 94% | -3 pp |
| Cabinet | EEF path | 0.9010 | 0.9145 | +0.0135 m（raw group mean） |
| Cabinet | Action TV | 13.3341 | 15.9711 | +2.6370 |
| Cabinet | AUC20 | 0.2821 | 0.3016 | +0.0195 |
| Cabinet | W at lift | 0.000 | 0.290 | +0.290 |
| Plate | Success | 99% | 100% | +1 pp |
| Plate | EEF path | 0.7161 | 0.6981 | -0.0180 m（raw group mean） |
| Plate | Action TV | 12.0574 | 12.4455 | +0.3881 |
| Plate | AUC20 | -0.0026 | 0.0518 | +0.0544 |
| Plate | Wrong15 | 0.670 | 0.466 | -0.204 |
| Plate | W at lift | 0.000 | 0.230 | +0.230 |

使用严格 paired、joint-success 分析时，新版相对旧版：

- Cabinet：pre-lift path `+0.0173 m`，pre-lift Action TV `+1.843`，AUC20 `+0.0196`。
- Plate：pre-lift path `+0.0316 m`，pre-lift Action TV `+1.304`，AUC20 `+0.0551`，Wrong15 `-0.206`。

这说明高温度和更大 lambda 确实恢复了 delayed-phase legibility，但也把相当多 steering 放进了共同 pickup 阶段。

### 5.7 实验二支持的结论

1. **Temperature calibration 是必要的。** `T=0.1` 使弱 residual evidence 在多个 chunks 后快速饱和；仅提高 lambda 无法恢复 lift 阶段的 steering。
2. **`T=1, lambda=2` 显著改善了旧 Belief 的 geometric legibility。** 尤其 plate 的 AUC20 从负值提高到 `0.0518`，Wrong15 从 `0.670` 降到 `0.466`。
3. **Belief 可以在 close 时保留比 Time-decay 更高的 steering。** 这说明它没有再像低温度版本一样在很早阶段完全关闭。
4. **Geometric legibility 与任务性能之间存在 trade-off。** 更强 Belief 改善了 AUC，但 cabinet 成功率下降、路径和 Action TV 上升。
5. **当前 observer 会从人眼难以察觉的 pickup motion difference 中积累较强证据。** 到 lift 时 positive belief 已达到 `0.855/0.885`，并非理论期望的 0.5。

### 5.8 实验二不能支持的结论

1. **不能声称 Belief 比 Time-decay 更 legible。** Cabinet 上 Time-decay AUC 显著更高；Plate 上两者相当。
2. **不能声称 Belief 在 shared prefix 全程保持高 steering。** 它只在 close 时更高，到 lift 时已经低于 Time-decay。
3. **不能声称当前 reach-and-grasp 是统计意义上的相同前缀。** 人眼难以区分不等于 `P(prefix | positive) = P(prefix | negative)`。
4. **不能把 `T=1, lambda=2` 作为已经确定的最终参数。** 它在 cabinet 上出现可靠的成功率损失和更大运动代价。
5. **不能用 simulation geometric AUC 直接声称人类更容易识别意图。** 这需要独立 observer 或 human study。
6. **不能通过把 Time-decay 改为 `gamma=0.5` 就宣称方法胜出。** 这会形成明显的 baseline tuning 质疑。

---

## 6. 两个实验合起来说明了什么

### 6.1 论文层面的核心假设

Time-decay 的隐含假设是：

> 任务歧义会随着 elapsed time 按一个固定速度下降，因此 steering 也应按固定时间表下降。

但 manipulation task 的意图揭示时刻是 state-dependent 的：

- 如果目标从最初 reach 就能区分，强 early steering 可能是多余的，并增加 OOD motion、路径和失败风险。
- 如果两个任务先共享一段 prerequisite skill，steering 应在 shared prefix 中保留到真正的 destination branch。
- 如果 shared prefix 长度在任务间变化，一个固定 `gamma` 无法同时做到“早期明确时快速关闭”和“延迟明确时持续保留”。

Belief weighting 的理论优势不是某一组参数比某一组 `gamma` 更好，而是：

> 它试图用当前轨迹的 posterior ambiguity 代替 wall-clock/chunk index，解耦 steering intensity 与固定执行时间。

### 6.2 当前证据对这一假设的支撑程度

#### 已经得到支撑的部分

- 早期分叉实验表明，当 observer 很快确信 positive instruction 后，Belief 自动回到接近 Base 的运动分布，而 Time-decay 仍施加明显 steering。
- Time-decay 在至少一个早期明确任务中造成了可靠的额外路径和 Action TV，并伴随成功率下降。
- delayed-destination 实验表明，Belief 的实际行为高度依赖 posterior calibration；提高 temperature 可以把 steering 延迟到 grasp/lift 附近。
- 一旦 lift 阶段恢复 steering，几何目标表达相对 Base 确实改善。

#### 尚未得到支撑的部分

- 当前没有证明一个固定的 Belief 配置可以同时在两个实验上优于一个经过公平 validation 的 Time-decay 配置。
- 当前没有证明同一个 shared prefix 在 candidate-conditioned policy 下真的不可区分。
- 当前没有证明 Belief observer 与 human observer 的意图判断一致。
- 当前没有证明优势来自“更正确的 steering timing”，而不是两种方法使用了不同的累计 guidance budget。

---

## 7. 为什么 shared prefix 的 belief 会从 0.5 上升到约 0.8

两个 candidate 的初始 prior 被设为相同，因此第一个 chunk 前是 `0.5/0.5`。但保持 0.5 还要求：

```text
P(executed prefix | positive instruction)
approximately equals
P(executed prefix | negative instruction)
```

当前数据没有满足这一强条件。可能原因包括：

1. 两组 demonstrations 的 grasp approach、wrist orientation 或速度存在系统性差异。
2. Policy 会为不同的后续 destination 提前选择略有区别的 grasp pose；这种 preparation motion 可能是真实信息，而不是纯噪声。
3. Belief residual 不是严格归一化的 trajectory likelihood，连续 chunks 的 evidence 可能相关，累计后产生过度自信。
4. Steering 自身制造了更偏向 positive prompt 的动作，下一次 observer 又用这段动作更新 belief，形成 self-reinforcing feedback。
5. 如果 observer 评分包含未真正执行的 predicted suffix，或者 action chunk 跨过了 semantic branch，可能提前获得未来意图信息。正式版本需要继续审计 observer 只使用已执行前缀。

提高数据量可能缓解第 1 点，但**只有 matched-prefix data 才直接解决问题**。单纯增加独立 teleoperation demos 可能反而更准确地学习操作者在两类任务中的微小习惯差异。

建议的数据收集/构造方式：

- 相同 initial state 下成对收集两个 destination。
- 使用同一个 scripted/replayed controller 完成 reach-and-grasp，抓稳后才分叉。
- 或在训练 window 层面，把完全相同的 pickup prefix 同时赋予两个完整任务标签。
- 两指令 demo 数量、操作者、采集顺序和初始状态严格平衡。
- 在 held-out prefix 上训练独立 instruction classifier；若抓取前准确率显著高于 50%，则该数据并不是有效的 shared-prefix benchmark。

---

## 8. Time-decay baseline 应如何公平设置

Legibility Diffuser 在每次执行 open-loop action chunk 后衰减 guidance，但原论文同时指出最佳 decay rate 与任务相关，长 horizon 往往需要更接近 1 的 `gamma`。论文对多个 gamma 进行了消融，而不是把 `0.5` 作为所有 simulation task 的通用参数。参见 [Legibility Diffuser 原论文](https://mbronars.github.io/LegibilityDiffuser.pdf)。

对于：

```text
W_c = W0 * gamma^c
```

其 chunk half-life 为：

```text
h_half = log(0.5) / log(gamma)
```

- `gamma=0.9`：half-life 约 6.58 chunks。
- `gamma=0.5`：half-life 为 1 chunk。

当前 bowl 任务约在第 10 个 query 附近 lift；如果直接把 `gamma` 改为 0.5 且保持 `W0=1`，lift 时理论权重会接近 `0.001`。这几乎必然让 Time-decay 在真正 destination branch 时没有 steering，但不能构成公平的主结论，因为 baseline 明显未针对当前 action-chunk frequency 和短时程任务校准。

正式论文应至少包含：

1. 原论文对应设置，作为 reproduction point。
2. 在独立 validation tasks 上选择的 global `W0, gamma`。
3. 完整 success-legibility-efficiency Pareto sweep，而不只报告一个有利点。
4. 可选的 per-task oracle gamma，作为对 Time-decay 非常宽松的上界。
5. Matched guidance budget：比较相近的 `sum_c W_c`，确认优势来自 timing 而不是总 steering 更强。

---

## 9. 建议的下一阶段参数与实验设计

### 9.1 不建议立即把 `T=1, lambda=2` 固定为论文参数

它比旧参数更有 legibility，但在 cabinet 上成功率从 Base 的 100% 降至 94%，并且比 Time-decay 路径更长、Action TV 更高。它是有价值的 sensitivity point，不是已经验证的最优点。

### 9.2 参数应在独立 validation set 上冻结

建议给两类方法相同调参预算：

- Time-decay：
  - `W0` 可选 `{0.5, 1, 2}`。
  - `gamma` 可选 `{0.5, 0.75, 0.9, 0.95}`，或直接用 chunk half-life 参数化。
- Belief：
  - `T` 可选 `{0.3, 0.5, 1, 2}`。
  - `lambda` 可选 `{0.5, 1, 2}`。
  - 保留 `first chunk W=0` 作为主配置，同时做一次 ablation。

可使用带约束的 validation objective：

```text
maximize independent legibility metric
subject to success >= Base - delta
and path overhead <= epsilon
```

例如预注册 `delta=2 percentage points`、`epsilon=10%`，具体阈值由论文应用需求决定，而不是看 test result 后再选择。

### 9.3 最重要的新实验：系统改变意图揭示时间

不应只依赖一个 early case 和一个 delayed case。建议构造至少三个 ambiguity-resolution bins：

| Bin | 场景特征 |
|---|---|
| Early | 起始 reaching direction 已能区分目标 |
| Medium | reach 相近，但 grasp approach/物体选择前分叉 |
| Late | 完全共享 reach-and-grasp，lift 后才选择 destination |

然后使用一套冻结的全局参数测试所有 bins。最有力的论文结果应是：

> 随着真实 branch time 变化，Belief 的 steering peak 自动随 uncertainty 移动；Time-decay 若不重新调整 gamma，则无法同时维持 early-task efficiency 和 late-task legibility。

### 9.4 Observer calibration 不能以“强制 shared prefix belief=0.5”为目标

Temperature 应在独立 calibration set 上通过 NLL、Brier score 或 ECE 选择，而不是直接观察 test trajectory 后把 belief 调到希望的数值。

如果论文最终讨论 human legibility，更理想的做法是：

- 给人类只播放截至当前时刻的 trajectory prefix；
- 收集人类对 candidate goal 的概率判断；
- 用这些概率校准 policy observer；
- 将几何 observer、policy residual observer 和 human observer 分开报告。

如果人类是 0.5 而 policy observer 是 0.85，这说明 observer 与 human inference 不一致，不能只靠提高 temperature 隐藏。

---

## 10. 建议向导师重点讨论的问题

1. 论文核心 claim 是否应确定为“state-dependent ambiguity scheduling”，而不是“某组参数胜过 gamma=0.5”。
2. 是否值得重新构造 matched-prefix dataset，使抓取前的 candidate classification accuracy 接近 chance。
3. Belief observer 应以 policy likelihood 为目标，还是需要用少量 human prefix judgments 做 calibration。
4. 正式 baseline 是使用 paper setting、global validation optimum，还是同时给出 per-task oracle time decay。
5. 是否将当前 `T=1, lambda=2` 结果作为 calibration ablation，而不是 main result。
6. 是否将主要结果展示为 Pareto frontier，而不是只比较单点 success 和 legibility。
7. Real-robot human study 中应如何将 intention-recognition time、confidence AUC、task success 和运动效率联合起来。

---

## 11. 当前最稳妥的论文式结论

可以写：

> Time-based guidance schedules couple intent expressiveness to elapsed execution time and therefore cannot directly account for when task ambiguity is actually resolved. In an early-disambiguating LIBERO task pair, belief-conditioned steering rapidly reduced unnecessary guidance and recovered base-policy efficiency, whereas fixed time decay increased motion cost and reduced success on one task. In a delayed-destination task pair, posterior calibration was essential: increasing the belief temperature restored guidance near grasp and improved geometric goal expression over the base and an overconfident belief configuration. However, the calibrated belief method did not outperform a slowly decaying time baseline and incurred additional manipulation cost, indicating that the current policy-based observer still extracts—and may amplify—subtle instruction-specific evidence during the nominally shared pickup phase.

对应中文：

> 时间调度将意图表达强度与执行时间绑定，无法直接反映任务歧义在何时真正消失。在早期即可区分的 LIBERO 任务中，Belief 调度迅速减少不必要的 guidance，使运动效率回到接近 Base；固定 Time-decay 则在其中一个任务上增加运动代价并降低成功率。在延迟目的地分叉任务中，posterior calibration 是必要条件：提高温度能够恢复抓取附近的 steering，并相对 Base 和过度自信的旧 Belief 改善几何目标表达。但当前校准后的 Belief 尚未优于慢速 Time-decay，且引入额外 manipulation cost，说明现有 policy-based observer 仍会在名义上的 shared pickup phase 中提取并可能放大细微的指令相关证据。

目前不应写：

> Belief-weighted steering is superior to time-decay steering in delayed-intent manipulation.

因为现有数据并不支持这一普遍性结论。

---

## 12. 数据与分析文件

### 实验一

- 正式汇总：`artifacts/goal_cream_bowl_steering_formal_100/formal_summary.md`
- 机器可读汇总：`artifacts/goal_cream_bowl_steering_formal_100/formal_summary.json`
- Episode 表：`artifacts/goal_cream_bowl_steering_formal_100/formal_episodes.csv`
- 运行脚本：`examples/libero/run_goal_cream_bowl_steering_formal.sh`

### 实验二

- 原始 `T=0.1, lambda=1` 正式输出：`/workspace/openpi-evaluations/goal_bowl_destination_steering_formal_100`
- 原始参数 phase report：`artifacts/goal_bowl_destination_steering_formal_original_phase/phase_legibility.md`
- `T=0.1/0.3/1.0, lambda=2` smoke tests：`artifacts/goal_bowl_destination_belief_strength`
- `T=1.0, lambda=2` 正式输出：`/workspace/openpi-evaluations/goal_bowl_destination_belief_t1_lambda2_formal_100`
- 合并正式汇总：`artifacts/goal_bowl_destination_steering_formal_t1_lambda2_combined/formal_summary.md`
- 合并 phase report：`artifacts/goal_bowl_destination_steering_formal_t1_lambda2_combined/phase_legibility.md`
- Phase analysis：`examples/libero/analyze_goal_bowl_destination_phases.py`
- 正式运行脚本：`examples/libero/run_goal_bowl_destination_steering_formal.sh`
- Belief-only 正式运行脚本：`examples/libero/run_goal_bowl_destination_belief_formal.sh`

### 排除项

`/workspace/openpi-evaluations/goal_bowl_destination_steering_formal_t1_lambda2_100` 是曾启动但随后停止的冗余全 600 条运行。由于 Base/Time-decay controls 已经存在，正式分析只复用旧 400 条 controls 并加入新 200 条 Belief；该不完整目录没有进入任何统计。
