# Language-conditioned motion legibility 项目完整复述、实验结果与修改建议

## 文档目的

本文档完整整理了目前围绕该论文项目已经讨论和完成的工作，包括：

1. 论文最初的研究问题和整体 scientific story；
2. 真实机器人数据是如何采集、配对和划分的；
3. 哪些轨迹参与了微调，哪些轨迹只用于测试；
4. checkpoint、数据转换和实际评估流程；
5. residual、model belief、Early-AUC 和几何 observer 的含义；
6. 训练集、测试集、四种弯曲条件以及稳健性实验的完整结果；
7. 为什么当前测试成功率较低，以及哪些解释已被数据支持或排除；
8. 是否应该加入刻意 legible 的轨迹继续微调；
9. 方块是否应该全程固定，以及下一轮更合理的受控实验设计；
10. 当前论文可以声称什么、不能声称什么，以及建议优先修改的地方。

这不是逐句聊天记录，而是把此前对话重组为一份可以直接用于实验规划、论文方法部分和组会讨论的
完整技术说明。

---

## 1. 论文项目的整体定位

### 1.1 核心研究问题

项目最初提出的核心问题不是简单地“发明一个新的 legibility metric”，而是：

> 当 generalist robot policy 的任务目标由自然语言定义时，应该如何重新理解和测量 motion
> legibility？一个 VLA 自己学到的 instruction-conditioned action model，能否反过来充当观察者，
> 从机器人正在执行的动作中推断自然语言意图？这种模型观察者与人类观察者在什么情况下相符，
> 又会在什么情况下失效？

经典 motion legibility 通常假设一个有限的几何目标集合，例如左边物体和右边物体，并使用人为设计的
cost 或 rationality model 描述观察者认为机器人“应该如何朝某个目标移动”：

\[
P(g\mid \xi)\propto P(\xi\mid g)P(g),
\qquad
P(\xi\mid g)\propto \exp[-C(\xi,g)].
\]

VLA 提供了另一种可能。它本身学习：

\[
\pi_\theta(a\mid o,\ell),
\]

其中 \(o\) 是视觉和机器人状态，\(a\) 是动作，\(\ell\) 是自然语言任务。因此可以反过来询问：

> 已观察到的动作，在“pick up the left block”和“pick up the right block”两个语言条件下，哪一个
> 更符合该 VLA 学到的行为分布？

由此得到一个 model-native observer：

\[
q_\theta(\ell\mid \xi_{0:t})
\propto
\operatorname{Compat}_\theta(\xi_{0:t},\ell)P(\ell).
\]

这里必须谨慎：目前使用的 flow-matching residual 并不是严格的 normalized likelihood。因此更合适的
名称是：

- instruction-conditioned action compatibility；
- predictability proxy；
- model-derived intent belief；
- posterior proxy。

不应在没有额外理论或校准证据时直接称其为真实的 \(P(\xi\mid\ell)\)。

### 1.2 为什么语言不只是把几何目标换一个名字

如果只比较“pick up the left block”和“pick up the right block”，语言暂时只是表达两个空间目标。
但论文更广泛的意义在于，语言还能表示传统 endpoint 无法完整表达的意图，例如：

- `pour from the mug into the bowl`；
- `hand the mug to the person`。

两项任务可能接触同一个物体、共享相同的初始 reach，甚至具有相同的早期 endpoint，但要求不同的：

- grasp position；
- wrist orientation；
- temporal ordering；
- future action；
- sequential task structure。

因此项目更重要的理论判断是：

> Language makes intent richer than a terminal geometric state, but a linguistic distinction can become
> legible only when it induces an observable behavioral distinction.

### 1.3 原计划中的论文结构

此前讨论出的第一篇论文主线是“measurement + characterization + human validation”，而不是主动
steering。建议的结构为：

1. **Introduction**：为什么 VLA 改变了 motion legibility 的观察者假设；
2. **Formalism**：如何从 instruction-conditioned action compatibility 得到 intent belief；
3. **LIBERO temporal emergence**：共享动作前缀时保持模糊，任务动作分叉后 belief 才分离；
4. **Controlled real-robot study**：研究 spatial、action-semantic 和 sequential intent；
5. **Online human study**：让人与模型观看相同视频 prefix，比较两者判断；
6. **Discussion**：candidate-set dependence、prompt sensitivity、distribution shift 和 observer failure。

第一篇论文回答：

> 我们能否测量并理解 language-conditioned legibility？

后续第二篇论文才回答：

> 我们能否在不损害任务成功和动作自然性的前提下，主动 steering VLA 生成更 legible 的动作？

### 1.4 最初建议的三项贡献

1. 形式化 language-conditioned motion legibility，并提出直接来自 VLA action model 的 observer；
2. 系统研究 spatial、action-semantic 和 sequential language intent 所造成的不同形式的动作歧义；
3. 使用相同视觉输入和相同轨迹 prefix，比较 VLA-derived belief 与人类判断，并识别一致与失效情况。

### 1.5 必须收紧的论文 claim

此前已经明确了三条写作边界：

1. residual 是 compatibility proxy，不是严格 likelihood；
2. 语言空间可以是 open-ended，但一次推断仍依赖 scene-specific candidate set \(\mathcal L\)；
3. 不能预设 demonstration-trained policy 等同于 human observer；这必须通过 user study 验证。

### 1.6 此前讨论过的论文标题与一句话 thesis

曾提出的标题包括：

- **From Motion to Language: Measuring Intent Legibility in Vision-Language-Action Policies**；
- **Language-Conditioned Motion Legibility in Generalist Robot Policies**。

论文的一句话 thesis 可以概括为：

> 传统方法使用人为设计的 geometric cost 模拟观察者如何从运动推断目标；VLA 自己学习了语言条件下什么动作
> 是合理的，因此可以提供一种 model-native intent observer。论文需要刻画该 observer 在 shared-prefix、
> spatial、semantic 和 sequential intent 中何时有效，并通过人类实验验证它是否真的接近人的判断。

### 1.7 原计划中的三类核心实验

#### LIBERO：意图何时从共享前缀中出现

例如三个任务都先执行：

\[
\text{reach bowl}\rightarrow\text{grasp}\rightarrow\text{lift},
\]

之后才分别把 bowl 放到 plate、stove 或 cabinet。理想 observer 应在共享前缀中保持高不确定性，在 bowl 开始
向不同目的地移动时才逐渐分离候选 belief。这一部分主要验证 temporal structure，而不是要求模型在尚无可观察
差异时提前猜出目标。

#### Controlled real robot：语言定义的不同意图类型

最初规划的 spatial 条件不仅包括本轮 C0-C3，还提出了更严格的四类对照：

1. direct/efficient；
2. in-distribution legible，即自然出现的早期 directional bias；
3. OOD legible，即明显朝真实目标方向 exaggerate；
4. OOD irrelevant control，即路径长度和弯曲幅度相近，但方向不帮助区分目标。

第四类 control 很重要，因为它可以检测 metric 是否只是 motion-magnitude detector。

action-semantic 实验则建议比较 `pour from the mug into the bowl` 和 `hand the mug to the person`。两项任务共享
物体和初始 reach，但可能通过 grasp pose、handle use 和 wrist orientation 提前表达未来用途。

sequential/compositional 实验强调：两个完整语言任务可以共享一个较长 subtask，所以语言语义不同并不保证每个
时刻都行为可分。只有当语言差异导致可观察动作差异时，legibility 才应该出现。

paraphrase 不建议作为与 spatial/semantic/sequential 平行的第四个主任务，而更适合作为 robustness analysis。
如果 `put the mug in the cabinet`、`store the mug inside the cabinet` 和 `put the mug away in the cupboard` 在相同
轨迹上得到明显不同 residual，就说明 observer 同时受到 language-realization bias 影响。

#### Human study：人与模型看相同的信息

最初建议复用前面所有关键轨迹，不另外设计一套与模型实验脱节的 user-study stimuli。人类参与者只观看与 VLA
一致的固定 RGB camera streams，从而尽可能对齐视觉证据。可在 20%、40%、60%、80% 等 prefix 截断，但同一
参与者不连续观看同一轨迹越来越长的多个版本，以避免学习和判断承诺。

主要问题不是抽象地问“这个动作有多 legible”，而是直接问：

> What is the robot trying to do?

随后记录候选任务选择和 confidence。较大规模研究可考虑约 100–200+ 名在线参与者，并使用 mixed-effects
模型处理 participant 与 trajectory 的随机效应。

### 1.8 Predictability、legibility 与 efficiency 的区分

此前反复强调三者不能混为一谈：

- **predictability/compatibility**：真实任务下，该动作是否像模型熟悉的任务执行；
- **legibility**：相对于竞争候选，该动作是否更早、更明确地排除错误意图；
- **efficiency**：路径长度、完成时间、jerk、成功率等物理代价。

一个动作可能不够典型、路径更长，却通过让错误解释迅速变得不合理而更加 legible。因此真实机器人实验应独立
记录 path length、completion time、success 和可选的 jerk，而不能把低 residual 直接等同于高效率。

### 1.9 最初建议的五张论文图

1. **Figure 1**：LIBERO bowl 的多帧轨迹和三条候选 belief curve，展示 shared prefix 后的分叉；
2. **Figure 2**：\(\xi_{0:t}\rightarrow\) VLA under multiple instructions \(\rightarrow D_i\rightarrow q_t(\ell)\)
   的 formalism diagram；
3. **Figure 3**：大规模 held-out LIBERO progress、true-intent posterior、entropy 和 recognition time；
4. **Figure 4**：真实机器人 spatial/semantic taxonomy、受控轨迹和 belief curves；
5. **Figure 5**：human 与 VLA posterior curve、correlation 和 failure cases。

---

## 2. 当前真实机器人数据是如何采集的

### 2.1 空间布局与采集原则

每次在桌面上放置两个相邻木块。两个木块总体位于机械臂基座中线附近，但具体位置在该区域内随机变化。
这样设置的原因是：

- 抓左块时，机械臂的初始运动通常自然地略微向左偏；
- 抓右块时，机械臂的初始运动通常自然地略微向右偏；
- 避免两个方块都位于很左或很右的位置，导致无论执行哪个指令，机械臂一开始都必须朝同一方向移动；
- 保留一定 layout 多样性，而不是只让模型记忆单一固定像素位置。

每一对轨迹共享完全相同的两个方块位置，一条执行左指令，一条执行右指令。使用的指令为：

- `pick up the left block`；
- `pick up the right block`。

### 2.2 微调训练轨迹

训练数据位于：

```text
/home/hci-lab/repos/droid/data/success/2026-08-09
```

最初描述路径时曾写成 `succuess`，实际目录名是 `success`。只有 **2026 年 8 月 9 日** 的数据有效，
其他日期均被明确排除。

有效训练数据共有：

- 32 条轨迹；
- 16 对相邻采集的左右任务；
- 每一对共享相同方块布局；
- 左右指令各一条，但一对内部的时间顺序不一定总是左后右或右后左；
- 所有轨迹都是正常 teleoperation task completion demonstration。

采集训练数据时，操作者没有刻意构造 legible 或 communicative motion，也没有为了让观察者更早识别目标
而夸大弯曲。操作者只是按照常人的任务执行思路，正常抓取指令指定的左块或右块。因此，这 32 条数据表示：

\[
\text{ordinary task-directed demonstrations},
\]

而不是：

\[
\text{explicitly communicative demonstrations}.
\]

### 2.3 训练数据转换与 checkpoint

DROID 原始轨迹在训练前使用项目中的转换流程转换为 LeRobot 格式。相关命令形式为：

```bash
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /path_to_your_droid_repo/data \
  --repo-id peopleandrobots/spatial \
  --skip-invalid \
  --overwrite
```

本地训练用 LeRobot dataset 为 `peopleandrobots/spatial`，经核对其中正好包含上述 32 条有效 episode，
转换记录中 `include_failures` 为 false，因此 failure 目录的测试轨迹没有参与微调。

训练完成的 checkpoint 位于：

```text
/workspace/checkpoints/pi05_droid_finetune/spatial/39999
```

使用的配置为 `pi05_droid_finetune`。模型 action horizon 为 16，模型内部 action dimension 为 32；
DROID 的物理有效动作是前 8 维，即 7 维 joint velocity 加 1 维 gripper action。

### 2.4 留出测试轨迹

测试数据位于：

```text
/home/hci-lab/repos/droid/data/failure/2026-08-09
```

同样只有 **2026 年 8 月 9 日** 的数据有效，failure 目录中其他日期全部忽略。有效测试数据共有：

- 8 条轨迹；
- 4 对左右任务；
- 每一对共享相同的方块布局；
- 指令仍然是同样的 left/right block；
- 按采集时间依次代表四级人为设计的运动条件。

四个条件和精确时间顺序如下：

| 条件 | 采集时间 | 目标 | 用户设计意图 |
|---|---|---|---|
| C0 natural/direct | 19:29:02 | left | 正常 teleop，未刻意弯曲 |
| C0 natural/direct | 19:29:23 | right | 正常 teleop，未刻意弯曲 |
| C1 mild exaggeration | 19:30:32 | left | 轻微朝正确方向弯曲 |
| C1 mild exaggeration | 19:31:21 | right | 轻微朝正确方向弯曲 |
| C2 strong exaggeration | 19:33:31 | left | 明显弯曲 |
| C2 strong exaggeration | 19:33:54 | right | 明显弯曲 |
| C3 extreme exaggeration | 19:34:35 | right | 非常弯曲，甚至略显不自然 |
| C3 extreme exaggeration | 19:35:12 | left | 非常弯曲，甚至略显不自然 |

用户对这些数据的原始判断是：除了最前面的 C0 两条以外，后面六条都是刻意构造的弯曲动作，因此从
行为风格上大概率不在原始训练 demonstration 的范围内。需要注意的是，这个“是否 OOD”的最初标签来自
人为设计意图；后面的 residual 和几何统计进一步对这种偏移进行了量化。

### 2.5 当前数据设计中的关键混杂

当前 C0-C3 每个条件只有一对轨迹，并且：

- 弯曲等级随采集时间固定递增；
- 每一级使用不同 layout；
- layout、时间、操作者状态和弯曲程度共同变化；
- 没有同一个 layout 下对 C0-C3 的完整重复；
- 没有多个独立 repetition。

因此当前数据是很有价值的 pilot，但不能把 C0-C3 的差异解释为“弯曲程度造成的因果效应”。

---

## 3. 对话中依次提出的问题

整个讨论经历了以下几个阶段：

1. 首先提出完整的 HRI 论文架构，希望研究 VLA 能否作为 language-conditioned legibility observer；
2. 描述 32 条普通训练轨迹和 8 条不同弯曲程度的测试轨迹，询问应评估哪些方面；
3. 指定 checkpoint 和数据转换方式，要求实际运行实验并报告结果；
4. 在看到测试准确率较低后，追问是否因为训练时没有加入明显弯曲的 legible 轨迹；
5. 进一步追问，当 true 和 wrong instruction residual 都变大时，为何 wrong residual 没有变得更多；
6. 询问是否应该刻意收集 legible demonstrations 用于微调，以及方块位置是否应该全部固定；
7. 询问 Early-AUC 的具体含义；
8. 询问几何 observer 的定义、计算方法和限制。

以下章节把这些问题和答案统一到同一套符号与实验结果中。

---

## 4. 实际 VLA observer 是如何实现的

### 4.1 为什么没有把 failure 数据转换进训练用 LeRobot dataset

评估时没有使用 `--overwrite` 重新生成包含 failure 的训练数据集，也没有把测试轨迹写进
`peopleandrobots/spatial`。这是为了避免测试污染和意外覆盖训练数据。

评估器直接原地读取 failure 的 HDF5 和 MP4，但严格复现训练转换过程所使用的数据表示：

- 从 HDF5 读取 joint position、gripper state、joint velocity 和 gripper action；
- 自动识别一台 exterior camera 和一台 wrist camera；
- 解码对应 MP4；
- 把原始 RGB 图像以 PIL bicubic 从 640×480 resize 到 320×180；
- 使用 checkpoint 自带的 `assets/droid/norm_stats.json`；
- 使用 `pi05_droid_finetune` 配置中完全相同的 data transform、quantile normalization 和 model transform。

因此，failure 数据保持严格 held-out，同时输入表示与训练流程对齐。

### 4.2 轨迹有效区间与抓取边界

对每条轨迹定义：

- motion start：`movement_enabled` 首次为 true 的时刻；
- grasp start：motion start 之后 `action/gripper_position > 0.1` 的第一个时刻。

主要 legibility 分析使用 motion start 到 grasp start 之间的 pre-grasp 动作。原因是空间目标识别的核心问题是：

> 在机器人真正抓到物体之前，观察者能否从运动中提前判断它要抓左块还是右块？

同时也计算了 full-trajectory final accuracy，用于确认加入抓取后的动作是否会改变最终判断。

### 4.3 Action chunk

模型 action horizon 为 16，因此每条轨迹被划分为非重叠的 16-step chunk。实现中特别处理了两个问题：

1. chunk 在 grasp onset 处强制切断，不允许一个 chunk 同时跨越 pre-grasp 和 post-grasp；
2. 每个 phase 的最后一个不足 16 步的 chunk 使用最后动作 padding，但通过 `executed_steps` mask 排除
   padding residual。

每个 chunk 的证据被赋给 chunk 的结束时刻，而不是起始时刻。因此在画 prefix belief 时，不会把尚未执行的
未来动作错误地算入更早的轨迹前缀。这一点已经通过测试验证。

### 4.4 Flow-matching residual compatibility

对 observation \(o_t\)、实际 action chunk \(a_t\) 和候选指令 \(\ell\)，计算 flow-matching residual：

\[
D_\theta(a_t,o_t,\ell)
=
\mathbb E_{\tau,\epsilon}
\left[
\left\|
v_\theta(x_\tau,o_t,\ell,\tau)-v^*(x_\tau,a_t,\epsilon,\tau)
\right\|^2
\right].
\]

其中 residual 越小，表示实际动作越符合模型在该 observation 和 instruction 下学到的 action field。
定义 compatibility score：

\[
s_t(\ell)=-D_\theta(a_t,o_t,\ell).
\]

对两个候选指令始终使用相同的 flow time 和相同 noise，即 common random numbers。这样 residual difference
主要反映语言条件的差异，而不是两次随机采样的差异。

主设置为：

- 候选顺序固定为 left、right；
- 每个 chunk 使用 3 个独立 seed：0、1、2；
- 每个 seed 8 个 flow sample；
- \(\tau\in[0.3,0.7]\)；
- 使用前 8 个 DROID action dimensions。

另外加入了 candidate-order 等变性测试，确认交换左右候选只会交换输出的两列，不会因候选顺序造成
标签偏差。

### 4.5 从 residual 到累计 model belief

对每个候选指令累计到当前时刻的 compatibility：

\[
S_t(\ell)=-\sum_{k\le t}D_k(\ell).
\]

在均匀 prior 下，二分类 belief 为：

\[
q_t(L)=
\frac{\exp(S_t(L)/T)}
{\exp(S_t(L)/T)+\exp(S_t(R)/T)},
\qquad q_t(R)=1-q_t(L).
\]

温度 \(T\) 只使用 32 条微调轨迹做描述性校准，主实验拟合得到：

\[
T\approx0.003664.
\]

这个温度只影响 belief 数值的软硬程度和曲线显示，不改变：

- 哪个 residual 更小；
- raw margin 的符号；
- 最终 top-1 是否正确。

正式研究中更严格的做法应当是使用独立 validation set 校准温度，而不是微调训练轨迹。

### 4.6 True residual、wrong residual 和 margin

对于真实指令 \(\ell^*\) 和另一个错误候选 \(\ell^-\)，定义：

\[
D_{\mathrm{true}}=D(\ell^*),
\qquad
D_{\mathrm{wrong}}=D(\ell^-),
\]

以及局部 margin：

\[
m=D_{\mathrm{wrong}}-D_{\mathrm{true}}.
\]

因此：

- \(m>0\)：真实指令 residual 更小，该 chunk 支持真实目标；
- \(m<0\)：错误指令 residual 更小，该 chunk 支持错误目标；
- \(m\approx0\)：模型无法利用该 chunk 区分两个指令。

对于一条刻意弯曲但仍然 legible 的轨迹，理想结果不一定要求 \(D_{\mathrm{true}}\) 保持很低。可能出现：

\[
D_{\mathrm{true}}\uparrow,
\qquad
D_{\mathrm{wrong}}\uparrow\uparrow,
\]

只要错误指令的 residual 增加得更多，margin 仍然明显为正，真实目标 belief 就会升高。这正是
predictability 和 legibility 的区别：

- absolute \(D_{\mathrm{true}}\) 衡量动作是否像模型熟悉的真实任务行为；
- relative margin 衡量动作是否比另一个候选更能区分真实意图。

### 4.7 Early-AUC

Early-AUC 是抓取前“真实目标 belief 随轨迹进度变化的曲线面积”，不是常见的 ROC-AUC。

先把 pre-grasp trajectory progress 归一化为 \(p\in[0,1]\)，没有动作证据时从均匀先验 0.5 开始，
然后计算：

\[
\operatorname{Early\text{-}AUC}
=
\int_0^1q_p(\ell^*)\,dp.
\]

当前离散 chunk 使用梯形积分。直观解释为：

- 接近 1：很早就判断正确并持续保持；
- 接近 0.5：总体无法区分，或正确与错误证据互相抵消；
- 小于 0.5：大部分时间偏向错误指令；
- 两条轨迹最终都正确时，更早被识别的轨迹 Early-AUC 更高。

Early-AUC 不等于动作效率，也不保证 absolute compatibility 良好。因此必须与最终正确率和 true residual
一起报告。

此外，当前分析把持续识别阈值设为 0.75，并可定义 recognition progress：

\[
p_{\mathrm{recognition}}
=
\min\{p:q_p(\ell^*)\ge 0.75\ \text{且此后始终不低于 }0.75\}.
\]

如果轨迹从未达到并保持该阈值，则 recognition progress 记为缺失。相比单独的最终正确率，它能区分“很早确认”
和“最后一刻才猜对”。

---

## 5. 几何 observer 是如何设置的

### 5.1 候选目标位置

对每一对共享 layout 的左右轨迹，用各自 grasp start 时的末端执行器位置近似两个候选目标：

\[
g_L=\text{left trajectory grasp-onset end-effector position},
\]

\[
g_R=\text{right trajectory grasp-onset end-effector position}.
\]

### 5.2 即时距离证据

对某个 chunk 结束时的末端位置 \(x_t\)，计算：

\[
d_L(t)=\|x_t-g_L\|_2,
\qquad
d_R(t)=\|x_t-g_R\|_2.
\]

定义支持左目标的几何 logit：

\[
z_t^{\mathrm{geo}}=d_R(t)-d_L(t).
\]

因此：

- \(z_t^{\mathrm{geo}}>0\)：当前位置更接近左目标；
- \(z_t^{\mathrm{geo}}<0\)：当前位置更接近右目标；
- \(z_t^{\mathrm{geo}}\approx0\)：当前几何位置无法区分。

转换为 belief：

\[
q_t^{\mathrm{geo}}(L)=\sigma(\alpha z_t^{\mathrm{geo}}),
\qquad
q_t^{\mathrm{geo}}(R)=1-q_t^{\mathrm{geo}}(L).
\]

尺度 \(\alpha\) 只用 32 条微调轨迹的 pre-grasp chunk 通过二分类交叉熵拟合，结果为：

\[
\alpha\approx51.05\ \mathrm{m}^{-1}.
\]

几何 Early-AUC 使用与 VLA 完全相同的 progress 和积分方式。

### 5.3 它能测什么、不能测什么

该基线只回答：

> 当前末端执行器更接近哪个已知候选目标？

它不使用：

- RGB 图像；
- 语言语义；
- joint velocity；
- gripper 动作；
- grasp orientation；
- 路径是否自然或高效；
- policy training distribution。

因此，它适合当前 left/right spatial intent，但无法区分“pour the mug”和“hand over the mug”这类共享
空间目标、依赖动作语义的任务。

### 5.4 当前几何基线的重要限制

候选目标坐标来自测试 pair 完整轨迹的 grasp-onset 位置。也就是说，分析早期 prefix 时，baseline 已经知道
两条完整轨迹最后在哪里抓取。经典几何 legibility 通常也假设候选目标已知，所以它仍是有意义的 baseline，
但论文必须明确说明这一点。

更严格的版本应当：

- 在第一帧中手工标注两个木块中心；或
- 使用视觉检测、AprilTag 或固定标定获得目标坐标；
- 在整条 trajectory 中固定这些 candidate coordinates；
- 不从测试轨迹未来的 grasp endpoint 估计目标。

---

## 6. 主实验结果

### 6.1 汇总结果

| 数据集 | 轨迹数 | pre-grasp 局部 chunk 正确率 | pre-grasp 最终正确率 | full 最终正确率 | Early-AUC | 几何最终正确率 | 几何 Early-AUC | 平均 true residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 微调训练轨迹 | 32 | 0.843 | 31/32 = 0.969 | 0.969 | 0.854 | 1.000 | 0.676 | 0.0150 |
| 留出测试轨迹 | 8 | 0.444 | 3/8 = 0.375 | 0.375 | 0.507 | 1.000 | 0.838 | 0.2542 |

最直接的观察是：

1. checkpoint 对参与过微调的 32 条轨迹识别很强；
2. 对 8 条未参与微调的留出轨迹，最终只正确 3 条；
3. 加入抓取后的动作没有挽救判断，pre-grasp 和 full accuracy 都是 3/8；
4. 测试 Early-AUC 0.507，整体几乎相当于一直保持 50/50；
5. 几何 observer 对相同测试轨迹为 8/8，表明轨迹本身确实包含清晰的空间目标信息；
6. 测试 true residual 是训练轨迹的 16.9 倍，说明 model compatibility 发生巨大分布偏移。

训练轨迹是微调时实际见过的 episode，因此 31/32 只能被称为 in-distribution reference，不能被当作
held-out generalization result。训练与测试之间的巨大差距既可能包含过拟合，也可能包含跨 session 或
轨迹分布变化。

### 6.2 四个条件的平均结果

| 条件 | 设计含义 | 路径比 | 最大直线偏离 | VLA Early-AUC | 最终正确率 | 几何 Early-AUC | 平均 true residual |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | 正常、直接 | 1.106 | 0.077 m | 0.075 | 0/2 | 0.749 | 0.200 |
| C1 | 轻微 exaggeration | 1.202 | 0.103 m | 0.800 | 1/2 | 0.909 | 0.205 |
| C2 | 明显 exaggeration | 1.449 | 0.214 m | 0.497 | 1/2 | 0.826 | 0.266 |
| C3 | 极端 exaggeration | 2.239 | 0.353 m | 0.656 | 1/2 | 0.867 | 0.345 |

路径比和最大直线偏离随人为弯曲等级总体增加，说明 C0-C3 的运动学操纵确实产生了预期效果。同时，
true residual 从 C0/C1 的约 0.20 增加到 C3 的 0.345，说明越夸张的动作整体越不符合 checkpoint 学到的
policy distribution。

但是 VLA legibility 并未单调增加：

- C0 最差；
- C1 最高；
- C2 回到约 0.5；
- C3 虽高于 0.5，但只有一侧正确。

测试集路径比与 VLA Early-AUC 的 Spearman correlation 只有 0.286；最大直线偏离与 VLA Early-AUC
的相关为 0.167；几何 observer AUC 与 VLA AUC 的相关为 -0.095。样本只有 8 条，这些只能作为描述，
不能做强统计推断。

### 6.3 逐条测试轨迹结果

下表中的 mean margin 为：

\[
\overline{D_{\mathrm{wrong}}-D_{\mathrm{true}}}.
\]

正值支持真实指令，负值支持错误指令。

| 条件 | 目标 | true residual | wrong residual | mean margin | Early-AUC | 最终 belief | 最终正确 | seed 符号一致率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | left | 0.1699 | 0.1488 | -0.0211 | 0.106 | ~0 | 否 | 1.000 |
| C0 | right | 0.2306 | 0.2045 | -0.0262 | 0.045 | 0 | 否 | 1.000 |
| C1 | left | 0.1954 | 0.1996 | +0.0042 | 0.951 | 0.999 | 是 | 1.000 |
| C1 | right | 0.2137 | 0.2135 | -0.0002 | 0.650 | 0.410 | 否 | 0.905 |
| C2 | left | 0.2280 | 0.2988 | +0.0708 | 0.948 | 1.000 | 是 | 1.000 |
| C2 | right | 0.3048 | 0.2884 | -0.0164 | 0.047 | ~0 | 否 | 0.952 |
| C3 | right | 0.4300 | 0.4425 | +0.0124 | 0.855 | 1.000 | 是 | 0.909 |
| C3 | left | 0.2607 | 0.2393 | -0.0214 | 0.457 | ~0 | 否 | 1.000 |

该表直接验证了用户提出的机制：很多测试轨迹在两个语言条件下 residual 都很高，但 wrong residual 并没有
稳定地比 true residual 更高。五条最终错误轨迹中，错误指令反而具有更低或近似相同的平均 residual。

几个典型例子：

- C2-left 是最符合“legible separation”的例子：虽然 true residual 0.228 已远高于训练均值，但 wrong
  residual 增加到 0.299，产生明显正 margin 0.071，因此模型强烈支持真实目标；
- C2-right 则相反：true residual 0.305，wrong residual 0.288，模型稳定认为错误指令更合适；
- C0 两条正常轨迹的 true residual 已经很高，并且两条都出现负 margin，因此不能把所有失败归因于大弯曲；
- C3-right 虽然 absolute residual 极大，但仍有小幅正 margin，所以 relative belief 很高；这也说明只看
  posterior 而不看 absolute energy 会产生误导。

### 6.4 稳健性实验

| 设置 | action dims | sample × seed | flow-time | 训练最终正确率 | 测试最终正确率 | 测试 Early-AUC | 测试/训练 true energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 主设置 | 8 | 8 × 3 | [0.3, 0.7] | 0.969 | 0.375 | 0.507 | 16.9× |
| 仅 joint action | 7 | 8 × 3 | [0.3, 0.7] | 0.969 | 0.375 | 0.505 | 15.7× |
| 宽 flow-time | 8 | 24 × 1 | [0.1, 0.9] | 0.969 | 0.500 | 0.565 | 20.9× |

结论如下：

- 排除 gripper residual 后，8 条测试轨迹的正确/错误模式与主设置完全相同；
- 把 24 个 sample 分布到更宽的 \([0.1,0.9]\) flow-time，只挽救了 1 条测试轨迹；
- 三种设置的训练正确率始终为 31/32；
- 测试始终只有 3/8 到 4/8；
- 主设置测试 chunk 的 seed-margin 符号一致率为 97.1%。

因此“总体准确率接近随机”并不意味着 Monte Carlo 随机噪声导致模型碰巧猜错。更准确的表达是：

> 聚合性能处在 chance level 附近，但模型在许多 chunk 上会跨 seed 稳定地产生错误方向的 residual
> ordering；这是结构化、可复现的模型偏差，而不是数值采样随机性。

### 6.5 输入和目标位置分布检查

为了判断低准确率是否来自简单数值越界，进行了以下检查：

| 检查量 | 微调轨迹 | 测试轨迹 |
|---|---:|---:|
| action 超出 checkpoint 1%-99% quantile 区间的比例 | 0.009 | 0.003 |
| state 超出 checkpoint 1%-99% quantile 区间的比例 | 0.004 | 0.000 |
| 平均 joint-velocity L2 norm | 0.514 | 0.483 |

测试动作和状态并没有比微调轨迹更频繁地超出 normalization 区间，平均 joint velocity 也没有更大。
因此当前失败不能用“弯曲轨迹速度过大，导致动作数值超出模型范围”简单解释。

目标位置方面：

- 8 个测试 grasp location 中 7 个位于所有训练 grasp location 的轴对齐包围盒内；
- 唯一越界的点只在一个坐标上超出约 1.9 mm；
- 每个测试目标与最近同侧训练目标的距离为 4.6–30.8 mm。

所以测试目标位置并不是显著超出训练工作空间。但是，“位置在数值范围内”并不代表对应的 RGB appearance、
初始关节姿态、背景、光照、手腕视角和动作序列联合分布与训练一致。

---

## 7. 为什么识别成功率低

### 7.1 用户提出的解释哪里是正确的

用户提出的核心解释是：训练集中没有刻意加入大幅 legible 弯曲，因此对于测试中的弯曲 chunk，模型只能发现
它们在两个语言条件下都很不典型，导致两个 residual 都增大，却没有学会让 wrong-instruction residual 增大得
更多。

这一解释得到部分支持：

1. C2、C3 的 absolute true residual 明显高于 C0、C1；
2. C3 的平均路径比达到 2.239，true residual 也达到最高的 0.345；
3. C2-right、C3-left 等轨迹确实表现为两个 residual 都很大，但 wrong residual 没有更大，甚至更小；
4. 训练数据只有普通 task completion，没有明确教模型“这种夸张方向偏置是在沟通真实目标”。

因此可以合理地说：

> natural-only demonstrations 没有保证 VLA action field 会把人为设计的 communicative exaggeration
> 映射成更大的 instruction-separation margin。

### 7.2 为什么它不能解释全部结果

最关键的反例是 C0：两条没有刻意弯曲、采集方式与普通 teleop 相同的测试轨迹全部错误，且 true residual
分别为 0.170 和 0.231，也已经是训练平均值 0.015 的十倍以上。

因此不能把低准确率完全归因为：

> 训练集没有大幅弯曲轨迹。

更完整的结论是：

> 整个 failure session 相对训练数据都产生了明显的 model compatibility shift；人为弯曲会进一步增加
> absolute residual，但不是测试泛化失败的唯一原因。

### 7.3 它是否呈现“随机性”

需要区分两种随机性：

1. **总体统计结果接近随机猜测**：是。3/8 和 Early-AUC 0.507 都在 chance-level 附近；
2. **模型每次随机采样都会随意改变答案**：不是。seed 符号一致率达到 97.1%。

模型的错误具有结构：

- C0-left 和 C0-right 都稳定错误；
- C2-left 稳定正确，C2-right 稳定错误；
- C3-right 稳定正确，C3-left 稳定错误。

这不是一个简单的“模型总猜左”或“模型总猜右”偏差，因为同一 pair 的左右都可能被反向判断。它更可能是
特定 observation、状态和动作组合与语言条件之间形成了错误但稳定的 compatibility ordering。

### 7.4 目前最可能的原因

根据现有证据，原因可能同时包括：

#### A. 数据量过小和 episode-level 记忆

微调只有 32 条轨迹，训练集识别又是对模型已经见过的 exact episodes 进行评估。31/32 可能体现强烈的
in-distribution fitting，而不是对“left/right motion semantics”的抽象泛化。

#### B. 跨 session 的视觉联合分布变化

即使方块坐标相近，failure 轨迹可能在以下方面不同：

- 光照和阴影；
- 物体的像素外观与相对遮挡；
- exterior 或 wrist camera 的细微变化；
- 机器人初始姿态；
- 操作者的速度曲线和微小停顿；
- 数据采集时间造成的背景或系统状态变化。

action/state quantile 检查只能排除明显低维数值越界，无法排除这些联合变化。

#### C. 训练数据缺少 communicative-motion variation

模型只学习了自然 task completion，可能认为大幅绕行在两个指令下都不正常。它没有被直接监督要求：

> 当动作先朝某一目标明显偏置时，应当降低另一个语言指令的 compatibility 更多。

#### D. Flow residual 的目标与人类 legibility 不完全一致

flow-matching model 被训练来预测 demonstration action field，不是被训练来模拟人类 observer。人类可以把一个
不自然但带有明显沟通目的的动作理解为 legible；VLA residual 可能只认为它不符合任何熟悉策略。

#### E. 累积 residual 与独立 chunk 假设

当前方法把不同 chunk 的 residual 加总为 trajectory evidence。这是一种实用 posterior proxy，但不是经过证明的
序列 likelihood factorization。非重叠 chunk 的边界位置也可能对个别轨迹产生影响。后续应加入 sliding-window
或多 offset chunk robustness。

#### F. 校准数据不独立

temperature 使用微调训练轨迹拟合，会使训练 belief 很饱和。它不改变 top-1 和 raw margin，但会影响 Early-AUC
曲线的数值尺度。正式实验应使用独立 validation trajectories 校准。

### 7.5 当前已基本排除的简单解释

目前结果不支持把失败主要归结为：

- gripper action 维度；
- 某一个 Monte Carlo seed；
- flow-time 只取 [0.3, 0.7]；
- joint velocity 明显过大；
- state/action 大量超出 checkpoint normalization；
- 方块位置远离训练工作空间；
- candidate left/right 的输入顺序 bug。

---

## 8. 是否应该加入刻意 legible 轨迹继续微调

答案取决于论文想回答的科学问题。

### 8.1 如果问题是“普通 demonstrations 是否自然产生 human-like observer”

在这一研究问题下，不应立刻把测试风格的弯曲轨迹加入训练，然后只报告新模型准确率提高。当前 negative result
本身很重要：

> 仅使用普通 task-completion demonstrations 微调出的 VLA，并没有自动成为对 communicative
> exaggeration 稳健的 intent observer，甚至对新的自然轨迹也缺乏可靠泛化。

如果先人为定义什么是 legible，再把这些轨迹加入训练，最后因为模型认为它们更 compatible 而声称模型“理解了
legibility”，会产生循环论证：

\[
\text{人为定义 legible}
\rightarrow
\text{加入训练}
\rightarrow
\text{模型拟合该分布}
\rightarrow
\text{用拟合结果证明 legible}.
\]

### 8.2 如果目标是构建更实用、更可靠的 observer

可以加入，但应把它定义为一个明确的实验因素：

> communicative-motion augmentation。

建议训练并比较至少两个 checkpoint：

1. **Natural-only**：只用普通 task completion；
2. **Natural + communicative**：加入方向正确的 mild/strong communicative trajectories。

最好再加入第三个 control：

3. **Natural + matched irrelevant curvature**：加入路径长度、速度和弯曲幅度相近，但弯曲方向不帮助区分目标的
   trajectory。

这样才能区分模型究竟学到了：

- “只要弯得大就认为更 legible”；还是
- “当轨迹方向更排斥错误目标时，才增加真实指令的相对证据”。

### 8.3 训练 augmentation 应包含什么

新的训练集应平衡包含：

- 自然直线轨迹；
- 自然存在的小幅 directional bias；
- mild communicative exaggeration；
- strong 但仍物理自然的 exaggeration；
- matched irrelevant-curvature controls；
- 左右目标数量平衡；
- 多个 layout；
- 多个采集 session；
- 如果可能，多个 operator；
- 相近的完成成功率、速度和路径时长分布。

极端且明显不真实的 C3 动作不一定适合作为主要训练分布。它们更适合作为 stress test，用来观察模型与人类在
强 OOD communicative motion 上何时分离。

### 8.4 不能再把现有 8 条当最终 test

现有 8 条已经用于观察结果、选择解释和提出改进，所以后续可以作为 development/pilot set，但不应在加入类似
轨迹训练后仍作为唯一的最终 test set。正式测试必须使用：

- 新采集的轨迹；
- 未参与训练或超参数选择的 layout；
- 新的时间顺序；
- 最好新的 session 或 operator；
- 预先固定的分析方案。

真正希望看到的不是简单的：

\[
D_{\mathrm{true}}\downarrow,
\]

而是：

\[
D_{\mathrm{wrong}}-D_{\mathrm{true}}\uparrow
\]

在新的 communicative trajectories 上稳定增大，同时 \(D_{\mathrm{true}}\) 不超过可信 OOD 范围，并且该变化与
人类判断方向一致。

---

## 9. 方块位置是否应该全程固定

### 9.1 不建议所有数据只使用一个固定 layout

完全固定方块位置会减少 nuisance variation，短期内可能提高测试准确率，但它无法区分模型到底学到了：

- 动作中的目标信息；
- 固定像素位置；
- 固定初始关节姿态；
- 固定 endpoint；
- 特定背景或相机画面模板。

如果论文目标包含 generalization 或 human-like motion understanding，只使用一个固定 layout 的外部效度很弱。

### 9.2 正确原则：对比内部固定，跨组有控制地变化

更合适的设计是：

> within-layout fixed, across-layout varied。

对每一个预定义 layout：

- 左右方块位置保持不变；
- left/right 指令都采集；
- C0/C1/C2/C3 或其他运动条件都在同一 layout 下采集；
- 每个 condition 有多个 repetition；
- condition 的采集顺序随机化或 Latin-square counterbalance。

跨 layout 时，再系统改变：

- 两块整体相对基座中线的平移；
- 两块之间的距离；
- 轻微深度差；
- 左右位置，但保持视觉可辨认和可安全抓取。

这样 condition comparison 不会与 layout 混杂，同时仍能测泛化。

### 9.3 固定 layout 可以作为诊断实验

虽然不建议整篇论文只用固定位置，但可以先做一个小型 fixed-layout diagnostic：

- 同一个 layout；
- 完全相同初始机械臂状态；
- 随机穿插 left/right；
- 只采多条自然 C0；
- 比较同一 session 和跨 session。

如果 fixed-layout 新自然轨迹仍然识别失败，说明问题不只是目标位置变化；如果 fixed-layout 成功而 unseen-layout
失败，则 layout/视觉泛化是主要瓶颈。这是定位原因的有效实验，不应与最终 benchmark 混为一谈。

### 9.4 推荐的 layout 划分

一个可执行的例子是：

- 8–12 个 training layouts；
- 2–4 个 validation layouts；
- 4–6 个完全未见的 test layouts；
- 每个 layout × target side × motion condition 至少 2–3 个 repetition。

可以进一步把测试分为：

1. **Seen-layout / new-trajectory**：layout 见过，但轨迹是新采集的；
2. **Unseen-layout / new-trajectory**：layout 和轨迹都未见；
3. **Cross-session**：相同 layout，但在不同日期重新采集；
4. **Cross-operator**：如果条件允许，由不同操作者 teleoperate。

这四种测试能分别诊断 episode memorization、layout 泛化、session shift 和 operator shift。

---

## 10. 建议优先修改的实验设计

### 10.1 第一优先级：先解决自然 held-out 泛化

在继续讨论 C1-C3 legibility 之前，最重要的是重新采集足够多的自然 C0 测试轨迹：

- 至少 8–12 个 layout；
- 每个 layout 一对 left/right，最好有多个 repetition；
- 与训练轨迹相同的普通 teleoperation 方式；
- condition 和 target 顺序随机；
- 记录并复位初始机器人状态；
- 保持 camera calibration、曝光和场景照明可追踪。

需要先回答：

> 这个 checkpoint 能否识别从同一任务分布重新采集的正常新轨迹？

如果自然 C0 仍接近 chance，那么直接加入弯曲训练数据可能只是掩盖基础 generalization 问题。

### 10.2 第二优先级：使用 factorial design 隔离曲率效应

建议把数据组织为：

\[
\text{layout}
\times
\text{target side}
\times
\text{motion condition}
\times
\text{repetition}.
\]

一个简化条件集可以是：

- C0 natural/direct；
- C1 mild goal-directed exaggeration；
- C2 strong but still natural goal-directed exaggeration；
- C-control matched irrelevant curvature。

C3 extreme 可以保留为单独的 OOD stress test，而不是与前三个条件一起拟合简单的单调趋势。

### 10.3 第三优先级：严格划分 train/validation/test

必须避免同一 layout 或几乎相同 trajectory 同时出现在训练和最终测试中。建议：

- 按 layout 分割，而不只是按 trajectory 随机分割；
- temperature 和 OOD threshold 只在 validation 上确定；
- test set 在模型和分析决策冻结前不查看；
- 现有 8 条只作为 pilot/development data。

### 10.4 第四优先级：同时报告相对 legibility 与绝对 compatibility

正式结果中，每条 trajectory 至少报告：

- true residual；
- wrong residual；
- residual margin；
- cumulative belief curve；
- Early-AUC；
- final accuracy；
- recognition time；
- path length、path ratio、最大 line deviation；
- task success 和 completion time。

建议使用训练或 validation energy distribution 建立 OOD gate。例如，当：

\[
D_{\mathrm{true}}>Q_{0.99}^{\mathrm{validation}}
\]

或两个 candidate residual 都过高时，把模型输出标记为“unsupported/OOD”，而不是强制解释两者中较小的一个
为可靠意图判断。

### 10.5 第五优先级：改进几何 baseline

把目标坐标改为来自第一帧可获得的信息，而不是完整测试轨迹未来的 grasp point：

- 手工标注；
- AprilTag；
- 相机标定后的 block center；
- object detector + depth/平面映射。

同时可以加入更接近经典 legibility 的 baseline，例如基于起点、候选目标和到目前为止 path cost 的
Boltzmann observer，而不只使用 instantaneous distance。

### 10.6 第六优先级：技术稳健性分析

建议补充：

- overlapping 16-step windows；
- 不同 window offsets；
- 每步或短窗口 temporal residual；
- 更多 Monte Carlo seeds；
- calibration-free raw-margin Early-AUC；
- bootstrap trajectory-level confidence intervals；
- left/right side-stratified results；
- 同一输入下 prompt paraphrase robustness；
- 图像、state 和 action 的受控 ablation，以定位联合分布偏移来源。

其中不能简单地打乱图像和动作后作强结论，因为 observation/action 必须物理对齐。更好的方法是重新采集受控的
same-layout、same-start trajectories，逐项改变 session、lighting 或 motion style。

---

## 11. 建议的下一轮完整实验路线

### Stage A：基础 generalization diagnosis

目标：确定当前失败是否首先来自新的普通轨迹。

1. 多 layout 采集新的 C0 natural left/right；
2. 同时包含 same-session 和 cross-session 重复；
3. 分别报告 seen-layout 与 unseen-layout；
4. 不更新 checkpoint，先对原模型做冻结评估；
5. 检查 absolute energy、margin 和视觉/状态变化。

### Stage B：受控 legibility characterization

目标：在相同 layout 下判断 communicative exaggeration 是否改变 VLA 与几何 observer 的 belief。

1. 每个 layout 采 C0、mild、strong、irrelevant control；
2. 左右目标完全平衡；
3. 随机化条件采集顺序；
4. 每个 cell 多次重复；
5. 保持任务成功率，并记录 path length、时间、jerk 等效率指标。

### Stage C：communicative augmentation

目标：判断加入 communicative demonstrations 是否改善新的未见轨迹，而不是仅拟合已有测试集。

1. 训练 natural-only checkpoint；
2. 训练 natural + communicative checkpoint；
3. 可加入 natural + irrelevant-curvature control checkpoint；
4. 在全新 layouts、全新 trajectories 上冻结比较；
5. 检查改善是否来自更低 true residual，还是更大的 wrong–true margin；
6. 检查是否损害自然轨迹的 task success 和 predictability。

### Stage D：human study

目标：验证 VLA observer 是否与人类意图判断一致，而不是只提高模型内部分类准确率。

让参与者只观看与 VLA 对齐的固定 RGB camera streams，并观看不同 trajectory prefix，例如：

\[
20\%,40\%,60\%,80\%.
\]

同一参与者不应连续观看同一轨迹的多个越来越长的 prefix，以避免记忆和 commitment。主要问题应当是：

> What is the robot trying to do?

选项为 left、right 和 not sure，并另外记录 confidence。最终比较：

- human true-intent proportion；
- VLA model belief；
- 几何 observer belief；
- human–model correlation；
- JS divergence；
- VLA entropy 与 human confidence；
- mixed-effects logistic regression，其中 participant 和 trajectory 为 random effects。

现有 8 条轨迹非常适合作为 model–human disagreement pilot，因为几何信息清楚、VLA 却常常稳定判断错误。

---

## 12. 当前结果对论文叙事的影响

### 12.1 当前可以声称的内容

基于这次 pilot，可以谨慎声称：

1. VLA flow residual 可以在微调轨迹上产生很强的 instruction separation；
2. 这种 separation 没有自动泛化到当前 8 条新轨迹；
3. 大幅弯曲会提高 absolute residual，但不会保证提高 relative legibility；
4. 几何 observer 能读出全部 8 条轨迹的正确空间目标，证明 motion 本身包含可用信息；
5. VLA 与几何 observer 的分歧是稳定的模型偏差，不是简单 Monte Carlo 噪声；
6. model-native observer 必须同时考虑 relative intent belief 和 absolute compatibility/OOD；
7. ordinary task demonstrations 并不必然产生 human-like communicative observer。

### 12.2 当前不能声称的内容

当前数据不支持声称：

1. C1 一定比 C0 更 legible，或 C3 一定比 C2 更 legible；
2. 弯曲程度对 VLA belief 具有因果作用；
3. checkpoint 的总体 intent-recognition accuracy 就是 37.5%；当前只有 8 条、且存在 session 混杂；
4. VLA observer 与人类一致；尚未收集 human judgment；
5. 模型错误完全由缺少弯曲训练数据导致；C0 反例否定了这一单因解释；
6. 几何 observer 是一个完全 online、只看当前 RGB 的 observer；它当前使用已知候选 endpoint；
7. residual 是精确、可跨数据集比较的 likelihood。

### 12.3 一个更诚实且更有研究价值的当前故事

目前最有说服力的叙事不是“VLA 已经能准确测量 legibility”，而是：

> VLA 的 instruction-conditioned action field 在训练内轨迹上能够形成强烈的目标证据，但这种
> model-native observer 在新的自然和 communicative trajectories 上可能因联合分布偏移而失效。
> 人类和简单几何 observer 仍可读出目标，说明生成动作的 policy compatibility 与观察者感知的
> legibility 并不等价。由此需要显式研究 natural demonstration、communicative augmentation、OOD
> detection 和 human validation 之间的关系。

这个 negative result 并不会破坏论文问题，反而揭示了“把 actor 直接当 observer”时最关键的科学边界。

---

## 13. 推荐修改清单（按优先级）

### 必须修改

1. 补采多组自然 C0 held-out trajectories，先验证基础泛化；
2. 同一 layout 下重复所有 motion conditions，消除 layout–condition 混杂；
3. 随机化或 counterbalance 采集顺序；
4. 增加 repetition，不能每个条件只有一对；
5. 把现有 8 条降级为 pilot/development set；
6. 按 layout/session 严格划分 train、validation、test；
7. 使用 validation 而不是训练 episode 校准 temperature；
8. 同时报告 absolute energy、relative margin 和 belief；
9. 为两个 residual 都很高的情况加入 OOD/unsupported 判断；
10. 在论文中避免把 residual 称为 normalized likelihood。

### 强烈建议修改

1. 用第一帧标注或检测到的 block coordinate 替代未来 grasp endpoint，构造几何 baseline；
2. 加入 irrelevant-curvature control；
3. 比较 natural-only 与 communicative-augmented checkpoint；
4. 增加 sliding-window/offset robustness；
5. 记录 initial robot pose、camera calibration、lighting/session 和 operator；
6. 对 C3 extreme 单独作为 OOD stress test，不强行假设其属于自然单调曲率序列；
7. 在 human study 前冻结模型、候选 prompt、temperature 和分析代码。

### 可以作为扩展

1. action-semantic task：pour versus handover；
2. sequential/shared-prefix task；
3. paraphrase sensitivity；
4. 多候选而非只有 left/right；
5. 经典 cost-based legibility baseline；
6. 用 human data 研究 model calibration，而不仅是 top-1 agreement；
7. 后续探索 belief-aware steering，但与本篇 observer characterization 分开。

---

## 14. 已实现的代码、测试与实验产物

### 14.1 代码

- `src/openpi/instruction_likelihood/droid_dataset.py`：原始 DROID HDF5/MP4 loader、日期/配对检查、
  motion/grasp segmentation 和 action chunk；
- `examples/droid/evaluate_spatial_legibility.py`：checkpoint residual scorer；
- `examples/droid/analyze_spatial_legibility.py`：belief、Early-AUC、几何 baseline、OOD 诊断和图表；
- `examples/droid/compare_spatial_legibility_robustness.py`：三种计分设置对比；
- `tests/instruction_likelihood/test_droid_dataset.py`：数据发现、抓取边界和 padding 测试；
- `src/openpi/models/pi0_test.py`：common-noise 和 candidate-order 测试。

最终 Ruff 检查通过，相关测试共 20 项通过。

### 14.2 主实验产物

目录：

```text
/home/hci-lab/repos/openpi/artifacts/spatial_legibility_eval_39999
```

主要文件：

- `REPORT.md`：英文实验报告；
- `RESULTS_ZH.md`：中文结果摘要；
- `ROBUSTNESS.md`：稳健性实验；
- `summary.json`：汇总数值；
- `episode_manifest.csv`：40 条轨迹及数据边界；
- `chunk_scores.csv`：912 行 per-seed、per-chunk 原始 residual；
- `chunk_scores_aggregated.csv`：seed 聚合后的 chunk 证据；
- `episode_metrics.csv`：逐轨迹结果；
- `condition_metrics.csv`：C0-C3 汇总；
- `heldout_belief_curves.png`：VLA 与几何 belief 曲线；
- `predictability_legibility_tradeoff.png`：absolute compatibility 与 relative legibility；
- `condition_summary.png`：条件级汇总图。

稳健性实验另存于：

```text
artifacts/spatial_legibility_eval_39999_joint_only
artifacts/spatial_legibility_eval_39999_wide_tau
```

---

## 15. 最终总结

当前项目已经完成了一次有价值的 end-to-end pilot：从 32 条自然 demonstrations 微调 checkpoint，使用 8 条
严格未参与训练的自然/人为弯曲轨迹，构造 VLA residual observer，并与几何 observer 比较。

实验支持用户关于“两个指令下 residual 都会因 OOD 弯曲而变大”的部分判断，但也表明问题更深：新的正常 C0
轨迹同样产生了很高 residual 并全部判断错误。因此，当前失败是 natural held-out generalization、session-level
joint distribution shift 和缺少 communicative-motion coverage 的组合，不能只通过“训练时没加大弯曲”解释。

下一步最合理的顺序是：

1. 先补采多 layout、多重复的自然 C0，定位基础泛化；
2. 再在同一 layout 内受控比较 natural、goal-directed curvature 和 irrelevant curvature；
3. 随后把 communicative augmentation 作为明确自变量训练新 checkpoint；
4. 最后在全新 test layouts 上与人类 observer 和无未来信息的几何 baseline 比较。

方块不应在整项研究中永远固定，但在每个受控比较 block 内必须固定；最好的原则是：

\[
\boxed{\text{同一 layout 内固定，跨 layout 有计划地变化，train/test 按 layout 隔离。}}
\]

如果后续结果显示 communicative augmentation 在全新 layout 上稳定增加
\(D_{\mathrm{wrong}}-D_{\mathrm{true}}\)，保持 true residual 在可信范围内，并与 human belief 同向变化，
那么才有充分证据说模型学到了可泛化的 communicative intent distinction，而不是简单记住了某些弯曲轨迹。
