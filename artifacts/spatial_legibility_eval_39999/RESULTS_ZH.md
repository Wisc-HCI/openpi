# Spatial legibility 实验结果（checkpoint 39999）

## 结论

这个 checkpoint 能非常好地区分参与微调的 32 条轨迹，但不能可靠地区分 8 条未参与训练的轨迹。
主设置下，训练轨迹在抓取前的最终指令识别为 **31/32（96.9%）**，留出轨迹只有
**3/8（37.5%）**；留出集 Early-AUC 为 **0.507**，接近随机基线 0.5。扩大 flow-time
采样范围后结果为 4/8，而去掉夹爪动作维度后仍为 3/8，因此主要结论不依赖这两项计分选择。

同一批留出轨迹上的几何观察者达到 **8/8**、Early-AUC **0.838**。这说明轨迹本身包含清晰的
左右目标信息，但当前 VLA residual observer 没有把这种信息稳定地泛化到未见轨迹。现有结果不支持
“该 VLA 可以直接作为可靠的人类 legibility observer”的结论；它更适合被报告为一个重要的
distribution-shift failure case，并在后续 user study 中检验这种偏差是否也出现在人类判断中。

## 评估协议

- 训练参考集：`success/2026-08-09` 的 32 条轨迹、16 对左右任务。
- 留出集：`failure/2026-08-09` 的 8 条轨迹、4 对，按采集时间标记为 C0-C3。
- 其他日期全部排除；留出轨迹没有写入或重新转换进微调数据集。
- 候选指令固定为 `pick up the left block` 和 `pick up the right block`。
- 每个 observation/action segment 使用 checkpoint 的原始 DROID transform 和 normalization。
- 轨迹按 16 个 action 的非重叠 chunk 计分；抓取边界处强制切分，尾部 padding 不参与 residual，
  因而某一 prefix 不使用未来动作证据。
- 主设置对每个 chunk 使用 3 个独立 seed、每个 seed 8 个 common-noise flow sample，
  flow time 为 [0.3, 0.7]。
- posterior temperature 只用 32 条微调轨迹校准，用于画 belief 曲线；最终正确率和 raw energy
  margin 的符号不依赖这个校准。

## 汇总结果

| 数据 | 轨迹数 | 抓取前最终正确率 | 局部 chunk 正确率 | Early-AUC | 几何基线正确率 | 几何 Early-AUC | 平均 true energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 微调轨迹 | 32 | 0.969 | 0.843 | 0.854 | 1.000 | 0.676 | 0.015 |
| 留出轨迹 | 8 | 0.375 | 0.444 | 0.507 | 1.000 | 0.838 | 0.254 |

留出轨迹的 true-instruction residual energy 是微调轨迹的 **16.9 倍**，而且 8 条轨迹全部位于
微调集 energy 分布的第 100 百分位。这意味着即便某条轨迹的相对 belief 很高，两个候选指令的
绝对 policy compatibility 仍可能都很差。这里的 residual energy 是 flow-matching compatibility
proxy，不是规范化 likelihood。

## C0-C3 描述性结果

| 条件 | 路径比 | 最大直线偏离 | VLA Early-AUC | 最终正确率 | 几何 Early-AUC | true energy |
|---|---:|---:|---:|---:|---:|---:|
| C0 正常/直接 | 1.106 | 0.077 m | 0.075 | 0/2 | 0.749 | 0.200 |
| C1 轻微弯曲 | 1.202 | 0.103 m | 0.800 | 1/2 | 0.909 | 0.205 |
| C2 明显弯曲 | 1.449 | 0.214 m | 0.497 | 1/2 | 0.826 | 0.266 |
| C3 极端弯曲 | 2.239 | 0.353 m | 0.656 | 1/2 | 0.867 | 0.345 |

曲率增加时，绝对 residual energy 整体升高，符合“越夸张越不像训练策略”的预期；但 VLA
legibility 并不单调增加。尤其是 C0 的两条正常轨迹都被判断成相反指令，因此留出失败不能只归因于
刻意弯曲造成的 OOD。路径比与 VLA Early-AUC 的 Spearman 相关只有 **0.286**，几何基线 AUC 与
VLA AUC 的相关为 **-0.095**。这些相关仅有 8 个样本，只能用于描述。

## 稳健性与分布审计

| 设置 | action 维度 | sample × seed | flow-time | 训练正确率 | 留出正确率 | 留出 AUC | 留出/训练 energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 主设置 | 8 | 8 × 3 | [0.3, 0.7] | 0.969 | 0.375 | 0.507 | 16.9× |
| 仅关节动作 | 7 | 8 × 3 | [0.3, 0.7] | 0.969 | 0.375 | 0.505 | 15.7× |
| 宽 flow-time | 8 | 24 × 1 | [0.1, 0.9] | 0.969 | 0.500 | 0.565 | 20.9× |

- 主设置留出 chunk 的 seed-margin 符号一致率为 **97.1%**，所以 3/8 不是明显的随机采样抖动。
- 去掉 gripper residual 后，8 条留出轨迹的正确/错误结果与主设置完全一致。
- 宽 flow-time 设置改变了 1/8 条轨迹的最终结果，但仍只有 chance-level 的 4/8。
- 留出集落在 checkpoint 动作 1%-99% normalization 区间外的比例为 **0.003**，低于微调轨迹的
  **0.009**；state 对应比例为 **0.000** 和 **0.004**。因此不是简单的原始关节值/速度越界。
- 8 个留出目标中 7 个位于训练目标的全局轴对齐包围盒内；唯一越界点只超出 **1.9 mm**。
  每个留出目标与最近的同侧训练目标距离为 **4.6-30.8 mm**。目标位置变化本身也不足以解释
  16.9 倍的 energy 差异。
- 已加入 candidate-order 等变性测试，确认交换左右候选只会交换输出列，不会产生标签顺序偏差。

综合来看，distribution shift 更可能来自完整的视觉、机器人状态和动作序列之间的联合关系，
而不是单一低维变量超出训练范围。凭现有 8 条轨迹还不能进一步定位是光照/画面、初始姿态、
teleop 动力学，还是模型对训练 episode 的记忆造成。

## 对论文与 user study 的含义

1. 论文中应把这个量称为 **instruction-conditioned action compatibility / posterior proxy**，而不是
   calibrated likelihood。
2. VLA belief 与 absolute energy 必须同时报告。建议在 energy 超出训练阈值时把 observer 标记为
   OOD/不可信，不能把两个很差候选中的相对胜者解释成可靠的意图判断。
3. 当前 C0-C3 每个条件只有一对轨迹，而且 layout 与采集时间随条件一起变化，无法对“弯曲程度的
   因果效应”做统计推断。正式实验需要每个条件有多组随机 layout，并随机化条件采集顺序。
4. 后续 human study 应把这 8 条轨迹作为 model-human disagreement pilot：让参与者只看匹配的
   prefix，比较 human true-intent proportion 与冻结后的 VLA belief，同时加入几何基线。
5. 最关键的新增采集应首先是多组全新 layout 的 C0 自然轨迹。只有先确认正常 held-out 数据能否
   泛化，才适合解释 C1-C3 的 communicative exaggeration 效果。

## 产物

- `REPORT.md`：英文主报告与逐轨迹表。
- `ROBUSTNESS.md`：三种计分设置的稳健性比较。
- `chunk_scores.csv`：每个 seed、每个 chunk 的原始 residual。
- `episode_metrics.csv`：40 条轨迹的聚合结果和 OOD 诊断。
- `condition_metrics.csv`：C0-C3 汇总。
- `heldout_belief_curves.png`：留出轨迹随时间变化的 VLA/几何 belief。
- `predictability_legibility_tradeoff.png`：absolute compatibility 与 relative legibility。
