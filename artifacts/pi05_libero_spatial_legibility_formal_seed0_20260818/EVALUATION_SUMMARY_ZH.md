# pi0.5-LIBERO 空间可读性正式单 seed 测评

完成时间：2026-08-18 13:22 CDT

## 实验范围

- 1 个 simulator seed（seed 0）
- 2 个目标间距：G1=10 cm、G2=12.5 cm
- 2 个物理布局：A、B
- 2 个真实目标：milk、orange juice
- 5 个轨迹条件：A1、L0、L1、L2、L3
- 共 40 条完整 episode；每条 290 步、256×256 双相机、HDF5+MP4
- 冻结 manifest SHA256：`456da6d01af8ec7a03e0ec7d694e9b96d0c4da7d35fbca7d0715852033cb92c9`
- checkpoint fingerprint：`52ee2b0ee4b640e775c665514feabacf7c1427b8bf699b51ea877e5b8f45c8e5`

## 主要结果

- 任务成功率：100%（40/40）
- 无碰撞率：100%（40/40）
- 主分析 L0–L3 balanced accuracy：C20=50.0%，C30=62.5%，C40=59.375%
- C30 matched Spearman rho：0.225（预注册目标 0.6）
- C30 中 L3>L0 的 matched group：50%（预注册目标 75%）
- C30 adjacent ordering success：62.5%（预注册目标 75%）
- L3 seed sign agreement：均值 90.25%，最小值 54%（要求每个条件至少 80%）
- 两个不稳定 L3 条件按规则从 10 seeds 扩展到了 50 seeds

平均 C30 随 level 从 L0 到 L3 上升（0.00212、0.00241、0.00447、0.00456），但这个 aggregate 趋势没有在 matched conditions 中稳定复现，因此不能据此声称单调空间可读性效应成立。

效率代价相对 matched L0：L1约+0.78%，L2约+2.95%，L3约+6.19%。几何 observer 的可读性和路径代价在每个 matched group 中均按预期增加。

## 重要失败与偏置

严格 sanity gate 未通过：方法学不变量（same prompt、candidate order、common noise、prefix cache）全部通过，但 orange-juice episode 的最终 pre-grasp chunk 仍错误偏向 milk；因此正式 audit 按设计拒绝通过。

描述性拆分显示明显的不对称：

- 所有五个 level 合计的 C30 accuracy：orange juice 90%，milk 35%
- Layout A 的 C30 accuracy：85%；Layout B：40%
- 左侧目标 C30 accuracy：75%；右侧目标：50%

这表明结果受到语义身份、物理布局或相机方向偏置影响，不能只解释成轨迹 exaggeration 的因果效果。

## 完整性状态

- 40 个 HDF5、40 个视频均存在；所有视频均为 290 帧
- 40 个 raw residual artifact 均通过 shape、dtype、finite、noise 复现和 residual-energy 精确重算
- source HDF5 SHA、checkpoint fingerprint、target index、实际间距和 matched initial simulator state 均通过复核
- 38 条使用 10 flow seeds；2 条预注册不稳定 L3 使用 50 flow seeds
- normalized physical actions 超出 checkpoint q01/q99 范围的比例为 0

由于 sanity gate 失败，这是一套完整、可复核的测量结果，但不是一个通过预注册有效性标准的正结果。不得通过事后更换 prompt、seed、布局或阈值将其改写为通过。

## 产物

- `pilot.json`：模型评分前的几何 pilot
- `frozen_geometry.json`：冻结场景和轨迹配置
- `rollouts/`：40 条 HDF5 和 MP4
- `observer/sanity_summary.json`：失败的 sanity gate 详情
- `observer/score_summary.json` 与 `observer/raw/`：seed-resolved residual
- `report/summary.json`：机器可读结果
- `report/report.md`：预注册指标报告
- `report/*.csv`：episode、chunk、aggregate 和 matched monotonicity 表
- `report/*.png`：11 类分析图
