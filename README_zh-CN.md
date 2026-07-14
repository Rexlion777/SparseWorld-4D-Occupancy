# SparseWorld 可靠性实验室

[简体中文](README_zh-CN.md) | [English](README.md)

**面向真实传感器退化的鲁棒 4D Occupancy 系统算法项目。**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](README.md#quick-start)
[![Core tests](https://github.com/Rexlion777/SparseWorld-4D-Occupancy/actions/workflows/core-tests.yml/badge.svg)](https://github.com/Rexlion777/SparseWorld-4D-Occupancy/actions/workflows/core-tests.yml)
[![Task](https://img.shields.io/badge/Task-4D%20Occupancy-7B61FF)](README.md#system-at-a-glance)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-00A6D6)](README.md#evaluation-protocol)

> 项目将研究型 SparseWorld 模型扩展为可审计感知系统：注入相机故障，追踪故障传播，使用因果时序记忆修复缺失相机特征，并同时评估恢复收益与虚假占据风险。

![R8 degradation recovery summary](assets/r8_result_card.svg)

## R8 核心结果

R8 缓存上一个有效时刻的同相机 FPN 特征，仅替换当前失效的前向相机组。在冻结的退化评测协议下：

| 传感器退化 | 相对退化基线的结果 |
|---|---:|
| 前向三相机缺失 | **FN 相对降低 23.2%** |
| 前向三相机缺失，4 s 未来时域 | **FN 相对降低 10.5%** |
| 单前向相机缺失 | **FN 相对降低 17.4%** |
| 运动模糊 | **FN 相对降低 7.1%** |
| 占据密度扩张 | **控制在 +2.3% 至 +6.0%** |

这些是故障恢复结果，不是 nuScenes 榜单结果。

![R8 BEV repair comparison](assets/r8/r8_bev_repair_comparison.png)

## 系统概览

- 每个样本 **30 张图像**：5 个时序帧 × 6 个环视相机。
- **1,040 个 Query** 表示 200 × 200 × 16，即 **640,000 个体素**。
- 输出 0 s、2 s、4 s、6 s 的语义占据。
- 故障集覆盖相机缺失、前向三相机缺失、后向相机缺失、运动模糊和低照度。
- 评测包含分区域、分类别、分时域 FN/FP，Occupancy 密度、时序稳定性与延迟。

## Template 1 四种可视化

| 环视观测 + 未来 Occupancy | Strict BEV |
|---|---|
| ![Template 1 overview](assets/template1/01_observation_future_overview.png) | ![Template 1 strict BEV](assets/template1/02_strict_bev.png) |
| **前视/自车视角** | **GT 与 SparseWorld 时序对比** |
| ![Template 1 front view](assets/template1/03_front_view.png) | ![Template 1 GT versus prediction](assets/template1/04_gt_vs_prediction_rollout.png) |

## Robot 当前工作的只读审计

2026-07-14 对 Robot 工作区进行了只读审计，未复制和提交其源码。当前方法主线为：

```text
历史 Query 记忆
→ 自车运动补偿
→ 相机投影与置信度加权 splat
→ 四层 FPN 重建/局部残差/时序传输或检索
→ R8 故障相机特征
→ 冻结 Occupancy Head
→ 特征与任务梯度对齐审计
```

审计结论：

1. 四层 FPN 重建契约、健康相机原样透传和零初始化一致性均已验证。
2. 特征重建误差降低，但 Occupancy 任务收益未能在独立窗口稳定泛化。
3. 特征损失与任务损失的平均梯度余弦为 `-0.487` 和 `-0.256`，审计窗口中负梯度比例均为 100%。
4. PCGrad 能去除直接对立分量，但仍未通过预注册的泛化和 false-free 门槛。

因此，**R8 仍是当前有证据支持的主线**。Q2F、残差、传输、检索和 PCGrad 作为研究证据保留，不冒充成功提升。完整公式、因子设计和结果见 [MCQM × R8 双语只读审计](docs/ROBOT_READONLY_AUDIT.md)。

## 项目代码与实验量

仓库保留 31 个有技术递进关系的实验阶段，约 5.6 万行 Python 代码，包含：

- SW1–SW4：模型 bring-up、几何、Query support 与语义激活。
- SW5–SW7：传感器故障注入、传播诊断与可靠性图。
- SW8–SW12：定向微调、贡献者路由与安全修复。
- SW13：因果特征记忆、密度约束与 R8 规模验证。
- SW14：可学习门控、残差、重排与负结果审计。
- MCQM/SWVIS：运动补偿 Query 记忆和 Template 1/4D 可视化。

建议从 [源码地图](docs/CODE_MAP.md) 开始阅读。

## 仓库结构

```text
SparseWorld-4D-Occupancy/
├── src/sparseworld_reliability/   # 轻量可测试可靠性核心
├── tests/                         # 因果、形状与指标契约
├── scripts/lidar_system_algorithm/
│   └── sparseworld_mainline/       # 31 个真实实验阶段
├── docs/                          # 架构、实验、结果、可复现性与只读审计
├── assets/                        # Template 1 与 R8 证据画廊
└── external/SparseWorld           # 上游子模块
```

## 快速测试

```bash
git clone --recurse-submodules https://github.com/Rexlion777/SparseWorld-4D-Occupancy.git
cd SparseWorld-4D-Occupancy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

完整 nuScenes/SparseWorld 链路额外需要 CUDA/MMCV 环境、数据集和 checkpoint，详见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 上游归属

本项目扩展自 [MSunDYY/SparseWorld](https://github.com/MSunDYY/SparseWorld)。传感器退化、因果特征记忆、评测契约、可靠性诊断与结果解读是本项目的扩展工作。
