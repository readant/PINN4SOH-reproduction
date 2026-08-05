# PINN for Battery SOH Estimation - 论文解读

> **论文**: Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis
> **期刊**: Nature Communications, 2024, 15:4332
> **作者**: Fujin Wang, Zhi Zhai, Zhibin Zhao, Yi Ding, Xuefeng Chen (西安交通大学)

---

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-problem.md](01-problem.md) | 问题定义与研究动机 |
| [02-feature-extraction.md](02-feature-extraction.md) | 通用特征提取方法 |
| [03-pinn-architecture.md](03-pinn-architecture.md) | PINN模型架构设计 |
| [04-loss-functions.md](04-loss-functions.md) | 三重损失函数 |
| [05-training-strategy.md](05-training-strategy.md) | 训练与迁移学习策略 |
| [06-experiments.md](06-experiments.md) | 实验结果与结论 |
| [07-code-mapping.md](07-code-mapping.md) | 论文方法与代码实现对应关系 |
| [08-verification-guide.md](08-verification-guide.md) | 完整验证流程（环境 → 实验 → 汇总） |
| [09-data-preprocessing.md](09-data-preprocessing.md) | 数据预处理与 CSV 格式 |
| [10-training-configs.md](10-training-configs.md) | 各数据集训练配置 |
| [11-transfer-learning.md](11-transfer-learning.md) | 迁移学习机制 |
| [12-visualization.md](12-visualization.md) | 结果可视化 |
| [13-checklist.md](13-checklist.md) | 验证检查清单 |
| [14-verification-report.md](14-verification-report.md) | 环境迁移验证报告 |
| [CHANGELOG.md](CHANGELOG.md) | 改动记录（动机 / 文件清单 / 验证方式） |
| [GIT_USAGE.md](GIT_USAGE.md) | Git 仓库使用说明 |

> **数据源纪律**：涉及实验数字的文档，一律以 `PINN4SOH/results/SUMMARY.md` 为准
> （由 `results_summary.py` 从原始产物生成）。

---

## 一句话概括

用物理信息神经网络（PINN）估计锂电池健康状态（SOH），将退化动力学方程直接嵌入神经网络训练，实现精度与泛化的兼顾。

## 核心创新点

```
传统方法:
  数据驱动 ──→ 精度高，泛化差
  物理模型 ──→ 稳定，计算成本高

本文方法 (B2架构):
  物理方程 ⊕ 神经网络 ──→ 精度高 + 稳定 + 可迁移
```

## 关键数字

| 项目 | 数值 | 说明 |
|------|------|------|
| MAPE | 论文 0.87% / 本仓库实测 **0.88%** | 四数据集平均；实测见 `PINN4SOH/results/SUMMARY.md` |
| 数据规模 | **387** 只电池、310,705 个样本 | 与论文口径一致 |
| 网络输入维度 | **17** | 16 个统计特征 + 1 个周期序号 |
| 模型结构 | 解网络 `Solution_u` + 动力学网络 `dynamical_F` | 代码中没有名为 G 的网络 |
