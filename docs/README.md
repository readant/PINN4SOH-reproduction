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
| [06-experiments.md](06-experiments.md) | 实验验证与结果分析 |
| [07-code-mapping.md](07-code-mapping.md) | 论文方法与代码实现对应关系 |

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

- **MAPE**: 0.87%（4个数据集平均）
- **数据规模**: 387个电池，310,705个样本
- **特征维度**: 16维（从充电曲线提取）
- **模型结构**: 解网络F + 动力学网络G
