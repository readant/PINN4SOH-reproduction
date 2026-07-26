# PINN4SOH 初学者指南

> 从看论文到写代码，一步步带你入门物理信息神经网络（PINN）在电池健康管理中的应用

## 学习目标

完成本指南后，你将能够：

1. 理解 PINN 的核心思想：物理约束 + 神经网络
2. 读懂 PINN4SOH 论文的关键公式和方法
3. 看懂代码实现，知道每行代码对应论文的哪个部分
4. 跑通第一个训练实验
5. 了解基于此论文的创新研究方向

## 前置知识

| 知识 | 要求 | 推荐资源 |
|------|------|----------|
| Python 基础 | 能读懂 Python 代码 | Python 官方教程 |
| PyTorch 基础 | 了解 Tensor、nn.Module、自动微分 | PyTorch 官方教程 |
| 深度学习基础 | 了解神经网络、损失函数、优化器 | 李宏毅机器学习 |
| 锂电池基础 | 了解 SOH、充放电曲线 | 论文 §1 即可 |

## 推荐学习顺序

```
第 1 天：阅读 01-paper-overview.md（30 分钟）
         ↓
第 2 天：阅读 02-core-method.md（1 小时）
         ↓
第 3 天：阅读 03-code-walkthrough.md（1.5 小时）
         ↓
第 4 天：动手运行 04-run-experiment.md（2 小时）
         ↓
第 5 天：阅读 05-understand-results.md（30 分钟）
         ↓
第 6 天：探索 06-innovation-paths.md（按兴趣选择）
         ↓
随时查阅：07-glossary.md 术语表
```

## 预计学习时间

- 快速浏览：3-5 天
- 深入学习：1-2 周
- 动手实践 + 创新探索：1-2 月

## 文档导航

| 序号 | 文档 | 内容 | 预计时间 |
|------|------|------|----------|
| 01 | [论文概览](01-paper-overview.md) | 一篇论文读懂 PINN4SOH | 30 min |
| 02 | [核心方法](02-core-method.md) | PINN 是怎么工作的 | 1 hour |
| 03 | [代码走读](03-code-walkthrough.md) | 从公式到 Python 代码 | 1.5 hours |
| 04 | [动手实验](04-run-experiment.md) | 跑通第一个训练 | 2 hours |
| 05 | [理解结果](05-understand-results.md) | 模型预测了什么 | 30 min |
| 06 | [创新方向](06-innovation-paths.md) | 研究机会与起步方案 | 1 hour |
| 07 | [术语表](07-glossary.md) | 关键概念速查 | 随时 |

## 论文信息

- **标题**: Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis
- **作者**: Fujin Wang, Zhi Zhai, Zhibin Zhao, Yi Di, Xuefeng Chen
- **期刊**: Nature Communications, 2024, 15:4332
- **DOI**: [10.1038/s41467-024-48779-z](https://doi.org/10.1038/s41467-024-48779-z)
- **代码**: [GitHub - wang-fujin/PINN4SOH](https://github.com/wang-fujin/PINN4SOH)
