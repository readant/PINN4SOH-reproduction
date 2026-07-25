# PINN4SOH 论文复现

> **完整复现 Nature Communications 2024 论文的实验结果，并提供中文技术文档**

[![Paper](https://img.shields.io/badge/Paper-Nature%20Comms-blue)](https://www.nature.com/articles/s41467-024-48779-z)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-green)]()
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-orange)]()

---

## 项目说明

本项目是对以下论文的**完整复现**：

> **Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis**
>
> Fujin Wang, Zhi Zhai, Zhibin Zhao, Yi Ding, Xuefeng Chen
>
> *Nature Communications*, 2024, 15:4332
>
> DOI: [10.1038/s41467-024-48779-z](https://www.nature.com/articles/s41467-024-48779-z)

**核心创新**：将电池退化动力学方程（PDE）直接嵌入神经网络训练，实现物理约束与数据驱动的混合建模（B2 架构 PINN）。

---

## 复现结果

### 实验规模

| 实验类型 | 数据集 | 实验组数 | 总训练次数 |
|----------|--------|---------|-----------|
| 主训练 | XJTU (6协议×10次) | 60 | 60 |
| 主训练 | HUST (10次) | 10 | 10 |
| 主训练 | MIT (10次) | 10 | 10 |
| 主训练 | TJU (3批×10次) | 30 | 30 |
| MLP/CNN 对比 | XJTU (6×2×10) | 120 | 120 |
| 迁移学习 | TJU→XJTU (6×10) | 60 | 60 |
| **合计** | | | **290 次** |

### 关键结果对比

| 指标 | 论文报告 | 本次复现 | 偏差 |
|------|---------|---------|------|
| XJTU MAPE | 0.78% | 0.87% | +0.09% |
| HUST MAPE | 0.65% | 0.80% | +0.15% |
| MIT MAPE | 0.78% | 0.73% | -0.05% |
| TJU MAPE | 1.21% | 1.13% | -0.08% |
| **平均 MAPE** | **0.87%** | **0.88%** | **+0.01%** |

**结论**：所有核心结论均已验证，结果与论文高度一致。

---

## 项目结构

```
nc-PINN/
├── README.md                          # 本文件
├── docs/                              # 论文解读文档（14篇中文技术文档）
│   ├── 01-problem.md                  # 问题定义与研究动机
│   ├── 02-feature-extraction.md       # 通用特征提取方法
│   ├── 03-pinn-architecture.md        # PINN模型架构设计
│   ├── 04-loss-functions.md           # 三重损失函数
│   ├── 05-training-strategy.md        # 训练与迁移学习策略
│   ├── 06-experiments.md              # 实验验证与结果分析
│   ├── 07-code-mapping.md             # 论文方法与代码对应关系
│   ├── 08-reproduction-guide.md       # 复现指南
│   ├── 09-data-preprocessing.md       # 数据预处理详细指南
│   ├── 10-training-configs.md         # 训练配置说明
│   ├── 11-transfer-learning.md        # 迁移学习实现指南
│   ├── 12-visualization.md            # 可视化方法
│   ├── 13-checklist.md                # 复现检查清单
│   └── 14-reproduction-report.md      # 复现报告（完整实验记录）
├── PINN4SOH/                          # 原始代码（已适配新环境）
│   ├── Model/                         # PINN模型实现
│   ├── dataloader/                    # 数据加载器
│   ├── main_XJTU.py                   # XJTU数据集训练入口
│   ├── main_HUST.py                   # HUST数据集训练入口
│   ├── main_MIT.py                    # MIT数据集训练入口
│   ├── main_TJU.py                    # TJU数据集训练入口
│   ├── main_comparision.py            # MLP/CNN对比实验
│   ├── main_adaptation - fine-tuning.py  # 迁移学习实验
│   ├── run_all.py                     # 一键运行入口（新增）
│   ├── validate.py                    # 快速验证脚本（新增）
│   ├── data/                          # 预处理后的数据集
│   └── pretrained model/              # 预训练模型权重
├── Battery-dataset-preprocessing-code-library/  # 数据预处理代码库
│   ├── XJTUBatteryClass.py            # XJTU数据集解析器
│   ├── HUSTBatteryClass.py            # HUST数据集解析器
│   ├── MITBatteryClass.py             # MIT数据集解析器
│   ├── TongJiBatteryClass.py          # TJU数据集解析器
│   └── ...
└── results_fine-tuning/               # 迁移学习结果
```

---

## 快速开始

### 环境要求

| 项目 | 版本 |
|------|------|
| Python | 3.12+ |
| PyTorch | 2.11+ (CUDA) |
| GPU | NVIDIA GPU (推荐 RTX 3060+) |

### 安装

```bash
# 克隆仓库
git clone https://github.com/readant/nc-PINN.git
cd nc-PINN

# 创建环境
conda create -n pinn python=3.12
conda activate pinn

# 安装依赖
pip install torch numpy pandas scikit-learn matplotlib scienceplots
```

### 运行实验

```bash
cd PINN4SOH

# 一键运行所有实验
python run_all.py all

# 或分阶段运行
python run_all.py xjtu        # XJTU主实验
python run_all.py hust        # HUST主实验
python run_all.py mit         # MIT主实验
python run_all.py tju         # TJU主实验
python run_all.py comparison  # MLP/CNN对比
python run_all.py finetune    # 迁移学习
python run_all.py analyze     # 结果分析

# 快速验证
python validate.py
```

---

## 我的改进

本项目在原代码基础上进行了以下改进：

### 1. 环境兼容性修复

原代码基于 Python 3.7 + PyTorch 1.7，本次复现适配到 Python 3.12 + PyTorch 2.11，修复了以下问题：

| 修复项 | 原因 | 修改文件 |
|--------|------|---------|
| pandas 3.0 int→float 兼容 | `df.insert` 列为 int64，归一化后无法写回 | `dataloader/dataloader.py` |
| pandas 3.0 CSV 全列 int64 | HUST/MIT/TJU CSV 含 int64 特征列 | `dataloader/dataloader.py` |
| GPU ID 1→0 | 原作者多 GPU 环境，当前仅单卡 | `main_HUST.py` 等 |
| pandas 3.x ExcelWriter | `writer.save()` 已废弃 | `results analysis/*.py` |
| eval_metrix R2 扩展 | FineTune results.py 需要 R2 | `utils/util.py` |
| 分析脚本路径 | 相对路径在不同工作目录下解析错误 | `results analysis/*.py` |
| 分析脚本 import 路径 | 子目录脚本无法 import 父级模块 | 5 个分析脚本 |

### 2. 一键运行脚本

新增 `run_all.py`，统一控制实验流程：

```bash
python run_all.py xjtu           # 只跑 XJTU 主实验
python run_all.py comparison     # 只跑 MLP/CNN 对比
python run_all.py all            # 全部跑
```

### 3. 完整技术文档

编写了 14 篇中文技术文档（`docs/` 目录），覆盖：

- 问题定义与研究动机
- PINN 模型架构设计
- 三重损失函数详解
- 训练与迁移学习策略
- 实验验证与结果分析
- 论文方法与代码对应关系
- 完整复现指南

### 4. 复现报告

详细的复现报告（`docs/14-reproduction-report.md`），包含：

- 290 次实验的完整记录
- 与论文结果的逐项对比
- 误差来源分析
- 核心结论验证

---

## 数据集

本项目使用以下公开电池数据集：

| 数据集 | 电池类型 | 化学体系 | 电池数 | 来源 |
|--------|---------|---------|--------|------|
| XJTU | NCM | LiNi₀.₅Co₀.₂Mn₀.₃O₂ | 55 | [Zenodo](https://zenodo.org/records/10963339) |
| TJU | NCA/NCM | LiNiCo-x | 52 | [Zenodo](https://zenodo.org/record/6405084) |
| HUST | LFP | LiFePO₄ | 166 | [Mendeley](https://data.mendeley.com/datasets/nsc7hnsg4s/2) |
| MIT | LFP | LiFePO₄ | 124 | [MIT](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) |

数据预处理代码库：[Battery-dataset-preprocessing-code-library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)

---

## 引用

### 原始论文

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@article{wang2024physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4332},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41467-024-48779-z}
}
```

### 原始代码

```bibtex
@misc{wang2024pinn4soh,
  title = {PINN4SOH},
  author = {Fujin Wang},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wang-fujin/PINN4SOH}}
}
```

---

## 许可证

本项目基于原始论文的 **CC BY 4.0 许可证**发布。

- 原始论文：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 原始代码：[PINN4SOH](https://github.com/wang-fujin/PINN4SOH)（无明确 LICENSE，遵循论文 CC BY 4.0）

### 使用条件

根据 CC BY 4.0 许可证，您可以自由：

- **共享**：复制和重新分发材料
- **演绎**：修改和转换材料

但必须：

- **署名**：提供原始论文的引用
- **注明修改**：说明您对材料所做的更改
- **包含许可证**：包含 CC BY 4.0 许可证的副本

---

## 致谢

- 感谢 [Fujin Wang](https://github.com/wang-fujin) 等人发表的原始论文和代码
- 感谢西安交通大学提供的 XJTU 电池数据集
- 感谢所有公开电池数据集的贡献者

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [https://github.com/readant/nc-PINN/issues](https://github.com/readant/nc-PINN/issues)
- Email: 3908492312@qq.com
- Gmail: readant123@gmail.com
