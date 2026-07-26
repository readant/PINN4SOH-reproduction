# PINN4SOH 代码验证与研究探索

> **基于官方代码的环境迁移验证，以及 PINN 在电池健康管理领域的学习与创新研究**

[![Paper](https://img.shields.io/badge/Paper-Nature%20Comms-blue)](https://www.nature.com/articles/s41467-024-48779-z)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-green)]()
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-orange)]()

---

## 项目说明

本项目**不是**独立复现，而是基于论文作者的官方开源代码进行验证与研究：

| 项目 | 说明 |
|------|------|
| **代码来源** | [PINN4SOH 官方仓库](https://github.com/wang-fujin/PINN4SOH) + [数据预处理库](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library) |
| **主要工作** | 环境迁移（Python 3.7→3.12, PyTorch 1.7→2.11）+ 代码修复 + 结果验证 |
| **仓库目的** | 学习 PINN 论文方法，探索电池健康管理领域的创新方向 |

### 原始论文

> **Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis**
>
> Fujin Wang, Zhi Zhai, Zhibin Zhao, Yi Ding, Xuefeng Chen
>
> *Nature Communications*, 2024, 15:4332
>
> DOI: [10.1038/s41467-024-48779-z](https://www.nature.com/articles/s41467-024-48779-z)

---

## 验证结果

在 Python 3.12 + PyTorch 2.11 环境下，官方代码核心算法有效性得到验证：

| 指标 | 论文报告 | 本环境验证 | 偏差 |
|------|---------|-----------|------|
| XJTU MAPE | 0.78% | 0.87% | +0.09% |
| HUST MAPE | 0.65% | 0.80% | +0.15% |
| MIT MAPE | 0.78% | 0.73% | -0.05% |
| TJU MAPE | 1.21% | 1.13% | -0.08% |
| **平均 MAPE** | **0.87%** | **0.88%** | **+0.01%** |

---

## 主要改进

### 1. 环境兼容性修复

原代码基于 Python 3.7 + PyTorch 1.7，本项目适配到 Python 3.12 + PyTorch 2.11：

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
cd PINN4SOH
python run_all.py all            # 运行所有实验
python run_all.py xjtu           # 只跑 XJTU 主实验
python run_all.py comparison     # 只跑 MLP/CNN 对比
python run_all.py finetune       # 迁移学习
```

### 3. 技术文档

编写了 14 篇中文技术文档（`docs/` 目录），覆盖论文方法解读、代码对应关系、复现指南等。

---

## 快速开始

### 环境要求

| 项目 | 版本 |
|------|------|
| Python | 3.12+ |
| PyTorch | 2.11+ (CUDA) |
| GPU | NVIDIA GPU (推荐 RTX 3060+) |

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/readant/nc-PINN.git
cd nc-PINN

# 创建环境
conda create -n pinn python=3.12
conda activate pinn

# 安装依赖
pip install torch numpy pandas scikit-learn matplotlib scienceplots

# 运行实验
cd PINN4SOH
python run_all.py all
```

---

## 项目结构

```
nc-PINN/
├── README.md                          # 本文件
├── docs/                              # 论文解读文档（14篇）
├── PINN4SOH/                          # 官方代码（已适配新环境）
│   ├── Model/                         # PINN模型实现
│   ├── dataloader/                    # 数据加载器
│   ├── main_*.py                      # 各数据集训练入口
│   ├── run_all.py                     # 一键运行入口（新增）
│   ├── validate.py                    # 快速验证脚本（新增）
│   ├── data/                          # 预处理后的数据集
│   └── pretrained model/              # 预训练模型权重
└── Battery-dataset-preprocessing-code-library/  # 数据预处理代码库
```

---

## 数据集

本项目使用以下公开电池数据集：

| 数据集 | 电池类型 | 化学体系 | 电池数 | 来源 |
|--------|---------|---------|--------|------|
| XJTU | NCM | LiNi₀.₅Co₀.₂Mn₀.₃O₂ | 55 | [Zenodo](https://zenodo.org/records/10963339) |
| TJU | NCA/NCM | LiNiCo-x | 52 | [Zenodo](https://zenodo.org/record/6405084) |
| HUST | LFP | LiFePO₄ | 166 | [Mendeley](https://data.mendeley.com/datasets/nsc7hnsg4s/2) |
| MIT | LFP | LiFePO₄ | 124 | [MIT](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) |

---

## 引用

### 原始论文

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
- 原始代码：[PINN4SOH](https://github.com/wang-fujin/PINN4SOH)

根据 CC BY 4.0 许可证，使用时需：署名原始论文、注明修改、包含许可证副本。

---

## 致谢

- 感谢 [Fujin Wang](https://github.com/wang-fujin) 等人发表的原始论文和代码
- 感谢西安交通大学提供的 XJTU 电池数据集
- 感谢所有公开电池数据集的贡献者

---

## 联系方式

- GitHub Issues: [https://github.com/readant/nc-PINN/issues](https://github.com/readant/nc-PINN/issues)
- Email: 3908492312@qq.com
- Gmail: readant123@gmail.com
