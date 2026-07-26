# 04 - 动手实验：跑通第一个训练

> 从环境配置到运行训练，一步步带你跑通代码

## 环境要求

| 项目 | 版本 |
|------|------|
| Python | 3.12+ |
| PyTorch | 2.11+ (CUDA) |
| GPU | NVIDIA GPU (推荐 RTX 3060+) |

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/readant/nc-PINN.git
cd nc-PINN
```

### 2. 创建虚拟环境

```bash
conda create -n pinn python=3.12
conda activate pinn
```

### 3. 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib scienceplots
```

## 数据准备

数据已包含在 `PINN4SOH/data/` 目录下：

```
data/
├── XJTU data/          # 西安交通大学（55个电池）
├── HUST data/          # 华中科技大学（166个电池）
├── MIT data/           # MIT（124个电池）
└── TJU data/           # 天津大学（52个电池）
```

## 运行训练

### 方法一：一键运行（推荐）

```bash
cd PINN4SOH

# 运行所有实验
python run_all.py all

# 或只运行特定数据集
python run_all.py xjtu        # XJTU主实验（60组）
python run_all.py hust        # HUST主实验（10组）
python run_all.py mit         # MIT主实验（10组）
python run_all.py tju         # TJU主实验（30组）

# 其他实验
python run_all.py comparison  # MLP/CNN对比
python run_all.py finetune    # 迁移学习
python run_all.py small       # 小样本实验
python run_all.py analyze     # 结果分析
```

### 方法二：直接运行脚本

```bash
cd PINN4SOH

# XJTU 数据集训练
python main_XJTU.py

# HUST 数据集训练
python main_HUST.py

# MIT 数据集训练
python main_MIT.py

# TJU 数据集训练
python main_TJU.py
```

## 训练过程

### 输出示例

```
Epoch [1/200] Train Loss: 0.0234 Valid MSE: 0.0012
Epoch [2/200] Train Loss: 0.0189 Valid MSE: 0.0009
Epoch [3/200] Train Loss: 0.0156 Valid MSE: 0.0007
...
Epoch [45/200] Train Loss: 0.0023 Valid MSE: 0.0001
Early stopping at epoch 45
Best MSE: 0.0001
```

### 训练时间

| 数据集 | 实验组数 | 单组时间 | 总时间 |
|--------|----------|----------|--------|
| XJTU | 60 | ~2 分钟 | ~2 小时 |
| HUST | 10 | ~5 分钟 | ~50 分钟 |
| MIT | 10 | ~3 分钟 | ~30 分钟 |
| TJU | 30 | ~3 分钟 | ~1.5 小时 |

## 查看结果

### 结果目录

```
results/
├── XJTU results/        # XJTU 实验结果
├── HUST results/        # HUST 实验结果
├── MIT results/         # MIT 实验结果
├── TJU results/         # TJU 实验结果
└── smoke_test/          # 快速测试结果
```

### 单次实验结果

```
results/XJTU results/0-0/Experiment1/
├── model.pth            # 最优模型权重
├── true_label.npy       # 真实标签
├── pred_label.npy       # 预测结果
└── training_log.txt     # 训练日志
```

## 常见问题

### Q1: CUDA 内存不足

```python
# 在 main_XJTU.py 中减小 batch_size
args.batch_size = 128  # 默认 256，可改为 128 或 64
```

### Q2: 训练太慢

```python
# 使用 CPU 训练（更慢但不需要 GPU）
args.device = 'cpu'
```

### Q3: 迁移学习脚本无法运行

文件名包含空格，需要重命名：

```bash
cd PINN4SOH
mv "main_adaptation - fine-tuning.py" "main_adaptation_fine_tuning.py"
python main_adaptation_fine_tuning.py
```

## 下一步

阅读 [05-understand-results.md](05-understand-results.md) 了解如何理解和分析实验结果。
