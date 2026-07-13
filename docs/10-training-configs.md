# 10 - 训练配置与实验设置

## 配置文件设计

### 默认配置 (configs/default.yaml)

```yaml
# 模型配置
model:
  feature_dim: 16
  f_hidden_dims: [64, 64, 32]
  g_hidden_dims: [64, 64, 32]

# 训练配置
training:
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 1e-5
  batch_size: 64
  epochs: 200
  early_stopping_patience: 20
  alpha: 0.1  # PDE损失权重
  beta: 0.1   # 单调性损失权重

# 数据配置
data:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  V_end: 4.2  # 充电截止电压

# 实验配置
experiment:
  n_runs: 10  # 重复实验次数
  save_dir: results/
```

### 数据集专用配置

#### XJTU配置 (configs/xjtu.yaml)

```yaml
# XJTU数据集特点:
# - 55个NCM电池
# - 6种充放电协议
# - 样本量相对较小

model:
  feature_dim: 16
  f_hidden_dims: [64, 64, 32]
  g_hidden_dims: [64, 64, 32]

training:
  learning_rate: 0.001
  alpha: 0.1
  beta: 0.1
  epochs: 200
  batch_size: 32  # 较小的batch size

data:
  V_end: 4.2
```

#### TJU配置 (configs/tju.yaml)

```yaml
# TJU数据集特点:
# - 52个电池 (NCA/NCM/NCM+NCA)
# - 多种化学体系
# - 快充策略

model:
  feature_dim: 16
  f_hidden_dims: [64, 64, 32]
  g_hidden_dims: [64, 64, 32]

training:
  learning_rate: 0.001
  alpha: 0.1
  beta: 0.1
  epochs: 200
  batch_size: 64

data:
  V_end: 4.2
```

#### HUST配置 (configs/hust.yaml)

```yaml
# HUST数据集特点:
# - 166个LFP电池
# - 样本量大
# - 快充实验

model:
  feature_dim: 16
  f_hidden_dims: [64, 64, 32]
  g_hidden_dims: [64, 64, 32]

training:
  learning_rate: 0.001
  alpha: 0.1
  beta: 0.05  # LFP电池单调性更强
  epochs: 200
  batch_size: 128  # 较大的batch size

data:
  V_end: 4.2
```

#### MIT配置 (configs/mit.yaml)

```yaml
# MIT数据集特点:
# - 124个LFP电池
# - 数据质量高
# - 标准充放电

model:
  feature_dim: 16
  f_hidden_dims: [64, 64, 32]
  g_hidden_dims: [64, 64, 32]

training:
  learning_rate: 0.001
  alpha: 0.1
  beta: 0.05
  epochs: 200
  batch_size: 128

data:
  V_end: 4.2
```

---

## 训练脚本

### train.py

```python
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path

from src.model import PINN
from src.trainer import PINNTrainer
from src.data_utils import RealBatteryDataset, create_dataloader

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(config_path, dataset_name, run_id=0):
    """训练模型"""
    # 加载配置
    config = load_config(config_path)

    # 设置随机种子
    set_seed(run_id)

    # 数据路径
    data_dir = f"data/{dataset_name}/processed"

    # 加载数据
    features = np.load(f"{data_dir}/features.npy")
    soh = np.load(f"{data_dir}/soh.npy")
    cycles = np.load(f"{data_dir}/cycles.npy")
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")

    # 数据划分
    unique_bats = np.unique(battery_ids)
    np.random.shuffle(unique_bats)

    n_train = int(len(unique_bats) * config['data']['train_ratio'])
    n_val = int(len(unique_bats) * config['data']['val_ratio'])

    train_bats = set(unique_bats[:n_train])
    val_bats = set(unique_bats[n_train:n_train+n_val])
    test_bats = set(unique_bats[n_train+n_val:])

    # 创建掩码
    train_mask = np.isin(battery_ids, list(train_bats))
    val_mask = np.isin(battery_ids, list(val_bats))
    test_mask = np.isin(battery_ids, list(test_bats))

    # 归一化 (只用训练集)
    min_vals = features[train_mask].min(axis=0)
    max_vals = features[train_mask].max(axis=0)

    # 归一化特征
    features_norm = 2 * (features - min_vals) / (max_vals - min_vals + 1e-8) - 1

    # 归一化周期
    max_cycle = cycles[train_mask].max()
    cycles_norm = cycles / max_cycle

    # 创建数据集
    train_dataset = RealBatteryDataset(
        cycles_norm[train_mask],
        features_norm[train_mask],
        soh[train_mask],
        battery_ids[train_mask]
    )
    val_dataset = RealBatteryDataset(
        cycles_norm[val_mask],
        features_norm[val_mask],
        soh[val_mask],
        battery_ids[val_mask]
    )
    test_dataset = RealBatteryDataset(
        cycles_norm[test_mask],
        features_norm[test_mask],
        soh[test_mask],
        battery_ids[test_mask]
    )

    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # 创建模型
    model = PINN(
        feature_dim=config['model']['feature_dim'],
        f_hidden_dims=config['model']['f_hidden_dims'],
        g_hidden_dims=config['model']['g_hidden_dims']
    )

    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PINNTrainer(model, device=device)
    trainer.set_optimizer(
        optimizer=config['training']['optimizer'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 训练
    save_path = f"models/{dataset_name}_run{run_id}.pth"
    os.makedirs("models", exist_ok=True)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        alpha=config['training']['alpha'],
        beta=config['training']['beta'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        save_path=save_path
    )

    # 测试
    metrics, predictions, ground_truth = trainer.test(test_loader)

    print(f"\n{dataset_name} Run {run_id}:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    return metrics, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--run_id', type=int, default=0)
    args = parser.parse_args()

    metrics, history = train(args.config, args.dataset, args.run_id)
```

---

## 实验运行脚本

### run_experiments.sh

```bash
#!/bin/bash

# 数据集列表
DATASETS=("XJTU" "TJU" "HUST" "MIT")
CONFIGS=("configs/xjtu.yaml" "configs/tju.yaml" "configs/hust.yaml" "configs/mit.yaml")
N_RUNS=10

# 创建结果目录
mkdir -p results
mkdir -p models

# 运行实验
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    CONFIG=${CONFIGS[$i]}

    echo "=========================================="
    echo "Running experiments on $DATASET"
    echo "=========================================="

    for run in $(seq 1 $N_RUNS); do
        echo "Run $run/$N_RUNS"
        python train.py --config $CONFIG --dataset $DATASET --run_id $run
    done

    echo ""
done

echo "All experiments completed!"
```

### run_transfer_experiments.sh

```bash
#!/bin/bash

# 迁移实验配置
SOURCE_DATASETS=("XJTU" "HUST")
TARGET_DATASETS=("TJU" "MIT")
N_TARGET_BATTERIES=(1 2 3 4 5)

for SOURCE in "${SOURCE_DATASETS[@]}"; do
    for TARGET in "${TARGET_DATASETS[@]}"; do
        echo "Transfer: $SOURCE -> $TARGET"

        for N_BATS in "${N_TARGET_BATTERIES[@]}"; do
            echo "  Using $N_BATS target batteries"
            python finetune.py \
                --source $SOURCE \
                --target $TARGET \
                --n_target_batteries $N_BATS
        done
    done
done
```

---

## 结果分析脚本

### analyze_results.py

```python
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_experiment_results(results_dir):
    """分析实验结果"""
    results = []

    for result_file in Path(results_dir).glob("*.npz"):
        data = np.load(result_file)
        results.append({
            'dataset': data['dataset'],
            'run_id': data['run_id'],
            'MAE': data['MAE'],
            'MAPE': data['MAPE'],
            'RMSE': data['RMSE']
        })

    df = pd.DataFrame(results)

    # 计算统计量
    summary = df.groupby('dataset').agg({
        'MAE': ['mean', 'std'],
        'MAPE': ['mean', 'std'],
        'RMSE': ['mean', 'std']
    })

    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    print(summary.to_string())

    return summary

def compare_with_paper(results, paper_results):
    """与论文结果对比"""
    print("\n" + "="*60)
    print("与论文结果对比")
    print("="*60)

    for dataset in results.index:
        our_mape = results.loc[dataset, ('MAPE', 'mean')]
        paper_mape = paper_results[dataset]

        diff = abs(our_mape - paper_mape) / paper_mape * 100

        print(f"{dataset}:")
        print(f"  Our MAPE: {our_mape:.2f}%")
        print(f"  Paper MAPE: {paper_mape:.2f}%")
        print(f"  Difference: {diff:.1f}%")
        print()

# 论文报告的结果
PAPER_RESULTS = {
    'XJTU': 0.85,
    'TJU': 1.21,
    'HUST': 0.65,
    'MIT': 0.78
}

if __name__ == "__main__":
    results = analyze_experiment_results("results/")
    compare_with_paper(results, PAPER_RESULTS)
```

---

## 调参建议

### 超参数搜索空间

| 参数 | 搜索范围 | 推荐值 |
|------|---------|--------|
| learning_rate | [1e-4, 1e-2] | 1e-3 |
| alpha | [0.01, 1.0] | 0.1 |
| beta | [0.0, 0.5] | 0.1 |
| batch_size | [32, 128] | 64 |
| f_hidden_dims | [[32,32], [64,64,32], [128,64,32]] | [64,64,32] |

### 调参策略

1. **先固定其他参数，调learning_rate**
   - 从1e-3开始
   - 如果loss震荡，减小lr
   - 如果loss下降太慢，增大lr

2. **再调alpha (PDE权重)**
   - 从0.1开始
   - 如果PDE loss太大，增大alpha
   - 如果数据拟合差，减小alpha

3. **最后调beta (单调性权重)**
   - 从0.0开始
   - 如果预测不单调，增大beta
   - 如果影响拟合精度，减小beta

### 常见问题诊断

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss不下降 | lr太大或太小 | 调整learning_rate |
| Val loss上升 | 过拟合 | 增加正则化、减小模型 |
| 预测全是常数 | alpha太大 | 减小alpha |
| 预测不单调 | beta太小 | 增大beta |
| 训练太慢 | batch size太小 | 增大batch_size |
