# 11 - 迁移学习实现指南

## 迁移学习原理

### 核心假设

```
G(Θ) - 动力学网络:
├── 学习的是退化动力学 du/dt = g(...)
├── 这是电池的固有物理规律
├── 与充放电协议无关
├── 与数据集无关
└── 可以跨场景迁移

F(Φ) - 解网络:
├── 学习的是特征→SOH的映射
├── 与具体场景相关
├── 不同化学体系需要不同的F
└── 需要在目标域微调
```

### 迁移策略

```
阶段1: 预训练
┌─────────────────────────────────────┐
│  源域大数据集                        │
│  ├── 全量训练PINN                    │
│  ├── F(Φ)和G(Θ)都训练              │
│  └── 保存预训练模型                  │
└─────────────────────────────────────┘
              │
              ▼
阶段2: 微调
┌─────────────────────────────────────┐
│  目标域小数据集                      │
│  ├── 加载预训练模型                  │
│  ├── 冻结G(Θ)                       │
│  ├── 只微调F(Φ)                     │
│  └── 用少量目标域数据训练            │
└─────────────────────────────────────┘
```

---

## 迁移学习实现

### finetune.py

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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def pretrain(source_dataset, config):
    """阶段1: 在源域预训练"""
    print(f"\n{'='*60}")
    print(f"Pre-training on {source_dataset}")
    print(f"{'='*60}")

    # 加载数据
    data_dir = f"data/{source_dataset}/processed"
    features = np.load(f"{data_dir}/features.npy")
    soh = np.load(f"{data_dir}/soh.npy")
    cycles = np.load(f"{data_dir}/cycles.npy")
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")

    # 数据划分
    unique_bats = np.unique(battery_ids)
    np.random.shuffle(unique_bats)

    n_train = int(len(unique_bats) * 0.6)
    n_val = int(len(unique_bats) * 0.2)

    train_bats = set(unique_bats[:n_train])
    val_bats = set(unique_bats[n_train:n_train+n_val])

    train_mask = np.isin(battery_ids, list(train_bats))
    val_mask = np.isin(battery_ids, list(val_bats))

    # 归一化
    min_vals = features[train_mask].min(axis=0)
    max_vals = features[train_mask].max(axis=0)
    features_norm = 2 * (features - min_vals) / (max_vals - min_vals + 1e-8) - 1

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

    train_loader = create_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=64, shuffle=False)

    # 创建模型
    model = PINN(feature_dim=16)

    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PINNTrainer(model, device=device)
    trainer.set_optimizer(lr=0.001, weight_decay=1e-5)

    save_path = f"models/pretrained_{source_dataset}.pth"
    trainer.train(
        train_loader, val_loader,
        epochs=200,
        alpha=config['training']['alpha'],
        beta=config['training']['beta'],
        save_path=save_path
    )

    # 保存归一化参数
    np.save(f"models/norm_{source_dataset}.npz", {
        'min_vals': min_vals,
        'max_vals': max_vals,
        'max_cycle': max_cycle
    })

    print(f"Pre-trained model saved to {save_path}")
    return model, min_vals, max_vals, max_cycle

def finetune(source_model_path, target_dataset, n_target_batteries, config):
    """阶段2: 在目标域微调"""
    print(f"\n{'='*60}")
    print(f"Fine-tuning on {target_dataset}")
    print(f"Using {n_target_batteries} target batteries")
    print(f"{'='*60}")

    # 加载目标域数据
    data_dir = f"data/{target_dataset}/processed"
    features = np.load(f"{data_dir}/features.npy")
    soh = np.load(f"{data_dir}/soh.npy")
    cycles = np.load(f"{data_dir}/cycles.npy")
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")

    # 加载归一化参数 (使用源域的)
    source_dataset = Path(source_model_path).stem.replace("pretrained_", "")
    norm_params = np.load(f"models/norm_{source_dataset}.npz")
    min_vals = norm_params['min_vals']
    max_vals = norm_params['max_vals']
    max_cycle = norm_params['max_cycle']

    # 归一化
    features_norm = 2 * (features - min_vals) / (max_vals - min_vals + 1e-8) - 1
    cycles_norm = cycles / max_cycle

    # 选择目标域电池
    unique_bats = np.unique(battery_ids)
    np.random.shuffle(unique_bats)
    target_bats = unique_bats[:n_target_batteries]
    other_bats = unique_bats[n_target_batteries:]

    # 用部分目标域数据微调，其余测试
    target_mask = np.isin(battery_ids, target_bats)
    test_mask = np.isin(battery_ids, other_bats)

    # 创建数据集
    finetune_dataset = RealBatteryDataset(
        cycles_norm[target_mask],
        features_norm[target_mask],
        soh[target_mask],
        battery_ids[target_mask]
    )
    test_dataset = RealBatteryDataset(
        cycles_norm[test_mask],
        features_norm[test_mask],
        soh[test_mask],
        battery_ids[test_mask]
    )

    finetune_loader = create_dataloader(finetune_dataset, batch_size=32, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

    # 加载预训练模型
    model = PINN(feature_dim=16)
    model.load_state_dict(torch.load(source_model_path))

    # 冻结动力学网络
    model.freeze_dynamics()
    print("Dynamics network G(Θ) frozen")

    # 微调
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PINNTrainer(model, device=device)
    trainer.set_optimizer(lr=1e-4, weight_decay=1e-5)  # 较小的学习率

    save_path = f"models/finetuned_{source_dataset}_to_{target_dataset}_n{n_target_batteries}.pth"
    trainer.fine_tune(
        finetune_loader, finetune_loader,  # 用微调数据做验证
        epochs=200,
        alpha=0.1,
        beta=0.1,
        save_path=save_path
    )

    # 测试
    metrics, predictions, ground_truth = trainer.test(test_loader)

    print(f"\nResults:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    # 保存结果
    np.savez(f"results/transfer_{source_dataset}_to_{target_dataset}_n{n_target_batteries}.npz",
             dataset=target_dataset,
             n_target_batteries=n_target_batteries,
             MAE=metrics['MAE'],
             MAPE=metrics['MAPE'],
             RMSE=metrics['RMSE'],
             predictions=predictions,
             ground_truth=ground_truth)

    return metrics

def direct_training(target_dataset, n_target_batteries, config):
    """直接训练 (对比基线)"""
    print(f"\n{'='*60}")
    print(f"Direct training on {target_dataset}")
    print(f"Using {n_target_batteries} target batteries")
    print(f"{'='*60}")

    # 加载数据
    data_dir = f"data/{target_dataset}/processed"
    features = np.load(f"{data_dir}/features.npy")
    soh = np.load(f"{data_dir}/soh.npy")
    cycles = np.load(f"{data_dir}/cycles.npy")
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")

    # 归一化
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    features_norm = 2 * (features - min_vals) / (max_vals - min_vals + 1e-8) - 1

    max_cycle = cycles.max()
    cycles_norm = cycles / max_cycle

    # 选择电池
    unique_bats = np.unique(battery_ids)
    np.random.shuffle(unique_bats)
    target_bats = unique_bats[:n_target_batteries]
    other_bats = unique_bats[n_target_batteries:]

    target_mask = np.isin(battery_ids, target_bats)
    test_mask = np.isin(battery_ids, other_bats)

    # 创建数据集
    train_dataset = RealBatteryDataset(
        cycles_norm[target_mask],
        features_norm[target_mask],
        soh[target_mask],
        battery_ids[target_mask]
    )
    test_dataset = RealBatteryDataset(
        cycles_norm[test_mask],
        features_norm[test_mask],
        soh[test_mask],
        battery_ids[test_mask]
    )

    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

    # 训练
    model = PINN(feature_dim=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PINNTrainer(model, device=device)
    trainer.set_optimizer(lr=0.001, weight_decay=1e-5)

    trainer.train(
        train_loader, train_loader,
        epochs=200,
        alpha=0.1,
        beta=0.1
    )

    # 测试
    metrics, predictions, ground_truth = trainer.test(test_loader)

    print(f"\nResults:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--n_target_batteries', type=int, default=1)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    # 预训练
    pretrained_model, _, _, _ = pretrain(args.source, config)

    # 微调
    finetune(
        f"models/pretrained_{args.source}.pth",
        args.target,
        args.n_target_batteries,
        config
    )

    # 直接训练对比
    direct_training(args.target, args.n_target_batteries, config)
```

---

## 迁移实验矩阵

### 论文中的迁移组合

| 源域 | 目标域 | 化学体系 | 预期效果 |
|------|--------|---------|---------|
| XJTU | TJU | NCM → NCA/NCM | 好 |
| TJU | XJTU | NCA/NCM → NCM | 好 |
| HUST | MIT | LFP → LFP | 好 |
| MIT | HUST | LFP → LFP | 好 |
| XJTU | HUST | NCM → LFP | 差 |
| HUST | XJTU | LFP → NCM | 差 |
| XJTU | MIT | NCM → LFP | 差 |
| MIT | XJTU | LFP → NCM | 差 |

### 运行所有迁移实验

```bash
#!/bin/bash

# 同族迁移 (预期效果好)
python finetune.py --source XJTU --target TJU --n_target_batteries 1
python finetune.py --source TJU --target XJTU --n_target_batteries 1
python finetune.py --source HUST --target MIT --n_target_batteries 1
python finetune.py --source MIT --target HUST --n_target_batteries 1

# 异族迁移 (预期效果差)
python finetune.py --source XJTU --target HUST --n_target_batteries 1
python finetune.py --source HUST --target XJTU --n_target_batteries 1
```

---

## 结果分析

### 迁移学习结果汇总

```python
def analyze_transfer_results():
    """分析迁移学习结果"""
    results = []

    for result_file in Path("results/").glob("transfer_*.npz"):
        data = np.load(result_file)
        results.append({
            'source': str(result_file).split('_')[1],
            'target': str(result_file).split('_')[3],
            'n_batteries': int(str(result_file).split('_n')[1].replace('.npz', '')),
            'MAPE': data['MAPE']
        })

    df = pd.DataFrame(results)

    # 按源域-目标域分组
    print("\n迁移学习结果:")
    print(df.pivot_table(
        values='MAPE',
        index=['source', 'target'],
        columns='n_batteries',
        aggfunc='mean'
    ))

    return df
```

### 预期结果格式

```
迁移学习结果 (MAPE %):

源域 → 目标域    | 1电池 | 2电池 | 3电池 | 4电池 | 5电池
----------------|-------|-------|-------|-------|-------
XJTU → TJU      | 1.8   | 1.5   | 1.3   | 1.2   | 1.1
TJU → XJTU      | 1.9   | 1.6   | 1.4   | 1.2   | 1.1
HUST → MIT      | 0.9   | 0.8   | 0.7   | 0.7   | 0.6
MIT → HUST      | 0.8   | 0.7   | 0.7   | 0.6   | 0.6
XJTU → HUST     | 3.5   | 2.8   | 2.3   | 2.0   | 1.8
HUST → XJTU     | 3.2   | 2.5   | 2.1   | 1.9   | 1.7
```
