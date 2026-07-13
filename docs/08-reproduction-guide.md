# 08 - 论文复现路线图

## 当前状态评估

### 已完成的部分

| 模块 | 状态 | 完成度 |
|------|------|--------|
| 特征提取 | ✅ 已实现 | 90% |
| PINN模型架构 | ✅ 已实现 | 95% |
| 损失函数 | ✅ 已修复 | 90% |
| 训练流程 | ✅ 已实现 | 85% |
| 合成数据演示 | ✅ 可运行 | 100% |

### 缺失的关键部分

| 模块 | 状态 | 优先级 |
|------|------|--------|
| 真实数据处理 | ❌ 缺失 | **P0** |
| 数据集下载与预处理 | ❌ 缺失 | **P0** |
| 多数据集实验 | ❌ 缺失 | **P1** |
| 迁移学习实验 | ❌ 缺失 | **P1** |
| 小样本实验 | ❌ 缺失 | **P2** |
| 结果可视化与对比 | ❌ 缺失 | **P2** |

---

## 复现路线图

### 阶段一：获取真实数据（1-2天）

#### 1.1 下载公开数据集

论文使用了4个数据集，其中3个是公开的：

| 数据集 | 下载地址 | 格式 |
|--------|---------|------|
| MIT | https://data.matr.io/1/ | CSV |
| HUST | https://github.com/wang-fujin/PINN4SOH | MAT |
| TJU | https://github.com/wang-fujin/PINN4SOH | MAT |
| XJTU | https://doi.org/10.5281/zenodo.10963339 | MAT |

#### 1.2 数据预处理

创建 `data_preprocessing.py`，实现：

```python
# 1. 读取原始数据
# 2. 提取充电曲线数据
# 3. 提取16维特征
# 4. 计算SOH（当前容量/初始容量）
# 5. 保存为统一格式
```

#### 1.3 数据存储结构

```
data/
├── MIT/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的特征和标签
│       ├── features.npy  # (n_samples, 16)
│       ├── soh.npy       # (n_samples,)
│       ├── cycles.npy    # (n_samples,)
│       └── battery_ids.npy  # (n_samples,)
├── HUST/
│   └── ...
├── TJU/
│   └── ...
└── XJTU/
    └── ...
```

---

### 阶段二：实现完整数据管道（2-3天）

#### 2.1 创建数据加载模块

创建 `src/real_data_utils.py`：

```python
class RealBatteryDataset(Dataset):
    """真实电池数据集"""
    
    def __init__(self, data_dir, battery_ids=None):
        """
        参数:
            data_dir: 数据目录
            battery_ids: 指定加载哪些电池的数据
        """
        self.features = np.load(f"{data_dir}/features.npy")
        self.soh = np.load(f"{data_dir}/soh.npy")
        self.cycles = np.load(f"{data_dir}/cycles.npy")
        self.battery_ids = np.load(f"{data_dir}/battery_ids.npy")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.cycles[idx], dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.soh[idx], dtype=torch.float32),
            torch.tensor(self.battery_ids[idx], dtype=torch.long)
        )
```

#### 2.2 实现数据划分策略

论文使用**按电池级别划分**：

```python
def split_by_battery(data_dir, train_ratio=0.6, val_ratio=0.2):
    """按电池级别划分数据集"""
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")
    unique_batteries = np.unique(battery_ids)
    
    np.random.shuffle(unique_batteries)
    n_train = int(len(unique_batteries) * train_ratio)
    n_val = int(len(unique_batteries) * val_ratio)
    
    train_bats = unique_batteries[:n_train]
    val_bats = unique_batteries[n_train:n_train+n_val]
    test_bats = unique_batteries[n_train+n_val:]
    
    return train_bats, val_bats, test_bats
```

---

### 阶段三：实现完整训练流程（2-3天）

#### 3.1 创建训练脚本

创建 `train_real_data.py`：

```python
def train_on_real_data(dataset_name, config):
    """在真实数据上训练"""
    
    # 1. 加载数据
    data_dir = f"data/{dataset_name}/processed"
    train_bats, val_bats, test_bats = split_by_battery(data_dir)
    
    # 2. 创建数据集
    train_dataset = RealBatteryDataset(data_dir, train_bats)
    val_dataset = RealBatteryDataset(data_dir, val_bats)
    test_dataset = RealBatteryDataset(data_dir, test_bats)
    
    # 3. 归一化
    # 使用训练集计算归一化参数
    min_vals, max_vals = compute_normalization_params(train_dataset.features)
    
    # 4. 创建模型
    model = PINN(feature_dim=16)
    
    # 5. 训练
    trainer = PINNTrainer(model)
    history = trainer.train(
        train_loader, val_loader,
        epochs=config['epochs'],
        alpha=config['alpha'],
        beta=config['beta']
    )
    
    # 6. 测试
    metrics, preds, truths = trainer.test(test_loader)
    
    return metrics, history
```

#### 3.2 配置管理系统

创建 `configs/` 目录：

```
configs/
├── default.yaml       # 默认配置
├── xjtu.yaml          # XJTU数据集配置
├── tju.yaml           # TJU数据集配置
├── hust.yaml          # HUST数据集配置
└── mit.yaml           # MIT数据集配置
```

每个配置文件包含：

```yaml
# configs/xjtu.yaml
dataset: XJTU
feature_dim: 16
f_hidden_dims: [64, 64, 32]
g_hidden_dims: [64, 64, 32]
learning_rate: 0.001
weight_decay: 1e-5
alpha: 0.1
beta: 0.1
epochs: 200
batch_size: 64
early_stopping_patience: 20
```

---

### 阶段四：实现对比实验（3-4天）

#### 4.1 实现MLP和CNN基线

创建 `src/baselines.py`：

```python
class MLP(nn.Module):
    """多层感知机基线"""
    def __init__(self, input_dim=17, hidden_dims=[64, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, t, x):
        return self.network(torch.cat([t, x], dim=1))

class CNN(nn.Module):
    """CNN基线"""
    # 实现1D CNN...
```

#### 4.2 实现多次重复实验

```python
def run_experiments(dataset_name, n_runs=10):
    """运行10次重复实验"""
    results = []
    
    for run in range(n_runs):
        # 设置随机种子
        set_seed(run)
        
        # 训练并测试
        metrics = train_and_test(dataset_name)
        results.append(metrics)
    
    # 计算均值和标准差
    mean_metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std_metrics = {k: np.std([r[k] for r in results]) for k in results[0]}
    
    return mean_metrics, std_metrics
```

---

### 阶段五：实现迁移学习（2-3天）

#### 5.1 预训练阶段

```python
def pretrain(source_dataset):
    """在源域数据集上预训练"""
    model = PINN(feature_dim=16)
    trainer = PINNTrainer(model)
    
    # 全量训练
    trainer.train(train_loader, val_loader, epochs=200)
    
    # 保存预训练模型
    torch.save(model.state_dict(), f"pretrained/{source_dataset}.pth")
    
    return model
```

#### 5.2 微调阶段

```python
def finetune(source_model, target_dataset, n_target_batteries=1):
    """在目标域数据集上微调"""
    # 加载预训练模型
    model = PINN(feature_dim=16)
    model.load_state_dict(source_model)
    
    # 冻结动力学网络
    model.freeze_dynamics()
    
    # 只用1个目标域电池微调
    target_bats = get_battery_ids(target_dataset)[:n_target_batteries]
    target_dataset = RealBatteryDataset(target_dataset, target_bats)
    
    trainer = PINNTrainer(model)
    trainer.fine_tune(target_loader, epochs=200)
    
    return model
```

---

### 阶段六：结果可视化与论文对标（2-3天）

#### 6.1 生成论文中的图表

```python
def plot_fig4(results):
    """生成Fig.4: SOH估计结果"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, dataset in enumerate(['XJTU', 'TJU', 'HUST', 'MIT']):
        axes[i].scatter(results[dataset]['true'], 
                       results[dataset]['pred'], alpha=0.5)
        axes[i].plot([0.7, 1.0], [0.7, 1.0], 'r--')
        axes[i].set_title(dataset)
    
    plt.savefig('fig4_soh_estimation.png')

def plot_fig5(small_sample_results):
    """生成Fig.5: 小样本实验"""
    # ...

def plot_fig6(transfer_results):
    """生成Fig.6: 迁移实验"""
    # ...
```

#### 6.2 生成对比表格

```python
def generate_table2(all_results):
    """生成Table 2: 四个数据集的结果"""
    print("| Dataset | Batch | Ours MAPE | MLP MAPE | CNN MAPE |")
    print("|---------|-------|-----------|----------|----------|")
    for dataset in ['XJTU', 'TJU', 'HUST', 'MIT']:
        for batch in all_results[dataset].keys():
            print(f"| {dataset} | {batch} | ... | ... | ... |")
```

---

## 推荐的复现顺序

### 第一周：数据准备

```
Day 1-2: 下载并理解数据格式
Day 3-4: 实现数据预处理
Day 5:   验证特征提取正确性
Day 6-7: 创建数据加载管道
```

### 第二周：基础实验

```
Day 1-2: 在XJTU数据集上训练
Day 3-4: 在TJU数据集上训练
Day 5:   在HUST数据集上训练
Day 6:   在MIT数据集上训练
Day 7:   对比结果，与论文Table 2对比
```

### 第三周：高级实验

```
Day 1-2: 实现MLP和CNN基线
Day 3-4: 多次重复实验（10次）
Day 5-6: 小样本实验
Day 7:   生成Fig.4和Fig.5
```

### 第四周：迁移学习

```
Day 1-2: 实现预训练流程
Day 3-4: 实现微调流程
Day 5-6: 运行跨数据集迁移实验
Day 7:   生成Fig.6
```

---

## 关键注意事项

### 1. 数据划分

**必须按电池级别划分**，不能随机划分样本：

```python
# 错误做法
train_idx = np.random.choice(n_samples, int(0.6*n_samples))

# 正确做法
train_bats = np.random.choice(unique_batteries, n_train, replace=False)
train_mask = np.isin(battery_ids, train_bats)
```

### 2. 归一化参数

**只用训练集计算归一化参数**：

```python
# 错误做法
min_vals = np.min(all_features)
max_vals = np.max(all_features)

# 正确做法
min_vals = np.min(train_features)
max_vals = np.max(train_features)
# 用训练集的min/max归一化所有数据
```

### 3. 单调性损失

**必须按电池分组计算**：

```python
# 错误做法
sorted_idx = torch.argsort(cycle_indices)
mono_loss = ReLU(sorted_u[1:] - sorted_u[:-1])

# 正确做法
for bat in unique_batteries:
    bat_mask = (battery_ids == bat)
    bat_u = predicted_u[bat_mask]
    bat_cycles = cycle_indices[bat_mask]
    sorted_idx = torch.argsort(bat_cycles)
    mono_loss += ReLU(bat_u[sorted_idx][1:] - bat_u[sorted_idx][:-1])
```

### 4. 评估指标

使用MAE、MAPE、RMSE三个指标：

```python
def calculate_metrics(preds, truths):
    mae = np.mean(np.abs(preds - truths))
    mape = np.mean(np.abs((preds - truths) / (truths + 1e-8))) * 100
    rmse = np.sqrt(np.mean((preds - truths) ** 2))
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
```

---

## 预期结果对标

### Table 2 对标

| 数据集 | 论文MAPE | 目标MAPE | 差距容忍 |
|--------|---------|---------|---------|
| XJTU | 0.85% | < 1.5% | 可接受 |
| TJU | 1.21% | < 2.0% | 可接受 |
| HUST | 0.65% | < 1.0% | 可接受 |
| MIT | 0.78% | < 1.2% | 可接受 |

**注意**: 由于随机种子和实现细节差异，结果可能有10-20%的偏差，这是正常的。

### Fig.4 对标

预测散点图应该：
- 点沿对角线分布
- 没有明显的系统偏差
- 在SOH=0.7~1.0范围内均匀分布

### Fig.5 对标

小样本实验应该：
- PINN在1个电池时优势最明显
- 随着训练电池数增加，差距缩小
- PINN始终优于MLP和CNN
