# 07 - 论文方法与代码实现对应关系

## 文件结构

```
nc-PINN/
├── src/
│   ├── feature_extraction.py  ← 特征提取模块
│   ├── model.py               ← PINN模型架构
│   ├── loss.py                ← 三重损失函数
│   ├── data_utils.py          ← 数据处理
│   └── trainer.py             ← 训练流程
├── demo.py                    ← 完整训练演示
└── docs/                      ← 本文档
```

## 模块对应关系

### 1. 特征提取

| 论文描述 | 代码文件 | 关键函数 |
|---------|---------|---------|
| 截取电压区间 [V_end-0.2, V_end] | `feature_extraction.py` | `extract_features()` |
| 截取电流区间 [0.1A, 0.5A] | `feature_extraction.py` | `extract_features()` |
| 16个统计特征 | `feature_extraction.py` | `extract_features()` |
| 特征归一化 | `feature_extraction.py` | `min_max_normalize()` |

```python
# 论文: 从充电曲线提取16个统计特征
# 代码: feature_extraction.py

def extract_features(voltage, current, time, V_end=4.2):
    # 截取电压区间
    V_start = V_end - 0.2
    mask_voltage = (voltage >= V_start) & (voltage <= V_end)
    
    # 截取电流区间
    cv_mask = (current >= 0.1) & (current <= 0.5)
    
    # 提取8个电压特征 + 8个电流特征 = 16维
    features = []
    features.append(np.mean(v_selected))      # 1. mean
    features.append(np.std(v_selected))       # 2. std
    features.append(kurtosis(v_selected))     # 3. kurtosis
    features.append(skew(v_selected))         # 4. skewness
    features.append(charge_time)              # 5. charging time
    features.append(accumulated_charge)       # 6. accumulated charge
    features.append(slope_v)                  # 7. curve slope
    features.append(entropy_v)                # 8. curve entropy
    # ... 电流特征类似
```

### 2. PINN模型

| 论文描述 | 代码文件 | 关键类/函数 |
|---------|---------|------------|
| 解网络 F(Φ) | `model.py` | `SolutionNetwork` |
| 动力学网络 G(Θ) | `model.py` | `DynamicsNetwork` |
| PDE约束 H = ∂F/∂t - G | `model.py` | `PINN.forward()` |
| 自动微分计算梯度 | `model.py` | `torch.autograd.grad()` |
| 微调模式 | `model.py` | `fine_tune_mode()` |

```python
# 论文: PINN由解网络和动力学网络组成
# 代码: model.py

class SolutionNetwork(nn.Module):
    """解网络 F(Φ): 映射特征到SOH"""
    def forward(self, t, x):
        input_data = torch.cat([t, x], dim=1)  # t(1维) + x(16维)
        u = self.network(input_data)            # 输出SOH
        return u

class DynamicsNetwork(nn.Module):
    """动力学网络 G(Θ): 建模退化动力学"""
    def forward(self, t, x, u, u_t, u_x):
        input_data = torch.cat([t, x, u, u_t, u_x], dim=1)
        g = self.network(input_data)            # 输出退化速率
        return g

class PINN(nn.Module):
    def forward(self, t, x):
        # 解网络前向传播
        u = self.solution_net(t, x)
        
        # 自动微分计算梯度
        u_t = torch.autograd.grad(u, t, ...)[0]  # ∂u/∂t
        u_x = torch.autograd.grad(u, x, ...)[0]  # ∂u/∂x
        
        # 动力学网络
        g = self.dynamics_net(t, x, u, u_t, u_x)
        
        # PDE约束
        H = u_t - g  # 应该趋近于0
        
        return u, H, u_t, u_x
```

### 3. 损失函数

| 论文公式 | 代码文件 | 关键函数 |
|---------|---------|---------|
| $L_{data} = \sum|u_i - \hat{u}_i|^2$ | `loss.py` | `data_loss()` |
| $L_{PDE} = \sum|H_i|^2$ | `loss.py` | `pde_loss()` |
| $L_{mono} = \sum_j\sum_k \text{ReLU}(\hat{u}_{k+1}-\hat{u}_k)$ | `loss.py` | `monotonicity_loss()` |
| $L = L_{data} + \alpha L_{PDE} + \beta L_{mono}$ | `loss.py` | `total_loss()` |

```python
# 论文: 三重损失函数
# 代码: loss.py

def data_loss(predicted_u, true_u):
    """L_data = Σ|u - ũ|²"""
    return torch.mean((predicted_u - true_u) ** 2)

def pde_loss(H):
    """L_PDE = Σ|H|²"""
    return torch.mean(H ** 2)

def monotonicity_loss(predicted_u, cycle_indices, battery_ids):
    """L_mono = Σ_j Σ_k ReLU(ũ_{k+1} - ũ_k)"""
    # 按电池分组计算
    for bat in torch.unique(battery_ids):
        mask = (battery_ids == bat)
        bat_u = predicted_u[mask]
        # 同一电池内按周期排序后计算单调性
        sorted_u = bat_u[torch.argsort(cycle_indices[mask])]
        mono_loss += torch.mean(F.relu(sorted_u[1:] - sorted_u[:-1]))
    return mono_loss / n_batteries

def total_loss(predicted_u, true_u, H, cycle_indices, battery_ids, alpha, beta):
    """总损失"""
    L_data = data_loss(predicted_u, true_u)
    L_pde = pde_loss(H)
    L_mono = monotonicity_loss(predicted_u, cycle_indices, battery_ids)
    return L_data + alpha * L_pde + beta * L_mono
```

### 4. 数据处理

| 论文描述 | 代码文件 | 关键函数 |
|---------|---------|---------|
| 合成数据生成 | `data_utils.py` | `generate_synthetic_data()` |
| 按电池划分数据集 | `data_utils.py` | `split_data()` |
| Dataset封装 | `data_utils.py` | `BatteryDataset` |
| DataLoader创建 | `data_utils.py` | `create_dataloader()` |

```python
# 论文: 训练集/验证集/测试集按6:2:2划分
# 代码: data_utils.py

def split_data(cycles, features, soh, battery_ids, ...):
    """按电池级别划分，确保同一电池不跨集合"""
    unique_batteries = np.unique(battery_ids)
    # 按电池比例划分
    train_bats = unique_batteries[:n_train]
    val_bats = unique_batteries[n_train:n_train+n_val]
    test_bats = unique_batteries[n_train+n_val:]
    ...
```

### 5. 训练流程

| 论文描述 | 代码文件 | 关键方法 |
|---------|---------|---------|
| 训练循环 | `trainer.py` | `PINNTrainer.train()` |
| 验证评估 | `trainer.py` | `PINNTrainer.validate()` |
| 测试评估 | `trainer.py` | `PINNTrainer.test()` |
| 微调训练 | `trainer.py` | `PINNTrainer.fine_tune()` |
| 早停机制 | `trainer.py` | `train()` 内部实现 |
| 模型保存 | `trainer.py` | `torch.save()` |

```python
# 论文: 训练过程包含早停和模型保存
# 代码: trainer.py

class PINNTrainer:
    def train(self, train_loader, val_loader, epochs, alpha, beta, 
              early_stopping_patience, save_path):
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, _ = self.train_epoch(train_loader, alpha, beta)
            
            # 验证
            if (epoch + 1) % 10 == 0:
                val_metrics = self.validate(val_loader)
                
                # 早停检查
                if val_metrics['RMSE'] < best_val_rmse:
                    best_val_rmse = val_metrics['RMSE']
                    patience_counter = 0
                    torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
```

## 代码运行流程

```
demo.py
  │
  ├── 1. generate_synthetic_data()
  │       └── 生成合成电池退化数据
  │
  ├── 2. split_data()
  │       └── 按电池级别划分数据集
  │
  ├── 3. compute_normalization_params() + min_max_normalize()
  │       └── 特征归一化
  │
  ├── 4. BatteryDataset() + create_dataloader()
  │       └── 构建数据加载器
  │
  ├── 5. PINN()
  │       └── 创建PINN模型
  │
  ├── 6. PINNTrainer() + train()
  │       └── 训练模型（含早停）
  │
  ├── 7. test()
  │       └── 测试评估
  │
  └── 8. plot_results()
          └── 可视化结果
```

## 已修复的问题

| 问题 | 修复内容 |
|------|---------|
| loss.py导入顺序 | `import numpy` 移到文件开头 |
| 单调性损失跨电池 | 改为按battery_ids分组计算 |
| 特征未归一化 | 添加特征和周期归一化 |
| 数据泄露 | split_data改为按电池级别划分 |
| 合成数据特征弱 | 增强特征-SOH耦合强度 |
| PDE权重过大 | α从1.0降为0.1 |
