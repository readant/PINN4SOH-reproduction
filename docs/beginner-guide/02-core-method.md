# 02 - 核心方法：PINN 是怎么工作的

> 深入理解 PINN 的架构、损失函数和训练策略

## 整体架构

PINN4SOH 由两个子网络组成：

```
                    ┌─────────────────┐
  17维特征 ───────→ │  Solution_u     │ ──────→ u (预测SOH)
  (16特征 + t)      │  解网络 F(Φ)    │
                    └────────┬────────┘
                             │
                        自动微分
                             ↓
                    u_t = ∂u/∂t, u_x = ∂u/∂x
                             │
                    ┌────────┴────────┐
  [xt, u, u_x, u_t]→│  dynamical_F   │ ──────→ g (退化速率)
                    │  动力学网络 G(Θ) │
                    └────────┬────────┘
                             │
                    H = u_t - g = 0  ← PDE 约束
```

## 两个子网络

### 1. Solution_u（解网络 F(Φ)）

**作用**：将特征映射到 SOH

```python
class Solution_u(nn.Module):
    def __init__(self):
        # 编码器：17维 → 32维
        self.encoder = MLP(input_dim=17, output_dim=32, 
                          layers_num=3, hidden_dim=60, droupout=0.2)
        # 预测器：32维 → 1维
        self.predictor = Predictor(input_dim=32)
```

**输入**：17 维向量 = 16 个统计特征 + 1 个周期索引 t

**输出**：1 个标量 = 预测的 SOH 值 u

**激活函数**：使用正弦函数 `Sin`（而非传统的 ReLU），有助于捕捉周期性退化模式

### 2. dynamical_F（动力学网络 G(Θ)）

**作用**：建模电池退化动力学

```python
# 输入：35维 = 17维原始输入 + 1维u + 16维u_x + 1维u_t
self.dynamical_F = MLP(input_dim=35, output_dim=1,
                       layers_num=3, hidden_dim=60, droupout=0.2)
```

**核心思想**：通过自动微分获取梯度，让网络学习退化速率

## 三重损失函数

### L_data：数据损失

```python
loss1 = 0.5 * MSE(u1, y1) + 0.5 * MSE(u2, y2)
```

- u1, u2：相邻周期的预测 SOH
- y1, y2：相邻周期的真实 SOH
- 作用：确保预测值接近真实值

### L_PDE：物理约束损失

```python
f_target = torch.zeros_like(f1)  # 目标：H = 0
loss2 = 0.5 * MSE(f1, f_target) + 0.5 * MSE(f2, f_target)
```

- f = u_t - g：PDE 残差
- 目标：让残差趋近于 0
- 作用：强制模型遵守物理定律

### L_mono：单调性损失

```python
loss3 = ReLU((u2 - u1) * (y1 - y2)).sum()
```

**物理含义**：SOH 应随周期单调递减

```
如果 y2 < y1（真实SOH递减），但 u2 > u1（预测SOH递增）
→ (u2-u1) > 0, (y1-y2) > 0
→ 乘积 > 0 → ReLU 产生惩罚
```

### 总损失

```python
loss = L_data + α * L_PDE + β * L_mono
```

| 超参数 | 默认值 | 作用 |
|--------|--------|------|
| α | 0.7 | PDE 损失权重 |
| β | 0.2 | 单调性损失权重 |

## 特征提取方法

### 为什么选择充电数据？

```
放电数据的问题：
├── 用户放电策略不同（有人用到 20%，有人用到 10%）
├── 很少完全放电
└── 数据不完整、不可比

充电数据的优势：
├── 一旦开始充电，大概率充满
├── 充电过程相对固定和规律
└── 充满电前的数据在大多数数据集中都存在
```

### 16 个统计特征

从充满电前的一小段电压/电流/温度数据中提取：

| 序号 | 特征 | 含义 |
|------|------|------|
| 1-4 | mean, std, max, min | 基础统计量 |
| 5-8 | skewness, kurtosis, RMS, peak-to-peak | 分布特征 |
| 9-12 | entropy, energy, crest factor, shape factor | 信号特征 |
| 13-16 | slope, intercept, R², area | 趋势特征 |

## 训练策略

### 双优化器

```python
# 解网络：使用学习率调度器（warmup + 余弦退火）
optimizer1 = Adam(solution_u.parameters(), lr=base_lr)

# 动力学网络：固定学习率
optimizer2 = Adam(dynamical_F.parameters(), lr=lr_F)
```

### 学习率调度

```
阶段1（Warmup）：0.002 → 0.01（线性增长）
阶段2（Cosine）：0.01 → 0.0002（余弦退火）
```

### 早停机制

```python
if valid_mse < best_valid_mse:
    best_valid_mse = valid_mse
    save_model()
    cnt = 0
else:
    cnt += 1
    if cnt >= early_stop:  # 默认 20 轮
        break
```

## 下一步

阅读 [03-code-walkthrough.md](03-code-walkthrough.md) 看看这些方法是如何用代码实现的。
