# 03 - PINN 模型架构

> 本文档描述**代码中的真实实现**。类名与论文符号的对应关系：
> **`Solution_u` = 解网络（输出 SOH 的 u）**，**`dynamical_F` = 动力学网络（拟合退化速率 F）**。
> 代码里没有名为 G 的网络 —— 论文正文中的 G 与代码的 `dynamical_F` 是同一个对象。

## 整体结构

```
                    ┌──────────────────────────────────────────┐
                    │                PINN 模型                  │
                    │                                          │
   t (周期)  ──────→│  ┌────────────────────┐                  │
   x (16维特征) ──→│  │ 解网络 Solution_u   │──→ u (预测SOH)   │
                    │  └────────────────────┘        │          │
                    │           │                    │          │
                    │           │ 自动微分            │          │
                    │           ▼                    │          │
                    │       u_t = ∂u/∂t              │          │
                    │       u_x = ∂u/∂x              │          │
                    │           │                    │          │
                    │           ▼                    │          │
                    │  ┌────────────────────┐        │          │
                    │  │ 动力学网络          │←───────┘          │
                    │  │ dynamical_F        │                   │
                    │  └────────────────────┘                   │
                    │           │                               │
                    │           ▼                               │
                    │       F (退化速率)                         │
                    │           │                               │
                    │           ▼                               │
                    │    f = u_t − F = 0   ← PDE 残差            │
                    └──────────────────────────────────────────┘
```

## 两个子网络

### 解网络：`Solution_u`

**作用**：建立 (周期, 特征) 到 SOH 的映射

```
输入: x (16维特征) 与 t (1维周期) 拼接 = 17维
  │
  ▼
encoder : MLP(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)
          即 17 → 60 → 60 → 32，隐藏层后接 Sin 激活，中间层带 Dropout(0.2)
  │
  ▼
predictor: Predictor(input_dim=32)
           = Dropout(0.2) → Linear(32→32) → Sin → Linear(32→1)
  │
  ▼
输出: u (SOH 预测值)
```

- 输出端**没有额外的 Sigmoid**：网络直接回归 SOH 数值，并不把输出强制约束在 (0, 1)
  （标签本身已归一化在约 0.7~1.0 区间）
- 激活函数是 `Sin`（`torch.sin`），不是 ReLU / Tanh
- ⚠️ 本文档旧版本写的「全连接层 64 → 64 → 32」「17 → 64 → 64 → 32 → 1」与代码不符，
  正确结构为 **17 → 60 → 60 → 32 →(predictor)→ 1**

### 动力学网络：`dynamical_F`

**作用**：建模 SOH 的变化率

$$\frac{\partial u}{\partial t} = F(t, x, u, u_t, u_x)$$

```
输入: concat(x[16], t[1], u[1], u_x[16], u_t[1]) = 35维
  │
  ▼
MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60)
    即 35 → 60 → 60 → 1
  │
  ▼
输出: F (SOH 变化率)
```

- 输入包含梯度信息 (u_t, u_x)，这是 PINN 的核心特征
- ⚠️ 旧版本写的「35 → 64 → 64 → 32 → 1」与代码不符，正确为 **35 → 60 → 60 → 1**

## PDE 残差的实现

### 残差定义

$$f = \frac{\partial u}{\partial t} - F(t, x, u, u_t, u_x) \;\longrightarrow\; 0$$

> 旧版本文档写作 $H = \partial F/\partial t - G = 0$，把 u 与 F 写反了，已更正。

### 代码实现（`Model/Model.py` → `PINN.forward()`）

```python
x = xt[:, 0:-1]                                    # 前 16 列 = 统计特征
t = xt[:, -1:]                                     # 最后 1 列 = 周期序号
u = self.solution_u(torch.cat((x, t), dim=1))      # 解网络前向

u_t = grad(u.sum(), t, create_graph=True, allow_unused=True)[0]   # ∂u/∂t
u_x = grad(u.sum(), x, create_graph=True, allow_unused=True)[0]   # ∂u/∂x

F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))         # 动力学网络
f = u_t - F                                                        # PDE 残差
return u, f
```

注意：`dynamical_F` 的输入用的是完整的 `xt`（17 维），与 `u, u_x, u_t` 拼接后共 **35 维**。

### 为什么只用一阶偏导

| 阶数 | 精度 | 计算复杂度 | 本文选择 |
|------|------|-----------|---------|
| 一阶 | 够用 | 低 | **选用** |
| 二阶 | 更高 | 高 | 未采用 |
| 更高阶 | 边际提升 | 很高 | 不必要 |

> 论文实验表明一阶偏导已足以刻画退化动力学。

## 模型的两种训练模式

### 全量训练 —— `PINN.Train()`

```
Solution_u  : 可训练 ✓
dynamical_F : 可训练 ✓
优化器      : optimizer1(Solution_u) + optimizer2(dynamical_F)
```

### 微调 / 迁移 —— `AdaPINN.Adaptation()`

```
Solution_u  : 可训练 ✓
dynamical_F : 冻结   ✗   （requires_grad = False）
优化器      : ada_optimizer(Solution_u)
```

代码位置：`PINN4SOH/main_finetune.py`

```python
for param in self.dynamical_F.parameters():   # freeze the dynamical_F
    param.requires_grad = False
```

物理意义：

- `dynamical_F` 学到的是**通用退化动力学**，与具体电池/工况关系较弱 → 冻结复用
- `Solution_u` 学的是**特定场景下的特征 → SOH 映射** → 需在新数据上适配

## 前向传播完整流程

```
输入: xt (17 列 = 16 特征 + 1 周期序号)
  │
  ├──→ Solution_u(x, t) = u                      # 预测 SOH
  ├──→ ∂u/∂t = u_t                               # 自动微分
  ├──→ ∂u/∂x = u_x                               # 自动微分
  └──→ dynamical_F(x, t, u, u_x, u_t) = F        # 退化速率
  │
  ▼
输出: (u, f = u_t − F)
```

训练时 `u` 供数据损失、`f` 供 PDE 损失；推理时只取 `u`。
