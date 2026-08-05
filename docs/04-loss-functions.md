# 04 - 三重损失函数

> 本文档以 `Model/Model.py::PINN.train_one_epoch()` 的实际实现为准。
> 代码里的三个损失分别是 `loss1` / `loss2` / `loss3`。

## 总览

$$L = L_{data} + \alpha \cdot L_{PDE} + \beta \cdot L_{phys}$$

```
损失函数
├── loss1 (L_data):  数据拟合 ────→ 预测值接近真实值
├── loss2 (L_PDE):   物理约束 ────→ 满足退化动力学方程  u_t − F → 0
└── loss3 (L_phys):  方向一致性 ──→ 预测的升降方向与真实一致（允许容量再生）
```

⚠️ 旧版本把第三项称为「单调性损失 L_mono、SOH 随周期单调递减」，**与代码不符**。
实际 `loss3` 惩罚的是"预测变化方向与真实变化方向相反"，当真实出现容量再生
（SOH 回升）时它同样要求预测回升。详见第 3 节。

## 0. 样本对的组织方式

三个损失都在**成对样本**上计算。配对方由 `dataloader/dataloader.py::DF.load_one_battery()` 给出：

```python
x1 = x[:-1]      # 第 1 … T-1 个周期
x2 = x[1:]       # 第 2 … T   个周期
y1 = y[:-1]
y2 = y[1:]
```

即 **(x1, y1) 与 (x2, y2) 是同一只电池内相邻周期的样本对**。
训练时每个 batch 取出这两组，分别前向传播得到 `(u1, f1)` 与 `(u2, f2)`。

## 1. 数据损失 loss1

### 公式

$$L_{data} = \frac{1}{2}\,\mathrm{MSE}(u_1, y_1) + \frac{1}{2}\,\mathrm{MSE}(u_2, y_2)$$

代码：

```python
loss1 = 0.5*self.loss_func(u1, y1) + 0.5*self.loss_func(u2, y2)
```

- `self.loss_func = nn.MSELoss()`（**均值**，不是求和）
- 两个分支各占 1/2 权重
- ⚠️ 旧版本写作 $L_{data} = \sum |u_i - \hat u_i|^2$，既漏掉了成对结构与 0.5 系数，
  也把 MSE 误写成求和，已更正

### 作用

最基础的监督信号，让网络学会从特征预测 SOH。

## 2. PDE 残差损失 loss2

### 公式

$$L_{PDE} = \frac{1}{2}\,\mathrm{MSE}(f_1, \mathbf{0}) + \frac{1}{2}\,\mathrm{MSE}(f_2, \mathbf{0}),
\qquad f = \frac{\partial u}{\partial t} - F(t,x,u,u_t,u_x)$$

代码：

```python
f_target = torch.zeros_like(f1)
loss2 = 0.5*self.loss_func(f1, f_target) + 0.5*self.loss_func(f2, f_target)
```

其中 `f1` / `f2` 由 `PINN.forward()` 返回（见文档 03）。
这里 `F` 即 `dynamical_F` —— **旧版本写作 $H = \partial F/\partial t - G$ 是符号错位**。

### 作用

**这是 PINN 的核心**：让网络预测满足退化动力学方程，而不仅是拟合数据。

| 没有 PDE 约束 | 有 PDE 约束 |
|--------------|------------|
| 纯数据拟合 | 物理 + 数据联合驱动 |
| 需要大量数据 | 小样本也能学好 |
| 可能过拟合 | 泛化能力更强 |
| 预测可能违反物理规律 | 预测遵循退化规律 |

**物理意义**：模型学到的不只是"特征 → SOH"的映射，而是退化动力学本身。

## 3. 方向一致性损失 loss3

### 真实公式

$$L_{phys} = \sum \mathrm{ReLU}\big((u_2 - u_1)\cdot(y_1 - y_2)\big)$$

代码：

```python
# physics loss  u2-u1<0, considering capacity regeneration effect
loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()
```

### 它到底约束什么

| 真实变化 | `y1 − y2` | 要求预测 | 若预测相反 |
|---------|-----------|---------|-----------|
| SOH 下降（放电老化） | > 0 | u₂ < u₁（也下降） | 乘积 > 0 → 被惩罚 |
| SOH 上升（容量再生） | < 0 | u₂ > u₁（也上升） | 乘积 > 0 → 被惩罚 |
| 方向一致 | — | — | 乘积 ≤ 0 → ReLU 后为 0 |

因此第三项是**允许容量再生**的方向一致性软约束，而不是"单调递减约束"。
代码注释 `considering capacity regeneration effect` 也印证了这一点。

### 关于 `.sum()`（已知问题）

```python
loss3 = self.relu(...).sum()      # 注意是求和，不是求均值
```

由于用求和，`loss3` 的量级随 batch size 线性变化：同样的数据分布，
batch size 翻倍会使 `loss3` 约翻倍，等效于把 `β` 放大。因此：

- 更换 batch size 后，**`β` 必须重新标定**，不能直接沿用旧值
- 若改为 `.mean()`，则所有已发布基线的 `β` 都需重跑重标，属单独的排期工作
  （详见 `docs/CHANGELOG.md` 的「本次未改动」小节）

## 4. 总损失

```python
loss = loss1 + self.alpha*loss2 + self.beta*loss3
```

## 5. 超参数的实际取值

各数据集的默认值（取自对应 `main_*.py` 的 `get_args()`）：

| 数据集 | α (`--alpha`) | β (`--beta`) |
|--------|--------------|-------------|
| XJTU | 0.7 | 0.2 |
| HUST | 0.5 | 0.2 |
| MIT | 1.0 | 0.02 |
| TJU | 1.0 | 0.05 |

> ⚠️ 旧版本给的「α 典型值 0.1~1.0、β 典型值 0.0~1.0」是泛泛区间，
> 与仓库实际取值无关，已替换为真实默认值。
> 另：`main_XJTU.py::small_sample()` 内部会把 α 覆盖为 0.5、β 覆盖为 10（沿用原作者设定）。

调参影响（定性）：

- α 过大 → 物理约束过强，难以拟合数据；α 过小 → 退化为纯数据驱动
- β 过大 → 影响拟合精度；β 过小 → 方向约束形同虚设
- **α/β 目前均为手工设定**，论文未给出选取依据 —— 这是本项目规划中的创新切入点之一
  （自动超参优化，见文档 06 与 `nc-pinn/docs/beginner-guide/06-innovation-paths.md`）

## 6. 完整实现

代码位置：`Model/Model.py::PINN.train_one_epoch()`

```python
def train_one_epoch(self, epoch, dataloader):
    self.train()
    for iter, (x1, x2, y1, y2) in enumerate(dataloader):
        x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
        u1, f1 = self.forward(x1)
        u2, f2 = self.forward(x2)

        loss1 = 0.5*self.loss_func(u1, y1) + 0.5*self.loss_func(u2, y2)      # 数据
        f_target = torch.zeros_like(f1)
        loss2 = 0.5*self.loss_func(f1, f_target) + 0.5*self.loss_func(f2, f_target)  # PDE
        loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()                 # 方向一致性

        loss = loss1 + self.alpha*loss2 + self.beta*loss3

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer1.step()      # 更新 Solution_u
        self.optimizer2.step()      # 更新 dynamical_F
    return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg
```

> 旧版本文档在此处给出的是 `def total_loss(predicted_u, true_u, H, cycle_indices, battery_ids, ...)`
> 这样一个**仓库中并不存在**的函数，已替换为真实实现。
