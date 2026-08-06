# 03 - 代码走读：从论文公式到 Python 代码

> 逐行解读核心代码，建立论文公式与代码的对应关系

## 项目文件结构

```
PINN4SOH/
├── Model/
│   └── Model.py              # PINN 模型定义（核心）
├── dataloader/
│   └── dataloader.py          # 数据加载与特征提取
├── main_XJTU.py               # XJTU 数据集训练脚本
├── main_HUST.py               # HUST 数据集训练脚本
├── main_MIT.py                # MIT 数据集训练脚本
├── main_TJU.py                # TJU 数据集训练脚本
├── main_comparision.py        # MLP/CNN 对比实验
├── run_all.py                 # 一键运行脚本
├── utils/
│   └── util.py                # 工具函数
├── data/                      # 预处理后的数据集
└── results/                   # 实验结果
```

## 论文公式 ↔ 代码对应表

| 论文公式 | 代码位置 | 说明 |
|----------|----------|------|
| u = F(t, x) | `Model.py` Solution_u | 解网络前向传播 |
| g = G(t, x, u, u_t, u_x) | `Model.py` dynamical_F | 动力学网络前向传播 |
| H = ∂u/∂t - g = 0 | `Model.py` forward() | PDE 残差计算 |
| L_data = MSE(û, y) | `Model.py` loss1 | 数据损失 |
| L_PDE = MSE(H, 0) | `Model.py` loss2 | PDE 损失 |
| L_mono = ReLU((û₂-û₁)(y₁-y₂)) | `Model.py` loss3 | 单调性损失 |
| x = [f₁, f₂, ..., f₁₆, t] | `dataloader.py` load_one_battery | 特征向量构建 |

## 核心代码解读

### 1. 模型定义（Model.py）

#### Solution_u - 解网络

```python
class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        # 编码器：17维 → 32维
        self.encoder = MLP(input_dim=17, output_dim=32, 
                          layers_num=3, hidden_dim=60, droupout=0.2)
        # 预测器：32维 → 1维
        self.predictor = Predictor(input_dim=32)
    
    def forward(self, xt):
        # xt: [batch_size, 17] (16特征 + 周期索引t)
        x = self.encoder(xt)      # 编码
        x = self.predictor(x)     # 预测
        return x                  # 输出: SOH预测值
```

**对应论文**：u = F(t, x)

#### MLP - 通用多层感知机

```python
class MLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, 
                 layers_num=4, hidden_dim=50, droupout=0.2):
        super(MLP, self).__init__()
        # 构建网络层
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < layers_num - 1:
                layers.append(Sin())  # 正弦激活函数
                layers.append(nn.Dropout(droupout))
        
        self.mlp = nn.Sequential(*layers)
        # Xavier 正态初始化
        self.apply(self._init_weights)
    
    def forward(self, x):
        return self.mlp(x)
```

#### PINN - 完整模型

```python
class PINN(nn.Module):
    def __init__(self, args):
        super(PINN, self).__init__()
        # 解网络
        self.solution_u = Solution_u()
        # 动力学网络
        self.dynamical_F = MLP(input_dim=35, output_dim=1,
                              layers_num=args.F_layers_num,
                              hidden_dim=args.F_hidden_dim, droupout=0.2)
    
    def forward(self, xt):
        xt.requires_grad = True  # 开启自动微分
        
        x = xt[:, 0:-1]   # 17维特征
        t = xt[:, -1:]     # 周期索引
        
        # 解网络前向传播
        u = self.solution_u(torch.cat((x, t), dim=1))
        
        # 自动微分：计算梯度
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True)[0]  # ∂u/∂t
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True)[0]  # ∂u/∂x
        
        # 动力学网络前向传播
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        
        # PDE 残差
        f = u_t - F  # H = ∂u/∂t - G(...)
        
        return u, f
```

**关键点**：
- `xt.requires_grad = True`：开启自动微分
- `grad(u.sum(), t)`：计算 u 对 t 的偏导数
- `f = u_t - F`：PDE 残差，目标是让 f 趋近于 0

#### 损失函数

```python
def train_one_epoch(self, epoch, dataloader):
    for iter, (x1, x2, y1, y2) in enumerate(dataloader):
        # 前向传播
        u1, f1 = self.forward(x1)  # 相邻周期 i
        u2, f2 = self.forward(x2)  # 相邻周期 i+1
        
        # L_data: 数据损失
        loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
        
        # L_PDE: PDE约束损失
        f_target = torch.zeros_like(f1)
        loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)
        
        # L_mono: 单调性损失
        loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()
        
        # 总损失
        loss = loss1 + alpha * loss2 + beta * loss3
        
        # 反向传播
        loss.backward()
        optimizer1.step()
        optimizer2.step()
```

### 2. 数据加载（dataloader.py）

#### 特征提取流程

```python
def read_one_csv(self, file_name, nominal_capacity=None):
    # 1. 读取 CSV
    df = pd.read_csv(file_name)
    
    # 2. 添加周期索引
    df.insert(df.shape[1]-1, 'cycle index', np.arange(df.shape[0]))
    
    # 3. 3-Sigma 异常值剔除
    df = self.delete_3_sigma(df)
    
    # 4. 容量归一化 → SOH
    df['capacity'] = df['capacity'] / nominal_capacity
    
    return df
```

#### 构建相邻周期对

```python
def load_one_battery(self, path, nominal_capacity=None):
    df = self.read_one_csv(path, nominal_capacity)
    
    x = df.iloc[:, :-1].values   # 17维特征（含cycle index）
    y = df.iloc[:, -1].values    # SOH值
    
    # 构建相邻周期对
    x1, x2 = x[:-1], x[1:]     # 相邻周期特征
    y1, y2 = y[:-1], y[1:]     # 相邻周期SOH
    
    return (x1, y1), (x2, y2)
```

### 3. 训练脚本（main_XJTU.py）

```python
def main():
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    
    for i in range(6):          # 6 个 batch
        batch = batchs[i]
        for e in range(10):     # 每个 batch 重复 10 次
            # 加载数据
            dataloader = load_data(args)
            
            # 创建模型
            pinn = PINN(args)
            
            # 训练
            pinn.Train(trainloader=..., validloader=..., testloader=...)
```

## 代码运行流程

```
1. 数据加载
   CSV文件 → read_one_csv() → 异常值剔除 → SOH计算
   ↓
2. 特征构建
   统计特征提取 → 归一化 → 相邻周期对
   ↓
3. 模型初始化
   Solution_u + dynamical_F
   ↓
4. 训练循环
   for epoch in range(200):
       for batch in dataloader:
           前向传播 → 计算损失 → 反向传播 → 更新参数
   ↓
5. 早停检查
   验证集MSE不再改善 → 保存最优模型
   ↓
6. 结果保存
   model.pth + true_label.npy + pred_label.npy
```

## 下一步

阅读 [04-run-experiment.md](04-run-experiment.md) 动手跑通第一个训练实验。
