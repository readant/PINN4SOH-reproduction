# 07 - 术语表：关键概念速查

> 随时查阅论文和代码中的关键术语

## 电池相关术语

| 术语 | 英文 | 定义 |
|------|------|------|
| **SOH** | State of Health | 电池健康状态 = 当前容量 / 初始容量 × 100% |
| **RUL** | Remaining Useful Life | 剩余使用寿命 |
| **SOC** | State of Charge | 荷电状态（当前电量百分比） |
| **NCM** | Nickel Cobalt Manganese | 三元锂电池（镍钴锰） |
| **NCA** | Nickel Cobalt Aluminum | 三元锂电池（镍钴铝） |
| **LFP** | Lithium Iron Phosphate | 磷酸铁锂电池 |
| **SEI** | Solid Electrolyte Interphase | 固体电解质界面膜 |
| **EIS** | Electrochemical Impedance Spectroscopy | 电化学阻抗谱 |
| **IC/DV** | Incremental Capacity / Differential Voltage | 增量容量/微分电压曲线 |

## 深度学习术语

| 术语 | 英文 | 定义 |
|------|------|------|
| **PINN** | Physics-Informed Neural Network | 物理信息神经网络 |
| **PDE** | Partial Differential Equation | 偏微分方程 |
| **MLP** | Multi-Layer Perceptron | 多层感知机 |
| **CNN** | Convolutional Neural Network | 卷积神经网络 |
| **MAPE** | Mean Absolute Percentage Error | 平均绝对百分比误差 |
| **MAE** | Mean Absolute Error | 平均绝对误差 |
| **MSE** | Mean Squared Error | 均方误差 |
| **RMSE** | Root Mean Squared Error | 均方根误差 |
| **R²** | R-squared | 决定系数 |

## PINN4SOH 专用术语

| 术语 | 定义 | 代码位置 |
|------|------|----------|
| **Solution_u** | 解网络 F(Φ)：将特征映射到 SOH | `Model.py` |
| **dynamical_F** | 动力学网络 G(Θ)：建模退化动力学 | `Model.py` |
| **L_data** | 数据损失：预测值 vs 真实值 | `Model.py` loss1 |
| **L_PDE** | PDE 约束损失：强制遵守物理定律 | `Model.py` loss2 |
| **L_mono** | 单调性损失：SOH 应随周期递减 | `Model.py` loss3 |
| **α (alpha)** | PDE 损失权重（默认 0.7） | `main_*.py` |
| **β (beta)** | 单调性损失权重（默认 0.2） | `main_*.py` |
| **B2 架构** | 混合方法：物理方程与神经网络融合 | 论文 §1 |

## 代码中的关键变量

| 变量 | 类型 | 含义 |
|------|------|------|
| `x` | Tensor [batch, 16] | 16 个统计特征 |
| `t` | Tensor [batch, 1] | 周期索引 |
| `xt` | Tensor [batch, 17] | 完整输入 = [x, t] |
| `u` | Tensor [batch, 1] | 预测 SOH 值 |
| `u_t` | Tensor [batch, 1] | ∂u/∂t（SOH 对时间的导数） |
| `u_x` | Tensor [batch, 16] | ∂u/∂x（SOH 对特征的导数） |
| `f` | Tensor [batch, 1] | PDE 残差 = u_t - g |
| `y` | Tensor [batch, 1] | 真实 SOH 值 |

## 16 个统计特征

| 序号 | 特征名 | 含义 |
|------|--------|------|
| 1 | mean | 均值 |
| 2 | std | 标准差 |
| 3 | max | 最大值 |
| 4 | min | 最小值 |
| 5 | skewness | 偏度 |
| 6 | kurtosis | 峰度 |
| 7 | RMS | 均方根 |
| 8 | peak-to-peak | 峰峰值 |
| 9 | entropy | 信息熵 |
| 10 | energy | 能量 |
| 11 | crest factor | 波峰因子 |
| 12 | shape factor | 形状因子 |
| 13 | slope | 斜率 |
| 14 | intercept | 截距 |
| 15 | R² | 拟合优度 |
| 16 | area | 面积 |

## 论文中的关键公式

### SOH 定义
```
SOH = Q_current / Q_nominal × 100%
```

### 解网络
```
u_i = f(t_i, x_i)
```
其中 t 是周期索引，x 是 16 维特征向量。

### 动力学网络
```
g_i = G(t_i, x_i, u_i, ∂u/∂t_i, ∂u/∂x_i)
```

### PDE 约束
```
H = ∂u/∂t - G(...) = 0
```

### 总损失
```
L = L_data + α × L_PDE + β × L_mono
```

其中：
- L_data = MSE(û, y)
- L_PDE = MSE(H, 0)
- L_mono = ReLU((û₂-û₁)(y₁-y₂))

## 常见缩写

| 缩写 | 全称 |
|------|------|
| SOH | State of Health |
| RUL | Remaining Useful Life |
| SOC | State of Charge |
| PINN | Physics-Informed Neural Network |
| PDE | Partial Differential Equation |
| MLP | Multi-Layer Perceptron |
| CNN | Convolutional Neural Network |
| RNN | Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | R-squared |
| XJTU | Xi'an Jiaotong University |
| HUST | Huazhong University of Science and Technology |
| MIT | Massachusetts Institute of Technology |
| TJU | Tianjin University |
