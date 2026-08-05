# 07 - 论文方法与代码实现对应关系

> 本文档对照**本仓库的真实代码**（`PINN4SOH/` 目录）与论文方法，逐项给出实现位置。
> 文档中出现的每一个路径、类名、函数名都可在仓库中直接检索到。

---

## 一、仓库真实结构

```
nc-PINN/
├── README.md
├── requirements.txt
├── LICENSE                          # CC BY 4.0（继承自原始论文）
├── docs/                            # 本套解读文档（01–14）
├── PINN4SOH/                        # 论文官方代码（已迁移到 Python 3.12 + PyTorch 2.x）
│   ├── Model/
│   │   ├── Model.py                 # 核心：Sin / MLP / Predictor / Solution_u / LR_Scheduler / PINN
│   │   └── Compare_Models.py        # 基线：ResBlock / MLP / CNN
│   ├── dataloader/
│   │   └── dataloader.py            # 基类 DF + XJTUdata / HUSTdata / MITdata / TJUdata
│   ├── utils/
│   │   ├── util.py                  # set_seed / get_logger / AverageMeter / eval_metrix / write_to_txt
│   │   └── split.py                 # uniform_split / natural_key / describe_split / check_leakage
│   ├── main_XJTU.py                 # XJTU 入口（6 个充放电协议 × 10 次）
│   ├── main_HUST.py / main_MIT.py / main_TJU.py
│   ├── main_comparision.py          # MLP / CNN 基线对比
│   ├── main_finetune.py             # 迁移学习（原 main_adaptation - fine-tuning.py，已重命名）
│   ├── run_all.py                   # 一键调度入口
│   ├── results_summary.py           # 实验结果汇总（仓库唯一数据源生成器）
│   ├── verify_fixes.py              # 修复回归验证（含端到端训练）
│   ├── check_split.py               # 数据划分规则体检
│   ├── check_consistency.py         # 文档-代码一致性校验
│   ├── validate.py / smoke_test.py  # 快速验证 / 冒烟测试
│   ├── results analysis/            # 原作者的结果分析脚本（处理 results/Ours/ 结果包）
│   ├── plotter/                     # 论文插图：Figure 2 / 4a / 4b / 5 / 6
│   ├── data/                        # 预处理后的四个数据集
│   ├── results/                     # 实验产物（含 SUMMARY.md —— 唯一数据源）
│   └── pretrained model/            # 预训练权重（11 个 .pth + 8 个 mu_var_pi_*.npz）
└── Battery-dataset-preprocessing-code-library/   # 官方数据预处理代码库
```

---

## 二、论文方法与代码对应表

### 1. 特征提取（论文 §Method「通用特征提取」）

| 论文描述 | 代码位置 | 关键实现 |
|---------|---------|---------|
| 从充电末段提取特征 | `dataloader/dataloader.py` → `DF.read_one_csv()` | 截取电压区间、计算统计量 |
| 16 个统计特征（8 电压 + 8 电流） | `DF.load_one_battery()` | 返回特征矩阵 |
| 拼接周期序号构成网络输入 | `DF.load_all_battery()` | 输出 17 列 = 16 特征 + 1 周期序号 |
| 归一化 | `DF.delete_3_sigma()` / `normalization_method` | `min-max` 或 `z-score` |

> **特征维度说明**：网络输入是 **17 维** = 16 个统计特征 + 1 个周期序号（cycle index）。
> 在 `PINN.forward()` 中，前 16 列作为空间变量 `x`，最后一列作为时间变量 `t`。
> 这一点在阅读代码时最容易误判（见 `Model.py` 的切片 `xt[:, 0:-1]` 与 `xt[:, -1:]`）。

### 2. 模型结构（论文 Fig.1c / §Method）

| 论文描述 | 代码位置 | 关键类 / 实现 |
|---------|---------|--------------|
| 解网络（估计 SOH 的网络） | `Model/Model.py` | **`Solution_u`**（内含 `MLP` 编码器 + `Predictor` 头） |
| 动力学网络（拟合退化速率） | `Model/Model.py` | **`dynamical_F`**，类型为 `MLP(input_dim=35, ...)` |
| 前向传播与自动微分 | `Model/Model.py` | `PINN.forward()`：`grad(u.sum(), t)` → `u_t`；`grad(u.sum(), x)` → `u_x` |
| PDE 残差 | `Model/Model.py` | `f = u_t - F`，其中 `F = dynamical_F(cat([xt, u, u_x, u_t]))` |
| 激活函数 | `Model/Model.py` | `Sin` 类（`torch.sin`），置于每个隐藏层后 |

**实际网络尺寸**

```
Solution_u
├── encoder  : MLP(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)
│              → 结构为 17 → 60 → 60 → 32，激活全部为 Sin
└── predictor: Predictor(input_dim=32) → Dropout(0.2) → Linear(32→32) → Sin → Linear(32→1)

dynamical_F  : MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60)
               → 输入为 concat(x[16], t[1], u[1], u_x[16], u_t[1]) = 35 维
```

> **符号提示**：论文正文中 <i>F</i> 指动力学方程、<i>u</i> 指 SOH。
> 代码里 **`dynamical_F` 对应方程 F**，**`Solution_u` 对应解 u** ——
> 不要写成"解网络 F"，历史文档曾有此误。

### 3. 损失函数（论文 §Method「Loss function」）

| 论文描述 | 代码位置 | 实现 |
|---------|---------|------|
| 数据损失 L_data | `Model/Model.py` → `PINN.train_one_epoch()` | `loss1 = 0.5·MSE(u1, y1) + 0.5·MSE(u2, y2)` |
| PDE 残差损失 L_PDE | 同上 | `loss2 = 0.5·MSE(f1, 0) + 0.5·MSE(f2, 0)` |
| 物理（容量再生）约束 | 同上 | `loss3 = relu((u2-u1)·(y1-y2)).sum()` |
| 总损失 | 同上 | `loss = loss1 + alpha·loss2 + beta·loss3` |

**关于 loss3 的准确表述**：它约束的是"预测的变化方向与真实变化方向一致" ——
当真实 SOH 下降（y1 > y2）时预测也必须下降，反之亦然；两者方向相反则被惩罚。
因此它不是"单调递减约束"，而是**允许容量再生**的方向一致性软约束。
另需注意此项用 `.sum()` 而非 `.mean()`，其量级随 batch size 漂移（见 §五）。

### 4. 训练策略（论文 §Method「Training」）

| 论文描述 | 代码位置 | 实现 |
|---------|---------|------|
| 两套优化器 | `Model/Model.py` → `PINN.__init__` | `optimizer1`（解网络）、`optimizer2`（动力学网络） |
| 学习率 warmup + 余弦衰减 | `Model/Model.py` | `LR_Scheduler`（warmup 段线性、衰减段余弦） |
| 早停 | `Model/Model.py` → `PINN.Train()` | 依据**验证集 MSE** 选最优并早停 |
| 最优模型保存 | `PINN.Train()` | `model.pth` + `true_label.npy` / `pred_label.npy` |
| 模型加载 | `PINN.load_model()` | `torch.load(..., weights_only=True)` |

### 5. 迁移学习（论文 Fig.6）

| 论文描述 | 代码位置 | 实现 |
|---------|---------|------|
| 源域预训练 + 目标域微调 | `main_finetune.py` | `class AdaPINN(PINN)` → `Adaptation()` |
| 单次迁移任务 | `main_finetune.py` | `one_adaptation_task(args, source, target, ...)` |
| 四组标准迁移 | `main_finetune.py` | `FineTune_TJU2XJTU / FineTune_XJTU2TJU / FineTune_HUST2MIT / FineTune_MIT2HUST` |
| 跨化学体系 8 组 | `main_finetune.py` | `FineTune()` |

### 6. 基线对比（论文 Table 2）

| 论文描述 | 代码位置 | 实现 |
|---------|---------|------|
| MLP 基线 | `Model/Compare_Models.py` | `class MLP` |
| CNN 基线 | `Model/Compare_Models.py` | `class CNN`（含 `ResBlock`） |
| 对比实验入口 | `main_comparision.py` | 通过 `setattr(args, 'model', 'MLP'/'CNN')` 切换 |

### 7. 结果分析与绘图

| 论文描述 | 代码位置 | 实现 |
|---------|---------|------|
| Table 各数据集指标 | `results analysis/{XJTU,HUST,MIT,TJU} results.py` | `class Results`（输出 `.xlsx`） |
| Table 2 方法对比 | `results analysis/Comparision results.py` | `class Results` |
| Fig.2 / 4a / 4b / 5 / 6 | `plotter/Figure *.py` | 需配合 `plotter/color.py` 配色 |
| **本仓库实验结果汇总** | `results_summary.py` | 生成 `results/SUMMARY.md`（仓库唯一数据源） |

---

## 三、入口脚本与产物对应关系

| 入口 | 产物目录 | 实验规模 |
|------|---------|---------|
| `main_XJTU.py` | `results/XJTU results/{0-0 … 5-5}/Experiment{1..10}` | 60 次 |
| `main_HUST.py` | `results/HUST results/Experiment{1..10}` | 10 次 |
| `main_MIT.py` | `results/MIT results/Experiment{1..10}` | 10 次 |
| `main_TJU.py` | `results/TJU results/{0-0,1-1,2-2}/Experiment{1..10}` | 30 次 |
| `main_comparision.py` | `results/XJTU-MLP results/`、`results/XJTU-CNN results/` | 60 + 60 次 |
| `main_finetune.py` | `results_fine-tuning/<源>-<目标>/` | 标准四组共 110 次 |
| `results_summary.py` | `results/SUMMARY.md`、`results/summary.csv` | 汇总以上全部 |

> **工作目录约束**：所有 `main_*.py` 与汇总脚本都假设 **cwd = `PINN4SOH/`**，
> 内部使用 `data/...`、`results/...` 等相对路径。
> `results analysis/*.py` 是例外，它们假设 **cwd = `results analysis/`**（用 `../results/...`）。
> 从错误目录运行会因路径解析失败而报错。

---

## 四、论文声明与本仓库实测的对照

实测值来自 `results/SUMMARY.md`（由 `results_summary.py` 从原始产物生成）。

| 结论 | 论文 | 本仓库实测 | 判定 |
|------|------|-----------|------|
| XJTU MAPE | 0.78% | 0.87 ± 0.25% | 接近，偏差 +0.09 pp |
| HUST MAPE | 0.65% | 0.80 ± 0.02% | 接近，偏差 +0.15 pp |
| MIT MAPE | 0.78% | 0.73 ± 0.04% | 接近，偏差 −0.05 pp |
| TJU MAPE | 1.21% | 1.13 ± 0.29% | 接近，偏差 −0.08 pp |
| PINN 优于 MLP/CNN | 报告为显著优势 | XJTU 上 0.87% vs 2.23% / 2.53% | 一致 |
| 迁移学习有效 | 微调显著降低跨域误差 | TJU→XJTU：1.03%（源域直测约 11%） | 一致 |

**完整实测表以 `results/SUMMARY.md` 为准**，本文不复制全部数字，避免再次出现多处数值不一致。

---

## 五、阅读代码时的已知陷阱

1. **输入维度**：`xt` 是 17 列（16 特征 + 1 周期序号），不是 16 列。
2. **F/G 命名**：代码里只有 `dynamical_F`，没有名为 G 的网络；论文中的 G 与代码的 F 指同一对象。
3. **`loss3` 用 `.sum()`**：损失量级随 batch size 变化，更换 batch size 后 `beta` 需重新标定。
4. **早停判据**：只依据验证集；`testloader` 仅用于评估，不参与模型选择。
5. **数据划分**：四个数据集各有一套 legacy 规则（默认）；`--split_strategy uniform` 可切换到
   `utils/split.py` 的统一规则，但**切换划分等价于更换实验条件**，不可与已有基线直接对比。
6. **结果目录有两套**：`results/`（本仓库实测）与 `results/Ours/`（原作者提供的结果包，需解压）；
   `results analysis/*.py` 读取的是后者，`results_summary.py` 读取的是前者。**两者不可混用。**
