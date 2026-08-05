# 09 - 数据预处理说明

> 本文档描述**本仓库实际使用的**数据预处理链路与结果格式。
> 历史版本曾给出的 `BatteryDataPreprocessor`、`extract_features_from_charge()` 等类与函数**在仓库中并不存在**，已全部删除并替换为真实实现说明。

## 一、整体链路

```
原始数据（各来源下载）
   │
   │  ① 官方预处理代码库
   ▼
Battery-dataset-preprocessing-code-library/
   ├── XJTUBatteryClass.py        # XJTU 数据集
   ├── HUSTBatteryClass.py        # HUST 数据集
   ├── MITBatteryClass.py         # MIT 数据集
   ├── TongJiBatteryClass.py      # TJU 数据集
   ├── RWTHBatteryClass.py        # RWTH（本仓库未使用）
   └── CALCEBatteryClass.py       # CALCE（本仓库未使用）
   │
   │  ② 输出为每只电池一个 CSV
   ▼
PINN4SOH/data/{XJTU,HUST,MIT,TJU} data/*.csv     ← 387 个文件
   │
   │  ③ 训练时由 dataloader 读取
   ▼
PINN4SOH/dataloader/dataloader.py :: DF.read_one_csv / load_one_battery / load_all_battery
   │
   ▼
TensorDataset(x1, x2, y1, y2) → DataLoader
```

**本仓库不重写预处理逻辑**：CSV 由官方预处理库产出后直接入库，
训练侧只做归一化与异常点剔除（见下）。

## 二、CSV 文件格式

每只电池一个 CSV，**最后一列是容量（转为 SOH 后作为标签）**，其余列为从充电曲线提取的特征。

| 位置 | 内容 |
|------|------|
| 第 1 … 16 列 | 统计特征（电压段 8 个 + 电流段 8 个，见 `02-feature-extraction.md`） |
| 倒数第 2 列 | `cycle index`（周期序号，读取时插入） |
| 最后一列 | `capacity`（容量，用于计算 SOH） |

## 三、训练侧的预处理步骤

实现位置：`dataloader/dataloader.py::DF.read_one_csv()`

### 1. 读入并统一为 float64

```python
df = pd.read_csv(file_name)
df = df.astype({col: 'float64' for col in df.select_dtypes(include=['int64']).columns})
```

> **为什么需要这一步**：pandas ≥ 2.0 不允许把 float64 赋给 int64 列。
> 归一化会把所有列变成 float，而原始 CSV 中存在 int64 列（如 HUST 的 `CC charge time`），
> 不统一 dtype 会在归一化时抛 `TypeError`。这是本仓库为环境迁移所做的适配之一。

### 2. 插入周期序号

```python
df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0], dtype=np.float64))
```

在**容量列之前**插入，因此特征列变为 16 + 1 = **17 列**。

### 3. 3σ 异常点剔除

```python
df = self.delete_3_sigma(df)
```

实现要点：先把 ±inf 替换为 NaN 并丢弃，再对**每一列**独立计算
`|x − mean| > 3σ` 的下标，取并集后整行删除。因此任一列出现离群值都会导致该周期被剔除。

### 4. 计算 SOH 并归一化

```python
df['capacity'] = df['capacity'] / nominal_capacity      # → SOH
f_df = df.iloc[:, :-1]                                  # 所有特征列（含 cycle index）
if normalization_method == 'min-max':
    f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
elif normalization_method == 'z-score':
    f_df = (f_df - f_df.mean()) / f_df.std()
df.iloc[:, :-1] = f_df
```

⚠️ **两点容易被写错的事实**：

1. **min-max 的映射区间是 [−1, 1]，不是 [0, 1]** —— 代码中先线性映射到 [0,1] 再乘 2 减 1。
2. **归一化只作用于特征列**，`capacity`（→ SOH 标签）只做 `除以额定容量` 的换算，不再标准化。
   因此 SOH 标签保持在物理量纲（约 0.7 ~ 1.0）。

### 5. 构造相邻周期样本对

```python
x = df.iloc[:, :-1].values      # 17 列特征
y = df.iloc[:, -1].values       # SOH
x1, x2 = x[:-1], x[1:]
y1, y2 = y[:-1], y[1:]
return (x1, y1), (x2, y2)
```

同一只电池内的**相邻周期**构成一对，这是 `loss3`（方向一致性约束）能够工作的前提
—— 只有成对的相邻周期才能比较"预测的变化方向"与"真实的变化方向"。

## 四、四个数据集的读取入口

| 类（`dataloader/dataloader.py`） | 目录 | 说明 |
|----------------------------------|------|------|
| `XJTUdata` | `data/XJTU data` | `read_one_batch(batch)` 按充放电协议（2C/3C/R2.5/R3/RW/satellite）分批 |
| `HUSTdata` | `data/HUST data` | 目录内全部电池 |
| `MITdata` | `data/MIT data` | `read_one_batch(batch)` 按 3 个测试日期分批 |
| `TJUdata` | `data/TJU data` | `read_one_batch(batch)` 按 3 个 Dataset 分批 |

所有类共用基类 `DF`，因此 `read_one_csv` / `delete_3_sigma` / `load_one_battery` 行为一致。
`read_all()` 返回的字典键因数据集而异（`train_2` / `valid_2` / `test_3` 等），
各 `main_*.py` 负责取出正确的键。

## 五、训练/测试划分

划分**不在预处理阶段完成**，而由各 `main_*.py` 在加载文件列表时决定：

| 数据集 | legacy 规则（默认） | 说明 |
|--------|-------------------|------|
| XJTU | 文件名含 `'4'` 或 `'8'` → 测试 | 子串匹配，实测 3C 批会多纳入 1 只电池 |
| HUST | 硬编码 20 个 `test_id` 字符串 | 新增电池需手工维护 |
| MIT | `id % 5 == 0` | 按电池编号取模 |
| TJU | 枚举下标个位为 (5,9)/(4,8)/(5,9) | 依赖 `os.listdir` 顺序 |

统一规则见 `utils/split.py::uniform_split`，通过 `--split_strategy uniform` 启用。
两种划分的对照可用 `python check_split.py` 复现。

⚠️ 切换划分等价于**更换实验条件**，新结果不可与既有基线直接比较。

## 六、复现预处理结果的方法

```bash
# 1) 从各数据集来源下载原始数据（见 PINN4SOH/data/README.md）
# 2) 用官方预处理库处理 —— 具体用法见该库自身的 README
cd Battery-dataset-preprocessing-code-library
# 3) 处理后的 CSV 放入 PINN4SOH/data/<数据集> data/
# 4) 校验文件数与格式
cd PINN4SOH
python check_consistency.py        # 会核对四个数据集的 CSV 数量
```

## 七、注意事项

1. **不要手工修改 `data/` 下的 CSV**：任何预处理改动都应回到官方预处理库完成，
   以保证与论文口径一致。
2. **不要遍历读取全部 CSV**：387 个文件，按需指定路径即可。
3. **`delete_3_sigma` 会删行**：同一电池在不同数据集中的有效周期数可能不同，
   比较指标时注意样本量。
4. **归一化区间是 [−1, 1]**：阅读代码或复现时不要按 [0, 1] 理解。
