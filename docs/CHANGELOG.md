# 改动说明 / CHANGELOG

本文件记录 PINN4SOH 代码仓库的实质性改动，便于回溯与复现。
格式：每条记录含日期、改动范围、动机、文件清单、验证方式、未改动项。

---

## 2026-07-27 — 可复现性 + 统一数据划分 + 已知缺陷修复

### 背景
此前项目完成了官方 PINN4SOH 仓库的 Python 3.7→3.12 / PyTorch 1.7→2.x 环境迁移，并跑通约 290 次训练（平均 MAPE 0.88%，论文 0.87%）。但在代码审计中发现若干影响研究结论可靠性的隐患，本次集中修复其中 3 条核心问题（对应审计报告的「地雷③ ④ ⑥ ⑦ ⑧」），其余（文件名空格、loss3 尺度、工作目录假设）留待后续。

### 改动范围

#### 1. 可复现性（审计地雷④）
- **新增** `PINN4SOH/utils/util.py :: set_seed(seed)`：统一固定 `random` / `numpy` / `torch`（CPU + CUDA）种子，并开启 `torch.backends.cudnn.deterministic=True`、`benchmark=False`。
- **修改** `PINN4SOH/Model/Model.py`：`PINN.__init__` 中调用 `set_seed(args.seed)` 完成播种。
- **修改** 四个入口 `main_{XJTU,HUST,MIT,TJU}.py`：新增命令行参数 `--seed`（默认 420）。多实验场景可用 `420 + e`（e 为实验序号）递增，既保证实验间有差异、又保证每一次可复现。

#### 2. 统一数据划分（审计地雷③）
- **新增** `PINN4SOH/utils/split.py`：
  - `uniform_split(file_list, test_ratio=0.2)`：对电池文件按**自然序**排序后，以固定步长在编号范围内**等步长均匀抽取**测试电池（电池级划分、无硬编码、跨数据集一致），消除原 `os.listdir` 顺序依赖与子串误伤。
  - `describe_split(name, train_list, test_list)` / `check_leakage(train_list, test_list)`：生成人类可读的划分摘要与泄漏检查，便于写入日志、事后核对。
- **修改** 四个入口：新增 `--split_strategy {legacy,uniform}`，**默认 `legacy`**，保持已有 290 次实验基线不变；显式传 `uniform` 才启用统一规则。
- **实证**：原 legacy 规则 `'4' in file or '8' in file` 是子串匹配，会误伤 `3C_battery-14.csv`（编号含 4 但非 4 号）、`battery-24`、`battery-40~49` 等；`uniform_split` 已修正为按编号精确定位。

#### 3. 已知缺陷修复（审计地雷⑥ ⑦ + logger 泄漏⑧ + 健壮性）
- **修改** `PINN4SOH/Model/Model.py`：
  - `LR_Scheduler.step()`：修复 `constant_predictor_lr` 分支下 `lr` 未定义（`UnboundLocalError`）隐患——改用 `self.base_lr`；常规分支对 `param_group` 名称改用 `.get('name')` 防御 `KeyError`。
  - `load_model()`：`torch.load` 补 `weights_only=True`，兼容 PyTorch 2.6+ 默认值变更并消除安全风险。
- **修改** `PINN4SOH/utils/util.py`：
  - `get_logger`：改为 `name` 参数化单例，并在重复调用时去重 handler，根治循环多次实验导致的 handler 累积泄漏。
  - `write_to_txt`：增加目录自动创建。原 `dataloader.load_all_battery` 在目标目录不存在时会崩溃（潜伏的健壮性问题），顺手修复。

### 新增文件清单
| 文件 | 用途 |
| --- | --- |
| `PINN4SOH/utils/split.py` | 统一数据划分模块（uniform_split 等） |
| `PINN4SOH/verify_fixes.py` | 修复验证脚本（11 项断言，含端到端训练） |
| `PINN4SOH/check_split.py` | 四数据集划分规则体检脚本（可保留作回归用） |

### 修改文件清单
- `PINN4SOH/utils/util.py`（set_seed、get_logger、write_to_txt）
- `PINN4SOH/Model/Model.py`（播种、LR_Scheduler、torch.load）
- `PINN4SOH/main_XJTU.py` / `main_HUST.py` / `main_MIT.py` / `main_TJU.py`（--seed、--split_strategy）

### 验证方式
```bash
# 项目实际运行环境：conda 环境 main（torch 2.11.0+cu128，CUDA available）
# 注意：managed Python 3.13.12 未装 torch，不能用于跑训练
cd PINN4SOH
/d/APP/miniconda3/envs/main/python.exe verify_fixes.py --e2e
```
结果：**11/11 通过**（随机种子 3 项 + 统一划分 3 项 + 已知缺陷 3 项 + 端到端 legacy/uniform 各 1 项）。
XJTU 2C 跑 2 epoch 训练正常收敛（Test MAPE 0.61%）。

### 本次未改动（已记入待办）
- **地雷①**：`run_all.py` 的 finetune 分支仍为**空操作**——因 `main_adaptation - fine-tuning.py` 文件名含空格无法 import。需先重命名该文件再接线。
- **地雷⑤**：loss3 仍使用 `.sum()` 而非 `.mean()`，尺度随 batch_size 漂移。改为 `.mean()` 会同时改变 β 标定，需重跑基线，单独排期。

### 使用提示
```bash
# 复现原有基线行为（默认）
python main_XJTU.py --batch 2C

# 启用统一划分 + 可复现种子
python main_XJTU.py --batch 2C --split_strategy uniform --seed 420
```
