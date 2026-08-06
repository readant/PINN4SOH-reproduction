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
python main_XJTU.py --batch 2C --split_strategy uniform --seed 42
```

---
---

# 第二阶段：代码缺陷复查与文档一致性整改

## 2026-08-02 — 代码缺陷修复与数据划分去偏置

### 背景

完成首轮修复后对代码做系统性复查，发现若干影响结论可靠性的实现缺陷：
早停判据与测试集耦合、命令行开关实际无法关闭、统一划分存在固定位置偏置、测试中存在恒真断言。

### 改动

#### 1. 早停判据与测试集解耦（`PINN4SOH/Model/Model.py::Train`）
- 早停与"最优模型"判定改为**只依赖验证集**；`testloader` 仅用于评估与落盘预测，
  不再参与任何模型选择，消除测试集泄漏式调参。
- 修复的具体缺陷：`validloader=None` 而 `testloader` 有效时，`valid_mse` 恒为初值 10、
  判定条件恒真、`early_stop` 永远归零 → **训练永不早停并逐轮覆盖保存**。
- `validloader` 与 `testloader` 同时缺失时显式抛 `ValueError`，不再静默退化。

#### 2. 命令行开关修复（四个 `main_*.py`）
- `--deterministic` 由 `type=bool` 改为 `argparse.BooleanOptionalAction`。
  原写法下 `--deterministic False` 会被解析为 `True`（非空字符串即真），
  等同于**无法从命令行关闭**确定性模式。

#### 3. 统一数据划分去偏置（`PINN4SOH/utils/split.py`）
- `uniform_split` 的测试索引由 `int(i * step)`（恒从索引 0 起算）改为取每段中点
  `int((i + 0.5) * step)`。原实现使**每个划分单元中编号最小的电池恒定进入测试集**
  （XJTU 6 个 batch 全部命中 1 号电池），构成系统性偏置。
- 同步修正 docstring 中"符合原论文取 4/8 号意图"这一无依据表述，
  并补充说明"划分单元仍由调用方决定"的适用范围与局限。

#### 4. 其他健壮性修复（`PINN4SOH/utils/util.py`）
- `write_to_txt` 显式声明 `encoding='utf-8'`（Windows 默认 ANSI 代码页）。
- `set_seed` 移除运行时设置 `PYTHONHASHSEED` 的语句 —— 该变量只在进程启动前生效，
  运行时赋值不产生任何效果，保留会造成"已固定哈希种子"的错觉。

### 验证方式

```bash
cd PINN4SOH
/d/APP/miniconda3/envs/main/python.exe verify_fixes.py          # 10/10 通过
/d/APP/miniconda3/envs/main/python.exe verify_fixes.py --e2e    # 12/12 通过
```

新增回归项：早停策略三场景（无验证集须跑满 / 有验证集按容忍次数停止 / 两者全缺显式报错）；
`uniform_split` 的断言由恒真的 `max(ids) <= 20` 替换为"不含最小编号 + 相邻间隔不超过 2 倍步长"。

### 本次未改动

`loss3` 仍使用 `.sum()` 而非 `.mean()`，尺度随 batch size 漂移。改 `.mean()` 会同时改变
β 的标定，需重跑全部基线，单独排期。

---

## 2026-08-03 — 一键脚本链路修复与迁移学习接线

### 改动

#### 1. `PINN4SOH/run_all.py`
- `_run_script()` 增加 `check=True` 与 `cwd` 参数：子进程失败不再被静默吞掉
  （原实现不检查返回码，分析脚本全部失败仍会打印"完成"）。
- `run_analyze()` 改为调用 `results_summary.py`。原实现调用的 `results analysis/*.py`
  读取的是 `../results/Ours/<数据集> results/`（原作者提供的结果包，需解压且目录名匹配），
  以 PINN4SOH 为 cwd 时路径必然解析失败。
- 新增 `analyze-original` 阶段处理原作者结果包：先解压 `results/Ours/*.zip`，
  再校验目录名，不匹配时给出明确指引并以非零状态结束。
- `all` 阶段任一子阶段返回失败即中止后续阶段。

#### 2. 迁移学习入口重命名
- `main_adaptation - fine-tuning.py` → **`PINN4SOH/main_finetune.py`**。
  原文件名含空格与连字符，Python 无法 import，只能 subprocess 调用，
  导致 `run_all.py` 的 finetune 分支**长期是空操作**（只打印提示却照常返回）。
- 新增 `run_standard_tasks()`（供 run_all 调用）与命令行入口：
  `python main_finetune.py [standard|tju2xjtu|xjtu2tju|hust2mit|mit2hust|cross]`。

#### 3. 冒烟测试对齐新能力（`PINN4SOH/smoke_test.py`）
- 原实现硬编码 legacy 的子串划分（`'4' in f or '8' in f`）且不固定随机种子，
  与仓库新增的可复现能力脱节。现改用 `uniform_split` + `set_seed(42)`，
  并补充测试集加载与评估。

### 验证方式

```bash
cd PINN4SOH
python run_all.py analyze      # 退出码 0，生成 results/SUMMARY.md
python main_finetune.py        # 正常进入标准四组迁移流程
python smoke_test.py           # 1 epoch 冒烟通过
```

---

## 2026-08-04 — 新增结果汇总脚本（仓库唯一数据源）

### 改动
- 新增 `PINN4SOH/results_summary.py`：扫描 `results/` 与 `results_fine-tuning/` 下
  所有含 `true_label.npy` + `pred_label.npy` 的实验目录，汇总为
  `results/SUMMARY.md`（人读）与 `results/summary.csv`（机读）。

### 动机

仓库中存在两套结果目录：

| 目录 | 归属 |
|------|------|
| `results/` | **本仓库实测**（本机训练产物） |
| `results/Ours/`、`results analysis/processed results/` | **原作者提供**（对应论文数值） |

前期文档混用了两者的数字，导致同一指标出现多个互相矛盾的取值
（如 XJTU 上 MLP 的 MAPE 在仓库内曾同时存在 1.26% / 2.11% / 2.23% 三种写法）。

### 约定
- 逐个扫描实验目录，**不写死编号**；
- MAPE 统一输出为百分比（×100），与论文口径一致；
- 产物缺失的分组标注"无产物"，**不估计、不填充、不引用他处数字**。

### 验证方式
输出的四数据集 MAPE 与 `results/*.xlsx`、各 `Experiment*/` 原始产物逐项复核一致。

---

## 2026-08-05 — 文档与代码一致性整改

### 背景

多篇文档描述的是**仓库中并不存在**的实现：虚构的 `src/` 目录结构、
`SolutionNetwork` / `DynamicsNetwork` 类、`BatteryDataPreprocessor` 类、
`extract_features_from_charge()` 函数，以及 `data/*/processed/` 目录约定。
这些内容对读者具有误导性，且无法通过任何校验。

### 改动

| 文件 | 改动 |
|------|------|
| `docs/07-code-mapping.md` | **重写**。改为对照真实目录结构与类名（`Solution_u` / `dynamical_F` / `DF` / `AdaPINN`），逐条给出论文方法与代码位置的映射 |
| `docs/08-verification-guide.md` | **重写并更名**（原 `08-reproduction-guide.md`）。改为可执行的验证流程：环境 → 干跑校验 → 冒烟 → 端到端 → 结果汇总 → 论文比对 |
| `docs/09-data-preprocessing.md` | **重写**。删除虚构类与函数，改为真实预处理链路（官方预处理库 → CSV → `DF.read_one_csv` 五步），并显式标出"min-max 归一化区间是 [-1,1] 而非 [0,1]"这一易错点 |
| `docs/13-checklist.md` | **重写**。由"从零复现清单"改为本项目可执行的验证检查清单 |
| `docs/03-pinn-architecture.md` | 修正网络结构为 **17 → 60 → 60 → 32**（原写 64→64→32）；符号更正为 `Solution_u` / `dynamical_F`；移除并不存在的 Sigmoid 输出层；更正 PDE 残差为 `f = u_t − F` |
| `docs/04-loss-functions.md` | 更正 PDE 公式、`L_data` 的成对结构与 0.5 系数、`loss3` 的真实语义（方向一致性，允许容量再生，非单调递减）；补充四个数据集 α/β 的实际默认值 |
| `docs/14-verification-report.md` | **更名**（原 `14-reproduction-report.md`）并更新实验规模表与数据源声明 |
| `docs/README.md` | 索引补全 08–14 与 CHANGELOG / GIT_USAGE；关键数字表更正 |

### 新增

- `PINN4SOH/check_consistency.py`：自动校验
  ① 文档中引用的 `.py` 是否真实存在；② `data/` 数据集文件数与口径是否一致；
  ③ 关键代码事实（类名、`--seed` 默认值、`BooleanOptionalAction`、`main_finetune.py`）；
  ④ 是否残留已废弃的类名/函数名；⑤ 文档中的实测数字能否在 `results/SUMMARY.md` 溯源。

### 验证方式

```bash
cd PINN4SOH
python check_consistency.py      # 6/6 通过
```

---

## 2026-08-06 — 统一实验数据源与文档数字更正

### 背景

仓库内同一指标曾出现多套互相矛盾的数字，且部分数字**没有任何实验产物支撑**
（例如"1 周期 PINN MAPE ~1.8%""HUST→XJTU 迁移 1.52%"），另有个别把论文报告值
当作本仓库实测值使用的情况。

### 改动

#### 1. `README.md`
- 数据集规模更正为 **XJTU 55 / HUST 77 / MIT 125 / TJU 130**（合计 387，与论文口径一致）。
  原表写作 HUST 166 / TJU 52 / MIT 124，合计 397，与论文矛盾且与 `data/` 实际文件数不符。
- clone 与 Issues 链接更正为真实远端 `readant/PINN4SOH-reproduction`
  （原写作不存在的 `readant/nc-PINN`）。
- 补充"质量保障"与"数据源纪律"两节。

#### 2. `docs/06-experiments.md`
- 全部数字替换为 `results/SUMMARY.md` 的实测值；
- 删除小样本实验与未执行迁移组合的数字，改为明确列出"本仓库尚未执行的实验"；
- 更正数据集规模表与实验规模表。

#### 3. `nc-pinn/docs/beginner-guide/`
- `01-paper-overview.md`：数据集表更正；基线对比改为本仓库实测值（基线仅覆盖 XJTU）；
- `05-understand-results.md`：区分"论文报告值"与"本仓库实测值"；
  删除无产物支撑的 Fig.5 / Fig.6 数字，改为"尚未执行"并给出补做命令。

#### 4. `docs/GIT_USAGE.md`
- 重写：更正仓库名（原写"解剖论文/"）、移除已不存在的目录、补充分支说明与
  版本控制边界、常见问题。

### 新增

| 文件 | 用途 |
| --- | --- |
| `PINN4SOH/data/README.md` | 数据集来源、目录结构、CSV 格式与许可说明 |
| `PINN4SOH/results/SUMMARY.md` | 由 `results_summary.py` 生成（本次首次产出） |
| `.github/workflows/ci.yml` | 最小 CI：语法检查 + 一致性校验 |

### 验证方式

```bash
cd PINN4SOH
python check_consistency.py      # 6/6 通过
python results_summary.py        # 重新生成 SUMMARY.md
```

文档中出现的每个实测 MAPE 均可在 `results/SUMMARY.md` 中定位。
