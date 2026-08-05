# PINN4SOH 环境迁移与验证报告

> **论文**：Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis
> **期刊**：Nature Communications, 2024, 15:4332
> **作者**：Fujin Wang, Zhi Zhai, Zhibin Zhao, Yi Ding, Xuefeng Chen（西安交通大学）
> **本仓库工作**：Python 3.7 → 3.12 / PyTorch 1.7 → 2.x 环境迁移、代码缺陷修复、结果验证
> **报告日期**：2026 年 7 月（文档整理于 2026 年 10 月）

> **数据源声明**：本报告中的实验数值均取自 `PINN4SOH/results/SUMMARY.md`
> （由 `results_summary.py` 从 `results/` 与 `results_fine-tuning/` 的原始产物自动汇总）。
> 文件名原为 `14-reproduction-report.md`，因项目定位是"在官方代码上做迁移与验证"而非从零复现，已更名。

---

## 1. 验证环境

| 项目 | 配置 |
|------|------|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA RTX 5060 Laptop GPU |
| OS | Windows 11 |
| 关键依赖 | numpy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, scienceplots 2.2.2 |

### 环境适配修复记录

| 修复项 | 原因 | 修改文件 |
|--------|------|---------|
| pandas ≥2.0 int64/float64 赋值限制 | `df` 含 int64 特征列（如 HUST 的 `CC charge time`），归一化写回时抛错 | `dataloader/dataloader.py` |
| GPU ID 1 → 0 | 原作者使用多卡环境，本机仅单卡，写死 `'1'` 会导致 GPU 不可见 | `main_HUST.py` 等四个入口 |
| pandas 3.x `ExcelWriter.save()` 废弃 | 结果分析脚本报错 | `results analysis/*.py` |
| `eval_metrix` 增加 R² 版本 | FineTune 分析脚本需要 R² | `utils/util.py` + 相关分析脚本 |
| 分析脚本 import 路径 | 子目录脚本无法 import 父级模块 | 5 个分析脚本添加 `sys.path` |

### 代码缺陷修复记录

| 类别 | 问题 | 修改文件 |
|------|------|---------|
| 可复现性 | 未固定随机种子（仅 `train_test_split` 有 `random_state`） | `utils/util.py::set_seed` + 四个入口 `--seed` |
| 可复现性 | 数据划分依赖 `os.listdir` 顺序、XJTU 用子串匹配 | 新增 `utils/split.py` + `--split_strategy` |
| 正确性 | `LR_Scheduler.step()` 在 constant 分支下 `UnboundLocalError` | `Model/Model.py` |
| 正确性 | 早停判据耦合 `testloader`，`validloader` 缺失时永不早停 | `Model/Model.py::Train` |
| 兼容性 | `torch.load` 未声明 `weights_only`（PyTorch 2.6+ 默认值变更） | `Model/Model.py::load_model` |
| 健壮性 | `get_logger` 重复调用累积 handler，文件句柄泄漏 | `utils/util.py` |
| 健壮性 | `write_to_txt` 未指定编码（Windows 下为 ANSI） | `utils/util.py` |
| 健壮性 | `write_to_txt` 未创建目标目录 | `utils/util.py` |
| 工程性 | 迁移学习入口文件名含空格，无法 import | 重命名为 `main_finetune.py` |

完整改动清单（含动机、验证方式、未改动项）见 `docs/CHANGELOG.md`。

---

## 2. 实验规模

| 实验类型 | 数据集 | 实验次数 |
|----------|--------|---------|
| 主训练 | XJTU（6 协议 × 10 次） | 60 |
| 主训练 | HUST（10 次） | 10 |
| 主训练 | MIT（10 次） | 10 |
| 主训练 | TJU（3 批 × 10 次） | 30 |
| 基线对比 | XJTU-MLP（6 协议 × 10 次） | 60 |
| 基线对比 | XJTU-CNN（6 协议 × 10 次） | 60 |
| 迁移学习 | TJU→XJTU / XJTU→TJU / HUST→MIT / MIT→HUST | 110 |
| **合计** | | **340 次训练** |

> 全部在 RTX 5060 上完成。单次训练 70–200 epoch（含早停），整体耗时约数小时。

---

## 3. 实验结果

### 3.1 四个数据集主训练（PINN）

| 数据集 | 实验次数 | MAPE (%) | MAE | RMSE | R² | 论文 MAPE | 偏差 |
|--------|---------|----------|-----|------|----|-----------|------|
| XJTU | 60 | 0.87 ± 0.25 | 0.0080 | 0.0113 | 0.9072 | 0.78% | +0.09 pp |
| HUST | 10 | 0.80 ± 0.02 | 0.0078 | 0.0094 | 0.9849 | 0.65% | +0.15 pp |
| MIT | 10 | 0.73 ± 0.04 | 0.0069 | 0.0097 | 0.8876 | 0.78% | −0.05 pp |
| TJU | 30 | 1.13 ± 0.29 | 0.0090 | 0.0123 | 0.9514 | 1.21% | −0.08 pp |

### 3.2 XJTU 逐协议明细

| Batch | 协议 | MAPE (%) |
|-------|------|----------|
| 1 | 2C | 0.62 ± 0.08 |
| 2 | 3C | 1.27 ± 0.10 |
| 3 | R2.5 | 0.86 ± 0.11 |
| 4 | R3 | 0.78 ± 0.12 |
| 5 | RW | 1.06 ± 0.08 |
| 6 | satellite | 0.65 ± 0.04 |

### 3.3 基线对比（XJTU）

| 方法 | 实验次数 | MAPE (%) | MAE | RMSE | R² |
|------|---------|----------|-----|------|----|
| **PINN** | 60 | **0.87 ± 0.25** | 0.0080 | 0.0113 | 0.9072 |
| MLP | 60 | 2.23 ± 0.91 | 0.0207 | 0.0249 | 0.5368 |
| CNN | 60 | 2.53 ± 0.98 | 0.0236 | 0.0329 | 0.0693 |

### 3.4 迁移学习

| 源域 → 目标域 | 实验次数 | 微调后 MAPE (%) |
|--------------|---------|----------------|
| TJU → XJTU | 60 | 1.03 ± 0.31 |
| XJTU → TJU | 30 | 3.61 ± 3.04 |
| HUST → MIT | 10 | 2.40 ± 0.10 |
| MIT → HUST | 10 | 3.98 ± 0.13 |

以 TJU → XJTU 为例，源域模型直接在目标域测试时 MAPE 约 **11%**（见各
`Experiment*/logging.txt` 中的 `Source only` 行），微调后降至 **1.03%**，降幅约一个数量级。

---

## 4. 与论文结论的一致性判定

| 结论 | 论文表述 | 本仓库验证 | 判定 |
|------|---------|-----------|------|
| 四数据集平均精度 | MAPE 0.87% | 0.88%（偏差 +0.01 pp） | ✅ 一致 |
| PINN 优于 MLP / CNN | 显著优于基线 | XJTU 上 0.87% vs 2.23% / 2.53%，R² 0.91 vs 0.54 / 0.07 | ✅ 一致 |
| 迁移学习有效 | 微调显著降低跨域误差 | TJU→XJTU：约 11% → 1.03% | ✅ 一致（限已完成四组） |
| 小样本场景优势明显 | 论文 Fig.5 | **尚未执行该实验** | ⚠️ 未验证 |
| 跨化学体系迁移规律 | 论文 Fig.6 | **尚未执行（8 组）** | ⚠️ 未验证 |

### 误差来源说明

本仓库实测与论文报告值相差均在 ±0.15 pp 以内，可能原因：

1. **训练随机性**：论文未公开全部随机种子设置，本仓库以 10–60 次重复的均值作比；
2. **硬件差异**：GPU 型号不同，浮点累加顺序存在差异；
3. **超参数**：论文未公开全部训练细节，本仓库沿用官方代码默认值；
4. **测试集构造**：默认沿用 legacy 划分规则，其中 XJTU 3C 批因文件名子串匹配
   多纳入 1 只测试电池（详见 `docs/CHANGELOG.md` 与 `check_split.py` 输出）。

---

## 5. 产出文件清单

```
PINN4SOH/
├── results/
│   ├── XJTU results/            # 60 组 PINN（6 协议 × 10 次）
│   ├── HUST results/            # 10 组
│   ├── MIT results/             # 10 组
│   ├── TJU results/             # 30 组（3 批 × 10 次）
│   ├── XJTU-MLP results/        # 60 组 MLP 基线
│   ├── XJTU-CNN results/        # 60 组 CNN 基线
│   ├── SUMMARY.md               # ★ 全部指标的汇总（唯一数据源）
│   └── summary.csv              # 机器可读版本
├── results_fine-tuning/         # 四组迁移学习共 110 组
├── pretrained model/            # 预训练权重
└── results_summary.py           # 汇总脚本（生成上述 SUMMARY.md）
```

---

## 6. 复现本报告的方法

```bash
cd PINN4SOH

# 快速验证（约 3 分钟，含端到端训练）
python verify_fixes.py --e2e
python check_consistency.py

# 汇总实验指标（读已有产物，约 10 秒）
python results_summary.py

# 完整重跑全部实验（约 340 次训练，耗时长）
python run_all.py all
```

---

## 7. 后续工作

1. **补齐未验证的论文结论**：小样本实验（Fig.5）、跨化学体系迁移（8 组）；
2. **扩展基线范围**：当前 MLP/CNN 仅覆盖 XJTU，需扩展到其余三个数据集；
3. **统一划分下的复测**：若采用 `--split_strategy uniform`，需在同一划分下重跑全部对比方法；
4. **结果包与仓库分离**：将可再生成的 `.npy` 产物改为按需解压，缩减仓库体积（见 `docs/GIT_USAGE.md` §5）。
