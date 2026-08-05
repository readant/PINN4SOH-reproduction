# 08 - 验证指南

> 本文档说明**如何在本机完整走一遍本项目的验证流程**：从环境搭建到结果汇总，
> 每一步都可执行、可核对。
> 文件名原为 `08-reproduction-guide.md`，因项目定位是"在官方代码基础上做环境迁移与
> 缺陷修复验证"（而非从零复现），已更名为验证指南。

## 一、前置条件

| 项目 | 要求 |
|------|------|
| Python | 3.12+ |
| PyTorch | 2.x（CUDA 版本，可用 GPU） |
| 依赖 | `pip install -r requirements.txt` |
| 数据 | `PINN4SOH/data/` 下四数据集共 387 个 CSV（随仓库提供） |
| 预训练权重 | `PINN4SOH/pretrained model/` 下 11 个 `.pth` |
| 工作目录 | 所有命令在 `PINN4SOH/` 下执行 |

确认环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.x.x+cuXXX True
```

## 二、验证流程（建议顺序）

### 第 1 步：干跑校验（不需要 GPU 训练，约 1 分钟）

```bash
cd PINN4SOH
python check_consistency.py     # 文档与代码是否一致
python check_split.py           # 数据划分规则体检
python verify_fixes.py          # 修复项回归（10 项）
```

期望结果：

- `check_consistency.py` → **6/6 通过**
- `check_split.py` → 打印 legacy / uniform 两套划分的对照表
- `verify_fixes.py` → **10/10 通过**

### 第 2 步：冒烟测试（1 epoch，约 1 分钟）

```bash
python smoke_test.py
```

期望：加载 XJTU 2C（uniform 划分，8 只电池中 2 只为测试），
完成 1 轮训练并输出测试指标，退出码 0。

### 第 3 步：端到端验证（含真实训练，约 2 分钟）

```bash
python verify_fixes.py --e2e
```

期望：**12/12 通过**，其中包含 legacy / uniform 两种划分下各 2 epoch 的 XJTU 2C 训练。

### 第 4 步：结果汇总（约 10 秒）

```bash
python results_summary.py
```

期望：生成

- `results/SUMMARY.md` —— 仓库内**唯一**的实验数据源
- `results/summary.csv` —— 机器可读版本

该脚本从 `results/` 与 `results_fine-tuning/` 的原始
`true_label.npy` / `pred_label.npy` 逐目录汇总，**不写死实验编号、不做任何估计**。

### 第 5 步（可选）：重跑实验

完整重跑耗时很长（约 400 次训练），按需选择：

```bash
python run_all.py xjtu          # XJTU 主实验（60 次）
python run_all.py hust          # HUST（10 次）
python run_all.py mit           # MIT（10 次）
python run_all.py tju           # TJU（30 次）
python run_all.py comparison    # MLP/CNN 基线（120 次）
python run_all.py finetune      # 迁移学习标准四组（110 次）
python run_all.py analyze       # 汇总（等价于 results_summary.py）
```

> `run_all.py` 的任一阶段返回非零状态时，整体以非零退出码终止，不再静默跳过。

## 三、与论文结果比对

汇总后打开 `results/SUMMARY.md`，与论文报告值对照：

| 数据集 | 论文报告 MAPE | 本仓库实测 MAPE | 允许偏差 |
|--------|--------------|----------------|---------|
| XJTU | 0.78% | 0.87 ± 0.25% | ±0.2 pp |
| HUST | 0.65% | 0.80 ± 0.02% | ±0.2 pp |
| MIT | 0.78% | 0.73 ± 0.04% | ±0.2 pp |
| TJU | 1.21% | 1.13 ± 0.29% | ±0.2 pp |
| **平均** | **0.87%** | **0.88%** | **±0.1 pp** |

四项均在允许偏差内，平均仅差 0.01 pp → 论文的精度声明在新环境下成立。

偏差的合理来源（详见文档 06 §6）：

1. 训练随机性（论文未公开全部随机种子）；
2. GPU 型号不同导致的浮点累加顺序差异；
3. 论文未公开全部训练细节，本仓库沿用官方默认超参；
4. legacy 划分规则本身的缺陷（XJTU 3C 批多纳入 1 只测试电池）。

## 四、本仓库尚未验证的论文结论

| 论文结论 | 状态 | 补做方式 |
|---------|------|---------|
| 小样本场景 PINN 优势更明显（Fig.5） | **未验证** | `python run_all.py small` |
| 跨化学体系迁移（异族，如 XJTU↔HUST） | **未验证** | `python main_finetune.py cross` |
| HUST / MIT 上的 MLP、CNN 基线 | **未验证** | 需扩展 `main_comparision.py` 的数据集范围 |

**这三项是当前验证工作的空白点，也是下一步最直接的补充实验。**

## 五、环境迁移过程中修复的问题

完整清单与验证方式见 `docs/CHANGELOG.md`。要点：

| 类别 | 问题 | 影响 |
|------|------|------|
| 环境兼容 | pandas ≥2.0 的 int64/float64 赋值限制 | 加载 HUST 等数据集会直接报错 |
| 环境兼容 | 原代码写死 `CUDA_VISIBLE_DEVICES='1'` | 单卡机器上 GPU 不可见 |
| 环境兼容 | `writer.save()` 在 pandas 3.x 已废弃 | 结果分析脚本报错 |
| 可复现性 | 未固定随机种子 | 同一配置两次运行结果不同，"10 次取均值"无法被他人复现 |
| 可复现性 | 数据划分依赖 `os.listdir` 顺序 | 不同文件系统上划分结果可能不同 |
| 正确性 | `LR_Scheduler.step()` 在 constant 分支下崩溃 | 特定配置下训练中断 |
| 正确性 | 早停判据耦合 `testloader` | `validloader` 缺失时永不早停；存在用测试集选模型的风险 |
| 健壮性 | `get_logger` 重复调用累积 handler | 循环实验时日志重复、文件句柄泄漏 |
| 健壮性 | `write_to_txt` 未指定编码 | Windows 下中文日志编码不一致 |
| 工程性 | 迁移学习入口文件名含空格 | 无法 import，一键脚本的 finetune 阶段长期为空操作 |

## 六、常见问题

**Q：`FileNotFoundError: data/XJTU data`**
A：当前目录不是 `PINN4SOH/`。所有训练脚本都以 `PINN4SOH/` 为相对路径基准。

**Q：`results analysis/*.py` 报找不到 `../results/Ours/XJTU results/`**
A：这些脚本处理的是**原作者提供**的结果包，需先解压 `results/Ours/*.zip`，
且解压后目录名要与脚本默认值匹配（仓库随附的 XJTU.zip 解压得到的是
`XJTU results (small sample N)/`，名称并不匹配）。
汇总本仓库自己的结果请用 `python results_summary.py`。

**Q：为什么 `results/XJTU_results.xlsx` 里的 MAPE 是 0.0061 而不是 0.61%？**
A：`sklearn.metrics.mean_absolute_percentage_error` 返回的是**小数比例**，
需要乘 100 才是百分比。`results_summary.py` 统一输出百分比，是推荐的口径。

**Q：能否用 uniform 划分的结果与已有基线对比？**
A：不能。划分变化等价于实验条件变化，两套结果不可直接比较。
切换后需在同一划分下重跑全部对比方法。

## 七、相关文档

- 改动记录与验证方式 → `docs/CHANGELOG.md`
- 逐项检查清单 → `docs/13-checklist.md`
- 实验数据与结论 → `docs/06-experiments.md`、`results/SUMMARY.md`
- 代码与论文对应关系 → `docs/07-code-mapping.md`
