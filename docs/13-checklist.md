# 13 - 验证检查清单

> 本清单对应本项目的实际工作方式：**在论文官方代码基础上完成环境迁移与缺陷修复，
> 并验证论文核心结论在新环境下依然成立**。
> 每一项都可执行、可核对；命令默认在 `PINN4SOH/` 目录下运行。

---

## 阶段一：环境准备

- [ ] Python 3.12+
- [ ] PyTorch 2.x（CUDA 版本，需能识别 GPU）
- [ ] 安装依赖：`pip install -r ../requirements.txt`
- [ ] 确认环境可用：
      `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
      → 期望输出 `2.x.x+cuXXX True`

> 注意：本机的 managed Python 3.13.12 未安装 torch，不能用于训练。
> 项目实际运行环境为 conda 环境 `main`（torch 2.11.0+cu128）。

---

## 阶段二：代码与数据就位

- [ ] `PINN4SOH/` 下四个数据集目录存在：
      `data/XJTU data`(55)、`data/HUST data`(77)、`data/MIT data`(125)、`data/TJU data`(130)
      —— 合计 387 个 CSV，与论文「387 只电池」一致
- [ ] 预训练权重存在：`pretrained model/` 下 11 个 `.pth` + 8 个 `mu_var_pi_*.npz`
- [ ] 数据集来源与许可已确认（各数据集条款不同，见根 README「数据集」小节）
- [ ] 从正确的目录运行：所有 `main_*.py` 与汇总脚本必须在 `PINN4SOH/` 下执行

---

## 阶段三：修复项确认（逐条对应 `docs/CHANGELOG.md`）

- [ ] 可复现性：`utils/util.py::set_seed` 存在，四个 `main_*.py` 均有 `--seed`
- [ ] 统一划分：`utils/split.py::uniform_split` 存在，四个 main 均有 `--split_strategy`
- [ ] `LR_Scheduler.step()` 在 constant 分支下不再抛 `UnboundLocalError`
- [ ] `PINN.load_model()` 使用 `torch.load(..., weights_only=True)`
- [ ] `get_logger` 重复调用不再累积 handler
- [ ] 早停判据只依据验证集，`testloader` 不参与模型选择
- [ ] `--deterministic` 可通过 `--no-deterministic` 关闭（原 `type=bool` 写法无法关闭）
- [ ] `write_to_txt` 显式使用 `encoding='utf-8'`
- [ ] 迁移学习入口 `main_finetune.py` 可直接 import / 直接运行

---

## 阶段四：自动化验证（推荐每次改动后执行）

- [ ] `python verify_fixes.py`　　　　　　　→ **10/10 通过**
- [ ] `python verify_fixes.py --e2e`　　　　→ **12/12 通过**（含 XJTU 2C 两种划分各 2 epoch 训练）
- [ ] `python check_split.py`　　　　　　　 → 打印 legacy / uniform 划分对照
- [ ] `python check_consistency.py`　　　　→ 文档中引用的路径 / 类名 / 声明与代码一致
- [ ] `python smoke_test.py`　　　　　　　 → 1 epoch 冒烟通过，退出码 0
- [ ] `python results_summary.py`　　　　  → 生成 `results/SUMMARY.md`

---

## 阶段五：实验结果核对

- [ ] `python run_all.py analyze` 正常退出（退出码 0）
- [ ] `results/SUMMARY.md` 中四数据集 MAPE 与论文偏差在 ±0.2 pp 以内
- [ ] XJTU 六个协议各自 MAPE 均已记录（不遗漏 batch）
- [ ] 基线对比（MLP / CNN）与本仓库 PINN 在同一划分下对比
- [ ] 迁移学习四组的源域直测 / 微调后指标均已记录

> **数据源纪律**：仓库内任何文档引用实验数字时，一律以 `results/SUMMARY.md` 为准。
> `results analysis/processed results/*.xlsx` 是**原作者提供**的结果（对应论文数值），
> 与本仓库实测不是同一套数据，不可混用。

---

## 阶段六：可复现性检查

- [ ] 固定 `--seed` 后，两次运行的初始权重逐位一致（`verify_fixes.py` 已覆盖）
- [ ] 多次实验使用 `seed + e` 递增，而非全部使用同一种子（否则均值无统计意义）
- [ ] 更换 `--split_strategy` 时，明确标注该结果使用了哪套划分
- [ ] 记录运行环境（torch 版本、GPU 型号），写入实验日志

---

## 阶段七：交付前检查

- [ ] `git status` 无未跟踪的临时产物（`results/verify_fixes_*`、`*_logger_probe.txt` 等应被忽略）
- [ ] 文档中出现的每个数字都能在 `results/SUMMARY.md` 或论文中找到出处
- [ ] 文档中出现的每个文件路径、类名在仓库中真实存在（`check_consistency.py` 已覆盖）
- [ ] `docs/CHANGELOG.md` 已记录本次改动（动机 / 文件清单 / 验证方式 / 未改动项）

---

## 常见问题排查

### 数据相关

- [ ] 充电段数据是否完整（特征提取依赖充电末段）
- [ ] SOH 是否落在合理区间（约 0.7 ~ 1.0）
- [ ] 特征中是否含 NaN / Inf（`DF.delete_3_sigma()` 负责剔除异常点）

### 模型相关

- [ ] 训练是否收敛（观察 `logging.txt` 中 loss 曲线）
- [ ] 预测值范围是否合理
- [ ] PDE 残差是否随训练下降（`loss2` 应持续减小）

### 训练相关

- [ ] 学习率是否合适（warmup + 余弦衰减，见 `LR_Scheduler`）
- [ ] 是否触发早停（`early stop at epoch N`）
- [ ] 是否过拟合（比较 valid MSE 与 test 指标走势）

### 路径相关

- [ ] 报 `FileNotFoundError: data/...` → 当前不在 `PINN4SOH/` 目录
- [ ] 报 `<仓库根>/results 不存在` → 在跑 `results analysis/*.py`，需 `cd results analysis` 或改用
      `python run_all.py analyze`
