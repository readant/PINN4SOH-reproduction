# Git 仓库使用说明

## 一、仓库基本信息

| 项目 | 内容 |
|------|------|
| 本地目录 | `E:\Projects0\Python\nc-PINN`（仓库根） |
| 远端 origin | `https://github.com/readant/PINN4SOH-reproduction.git` |
| 主线分支 | `main` |
| 作者 | readant |

## 二、仓库结构

本仓库采用**单一顶层仓库**管理，两个上游子仓库的历史以 `.git.backup/` 形式保留：

```
nc-PINN/                                  ← 顶层 Git 仓库（日常只在此提交）
│
├── .git/                                 ← 顶层版本库
├── .gitignore
│
├── README.md
├── requirements.txt
├── docs/                                 ← 论文解读（01–14）+ CHANGELOG + 本文件
│
├── PINN4SOH/                             ← 论文官方代码（已纳入顶层仓库）
│   └── .git.backup/                      ← [存档] 原作者仓库完整历史，不纳入顶层管理
│
└── Battery-dataset-preprocessing-code-library/
    └── .git.backup/                      ← [存档] 原作者仓库完整历史，不纳入顶层管理
```

### 为什么这样组织

1. 原始代码来自两个不同的 GitHub 仓库，各自有独立历史；
2. 本仓库的工作是**在新环境下验证官方代码并修复其缺陷**，而非持续跟随上游更新，
   因此不需要 submodule 的双步提交流程；
3. 用 `.git.backup/` 保留上游历史供查阅，同时已被 `.gitignore` 排除，不进入顶层版本库。

> 注：`.git.backup/` 是完整的嵌套版本库，直接在仓库根执行 `git status` 时不会显示其内部变化，
> 但递归扫描工具（部分 IDE、打包脚本）可能误入。若需归档，建议先打包成 zip 再删除目录。

## 三、分支说明

| 分支 | 状态 | 内容 |
|------|------|------|
| `main` | **主线**，日常在此提交 | `docs/`（论文解读）、`PINN4SOH/`（修复与验证）、`README.md` |
| `my-reproduction` | 历史存档分支 | `nc-pinn/docs/beginner-guide/`（8 篇初学者指南）、`nc-pinn/docs/onther/`（早期分析）、`nc-PINN.pdf` |

**关于 `my-reproduction`**：该分支与 `main` **没有共同祖先**（两条独立初始提交），
因此无法用常规方式 merge 或 diff。其内容已作为普通文件归并进 `main`，
分支本身保留仅作历史存档 —— 请勿在此分支上继续开发，也不要尝试合并两条历史。

## 四、日常工作流

### 提交修改

所有工作都在顶层仓库进行：

```bash
cd E:/Projects0/Python/nc-PINN
git add -A
git commit -m "fix: 修复 xxx"
```

**不需要**进入 `PINN4SOH/` 或 `Battery-dataset-preprocessing-code-library/` 单独提交。

### 提交信息规范

统一使用 `<类型>: <中文简述>` 的前缀格式：

| 类型 | 用途 |
|------|------|
| `feat` | 新增功能（如统一数据划分模块） |
| `fix` | 修复缺陷 |
| `docs` | 文档改动 |
| `test` | 验证脚本 |
| `refactor` | 重构（行为不变） |
| `chore` | 仓库配置、忽略规则等杂项 |
| `results` | 实验产物入库 |

示例：`fix: 修复早停判据耦合测试集的问题`、`docs: 更新论文解读中的网络结构描述`

**正文（body）**：改动较大时补一段说明范围与要点，与标题之间空一行。例如：

```
feat: 纳入 PINN4SOH 官方代码与数据预处理代码库

- PINN4SOH：Model / dataloader / main_* / plotter / results analysis
- Battery-dataset-preprocessing-code-library：四个数据集的预处理类
- 已应用 Python 3.12 与 PyTorch 2.x 的环境适配修复
```

> 仓库的历史提交已按本规范统一：描述一律改为中文、前缀归入上表集合、
> 较大的提交补齐正文；**提交时间与文件内容均保持原样**，未做任何改写。

### 查看历史与改动

```bash
git log --oneline -20
git log -p -- PINN4SOH/Model/Model.py
git diff HEAD~1 -- PINN4SOH/main_XJTU.py
```

## 五、版本控制边界

### 纳入版本控制

- 源代码、脚本、文档、配置（`PINN4SOH/**/*.py`、`docs/` 等）
- `PINN4SOH/data/` 下预处理后的数据集（387 个 CSV）—— 保证克隆后可直接运行
- `PINN4SOH/results/` 下的实验产物（`true_label.npy` / `pred_label.npy` / `model.pth`）
  —— 作为结果可复核的证据链
- 预训练权重 `PINN4SOH/pretrained model/`

### 排除在版本控制之外

| 路径 | 原因 |
|------|------|
| `.workbuddy/`、`.mimocode/` | 本地工具/代理状态目录 |
| `__pycache__/`、`*.pyc`、`.mypy_cache/` | 可再生成的缓存 |
| `PINN4SOH/.git.backup/`、`Battery-*/  .git.backup/` | 上游历史存档 |
| `PINN4SOH/results/smoke_test/`、`results/verify_fixes_*/` | 冒烟与回归产生的临时产物 |
| `results/Ours/*.zip`、`results/MLP/*.zip`、`results/CNN/*.zip` | 大体积结果包（36 MB），需要时解压 |
| `results_fine-tuning/` 下的中间产物 | 可重新训练生成 |
| `*.pdf` | 论文 PDF（有 DOI 可获取） |

### 关于仓库体量

仓库当前追踪约 1300 个文件（含 387 个数据集 CSV 与 460 余个结果 `.npy`）。
这是"克隆即可复现"与"仓库体积"之间的取舍：当前选择**优先保证可复现性**。
若后续需要瘦身，建议的优先顺序是：

1. 将 `results/**/*.npy` 改为按需解压的压缩包（可再生成，且体量最大）；
2. 将 `data/` 改为下载脚本 + `data/README.md` 说明来源与许可；
3. 保留 `results/SUMMARY.md` 与 `summary.csv`（汇总指标，体积可忽略）。

## 六、数据与产物的两套目录（重要）

| 目录 | 归属 | 说明 |
|------|------|------|
| `PINN4SOH/results/` | **本仓库实测** | 本机训练产物，`results_summary.py` 读取此处 |
| `PINN4SOH/results/Ours/` | **原作者提供** | 随仓库附带的原始结果包（zip，需解压），对应论文数值 |
| `PINN4SOH/results analysis/processed results/` | **原作者提供** | 已处理好的 `.xlsx`，`results analysis/*.py` 的输出目标 |

**两套数据不可混用**。文档中引用实验数字时，一律以 `results/SUMMARY.md` 为准。

## 七、常见问题

**Q：`git status` 出现 `results/verify_fixes_e2e_legacy/` 等未跟踪目录？**
A：这是 `verify_fixes.py` 的正常产物，已在 `.gitignore` 中排除。若仍出现，说明忽略规则未生效，
检查 `.gitignore` 中 `PINN4SOH/results/verify_fixes_*/` 一行。

**Q：为什么 `PINN4SOH/` 里没有 `.git`？**
A：原始 `.git` 已重命名为 `.git.backup` 并排除在顶层仓库外，避免嵌套版本库互相干扰。

**Q：能否直接 `git pull` 上游官方仓库的更新？**
A：不建议。本仓库对官方代码有大量修改（环境迁移 + 缺陷修复），直接拉取会产生大面积冲突。
如需参考上游改动，请到 `.git.backup/` 中查阅其历史。
