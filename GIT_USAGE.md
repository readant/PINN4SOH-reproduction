# Git 仓库使用说明

## 仓库结构

本目录包含三套 Git 历史，采用**表层统一 + 底层备份**的方式组织：

```
解剖论文/                              ← 顶层 Git 仓库（你日常使用的）
│
├── .git/                              ← 顶层仓库的版本库
├── .gitignore
│
├── PINN4SOH/                          ← 论文源码（已纳入顶层仓库）
│   └── .git.backup/                   ← [备份] 原作者仓库完整历史
│
├── Battery-dataset-preprocessing-code-library/
│   └── .git.backup/                   ← [备份] 原作者仓库完整历史
│
├── docs/                              ← 你写的论文解读文档
├── model.pth                          ← 预训练模型
├── nc-PINN.pdf                        ← 论文 PDF
└── results of reviewer/               ← 审稿结果
```

### 设计思路

- **顶层仓库**（`解剖论文/.git`）：统一管理所有文件，包括你对源码的修改、注释文档、新增的工具脚本等。日常工作只需要在这个仓库提交。
- **子仓库备份**（`.git.backup`）：保留原作者仓库的完整 commit 历史，供你分析论文代码演进过程时查阅。**它们不会被纳入顶层仓库的版本管理**（已在 `.gitignore` 中排除）。

### 为什么这样设计

1. 原始代码来自两个不同的 GitHub 仓库，各自有独立的 Git 历史
2. 你在这个项目中的工作是**复现论文**，需要修改源码、添加文档，而不是持续拉取上游更新
3. 一个统一的顶层仓库让日常提交更简单，不需要处理 Git submodule 的两步提交
4. 原仓库历史作为备份保留，需要时可以随时查阅

---

## 日常工作流

### 提交修改

所有工作都在顶层仓库进行，和普通 Git 仓库一样：

```bash
cd 解剖论文
git add -A
git commit -m "修复了 xxx 问题"
```

**不需要**进入 `PINN4SOH/` 或 `Battery-dataset/` 目录单独提交。

### 查看提交历史

```bash
git log --oneline
```

输出示例：

```
4594a66 add: PINN4SOH paper source (14 fixes applied) + Battery dataset library
e76f5fa init: project scaffolding, docs, and pretrained model
```

### 查看某个文件的改动

```bash
git log -p -- PINN4SOH/dataloader/dataloader.py
git diff HEAD~1 -- PINN4SOH/main_HUST.py
```

### 查看当前状态

```bash
git status
```

---

## 查阅原仓库历史

当你需要查看原作者代码的提交记录时：

```bash
# 查看 PINN4SOH 原始提交日志
git --git-dir=PINN4SOH/.git.backup log --oneline

# 查看某个文件的原始版本
git --git-dir=PINN4SOH/.git.backup show HEAD:dataloader/dataloader.py

# 对比你的修改和原始版本
git --git-dir=PINN4SOH/.git.backup show HEAD:dataloader/dataloader.py > /tmp/original.py
diff /tmp/original.py PINN4SOH/dataloader/dataloader.py
```

```bash
# 查看 Battery-dataset 原始提交日志
git --git-dir=Battery-dataset-preprocessing-code-library/.git.backup log --oneline
```

**注意**：直接 `cd PINN4SOH && git log` 不会工作，因为 `PINN4SOH/.git` 已被重命名为 `.git.backup`，且该目录被顶层 `.gitignore` 排除。需要显式指定 `--git-dir` 参数。

---

## 恢复原仓库（如果需要）

如果你想恢复子仓库为独立 Git 仓库（例如需要提交回上游），可以：

```bash
# 恢复 PINN4SOH 为独立仓库
cd PINN4SOH
mv .git.backup .git
# 此时 PINN4SOH 恢复为独立 Git 仓库，可以 git commit / git push 等
```

恢复后注意：顶层 `.gitignore` 中的 `PINN4SOH/.git.backup/` 规则不再生效，但如果你打算长期使用独立仓库，应该改用 `git submodule` 方案管理。

---

## 文件变更速查

以下是你在原作者源码上做的修改汇总，方便回溯：

| 文件 | 修改 |
|------|------|
| `PINN4SOH/dataloader/dataloader.py` | 修复 pandas 3.0 类型兼容（cycle_index dtype） |
| `PINN4SOH/main_HUST.py` | ① GPU ID `1`→`0` ② 激活 `main()` 入口 |
| `PINN4SOH/main_MIT.py` | ① 输出路径修正 ② 激活 `main()` 入口 |
| `PINN4SOH/main_TJU.py` | 激活 `main()` 入口 |
| `PINN4SOH/main_XJTU.py` | 输出路径统一 |
| `PINN4SOH/main_comparision.py` | 输出路径统一 |
| `PINN4SOH/utils/util.py` | 新增 `eval_metrix_with_r2()` 函数 |
| `PINN4SOH/results analysis/XJTU results.py` | 输入路径修正 |
| `PINN4SOH/results analysis/Comparision results.py` | 输入路径修正 |
| `PINN4SOH/results analysis/MIT results.py` | 输入路径修正 + R2 函数适配 |
| `PINN4SOH/results analysis/HUST results.py` | R2 函数适配 |
| `PINN4SOH/results analysis/TJU results.py` | R2 函数适配 |
| `PINN4SOH/results analysis/FineTune results.py` | R2 函数适配 |
| `PINN4SOH/run_all.py` | **[新增]** 一键复现入口 |
| `PINN4SOH/plotter/Figure 6.py` | **[新增]** 迁移学习 Fig 6 绘图 |
| `PINN4SOH/smoke_test.py` | **[新增]** 冒烟测试脚本 |
