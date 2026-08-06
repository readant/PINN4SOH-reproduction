"""
文档 / 代码一致性校验
=====================

把"文档里写的东西"与"仓库里真实存在的东西"做机器比对，防止以下问题复发：

* 文档引用了不存在的脚本或模块（历史上曾出现描述虚构 `src/` 结构的文档）；
* 文档里的数据集规模与 `data/` 目录实际文件数不符（历史上曾写错 HUST/TJU 数量）；
* 文档描述了已被删除的旧类名 / 旧函数名；
* 文档中的实测 MAPE 在本仓库结果汇总里找不到出处。

用法（必须从 PINN4SOH 目录运行）：
    python check_consistency.py

退出码 0 = 全部通过，1 = 存在失败项。
"""

import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))          # .../nc-PINN/PINN4SOH
REPO = os.path.dirname(BASE)                                # .../nc-PINN

PASS, FAIL = '[PASS]', '[FAIL]'
results = []


def record(name, ok, detail=''):
    results.append(ok)
    print(f'{PASS if ok else FAIL} {name}' + (f'  -- {detail}' if detail else ''))


# ---------------------------------------------------------------- 工具

def repo_markdown_files():
    """返回需要参与校验的 Markdown 文件（相对仓库根）。"""
    files = []
    for pattern in ('README.md', 'docs/*.md', 'nc-pinn/docs/*.md',
                    'nc-pinn/docs/beginner-guide/*.md'):
        import glob
        files.extend(os.path.relpath(p, REPO)
                     for p in sorted(glob.glob(os.path.join(REPO, pattern))))
    p = os.path.join(BASE, 'README.md')
    if os.path.exists(p):
        files.append(os.path.relpath(p, REPO))
    return files


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def index_python_files():
    """建立仓库内所有 .py 的 basename 索引与相对路径集合。"""
    basenames, relpaths = set(), set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in ('.git', '.git.backup', '__pycache__', 'node_modules',
                                '.mypy_cache', '.workbuddy', '.vscode')]
        for fn in files:
            if fn.endswith('.py'):
                full = os.path.join(root, fn)
                basenames.add(fn)
                relpaths.add(os.path.relpath(full, REPO).replace('\\', '/'))
                relpaths.add(os.path.relpath(full, BASE).replace('\\', '/'))
    return basenames, relpaths


# ---------------------------------------------------------------- 1. 文档引用

# 历史文件名 / 原作者文档中的笔误：允许出现在文档中
# （均已在 PINN4SOH/README.md 的「本仓库补充说明」中显式标注差异）
HISTORICAL = {
    'main_adaptation - fine-tuning.py',
    'main_adaptation_fine_tuning.py',
    'main_comparison.py',
}

def is_review_doc(rel):
    """
    评审 / 整改类文档。

    这类文档按定义就要引用"曾经被写错、现已删除"的东西（如审查报告会列举
    文档里虚构过的文件），因此不参与"引用必须存在"与"不得含废弃符号"两项检查。
    """
    base = os.path.basename(rel)
    return ('审查报告' in base) or ('整改' in base)


CODE_TOKEN = re.compile(r'`([^`\n]{3,80}?\.py(?::\d+(?:-\d+)?)?)`')


def test_referenced_scripts_exist():
    """
    扫描文档中反引号内的 `*.py` 引用，逐个确认文件在仓库中真实存在。

    只校验以 `.py` 结尾的引用（这是历史上出错最多的一类：文档描述过不存在的模块）。
    跳过：含通配符 `*` 的目录约定、含 `{}` 的缩写写法（如 `main_{XJTU,HUST,MIT,TJU}.py`）、
    已归档的历史文件名，以及评审类文档。
    """
    basenames, relpaths = index_python_files()
    missing, checked = [], 0
    for rel in repo_markdown_files():
        if is_review_doc(rel):
            continue
        text = read(os.path.join(REPO, rel))
        for token in CODE_TOKEN.findall(text):
            token = token.strip()
            token = re.sub(r':\d+(-\d+)?$', '', token)     # 去掉行号后缀
            if '*' in token or '{' in token or '}' in token or token in HISTORICAL:
                continue
            exists = (token in relpaths
                      or os.path.exists(os.path.join(REPO, token))
                      or os.path.exists(os.path.join(BASE, token)))
            # 含空格的 token 通常是命令行片段（如 `cd PINN4SOH && python xxx.py`），
            # 只有在"确实存在同名文件"（如 `Figure 4a.py`）时才按文件引用处理
            if ' ' in token and not exists:
                continue
            checked += 1
            if not exists:
                exists = os.path.basename(token) in basenames
            if not exists:
                missing.append(f'{rel}: {token}')
    uniq = sorted(set(missing))
    detail = f'检查 {checked} 处引用，缺失 {len(uniq)} 处'
    if uniq:
        detail += '：' + '；'.join(uniq[:5])
    return not uniq, detail


# ---------------------------------------------------------------- 2. 数据集规模

EXPECTED_DATASETS = {
    'data/XJTU data': 55,
    'data/HUST data': 77,
    'data/MIT data': 125,
    'data/TJU data': 130,
}


def test_dataset_counts():
    """data/ 下的实际 CSV 数必须与文档口径一致（合计 387，与论文一致）。"""
    problems, total = [], 0
    for rel, expected in EXPECTED_DATASETS.items():
        root = os.path.join(BASE, rel)
        n = 0
        for _, _, files in os.walk(root):
            n += sum(1 for f in files if f.endswith('.csv'))
        total += n
        if n != expected:
            problems.append(f'{rel} 实际 {n}，期望 {expected}')
    if total != 387:
        problems.append(f'合计 {total}，期望 387')
    detail = f'合计 {total} 个 CSV'
    if problems:
        detail += '；' + '；'.join(problems)
    return not problems, detail


# ---------------------------------------------------------------- 3. 代码事实

def test_code_facts():
    """文档中陈述的关键代码事实必须与实现一致。"""
    problems = []
    model = read(os.path.join(BASE, 'Model', 'Model.py'))
    util = read(os.path.join(BASE, 'utils', 'util.py'))
    split = read(os.path.join(BASE, 'utils', 'split.py'))

    facts = [
        ('Model.py 含 class Solution_u', 'class Solution_u' in model),
        ('Model.py 含 dynamical_F 且输入 35 维', 'self.dynamical_F = MLP(input_dim=35' in model
         or 'dynamical_F = MLP(input_dim=35' in model),
        ('Model.py 含 Sin 激活类', 'class Sin(' in model),
        ('util.py 含 set_seed', 'def set_seed(' in util),
        ('util.py 的 write_to_txt 指定 utf-8',
         "encoding='utf-8'" in util or 'encoding="utf-8"' in util),
        ('split.py 含 uniform_split', 'def uniform_split(' in split),
    ]
    for name, ok in facts:
        if not ok:
            problems.append(name)

    # 四个 main 的 seed 默认值必须为 42（CHANGELOG 曾误写为 420）
    seed_re = re.compile(r"'--seed',\s*type=int,\s*default=(\d+)")
    for main in ('main_XJTU.py', 'main_HUST.py', 'main_MIT.py', 'main_TJU.py'):
        text = read(os.path.join(BASE, main))
        m = seed_re.search(text)
        if not m:
            problems.append(f'{main} 未找到 --seed 默认值')
        elif m.group(1) != '42':
            problems.append(f'{main} --seed 默认值为 {m.group(1)}，文档口径为 42')
        if 'BooleanOptionalAction' not in text:
            problems.append(f'{main} 的 --deterministic 未使用 BooleanOptionalAction')

    # 迁移学习入口必须存在且可导入（原文件名含空格）
    if not os.path.exists(os.path.join(BASE, 'main_finetune.py')):
        problems.append('缺少 main_finetune.py')
    else:
        ft = read(os.path.join(BASE, 'main_finetune.py'))
        if 'def run_standard_tasks(' not in ft:
            problems.append('main_finetune.py 缺少 run_standard_tasks()')

    return not problems, ('全部一致' if not problems else '；'.join(problems))


def test_uniform_split_unbiased():
    """uniform_split 不得让编号最小的电池恒定进入测试集（回归项）。"""
    sys.path.insert(0, BASE)
    from utils.split import uniform_split
    files = [f'battery-{i}.csv' for i in range(1, 21)]
    _, test = uniform_split(files, test_ratio=0.2)
    ids = sorted(int(os.path.basename(f).split('-')[-1].split('.')[0]) for f in test)
    return (bool(ids) and ids[0] != 1), f'测试电池编号 {ids}'


# ---------------------------------------------------------------- 4. 过时表述

# 这些符号在代码中从未存在过，只出现在历史文档里
STALE_SYMBOLS = ['SolutionNetwork', 'DynamicsNetwork',
                 'extract_features_from_charge', 'BatteryDataPreprocessor']


# 澄清语境标记：这些行是在说明"某物不存在/已删除"，属正当提及，予以豁免
EXEMPT_MARKERS = ('不存在', '已删除', '历史版本', '已更名', '已废弃',
                  '从未存在', '重命名', '原名')


def test_no_stale_symbols():
    """
    已删除/从未存在的类名与函数名不得出现在项目文档中。

    豁免两类：
    - `docs/CHANGELOG.md`：它按职责必须记录"删除了哪些符号"；
    - 含澄清语境的单行（如"这些类并不存在，已全部删除"）。
    """
    hits, lines_scanned = [], 0
    for rel in repo_markdown_files():
        if is_review_doc(rel) or os.path.basename(rel) == 'CHANGELOG.md':
            continue
        for lineno, line in enumerate(read(os.path.join(REPO, rel)).splitlines(), 1):
            lines_scanned += 1
            if any(m in line for m in EXEMPT_MARKERS):
                continue
            for sym in STALE_SYMBOLS:
                if sym in line:
                    hits.append(f'{rel}:{lineno}: {sym}')
    detail = f'扫描 {lines_scanned} 行'
    if hits:
        detail += '；命中 ' + '；'.join(sorted(set(hits))[:5])
    return not hits, detail


# ---------------------------------------------------------------- 5. 数字可溯源

def test_measured_numbers_traceable():
    """
    文档中声称的"本仓库实测 MAPE"必须能在 results/SUMMARY.md 中找到出处。

    只检查四个主实验数据集的实测值；论文报告值（0.78/0.65/0.78/1.21）
    不属于"实测"，由 06/README 显式标注来源。
    """
    summary_path = os.path.join(BASE, 'results', 'SUMMARY.md')
    if not os.path.exists(summary_path):
        return False, 'results/SUMMARY.md 不存在，请先运行 results_summary.py'
    summary = read(summary_path)

    targets = ['0.87', '0.80', '0.73', '1.13']
    missing = [t for t in targets if t not in summary]
    return not missing, (f'SUMMARY.md 覆盖 {len(targets) - len(missing)}/{len(targets)} 项'
                         + ('' if not missing else f'，缺少 {missing}'))


# ---------------------------------------------------------------- 主流程

def main():
    print('=' * 74)
    print('文档 / 代码一致性校验')
    print('=' * 74)
    print(f'仓库根：{REPO}\n')

    print('--- 文档引用的脚本 ---')
    ok, detail = test_referenced_scripts_exist()
    record('文档中引用的 .py 均真实存在', ok, detail)

    print('\n--- 数据集规模 ---')
    ok, detail = test_dataset_counts()
    record('data/ 实际文件数与口径一致', ok, detail)

    print('\n--- 代码事实 ---')
    ok, detail = test_code_facts()
    record('关键类名 / 参数默认值 / 入口文件一致', ok, detail)
    ok, detail = test_uniform_split_unbiased()
    record('uniform_split 无"最小编号恒入测试集"偏置', ok, detail)

    print('\n--- 过时表述 ---')
    ok, detail = test_no_stale_symbols()
    record('文档中无已废弃的类名 / 函数名', ok, detail)

    print('\n--- 数字可溯源 ---')
    ok, detail = test_measured_numbers_traceable()
    record('实测 MAPE 可在 results/SUMMARY.md 溯源', ok, detail)

    print('\n' + '=' * 74)
    passed = sum(results)
    total = len(results)
    print(f'结果: {passed}/{total} 通过')
    print('全部通过。' if passed == total else '存在失败项，请检查上面标记为 [FAIL] 的条目。')
    print('=' * 74)
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
