"""
实验结果统一汇总（本仓库唯一数据源）
=====================================

从 `results/` 与 `results_fine-tuning/` 的原始产物（`true_label.npy` /
`pred_label.npy`）一键汇总全部指标，输出：

    results/SUMMARY.md    人类可读表格 —— 仓库内所有文档引用本文件
    results/summary.csv   机器可读明细

约定
----
1. 逐个扫描 Experiment 目录（**不写死编号**），只统计两个 npy 同时存在的目录；
2. MAPE 一律以**百分比**输出（×100），与论文口径一致，避免"0.0061 vs 0.61%"的歧义；
3. 产物缺失的分组明确标注"无产物"，**不估计、不填充、不引用他处数字**。

与 `results analysis/processed results/*.xlsx` 的区别
-----------------------------------------------------
后者是**原作者随仓库提供**的处理结果（对应论文数据），不是本仓库跑出来的。
两者不可混用：本文件处理的是本仓库 `results/` 下的真实产物。

用法（从 PINN4SOH 目录运行）：
    python results_summary.py
"""

import csv
import glob
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# 论文报告的 MAPE（%），仅用于并列对照，不参与任何计算
PAPER_MAPE = {'XJTU': 0.78, 'HUST': 0.65, 'MIT': 0.78, 'TJU': 1.21}
PAPER_AVG = 0.87

GROUPS = [
    ('主实验（PINN）', [
        ('XJTU', 'results/XJTU results/*/Experiment*'),
        ('HUST', 'results/HUST results/Experiment*'),
        ('MIT',  'results/MIT results/Experiment*'),
        ('TJU',  'results/TJU results/*/Experiment*'),
    ]),
    ('基线模型（仅 XJTU）', [
        ('XJTU-MLP', 'results/XJTU-MLP results/*/Experiment*'),
        ('XJTU-CNN', 'results/XJTU-CNN results/*/Experiment*'),
    ]),
    ('迁移学习（fine-tuning）', [
        ('TJU->XJTU', 'results_fine-tuning/TJU-XJTU/*/Experiment*'),
        ('XJTU->TJU', 'results_fine-tuning/XJTU-TJU/*/Experiment*'),
        ('HUST->MIT', 'results_fine-tuning/HUST-MIT/Experiment*'),
        ('MIT->HUST', 'results_fine-tuning/MIT-HUST/Experiment*'),
    ]),
]

XJTU_BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']


def _r2(pred, true):
    """决定系数（单输出回归，等价于 sklearn.metrics.r2_score）。"""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _metrics(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    mae = float(np.mean(np.abs(pred - true)))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return mae, mape, rmse, _r2(pred, true)


def collect(pattern):
    """按 glob 收集所有含预测/标签两份 npy 的实验目录。"""
    dirs = []
    for d in sorted(glob.glob(os.path.join(BASE, pattern))):
        if (os.path.exists(os.path.join(d, 'true_label.npy'))
                and os.path.exists(os.path.join(d, 'pred_label.npy'))):
            dirs.append(d)
    return dirs


def summarize(label, pattern):
    dirs = collect(pattern)
    if not dirs:
        return {'name': label, 'n': 0, 'mape': None, 'mape_std': None,
                'mae': None, 'rmse': None, 'r2': None}
    rows = []
    for d in dirs:
        pred = np.load(os.path.join(d, 'pred_label.npy'))
        true = np.load(os.path.join(d, 'true_label.npy'))
        rows.append(_metrics(pred, true))
    a = np.array(rows)
    return {'name': label, 'n': len(dirs),
            'mape': float(a[:, 1].mean()), 'mape_std': float(a[:, 1].std()),
            'mae': float(a[:, 0].mean()), 'rmse': float(a[:, 2].mean()),
            'r2': float(np.nanmean(a[:, 3]))}


def fmt(row, paper=None):
    if row['n'] == 0:
        return f"| {row['name']} | 无产物 | - | - | - | - |"
    dev = ''
    if paper is not None:
        dev = f" | {row['mape'] - paper:+.2f} |"
    return (f"| {row['name']} | {row['n']} | **{row['mape']:.2f} ± {row['mape_std']:.2f}** | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |{dev}")


def main():
    print('=' * 78)
    print('本仓库实验结果汇总（数据源：results/ 与 results_fine-tuning/ 原始产物）')
    print('=' * 78)

    lines = [
        '# 实验结果汇总（本仓库实测）',
        '',
        '> 本文件由 `results_summary.py` 从 `results/` 与 `results_fine-tuning/` 的原始产物',
        '> 自动生成，是仓库内**唯一**的实验数据源。',
        '> 其它文档（README、docs/*、beginner-guide/*）引用实验数字时一律以本文件为准。',
        '',
        '**注意**：`results analysis/processed results/*.xlsx` 是**原作者随仓库提供**的处理结果，',
        '对应论文发表的数值，不属于本仓库的实测结果，两者不可混用。',
        '',
        '口径：MAPE 为百分比（%），逐实验计算后取均值±标准差；MAE/RMSE/R² 为逐实验均值的均值。',
        '',
    ]
    csv_rows = [['group', 'name', 'n_experiments', 'mape_mean', 'mape_std', 'mae', 'rmse', 'r2']]

    for group_name, items in GROUPS:
        print(f'\n--- {group_name} ---')
        if group_name.startswith('主实验'):
            lines += [f'## {group_name}', '',
                      '| 数据集 | 实验次数 | MAPE (%) | MAE | RMSE | R² | 与论文偏差 (pp) |',
                      '|--------|---------|----------|-----|------|----|----------------|']
        else:
            lines += [f'## {group_name}', '',
                      '| 组合 | 实验次数 | MAPE (%) | MAE | RMSE | R² |',
                      '|------|---------|----------|-----|------|----|']
        main_mapes = []
        for label, pattern in items:
            row = summarize(label, pattern)
            paper = PAPER_MAPE.get(label) if group_name.startswith('主实验') else None
            if group_name.startswith('主实验') and row['n']:
                main_mapes.append(row['mape'])
            lines.append(fmt(row, paper))
            csv_rows.append([group_name, row['name'], row['n'],
                             '' if row['mape'] is None else round(row['mape'], 4),
                             '' if row['mape_std'] is None else round(row['mape_std'], 4),
                             '' if row['mae'] is None else round(row['mae'], 6),
                             '' if row['rmse'] is None else round(row['rmse'], 6),
                             '' if row['r2'] is None else round(row['r2'], 4)])
            if row['n']:
                print(f"  {row['name']:10s} n={row['n']:3d}  MAPE={row['mape']:.2f}±{row['mape_std']:.2f}%"
                      f"  MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  R2={row['r2']:.4f}")
            else:
                print(f"  {row['name']:10s} 无产物")
        if group_name.startswith('主实验') and main_mapes:
            avg = float(np.mean(main_mapes))
            lines += ['', f'**四数据集平均 MAPE：{avg:.2f}%**（论文报告 {PAPER_AVG}%，'
                          f'偏差 {avg - PAPER_AVG:+.2f} pp）', '']
            csv_rows.append([group_name, 'AVERAGE', '', round(avg, 4), '', '', '', ''])

    # XJTU 逐 batch 明细
    lines += ['## XJTU 逐充放电协议明细', '',
              '| Batch | 协议 | 实验次数 | MAPE (%) |',
              '|-------|------|---------|----------|']
    print('\n--- XJTU 逐 batch ---')
    for i, name in enumerate(XJTU_BATCHES):
        row = summarize(name, f'results/XJTU results/{i}-{i}/Experiment*')
        cell = f"{row['mape']:.2f} ± {row['mape_std']:.2f}" if row['n'] else '无产物'
        lines.append(f'| {i + 1} | {name} | {row["n"]} | {cell} |')
        csv_rows.append(['XJTU-batch', f'{i + 1}-{name}', row['n'],
                         '' if row['mape'] is None else round(row['mape'], 4),
                         '' if row['mape_std'] is None else round(row['mape_std'], 4), '', '', ''])
        print(f"  batch{i + 1} {name:10s} n={row['n']:3d}  MAPE={cell}")

    lines += ['', '---', '',
              f'生成方式：`cd PINN4SOH && python results_summary.py`（脚本路径：`PINN4SOH/results_summary.py`）', '']

    md_path = os.path.join(BASE, 'results', 'SUMMARY.md')
    csv_path = os.path.join(BASE, 'results', 'summary.csv')
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print('\n' + '=' * 78)
    print(f'已写出：{os.path.relpath(md_path, BASE)}')
    print(f'已写出：{os.path.relpath(csv_path, BASE)}')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
