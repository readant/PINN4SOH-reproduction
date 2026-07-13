'''
[新增] 迁移学习结果可视化 — 论文 Figure 6
绘制不同跨数据集迁移任务中 Source Only 与 Fine-tuning 方法的 MAPE 对比，
展示物理约束动力学网络 G(Theta) 在跨电池知识迁移中的效果。

数据来源: results_fine-tuning/ 目录下 FineTune results.py 生成的 Excel 文件
依赖: scienceplots, seaborn, pandas, matplotlib
'''

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])
import pandas as pd
import numpy as np
import seaborn as sns
import os

# ---------------------------------------------------------
# 配置区：指定要分析的迁移任务
# ---------------------------------------------------------
# 迁移任务格式: (source, target, 类型标签)
# "同族" = 化学体系相近 (NCM<->NCA, LFP<->LFP)
# "异族" = 化学体系不同 (NCM<->LFP)
transfer_tasks = [
    ('XJTU', 'TJU',  '同族 NCM↔NCA'),
    ('TJU',  'XJTU', '同族 NCM↔NCA'),
    ('HUST', 'MIT',  '同族 LFP↔LFP'),
    ('MIT',  'HUST', '同族 LFP↔LFP'),
    # 异族迁移也可加入（如 XJTU↔HUST、HUST↔XJTU），
    # 需先在 main_adaptation - fine-tuning.py 中运行对应的 FineTune_*() 函数
]

RESULTS_DIR = '../results_fine-tuning'
ANALYSIS_DIR = '../results analysis/processed results (fine tuning)'

# ---------------------------------------------------------
# 收集各任务的 MAPE 数据
# ---------------------------------------------------------
all_data = []  # 每行: {task, method, MAPE}

for source, target, label in transfer_tasks:
    excel_path = os.path.join(ANALYSIS_DIR, f'Ours-FineTune-{source}-{target}.xlsx')

    if not os.path.exists(excel_path):
        print(f"[跳过] 缺少预处理结果文件: {excel_path}")
        print(f"  请先运行 FineTune results.py 生成此文件")
        continue

    for batch in range(3):
        try:
            # 读取 per-battery 结果（多次实验平均值）
            df = pd.read_excel(excel_path, sheet_name=f'battery_mean_{batch}', engine='openpyxl')
            if 'MAPE' not in df.columns:
                continue
            mape_vals = df['MAPE'].values * 100  # 转换为百分比
            for v in mape_vals:
                if not np.isnan(v):
                    all_data.append({'任务': f'{source}→{target}\n{label}',
                                     'MAPE (%)': v,
                                     '方法': 'Fine-tuning'})
        except ValueError:
            continue

    # 读取 Source Only 结果（适配前）
    # Source Only 数据保存在同一 Excel 的 source_only 表中
    try:
        for batch in range(3):
            df_src = pd.read_excel(excel_path, sheet_name=f'source_only_{batch}', engine='openpyxl')
            if 'mape' in df_src.columns:
                for _, row in df_src.iterrows():
                    all_data.append({'任务': f'{source}→{target}\n{label}',
                                     'MAPE (%)': row['mape'] * 100,
                                     '方法': 'Source Only'})
    except (ValueError, FileNotFoundError):
        # 如果没有 source_only 表，尝试从日志目录直接读取
        sim_root = os.path.join(RESULTS_DIR, f'{source}-{target}')
        if os.path.exists(sim_root):
            for exp_dir in os.listdir(sim_root):
                exp_path = os.path.join(sim_root, exp_dir)
                log_path = os.path.join(exp_path, 'logging.txt')
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        for line in f:
                            if 'Source only:' in line:
                                parts = line.split('MAPE:')[1].split(',')[0].strip()
                                try:
                                    mape = float(parts)
                                    all_data.append({'任务': f'{source}→{target}\n{label}',
                                                     'MAPE (%)': mape * 100,
                                                     '方法': 'Source Only'})
                                except ValueError:
                                    pass

df_plot = pd.DataFrame(all_data)

if df_plot.empty:
    print("未找到任何迁移学习结果数据。请先运行迁移学习实验和分析脚本。")
    exit(1)

# ---------------------------------------------------------
# 绘图
# ---------------------------------------------------------
# 为 方法 分配配色
method_colors = {
    'Source Only': '#E88B74',   # 橙色/红色
    'Fine-tuning': '#6BAED6',   # 蓝色
}

unique_tasks = df_plot['任务'].unique()
unique_methods = df_plot['方法'].unique()

fig, ax = plt.subplots(figsize=(len(unique_tasks) * 1.6, 3.5), dpi=200)

sns.violinplot(
    x='任务', y='MAPE (%)', hue='方法', data=df_plot,
    palette=method_colors,
    scale='width',
    inner='quartile',
    dodge=True,
    saturation=0.9,
    linewidth=0.5,
    ax=ax
)

# 标注均值
mean_df = df_plot.groupby(['任务', '方法'])['MAPE (%)'].mean().reset_index()
for task_i, task in enumerate(unique_tasks):
    for method_j, method in enumerate(unique_methods):
        subset = mean_df[(mean_df['任务'] == task) & (mean_df['方法'] == method)]
        if subset.empty:
            continue
        mean_val = subset['MAPE (%)'].values[0]
        # 小提琴图的 x 位置偏移
        n_methods = len(unique_methods)
        offset = (method_j - (n_methods - 1) / 2) * 0.2
        x_pos = task_i + offset
        ax.plot([x_pos - 0.06, x_pos + 0.06], [mean_val, mean_val],
                color='black', linewidth=1.2)

plt.xlabel(None)
plt.ylabel('MAPE (%)')
plt.title('Transfer Learning: Source Only vs Fine-tuning')
plt.legend(loc='upper right', frameon=False, fontsize=8)
plt.tight_layout()

# 保存
save_path = 'Figure 6 - Transfer Learning.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[保存] {save_path}")
plt.show()
print('Done.')
