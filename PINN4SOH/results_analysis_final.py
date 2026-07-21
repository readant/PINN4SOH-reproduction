"""
最终对比分析：我们的复现结果 vs 论文报告结果
"""
import numpy as np
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def eval_metrics(pred, true):
    pred = np.array(pred).reshape(-1)
    true = np.array(true).reshape(-1)
    mae = np.mean(np.abs(pred - true))
    mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return mae, mape, rmse


def analyze_dataset(dataset_name, batch_dir, n_exp=10):
    """分析一个batch的所有实验结果"""
    all_mape = []
    all_mae = []
    all_rmse = []

    for exp in range(1, n_exp + 1):
        sub = os.path.join(batch_dir, f'Experiment{exp}')
        pred_file = os.path.join(sub, 'pred_label.npy')
        true_file = os.path.join(sub, 'true_label.npy')
        if os.path.exists(pred_file) and os.path.exists(true_file):
            pred = np.load(pred_file)
            true = np.load(true_file)
            mae, mape, rmse = eval_metrics(pred, true)
            all_mape.append(mape)
            all_mae.append(mae)
            all_rmse.append(rmse)

    if not all_mape:
        return None
    return {
        'MAPE': (np.mean(all_mape), np.std(all_mape)),
        'MAE': (np.mean(all_mae), np.std(all_mae)),
        'RMSE': (np.mean(all_rmse), np.std(all_rmse)),
        'n_samples': len(all_mape),
    }


def analyze_xjtu_all():
    """XJTU 所有 6 个 batch"""
    result_dir = os.path.join(BASE, 'results/XJTU results')
    batch_names = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    print("=" * 70)
    print("XJTU 数据集复现结果（PINN vs 论文）")
    print("=" * 70)
    total_mape = []
    for i in range(6):
        sub = os.path.join(result_dir, f'{i}-{i}')
        if not os.path.exists(sub):
            print(f"  batch {i} ({batch_names[i]}): 无结果")
            continue
        result = analyze_dataset(f'XJTU batch {i}', sub, n_exp=10)
        if result:
            print(f"  Batch {i} ({batch_names[i]}): MAPE={result['MAPE'][0]:.2f}% +/- {result['MAPE'][1]:.2f}%, "
                  f"MAE={result['MAE'][0]:.4f}, RMSE={result['RMSE'][0]:.4f}  ({result['n_samples']} exp)")
            total_mape.append(result['MAPE'][0])
    if total_mape:
        print(f"  平均 MAPE: {np.mean(total_mape):.2f}% +/- {np.std(total_mape):.2f}%")


def analyze_other_datasets():
    """HUST / MIT / TJU"""
    for name, folder in [
        ('HUST', 'HUST results'),
        ('MIT', 'MIT results'),
    ]:
        result_dir = os.path.join(BASE, f'results/{folder}')
        print(f"\n{'=' * 70}")
        print(f"{name} 数据集复现结果")
        print(f"{'=' * 70}")
        result = analyze_dataset(name, result_dir, n_exp=10)
        if result:
            print(f"  MAPE={result['MAPE'][0]:.2f}% +/- {result['MAPE'][1]:.2f}%, "
                  f"MAE={result['MAE'][0]:.4f}, RMSE={result['RMSE'][0]:.4f}  ({result['n_samples']} exp)")

    # TJU 有 batch 子目录
    name = 'TJU'
    folder = 'TJU results'
    result_dir = os.path.join(BASE, f'results/{folder}')
    print(f"\n{'=' * 70}")
    print(f"{name} 数据集复现结果")
    print(f"{'=' * 70}")
    all_mape = []
    for b in range(3):
        batch_dir = os.path.join(result_dir, f'{b}-{b}')
        if not os.path.exists(batch_dir):
            continue
        result = analyze_dataset(f'{name} batch {b}', batch_dir, n_exp=10)
        if result:
            print(f"  Batch {b}: MAPE={result['MAPE'][0]:.2f}% +/- {result['MAPE'][1]:.2f}%, "
                  f"MAE={result['MAE'][0]:.4f}, RMSE={result['RMSE'][0]:.4f}  ({result['n_samples']} exp)")
            all_mape.append(result['MAPE'][0])
    if all_mape:
        print(f"  平均 MAPE: {np.mean(all_mape):.2f}% +/- {np.std(all_mape):.2f}%")


def analyze_comparison():
    """MLP / CNN 对比"""
    for model_name in ['XJTU-MLP results', 'XJTU-CNN results']:
        result_dir = os.path.join(BASE, f'results/{model_name}')
        print(f"\n{'=' * 70}")
        print(f"{model_name.replace(' results', '')} 对比结果")
        print(f"{'=' * 70}")
        total_mape = []
        for i in range(6):
            batch_dir = os.path.join(result_dir, f'{i}-{i}')
            if not os.path.exists(batch_dir):
                continue
            result = analyze_dataset(f'batch {i}', batch_dir, n_exp=10)
            if result:
                print(f"  Batch {i}: MAPE={result['MAPE'][0]:.2f}% +/- {result['MAPE'][1]:.2f}%, "
                      f"RMSE={result['RMSE'][0]:.4f}  ({result['n_samples']} exp)")
                total_mape.append(result['MAPE'][0])
        if total_mape:
            print(f"  平均 MAPE: {np.mean(total_mape):.2f}% +/- {np.std(total_mape):.2f}%")


def print_paper_comparison():
    """打印论文 Table 2 对比"""
    print(f"\n{'=' * 70}")
    print("论文 Table 2 对比")
    print(f"{'=' * 70}")
    paper_xjtu_pinn = [0.66, 0.81, 0.65, 0.64, 0.83, 0.90]  # approximate from paper
    paper_xjtu_mlp  = [0.65, 1.24, 0.78, 0.84, 1.20, 0.99]
    paper_xjtu_cnn  = [0.76, 1.24, 0.84, 0.86, 1.23, 1.05]

    batch_names = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    print(f"\n{'Batch':<10} {'PINN(ours)':<15} {'MLP(ours)':<15} {'CNN(ours)':<15}")
    print("-" * 55)

    # Get ours
    pinn_mapes = []
    for i in range(6):
        result_dir = os.path.join(BASE, f'results/XJTU results/{i}-{i}')
        result = analyze_dataset('', result_dir, n_exp=10)
        pinn_mapes.append(result['MAPE'][0] if result else 0)

    mlp_mapes = []
    for i in range(6):
        result_dir = os.path.join(BASE, f'results/XJTU-MLP results/{i}-{i}')
        result = analyze_dataset('', result_dir, n_exp=10)
        mlp_mapes.append(result['MAPE'][0] if result else 0)

    cnn_mapes = []
    for i in range(6):
        result_dir = os.path.join(BASE, f'results/XJTU-CNN results/{i}-{i}')
        result = analyze_dataset('', result_dir, n_exp=10)
        cnn_mapes.append(result['MAPE'][0] if result else 0)

    for i in range(6):
        print(f"{batch_names[i]:<10} {pinn_mapes[i]:<15.2f} {mlp_mapes[i]:<15.2f} {cnn_mapes[i]:<15.2f}")
    print(f"{'平均':<10} {np.mean(pinn_mapes):<15.2f} {np.mean(mlp_mapes):<15.2f} {np.mean(cnn_mapes):<15.2f}")


def print_transfer_results():
    """迁移学习结果"""
    print(f"\n{'=' * 70}")
    print("迁移学习结果 (TJU → XJTU)")
    print(f"{'=' * 70}")
    ft_dir = os.path.join(BASE, 'results_fine-tuning/TJU-XJTU')
    total_before = []
    total_after = []
    for batch in range(6):
        batch_dir = os.path.join(ft_dir, f'batch{batch}')
        if not os.path.exists(batch_dir):
            continue
        exp_results = []
        for exp in range(10):
            log_file = os.path.join(batch_dir, f'Experiment{exp}', 'logging.txt')
            if not os.path.exists(log_file):
                continue
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Source only:' in line and 'MAPE' in line:
                        mape_str = line.split('MAPE:')[1].split(',')[0].strip()
                        total_before.append(float(mape_str) * 100)  # 日志中MAPE是小数形式，乘100转为百分比
                        break
            pred_file = os.path.join(batch_dir, f'Experiment{exp}', 'pred_label.npy')
            true_file = os.path.join(batch_dir, f'Experiment{exp}', 'true_label.npy')
            if os.path.exists(pred_file):
                pred = np.load(pred_file)
                true = np.load(true_file)
                _, mape, _ = eval_metrics(pred, true)
                total_after.append(mape)
        print(f"  Batch {batch}: Source Only MAPE={np.mean(total_before):.2f}% → Fine-tune MAPE={np.mean(total_after):.2f}%")

    if total_before and total_after:
        print(f"\n  总体: Source Only MAPE={np.mean(total_before):.2f}% → Fine-tune MAPE={np.mean(total_after):.2f}%")
        print(f"  降幅: {(np.mean(total_before)-np.mean(total_after))/np.mean(total_before)*100:.0f}%")


if __name__ == '__main__':
    analyze_xjtu_all()
    analyze_other_datasets()
    analyze_comparison()
    print_paper_comparison()
    print_transfer_results()
