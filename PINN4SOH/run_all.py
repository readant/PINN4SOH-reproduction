"""
[新增] 论文《PINN for Battery SOH》一键复现入口。
原作者将各实验入口分散在独立的 main_*.py 文件中，且多数注释掉了。
本脚本统一控制实验流程，按需运行不同阶段。

用法:
    python run_all.py xjtu           # 只跑 XJTU 主实验
    python run_all.py hust           # 只跑 HUST 主实验
    python run_all.py mit            # 只跑 MIT 主实验
    python run_all.py tju            # 只跑 TJU 主实验
    python run_all.py comparison     # 只跑 MLP/CNN 对比（XJTU）
    python run_all.py finetune       # 只跑迁移学习
    python run_all.py small          # 只跑小样本实验（XJTU）
    python run_all.py analyze        # 只跑结果分析 + 绘图
    python run_all.py all            # 全部跑（耗时较长）

依赖环境: conda activate main
"""
import argparse
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_script(filename):
    """通过 subprocess 运行独立的 Python 脚本"""
    script = os.path.join(BASE_DIR, filename)
    subprocess.run([sys.executable, script], cwd=BASE_DIR)


def run_xjtu():
    """XJTU 数据集主实验（6 batch x 10 次 = 60 组，约数小时）"""
    print("=" * 60)
    print("Running XJTU main experiments...")
    from main_XJTU import main
    main()


def run_hust():
    """HUST 数据集主实验（10 次，约数小时）"""
    print("=" * 60)
    print("Running HUST main experiments...")
    from main_HUST import main
    main()


def run_mit():
    """MIT 数据集主实验（10 次）"""
    print("=" * 60)
    print("Running MIT main experiments...")
    from main_MIT import main
    main()


def run_tju():
    """TJU 数据集主实验（3 batch x 10 次 = 30 组）"""
    print("=" * 60)
    print("Running TJU main experiments...")
    from main_TJU import main
    main()


def run_comparison():
    """MLP/CNN 对比实验（XJTU, 6 batch x 2 model x 10 次）"""
    print("=" * 60)
    print("Running MLP/CNN comparison experiments...")
    _run_script('main_comparision.py')


def run_finetune():
    """
    [暂待运行] 迁移学习实验（4种跨数据集迁移，各 10 次）。
    原文件名含空格和连字符（main_adaptation - fine-tuning.py），无法直接 import。
    需先运行内部实验函数，可参考下方注释。
    """
    print("=" * 60)
    print("Running fine-tuning experiments...")
    print("提示: 迁移学习函数位于 main_adaptation - fine-tuning.py 中。")
    print("请手动取消该文件 __main__ 段中 FineTune_*() 的注释后单独运行。")

    # 原作者的文件名包含空格和连字符，Python 无法 import。
    # 如需通过本脚本运行，请先将文件重命名为 main_adaptation.py，
    # 然后取消下方注释：
    #
    # from main_adaptation import FineTune
    # FineTune()


def run_small_sample():
    """小样本实验（XJTU）"""
    print("=" * 60)
    print("Running small sample experiments...")
    from main_XJTU import small_sample
    small_sample()


def run_analyze():
    """运行结果分析，生成 Excel 表格"""
    print("=" * 60)
    print("Running results analysis...")
    analysis_scripts = [
        'results analysis/XJTU results.py',
        'results analysis/HUST results.py',
        'results analysis/MIT results.py',
        'results analysis/TJU results.py',
        'results analysis/Comparision results.py',
    ]
    for script in analysis_scripts:
        if os.path.exists(os.path.join(BASE_DIR, script)):
            print(f"  Running {script}...")
            _run_script(script)
    print("\n分析完成。结果保存在 results/ 目录下的 .xlsx 文件中。")


ALL_STAGES = {
    'xjtu': run_xjtu,
    'hust': run_hust,
    'mit': run_mit,
    'tju': run_tju,
    'comparison': run_comparison,
    'finetune': run_finetune,
    'small': run_small_sample,
    'analyze': run_analyze,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN4SOH 一键复现入口')
    parser.add_argument('stage', nargs='?', default='all',
                        choices=['all', 'xjtu', 'hust', 'mit', 'tju',
                                 'comparison', 'finetune', 'small', 'analyze'],
                        help='选择运行的实验阶段 (default: all)')
    args = parser.parse_args()

    if args.stage == 'all':
        stages = ['xjtu', 'hust', 'mit', 'tju', 'comparison', 'finetune', 'small', 'analyze']
        for stage in stages:
            print(f"\n{'#' * 60}")
            print(f"# Stage: {stage}")
            print(f"{'#' * 60}")
            ALL_STAGES[stage]()
    else:
        ALL_STAGES[args.stage]()
