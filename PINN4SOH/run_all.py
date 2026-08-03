"""
一键实验入口（本仓库统一调度）
==============================
原作者将各实验入口分散在独立的 main_*.py 中且多数被注释掉，本脚本统一调度。

用法（必须先 cd 到 PINN4SOH）:
    python run_all.py xjtu              # XJTU 主实验（6 batch x 10 次）
    python run_all.py hust              # HUST 主实验（10 次）
    python run_all.py mit               # MIT 主实验（10 次）
    python run_all.py tju               # TJU 主实验（3 batch x 10 次）
    python run_all.py comparison        # MLP/CNN 对比（XJTU）
    python run_all.py finetune          # 迁移学习（标准四组，各 10 次）
    python run_all.py small             # 小样本实验（XJTU）
    python run_all.py analyze           # 汇总实验结果 -> results/SUMMARY.md
    python run_all.py analyze-original  # （可选）原作者分析脚本，需先解压 results/Ours/*.zip
    python run_all.py all               # 全部跑（耗时很长，含约 400 次训练）

依赖环境: conda activate main
说明: 任一阶段失败都以非零退出码终止，不再静默跳过。
"""
import argparse
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_script(filename, cwd=None):
    """
    通过 subprocess 运行独立 Python 脚本。

    [修复] 原实现不检查返回码，脚本失败后调用方仍会打印"完成"。
    现改为 check=True：非零退出码抛出 CalledProcessError，失败显式暴露。

    :param cwd: 子进程工作目录。`results analysis/*.py` 用 '../results/...'
                这类相对路径，必须在自己的目录下运行。
    """
    script = os.path.join(BASE_DIR, filename)
    subprocess.run([sys.executable, script], cwd=cwd or BASE_DIR, check=True)


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
    迁移学习实验 —— 标准四组（TJU->XJTU / XJTU->TJU / HUST->MIT / MIT->HUST）。

    [修复] 原实现为空操作：入口文件名 `main_adaptation - fine-tuning.py` 含空格
    与连字符，无法 import，此分支只打印一条提示却照常返回，使用者会误以为已运行。
    该文件已重命名为 `main_finetune.py`，此处直接调用其 run_standard_tasks()。
    """
    print("=" * 60)
    print("Running fine-tuning experiments (4 standard tasks)...")
    from main_finetune import run_standard_tasks
    run_standard_tasks()


def run_small_sample():
    """小样本实验（XJTU）"""
    print("=" * 60)
    print("Running small sample experiments...")
    from main_XJTU import small_sample
    small_sample()


def run_analyze():
    """
    汇总本仓库实验结果，生成 `results/SUMMARY.md` 与 `results/summary.csv`。

    [修复] 原实现调用 `results analysis/*.py`，但这些脚本读取的是
    `../results/Ours/<数据集> results/`（原作者随仓库提供的结果包，需先解压、
    且目录名要匹配），以 PINN4SOH 为 cwd 时路径必然解析失败；同时子进程返回码
    未被检查，失败仍打印"分析完成"。
    现改为汇总本仓库 `results/` 下的真实产物（唯一数据源），
    处理作者提供的结果包请用 analyze-original。
    """
    print("=" * 60)
    print("Summarizing experiment results ...")
    _run_script('results_summary.py')


def run_analyze_original():
    """
    （可选）运行原作者提供的 `results analysis/*.py`，处理 `results/Ours/` 下的结果包。

    前置条件：`results/Ours/*.zip` 已解压，且解压后的目录名与脚本默认 root
    （`../results/Ours/<数据集> results/`）一致。本函数会在解压后检查该目录，
    不满足时给出明确提示并以非零状态结束，而不是静默成功。
    """
    print("=" * 60)
    print("Running original results analysis ...")

    analysis_dir = os.path.join(BASE_DIR, 'results analysis')
    ours_dir = os.path.join(BASE_DIR, 'results', 'Ours')
    _extract_ours_results(ours_dir)

    if not os.path.isdir(os.path.join(ours_dir, 'XJTU results')):
        print(f"  [失败] 未找到 {os.path.join(ours_dir, 'XJTU results')}。")
        print("         `results analysis/*.py` 默认读取 `../results/Ours/<数据集> results/`，")
        print("         而仓库随附的 XJTU.zip 解压后得到的是 `XJTU results (small sample N)/`，")
        print("         目录名不匹配。请先核对结果包内容，或直接使用 `python run_all.py analyze`")
        print("         汇总本仓库 results/ 下的实测产物。")
        return False

    scripts = ['XJTU results.py', 'HUST results.py', 'MIT results.py',
               'TJU results.py', 'Comparision results.py']
    failed = []
    for script in scripts:
        if not os.path.exists(os.path.join(analysis_dir, script)):
            print(f"  [跳过] 未找到 {script}")
            continue
        print(f"  Running {script} ...")
        try:
            _run_script(os.path.join('results analysis', script), cwd=analysis_dir)
        except subprocess.CalledProcessError as exc:
            failed.append(script)
            print(f"  [失败] {script} 退出码 {exc.returncode}")

    if failed:
        print(f"\n分析未全部完成，失败：{', '.join(failed)}")
        return False
    print("\n分析完成。结果保存在 results analysis/processed results/ 下的 .xlsx 文件中。")
    return True


def _extract_ours_results(ours_dir):
    """解压 results/Ours/*.zip（zip 未纳入版本控制，见 .gitignore）。"""
    import zipfile
    if not os.path.isdir(ours_dir):
        print(f"  [提示] 未找到 {os.path.relpath(ours_dir, BASE_DIR)}。")
        return
    for name in sorted(os.listdir(ours_dir)):
        if not name.lower().endswith('.zip'):
            continue
        print(f"  解压 {name} ...")
        try:
            with zipfile.ZipFile(os.path.join(ours_dir, name)) as zf:
                zf.extractall(ours_dir)
        except zipfile.BadZipFile:
            print(f"  [警告] {name} 不是有效的 zip，已跳过")


ALL_STAGES = {
    'xjtu': run_xjtu,
    'hust': run_hust,
    'mit': run_mit,
    'tju': run_tju,
    'comparison': run_comparison,
    'finetune': run_finetune,
    'small': run_small_sample,
    'analyze': run_analyze,
    'analyze-original': run_analyze_original,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN4SOH 实验统一入口')
    parser.add_argument('stage', nargs='?', default='all',
                        choices=['all'] + list(ALL_STAGES),
                        help='选择运行的实验阶段 (default: all)')
    args = parser.parse_args()

    if args.stage == 'all':
        stages = ['xjtu', 'hust', 'mit', 'tju', 'comparison', 'finetune', 'small', 'analyze']
        for stage in stages:
            print(f"\n{'#' * 60}")
            print(f"# Stage: {stage}")
            print(f"{'#' * 60}")
            if ALL_STAGES[stage]() is False:
                print(f"\n[中止] 阶段 {stage} 未成功完成，后续阶段不再执行。")
                raise SystemExit(1)
    else:
        if ALL_STAGES[args.stage]() is False:
            raise SystemExit(1)
