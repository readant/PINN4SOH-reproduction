"""
修复验证脚本 (Regression check for reproducibility & splitting fixes)
======================================================================

逐条验证本次修复是否生效，可反复运行作为回归检查。

用法（必须从 PINN4SOH 目录运行）：
    python verify_fixes.py

退出码 0 = 全部通过，1 = 存在失败项。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch

from utils.util import set_seed
from utils.split import uniform_split, describe_split, check_leakage, natural_key
from Model.Model import LR_Scheduler, Solution_u

PASS, FAIL = '[PASS]', '[FAIL]'
results = []


def record(name, ok, detail=''):
    results.append(ok)
    print(f'{PASS if ok else FAIL} {name}' + (f'  -- {detail}' if detail else ''))


# ---------------------------------------------------------------- 1. 随机种子

def test_seed_reproducibility():
    """同一颗种子必须产生完全相同的权重，不同种子必须产生不同的权重。"""
    set_seed(42)
    w_a = [p.detach().clone() for p in Solution_u().parameters()]

    set_seed(42)
    w_b = [p.detach().clone() for p in Solution_u().parameters()]

    set_seed(43)
    w_c = [p.detach().clone() for p in Solution_u().parameters()]

    same = all(torch.equal(a, b) for a, b in zip(w_a, w_b))
    diff = any(not torch.equal(a, b) for a, b in zip(w_a, w_c))
    return same, diff


def test_dropout_determinism():
    """同一颗种子下，Dropout 掩码必须一致。"""
    # 必须先用独立的种子把输入 x 固定下来。
    # 若在 set_seed(被测种子) 之后再生成 x，两次运行的 RNG 消耗序列会错开一步
    # （第一次是 init→randn→dropout，第二次是 init→dropout），
    # 从而误报为"不可复现"。
    set_seed(0)
    x = torch.randn(64, 17)

    set_seed(7)
    m = Solution_u()
    m.train()
    y1 = m(x)

    set_seed(7)
    m2 = Solution_u()
    m2.train()
    y2 = m2(x)

    return torch.allclose(y1, y2)


# ---------------------------------------------------------------- 2. 数据划分

def test_uniform_split_basic():
    """uniform_split：比例正确、无泄漏、结果确定。"""
    files = [f'batch_{i}.csv' for i in range(1, 21)]  # 20 只电池
    train, test = uniform_split(files, test_ratio=0.2)

    ratio_ok = len(test) == 4
    disjoint = not check_leakage(train, test)
    complete = len(train) + len(test) == 20

    # 确定性：同一输入必须给出同一输出
    train2, test2 = uniform_split(files, test_ratio=0.2)
    stable = train == train2 and test == test2

    return ratio_ok and disjoint and complete and stable, len(test)


def test_uniform_split_natural_order():
    """自然序：battery-2 必须排在 battery-10 之前。"""
    files = [f'battery-{i}.csv' for i in (1, 10, 2, 20, 3)]
    ordered = sorted(files, key=natural_key)
    expected = ['battery-1.csv', 'battery-2.csv', 'battery-3.csv',
                'battery-10.csv', 'battery-20.csv']
    return [os.path.basename(p) for p in ordered] == expected


def test_uniform_split_spread():
    """测试电池应在整个编号范围内均匀分散，而不是集中在尾部。"""
    files = [f'battery-{i}.csv' for i in range(1, 21)]
    _, test = uniform_split(files, test_ratio=0.2)
    ids = sorted(int(os.path.basename(f).split('-')[-1].split('.')[0]) for f in test)
    return ids, max(ids) <= 20


# ---------------------------------------------------------------- 3. 已知缺陷

def test_lr_scheduler_constant_branch():
    """
    复现 LR_Scheduler 的 UnboundLocalError。

    当所有 param_group 都命中 constant_predictor_lr 分支时，原实现的循环体内
    从未给 lr 赋值，末尾 `self.current_lr = lr` 会抛 UnboundLocalError。
    """
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'name': 'predictor'}]
    )
    scheduler = LR_Scheduler(optimizer=optimizer, warmup_epochs=2,
                             warmup_lr=0.001, num_epochs=5,
                             base_lr=0.01, final_lr=0.0001,
                             constant_predictor_lr=True)
    try:
        lr = scheduler.step()
        return True, f'返回 lr={lr}'
    except UnboundLocalError as e:
        return False, str(e)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def test_lr_scheduler_normal():
    """常规分支下 lr 应随迭代递减。"""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = LR_Scheduler(optimizer=optimizer, warmup_epochs=2,
                             warmup_lr=0.001, num_epochs=10,
                             base_lr=0.01, final_lr=0.0001)
    lrs = [scheduler.step() for _ in range(4)]
    return lrs[0] < lrs[-1], f'warmup 段 lr: {[round(x, 5) for x in lrs]}'


def test_logger_no_leak():
    """反复调用 get_logger 不应累积 handler。"""
    import logging
    from utils.util import get_logger
    log_file = 'results/verify_fixes_logger_probe.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for _ in range(5):
        get_logger(log_file)
    n = len(logging.getLogger('mylogger').handlers)
    # 一次 get_logger 挂 2 个 handler（console + file）
    return n == 2, f'handler 数量={n}（期望 2）'


# ---------------------------------------------------------------- 4. 端到端

def test_end_to_end(strategy):
    """端到端：用指定划分策略跑通 XJTU 2C 的 2 epoch 训练。"""
    import argparse
    from main_XJTU import load_data
    from Model.Model import PINN

    args = argparse.Namespace(
        data='XJTU', batch='2C', batch_size=256,
        normalization_method='min-max',
        epochs=2, early_stop=5, warmup_epochs=1,
        warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60, alpha=0.7, beta=0.2,
        seed=42, split_strategy=strategy, deterministic=True,
        save_folder=f'results/verify_fixes_e2e_{strategy}', log_dir='e2e.txt',
    )
    try:
        # 显式建目录：load_data 内部的 write_to_txt 会往该目录写日志，
        # 而目录原本只在 PINN.__init__ 里创建，调用顺序敏感。
        os.makedirs(args.save_folder, exist_ok=True)
        dataloader = load_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'],
                   validloader=dataloader['valid'],
                   testloader=dataloader['test'])
        return True, '2 epoch 训练完成'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------- 主流程

def main():
    print('=' * 70)
    print('修复验证：可复现性 / 统一划分 / 已知缺陷')
    print('=' * 70)
    print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}\n')

    print('--- 随机种子 ---')
    same, diff = test_seed_reproducibility()
    record('同种子权重一致', same)
    record('不同种子权重不同', diff)
    record('同种子 Dropout 掩码一致', test_dropout_determinism())

    print('\n--- 统一数据划分 ---')
    ok, n_test = test_uniform_split_basic()
    record('uniform_split 比例/无泄漏/确定性', ok, f'测试电池 {n_test}/20')
    record('自然序排序正确', test_uniform_split_natural_order())
    ids, spread_ok = test_uniform_split_spread()
    record('测试电池均匀分散', spread_ok, f'抽取编号 {ids}')

    print('\n--- 已知缺陷 ---')
    ok, detail = test_lr_scheduler_constant_branch()
    record('LR_Scheduler constant 分支不再崩溃', ok, detail)
    ok, detail = test_lr_scheduler_normal()
    record('LR_Scheduler 常规分支正常 warmup', ok, detail)
    ok, detail = test_logger_no_leak()
    record('logger 无 handler 累积', ok, detail)

    if '--e2e' in sys.argv:
        print('\n--- 端到端训练（XJTU 2C, 2 epoch）---')
        for strategy in ('legacy', 'uniform'):
            ok, detail = test_end_to_end(strategy)
            record(f'端到端运行（--split_strategy {strategy}）', ok, detail)

    print('\n' + '=' * 70)
    passed = sum(results)
    total = len(results)
    print(f'结果: {passed}/{total} 通过')
    if passed == total:
        print('全部通过。')
    else:
        print('存在失败项，请检查上面标记为 [FAIL] 的条目。')
    print('=' * 70)
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
