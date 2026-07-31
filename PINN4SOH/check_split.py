"""
数据划分规则体检脚本 (一次性校验工具)
=====================================

对比 legacy（各数据集原有规则）与 uniform（utils/split.py 统一规则）的划分结果，
直观展示原实现的问题，并验证新规则的正确性。

用法（必须从 PINN4SOH 目录运行）：
    python check_split.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.split import uniform_split, describe_split, check_leakage


# ---------------------------------------------------------------- legacy 规则
# 以下函数逐行复刻四个 main_*.py 里的原始划分逻辑，未做任何修正

def xjtu_legacy(root, batch):
    train_list, test_list = [], []
    for f in os.listdir(root):
        if batch in f:
            # 原实现：对整个文件名做子串匹配
            if '4' in f or '8' in f:
                test_list.append(os.path.join(root, f))
            else:
                train_list.append(os.path.join(root, f))
    return train_list, test_list


def hust_legacy(root):
    test_id = ['1-4', '1-8', '2-4', '2-8',
               '3-4', '3-8', '4-4', '4-8',
               '5-4', '5-7', '6-4', '6-8',
               '7-4', '7-8', '8-4', '8-8',
               '9-4', '9-8', '10-4', '10-8']
    train_list, test_list = [], []
    for f in os.listdir(root):
        if f[:-4] in test_id:
            test_list.append(os.path.join(root, f))
        else:
            train_list.append(os.path.join(root, f))
    return train_list, test_list


def mit_legacy(root):
    train_list, test_list = [], []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        for f in os.listdir(batch_root):
            bid = int(f.split('-')[-1].split('.')[0])
            if bid % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    return train_list, test_list


def tju_legacy(root, batch_idx):
    mod = [(5, 9), (4, 8), (5, 9)]
    batchs = os.listdir(root)
    batch_root = os.path.join(root, batchs[batch_idx])
    files = os.listdir(batch_root)
    train_list, test_list = [], []
    for i, f in enumerate(files):
        bid = i + 1
        if bid % 10 == mod[batch_idx][0] or bid % 10 == mod[batch_idx][1]:
            test_list.append(os.path.join(batch_root, f))
        else:
            train_list.append(os.path.join(batch_root, f))
    return train_list, test_list


# ---------------------------------------------------------------- uniform 规则

def xjtu_uniform(root, batch):
    files = [os.path.join(root, f) for f in os.listdir(root) if batch in f]
    return uniform_split(files)


def hust_uniform(root):
    files = [os.path.join(root, f) for f in os.listdir(root)]
    return uniform_split(files)


def mit_uniform(root):
    files = []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files.extend(os.path.join(batch_root, f) for f in os.listdir(batch_root))
    return uniform_split(files)


def tju_uniform(root, batch_idx):
    batchs = os.listdir(root)
    batch_root = os.path.join(root, batchs[batch_idx])
    files = [os.path.join(batch_root, f) for f in os.listdir(batch_root)]
    return uniform_split(files)


# ---------------------------------------------------------------- 报告

def report(title, legacy, uniform):
    lt, lv = legacy
    ut, uv = uniform
    lr = len(lv) / (len(lt) + len(lv)) if (len(lt) + len(lv)) else 0
    ur = len(uv) / (len(ut) + len(uv)) if (len(ut) + len(uv)) else 0
    print(f'{title:<34} legacy 测试 {len(lv):>3} ({lr:>5.1%})   '
          f'uniform 测试 {len(uv):>3} ({ur:>5.1%})')
    leak = check_leakage(ut, uv)
    if leak:
        print(f'{"":<34} !! 泄漏: {sorted(leak)[:5]}')
    return lr, ur


def main():
    print('=' * 84)
    print('数据划分规则体检：legacy（原实现） vs uniform（统一规则）')
    print('=' * 84)

    xjtu_root = 'data/XJTU data'
    if os.path.isdir(xjtu_root):
        print('\n--- XJTU（按 batch 划分，理想测试占比 20%）---')
        for batch in ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']:
            legacy = xjtu_legacy(xjtu_root, batch)
            if not legacy[0] and not legacy[1]:
                continue
            report(f'XJTU / {batch}', legacy, xjtu_uniform(xjtu_root, batch))

    hust_root = 'data/HUST data'
    if os.path.isdir(hust_root):
        print('\n--- HUST（全局划分）---')
        report('HUST', hust_legacy(hust_root), hust_uniform(hust_root))

    mit_root = 'data/MIT data'
    if os.path.isdir(mit_root):
        print('\n--- MIT（全局划分）---')
        report('MIT', mit_legacy(mit_root), mit_uniform(mit_root))

    tju_root = 'data/TJU data'
    if os.path.isdir(tju_root):
        print('\n--- TJU（按 Dataset 划分）---')
        for i, name in enumerate(os.listdir(tju_root)):
            report(f'TJU / {name[:28]}',
                   tju_legacy(tju_root, i), tju_uniform(tju_root, i))

    # XJTU 子串匹配误伤清单
    print('\n' + '=' * 84)
    print('XJTU legacy 子串匹配误伤明细（编号含 4 或 8 即被判为测试）')
    print('=' * 84)
    for batch in ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']:
        _, test = xjtu_legacy(xjtu_root, batch)
        ids = sorted(int(os.path.basename(f).split('-')[-1].split('.')[0])
                     for f in test)
        if ids:
            print(f'  {batch:<10} 测试电池编号: {ids}')

    print('\n对照：XJTU uniform 抽取的测试电池编号')
    for batch in ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']:
        _, test = xjtu_uniform(xjtu_root, batch)
        ids = sorted(int(os.path.basename(f).split('-')[-1].split('.')[0])
                     for f in test)
        if ids:
            print(f'  {batch:<10} 测试电池编号: {ids}')


if __name__ == '__main__':
    main()
