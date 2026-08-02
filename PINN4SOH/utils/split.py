"""
统一数据划分模块 (Unified train/test splitting)
================================================

解决什么问题
------------
原代码里四个数据集各写了一套划分规则，彼此不一致、且部分实现有缺陷：

* **XJTU**：`if '4' in file or '8' in file` —— 这是对整个文件名的**子串匹配**。
  实测（见 `check_split.py`）6 个 batch 中仅 3C 批被误伤：`3C_battery-14.csv`
  因编号含 '4' 被判入测试集，使该批测试电池由 2 只变为 3 只（3/15 = 20%）。
  其余批次的电池编号（1–15）不含 24/40–49 这类编号，因此未受影响。
* **HUST**：硬编码 20 个 test_id 字符串，其中 `'5-7'` 与其余 `-4/-8` 的模式不符，
  实测与文件名不匹配，疑似笔误；且新增电池后必须手工维护这份列表。
* **MIT**：`id % 5 == 0`，依赖从文件名解析编号。
* **TJU**：按枚举下标个位是否为 `(5,9)/(4,8)/(5,9)`，且**依赖 os.listdir 的返回顺序**，
  在不同文件系统上可能得到不同划分。

这四套规则让"跨数据集横向对比"失去公平性 —— 论文 Table 里把四个数据集的
MAPE 放在一起比较，但它们的测试集构造方式完全不同。

设计
----
`uniform_split()` 用**同一套规则**处理所有数据集：

1. 按**自然序**（battery-2 排在 battery-10 之前）对电池文件排序，消除
   `os.listdir` 顺序依赖；
2. 以固定步长取每段**中点**在整个编号范围内均匀抽取测试电池，而不是截取尾部 ——
   这样测试电池的老化程度分布与训练集一致，同时避免"编号最小/最大的电池
   恒定落入某一侧"的系统性偏置；
3. 划分粒度是**电池级**（不是样本级），保证同一只电池不会同时出现在
   训练集和测试集里，避免信息泄漏。

适用范围与局限
--------------
本模块统一的是**函数内部的抽取规则**；一次调用处理哪些文件（即"划分单元"）
仍由调用方决定 —— XJTU 传单个 batch、HUST 传全体、MIT 传 3 个批次合并后的
全体、TJU 传单个 Dataset。因此它并未把四个数据集的划分单元也统一起来，
跨数据集的横向对比仍需在报告或论文中显式说明各自的划分单元。

向后兼容
--------
默认仍走各数据集原有的 legacy 规则（`--split_strategy legacy`），
已有的 290 次实验结果与基线对比不受影响。显式传
`--split_strategy uniform` 才启用统一规则。
"""

import os
import re


def natural_key(path):
    """
    自然排序键：把文件名中的数字按数值大小比较。

    字典序下 'battery-10' < 'battery-2'，会让编号顺序变得混乱；
    自然序得到 1, 2, 3, ... 10, 11，划分结果才符合直觉。
    """
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def uniform_split(file_list, test_ratio=0.2):
    """
    在"一个划分单元"内部，按统一规则切分训练/测试电池。

    一个划分单元可以是：XJTU 的某个 batch（如 '2C'）、HUST 的全体、
    MIT 的全体、TJU 的某个 Dataset —— 由调用方决定传入哪些文件。

    :param file_list: 文件路径列表（同一划分单元内的全部电池）
    :param test_ratio: 测试电池占比，默认 0.2
    :return: (train_list, test_list)
    """
    if not file_list:
        raise ValueError('uniform_split 收到空的文件列表')

    files = sorted(file_list, key=natural_key)
    n = len(files)
    n_test = max(1, int(round(n * test_ratio)))
    if n_test >= n:
        raise ValueError(
            f'测试集占满了全部电池（共 {n} 个，测试 {n_test} 个）。'
            f'请调低 test_ratio。'
        )

    # 把 [0, n) 等分为 n_test 段，取每段中点作为测试索引。
    # 早期实现用 int(i * step) 从索引 0 起算，会让"编号最小的电池"在每个划分单元里
    # 恒定进入测试集（XJTU 6 个 batch 全部命中 1 号），构成系统性偏置。
    step = n / n_test
    test_idx = {min(int((i + 0.5) * step), n - 1) for i in range(n_test)}

    test_list = [f for i, f in enumerate(files) if i in test_idx]
    train_list = [f for i, f in enumerate(files) if i not in test_idx]
    return train_list, test_list


def describe_split(name, train_list, test_list):
    """生成一行人类可读的划分摘要，便于写入日志、事后核对。"""
    total = len(train_list) + len(test_list)
    ratio = len(test_list) / total if total else 0.0
    return (f'[split] {name}: 共 {total} 只电池 | '
            f'训练 {len(train_list)} | 测试 {len(test_list)} | '
            f'测试占比 {ratio:.1%}')


def check_leakage(train_list, test_list):
    """
    检查训练集与测试集是否存在电池级别的重叠（信息泄漏）。

    :return: 重叠的文件名集合，为空表示无泄漏
    """
    return set(os.path.basename(p) for p in train_list) & \
           set(os.path.basename(p) for p in test_list)
