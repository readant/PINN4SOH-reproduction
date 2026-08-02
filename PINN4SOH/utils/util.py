from sklearn import metrics
import numpy as np
import logging
import random
import os
import torch


def set_seed(seed=42, deterministic=True):
    """
    [可复现性修复] 固定所有随机源。

    原代码只固定了 train_test_split 的 random_state=420，而模型权重初始化
    (xavier_normal_)、Dropout 掩码、DataLoader 的 shuffle 顺序全部依赖未初始化的
    全局随机状态 —— 导致同一份配置两次运行会给出不同结果，"10 次实验取平均"
    也无法被他人复现。

    注意：不要对所有实验使用同一个种子。10 次 Experiment 若种子相同，得到的是
    10 个完全一样的结果，均值毫无统计意义。正确做法是每次实验用
    `base_seed + experiment_index`（见各 main 的 --seed 用法）。

    :param seed: 随机种子
    :param deterministic: 是否强制 cuDNN 确定性算法。True 可保证逐位可复现，
                           但会牺牲部分训练速度；仅在需要严格复现时开启。
    """
    seed = int(seed)
    random.seed(seed)
    # 注意：PYTHONHASHSEED 只对"进程启动前"设置生效，运行时再赋值不会改变
    # 已启动解释器的哈希随机化，因此这里不写这一行，避免造成"已固定哈希种子"
    # 的错觉。若确需逐位复现字典/集合迭代顺序，请在启动命令中设置该环境变量。
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _close_logger_handlers(logger):
    """安全移除并关闭 logger 上的所有 handler（供 get_logger / clear_logger 复用）。"""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def get_logger(log_name='log.txt'):
    logger = logging.getLogger('mylogger')
    logger.setLevel(level=logging.DEBUG)
    # [修复] getLogger('mylogger') 是进程级单例，原实现每次调用都 addHandler，
    # 循环跑 10 次实验会累积 10 组 handler：日志重复打印 N 遍，且 FileHandler
    # 一直持有文件句柄造成泄漏。这里在挂载新 handler 前先彻底清理旧的。
    _close_logger_handlers(logger)
    formatter = logging.Formatter('%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))

    return [MAE,MAPE,MSE,RMSE]


def eval_metrix_with_r2(true_label, pred_label):
    """
    [补充] FineTune results.py 需要额外返回 R2。
    不修改原 eval_metrix 签名以避免破坏 Model.py 等调用方（它们只期望 4 个返回值）。
    """
    MAE, MAPE, MSE, RMSE = eval_metrix(true_label, pred_label)
    R2 = metrics.r2_score(true_label, pred_label)
    return [MAE, MAPE, MSE, RMSE, R2]

def write_to_txt(txt_name,txt):
    # [健壮性] 原实现假设目标目录已存在。目录实际由 PINN.__init__ 创建，
    # 而 load_all_battery 在 PINN 之前就会调用本函数写日志 ——
    # 一旦调用方忘记提前建目录（如自定义脚本），就会 FileNotFoundError。
    # 这里兜底创建，让调用顺序不再敏感。
    dir_name = os.path.dirname(txt_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    # [跨平台] 显式声明 utf-8：Windows 下 open() 默认使用 ANSI 代码页（GBK），
    # 写入中文日志会与其它平台/工具产生编码不一致。
    with open(txt_name, 'a', encoding='utf-8') as f:
        f.write(txt)
        f.write('\n')

if __name__ == '__main__':
    for i in range(3):
        logger_name = f'log_{i}.txt'
        if i == 1:
            logger_name = None
        logger = get_logger(log_name=logger_name)
        logger.info(f'time: {i}, This is a log info')
        logger.removeHandler(logger.handlers[0])
        logger.handlers.clear()

