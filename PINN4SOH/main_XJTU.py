from dataloader.dataloader import XJTUdata
from Model.Model import PINN
from utils.split import uniform_split, describe_split
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    files = os.listdir(root)
    batch_files = [os.path.join(root, f) for f in files if args.batch in f]

    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        train_list, test_list = uniform_split(batch_files)
    else:
        train_list, test_list = [], []
        for path in batch_files:
            # 原实现：对文件名做子串匹配。编号里含 4 或 8 的电池一律划入测试集
            #（如 3C batch 的 14 号电池会因含 '4' 被误判为测试集）。
            # 保留原行为以维持历史基线，改用 --split_strategy uniform 可规避。
            if '4' in os.path.basename(path) or '8' in os.path.basename(path):
                test_list.append(path)
            else:
                train_list.append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]

    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        print(describe_split(f'XJTU/{args.batch}', train_list, test_list))

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader

def main():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    base_seed = args.seed
    for i in range(6):
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            # [可复现性] 每次实验使用 seed+e：10 次实验彼此不同（统计有意义），
            # 但整轮实验可被完整复现。
            setattr(args, 'seed', base_seed + e)
            save_folder = 'results/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)  # [路径统一] 原为 'results of reviewer/...'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)

            dataloader = load_data(args)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

def small_sample():
    args = get_args()
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    base_seed = args.seed
    for n in [1,2,3,4]:
        for i in range(6):
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            setattr(args,'alpha',0.5)
            setattr(args,'beta',10)
            for e in range(10):
                setattr(args, 'seed', base_seed + e)
                save_folder = f'results/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_data(args,small_sample=n)
                pinn = PINN(args)
                pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'],
                           testloader=dataloader['test'])

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    # 可复现性相关 (reproducibility)
    parser.add_argument('--seed', type=int, default=42,
                        help='基础随机种子。第 e 次实验使用 seed+e，'
                             '使 10 次实验彼此不同（统计有意义）但整轮可完整复现')
    parser.add_argument('--split_strategy', type=str, default='legacy',
                        choices=['legacy', 'uniform'],
                        help='legacy: 沿用原划分规则（保持历史基线不变）；'
                             'uniform: 使用 utils/split.py 的统一规则')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='是否强制 cuDNN 确定性算法。True 可逐位复现，但会略降训练速度')

    parser.add_argument('--log_dir', type=str, default='text log.txt', help='log dir, if None, do not save')
    # [路径统一] 原代码保存到 'results of reviewer/XJTU results/'，与 results/ 目录结构不一致
    # 统一使用 'results/XJTU results/'，与 main_HUST/main_TJU/main_MIT 保持一致
    parser.add_argument('--save_folder', type=str, default='results/XJTU results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

