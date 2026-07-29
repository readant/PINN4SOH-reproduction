
from dataloader.dataloader import XJTUdata,MITdata,HUSTdata,TJUdata
from Model.Model import PINN
from utils.split import uniform_split, describe_split
import argparse
import os
# [环境适配] 原代码: CUDA_VISIBLE_DEVICES = '1'（作者使用多 GPU 系统）
# 当前环境仅一张 RTX 5060（device 0），设为 '1' 会导致 GPU 不可见
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for HUST dataset')
    parser.add_argument('--data', type=str, default='HUST', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    # model related
    # parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    # parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
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

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results/HUST results', help='save folder')

    args = parser.parse_args()

    return args


def load_HUST_data(args,small_sample=None):
    root = 'data/HUST data'
    data = HUSTdata(root=root,args=args)
    files = [os.path.join(root, f) for f in os.listdir(root)]

    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        train_list, test_list = uniform_split(files)
    else:
        # 原实现：硬编码 20 个测试电池 ID，得到 26.0% 的测试占比，
        # 高于其余数据集的 ~20%；其中 '5-7' 与其余 '-4/-8' 的模式不一致，
        # 疑似笔误。且新增电池后必须手工维护这份列表。
        test_id = ['1-4','1-8','2-4','2-8',
                   '3-4','3-8','4-4','4-8',
                   '5-4','5-7','6-4','6-8',
                   '7-4','7-8','8-4','8-8',
                   '9-4','9-8','10-4','10-8']
        train_list, test_list = [], []
        for path in files:
            if os.path.basename(path)[:-4] in test_id:
                test_list.append(path)
            else:
                train_list.append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        print(describe_split('HUST', train_list, test_list))

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def main():
    args = get_args()
    base_seed = args.seed
    for e in range(10):
        # [可复现性] 每次实验使用 seed+e：10 次实验彼此不同（统计有意义），
        # 但整轮实验可被完整复现。
        setattr(args, 'seed', base_seed + e)
        setattr(args, 'save_folder', f'results/HUST results/Experiment{e + 1}')
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        dataloader = load_HUST_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])


def small_sample():
    args = get_args()
    base_seed = args.seed
    for n in [1,2,3,4]:
        for e in range(10):
            setattr(args, 'seed', base_seed + e)
            setattr(args,'save_folder',f'results/HUST results (small sample {n})/Experiment{e+1}')
            setattr(args,'batch_size',128)
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            dataloader = load_HUST_data(args,small_sample=n)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])


if __name__ == '__main__':
    # [入口激活] 原作者注释了入口，release 模式需手动调用。取消注释以直接运行
    main()
    # small_sample()
