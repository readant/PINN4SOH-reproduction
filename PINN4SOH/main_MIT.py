
from dataloader.dataloader import XJTUdata,MITdata,HUSTdata,TJUdata
from Model.Model import PINN
from utils.split import uniform_split, describe_split
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MIT', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')

    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')

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
    parser.add_argument('--save_folder', type=str, default='results/MIT results', help='save folder')

    args = parser.parse_args()

    return args

def load_MIT_data(args,small_sample=None):
    root = 'data/MIT data'
    all_files = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        all_files.extend(os.path.join(batch_root, f) for f in os.listdir(batch_root))

    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        train_list, test_list = uniform_split(all_files)
    else:
        # 原实现：从文件名解析电池编号，编号能被 5 整除的划入测试集。
        # 三个批次各自独立取模，整体测试占比 18.4%，与其他数据集不一致。
        train_list, test_list = [], []
        for path in all_files:
            bid = int(os.path.basename(path).split('-')[-1].split('.')[0])
            if bid % 5 == 0:
                test_list.append(path)
            else:
                train_list.append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    if getattr(args, 'split_strategy', 'legacy') == 'uniform':
        print(describe_split('MIT', train_list, test_list))
    data = MITdata(root=root,args=args)
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
        setattr(args, 'save_folder', f'results/MIT results/Experiment{e + 1}')  # [路径统一] 原为 'revise_results/...'
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        dataloader = load_MIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

def small_sample():
    args = get_args()
    num_battery = 2
    base_seed = args.seed
    for e in range(10):
        setattr(args, 'seed', base_seed + e)
        setattr(args, 'save_folder',
                f'results/MIT results (small sample {num_battery})/Experiment{e + 1}')
        setattr(args, 'batch_size', 128)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        dataloader = load_MIT_data(args, small_sample=num_battery)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


if __name__ == '__main__':
    # [入口激活] 原作者注释了入口，取消注释以直接运行
    main()
