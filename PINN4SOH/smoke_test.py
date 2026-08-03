"""
Smoke test for PINN4SOH - runs 1 epoch on XJTU 2C batch.

用途：改动代码后快速确认「数据加载 → 模型构建 → 单轮训练 → 测试评估」链路完好。

[修复] 原实现硬编码 legacy 的子串划分（`'4' in f or '8' in f`）且不固定随机种子，
与本仓库新增的可复现能力脱节 —— 同一个脚本两次运行结果不可比。
现统一改用 utils/split.py 的 uniform_split，并固定随机种子。
"""
import os
import argparse
from dataloader.dataloader import XJTUdata
from Model.Model import PINN
from utils.split import uniform_split, describe_split
from utils.util import set_seed

SEED = 42
set_seed(SEED)

args = argparse.Namespace()
args.batch_size = 256
args.normalization_method = 'min-max'
args.epochs = 1
args.early_stop = 5
args.warmup_epochs = 1
args.warmup_lr = 0.002
args.lr = 0.01
args.final_lr = 0.0002
args.lr_F = 0.001
args.F_layers_num = 3
args.F_hidden_dim = 60
args.alpha = 0.7
args.beta = 0.2
args.save_folder = 'results/smoke_test'
args.log_dir = 'smoke_test.txt'
args.batch = '2C'

os.makedirs(args.save_folder, exist_ok=True)

root = 'data/XJTU data'
data = XJTUdata(root=root, args=args)

files = [os.path.join(root, f) for f in os.listdir(root) if args.batch in f]
train_list, test_list = uniform_split(files)
print(describe_split(f'XJTU/{args.batch}', train_list, test_list))
print(f"Train files: {[os.path.basename(f) for f in train_list]}")

dataloader = data.read_all(specific_path_list=train_list)
test_loader = data.read_all(specific_path_list=test_list)
print(f"Train loader batches: {len(dataloader['train_2'])}")
print(f"Valid loader batches: {len(dataloader['valid_2'])}")

print("\nCreating model...")
pinn = PINN(args)
count_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)
print(f"Trainable params: {count_params}")
print("Data loading and model creation OK!")

print("\nTraining 1 epoch (smoke test)...")
pinn.Train(
    trainloader=dataloader['train_2'],
    validloader=dataloader['valid_2'],
    testloader=test_loader['test_3'],
)
print("\nSmoke test passed!")
