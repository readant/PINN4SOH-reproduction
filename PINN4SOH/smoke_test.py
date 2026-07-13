"""Smoke test for PINN4SOH - runs 1 epoch on XJTU 2C batch."""
import os
import argparse
from dataloader.dataloader import XJTUdata
from Model.Model import PINN

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

train_list = []
files = os.listdir(root)
for file in files:
    if args.batch in file:
        if '4' in file or '8' in file:
            pass  # skip test batteries
        else:
            train_list.append(os.path.join(root, file))

print(f"Train files: {len(train_list)}")
print(f"Files: {[os.path.basename(f) for f in train_list]}")

dataloader = data.read_all(specific_path_list=train_list)
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
    testloader=None
)
print("\nSmoke test passed!")
