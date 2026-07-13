"""
快速验证脚本：XJTU 2C batch，500 epochs，1次实验。
确认训练流程和 GPU 正常工作，并输出测试集指标。
"""
import os
import argparse
import numpy as np
from dataloader.dataloader import XJTUdata
from Model.Model import PINN

# ---------- 配置 ----------
BATCH = '2C'
EPOCHS = 500
SAVE_DIR = 'results/XJTU results/0-0/Experiment1'

os.makedirs(SAVE_DIR, exist_ok=True)

args = argparse.Namespace(
    batch_size=256,
    normalization_method='min-max',
    epochs=EPOCHS,
    early_stop=20,
    warmup_epochs=30,
    warmup_lr=0.002,
    lr=0.01,
    final_lr=0.0002,
    lr_F=0.001,
    F_layers_num=3,
    F_hidden_dim=60,
    alpha=0.7,
    beta=0.2,
    save_folder=SAVE_DIR,
    log_dir='training_log.txt',
    batch=BATCH,
)

# ---------- 加载数据 ----------
print(f"=== Loading XJTU dataset, batch={BATCH} ===")
root = 'data/XJTU data'
data = XJTUdata(root=root, args=args)

train_list, test_list = [], []
for f in os.listdir(root):
    if BATCH in f:
        if '4' in f or '8' in f:
            test_list.append(os.path.join(root, f))
        else:
            train_list.append(os.path.join(root, f))

loader_dict = data.read_all(specific_path_list=train_list)
test_dict = data.read_all(specific_path_list=test_list)

print(f"  Train batteries: {len(train_list)}")
print(f"  Test batteries:  {len(test_list)}")
print(f"  Train batches:   {len(loader_dict['train_2'])}")
print(f"  Test batches:    {len(test_dict['test_3'])}")

# ---------- 训练 ----------
print(f"\n=== Training PINN ({EPOCHS} epochs) ===")
model = PINN(args)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params}")

model.Train(
    trainloader=loader_dict['train_2'],
    validloader=loader_dict['valid_2'],
    testloader=test_dict['test_3'],
)

# ---------- 评估 ----------
print(f"\n=== Evaluation ===")
true = np.load(os.path.join(SAVE_DIR, 'true_label.npy')).reshape(-1)
pred = np.load(os.path.join(SAVE_DIR, 'pred_label.npy')).reshape(-1)

mae  = np.mean(np.abs(pred - true))
mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
rmse = np.sqrt(np.mean((pred - true) ** 2))

print(f"  MAE:  {mae:.6f}")
print(f"  MAPE: {mape:.4f}%")
print(f"  RMSE: {rmse:.6f}")
print(f"\nResults saved to: {SAVE_DIR}/")
print("=== Quick validation done! ===")
