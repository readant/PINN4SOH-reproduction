# 数据集说明

本目录存放**预处理后**的四个公开电池数据集，供 `PINN4SOH/` 下的训练脚本直接读取。
所有 `main_*.py` 均以本目录为相对路径基准（`data/<数据集> data/`），因此必须在
`PINN4SOH/` 目录下运行。

## 目录与规模

| 目录 | 化学体系 | CSV 文件数 | 来源 | 许可 |
|------|---------|-----------|------|------|
| `XJTU data/` | NCM（LiNi₀.₅Co₀.₂Mn₀.₃O₂） | 55 | 西安交通大学，[Zenodo 10963339](https://zenodo.org/records/10963339) | 见 Zenodo 页面 |
| `HUST data/` | LFP / 石墨 | 77 | 华中科技大学，[Mendeley nsc7hnsg4s/2](https://data.mendeley.com/datasets/nsc7hnsg4s/2) | CC BY 4.0 |
| `MIT data/` | LFP / 石墨 | 125 | Severson et al.，[data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) | CC BY 4.0 |
| `TJU data/` | NCA / NCM / NCM·NCA | 130 | Zhu et al.，[Zenodo 6405084](https://zenodo.org/record/6405084) | 见 Zenodo 页面 |
| **合计** | — | **387** | — | — |

387 与论文「387 只电池、310,705 个样本」一致。

## 目录结构

```
data/
├── XJTU data/          2C_battery-1.csv … satellite_battery-15.csv（6 种充放电协议）
├── HUST data/          1-1.csv … 10-8.csv
├── MIT data/
│   ├── 2017-05-12/     46 个 CSV
│   ├── 2017-06-30/     43 个 CSV
│   └── 2018-04-12/     36 个 CSV
└── TJU data/
    ├── Dataset_1_NCA_battery/          66 个 CSV
    ├── Dataset_2_NCM_battery/          55 个 CSV
    └── Dataset_3_NCM_NCA_battery/       9 个 CSV
```

## 数据格式

每个 CSV 为一只电池的循环数据，**最后一列是 SOH 标签**，其余列为特征
（由官方预处理代码从充电末段提取的统计量）。`dataloader/dataloader.py::DF.load_one_battery()`
读取后将特征与标签拆分为 `x` / `y`，并构造相邻周期样本对 `(x1,y1)` 与 `(x2,y2)`。

## 数据从哪来

原始数据需从上述各来源自行下载，再经官方预处理代码库
[Battery-dataset-preprocessing-code-library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)
（本仓库根目录下同名文件夹）处理，得到本目录中的 CSV。

## 注意事项

1. **再分发需逐项确认许可**：四个数据集条款不同（HUST、MIT 为 CC BY 4.0，
   XJTU 与 TJU 需按各自来源页面的要求署名）。若将本仓库公开，请先核对。
2. **不要遍历读取**：本目录共 387 个文件，全量读取耗时且无必要；
   `dataloader` 已按需指定路径读取。
3. **不要手工修改 CSV**：任何预处理改动都应回到官方预处理代码库中完成，
   以保证与论文口径一致。
4. 训练脚本的**训练/测试划分**不写在本目录，而由各 `main_*.py` 的划分规则决定
   （默认 legacy，可用 `--split_strategy uniform` 切换到 `utils/split.py` 的统一规则）。
