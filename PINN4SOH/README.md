# PINN4SOH

> ### 本仓库补充说明（阅读下方官方文档前请先看这里）
>
> 本文件是**原作者提供的英文 README**，此处保留原文以便与上游对照。
> 本仓库对其代码做了环境迁移与缺陷修复，因此下列描述与实际状态存在差异：
>
> | 原 README 的说明 | 本仓库实际情况 |
> |-----------------|---------------|
> | Python 3.7.10 + PyTorch 1.7.1 | 已迁移到 **Python 3.12 + PyTorch 2.x（CUDA）** |
> | `conda create -n new_environment python=3.7.10` | 本仓库使用 conda 环境 `main` |
> | `main_comparison.py` | 实际文件名为 **`main_comparision.py`**（原作者文件名拼写如此，未改动以保持与上游一致） |
> | `main_adaptation - fine-tuning.py` | 已重命名为 **`main_finetune.py`**（原文件名含空格与连字符，无法被 import） |
> | 运行 `results analysis/XJTU results.py` 处理结果 | 该脚本读取 `../results/Ours/...`（**原作者提供**的结果包，需先解压）；汇总**本仓库实测**结果请用 `python results_summary.py` |
> | 数据集为预处理后的 CSV | 本仓库 `data/` 下共 387 个 CSV，来源与许可见 `data/README.md` |
>
> 新增的一键入口、验证脚本与文档索引，请见仓库根目录 `README.md` 与
> `docs/08-verification-guide.md`、`docs/CHANGELOG.md`。

This code is for our paper: [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](https://www.nature.com/articles/s41467-024-48779-z)

> [!IMPORTANT]
Summary of articles using the XJTU Battery Dataset:
<https://github.com/wang-fujin/XJTU-Battery-Dataset-Papers-Summary>

# 1. System requirements

python version: 3.7.10

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  1.7.1   |
|    sklearn     |  0.24.2  |
|     numpy      |  1.20.3  |
|     pandas     |  1.3.5   |
|   matplotlib   |  3.3.4   |
|  scienceplots  |          |

# 2. Installation guide

If you are not familiar with Python and Pytorch framework,
you can install Anaconda first and use Anaconda to quickly configure the environment.

## 2.1 Create environment

```angular2html
conda create -n new_environment python=3.7.10
```

## 2.2 Activate environment

```angular2html
conda activate new_environment
```

## 2.3 Install dependencies

```angular2html
conda install pytorch=1.7.1
conda install scikit-learn=0.24.2 numpy=1.20.3 pandas=1.3.5 matplotlib=3.3.4
pip install scienceplots      # for beautiful plots
```

# 3. Demo

We provide a detailed demo of our code running on the XJTU dataset.

1. Run the `main_XJTU.py` file to train our model. The program will generate a folder named `results` and save the results in it.
2. Run the `main_comparison.py` file. You can change `setattr(args,'model','MLP')` to select the CNN or MLP model. It will generate a folder in the `results` to save the results of the corresponding model (CNN or MLP).
3. Run the `results analysis/XJTU results.py` file. It will process the results in Step one and generate the `XJTU_results.xlsx` file. At the same time, the results of each batch in the XJTU dataset will also be printed on the Command Console, corresponding to the results in Table 2 of our manuscript.
4. Run the `results analysis/Comparision results.py` file to generate the `XJTU-MLP_results.xlsx` file and save it in the `results` folder. The results of each batch in the XJTU data set will also be printed on the Command Console, corresponding to the results in Table 2 of our manuscript.

**Note: As we all know, the training process of neural network models is random,
and the volatility of regression models is often greater than that of classification models.
Therefore, the results obtained from the above process are not expected to be exactly identical to those mentioned in our manuscript.
However, it is evident that the results obtained from our method are superior to those of MLP and CNN.**

In addition, we also provide the results of our training,
which are saved in the `results` folder and `results analysis` folder.
These results correspond exactly to the data in our manuscript.

What's more, we also provide the codes corresponding to the Figures in our manuscript,
which are saved in the `plotter` folder.
You can use these codes to draw the Figures in the manuscript.

# 4.  Additional information

The data in the `data` folder is preprocessed data.
Raw data can be obtained from the following links:

1. XJTU dataset: [link](https://wang-fujin.github.io/)
2. TJU dataset: [link](https://zenodo.org/record/6405084)
3. HUST dataset: [link](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
4. MIT dataset: [link](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)

The code for **reading and preprocessing** the dataset is publicly available at [https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)

---

We generated a comprehensive dataset consisting of 55 lithium-nickel-cobalt-manganese-oxide (NCM) batteries.

It is available at: [Link](https://wang-fujin.github.io/)

Zenodo link: [https://zenodo.org/records/10963339](https://zenodo.org/records/10963339).

![https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png](https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png)

![https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png](https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png)

# 5. Citation

If you find it useful, please cite our paper:

```bibtex
@article{wang2024physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4332},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
