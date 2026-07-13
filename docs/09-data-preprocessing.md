# 09 - 数据预处理详细指南

## 数据集格式分析

### MIT数据集

```
格式: CSV文件
结构: 每个电池一个CSV文件
关键列:
├── cycle_number: 周期编号
├── voltage: 电压序列
├── current: 电流序列
├── time: 时间序列
├── capacity: 容量
└── temperature: 温度

SOH计算: Q_current / Q_initial
```

### HUST/TJU数据集

```
格式: MAT文件 (MATLAB)
结构: 每个电池一个MAT文件
关键字段:
├── cycle: 周期信息
├── voltage: 电压
├── current: 电流
├── time: 时间
└── capacity: 容量

需要使用scipy.io.loadmat读取
```

### XJTU数据集

```
格式: MAT文件
结构: 按batch组织
包含: 55个NCM电池的完整退化数据
```

---

## 特征提取实现

### 从充电曲线提取特征

```python
import numpy as np
from scipy.stats import kurtosis, skew

def extract_features_from_charge(voltage, current, time, V_end=4.2):
    """
    从充电曲线提取16维特征

    参数:
        voltage: 电压序列 (V)
        current: 电流序列 (A)
        time: 时间序列 (s)
        V_end: 充电截止电压 (V)

    返回:
        features: 16维特征向量
    """
    # 1. 截取电压区间 [V_end-0.2, V_end]
    V_start = V_end - 0.2
    voltage_mask = (voltage >= V_start) & (voltage <= V_end)

    # 2. 截取电流区间 [0.1A, 0.5A] (恒压充电阶段)
    current_mask = (current >= 0.1) & (current <= 0.5)

    # 3. 组合掩码
    combined_mask = voltage_mask & current_mask

    # 4. 如果数据不足，只用电压掩码
    if np.sum(combined_mask) < 10:
        combined_mask = voltage_mask

    # 5. 提取数据
    v_selected = voltage[combined_mask]
    i_selected = current[combined_mask]
    t_selected = time[combined_mask]

    # 6. 边界检查
    if len(v_selected) < 5:
        return np.zeros(16)

    features = []

    # 电压特征 (8个)
    features.append(np.mean(v_selected))           # 1. mean
    features.append(np.std(v_selected))            # 2. std
    features.append(kurtosis(v_selected))          # 3. kurtosis
    features.append(skew(v_selected))              # 4. skewness

    if len(t_selected) >= 2:
        charge_time = t_selected[-1] - t_selected[0]
    else:
        charge_time = 0
    features.append(charge_time)                   # 5. charging time

    delta_t = np.diff(t_selected)
    if len(delta_t) > 0:
        avg_current = np.mean(i_selected[:-1])
        accumulated_charge = np.sum(avg_current * delta_t) / 3600
    else:
        accumulated_charge = 0
    features.append(accumulated_charge)            # 6. accumulated charge

    if len(v_selected) >= 2:
        slope_v = np.polyfit(t_selected, v_selected, 1)[0]
    else:
        slope_v = 0
    features.append(slope_v)                       # 7. curve slope

    hist_v, _ = np.histogram(v_selected, bins=10, density=True)
    hist_v = hist_v[hist_v > 0]
    entropy_v = -np.sum(hist_v * np.log(hist_v))
    features.append(entropy_v)                     # 8. curve entropy

    # 电流特征 (8个)
    features.append(np.mean(i_selected))           # 9. mean
    features.append(np.std(i_selected))            # 10. std
    features.append(kurtosis(i_selected))          # 11. kurtosis
    features.append(skew(i_selected))              # 12. skewness

    if len(t_selected) >= 2:
        charge_time_i = t_selected[-1] - t_selected[0]
    else:
        charge_time_i = 0
    features.append(charge_time_i)                 # 13. charging time

    if len(delta_t) > 0:
        accumulated_charge_i = np.sum(i_selected[:-1] * delta_t) / 3600
    else:
        accumulated_charge_i = 0
    features.append(accumulated_charge_i)          # 14. accumulated charge

    if len(t_selected) >= 2:
        slope_i = np.polyfit(t_selected, i_selected, 1)[0]
    else:
        slope_i = 0
    features.append(slope_i)                       # 15. curve slope

    hist_i, _ = np.histogram(i_selected, bins=10, density=True)
    hist_i = hist_i[hist_i > 0]
    entropy_i = -np.sum(hist_i * np.log(hist_i))
    features.append(entropy_i)                     # 16. curve entropy

    return np.array(features)
```

### 完整的预处理流程

```python
import os
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

class BatteryDataPreprocessor:
    """电池数据预处理器"""

    def __init__(self, V_end=4.2):
        self.V_end = V_end

    def process_dataset(self, dataset_name, raw_dir, output_dir):
        """处理整个数据集"""
        os.makedirs(output_dir, exist_ok=True)

        if dataset_name == 'MIT':
            self._process_mit(raw_dir, output_dir)
        elif dataset_name in ['HUST', 'TJU', 'XJTU']:
            self._process_mat_dataset(dataset_name, raw_dir, output_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _process_mit(self, raw_dir, output_dir):
        """处理MIT数据集"""
        all_features = []
        all_soh = []
        all_cycles = []
        all_battery_ids = []

        battery_idx = 0
        for file in Path(raw_dir).glob("*.csv"):
            print(f"Processing {file.name}...")

            df = pd.read_csv(file)

            # 获取初始容量
            initial_capacity = df['capacity'].iloc[0]

            # 按周期分组
            for cycle_num in df['cycle_number'].unique():
                cycle_data = df[df['cycle_number'] == cycle_num]

                # 提取充电数据
                charge_mask = cycle_data['current'] > 0
                if charge_mask.sum() < 10:
                    continue

                voltage = cycle_data.loc[charge_mask, 'voltage'].values
                current = cycle_data.loc[charge_mask, 'current'].values
                time = cycle_data.loc[charge_mask, 'time'].values
                capacity = cycle_data.loc[charge_mask, 'capacity'].values

                # 提取特征
                features = extract_features_from_charge(
                    voltage, current, time, self.V_end
                )

                # 计算SOH
                current_capacity = capacity[-1] if len(capacity) > 0 else 0
                soh = current_capacity / initial_capacity

                if soh >= 0.7:  # 过滤异常值
                    all_features.append(features)
                    all_soh.append(soh)
                    all_cycles.append(cycle_num)
                    all_battery_ids.append(battery_idx)

            battery_idx += 1

        # 保存处理后的数据
        np.save(f"{output_dir}/features.npy", np.array(all_features))
        np.save(f"{output_dir}/soh.npy", np.array(all_soh))
        np.save(f"{output_dir}/cycles.npy", np.array(all_cycles))
        np.save(f"{output_dir}/battery_ids.npy", np.array(all_battery_ids))

        print(f"Processed {battery_idx} batteries, {len(all_features)} samples")

    def _process_mat_dataset(self, dataset_name, raw_dir, output_dir):
        """处理HUST/TJU/XJTU数据集"""
        all_features = []
        all_soh = []
        all_cycles = []
        all_battery_ids = []

        battery_idx = 0
        for file in Path(raw_dir).glob("*.mat"):
            print(f"Processing {file.name}...")

            mat_data = sio.loadmat(file)
            battery_data = mat_data['battery']

            # 获取初始容量
            initial_capacity = None
            for cycle_info in battery_data[0, 0]['cycle'][0, :]:
                if cycle_info['type'] == 'discharge':
                    initial_capacity = cycle_info['capacity'][0, -1]
                    break

            if initial_capacity is None:
                continue

            # 遍历每个周期
            for cycle_idx in range(battery_data[0, 0]['cycle'].shape[1]):
                cycle_info = battery_data[0, 0]['cycle'][0, cycle_idx]

                # 只处理充电周期
                if cycle_info['type'] != 'charge':
                    continue

                voltage = cycle_info['voltage'][0, :]
                current = cycle_info['current'][0, :]
                time = cycle_info['time'][0, :]

                # 提取特征
                features = extract_features_from_charge(
                    voltage, current, time, self.V_end
                )

                # 计算SOH
                capacity = cycle_info['capacity'][0, -1]
                soh = capacity / initial_capacity

                if soh >= 0.7:
                    all_features.append(features)
                    all_soh.append(soh)
                    all_cycles.append(cycle_idx)
                    all_battery_ids.append(battery_idx)

            battery_idx += 1

        # 保存数据
        np.save(f"{output_dir}/features.npy", np.array(all_features))
        np.save(f"{output_dir}/soh.npy", np.array(all_soh))
        np.save(f"{output_dir}/cycles.npy", np.array(all_cycles))
        np.save(f"{output_dir}/battery_ids.npy", np.array(all_battery_ids))

        print(f"Processed {battery_idx} batteries, {len(all_features)} samples")
```

---

## 数据验证脚本

```python
def verify_processed_data(data_dir):
    """验证处理后的数据"""
    features = np.load(f"{data_dir}/features.npy")
    soh = np.load(f"{data_dir}/soh.npy")
    cycles = np.load(f"{data_dir}/cycles.npy")
    battery_ids = np.load(f"{data_dir}/battery_ids.npy")

    print(f"Features shape: {features.shape}")
    print(f"SOH range: [{soh.min():.3f}, {soh.max():.3f}]")
    print(f"Cycles range: [{cycles.min()}, {cycles.max()}]")
    print(f"Number of batteries: {len(np.unique(battery_ids))}")
    print(f"Samples per battery: {np.bincount(battery_ids).mean():.0f} ± "
          f"{np.bincount(battery_ids).std():.0f}")

    # 检查特征是否有异常值
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")

    # 检查SOH分布
    print(f"SOH mean: {soh.mean():.3f}")
    print(f"SOH std: {soh.std():.3f}")
```

---

## 常见问题与解决方案

### 问题1: 特征提取返回全零

**原因**: 充电数据不足或格式异常

**解决**:
```python
# 检查数据
if len(voltage) < 10:
    print(f"Warning: Battery {battery_id} has insufficient data")
    continue
```

### 问题2: SOH值异常

**原因**: 初始容量计算错误或容量数据缺失

**解决**:
```python
# 验证SOH范围
if soh < 0.5 or soh > 1.1:
    print(f"Warning: Abnormal SOH {soh:.3f} for battery {battery_id}")
    continue
```

### 问题3: 内存不足

**原因**: 数据量太大

**解决**:
```python
# 分批处理
def process_in_batches(data, batch_size=1000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        results.append(extract_features_batch(batch))
    return np.concatenate(results)
```

### 问题4: 时间序列不连续

**原因**: 数据采集间隔不一致

**解决**:
```python
# 插值处理
from scipy.interpolate import interp1d

def interpolate_time_series(time, values, target_points=100):
    f = interp1d(time, values, kind='linear')
    new_time = np.linspace(time.min(), time.max(), target_points)
    return new_time, f(new_time)
```
