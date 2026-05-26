"""
数据加载与预处理模块。

功能：
  - 解析 CSV 文件中嵌套列表字符串为数值张量
  - 变长目标数（N=2/3/4）零填充至最大目标数
  - 按特征维度 z-score 标准化
  - 构建 PyTorch Dataset 并同步划分训练/验证/测试集
"""

import ast
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


def parse_nested_list_string(s: str):
    """将嵌套列表字符串解析为 Python 列表。"""
    return ast.literal_eval(s)


def pad_step(sample: np.ndarray, max_len: int) -> np.ndarray:
    """
    将单个时间步的特征填充至固定长度。

    参数:
        sample: 一维数组，长度因 N 而异 (N=2→15, N=3→24, N=4→34)
        max_len: 目标长度 (34)

    返回:
        shape (max_len,) 的填充后数组
    """
    actual_len = len(sample)
    if actual_len < max_len:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:actual_len] = sample
        return padded
    return sample.astype(np.float32)


def parse_csv_to_tensor(filepath: str, max_step_features: int):
    """
    解析单个 CSV 文件为三维张量 (样本数, 时间步, max_step_features)。

    新数据集每步包含: N×6 状态 + N 距离 + C(N,2) 夹角。
    N=2→15, N=3→24, N=4→34。pad 至 34。

    参数:
        filepath: CSV 文件路径
        max_step_features: 单步最大特征数 (34)

    返回:
        numpy 数组, shape (num_samples, 10, max_step_features)
    """
    samples = []
    masks = []

    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过列名行
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = parse_nested_list_string(line)
            time_steps_data = []
            time_steps_mask = []

            for step in raw:
                padded = pad_step(np.array(step, dtype=np.float32), max_step_features)
                time_steps_data.append(padded)

                # 构建有效性掩码（非零位置为真实特征）
                m = (padded != 0).astype(np.float32)
                time_steps_mask.append(m)

            samples.append(np.stack(time_steps_data, axis=0))   # (10, max_step_features)
            masks.append(np.stack(time_steps_mask, axis=0))      # (10, max_step_features)

    data = np.stack(samples, axis=0)  # (num_samples, 10, max_step_features)
    masks = np.stack(masks, axis=0)   # (num_samples, 10, max_step_features)
    return data, masks


def parse_y_csv(filepath: str, config: dict = None):
    """
    解析 Y.csv 标签文件，并应用标签变换。

    参数:
        filepath: CSV 文件路径
        config: 全局配置（用于读取 label_transform 选项）

    返回:
        n_labels: (num_samples,) int64，值为 0/1/2
        dist_labels: (num_samples,) float32 —— log(dist) 或原始 dist
        phi_labels:  (num_samples,) or (num_samples, 2) float32 —— cos/sin 双通道或原始 phi
        label_params: 变换参数 dict（用于逆变换）
    """
    n_list, dist_list, phi_list = [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = parse_nested_list_string(line)
            n_list.append(raw[0])
            dist_list.append(raw[1])
            phi_list.append(raw[2])

    n_labels = np.array(n_list, dtype=np.int64) - 2
    dist_raw = np.array(dist_list, dtype=np.float32)
    phi_raw = np.array(phi_list, dtype=np.float32)

    label_params = {"dist_mean": float(dist_raw.mean()),
                    "dist_std": float(dist_raw.std()),
                    "phi_mean": float(phi_raw.mean()),
                    "phi_std": float(phi_raw.std())}

    # 标签变换
    use_log = config and config.get("label_transform", {}).get("dist_log", False)
    use_cs = config and config.get("label_transform", {}).get("phi_cos_sin", False)

    if use_log:
        dist_labels = np.log(dist_raw + 1e-8).astype(np.float32)
        label_params["dist_log_mean"] = float(dist_labels.mean())
        label_params["dist_log_std"] = float(dist_labels.std())
    else:
        dist_labels = dist_raw

    if use_cs:
        cos_phi = np.cos(phi_raw).astype(np.float32)
        sin_phi = np.sin(phi_raw).astype(np.float32)
        phi_labels = np.stack([cos_phi, sin_phi], axis=-1)  # (N, 2)
    else:
        phi_labels = phi_raw

    return n_labels, dist_labels, phi_labels, label_params


class IntentRecognitionDataset(Dataset):
    """
    意图识别数据集 (v2)。

    每个样本:
        - x_real:       (10, max_targets * 6)
        - x_recurrence: (10, max_targets * 6)
        - mask:         (10, max_targets)
        - n_label:      int (0/1/2)
        - dist_label:   float —— log(min_distance) 或原始值
        - phi_label:    float or (2,) —— cos/sin 双通道或原始 phi
    """

    def __init__(self, real_data, rec_data, masks, n_labels, dist_labels, phi_labels):
        self.real_data = torch.from_numpy(real_data).float()
        self.rec_data = torch.from_numpy(rec_data).float()
        self.masks = torch.from_numpy(masks).float()
        self.n_labels = torch.from_numpy(n_labels).long()
        self.dist_labels = torch.from_numpy(dist_labels).float()
        self.phi_labels = torch.from_numpy(phi_labels).float()
        self._phi_is_2d = (phi_labels.ndim == 2)

    def __len__(self):
        return len(self.real_data)

    def __getitem__(self, idx):
        return {
            "x_real": self.real_data[idx],
            "x_recurrence": self.rec_data[idx],
            "mask": self.masks[idx],
            "n_label": self.n_labels[idx],
            "dist_label": self.dist_labels[idx],
            "phi_label": self.phi_labels[idx],
        }


def compute_grouped_statistics(data: np.ndarray, max_targets: int):
    """
    按物理量分组计算均值和标准差（仅对非零元素）。

    每步 34 个特征分三组：
      - 状态 (0:24): 6 物理量 × 4 目标，统计按 6 物理量做，tile 至 24
      - 距离 (24:28): 1 物理量，tile 至 4
      - 夹角 (28:34): 1 物理量，tile 至 6

    参数:
        data: (num_samples, 10, 34)

    返回:
        mean, std: shape (34,) —— tile 后的均值/标准差
    """
    D = data.shape[-1]  # 34
    num_samples = data.shape[0]

    # --- 状态部分：reshape 为 (N, 10, 4, 6)，按第 3 维(6)统计 ---
    states = data[:, :, :24].reshape(num_samples, -1, max_targets, 6)
    states_flat = states.reshape(-1, 6)
    state_mask = (states_flat != 0).any(axis=1)
    state_valid = states_flat[state_mask]
    state_mean_6 = state_valid.mean(axis=0).astype(np.float32)  # (6,)
    state_std_6 = state_valid.std(axis=0).astype(np.float32)     # (6,)

    # --- 距离部分：reshape 为 (N, 10, 4)，按标量统计 ---
    dists = data[:, :, 24:28].reshape(-1, max_targets)
    dist_flat = dists.reshape(-1)
    dist_nonzero = dist_flat[dist_flat != 0]
    dist_mean = dist_nonzero.mean().astype(np.float32) if len(dist_nonzero) > 0 else np.float32(0)
    dist_std = dist_nonzero.std().astype(np.float32) if len(dist_nonzero) > 0 else np.float32(1)

    # --- 夹角部分：reshape 为 (N, 10, 6)，按标量统计 ---
    angles = data[:, :, 28:34].reshape(-1, 6)
    angle_flat = angles.reshape(-1)
    angle_nonzero = angle_flat[angle_flat != 0]
    angle_mean = angle_nonzero.mean().astype(np.float32) if len(angle_nonzero) > 0 else np.float32(0)
    angle_std = angle_nonzero.std().astype(np.float32) if len(angle_nonzero) > 0 else np.float32(1)

    # --- 组装为 (34,) ---
    mean = np.zeros(D, dtype=np.float32)
    std = np.zeros(D, dtype=np.float32)
    mean[:24] = np.tile(state_mean_6, max_targets)
    std[:24] = np.tile(state_std_6, max_targets)
    mean[24:28] = dist_mean
    std[24:28] = dist_std
    mean[28:34] = angle_mean
    std[28:34] = angle_std
    return mean, std


def normalize_position(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """按组标准化。零填充位置保持为零。"""
    data = data.copy()
    mask = data != 0
    mean = mean.reshape(1, 1, -1).astype(data.dtype)
    std = std.reshape(1, 1, -1).astype(data.dtype)
    data = (data - mean) / (std + 1e-8)
    data[~mask] = 0.0
    return data


def load_and_preprocess(data_dir: str, config: dict):
    """
    完整的加载与预处理流程。

    返回:
        dataset: IntentRecognitionDataset 实例
        train_indices, val_indices, test_indices: 划分索引
        stats: 统计量字典
    """
    cfg_data = config["data"]
    max_step_features = cfg_data["max_step_features"]

    import os

    # 解析数据
    real_path = os.path.join(data_dir, cfg_data["x_real_file"])
    rec_path = os.path.join(data_dir, cfg_data["x_recurrence_file"])
    y_path = os.path.join(data_dir, cfg_data["y_file"])

    print(f"[数据加载] 解析 {real_path} ...")
    X_real, masks_real = parse_csv_to_tensor(real_path, max_step_features)
    print(f"  X_real 形状: {X_real.shape}")

    print(f"[数据加载] 解析 {rec_path} ...")
    X_recurrence, masks_rec = parse_csv_to_tensor(rec_path, max_step_features)
    print(f"  X_recurrence 形状: {X_recurrence.shape}")

    print(f"[数据加载] 解析 {y_path} ...")
    n_labels, dist_labels, phi_labels, label_params = parse_y_csv(y_path, config)
    print(f"  标签数量: {len(n_labels)}")
    if config.get("label_transform", {}).get("dist_log", False):
        print(f"  [变换] min_distance → log(dist)")
    if config.get("label_transform", {}).get("phi_cos_sin", False):
        print(f"  [变换] phi → (cosφ, sinφ) 双通道")

    # 掩码
    masks = masks_real

    # 按特征位置分别标准化（34 个位置各含不同物理量，量级差异大）
    real_mean, real_std = compute_grouped_statistics(X_real, cfg_data["max_targets"])
    rec_mean, rec_std = compute_grouped_statistics(X_recurrence, cfg_data["max_targets"])
    print(f"\n[标准化] X_real 前6位置均值(状态): {real_mean[:6]}")
    print(f"[标准化] X_real 距离位置均值: {real_mean[24:28]}")
    print(f"[标准化] X_real 夹角位置均值: {real_mean[28:]}")
    print(f"[标准化] X_real 夹角位置标准差: {real_std[28:]}")

    X_real = normalize_position(X_real, real_mean, real_std)
    X_recurrence = normalize_position(X_recurrence, rec_mean, rec_std)

    # 标签统计
    print(f"\n[标签统计]")
    print(f"  N=2: {(n_labels == 0).sum()} ({(n_labels == 0).mean()*100:.1f}%)")
    print(f"  N=3: {(n_labels == 1).sum()} ({(n_labels == 1).mean()*100:.1f}%)")
    print(f"  N=4: {(n_labels == 2).sum()} ({(n_labels == 2).mean()*100:.1f}%)")
    if dist_labels.ndim == 1:
        print(f"  dist_label: [{dist_labels.min():.4f}, {dist_labels.max():.4f}]")
    if phi_labels.ndim == 1:
        print(f"  phi_label: [{phi_labels.min():.4f}, {phi_labels.max():.4f}]")

    # 构建数据集
    dataset = IntentRecognitionDataset(X_real, X_recurrence, masks,
                                       n_labels, dist_labels, phi_labels)

    # 同步划分
    split_cfg = config["split"]
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=split_cfg["val_ratio"] + split_cfg["test_ratio"],
        random_state=split_cfg["random_seed"],
        shuffle=True,
    )
    val_ratio_in_temp = split_cfg["val_ratio"] / (split_cfg["val_ratio"] + split_cfg["test_ratio"])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_ratio_in_temp,
        random_state=split_cfg["random_seed"],
        shuffle=True,
    )

    print(f"\n[数据集划分] 训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")

    stats = {
        "real_mean": real_mean.tolist(),
        "real_std": real_std.tolist(),
        "rec_mean": rec_mean.tolist(),
        "rec_std": rec_std.tolist(),
        "label_params": label_params,
        "label_transform": config.get("label_transform", {}),
    }

    return dataset, train_idx, val_idx, test_idx, stats


def create_dataloaders(dataset, train_idx, val_idx, test_idx, config: dict):
    """创建训练/验证/测试 DataLoader。"""
    cfg = config["training"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
