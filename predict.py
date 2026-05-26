"""
推理脚本。

对新的输入数据进行预测。支持两种模式：
  1. 从 CSV 文件加载并批量推理
  2. 从原始嵌套列表数据进行单样本推理

用法示例:
  python predict.py --real Data/X_real.csv --rec Data/X_recurrence.csv
  python predict.py --raw-real "[[...], [...], ...]" --raw-rec "[[...], [...], ...]"
"""

import os
import sys
import ast
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import (
    parse_csv_to_tensor, parse_nested_list_string,
    pad_step, normalize_position, _append_nearest_farthest_angle,
)
from models.conv_transformer import Conv2DTransformer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(config: dict, device: torch.device):
    """加载训练好的最佳模型。"""
    model_path = os.path.join(
        config["logging"]["model_save_dir"],
        config["logging"]["best_model_name"],
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 train.py")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = Conv2DTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint["stats"]
    return model, stats


def preprocess_sample(raw_real, raw_rec, config, stats):
    """
    将原始嵌套列表数据预处理为模型输入张量。

    参数:
        raw_real: 长度为 10 的列表，每个元素含 N×6+N+C(N,2) 个浮点数
        raw_rec: 同上
        config: 全局配置
        stats: 标准化统计量字典

    返回:
        x_real: (1, 10, max_step_features) torch.Tensor
        x_recurrence: (1, 10, max_step_features) torch.Tensor
        mask: (1, 10, max_step_features) torch.Tensor
    """
    def process_one(raw, mean, std):
        mean_arr = np.array(mean).reshape(1, -1)
        std_arr = np.array(std).reshape(1, -1)
        time_steps_data = []
        time_steps_mask = []
        max_len = config["data"]["max_step_features"]
        for step in raw:
            step_arr = np.array(step, dtype=np.float32)
            step_with_nf = _append_nearest_farthest_angle(step_arr)
            padded = pad_step(step_with_nf, max_len)
            mask_vals = padded != 0
            padded = padded.copy()
            padded = (padded - mean_arr) / (std_arr + 1e-8)
            padded[~mask_vals] = 0.0
            time_steps_data.append(padded)
            time_steps_mask.append(mask_vals.astype(np.float32))

        return (np.stack(time_steps_data, axis=0),
                np.stack(time_steps_mask, axis=0))

    real_data, mask_real = process_one(raw_real, stats["real_mean"], stats["real_std"])
    rec_data, mask_rec = process_one(raw_rec, stats["rec_mean"], stats["rec_std"])

    x_real = torch.from_numpy(real_data).float().unsqueeze(0)
    x_rec = torch.from_numpy(rec_data).float().unsqueeze(0)
    mask = torch.from_numpy(mask_real).float().unsqueeze(0)

    return x_real, x_rec, mask


def predict_single(model, x_real, x_rec, mask, device, stats=None):
    """对单个样本推理，输出物理空间的预测值。"""
    x_real = x_real.to(device)
    x_rec = x_rec.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        n_logits, dist_pred_raw, phi_pred_raw = model(x_real, x_rec, mask)

    n_probs = torch.softmax(n_logits, dim=-1).cpu().numpy()[0]
    n_pred = int(np.argmax(n_probs)) + 2

    # 逆变换
    lt = stats.get("label_transform", {}) if stats else {}
    if lt.get("dist_log", False):
        dist_val = float(np.exp(dist_pred_raw.cpu().item()))
    else:
        dist_val = dist_pred_raw.cpu().item()

    if lt.get("phi_cos_sin", False):
        cs = phi_pred_raw.cpu().numpy()[0]
        phi_val = float(np.arctan2(cs[1], cs[0]) % (2 * np.pi))
    else:
        phi_val = phi_pred_raw.cpu().item()

    return {
        "N": n_pred,
        "N_probabilities": {"N=2": float(n_probs[0]),
                            "N=3": float(n_probs[1]),
                            "N=4": float(n_probs[2])},
        "min_distance": round(dist_val, 4),
        "phi": round(phi_val, 4),
    }


def predict_batch(model, real_csv_path, rec_csv_path, config, stats, device):
    """从 CSV 文件批量推理，输出物理空间预测值。"""
    max_step_features = config["data"]["max_step_features"]

    print(f"[数据加载] 解析 {real_csv_path} ...")
    real_data, _ = parse_csv_to_tensor(real_csv_path, max_step_features)
    rec_data, _ = parse_csv_to_tensor(rec_csv_path, max_step_features)

    real_data = normalize_position(real_data,
                                   np.array(stats["real_mean"]),
                                   np.array(stats["real_std"]))
    rec_data = normalize_position(rec_data,
                                   np.array(stats["rec_mean"]),
                                   np.array(stats["rec_std"]))

    x_real = torch.from_numpy(real_data).float()
    x_rec = torch.from_numpy(rec_data).float()
    mask = torch.zeros(x_real.size(0), config["data"]["time_steps"], max_step_features)

    batch_size = config["training"]["batch_size"]
    n_preds, dist_preds, phi_preds = [], [], []
    lt = stats.get("label_transform", {})

    for i in range(0, len(x_real), batch_size):
        end = min(i + batch_size, len(x_real))
        xr = x_real[i:end].to(device)
        xc = x_rec[i:end].to(device)
        mk = mask[i:end].to(device)

        with torch.no_grad():
            n_logits, dist_pred_raw, phi_pred_raw = model(xr, xc, mk)

        n_pred = torch.argmax(n_logits, dim=-1).cpu().numpy() + 2
        n_preds.extend(n_pred.tolist())

        dp = dist_pred_raw.squeeze(-1).cpu().numpy()
        if lt.get("dist_log", False):
            dp = np.exp(dp)
        dist_preds.extend(dp.tolist())

        pp = phi_pred_raw.cpu().numpy()
        if lt.get("phi_cos_sin", False):
            phi_vals = np.arctan2(pp[:, 1], pp[:, 0]) % (2 * np.pi)
            phi_preds.extend(phi_vals.tolist())
        else:
            phi_preds.extend(pp.squeeze(-1).tolist())

    return n_preds, dist_preds, phi_preds


def main():
    parser = argparse.ArgumentParser(description="意图识别模型推理")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--real", help="X_real CSV 文件路径（批量模式）")
    parser.add_argument("--rec", help="X_recurrence CSV 文件路径（批量模式）")
    parser.add_argument("--raw-real", help="原始 X_real 嵌套列表字符串（单样本模式）")
    parser.add_argument("--raw-rec", help="原始 X_recurrence 嵌套列表字符串（单样本模式）")
    parser.add_argument("--output", help="批量模式下的输出 CSV 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    model, stats = load_model(config, device)
    print("[模型] 已加载最佳模型")

    if args.raw_real and args.raw_rec:
        # 单样本模式
        raw_real = parse_nested_list_string(args.raw_real)
        raw_rec = parse_nested_list_string(args.raw_rec)

        x_real, x_rec, mask = preprocess_sample(
            raw_real, raw_rec, config, stats
        )
        result = predict_single(model, x_real, x_rec, mask, device, stats)
        print(f"\n预测结果:")
        print(f"  目标数量 N: {result['N']}")
        print(f"  类别概率: {result['N_probabilities']}")
        print(f"  最小距离: {result['min_distance']} km")
        print(f"  夹角 phi: {result['phi']} rad ({result['phi']*180/np.pi:.2f}°)")

    elif args.real and args.rec:
        # 批量模式
        n_preds, dist_preds, phi_preds = predict_batch(
            model, args.real, args.rec, config, stats, device
        )
        print(f"\n批量推理完成，共 {len(n_preds)} 条样本。")

        if args.output:
            import pandas as pd
            df = pd.DataFrame({
                "N_pred": n_preds,
                "min_distance_pred": dist_preds,
                "phi_pred": phi_preds,
            })
            df.to_csv(args.output, index=False)
            print(f"结果已保存至 {args.output}")
        else:
            # 打印前 10 条
            print("\n前 10 条预测结果:")
            print(f"{'序号':<6} {'N':<4} {'min_distance':<14} {'phi':<10}")
            for i in range(min(10, len(n_preds))):
                print(f"{i:<6} {n_preds[i]:<4} {dist_preds[i]:<14.4f} {phi_preds[i]:<10.4f}")
    else:
        print("请指定 --real/--rec（批量模式）或 --raw-real/--raw-rec（单样本模式）")


if __name__ == "__main__":
    main()
