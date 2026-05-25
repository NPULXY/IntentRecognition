"""
评估脚本。

加载训练好的最佳模型，在测试集上计算详细评估指标：
  - N 分类：准确率、精确率、召回率、F1（各类别及宏平均）
  - min_distance 回归：MAE、MSE、RMSE、R²
  - phi 回归：MAE、MSE、RMSE、R²
"""

import os
import sys
import yaml
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_and_preprocess, create_dataloaders
from models.conv_transformer import Conv2DTransformer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config_path: str = "config.yaml"):
    """在测试集上评估最佳模型。"""
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ---- 加载数据 ----
    dataset, train_idx, val_idx, test_idx, stats = load_and_preprocess(
        config["data"]["data_dir"], config
    )
    _, _, test_loader = create_dataloaders(dataset, train_idx, val_idx, test_idx, config)

    # ---- 加载模型 ----
    model_path = os.path.join(
        config["logging"]["model_save_dir"],
        config["logging"]["best_model_name"],
    )
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型。")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = Conv2DTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint.get("stats", {})

    print(f"\n[模型加载] {model_path}")
    print(f"[训练轮次] Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

    # ---- 收集预测结果（变换空间） ----
    all_n_pred, all_n_label = [], []
    all_dist_pred_raw, all_dist_label_raw = [], []
    all_phi_pred_raw, all_phi_label_raw = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_real = batch["x_real"].to(device)
            x_rec = batch["x_recurrence"].to(device)
            mask = batch["mask"].to(device)
            n_label = batch["n_label"].to(device)
            dist_label = batch["dist_label"].to(device)
            phi_label = batch["phi_label"].to(device)

            n_logits, dist_pred, phi_pred = model(x_real, x_rec, mask)

            n_pred = torch.argmax(n_logits, dim=-1)

            all_n_pred.append(n_pred.cpu().numpy())
            all_n_label.append(n_label.cpu().numpy())
            all_dist_pred_raw.append(dist_pred.squeeze(-1).cpu().numpy())
            all_dist_label_raw.append(dist_label.cpu().numpy())
            all_phi_pred_raw.append(phi_pred.cpu().numpy())
            all_phi_label_raw.append(phi_label.cpu().numpy())

    n_pred = np.concatenate(all_n_pred)
    n_label = np.concatenate(all_n_label)
    dist_pred_raw = np.concatenate(all_dist_pred_raw)
    dist_label_raw = np.concatenate(all_dist_label_raw)
    phi_pred_raw = np.concatenate(all_phi_pred_raw)
    phi_label_raw = np.concatenate(all_phi_label_raw)

    # ---- 逆变换至物理空间 ----
    lt = stats.get("label_transform", {})
    if lt.get("dist_log", False):
        dist_pred = np.exp(dist_pred_raw)
        dist_label = np.exp(dist_label_raw)
    else:
        dist_pred = dist_pred_raw
        dist_label = dist_label_raw

    if lt.get("phi_cos_sin", False):
        # phi_pred_raw: (N, 2) cos/sin, phi_label_raw: (N, 2) cos/sin
        cos_p, sin_p = phi_pred_raw[:, 0], phi_pred_raw[:, 1]
        cos_l, sin_l = phi_label_raw[:, 0], phi_label_raw[:, 1]
        phi_pred = np.arctan2(sin_p, cos_p) % (2 * np.pi)
        phi_label = np.arctan2(sin_l, cos_l) % (2 * np.pi)
    else:
        phi_pred = phi_pred_raw
        phi_label = phi_label_raw

    # ---- N 分类指标 ----
    n_acc = accuracy_score(n_label, n_pred)
    n_precision, n_recall, n_f1, _ = precision_recall_fscore_support(
        n_label, n_pred, average=None, labels=[0, 1, 2]
    )
    n_precision_macro, n_recall_macro, n_f1_macro, _ = precision_recall_fscore_support(
        n_label, n_pred, average="macro"
    )
    cm = confusion_matrix(n_label, n_pred)

    # ---- min_distance 回归指标 ----
    dist_mae = mean_absolute_error(dist_label, dist_pred)
    dist_mse = mean_squared_error(dist_label, dist_pred)
    dist_rmse = np.sqrt(dist_mse)
    dist_r2 = r2_score(dist_label, dist_pred)

    # ---- phi 回归指标 ----
    phi_mae = mean_absolute_error(phi_label, phi_pred)
    phi_mse = mean_squared_error(phi_label, phi_pred)
    phi_rmse = np.sqrt(phi_mse)
    phi_r2 = r2_score(phi_label, phi_pred)

    # ---- 输出结果 ----
    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")

    print(f"\n── N 分类 (目标数量) ──")
    print(f"  整体准确率: {n_acc:.4f}")
    print(f"  混淆矩阵:")
    print(f"           预测=2  预测=3  预测=4")
    for i, true_n in enumerate([2, 3, 4]):
        print(f"  真实={true_n}  {cm[i, 0]:5d}   {cm[i, 1]:5d}   {cm[i, 2]:5d}")
    print(f"  各类别指标:")
    for i, n_val in enumerate([2, 3, 4]):
        print(f"    N={n_val}: Precision={n_precision[i]:.4f}, "
              f"Recall={n_recall[i]:.4f}, F1={n_f1[i]:.4f}")
    print(f"  宏平均: Precision={n_precision_macro:.4f}, "
          f"Recall={n_recall_macro:.4f}, F1={n_f1_macro:.4f}")

    print(f"\n── min_distance 回归 ──")
    print(f"  MAE:  {dist_mae:.4f} km")
    print(f"  MSE:  {dist_mse:.4f} km^2")
    print(f"  RMSE: {dist_rmse:.4f} km")
    print(f"  R2:   {dist_r2:.4f}")

    print(f"\n── phi 回归 ──")
    print(f"  MAE:  {phi_mae:.4f} rad")
    print(f"  MSE:  {phi_mse:.4f} rad^2")
    print(f"  RMSE: {phi_rmse:.4f} rad")
    print(f"  R2:   {phi_r2:.4f}")

    print(f"\n{'='*60}")

    # 保存结果
    results = {
        "n_accuracy": float(n_acc),
        "n_precision_per_class": n_precision.tolist(),
        "n_recall_per_class": n_recall.tolist(),
        "n_f1_per_class": n_f1.tolist(),
        "n_precision_macro": float(n_precision_macro),
        "n_recall_macro": float(n_recall_macro),
        "n_f1_macro": float(n_f1_macro),
        "confusion_matrix": cm.tolist(),
        "dist_mae": float(dist_mae),
        "dist_mse": float(dist_mse),
        "dist_rmse": float(dist_rmse),
        "dist_r2": float(dist_r2),
        "phi_mae": float(phi_mae),
        "phi_mse": float(phi_mse),
        "phi_rmse": float(phi_rmse),
        "phi_r2": float(phi_r2),
    }

    import json
    results_path = os.path.join(config["logging"]["log_dir"], "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[结果保存] {results_path}")


if __name__ == "__main__":
    evaluate()
