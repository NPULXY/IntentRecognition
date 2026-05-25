"""
训练脚本。

功能:
  - 加载配置和数据
  - 构建 Conv2D + Transformer 模型
  - 训练循环（含早停和学习率调度）
  - 保存最佳模型和训练日志
"""

import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_and_preprocess, create_dataloaders
from models.conv_transformer import Conv2DTransformer, MultiTaskLoss


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_dirs(config: dict):
    """创建日志和模型保存目录。"""
    for key in ["log_dir", "model_save_dir"]:
        path = config["logging"][key]
        os.makedirs(path, exist_ok=True)


def compute_metrics(n_logits, dist_pred, phi_pred, n_label, dist_label, phi_label,
                    stats=None):
    """计算人类可读的评估指标（逆变换后）。"""
    n_pred = torch.argmax(n_logits, dim=-1)
    n_acc = (n_pred == n_label).float().mean().item()

    # 逆变换 min_distance
    lt = stats.get("label_transform", {}) if stats else {}
    if lt.get("dist_log", False):
        dist_pred_real = torch.exp(dist_pred.squeeze(-1))
        dist_label_real = torch.exp(dist_label)
    else:
        dist_pred_real = dist_pred.squeeze(-1)
        dist_label_real = dist_label
    dist_mae = torch.abs(dist_pred_real - dist_label_real).mean().item()

    # 逆变换 phi
    if lt.get("phi_cos_sin", False):
        cos_p, sin_p = phi_pred[:, 0], phi_pred[:, 1]
        cos_l, sin_l = phi_label[:, 0], phi_label[:, 1]
        phi_pred_real = torch.atan2(sin_p, cos_p) % (2 * torch.pi)
        phi_label_real = torch.atan2(sin_l, cos_l) % (2 * torch.pi)
        # 处理角度环绕
        diff = torch.abs(phi_pred_real - phi_label_real)
        diff = torch.min(diff, 2 * torch.pi - diff)
        phi_mae = diff.mean().item()
    else:
        phi_pred_real = phi_pred.squeeze(-1)
        phi_label_real = phi_label
        phi_mae = torch.abs(phi_pred_real - phi_label_real).mean().item()

    return n_acc, dist_mae, phi_mae


def validate(model, dataloader, criterion, device, stats=None):
    """在验证集上评估模型。"""
    model.eval()
    total_loss = 0.0
    total_n_acc = 0.0
    total_dist_mae = 0.0
    total_phi_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x_real = batch["x_real"].to(device)
            x_rec = batch["x_recurrence"].to(device)
            mask = batch["mask"].to(device)
            n_label = batch["n_label"].to(device)
            dist_label = batch["dist_label"].to(device)
            phi_label = batch["phi_label"].to(device)

            n_logits, dist_pred, phi_pred = model(x_real, x_rec, mask)
            loss, _ = criterion(n_logits, dist_pred, phi_pred,
                               n_label, dist_label, phi_label)
            n_acc, dist_mae, phi_mae = compute_metrics(
                n_logits, dist_pred, phi_pred,
                n_label, dist_label, phi_label, stats
            )

            total_loss += loss.item()
            total_n_acc += n_acc
            total_dist_mae += dist_mae
            total_phi_mae += phi_mae
            num_batches += 1

    return (total_loss / num_batches,
            total_n_acc / num_batches,
            total_dist_mae / num_batches,
            total_phi_mae / num_batches)


def train(config_path: str = "config.yaml"):
    """主训练流程。"""
    config = load_config(config_path)
    setup_dirs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ---- 数据加载 ----
    dataset, train_idx, val_idx, test_idx, stats = load_and_preprocess(
        config["data"]["data_dir"], config
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, train_idx, val_idx, test_idx, config
    )

    # ---- 模型、损失、优化器 ----
    model = Conv2DTransformer(config).to(device)
    criterion = MultiTaskLoss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["training"]["lr_scheduler_patience"],
        factor=config["training"]["lr_scheduler_factor"],
        verbose=True,
    )

    num_epochs = config["training"]["num_epochs"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    log_interval = config["logging"]["log_interval"]

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_n_acc": [],
               "val_dist_mae": [], "val_phi_mae": []}

    print(f"\n{'='*60}")
    print(f"[训练开始] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[模型参数] {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    # ---- 训练循环 ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_n = 0.0
        epoch_loss_dist = 0.0
        epoch_loss_phi = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x_real = batch["x_real"].to(device)
            x_rec = batch["x_recurrence"].to(device)
            mask = batch["mask"].to(device)
            n_label = batch["n_label"].to(device)
            dist_label = batch["dist_label"].to(device)
            phi_label = batch["phi_label"].to(device)

            optimizer.zero_grad()
            n_logits, dist_pred, phi_pred = model(x_real, x_rec, mask)
            loss, loss_components = criterion(
                n_logits, dist_pred, phi_pred,
                n_label, dist_label, phi_label
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_n += loss_components["loss_n"]
            epoch_loss_dist += loss_components["loss_dist"]
            epoch_loss_phi += loss_components["loss_phi"]
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                n_acc, dist_mae, phi_mae = compute_metrics(
                    n_logits, dist_pred, phi_pred,
                    n_label, dist_label, phi_label, stats
                )
                print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | N Acc: {n_acc:.3f} | "
                      f"Dist MAE: {dist_mae:.4f} | Phi MAE: {phi_mae:.4f}")

        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # ---- 验证 ----
        val_loss, val_n_acc, val_dist_mae, val_phi_mae = validate(
            model, val_loader, criterion, device, stats
        )
        history["val_loss"].append(val_loss)
        history["val_n_acc"].append(val_n_acc)
        history["val_dist_mae"].append(val_dist_mae)
        history["val_phi_mae"].append(val_phi_mae)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  --- Epoch {epoch:3d} 完成 | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val N Acc: {val_n_acc:.4f} | Val Dist MAE: {val_dist_mae:.4f} | "
              f"Val Phi MAE: {val_phi_mae:.4f} | LR: {lr:.2e} | 耗时: {elapsed:.1f}s ---")

        # ---- 早停与模型保存 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_path = os.path.join(
                config["logging"]["model_save_dir"],
                config["logging"]["best_model_name"],
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "stats": stats,
                    "config": config,
                },
                save_path,
            )
            print(f"  >>> 保存最佳模型 (Epoch {epoch}, Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  未改善 ({epochs_no_improve}/{early_stopping_patience})")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[早停触发] 在 Epoch {epoch} 停止训练，最佳 Epoch: {best_epoch}")
            break

    # ---- 训练结束 ----
    print(f"\n{'='*60}")
    print(f"[训练结束] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[最佳模型] Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # 保存训练历史
    history_path = os.path.join(config["logging"]["log_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[日志] 训练历史已保存至 {history_path}")

    return model, test_loader, config, stats


if __name__ == "__main__":
    train()
