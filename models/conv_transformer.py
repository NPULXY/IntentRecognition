"""
Conv2D + Transformer 混合模型 (v4)。

关键改进：N 分类使用独立浅层分支（从 Conv 特征直接预测），
避免回归梯度干扰 N 分类的学习。

架构：
  1. Conv2D 阶段：共享卷积提取时空特征
  2. N 分支：Conv 特征 → 轻量 MLP → N 分类（早分支，不受回归影响）
  3. Transformer 阶段：Conv 特征 → Transformer → 回归输出
  4. 多任务输出：N(分类) + dist(回归) + phi_cos_sin(回归)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU 基本块。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, padding: int,
                 stride: tuple = (1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv2DTransformer(nn.Module):
    """
    Conv2D + Transformer 混合模型 (v4)。

    N 分类使用独立早分支，避免回归梯度干扰。

    输入:
        x_real:       (B, 10, max_N * 6)
        x_recurrence: (B, 10, max_N * 6)

    输出:
        n_logits:    (B, 3)  —— N 分类 logits（早分支）
        dist_pred:   (B, 1)  —— min_distance 预测
        phi_cs:      (B, 2)  —— (cosφ, sinφ) 预测
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        cfg_data = config["data"]

        max_step_features = cfg_data["max_step_features"]
        self.input_width = max_step_features * 2    # real + recurrence

        conv_channels = cfg["conv_channels"]
        conv_kernel = cfg["conv_kernel"]
        conv_padding = cfg["conv_padding"]

        # ---- 共享 Conv2D 阶段 ----
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(ConvBlock(in_ch, out_ch, conv_kernel, conv_padding))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*layers)
        self.feature_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        pooled_width = self.input_width
        for _ in range(1):
            pooled_width = pooled_width // 2
        self.pooled_width = pooled_width
        self.conv_out_channels = conv_channels[-1]

        # ---- N 分类早分支（独立，浅层） ----
        n_hidden = conv_channels[-1] // 2
        self.n_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),       # (B, C, 1, 1)
            nn.Flatten(),                         # (B, C)
            nn.Linear(conv_channels[-1], n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, 3),              # (B, 3)
        )

        # ---- Transformer 阶段（回归专用） ----
        d_model = cfg["d_model"]
        nhead = cfg["nhead"]
        num_layers = cfg["num_encoder_layers"]
        dim_feedforward = cfg["dim_feedforward"]
        dropout = cfg["dropout"]

        self.input_proj = nn.Linear(self.conv_out_channels * self.pooled_width, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- 回归输出头 ----
        hidden_dim = cfg["hidden_dim"]
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x_real: torch.Tensor, x_recurrence: torch.Tensor,
                mask: torch.Tensor = None):
        B = x_real.size(0)
        time_steps = x_real.size(1)

        x = torch.cat([x_real, x_recurrence], dim=-1)
        x = x.unsqueeze(1)

        # 共享 Conv2D
        conv_feat = self.conv_blocks(x)          # (B, C_out, 10, W)
        conv_feat = self.feature_pool(conv_feat)  # (B, C_out, 10, W/2)

        # ---- N 分类早分支 ----
        n_logits = self.n_branch(conv_feat)       # (B, 3)

        # ---- Transformer 回归分支 ----
        # conv_feat → sequence → Transformer
        seq = conv_feat.permute(0, 2, 1, 3).reshape(B, time_steps, -1)
        seq = self.input_proj(seq)
        seq = self.pos_encoder(seq)
        seq = self.transformer_encoder(seq)

        pooled = seq.mean(dim=1)
        shared = self.shared_fc(pooled)

        dist_pred = self.dist_head(shared)         # (B, 1)
        phi_cs = self.phi_head(shared)             # (B, 2)

        return n_logits, dist_pred, phi_cs


class MultiTaskLoss(nn.Module):
    """
    多任务组合损失 (v4):
      - N: 交叉熵（早分支，独立梯度路径）
      - min_distance: Huber 损失
      - phi: MSE on (cosφ, sinφ) 双通道
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["loss"]
        self.w_n = cfg["n_weight"]
        self.w_dist = cfg["dist_weight"]
        self.w_phi = cfg["phi_weight"]
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, n_logits, dist_pred, phi_cs_pred,
                n_label, dist_label, phi_cs_label):
        loss_n = self.ce_loss(n_logits, n_label)
        loss_dist = self.huber_loss(dist_pred.squeeze(-1), dist_label)
        loss_phi = self.mse_loss(phi_cs_pred, phi_cs_label)

        total = self.w_n * loss_n + self.w_dist * loss_dist + self.w_phi * loss_phi
        return total, {"loss_n": loss_n.item(),
                       "loss_dist": loss_dist.item(),
                       "loss_phi": loss_phi.item()}
