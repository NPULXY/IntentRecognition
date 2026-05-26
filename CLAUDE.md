# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航天器相对运动交会/博弈的**意图识别**深度学习项目。输入追踪航天器的真实轨迹 (`X_real`) 和 CW 方程递推预测轨迹 (`X_recurrence`)，输出三个标签：目标数量 N（分类）、全局最小距离 min_distance（回归）、最近与最远目标夹角 phi（回归）。

数据规模 23,787 样本，N 分布为 2(46%)/3(31%)/4(23%)。每个样本 10 个时间步，N 可变导致输入变长。

## 数据集

当前 `Dataset/` 使用**增强版数据集**，每步包含：

| N | 状态数 (x,y,z,vx,vy,vz) | 距离数 (到原点) | 夹角数 (两两) | 单步总特征 |
|:-:|:---:|:---:|:---:|:---:|
| 2 | 12 | 2 | 1 | 15 |
| 3 | 18 | 3 | 3 | 24 |
| 4 | 24 | 4 | 6 | **34** |

距离和夹角是确定性几何量，从状态值派生。原版数据集（仅 N×6 状态，无几何特征）在 `Dataset.zip` 中。

## 常用命令

```bash
python train.py              # 训练模型（读取 config.yaml）
python evaluate.py           # 在测试集上评估已保存的最佳模型
python predict.py --real <path> --rec <path> --output <path>  # 批量推理
python predict.py --raw-real "<nested_list>" --raw-rec "<nested_list>"  # 单样本推理
```

所有脚本从项目根目录运行，自动将根目录加入 `sys.path`。

## 架构

### 数据流

```
CSV (嵌套列表字符串) → ast.literal_eval 逐行解析 → pad_step 至 34
→ 分组 z-score (状态/距离/夹角各自归一化) → IntentRecognitionDataset
→ 同步随机划分 (70/15/15, seed=42) → DataLoader
```

CSV 每行是**单列嵌套列表字符串**，逗号在内层列表中，因此**不能用 `pd.read_csv` 默认逗号分隔**，必须逐行 `ast.literal_eval` 解析。三个 CSV 行序严格对应，划分时使用同一 `random_state` 和同一索引数组。

### 标准化策略

**分组标准化**（`compute_grouped_statistics`）：34 个特征分三组：
- **状态 (0:24)**：6 物理量 × 4 目标，按 6 物理量分别统计后 tile
- **距离 (24:28)**：同物理量（km），全局统计后 tile
- **夹角 (28:34)**：同物理量（rad），全局统计后 tile

零填充位置标准化后强制恢复为 0，使模型能识别 padding。

### 模型: Conv2D + Transformer v4 (`models/conv_transformer.py`)

核心改进：**N 分类使用独立的早分支**（从 Conv 特征直接经轻量 MLP 预测），避免回归任务的梯度干扰 N 分类学习。

```
X_real (B,10,34) + X_recurrence (B,10,34)
  → cat → (B,10,68) → unsqueeze → (B,1,10,68)
  → 4×ConvBlock(1→48→96→192→256) → (B,256,10,68)
  → MaxPool2d((1,2)) → (B,256,10,34)
  → ──┬─ [N早分支] AdaptiveAvgPool2d → Flatten → Linear(256,128) → Linear(128,3)
       │
       └─ [回归分支] reshape → (B,10,8704) → Linear → (B,10,256)
           → PositionalEncoding → 3×TransformerEncoder(d=256, h=8)
           → mean(dim=1) → (B,256) → shared_fc → (B,256)
           → ├─ dist_head: Linear(256,128)→ReLU→Linear(128,1)         # min_distance
              └─ phi_head:  Linear(256,128)→ReLU→Linear(128,2)         # (cosφ, sinφ)
```

关键设计：
- **N 早分支独立**：N 分类与回归分支不共享梯度路径（除 Conv 层），彻底解决训练中 N 准确率崩塌的问题
- **仅特征维池化**：时间维始终保持 10，使 Transformer self-attention 能跨时间步建模
- **phi 用 cos/sin 双通道编码**：输出 (cosφ, sinφ)，推理时 atan2 还原，避免 [0,π] 边界问题

训练: AdamW(lr=1e-3, wd=1e-4) + ReduceLROnPlateau(patience=15, factor=0.5) + 早停(patience=30) + 梯度裁剪(max_norm=1.0)。损失: `1.0×CE(N) + 1.0×Huber(dist) + 1.0×MSE(cos_sin_phi)`。

### checkpoint 内容

`best_model.pt` 包含 `model_state_dict`、`stats`（标准化参数，含分组统计量）、`config` 和 `label_transform`，确保 evaluate/predict 可独立运行。

## 实验结果（最佳：v4 + 原版数据）

| 任务 | 指标 | 值 |
|------|------|-----|
| N 分类 | 准确率 | 100% |
| min_distance | MAE / R² | 0.63 km / 0.42 |
| phi | MAE / R² | 0.55 rad / ~0（完全未学到）|

## 已知问题与改进方向

- **N 分类已解决**：v4 早分支架构保证训练全程 100%。
- **min_distance R²≈0.42**：中等水平，比猜均值好 26%。仍有提升空间。
- **phi R²≈0 是本任务核心瓶颈**。phi = "全局最小距离时刻，最近与最远目标位置向量夹角"。这需要：
  1. 找到全局最小距离对应的时刻 → argmin
  2. 在该时刻找到最近/最远目标 → argmin/argmax
  3. 查找对应夹角 → lookup

  这些都是离散选择操作，神经网络天生不擅长。显式嵌入每步距离和两两夹角（增强数据集）**没有改善** phi，因为数据虽有但模型无法学会"选择"。`phi-improvement` 分支尝试预计算每步候选 phi 也无效——仍需选对时间步。

  突破方向可能需要**两阶段模型**（先预测 min_distance 时间位置，再查找 phi）或**软注意力机制**替代硬 argmin/argmax。

## 实验历史

| 版本 | 数据 | 架构 | N | dist R² | phi R² | 备注 |
|------|------|------|:--:|:-------:|:------:|------|
| v1 | 原版 | 2.6M, 单头共享 | 100% | 0.41 | ~0 | N 在 epoch 8 崩塌 |
| v2 | 原版 | 16.8M + log-dist + cos/sin | 45% | — | — | 完全崩溃 |
| v2b | 原版 | 6.6M + log-dist + cos/sin | 100% | 0.005 | ~0 | log 反而恶化 |
| v3 | 原版 | v1 + cos/sin phi | 100% | — | ~0 | N 在 epoch 18 崩塌 |
| **v4** | **原版** | **N 早分支 + cos/sin phi** | **100%** | **0.42** | **~0** | **最佳** |
| — | 增强版 | v4 + 更大 Conv | 100% | 0.41 | ~0 | 几何特征未提升 |
| — | 增强版 | + 候选 phi 特征 | 100% | ~0 | ~0 | phi-improvement 分支 |

## 分支

- `master`：当前主分支，适配增强版数据集（34 特征），v4 架构
- `phi-improvement`：预计算每步最近最远夹角作为第 35 维特征的实验（失败）
