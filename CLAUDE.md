# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航天器相对运动交会/博弈的**意图识别**深度学习项目。输入追踪航天器的真实轨迹 (`X_real`) 和 CW 方程递推预测轨迹 (`X_recurrence`)，输出三个标签：目标数量 N（分类）、全局最小距离 min_distance（回归）、最近与最远目标夹角 phi（回归）。

数据规模 23,787 样本，N 分布为 2(46%)/3(31%)/4(23%)。每个样本 10 个时间步，每步包含 N×6 维状态 (x,y,z,vx,vy,vz)。N 可变导致输入变长。

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
CSV (嵌套列表字符串) → ast.literal_eval 逐行解析 → pad 至 max_N=4 → 按 6 特征维分别 z-score
→ IntentRecognitionDataset → 同步随机划分 (70/15/15, seed=42) → DataLoader
```

CSV 每行是**单列嵌套列表字符串**，逗号在内层列表中，因此**不能用 `pd.read_csv` 默认逗号分隔**，必须逐行 `ast.literal_eval` 解析。三个 CSV 行序严格对应，划分时使用同一 `random_state` 和同一索引数组。

### 标准化策略

**按特征维度分别 z-score**（非全局标量）。reshape 为 `(N, 10, 4, 6)`，在最后一维（6 个物理量：x,y,z,vx,vy,vz）上分别计算均值/标准差（仅对非零位置），然后 tile 回 24 维。这样位置 (km 量级) 和速度 (km/s 量级) 各自被正确归一化。**零填充位置在标准化后强制恢复为 0**，使模型能通过零值识别 padding。

### 模型: Conv2D + Transformer v4 (`models/conv_transformer.py`)

核心改进：**N 分类使用独立的早分支**（从 Conv 特征直接经轻量 MLP 预测），避免回归任务的梯度干扰 N 分类学习。

```
X_real (B,10,24) + X_recurrence (B,10,24)
  → cat → (B,10,48) → unsqueeze → (B,1,10,48)
  → 3×ConvBlock(1→32→64→128) → (B,128,10,48)
  → MaxPool2d((1,2)) → (B,128,10,24)              # 仅特征维池化，保持时间维
  → ──┬─ [N早分支] AdaptiveAvgPool2d → Flatten → Linear(128,64) → Linear(64,3)  # N 分类
       │
       └─ [回归分支] reshape → (B,10,3072) → Linear → (B,10,256)
           → PositionalEncoding → 3×TransformerEncoder(d=256, h=8)
           → mean(dim=1) → (B,256) → shared_fc → (B,256)
           → ├─ dist_head: Linear(256,128)→ReLU→Linear(128,1)           # min_distance
              └─ phi_head:  Linear(256,128)→ReLU→Linear(128,2)           # (cosφ, sinφ)
```

关键设计：
- **N 早分支独立**：N 分类与回归分支不共享梯度路径（除 Conv 层），彻底解决训练中 N 准确率崩塌的问题
- **仅特征维池化**：时间维始终保持 10，使 Transformer self-attention 能跨时间步建模
- **phi 用 cos/sin 双通道编码**：输出 (cosφ, sinφ)，推理时 atan2 还原，避免 [0,π] 边界问题

训练: AdamW(lr=1e-3, wd=1e-4) + ReduceLROnPlateau(patience=15, factor=0.5) + 早停(patience=30) + 梯度裁剪(max_norm=1.0)。损失: `1.0×CE(N) + 1.0×Huber(dist) + 1.0×MSE(cos_sin_phi)`。

### checkpoint 内容

`best_model.pt` 包含 `model_state_dict`、`stats`（标准化参数）、`config` 和标签变换参数 `label_transform`，确保 evaluate/predict 可独立运行，无需重新计算统计量。

## 实验结果

| 任务 | 指标 | 值 |
|------|------|-----|
| N 分类 | 准确率 | 100%（零填充直接暴露目标数）|
| min_distance | MAE / R² | 0.63 km / 0.42（比猜均值好 26%）|
| phi | MAE / R² | 0.55 rad / ~0（完全未学到）|

## 已知问题与改进方向

- **N 分类已解决**：v4 早分支架构保证训练全程 100%。
- **min_distance R²≈0.42**：中等水平。可尝试按 N 分组训练子模型、增大 Transformer 容量、或显式计算几何特征。
- **phi R²≈0 是本任务核心瓶颈**。phi 是"最近与最远目标位置向量夹角"——这是高阶几何量，需先定位最近/最远目标再计算角度，Conv2D 的局部特征不足以完成此推理链。最有效的改进方向是**显式几何特征工程**：在输入端直接计算每时间步各目标间距离和角度，作为额外特征通道送入模型，而非期望网络从 x/y/z 坐标中隐式学习几何关系。

## 实验历史

| 版本 | 方案 | 结果 |
|------|------|------|
| v1 | 2.6M 参数，Huber 损失，单头共享 | N=100%, dist R²=0.41, phi R²≈0，N 在 epoch 8 崩塌 |
| v2 | 16.8M 参数 + log-dist + cos/sin phi | 完全崩溃，N 始终 45% |
| v2b | 6.6M 参数 + log-dist + cos/sin phi | N=100%, dist R²=0.005（log 反而恶化）|
| v3 | v1 架构 + cos/sin phi | N 在 epoch 18 崩塌，与 v1 同模式 |
| v4 | v1 大小 + N 早分支 + cos/sin phi | **最佳**：N 不崩塌，dist R²=0.42，phi 仍未学到 |
