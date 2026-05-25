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

### 模型: Conv2D + Transformer (`models/conv_transformer.py`)

```
X_real (B,10,24) + X_recurrence (B,10,24)
  → cat → (B,10,48) → unsqueeze → (B,1,10,48)   # 单通道 2D 图
  → 3×ConvBlock(1→32→64→128) → (B,128,10,48)
  → MaxPool2d((1,2)) → (B,128,10,24)              # 仅特征维池化，保持时间维
  → reshape → (B,10, 128×24) → Linear → (B,10,256)
  → PositionalEncoding → 3×TransformerEncoder(d=256, h=8)
  → mean(dim=1) → (B,256) → shared_fc → (B,256)
  → ├─ n_head:   Linear(256,3)                              # N 分类
     ├─ dist_head: Linear(256,128)→ReLU→Linear(128,1)       # min_distance
     └─ phi_head:  Linear(256,128)→ReLU→Linear(128,1)→sigmoid×π  # phi∈[0,π]
```

关键设计：**不在时间维做池化**，保证进入 Transformer 的序列长度 = 10，使 self-attention 能跨时间步建模。参数量 ~260 万。

训练: AdamW(lr=1e-3, wd=1e-4) + ReduceLROnPlateau(patience=15, factor=0.5) + 早停(patience=30) + 梯度裁剪(max_norm=1.0)。损失: `1.0×CE(N) + 1.0×Huber(dist) + 1.0×Huber(phi)`。

### checkpoint 内容

`best_model.pt` 不仅保存 `model_state_dict`，还保存 `stats`（标准化参数）和 `config`，确保 evaluate/predict 可独立运行。

## 已知问题与改进方向

- **N 分类完美 (100%)**，因为零填充模式直接暴露了目标数。
- **min_distance R²≈0.41**：中等，比猜均值好 25%。
- **phi R²≈0 (完全失效)**：模型等价于预测均值。phi 是"最近与最远目标位置向量夹角"——这是高阶几何量，需先定位最近/最远目标再计算角度。Conv2D 的局部特征不足以完成此推理链。改进思路：显式计算每时间步的目标间距离/角度作为额外特征通道；用 cos/sin 双通道编码 phi；按 N 分组训练子模型。
