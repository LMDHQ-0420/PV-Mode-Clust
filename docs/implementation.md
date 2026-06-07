# Implementation Guide — 天气区制感知与 NWP 误差订正的软门控多专家光伏日前功率预测
> 生成时间：2026-06-07 | 策略：从头构建 | 状态：PENDING_REVIEW
> 关联实验设计：docs/idea_report.md Part 3

---

## 1 项目结构

> 旧 `code/` 代码弃用（仅借鉴关键函数逻辑），新代码直接覆盖 `code/`，保留 `code/dataset/`（原始 PVOD）。

### 1.1 完整目录树

```text
code/
├── dataset/                          # [KEEP] 原始 PVOD 数据（station*.csv, metadata.csv）
├── src/
│   ├── data/
│   │   ├── pvod_dataset.py           # PVOD 数据集类：读取、划分、滑窗，产出训练样本
│   │   ├── preprocess.py             # 原始 PVOD → 清洗对齐（异常值/缺失/夜间标记），落盘缓存
│   │   ├── fcm_regime.py             # FCM 软天气区制：训练中心、产出软隶属 u(t)（模块A）
│   │   ├── causal_vmd.py             # 因果滑窗 VMD：对历史功率/NWP 做无泄漏多尺度分解（模块C1）
│   │   └── transforms.py            # Max-Min 归一化（按训练集统计），反归一化
│   ├── models/
│   │   ├── corrector.py              # NWP 误差订正器 g(NWP,u)（模块B，第一阶段）
│   │   ├── experts.py               # 轻量专家 GRU/TCN（模块C2）
│   │   ├── gated_moe.py             # 软门控多专家整体模型（模块D，组装 A/C1/C2/D）
│   │   └── losses.py               # L_corr（订正）、L_pred（预测）
│   ├── trainers/
│   │   ├── corrector_trainer.py     # 第一阶段：训练订正器
│   │   └── predictor_trainer.py     # 第二阶段：训练专家+门控（冻结订正器）
│   ├── baselines/
│   │   ├── ts_library_wrap.py       # 强时序 SOTA 封装（DLinear/Informer/PatchTST/iTransformer/TimesNet）
│   │   ├── rf_baseline.py           # 历史功率 RF、RF+原始NWP（前身基线）
│   │   └── correction_baselines.py  # 线性/QM/RF 订正基线
│   └── utils/
│       ├── metrics.py               # RMSE/MAE/ACC/Q_R/r/R²（仅白天）
│       ├── seed.py                  # 固定随机种子
│       └── logger.py                # 训练日志 CSV
├── scripts/
│   ├── preprocess.sh                # 运行预处理，生成 data/processed/
│   ├── train_main.sh                # 训练本文完整方法（主实验）
│   ├── train_baselines.sh           # 训练所有 baseline
│   ├── ablation.sh                  # 批量消融 A–G
│   ├── evaluate.sh                  # 评估指定 checkpoint
│   └── interpret.sh                 # 导出可解释中间量、绘 4 张图
├── notebooks/
│   ├── 01_data_demo.ipynb           # 数据/预处理/FCM区制/VMD 可视化
│   ├── 02_model_demo.ipynb          # 模型结构/订正前后/中间表示
│   └── 03_results_demo.ipynb        # 主表/消融/泄漏演示/可解释4图
├── configs/
│   ├── default.yaml                 # 主方法默认超参
│   └── ablation/                    # A–G 各变体 config（覆盖 default 部分项）
├── data/                            # [gitignored] processed/ 缓存 + FCM/VMD 中间产物
├── results/                         # [gitignored] checkpoints/eval/predictions/ablation/figures
├── logs/                            # [gitignored] 训练曲线
├── README.md                        # 项目说明、环境、运行命令（位置：code/，见阶段E确认）
└── requirements.txt
```

### 1.2 各文件功能表

| 文件 | 功能 | 输入 | 输出 | 被谁调用 |
|------|------|------|------|---------|
| `src/data/preprocess.py` | 原始 PVOD 清洗对齐 | `dataset/station*.csv` | `data/processed/{sid}.csv` | `scripts/preprocess.sh` |
| `src/data/pvod_dataset.py` | 数据集类：划分+滑窗 | processed csv | `(x, y, aux)` tensor | trainer/eval |
| `src/data/fcm_regime.py` | FCM 软区制 | 气象向量 | 软隶属 `u(t)` [N,K]、中心 | dataset / model |
| `src/data/causal_vmd.py` | 因果滑窗 VMD | 历史功率/NWP 序列 | 多尺度模态 `ν(t)` | dataset |
| `src/data/transforms.py` | 归一化 | 原始数值 | 归一化值 + 统计量 | dataset / eval |
| `src/models/corrector.py` | NWP 误差订正器 | `[NWP, u]` | 订正后 NWP | corrector_trainer / gated_moe |
| `src/models/experts.py` | 轻量专家 GRU/TCN | `z_t` | 各专家预测 | gated_moe |
| `src/models/gated_moe.py` | 软门控多专家完整模型 | `z_t, u` | 日前功率 ŷ | predictor_trainer / eval |
| `src/models/losses.py` | 损失函数 | pred,target | 标量 | trainers |
| `src/trainers/corrector_trainer.py` | 阶段一训练 | corrector,data | 订正器 ckpt | train_main.sh |
| `src/trainers/predictor_trainer.py` | 阶段二训练 | gated_moe,data | 完整模型 ckpt | train_main.sh |
| `src/baselines/ts_library_wrap.py` | 强时序 SOTA 封装 | x | 预测 | train_baselines.sh |
| `src/baselines/rf_baseline.py` | RF 基线 | 特征 | 预测 | train_baselines.sh |
| `src/baselines/correction_baselines.py` | 订正基线 | NWP,LMD | 订正后 NWP | train_baselines.sh |
| `src/utils/metrics.py` | 评估指标 | pred,true,mask | dict | trainer/eval |
| `src/utils/seed.py` | 固定种子 | seed | — | 所有入口 |
| `src/utils/logger.py` | 日志 | epoch,metrics | CSV | trainers |
| `configs/default.yaml` | 集中超参 | — | — | 所有模块 |

**目录级约束**：

| 路径 | 关键约束 |
|-----|---------|
| `src/data/` | 只处理数据与特征（含 FCM/VMD），不含模型训练逻辑 |
| `src/models/` | 只定义结构，不含训练循环 |
| `src/trainers/` | 不定义模型，通过参数接收 |
| `src/baselines/` | 输入输出接口与主模型一致，便于替换评估 |
| `src/utils/` | 无状态工具 |
| `scripts/` | 只拼参数调 Python 模块 |
| `notebooks/` | 关键步骤可视化，不被生产代码依赖 |

> 设计依据：模块边界对应 idea_report Part 2 Method 的四模块（A=fcm_regime, B=corrector, C1=causal_vmd, C2=experts, D=gated_moe），一一可追溯，便于消融时按模块开关。

---

## 2 数据流

```text
原始文件（code/dataset/station{00..09}.csv, metadata.csv）
  → 读取与清洗（preprocess.py）
      字段：date_time + nwp_*(7) + lmd_*(6) + power
      清洗：箱线图法去异常值（Q1-1.5IQR ~ Q3+1.5IQR），缺失前向填充，
            标记夜间（lmd_totalirrad==0），不删行（前身验证行数一致）
      落盘：data/processed/{sid}.csv
  → 划分（pvod_dataset.py，按时间每站独立 7:1:2）
      train 段统计量用于归一化与 FCM/VMD 拟合（防泄漏）
  → 归一化（transforms.py，Max-Min，按 train 段统计）
  → FCM 软区制（fcm_regime.py）
      train 段拟合 K 个中心 → 全段算软隶属 u(t)∈[N,K]
  → 因果 VMD（causal_vmd.py）
      对历史功率/NWP 滑窗(τ≤t)分解 → 多尺度模态 ν(t)
  → 滑窗（pvod_dataset.py）
      look-back L（192/384）→ 预测 H=96；步长1
  → 模型输入
      x_nwp:  [B, L+H, D_nwp]   未来NWP预报（日前可得）
      x_hist: [B, L, D_hist]    历史功率+历史NWP+lag/统计特征
      nu:     [B, L, M]         因果VMD多尺度模态
      u:      [B, L+H, K]       软隶属（门控+订正条件）
      y:      [B, H]            未来96点功率（标签）
      lmd_paired: [B, L+H, D_p] 配对实测（仅训练阶段，订正监督用）
```

> 关键决策：(1) 所有统计量（归一化、FCM中心、VMD）**只用 train 段拟合**，从根上杜绝泄漏（RQ3）；(2) `x_nwp` 含未来段（日前预报可得），`lmd_paired` 仅训练用、推理不取（部署前提）；(3) 夜间点保留但评估时按 mask 排除。

---

## 3 各文件实现说明

> 阅读顺序：数据 → 特征(FCM/VMD) → 模型 → 损失 → 训练 → 工具 → 脚本 → baselines。函数只写签名+逻辑步骤，不贴代码。

### 3.1 `src/data/preprocess.py`

**文件职责**：原始 PVOD → 清洗对齐的 processed csv。

**`preprocess_station(sid: str, raw_dir: str, out_dir: str) -> None`**
- 输入：站点 id、原始目录、输出目录
- 逻辑：
  1. 读 `dataset/{sid}.csv`，解析 date_time 为 datetime。
  2. 对每个数值列用箱线图法（Q1−1.5·IQR, Q3+1.5·IQR）将异常值置 NaN。
  3. 缺失值前向填充 + 后向兜底（避免开头 NaN）。
  4. 新增 `is_day` 列（lmd_totalirrad>0）。
  5. 落盘 `data/processed/{sid}.csv`。

> 借鉴前身 `data_process.py` 的箱线图逻辑；不删行（前身验证 processed 与原始行数一致）。

**`build_pairing_index() -> dict`**
- 功能：返回 NWP–LMD 可配对字段映射（订正用）。
- 返回：`{'irrad':('nwp_globalirrad','lmd_totalirrad'), 'temp':('nwp_temperature','lmd_temperature'), 'pressure':('nwp_pressure','lmd_pressure')}`

> 依据数据验证：辐照(r=0.72–0.93)/温度(r=0.97)/气压(r=0.99)可配对；风速(r=0.55)/风向(r=0.33)不配对、仅作特征（idea_report 3.3）。

### 3.2 `src/data/transforms.py`

**`MinMaxNormalizer`**
- 初始化：`fit(train_df, cols)` 记录每列 min/max。
- `transform(df) -> df`：`(x−min)/(max−min)`。
- `inverse(col, values) -> values`：反归一化（评估前还原 power）。

> 借鉴前身 `utils.normalize_column`；统计量仅从 train 段计算，对 val/test 套用，防泄漏。

### 3.3 `src/data/fcm_regime.py`（模块 A）

**`FCMRegime`**
- 初始化参数：`n_clusters=K`（默认3）、`m=2.0`（模糊度 η）、`feature_cols`（用于聚类的 NWP 气象列）。
- `fit(train_features: ndarray) -> None`：在 train 段用 FCM（`skfuzzy.cmeans`）拟合 K 个中心 `c_k`，保存中心。
- `soft_membership(features: ndarray) -> ndarray`：对任意段，用**固定中心**算软隶属
  $u_k(t)=\frac{\|m_t-c_k\|^{-2/(m-1)}}{\sum_j\|m_t-c_j\|^{-2/(m-1)}}$，返回 `[N,K]`。
- 输入/输出 shape：features `[N, d]` → u `[N, K]`，逐行和为 1。

> 关键（推理一致性，idea_report 3.5）：聚类**只用 NWP 侧特征**（推理可得），中心 train 段固定；推理用 `soft_membership` 算新样本隶属，无需 LMD。借鉴前身 `fcm_clustering` 但改 argmax 硬标签为软隶属向量。⚠️ NWP-only 区制有效性在消融中验证。

### 3.4 `src/data/causal_vmd.py`（模块 C1）

**`causal_vmd_features(series: ndarray, K_modes: int, alpha: float, window: int, stride: int) -> ndarray`**
- 功能：对一维历史序列做**因果滑窗 VMD**，产出多尺度模态特征。
- 参数：`series`[T]、`K_modes`（模态数，默认5）、`alpha`（带宽惩罚）、`window`（滑窗长，默认=look-back L）、`stride`（步长，默认1）。
- 逻辑：
  1. 对每个时刻 t，取窗 `series[t-window+1 : t+1]`（仅过去）。
  2. 对该窗做 VMD（`vmdpy.VMD`），得 `K_modes` 个模态。
  3. 取各模态**在 t 时刻的值**（窗口末端），拼成 t 的模态特征向量 `[K_modes]`。
  4. 拼所有 t → `[T, K_modes]`。
- 返回：`[T, K_modes]` 多尺度特征。

> 核心（RQ3，idea_report 3.7）：每个时刻只用 τ≤t 的数据分解，从根上杜绝泄漏 [5,9]。VMD 优于 EMD（模态数可控、抑制混叠）。
> ⚠️ 性能：逐点滑窗 VMD 成本高，实现时缓存到 `data/processed/{sid}_vmd.npy`，避免每 epoch 重算。

**`global_vmd_features(series, K_modes, alpha) -> ndarray`**（消融变体 E 用）
- 功能：对**整条序列**（含测试段）一次性 VMD，**故意引入泄漏**，仅供消融对照。

> 这是消融 E 的实现入口，演示"虚高精度"陷阱（idea_report Part 3 §2）。

### 3.5 `src/data/pvod_dataset.py`

**`PVODDataset(Dataset)`**
- 初始化参数：`sid`、`split`（train/val/test）、`look_back L`、`horizon H=96`、`fcm`（已 fit 的 FCMRegime）、`normalizer`、`use_vmd`（bool）、`leak_vmd`（bool，消融E）、`return_lmd`（bool，训练阶段订正用）。
- 初始化逻辑：
  1. 读 `data/processed/{sid}.csv`，按时间 7:1:2 取对应 split 段。
  2. 套用 normalizer（train 统计量）。
  3. 算 FCM 软隶属 `u`（NWP 特征）。
  4. 若 use_vmd：加载/计算 VMD 模态 `ν`（leak_vmd 决定因果/全序列）。
  5. 构造 lag/滑动统计特征（借鉴前身 `extract_features`：hour、lag_1..4、rolling_change_rate）。
- `__len__`：滑窗样本数 = `len(seg) − L − H + 1`。
- `__getitem__(idx) -> dict`：返回
  - `x_nwp` [L+H, D_nwp]、`x_hist` [L, D_hist]、`nu` [L, M]、`u` [L+H, K]、`y` [H]、（若 return_lmd）`lmd_paired` [L+H, D_p]、`capacity`（标量，评估归一）、`is_day` [H]（白天掩码）。

> 借鉴前身 `extract_features` / `split_train_test`，但划分改为 7:1:2 时间顺序（Part 3 §0.7），并新增 u/ν/lmd 输出。

### 3.6 `src/models/corrector.py`（模块 B，第一阶段）

**`NWPCorrector(nn.Module)`**
- 初始化参数：`d_nwp`、`K`（区制数）、`hidden=128`、`d_paired`（可订正气象维度）。
- `forward(x_nwp, u) -> x_corr`：
  - 输入：`x_nwp` [B, T, d_nwp]、`u` [B, T, K]
  - 逻辑：
    1. 拼接 `[x_nwp, u]` → MLP/小型时序层，输出订正量 `ê` [B, T, d_paired]。
    2. `x_corr_paired = x_nwp_paired − ê`（仅订正可配对列）。
    3. 不可配对列（风速风向等）原样透传。
  - 输出：`x_corr` [B, T, d_nwp]（订正后 NWP，供第二阶段）。

> 核心（RQ1，idea_report 3.6）：仅吃 `[NWP, u]`（推理可得），不含未来 LMD。区制条件 u 使订正分天气自适应 [3,4]。

### 3.7 `src/models/experts.py`（模块 C2）

**`GRUExpert(nn.Module)` / `TCNExpert(nn.Module)`**
- 初始化参数：`d_in`（历史编码输入维）、`hidden`、`horizon H`、`d_fut`（未来段订正 NWP 维度=d_nwp）、（TCN）`kernel/levels`。
- `forward(z, x_fut) -> pred`：
  - 输入：`z` [B, L, d_in]（历史段特征：订正后历史 NWP + x_hist + nu）、`x_fut` [B, H, d_fut]（**未来段订正后 NWP**，即日前天气预报）。
  - 逻辑：1) 编码历史 `z`（GRU 末隐状态 / TCN 末步）得上下文向量 `ctx` [B, hidden]；
         2) 把 `ctx` 广播到 H 步，与未来段订正 NWP `x_fut` 逐步拼接 → [B, H, hidden+d_fut]；
         3) 共享 MLP 头逐步映射到功率 → `pred` [B, H]。
- 默认 GRU；config `expert.type ∈ {gru,tcn}` 切换。

> idea_report 3.7：K 个并行轻量专家，每个学一种区制下"特征→未来功率"。**关键修订（2026-06-08 代码审查 E-5）**：专家显式消费**未来段订正后 NWP**（日前天气预报），使订正在预测期真正生效（RQ1）、模型确为 NWP 气象驱动；原"专家仅用历史段"会让订正对预测无贡献，与立论矛盾，故修正。GRU 默认（稳快、序列不长够用），TCN 备选作加分消融。

### 3.8 `src/models/gated_moe.py`（模块 D，完整模型）

**`GatedMoEForecaster(nn.Module)`**
- 初始化参数：`corrector`（可冻结）、`K`、`expert_cfg`、`d_hist`、`M`（VMD模态数）、`use_vmd`、`gate_mode ∈ {soft,hard}`（消融C）、`single_expert`（消融D）。
- `forward(batch) -> (y_hat, aux)`：
  1. `x_corr = corrector(x_nwp, u)`（或 batch 已传订正后，视训练阶段）。`x_corr` 含历史段与未来段。
  2. 构造专家历史输入 `z = concat([x_corr 历史段, x_hist, nu])` [B, L, d_in]；
     取未来段订正 NWP `x_fut = x_corr[:, L:, :]` [B, H, d_nwp]（**日前预报，驱动预测**）。
  3. 每个专家 `f_k(z, x_fut)` → `pred_k` [B, H]，堆叠 [B, H, K]。
  4. 门控权重 `g`：soft=用 u 在预测段的隶属（或对 L 段池化）；hard=argmax(u) one-hot；single=K=1。
  5. `y_hat = Σ_k g_k · pred_k` [B, H]。
  6. `aux` 收集 `u`、各 `pred_k`、`x_corr`（可解释导出用）。
- 输出：`y_hat` [B, H]、`aux` dict。

> 核心（RQ2，idea_report 3.7）：u 同时作门控权重，软混合专家。`gate_mode/single_expert` 是消融 C/D 的开关。`aux` 为可解释 4 图预留接口（Part 3 §3）。

### 3.9 `src/models/losses.py`

**`corrector_loss(x_corr_paired, lmd_paired, mask) -> Tensor`**
- 公式：$\mathcal{L}_{corr}=\frac1n\sum\|x^{corr}-x^{lmd}\|^2$（仅白天 mask）。
- 输入：[B,T,d_p]，输出标量。

**`prediction_loss(y_hat, y, is_day) -> Tensor`**
- 公式：$\mathcal{L}_{pred}=\frac1n\sum(\hat p-p)^2$（白天 mask）。

> idea_report 3.6/3.7：两损失对应两阶段。仅白天计入，符合光伏惯例。

### 3.10 `src/trainers/corrector_trainer.py`（阶段一）

**`CorrectorTrainer`**：fit 订正器，监督 = train 段 LMD，早停于 val L_corr，存 `results/checkpoints/corrector_{sid}.pth`。

### 3.11 `src/trainers/predictor_trainer.py`（阶段二）

**`PredictorTrainer`**：加载并**冻结**订正器，训练专家+门控，监督 L_pred，早停于 val，存 `results/checkpoints/best_{sid}.pth`。
- config `finetune_corrector`（bool，默认False）控制是否解冻订正器微调。

> idea_report 3.7：两阶段（先订正后预测，冻结）便于逐组件消融。⚠️ finetune 选项待实验定夺。

### 3.12 `src/utils/metrics.py`

| 函数 | 指标 | 公式 | 方向 |
|-----|------|------|------|
| `rmse(p,t,C)` | RMSE | $\sqrt{\frac1n\sum((t-p)/C)^2}$ | ↓ |
| `mae(p,t,C)` | MAE | $\frac1n\sum|t-p|/C$ | ↓ |
| `acc(p,t,C)` | 准确率 | $(1-\text{RMSE})\times100$ | ↑ |
| `qr(p,t,C)` | 合格率 | $|t-p|/C<0.25$ 占比×100 | ↑ |
| `pearson(p,t)` | r | Pearson | ↑ |
| `r2(p,t)` | R² | 决定系数 | ↑ |

- 所有函数接受白天 mask，仅在 is_day 上计算。`C`=容量（metadata）。

> 借鉴前身 `utils.evaluation`；口径对齐 idea_report Part 3 §0.7（前身国标 + 文献 r/R²）。

**`day_ahead_rolling(model, dataset) -> per_day_metrics`**
- 实现 Part 3 §0.7 日前协议：测试集按天滚动，每天取 96 点预测，按完整天算指标再平均。

### 3.13 `src/utils/seed.py` / `logger.py`

- `set_seed(seed)`：固定 numpy/torch/cuda 种子。
- `Logger.log(epoch, metrics)`：追加写 `logs/train_{timestamp}.csv`。

### 3.14 `configs/default.yaml`

| 参数 | 块 | 默认 | 说明 |
|-----|----|------|------|
| `data.look_back` | data | 192 | 回看 2 天（敏感性附 384）Part3 §0.7 |
| `data.horizon` | data | 96 | 日前 1 天 |
| `data.split` | data | [0.7,0.1,0.2] | 时间顺序 |
| `fcm.K` | fcm | 3 | 区制数（消融G扫2-5） |
| `fcm.m` | fcm | 2.0 | 模糊度 |
| `vmd.K_modes` | vmd | 5 | 模态数 |
| `vmd.alpha` | vmd | 2000 | 带宽惩罚 |
| `expert.type` | model | gru | gru/tcn |
| `model.gate_mode` | model | soft | soft/hard（消融C） |
| `model.single_expert` | model | false | 消融D |
| `model.use_vmd` | model | true | 消融F关 |
| `model.leak_vmd` | model | false | 消融E开（泄漏演示） |
| `model.use_corrector` | model | true | 消融B关 |
| `train.lr` | train | 1e-3 | Adam，Part3 §0.7 |
| `train.batch_size` | train | 64 | 同上 |
| `train.patience` | train | 10 | 早停 |
| `train.seeds` | train | [0,1,2,3,4] | 5种子均值±std |

> 每个消融变体对应 `configs/ablation/{variant}.yaml`，仅覆盖相关开关。**每个开关都直通一个消融变体**，保证 Part 3 §2 全覆盖。

### 3.15 `scripts/`

- `preprocess.sh`：跑 preprocess.py 生成 data/processed/ + VMD 缓存。
- `train_main.sh`：两阶段训本文方法（10站×5种子）。
- `train_baselines.sh`：训强时序 SOTA + RF + 订正基线。
- `ablation.sh`：循环 A–G config，汇总 `results/ablation/summary.csv`。
- `evaluate.sh`：日前协议评估，出 eval json + predictions csv。
- `interpret.sh`：加载最优模型，导出 aux 中间量，绘可解释 4 图到 `results/figures/`。

### 3.16 `src/baselines/`

**`ts_library_wrap.py`**：封装 Time Series Library 的 DLinear/Informer/PatchTST/iTransformer/TimesNet，统一 `forward(x)->[B,H]` 接口，统一超参（§0.7）。
**`rf_baseline.py`**：历史功率 RF、RF+原始NWP（借鉴前身 `rf.py`）。
**`correction_baselines.py`**：线性/均值订正、分位数映射 QM、RF 订正——产出订正后 NWP 再接预测器。

> 接口与主模型一致，便于在评估脚本中替换。覆盖 Part 3 §1 三组对比。

---

## 4 数据下载与准备

### 4.1 数据集

| 数据集 | 类型 | 来源 | 下载链接 | 存放路径 |
|-------|------|------|---------|---------|
| PVOD | 已就绪 | Yao 2021 [1] | （本地已有） | `code/dataset/` |

### 4.2 准备步骤

```bash
# 数据已在 code/dataset/，仅需预处理生成缓存
bash scripts/preprocess.sh
```

### 4.3 处理后目录结构

```text
code/data/
└── processed/
    ├── station00.csv     # 清洗对齐后（含 is_day）
    ├── station00_vmd.npy # 因果VMD模态缓存 [T, K_modes]
    └── ... (10站)
```

### 4.4 数据字段说明

| 字段 | 类型 | 单位 | 含义 | 正常范围 |
|-----|------|------|------|---------|
| nwp_globalirrad | float | W/m² | 预报水平辐照 | [0, ~950] |
| lmd_totalirrad | float | W/m² | 实测总辐照（订正真值） | [0, ~1120] |
| nwp_temperature/humidity/pressure/windspeed/winddirection | float | ℃/%/hPa/(m/s)/° | 预报气象 | — |
| lmd_temperature/pressure/... | float | 同上 | 实测气象 | — |
| power | float | kW | 发电功率（标签） | [0, capacity] |
| is_day | bool | — | 白天标记（lmd辐照>0） | — |

> 配对相减仅用辐照/温度/气压（r 高）；容量取自 metadata.csv 的 Capacity。

---

## 5 results 文件格式规范

### 5.1 模型权重 `results/checkpoints/best_{sid}.pth`
- PyTorch state_dict，含 corrector + experts + gate；验证集最优 epoch。

### 5.2 训练曲线 `logs/train_{timestamp}.csv`

| 字段 | 类型 | 含义 |
|-----|------|------|
| epoch | int | 从1起 |
| stage | str | corrector / predictor |
| train_loss | float | 训练损失 |
| val_loss | float | 验证损失 |
| val_acc | float | 验证准确率(%) |
| lr | float | 学习率 |

### 5.3 评估结果 `results/eval_{timestamp}.json`

| 字段 | 类型 | 单位 | 含义 | 方向 |
|-----|------|------|------|------|
| rmse/mae | float | 归一 | 误差 | ↓ |
| acc/qr | float | % | 准确率/合格率 | ↑ |
| r/r2 | float | — | 相关/决定系数 | ↑ |
| station | str | — | 站点 | — |
| split | str | — | test | — |
| seed | int | — | 随机种子 | — |
| checkpoint | str | — | 权重路径 | — |

### 5.4 逐样本预测 `results/predictions_{sid}_{timestamp}.csv`

| 字段 | 类型 | 单位 | 含义 |
|-----|------|------|------|
| day_id | int | — | 测试集第几天 |
| step | int | — | 当日第几点(1-96) |
| true_power | float | kW | 真实功率 |
| pred_power | float | kW | 预测功率 |
| abs_error | float | kW | 绝对误差 |
| is_day | bool | — | 白天掩码 |

### 5.5 消融汇总 `results/ablation/summary.csv`

| 字段 | 含义 |
|-----|------|
| variant | A–G（对应 ablation.sh --variant） |
| acc_mean / acc_std | 10站×5种子 ACC 均值±标准差 |
| rmse_mean / mae_mean / r2_mean | 其他指标均值 |
| delta_acc | 相对完整模型A的ΔACC |
| notes | 变体说明 |

### 5.6 可解释中间量 `results/figures/` + `results/interpret/`
- `regime_membership_{sid}.csv`：u(t) 时序（图1）
- `expert_weight_heatmap.csv`：天气×专家权重（图2）
- `correction_compare_{sid}.csv`：订正前/后/LMD（图3）
- `case_{sunny|cloudy}.csv`：单日各专家预测+权重（图4）
- 对应 PNG 存 `results/figures/fig{1-4}_*.png`

> 这些由 interpret.sh 从 gated_moe 的 `aux` 导出，对应 Part 3 §3 四图。

---

## 6 实现顺序

```
requirements.txt
  → configs/default.yaml
  → README.md（初稿：项目内容+环境，运行命令占位）
  → src/utils/（seed, metrics, logger）
  → src/data/preprocess.py  → scripts/preprocess.sh（先把数据跑通）
  → src/data/transforms.py
  → src/data/fcm_regime.py
  → src/data/causal_vmd.py
  → notebooks/01_data_demo.ipynb（数据/FCM区制/VMD 可视化）
  → src/data/pvod_dataset.py
  → src/models/corrector.py
  → src/models/experts.py
  → src/models/gated_moe.py
  → src/models/losses.py
  → notebooks/02_model_demo.ipynb（结构/订正前后）
  → src/trainers/corrector_trainer.py
  → src/trainers/predictor_trainer.py
  → src/baselines/（ts_library_wrap, rf_baseline, correction_baselines）
  → scripts/（train_main, train_baselines, ablation, evaluate, interpret）
  → notebooks/03_results_demo.ipynb（主表/消融/泄漏演示/可解释4图）
```

每完成一个文件，立即在 `docs/dev_log.md` 更新进度与日志；影响运行/环境的同步更新 `README.md`，关键步骤同步补 `notebooks/`。
`✅ Done` 仅在文件写完且运行验证无报错后标记（按用户运行策略）。
