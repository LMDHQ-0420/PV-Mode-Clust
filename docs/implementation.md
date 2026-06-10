# Implementation Guide — 天气区制感知的可学习软门控异质多专家光伏日前功率预测（NWP 误差自适应订正）
> 生成时间：2026-06-07 | 最后修订：2026-06-08（E-8 回溯，按重塑后方法重写）| 策略：从头构建 | 状态：PENDING_REVIEW
> 关联实验设计：docs/idea_report.md Part 3
>
> **重塑要点（相对初版）**：① 删除因果 VMD 模块（实证无增益，仅留泄漏演示消融）；② 订正器改为
> 自适应辅助机制（只订辐照 + 大误差区掩码 + 残差门控 + 幅度约束）；③ 专家改为结构异质（短窗 GRU /
> 长窗 TCN / 频域分支）；④ 门控改为可学习（FCM 先验 + MLP 残差）；⑤ 训练改为**单阶段端到端联合**
> （L = L_pred + λ·L_corr），删两阶段冻结。

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
│   │   ├── causal_vmd.py             # [仅消融H] 全序列/因果 VMD，泄漏演示用，不入主流程
│   │   └── transforms.py            # Max-Min 归一化（按训练集统计），反归一化
│   ├── models/
│   │   ├── corrector.py              # 自适应辐照订正：corr=NWP+gate·Δ，仅大误差区+幅度约束（模块B）
│   │   ├── experts.py               # 异质专家：短窗GRU/长窗TCN/频域分支（模块C）
│   │   ├── gated_moe.py             # 可学习软门控异质MoE整体模型（组装 A/B/C/D，辐照锚定可选）
│   │   └── losses.py               # L_pred（预测,可选Huber）、L_corr（订正辅助，仅大误差区+幅度惩罚）
│   ├── stacking.py                  # 模块E：深度MoE×RF 互补 stacking（验证集拟合权重）—SOTA最终模型
│   ├── trainers/
│   │   └── joint_trainer.py         # 单阶段端到端联合训练（L=L_pred+λL_corr）
│   ├── baselines/
│   │   ├── dl_baselines.py          # 自包含DL baseline（LSTM/LSTNet/TCN/NBEATS/NHiTS/Crossformer/NWPLSTMbaseline）
│   │   ├── ts_library_wrap.py       # 统一训练封装（DLinear + dl_baselines + 可选TSLib系）
│   │   ├── rf_baseline.py           # 历史功率 RF、RF+NWP（日前96步多步，同口径）
│   │   └── correction_baselines.py  # 线性/QM/RF 订正基线（订正质量评估）
│   └── utils/
│       ├── metrics.py               # RMSE/MAE/ACC/Q_R/r/R²（仅白天）
│       ├── seed.py                  # 固定随机种子
│       └── logger.py                # 训练日志 CSV
├── scripts/
│   ├── preprocess.sh                # 运行预处理，生成 data/processed/
│   ├── train_main.sh                # 训练本文完整方法（主实验）
│   ├── train_baselines.sh           # 训练所有 baseline
│   ├── ablation.sh                  # 批量消融 A–H
│   ├── evaluate.sh                  # 评估指定 checkpoint
│   └── interpret.sh                 # 导出可解释中间量、绘 4 张图
├── notebooks/
│   ├── 01_data_demo.ipynb           # 数据/预处理/FCM区制/VMD 可视化
│   ├── 02_model_demo.ipynb          # 模型结构/订正前后/中间表示
│   └── 03_results_demo.ipynb        # 主表/消融/泄漏演示/可解释4图
├── configs/
│   ├── default.yaml                 # 主方法默认超参
│   └── ablation/                    # A–H 各变体 config（覆盖 default 部分项）
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
| `src/data/causal_vmd.py` | [仅消融H] VMD | 功率序列 | 模态特征 | dataset（leak_vmd 开关） |
| `src/data/transforms.py` | 归一化 | 原始数值 | 归一化值 + 统计量 | dataset / eval |
| `src/models/corrector.py` | 自适应辐照订正 | `[NWP辐照, u]` | 订正后辐照 + 残差门控γ | gated_moe |
| `src/models/experts.py` | 异质专家(GRU/TCN/频域) | `z_t, x_fut` | 各专家预测 | gated_moe |
| `src/models/gated_moe.py` | 可学习软门控异质MoE | `batch` | 日前功率 ŷ + aux | joint_trainer / eval |
| `src/models/losses.py` | 损失函数 | pred,target,mask | 标量 | joint_trainer |
| `src/trainers/joint_trainer.py` | 单阶段联合训练 | model,data | 完整模型 ckpt | train_main.sh |
| `src/baselines/dl_baselines.py` | 自包含DL baseline（7模型） | x_hist(±x_nwp_fut) | [B,H]预测 | train_baselines.sh |
| `src/baselines/ts_library_wrap.py` | 统一训练/评估封装 | x | 预测+JSON | train_baselines.sh |
| `src/baselines/rf_baseline.py` | RF 基线(日前多步) | 窗口特征 | [B,H]预测 | train_baselines.sh |
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

> 设计依据：模块边界对应 idea_report Part 2 Method 的模块（A=fcm_regime, B=corrector 自适应订正, C=experts 异质专家, D=gated_moe 可学习软门控），一一可追溯，便于消融时按模块开关。VMD 已退出主流程，仅 causal_vmd.py 保留供消融 H（泄漏演示）。

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
      train 段统计量用于归一化与 FCM 拟合（防泄漏）
  → 归一化（transforms.py，Max-Min，按 train 段统计）
  → FCM 软区制（fcm_regime.py）
      train 段拟合 K 个中心 → 全段算软隶属 u(t)∈[N,K]
  → 大误差区掩码（pvod_dataset.py）
      train 段辐照误差 |NWP-LMD| 的分位阈值 → 全段标记 m_t∈{0,1}
  → 滑窗（pvod_dataset.py）
      look-back L（192/384）→ 预测 H=96；步长1
  → 模型输入
      x_nwp:  [B, L+H, D_nwp]   未来NWP预报（日前可得）
      x_hist: [B, L, D_hist]    历史功率+历史NWP+lag/统计特征
      u:      [B, L+H, K]       软隶属（门控先验+订正条件）
      y:      [B, H]            未来96点功率（标签）
      irrad_lmd: [B, L+H]       配对实测辐照（仅训练，订正监督）
      big_err_mask: [B, L+H]    大误差区掩码（仅训练，订正损失用）
      is_day: [B, H]            白天掩码（评估用）
```

> 关键决策：(1) 所有统计量（归一化、FCM中心、大误差区阈值）**只用 train 段拟合**，从根上杜绝泄漏；(2) `x_nwp` 含未来段（日前预报可得），订正监督用的 LMD 辐照仅训练取、推理不取（部署前提）；(3) 夜间点保留但评估按 mask 排除；(4) **VMD 已退出主流程**（仅 leak_vmd 消融时由 causal_vmd 提供模态）。

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

> 这是消融 H 的实现入口，演示"虚高精度"陷阱（idea_report Part 3 §2）。

### 3.5 `src/data/pvod_dataset.py`

**`PVODDataset(Dataset)`**
- 初始化参数：`sid`、`split`（train/val/test）、`look_back L`、`horizon H=96`、`fcm`（已 fit 的 FCMRegime）、`normalizer`、`big_err_quantile`（大误差区分位，默认0.6）、`leak_vmd`（bool，消融H）、`return_lmd`（bool，订正监督用）。
- 初始化逻辑：
  1. 读 `data/processed/{sid}.csv`，按时间 7:1:2 取对应 split 段。
  2. 套用 normalizer（train 统计量）。
  3. 算 FCM 软隶属 `u`（NWP 特征）。
  4. **大误差区掩码**：在 train 段算白天辐照误差 |NWP_irrad−LMD_irrad| 的 `big_err_quantile` 分位阈值，对全段标记 `big_err_mask`（>阈值且白天→1）。
  5. 构造 lag/滑动统计特征（借鉴前身 `extract_features`：hour、lag_1..4、rolling_change_rate）。
  6. 若 `leak_vmd`（仅消融H）：加载/计算全序列 VMD 模态拼进 x_hist。
- `__len__`：滑窗样本数 = `len(seg) − L − H + 1`。
- `__getitem__(idx) -> dict`：返回
  - `x_nwp` [L+H, D_nwp]、`x_hist` [L, D_hist]、`u` [L+H, K]、`y` [H]、（若 return_lmd）`irrad_lmd` [L+H]、`big_err_mask` [L+H]、`capacity`（标量）、`is_day` [H]（白天掩码）。

> 借鉴前身 `extract_features` / `split_train_test`，但划分改为 7:1:2 时间顺序（Part 3 §0.7），并新增 u/ν/lmd 输出。

### 3.6 `src/models/corrector.py`（模块 B，自适应辐照订正）

**`AdaptiveIrradCorrector(nn.Module)`**
- 初始化参数：`d_nwp`、`K`（区制数）、`hidden=128`、`irrad_idx`（辐照列在 d_nwp 中的索引，**只订这一列**）。
- `forward(x_nwp, u) -> (x_corr, gamma, delta)`：
  - 输入：`x_nwp` [B, T, d_nwp]、`u` [B, T, K]
  - 逻辑：
    1. 拼接 `[x_nwp_irrad, u]` → MLP，输出两路：订正量 `Δ` [B, T, 1] 与残差门控 logit；`gamma = sigmoid(logit)` ∈[0,1]。
    2. `x_corr_irrad = x_nwp_irrad + gamma · Δ`（仅订正辐照列；`+` 号，残差形式）。
    3. 其余列（温度/气压/风等）原样透传。
  - 输出：`x_corr` [B, T, d_nwp]（仅辐照列被订正）、`gamma` [B,T,1]（残差门控,可解释）、`delta` [B,T,1]。

> 核心（RQ2，idea_report 3.6）：仅吃 `[NWP辐照, u]`（推理可得），仅订辐照（与功率强相关），残差门控 `gamma` 让订正"无话可说时闭嘴"。订正辅助损失仅在大误差区（掩码 m_t）计入 + 幅度惩罚 β(γΔ)²（见 3.9）。**端到端联合训练**，不再两阶段冻结。`gamma` 导出供可解释图3。

### 3.7 `src/models/experts.py`（模块 C，异质专家——以未来 NWP 为主序列）

**关键（实证修正）**：日前 PV 中未来 NWP 是主导预测因子，故所有专家以**未来段订正 NWP 序列 `x_fut` [B,H,d_fut] 为主建模序列**，历史 `z` 经 `_HistEncoder`（GRU）编码为上下文 `ctx [B,hidden]` 作逐步条件（广播到 H 步与 x_fut 拼接）。统一接口 `forward(z, x_fut) -> [B,H]`。三类专家在"如何处理 x_fut 序列"上结构异质：

**`LocalConvExpert`**（小核因果卷积，捕局部快速波动）
- `[x_fut ‖ ctx广播]` → 两层小核(kernel=3)因果卷积 → 逐步线性头。

**`DilatedTCNExpert`**（大膨胀感受野，捕日内长程趋势）
- `[x_fut ‖ ctx广播]` → levels=4 膨胀因果卷积(dilation 1/2/4/8) → 逐步线性头。

**`DirectMLPExpert`**（逐步 MLP，类 RF 直接映射）
- 每步 `[x_fut_h ‖ ctx]` → 3 层 MLP → 功率。

- `build_experts(expert_types, ...)` 按 `model.expert_types`（默认 `[local_conv, dilated_tcn, direct_mlp]`）构造；消融 D（同构）时全用同一类型。注册表兼容旧名 `short_gru/long_tcn/freq`（映射到新实现）。

> idea_report 3.7：以未来 NWP 为主序列的修正使大站 +1.7 ACC；结构异质防塌缩 [10,14]。

### 3.8 `src/models/gated_moe.py`（模块 D，可学习软门控异质 MoE）

**`GatedMoEForecaster(nn.Module)`**
- 初始化参数：`corrector`、`K`、`expert_types`（异质专家类型列表）、`d_nwp`、`d_hist`、`horizon`、`gate_mode ∈ {soft,hard}`（消融B）、`learnable_gate`（bool，消融C关）、`single_expert`（消融E）、`use_corrector`（消融F关）、`gate_hidden`。
- `forward(batch) -> (y_hat, aux)`：
  1. 若 `use_corrector`：`x_corr, gamma, delta = corrector(x_nwp, u)`；否则 `x_corr=x_nwp`。
  2. 专家历史输入 `z = concat([x_corr 历史段, x_hist])` [B, L, d_in]；`x_fut = x_corr[:, L:, :]` [B,H,d_nwp]。
  3. 各异质专家 `f_k(z, x_fut)` → `pred_k` [B,H]，堆叠 [B,H,K]。
  4. **可学习软门控**：`u_fut = u[:, -H:, :].mean(1)` [B,K]；
     - learnable_gate=True：`g = softmax(α·log(u_fut+ε) + MLP_ψ(x_nwp_pool))`，α 可学习标量；
     - learnable_gate=False（消融C）：`g = u_fut`（固定外部隶属）；
     - gate_mode=hard（消融B）：对 g 取 argmax one-hot；
     - single_expert=True（消融E）：K=1，g=1。
  5. `moe_out = Σ_k g_k · pred_k` [B,H]。
  6. 辐照锚定（`irrad_anchor`，默认 False）：True 时 `y_hat = softplus(gain)·x_corr未来辐照 + moe_out`；False 时 `y_hat = moe_out`。
  7. `aux` 收集 `u`、`g`、各 `pred_k`、`x_corr`、`gamma`（可解释导出用）。
- 输出：`y_hat` [B,H]、`aux` dict。

> 核心（RQ1，idea_report 3.7）：FCM 软隶属作可解释先验 + 可学习残差门控，软混合异质专家。`gate_mode/learnable_gate/single_expert/use_corrector/irrad_anchor` 直通消融 B/C/E/F/I；同构专家消融 D 在 expert_types 层面切换。`irrad_anchor` 实证有害（默认关），作负结果对照。`aux` 为可解释 5 图预留接口（Part 3 §3）。

### 3.8b `src/stacking.py`（模块 E，互补 stacking — SOTA 最终模型）

**`run_stacking(cfg, sid, seed)`**：训深度 MoE（JointTrainer）+ RF（日前多步），在验证集按 RMSE 拟合凸组合权重 `w*∈[0,1]`，测试集用固定 `w*` 评估 `y=w*·deep+(1-w*)·rf`。
- `fit_blend_weight(deep_val, rf_val, true_val, day_val, cap)`：val 上 0.05 步长扫 w，取 RMSE 最小。
- 落 `results/stack_{sid}_seed{seed}.json`：含 deep/rf/stack 三者全指标 + `blend_weight`，顶层字段=stack。
- 被 `src.run stack` 子命令 / `scripts/train_main.sh` 调用。

> 依据（RQ4）：深度模型与 RF 互补，凸组合稳定超越各自（station04 +1.4 ACC）。w* 仅 val 拟合，防泄漏。集成是 PV 文献达 SOTA 标准手段 [6]。

### 3.9 `src/models/losses.py`

**`prediction_loss(y_hat, y, is_day) -> Tensor`**
- 公式：$\mathcal{L}_{pred}=\frac1n\sum(\hat p-p)^2$（白天 mask）。

**`corrector_loss(x_corr_irrad, lmd_irrad, big_err_mask, gamma_delta, beta) -> Tensor`**
- 公式：$\mathcal{L}_{corr}=\dfrac{\sum_t m_t(x^{corr}_t-x^{lmd}_t)^2}{\sum_t m_t} + \beta\cdot\frac1n\sum_t(\gamma_t\Delta_t)^2$。
- 第一项仅在大误差区掩码 `m_t` 计入（订正监督），第二项为幅度惩罚（抑制乱订正）。

**总损失**（joint_trainer 内）：$\mathcal{L}=\mathcal{L}_{pred}+\lambda\mathcal{L}_{corr}$。

> idea_report 3.6/3.7：单阶段联合，订正以辅助损失形式服务预测。仅白天计入，符合光伏惯例。

### 3.10 `src/trainers/joint_trainer.py`（单阶段端到端联合）

**`JointTrainer`**：一次性训练订正器+异质专家+可学习门控，最小化 $\mathcal{L}=\mathcal{L}_{pred}+\lambda\mathcal{L}_{corr}$，早停于 val L_pred，存 `results/checkpoints/best_{sid}.pth`。
- config `loss.lambda_corr`（订正辅助权重，默认 0.1）、`corrector.beta`（幅度惩罚，默认 0.01）、`corrector.big_err_quantile`（大误差区阈值分位，默认 0.6）。
- 大误差区掩码 `m_t` 在 train 段按辐照误差分位预计算（dataset 提供），随 batch 取出。

> idea_report 3.7：单阶段联合替代原两阶段冻结——订正梯度同时回传，使订正以降低功率误差为目标。删除原 corrector_trainer/predictor_trainer。

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
| `model.expert_types` | model | [short_gru,long_tcn,freq] | 异质专家；消融D设为同构 |
| `model.short_window` | model | 48 | 短窗GRU回看长度 |
| `model.gate_mode` | model | soft | soft/hard（消融B） |
| `model.learnable_gate` | model | true | 可学习残差门控；消融C关 |
| `model.gate_hidden` | model | 64 | 门控 MLP 宽度 |
| `model.expert_hidden` | model | 128 | 专家隐层（实证>128 无益）|
| `model.single_expert` | model | false | 消融E（K=1） |
| `model.use_corrector` | model | true | 消融F关 |
| `model.irrad_anchor` | model | false | 辐照锚定头（消融I开，实证有害默认关）|
| `model.leak_vmd` | model | false | 消融H开（全序列VMD泄漏演示） |
| `corrector.beta` | corrector | 0.01 | 订正量幅度惩罚 |
| `corrector.big_err_quantile` | corrector | 0.6 | 大误差区阈值分位 |
| `loss.lambda_corr` | loss | 0.1 | 订正辅助损失权重 |
| `loss.huber_delta` | loss | 0.0 | >0 用Huber（实证无益，默认MSE）|
| `train.lr` | train | 3e-3 | Adam（调参定，高LR+cosine更优）|
| `train.cosine_lr` | train | true | cosine 退火 |
| `train.max_epochs` | train | 150 | 调参定 |
| `train.batch_size` | train | 64 | Part3 §0.7 |
| `train.patience` | train | 20 | 早停 |
| `train.seeds` | train | [0,1,2,3,4] | 5种子均值±std |

> 每个消融变体对应 `configs/ablation/{variant}.yaml`，仅覆盖相关开关。**每个开关都直通一个消融变体**（B硬门控/C固定门控/D同构/E单专家/F w/o订正/F2盲目订正/G K敏感/H泄漏演示），保证 Part 3 §2 全覆盖。

### 3.15 `scripts/`

- `preprocess.sh`：跑 preprocess.py 生成 data/processed/。
- `train_main.sh`：**单阶段联合**训本文方法（10站×5种子）。
- `train_baselines.sh`：训强时序 SOTA + RF（日前多步）+ 订正基线。
- `ablation.sh`：循环 A–H config，汇总 `results/ablation/summary.csv`。
- `evaluate.sh`：日前协议评估，出 eval json + predictions csv。
- `interpret.sh`：加载最优模型，导出 aux 中间量（u/g/γ/各专家预测/订正前后），绘可解释 5 图。

### 3.16 `src/baselines/`

**`dl_baselines.py`**：7个自包含深度学习 baseline，覆盖四大家族：
- **RNN**：`LSTMForecaster`（双层LSTM）、`NWPLSTMBaseline`（Encoder-Decoder LSTM，消费未来NWP，与本文同信息量）
- **Conv**：`LSTNetForecaster`（Conv+GRU+skip-GRU+AR），`TCNForecaster`（膨胀因果卷积，dilations=[1,4,16,64]）
- **MLP-basis**：`NBEATSForecaster`（2栈×3块），`NHiTSForecaster`（池化尺度[1,4,16]，分层插值）
- **Attention**：`CrossformerForecaster`（patch_size=16/stride=8 跨时间联合patch编码，简化版）

**`ts_library_wrap.py`**：统一训练/评估封装（Adam+cosine LR，patience=20早停）。`DL_MODELS` 调 `build_dl_model`（无外部依赖）；`TSLIB_MODELS` 调 `TSLibWrapper`（需 `TSLIB_PATH`）；NWP-aware 模型额外传 `b["x_nwp"][:, L:, :]`。

**`rf_baseline.py`**：历史功率 RF、RF+（未来段）NWP，**必须采用日前 96 步协议**（§0.7），与主方法/强时序 SOTA 同口径。
- **关键（2026-06-08 代码审查 E-8 修订）**：严禁用"前 1~4 个点的真实功率"做特征（`lag_1=power.shift(1)` 那种逐点/持续性预测）——那相当于预测 t 点时偷看了 t-1（15min 前）真值，是超短期而非日前，对深度 baseline 不公平、且违背日前定义。
- **正确做法（直接多步）**：用 PVODDataset 同样的滑窗样本，把**历史窗口 [L] 的功率/历史NWP 展平 + 未来段 [H] 的 NWP 预报**拼成一条特征向量，RF **一次性回归未来 96 维**（`RandomForestRegressor` 原生支持多目标输出 Y[N,H]）。
  - `rf_hist`：仅历史功率窗口（展平）→ 预测 [H]。
  - `rf_nwp`：历史功率窗口 + 未来段 NWP 预报（展平）→ 预测 [H]。未来 NWP 是日前可得的合法输入。
- 评估走 `day_ahead_rolling`（与主方法完全一致），仅白天、容量归一。

**`correction_baselines.py`**：线性/均值订正、分位数映射 QM、RF 订正——产出订正后 NWP，评估订正质量（对 LMD）。此文件不涉及功率预测的日前协议，保持现状。

> 接口与主模型一致，便于在评估脚本中替换。覆盖 Part 3 §1 三组对比。**所有功率预测类 baseline 一律日前 96 步同口径，这是公平对比的前提。**

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
| variant | A–H（对应 ablation.sh --variant） |
| acc_mean / acc_std | 10站×5种子 ACC 均值±标准差 |
| rmse_mean / mae_mean / r2_mean | 其他指标均值 |
| delta_acc | 相对完整模型A的ΔACC |
| notes | 变体说明 |

### 5.6 可解释中间量 `results/figures/` + `results/interpret/`
- `regime_membership_{sid}.csv`：u(t) 时序（图1）
- `expert_weight_heatmap.csv`：天气×最终门控 g 权重（图2）
- `correction_compare_{sid}.csv`：订正前/后/LMD + 残差门控 γ（图3）
- `case_{sunny|cloudy}.csv`：单日各专家预测+门控权重（图4）
- `gate_entropy_acc.csv`：隶属熵分层 软vs硬门控 ACC（图5，RQ3）
- 对应 PNG 存 `results/figures/fig{1-5}_*.png`

> 这些由 interpret.sh 从 gated_moe 的 `aux`（u/g/γ/各专家预测/订正前后）导出，对应 Part 3 §3 五图。

---

## 6 实现顺序

```
requirements.txt
  → configs/default.yaml
  → README.md（初稿：项目内容+环境，运行命令占位）
  → src/utils/（seed, metrics, logger, config）
  → src/data/preprocess.py  → scripts/preprocess.sh（先把数据跑通）
  → src/data/transforms.py
  → src/data/fcm_regime.py
  → src/data/causal_vmd.py（仅消融H保留）
  → notebooks/01_data_demo.ipynb（数据/FCM区制 可视化）
  → src/data/pvod_dataset.py（含大误差区掩码预计算；删 VMD 主流程依赖）
  → src/models/corrector.py（自适应辐照订正 + 残差门控）
  → src/models/experts.py（异质：short_gru/long_tcn/freq）
  → src/models/gated_moe.py（可学习软门控）
  → src/models/losses.py（L_pred + L_corr 含大误差区+幅度惩罚）
  → notebooks/02_model_demo.ipynb（结构/订正前后/门控）
  → src/trainers/joint_trainer.py（单阶段联合 L=L_pred+λL_corr）
  → src/baselines/（ts_library_wrap, rf_baseline 日前多步, correction_baselines）
  → scripts/（train_main, train_baselines, ablation, evaluate, interpret）
  → notebooks/03_results_demo.ipynb（主表/消融/泄漏演示/可解释5图）
```

每完成一个文件，立即在 `docs/dev_log.md` 更新进度与日志；影响运行/环境的同步更新 `README.md`，关键步骤同步补 `notebooks/`。
`✅ Done` 仅在文件写完且运行验证无报错后标记（按用户运行策略）。
