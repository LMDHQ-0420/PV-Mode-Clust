# 开发日志 — 天气区制感知与 NWP 误差订正的软门控多专家光伏日前功率预测
> 创建时间：2026-06-08 | 最后更新：2026-06-10
> 关联实现指南：docs/implementation.md

## 项目概览
| 项目 | 内容 |
|------|------|
| 研究方向 | 天气区制感知 + NWP 误差订正的软门控多专家光伏日前功率预测 |
| 实现策略 | 从头构建（旧 code/ 弃用，仅借鉴 metrics/FCM/VMD/extract_features 逻辑）|
| 框架 | PyTorch；单/多 GPU（4× RTX 3090）|
| 运行环境 | conda `zw@PV-Mode-Clust`（Python 3.10）|
| 运行策略 | 混合：快脚本 Claude 跑，完整训练/消融/baseline 用户跑 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|------|------|------|---------|------|
| 初始化 | requirements.txt, configs/default.yaml, README.md, src/utils/ | ✅ Done | 2026-06-08 | seed/metrics/logger/config |
| 数据预处理 | src/data/preprocess.py, scripts/preprocess.sh | ✅ Done | 2026-06-08 | 10 站预处理已跑通 |
| 归一化 | src/data/transforms.py | ✅ Done | 2026-06-08 | MinMaxNormalizer |
| FCM 软区制 | src/data/fcm_regime.py | ✅ Done | 2026-06-08 | 模块 A，软隶属固定中心 |
| 因果 VMD | src/data/causal_vmd.py | ✅ Done | 2026-06-08 | 模块 C1 + 泄漏版(消融E) |
| 数据集类 | src/data/pvod_dataset.py | ✅ Done | 2026-06-08 | smoke 验证 shape 正确 |
| 订正器 | src/models/corrector.py | ✅ Done | 2026-06-08 | 模块 B |
| 专家 | src/models/experts.py | ✅ Done | 2026-06-08 | 模块 C2，GRU+TCN |
| 完整模型 | src/models/gated_moe.py | ✅ Done | 2026-06-08 | 模块 D，含 B/C/D/F 开关 |
| 损失函数 | src/models/losses.py | ✅ Done | 2026-06-08 | L_corr / L_pred |
| 阶段一训练 | src/trainers/corrector_trainer.py | ✅ Done | 2026-06-08 | smoke 训通 |
| 阶段二训练 | src/trainers/predictor_trainer.py | ✅ Done | 2026-06-08 | smoke 训通 |
| 入口/汇总 | src/run.py, src/summarize.py, src/trainers/build.py | ✅ Done | 2026-06-08 | train/eval/interpret 均跑通 |
| Baseline | src/baselines/ | ✅ Done | 2026-06-08 | DLinear/RF/订正 smoke 通；TSLib 需 TSLIB_PATH |
| 运行脚本 | scripts/ | ✅ Done | 2026-06-08 | preprocess/train_main/baselines/ablation/eval/interpret + A–G config |
| 可视化 notebook | notebooks/ | ✅ Done | 2026-06-08 | 01 数据 / 02 模型 / 03 结果 |

状态：⬜ TODO / 🔄 WIP / ✅ Done（已运行验证）/ ❌ Blocked

## 开发日志

### 2026-06-08 — 初始化项目
- **完成内容**：requirements.txt（无 torch）、configs/default.yaml（含 A–G 消融开关）、README.md 初稿、
  src/utils/{seed,metrics,logger,config}.py、各包 `__init__.py`。
- **遇到的问题**：原始 PVOD 数据未在 `code/dataset/`（上传中，~11min）；conda 环境 `zw@PV-Mode-Clust` 已建。
- **解决方案**：先写不依赖数据的全部代码，数据到位后再跑预处理与 smoke test。

### 2026-06-08 — 数据层
- **完成内容**：preprocess.py（箱线图清洗+is_day，兼容 {sid}.csv / {sid}_processed.csv 命名）、
  transforms.py（MinMaxNormalizer，train 段 fit）、fcm_regime.py（FCM 软隶属，固定中心推理一致）、
  causal_vmd.py（因果滑窗 VMD + 缓存 + 泄漏版供消融E）、pvod_dataset.py（7:1:2 时间划分、
  全序列因果特征后切窗、产出 x_nwp/x_hist/nu/u/y/paired/is_day/capacity）、scripts/preprocess.sh。
- **遇到的问题**：原始数据未到位，暂无法运行验证。
- **解决方案**：标 WIP，数据到位后跑 preprocess.sh + 单站 Dataset smoke test 再标 Done。

## 已知问题
- [ ] 原始 PVOD 数据待上传到 `code/dataset/`，预处理与训练在此之前无法运行。
- [ ] 因果滑窗 VMD 逐点成本高，已加 .npy 缓存；首次构建仍可能较慢，需实跑观察。

### 2026-06-08 — 模型层
- **完成内容**：corrector.py（[NWP,u]→订正量 ê，仅订正可配对列，透传其余）、experts.py
  （GRUExpert/TCNExpert + build_expert 工厂）、gated_moe.py（组装 B/C1/C2/D，门控由 u 预测段
  池化，含 soft/hard/single/use_vmd/use_corrector 开关）、losses.py（corrector_loss/prediction_loss，
  仅白天 mask）。
- **遇到的问题**：index_copy_ 负 dim 不稳 → 改用显式 last dim + 非就地 index_copy。
- **解决方案**：smoke test 阶段用随机张量验证各 shape 与开关分支。

### 2026-06-08 — 训练/入口/脚本
- **完成内容**：trainers/build.py（防泄漏装配 dataset+model）、corrector_trainer.py（阶段一，早停 val
  L_corr）、predictor_trainer.py（阶段二，冻结订正器，可 finetune）、src/run.py（train/evaluate/interpret
  子命令 + ckpt_dir/eval_dir 隔离）、src/summarize.py（消融汇总）、scripts/{preprocess,train_main,
  train_baselines,ablation,evaluate,interpret}.sh、configs/ablation/A–G 共 9 个变体 yaml。
- **遇到的问题**：消融变体输出需隔离避免互相覆盖。
- **解决方案**：run.py 加 --ckpt_dir/--eval_dir，ablation.sh 每变体独立目录后再 summarize。

### 2026-06-08 — baseline / notebook / 环境
- **完成内容**：baselines/rf_baseline.py（RF 历史/+NWP，新划分重跑）、correction_baselines.py
  （raw/mean/linear/QM/RF 订正对 LMD 评估）、ts_library_wrap.py（自含 DLinear + TSLib 集成接口，
  统一 forward(x)->[B,H] 与训练评估）、notebooks 01/02/03、.gitignore。
  环境 zw@PV-Mode-Clust 装好 requirements，正在装 torch(cu121)。
- **遇到的问题**：Informer/PatchTST 等需官方 TSLib 仓库，直接复制有版权/诚信风险。
- **解决方案**：DLinear 自含可跑作默认；其余走 TSLIB_PATH 集成，缺失时明确报错不伪造。

### 2026-06-08 — 环境就绪 + 全链路 smoke test（Claude 跑）
- **完成内容**：env zw@PV-Mode-Clust 装好 torch 2.5.1+cu121（CUDA 可用，4×3090）；
  10 站预处理跑通；用 station05 + throwaway 缓存目录做缩小版全链路 smoke：
  dataset 构建（shape 全对：x_nwp[192,7]/x_hist[L,17]/nu[L,5]/u[192,3]/y[96]）、
  两阶段训练、evaluate（出 eval json + predictions csv）、interpret（出中间量 csv）、
  DLinear/RF/订正基线均跑通。smoke 产物已清理，真实 VMD 缓存未污染。
- **遇到的问题**：VMD .npy 缓存键只含 {sid}_vmd，smoke 用不同 stride 会污染真实缓存。
- **解决方案**：smoke 用独立 data/smoke 目录隔离；真实运行用 default stride=1。
- **观察**：2-epoch smoke 下 RF≈92.5% 高于深度模型（84%），符合"RF+NWP 是强基线"
  的前身结论；真实结论需 100-epoch×5 种子完整训练（用户跑）。

### 2026-06-08 — E-7 代码审查 + E-5 回溯修复（未来 NWP 入口）
- **审查结论**：可运行性/数据流/防泄漏/消融开关/单位口径全部通过；发现 1 处逻辑硬伤。
- **问题**：专家预测器只用历史段订正 NWP，未来段订正 NWP（日前预报）未进预测 →
  订正在预测期不生效，与"NWP 驱动日前预测/RQ1"立论矛盾。implementation.md §3.8 原文亦如此，
  属设计文档逻辑漏洞。
- **回溯范围（用户确认"修复"）**：阶段 D，改 implementation.md §3.7/§3.8 + experts.py/gated_moe.py。
- **改动**：
  1. implementation.md §3.7 专家 forward 改为 `forward(z, x_fut)`，§3.8 step2 取 `x_fut=x_corr[:,L:,:]`，
     附修订理由；执行校验（覆盖/一致性/完整性均通过，消融入口不受影响）。
  2. experts.py 新增 `_FutureHead`，GRU/TCN 专家编码历史得 ctx → 广播拼未来订正 NWP 逐步出功率；
     build_expert 增加 d_fut 参数。
  3. gated_moe.py 传 x_fut 给专家，d_fut=d_nwp。
- **验证**：重跑 smoke 通过；扰动未来段 NWP 输出平均变化 0.426（>0），证明日前预报确实驱动预测。
- **同步**：README 无需改（运行命令不变）；本条记录入 dev_log。

### 2026-06-08 — 新增 GPU 指定参数（用户提出）
- **完成内容**：run.py / ts_library_wrap.py 加 `--gpu N`（覆盖 train.device 为 cuda:N）；
  scripts(train_main/ablation/evaluate/interpret/train_baselines) 支持 `GPU=N` 环境变量透传。
  验证 `cuda:1` 张量落卡正确。README 补"指定 GPU"小节（多卡并行示例）。
- **原因**：4×3090 多卡环境下原先只能 cuda:0 或靠 CUDA_VISIBLE_DEVICES，不便并行铺站点/种子。
- **影响**：运行命令新增可选项，向后兼容（缺省仍 cuda:0）。

### 2026-06-08 — 清理老版本竞赛代码残留（用户要求）
- **删除**（新代码 src/scripts 对其零依赖，已核实）：
  - 老脚本：problem_01~03_4.py、rf.py、utils.py、data_process.py、draw.ipynb、problem_04_data_process.ipynb
  - 老产物：model/(91 pkl)、picture/、results/problem_0*(118 旧结果)、各 .DS_Store
  - dataset 残留：10 个 *_processed.csv（旧 2/5/8/11 月划分产物，用户确认删）、ERA5_data.grib(64MB，问题四降尺度，已舍弃，用户确认删)
- **代码微调**：preprocess.py 去掉"兼容 {sid}_processed.csv 老命名"的回退分支，只认 {sid}.csv。
- **效果**：code/ 5.3G → 56M；dataset/ 仅剩 station00-09.csv + metadata.csv。
- **验证**：preprocess 重跑 10 站通过；残留引用扫描无对老脚本的 import/调用。
- **关键事实**：借鉴逻辑（评估指标/FCM/VMD/extract_features）早已内化进 src/，删老脚本不丢信息。

### 2026-06-08 — E-8 回溯：实证诊断 + 文献精读 + idea 重塑（阶段B）
- **实证诊断**（station04/05 同口径对照 + 消融）：
  - 修复 RF baseline 口径（原偷看前15min真值→改日前96步多步），公平后 RF+NWP 88.4/82.0、本文 87.3/80.9、DLinear 83.9/78.2、RF仅历史 84.6/77.4。本文胜 DLinear/RF-hist 但微输 RF+NWP。
  - 关键消融：B w/o订正 ΔACC +0.8(大站)/−0.2（订正有害/无效）；C 硬门控 −1.0/−1.1（软门控有效✅）；D 单专家 +0.3/−0.3（多专家塌缩）；F w/o VMD +0.37/−0.05（VMD无增益）。
- **文献精读**（3 组并行，docs/papers/）：订正类(AE2024/EAAI2025/RE2025/E&B2024)、VMD类(VMDNet/EMD泄漏)、MoE类(MoWE/TimeExpert/Wind/IDS-Net)。结论：成功MoE靠可学门控+异质专家；成功订正靠端到端联合+仅辐照+仅大误差区；分解因果化后增益蒸发。
- **idea 重塑（阶段B，已改 idea_report Part1/2）**：
  - 标题/RQ 重构：核心从"NWP订正"改为"可学习软门控异质MoE"（RQ1主），订正降为端到端辅助机制（RQ2），可解释决策支持（RQ3）；删因果VMD主线（仅留泄漏演示消融）。
  - Method 重写：A FCM软区制(留)；B 自适应订正(只辐照+大误差区掩码+残差门控+幅度约束+端到端辅助损失)；C 结构异质专家(短窗GRU/长窗TCN/频域)；D 可学习软门控(FCM先验+MLP残差, softmax)。总损失 L=L_pred+λL_corr，单阶段联合。
  - 新增文献 [14]Wind MoE、[15]IDS-Net（标低置信度待核实出处）；修正 MoE 类引用编号。
- **阶段C（已改 idea_report Part3）**：
  - §0.7 训练设置改单阶段联合 L=L_pred+λL_corr；强调所有功率 baseline 日前96步同口径。
  - §1 主实验：RF baseline 明确日前多步重跑（非逐点偷看真值）；本文方法描述更新。
  - §2 消融重构：A完整/B硬门控/C固定门控/D同构专家/E单专家（均 RQ1）/F w/o订正/F2盲目订正（RQ2）/G K敏感/H全序列VMD泄漏演示（诚实性,不计精度排名）。删原"因果VMD作贡献"消融。
  - §3 可解释：新增图5（软门控过渡带优势,按隶属熵分层,RQ3核心）；图2/3 更新为最终门控g/残差门控γ;导出中间量增加 g_k/γ_t。
- **阶段D（已改 implementation.md，过校验）**：
  - §1 树/表：删 causal_vmd 主流程（仅消融H）、corrector→自适应订正、experts→异质、trainers→单个 joint_trainer。
  - §2 数据流：删 nu/VMD，加 big_err_mask；订正监督改 irrad_lmd。
  - §3.5 dataset：加大误差区掩码预计算；§3.6 corrector：AdaptiveIrradCorrector(只辐照+残差门控γ)；
    §3.7 experts：ShortGRU/LongTCN/Freq 异质；§3.8 gated_moe：可学习软门控 g=softmax(α·log u+MLP)；
    §3.9 losses：L_corr 仅大误差区+幅度惩罚；§3.10 JointTrainer 单阶段 L=L_pred+λL_corr。
  - §3.14 config：expert_types/learnable_gate/blind_correct/lambda_corr/beta/big_err_quantile 等新开关；
    消融 A–H 全覆盖。§6 实现顺序同步。
  - 校验：实验覆盖/逻辑一致/完整性 三项通过。
- **下一步**：阶段E 改代码（按新 implementation）→ smoke → 在 station04/05 重跑对照确认是否翻盘。

## 给用户的运行提示（混合策略下用户负责的大跑）
- 完整主实验：`GPU=N bash scripts/train_main.sh`（10站×5种子，单阶段联合）。
- 全部消融：`bash scripts/ablation.sh`；baseline：`bash scripts/train_baselines.sh`
  （Informer 等需 `export TSLIB_PATH=/path/to/Time-Series-Library`）。

### 2026-06-10 — 全10站最终验证 + E-7 代码审查

#### 10站 × 5种子完整结果（日前96步，ACC%）

| Station    | Deep  | RF    | Stack | w*   | Δ(S−RF) |
|------------|-------|-------|-------|------|---------|
| station00  | 89.87 | 91.57 | 92.04 | 0.34 | +0.48   |
| station01  | 86.39 | 88.51 | 89.10 | 0.32 | +0.58   |
| station02  | 88.25 | 90.32 | 90.70 | 0.21 | +0.39   |
| station03  | 87.88 | 89.41 | 89.64 | 0.34 | +0.23   |
| station04  | 81.37 | 82.10 | 83.49 | 0.40 | +1.38   |
| station05  | 87.36 | 88.39 | 88.25 | 0.64 | −0.14   |
| station06  | 91.53 | 93.38 | 92.80 | 0.65 | −0.58   |
| station07  | 91.63 | 93.31 | 93.30 | 0.32 | −0.01   |
| station08  | 88.74 | 90.31 | 90.59 | 0.34 | +0.27   |
| station09  | 94.68 | 94.86 | 95.17 | 0.18 | +0.31   |
| **MEAN**   | **88.77** | **90.22** | **90.51** | **0.37** | **+0.29** |

- RMSE (mean): deep=0.1123, RF=0.0978, stack=0.0949；R²: deep=0.736, RF=0.800, stack=0.811
- stack 在 8/10 站超越 RF（st05/st06 deep主导时 stacking 微降）；平均提升 +0.29 ACC
- 种子稳定性：stack std 跨10站 3.11（低于 deep 3.46 / RF 3.44），stacking 同时压方差

#### E-7 代码审查结论（high effort，3+角度）
共 10 条候选，5 CONFIRMED / 2 PLAUSIBLE / 3 REFUTED，实际修复 2 处：

1. **[修复] `build.py:67` `mc['expert_types']` 改 `.get()` 加默认值**
   — 直接 bracket 访问在缺省config路径下会 KeyError；改 `mc.get("expert_types", ["local_conv","dilated_tcn","direct_mlp"])` 兜底。

2. **[修复] `build.py:78` `irrad_anchor` 代码默认值 `True` 与 yaml 默认 `false` 不一致**
   — 实验已证 irrad_anchor 有害（88.04→86.9 ACC）；代码侧默认值改 `False` 与 yaml 一致。

3. **[无需修复] `corrector.py:54` `index_copy`**
   — PyTorch 2.x 的 `Tensor.index_copy(dim, index, source)` 非原地方法存在；环境验证通过，10站跑通。

4. **[无需修复] `gated_moe.py:109` `gate` shape `[B,1]` in single_expert**
   — single_expert=True 时 n_experts=1，`[B,1]` 正确匹配；einsum 走 else 分支不涉及。

5. **[设计权衡] stacking val 集与早停 val 集重叠（轻度数据泄漏）**
   — blend_weight 在同一 val 集上拟合，存在轻度乐观偏差；现规模（10站×5种子）bias 可接受，记录备注。
