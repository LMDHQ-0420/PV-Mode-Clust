# 天气区制感知与 NWP 误差订正的软门控多专家光伏日前功率预测

> 关联设计：`../docs/idea_report.md`（idea / 实验设计）、`../docs/implementation.md`（实现指南）

## 项目主要内容

围绕 PVOD（河北 10 站，含 NWP 预报 + LMD 实测）数据，做**日前 96 点（15min × 1 天）光伏功率预测**。核心方法由四模块组成：

- **A. FCM 软天气区制**：仅用 NWP 侧气象特征做模糊 C 均值聚类，产出软隶属 `u(t)`，刻画"当前属于哪种天气区制"。
- **B. NWP 误差订正器**：以 `[NWP, u]` 为输入（推理可得），训练时用 LMD 实测做监督，分天气区制订正 NWP 偏差。
- **C. 多尺度专家**：因果滑窗 VMD 对历史功率做无泄漏分解作为特征；K 个轻量 GRU/TCN 专家，每个学一种区制下的"特征→未来功率"。
- **D. 软门控混合**：用区制软隶属 `u` 作门控权重，软混合各专家输出，得日前功率 ŷ，并预留可解释中间量输出。

两阶段训练：先训订正器（阶段一），再冻结订正器训专家+门控（阶段二）。

## 环境配置

环境：conda 环境 `zw@PV-Mode-Clust`（Python 3.10）；设备 RTX 3090（CUDA）。

```bash
conda activate zw@PV-Mode-Clust

# PyTorch 按官网指引装（匹配本机 CUDA 版本），例如 CUDA 12.1：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 其余依赖：
pip install -r requirements.txt
```

> `requirements.txt` 不含 torch/torchvision/torchaudio——这三者按官网命令单独装以匹配 CUDA。

## 数据准备

原始 PVOD 数据放在 `dataset/`：

```
code/dataset/
├── metadata.csv          # 含 Station_ID, Capacity 等
├── station00.csv         # 每站时序（date_time + nwp_*(7) + lmd_*(6) + power）
└── ... station09.csv
```

预处理（清洗对齐 + 因果 VMD 缓存）：

```bash
bash scripts/preprocess.sh
```

生成 `data/processed/{sid}.csv` 与 `data/processed/{sid}_vmd.npy`。

## 详细运行命令

所有命令在 `code/` 下、激活 `zw@PV-Mode-Clust` 环境后运行。

```bash
# 0) 预处理（清洗对齐 → data/processed/）。首次训练时按站懒生成因果 VMD 缓存
bash scripts/preprocess.sh

# 1) 主方法两阶段训练 + 评估（缺省 10 站 × 5 种子；可传站点子集）
bash scripts/train_main.sh                 # 全量
bash scripts/train_main.sh station05       # 单站快速验证

# 2) baseline 训练评估（RF / 订正基线 / 强时序 SOTA）
bash scripts/train_baselines.sh
#   说明：DLinear 自含可直接跑；Informer/PatchTST/iTransformer/TimesNet 需先
#   git clone thuml/Time-Series-Library 并设 TSLIB_PATH 指向它：
#   export TSLIB_PATH=/path/to/Time-Series-Library

# 3) 批量消融 A–G（每变体独立目录，自动汇总到 results/ablation/summary.csv）
bash scripts/ablation.sh                   # 全量（量大）
bash scripts/ablation.sh station05         # 单站验证

# 4) 单独评估某站 best checkpoint（日前协议）
bash scripts/evaluate.sh station00 0

# 5) 导出可解释中间量（区制隶属 / 门控权重 / 订正前后）→ results/interpret/
bash scripts/interpret.sh station00 0
```

可视化：`notebooks/01_data_demo.ipynb`（数据/FCM/VMD）、`02_model_demo.ipynb`（模型/订正）、
`03_results_demo.ipynb`（主表/消融/可解释 4 图）。

### 指定 GPU

本机 4× RTX 3090，两种方式指定卡：

```bash
# 方式 1：脚本用环境变量 GPU（推荐，便于并行铺多卡）
GPU=0 bash scripts/train_main.sh station00 station01 &
GPU=1 bash scripts/train_main.sh station02 station03 &
GPU=2 bash scripts/ablation.sh station04 &

# 方式 2：直接调 python，用 --gpu
python -m src.run train --config configs/default.yaml --station station00 --seed 0 --gpu 1
python -m src.baselines.ts_library_wrap --config configs/default.yaml \
  --station station00 --model DLinear --gpu 2

# 也可用 CUDA_VISIBLE_DEVICES 隔离（此时进程内 --gpu 0 即物理 3 号卡）
CUDA_VISIBLE_DEVICES=3 bash scripts/train_main.sh station05
```

> `--gpu N` 覆盖 config 的 `train.device` 为 `cuda:N`；脚本读环境变量 `GPU` 透传。
> 缺省（不指定）用 `configs/default.yaml` 的 `train.device: cuda`（即 cuda:0）。

## 目录结构

见 `../docs/implementation.md` §1.1。
