#!/usr/bin/env bash
# 训练所有 baseline：强时序 SOTA + RF + 订正基线。
# 统一在新 7:1:2 时间划分下重跑（与主方法公平可比，见 user_requirements 阶段C）。
# 用法：bash scripts/train_baselines.sh [station ...]
set -e
cd "$(dirname "$0")/.."

GPU_ARG=""
[ -n "$GPU" ] && GPU_ARG="--gpu $GPU"

STATIONS=("$@")
if [ ${#STATIONS[@]} -eq 0 ]; then
  STATIONS=(station00 station01 station02 station03 station04 \
            station05 station06 station07 station08 station09)
fi

# 强时序 SOTA（Time Series Library 封装）
TS_MODELS=(DLinear Informer PatchTST iTransformer TimesNet)

for sid in "${STATIONS[@]}"; do
  echo "==== baselines: $sid ===="
  # RF 基线（历史功率 / +原始NWP）
  python -m src.baselines.rf_baseline --config configs/default.yaml --station "$sid"
  # 订正基线（线性 / QM / RF 订正）
  python -m src.baselines.correction_baselines --config configs/default.yaml --station "$sid"
  # 强时序 SOTA
  for m in "${TS_MODELS[@]}"; do
    python -m src.baselines.ts_library_wrap --config configs/default.yaml \
      --station "$sid" --model "$m" $GPU_ARG
  done
done
echo "baseline 训练完成。"
