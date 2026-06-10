#!/usr/bin/env bash
# 训练所有 baseline：RF + 订正基线 + 自包含DL系 + TSLib系（可选）。
# 统一日前96步同口径，10站×5种子。
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
SEEDS=(0 1 2 3 4)

# 自包含 DL baselines（无外部依赖）
DL_MODELS=(DLinear LSTM LSTNet TCN NBEATS NHiTS Crossformer NWPLSTMbaseline)

# TSLib 系（需 export TSLIB_PATH=...）
TSLIB_MODELS=(Informer PatchTST iTransformer TimesNet)

for sid in "${STATIONS[@]}"; do
  echo "==== baselines: $sid ===="

  # RF 基线（无种子，确定性）
  python -m src.baselines.rf_baseline --config configs/default.yaml --station "$sid"

  # 订正基线（无种子）
  python -m src.baselines.correction_baselines --config configs/default.yaml --station "$sid"

  # 自包含 DL baselines（5 种子）
  for m in "${DL_MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
      python -m src.baselines.ts_library_wrap \
        --config configs/default.yaml --station "$sid" \
        --model "$m" --seed "$s" $GPU_ARG
    done
  done

  # TSLib 系（5 种子，需 TSLIB_PATH）
  if [ -n "$TSLIB_PATH" ]; then
    for m in "${TSLIB_MODELS[@]}"; do
      for s in "${SEEDS[@]}"; do
        python -m src.baselines.ts_library_wrap \
          --config configs/default.yaml --station "$sid" \
          --model "$m" --seed "$s" $GPU_ARG
      done
    done
  else
    echo "[跳过 TSLib 模型] 未设置 TSLIB_PATH；如需运行请先: export TSLIB_PATH=/path/to/Time-Series-Library"
  fi
done
echo "baseline 训练完成。"
