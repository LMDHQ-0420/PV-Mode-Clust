#!/usr/bin/env bash
# 主方法两阶段训练：10 站 × 5 种子。
# 用法：bash scripts/train_main.sh [station ...]   (缺省全 10 站)
# GPU：设环境变量 GPU=2 指定卡，例如 GPU=1 bash scripts/train_main.sh station05
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

for sid in "${STATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "==== train $sid seed=$seed ${GPU:+(gpu=$GPU)} ===="
    python -m src.run train    --config configs/default.yaml --station "$sid" --seed "$seed" $GPU_ARG
    python -m src.run evaluate --config configs/default.yaml --station "$sid" --seed "$seed" $GPU_ARG
  done
done
echo "主方法训练+评估完成。"
