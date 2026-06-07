#!/usr/bin/env bash
# 导出可解释中间量（区制隶属 / 门控权重 / 订正前后），供 notebooks 绘 4 图。
# 用法：bash scripts/interpret.sh station00 [seed]
set -e
cd "$(dirname "$0")/.."
SID=${1:-station00}
SEED=${2:-0}
GPU_ARG=""; [ -n "$GPU" ] && GPU_ARG="--gpu $GPU"
python -m src.run interpret --config configs/default.yaml --station "$SID" --seed "$SEED" $GPU_ARG
