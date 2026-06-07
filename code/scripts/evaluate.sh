#!/usr/bin/env bash
# 日前协议评估指定站点的 best checkpoint。
# 用法：bash scripts/evaluate.sh station00 [seed]
set -e
cd "$(dirname "$0")/.."
SID=${1:-station00}
SEED=${2:-0}
GPU_ARG=""; [ -n "$GPU" ] && GPU_ARG="--gpu $GPU"
python -m src.run evaluate --config configs/default.yaml --station "$SID" --seed "$SEED" $GPU_ARG
