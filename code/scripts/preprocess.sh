#!/usr/bin/env bash
# 预处理：清洗对齐原始 PVOD → data/processed/{sid}.csv
# 因果 VMD 缓存在首次构建 Dataset 时按站懒生成（{sid}_vmd.npy）。
set -e
cd "$(dirname "$0")/.."   # 切到 code/

python -m src.data.preprocess \
  --raw_dir dataset \
  --out_dir data/processed \
  --metadata dataset/metadata.csv

echo "预处理完成 → data/processed/"
