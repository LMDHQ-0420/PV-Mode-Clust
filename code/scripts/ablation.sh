#!/usr/bin/env bash
# 批量消融 A–G：每个变体独立 ckpt/eval 目录，训练+评估后汇总到 summary.csv。
# 用法：bash scripts/ablation.sh [station ...]   (缺省全 10 站，5 种子)
# 注意：完整跑量大，建议先用单站验证：bash scripts/ablation.sh station00
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

# 变体名:config:说明
VARIANTS=(
  "A:configs/ablation/A_full.yaml:完整模型"
  "B:configs/ablation/B_wo_corrector.yaml:w/o订正(RQ1)"
  "C:configs/ablation/C_hard_gate.yaml:硬门控(RQ2)"
  "D:configs/ablation/D_single_expert.yaml:单专家(RQ2)"
  "E:configs/ablation/E_leak_vmd.yaml:全序列VMD泄漏(RQ3)"
  "F:configs/ablation/F_wo_vmd.yaml:w/o VMD(RQ3)"
  "G2:configs/ablation/G_K2.yaml:K=2"
  "G4:configs/ablation/G_K4.yaml:K=4"
  "G5:configs/ablation/G_K5.yaml:K=5"
)

SUMMARY=results/ablation/summary.csv
rm -f "$SUMMARY"

for entry in "${VARIANTS[@]}"; do
  IFS=":" read -r name cfg note <<< "$entry"
  ckpt_dir="results/ablation/$name/checkpoints"
  eval_dir="results/ablation/$name/eval"
  echo "######## 变体 $name ($note) ########"
  for sid in "${STATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python -m src.run train    --config configs/default.yaml --override "$cfg" \
        --station "$sid" --seed "$seed" --ckpt_dir "$ckpt_dir" $GPU_ARG
      python -m src.run evaluate --config configs/default.yaml --override "$cfg" \
        --station "$sid" --seed "$seed" --ckpt_dir "$ckpt_dir" --eval_dir "$eval_dir" $GPU_ARG
    done
  done
  python -m src.summarize --eval_dir "$eval_dir" --variant "$name" \
    --notes "$note" --out_csv "$SUMMARY"
done
echo "消融完成 → $SUMMARY"
