"""汇总评估结果 → 消融汇总表 / 主表。

读取 results/ 下 eval_*.json，按 (variant, station) 聚合 10 站×5 种子，
产出 results/ablation/summary.csv（implementation §5.5）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def collect(eval_dir: str) -> pd.DataFrame:
    """读取目录下所有 eval_*.json 为一张长表。"""
    rows = []
    for p in glob.glob(os.path.join(eval_dir, "eval_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def summarize_variant(eval_dir: str, variant: str, notes: str,
                      out_csv: str, baseline_acc: float | None = None) -> dict:
    """聚合一个变体的 10 站×5 种子结果，追加写 summary.csv。"""
    df = collect(eval_dir)
    if df.empty:
        print(f"[summarize] {eval_dir} 无 eval json")
        return {}
    row = {
        "variant": variant,
        "acc_mean": float(df["acc"].mean()), "acc_std": float(df["acc"].std()),
        "rmse_mean": float(df["rmse"].mean()),
        "mae_mean": float(df["mae"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "notes": notes,
    }
    row["delta_acc"] = (row["acc_mean"] - baseline_acc) if baseline_acc is not None else 0.0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = not os.path.exists(out_csv)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=header, index=False)
    print(f"[summarize] {variant}: ACC={row['acc_mean']:.3f}±{row['acc_std']:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="该变体的 eval json 目录")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--out_csv", default="results/ablation/summary.csv")
    ap.add_argument("--baseline_acc", type=float, default=None)
    args = ap.parse_args()
    summarize_variant(args.eval_dir, args.variant, args.notes,
                      args.out_csv, args.baseline_acc)


if __name__ == "__main__":
    main()
