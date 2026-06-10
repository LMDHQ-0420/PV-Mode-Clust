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


def summarize_main(eval_dir: str = "results", out_csv: str = "results/main_table.csv"):
    """聚合 stack_*.json 主表：deep / rf / stack 三者 10站×5种子均值±std。"""
    rows = []
    for p in glob.glob(os.path.join(eval_dir, "stack_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"[summarize_main] {eval_dir} 无 stack json")
        return
    df = pd.DataFrame(rows)
    out = []
    for method in ("deep", "rf", "stack"):
        sub = pd.DataFrame([r[method] for r in rows])
        out.append({
            "method": method,
            "acc_mean": float(sub["acc"].mean()), "acc_std": float(sub["acc"].std()),
            "rmse_mean": float(sub["rmse"].mean()), "mae_mean": float(sub["mae"].mean()),
            "r2_mean": float(sub["r2"].mean()), "qr_mean": float(sub["qr"].mean()),
        })
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(res.to_string(index=False))
    print(f"[summarize_main] 平均拟合权重 w={df['blend_weight'].mean():.2f} → {out_csv}")


def summarize_baselines(base_dir: str = "results/baselines",
                        out_csv: str = "results/baselines_table.csv"):
    """聚合 results/baselines/*/eval_*.json → model × station 均值表。

    输出行=模型，列=站点（mean±std）+ MEAN 列（跨站均值）。
    RF 结果来自 results/rf_*.json（单种子，无 std）。
    """
    rows = []
    # DL baselines (results/baselines/{Model}/eval_{sid}_seed{s}.json)
    for p in glob.glob(os.path.join(base_dir, "*", "eval_*.json")):
        model_name = os.path.basename(os.path.dirname(p))
        with open(p) as f:
            d = json.load(f)
        rows.append({"model": model_name,
                     "station": d.get("station", "?"),
                     "seed": d.get("seed", 0),
                     "acc": float(d["acc"]),
                     "rmse": float(d["rmse"])})
    # RF baselines (results/rf_{sid}.json — deterministic, no seed)
    for p in glob.glob(os.path.join(os.path.dirname(base_dir), "rf_*.json")):
        with open(p) as f:
            d = json.load(f)
        for tag in ("rf_hist", "rf_nwp"):
            if tag in d:
                rows.append({"model": tag,
                             "station": d.get("station", "?"),
                             "seed": 0,
                             "acc": float(d[tag]["acc"]),
                             "rmse": float(d[tag]["rmse"])})

    if not rows:
        print(f"[summarize_baselines] {base_dir} 无评估结果")
        return

    df = pd.DataFrame(rows)
    agg = (df.groupby(["model", "station"])["acc"]
             .agg(mean="mean", std="std")
             .reset_index())
    agg["acc_str"] = agg.apply(
        lambda r: f"{r['mean']:.2f}" if np.isnan(r['std']) else f"{r['mean']:.2f}±{r['std']:.2f}",
        axis=1)
    table = agg.pivot(index="model", columns="station", values="acc_str")
    mean_by_model = df.groupby("model")["acc"].mean()
    table.insert(0, "MEAN", mean_by_model.map(lambda v: f"{v:.2f}"))

    ordered = ["rf_hist", "rf_nwp", "DLinear", "LSTM", "LSTNet", "TCN",
               "NBEATS", "NHiTS", "Crossformer", "NWPLSTMbaseline",
               "Informer", "PatchTST", "iTransformer", "TimesNet"]
    table = table.reindex([m for m in ordered if m in table.index])

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    table.to_csv(out_csv)
    print(table.to_string())
    print(f"\n[summarize_baselines] → {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ablation",
                    choices=["ablation", "main", "baselines"])
    ap.add_argument("--eval_dir", default="results", help="eval json 目录")
    ap.add_argument("--variant", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--baseline_acc", type=float, default=None)
    args = ap.parse_args()
    if args.mode == "main":
        summarize_main(args.eval_dir, args.out_csv or "results/main_table.csv")
    elif args.mode == "baselines":
        summarize_baselines(
            base_dir=os.path.join(args.eval_dir, "baselines"),
            out_csv=args.out_csv or "results/baselines_table.csv")
    else:
        summarize_variant(args.eval_dir, args.variant, args.notes,
                          args.out_csv or "results/ablation/summary.csv",
                          args.baseline_acc)


if __name__ == "__main__":
    main()
