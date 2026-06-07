"""订正基线：线性 / 均值 / 分位数映射(QM) / RF 订正 —— 产出订正后 NWP 再评估订正质量。

对照本文 NWPCorrector（RQ1）：这些基线不分天气区制，统一订正可配对列（辐照/温度/气压）。
评估指标：订正后 NWP 对 LMD 实测的 RMSE/MAE（仅白天），越低越好；落 json 供主表对比。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..data.pvod_dataset import paired_cols
from ..utils.config import load_config


def _split_idx(n, cfg):
    r_tr, r_va, _ = cfg["data"]["split"]
    return int(n * r_tr), int(n * (r_tr + r_va))


def _eval_corr(nwp_corr, lmd, is_day):
    m = is_day.astype(bool)
    e = (nwp_corr[m] - lmd[m])
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e)))}


def run_corrections(cfg, sid):
    proc = cfg["data"]["processed_dir"]
    df = pd.read_csv(os.path.join(proc, f"{sid}.csv"))
    nwp_p, lmd_p = paired_cols(df)
    if not nwp_p:
        print(f"[corr] {sid}: 无可配对列，跳过")
        return
    n = len(df)
    i_tr, i_va = _split_idx(n, cfg)
    is_day = df["is_day"].values
    is_day_test = is_day[i_va:]

    results = {}
    for nc, lc in zip(nwp_p, lmd_p):
        nwp = df[nc].values.astype(float)
        lmd = df[lc].values.astype(float)
        tr = slice(0, i_tr); te = slice(i_va, n)

        out = {}
        # 1) 原始（不订正）
        out["raw"] = _eval_corr(nwp[te], lmd[te], is_day_test)
        # 2) 均值偏移订正：减训练段平均偏差
        bias = np.mean((nwp[tr] - lmd[tr])[is_day[tr].astype(bool)])
        out["mean"] = _eval_corr(nwp[te] - bias, lmd[te], is_day_test)
        # 3) 线性订正：lmd ≈ a*nwp + b（train 拟合）
        mtr = is_day[tr].astype(bool)
        a, b = np.polyfit(nwp[tr][mtr], lmd[tr][mtr], 1)
        out["linear"] = _eval_corr(a * nwp[te] + b, lmd[te], is_day_test)
        # 4) 分位数映射 QM：把 nwp 分布映射到 lmd 分布（train 经验分位）
        qs = np.linspace(0, 1, 101)
        nq = np.quantile(nwp[tr][mtr], qs)
        lq = np.quantile(lmd[tr][mtr], qs)
        qm = np.interp(nwp[te], nq, lq)
        out["qm"] = _eval_corr(qm, lmd[te], is_day_test)
        # 5) RF 订正：以 nwp 全列预测 lmd
        nwp_cols = [c for c in df.columns if c.startswith("nwp_")]
        rf = RandomForestRegressor(n_estimators=100, random_state=cfg["seed"], n_jobs=-1)
        rf.fit(df[nwp_cols].values[tr][mtr], lmd[tr][mtr])
        rfp = rf.predict(df[nwp_cols].values[te])
        out["rf"] = _eval_corr(rfp, lmd[te], is_day_test)

        results[nc] = out

    eval_dir = os.path.join(cfg["paths"]["eval"], "baselines", "correction")
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, f"correction_{sid}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[corr] {sid}: 订正基线评估 → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--station", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_corrections(cfg, args.station)


if __name__ == "__main__":
    main()
