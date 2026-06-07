"""RF 基线：历史功率 RF、RF+原始 NWP（前身基线，新 7:1:2 划分下重跑）。

借鉴前身 code/rf.py 的特征工程与 RF 训练，但：
  - 划分改为 7:1:2 时间顺序（与主方法一致，公平可比）；
  - 评估口径走 src.utils.metrics（仅白天，容量归一）；
  - 直接做日前形式：用历史特征预测，逐点回归后按测试段整体评估。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..utils import metrics as M
from ..utils.config import load_config
from ..trainers.build import load_capacity


def _features(df: pd.DataFrame, use_nwp: bool) -> pd.DataFrame:
    """构造 RF 特征（因果 lag/滑动统计 + 可选原始 NWP）。"""
    out = df.copy()
    out["date_time"] = pd.to_datetime(out["date_time"])
    out["hour"] = out["date_time"].dt.hour
    for lag in range(1, 5):
        out[f"lag_{lag}"] = out["power"].shift(lag)
    w = 4
    shifted = out["power"].shift(w - 1)
    for i in range(1, 5):
        out[f"rcr_{i}"] = (shifted.rolling(w).max().shift((i - 1) * 2)
                           - shifted.rolling(w).min().shift((i - 1) * 2)) / (w - 1)
    cols = ["hour"] + [f"lag_{l}" for l in range(1, 5)] + [f"rcr_{i}" for i in range(1, 5)]
    if use_nwp:
        nwp_cols = [c for c in df.columns if c.startswith("nwp_")]
        cols += nwp_cols
    out = out.fillna(0.0)
    return out[["date_time", "power", "is_day"] + cols], cols


def run_rf(cfg, sid, use_nwp: bool, tag: str):
    proc = cfg["data"]["processed_dir"]
    df = pd.read_csv(os.path.join(proc, f"{sid}.csv"))
    feat_df, cols = _features(df, use_nwp)

    n = len(feat_df)
    r_tr, r_va, _ = cfg["data"]["split"]
    i_tr = int(n * r_tr)
    i_va = int(n * (r_tr + r_va))
    train = feat_df.iloc[:i_tr]
    test = feat_df.iloc[i_va:]

    model = RandomForestRegressor(n_estimators=100, random_state=cfg["seed"], n_jobs=-1)
    model.fit(train[cols].values, train["power"].values)
    pred = model.predict(test[cols].values)

    cap = load_capacity(os.path.join(cfg["data"]["raw_dir"], "metadata.csv"), sid)
    res = M.compute_all(pred, test["power"].values, cap, test["is_day"].values)
    res.update({"station": sid, "split": "test", "model": tag})

    eval_dir = os.path.join(cfg["paths"]["eval"], "baselines", tag)
    os.makedirs(eval_dir, exist_ok=True)
    out = os.path.join(eval_dir, f"eval_{sid}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[rf:{tag}] {sid}: ACC={res['acc']:.3f} → {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--station", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_rf(cfg, args.station, use_nwp=False, tag="rf_hist")
    run_rf(cfg, args.station, use_nwp=True, tag="rf_nwp")


if __name__ == "__main__":
    main()
