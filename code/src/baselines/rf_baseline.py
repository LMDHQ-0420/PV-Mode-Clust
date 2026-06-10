"""RF 基线：历史功率 RF、RF+未来段 NWP —— **日前 96 步直接多步**，与主方法同口径。

E-8 修订（2026-06-08）：原实现用 lag_1..4=前几个点的真实功率做逐点回归，相当于预测 t
点时偷看 t-1（15min 前）真值，是超短期/持续性预测，对日前深度 baseline 不公平。
现改为日前直接多步：复用 PVODDataset 的滑窗样本（与主方法完全相同的 split/样本/评估），
把历史窗口（+可选未来段 NWP 预报）展平成特征，RF 一次性回归未来 96 维。

  - rf_hist：仅历史功率窗口 [L] 展平 → 预测 [H]。
  - rf_nwp ：历史功率窗口 [L] + 未来段 NWP 预报 [H, d_nwp] 展平 → 预测 [H]。
评估走 day_ahead_rolling（仅白天、容量归一），与主方法一致。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..trainers.build import build_station_data
from ..utils import metrics as M
from ..utils.config import load_config


def _collect_xy(ds, use_nwp: bool):
    """从 PVODDataset 取全部样本，展平为 RF 特征矩阵 X 与多输出标签 Y。

    历史功率窗口：x_hist 的第 0 列（power，归一化）[L]。
    未来段 NWP：x_nwp 的未来 H 段 [H, d_nwp]（日前可得）。

    Returns:
        X [N, F], Y [N, H], is_day [N, H]
    """
    L = ds.L
    Xs, Ys, Ds = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        x_hist = s["x_hist"].numpy()           # [L, d_hist]，第0列为历史功率
        x_nwp = s["x_nwp"].numpy()             # [L+H, d_nwp]
        y = s["y"].numpy()                     # [H]
        d = s["is_day"].numpy()                # [H]
        feat = [x_hist[:, 0]]                   # 历史功率 [L]
        if use_nwp:
            feat.append(x_nwp[L:, :].reshape(-1))  # 未来段 NWP 预报展平 [H*d_nwp]
        Xs.append(np.concatenate(feat))
        Ys.append(y)
        Ds.append(d)
    return np.stack(Xs), np.stack(Ys), np.stack(Ds)


def run_rf(cfg, sid, use_nwp: bool, tag: str):
    # 复用主方法的数据装配，保证 split/样本/归一化/容量完全同口径
    datasets, meta = build_station_data(cfg, sid)
    normalizer = meta["normalizer"]
    cap = meta["capacity"]

    Xtr, Ytr, _ = _collect_xy(datasets["train"], use_nwp)
    Xte, Yte, Dte = _collect_xy(datasets["test"], use_nwp)

    # RandomForestRegressor 原生支持多目标输出（Y [N, H]），一片森林同时回归 96 维，
    # 等价于但远快于 MultiOutputRegressor 包 96 个独立森林。
    model = RandomForestRegressor(n_estimators=100, random_state=cfg["seed"], n_jobs=-1)
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte)                  # [N, H]，归一化空间

    # 反归一化到 kW，再走日前协议评估（与主方法一致）
    pred_kw = normalizer.inverse("power", pred)
    true_kw = normalizer.inverse("power", Yte)
    res = M.day_ahead_rolling(pred_kw, true_kw, Dte, cap, cfg["data"]["horizon"])
    res.update({"station": sid, "split": "test", "model": tag})

    eval_dir = os.path.join(cfg["paths"]["eval"], "baselines", tag)
    os.makedirs(eval_dir, exist_ok=True)
    out = os.path.join(eval_dir, f"eval_{sid}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[rf:{tag}] {sid}: ACC={res['acc']:.3f} RMSE={res['rmse']:.4f} → {out}")
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
