"""Stacking 集成 —— 深度区制 MoE + RF 的互补凸组合（本文 SOTA 最终模型）。

依据（实验验证，2026-06-08）：单深度模型在该数据集只能逼平 RF+NWP；两者互补
（深度模型强于时序/区制结构，RF 强于未来 NWP→功率的直接非线性映射），其凸组合
y = w·deep + (1-w)·rf 在两站均稳定超过各自，w 在验证集上按 RMSE 最优拟合。
这是 PV 预测文献的标准 SOTA 手段（集成/stacking）。

产出 eval json（含 deep/rf/stack 三者指标 + 拟合权重 w），供主表。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

from .baselines.rf_baseline import _collect_xy
from .trainers.build import build_model, build_station_data
from .trainers.joint_trainer import JointTrainer
from .utils import metrics as M


@torch.no_grad()
def _deep_predict(model, ds_split, cfg, device, normalizer):
    """返回该 split 的 (pred_kw, true_kw, is_day)。"""
    P, T, D = [], [], []
    for b in DataLoader(ds_split, batch_size=cfg["train"]["batch_size"]):
        b = {k: v.to(device) for k, v in b.items()}
        yh, _ = model(b)
        P.append(normalizer.inverse("power", yh.cpu().numpy()))
        T.append(normalizer.inverse("power", b["y"].cpu().numpy()))
        D.append(b["is_day"].cpu().numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(D)


def _rf_predict(datasets, normalizer, seed):
    """训练 RF（历史窗+未来NWP，日前多步），返回 {split: pred_kw}。"""
    Xtr, Ytr, _ = _collect_xy(datasets["train"], use_nwp=True)
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, Ytr)
    out = {}
    for sp in ("val", "test"):
        Xs, _, _ = _collect_xy(datasets[sp], use_nwp=True)
        out[sp] = normalizer.inverse("power", rf.predict(Xs))
    return out


def fit_blend_weight(deep_val, rf_val, true_val, day_val, cap) -> float:
    """在验证集上按 RMSE 扫描最优凸组合权重 w∈[0,1]（步长0.05）。"""
    best_w, best = 0.5, float("inf")
    for w in np.linspace(0, 1, 21):
        blend = w * deep_val + (1 - w) * rf_val
        r = M.rmse(blend, true_val, cap, day_val)
        if r < best:
            best, best_w = r, float(w)
    return best_w


def run_stacking(cfg, sid, seed, device=None):
    """训练深度模型 + RF，拟合权重，评估 stacking，落 eval json。"""
    device = device or (cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    datasets, meta = build_station_data(cfg, sid)
    normalizer, cap = meta["normalizer"], meta["capacity"]

    # 深度模型
    _, model = build_model(cfg, meta["dims"])
    JointTrainer(model, datasets, cfg, sid, cap, device).fit()
    model.eval()
    dv = _deep_predict(model, datasets["val"], cfg, device, normalizer)
    dt = _deep_predict(model, datasets["test"], cfg, device, normalizer)

    # RF
    rf = _rf_predict(datasets, normalizer, seed)

    # 拟合权重（val）→ 评估（test）
    w = fit_blend_weight(dv[0], rf["val"], dv[1], dv[2], cap)
    blend = w * dt[0] + (1 - w) * rf["test"]
    H = cfg["data"]["horizon"]
    res_stack = M.day_ahead_rolling(blend, dt[1], dt[2], cap, H)
    res_deep = M.compute_all(dt[0], dt[1], cap, dt[2])
    res_rf = M.compute_all(rf["test"], dt[1], cap, dt[2])

    out = {
        "station": sid, "seed": seed, "blend_weight": w,
        "stack": res_stack, "deep": res_deep, "rf": res_rf,
        **{k: res_stack[k] for k in ("acc", "rmse", "mae", "r", "r2", "qr")},  # 顶层=stack
    }
    eval_dir = cfg["paths"]["eval"]
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, f"stack_{sid}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[stack] {sid} seed{seed}: deep={res_deep['acc']:.2f} rf={res_rf['acc']:.2f} "
          f"w={w:.2f} STACK={res_stack['acc']:.2f} → {path}")
    return out
