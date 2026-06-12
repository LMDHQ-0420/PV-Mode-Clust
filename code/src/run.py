"""统一入口 —— 训练 / 评估 / 可解释导出 / stacking。

子命令：
  train     单站端到端训练（异质专家+可学习软门控），可指定 seed。
  evaluate  加载 best_{sid}.pth，日前协议评估，输出 eval json + predictions csv。
  interpret 导出中间量（区制隶属/门控权重）供可解释性图。
  stack     深度模型 × RF 互补 stacking。

被 scripts/*.sh 调用。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .trainers.build import build_model, build_station_data
from .trainers.joint_trainer import JointTrainer
from .utils import metrics as M
from .utils.config import load_config
from .utils.logger import Logger
from .utils.seed import set_seed


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------------- train ----------------
def cmd_train(cfg, sid, seed):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    model = build_model(cfg, meta["dims"])

    log_path = os.path.join(cfg["paths"]["logs"], f"train_{sid}_seed{seed}_{_ts()}.csv")
    logger = Logger(log_path)

    trainer = JointTrainer(model, datasets, cfg, sid, meta["capacity"], device, logger)
    ckpt = trainer.fit()
    print(f"[train] {sid} seed{seed} → {ckpt}")
    return ckpt


# ---------------- evaluate ----------------
@torch.no_grad()
def cmd_evaluate(cfg, sid, seed, ckpt=None):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    model = build_model(cfg, meta["dims"])
    ckpt = ckpt or os.path.join(cfg["paths"]["checkpoints"], f"best_{sid}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.to(device).eval()

    normalizer = meta["normalizer"]
    cap = meta["capacity"]
    loader = DataLoader(datasets["test"], batch_size=cfg["train"]["batch_size"])

    preds, trues, days = [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        y_hat, _ = model(b)
        preds.append(normalizer.inverse("power", y_hat.cpu().numpy()))
        trues.append(normalizer.inverse("power", b["y"].cpu().numpy()))
        days.append(b["is_day"].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    days = np.concatenate(days)

    res = M.day_ahead_rolling(preds, trues, days, cap, cfg["data"]["horizon"])
    res.update({"station": sid, "split": "test", "seed": seed, "checkpoint": ckpt})

    eval_dir = cfg["paths"]["eval"]
    os.makedirs(eval_dir, exist_ok=True)
    out_json = os.path.join(eval_dir, f"eval_{sid}_seed{seed}.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    # 逐天逐步预测明细
    H = cfg["data"]["horizon"]
    rows = []
    for d in range(preds.shape[0]):
        for s in range(H):
            rows.append({"day_id": d, "step": s + 1,
                         "true_power": float(trues[d, s]),
                         "pred_power": float(preds[d, s]),
                         "abs_error": float(abs(preds[d, s] - trues[d, s])),
                         "is_day": int(days[d, s])})
    pd.DataFrame(rows).to_csv(
        os.path.join(eval_dir, f"predictions_{sid}_seed{seed}.csv"), index=False)
    print(f"[eval] {sid} seed{seed}: ACC={res['acc']:.3f} RMSE={res['rmse']:.4f} → {out_json}")
    return res


# ---------------- interpret ----------------
@torch.no_grad()
def cmd_interpret(cfg, sid, seed, ckpt=None):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    model = build_model(cfg, meta["dims"])
    ckpt = ckpt or os.path.join(cfg["paths"]["checkpoints"], f"best_{sid}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.to(device).eval()

    out_dir = cfg["paths"]["interpret"]
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(datasets["test"], batch_size=cfg["train"]["batch_size"])
    H = cfg["data"]["horizon"]

    u_list, gate_list, true_list, pred_list, day_list = [], [], [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        y_hat, aux = model(b)
        u_list.append(aux["u"][:, -H:, :].cpu().numpy().mean(axis=1))  # [B, K]
        gate_list.append(aux["gate"].cpu().numpy())                     # [B, n_experts]
        true_list.append(b["y"].cpu().numpy())
        pred_list.append(y_hat.cpu().numpy())
        day_list.append(b["is_day"].cpu().numpy())

    u_arr = np.concatenate(u_list)
    gate_arr = np.concatenate(gate_list)
    trues = np.concatenate(true_list)
    preds_arr = np.concatenate(pred_list)
    days = np.concatenate(day_list)

    # 图1：每天区制软隶属均值
    pd.DataFrame(u_arr, columns=[f"regime_{k}" for k in range(u_arr.shape[1])]).to_csv(
        os.path.join(out_dir, f"regime_membership_{sid}.csv"), index=False)

    # 图2/3：每天专家门控权重
    pd.DataFrame(gate_arr, columns=[f"expert_{k}" for k in range(gate_arr.shape[1])]).to_csv(
        os.path.join(out_dir, f"expert_weight_{sid}.csv"), index=False)

    # 图4：软门控 vs 硬门控过渡带分析所需的逐天信息（entropy + error）
    from scipy.stats import entropy as scipy_entropy
    ent = np.array([scipy_entropy(u_arr[i] + 1e-8) for i in range(len(u_arr))])
    day_err = np.sqrt(((preds_arr - trues) ** 2 * days).sum(axis=1) / (days.sum(axis=1) + 1e-8))
    pd.DataFrame({
        "entropy": ent,
        "rmse_day": day_err,
        "dominant_regime": u_arr.argmax(axis=1),
    }).to_csv(os.path.join(out_dir, f"entropy_error_{sid}.csv"), index=False)

    print(f"[interpret] {sid}: 中间量导出 → {out_dir}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate", "interpret", "stack"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default=None, help="消融变体 yaml 路径")
    ap.add_argument("--station", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.ckpt_dir:
        cfg["paths"]["checkpoints"] = args.ckpt_dir
    if args.eval_dir:
        cfg["paths"]["eval"] = args.eval_dir
    if args.gpu is not None:
        cfg["train"]["device"] = f"cuda:{args.gpu}"

    if args.cmd == "train":
        cmd_train(cfg, args.station, args.seed)
    elif args.cmd == "evaluate":
        cmd_evaluate(cfg, args.station, args.seed, args.ckpt)
    elif args.cmd == "stack":
        from .stacking import run_stacking
        run_stacking(cfg, args.station, args.seed)
    else:
        cmd_interpret(cfg, args.station, args.seed, args.ckpt)


if __name__ == "__main__":
    main()
