"""统一入口 —— 训练 / 评估 / 可解释导出。

子命令：
  train     单站单阶段联合训练（订正+异质专家+可学习门控），可指定 seed。
  evaluate  加载 best_{sid}.pth，日前协议评估，落 eval json + predictions csv。
  interpret 导出 aux 中间量（区制隶属/门控权重/订正对比/个案）供 5 图。

被 scripts/*.sh 调用，不含业务逻辑以外的参数拼装。
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
    corrector, model = build_model(cfg, meta["dims"])

    log_path = os.path.join(cfg["paths"]["logs"], f"train_{sid}_seed{seed}_{_ts()}.csv")
    logger = Logger(log_path)

    trainer = JointTrainer(model, datasets, cfg, sid, meta["capacity"], device, logger)
    ckpt = trainer.fit()
    print(f"[train] {sid} seed{seed}: joint done → {ckpt}")
    return ckpt


# ---------------- evaluate ----------------
@torch.no_grad()
def cmd_evaluate(cfg, sid, seed, ckpt=None):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    _, model = build_model(cfg, meta["dims"])
    ckpt = ckpt or os.path.join(cfg["paths"]["checkpoints"], f"best_{sid}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    normalizer = meta["normalizer"]
    cap = meta["capacity"]
    loader = DataLoader(datasets["test"], batch_size=cfg["train"]["batch_size"])

    preds, trues, days = [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        y_hat, _ = model(b)
        # 反归一化到 kW
        preds.append(normalizer.inverse("power", y_hat.cpu().numpy()))
        trues.append(normalizer.inverse("power", b["y"].cpu().numpy()))
        days.append(b["is_day"].cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues); days = np.concatenate(days)

    res = M.day_ahead_rolling(preds, trues, days, cap, cfg["data"]["horizon"])
    res.update({"station": sid, "split": "test", "seed": seed, "checkpoint": ckpt})

    eval_dir = cfg["paths"]["eval"]
    os.makedirs(eval_dir, exist_ok=True)
    out_json = os.path.join(eval_dir, f"eval_{sid}_seed{seed}_{_ts()}.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    # 逐样本预测（按天展开）
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
        os.path.join(eval_dir, f"predictions_{sid}_{_ts()}.csv"), index=False)
    print(f"[eval] {sid}: ACC={res['acc']:.3f} RMSE={res['rmse']:.4f} → {out_json}")
    return res


# ---------------- interpret ----------------
@torch.no_grad()
def cmd_interpret(cfg, sid, seed, ckpt=None):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    _, model = build_model(cfg, meta["dims"])
    ckpt = ckpt or os.path.join(cfg["paths"]["checkpoints"], f"best_{sid}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    out_dir = cfg["paths"]["interpret"]
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(datasets["test"], batch_size=cfg["train"]["batch_size"])
    H = cfg["data"]["horizon"]
    irrad_idx = meta["dims"]["irrad_idx"]

    u_list, gate_list = [], []
    corr_list, nwp_list, lmd_list, gamma_list = [], [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        y_hat, aux = model(b)
        u_list.append(aux["u"][:, -H:, :].cpu().numpy().mean(axis=1))   # [B,K]
        gate_list.append(aux["gate"].cpu().numpy())                      # [B,n_experts]
        if aux["x_corr"] is not None and aux["gamma"] is not None:
            corr_list.append(aux["x_corr"][:, -H:, irrad_idx].cpu().numpy())  # 订正后辐照
            nwp_list.append(b["x_nwp"][:, -H:, irrad_idx].cpu().numpy())      # 原始辐照
            lmd_list.append(b["irrad_lmd"][:, -H:].cpu().numpy())             # LMD 辐照
            gamma_list.append(aux["gamma"][:, -H:, 0].cpu().numpy())          # 残差门控

    u_arr = np.concatenate(u_list)
    gate_arr = np.concatenate(gate_list)
    # 图1：区制隶属
    pd.DataFrame(u_arr, columns=[f"regime_{k}" for k in range(u_arr.shape[1])]).to_csv(
        os.path.join(out_dir, f"regime_membership_{sid}.csv"), index=False)
    # 图2：天气×专家最终门控权重 g
    pd.DataFrame(gate_arr, columns=[f"expert_{k}" for k in range(gate_arr.shape[1])]).to_csv(
        os.path.join(out_dir, f"expert_weight_{sid}.csv"), index=False)
    # 图3：订正前后对比 + 残差门控 γ
    if corr_list:
        corr = np.concatenate(corr_list).mean(axis=0)
        nwp = np.concatenate(nwp_list).mean(axis=0)
        lmd = np.concatenate(lmd_list).mean(axis=0)
        gamma = np.concatenate(gamma_list).mean(axis=0)
        pd.DataFrame({"nwp_raw": nwp, "nwp_corrected": corr,
                      "lmd_true": lmd, "gamma": gamma}).to_csv(
            os.path.join(out_dir, f"correction_compare_{sid}.csv"), index=False)
    print(f"[interpret] {sid}: 中间量导出 → {out_dir}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate", "interpret", "stack"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default=None, help="消融变体 yaml")
    ap.add_argument("--station", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ckpt_dir", default=None, help="覆盖 paths.checkpoints（消融隔离）")
    ap.add_argument("--eval_dir", default=None, help="覆盖 paths.eval（消融隔离）")
    ap.add_argument("--gpu", type=int, default=None,
                    help="指定 GPU id（如 0/1/2/3）；缺省用 config 的 train.device（cuda:0）")
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
