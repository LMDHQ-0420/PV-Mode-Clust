"""时序 baseline 统一封装：DLinear / 自包含DL系 / TSLib系。

自包含（无外部依赖）：DLinear, LSTM, LSTNet, TCN, NBEATS, NHiTS, Crossformer, NWPLSTMbaseline。
TSLib依赖（需 TSLIB_PATH）：Informer, PatchTST, iTransformer, TimesNet。

输入构造：x_hist [B, L, d_hist]（± x_nwp_fut [B, H, d_nwp] 对 NWP-aware 模型）→ [B, H]。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..trainers.build import build_station_data
from ..utils import metrics as M
from ..utils.config import load_config
from ..utils.seed import set_seed
from .dl_baselines import build_dl_model, NWP_AWARE_MODELS

TSLIB_MODELS = {"Informer", "PatchTST", "iTransformer", "TimesNet"}
DL_MODELS = {"LSTM", "LSTNet", "TCN", "NBEATS", "NHiTS", "Crossformer", "NWPLSTMbaseline"}


# ---------------- 自包含 DLinear ----------------
class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):  # x [B, L, C]
        pad = (self.kernel - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.transpose(1, 2)).transpose(1, 2)


class DLinear(nn.Module):
    """DLinear：序列分解 + 通道独立线性映射（Zeng et al. 2023）。

    这里聚焦预测 power 单变量：从 [B, L, C] 取全部通道分解后线性到 [B, H]，
    再对通道求和投影到功率（简化但忠实 DLinear 思想）。
    """

    def __init__(self, seq_len: int, pred_len: int, channels: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        self.lin_seasonal = nn.Linear(seq_len, pred_len)
        self.lin_trend = nn.Linear(seq_len, pred_len)
        self.proj = nn.Linear(channels, 1)

    def forward(self, x):  # x [B, L, C] → [B, H]
        trend = self.decomp(x)
        seasonal = x - trend
        s = self.lin_seasonal(seasonal.transpose(1, 2))   # [B, C, H]
        t = self.lin_trend(trend.transpose(1, 2))         # [B, C, H]
        out = (s + t).transpose(1, 2)                     # [B, H, C]
        return self.proj(out).squeeze(-1)                 # [B, H]


# ---------------- TSLib 集成 ----------------
def _build_tslib_model(name, cfg, d_in):
    """从官方 Time-Series-Library 构建模型（需 tslib_path）。"""
    tslib = cfg.get("baselines", {}).get("tslib_path") or os.environ.get("TSLIB_PATH")
    if not tslib or not os.path.isdir(tslib):
        raise RuntimeError(
            f"模型 {name} 需要 Time-Series-Library：请 clone thuml/Time-Series-Library，"
            f"并在 configs.baselines.tslib_path 或环境变量 TSLIB_PATH 指向它。"
        )
    sys.path.insert(0, tslib)
    import importlib
    mod = importlib.import_module(f"models.{name}")
    from argparse import Namespace
    H = cfg["data"]["horizon"]; L = cfg["data"]["look_back"]
    conf = Namespace(
        task_name="long_term_forecast", seq_len=L, label_len=L // 2, pred_len=H,
        enc_in=d_in, dec_in=d_in, c_out=1, d_model=64, n_heads=4, e_layers=2,
        d_layers=1, d_ff=128, factor=3, dropout=0.1, embed="timeF", freq="t",
        activation="gelu", output_attention=False, moving_avg=25,
        top_k=5, num_kernels=6, distil=True, patch_len=16, stride=8,
    )
    return mod.Model(conf)


class TSLibWrapper(nn.Module):
    """把 TSLib 模型统一成 forward(x)->[B,H]（只取 power 通道输出）。"""

    def __init__(self, name, cfg, d_in):
        super().__init__()
        self.name = name
        self.core = _build_tslib_model(name, cfg, d_in)
        self.H = cfg["data"]["horizon"]

    def forward(self, x):  # x [B, L, C]
        B, L, C = x.shape
        x_mark = torch.zeros(B, L, 4, device=x.device)
        dec_inp = torch.zeros(B, self.H, C, device=x.device)
        dec_mark = torch.zeros(B, self.H, 4, device=x.device)
        out = self.core(x, x_mark, dec_inp, dec_mark)  # [B, H, c_out]
        return out[..., 0]


def build_ts_model(name, cfg, d_in):
    L = cfg["data"]["look_back"]; H = cfg["data"]["horizon"]
    if name == "DLinear":
        return DLinear(L, H, d_in)
    if name in DL_MODELS:
        return build_dl_model(name, cfg, d_in, d_nwp=7)
    if name in TSLIB_MODELS:
        return TSLibWrapper(name, cfg, d_in)
    raise ValueError(f"未知时序 baseline：{name}")


# ---------------- 通用训练/评估 ----------------
def _x_from_batch(b):
    """baseline 输入：历史 NWP + 历史功率（x_hist 已含），取 [B, L, d_hist]。"""
    return b["x_hist"]


def train_eval(cfg, sid, name, seed):
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    datasets, meta = build_station_data(cfg, sid)
    d_in = meta["dims"]["d_hist"]
    model = build_ts_model(name, cfg, d_in).to(device)

    tc = cfg["train"]
    L = cfg["data"]["look_back"]
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    from ..models.losses import prediction_loss
    nwp_aware = name in NWP_AWARE_MODELS

    def _forward(b):
        if nwp_aware:
            return model(_x_from_batch(b), b["x_nwp"][:, L:, :])
        return model(_x_from_batch(b))

    def run(ds, train):
        loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=train)
        model.train(train)
        tot, cnt = 0.0, 0
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            yh = _forward(b)
            loss = prediction_loss(yh, b["y"], b["is_day"])
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
            tot += loss.item() * b["y"].size(0); cnt += b["y"].size(0)
        return tot / max(cnt, 1)

    best, bad, best_state = float("inf"), 0, None
    for ep in range(1, tc["max_epochs"] + 1):
        run(datasets["train"], True)
        with torch.no_grad():
            va = run(datasets["val"], False)
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)

    # 评估（反归一化）
    model.eval()
    normalizer = meta["normalizer"]; cap = meta["capacity"]
    preds, trues, days = [], [], []
    with torch.no_grad():
        for b in DataLoader(datasets["test"], batch_size=tc["batch_size"]):
            b = {k: v.to(device) for k, v in b.items()}
            yh = _forward(b)
            preds.append(normalizer.inverse("power", yh.cpu().numpy()))
            trues.append(normalizer.inverse("power", b["y"].cpu().numpy()))
            days.append(b["is_day"].cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues); days = np.concatenate(days)
    res = M.day_ahead_rolling(preds, trues, days, cap, cfg["data"]["horizon"])
    res.update({"station": sid, "model": name, "seed": seed})

    eval_dir = os.path.join(cfg["paths"]["eval"], "baselines", name)
    os.makedirs(eval_dir, exist_ok=True)
    out = os.path.join(eval_dir, f"eval_{sid}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[ts:{name}] {sid}: ACC={res['acc']:.3f} → {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--station", required=True)
    ap.add_argument("--model", required=True,
                    help="DLinear/Informer/PatchTST/iTransformer/TimesNet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None, help="指定 GPU id")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.gpu is not None:
        cfg["train"]["device"] = f"cuda:{args.gpu}"
    train_eval(cfg, args.station, args.model, args.seed)


if __name__ == "__main__":
    main()
