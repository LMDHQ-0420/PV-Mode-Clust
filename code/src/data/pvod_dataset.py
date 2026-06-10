"""PVOD 数据集类：读取 processed csv → 划分 → 归一化 → FCM 区制 → 大误差区掩码 → 滑窗。

E-8 重塑（2026-06-08）：VMD 退出主流程（仅 leak_vmd 消融时拼入 x_hist）；订正监督改为
仅辐照（irrad_lmd）+ 大误差区掩码（big_err_mask）。

产出每个样本（implementation §2 数据流）：
  x_nwp:        [L+H, D_nwp]   未来+历史 NWP 预报（日前可得）
  x_hist:       [L, D_hist]    历史功率 + 历史 NWP + lag/统计特征（leak_vmd 时附 VMD 模态）
  u:            [L+H, K]       软隶属（门控先验 + 订正条件）
  y:            [H]            未来 96 点功率（归一化标签）
  irrad_lmd:    [L+H]          LMD 实测辐照（订正监督，归一化）
  big_err_mask: [L+H]          大误差区掩码（订正损失用）
  is_day:       [H]            白天掩码（未来段）
  capacity:     标量           装机容量（kW，评估反归一用）

防泄漏：normalizer / FCM 中心 / 大误差区阈值均只用 train 段拟合并外部传入。
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from .fcm_regime import FCMRegime
from .preprocess import PAIRING_INDEX
from .transforms import MinMaxNormalizer

IRRAD_NWP = "nwp_globalirrad"
IRRAD_LMD = "lmd_totalirrad"


def list_nwp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("nwp_")]


def list_lmd_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("lmd_")]


def paired_cols(df: pd.DataFrame):
    """返回实际存在的 (nwp_side, lmd_side) 配对列两组列表（订正基线仍用）。"""
    nwp_side, lmd_side = [], []
    for _, (n, l) in PAIRING_INDEX.items():
        if n in df.columns and l in df.columns:
            nwp_side.append(n)
            lmd_side.append(l)
    return nwp_side, lmd_side


def _engineer_features(power: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """lag/滑动统计/时间特征（借鉴前身 extract_features，全部因果）。返回 [T, 9]。"""
    T = len(power)
    feats = [hour / 23.0]
    for lag in range(1, 5):
        feats.append(np.concatenate([np.zeros(lag), power[:-lag]]) if lag < T
                     else np.zeros(T))
    w = 4
    shifted = pd.Series(power).shift(w - 1)
    for i in range(1, 5):
        cr = (shifted.rolling(w).max().shift((i - 1) * 2)
              - shifted.rolling(w).min().shift((i - 1) * 2)) / (w - 1)
        feats.append(cr.fillna(0.0).values)
    return np.stack(feats, axis=1)


def fit_normalizer_fcm(train_df: pd.DataFrame, fcm_cfg: dict):
    """在 train 段拟合 normalizer 与 FCM（外部调用，防泄漏）。"""
    nwp_cols = list_nwp_cols(train_df)
    lmd_cols = list_lmd_cols(train_df)
    norm_cols = list(dict.fromkeys(nwp_cols + lmd_cols + ["power"]))
    normalizer = MinMaxNormalizer().fit(train_df, norm_cols)

    fcm = FCMRegime(n_clusters=fcm_cfg["K"], m=fcm_cfg["m"],
                    feature_cols=fcm_cfg["feature_cols"],
                    error=fcm_cfg.get("error", 0.005),
                    maxiter=fcm_cfg.get("maxiter", 1000))
    tr_norm = normalizer.transform(train_df)
    fcm.fit(tr_norm[fcm_cfg["feature_cols"]].values)
    return normalizer, fcm


class PVODDataset(Dataset):
    """单站、单 split 的滑窗数据集。"""

    def __init__(self, sid: str, split: str, df_full: pd.DataFrame,
                 normalizer: MinMaxNormalizer, fcm: FCMRegime,
                 cfg: dict, capacity: float,
                 leak_vmd: bool = False, return_lmd: bool = True,
                 vmd_cache_dir: str = "data/processed"):
        self.sid = sid
        self.split = split
        self.L = cfg["data"]["look_back"]
        self.H = cfg["data"]["horizon"]
        self.capacity = float(capacity)
        self.leak_vmd = leak_vmd
        self.return_lmd = return_lmd
        self.big_err_quantile = cfg.get("corrector", {}).get("big_err_quantile", 0.6)

        df_full = df_full.copy()
        df_full["date_time"] = pd.to_datetime(df_full["date_time"])
        df_full = df_full.sort_values("date_time").reset_index(drop=True)
        n = len(df_full)
        r_tr, r_va, r_te = cfg["data"]["split"]
        i_tr = int(n * r_tr)
        i_va = int(n * (r_tr + r_va))
        bounds = {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, n)}
        lo, hi = bounds[split]

        df_norm = normalizer.transform(df_full)
        self.nwp_cols = list_nwp_cols(df_full)
        self.fcm_cols = cfg["fcm"]["feature_cols"]
        # 辐照列索引（在 nwp_cols 中），供模型订正定位
        self.irrad_idx = self.nwp_cols.index(IRRAD_NWP) if IRRAD_NWP in self.nwp_cols else 0

        power_all = df_norm["power"].values.astype(float)
        hour_all = pd.to_datetime(df_full["date_time"]).dt.hour.values.astype(float)
        nwp_all = df_norm[self.nwp_cols].values.astype(float)
        eng_all = _engineer_features(power_all, hour_all)
        hist_all = np.concatenate([power_all[:, None], nwp_all, eng_all], axis=1)

        # leak_vmd 消融：全序列 VMD 模态拼进 x_hist（故意泄漏，仅演示）
        if leak_vmd:
            from .causal_vmd import global_vmd_features
            vcfg = cfg["vmd"]
            cache = os.path.join(vmd_cache_dir, f"{sid}_vmd_leak.npy")
            if os.path.exists(cache) and len(np.load(cache)) == n:
                nu = np.load(cache)
            else:
                os.makedirs(vmd_cache_dir, exist_ok=True)
                nu = global_vmd_features(power_all, vcfg["K_modes"], vcfg["alpha"])
                np.save(cache, nu)
            hist_all = np.concatenate([hist_all, nu], axis=1)

        self.hist_all = hist_all
        self.nwp_all = nwp_all
        self.u_all = fcm.soft_membership(df_norm[self.fcm_cols].values)
        self.power_all = power_all
        self.is_day_all = (df_full["is_day"].values.astype(float)
                           if "is_day" in df_full else np.ones(n))

        # 辐照 LMD（订正监督）+ 大误差区掩码
        if IRRAD_LMD in df_norm.columns:
            self.irrad_lmd_all = df_norm[IRRAD_LMD].values.astype(float)
            irrad_nwp_all = nwp_all[:, self.irrad_idx]
            err = np.abs(irrad_nwp_all - self.irrad_lmd_all)
            # 阈值只用 train 段白天样本算（防泄漏）
            tr_day = (self.is_day_all[:i_tr] > 0)
            thr = (np.quantile(err[:i_tr][tr_day], self.big_err_quantile)
                   if tr_day.any() else 0.0)
            self.big_err_mask_all = ((err > thr) & (self.is_day_all > 0)).astype(float)
        else:
            self.irrad_lmd_all = np.zeros(n)
            self.big_err_mask_all = np.zeros(n)

        self.lo, self.hi = lo, hi
        self.seg_len = hi - lo
        self.n_samples = max(0, self.seg_len - self.L - self.H + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        a = self.lo + idx
        h0, h1 = a, a + self.L
        f0, f1 = a + self.L, a + self.L + self.H
        full0, full1 = a, a + self.L + self.H

        out = {
            "x_nwp": self.nwp_all[full0:full1],            # [L+H, D_nwp]
            "x_hist": self.hist_all[h0:h1],                # [L, D_hist]
            "u": self.u_all[full0:full1],                  # [L+H, K]
            "y": self.power_all[f0:f1],                    # [H]
            "is_day": self.is_day_all[f0:f1],              # [H]
            "irrad_lmd": self.irrad_lmd_all[full0:full1],  # [L+H]
            "big_err_mask": self.big_err_mask_all[full0:full1],  # [L+H]
            "capacity": np.float32(self.capacity),
        }
        if torch is not None:
            out = {k: (torch.as_tensor(v, dtype=torch.float32)
                       if not np.isscalar(v) else torch.tensor(float(v)))
                   for k, v in out.items()}
        return out
