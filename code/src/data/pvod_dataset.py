"""PVOD 数据集类：读取 processed csv → 划分 → 归一化 → FCM 区制 → 因果 VMD → 滑窗。

产出每个样本（implementation §2 数据流）：
  x_nwp:      [L+H, D_nwp]   未来+历史 NWP 预报（日前可得）
  x_hist:     [L, D_hist]    历史功率 + 历史 NWP + lag/统计特征
  nu:         [L, M]         因果 VMD 多尺度模态（对历史功率）
  u:          [L+H, K]       软隶属（门控 + 订正条件）
  y:          [H]            未来 96 点功率（归一化标签）
  lmd_paired: [L+H, D_p]     配对实测（订正监督，训练阶段；归一化）
  nwp_paired: [L+H, D_p]     配对 NWP（订正输入侧，归一化）
  is_day:     [H]            白天掩码（未来段）
  capacity:   标量           装机容量（kW，评估反归一用）

防泄漏：normalizer / FCM 中心 / （因果）VMD 统计均只用 train 段拟合并外部传入。
借鉴前身 extract_features / split_train_test，划分改为 7:1:2 时间顺序。
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # 允许无 torch 时被纯数据脚本 import
    torch = None
    Dataset = object

from .causal_vmd import causal_vmd_features, global_vmd_features
from .fcm_regime import FCMRegime
from .preprocess import PAIRING_INDEX
from .transforms import MinMaxNormalizer


def list_nwp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("nwp_")]


def list_lmd_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("lmd_")]


def paired_cols(df: pd.DataFrame):
    """返回实际存在的 (nwp_side, lmd_side) 配对列两组列表，顺序对齐。"""
    nwp_side, lmd_side = [], []
    for _, (n, l) in PAIRING_INDEX.items():
        if n in df.columns and l in df.columns:
            nwp_side.append(n)
            lmd_side.append(l)
    return nwp_side, lmd_side


def _engineer_features(power: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """lag/滑动统计/时间特征（借鉴前身 extract_features，全部因果）。

    返回 [T, n_eng]：hour_norm, lag_1..4, rolling_change_rate_1..4。
    """
    T = len(power)
    feats = []
    feats.append(hour / 23.0)  # hour 归一化
    for lag in range(1, 5):
        feats.append(np.concatenate([np.zeros(lag), power[:-lag]]) if lag < T
                     else np.zeros(T))
    # rolling change rate：窗口内 (max-min)/(w-1)，对历史功率，shift 保证因果
    w = 4
    s = pd.Series(power)
    shifted = s.shift(w - 1)
    for i in range(1, 5):
        cr = (shifted.rolling(w).max().shift((i - 1) * 2)
              - shifted.rolling(w).min().shift((i - 1) * 2)) / (w - 1)
        feats.append(cr.fillna(0.0).values)
    return np.stack(feats, axis=1)  # [T, 9]


def fit_normalizer_fcm(train_df: pd.DataFrame, fcm_cfg: dict):
    """在 train 段拟合 normalizer 与 FCM（外部调用，防泄漏）。

    Returns:
        (normalizer, fcm)
    """
    nwp_cols = list_nwp_cols(train_df)
    lmd_cols = list_lmd_cols(train_df)
    norm_cols = list(dict.fromkeys(nwp_cols + lmd_cols + ["power"]))
    normalizer = MinMaxNormalizer().fit(train_df, norm_cols)

    fcm = FCMRegime(n_clusters=fcm_cfg["K"], m=fcm_cfg["m"],
                    feature_cols=fcm_cfg["feature_cols"],
                    error=fcm_cfg.get("error", 0.005),
                    maxiter=fcm_cfg.get("maxiter", 1000))
    # FCM 用归一化后的 NWP 特征（尺度一致）
    tr_norm = normalizer.transform(train_df)
    fcm.fit(tr_norm[fcm_cfg["feature_cols"]].values)
    return normalizer, fcm


class PVODDataset(Dataset):
    """单站、单 split 的滑窗数据集。"""

    def __init__(self, sid: str, split: str, df_full: pd.DataFrame,
                 normalizer: MinMaxNormalizer, fcm: FCMRegime,
                 cfg: dict, capacity: float,
                 use_vmd: bool = True, leak_vmd: bool = False,
                 return_lmd: bool = True, vmd_cache_dir: str = "data/processed"):
        """
        Args:
            df_full:    该站完整 processed df（用于按比例切 split；VMD 在全序列上按
                        因果方式算后再切，保证窗口跨 split 边界时仍只用过去）。
            normalizer: 已在 train 段 fit。
            fcm:        已在 train 段 fit。
            cfg:        config dict（含 data/fcm/vmd）。
            capacity:   装机容量（kW）。
        """
        self.sid = sid
        self.split = split
        self.L = cfg["data"]["look_back"]
        self.H = cfg["data"]["horizon"]
        self.capacity = float(capacity)
        self.use_vmd = use_vmd
        self.leak_vmd = leak_vmd
        self.return_lmd = return_lmd

        df_full = df_full.copy()
        df_full["date_time"] = pd.to_datetime(df_full["date_time"])
        df_full = df_full.sort_values("date_time").reset_index(drop=True)
        n = len(df_full)
        r_tr, r_va, r_te = cfg["data"]["split"]
        i_tr = int(n * r_tr)
        i_va = int(n * (r_tr + r_va))
        bounds = {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, n)}
        lo, hi = bounds[split]

        # 归一化全序列（统计量来自 train），再取本 split 段 [lo, hi)
        df_norm = normalizer.transform(df_full)

        self.nwp_cols = list_nwp_cols(df_full)
        self.lmd_cols = list_lmd_cols(df_full)
        self.nwp_paired_cols, self.lmd_paired_cols = paired_cols(df_full)
        self.fcm_cols = cfg["fcm"]["feature_cols"]

        # ---- 全序列特征矩阵（因果，可安全切窗）----
        power_all = df_norm["power"].values.astype(float)
        hour_all = pd.to_datetime(df_full["date_time"]).dt.hour.values.astype(float)
        nwp_all = df_norm[self.nwp_cols].values.astype(float)            # [N, D_nwp]
        eng_all = _engineer_features(power_all, hour_all)                # [N, 9]
        # x_hist = 历史功率 + 历史 NWP + 工程特征
        self.hist_all = np.concatenate(
            [power_all[:, None], nwp_all, eng_all], axis=1)              # [N, D_hist]
        self.nwp_all = nwp_all                                           # [N, D_nwp]
        self.u_all = fcm.soft_membership(df_norm[self.fcm_cols].values)  # [N, K]
        self.power_all = power_all
        self.is_day_all = df_full["is_day"].values.astype(float) if "is_day" in df_full \
            else np.ones(n)
        if self.nwp_paired_cols:
            self.nwp_paired_all = df_norm[self.nwp_paired_cols].values.astype(float)
            self.lmd_paired_all = df_norm[self.lmd_paired_cols].values.astype(float)
        else:
            self.nwp_paired_all = np.zeros((n, 0))
            self.lmd_paired_all = np.zeros((n, 0))

        # ---- 因果 VMD（全序列上算，缓存）----
        if use_vmd:
            self.nu_all = self._load_or_compute_vmd(
                power_all, cfg, vmd_cache_dir, sid, leak_vmd)            # [N, M]
        else:
            self.nu_all = np.zeros((n, cfg["vmd"]["K_modes"]))

        self.lo, self.hi = lo, hi
        self.seg_len = hi - lo
        # 样本数：本 split 段内可取的滑窗数
        self.n_samples = max(0, self.seg_len - self.L - self.H + 1)

    # ---------- VMD ----------
    def _load_or_compute_vmd(self, power_all, cfg, cache_dir, sid, leak):
        os.makedirs(cache_dir, exist_ok=True)
        tag = "vmd_leak" if leak else "vmd"
        cache = os.path.join(cache_dir, f"{sid}_{tag}.npy")
        if os.path.exists(cache):
            arr = np.load(cache)
            if len(arr) == len(power_all):
                return arr
        vcfg = cfg["vmd"]
        if leak:
            arr = global_vmd_features(power_all, vcfg["K_modes"], vcfg["alpha"],
                                      vcfg.get("tau", 0.0), vcfg.get("DC", 0),
                                      vcfg.get("init", 1), vcfg.get("tol", 1e-7))
        else:
            arr = causal_vmd_features(power_all, vcfg["K_modes"], vcfg["alpha"],
                                      vcfg.get("window", self.L), vcfg.get("stride", 1),
                                      vcfg.get("tau", 0.0), vcfg.get("DC", 0),
                                      vcfg.get("init", 1), vcfg.get("tol", 1e-7))
        np.save(cache, arr)
        return arr

    # ---------- Dataset 接口 ----------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        # 在本 split 段内定位：历史 [a, a+L)，未来 [a+L, a+L+H)
        a = self.lo + idx
        h0, h1 = a, a + self.L              # history
        f0, f1 = a + self.L, a + self.L + self.H  # future
        full0, full1 = a, a + self.L + self.H     # L+H span

        out = {
            "x_nwp": self.nwp_all[full0:full1],          # [L+H, D_nwp]
            "x_hist": self.hist_all[h0:h1],              # [L, D_hist]
            "nu": self.nu_all[h0:h1],                    # [L, M]
            "u": self.u_all[full0:full1],                # [L+H, K]
            "y": self.power_all[f0:f1],                  # [H]
            "is_day": self.is_day_all[f0:f1],            # [H]
            "nwp_paired": self.nwp_paired_all[full0:full1],  # [L+H, D_p]
            "lmd_paired": self.lmd_paired_all[full0:full1],  # [L+H, D_p]
            "capacity": np.float32(self.capacity),
        }
        if torch is not None:
            out = {k: (torch.as_tensor(v, dtype=torch.float32)
                       if not np.isscalar(v) else torch.tensor(float(v)))
                   for k, v in out.items()}
        return out
