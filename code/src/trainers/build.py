"""装配辅助 —— 从 config + 站点 id 构建 datasets / normalizer / fcm / 模型。

集中处理"train 段拟合统计量、防泄漏"的流程，供两阶段 trainer 与 evaluate 复用。
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

from ..data.pvod_dataset import (PVODDataset, fit_normalizer_fcm,
                                 list_nwp_cols, paired_cols)
from ..models.corrector import NWPCorrector
from ..models.gated_moe import GatedMoEForecaster


def load_capacity(metadata_path: str, sid: str) -> float:
    """从 metadata.csv 读站点装机容量（kW）。"""
    meta = pd.read_csv(metadata_path)
    id_col = "Station_ID" if "Station_ID" in meta.columns else meta.columns[0]
    cap_col = "Capacity" if "Capacity" in meta.columns else None
    row = meta[meta[id_col].astype(str) == str(sid)]
    if cap_col is None or row.empty:
        return 1.0
    cap = float(row.iloc[0][cap_col])
    # 前身约定：评估时容量除以 1e3（kW），与 power 同量纲
    return cap / 1e3


def build_station_data(cfg: Dict, sid: str) -> Tuple[Dict, Dict]:
    """构建单站 train/val/test 三个 dataset，返回 (datasets, meta)。

    Returns:
        datasets: {'train':..., 'val':..., 'test':...}
        meta:     {'normalizer','fcm','capacity','dims'}（dims 含 d_nwp/d_hist/M/K/H/d_paired/paired_idx）
    """
    proc_dir = cfg["data"]["processed_dir"]
    df_full = pd.read_csv(os.path.join(proc_dir, f"{sid}.csv"))

    # train 段切片用于 fit
    n = len(df_full)
    r_tr = cfg["data"]["split"][0]
    train_df = df_full.iloc[: int(n * r_tr)].copy()
    normalizer, fcm = fit_normalizer_fcm(train_df, cfg["fcm"])

    cap = load_capacity(os.path.join(cfg["data"]["raw_dir"], "metadata.csv"), sid)

    common = dict(normalizer=normalizer, fcm=fcm, cfg=cfg, capacity=cap,
                  use_vmd=cfg["model"]["use_vmd"], leak_vmd=cfg["model"]["leak_vmd"],
                  vmd_cache_dir=proc_dir)
    datasets = {
        sp: PVODDataset(sid, sp, df_full, return_lmd=True, **common)
        for sp in ("train", "val", "test")
    }

    nwp_cols = list_nwp_cols(df_full)
    nwp_paired, lmd_paired = paired_cols(df_full)
    paired_idx = [nwp_cols.index(c) for c in nwp_paired]
    ds0 = datasets["train"]
    dims = dict(
        d_nwp=len(nwp_cols),
        d_hist=ds0.hist_all.shape[1],
        M=cfg["vmd"]["K_modes"],
        K=cfg["fcm"]["K"],
        H=cfg["data"]["horizon"],
        d_paired=len(nwp_paired),
        paired_idx=paired_idx,
    )
    meta = dict(normalizer=normalizer, fcm=fcm, capacity=cap, dims=dims)
    return datasets, meta


def build_models(cfg: Dict, dims: Dict):
    """按 dims 造 corrector 与 gated_moe。"""
    corrector = NWPCorrector(
        d_nwp=dims["d_nwp"], K=dims["K"], d_paired=dims["d_paired"],
        paired_idx=dims["paired_idx"], hidden=cfg["model"]["corrector_hidden"],
    )
    model = GatedMoEForecaster(
        corrector=corrector, K=dims["K"], expert_cfg=cfg["model"]["expert"],
        d_nwp=dims["d_nwp"], d_hist=dims["d_hist"], M=dims["M"], horizon=dims["H"],
        use_vmd=cfg["model"]["use_vmd"], use_corrector=cfg["model"]["use_corrector"],
        gate_mode=cfg["model"]["gate_mode"], single_expert=cfg["model"]["single_expert"],
    )
    return corrector, model
