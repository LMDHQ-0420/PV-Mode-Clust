"""装配辅助 —— 从 config + 站点 id 构建 datasets / normalizer / fcm / 模型。

集中处理"train 段拟合统计量、防泄漏"的流程，供 joint_trainer 与 evaluate 复用。
E-8 重塑：自适应订正器 + 可学习软门控异质 MoE；VMD 退出主流程（仅 leak_vmd 消融）。
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

from ..data.pvod_dataset import (PVODDataset, fit_normalizer_fcm, list_nwp_cols)
from ..models.corrector import AdaptiveIrradCorrector
from ..models.gated_moe import GatedMoEForecaster


def load_capacity(metadata_path: str, sid: str) -> float:
    """从 metadata.csv 读站点装机容量（kW），评估时 /1e3 与 power 同量纲。"""
    meta = pd.read_csv(metadata_path)
    id_col = "Station_ID" if "Station_ID" in meta.columns else meta.columns[0]
    cap_col = "Capacity" if "Capacity" in meta.columns else None
    row = meta[meta[id_col].astype(str) == str(sid)]
    if cap_col is None or row.empty:
        return 1.0
    return float(row.iloc[0][cap_col]) / 1e3


def build_station_data(cfg: Dict, sid: str) -> Tuple[Dict, Dict]:
    """构建单站 train/val/test 三个 dataset，返回 (datasets, meta)。"""
    proc_dir = cfg["data"]["processed_dir"]
    df_full = pd.read_csv(os.path.join(proc_dir, f"{sid}.csv"))

    n = len(df_full)
    r_tr = cfg["data"]["split"][0]
    train_df = df_full.iloc[: int(n * r_tr)].copy()
    normalizer, fcm = fit_normalizer_fcm(train_df, cfg["fcm"])
    cap = load_capacity(os.path.join(cfg["data"]["raw_dir"], "metadata.csv"), sid)

    common = dict(normalizer=normalizer, fcm=fcm, cfg=cfg, capacity=cap,
                  leak_vmd=cfg["model"].get("leak_vmd", False),
                  vmd_cache_dir=proc_dir)
    datasets = {sp: PVODDataset(sid, sp, df_full, return_lmd=True, **common)
                for sp in ("train", "val", "test")}

    nwp_cols = list_nwp_cols(df_full)
    ds0 = datasets["train"]
    dims = dict(
        d_nwp=len(nwp_cols),
        d_hist=ds0.hist_all.shape[1],
        K=cfg["fcm"]["K"],
        H=cfg["data"]["horizon"],
        irrad_idx=ds0.irrad_idx,
    )
    meta = dict(normalizer=normalizer, fcm=fcm, capacity=cap, dims=dims)
    return datasets, meta


def build_model(cfg: Dict, dims: Dict):
    """按 dims + config 造 AdaptiveIrradCorrector + GatedMoEForecaster。"""
    mc = cfg["model"]
    corrector = AdaptiveIrradCorrector(
        d_nwp=dims["d_nwp"], K=dims["K"], irrad_idx=dims["irrad_idx"],
        hidden=mc.get("corrector_hidden", 128),
    )
    # 同构消融 D：expert_types 设为同一类型重复 K 次（在 config 里覆盖）
    expert_types = mc.get("expert_types", ["local_conv", "dilated_tcn", "direct_mlp"])
    model = GatedMoEForecaster(
        corrector=corrector, expert_types=expert_types,
        d_nwp=dims["d_nwp"], d_hist=dims["d_hist"], K=dims["K"], horizon=dims["H"],
        gate_mode=mc.get("gate_mode", "soft"),
        learnable_gate=mc.get("learnable_gate", True),
        single_expert=mc.get("single_expert", False),
        use_corrector=mc.get("use_corrector", True),
        expert_hidden=mc.get("expert_hidden", 64),
        gate_hidden=mc.get("gate_hidden", 64),
        short_window=mc.get("short_window", 48),
        irrad_anchor=mc.get("irrad_anchor", False),
        irrad_idx=dims["irrad_idx"],
    )
    return corrector, model
