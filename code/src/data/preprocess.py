"""原始 PVOD → 清洗对齐的 processed csv。

借鉴前身 code/ 的箱线图异常值处理；不删行（前身验证 processed 与原始行数一致）。
字段：date_time + nwp_*(7) + lmd_*(6) + power（具体列名以实际 csv 为准，按前缀识别）。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# NWP–LMD 可配对字段（订正用）：r 高的辐照/温度/气压才相减做误差，
# 风速(r=0.55)/风向(r=0.33) 不配对，仅作特征（idea_report 3.3 / 数据验证结论）。
PAIRING_INDEX = {
    "irrad": ("nwp_globalirrad", "lmd_totalirrad"),
    "temp": ("nwp_temperature", "lmd_temperature"),
    "pressure": ("nwp_pressure", "lmd_pressure"),
}


def build_pairing_index() -> dict:
    """返回 NWP–LMD 可配对字段映射（订正用）。"""
    return dict(PAIRING_INDEX)


def _iqr_clip_to_nan(df: pd.DataFrame, cols) -> pd.DataFrame:
    """箱线图法（Q1-1.5IQR, Q3+1.5IQR）将异常值置 NaN（不删行）。"""
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = s.mask((s < lo) | (s > hi), np.nan)
    return out


def preprocess_station(sid: str, raw_dir: str, out_dir: str) -> str:
    """清洗单个站点并落盘。

    Args:
        sid:     站点 id（如 'station00'）。
        raw_dir: 原始目录（含 {sid}.csv）。
        out_dir: 输出目录（data/processed）。

    Returns:
        落盘路径。

    逻辑（implementation §3.1）：
      1. 读 {sid}.csv，解析 date_time 为 datetime。
      2. 数值列箱线图法将异常值置 NaN。
      3. 缺失值前向填充 + 后向兜底（避免开头 NaN），再剩余填 0。
      4. 新增 is_day 列（lmd_totalirrad>0；缺则退化用 nwp_globalirrad>0）。
      5. 落盘 data/processed/{sid}.csv。
    """
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(raw_dir, f"{sid}.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"未找到 {sid} 原始文件：{src}")

    df = pd.read_csv(src)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    # 数值列（除时间外）做箱线图清洗
    num_cols = [c for c in df.columns if c != "date_time" and
                pd.api.types.is_numeric_dtype(df[c])]
    df = _iqr_clip_to_nan(df, num_cols)

    # 缺失填充：前向 → 后向 → 0 兜底
    df[num_cols] = df[num_cols].ffill().bfill().fillna(0.0)

    # 白天标记：优先 lmd 实测辐照，缺则用 nwp 预报辐照
    if "lmd_totalirrad" in df.columns:
        df["is_day"] = (df["lmd_totalirrad"] > 0).astype(int)
    elif "nwp_globalirrad" in df.columns:
        df["is_day"] = (df["nwp_globalirrad"] > 0).astype(int)
    else:
        df["is_day"] = 1

    out_path = os.path.join(out_dir, f"{sid}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--metadata", default="dataset/metadata.csv")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="指定站点；缺省则从 metadata 读取全部")
    args = ap.parse_args()

    if args.stations:
        stations = args.stations
    else:
        meta = pd.read_csv(args.metadata)
        col = "Station_ID" if "Station_ID" in meta.columns else meta.columns[0]
        stations = [str(s) for s in meta[col].tolist()]

    for sid in stations:
        path = preprocess_station(sid, args.raw_dir, args.out_dir)
        df = pd.read_csv(path)
        print(f"[preprocess] {sid}: {len(df)} 行 → {path}")


if __name__ == "__main__":
    main()
