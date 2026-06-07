"""Max-Min 归一化 —— 统计量仅从 train 段计算，对 val/test 套用（防泄漏）。

借鉴前身 code/utils.py:normalize_column / inverse_normalize_column，
封装为可 fit/transform/inverse 的类，便于按列保存统计量。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class MinMaxNormalizer:
    """按列 Max-Min 归一化，统计量从训练段拟合。"""

    def __init__(self):
        self.min_: Dict[str, float] = {}
        self.max_: Dict[str, float] = {}
        self.cols: List[str] = []

    def fit(self, train_df: pd.DataFrame, cols: List[str]) -> "MinMaxNormalizer":
        """在训练段记录每列 min/max。"""
        self.cols = list(cols)
        for c in cols:
            self.min_[c] = float(train_df[c].min())
            self.max_[c] = float(train_df[c].max())
        return self

    def _scale(self, c: str, x):
        rng = self.max_[c] - self.min_[c]
        if rng == 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        return (np.asarray(x, dtype=float) - self.min_[c]) / rng

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 df 中已 fit 的列做归一化，返回新 df。"""
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                out[c] = self._scale(c, out[c].values)
        return out

    def inverse(self, col: str, values) -> np.ndarray:
        """反归一化（评估前还原 power 等）。"""
        rng = self.max_[col] - self.min_[col]
        return np.asarray(values, dtype=float) * rng + self.min_[col]
