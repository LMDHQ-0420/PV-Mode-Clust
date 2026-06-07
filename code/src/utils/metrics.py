"""评估指标 —— 仅在白天（is_day）样本上计算。

口径对齐 idea_report Part 3 §0.7：
  - 国标系：RMSE / MAE / ACC / Q_R，误差均以装机容量 C 归一；
  - 文献系：Pearson r / R²。
所有函数接受 numpy array（已反归一化到 kW）+ 白天掩码，返回 float。
借鉴前身 code/utils.py:evaluation 的口径，但拆成独立无状态函数。
"""
from __future__ import annotations

import numpy as np


def _mask(pred: np.ndarray, true: np.ndarray, is_day: np.ndarray | None):
    """按白天掩码筛选，返回 (pred_day, true_day)。"""
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    if is_day is None:
        return pred, true
    m = np.asarray(is_day).ravel().astype(bool)
    return pred[m], true[m]


def rmse(pred, true, capacity: float, is_day=None) -> float:
    """归一化 RMSE：sqrt(mean(((t-p)/C)^2))，越低越好。"""
    p, t = _mask(pred, true, is_day)
    if len(p) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(((t - p) / capacity) ** 2)))


def mae(pred, true, capacity: float, is_day=None) -> float:
    """归一化 MAE：mean(|t-p|/C)，越低越好。"""
    p, t = _mask(pred, true, is_day)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p) / capacity))


def acc(pred, true, capacity: float, is_day=None) -> float:
    """国标准确率 C_R = (1 - RMSE) * 100，越高越好。"""
    r = rmse(pred, true, capacity, is_day)
    return float((1 - r) * 100)


def qr(pred, true, capacity: float, is_day=None) -> float:
    """合格率 Q_R：|t-p|/C < 0.25 的占比 ×100，越高越好。"""
    p, t = _mask(pred, true, is_day)
    if len(p) == 0:
        return float("nan")
    ok = (np.abs(t - p) / capacity < 0.25).astype(float)
    return float(np.mean(ok) * 100)


def pearson(pred, true, is_day=None) -> float:
    """Pearson 相关系数 r，越高越好。"""
    p, t = _mask(pred, true, is_day)
    if len(p) < 2:
        return float("nan")
    pm, tm = p.mean(), t.mean()
    num = np.sum((p - pm) * (t - tm))
    den = np.sqrt(np.sum((p - pm) ** 2) * np.sum((t - tm) ** 2))
    return float(num / den) if den != 0 else 0.0


def r2(pred, true, is_day=None) -> float:
    """决定系数 R²，越高越好。"""
    p, t = _mask(pred, true, is_day)
    if len(p) < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def compute_all(pred, true, capacity: float, is_day=None) -> dict:
    """一次性算齐 6 个指标，返回 dict。"""
    return {
        "rmse": rmse(pred, true, capacity, is_day),
        "mae": mae(pred, true, capacity, is_day),
        "acc": acc(pred, true, capacity, is_day),
        "qr": qr(pred, true, capacity, is_day),
        "r": pearson(pred, true, is_day),
        "r2": r2(pred, true, is_day),
    }


def day_ahead_rolling(preds, trues, is_day, capacity: float, horizon: int = 96) -> dict:
    """日前协议（Part 3 §0.7）：测试集按天滚动评估。

    将逐样本展平的预测拼成 [num_samples, horizon]，先汇总所有白天点算整体指标
    （等价于"按完整天算指标再平均"的同口径全局版本）。

    Args:
        preds:    [N, H] 预测功率（kW，已反归一化）。
        trues:    [N, H] 真实功率（kW）。
        is_day:   [N, H] 白天掩码。
        capacity: 装机容量（kW）。
        horizon:  每天预测点数（96）。

    Returns:
        指标 dict（与 compute_all 同字段）。
    """
    preds = np.asarray(preds, dtype=float).reshape(-1)
    trues = np.asarray(trues, dtype=float).reshape(-1)
    mask = np.asarray(is_day).reshape(-1)
    return compute_all(preds, trues, capacity, mask)
