"""模块 C1：因果滑窗 VMD。

核心（RQ3，idea_report 3.7）：每个时刻只用 τ≤t 的数据分解，从根上杜绝泄漏 [5,9]。
逐点滑窗 VMD 成本高 → 缓存到 data/processed/{sid}_vmd.npy，避免每 epoch 重算。
另提供 global_vmd_features（全序列 VMD，**故意泄漏**）供消融 E 对照。
"""
from __future__ import annotations

import numpy as np

try:
    from vmdpy import VMD
except ImportError:  # 允许无 vmdpy 时 import 文件
    VMD = None


def _vmd_modes(window: np.ndarray, K_modes: int, alpha: float,
               tau: float, DC: int, init: int, tol: float) -> np.ndarray:
    """对一维窗口做 VMD，返回 [K_modes, len] 模态（按中心频率排序）。"""
    if VMD is None:
        raise ImportError("需要 vmdpy：pip install vmdpy")
    x = np.asarray(window, dtype=float)
    # vmdpy 要求偶数长度，奇数则丢首点
    if len(x) % 2 == 1:
        x = x[1:]
    u, u_hat, omega = VMD(x, alpha, tau, K_modes, DC, init, tol)
    return np.asarray(u, dtype=float)  # [K_modes, len(x)]


def causal_vmd_features(series: np.ndarray, K_modes: int = 5, alpha: float = 2000.0,
                        window: int = 192, stride: int = 1,
                        tau: float = 0.0, DC: int = 0, init: int = 1,
                        tol: float = 1e-7) -> np.ndarray:
    """对一维历史序列做因果滑窗 VMD，产出每时刻的多尺度模态特征。

    逻辑（implementation §3.4）：
      对每个 t，取窗 series[t-window+1 : t+1]（仅过去），VMD 得 K_modes 个模态，
      取各模态在窗末端（t 时刻）的值，拼成 [K_modes] 向量；拼所有 t → [T, K_modes]。
      不足一窗的前缀用首个可计算窗的结果回填（仍只含过去信息）。

    Args:
        series:  [T] 一维历史序列（如历史功率）。
        K_modes: 模态数。
        window:  滑窗长（默认 = look_back）。
        stride:  步长（默认 1，逐点）。

    Returns:
        [T, K_modes] 多尺度特征。
    """
    x = np.asarray(series, dtype=float).ravel()
    T = len(x)
    feats = np.zeros((T, K_modes), dtype=float)
    first_valid = None

    for t in range(window - 1, T, stride):
        win = x[t - window + 1: t + 1]
        modes = _vmd_modes(win, K_modes, alpha, tau, DC, init, tol)  # [K_modes, L']
        end_val = modes[:, -1]  # 窗末端 = t 时刻
        feats[t] = end_val
        if first_valid is None:
            first_valid = end_val
        # stride>1 时回填中间点为同窗末端值（仍因果）
        if stride > 1:
            lo = max(window - 1, t - stride + 1)
            feats[lo:t] = end_val

    # 前缀（不足一窗）用首个有效窗回填，保证全长可用且不引入未来信息
    if first_valid is not None and window - 1 > 0:
        feats[: window - 1] = first_valid
    return feats


def global_vmd_features(series: np.ndarray, K_modes: int = 5, alpha: float = 2000.0,
                        tau: float = 0.0, DC: int = 0, init: int = 1,
                        tol: float = 1e-7) -> np.ndarray:
    """对**整条序列**（含测试段）一次性 VMD，**故意引入泄漏**——仅供消融 E 对照。

    演示"虚高精度"陷阱（idea_report Part 3 §2）：每个 t 的模态都用到了全序列
    （含 t 之后）信息。

    Returns:
        [T, K_modes]，每行取对应时刻各模态值。
    """
    x = np.asarray(series, dtype=float).ravel()
    T = len(x)
    odd = T % 2 == 1
    modes = _vmd_modes(x, K_modes, alpha, tau, DC, init, tol)  # [K_modes, T or T-1]
    feats = modes.T  # [T', K_modes]
    if odd:  # _vmd_modes 丢了首点，前面补一行
        feats = np.vstack([feats[0:1], feats])
    return feats[:T]
