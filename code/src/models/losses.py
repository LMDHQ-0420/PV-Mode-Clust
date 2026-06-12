"""损失函数 —— L = L_pred（仅白天 MSE / Huber）。"""
from __future__ import annotations

import torch


def prediction_loss(y_hat: torch.Tensor, y: torch.Tensor,
                    is_day: torch.Tensor, huber_delta: float = 0.0) -> torch.Tensor:
    """L_pred（仅白天）。huber_delta>0 用 Huber，否则 MSE。

    Args:
        y_hat/y: [B, H]; is_day: [B, H]; huber_delta: Huber 阈值（0=MSE）。
    """
    if huber_delta > 0:
        err = (y_hat - y).abs()
        per = torch.where(err <= huber_delta,
                          0.5 * err ** 2,
                          huber_delta * (err - 0.5 * huber_delta))
    else:
        per = (y_hat - y) ** 2
    return (per * is_day).sum() / (is_day.sum() + 1e-8)
