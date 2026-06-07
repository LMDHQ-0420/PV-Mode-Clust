"""损失函数 —— 两阶段各一，均仅在白天 mask 上计入（光伏惯例）。

idea_report 3.6/3.7：
  L_corr：订正后可配对 NWP 对齐 LMD 实测（阶段一监督）。
  L_pred：预测功率对齐真实功率（阶段二监督）。
"""
from __future__ import annotations

import torch


def corrector_loss(corr_paired: torch.Tensor, lmd_paired: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """L_corr = mean(||x_corr - x_lmd||^2)（仅白天）。

    Args:
        corr_paired: [B, T, d_p] 订正后可配对列（归一化）。
        lmd_paired:  [B, T, d_p] 对应 LMD 实测（归一化）。
        mask:        [B, T] 白天掩码（1/0）。

    Returns:
        标量损失。
    """
    diff2 = (corr_paired - lmd_paired) ** 2          # [B, T, d_p]
    m = mask.unsqueeze(-1)                           # [B, T, 1]
    num = (diff2 * m).sum()
    den = m.sum() * corr_paired.shape[-1] + 1e-8
    return num / den


def prediction_loss(y_hat: torch.Tensor, y: torch.Tensor,
                    is_day: torch.Tensor) -> torch.Tensor:
    """L_pred = mean((p_hat - p)^2)（仅白天）。

    Args:
        y_hat:  [B, H] 预测功率（归一化）。
        y:      [B, H] 真实功率（归一化）。
        is_day: [B, H] 白天掩码。

    Returns:
        标量损失。
    """
    diff2 = (y_hat - y) ** 2
    num = (diff2 * is_day).sum()
    den = is_day.sum() + 1e-8
    return num / den
