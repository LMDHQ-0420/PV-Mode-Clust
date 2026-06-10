"""损失函数 —— 单阶段联合：L = L_pred + λ·L_corr。

E-8 重塑（2026-06-08）：
  L_pred：预测功率对齐真实功率（仅白天）。
  L_corr：订正后辐照在**大误差区**对齐 LMD（仅 m_t=1）+ 订正量幅度惩罚 β(γΔ)²。
"""
from __future__ import annotations

import torch


def prediction_loss(y_hat: torch.Tensor, y: torch.Tensor,
                    is_day: torch.Tensor, huber_delta: float = 0.0) -> torch.Tensor:
    """L_pred（仅白天）。huber_delta>0 用 Huber（对 PV 异常值更鲁棒），否则 MSE。

    Args:
        y_hat/y: [B, H]；is_day: [B, H]；huber_delta: Huber 阈值（0=MSE）。
    """
    if huber_delta > 0:
        err = (y_hat - y).abs()
        per = torch.where(err <= huber_delta,
                          0.5 * err ** 2,
                          huber_delta * (err - 0.5 * huber_delta))
    else:
        per = (y_hat - y) ** 2
    return (per * is_day).sum() / (is_day.sum() + 1e-8)


def corrector_loss(x_corr_irrad: torch.Tensor, irrad_lmd: torch.Tensor,
                   big_err_mask: torch.Tensor, gamma_delta: torch.Tensor,
                   beta: float = 0.01) -> torch.Tensor:
    """L_corr = 大误差区 MSE(订正辐照, LMD辐照) + β·mean((γΔ)^2)。

    Args:
        x_corr_irrad: [B, T] 订正后辐照（归一化）。
        irrad_lmd:    [B, T] LMD 实测辐照（归一化）。
        big_err_mask: [B, T] 大误差区掩码（1/0）。
        gamma_delta:  [B, T] 残差门控×订正量 γ·Δ（幅度惩罚项）。
        beta:         幅度惩罚系数。

    Returns:
        标量损失。
    """
    sup = ((x_corr_irrad - irrad_lmd) ** 2 * big_err_mask).sum() / (big_err_mask.sum() + 1e-8)
    amp = (gamma_delta ** 2).mean()
    return sup + beta * amp
