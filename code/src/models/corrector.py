"""模块 B：自适应 NWP 辐照订正（端到端辅助机制）。

E-8 重塑（2026-06-08）：原"区制条件订正全部可配对列、两阶段冻结"实证无效甚至有害。
按文献重做四点：(1) 仅订辐照（与功率强相关）；(2) 残差门控 gamma 让订正只在必要时生效；
(3) 仅大误差区监督 + 幅度惩罚（损失侧，见 losses.py）；(4) 端到端联合训练（trainer 侧）。

corrected_irrad = NWP_irrad + gamma * Delta，其余列透传。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveIrradCorrector(nn.Module):
    """自适应辐照订正器：以 [NWP辐照, u] 为条件，输出订正量与残差门控。"""

    def __init__(self, d_nwp: int, K: int, irrad_idx: int, hidden: int = 128):
        """
        Args:
            d_nwp:     NWP 总维度。
            K:         区制数（u 维度）。
            irrad_idx: 辐照列在 d_nwp 中的索引（只订这一列）。
            hidden:    隐藏层宽度。
        """
        super().__init__()
        self.d_nwp = d_nwp
        self.irrad_idx = irrad_idx
        self.net = nn.Sequential(
            nn.Linear(1 + K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_delta = nn.Linear(hidden, 1)   # 订正量 Δ
        self.head_gate = nn.Linear(hidden, 1)    # 残差门控 logit

    def forward(self, x_nwp: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_nwp: [B, T, d_nwp] NWP 预报（归一化）。
            u:     [B, T, K]     区制软隶属。

        Returns:
            x_corr: [B, T, d_nwp]，仅辐照列被订正，其余透传。
            gamma:  [B, T, 1]     残差门控 ∈[0,1]（可解释）。
            delta:  [B, T, 1]     订正量。
        """
        last = x_nwp.dim() - 1
        irrad = x_nwp[..., self.irrad_idx:self.irrad_idx + 1]   # [B, T, 1]
        h = self.net(torch.cat([irrad, u], dim=-1))
        delta = self.head_delta(h)                               # [B, T, 1]
        gamma = torch.sigmoid(self.head_gate(h))                 # [B, T, 1]
        corr_irrad = irrad + gamma * delta                       # 残差形式

        x_corr = x_nwp.index_copy(
            last, torch.tensor([self.irrad_idx], device=x_nwp.device), corr_irrad)
        return x_corr, gamma, delta
