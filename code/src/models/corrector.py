"""模块 B：NWP 误差订正器 g(NWP, u)（第一阶段）。

核心（RQ1，idea_report 3.6）：仅吃 [NWP, u]（推理可得），不含未来 LMD。
区制条件 u 使订正分天气自适应 [3,4]。订正后 NWP 供第二阶段预测使用。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NWPCorrector(nn.Module):
    """以 [NWP, u] 为条件，逐时刻订正可配对气象列。

    仅订正可配对列（辐照/温度/气压，r 高），不可配对列（风速/风向等）原样透传。
    输出订正量 ê 后做 x_corr_paired = x_nwp_paired - ê。
    """

    def __init__(self, d_nwp: int, K: int, d_paired: int,
                 paired_idx: list[int], hidden: int = 128):
        """
        Args:
            d_nwp:      NWP 总维度。
            K:          区制数（u 维度）。
            d_paired:   可配对（可订正）列数。
            paired_idx: 可配对列在 d_nwp 中的索引位置。
            hidden:     隐藏层宽度。
        """
        super().__init__()
        self.d_nwp = d_nwp
        self.d_paired = d_paired
        self.register_buffer("paired_idx",
                             torch.tensor(paired_idx, dtype=torch.long))
        self.net = nn.Sequential(
            nn.Linear(d_nwp + K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_paired),
        )

    def forward(self, x_nwp: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_nwp: [B, T, d_nwp] NWP 预报（归一化）。
            u:     [B, T, K]     区制软隶属。

        Returns:
            x_corr:        [B, T, d_nwp] 订正后 NWP（不可配对列透传）。
            corr_paired:   [B, T, d_paired] 订正后的可配对列（订正监督用）。
            e_hat:         [B, T, d_paired] 预测的订正量。
        """
        last = x_nwp.dim() - 1
        h = torch.cat([x_nwp, u], dim=-1)        # [B, T, d_nwp+K]
        e_hat = self.net(h)                        # [B, T, d_paired]
        x_corr = x_nwp.clone()
        paired = x_nwp.index_select(last, self.paired_idx)   # [B, T, d_paired]
        corr_paired = paired - e_hat
        x_corr = x_corr.index_copy(last, self.paired_idx, corr_paired)
        return x_corr, corr_paired, e_hat
