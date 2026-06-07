"""模块 C2：轻量专家 GRU / TCN。

idea_report 3.7：K 个并行轻量专家，每个学一种区制下"特征→未来功率"。
GRU 默认（稳快、序列不长够用），TCN 备选作加分消融。

关键（2026-06-08 代码审查 E-5 修订）：专家显式消费**未来段订正后 NWP**（日前天气预报），
使订正在预测期真正生效（RQ1）、模型确为 NWP 气象驱动。
  forward(z, x_fut)：z [B, L, d_in] 历史特征 + x_fut [B, H, d_fut] 未来订正 NWP → pred [B, H]。
逻辑：编码历史得上下文 ctx → 广播到 H 步并与 x_fut 逐步拼接 → 共享 MLP 头逐步出功率。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FutureHead(nn.Module):
    """逐步预测头：拼 [上下文广播, 未来订正 NWP] → 每步功率。"""

    def __init__(self, hidden: int, d_fut: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden + d_fut, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, ctx: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        """ctx [B, hidden], x_fut [B, H, d_fut] → [B, H]。"""
        H = x_fut.shape[1]
        ctx_b = ctx.unsqueeze(1).expand(-1, H, -1)        # [B, H, hidden]
        h = torch.cat([ctx_b, x_fut], dim=-1)              # [B, H, hidden+d_fut]
        return self.mlp(h).squeeze(-1)                     # [B, H]


class GRUExpert(nn.Module):
    """单个 GRU 专家：编码历史窗口得上下文，逐步头结合未来订正 NWP 出 H 点。"""

    def __init__(self, d_in: int, hidden: int, horizon: int, d_fut: int,
                 num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=num_layers, batch_first=True)
        self.head = _FutureHead(hidden, d_fut)

    def forward(self, z: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        """z [B, L, d_in], x_fut [B, H, d_fut] → [B, H]。"""
        out, h = self.gru(z)
        ctx = out[:, -1, :]            # 末时刻隐状态 [B, hidden]
        return self.head(ctx, x_fut)


class _Chomp1d(nn.Module):
    """裁掉因果卷积右侧 padding，保证因果。"""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TCNExpert(nn.Module):
    """单个 TCN 专家：膨胀因果卷积堆叠得上下文，逐步头结合未来订正 NWP 出 H 点。"""

    def __init__(self, d_in: int, hidden: int, horizon: int, d_fut: int,
                 kernel: int = 3, levels: int = 3):
        super().__init__()
        layers = []
        ch_in = d_in
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel - 1) * dilation
            layers += [
                nn.Conv1d(ch_in, hidden, kernel, padding=pad, dilation=dilation),
                _Chomp1d(pad),
                nn.ReLU(),
            ]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = _FutureHead(hidden, d_fut)

    def forward(self, z: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        """z [B, L, d_in], x_fut [B, H, d_fut] → [B, H]。"""
        x = z.transpose(1, 2)          # [B, d_in, L]
        y = self.tcn(x)                # [B, hidden, L]
        ctx = y[:, :, -1]              # [B, hidden]
        return self.head(ctx, x_fut)


def build_expert(expert_cfg: dict, d_in: int, horizon: int, d_fut: int) -> nn.Module:
    """按 config 造一个专家。

    Args:
        d_in:  历史段编码输入维度。
        d_fut: 未来段订正 NWP 维度（= d_nwp）。
    """
    t = expert_cfg.get("type", "gru").lower()
    hidden = expert_cfg.get("hidden", 64)
    if t == "gru":
        return GRUExpert(d_in, hidden, horizon, d_fut,
                         num_layers=expert_cfg.get("num_layers", 1))
    if t == "tcn":
        return TCNExpert(d_in, hidden, horizon, d_fut,
                         kernel=expert_cfg.get("tcn_kernel", 3),
                         levels=expert_cfg.get("tcn_levels", 3))
    raise ValueError(f"未知专家类型：{t}")
