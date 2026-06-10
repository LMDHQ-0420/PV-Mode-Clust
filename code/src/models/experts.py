"""模块 C：结构异质专家（防同构塌缩）。

E-8 重塑 + 架构修正（2026-06-08）：
诊断——日前 PV 中**未来段 NWP（辐照）是每一步功率的主导预测因子**，RF 之所以强是因为它
对未来 NWP 窗口有完整逐步访问。原专家把"历史"当主序列、未来 NWP 仅经薄头注入，方向反了。

修正：所有专家以**未来段订正 NWP 序列 x_fut [B,H,d_fut] 为主序列**建模，历史 z 编码为
上下文 ctx 作 FiLM/拼接条件。异质性体现在"如何处理未来 NWP 序列"：
  - LocalConvExpert：小核因果卷积——局部天气平滑（多云突变）；
  - DilatedTCNExpert：大膨胀感受野——日内长程趋势（晴天）；
  - DirectMLPExpert：逐步 MLP（类 RF 直接映射）+ 历史统计条件。

统一接口 forward(z, x_fut) -> [B,H]。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _HistEncoder(nn.Module):
    """历史 z [B,L,d_in] → 上下文 ctx [B,hidden]（GRU 末隐状态 + 全局统计）。"""

    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(z)
        return out[:, -1, :]


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class LocalConvExpert(nn.Module):
    """小核卷积处理未来 NWP（局部天气），历史 ctx 作条件。"""

    def __init__(self, d_in: int, d_fut: int, hidden: int, horizon: int,
                 kernel: int = 3, **kw):
        super().__init__()
        self.hist = _HistEncoder(d_in, hidden)
        pad = kernel - 1
        self.conv = nn.Sequential(
            nn.Conv1d(d_fut + hidden, hidden, kernel, padding=pad), _Chomp1d(pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=pad), _Chomp1d(pad), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, z, x_fut):
        ctx = self.hist(z)                                  # [B,hidden]
        H = x_fut.shape[1]
        x = torch.cat([x_fut, ctx.unsqueeze(1).expand(-1, H, -1)], dim=-1)  # [B,H,d_fut+hidden]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)    # [B,H,hidden]
        return self.head(y).squeeze(-1)


class DilatedTCNExpert(nn.Module):
    """大膨胀感受野 TCN 处理未来 NWP（长程趋势），历史 ctx 作条件。"""

    def __init__(self, d_in: int, d_fut: int, hidden: int, horizon: int,
                 kernel: int = 3, levels: int = 4, **kw):
        super().__init__()
        self.hist = _HistEncoder(d_in, hidden)
        layers = []
        ch = d_fut + hidden
        for i in range(levels):
            dil = 2 ** i
            pad = (kernel - 1) * dil
            layers += [nn.Conv1d(ch, hidden, kernel, padding=pad, dilation=dil),
                       _Chomp1d(pad), nn.ReLU()]
            ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, z, x_fut):
        ctx = self.hist(z)
        H = x_fut.shape[1]
        x = torch.cat([x_fut, ctx.unsqueeze(1).expand(-1, H, -1)], dim=-1)
        y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(y).squeeze(-1)


class DirectMLPExpert(nn.Module):
    """逐步 MLP（类 RF 直接映射）：每步未来 NWP + 历史 ctx → 功率。"""

    def __init__(self, d_in: int, d_fut: int, hidden: int, horizon: int, **kw):
        super().__init__()
        self.hist = _HistEncoder(d_in, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_fut + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, x_fut):
        ctx = self.hist(z)
        H = x_fut.shape[1]
        x = torch.cat([x_fut, ctx.unsqueeze(1).expand(-1, H, -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


_EXPERT_REGISTRY = {
    "local_conv": LocalConvExpert,
    "dilated_tcn": DilatedTCNExpert,
    "direct_mlp": DirectMLPExpert,
    # 兼容旧名（映射到新实现，保持 config 不炸）
    "short_gru": LocalConvExpert,
    "long_tcn": DilatedTCNExpert,
    "freq": DirectMLPExpert,
}


def build_experts(expert_types, d_in: int, horizon: int, d_fut: int,
                  hidden: int = 64, short_window: int = 48) -> nn.ModuleList:
    """按类型列表构造异质专家（统一以未来 NWP 为主序列）。"""
    experts = []
    for t in expert_types:
        cls = _EXPERT_REGISTRY[t]
        experts.append(cls(d_in=d_in, d_fut=d_fut, hidden=hidden, horizon=horizon))
    return nn.ModuleList(experts)
