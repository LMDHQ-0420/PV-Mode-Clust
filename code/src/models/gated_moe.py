"""可学习软门控异质多专家预测模型（模块 A+B+C+D）。

结构：
  A. FCM 软区制隶属（在 build.py 中拟合，batch 里以 u 传入）
  B. 结构异质专家：local_conv / dilated_tcn / direct_mlp
  C. 可学习软门控：g = softmax(α·log u + MLP(x_nwp))
  D. 互补 Stacking（在 stacking.py 中完成）

消融开关：
  gate_mode=hard      → 变体 B（argmax one-hot）
  learnable_gate=False → 变体 C（固定 FCM 隶属）
  expert_types 同构    → 变体 D
  single_expert=True  → 变体 E（K=1）
  irrad_anchor=True   → 变体 I（物理头，负结果对照）
  leak_vmd=True       → 变体 H（VMD 泄漏演示）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .experts import build_experts


class GatedMoEForecaster(nn.Module):
    def __init__(self, expert_types, d_nwp: int, d_hist: int, K: int,
                 horizon: int, gate_mode: str = "soft", learnable_gate: bool = True,
                 single_expert: bool = False, expert_hidden: int = 64,
                 gate_hidden: int = 64, short_window: int = 48,
                 irrad_anchor: bool = False, irrad_idx: int = 0):
        """
        Args:
            expert_types:   异质专家类型列表。
            K:              区制数 / 门控先验维度。
            learnable_gate: True=FCM先验+可学习残差；False=固定 u（消融 C）。
            irrad_anchor:   True=辐照锚定物理头（消融 I，负结果对照）。
        """
        super().__init__()
        self.K = K
        self.horizon = horizon
        self.gate_mode = gate_mode
        self.learnable_gate = learnable_gate
        self.single_expert = single_expert
        self.irrad_anchor = irrad_anchor
        self.irrad_idx = irrad_idx
        self.d_fut = d_nwp

        d_in = d_nwp + d_hist
        self.d_in = d_in
        types = [expert_types[0]] if single_expert else list(expert_types)
        self.n_experts = len(types)
        self.experts = build_experts(types, d_in, horizon, self.d_fut,
                                     hidden=expert_hidden, short_window=short_window)

        # 区制(K)→专家(n_experts) 先验映射
        self.prior_proj = (None if K == self.n_experts
                           else nn.Linear(K, self.n_experts, bias=False))

        # 可学习门控残差
        if learnable_gate and not single_expert:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_nwp, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, self.n_experts),
            )
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.gate_mlp = None

        # 辐照锚定头（消融 I）
        self.gain = nn.Parameter(torch.tensor(1.0))

    def _gate(self, u: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """计算门控权重 g: [B, n_experts]。"""
        H = self.horizon
        u_fut = u[:, -H:, :].mean(dim=1)
        u_fut = u_fut / (u_fut.sum(-1, keepdim=True) + 1e-8)
        prior_p = u_fut if self.prior_proj is None else torch.softmax(self.prior_proj(u_fut), -1)
        if self.gate_mlp is not None:
            prior = self.alpha * torch.log(prior_p + 1e-6)
            resid = self.gate_mlp(x_nwp.mean(dim=1))
            g = torch.softmax(prior + resid, dim=-1)
        else:
            g = prior_p
        if self.gate_mode == "hard":
            idx = g.argmax(dim=-1)
            g = torch.zeros_like(g).scatter_(-1, idx[:, None], 1.0)
        return g

    def forward(self, batch: dict):
        x_nwp = batch["x_nwp"]
        x_hist = batch["x_hist"]
        u = batch["u"]
        B = x_nwp.shape[0]
        L = x_hist.shape[1]

        # 专家输入：历史段 [NWP, x_hist] + 未来段 NWP
        z = torch.cat([x_nwp[:, :L, :], x_hist], dim=-1)   # [B, L, d_in]
        x_fut = x_nwp[:, L:, :]                              # [B, H, d_nwp]

        preds = torch.stack([f(z, x_fut) for f in self.experts], dim=-1)  # [B, H, n_experts]

        if self.single_expert:
            moe_out = preds[:, :, 0]
            gate = torch.ones(B, 1, device=x_nwp.device)
        else:
            gate = self._gate(u, x_nwp)
            moe_out = torch.einsum("bhk,bk->bh", preds, gate)

        # 辐照锚定（消融 I 负结果对照）
        if self.irrad_anchor:
            irrad_fut = x_nwp[:, L:, self.irrad_idx]
            y_hat = nn.functional.softplus(self.gain) * irrad_fut + moe_out
        else:
            y_hat = moe_out

        aux = {"u": u, "gate": gate, "preds": preds}
        return y_hat, aux
