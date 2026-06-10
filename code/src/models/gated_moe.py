"""模块 D：可学习软门控异质多专家完整模型（组装 A/B/C/D）。

E-8 重塑（2026-06-08）：
  - 门控：FCM 软隶属作可解释先验 + 可学习残差 MLP，g=softmax(α·log u + MLP(x_nwp))；
  - 专家：结构异质（short_gru/long_tcn/freq），防同构塌缩；
  - 订正：自适应辐照订正（残差门控），端到端联合；
  - VMD：退出主流程。

消融开关：gate_mode(soft/hard,B) / learnable_gate(C) / expert_types同构(D) /
single_expert(E) / use_corrector(F) / blind_correct(F2,在订正器外围逻辑)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .experts import build_experts


class GatedMoEForecaster(nn.Module):
    def __init__(self, corrector, expert_types, d_nwp: int, d_hist: int, K: int,
                 horizon: int, gate_mode: str = "soft", learnable_gate: bool = True,
                 single_expert: bool = False, use_corrector: bool = True,
                 expert_hidden: int = 64, gate_hidden: int = 64,
                 short_window: int = 48, irrad_anchor: bool = True,
                 irrad_idx: int = 0):
        """
        Args:
            corrector:      AdaptiveIrradCorrector 实例。
            expert_types:   异质专家类型列表（single_expert 时取首个、K=1）。
            K:              区制数 / 门控维度。
            learnable_gate: True=FCM先验+可学习残差；False=固定 u（消融C）。
        """
        super().__init__()
        self.corrector = corrector
        self.K = K
        self.horizon = horizon
        self.gate_mode = gate_mode
        self.learnable_gate = learnable_gate
        self.single_expert = single_expert
        self.use_corrector = use_corrector
        self.d_fut = d_nwp

        d_in = d_nwp + d_hist          # 专家历史输入：订正后历史 NWP + x_hist
        self.d_in = d_in
        types = [expert_types[0]] if single_expert else list(expert_types)
        self.n_experts = len(types)
        self.experts = build_experts(types, d_in, horizon, self.d_fut,
                                     hidden=expert_hidden, short_window=short_window)

        # 区制(K)→专家(n_experts) 先验映射：K==n_experts 时为恒等，否则用学习线性
        self.prior_proj = (None if K == self.n_experts
                           else nn.Linear(K, self.n_experts, bias=False))

        # 可学习门控：从 NWP 池化特征算残差 logits [n_experts]
        if learnable_gate and not single_expert:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_nwp, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, self.n_experts),
            )
            self.alpha = nn.Parameter(torch.tensor(1.0))   # 先验强度
        else:
            self.gate_mlp = None

        # 辐照锚定头（物理先验：PV 功率 ≈ 效率×辐照）：y = softplus(gain)·irrad_fut + MoE残差
        # 让 MoE 只学"对物理基线的修正"，给与 RF 同等的逐步辐照直达，且带正确归纳偏置。
        self.irrad_anchor = irrad_anchor
        self.irrad_idx = irrad_idx
        self.gain = nn.Parameter(torch.tensor(1.0))

    def _gate(self, u: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """算门控权重 g [B, n_experts]。"""
        H = self.horizon
        u_fut = u[:, -H:, :].mean(dim=1)                 # [B, K] 预测段平均隶属
        u_fut = u_fut / (u_fut.sum(-1, keepdim=True) + 1e-8)
        prior_p = u_fut if self.prior_proj is None else torch.softmax(self.prior_proj(u_fut), -1)
        if self.gate_mlp is not None:
            prior = self.alpha * torch.log(prior_p + 1e-6)  # 可解释先验（映射到专家空间）
            resid = self.gate_mlp(x_nwp.mean(dim=1))      # 可学习残差 [B,n_experts]
            g = torch.softmax(prior + resid, dim=-1)
        else:
            g = prior_p                                   # 固定外部门控（消融C）
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

        # 模块 B：自适应订正
        if self.use_corrector and self.corrector is not None:
            x_corr, gamma, delta = self.corrector(x_nwp, u)
        else:
            x_corr, gamma, delta = x_nwp, None, None

        # 专家输入：历史段 [订正后NWP, x_hist] + 未来段订正 NWP
        z = torch.cat([x_corr[:, :L, :], x_hist], dim=-1)   # [B, L, d_in]
        x_fut = x_corr[:, L:, :]                              # [B, H, d_nwp]

        preds = torch.stack([f(z, x_fut) for f in self.experts], dim=-1)  # [B,H,n_experts]

        if self.single_expert:
            moe_out = preds[:, :, 0]
            gate = torch.ones(B, 1, device=x_nwp.device)
        else:
            gate = self._gate(u, x_nwp)                       # [B, n_experts]
            moe_out = torch.einsum("bhk,bk->bh", preds, gate)

        # 辐照锚定：y = softplus(gain)·(订正后)未来辐照 + MoE 残差
        if self.irrad_anchor:
            irrad_fut = x_corr[:, L:, self.irrad_idx]         # [B, H] 订正后未来辐照
            y_hat = nn.functional.softplus(self.gain) * irrad_fut + moe_out
        else:
            y_hat = moe_out

        aux = {"u": u, "gate": gate, "preds": preds,
               "x_corr": x_corr, "gamma": gamma, "delta": delta}
        return y_hat, aux
