"""模块 D：软门控多专家完整模型（组装 A/B/C1/C2/D）。

核心（RQ2，idea_report 3.7）：区制软隶属 u 同时作门控权重，软混合 K 个专家。
gate_mode / single_expert / use_vmd / use_corrector 是消融 B/C/D/F 的开关。
aux 收集 u / 各专家预测 / x_corr，为可解释 4 图预留接口（Part 3 §3）。

数据流（见 implementation §3.8）：
  x_corr = corrector(x_nwp, u)             （use_corrector=False 时直接用 x_nwp）
  z = concat([x_corr 历史段, x_hist, nu])   作为专家输入 [B, L, d_in]
  pred_k = f_k(z)  → 堆叠 [B, H, K]
  g = 门控权重（soft: u 在预测段池化；hard: argmax one-hot；single: K=1）
  y_hat = Σ_k g_k · pred_k                  [B, H]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .experts import build_expert


class GatedMoEForecaster(nn.Module):
    def __init__(self, corrector, K: int, expert_cfg: dict,
                 d_nwp: int, d_hist: int, M: int, horizon: int,
                 use_vmd: bool = True, use_corrector: bool = True,
                 gate_mode: str = "soft", single_expert: bool = False):
        """
        Args:
            corrector:     NWPCorrector 实例（可冻结）。
            K:             区制数 / 专家数。
            d_nwp/d_hist:  NWP、历史特征维度。
            M:             VMD 模态数。
            horizon:       预测步数 H。
        """
        super().__init__()
        self.corrector = corrector
        self.K = K
        self.horizon = horizon
        self.use_vmd = use_vmd
        self.use_corrector = use_corrector
        self.gate_mode = gate_mode
        self.single_expert = single_expert

        # 专家历史输入维度：x_corr(历史段, d_nwp) + x_hist(d_hist) + nu(M if use_vmd)
        d_in = d_nwp + d_hist + (M if use_vmd else 0)
        self.d_in = d_in
        self.d_fut = d_nwp   # 未来段订正后 NWP（日前预报）驱动预测
        n_experts = 1 if single_expert else K
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [build_expert(expert_cfg, d_in, horizon, self.d_fut)
             for _ in range(n_experts)]
        )

    def _gate(self, u: torch.Tensor) -> torch.Tensor:
        """由软隶属 u [B, L+H, K] 算门控权重 g [B, K]。

        soft：对未来段（预测段）的隶属做时间平均；
        hard：对该平均取 argmax 的 one-hot；
        single：返回全 1（只有 1 个专家，外部按 n_experts 处理）。
        """
        H = self.horizon
        u_future = u[:, -H:, :]                 # [B, H, K]
        g = u_future.mean(dim=1)               # [B, K]
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        if self.gate_mode == "hard":
            idx = g.argmax(dim=-1)             # [B]
            g = torch.zeros_like(g).scatter_(-1, idx[:, None], 1.0)
        return g

    def forward(self, batch: dict):
        """
        Args:
            batch: 含 x_nwp[B,L+H,d_nwp], x_hist[B,L,d_hist], nu[B,L,M], u[B,L+H,K]。

        Returns:
            y_hat: [B, H]
            aux:   dict（u, x_corr, corr_paired, e_hat, preds[B,H,n_experts], gate[B,K]）
        """
        x_nwp = batch["x_nwp"]
        x_hist = batch["x_hist"]
        nu = batch["nu"]
        u = batch["u"]
        B, LH, _ = x_nwp.shape
        L = x_hist.shape[1]

        # ---- 模块 B：订正 ----
        if self.use_corrector and self.corrector is not None:
            x_corr, corr_paired, e_hat = self.corrector(x_nwp, u)
        else:
            x_corr, corr_paired, e_hat = x_nwp, None, None

        # ---- 组装专家历史输入 z（历史段）+ 未来段订正 NWP（日前预报，驱动预测）----
        x_corr_hist = x_corr[:, :L, :]                 # [B, L, d_nwp]
        x_fut = x_corr[:, L:, :]                        # [B, H, d_nwp] 未来段订正 NWP
        parts = [x_corr_hist, x_hist]
        if self.use_vmd:
            parts.append(nu)                           # [B, L, M]
        z = torch.cat(parts, dim=-1)                   # [B, L, d_in]

        # ---- 模块 C2：各专家（历史上下文 + 未来订正 NWP）----
        preds = torch.stack([f(z, x_fut) for f in self.experts], dim=-1)  # [B, H, n_experts]

        # ---- 模块 D：门控混合 ----
        if self.single_expert:
            y_hat = preds[:, :, 0]                     # 单专家直接输出
            gate = torch.ones(B, 1, device=x_nwp.device)
        else:
            gate = self._gate(u)                       # [B, K]
            y_hat = torch.einsum("bhk,bk->bh", preds, gate)  # [B, H]

        aux = {
            "u": u, "x_corr": x_corr, "corr_paired": corr_paired,
            "e_hat": e_hat, "preds": preds, "gate": gate,
        }
        return y_hat, aux
