"""自包含深度学习 baseline 合集（无 TSLib 依赖）。

覆盖 RNN / Conv / MLP-basis / Attention 四大家族，均与主方法同口径（日前96步，7:1:2划分）。

模型列表：
  LSTM            — 双层 LSTM seq2seq
  LSTNet          — Conv + GRU + skip GRU + AR（Lai et al. 2018）
  TCN             — 膨胀因果卷积（Bai et al. 2018）
  NBEATS          — 多栈 MLP 基展开（Oreshkin et al. 2020）
  NHiTS           — 多尺度池化 MLP（Challu et al. 2023）
  Crossformer     — 跨时间 patch attention（简化版，Zhang & Yan 2023）
  NWPLSTMbaseline — Encoder-Decoder LSTM，消费未来NWP（与主方法同信息量）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NWP_AWARE_MODELS = {"NWPLSTMbaseline"}


# ─────────────────────────── LSTM ───────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, d_hist: int, horizon: int, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_hist, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)
        # forget gate bias = 1.0 (减缓梯度消失)
        for name, p in self.lstm.named_parameters():
            if "bias_hh" in name:
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x):           # x [B, L, C] → [B, H]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ─────────────────────────── LSTNet ──────────────────────────
class LSTNetForecaster(nn.Module):
    def __init__(self, d_hist: int, seq_len: int, horizon: int,
                 n_filters: int = 32, conv_k: int = 8,
                 rnn_hidden: int = 64, skip: int = 96, ar_window: int = 48):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, (conv_k, d_hist))
        conv_out_len = seq_len - conv_k + 1          # e.g. 192-8+1=185

        self.gru = nn.GRU(n_filters, rnn_hidden, batch_first=True)

        # skip GRU：取 conv 输出最后 skip 步
        self.skip = skip
        self.skip_len = min(skip, conv_out_len)
        self.skip_gru = nn.GRU(n_filters, rnn_hidden // 2, batch_first=True)

        # AR 分量：power 通道最后 ar_window 步的线性组合
        self.ar_window = ar_window
        self.ar = nn.Linear(ar_window, horizon)

        self.out = nn.Linear(rnn_hidden + rnn_hidden // 2, horizon)

    def forward(self, x):           # x [B, L, C] → [B, H]
        B = x.size(0)
        # Conv 特征
        c = self.conv(x.unsqueeze(1)).squeeze(-1)   # [B, n_filters, L']
        c = F.relu(c).transpose(1, 2)               # [B, L', n_filters]

        _, h_gru = self.gru(c)                       # h [1, B, rnn_hidden]
        gru_out = h_gru.squeeze(0)                   # [B, rnn_hidden]

        skip_in = c[:, -self.skip_len:, :]           # [B, skip_len, n_filters]
        _, h_skip = self.skip_gru(skip_in)
        skip_out = h_skip.squeeze(0)                 # [B, rnn_hidden//2]

        feat = torch.cat([gru_out, skip_out], dim=-1)  # [B, rnn_hidden + rnn_hidden//2]
        out = self.out(feat)                           # [B, H]

        # AR 分量（power = 第 0 列）
        ar_in = x[:, -self.ar_window:, 0]            # [B, ar_window]
        out = out + self.ar(ar_in)
        return out


# ─────────────────────────── TCN ─────────────────────────────
class _CausalConvBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):           # x [B, C, L]
        r = x
        x = F.pad(x, (self.pad, 0))
        x = self.drop(self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)))
        x = F.pad(x, (self.pad, 0))
        x = self.drop(self.act(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2)))
        return x + r


class TCNForecaster(nn.Module):
    def __init__(self, d_hist: int, horizon: int, channels: int = 64,
                 kernel: int = 3, dilations=(1, 4, 16, 64), dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(d_hist, channels, 1)
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(channels, kernel, d, dropout) for d in dilations]
        )
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):           # x [B, L, C] → [B, H]
        x = self.proj(x.transpose(1, 2))   # [B, channels, L]
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, :, -1])      # 取最后时间步


# ─────────────────────────── N-BEATS ─────────────────────────
class _NBEATSBlock(nn.Module):
    def __init__(self, input_len: int, horizon: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast_head = nn.Linear(hidden, input_len)
        self.forecast_head = nn.Linear(hidden, horizon)

    def forward(self, x):           # x [B, L] → backcast [B, L], forecast [B, H]
        h = self.mlp(x)
        return self.backcast_head(h), self.forecast_head(h)


class NBEATSForecaster(nn.Module):
    def __init__(self, d_hist: int, seq_len: int, horizon: int,
                 n_stacks: int = 2, n_blocks: int = 3, hidden: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(d_hist, 1)   # 压通道 → 单变量
        self.stacks = nn.ModuleList([
            nn.ModuleList([_NBEATSBlock(seq_len, horizon, hidden) for _ in range(n_blocks)])
            for _ in range(n_stacks)
        ])

    def forward(self, x):           # x [B, L, C] → [B, H]
        # 投影到单变量
        residual = self.input_proj(x).squeeze(-1)   # [B, L]
        forecast = torch.zeros(residual.size(0), self.stacks[0][0].forecast_head.out_features,
                               device=x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, fcast = block(residual)
                residual = residual - backcast.detach()
                forecast = forecast + fcast
        return forecast


# ─────────────────────────── N-HiTS ──────────────────────────
class _NHiTSBlock(nn.Module):
    def __init__(self, input_len: int, horizon: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast_head = nn.Linear(hidden, input_len)
        self.forecast_head = nn.Linear(hidden, horizon)

    def forward(self, x):
        h = self.mlp(x)
        return self.backcast_head(h), self.forecast_head(h)


class NHiTSForecaster(nn.Module):
    def __init__(self, d_hist: int, seq_len: int, horizon: int,
                 pool_sizes=(1, 4, 16), n_blocks: int = 2, hidden: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(d_hist, 1)
        self.pool_sizes = pool_sizes
        self.horizon = horizon
        self.stacks = nn.ModuleList()
        for ps in pool_sizes:
            pooled_len = seq_len // ps
            self.stacks.append(nn.ModuleList(
                [_NHiTSBlock(pooled_len, horizon, hidden) for _ in range(n_blocks)]
            ))
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(ps, stride=ps) for ps in pool_sizes]
        )

    def forward(self, x):           # x [B, L, C] → [B, H]
        uni = self.input_proj(x).squeeze(-1)         # [B, L]
        forecast = torch.zeros(uni.size(0), self.horizon, device=x.device)
        residual = uni
        for pool_op, stack in zip(self.pools, self.stacks):
            pooled = pool_op(residual.unsqueeze(1)).squeeze(1)   # [B, L//ps]
            for block in stack:
                bc, fc = block(pooled)
                pooled = pooled - bc.detach()
                forecast = forecast + fc
            # 把 backcast 上采样回原长后从 residual 减去
            bc_up = F.interpolate(pooled.unsqueeze(1), size=residual.size(1),
                                  mode="linear", align_corners=False).squeeze(1)
            residual = residual - bc_up.detach()
        return forecast


# ─────────────────────────── Crossformer (简化) ──────────────
class CrossformerForecaster(nn.Module):
    def __init__(self, d_hist: int, seq_len: int, horizon: int,
                 patch_size: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        n_patches = (seq_len - patch_size) // stride + 1
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = n_patches

        self.patch_embed = nn.Linear(patch_size * d_hist, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x):           # x [B, L, C] → [B, H]
        B, L, C = x.shape
        # 提取 patch
        patches = []
        for i in range(self.n_patches):
            s = i * self.stride
            patches.append(x[:, s:s + self.patch_size, :].reshape(B, -1))  # [B, patch*C]
        patches = torch.stack(patches, dim=1)   # [B, n_patches, patch*C]
        tokens = self.patch_embed(patches)       # [B, n_patches, d_model]
        tokens = self.encoder(tokens)            # [B, n_patches, d_model]
        return self.head(tokens.reshape(B, -1))  # [B, H]


# ─────────────────────────── NWP-aware LSTM ──────────────────
class NWPLSTMBaseline(nn.Module):
    """Encoder-Decoder LSTM，与主方法同信息量（历史+未来NWP），无 MoE/订正。"""

    def __init__(self, d_hist: int, d_nwp: int, horizon: int,
                 hidden: int = 128, n_layers: int = 2, nwp_proj: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(d_hist, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.nwp_proj = nn.Linear(d_nwp, nwp_proj)
        self.decoder = nn.LSTM(nwp_proj, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden, 1)
        # forget gate bias
        for lstm in (self.encoder, self.decoder):
            for name, p in lstm.named_parameters():
                if "bias_hh" in name:
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x_hist, x_nwp_fut):  # x_hist [B,L,C], x_nwp_fut [B,H,d_nwp] → [B,H]
        _, (h, c) = self.encoder(x_hist)
        dec_in = F.relu(self.nwp_proj(x_nwp_fut))   # [B, H, nwp_proj]
        dec_out, _ = self.decoder(dec_in, (h, c))    # [B, H, hidden]
        return self.out_proj(dec_out).squeeze(-1)    # [B, H]


# ─────────────────────────── 工厂 ────────────────────────────
def build_dl_model(name: str, cfg: dict, d_hist: int, d_nwp: int = 7) -> nn.Module:
    L = cfg["data"]["look_back"]
    H = cfg["data"]["horizon"]
    if name == "LSTM":
        return LSTMForecaster(d_hist, H)
    if name == "LSTNet":
        return LSTNetForecaster(d_hist, L, H)
    if name == "TCN":
        return TCNForecaster(d_hist, H)
    if name == "NBEATS":
        return NBEATSForecaster(d_hist, L, H)
    if name == "NHiTS":
        return NHiTSForecaster(d_hist, L, H)
    if name == "Crossformer":
        return CrossformerForecaster(d_hist, L, H)
    if name == "NWPLSTMbaseline":
        return NWPLSTMBaseline(d_hist, d_nwp, H)
    raise ValueError(f"未知 DL baseline: {name}")
