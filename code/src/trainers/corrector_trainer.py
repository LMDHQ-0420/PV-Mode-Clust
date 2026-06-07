"""阶段一：训练 NWP 误差订正器。

监督 = train 段 LMD（仅白天），早停于 val L_corr，存 results/checkpoints/corrector_{sid}.pth。
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from ..models.losses import corrector_loss
from ..utils.logger import Logger


class CorrectorTrainer:
    def __init__(self, corrector, datasets, cfg, sid, device=None, logger=None):
        self.corrector = corrector
        self.datasets = datasets
        self.cfg = cfg
        self.sid = sid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.corrector.to(self.device)
        self.logger = logger
        tc = cfg["train"]
        self.opt = torch.optim.Adam(corrector.parameters(), lr=tc["lr"],
                                    weight_decay=tc["weight_decay"])
        self.bs = tc["batch_size"]
        self.patience = tc["patience"]
        self.max_epochs = tc["max_epochs"]
        self.grad_clip = tc["grad_clip"]

    def _run_epoch(self, ds, train: bool):
        loader = DataLoader(ds, batch_size=self.bs, shuffle=train)
        self.corrector.train(train)
        total, count = 0.0, 0
        for batch in loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            _, corr_paired, _ = self.corrector(b["x_nwp"], b["u"])
            # 白天 mask 对齐到 L+H 段：用 lmd_totalirrad>0 不可得时退化为全 1；
            # 这里用 nwp 辐照>0 近似（归一化后 >0），保证仅白天监督。
            mask = (b["x_nwp"][:, :, 0] > 1e-6).float()  # [B, L+H]
            loss = corrector_loss(corr_paired, b["lmd_paired"], mask)
            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), self.grad_clip)
                self.opt.step()
            total += loss.item() * b["x_nwp"].size(0)
            count += b["x_nwp"].size(0)
        return total / max(count, 1)

    def fit(self) -> str:
        best_val, best_state, bad = float("inf"), None, 0
        ckpt_dir = self.cfg["paths"]["checkpoints"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, f"corrector_{self.sid}.pth")
        for ep in range(1, self.max_epochs + 1):
            tr = self._run_epoch(self.datasets["train"], True)
            with torch.no_grad():
                va = self._run_epoch(self.datasets["val"], False)
            if self.logger:
                self.logger.log({"epoch": ep, "stage": "corrector",
                                 "train_loss": tr, "val_loss": va,
                                 "val_acc": "", "lr": self.opt.param_groups[0]["lr"]})
            if va < best_val - 1e-6:
                best_val, best_state, bad = va, {k: v.detach().cpu().clone()
                                                 for k, v in self.corrector.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.corrector.load_state_dict(best_state)
        torch.save(self.corrector.state_dict(), ckpt)
        return ckpt
