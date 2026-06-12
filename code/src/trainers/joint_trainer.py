"""端到端训练器：L = L_pred（仅预测损失，早停于 val L_pred）。"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.losses import prediction_loss
from ..utils import metrics as M


class JointTrainer:
    def __init__(self, model, datasets, cfg, sid, capacity, device=None, logger=None):
        self.model = model
        self.datasets = datasets
        self.cfg = cfg
        self.sid = sid
        self.capacity = capacity
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = logger
        tc = cfg["train"]
        self.opt = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                    weight_decay=tc["weight_decay"])
        self.bs = tc["batch_size"]
        self.patience = tc["patience"]
        self.max_epochs = tc["max_epochs"]
        self.grad_clip = tc["grad_clip"]
        self.sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.max_epochs)
                      if tc.get("cosine_lr", False) else None)
        self.huber_delta = cfg.get("loss", {}).get("huber_delta", 0.0)

    def _run_epoch(self, ds, train: bool):
        loader = DataLoader(ds, batch_size=self.bs, shuffle=train)
        self.model.train(train)
        tot, cnt = 0.0, 0
        preds_all, trues_all, day_all = [], [], []
        for batch in loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            y_hat, aux = self.model(b)
            loss = prediction_loss(y_hat, b["y"], b["is_day"], self.huber_delta)
            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
            tot += loss.item() * b["y"].size(0)
            cnt += b["y"].size(0)
            if not train:
                preds_all.append(y_hat.detach().cpu().numpy())
                trues_all.append(b["y"].detach().cpu().numpy())
                day_all.append(b["is_day"].detach().cpu().numpy())
        avg = tot / max(cnt, 1)
        val_acc = ""
        if not train and preds_all:
            p = np.concatenate(preds_all)
            t = np.concatenate(trues_all)
            d = np.concatenate(day_all)
            val_acc = M.acc(p, t, capacity=1.0, is_day=d)
        return avg, val_acc

    def fit(self) -> str:
        best_val, best_state, bad = float("inf"), None, 0
        ckpt_dir = self.cfg["paths"]["checkpoints"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, f"best_{self.sid}.pth")
        for ep in range(1, self.max_epochs + 1):
            tr, _ = self._run_epoch(self.datasets["train"], True)
            if self.sched is not None:
                self.sched.step()
            with torch.no_grad():
                va, va_acc = self._run_epoch(self.datasets["val"], False)
            if self.logger:
                self.logger.log({"epoch": ep, "train_loss": tr, "val_loss": va,
                                 "val_acc": va_acc, "lr": self.opt.param_groups[0]["lr"]})
            if va < best_val - 1e-6:
                best_val, bad = va, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), ckpt)
        return ckpt
