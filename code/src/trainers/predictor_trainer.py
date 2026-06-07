"""阶段二：训练专家 + 门控（默认冻结订正器）。

监督 L_pred（仅白天），早停于 val L_pred，存 results/checkpoints/best_{sid}.pth。
config train.finetune_corrector 控制是否解冻订正器一起微调。
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.losses import prediction_loss
from ..utils import metrics as M
from ..utils.logger import Logger


class PredictorTrainer:
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

        self.finetune = tc["finetune_corrector"]
        if not self.finetune and model.corrector is not None:
            for p in model.corrector.parameters():
                p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
        self.bs = tc["batch_size"]
        self.patience = tc["patience"]
        self.max_epochs = tc["max_epochs"]
        self.grad_clip = tc["grad_clip"]

    def _run_epoch(self, ds, train: bool):
        loader = DataLoader(ds, batch_size=self.bs, shuffle=train)
        self.model.train(train)
        total, count = 0.0, 0
        preds_all, trues_all, day_all = [], [], []
        for batch in loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            y_hat, _ = self.model(b)
            loss = prediction_loss(y_hat, b["y"], b["is_day"])
            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
                self.opt.step()
            total += loss.item() * b["y"].size(0)
            count += b["y"].size(0)
            if not train:
                preds_all.append(y_hat.detach().cpu().numpy())
                trues_all.append(b["y"].detach().cpu().numpy())
                day_all.append(b["is_day"].detach().cpu().numpy())
        avg = total / max(count, 1)
        val_acc = ""
        if not train and preds_all:
            # 反归一化到 kW 再算 ACC（normalizer 在 dataset 外，evaluate 用；
            # 这里用归一化口径近似 val_acc，仅作早停参考显示）
            p = np.concatenate(preds_all); t = np.concatenate(trues_all)
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
            with torch.no_grad():
                va, va_acc = self._run_epoch(self.datasets["val"], False)
            if self.logger:
                self.logger.log({"epoch": ep, "stage": "predictor",
                                 "train_loss": tr, "val_loss": va,
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
