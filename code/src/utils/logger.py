"""训练日志 —— 按 epoch 追加写 CSV。"""
from __future__ import annotations

import csv
import os
from typing import Dict


class Logger:
    """逐 epoch 把指标追加到 logs/train_{timestamp}.csv。

    字段：epoch, stage, train_loss, val_loss, val_acc, lr（见 implementation §5.2）。
    首次写入时自动写表头，列以首条记录的 keys 为准。
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._header_written = os.path.exists(log_path) and os.path.getsize(log_path) > 0
        self._fieldnames = None

    def log(self, metrics: Dict) -> None:
        """追加一条记录。"""
        if self._fieldnames is None:
            self._fieldnames = list(metrics.keys())
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: metrics.get(k, "") for k in self._fieldnames})
