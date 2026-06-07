"""模块 A：FCM 软天气区制。

train 段拟合 K 个中心，对任意段用**固定中心**算软隶属 u(t)∈[N,K]（推理一致）。
关键（idea_report 3.5）：聚类只用 NWP 侧特征（推理可得），不依赖 LMD。
借鉴前身 fcm_clustering，但改 argmax 硬标签为软隶属向量。
"""
from __future__ import annotations

from typing import List

import numpy as np

try:
    from skfuzzy.cluster import cmeans
except ImportError:  # 允许在无 skfuzzy 环境下 import 文件
    cmeans = None


class FCMRegime:
    """模糊 C 均值软天气区制。

    Attributes:
        centers_: [K, d] 聚类中心（train 段固定）。
    """

    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 feature_cols: List[str] | None = None,
                 error: float = 0.005, maxiter: int = 1000):
        self.K = n_clusters
        self.m = m
        self.feature_cols = feature_cols or []
        self.error = error
        self.maxiter = maxiter
        self.centers_ = None  # [K, d]

    def fit(self, train_features: np.ndarray) -> "FCMRegime":
        """在 train 段拟合 K 个中心。

        Args:
            train_features: [N, d] NWP 侧气象特征（已选列）。
        """
        if cmeans is None:
            raise ImportError("需要 scikit-fuzzy：pip install scikit-fuzzy")
        X = np.asarray(train_features, dtype=float)
        # skfuzzy 约定：data shape = [features, samples]
        cntr, u, u0, d, jm, p, fpc = cmeans(
            X.T, self.K, self.m,
            error=self.error, maxiter=self.maxiter, init=None,
        )
        self.centers_ = np.asarray(cntr, dtype=float)  # [K, d]
        return self

    def soft_membership(self, features: np.ndarray) -> np.ndarray:
        """用固定中心对任意段算软隶属。

        u_k(t) = ||m_t - c_k||^(-2/(m-1)) / Σ_j ||m_t - c_j||^(-2/(m-1))

        Args:
            features: [N, d] 气象特征。

        Returns:
            u: [N, K]，逐行和为 1。
        """
        if self.centers_ is None:
            raise RuntimeError("FCMRegime 未 fit")
        X = np.asarray(features, dtype=float)              # [N, d]
        # 到各中心的欧氏距离 [N, K]
        dist = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        dist = np.fmax(dist, 1e-10)                        # 防除零
        power = -2.0 / (self.m - 1.0)
        num = dist ** power                                # [N, K]
        u = num / np.sum(num, axis=1, keepdims=True)
        return u
