"""固定随机种子，保证可复现（numpy / torch / cuda）。"""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """固定所有随机源。

    Args:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 研究代码：确定性优先于极致速度，但不强制 deterministic 算法以免报错
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
