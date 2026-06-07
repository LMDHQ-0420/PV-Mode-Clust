"""配置加载 —— 读 YAML，支持 ablation 变体覆盖。"""
from __future__ import annotations

import copy
from typing import Any, Dict

import yaml


def _deep_update(base: Dict, override: Dict) -> Dict:
    """递归合并 override 到 base（就地修改 base 的深拷贝外层调用保证）。"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, override_path: str | None = None) -> Dict[str, Any]:
    """加载默认配置，可选叠加一个变体配置（仅覆盖相关项）。

    Args:
        path:          default.yaml 路径。
        override_path: 消融变体 yaml 路径（可为 None）。

    Returns:
        合并后的 config dict。
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            ov = yaml.safe_load(f) or {}
        cfg = _deep_update(copy.deepcopy(cfg), ov)
    return cfg
