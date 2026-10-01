"""配置定义与加载（同时支持 JSON 与 YAML）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ModelConfig:
    """统一分类模型配置。"""

    type: str = "mlp"  # "mlp" | "cnn"
    # --- SklearnMLP ---
    hidden_layer_sizes: tuple = (256, 128)
    mlp_max_iter: int = 40
    alpha: float = 1e-4  # L2 正则
    # --- TorchCNN ---
    cnn_epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    conv_channels: tuple = (16, 32)
    # --- 通用训练选项（防御方可通过 train_config 覆盖）---
    label_smoothing: float = 0.0  # CNN 有效；MLP 忽略
    augmentation: bool = False    # CNN 有效；防御 data_augmentation 也会开启


@dataclass
class DataConfig:
    """数据集配置。"""

    train_size: int = 8000
    test_size: int = 2000
    cache_dir: str = "data"
    normalize: bool = True
    synthetic: bool = False  # True 强制使用离线合成数据
    seed: int = 42


@dataclass
class Config:
    """顶层配置。"""

    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    poison_rate: float = 0.05  # PoisonRate 上限（投毒预算比例）
    delta: float = 0.05        # 攻击成功判定阈值（个百分点）
    attacks: list = field(default_factory=lambda: ["all"])
    defenses: list = field(default_factory=lambda: ["all"])
    output_dir: str = "results"
    enforce_budget: bool = True  # 攻击超出预算时截断并告警
    # 外部选手提交目录（可选）
    attack_submissions: list = field(default_factory=list)
    defense_submissions: list = field(default_factory=list)


def _try_load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | os.PathLike | None = None, **overrides: Any) -> Config:
    """从 JSON/YAML 文件加载配置，再用 overrides 覆盖顶层字段。"""
    cfg_dict: dict = {}
    if path is not None:
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        if path.endswith((".yaml", ".yml")):
            cfg_dict = _try_load_yaml(path)
            if cfg_dict is None:
                raise RuntimeError(
                    "读取 YAML 配置需要 PyYAML，请 `pip install pyyaml` 或改用 .json 配置"
                )
        else:
            with open(path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
    cfg_dict.update(overrides)
    return _build_config(cfg_dict)


def _build_config(d: dict) -> Config:
    model_cfg = ModelConfig(**(d.get("model") or {}))
    data_cfg = DataConfig(**(d.get("data") or {}))
    return Config(
        seed=d.get("seed", 42),
        model=model_cfg,
        data=data_cfg,
        poison_rate=d.get("poison_rate", 0.05),
        delta=d.get("delta", 0.05),
        attacks=d.get("attacks", ["all"]),
        defenses=d.get("defenses", ["all"]),
        output_dir=d.get("output_dir", "results"),
        enforce_budget=d.get("enforce_budget", True),
        attack_submissions=d.get("attack_submissions", []),
        defense_submissions=d.get("defense_submissions", []),
    )


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)
