"""攻击/防御环境 Env：提供统一训练与推理接口及工具。"""

from __future__ import annotations

import numpy as np

from .config import Config, ModelConfig
from .model import Model, train_model
from .utils import get_logger, set_seed

log = get_logger("env")


class Env:
    """攻防双方共享的运行环境。

    - ``env.train(...)`` 是赛题规定的统一训练接口。
    - 返回的 model 具有 ``predict`` / ``predict_proba``。
    - 另暴露数据维度、类别数、默认训练配置等元信息，方便方法实现。
    """

    def __init__(self, config: Config):
        self.config = config
        self.num_classes = 10
        self.image_shape = (28, 28)
        self.input_channels = 1
        self.rng = set_seed(config.seed)

    # ---- 赛题统一接口 ---- #
    def train(
        self,
        images,
        labels,
        sample_weights=None,
        config: ModelConfig | dict | None = None,
    ) -> Model:
        """统一训练接口。

        防御方可通过输出中的 ``train_config`` 传回部分覆盖（如 label_smoothing、
        augmentation）；本方法会将其合并到默认 ModelConfig 上。
        """
        cfg = self._merge_config(config)
        return train_model(images, labels, sample_weights, cfg, seed=self.config.seed)

    def _merge_config(self, override) -> ModelConfig:
        base = ModelConfig(**self.config.model.__dict__)
        if override is None:
            return base
        if isinstance(override, ModelConfig):
            return override
        if isinstance(override, dict):
            for k, v in override.items():
                if hasattr(base, k) and v is not None:
                    setattr(base, k, v)
            return base
        return base

    # ---- 元信息 / 工具 ---- #
    @property
    def default_train_config(self) -> ModelConfig:
        return ModelConfig(**self.config.model.__dict__)

    @property
    def poison_budget_fraction(self) -> float:
        return self.config.poison_rate

    def poison_budget(self, n_train: int) -> int:
        """给定训练集大小，返回可修改样本数上限。"""
        return max(1, int(self.config.poison_rate * n_train))

    def train_probe(self, images, labels, config: ModelConfig | dict | None = None) -> Model:
        """训练一个轻量探针模型（防御方可用，例如估计逐样本 loss）。"""
        probe_cfg = self._merge_config(config)
        # 探针更快：MLP 少迭代；CNN 少 epoch
        if probe_cfg.type == "mlp":
            probe_cfg.mlp_max_iter = min(probe_cfg.mlp_max_iter, 15)
        else:
            probe_cfg.cnn_epochs = min(probe_cfg.cnn_epochs, 3)
        return train_model(images, labels, None, probe_cfg, seed=self.config.seed)
