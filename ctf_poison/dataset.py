"""数据集加载：MNIST（fetch_openml 带缓存）+ 离线合成兜底。"""

from __future__ import annotations

import os

import numpy as np

from .config import DataConfig
from .utils import get_logger

log = get_logger("dataset")


def _to_image_format(images: np.ndarray, normalize: bool) -> np.ndarray:
    """转为 [N,1,28,28] float32，按需归一化到 [0,1]。"""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        # [N, 784] -> [N,1,28,28]
        n = images.shape[0]
        images = images.reshape(n, 1, 28, 28)
    elif images.ndim == 3:
        # [N,28,28] -> [N,1,28,28]
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim == 4:
        pass
    else:
        raise ValueError(f"不支持的图像形状: {images.shape}")
    if normalize:
        if images.max() > 1.5:
            images = images / 255.0
        images = np.clip(images, 0.0, 1.0)
    return images


def _load_mnist_openml(cfg: DataConfig):
    """从 OpenML 拉 MNIST，缓存到 cfg.cache_dir。"""
    from sklearn.datasets import fetch_openml

    os.makedirs(cfg.cache_dir, exist_ok=True)
    log.info("从 OpenML 加载 MNIST（缓存目录 %s）...", cfg.cache_dir)
    bunch = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=cfg.cache_dir,
    )
    X = bunch.data.astype(np.float32)  # [N, 784]
    y = bunch.target.astype(np.int64)  # 字符串 '0'..'9' -> int
    return X, y


def _make_synthetic(cfg: DataConfig, n_total: int):
    """离线合成 10 类“类数字”图像数据（无网络时兜底）。

    为每个类别生成一张随机稀疏模板，样本 = 类模板 + 高斯噪声，
    保证 MLP/CNN 能学到较高准确率，使攻击/防御指标可观测。
    """
    rng = np.random.default_rng(cfg.seed)
    n_classes = 10
    log.warning("使用合成数据（n=%d）。如需真实 MNIST 请联网重试。", n_total)

    # 每类一个随机模板：在 28x28 上撒几个高斯块
    templates = np.zeros((n_classes, 28, 28), dtype=np.float32)
    for c in range(n_classes):
        for _ in range(3 + c % 3):
            cy, cx = rng.integers(4, 24, size=2)
            sigma = rng.uniform(1.5, 3.0)
            yy, xx = np.mgrid[0:28, 0:28]
            templates[c] += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    templates = np.clip(templates / templates.max(axis=(1, 2), keepdims=True), 0, 1)

    per_class = n_total // n_classes
    X = np.zeros((per_class * n_classes, 28, 28), dtype=np.float32)
    y = np.zeros(per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        s, e = c * per_class, (c + 1) * per_class
        noise = rng.normal(0.0, 0.25, size=(per_class, 28, 28)).astype(np.float32)
        X[s:e] = np.clip(templates[c][None] + noise, 0.0, 1.0)
        y[s:e] = c

    # 打乱
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    return X.astype(np.float32), y.astype(np.int64)


def load_dataset(cfg: DataConfig) -> tuple[dict, dict]:
    """加载训练集与测试集。

    返回 (train, test)，各自为 ``{"images": [N,1,28,28], "labels": [N]}``。
    """
    rng = np.random.default_rng(cfg.seed)
    total_needed = cfg.train_size + cfg.test_size

    X, y = None, None
    if not cfg.synthetic:
        try:
            X, y = _load_mnist_openml(cfg)
        except Exception as e:  # 网络失败等
            log.warning("OpenML 加载失败：%s，回退到合成数据。", e)
            X, y = None, None

    if X is None:
        X, y = _make_synthetic(cfg, max(total_needed * 3, 20000))

    n = X.shape[0]
    total_needed = min(total_needed, n)
    perm = rng.permutation(n)[:total_needed]
    X, y = X[perm], y[perm]
    tr = cfg.train_size
    if tr >= total_needed:
        tr = max(total_needed - 500, int(total_needed * 0.8))
    Xtr, ytr = X[:tr], y[:tr]
    Xte, yte = X[tr:total_needed], y[tr:total_needed]

    train = {
        "images": _to_image_format(Xtr, cfg.normalize),
        "labels": ytr.astype(np.int64),
    }
    test = {
        "images": _to_image_format(Xte, cfg.normalize),
        "labels": yte.astype(np.int64),
    }
    log.info(
        "数据就绪: train=%s test=%s (label 范围 %d-%d)",
        train["images"].shape, test["images"].shape, int(y.min()), int(y.max()),
    )
    return train, test
