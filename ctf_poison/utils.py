"""通用工具：随机种子、日志、形状校验、修改样本计数。"""

from __future__ import annotations

import logging
import sys

import numpy as np

_LOGGER = None


def get_logger(name: str = "ctf_poison") -> logging.Logger:
    """返回一个带统一格式的 logger（重复调用不会重复添加 handler）。"""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER.getChild(name) if name != "ctf_poison" else _LOGGER
    logger = logging.getLogger("ctf_poison")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    _LOGGER = logger
    return logger


def set_seed(seed: int) -> np.random.Generator:
    """设置 numpy 随机种子并返回一个 Generator。"""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def as_float01(images: np.ndarray) -> np.ndarray:
    """把图像统一到 [0,1] float32，形状 [N,1,H,W]。"""
    images = np.asarray(images)
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    if images.max() > 1.5:  # 看起来是 [0,255]
        images = images / 255.0
    return images


def flatten_images(images: np.ndarray) -> np.ndarray:
    """[N,1,H,W] -> [N, H*W]。"""
    images = np.asarray(images)
    return images.reshape(images.shape[0], -1)


def validate_dataset(images: np.ndarray, labels: np.ndarray, name: str = "data") -> None:
    """校验数据集形状与一致性，失败抛出 ValueError。"""
    images = np.asarray(images)
    labels = np.asarray(labels)
    if images.ndim < 2:
        raise ValueError(f"[{name}] images 至少 2 维，得到 {images.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"[{name}] images 与 labels 数量不一致: {images.shape[0]} vs {labels.shape[0]}"
        )
    if labels.ndim != 1:
        raise ValueError(f"[{name}] labels 必须是 1 维，得到 {labels.shape}")


def ensure_shapes_match(
    ref_images: np.ndarray, ref_labels: np.ndarray,
    new_images: np.ndarray, new_labels: np.ndarray, name: str = "output",
) -> None:
    """保证攻防输出规模与输入一致（赛题要求）。"""
    if new_images.shape != ref_images.shape:
        raise ValueError(
            f"[{name}] images 形状 {new_images.shape} 与输入 {ref_images.shape} 不一致"
        )
    if new_labels.shape != ref_labels.shape:
        raise ValueError(
            f"[{name}] labels 形状 {new_labels.shape} 与输入 {ref_labels.shape} 不一致"
        )


def count_modified(
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    new_images: np.ndarray,
    new_labels: np.ndarray,
    img_tol: float = 1e-6,
) -> dict:
    """统计被修改的样本数。

    返回::

        {
          "label_changed": 仅标签被改的样本数,
          "image_changed": 仅像素被改的样本数,
          "any_modified":  标签或像素被改的样本数（用于 PoisonRate）,
        }
    """
    clean_images = np.asarray(clean_images)
    new_images = np.asarray(new_images)
    clean_labels = np.asarray(clean_labels)
    new_labels = np.asarray(new_labels)

    label_diff = new_labels != clean_labels
    n_label = int(np.sum(label_diff))

    if new_images.shape == clean_images.shape:
        img_diff = np.any(
            np.abs(new_images.astype(np.float64) - clean_images.astype(np.float64)) > img_tol,
            axis=tuple(range(1, new_images.ndim)),
        )
        n_img = int(np.sum(img_diff))
        n_any = int(np.sum(label_diff | img_diff))
    else:
        # 形状不一致（例如防御删除了样本），全部视为被改
        n_img = int(new_images.shape[0])
        n_any = int(new_images.shape[0])
    return {"label_changed": n_label, "image_changed": n_img, "any_modified": n_any}


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
