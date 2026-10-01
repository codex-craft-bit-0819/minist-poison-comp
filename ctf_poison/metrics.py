"""攻防指标计算（严格对齐赛题公式）。"""

from __future__ import annotations

import numpy as np

from .utils import count_modified


def accuracy(model, images: np.ndarray, labels: np.ndarray) -> float:
    """分类准确率。"""
    pred = model.predict(images)
    labels = np.asarray(labels)
    return float(np.mean(np.asarray(pred) == labels))


def count_poison(
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    new_images: np.ndarray,
    new_labels: np.ndarray,
    img_tol: float = 1e-6,
) -> dict:
    """统计投毒修改情况（委托给 utils.count_modified）。"""
    return count_modified(clean_images, clean_labels, new_images, new_labels, img_tol)


def poison_rate(
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    new_images: np.ndarray,
    new_labels: np.ndarray,
    img_tol: float = 1e-6,
) -> float:
    """PoisonRate = N_modified / N_train。"""
    info = count_poison(clean_images, clean_labels, new_images, new_labels, img_tol)
    n_train = max(int(clean_labels.shape[0]), 1)
    return info["any_modified"] / n_train


def untargeted_asr(acc_clean: float, acc_attack: float) -> float:
    """非定向攻击成功率：归一化整体性能下降，越大越成功。"""
    denom = max(acc_clean, 1e-8)
    return float(max(0.0, (acc_clean - acc_attack) / denom))


def recovery_rate(acc_clean: float, acc_attack: float, acc_defense: float):
    """防御恢复率，截断到 [0,1]。

    若攻击未造成有效下降（分母≈0）返回 None，由调用方单独处理。
    """
    denom = acc_clean - acc_attack
    if denom <= 1e-6:
        return None
    return float(np.clip((acc_defense - acc_attack) / denom, 0.0, 1.0))


def effective_asr(acc_clean: float, acc_defense: float) -> float:
    """防御后残余攻击效果（用于全对阵矩阵的攻击-对抗意义）。"""
    denom = max(acc_clean, 1e-8)
    return float(max(0.0, (acc_clean - acc_defense) / denom))


def attack_success(acc_clean: float, acc_attack: float, delta: float = 0.05) -> bool:
    """攻击成功判定：Acc_attack < Acc_clean - Δ。"""
    return bool(acc_attack < acc_clean - delta)
