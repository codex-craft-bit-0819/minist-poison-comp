"""防御基类与共享工具。"""

import numpy as np

from ctf_poison.registry import BaseDefense
from ctf_poison.utils import flatten_images


def oof_predict_proba(images, labels, seed=42, cv=3, num_classes=None):
    """用交叉验证得到 out-of-fold 预测概率（LogReg 探针，快速）。

    用于 confident_learning / loss_reweight 等需要无偏逐样本估计的防御。

    返回 [N, num_classes] 矩阵，列对应原始标签值 0..num_classes-1，
    缺失类别列补 0，保证下游可按原始标签直接索引（即使某类被攻击整体抹掉）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = flatten_images(images)
    y = np.asarray(labels).astype(np.int64)
    clf = LogisticRegression(max_iter=200, C=1.0)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)

    if num_classes is None:
        num_classes = int(y.max()) + 1
    classes = np.unique(y)  # cross_val_predict 列序 = sorted(unique(y))
    full = np.zeros((len(y), num_classes), dtype=np.float32)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    return full


__all__ = ["BaseDefense", "oof_predict_proba"]
