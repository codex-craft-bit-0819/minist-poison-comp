"""孤立森林过滤（Isolation Forest Filter）。

在特征空间检测离群样本（含噪声扰动/异常像素的投毒样本通常更孤立），
移除最异常的一小部分，保留剩余样本训练。
"""

import numpy as np

from .base import BaseDefense
from ctf_poison.utils import flatten_images


class IsolationFilter(BaseDefense):
    name = "isolation_filter"
    description = "孤立森林检测并移除离群样本"
    params = {"contamination": 0.06, "seed": 42}

    def defend(self, env, train):
        from sklearn.ensemble import IsolationForest

        images = train["images"]
        labels = train["labels"]
        X = flatten_images(images)
        contam = float(self.params["contamination"])
        clf = IsolationForest(
            contamination=contam, random_state=self.params["seed"], n_jobs=-1
        )
        mask = clf.fit_predict(X) == 1  # 1 = 正常, -1 = 离群
        # 至少保留 70%
        if mask.sum() < int(0.7 * len(labels)):
            order = np.argsort(-clf.score_samples(X))  # 越小越异常 -> 取分数高的
            keep_n = max(int(0.7 * len(labels)), 1)
            mask = np.zeros(len(labels), dtype=bool)
            mask[np.argsort(clf.score_samples(X))[-keep_n:]] = True
        return {
            "images": images[mask].copy(),
            "labels": labels[mask].copy(),
            "sample_weights": None,
            "train_config": None,
        }
