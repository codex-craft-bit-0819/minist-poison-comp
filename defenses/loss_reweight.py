"""损失降权鲁棒训练（Loss-based Reweighting）。

用交叉验证 OOF 概率计算每个样本的交叉熵损失，对高损失（可能被投毒）
样本降权，实现 trim-loss 风格的鲁棒训练。数据本身不变，返回 sample_weights。
"""

import numpy as np

from .base import BaseDefense, oof_predict_proba


class LossReweight(BaseDefense):
    name = "loss_reweight"
    description = "按 OOF 损失对可疑样本降权（鲁棒训练）"
    params = {"seed": 42, "drop_fraction": 0.08, "low_weight": 0.1, "cv": 3}

    def defend(self, env, train):
        images = train["images"]
        labels = train["labels"].copy()
        proba = oof_predict_proba(
            images, labels, seed=self.params["seed"], cv=self.params["cv"],
            num_classes=getattr(env, "num_classes", None),
        )
        eps = 1e-8
        true_proba = proba[np.arange(len(labels)), labels]
        loss = -np.log(true_proba + eps)

        frac = float(self.params["drop_fraction"])
        thresh = np.quantile(loss, 1.0 - frac)
        low_w = float(self.params["low_weight"])
        weights = np.where(loss >= thresh, low_w, 1.0).astype(np.float32)
        return {
            "images": images.copy(),
            "labels": labels,
            "sample_weights": weights,
            "train_config": None,
        }
