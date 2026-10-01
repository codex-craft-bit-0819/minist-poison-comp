"""置信学习标签纠错（Confident Learning Label Correction）。

用交叉验证 OOF 概率找出“模型对给定标签非常不认可”的样本（给定标签的
预测概率低于阈值），把其标签纠正为预测类别。采用保守准则，避免在干净
数据上误改正确标签；对标签翻转类攻击有效（自实现，不依赖 cleanlab）。
"""

import numpy as np

from .base import BaseDefense, oof_predict_proba


class ConfidentLearning(BaseDefense):
    name = "confident_learning"
    description = "基于交叉验证置信度保守地纠正可疑标签"
    params = {"seed": 42, "given_label_prob_thresh": 0.1, "cv": 3}

    def defend(self, env, train):
        images = train["images"].copy()
        labels = train["labels"].copy()
        proba = oof_predict_proba(
            images, labels, seed=self.params["seed"], cv=self.params["cv"],
            num_classes=getattr(env, "num_classes", None),
        )
        pred = proba.argmax(axis=1)
        # 仅当模型对“给定标签”几乎不认可时才纠正（保守，减少误改）
        given_prob = proba[np.arange(len(labels)), labels]
        thresh = float(self.params["given_label_prob_thresh"])
        mislabeled = (pred != labels) & (given_prob < thresh)
        labels[mislabeled] = pred[mislabeled]
        return {
            "images": images,
            "labels": labels,
            "sample_weights": None,
            "train_config": None,
        }
