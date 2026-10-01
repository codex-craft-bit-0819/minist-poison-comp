"""干净标签扰动（Clean-Label Perturbation）。

保持样本标签不变，但把图像向“另一个类别”的类中心混合，
使模型学到错误的特征-标签关联（干净标签投毒）。较难被标签级检测发现。
"""

import numpy as np

from .base import BaseAttack


class CleanLabelPerturb(BaseAttack):
    name = "clean_label_perturb"
    description = "保持标签，把图像向其他类中心混合（干净标签投毒）"
    params = {"seed": 99, "alpha": 0.6}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        num_classes = task.get("num_classes", 10)
        rng = np.random.default_rng(self.params["seed"])
        alpha = float(self.params["alpha"])

        # 计算各类中心
        flat = images.reshape(images.shape[0], -1)
        centroids = np.zeros((num_classes, flat.shape[1]), dtype=np.float32)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                centroids[c] = flat[mask].mean(axis=0)

        idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
        for i in idx:
            old = int(labels[i])
            tgt = rng.choice([c for c in range(num_classes) if c != old])
            flat[i] = (1 - alpha) * flat[i] + alpha * centroids[tgt]
        images = flat.reshape(images.shape).astype(np.float32)
        images = np.clip(images, 0.0, 1.0)
        return {"images": images, "labels": labels}
