"""高斯噪声注入（Gaussian Noise Injection）。

在 budget 个样本的像素上叠加高斯噪声（保持标签），破坏特征质量。
"""

import numpy as np

from .base import BaseAttack


class GaussianNoise(BaseAttack):
    name = "gaussian_noise"
    description = "对 budget 个样本叠加高斯噪声（不改标签），降低特征质量"
    params = {"seed": 11, "std": 0.35}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        rng = np.random.default_rng(self.params["seed"])
        std = float(self.params["std"])

        idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
        noise = rng.normal(0.0, std, size=images[idx].shape).astype(images.dtype)
        images[idx] = np.clip(images[idx] + noise, 0.0, 1.0)
        return {"images": images, "labels": labels}
