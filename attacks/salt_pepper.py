"""椒盐噪声（Salt & Pepper）。

对 budget 个样本随机将部分像素置为 0 或 1，模拟脉冲噪声。
"""

import numpy as np

from .base import BaseAttack


class SaltPepper(BaseAttack):
    name = "salt_pepper"
    description = "对 budget 个样本施加椒盐脉冲噪声（不改标签）"
    params = {"seed": 23, "amount": 0.25}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        rng = np.random.default_rng(self.params["seed"])
        amount = float(self.params["amount"])

        idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
        for i in idx:
            img = images[i]
            mask = rng.random(img.shape)
            img = np.where(mask < amount / 2, 0.0, img)
            img = np.where(mask > 1 - amount / 2, 1.0, img)
            images[i] = img
        return {"images": images, "labels": labels}
