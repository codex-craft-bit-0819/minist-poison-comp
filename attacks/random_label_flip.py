"""随机标签翻转（Random Label Flip）—— 赛题最小示例的同款。"""

import numpy as np

from .base import BaseAttack


class RandomLabelFlip(BaseAttack):
    name = "random_label_flip"
    description = "随机选取 budget 个样本，将标签翻转为其他类别"
    params = {"seed": 42}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        num_classes = task.get("num_classes", 10)
        rng = np.random.default_rng(self.params["seed"])

        idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
        for i in idx:
            old = int(labels[i])
            candidates = [c for c in range(num_classes) if c != old]
            labels[i] = rng.choice(candidates)
        return {"images": images, "labels": labels}
