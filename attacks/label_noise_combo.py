"""标签+噪声组合攻击（Label Flip + Noise Combo）。

对同一批 budget 样本同时翻转标签并注入噪声，最大化破坏强度。
"""

import numpy as np

from .base import BaseAttack


class LabelNoiseCombo(BaseAttack):
    name = "label_noise_combo"
    description = "对同一批样本同时翻转标签并注入高斯噪声"
    params = {"seed": 5, "std": 0.3}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        num_classes = task.get("num_classes", 10)
        rng = np.random.default_rng(self.params["seed"])
        std = float(self.params["std"])

        idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
        # 翻转标签
        for i in idx:
            old = int(labels[i])
            candidates = [c for c in range(num_classes) if c != old]
            labels[i] = rng.choice(candidates)
        # 注入噪声
        noise = rng.normal(0.0, std, size=images[idx].shape).astype(images.dtype)
        images[idx] = np.clip(images[idx] + noise, 0.0, 1.0)
        return {"images": images, "labels": labels}
