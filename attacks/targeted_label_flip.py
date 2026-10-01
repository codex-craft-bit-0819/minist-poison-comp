"""定向标签翻转（Targeted Label Flip）。

将源类别 src 的样本标签翻转为目标类别 dst，造成两类之间的混淆。
"""

import numpy as np

from .base import BaseAttack


class TargetedLabelFlip(BaseAttack):
    name = "targeted_label_flip"
    description = "把源类样本标签翻转为目标类，制造定向混淆"
    params = {"seed": 7, "src": 1, "dst": 7}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = int(task["poison_budget"])
        src = int(self.params["src"])
        dst = int(self.params["dst"])
        rng = np.random.default_rng(self.params["seed"])

        src_idx = np.where(labels == src)[0]
        if len(src_idx) == 0 or src == dst:
            # 退化为随机翻转
            idx = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
            for i in idx:
                labels[i] = (int(labels[i]) + 1) % 10
            return {"images": images, "labels": labels}

        n_flip = min(budget, len(src_idx))
        chosen = rng.choice(src_idx, size=n_flip, replace=False)
        labels[chosen] = dst
        return {"images": images, "labels": labels}
