"""攻击方提交示例（Random Label Flip）。

接口对齐赛题::

    def attack(env, task: dict) -> dict:
        task = {"images": ndarray, "labels": ndarray, "poison_budget": int, ...}
        return {"images": poisoned_images, "labels": poisoned_labels}

输出规模必须与原训练集一致；修改样本数不得超过 task["poison_budget"]。
"""

import numpy as np

NAME = "user_random_label_flip"


def attack(env, task: dict) -> dict:
    images = task["images"].copy()
    labels = task["labels"].copy()

    budget = int(task["poison_budget"])
    rng = np.random.default_rng(42)

    indices = rng.choice(len(labels), size=min(budget, len(labels)), replace=False)
    for idx in indices:
        old = int(labels[idx])
        candidates = [x for x in range(10) if x != old]
        labels[idx] = rng.choice(candidates)

    return {"images": images, "labels": labels}
