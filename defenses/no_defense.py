"""无防御基线（No Defense）-- 透传，等价于不清洗。"""

from .base import BaseDefense


class NoDefense(BaseDefense):
    name = "no_defense"
    description = "不做任何清洗，原样返回（基线）"

    def defend(self, env, train):
        return {
            "images": train["images"],
            "labels": train["labels"],
            "sample_weights": None,
            "train_config": None,
        }
