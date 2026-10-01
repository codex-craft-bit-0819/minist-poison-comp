"""防御方提交示例（No Defense 基线）。

接口对齐赛题::

    def defend(env, train: dict) -> dict:
        train = {"images": ndarray, "labels": ndarray}
        return {"images": clean_images, "labels": clean_labels,
                "sample_weights": None, "train_config": None}

返回规模可与输入不同（允许删除/扩充样本）。可选返回 sample_weights
与 train_config（部分覆盖统一模型训练配置，如 label_smoothing）。
"""

NAME = "user_no_defense"


def defend(env, train: dict) -> dict:
    return {
        "images": train["images"],
        "labels": train["labels"],
        "sample_weights": None,
        "train_config": None,
    }
