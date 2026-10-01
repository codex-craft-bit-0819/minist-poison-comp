"""标签平滑（Label Smoothing）。

通过 train_config 透传 label_smoothing 参数。TorchCNN 会启用 CrossEntropyLoss
的 label_smoothing；SklearnMLP 不支持则忽略（见 model.py 日志）。
数据本身不改动。
"""

from .base import BaseDefense


class LabelSmoothing(BaseDefense):
    name = "label_smoothing"
    description = "启用标签平滑训练（CNN 有效），缓解过拟合投毒"
    params = {"smoothing": 0.1}

    def defend(self, env, train):
        return {
            "images": train["images"].copy(),
            "labels": train["labels"].copy(),
            "sample_weights": None,
            "train_config": {"label_smoothing": float(self.params["smoothing"])},
        }
