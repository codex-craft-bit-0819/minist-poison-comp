"""数据增强（Data Augmentation）。

为每个样本生成一份随机平移副本并入训练集，扩大有效数据量、提升对噪声
投毒的鲁棒性。模型无关，对 MLP/CNN 均有效。
"""

import numpy as np

from .base import BaseDefense


class DataAugmentation(BaseDefense):
    name = "data_augmentation"
    description = "加入随机平移副本扩充训练集"
    params = {"seed": 42, "max_shift": 2, "copies": 1}

    def defend(self, env, train):
        images = train["images"]
        labels = train["labels"]
        rng = np.random.default_rng(self.params["seed"])
        max_shift = int(self.params["max_shift"])
        copies = int(self.params["copies"])

        new_imgs = [images]
        new_lbls = [labels]
        for _ in range(copies):
            aug = images.copy()
            n = aug.shape[0]
            sh = rng.integers(-max_shift, max_shift + 1, size=(n, 2))
            for i in range(n):
                dy, dx = int(sh[i, 0]), int(sh[i, 1])
                aug[i] = np.roll(aug[i], shift=(dy, dx), axis=(1, 2))
            new_imgs.append(aug)
            new_lbls.append(labels.copy())

        out_images = np.concatenate(new_imgs, axis=0).astype(np.float32)
        out_labels = np.concatenate(new_lbls, axis=0).astype(np.int64)
        return {
            "images": out_images,
            "labels": out_labels,
            "sample_weights": None,
            "train_config": None,
        }
