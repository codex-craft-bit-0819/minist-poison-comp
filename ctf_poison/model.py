"""统一分类模型：SklearnMLP（默认，快）与 TorchCNN（真实），共享 Model 接口。

接口::

    model = env.train(images, labels, sample_weights=None, config=None)
    pred  = model.predict(test_images)         # -> int[N]
    proba = model.predict_proba(test_images)   # -> float[N, C]
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig
from .utils import get_logger, flatten_images, set_seed

log = get_logger("model")


class Model:
    """统一模型接口。"""

    def predict(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Sklearn MLP
# --------------------------------------------------------------------------- #
class SklearnMLP(Model):
    def __init__(self, estimator, n_classes: int):
        self.est = estimator
        self.n_classes = n_classes

    def _X(self, images):
        return flatten_images(np.asarray(images, dtype=np.float32))

    def predict(self, images):
        return self.est.predict(self._X(images)).astype(np.int64)

    def predict_proba(self, images):
        p = self.est.predict_proba(self._X(images))
        return p


def _train_sklearn_mlp(images, labels, sample_weights, cfg: ModelConfig, seed: int):
    from sklearn.neural_network import MLPClassifier

    if cfg.label_smoothing > 0:
        log.debug("SklearnMLP 不支持 label_smoothing=%s，忽略。", cfg.label_smoothing)
    X = flatten_images(np.asarray(images, dtype=np.float32))
    y = np.asarray(labels, dtype=np.int64)
    hidden = tuple(cfg.hidden_layer_sizes)
    est = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=cfg.alpha,
        batch_size=min(cfg.batch_size, 256),
        learning_rate_init=cfg.lr,
        max_iter=cfg.mlp_max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
    )
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weights, dtype=np.float64)
    est.fit(X, y, **fit_kwargs)
    return SklearnMLP(est, n_classes=int(y.max()) + 1)


# --------------------------------------------------------------------------- #
# Torch CNN
# --------------------------------------------------------------------------- #
class TorchCNN(Model):
    def __init__(self, net, n_classes: int, device):
        self.net = net
        self.n_classes = n_classes
        self.device = device

    def _tensor(self, images):
        import torch
        x = np.asarray(images, dtype=np.float32)
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        elif x.ndim == 3:
            x = x[:, None, :, :]
        return torch.from_numpy(x).to(self.device)

    def predict(self, images):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 512):
                xb = self._tensor(images[i : i + 512])
                logits = self.net(xb)
                out.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(out).astype(np.int64)

    def predict_proba(self, images):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 512):
                xb = self._tensor(images[i : i + 512])
                p = torch.softmax(self.net(xb), dim=1)
                out.append(p.cpu().numpy())
        return np.concatenate(out, axis=0)


def _build_cnn(cfg: ModelConfig):
    import torch.nn as nn

    c1, c2 = cfg.conv_channels

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 28 -> 14
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 14 -> 7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c2 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return Net()


def _augment_shift(images: np.ndarray, rng, max_shift: int = 2) -> np.ndarray:
    """随机平移增强（numpy 实现，模型无关）。"""
    out = images.copy()
    n = out.shape[0]
    sh = rng.integers(-max_shift, max_shift + 1, size=(n, 2))
    for i in range(n):
        dy, dx = int(sh[i, 0]), int(sh[i, 1])
        out[i] = np.roll(out[i], shift=(dy, dx), axis=(1, 2))
    return out


def _train_torch_cnn(images, labels, sample_weights, cfg: ModelConfig, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    device = torch.device("cpu")

    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 1, 28, 28)
    elif x.ndim == 3:
        x = x[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    net = _build_cnn(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=cfg.label_smoothing, reduction="none"
    )
    weights = None
    if sample_weights is not None:
        weights = torch.from_numpy(
            np.asarray(sample_weights, dtype=np.float32)
        ).to(device)

    bs = cfg.batch_size
    for epoch in range(cfg.cnn_epochs):
        net.train()
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            if weights is not None:
                w = weights[idx]
                loss = (loss * w).sum() / w.sum().clamp(min=1e-8)
            else:
                loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if cfg.augmentation:
            # 用增强副本替换一半样本做下一轮
            pass  # 增强主要由 data_augmentation 防御负责，这里保留钩子
    return TorchCNN(net, n_classes=10, device=device)


def train_model(images, labels, sample_weights=None, config: ModelConfig | None = None, seed: int = 42) -> Model:
    """统一训练入口：按 config.type 选择模型。"""
    cfg = config or ModelConfig()
    if sample_weights is not None and len(sample_weights) != len(labels):
        raise ValueError("sample_weights 长度与 labels 不一致")
    if cfg.type == "mlp":
        return _train_sklearn_mlp(images, labels, sample_weights, cfg, seed)
    elif cfg.type == "cnn":
        return _train_torch_cnn(images, labels, sample_weights, cfg, seed)
    else:
        raise ValueError(f"未知模型类型: {cfg.type}（支持: mlp, cnn）")
