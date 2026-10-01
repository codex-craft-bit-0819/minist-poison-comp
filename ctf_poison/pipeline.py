"""单对阵流水线：攻击 -> 防御 -> 训练 -> 评测 -> 指标。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .env import Env
from .metrics import (
    accuracy,
    attack_success,
    count_poison,
    effective_asr,
    poison_rate,
    recovery_rate,
    untargeted_asr,
)
from .utils import ensure_shapes_match, get_logger

log = get_logger("pipeline")


@dataclass
class AttackMeta:
    """单次攻击的产物与统计。"""

    poisoned: dict
    poison_rate: float
    n_modified: int
    budget: int
    budget_violated: bool
    label_changed: int
    image_changed: int


@dataclass
class MatchupResult:
    """一对 (attack, defense) 的完整评测结果。"""

    attack_name: str
    defense_name: str
    acc_clean: float
    acc_attack: float
    acc_defense: float
    untargeted_asr: float
    recovery_rate: Optional[float]
    effective_asr: float
    poison_rate: float
    n_modified: int
    attack_success: bool
    budget: int
    budget_violated: bool
    train_time_defense: float = 0.0
    defense_changed_size: bool = False
    defense_n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "attack": self.attack_name,
            "defense": self.defense_name,
            "acc_clean": round(self.acc_clean, 6),
            "acc_attack": round(self.acc_attack, 6),
            "acc_defense": round(self.acc_defense, 6),
            "untargeted_asr": round(self.untargeted_asr, 6),
            "recovery_rate": None if self.recovery_rate is None else round(self.recovery_rate, 6),
            "effective_asr": round(self.effective_asr, 6),
            "poison_rate": round(self.poison_rate, 6),
            "n_modified": self.n_modified,
            "attack_success": self.attack_success,
            "budget": self.budget,
            "budget_violated": self.budget_violated,
            "defense_changed_size": self.defense_changed_size,
            "defense_n_samples": self.defense_n_samples,
            "train_time_defense_s": round(self.train_time_defense, 3),
        }


def _modified_mask(clean_images, clean_labels, new_images, new_labels, img_tol=1e-6):
    label_diff = new_labels != clean_labels
    img_diff = np.any(
        np.abs(new_images.astype(np.float64) - clean_images.astype(np.float64)) > img_tol,
        axis=tuple(range(1, new_images.ndim)),
    )
    return label_diff | img_diff


def _enforce_budget(env: Env, clean: dict, poisoned: dict, budget: int) -> tuple[dict, bool]:
    """若攻击超出预算，回退多余的修改样本到干净值。"""
    info = count_poison(
        clean["images"], clean["labels"], poisoned["images"], poisoned["labels"]
    )
    n_mod = info["any_modified"]
    if n_mod <= budget:
        return poisoned, False
    mask = _modified_mask(clean["images"], clean["labels"], poisoned["images"], poisoned["labels"])
    idx = np.where(mask)[0]
    keep = env.rng.choice(idx, size=budget, replace=False)
    revert = np.setdiff1d(idx, keep)
    images = poisoned["images"].copy()
    labels = poisoned["labels"].copy()
    images[revert] = clean["images"][revert]
    labels[revert] = clean["labels"][revert]
    log.warning(
        "攻击修改 %d 个样本超出预算 %d，已回退 %d 个。",
        n_mod, budget, len(revert),
    )
    return {"images": images, "labels": labels}, True


def run_attack(env: Env, attack, clean: dict, budget: int) -> AttackMeta:
    """执行攻击并统计投毒情况。"""
    task = {
        "images": clean["images"].copy(),
        "labels": clean["labels"].copy(),
        "poison_budget": budget,
        "num_classes": env.num_classes,
    }
    out = attack(env, task)
    if not isinstance(out, dict) or "images" not in out or "labels" not in out:
        raise ValueError(f"攻击 {attack} 必须返回 {{'images':..., 'labels':...}}")
    images = np.asarray(out["images"], dtype=np.float32)
    labels = np.asarray(out["labels"]).astype(clean["labels"].dtype)
    if images.ndim == 2:
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim == 3:
        images = images[:, None, :, :]
    poisoned = {"images": images, "labels": labels}
    try:
        ensure_shapes_match(clean["images"], clean["labels"], images, labels, name=f"attack::{getattr(attack, 'name', 'attack')}")
    except ValueError as e:
        raise ValueError(f"攻击输出规模必须与原训练集一致：{e}")

    violated = False
    if env.config.enforce_budget:
        poisoned, violated = _enforce_budget(env, clean, poisoned, budget)

    info = count_poison(clean["images"], clean["labels"], poisoned["images"], poisoned["labels"])
    pr = info["any_modified"] / max(len(clean["labels"]), 1)
    return AttackMeta(
        poisoned=poisoned,
        poison_rate=pr,
        n_modified=info["any_modified"],
        budget=budget,
        budget_violated=violated,
        label_changed=info["label_changed"],
        image_changed=info["image_changed"],
    )


def run_defense(env: Env, defense, poisoned: dict) -> dict:
    """执行防御，返回清洗后的数据（可含 sample_weights / train_config）。"""
    inp = {"images": poisoned["images"], "labels": poisoned["labels"]}
    out = defense(env, inp)
    if not isinstance(out, dict) or "images" not in out or "labels" not in out:
        raise ValueError(f"防御 {defense} 必须返回 {{'images':..., 'labels':...}}")
    images = np.asarray(out["images"], dtype=np.float32)
    labels = np.asarray(out["labels"]).astype(np.int64)
    if images.ndim == 2:
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim == 3:
        images = images[:, None, :, :]
    if images.shape[0] != labels.shape[0]:
        raise ValueError("防御输出 images 与 labels 数量不一致")
    result = {
        "images": images,
        "labels": labels,
        "sample_weights": out.get("sample_weights"),
        "train_config": out.get("train_config"),
    }
    return result


def train_and_eval(env: Env, data: dict, test: dict) -> tuple[float, object]:
    """在 data 上训练并在 test 上评测准确率。"""
    model = env.train(
        data["images"],
        data["labels"],
        sample_weights=data.get("sample_weights"),
        config=data.get("train_config"),
    )
    acc = accuracy(model, test["images"], test["labels"])
    return acc, model


def run_matchup(
    env: Env,
    attack,
    defense,
    clean_train: dict,
    clean_test: dict,
    acc_clean: float,
    delta: float = 0.05,
    cached: Optional[dict] = None,
) -> MatchupResult:
    """跑一对 (attack, defense)。

    ``cached`` 可传入已计算的 ``{"acc_attack":..., "poisoned":..., "meta":...}``，
    避免对同一攻击重复训练（由 Orchestrator 维护）。
    """
    budget = env.poison_budget(len(clean_train["labels"]))

    if cached is not None and cached.get("poisoned") is not None:
        meta = cached["meta"]
        acc_attack = cached["acc_attack"]
        poisoned = cached["poisoned"]
    else:
        meta = run_attack(env, attack, clean_train, budget)
        poisoned = meta.poisoned
        _, model_atk = train_and_eval(env, poisoned, clean_test)
        acc_attack = accuracy(model_atk, clean_test["images"], clean_test["labels"])

    defended = run_defense(env, defense, poisoned)
    t0 = time.time()
    acc_defense, _ = train_and_eval(env, defended, clean_test)
    dt = time.time() - t0

    pr = poison_rate(
        clean_train["images"], clean_train["labels"],
        poisoned["images"], poisoned["labels"],
    )
    rec = recovery_rate(acc_clean, acc_attack, acc_defense)
    return MatchupResult(
        attack_name=getattr(attack, "name", str(attack)),
        defense_name=getattr(defense, "name", str(defense)),
        acc_clean=acc_clean,
        acc_attack=acc_attack,
        acc_defense=acc_defense,
        untargeted_asr=untargeted_asr(acc_clean, acc_attack),
        recovery_rate=rec,
        effective_asr=effective_asr(acc_clean, acc_defense),
        poison_rate=pr,
        n_modified=meta.n_modified,
        attack_success=attack_success(acc_clean, acc_attack, delta),
        budget=budget,
        budget_violated=meta.budget_violated,
        train_time_defense=dt,
        defense_changed_size=defended["images"].shape[0] != len(clean_train["labels"]),
        defense_n_samples=defended["images"].shape[0],
    )
