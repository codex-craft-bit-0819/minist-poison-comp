"""全对阵编排：Attack x Defense 矩阵 + 聚合评分。

训练次数优化：Acc_clean 只算 1 次；每个攻击的 Acc_attack 只算 1 次并缓存复用，
因此总训练次数 = 1 + |A| + |A|*|D|。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import Config
from .dataset import load_dataset
from .env import Env
from .pipeline import MatchupResult, run_attack, run_matchup, train_and_eval
from .registry import (
    discover_attacks,
    discover_defenses,
    load_attack_submission,
    load_defense_submission,
)
from .utils import get_logger

log = get_logger("orchestrator")


@dataclass
class RunResult:
    """一次全对阵运行的全部结果。"""

    config: Config
    attack_names: list
    defense_names: list
    acc_clean: float
    per_matchup: list  # List[MatchupResult]
    matrices: dict  # name -> 2D ndarray (rows=attacks, cols=defenses)
    attack_metrics: dict  # per-attack: untargeted_asr, acc_attack, poison_rate, n_modified
    attack_scores: dict  # attack -> avg effective_asr over defenses
    defense_scores: dict  # defense -> avg recovery_rate over attacks
    defense_acc_scores: dict  # defense -> avg acc_defense
    elapsed: float = 0.0
    train_size: int = 0
    test_size: int = 0

    def save(self, output_dir: str):
        from .reporting import save_results

        return save_results(self, output_dir)


class Orchestrator:
    """攻防全对阵编排器。"""

    def __init__(self, config: Config, attacks=None, defenses=None):
        self.config = config
        self.env = Env(config)
        self.attacks = attacks if attacks is not None else self._select_attacks()
        self.defenses = defenses if defenses is not None else self._select_defenses()
        self._dedup_names()

    @classmethod
    def from_config(cls, path, **overrides) -> "Orchestrator":
        from .config import load_config

        cfg = load_config(path, **overrides) if path else load_config(None, **overrides)
        return cls(cfg)

    # ---- 方法选择 ---- #
    def _select_attacks(self) -> list:
        all_a = discover_attacks()
        names = self.config.attacks
        if not names or "all" in names:
            sel = [all_a[n] for n in sorted(all_a)]
        else:
            sel = []
            for n in names:
                if n not in all_a:
                    raise KeyError(f"未知攻击方法: {n}，可选: {sorted(all_a)}")
                sel.append(all_a[n])
        for sub in self.config.attack_submissions:
            sel.append(load_attack_submission(sub))
        return sel

    def _select_defenses(self) -> list:
        all_d = discover_defenses()
        names = self.config.defenses
        if not names or "all" in names:
            sel = [all_d[n] for n in sorted(all_d)]
        else:
            sel = []
            for n in names:
                if n not in all_d:
                    raise KeyError(f"未知防御方法: {n}，可选: {sorted(all_d)}")
                sel.append(all_d[n])
        for sub in self.config.defense_submissions:
            sel.append(load_defense_submission(sub))
        return sel

    def _dedup_names(self):
        # 外部提交可能与内置重名，加后缀去重
        seen_a, seen_d = set(), set()
        for a in self.attacks:
            n = a.name
            i = 1
            while n in seen_a:
                n = f"{a.name}#{i}"
                i += 1
            a.name = n
            seen_a.add(n)
        for d in self.defenses:
            n = d.name
            i = 1
            while n in seen_d:
                n = f"{d.name}#{i}"
                i += 1
            d.name = n
            seen_d.add(n)

    # ---- 单对阵（公开，供 CLI/外部调用）---- #
    def run_single(self, attack, defense) -> MatchupResult:
        """跑单个 (attack, defense) 对阵（不使用缓存，独立训练 Acc_clean/Acc_attack）。"""
        train, test = load_dataset(self.config.data)
        acc_clean, _ = train_and_eval(self.env, train, test)
        return run_matchup(
            self.env, attack, defense, train, test, acc_clean,
            delta=self.config.delta,
        )

    # ---- 全对阵 ---- #
    def run_all_vs_all(self) -> RunResult:
        t_start = time.time()
        train, test = load_dataset(self.config.data)
        acc_clean, _ = train_and_eval(self.env, train, test)
        log.info("Acc_clean = %.4f (训练集 %d, 测试集 %d)",
                 acc_clean, len(train["labels"]), len(test["labels"]))

        budget = self.env.poison_budget(len(train["labels"]))
        # 1) 每个攻击：跑一次攻击 + 算一次 Acc_attack，缓存
        attack_cache = {}
        attack_metrics = {}
        for a in self.attacks:
            t0 = time.time()
            meta = run_attack(self.env, a, train, budget)
            acc_atk, _ = train_and_eval(self.env, meta.poisoned, test)
            attack_cache[a.name] = {
                "poisoned": meta.poisoned,
                "acc_attack": acc_atk,
                "meta": meta,
            }
            attack_metrics[a.name] = {
                "acc_attack": acc_atk,
                "untargeted_asr": max(0.0, (acc_clean - acc_atk) / max(acc_clean, 1e-8)),
                "poison_rate": meta.poison_rate,
                "n_modified": meta.n_modified,
                "budget_violated": meta.budget_violated,
                "label_changed": meta.label_changed,
                "image_changed": meta.image_changed,
            }
            log.info(
                "[攻击] %-22s Acc_attack=%.4f ASR=%.4f PoisonRate=%.4f (%d/%d) %.1fs%s",
                a.name, acc_atk, attack_metrics[a.name]["untargeted_asr"],
                meta.poison_rate, meta.n_modified, budget, time.time() - t0,
                " [超预算已截断]" if meta.budget_violated else "",
            )

        # 2) 全对阵：每个 (attack, defense) 跑防御 + 训练 + 评测
        per_matchup = []
        attack_names = [a.name for a in self.attacks]
        defense_names = [d.name for d in self.defenses]
        for a in self.attacks:
            for d in self.defenses:
                t0 = time.time()
                res = run_matchup(
                    self.env, a, d, train, test, acc_clean,
                    delta=self.config.delta,
                    cached=attack_cache[a.name],
                )
                per_matchup.append(res)
                rec_str = "n/a" if res.recovery_rate is None else f"{res.recovery_rate:.3f}"
                log.info(
                    "[对阵] %-22s x %-22s Acc_def=%.4f Recover=%s EffASR=%.4f %.1fs",
                    a.name, d.name, res.acc_defense, rec_str, res.effective_asr,
                    time.time() - t0,
                )

        matrices = self._build_matrices(per_matchup, attack_names, defense_names)
        attack_scores, defense_scores, defense_acc_scores = self._aggregate(
            per_matchup, attack_names, defense_names
        )
        elapsed = time.time() - t_start
        log.info("全对阵完成: %d 个对阵, 耗时 %.1fs", len(per_matchup), elapsed)

        return RunResult(
            config=self.config,
            attack_names=attack_names,
            defense_names=defense_names,
            acc_clean=acc_clean,
            per_matchup=per_matchup,
            matrices=matrices,
            attack_metrics=attack_metrics,
            attack_scores=attack_scores,
            defense_scores=defense_scores,
            defense_acc_scores=defense_acc_scores,
            elapsed=elapsed,
            train_size=len(train["labels"]),
            test_size=len(test["labels"]),
        )

    def _build_matrices(self, per_matchup, attack_names, defense_names) -> dict:
        na, nd = len(attack_names), len(defense_names)
        ai = {n: i for i, n in enumerate(attack_names)}
        di = {n: i for i, n in enumerate(defense_names)}
        acc = np.full((na, nd), np.nan)
        rec = np.full((na, nd), np.nan)
        easr = np.full((na, nd), np.nan)
        for r in per_matchup:
            i, j = ai[r.attack_name], di[r.defense_name]
            acc[i, j] = r.acc_defense
            rec[i, j] = r.recovery_rate if r.recovery_rate is not None else np.nan
            easr[i, j] = r.effective_asr
        return {"acc_defense": acc, "recovery": rec, "effective_asr": easr}

    def _aggregate(self, per_matchup, attack_names, defense_names):
        attack_scores = {}
        for a in attack_names:
            vals = [r.effective_asr for r in per_matchup if r.attack_name == a]
            attack_scores[a] = float(np.mean(vals)) if vals else 0.0
        defense_scores = {}
        defense_acc_scores = {}
        for d in defense_names:
            recs = [r.recovery_rate for r in per_matchup
                    if r.defense_name == d and r.recovery_rate is not None]
            accs = [r.acc_defense for r in per_matchup if r.defense_name == d]
            defense_scores[d] = float(np.mean(recs)) if recs else None
            defense_acc_scores[d] = float(np.mean(accs)) if accs else 0.0
        return attack_scores, defense_scores, defense_acc_scores
