#!/usr/bin/env python
"""数据投毒攻防对抗后端 CLI。

常用命令::

    python main.py list
    python main.py run --config config.json
    python main.py matchup -a random_label_flip -d no_defense
    python main.py submission --attack submissions/attack_template --defense submissions/defense_template
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 确保能 import 当前目录下的包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_list(_args):
    from ctf_poison import discover_attacks, discover_defenses

    attacks = discover_attacks()
    defenses = discover_defenses()
    print("=" * 60)
    print(f"攻击方法池 ({len(attacks)}):")
    for n in sorted(attacks):
        a = attacks[n]
        print(f"  - {n:22s} {a.description}")
    print(f"\n防御方法池 ({len(defenses)}):")
    for n in sorted(defenses):
        d = defenses[n]
        print(f"  - {n:22s} {d.description}")
    print("=" * 60)


def _build_config(args):
    from ctf_poison import load_config

    cfg = load_config(args.config) if getattr(args, "config", None) else load_config(None)
    if getattr(args, "model", None):
        cfg.model.type = args.model
    if getattr(args, "train_size", None):
        cfg.data.train_size = args.train_size
    if getattr(args, "test_size", None):
        cfg.data.test_size = args.test_size
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
        cfg.data.seed = args.seed
    if getattr(args, "poison_rate", None) is not None:
        cfg.poison_rate = args.poison_rate
    if getattr(args, "attacks", None):
        cfg.attacks = args.attacks.split(",")
    if getattr(args, "defenses", None):
        cfg.defenses = args.defenses.split(",")
    if getattr(args, "output", None):
        cfg.output_dir = args.output
    return cfg


def cmd_run(args):
    from ctf_poison import Orchestrator

    cfg = _build_config(args)
    orch = Orchestrator(cfg)
    result = orch.run_all_vs_all()
    files = result.save(cfg.output_dir)
    _print_summary(result, files)


def cmd_matchup(args):
    from ctf_poison import Orchestrator, discover_attacks, discover_defenses

    cfg = _build_config(args)
    orch = Orchestrator(cfg)
    # 单对阵：强制只跑指定的一对
    attacks = discover_attacks()
    defenses = discover_defenses()
    if args.attack not in attacks:
        print(f"未知攻击: {args.attack}，可选: {sorted(attacks)}")
        sys.exit(1)
    if args.defense not in defenses:
        print(f"未知防御: {args.defense}，可选: {sorted(defenses)}")
        sys.exit(1)
    res = orch.run_single(attacks[args.attack], defenses[args.defense])
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))


def cmd_submission(args):
    """加载外部选手提交目录跑全对阵。"""
    from ctf_poison import (
        Orchestrator,
        load_attack_submission,
        load_defense_submission,
        discover_attacks,
        discover_defenses,
    )

    cfg = _build_config(args)
    attacks = []
    defenses = []
    # 内置池（可选）
    if args.include_builtin:
        a_all = discover_attacks()
        d_all = discover_defenses()
        attacks += [a_all[n] for n in sorted(a_all)]
        defenses += [d_all[n] for n in sorted(d_all)]
    # 外部提交
    if args.attack:
        for i, d in enumerate(args.attack):
            attacks.append(load_attack_submission(d, name=f"user_attack_{i}"))
    if args.defense:
        for i, d in enumerate(args.defense):
            defenses.append(load_defense_submission(d, name=f"user_defense_{i}"))
    if not attacks or not defenses:
        print("需要至少一个 --attack 和一个 --defense 提交目录（或 --include-builtin）")
        sys.exit(1)
    orch = Orchestrator(cfg, attacks=attacks, defenses=defenses)
    result = orch.run_all_vs_all()
    files = result.save(cfg.output_dir)
    _print_summary(result, files)


def _print_summary(result, files):
    print("\n" + "=" * 60)
    print(f"Acc_clean = {result.acc_clean:.4f}  (train={result.train_size}, test={result.test_size})")
    print(f"对阵数 = {len(result.per_matchup)}  耗时 = {result.elapsed:.1f}s")
    print("-" * 60)
    print("攻击榜（平均 EffectiveASR 降序）:")
    arows = sorted(result.attack_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (a, s) in enumerate(arows, 1):
        am = result.attack_metrics[a]
        print(f"  {rank}. {a:22s} eff_asr={s:.4f}  no_def_asr={am['untargeted_asr']:.4f}  "
              f"poison={am['poison_rate']:.4f}")
    print("-" * 60)
    print("防御榜（平均 RecoveryRate 降序）:")
    drows = sorted(
        result.defense_scores.items(),
        key=lambda x: (x[1] if x[1] is not None else -1), reverse=True,
    )
    for rank, (d, s) in enumerate(drows, 1):
        s_str = "n/a" if s is None else f"{s:.4f}"
        print(f"  {rank}. {d:22s} recovery={s_str}  avg_acc={result.defense_acc_scores[d]:.4f}")
    print("-" * 60)
    print("输出文件:")
    for k, v in files.items():
        print(f"  {k:16s} {v}")
    print("=" * 60)


def build_parser():
    p = argparse.ArgumentParser(
        description="数据投毒攻防对抗后端", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出已注册的攻击/防御方法").set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="运行全对阵 (Attack x Defense)")
    pr.add_argument("--config", default="config.json")
    pr.add_argument("--model", choices=["mlp", "cnn"], default=None)
    pr.add_argument("--train-size", type=int, default=None)
    pr.add_argument("--test-size", type=int, default=None)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--poison-rate", type=float, default=None)
    pr.add_argument("--attacks", default=None, help="逗号分隔的攻击名，默认 all")
    pr.add_argument("--defenses", default=None, help="逗号分隔的防御名，默认 all")
    pr.add_argument("--output", default=None, help="输出目录（覆盖配置）")
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("matchup", help="运行单个 (attack, defense) 对阵")
    pm.add_argument("-a", "--attack", required=True)
    pm.add_argument("-d", "--defense", required=True)
    pm.add_argument("--config", default="config.json")
    pm.add_argument("--model", choices=["mlp", "cnn"], default=None)
    pm.add_argument("--train-size", type=int, default=None)
    pm.add_argument("--test-size", type=int, default=None)
    pm.add_argument("--seed", type=int, default=None)
    pm.add_argument("--poison-rate", type=float, default=None)
    pm.add_argument("--output", default=None, help="输出目录（仅 matchup 时不会落盘矩阵，这里忽略）")
    pm.set_defaults(func=cmd_matchup)

    ps = sub.add_parser("submission", help="加载外部选手提交目录跑全对阵")
    ps.add_argument("-a", "--attack", nargs="+", help="attack.py 所在目录（可多个）")
    ps.add_argument("-d", "--defense", nargs="+", help="defense.py 所在目录（可多个）")
    ps.add_argument("--include-builtin", action="store_true", help="同时包含内置攻防池")
    ps.add_argument("--config", default="config.json")
    ps.add_argument("--model", choices=["mlp", "cnn"], default=None)
    ps.add_argument("--train-size", type=int, default=None)
    ps.add_argument("--test-size", type=int, default=None)
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--output", default=None, help="输出目录（覆盖配置）")
    ps.set_defaults(func=cmd_submission)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
