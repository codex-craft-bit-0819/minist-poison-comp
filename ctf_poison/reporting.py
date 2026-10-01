"""结果落盘：summary.json / 多张 CSV 矩阵 / 热力图 (HTML+PNG)。"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from .utils import get_logger

log = get_logger("reporting")


def _safe_float(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return float(x)


def save_results(result, output_dir: str) -> dict:
    """把 RunResult 写入 output_dir，返回写入文件清单。"""
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    # 1) summary.json
    summary = {
        "config": _config_dict(result.config),
        "acc_clean": round(result.acc_clean, 6),
        "train_size": result.train_size,
        "test_size": result.test_size,
        "elapsed_s": round(result.elapsed, 3),
        "n_attacks": len(result.attack_names),
        "n_defenses": len(result.defense_names),
        "n_matchups": len(result.per_matchup),
        "attack_names": result.attack_names,
        "defense_names": result.defense_names,
        "attack_metrics": {
            k: {
                "acc_attack": round(v["acc_attack"], 6),
                "untargeted_asr": round(v["untargeted_asr"], 6),
                "poison_rate": round(v["poison_rate"], 6),
                "n_modified": v["n_modified"],
                "budget_violated": v["budget_violated"],
                "label_changed": v["label_changed"],
                "image_changed": v["image_changed"],
            }
            for k, v in result.attack_metrics.items()
        },
        "attack_scores": {k: round(_safe_float(v), 6) for k, v in result.attack_scores.items()},
        "defense_scores": {k: round(_safe_float(v), 6) for k, v in result.defense_scores.items()},
        "defense_acc_scores": {k: round(_safe_float(v), 6) for k, v in result.defense_acc_scores.items()},
        "per_matchup": [r.to_dict() for r in result.per_matchup],
    }
    p = os.path.join(output_dir, "summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    files["summary"] = p

    # 2) CSV 矩阵
    for name in ("acc_defense", "recovery", "effective_asr"):
        mat = result.matrices[name]
        p = _write_matrix_csv(
            os.path.join(output_dir, f"matrix_{name}.csv"),
            mat, result.attack_names, result.defense_names,
        )
        files[name] = p

    # 3) per_matchup.csv
    p = os.path.join(output_dir, "per_matchup.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(result.per_matchup[0].to_dict().keys()))
        for r in result.per_matchup:
            w.writerow(list(r.to_dict().values()))
    files["per_matchup"] = p

    # 4) 排行榜
    _write_leaderboards(output_dir, result, files)

    # 5) 热力图
    try:
        _write_heatmap_png(result, os.path.join(output_dir, "heatmap.png"))
        files["heatmap_png"] = os.path.join(output_dir, "heatmap.png")
    except Exception as e:
        log.warning("绘制 PNG 热力图失败: %s", e)
    try:
        files["heatmap_html"] = _write_heatmap_html(
            result, os.path.join(output_dir, "heatmap.html")
        )
    except Exception as e:
        log.warning("生成 HTML 热力图失败: %s", e)

    log.info("结果已写入 %s （%d 个对阵）", output_dir, len(result.per_matchup))
    return files


def _config_dict(cfg) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def _write_matrix_csv(path, mat, row_names, col_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack \\ defense"] + list(col_names))
        for i, rn in enumerate(row_names):
            row = [rn]
            for j in range(len(col_names)):
                v = mat[i, j]
                row.append("" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.6f}")
            w.writerow(row)
    return path


def _write_leaderboards(output_dir, result, files):
    # 攻击榜（按平均 EffectiveASR 降序）
    p = os.path.join(output_dir, "leaderboard_attack.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attack", "avg_effective_asr", "untargeted_asr_no_defense",
                    "acc_attack", "poison_rate", "n_modified"])
        rows = []
        for a in result.attack_names:
            am = result.attack_metrics[a]
            rows.append((
                result.attack_scores.get(a, 0.0),
                am["untargeted_asr"],
                am["acc_attack"],
                am["poison_rate"],
                am["n_modified"],
                a,
            ))
        rows.sort(reverse=True)
        for rank, (score, uasr, acc_atk, pr, nm, a) in enumerate(rows, 1):
            w.writerow([rank, a, f"{score:.6f}", f"{uasr:.6f}",
                        f"{acc_atk:.6f}", f"{pr:.6f}", nm])
    files["leaderboard_attack"] = p

    # 防御榜（按平均 RecoveryRate 降序，并列出平均 Acc_defense）
    p = os.path.join(output_dir, "leaderboard_defense.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "defense", "avg_recovery_rate", "avg_acc_defense"])
        rows = []
        for d in result.defense_names:
            rows.append((
                result.defense_scores.get(d),
                result.defense_acc_scores.get(d, 0.0),
                d,
            ))
        # None 视为最低
        rows.sort(key=lambda x: (x[0] if x[0] is not None else -1), reverse=True)
        for rank, (rec, acc, d) in enumerate(rows, 1):
            rec_s = "n/a" if rec is None else f"{rec:.6f}"
            w.writerow([rank, d, rec_s, f"{acc:.6f}"])
    files["leaderboard_defense"] = p


def _write_heatmap_png(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(6 + 3 * len(result.defense_names) * 0.5,
                                            2 + 0.4 * len(result.attack_names)))
    titles = {
        "acc_defense": "Acc_defense (higher = better defense)",
        "recovery": "RecoveryRate (higher = better defense)",
        "effective_asr": "EffectiveASR (higher = better attack)",
    }
    for ax, key in zip(axes, ["acc_defense", "recovery", "effective_asr"]):
        mat = result.matrices[key]
        masked = np.ma.masked_invalid(mat)
        ax.imshow(masked, cmap="RdYlGn" if key != "effective_asr" else "RdYlGn_r",
                  aspect="auto", vmin=np.nanmin(mat) if not np.all(np.isnan(mat)) else 0,
                  vmax=np.nanmax(mat) if not np.all(np.isnan(mat)) else 1)
        ax.set_xticks(range(len(result.defense_names)))
        ax.set_xticklabels(result.defense_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(result.attack_names)))
        ax.set_yticklabels(result.attack_names, fontsize=8)
        ax.set_title(titles[key], fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "n/a"
                else:
                    txt = f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_heatmap_html(result, path):
    """纯 HTML 热力图（无需 matplotlib 也能看）。"""
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<style>",
             "body{font-family:sans-serif;margin:16px}",
             "table{border-collapse:collapse;margin:12px 0}",
             "th,td{border:1px solid #ccc;padding:5px 8px;text-align:center;font-size:12px}",
             "th{background:#f0f0f0}",
             "h2{margin-top:24px}",
             "</style></head><body>"]
    parts.append(f"<h1>攻防对抗全对阵结果</h1>")
    parts.append(f"<p>Acc_clean = {result.acc_clean:.4f} | "
                 f"训练集 {result.train_size} | 测试集 {result.test_size} | "
                 f"耗时 {result.elapsed:.1f}s</p>")
    titles = {
        "acc_defense": "Acc_defense（防御后准确率，越高防御越好）",
        "recovery": "RecoveryRate（恢复率，越高防御越好）",
        "effective_asr": "EffectiveASR（残余攻击效果，越高攻击越好）",
    }
    for key in ["acc_defense", "recovery", "effective_asr"]:
        mat = result.matrices[key]
        parts.append(f"<h2>{titles[key]}</h2>")
        parts.append("<table><thead><tr><th>attack \\ defense</th>")
        for d in result.defense_names:
            parts.append(f"<th>{d}</th>")
        parts.append("</tr></thead><tbody>")
        # 计算该矩阵的取值范围用于着色
        vals = mat[~np.isnan(mat)]
        lo, hi = (vals.min(), vals.max()) if vals.size else (0, 1)
        for i, a in enumerate(result.attack_names):
            parts.append("<tr><td style='text-align:left;font-weight:bold'>%s</td>" % a)
            for j in range(len(result.defense_names)):
                v = mat[i, j]
                if np.isnan(v):
                    parts.append("<td style='background:#eee'>n/a</td>")
                else:
                    t = 0.5 if hi == lo else (v - lo) / (hi - lo)
                    if key == "effective_asr":
                        t = 1 - t  # 高 ASR -> 红
                    # 绿(t=1) 红(t=0)
                    r = int(255 * (1 - t))
                    g = int(255 * t)
                    color = f"rgb({r},{g},120)"
                    parts.append(
                        f"<td style='background:{color};color:#000'>{v:.3f}</td>"
                    )
            parts.append("</tr>")
        parts.append("</tbody></table>")
    # 排行榜
    parts.append("<h2>攻击榜（平均 EffectiveASR 降序）</h2><table><thead><tr>"
                 "<th>rank</th><th>attack</th><th>avg_eff_asr</th>"
                 "<th>untargeted_asr(无防御)</th><th>poison_rate</th></tr></thead><tbody>")
    arows = sorted(
        [(result.attack_scores.get(a, 0), result.attack_metrics[a]["untargeted_asr"],
          result.attack_metrics[a]["poison_rate"], a) for a in result.attack_names],
        reverse=True,
    )
    for rank, (s, u, pr, a) in enumerate(arows, 1):
        parts.append(f"<tr><td>{rank}</td><td>{a}</td><td>{s:.4f}</td><td>{u:.4f}</td><td>{pr:.4f}</td></tr>")
    parts.append("</tbody></table>")
    parts.append("<h2>防御榜（平均 RecoveryRate 降序）</h2><table><thead><tr>"
                 "<th>rank</th><th>defense</th><th>avg_recovery</th><th>avg_acc_defense</th></tr></thead><tbody>")
    drows = sorted(
        [(result.defense_scores.get(d), result.defense_acc_scores.get(d, 0), d) for d in result.defense_names],
        key=lambda x: (x[0] if x[0] is not None else -1), reverse=True,
    )
    for rank, (rec, acc, d) in enumerate(drows, 1):
        rec_s = "n/a" if rec is None else f"{rec:.4f}"
        parts.append(f"<tr><td>{rank}</td><td>{d}</td><td>{rec_s}</td><td>{acc:.4f}</td></tr>")
    parts.append("</tbody></table></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return path
