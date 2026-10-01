# 手写数字分类数据投毒攻防对抗后端

基于赛题《手写数字分类数据投毒攻防对抗赛题》实现的攻防对抗评测后端。对**任意攻击方法 × 任意防御方法**组合，自动跑通 `攻击 → 防御 → 统一模型训练 → 隐藏测试集评测` 流水线，并计算攻击指标与防御指标，输出全对阵热力矩阵与排行榜。

## 功能特性

- **统一接口**：严格对齐赛题 `attack(env, task)` / `defend(env, train)` / `env.train(...).predict(...)`。
- **任意组合**：内置 6 攻击 × 6 防御池，支持加载外部选手提交（`attack.py` / `defense.py`）。
- **自动指标计算**：
  - 攻击侧：`UntargetedASR`、`AccuracyDrop`、`PoisonRate`、攻击成功判定。
  - 防御侧：`RecoveryRate`、`Acc_defense`、`EffectiveASR`（防御后残余破坏）。
- **全对阵矩阵**：`Acc_defense` / `RecoveryRate` / `EffectiveASR` 三张矩阵 + 攻防排行榜。
- **统一模型**：`Model-A` 默认 sklearn MLP（CPU 秒级），可选 PyTorch CNN（更真实）。
- **预算校验**：自动统计 `PoisonRate`，超出预算可截断并告警。

## 目录结构

```
1/
├── main.py                # CLI 入口
├── config.json            # 主配置
├── requirements.txt
├── ctf_poison/            # 后端核心
├── attacks/               # 内置攻击池 A0
├── defenses/              # 内置防御池 D0
├── submissions/           # 选手提交模板
└── results/               # 运行产物（自动生成）
```

## 安装

```bash
pip install -r requirements.txt
```

> 默认配置（`model=mlp`）只需 `numpy` + `scikit-learn` + `scipy` 即可运行。首次运行会从 OpenML 下载 MNIST 到 `data/` 缓存；无网络时自动回退到合成数据。

## 快速开始

```bash
# 1) 查看已注册的攻防方法
python main.py list

# 2) 跑全对阵（默认 6×6，MLP，8000 训练样本，约 1-2 分钟）
python main.py run --config config.json

# 3) 跑单个对阵
python main.py matchup -a random_label_flip -d confident_learning

# 4) 用真实 CNN
python main.py run --model cnn --train-size 10000

# 5) 加载外部选手提交跑全对阵
python main.py submission \
    --attack submissions/attack_template \
    --defense submissions/defense_template \
    --include-builtin
```

## Python API

```python
from ctf_poison import Orchestrator

orch = Orchestrator.from_config("config.json")
result = orch.run_all_vs_all()
result.save("results/")            # 落盘 JSON/CSV/热力图

# 直接取矩阵与评分
result.matrices["recovery"]        # [n_attacks, n_defenses]
result.attack_scores               # {attack: avg_effective_asr}
result.defense_scores              # {defense: avg_recovery_rate}
```

## 内置攻防方法池

### 攻击池 A0

| 名称 | 说明 |
|------|------|
| `random_label_flip` | 随机翻转 budget 个样本的标签 |
| `targeted_label_flip` | 把源类样本标签翻转为目标类（定向混淆） |
| `gaussian_noise` | 对样本叠加高斯噪声（不改标签） |
| `salt_pepper` | 椒盐脉冲噪声（不改标签） |
| `clean_label_perturb` | 保持标签，把图像向其他类中心混合（干净标签投毒） |
| `label_noise_combo` | 同批样本同时翻转标签 + 注入噪声 |

### 防御池 D0

| 名称 | 说明 |
|------|------|
| `no_defense` | 不清洗（基线） |
| `confident_learning` | 交叉验证置信度检测并纠正可疑标签 |
| `isolation_filter` | 孤立森林检测并移除离群样本 |
| `loss_reweight` | 按 OOF 损失对可疑样本降权（鲁棒训练） |
| `label_smoothing` | 启用标签平滑训练（CNN 有效） |
| `data_augmentation` | 加入随机平移副本扩充训练集 |

## 指标定义（对齐赛题）

- `Acc_clean`：干净训练集训练得到的准确率（全对阵只算 1 次）。
- `Acc_attack`：投毒训练集（无防御）训练得到的准确率（每个攻击算 1 次，缓存复用）。
- `Acc_defense`：防御后训练集训练得到的准确率（每个对阵算 1 次）。
- `PoisonRate = N_modified / N_train`，推荐 ≤ 5%。
- `UntargetedASR = max(0, (Acc_clean − Acc_attack) / Acc_clean)`，越高攻击越成功。
- `RecoveryRate = clip((Acc_defense − Acc_attack) / (Acc_clean − Acc_attack), 0, 1)`，越高防御越好；攻击无效（分母≈0）时记为 `n/a`。
- `EffectiveASR = max(0, (Acc_clean − Acc_defense) / Acc_clean)`，防御后残余破坏，用于全对阵矩阵的攻防对抗意义。
- 攻击成功判定：`Acc_attack < Acc_clean − Δ`（默认 Δ = 0.05）。

**聚合评分**：
- 攻击分 = 各防御下平均 `EffectiveASR`（+无防御 `UntargetedASR`）。
- 防御分 = 各攻击下平均 `RecoveryRate`。

## 训练开销优化

全对阵总训练次数 = `1 + |A| + |A|×|D|`（`Acc_clean` 与每个攻击的 `Acc_attack` 复用缓存），而非朴素的 `2×|A|×|D|`。默认 6×6 = 43 次训练。

## 扩展：自定义攻防

### 自定义攻击

在 `attacks/` 下新建 `my_attack.py`：

```python
import numpy as np
from .base import BaseAttack

class MyAttack(BaseAttack):
    name = "my_attack"
    description = "我的自定义攻击"
    params = {"seed": 0}

    def attack(self, env, task):
        images = task["images"].copy()
        labels = task["labels"].copy()
        budget = task["poison_budget"]
        # ... 你的投毒逻辑（修改不超过 budget 个样本） ...
        return {"images": images, "labels": labels}
```

注册表会自动发现。或按赛题目录规范放到 `submissions/` 下（见 `submissions/attack_template/`）。

### 自定义防御

在 `defenses/` 下新建 `my_defense.py`，或按 `submissions/defense_template/` 提交。防御输出可包含 `sample_weights` 与 `train_config`（部分覆盖训练配置，如 `label_smoothing`、`augmentation`）。

## 输出产物（`results/`）

- `summary.json`：全部对阵与指标的完整记录。
- `matrix_acc_defense.csv` / `matrix_recovery.csv` / `matrix_effective_asr.csv`：三张对阵矩阵。
- `per_matchup.csv`：每个对阵的明细。
- `leaderboard_attack.csv` / `leaderboard_defense.csv`：攻防排行榜。
- `heatmap.png` / `heatmap.html`：可视化热力图。
