"""数据投毒攻防对抗后端 (Data Poisoning Attack-Defense Adversarial Backend).

提供统一的分类模型训练/推理接口、攻防方法注册与自动发现、
全对阵 (Attack x Defense) 编排与指标计算。

典型用法::

    from ctf_poison import Orchestrator
    orch = Orchestrator.from_config("config.json")
    result = orch.run_all_vs_all()
    result.save("results/")
"""

from .config import Config, ModelConfig, DataConfig, load_config
from .env import Env
from .model import Model, train_model
from .metrics import (
    accuracy,
    count_modified,
    poison_rate,
    untargeted_asr,
    recovery_rate,
    effective_asr,
    attack_success,
)
from .registry import (
    BaseAttack,
    BaseDefense,
    AttackRegistry,
    DefenseRegistry,
    discover_attacks,
    discover_defenses,
    load_attack_submission,
    load_defense_submission,
)
from .pipeline import run_attack, run_defense, run_matchup, MatchupResult
from .orchestrator import Orchestrator, RunResult
from .reporting import save_results

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "load_config",
    "Env",
    "Model",
    "train_model",
    "accuracy",
    "count_modified",
    "poison_rate",
    "untargeted_asr",
    "recovery_rate",
    "effective_asr",
    "attack_success",
    "BaseAttack",
    "BaseDefense",
    "AttackRegistry",
    "DefenseRegistry",
    "discover_attacks",
    "discover_defenses",
    "load_attack_submission",
    "load_defense_submission",
    "run_attack",
    "run_defense",
    "run_matchup",
    "MatchupResult",
    "Orchestrator",
    "RunResult",
    "save_results",
]

__version__ = "0.1.0"
