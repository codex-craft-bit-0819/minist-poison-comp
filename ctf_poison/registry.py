"""攻防方法基类、注册表与自动发现。

- 内置方法放在 ``attacks/`` 与 ``defenses/`` 包下，继承 BaseAttack/BaseDefense，
  注册表会自动扫描发现。
- 选手提交（``attack.py`` / ``defense.py`` 中的裸函数）通过
  ``load_attack_submission`` / ``load_defense_submission`` 加载并包装。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
from typing import Callable

import numpy as np

from .utils import get_logger

log = get_logger("registry")


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class BaseAttack:
    """攻击方法基类。

    子类需设置 ``name`` 并实现 :meth:`attack`。``__init__`` 接收任意参数。
    """

    name: str = "base"
    description: str = ""
    params: dict = {}

    def __init__(self, **params):
        self.params = {**self.__class__.params, **params}

    def attack(self, env, task: dict) -> dict:
        raise NotImplementedError

    def __call__(self, env, task: dict) -> dict:
        return self.attack(env, task)

    def __repr__(self):
        return f"<Attack {self.name} params={self.params}>"


class BaseDefense:
    """防御方法基类。

    子类需设置 ``name`` 并实现 :meth:`defend`。返回 dict 可包含::

        {"images": ..., "labels": ..., "sample_weights": None, "train_config": None}
    """

    name: str = "base"
    description: str = ""
    params: dict = {}

    def __init__(self, **params):
        self.params = {**self.__class__.params, **params}

    def defend(self, env, train: dict) -> dict:
        raise NotImplementedError

    def __call__(self, env, train: dict) -> dict:
        return self.defend(env, train)

    def __repr__(self):
        return f"<Defense {self.name} params={self.params}>"


# --------------------------------------------------------------------------- #
# 函数适配器（包装选手提交的裸函数）
# --------------------------------------------------------------------------- #
class FunctionAttack(BaseAttack):
    """把 ``attack(env, task)`` 裸函数包装成 BaseAttack。"""

    def __init__(self, fn: Callable, name: str = "user_attack", description: str = ""):
        super().__init__()
        self._fn = fn
        self.name = name
        self.description = description or (fn.__doc__ or "").strip()

    def attack(self, env, task):
        out = self._fn(env, task)
        return _normalize_attack_output(task, out)


class FunctionDefense(BaseDefense):
    """把 ``defend(env, train)`` 裸函数包装成 BaseDefense。"""

    def __init__(self, fn: Callable, name: str = "user_defense", description: str = ""):
        super().__init__()
        self._fn = fn
        self.name = name
        self.description = description or (fn.__doc__ or "").strip()

    def defend(self, env, train):
        out = self._fn(env, train)
        return _normalize_defense_output(out)


def _normalize_attack_output(task: dict, out: dict) -> dict:
    if not isinstance(out, dict) or "images" not in out or "labels" not in out:
        raise ValueError("attack 必须返回 {'images':..., 'labels':...}")
    images = np.asarray(out["images"], dtype=np.float32)
    labels = np.asarray(out["labels"]).astype(task["labels"].dtype, copy=False)
    if images.ndim == 2:
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim == 3:
        images = images[:, None, :, :]
    return {"images": images, "labels": labels}


def _normalize_defense_output(out: dict) -> dict:
    if not isinstance(out, dict) or "images" not in out or "labels" not in out:
        raise ValueError("defend 必须返回 {'images':..., 'labels':...}")
    images = np.asarray(out["images"], dtype=np.float32)
    labels = np.asarray(out["labels"]).astype(np.int64)
    if images.ndim == 2:
        images = images.reshape(images.shape[0], 1, 28, 28)
    elif images.ndim == 3:
        images = images[:, None, :, :]
    result = {"images": images, "labels": labels}
    result["sample_weights"] = out.get("sample_weights")
    result["train_config"] = out.get("train_config")
    return result


# --------------------------------------------------------------------------- #
# 注册表与自动发现
# --------------------------------------------------------------------------- #
def _all_subclasses(cls):
    seen = set()
    stack = [cls]
    while stack:
        c = stack.pop()
        for sub in c.__subclasses__():
            if sub not in seen and sub.name != "base":
                seen.add(sub)
                stack.append(sub)
    return list(seen)


class _Registry:
    def __init__(self, base_cls, package_name, label):
        self.base_cls = base_cls
        self.package_name = package_name
        self.label = label
        self._extra: dict = {}
        self._discovered = False
        self._cache: dict = {}

    def _discover(self):
        try:
            pkg = importlib.import_module(self.package_name)
        except ImportError:
            self._discovered = True
            return
        for m in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module(f"{self.package_name}.{m.name}")
            except Exception as e:
                log.warning("导入 %s.%s 失败: %s", self.package_name, m.name, e)
        self._discovered = True

    def register(self, cls_or_obj, name: str | None = None):
        if isinstance(cls_or_obj, type):
            obj = cls_or_obj()
        else:
            obj = cls_or_obj
        nm = name or obj.name
        self._extra[nm] = obj
        self._cache[nm] = obj

    def all(self) -> dict:
        if not self._discovered:
            self._discover()
        result = dict(self._extra)
        for sub in _all_subclasses(self.base_cls):
            result[sub.name] = sub()
        self._cache = result
        return result

    def get(self, name: str):
        if name not in self._cache:
            self.all()
        if name in self._cache:
            return self._cache[name]
        raise KeyError(f"未找到 {self.label}: {name}")

    def names(self) -> list:
        return sorted(self.all().keys())


AttackRegistry = _Registry(BaseAttack, "attacks", "攻击方法")
DefenseRegistry = _Registry(BaseDefense, "defenses", "防御方法")


def discover_attacks() -> dict:
    return AttackRegistry.all()


def discover_defenses() -> dict:
    return DefenseRegistry.all()


# --------------------------------------------------------------------------- #
# 外部选手提交加载
# --------------------------------------------------------------------------- #
def _import_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_attack_submission(dir_path: str, name: str | None = None) -> BaseAttack:
    """从提交目录加载 attack.py（裸函数式或含 BaseAttack 子类）。"""
    attack_py = os.path.join(dir_path, "attack.py")
    if not os.path.exists(attack_py):
        attack_py = dir_path
    mod = _import_file("user_attack_" + os.path.basename(dir_path), attack_py)
    if hasattr(mod, "attack") and callable(mod.attack):
        nm = name or getattr(mod, "NAME", os.path.basename(os.path.dirname(dir_path) or dir_path))
        return FunctionAttack(mod.attack, name=nm)
    raise ValueError(f"{attack_py} 中未找到 attack(env, task) 函数")


def load_defense_submission(dir_path: str, name: str | None = None) -> BaseDefense:
    """从提交目录加载 defense.py（裸函数式或含 BaseDefense 子类）。"""
    defense_py = os.path.join(dir_path, "defense.py")
    if not os.path.exists(defense_py):
        defense_py = dir_path
    mod = _import_file("user_defense_" + os.path.basename(dir_path), defense_py)
    if hasattr(mod, "defend") and callable(mod.defend):
        nm = name or getattr(mod, "NAME", os.path.basename(os.path.dirname(dir_path) or dir_path))
        return FunctionDefense(mod.defend, name=nm)
    raise ValueError(f"{defense_py} 中未找到 defend(env, train) 函数")
