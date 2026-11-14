"""Configuration loading (plain YAML + dotted command-line overrides)."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self))


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply overrides of the form ``a.b.c=value`` (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str | os.PathLike | None = None, overrides: List[str] | None = None) -> Cfg:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides or [])
    return Cfg.wrap(cfg)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def output_path(cfg: Cfg, *parts: str) -> Path:
    p = resolve(cfg.output_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def add_pystk_to_path(cfg: Cfg) -> Path:
    """Make ``pystk2_gymnasium``, ``agents``, ``main`` and ``utils`` of the
    pystk_project importable."""
    src = Path(os.environ.get("PYSTK_SRC", "")) if os.environ.get("PYSTK_SRC") else resolve(cfg.pystk_src)
    src = src.resolve()
    if not (src / "pystk2_gymnasium").is_dir():
        raise FileNotFoundError(
            f"pystk_project/src not found at {src}. Set `pystk_src` in the config "
            "or the PYSTK_SRC environment variable."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
