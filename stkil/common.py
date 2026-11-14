"""Helpers shared by the training scripts: seeding, run directories, logging,
evaluation bookkeeping."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from .config import Cfg, output_path
from .evaluate import eval_seeds, evaluate, learner_act_fn

EVAL_FIELDS = ["env_steps", "grad_steps", "wall_time_s", "score", "finish_rate",
               "lap_steps_finished", "lap_steps_censored", "progress", "return", "extra"]


def versions() -> Dict[str, str]:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ["torch", "numpy", "gymnasium", "pysupertuxkart2", "scipy"]:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "n/a"
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def dataset_path(cfg: Cfg) -> Path:
    return output_path(cfg, "data", f"expert_{cfg.env.track}.npz")


def run_dir(cfg: Cfg, algo: str, seed: int) -> Path:
    d = output_path(cfg, "runs", algo, f"seed_{seed}", "x").parent
    d.mkdir(parents=True, exist_ok=True)
    return d


class RunLogger:
    def __init__(self, path: Path, cfg: Cfg, algo: str, seed: int):
        self.path = path
        self.t0 = time.time()
        with open(path / "config.json", "w") as f:
            json.dump({"algo": algo, "seed": seed, "versions": versions(), **cfg.to_dict()}, f, indent=2)
        self.csv_f = open(path / "eval.csv", "w", newline="")
        self.csv = csv.DictWriter(self.csv_f, fieldnames=EVAL_FIELDS)
        self.csv.writeheader()
        self.jsonl = open(path / "evals.jsonl", "w")

    def log_eval(self, env_steps: int, grad_steps: int, res: Dict, extra: Dict | None = None):
        row = {"env_steps": env_steps, "grad_steps": grad_steps, "wall_time_s": round(time.time() - self.t0, 1),
               **{k: res[k] for k in EVAL_FIELDS if k in res}, "extra": json.dumps(extra or {})}
        self.csv.writerow(row)
        self.csv_f.flush()
        self.jsonl.write(json.dumps({**row, "episodes": res["episodes"]}) + "\n")
        self.jsonl.flush()
        print(f"[eval] steps={env_steps:7d} score={res['score']:.3f} finish={res['finish_rate']:.2f} "
              f"lap={res['lap_steps_finished']:.1f} progress={res['progress']:.3f} "
              f"t={row['wall_time_s']}s {extra or ''}")

    def finish(self, final: Dict, extra: Dict | None = None):
        with open(self.path / "final.json", "w") as f:
            json.dump({**final, "wall_time_s": time.time() - self.t0, **(extra or {})}, f, indent=2)
        self.csv_f.close()
        self.jsonl.close()


def make_evaluator(cfg: Cfg, eval_env, normalizer) -> Callable:
    """Returns eval_fn(policy_vec, final=False) using the shared seed lists."""
    mid = eval_seeds(cfg, int(cfg.budget.eval_episodes), offset=0)
    fin = eval_seeds(cfg, int(cfg.budget.final_eval_episodes), offset=1000)

    def eval_fn(policy_vec, final: bool = False):
        act = learner_act_fn(eval_env, policy_vec, normalizer)
        return evaluate(eval_env, act, learner=True, seeds=fin if final else mid)

    return eval_fn
