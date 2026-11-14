"""Generate the expert dataset D0 (shared by DAgger and SAC+symmetric sampling).

    python scripts/generate_dataset.py [--config ...] [--set expert.name=team4 ...]

If expert.name == "best", the expert benchmark is read (and run if missing).
Writes data/expert_<track>.npz and data/expert_<track>.json.
"""

import argparse
import json

import _bootstrap  # noqa: F401
from benchmark_experts import benchmark
from stkil.common import dataset_path
from stkil.config import add_pystk_to_path, load_config, output_path
from stkil.dataset import generate_dataset
from stkil.env import SoloLapEnv
from stkil.experts import ExpertPolicy


def resolve_expert(cfg) -> str:
    name = cfg.expert.name
    if name != "best":
        return name
    path = output_path(cfg, "results", "expert_benchmark.json")
    if not path.exists():
        return benchmark(cfg)
    with open(path) as f:
        return json.load(f)["best"]


def main(cfg):
    name = resolve_expert(cfg)
    src = None if name == "mock" else str(add_pystk_to_path(cfg))
    env = SoloLapEnv(cfg)
    env.reset(seed=0)
    expert = ExpertPolicy(name, env.stk_env, src, int(cfg.expert.path_lookahead))
    meta = generate_dataset(cfg, env, expert, dataset_path(cfg))
    env.close()
    print(f"dataset: {meta['n_transitions']} transitions from expert {name} -> {dataset_path(cfg)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    main(load_config(a.config, a.set))
