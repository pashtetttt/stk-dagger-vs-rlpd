"""Benchmark the candidate expert agents alone on the track and pick the fastest.

    python scripts/benchmark_experts.py [--config configs/default.yaml] [--set key=value ...]

Writes results/expert_benchmark.json.
"""

import argparse
import json
import traceback

import _bootstrap  # noqa: F401
from stkil.config import add_pystk_to_path, load_config, output_path
from stkil.env import SoloLapEnv
from stkil.evaluate import eval_seeds, evaluate
from stkil.experts import ExpertPolicy


def benchmark(cfg, candidates=None):
    candidates = candidates or list(cfg.expert.candidates)
    src = None if cfg.env.backend == "mock" else str(add_pystk_to_path(cfg))
    env = SoloLapEnv(cfg)
    env.reset(seed=0)
    seeds = eval_seeds(cfg, int(cfg.expert.benchmark_episodes), offset=5000)
    results = {}
    for name in candidates:
        try:
            expert = ExpertPolicy(name, env.stk_env, src, int(cfg.expert.path_lookahead))
            res = evaluate(env, expert.act, learner=False, seeds=seeds, on_reset=expert.reset)
            res["errors"] = expert.n_errors
            results[name] = res
            print(f"{name}: score={res['score']:.3f} finish={res['finish_rate']:.2f} "
                  f"lap_steps={res['lap_steps_finished']:.1f} errors={expert.n_errors}")
        except Exception as e:
            traceback.print_exc()
            results[name] = {"error": repr(e), "score": float("-inf")}
    env.close()
    best = max(results, key=lambda k: results[k]["score"])
    out = {"track": cfg.env.track, "seeds": seeds, "best": best, "results": results}
    path = output_path(cfg, "results", "expert_benchmark.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"best expert: {best}  -> {path}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--candidates", nargs="*", default=None)
    a = p.parse_args()
    benchmark(load_config(a.config, a.set), a.candidates)
