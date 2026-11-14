"""Full pipeline (resumable): expert benchmark -> dataset -> all runs -> analysis.

    python scripts/run_all.py                       # default study (5 seeds, dagger + rlpd)
    python scripts/run_all.py --with-sac            # + SAC-from-scratch control
    python scripts/run_all.py --jobs 3              # runs in parallel processes
    python scripts/run_all.py --set budget.online_env_steps=100000 experiment.seeds=[0,1,2,3,4,5,6,7,8,9]

A run whose runs/<algo>/seed_<k>/final.json already exists is skipped, so the
command can be relaunched after an interruption.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from stkil.common import dataset_path, run_dir
from stkil.config import load_config, output_path

HERE = Path(__file__).resolve().parent


def sh(args, log: Path | None = None):
    print("$", " ".join(args), flush=True)
    if log is None:
        subprocess.run(args, check=True)
        return 0
    with open(log, "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"!! failed (see {log})", flush=True)
    else:
        print(f"ok: {log.parent}", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--with-sac", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    py = sys.executable
    common = (["--config", a.config] if a.config else []) + (["--set", *a.set] if a.set else [])

    if not output_path(cfg, "results", "expert_benchmark.json").exists() and cfg.expert.name == "best":
        sh([py, str(HERE / "benchmark_experts.py"), *common])
    if not dataset_path(cfg).exists():
        sh([py, str(HERE / "generate_dataset.py"), *common])

    algos = list(cfg.experiment.algos) + (["sac"] if a.with_sac and "sac" not in cfg.experiment.algos else [])
    todo = []
    for seed in cfg.experiment.seeds:          # interleave algos so partial results stay balanced
        for algo in algos:
            d = run_dir(cfg, algo, seed)
            if not (d / "final.json").exists():
                todo.append((algo, seed, d))
    print(f"{len(todo)} runs to do: {[(x, s) for x, s, _ in todo]}")

    threads = ["--threads", "1"] if a.jobs > 1 else []
    if a.jobs > 1:  # inherited by the child processes
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    def one(job):
        algo, seed, d = job
        return sh([py, str(HERE / "train.py"), "--algo", algo, "--seed", str(seed), *threads, *common],
                  log=d / "train.log")

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        codes = list(ex.map(one, todo))
    if any(codes):
        print(f"{sum(c != 0 for c in codes)} run(s) failed; relaunch to retry them.")

    if not a.skip_analysis:
        sh([py, str(HERE / "analyze.py"), *common])


if __name__ == "__main__":
    main()
