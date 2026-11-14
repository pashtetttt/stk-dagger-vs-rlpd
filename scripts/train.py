"""Train one method with one seed.

    python scripts/train.py --algo dagger --seed 0
    python scripts/train.py --algo rlpd   --seed 0
    python scripts/train.py --algo sac    --seed 0      # control: no prior data

Results go to runs/<algo>/seed_<seed>/ (config.json, eval.csv, evals.jsonl,
final.json, model.pt).
"""

import argparse

import _bootstrap  # noqa: F401
from stkil.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["dagger", "rlpd", "sac"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)

    import torch

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    if a.algo == "dagger":
        from stkil.dagger import train_dagger

        train_dagger(cfg, a.seed)
    else:
        from stkil.rlpd import train_rlpd

        train_rlpd(cfg, a.seed, algo=a.algo)


if __name__ == "__main__":
    main()
