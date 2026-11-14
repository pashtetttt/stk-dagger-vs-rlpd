"""Watch a trained policy (or the expert) drive one lap with the STK window.

    python scripts/watch.py --model runs/dagger/seed_0/model.pt
    python scripts/watch.py --expert team4
"""

import argparse

import torch

import _bootstrap  # noqa: F401
from stkil.config import add_pystk_to_path, load_config
from stkil.env import SoloLapEnv
from stkil.evaluate import learner_act_fn, run_episode
from stkil.experts import ExpertPolicy
from stkil.features import Normalizer
from stkil.networks import Actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--expert", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    cfg = load_config(a.config, a.set + ["env.render=true"])
    env = SoloLapEnv(cfg)
    env.reset(seed=a.seed)
    if a.expert:
        ex = ExpertPolicy(a.expert, env.stk_env, str(add_pystk_to_path(cfg)), int(cfg.expert.path_lookahead))
        res = run_episode(env, ex.act, learner=False, seed=a.seed, on_reset=ex.reset)
    else:
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        norm = Normalizer(ck["norm"]["mean"], ck["norm"]["std"])
        actor = Actor(len(norm.mean), len(cfg.action_keys), cfg.network.hidden,
                      cfg.network.log_std_min, cfg.network.log_std_max)
        actor.load_state_dict(ck["actor"])

        @torch.no_grad()
        def pol(f):
            return actor.deterministic(torch.as_tensor(f)[None]).numpy()[0]

        res = run_episode(env, learner_act_fn(env, pol, norm), learner=True, seed=a.seed)
    print(res)
    env.close()


if __name__ == "__main__":
    main()
