"""Expert dataset D0, shared by DAgger (iteration 0 = behaviour cloning) and by
SAC + symmetric sampling (offline half of every batch).

For every step we store
    obs (raw features), act_exec (executed action, projected on [-1,1]^d),
    act_expert (clean expert label), reward, next_obs, terminated, truncated, episode
A fraction of the episodes is executed with exploration noise to widen the state
distribution; the expert label stays clean (it is the expert's choice in the
visited state), so BC always imitates the clean expert while the RL transitions
stay consistent with what was really executed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from .actions import BINARY, dict_to_vec
from .config import Cfg
from .evaluate import episode_score
from .features import Normalizer


def _noisy(action: Dict, rng: np.random.RandomState, dc: Cfg, keys) -> Dict:
    a = dict(action)
    a["steer"] = float(np.clip(a["steer"] + rng.normal(0, dc.noise_steer_std), -1, 1))
    a["acceleration"] = float(np.clip(a["acceleration"] + rng.normal(0, dc.noise_accel_std), 0, 1))
    for k in keys:
        if k in BINARY and rng.rand() < dc.noise_binary_flip:
            a[k] = not a[k]
    return a


def generate_dataset(cfg: Cfg, env, expert, out_path: Path) -> Dict:
    dc = cfg.dataset
    keys = list(cfg.action_keys)
    rng = np.random.RandomState(int(dc.seed))
    n_eps = int(dc.n_episodes)
    n_noisy = int(round(n_eps * float(dc.noisy_fraction)))
    buf = {k: [] for k in ["obs", "act_exec", "act_expert", "reward", "next_obs", "terminated", "truncated", "episode"]}
    ep_stats = []
    t0 = time.time()
    for ep in range(n_eps):
        noisy = ep >= n_eps - n_noisy
        obs = env.reset(seed=int(dc.seed) + ep)
        expert.reset()
        f = env.features(obs)
        done, ret, info = False, 0.0, {}
        while not done:
            a_exp = expert.act(obs)
            a_exec = _noisy(a_exp, rng, dc, keys) if noisy else a_exp
            obs, r, term, trunc, info = env.step_dict(a_exec, learner=False)
            f2 = env.features(obs)
            buf["obs"].append(f)
            buf["act_exec"].append(dict_to_vec(info["executed_action"], keys))
            buf["act_expert"].append(dict_to_vec(a_exp, keys))
            buf["reward"].append(r)
            buf["next_obs"].append(f2)
            buf["terminated"].append(term)
            buf["truncated"].append(trunc)
            buf["episode"].append(ep)
            f, ret, done = f2, ret + r, term or trunc
        ep_stats.append({"episode": ep, "noisy": noisy, "finished": info["finished"],
                         "lap_steps": info["lap_steps"], "progress": info["progress"], "return": ret,
                         "score": episode_score(info["finished"], info["lap_steps"], info["progress"], env.max_steps)})
        print(f"[dataset] ep {ep:3d} noisy={noisy!s:5} finished={info['finished']!s:5} "
              f"lap_steps={info['lap_steps']} progress={info['progress']:.3f} return={ret:.1f}")
    data = {k: np.asarray(v) for k, v in buf.items()}
    for k in ("obs", "next_obs", "act_exec", "act_expert", "reward"):
        data[k] = data[k].astype(np.float32)
    norm = Normalizer.fit(data["obs"])
    meta = {"expert": expert.name, "n_transitions": int(len(data["reward"])), "episodes": ep_stats,
            "feature_dim": int(data["obs"].shape[1]), "action_keys": keys,
            "track": cfg.env.track, "wall_time_s": time.time() - t0,
            "expert_errors": expert.n_errors}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data, norm_mean=norm.mean, norm_std=norm.std)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_dataset(path: Path):
    z = np.load(path)
    data = {k: z[k] for k in z.files if not k.startswith("norm_")}
    norm = Normalizer(z["norm_mean"], z["norm_std"])
    return data, norm
