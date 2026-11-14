"""SAC with RLPD-style symmetric sampling (Ball, Smith, Kostrikov & Levine, 2023).

Every gradient batch of size B is made of
    round(B * offline_ratio) transitions sampled uniformly from the expert
    dataset D0 and the rest sampled uniformly from the online replay buffer
(offline_ratio = 0.5 -> "symmetric sampling"). Other RLPD ingredients kept:
LayerNorm in the critics, an ensemble of critics with a min over a random
subset for the target, no offline pre-training, several critic updates per
env step (UTD), target entropy = -0.5 * |A|. With offline_ratio = 0 and some
uniform-random warm-up steps this is plain SAC from scratch (control).

Time-outs are truncations: the target bootstraps through them; only real
terminations (lap finished) stop the bootstrap.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .common import RunLogger, dataset_path, make_evaluator, run_dir, set_seed
from .config import Cfg
from .dataset import load_dataset
from .env import SoloLapEnv
from .networks import Actor, EnsembleCritic


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.o = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, act_dim), np.float32)
        self.r = np.zeros((capacity,), np.float32)
        self.o2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros((capacity,), np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, o, a, r, o2, d):
        for x in zip(o, a, r, o2, d):
            self.add(*x)

    def sample_idx(self, rng, n):
        return rng.randint(0, self.size, size=n)


def rl_config(cfg: Cfg, algo: str) -> dict:
    rc = dict(cfg.rlpd)
    if algo == "sac":
        rc.update(dict(cfg.sac))
    return rc


def train_rlpd(cfg: Cfg, seed: int, algo: str = "rlpd"):
    set_seed(seed)
    rc = rl_config(cfg, algo)
    device = torch.device(cfg.experiment.device)
    out = run_dir(cfg, algo, seed)
    logger = RunLogger(out, cfg, algo, seed)

    data, norm = load_dataset(dataset_path(cfg))
    env, eval_env = SoloLapEnv(cfg), SoloLapEnv(cfg)
    keys = list(cfg.action_keys)
    obs_dim, act_dim = data["obs"].shape[1], len(keys)

    offline = ReplayBuffer(len(data["reward"]), obs_dim, act_dim)
    offline.add_batch(norm(data["obs"]), data["act_exec"], data["reward"], norm(data["next_obs"]),
                      data["terminated"].astype(np.float32))
    online = ReplayBuffer(int(rc["buffer_size"]), obs_dim, act_dim)

    actor = Actor(obs_dim, act_dim, cfg.network.hidden, cfg.network.log_std_min, cfg.network.log_std_max).to(device)
    critic = EnsembleCritic(obs_dim, act_dim, cfg.network.critic_hidden, int(rc["n_critics"]),
                            bool(cfg.network.critic_layernorm)).to(device)
    critic_t = copy.deepcopy(critic).requires_grad_(False)
    log_alpha = torch.tensor(float(np.log(float(rc["init_alpha"]))), dtype=torch.float32,
                             device=device, requires_grad=True)
    target_entropy = -float(rc["target_entropy_scale"]) * act_dim
    opt_a = torch.optim.Adam(actor.parameters(), lr=float(rc["actor_lr"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(rc["critic_lr"]))
    opt_al = torch.optim.Adam([log_alpha], lr=float(rc["alpha_lr"]))

    gamma, tau = float(rc["gamma"]), float(rc["tau"])
    B, utd = int(rc["batch_size"]), int(rc["utd"])
    ratio = float(rc["offline_ratio"])
    n_off = int(round(B * ratio)) if ratio > 0 else 0
    n_on = B - n_off
    m = int(rc["critic_subset"])
    n_crit = int(rc["n_critics"])
    rng = np.random.RandomState(seed)

    def batch():
        parts = []
        if n_off:
            i = offline.sample_idx(rng, n_off)
            parts.append((offline.o[i], offline.a[i], offline.r[i], offline.o2[i], offline.d[i]))
        if n_on:
            i = online.sample_idx(rng, n_on)
            parts.append((online.o[i], online.a[i], online.r[i], online.o2[i], online.d[i]))
        return [torch.as_tensor(np.concatenate(x), device=device) for x in zip(*parts)]

    stats = {}

    def update_critic(o, a, r, o2, d):
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = actor.sample(o2)
            q_t = critic_t(o2, a2)                                     # (n, B)
            sub = torch.as_tensor(rng.choice(n_crit, size=m, replace=False), device=device)
            q_t = q_t[sub].min(dim=0).values
            y = r + gamma * (1.0 - d) * (q_t - alpha * logp2)
        q = critic(o, a)
        loss = ((q - y[None]) ** 2).mean(dim=1).sum()
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_t.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        stats["critic_loss"] = loss.item()
        stats["q_mean"] = q.mean().item()

    def update_actor(o):
        a, logp = actor.sample(o)
        q = critic(o, a).mean(dim=0)
        alpha = log_alpha.exp()
        loss = (alpha.detach() * logp - q).mean()
        opt_a.zero_grad()
        loss.backward()
        opt_a.step()
        al_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
        opt_al.zero_grad()
        al_loss.backward()
        opt_al.step()
        stats["actor_loss"] = loss.item()
        stats["alpha"] = log_alpha.exp().item()
        stats["entropy"] = -logp.mean().item()

    @torch.no_grad()
    def policy_vec(f):
        return actor.deterministic(torch.as_tensor(f, device=device)[None]).cpu().numpy()[0]

    @torch.no_grad()
    def explore_vec(f):
        a, _ = actor.sample(torch.as_tensor(f, device=device)[None])
        return a.cpu().numpy()[0]

    eval_fn = make_evaluator(cfg, eval_env, norm)
    total = int(cfg.budget.online_env_steps)
    eval_every = int(cfg.budget.eval_every)
    random_steps = int(rc["random_steps"])
    learning_starts = max(int(rc["learning_starts"]), n_on)
    grad_steps, episode, ep_ret, ep_info = 0, 0, 0.0, []
    train_seed = lambda ep: 10_000_000 + seed * 100_000 + ep  # noqa: E731

    logger.log_eval(0, 0, eval_fn(policy_vec), {"offline_size": offline.size})
    obs = env.reset(seed=train_seed(episode))
    f = norm(env.features(obs))
    for t in range(1, total + 1):
        a = rng.uniform(-1, 1, size=act_dim).astype(np.float32) if t <= random_steps else explore_vec(f)
        obs, r, term, trunc, info = env.step_vec(a)
        f2 = norm(env.features(obs))
        online.add(f, a, r, f2, float(term))
        ep_ret += r
        f = f2
        if term or trunc:
            ep_info.append((info["finished"], info["lap_steps"], ep_ret))
            episode += 1
            ep_ret = 0.0
            obs = env.reset(seed=train_seed(episode))
            f = norm(env.features(obs))

        if online.size >= learning_starts:
            for _ in range(utd):
                o, a_, r_, o2, d_ = batch()
                update_critic(o, a_, r_, o2, d_)
                grad_steps += 1
            update_actor(batch()[0])

        if t % eval_every == 0:
            recent = ep_info[-10:]
            extra = {**{k: round(v, 4) for k, v in stats.items()}, "train_episodes": episode,
                     "train_finish_rate_last10": float(np.mean([e[0] for e in recent])) if recent else None}
            logger.log_eval(t, grad_steps, eval_fn(policy_vec), extra)

    final = eval_fn(policy_vec, final=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "norm": norm.state_dict()},
               out / "model.pt")
    logger.finish(final, {"env_steps": total, "grad_steps": grad_steps, "train_episodes": episode})
    env.close()
    eval_env.close()
    return final
