"""DAgger (Ross, Gordon & Bagnell, 2011) with the expert dataset D0 as
iteration 0.

    D <- D0 (expert states, clean expert labels)
    pi_0 <- BC(D)                                     # evaluated at 0 online steps
    for i = 1..N:
        beta_i = beta0 * decay^(i-1)
        run pi_mix = beta_i * expert + (1 - beta_i) * pi_{i-1}   (per-step mixture)
            for T/N env steps and label EVERY visited state with the expert
        D <- D U {(s, expert(s))}
        pi_i <- train on D (warm start, fixed number of gradient steps)
        evaluate pi_i (deterministic)

The loss is the mean-squared error between tanh(mean) of the shared Actor and
the expert action projected on [-1, 1]^d (binary targets are +/-0.95).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .actions import dict_to_vec
from .common import RunLogger, dataset_path, make_evaluator, run_dir, set_seed
from .config import Cfg, add_pystk_to_path
from .dataset import load_dataset
from .env import SoloLapEnv
from .experts import ExpertPolicy
from .networks import Actor


def resolve_expert_name(cfg: Cfg) -> str:
    import json

    meta = dataset_path(cfg).with_suffix(".json")
    with open(meta) as f:
        return json.load(f)["expert"]


def make_expert(cfg: Cfg, env: SoloLapEnv, name: str) -> ExpertPolicy:
    src = None if name == "mock" else str(add_pystk_to_path(cfg))
    return ExpertPolicy(name, env.stk_env, src, int(cfg.expert.path_lookahead))


def train_dagger(cfg: Cfg, seed: int, algo: str = "dagger"):
    set_seed(seed)
    dc = cfg.dagger
    device = torch.device(cfg.experiment.device)
    out = run_dir(cfg, algo, seed)
    logger = RunLogger(out, cfg, algo, seed)

    data, norm = load_dataset(dataset_path(cfg))
    env, eval_env = SoloLapEnv(cfg), SoloLapEnv(cfg)
    env.reset(seed=0)  # builds the feature extractor
    expert = make_expert(cfg, env, resolve_expert_name(cfg))
    keys = list(cfg.action_keys)
    obs_dim, act_dim = data["obs"].shape[1], len(keys)

    X = [norm(data["obs"])]
    Y = [data["act_expert"].astype(np.float32)]

    def new_actor():
        return Actor(obs_dim, act_dim, cfg.network.hidden, cfg.network.log_std_min,
                     cfg.network.log_std_max).to(device)

    actor = new_actor()
    opt = torch.optim.Adam(actor.parameters(), lr=float(dc.lr), weight_decay=float(dc.weight_decay))
    rng = np.random.RandomState(seed)
    grad_steps = 0

    def fit(n_steps: int) -> float:
        nonlocal grad_steps
        Xa = torch.as_tensor(np.concatenate(X), device=device)
        Ya = torch.as_tensor(np.concatenate(Y), device=device)
        losses = []
        actor.train()
        for _ in range(int(n_steps)):
            idx = torch.as_tensor(rng.randint(0, len(Xa), size=int(dc.batch_size)), device=device)
            loss = ((actor.deterministic(Xa[idx]) - Ya[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            grad_steps += 1
        return float(np.mean(losses[-200:]))

    @torch.no_grad()
    def policy_vec(f):
        actor.eval()
        return actor.deterministic(torch.as_tensor(f, device=device)[None]).cpu().numpy()[0]

    eval_fn = make_evaluator(cfg, eval_env, norm)

    # ---- iteration 0: behaviour cloning on D0 (0 online env steps) -------------
    loss = fit(dc.bc_init_steps)
    logger.log_eval(0, grad_steps, eval_fn(policy_vec),
                    {"iteration": 0, "beta": None, "bc_loss": loss, "dataset": int(len(Y[0]))})
    # BC baseline evaluated with the final protocol (same seeds as every final eval)
    bc_final = eval_fn(policy_vec, final=True)
    torch.save({"actor": actor.state_dict(), "norm": norm.state_dict()}, out / "model_bc.pt")

    n_iter = int(dc.n_iterations)
    steps_per_iter = int(cfg.budget.online_env_steps) // n_iter
    env_steps, episode = 0, 0
    train_seed = lambda ep: 10_000_000 + seed * 100_000 + ep  # noqa: E731
    obs = env.reset(seed=train_seed(episode))
    expert.reset()
    for it in range(1, n_iter + 1):
        beta = float(dc.beta0) * float(dc.beta_decay) ** (it - 1)
        newX, newY, n_expert_exec = [], [], 0
        t_it = time.time()
        for _ in range(steps_per_iter):
            a_exp = expert.act(obs)          # expert queried at every visited state
            f = env.features(obs)
            newX.append(norm(f))
            newY.append(dict_to_vec(a_exp, keys))
            if rng.rand() < beta:
                obs, _, term, trunc, _ = env.step_dict(a_exp, learner=False)
                n_expert_exec += 1
            else:
                obs, _, term, trunc, _ = env.step_vec(policy_vec(norm(f)))
            env_steps += 1
            if term or trunc:
                episode += 1
                obs = env.reset(seed=train_seed(episode))
                expert.reset()
        X.append(np.asarray(newX, dtype=np.float32))
        Y.append(np.asarray(newY, dtype=np.float32))
        if dc.retrain_from_scratch:
            actor = new_actor()
            opt = torch.optim.Adam(actor.parameters(), lr=float(dc.lr), weight_decay=float(dc.weight_decay))
            loss = fit(dc.bc_init_steps)
        else:
            loss = fit(dc.grad_steps_per_iter)
        logger.log_eval(env_steps, grad_steps, eval_fn(policy_vec),
                        {"iteration": it, "beta": beta, "bc_loss": loss,
                         "dataset": int(sum(len(y) for y in Y)),
                         "expert_exec_frac": n_expert_exec / steps_per_iter,
                         "collect_time_s": round(time.time() - t_it, 1)})

    final = eval_fn(policy_vec, final=True)
    torch.save({"actor": actor.state_dict(), "norm": norm.state_dict()}, out / "model.pt")
    logger.finish(final, {"env_steps": env_steps, "grad_steps": grad_steps,
                          "expert_errors": expert.n_errors, "bc_final": bc_final})
    env.close()
    eval_env.close()
    return final
