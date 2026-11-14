"""Evaluation protocol shared by all methods.

* deterministic policies (DAgger: the network output; SAC: tanh(mean));
* the same list of evaluation seeds for every method and every run
  (common random numbers -> paired comparisons are possible);
* metrics per episode: finished, lap_steps, progress, return, and the scalar
  ``score``:
      score = progress                                   if the lap is not finished
      score = 1 + (max_steps - lap_steps) / max_steps    if it is finished
  so any finisher beats any non-finisher and faster laps get higher scores.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .actions import vec_to_dict


def episode_score(finished: bool, lap_steps: Optional[int], progress: float, max_steps: int) -> float:
    if finished:
        return 1.0 + (max_steps - lap_steps) / max_steps
    return float(progress)


def run_episode(env, act: Callable[[Dict], Dict], learner: bool, seed: int,
                on_reset: Optional[Callable[[], None]] = None) -> Dict:
    obs = env.reset(seed=seed)
    if on_reset:
        on_reset()
    ret, done, info = 0.0, False, {"finished": False, "lap_steps": None, "progress": 0.0}
    while not done:
        obs, r, term, trunc, info = env.step_dict(act(obs), learner=learner)
        ret += r
        done = term or trunc
    return {
        "seed": int(seed),
        "finished": bool(info["finished"]),
        "lap_steps": info["lap_steps"],
        "progress": float(info["progress"]),
        "return": float(ret),
        "steps": int(env.t),
        "score": episode_score(info["finished"], info["lap_steps"], info["progress"], env.max_steps),
    }


def learner_act_fn(env, policy_vec: Callable[[np.ndarray], np.ndarray], normalizer) -> Callable[[Dict], Dict]:
    keys = env.action_keys

    def act(obs):
        a = policy_vec(normalizer(env.features(obs)))
        return vec_to_dict(a, keys, env.allow_fire)

    return act


def evaluate(env, act: Callable[[Dict], Dict], learner: bool, seeds: Sequence[int],
             on_reset: Optional[Callable[[], None]] = None) -> Dict:
    eps = [run_episode(env, act, learner, s, on_reset) for s in seeds]
    return summarize(eps, env.max_steps)


def summarize(eps: List[Dict], max_steps: int) -> Dict:
    fin = [e for e in eps if e["finished"]]
    return {
        "episodes": eps,
        "score": float(np.mean([e["score"] for e in eps])),
        "finish_rate": float(len(fin) / len(eps)),
        "lap_steps_finished": float(np.mean([e["lap_steps"] for e in fin])) if fin else float("nan"),
        "lap_steps_censored": float(np.mean([e["lap_steps"] if e["finished"] else max_steps for e in eps])),
        "progress": float(np.mean([e["progress"] for e in eps])),
        "return": float(np.mean([e["return"] for e in eps])),
    }


def eval_seeds(cfg, n: int, offset: int = 0) -> List[int]:
    base = int(cfg.budget.eval_seed_base)
    return [base + offset + i for i in range(n)]
