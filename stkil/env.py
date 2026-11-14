"""Single-kart, single-lap SuperTuxKart environment used by every method.

``SoloLapEnv`` wraps ``pystk2_gymnasium.envs.STKRaceEnv`` with
``num_kart=1`` (the kart is alone, no opponents, no opponent items) and:

* returns the raw observation dict (for the expert) and exposes
  ``features(obs)`` (for learned policies);
* adds a time limit that produces ``truncated=True`` (never ``terminated``) so
  that SAC bootstraps correctly on time-outs;
* optionally applies the same automatic-rescue rule to every learned policy;
* reports lap statistics in ``info`` (``finished``, ``lap_steps``, ``progress``).

Reward (from pystk2_gymnasium, with a single kart the position term is 0):
    r_t = (d_t - d_{t-1}) / 10 - 0.1 + 10 * 1[finished]
i.e. progress along the track, a constant time penalty and a finishing bonus,
which is aligned with "complete one lap as fast as possible".
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .actions import check_keys, sanitize_action, vec_to_dict
from .config import Cfg, add_pystk_to_path
from .features import FeatureExtractor


class AutoRescue:
    """Calls the rescue bird when the kart has been (almost) still for
    ``patience`` consecutive steps. Used identically for all learned policies."""

    def __init__(self, cfg: Cfg):
        self.enabled = bool(cfg.enabled)
        self.warmup = int(cfg.warmup_steps)
        self.threshold = float(cfg.speed_threshold)
        self.patience = int(cfg.patience)
        self.reset()

    def reset(self):
        self.slow = 0

    def __call__(self, action: Dict, obs: Dict, t: int) -> Dict:
        if not self.enabled:
            return action
        speed = float(np.linalg.norm(np.asarray(obs["velocity"], dtype=np.float32)))
        if t >= self.warmup and speed < self.threshold:
            self.slow += 1
        else:
            self.slow = 0
        if self.slow >= self.patience:
            self.slow = 0
            action = {**action, "rescue": True}
        return action


def _make_backend(cfg: Cfg):
    if cfg.env.backend == "mock":
        from .mock_env import MockSTKRaceEnv

        return MockSTKRaceEnv(track=cfg.env.track), 7
    if cfg.env.backend != "stk":
        raise ValueError(f"Unknown backend {cfg.env.backend}")
    add_pystk_to_path(cfg)
    import pystk2
    from pystk2_gymnasium.definitions import AgentSpec
    from pystk2_gymnasium.envs import STKRaceEnv
    from pystk2_gymnasium.utils import max_enum_value

    env = STKRaceEnv(
        agent=AgentSpec(use_ai=False, rank_start=0, name="learner"),
        track=cfg.env.track,
        num_kart=1,
        laps=int(cfg.env.laps),
        render_mode="human" if cfg.env.get("render", False) else None,
    )
    return env, int(max_enum_value(pystk2.Item))


class SoloLapEnv:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.env, self.n_item_types = _make_backend(cfg)
        self.max_steps = int(cfg.env.max_steps)
        self.allow_fire = bool(cfg.env.allow_fire)
        self.action_keys = list(cfg.action_keys)
        check_keys(self.action_keys)
        self.auto_rescue = AutoRescue(cfg.env.auto_rescue)
        self.extractor: Optional[FeatureExtractor] = None
        self.t = 0
        self.obs = None

    # ------------------------------------------------------------------ utils
    @property
    def stk_env(self):
        """The underlying STKRaceEnv (given to the expert agents)."""
        return self.env

    @property
    def track_length(self) -> float:
        return float(self.env.track.length)

    def _kart(self):
        return self.env.world.karts[0]

    def progress(self) -> float:
        d = max(float(self._kart().overall_distance), 0.0)
        return float(min(d / max(self.track_length * int(self.cfg.env.laps), 1.0), 1.0))

    def features(self, obs: Dict) -> np.ndarray:
        return self.extractor(obs)

    @property
    def feature_dim(self) -> int:
        if self.extractor is None:
            raise RuntimeError("call reset() once before reading feature_dim")
        return self.extractor.dim

    # ------------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None) -> Dict:
        obs, _ = self.env.reset(seed=seed)
        if self.extractor is None:
            fc = self.cfg.features
            self.extractor = FeatureExtractor(fc.n_paths, fc.n_items, self.n_item_types, self.track_length)
        self.t = 0
        self.auto_rescue.reset()
        self.obs = obs
        return obs

    def step_dict(self, action: Dict, learner: bool) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with an STK action dict. ``learner=True`` enables auto-rescue."""
        action = sanitize_action(action, allow_fire=self.allow_fire)
        if learner:
            action = self.auto_rescue(action, self.obs, self.t)
        obs, reward, terminated, _, _ = self.env.step(action)
        self.t += 1
        truncated = (not terminated) and self.t >= self.max_steps
        finished = bool(self._kart().has_finished_race)
        info = {
            "finished": finished,
            "lap_steps": self.t if finished else None,
            "progress": 1.0 if finished else self.progress(),
            "executed_action": action,
        }
        self.obs = obs
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step_vec(self, vec: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        return self.step_dict(vec_to_dict(vec, self.action_keys, self.allow_fire), learner=True)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
