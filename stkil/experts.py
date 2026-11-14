"""Expert policies = hand-written agents from osigaud/pystk_project/src/agents.

The team agents are stateful (stuck counters, drift state machines, ...) and
their ``reset()`` method resets the environment, so we never call it: a fresh
agent instance is created at the beginning of every episode instead.

When the expert is used as a *labeller* (DAgger), it is queried at every step
of the trajectory actually followed, so its internal state follows the visited
states even when the learner's action is executed.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Dict

import numpy as np

from .actions import sanitize_action

EXPERTS = {
    "team1": ("agents.team1.agent1", "Agent1"),
    "team2": ("agents.team2.agent2", "Agent2"),
    "team3": ("agents.team3.agent3", "Agent3"),
    "team4": ("agents.team4.agent4", "Agent4"),
    "team5": ("agents.team5.agent5", "Agent5"),
}

DEFAULT_ACTION = {"acceleration": 0.0, "steer": 0.0, "brake": False, "drift": False,
                  "nitro": False, "rescue": False, "fire": False}


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _copy_obs(obs: Dict) -> Dict:
    """Defensive copy so that an agent cannot modify the observation in place."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, tuple):
            out[k] = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
        elif isinstance(v, dict):
            out[k] = _copy_obs(v)
        else:
            out[k] = v
    return out


class ExpertPolicy:
    """Wraps a team agent; ``act(obs)`` returns a sanitized STK action dict."""

    def __init__(self, name: str, stk_env, pystk_src: str | None, path_lookahead: int = 3):
        self.name = name
        self.stk_env = stk_env
        self.path_lookahead = path_lookahead
        self.n_errors = 0
        if name == "mock":
            from .mock_env import MockExpert

            self._factory = lambda: MockExpert()
            self._cwd = None
        else:
            if name not in EXPERTS:
                raise ValueError(f"Unknown expert {name}; choose among {list(EXPERTS)}")
            # Some agents load their YAML config relative to src/main at import time
            self._cwd = os.path.join(pystk_src, "main")
            module, cls = EXPERTS[name]
            with _cwd(self._cwd):
                self._cls = getattr(importlib.import_module(module), cls)
            self._factory = lambda: self._cls(self.stk_env, path_lookahead=self.path_lookahead)
        self.agent = None
        self.reset()

    def reset(self):
        if self._cwd:
            with _cwd(self._cwd):
                self.agent = self._factory()
        else:
            self.agent = self._factory()

    def act(self, obs: Dict) -> Dict:
        try:
            action = self.agent.choose_action(_copy_obs(obs))
        except Exception as e:  # an expert crash must not kill a 1-hour run
            self.n_errors += 1
            if self.n_errors <= 5:
                print(f"[expert {self.name}] error: {e!r} -> default action")
            action = DEFAULT_ACTION
        return sanitize_action(action, allow_fire=True)
