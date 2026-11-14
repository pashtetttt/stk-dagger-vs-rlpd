"""Shared action representation.

Learned policies (DAgger and SAC/RLPD) output a vector ``a in [-1, 1]^d`` whose
components are listed in ``cfg.action_keys``:

* ``steer``        -> steer = a                      (in [-1, 1])
* ``acceleration`` -> acceleration = (a + 1) / 2     (in [0, 1])
* binary keys (``brake``, ``drift``, ``nitro``, ``rescue``, ``fire``) -> a > 0

Expert actions (dicts produced by the team agents) are projected onto the same
vector space, binary values being mapped to +1 / -1. This projection is used for
the DAgger labels and for the actions stored in the offline RL dataset.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

CONTINUOUS = ("steer", "acceleration")
BINARY = ("brake", "drift", "nitro", "rescue", "fire")
ALL_KEYS = CONTINUOUS + BINARY

# Value used for binary targets: tanh outputs can only approach +/-1
BINARY_TARGET = 0.95


def _scalar(x) -> float:
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


def sanitize_action(action: Dict, allow_fire: bool = False) -> Dict:
    """Turn any agent output into a well-formed STK action dict."""
    out = {
        "acceleration": float(np.clip(_scalar(action.get("acceleration", 0.0)), 0.0, 1.0)),
        "steer": float(np.clip(_scalar(action.get("steer", 0.0)), -1.0, 1.0)),
    }
    for k in BINARY:
        out[k] = bool(_scalar(action.get(k, 0)) > 0)
    if not allow_fire:
        out["fire"] = False
    return out


def dict_to_vec(action: Dict, keys: Sequence[str]) -> np.ndarray:
    """Project an STK action dict onto the learner's [-1, 1]^d space."""
    a = sanitize_action(action, allow_fire=True)
    v = np.zeros(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        if k == "steer":
            v[i] = a["steer"]
        elif k == "acceleration":
            v[i] = 2.0 * a["acceleration"] - 1.0
        elif k in BINARY:
            v[i] = BINARY_TARGET if a[k] else -BINARY_TARGET
        else:
            raise KeyError(f"Unknown action key {k}")
    return np.clip(v, -1.0, 1.0)


def vec_to_dict(vec: np.ndarray, keys: Sequence[str], allow_fire: bool = False) -> Dict:
    """Map a learner action vector to an STK action dict."""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    assert len(vec) == len(keys), (vec.shape, keys)
    out = {"acceleration": 0.0, "steer": 0.0, **{k: False for k in BINARY}}
    for v, k in zip(vec, keys):
        v = float(np.clip(v, -1.0, 1.0))
        if k == "steer":
            out["steer"] = v
        elif k == "acceleration":
            out["acceleration"] = (v + 1.0) / 2.0
        else:
            out[k] = v > 0.0
    if not allow_fire:
        out["fire"] = False
    return out


def check_keys(keys: List[str]) -> None:
    unknown = [k for k in keys if k not in ALL_KEYS]
    if unknown:
        raise ValueError(f"Unknown action keys {unknown}; allowed: {ALL_KEYS}")
    if "steer" not in keys or "acceleration" not in keys:
        raise ValueError("action_keys must contain steer and acceleration")
