"""Fixed-size observation features shared by all learned policies.

The raw pystk2 observation is a dict with variable-length sequences (paths,
items, karts). The expert agents consume this raw dict; learned policies consume
the vector below. All 3D vectors are expressed in the kart frame
(x: left, y: up, z: front), as provided by ``pystk2_gymnasium``.

Layout (D = 9 + 7 * n_paths + (3 + n_item_types) * n_items):
    velocity (3) | center_path (3) | center_path_distance (1)
    | max_steer_angle (1) | lap progress (1)
    | for each of the n_paths nearest path segments: start (3), end (3), width (1)
    | for each of the n_items nearest items: position (3), one-hot type
Opponent-related information (karts_position) is not used: the kart is alone.
"""

from __future__ import annotations

import numpy as np


def _rows(seq, width: int, n: int) -> np.ndarray:
    """Stack the first ``n`` entries of a sequence of vectors, pad by repeating
    the last row (or zeros when the sequence is empty)."""
    out = np.zeros((n, width), dtype=np.float32)
    if seq is None:
        return out
    seq = list(seq)
    if len(seq) == 0:
        return out
    arr = np.asarray(seq[:n], dtype=np.float32).reshape(-1, width)
    out[: len(arr)] = arr
    if len(arr) < n:
        out[len(arr):] = arr[-1]
    return out


class FeatureExtractor:
    def __init__(self, n_paths: int, n_items: int, n_item_types: int, track_length: float):
        self.n_paths = n_paths
        self.n_items = n_items
        self.n_item_types = int(n_item_types)
        self.track_length = float(max(track_length, 1.0))
        self.dim = 9 + 7 * n_paths + (3 + self.n_item_types) * n_items

    def __call__(self, obs: dict) -> np.ndarray:
        f = [
            np.asarray(obs["velocity"], dtype=np.float32).reshape(3),
            np.asarray(obs["center_path"], dtype=np.float32).reshape(3),
            np.asarray(obs["center_path_distance"], dtype=np.float32).reshape(1),
            np.asarray(obs["max_steer_angle"], dtype=np.float32).reshape(1),
            np.asarray(obs["distance_down_track"], dtype=np.float32).reshape(1) / self.track_length,
        ]
        starts = _rows(obs.get("paths_start"), 3, self.n_paths)
        ends = _rows(obs.get("paths_end"), 3, self.n_paths)
        widths = _rows(obs.get("paths_width"), 1, self.n_paths)
        f.append(np.concatenate([starts, ends, widths], axis=1).reshape(-1))

        items_pos = list(obs.get("items_position") or [])
        items_type = list(obs.get("items_type") or [])
        pos = np.zeros((self.n_items, 3), dtype=np.float32)
        onehot = np.zeros((self.n_items, self.n_item_types), dtype=np.float32)
        for i in range(min(self.n_items, len(items_pos))):
            pos[i] = np.asarray(items_pos[i], dtype=np.float32).reshape(3)
            t = int(np.asarray(items_type[i]).reshape(-1)[0]) if i < len(items_type) else -1
            if 0 <= t < self.n_item_types:
                onehot[i, t] = 1.0
        f.append(np.concatenate([pos, onehot], axis=1).reshape(-1))

        x = np.concatenate(f).astype(np.float32)
        assert x.shape == (self.dim,), (x.shape, self.dim)
        return np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)


class Normalizer:
    """Frozen standardisation fitted once on the expert dataset D0 and shared by
    every method (DAgger, SAC+symmetric sampling, SAC from scratch)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 10.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-3)
        self.clip = clip

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "clip": self.clip}
