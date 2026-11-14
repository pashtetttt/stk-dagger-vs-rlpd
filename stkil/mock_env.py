"""A tiny pure-numpy stand-in for ``STKRaceEnv`` (kinematic kart on a closed
loop). It produces observations with the same keys and kart-frame conventions
as pystk2_gymnasium (x: left, y: up, z: front) so that the whole pipeline
(dataset, DAgger, SAC+symmetric sampling, evaluation, analysis) can be
smoke-tested in a few minutes without SuperTuxKart. It is NOT used for any
result reported in the study.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _Track:
    def __init__(self, n_nodes=120, radius=60.0, width=8.0):
        phi = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        r = radius * (1.0 + 0.25 * np.sin(3 * phi))
        self.nodes = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
        seg = np.roll(self.nodes, -1, axis=0) - self.nodes
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        self.width = width
        self.n = n_nodes

    def project(self, p, hint):
        """Nearest segment around ``hint``; returns (idx, arc length, signed lateral offset vector)."""
        best = None
        for k in range(-6, 7):
            i = (hint + k) % self.n
            a, b = self.nodes[i], self.nodes[(i + 1) % self.n]
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            q = a + t * ab
            d = np.linalg.norm(p - q)
            if best is None or d < best[0]:
                best = (d, i, self.cum[i] + t * self.seg_len[i], q)
        _, i, s, q = best
        return i, s, q


class MockSTKRaceEnv:
    DT = 0.1
    WHEELBASE = 2.0
    MAX_STEER = 0.5      # rad
    A_MAX = 8.0          # m/s^2
    V_MAX = 22.0
    NITRO_BOOST = 1.25

    def __init__(self, track="mock", **_):
        self.default_track = track
        self.track = _Track()
        self.world = SimpleNamespace(karts=[SimpleNamespace(overall_distance=0.0, has_finished_race=False,
                                                            location=(0.0, 0.0, 0.0), position=1)])
        self.rng = np.random.RandomState(0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.rng = np.random.RandomState(seed)
        tr = self.track
        self.pos = tr.nodes[0].copy()
        d = tr.nodes[1] - tr.nodes[0]
        self.theta = float(np.arctan2(d[1], d[0]) + self.rng.uniform(-0.05, 0.05))
        self.v = 0.0
        self.node = 0
        self.s_prev = 0.0
        self.overall = 0.0
        self.energy = 0.0
        n_items = 8
        idx = self.rng.choice(tr.n, n_items, replace=False)
        self.items_pos = tr.nodes[idx] + self.rng.uniform(-2, 2, size=(n_items, 2))
        self.items_type = self.rng.randint(0, 7, size=n_items)
        k = self.world.karts[0]
        k.overall_distance, k.has_finished_race = 0.0, False
        return self._obs(), {}

    def _to_kart(self, p):
        d = np.asarray(p, dtype=np.float64) - self.pos
        f = np.array([np.cos(self.theta), np.sin(self.theta)])
        left = np.array([-np.sin(self.theta), np.cos(self.theta)])
        return np.array([d @ left, 0.0, d @ f], dtype=np.float32)

    def _obs(self):
        tr = self.track
        _, _, q = tr.project(self.pos, self.node)
        cp = self._to_kart(q)
        ids = [(self.node + k) % tr.n for k in range(20)]
        items = [self._to_kart(p) for p in self.items_pos]
        order = sorted(range(len(items)), key=lambda i: (items[i][2] < 0, np.linalg.norm(items[i])))
        return {
            "phase": 3,
            "velocity": np.array([0.0, 0.0, self.v], dtype=np.float32),
            "center_path": cp,
            "center_path_distance": np.array([np.linalg.norm(cp) * np.sign(cp[0])], dtype=np.float32),
            "max_steer_angle": np.array([self.MAX_STEER], dtype=np.float32),
            "distance_down_track": np.array([self.s_prev], dtype=np.float32),
            "energy": np.array([self.energy], dtype=np.float32),
            "paths_start": tuple(self._to_kart(tr.nodes[i]) for i in ids),
            "paths_end": tuple(self._to_kart(tr.nodes[(i + 1) % tr.n]) for i in ids),
            "paths_width": tuple(np.array([tr.width], dtype=np.float32) for _ in ids),
            "paths_distance": tuple(np.array([tr.cum[i], tr.cum[i] + tr.seg_len[i]], dtype=np.float32) for i in ids),
            "items_position": tuple(items[i] for i in order),
            "items_type": tuple(int(self.items_type[i]) for i in order),
            "karts_position": tuple(),
        }

    def step(self, action):
        steer = float(np.clip(action["steer"], -1, 1)) * self.MAX_STEER
        acc = float(np.clip(action["acceleration"], 0, 1))
        vmax = self.V_MAX * (self.NITRO_BOOST if action.get("nitro") else 1.0)
        a = acc * self.A_MAX - (6.0 if action.get("brake") else 0.0) - 0.3 * self.v
        self.v = float(np.clip(self.v + a * self.DT, 0.0, vmax))
        _, _, q = self.track.project(self.pos, self.node)
        if np.linalg.norm(self.pos - q) > self.track.width / 2:   # off-road: slow
            self.v = min(self.v, 5.0)
        if action.get("rescue"):                                   # teleport to centre line
            self.pos, self.v = q.copy(), 0.0
        yaw = self.v / self.WHEELBASE * np.tan(steer) * (0.6 if action.get("drift") else 1.0)
        self.theta += yaw * self.DT
        self.pos = self.pos + self.v * self.DT * np.array([np.cos(self.theta), np.sin(self.theta)])

        tr = self.track
        self.node, s, _ = tr.project(self.pos, self.node)
        ds = s - self.s_prev
        if ds < -tr.length / 2:
            ds += tr.length
        elif ds > tr.length / 2:
            ds -= tr.length
        self.s_prev = s
        last = max(self.overall, 0.0)
        self.overall += ds
        k = self.world.karts[0]
        k.overall_distance = self.overall
        finished = self.overall >= tr.length
        k.has_finished_race = finished
        reward = (max(self.overall, 0.0) - last) / 10.0 - 0.1 + 10.0 * finished
        return self._obs(), reward, finished, False, {"position": 1, "distance": max(self.overall, 0)}

    def close(self):
        pass


class MockExpert:
    """Pure-pursuit controller for the mock track."""

    def choose_action(self, obs):
        ends = obs["paths_end"]
        target = ends[min(3, len(ends) - 1)]
        angle = float(np.arctan2(target[0], target[2]))
        steer = float(np.clip(2.5 * angle, -1, 1))
        acc = 1.0 if abs(steer) < 0.6 else 0.6
        return {"steer": steer, "acceleration": acc, "brake": False, "drift": False,
                "nitro": abs(steer) < 0.15, "rescue": False, "fire": False}
