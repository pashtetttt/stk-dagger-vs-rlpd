"""Neural networks. The *same* actor class (MLP + tanh-squashed Gaussian head)
is used by DAgger and by SAC, so that the two methods differ only by how the
actor is trained, not by its capacity."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden: Sequence[int], layernorm: bool = False) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int],
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def _dist_params(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mu, log_std

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """tanh(mean): the action used for evaluation (and the BC output)."""
        mu, _ = self._dist_params(obs)
        return torch.tanh(mu)

    def sample(self, obs: torch.Tensor):
        """Reparameterised sample and its log-probability (with tanh correction)."""
        mu, log_std = self._dist_params(obs)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # log |d tanh(u)/du| = 2 (log 2 - u - softplus(-2u))  (numerically stable)
        logp = logp - (2 * (math.log(2.0) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class EnsembleCritic(nn.Module):
    """``n`` independent Q-networks (optionally with LayerNorm, as in RLPD)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int], n: int, layernorm: bool):
        super().__init__()
        self.qs = nn.ModuleList([mlp(obs_dim + act_dim, 1, hidden, layernorm) for _ in range(n)])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs], dim=0)  # (n, B)
