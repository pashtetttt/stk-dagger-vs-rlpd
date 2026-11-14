"""Statistics used in the report (numpy + scipy only).

Unit of replication = one training seed (independent run). For each run we
reduce the final evaluation (10 deterministic episodes on shared seeds) to one
number, then compare methods across seeds with
  * Welch's t-test (unequal variances),
  * the Mann-Whitney U test (no normality assumption; with n=5 vs 5 the smallest
    reachable two-sided p-value is 2/252 = 0.0079),
  * a bootstrap 95% CI of the difference of means,
  * Hedges' g effect size and the probability of improvement P(X > Y).
Secondary comparisons are corrected with Holm-Bonferroni.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as st


def iqm(x: Sequence[float]) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(np.mean(x))
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(np.mean(x[lo:hi]))


def bootstrap_ci(x: Sequence[float], fn=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    b = np.array([fn(x[rng.randint(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def describe(x: Sequence[float]) -> Dict:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": float(np.mean(x)), "std": sd, "ci95_t": [float(np.mean(x) - half), float(np.mean(x) + half)],
            "ci95_boot": list(bootstrap_ci(x)), "median": float(np.median(x)), "iqm": iqm(x),
            "min": float(np.min(x)), "max": float(np.max(x))}


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (a.mean() - b.mean()) / sp)


def prob_improvement(a, b) -> float:
    """P(X > Y) for X ~ a, Y ~ b (ties count 1/2)."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def compare(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n_a": len(a), "n_b": len(b), "mean_diff": float(a.mean() - b.mean())}
    if len(a) >= 2 and len(b) >= 2:
        if np.allclose(np.concatenate([a, b]), a[0]):
            out.update(welch_t=float("nan"), welch_p=1.0, mwu_U=float("nan"), mwu_p=1.0)
        else:
            t = st.ttest_ind(a, b, equal_var=False)
            u = st.mannwhitneyu(a, b, alternative="two-sided")
            out.update(welch_t=float(t.statistic), welch_p=float(t.pvalue),
                       mwu_U=float(u.statistic), mwu_p=float(u.pvalue))
        rng = np.random.RandomState(seed)
        diffs = [a[rng.randint(0, len(a), len(a))].mean() - b[rng.randint(0, len(b), len(b))].mean()
                 for _ in range(n_boot)]
        out["diff_ci95_boot"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        out["hedges_g"] = hedges_g(a, b)
        out["p_improvement"] = prob_improvement(a, b)
    return out


def holm(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
