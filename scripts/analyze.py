"""Aggregate all runs, compute statistics and produce figures + tables.

    python scripts/analyze.py [--config ...]

Reads runs/<algo>/seed_<k>/{eval.csv,final.json}, results/expert_benchmark.json
and data/expert_<track>.json. Writes
    results/stats.json             all numbers
    results/summary.md             tables ready to paste in the report
    results/figures/*.png          learning curves, final distributions
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from stkil.common import dataset_path
from stkil.config import load_config, output_path
from stkil.stats import bootstrap_ci, compare, describe, holm

LABELS = {"dagger": "DAgger", "rlpd": "SAC + symmetric sampling (RLPD)", "sac": "SAC from scratch", "bc": "BC (DAgger it. 0)"}
COLORS = {"dagger": "#2a78d6", "rlpd": "#eb6834", "sac": "#1baf7a", "bc": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(cfg):
    root = output_path(cfg, "runs", "x").parent
    runs = {}
    if not root.exists():
        return runs
    for algo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sd in sorted(algo_dir.glob("seed_*")):
            fin = sd / "final.json"
            if not fin.exists():
                continue
            with open(sd / "eval.csv") as f:
                rows = list(csv.DictReader(f))
            with open(fin) as f:
                final = json.load(f)
            curve = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
                     for k in ["env_steps", "score", "finish_rate", "lap_steps_finished", "progress", "return", "wall_time_s"]}
            runs.setdefault(algo_dir.name, []).append({"seed": int(sd.name.split("_")[1]), "curve": curve, "final": final})
    return runs


def final_metric(r, key, max_steps):
    v = r[key]
    if key == "lap_steps_finished" and not np.isfinite(v):
        return np.nan
    return float(v)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(runs, key, ylabel, path, refs, dt=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    style(ax)
    for algo in ["dagger", "rlpd", "sac"]:
        if algo not in runs:
            continue
        xs = runs[algo][0]["curve"]["env_steps"]
        Y = np.array([r["curve"][key] for r in runs[algo] if len(r["curve"][key]) == len(xs)])
        if dt:
            Y = Y * dt
        mean = np.nanmean(Y, axis=0)
        lo, hi = zip(*[bootstrap_ci(c[np.isfinite(c)], n_boot=2000) if np.isfinite(c).sum() > 1 else (np.nan, np.nan)
                       for c in Y.T])
        ax.plot(xs, mean, color=COLORS[algo], lw=2, marker="o", ms=4, label=f"{LABELS[algo]} (n={len(Y)})")
        ax.fill_between(xs, lo, hi, color=COLORS[algo], alpha=0.18, lw=0)
    for name, (val, ls) in refs.items():
        if val is not None and np.isfinite(val):
            ax.axhline(val * (dt or 1), color=MUTED, ls=ls, lw=1.2, label=name)
    ax.set_xlabel("online environment steps", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_final(per_algo, ylabel, path, ref=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = [a for a in ["bc", "dagger", "rlpd", "sac"] if a in per_algo]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
    style(ax)
    rng = np.random.RandomState(0)
    for i, a in enumerate(algos):
        v = np.asarray([x for x in per_algo[a] if np.isfinite(x)])
        if len(v) == 0:
            continue
        ax.boxplot([v], positions=[i], widths=0.45, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
                   capprops={"color": MUTED})
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=36, color=COLORS[a], zorder=3,
                   edgecolor="white", linewidth=1)
    if ref is not None and np.isfinite(ref):
        ax.axhline(ref, color=MUTED, ls="--", lw=1.2, label="expert")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(range(len(algos)))
    short = {"bc": "BC\n(DAgger it. 0)", "dagger": "DAgger", "rlpd": "SAC + sym.\nsampling (RLPD)", "sac": "SAC\nfrom scratch"}
    ax.set_xticklabels([short[a] for a in algos], fontsize=8)
    ax.set_ylabel(ylabel, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dagger_diag(cfg, path):
    """BC loss on the aggregated DAgger dataset at the end of each iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = output_path(cfg, "runs", "dagger", "x").parent
    losses, betas, xs = [], None, None
    for sd in sorted(root.glob("seed_*")):
        with open(sd / "eval.csv") as f:
            rows = list(csv.DictReader(f))
        ex = [json.loads(r["extra"]) for r in rows]
        losses.append([e["bc_loss"] for e in ex])
        betas = [e["beta"] for e in ex]
        xs = [int(r["env_steps"]) for r in rows]
    L = np.array(losses)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    style(ax)
    lo, hi = zip(*[bootstrap_ci(c, n_boot=2000) for c in L.T])
    ax.plot(xs, L.mean(0), color=COLORS["dagger"], lw=2, marker="o", ms=4, label=f"DAgger (n={len(L)})")
    ax.fill_between(xs, lo, hi, color=COLORS["dagger"], alpha=0.18, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x // 1000}k\n1/{round(1 / b)}" if b is not None else "0\nBC" for x, b in zip(xs, betas)],
                       fontsize=7)
    ax.set_xlabel("online environment steps  /  β (prob. of executing the expert)", color=INK)
    ax.set_ylabel("training MSE on aggregated data", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compose_panel(figdir, names, path, ncols=2):
    """All report figures in one compact grid, labelled (a), (b), ..."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    files = [figdir / f"{n}.png" for n in names if (figdir / f"{n}.png").exists()]
    nrows = (len(files) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 3.9 * nrows), dpi=150)
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(files):
            ax.imshow(mpimg.imread(files[i]))
            ax.set_title(f"({chr(97 + i)})", loc="left", fontsize=12, color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fmt(d, k="mean", p=3):
    if not d or d.get("n", 0) == 0:
        return "–"
    lo, hi = d["ci95_t"]
    return f"{d[k]:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]" if np.isfinite(lo) else f"{d[k]:.{p}f}"


def main(cfg):
    max_steps = int(cfg.env.max_steps)
    dt = float(cfg.env.step_dt)
    runs = load_runs(cfg)
    if not runs:
        print("no finished runs found")
        return
    figdir = output_path(cfg, "results", "figures", "x").parent
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- references ---------------------------------------------------------
    expert = {}
    bpath = output_path(cfg, "results", "expert_benchmark.json")
    if bpath.exists():
        with open(bpath) as f:
            b = json.load(f)
        expert = {"name": b["best"], **{k: b["results"][b["best"]][k] for k in
                                        ["score", "finish_rate", "lap_steps_finished", "progress"]},
                  "all": {k: {kk: v.get(kk) for kk in ["score", "finish_rate", "lap_steps_finished"]}
                          for k, v in b["results"].items()}}
    dmeta = {}
    dp = dataset_path(cfg).with_suffix(".json")
    if dp.exists():
        with open(dp) as f:
            dm = json.load(f)
        dmeta = {"expert": dm["expert"], "n_transitions": dm["n_transitions"], "n_episodes": len(dm["episodes"]),
                 "clean_finish_rate": float(np.mean([e["finished"] for e in dm["episodes"] if not e["noisy"]] or [np.nan])),
                 "noisy_finish_rate": float(np.mean([e["finished"] for e in dm["episodes"] if e["noisy"]] or [np.nan]))}

    # ---- per-seed final values ----------------------------------------------
    finals = {a: [r["final"] for r in rs] for a, rs in runs.items()}
    if "dagger" in runs:
        finals["bc"] = [r["final"]["bc_final"] for r in runs["dagger"] if "bc_final" in r["final"]]
    metrics = {"score": "score (↑)", "finish_rate": "finish rate (↑)",
               "lap_steps_finished": "lap steps, finished laps (↓)", "lap_steps_censored": "lap steps, censored (↓)",
               "progress": "lap progress (↑)"}
    per = {m: {a: [final_metric(f, m, max_steps) for f in fs] for a, fs in finals.items()} for m in metrics}
    auc = {a: [float(np.nanmean(r["curve"]["score"])) for r in rs] for a, rs in runs.items()}
    desc = {m: {a: describe(v) for a, v in per[m].items()} for m in metrics}
    desc["auc_score"] = {a: describe(v) for a, v in auc.items()}
    # exploratory (not pre-registered): area under the curve without the 0-step point,
    # where DAgger is already BC-initialised while SAC starts from a random actor
    auc5 = {a: [float(np.nanmean(r["curve"]["score"][1:])) for r in rs] for a, rs in runs.items()}
    desc["auc_score_from_5k"] = {a: describe(v) for a, v in auc5.items()}

    # ---- hypothesis tests ---------------------------------------------------
    tests = {}
    pairs = [("dagger", "rlpd"), ("dagger", "bc"), ("rlpd", "bc"), ("rlpd", "sac"), ("dagger", "sac")]
    for a, b in pairs:
        if a in per["score"] and b in per["score"]:
            tests[f"{a}_vs_{b}"] = {
                "final_score": compare(per["score"][a], per["score"][b]),
                "auc_score": compare(auc[a], auc[b]) if a in auc and b in auc else None,
                "final_lap_steps_censored": compare(per["lap_steps_censored"][a], per["lap_steps_censored"][b]),
            }
    # primary hypothesis: final score DAgger vs RLPD (no correction);
    # secondary family: every other (pair, metric) -> Holm-Bonferroni on MWU p-values
    fam = [(k, m) for k, v in tests.items() for m, t in v.items()
           if t is not None and "mwu_p" in t and not (k == "dagger_vs_rlpd" and m == "final_score")]
    for a, b in pairs:
        if f"{a}_vs_{b}" in tests and a in auc5 and b in auc5:
            tests[f"{a}_vs_{b}"]["auc_score_from_5k (exploratory)"] = compare(auc5[a], auc5[b])
    fam = [x for x in fam if "exploratory" not in x[1]]
    if fam:
        adj = holm([tests[k][m]["mwu_p"] for k, m in fam])
        for (k, m), p in zip(fam, adj):
            tests[k][m]["mwu_p_holm"] = p

    # ---- figures --------------------------------------------------------------
    refs_score = {"expert": (expert.get("score"), "--")}
    bc_curve = [r["curve"]["score"][0] for r in runs.get("dagger", [])]
    if bc_curve:
        refs_score["BC (mean)"] = (float(np.mean(bc_curve)), ":")
    plot_curves(runs, "score", "evaluation score (↑)", figdir / "learning_curve_score.png", refs_score)
    plot_curves(runs, "finish_rate", "finish rate (↑)", figdir / "learning_curve_finish_rate.png", {})
    plot_curves(runs, "lap_steps_finished", "lap time of finished laps [s] (↓)",
                figdir / "learning_curve_lap_time.png",
                {"expert": (expert.get("lap_steps_finished"), "--")}, dt=dt)
    if "dagger" in runs:
        plot_dagger_diag(cfg, figdir / "dagger_diagnostics.png")
    plot_final(per["score"], "final score (↑)", figdir / "final_score.png", expert.get("score"))
    plot_final({a: [x * dt for x in v] for a, v in per["lap_steps_finished"].items()},
               "final lap time [s], finished laps (↓)", figdir / "final_lap_time.png",
               (expert.get("lap_steps_finished") or np.nan) * dt)

    compose_panel(figdir, ["learning_curve_score", "learning_curve_lap_time", "learning_curve_finish_rate",
                           "dagger_diagnostics", "final_score", "final_lap_time"], figdir / "report_panel.png")

    # ---- outputs --------------------------------------------------------------
    out = {"config": cfg.to_dict(), "expert": expert, "dataset": dmeta,
           "n_runs": {a: len(r) for a, r in runs.items()}, "descriptive": desc, "tests": tests,
           "per_seed": {m: per[m] for m in metrics}, "auc_per_seed": auc, "auc_from_5k_per_seed": auc5,
           "wall_time_s": {a: describe([r["final"].get("wall_time_s", np.nan) for r in rs]) for a, rs in runs.items()}}
    with open(output_path(cfg, "results", "stats.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    L = ["# Results summary (auto-generated by scripts/analyze.py)", ""]
    L.append(f"Track `{cfg.env.track}`, time limit {max_steps} steps ({max_steps * dt:.0f} s at {dt} s/step), "
             f"online budget {cfg.budget.online_env_steps} env steps, final evaluation on "
             f"{cfg.budget.final_eval_episodes} shared seeds. Values: mean over training seeds "
             "[95% t-interval].")
    L.append("")
    if expert:
        L.append(f"Expert (`{expert['name']}`): score {expert['score']:.3f}, finish rate {expert['finish_rate']:.2f}, "
                 f"lap {expert['lap_steps_finished']:.1f} steps ({expert['lap_steps_finished'] * dt:.1f} s).")
        L.append("")
        L.append("| expert | score | finish rate | lap steps |")
        L.append("|---|---|---|---|")
        for k, v in expert["all"].items():
            L.append(f"| {k} | {v['score']:.3f} | {v['finish_rate']:.2f} | {v['lap_steps_finished']:.1f} |")
        L.append("")
    if dmeta:
        L.append(f"Dataset D0: {dmeta['n_transitions']} transitions, {dmeta['n_episodes']} episodes "
                 f"(finish rate clean {dmeta['clean_finish_rate']:.2f} / noisy {dmeta['noisy_finish_rate']:.2f}).")
        L.append("")
    algos = [a for a in ["bc", "dagger", "rlpd", "sac"] if a in desc["score"]]
    L.append("| method | n | " + " | ".join(metrics.values()) + " | AUC score (↑) | AUC from 5k (↑, expl.) |")
    L.append("|---|---|" + "---|" * (len(metrics) + 2))
    for a in algos:
        cells = [fmt(desc[m][a], p=1 if "lap_steps" in m else 3) for m in metrics]
        L.append(f"| {LABELS[a]} | {desc['score'][a].get('n', 0)} | " + " | ".join(cells)
                 + f" | {fmt(desc['auc_score'].get(a))} | {fmt(desc['auc_score_from_5k'].get(a))} |")
    L.append("")
    L.append("## Tests (unit = training seed)")
    L.append("")
    L.append("| comparison | metric | mean diff [boot 95% CI] | Welch p | MWU p | MWU p (Holm) | Hedges g | P(A>B) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in tests.items():
        for m, t in v.items():
            if not t or "welch_p" not in t:
                continue
            ci = t["diff_ci95_boot"]
            L.append(f"| {k.replace('_vs_', ' vs ')} | {m} | {t['mean_diff']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{t['welch_p']:.4f} | {t['mwu_p']:.4f} | {(format(t['mwu_p_holm'], '.4f') if 'mwu_p_holm' in t else ('– (primary)' if m == 'final_score' else '– (expl.)'))} | "
                     f"{t['hedges_g']:.2f} | {t['p_improvement']:.2f} |")
    L.append("")
    prim = tests.get("dagger_vs_rlpd", {}).get("final_score")
    if prim and "welch_p" in prim:
        sig = prim["welch_p"] < 0.05 and prim["mwu_p"] < 0.05
        better = "DAgger" if prim["mean_diff"] > 0 else "SAC + symmetric sampling"
        L.append("**Primary hypothesis (final score, DAgger vs SAC+symmetric sampling):** "
                 + (f"significant difference at α = 0.05 (both tests), in favour of {better}."
                    if sig else "no statistically significant difference at α = 0.05 with this number of seeds."))
    L.append("")
    L.append("Figures: `results/figures/learning_curve_score.png`, `learning_curve_finish_rate.png`, "
             "`learning_curve_lap_time.png`, `final_score.png`, `final_lap_time.png`.")
    with open(output_path(cfg, "results", "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    main(load_config(a.config, a.set))
