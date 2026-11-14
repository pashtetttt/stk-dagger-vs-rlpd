"""Build report/report.md (and report/report.pdf if pandoc is installed) from
report/report_template.md, configs, results/stats.json and results/summary.md.

    python scripts/make_report.py [--config ...] [--set report.repo_url=https://github.com/...]

The interpretation of the results is read from report/discussion.md (written by
hand once the results are known); hypothesis verdicts are computed here.
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess

import numpy as np

import _bootstrap  # noqa: F401
from stkil.common import dataset_path
from stkil.config import PROJECT_ROOT, add_pystk_to_path, load_config, output_path


def get(cfg, dotted):
    node = cfg
    for p in dotted.split("."):
        node = node[p]
    return node


def hparam_table(cfg):
    d, r, s_, b = cfg.dagger, cfg.rlpd, cfg.sac, cfg.budget
    rows = [
        ("DAgger", f"BC init {d.bc_init_steps} grad. steps; {d.n_iterations} iterations × "
                   f"{b.online_env_steps // d.n_iterations} env steps; β_i = {d.beta0}·{d.beta_decay}^(i−1) "
                   f"(per-step mixture); {d.grad_steps_per_iter} grad. steps/iteration, warm start; "
                   f"Adam lr {d.lr}, batch {d.batch_size}"),
        ("RLPD", f"offline/online ratio {r.offline_ratio}/{1 - r.offline_ratio}; {r.n_critics} critics "
                 f"MLP {cfg.network.critic_hidden} + LayerNorm, target min over {r.critic_subset}; "
                 f"UTD {r.utd}, 1 actor update/step; γ {r.gamma}, τ {r.tau}, batch {r.batch_size}; "
                 f"lr {r.actor_lr} (actor, critic, α); auto α (init {r.init_alpha}), "
                 f"target entropy −{r.target_entropy_scale}·|A|; learning starts {r.learning_starts}, "
                 f"no random steps"),
        ("SAC (control)", f"as RLPD with offline ratio {s_.offline_ratio}, {s_.random_steps} random steps, "
                          f"learning starts {s_.learning_starts}"),
        ("Shared", f"{b.online_env_steps} online steps; eval every {b.eval_every} steps "
                   f"({b.eval_episodes} ep.), final {b.final_eval_episodes} ep.; seeds {list(cfg.experiment.seeds)}"),
    ]
    out = ["| | hyper-parameters |", "|---|---|"]
    out += [f"| **{a}** | {b_} |" for a, b_ in rows]
    return "\n".join(out)


def verdicts(stats, dt):
    t = stats.get("tests", {})
    desc = stats.get("descriptive", {})
    ex = stats.get("expert", {})
    lines = []

    def sig(test, key="mwu_p_holm"):
        p = test.get(key, test.get("mwu_p"))
        return p is not None and p < 0.05 and test.get("welch_p", 1) < 0.05

    dv = t.get("dagger_vs_rlpd", {})
    if dv.get("final_score"):
        x = dv["final_score"]
        lines.append(f"* **Primary (final score):** mean difference DAgger − RLPD = {x['mean_diff']:.3f} "
                     f"[{x['diff_ci95_boot'][0]:.3f}, {x['diff_ci95_boot'][1]:.3f}], Welch p = {x['welch_p']:.4f}, "
                     f"MWU p = {x['mwu_p']:.4f}, Hedges g = {x['hedges_g']:.2f} → "
                     + ("significant." if x["welch_p"] < 0.05 and x["mwu_p"] < 0.05 else "not significant."))
    if dv.get("auc_score"):
        x = dv["auc_score"]
        lines.append(f"* **H1 (DAgger learns faster, AUC):** diff = {x['mean_diff']:.3f}, Holm-corrected MWU p = "
                     f"{x.get('mwu_p_holm', float('nan')):.4f} → "
                     + ("supported." if x["mean_diff"] > 0 and sig(x) else "not supported at α = 0.05."))
    lap = desc.get("lap_steps_finished", {})
    if ex.get("lap_steps_finished") and lap:
        e = ex["lap_steps_finished"]
        parts = []
        for a in ["dagger", "rlpd"]:
            if lap.get(a, {}).get("n"):
                lo, hi = lap[a]["ci95_t"]
                rel = "faster than" if hi < e else ("slower than" if lo > e else "not distinguishable from")
                parts.append(f"{a}: {lap[a]['mean'] * dt:.1f} s [95% CI {lo * dt:.1f}, {hi * dt:.1f}] — {rel} the expert")
        lines.append(f"* **H2 (lap time vs expert = {e * dt:.1f} s, finished laps only):** " + "; ".join(parts) + ".")
    for k, name in [("dagger_vs_bc", "DAgger vs BC"), ("rlpd_vs_bc", "RLPD vs BC"), ("rlpd_vs_sac", "RLPD vs SAC from scratch")]:
        x = t.get(k, {}).get("final_score")
        if x and "welch_p" in x:
            a_name, b_name = name.split(" vs ")
            better = a_name if x["mean_diff"] > 0 else b_name
            lines.append(f"* **H3 ({name}, final score):** diff = {x['mean_diff']:.3f}, Holm-corrected MWU p = "
                         f"{x.get('mwu_p_holm', float('nan')):.4f} → "
                         + (f"significant, in favour of {better}." if sig(x) else "not significant."))
    return "\n".join(lines)


LABEL = {"bc": "BC", "dagger": "DAgger", "rlpd": "RLPD", "sac": "SAC"}


def _ci(d, scale=1.0, p=3):
    if not d or not d.get("n"):
        return "–"
    lo, hi = d["ci95_t"]
    if not np.isfinite(lo):
        return f"{d['mean'] * scale:.{p}f}"
    return f"{d['mean'] * scale:.{p}f} [{lo * scale:.{p}f}, {hi * scale:.{p}f}]"


def results_tables(stats, cfg, dt):
    d = stats["descriptive"]
    algos = [a for a in ["bc", "dagger", "rlpd", "sac"] if a in d["score"]]
    L = ["| method | final score ↑ | finish rate ↑ | lap time [s] ↓ | AUC ↑ | AUC from 5k ↑ |",
         "|---|---|---|---|---|---|"]
    full = {"bc": "BC (DAgger it. 0)", "dagger": "DAgger", "rlpd": "SAC + sym. sampling (RLPD)",
            "sac": "SAC from scratch"}
    for a in algos:
        n_fin = d["lap_steps_finished"][a].get("n", 0)
        L.append(f"| {full[a]} | {_ci(d['score'][a])} | {_ci(d['finish_rate'][a], p=2)} | "
                 f"{_ci(d['lap_steps_finished'][a], dt, 1)}{'' if n_fin == d['score'][a]['n'] else f' ({n_fin} seeds)'} | "
                 f"{_ci(d['auc_score'].get(a)) if a != 'bc' else '–'} | "
                 f"{_ci(d.get('auc_score_from_5k', {}).get(a)) if a != 'bc' else '–'} |")
    L += ["", f"*Table 1.* Mean over {d['score']['dagger']['n']} seeds [95 % t-interval], final evaluation "
          f"after {cfg.budget.online_env_steps} online steps; lap time over finished laps; "
          "AUC = mean score over the learning curve (\"from 5k\": exploratory, without the 0-step point).", "",
          "| A vs B | metric | A − B [boot. 95 % CI] | Welch p | MWU p | Holm p | g | P(A>B) |",
          "|---|---|---|---|---|---|---|---|"]
    keys = [("dagger_vs_rlpd", "final_score"), ("dagger_vs_rlpd", "auc_score"),
            ("dagger_vs_rlpd", "auc_score_from_5k (exploratory)"), ("dagger_vs_bc", "final_score"),
            ("rlpd_vs_bc", "final_score"), ("rlpd_vs_sac", "final_score"), ("rlpd_vs_sac", "auc_score")]
    names = {"final_score": "final score", "auc_score": "AUC", "auc_score_from_5k (exploratory)": "AUC from 5k"}
    for k, m in keys:
        t = stats["tests"].get(k, {}).get(m)
        if not t or "welch_p" not in t:
            continue
        a, b = k.split("_vs_")
        holm_p = f"{t['mwu_p_holm']:.4f}" if "mwu_p_holm" in t else ("primary" if m == "final_score" else "expl.")
        ci = t["diff_ci95_boot"]
        L.append(f"| {LABEL[a]} vs {LABEL[b]} | {names[m]} | {t['mean_diff']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                 f"{t['welch_p']:.4f} | {t['mwu_p']:.4f} | {holm_p} | {t['hedges_g']:.2f} | {t['p_improvement']:.2f} |")
    L += ["", "*Table 2.* Key tests over training seeds (all tests: `results/summary.md`). Holm: correction over "
          "all secondary tests; \"expl.\": exploratory, uncorrected."]
    return "\n".join(L)


def expert_line(stats, dt):
    ex = stats.get("expert", {})
    if not ex:
        return ""
    parts = []
    for k, v in sorted(ex["all"].items(), key=lambda kv: -(kv[1].get("score") or -1)):
        lap = v.get("lap_steps_finished")
        fr = v.get("finish_rate", 0)
        parts.append(f"{k} {lap * dt:.1f} s" + ("" if fr == 1 else f" (finishes {fr:.0%}, crashes without opponents)"))
    return (f"Lap times: {', '.join(parts)}. The expert is **{ex['name']}** "
            f"({ex['lap_steps_finished'] * dt:.1f} s, deterministic).")


def main(cfg, repo_url):
    tpl = (PROJECT_ROOT / "report" / "report_template.md").read_text(encoding="utf-8")
    stats_p = output_path(cfg, "results", "stats.json")
    summ_p = output_path(cfg, "results", "summary.md")
    stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
    dt = float(cfg.env.step_dt)
    dmeta = {}
    if dataset_path(cfg).with_suffix(".json").exists():
        dmeta = json.loads(dataset_path(cfg).with_suffix(".json").read_text())

    commit = str(cfg.get("pystk_commit", "unknown"))
    try:
        src = add_pystk_to_path(cfg)
        commit = subprocess.run(["git", "-C", str(src.parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or commit
    except Exception:
        pass

    versions = "n/a"
    runs = output_path(cfg, "runs", "x").parent
    for c in sorted(runs.glob("*/seed_*/config.json")) if runs.exists() else []:
        v = json.loads(c.read_text()).get("versions")
        if v:
            versions = ", ".join(f"{k} {val}" for k, val in v.items())
            break

    ex = stats.get("expert", {})
    if ex:
        et = ["| agent | score | finish rate | lap steps | lap time [s] |", "|---|---|---|---|---|"]
        for k, v in ex["all"].items():
            lap = v.get("lap_steps_finished")
            ok = isinstance(lap, (int, float)) and np.isfinite(lap)
            et.append(f"| {k}{' **(expert)**' if k == ex['name'] else ''} | {v.get('score', float('nan')):.3f} | "
                      f"{v.get('finish_rate', float('nan'))} | {lap:.1f} | {lap * dt:.1f} |" if ok else
                      f"| {k} | {v.get('score')} | {v.get('finish_rate')} | – | – |")
        expert_table = "\n".join(et)
    else:
        expert_table = "*Expert benchmark not run yet (`python scripts/benchmark_experts.py`).*"

    if summ_p.exists() and stats:
        results = results_tables(stats, cfg, dt)
        prim = stats["tests"].get("dagger_vs_rlpd", {}).get("final_score", {})
        if "welch_p" in prim:
            lap = stats["descriptive"]["lap_steps_finished"]
            ex_lap = stats.get("expert", {}).get("lap_steps_finished")
            headline = ("**Main result.** " + ("DAgger" if prim["mean_diff"] > 0 else "SAC + symmetric sampling")
                        + f" obtains the higher final score (Welch p = {prim['welch_p']:.4f}, "
                        f"Mann–Whitney p = {prim['mwu_p']:.4f}). Mean lap time: RLPD "
                        f"{lap['rlpd']['mean'] * dt:.1f} s, DAgger {lap['dagger']['mean'] * dt:.1f} s, "
                        f"BC {lap['bc']['mean'] * dt:.1f} s"
                        + (f", expert {ex_lap * dt:.1f} s" if ex_lap else "") + ". See Sections 4–5.")
        else:
            headline = ""
    else:
        results = "*No results yet — run `python scripts/run_all.py` then `python scripts/make_report.py`.*"
        headline = "*Results pending.*"

    disc_p = PROJECT_ROOT / "report" / "discussion.md"
    discussion = disc_p.read_text(encoding="utf-8") if disc_p.exists() else \
        "*To be written from the results above (create `report/discussion.md`).*"

    subs = {
        "DATE": datetime.date.today().isoformat(),
        "N_SEEDS": str(max(stats.get("n_runs", {}).values() or [len(cfg.experiment.seeds)])),
        "HEADLINE": headline,
        "PYSTK_COMMIT": commit,
        "MAX_SECONDS": f"{int(cfg.env.max_steps) * dt:.0f}",
        "FEATURE_DIM": str(dmeta.get("feature_dim", "D")),
        "ACT_DIM": str(len(cfg.action_keys)),
        "ACTION_KEYS": ", ".join(f"`{k}`" for k in cfg.action_keys),
        "NOISY_PCT": f"{100 * float(cfg.dataset.noisy_fraction):.0f}",
        "DATASET_LINE": (f"D0 contains **{dmeta['n_transitions']} transitions** generated by `{dmeta['expert']}`."
                         if dmeta else ""),
        "EXPERT_TABLE": expert_table,
        "HPARAM_TABLE": hparam_table(cfg),
        "EXPERT_LINE": expert_line(stats, dt),
        "DATASET_N": str(dmeta.get("n_transitions", "?")),
        "RESULTS": results,
        "DISCUSSION": discussion,
        "VERSIONS": versions,
        "REPO_URL": repo_url or "*(add the URL of the public repository)*",
    }
    out = re.sub(r"\{\{cfg\.([\w.]+)\}\}", lambda m: str(get(cfg, m.group(1))), tpl)
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", v)
    md = PROJECT_ROOT / "report" / "report.md"
    md.write_text(out, encoding="utf-8")
    print(f"written {md}")
    if shutil.which("pandoc"):
        pdf = md.with_suffix(".pdf")
        hdr = md.parent / "_header.tex"
        hdr.write_text("\\usepackage{etoolbox}\n\\AtBeginEnvironment{longtable}{\\scriptsize}\n"
                       "\\usepackage{float}\n\\floatplacement{figure}{H}\n\\usepackage{titling}\n\\setlength{\\droptitle}{-2.5cm}\n", encoding="utf-8")
        engine = (["--pdf-engine=xelatex", "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans",
                   "-V", "monofont=DejaVu Sans Mono"] if shutil.which("xelatex") else [])
        engine += ["-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
                   "-H", str(hdr)]
        r = subprocess.run(["pandoc", str(md), "-o", str(pdf), "--resource-path", str(md.parent), *engine],
                           cwd=md.parent)
        print(f"written {pdf}" if r.returncode == 0 else "pandoc failed (a LaTeX engine is needed for PDF)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--repo-url", default=None)
    a = p.parse_args()
    main(load_config(a.config, a.set), a.repo_url)
