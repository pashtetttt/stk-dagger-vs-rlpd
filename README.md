# Mini-project 2 — DAgger vs symmetric sampling (RLPD) in SuperTuxKart

Empirical comparison of **DAgger** (interactive imitation learning) and **SAC with
RLPD-style symmetric sampling** (online RL with offline expert data) for a single
kart that must complete **one lap as fast as possible, alone on one track**.
Both methods use the same expert dataset, actor network, features, online budget
and evaluation seeds. See `report/` for the methodology and the conceptual comparison.

```
mp2_dagger_rlpd/
├── configs/default.yaml        every hyper-parameter of the study
├── stkil/                      library code
│   ├── env.py                  single-kart / single-lap wrapper (time limit, auto-rescue, lap stats)
│   ├── features.py             fixed-size observation features + frozen normaliser
│   ├── actions.py              shared [-1,1]^d action space <-> STK action dicts
│   ├── experts.py              team agents of pystk_project used as experts
│   ├── dataset.py              expert dataset D0 (executed actions + clean expert labels)
│   ├── networks.py             shared actor (tanh-Gaussian MLP), critic ensemble (LayerNorm)
│   ├── dagger.py               DAgger (iteration 0 = behaviour cloning)
│   ├── rlpd.py                 SAC + symmetric sampling (and SAC from scratch)
│   ├── evaluate.py             evaluation protocol and score
│   ├── stats.py                Welch / Mann–Whitney / bootstrap / Holm
│   └── mock_env.py             numpy stand-in for STK, only for the smoke test
├── scripts/
│   ├── smoke_test.py           end-to-end check without STK (≈ minutes)
│   ├── benchmark_experts.py    pick the fastest team agent on the track
│   ├── generate_dataset.py     build D0
│   ├── train.py                one (algo, seed) run
│   ├── run_all.py              whole study, resumable, parallel
│   ├── analyze.py              statistics, tables, figures
│   ├── make_report.py          report/report.md (+ PDF with pandoc)
│   └── watch.py                watch a trained policy / the expert drive
└── report/report_template.md   report (methodology, conceptual comparison; results auto-filled)
```

## 1. Install (Linux or WSL2 on Windows)

The SuperTuxKart bindings (`pysupertuxkart2`) are distributed for Linux; on Windows
use WSL2 (Ubuntu). This folder is expected next to the cloned course repository:

```
RL/
├── pystk_project/      git clone https://github.com/osigaud/pystk_project
└── mp2_dagger_rlpd/    this project
```

```bash
cd RL/mp2_dagger_rlpd
uv venv && source .venv/bin/activate          # or python -m venv .venv
uv pip install -r requirements.txt            # torch CPU is enough
# first STK start downloads the game assets (a few hundred MB)
```

**Windows: use WSL2.** The native Windows wheel of `pysupertuxkart2` 0.7.4 does not
bundle the DLLs it links against (SDL2, libpng16, libcurl, libfreetype, libharfbuzz,
libshaderc_shared, ...), so `import pystk2` fails with *DLL load failed*. Run the
study from Ubuntu in WSL2 (the Linux wheel is self-contained):

```bash
# PowerShell (admin), once:  wsl --install -d Ubuntu   then reboot
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv ~/venvs/stk --python 3.12 && source ~/venvs/stk/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, avoids ~2 GB of CUDA
cd /mnt/c/Users/<you>/Documents/RL/mp2_dagger_rlpd
uv pip install -r requirements.txt
python -c "import pystk2; print('pystk2 ok')"
```

The smoke test (`scripts/smoke_test.py`) does not need STK and also runs on native Windows.

If `pystk_project` lives elsewhere: `export PYSTK_SRC=/path/to/pystk_project/src`
or edit `pystk_src` in `configs/default.yaml`.

## 2. Check that everything runs

```bash
python scripts/smoke_test.py      # no STK needed, writes smoke_output/
python scripts/benchmark_experts.py --set expert.benchmark_episodes=1   # STK + the 5 team agents
```

## 3. Run the study

```bash
python scripts/run_all.py --with-sac --jobs 3
python scripts/make_report.py --repo-url https://github.com/<you>/<repo>
```

`run_all.py` does: expert benchmark → dataset D0 → `dagger`, `rlpd` (and `sac`
with `--with-sac`) for every seed → `analyze.py`. It is **resumable**: finished runs
(`runs/<algo>/seed_<k>/final.json`) are skipped, so just relaunch after an
interruption. Each run starts two STK processes (training + evaluation);
`--jobs N` runs N trainings in parallel (choose N ≤ number of cores / 2).

Useful overrides (any config key, YAML syntax):

```bash
python scripts/run_all.py --set experiment.seeds=[0,1,2,3,4]           # faster, less power
python scripts/run_all.py --set budget.online_env_steps=100000 rlpd.utd=4
python scripts/run_all.py --set env.track=lighthouse output_dir=out_lighthouse
```

Rough cost: one run = 50k training steps + ≈ 25k evaluation steps; RLPD also does
UTD × 50k critic updates. Measure the speed with the smoke test and the first real
run (`runs/*/seed_0/eval.csv`, column `wall_time_s`) before launching everything.

## 4. Outputs

* `results/expert_benchmark.json` – the 5 team agents alone on the track
* `data/expert_<track>.npz/.json` – dataset D0
* `runs/<algo>/seed_<k>/` – `config.json` (config + versions), `eval.csv`, `evals.jsonl`
  (per-episode), `final.json`, `model.pt` (`model_bc.pt` for BC), `train.log`
* `results/summary.md`, `results/stats.json`, `results/figures/*.png`
* `report/report.md` (`report.pdf` if pandoc + xelatex are installed)

Watch a policy: `python scripts/watch.py --model runs/rlpd/seed_0/model.pt`
or the expert: `python scripts/watch.py --expert team4`.

## 5. Main design decisions (details in the report)

* One fixed track (`abyss` by default), one lap, kart alone (`num_kart=1`), track
  items kept, items never fired; time limit 1500 steps = truncation (bootstrap kept).
* Same actor network for both methods; DAgger trains tanh(mean) by MSE, SAC trains
  the full tanh-Gaussian.
* D0 = 20 expert episodes, half of them with executed-action noise; DAgger uses the
  clean expert labels, RLPD the executed transitions.
* Same online budget (50k env steps), same evaluation seeds; DAgger iteration 0 is
  the BC baseline; SAC from scratch is the "no prior data" control.
* Statistics over training seeds (10 by default): Welch t-test + Mann–Whitney U,
  bootstrap CIs, Hedges g, probability of improvement, Holm correction for
  secondary tests.
* Plain PyTorch instead of rl-mind (allowed by the subject): the code follows the
  rl-mind principles (time-step accounting, separate evaluation, terminated vs
  truncated) but stays small and fully inspectable.

## 6. Troubleshooting

* `ModuleNotFoundError: pystk2_gymnasium / agents` → `pystk_src` / `PYSTK_SRC` is wrong.
* An expert prints `error: ...` → it crashed on some observation; the default action
  is used and the count is stored (`expert_errors`). Use another candidate
  (`--set expert.name=team4`).
* A package named `utils` installed in the venv would shadow `pystk_project/src/utils`
  (namespace package): uninstall it.
