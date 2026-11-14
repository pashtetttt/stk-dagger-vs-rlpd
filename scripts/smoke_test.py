"""End-to-end smoke test WITHOUT SuperTuxKart (pure-numpy mock track).

    python scripts/smoke_test.py

Runs the whole pipeline (expert benchmark, dataset, DAgger, SAC+symmetric
sampling, SAC from scratch, analysis) with tiny budgets in ./smoke_output.
It only checks that the code runs end to end; its numbers mean nothing.
Takes a few minutes on a laptop CPU.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

OVERRIDES = [
    "env.backend=mock", "env.max_steps=400", "output_dir=smoke_output",
    "expert.candidates=[mock]", "expert.benchmark_episodes=2",
    "dataset.n_episodes=4",
    "budget.online_env_steps=2000", "budget.eval_every=500", "budget.eval_episodes=1",
    "budget.final_eval_episodes=2",
    "dagger.bc_init_steps=300", "dagger.n_iterations=4", "dagger.grad_steps_per_iter=100",
    "rlpd.learning_starts=256", "rlpd.utd=1", "sac.random_steps=300", "sac.learning_starts=300",
    "experiment.seeds=[0,1]",
]

if __name__ == "__main__":
    subprocess.run([sys.executable, str(HERE / "run_all.py"), "--with-sac", "--set", *OVERRIDES], check=True)
    print("\nSMOKE TEST OK — see smoke_output/results/summary.md")
