**RLPD ends faster than DAgger.** With identical data, network and budget, SAC with
symmetric sampling reaches 57.7 s per lap [56.7, 58.7] and DAgger 60.9 s [59.8, 61.9]
(primary test: Welch p = 10⁻⁴, MWU p = 7·10⁻⁴; P(RLPD > DAgger) = 0.96). Both finish 100 % of the laps. The
evaluation turned out deterministic: the 10 episodes of a policy give identical times.
All the variance therefore comes from training, which is what the tests measure.

**DAgger degraded its own BC initialisation** (58.6 → 60.9 s, slower in all 10 seeds,
Holm p = 0.002). It never ran faster than the expert. Panel (d) shows when this
happens: the training MSE stays at its BC level while β ≥ 1/2, then doubles once the
learner drives most of the time. Score and finish rate collapse transiently around
20k steps (a, c), then plateau below BC. Our interpretation, not tested by ablation:
(i) *there was little distribution shift to fix*: BC already matched the expert to
within one step in 9/10 seeds, partly because the noisy half of D0 already
demonstrates recoveries (as in DART); (ii) *labels in learner-visited states are
harder to imitate*: team2 keeps a hidden stuck/recovery state and uses more track
information than our features, so its actions in off-trajectory states are not a
function of our observation; (iii) *aggregation dilutes the racing line*: at the end
only 11 858 of 61 858 training pairs come from the expert's own trajectories.
Possible fixes: slower β decay, retraining from scratch, re-weighting D0. None of
them would lift DAgger's ceiling above the expert.

**RLPD went beyond the demonstrator.** It finished every lap after only 5k online
steps. At 50k steps, 7 of 10 seeds were faster than the expert (best 55.9 s vs 58.5
s). The mean gain (0.8 s) is not significant against the expert or against BC (Holm
p = 0.50), so H2 is only partially supported. This is the key qualitative difference:
DAgger optimises agreement with the expert, RLPD optimises lap time. Training was
stable: no Q divergence, and α decayed to ≈ 0.003.

**Prior data make RL work here.** SAC without D0 finishes the lap in only 5 of 10
seeds and oscillates strongly (RLPD vs SAC: Holm p = 0.016 for the final score, 0.002
for the AUC). Symmetric sampling removes the exploration problem: the critic sees
high-value transitions from the first update.

**Speed and cost (H1 not supported).** The pre-registered AUC favours DAgger (1.549
vs 1.483, Holm p = 0.50), but only because DAgger starts from BC while SAC starts from
a random actor. Without the 0-step point, RLPD is ahead (1.604 vs 1.543, MWU
p = 0.076, exploratory). A DAgger run took 11 min with 50k expert queries; an RLPD run
took 34 min (98k critic updates) and needed a reward but no further expert access.

**Take-away.** When BC on a noise-augmented dataset already imitates the expert and a
dense reward exists, RL with prior data is preferable: it starts as fast and can only
improve on the expert. DAgger is the tool for compounding errors when no reward is
available, and it suffers when the expert depends on information the learner does
not observe.
