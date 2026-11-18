# Demo agents: candidates, budget, evaluation and selection rule

Written and committed before any training run whose result can be selected. The code that
implements it is `bench/train_demo.py` (training, validation, checkpoints) and
`bench/eval_demo.py` (evaluation and the rule), committed with this file.

## Format

The format the demo is played in: 4 seats (one visitor, three agents), blinds 1/2, stacks 200
(100 big blinds) reset before every hand, the engine's rules after the standard seat order (PR #2)
and the exact per-size legal mask. Training and evaluation both use it.

## Candidates

One network plays all four seats in self-play (`train_poker_dqn` with a single model, stacks
reset every hand). State: `encode_observation`, 151 floats (`docs/state_encoding.md`). Output: 7
Q-values. Only the layers the repository's models already use (Linear, ReLU, GRU).

| candidate | network | parameters | float32 size |
|---|---|---:|---:|
| `mlp-s` | `MLPModel`, 151 → 128 → 128 → 7, ReLU, no dropout | 36,871 | 0.14 MB |
| `mlp-l` | `MLPModel`, 151 → 512 → 512 → 256 → 7, ReLU, no dropout | 473,607 | 1.81 MB |
| `gru` | `GRUModel`, one GRU layer of 128 over the hand so far, then Linear 128 → 7 | 108,807 | 0.42 MB |

The `gru` sees the sequence of its own encoded states at each of its decisions in the current
hand, the current one last, starting empty every hand (`docs/state_encoding.md`, last section).
All three are far under the 8 MB limit.

## Training

- Seeds 1, 2 and 3 for every candidate: 9 runs. A seed fixes Python's `random`, NumPy, torch
  (weights and sampling) and the engine's deals (`GameEngine.seed(seed)` once, then the deals
  follow from it). One torch thread per run.
- Budget: **400,000 hands per run for `mlp-s` and `mlp-l`, 150,000 for `gru`.** In the pilots a
  GRU hand cost about 15 times an MLP hand on this CPU (17 against about 250 hands per second, one
  thread; the profiled GRU's self-play hands averaged 55 decisions, many of them one-big-blind
  raises), so
  400,000 GRU hands would take over 6 hours per run. The comparison between families is therefore
  at unequal hands, and is reported as such.
- DQN as in `ReinforcementLearningEnvironment` after this branch's fixes: every seat that acted
  gets its result; replay buffer of 100,000 transitions; one update per hand on a batch of 256;
  Adam, learning rate 1e-4; gamma 1.0 (a hand is the episode); Double DQN targets over legal
  actions only; target network copied every 200 hands; reward = the hand's result in big blinds
  × 0.01; epsilon-greedy from 1.0 to 0.05 over the first fifth of the run's hands (80,000 for the
  MLPs, 30,000 for the GRU), then 0.05.
- Opponents during training: only the network itself. No baseline bot is used in training.

## Checkpoints (validation deals)

Every tenth of the run (10 checkpoints: every 40,000 hands for the MLPs, 15,000 for the GRU) the
network plays greedily (no exploration, `eval()` mode) against each of the three baselines on
**validation deals**: 1,000 duplicate deals from seed 5000.
A checkpoint's validation score is the rule's score (below) on these deals. Each run keeps the
checkpoint with the best validation score (the earliest on a tie); that is the run's candidate
network. The evaluation deals are not used for this.

## Evaluation (evaluation deals)

`bench/winrate.py`'s method, unchanged: a deal is played 4 times with the same cards, the agent
taking each seat once and three copies of one baseline in the other seats; the deal's score is the
agent's mean result over the 4. Baselines `random`, `call` and `tight`, unchanged (their code; a
note added after the runs, under *Records*, says how the exact mask changes what they do); the
random bot in the agent's seat is the control.

- **40,000 duplicate deals** (160,000 hands) per candidate and baseline, from seed 7000 (deck seed
  7000 × 1,000,003 + deal number, the random bot's choices from `random.Random(7000)`).
- These deals are used only here: not in training (whose deals follow from seeds 1–3), not for
  validation (seed 5000), not for choosing checkpoints. (Added after the runs: the smoke test
  disclosed under *Budget* played pilot networks on the first 200–300 of them, and the parity
  records for the next stage are recorded from the selected agent's play on the first of them.)
- bb/100 = 100 × the mean deal score; 95% interval = mean ± 1.96 × sd / √40,000 over deals.

## Rule

1. Each of the 9 networks gets a **score**: the lowest, over the three baselines, of the lower end
   of its 95% interval on bb/100 on the evaluation deals.
2. Rank by score, highest first; ties by the mean bb/100 over the three baselines, then by fewer
   parameters.
3. **Seats:** the three top-ranked networks, one per agent seat, in rank order. Exception: if the
   top network is clearly best, meaning its score is higher than every other network's lowest
   point estimate of bb/100 over the three baselines, it plays all three seats.
4. Whether or not the selected agents beat the baselines, the results are reported as measured, and
   the rule, the candidates, the budget and the evaluation are not changed after the evaluation.

## Budget

At most about 6 hours of training wall-clock on the shared machine, counting elapsed time (runs in
parallel overlap), the pilot runs included; total CPU time is reported as well. If the agents are
still weak when the budget is spent, that is the result.

Pilot runs used seeds other than 1–3 (101–104) and the validation deals, to check the pipeline
and measure speed; their results cannot be selected and are not used. One exception, disclosed:
while testing `bench/eval_demo.py` and `bench/export_weights.py` end to end, three pilot networks
(`mlp-s` and `mlp-l` seed 101, `gru` seed 103; not candidates) were played on the first 200–300
evaluation deals. The candidates, the evaluation and the rule were fixed before that. The GRU's
hand budget was cut afterwards, from 400,000 to 150,000, for speed alone (a profile of a pilot GRU
on its own training hands: 17 hands per second, 55 decisions per hand), not on those results.

## Amendment (18 November, committed at 00:40:48 GMT in db21bf8, during training, before any candidate was evaluated)

From about 00:10 another job on the machine (twelve training processes) raised the load average
from about 10 to 200–360, and each of these runs now gets about a third of a core. At that rate
the GRU runs would need 5–7 more hours. So that the 6-hour budget holds: training stops at
**05:20 GMT** (the pilots took about 0.4 hours from 23:29; the runs started at 23:53). A run that
has not finished by then is stopped, and its candidate is the checkpoint it kept up to its last
validation, exactly as for a finished run; its `run.json` records that it was stopped and after
how many hands. Nothing else changes: candidates, seeds, validation, evaluation and the rule stay
as above. The `mlp-s` runs were started by hand at 00:17, alongside the others instead of after
them, for the same reason.

## Records (added after the runs)

- **Start times.** The nine runs' `run.json` have no absolute start time (`train_demo.py`
  records `started_utc` from the commit that added it, after the runs). Their start times are
  inferred: the commit time of each `run.json` (committed within seconds of the run ending, by a
  script that watched for it) minus its `wall_s`. That gives 23:53:28–23:53:41 GMT for the six
  runs started by `--all` (their directories were created at 23:53:16) and 00:17:23–00:17:34 for
  the three `mlp-s` runs.
- **Early look at results.** The evaluation was not run after all training ended. The six MLP
  candidates and the random-bot control were evaluated on the evaluation deals from 02:53 to
  03:14 GMT (`bench/raw/demo_eval_early_1.log`), and `gru` seed 2 from about 04:20 to 04:23
  (`demo_eval_early_2.log`), while GRU runs were still training (`gru` seed 2 ended at 04:15,
  seeds 3 and 1 at 04:36 and 04:40). Those partial tables, which already put `mlp-s` seed 3 first,
  were seen before the last two GRU runs ended. Nothing was changed after them: the training code,
  the budgets (the GRU runs ran their full 150,000 hands), the 05:20 cut-off (committed at
  00:40:48, before any candidate was evaluated) and the rule are as committed earlier, and the
  final evaluation reused those cells unchanged. The times come from the two logs and from the
  cell files' modification times on this machine (`bench/raw/demo_cells_times.txt`); the cells
  computed then have no `computed_utc` or `commit` field, since neither is recoverable from a
  committed record. Cells computed from now on record both.
- **Baselines and the exact mask** (a clarification added after the runs). The baselines' code is
  unchanged, but each chooses from the legal mask it is given, and the mask is now exact per raise
  size. So they behave as written for an exact mask, not as they did in PR #2's measurements, and
  PR #2's win rates are not comparable with these. `python bench/check_baseline_masks.py`
  (`bench/raw/check_baseline_masks.txt`, 2,000 deals against the selected agent): the old mask
  would have differed from the exact one in 45.5% of the random bot's decisions (so its choices
  are drawn from a different set), 9.2% of the calling bot's (whose choices do not change: it
  never raises) and 32.4% of the tight bot's, whose action changes in 12 of 43,502 decisions (it
  now calls where the old fallback made it raise half the pot).
