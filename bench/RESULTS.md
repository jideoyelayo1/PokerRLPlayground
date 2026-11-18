# Measurements

Every number here comes from a command in this repository. Raw output is in `bench/raw/`.
Re-run: `sh bench/run_benchmarks.sh` (tests and throughput), `sh bench/run_benchmarks.sh winrate` (the 2,000-hand training and win rate of section 5) and `sh bench/run_benchmarks.sh demo` (the demo agents, section 6).

**Code measured.** Section 6 (the demo agents) and the checks in section 2c were measured on this branch (`train/demo-agents`), after the training-loop fixes listed in 2c; each raw file names its commit. Sections 1, 3, 4 and 5 were measured on commit 2a3e3df, which gives the engine the standard hold'em seat order (section 2a), with one run of `sh bench/run_benchmarks.sh winrate`; section 5 is superseded by section 6 (it describes the training loop before those fixes, at 5,000-big-blind stacks). The numbers in earlier versions of this file (in the git history, up to commit 2c79230) were measured with the engine's old, non-standard seat order, and are not comparable cell for cell. Sections 2 and 2b describe faults found and fixed on older code (2b's check was run again on 2a3e3df); the seed-2 rerun in section 4 was not repeated.

**Machine.** Apple M1 Pro (8 performance + 2 efficiency cores, 16 GB), macOS 27.0.1 (Darwin 27.0.0).
Apple clang 21.0.0 (clang-2100.3.34.2), CMake 4.4.0, Release build (`-O3 -DNDEBUG`, C++20).
Python 3.14.6, torch 2.9.1 (CPU, 8 threads), numpy 2.4.1, pybind11 3.0.1; the extension was built by `setup.py` (`-O3`).
The machine was shared with other jobs during the runs; each raw file records the load average before and after.

## 1. Hand evaluator

`ctest --test-dir build` runs `test_hand_eval` and `eval_exhaustive` ([raw](raw/eval_exhaustive.txt)).

- **Known answers** (`PokerEngine/tests/test_hand_eval.cpp`): 48 checks covering every category, kicker vectors, the wheel and steel wheel, six-card straights, flush over straight, two sets of trips, three pairs, counterfeited two pair, board-plays splits and the category ladder. 0 failures.
- **Exhaustive cross-check** (`PokerEngine/tests/eval_exhaustive.cpp`). Every 5-card hand (2,598,960) through `evaluate5` and every 7-card hand (133,784,560) through `GameEngine::bestHandOf7`, each compared field by field (category and kicker vector) with a reference evaluator written for the test. The reference works from rank counts and suit bitmasks; it does not try the 21 five-card subsets. **0 disagreements** in 136,383,520 hands. A planted kicker bug in the reference gave 41,184 disagreements over the 5-card hands, so the comparison does catch errors.
- **Published tables**: category counts and distinct hand values per category all match.

| category | 7-card hands | distinct values (7-card) | 5-card hands | distinct values (5-card) |
|---|---:|---:|---:|---:|
| Straight flush | 41,584 | 10 | 40 | 10 |
| Four of a kind | 224,848 | 156 | 624 | 156 |
| Full house | 3,473,184 | 156 | 3,744 | 156 |
| Flush | 4,047,644 | 1,277 | 5,108 | 1,277 |
| Straight | 6,180,020 | 10 | 10,200 | 10 |
| Three of a kind | 6,461,620 | 575 | 54,912 | 858 |
| Two pair | 31,433,400 | 763 | 123,552 | 858 |
| One pair | 58,627,800 | 1,470 | 1,098,240 | 2,860 |
| High card | 23,294,460 | 407 | 1,302,540 | 1,277 |
| **Total** | **133,784,560** | **4,824** | **2,598,960** | **7,462** |

Wall time: 63 s on 10 threads.

## 2. Betting rounds (bug found and fixed)

`PokerEngine/tests/test_betting.cpp` drives the `step()` API, the one the training loop uses. It checks that:
- a street closes after one action per live seat;
- every seat acts exactly once per street when everyone calls or checks (2 and 4 seats, every blind position);
- in a stress test with six policy mixes, each with stacks reset every hand and as a running tournament (5,000 hands each, engine seed 3, policy seed 4), every hand finishes within the training loop's 2,000-step cap, the player to act always has a legal action, and chips are conserved.

On the engine before the fix, 36 of 124 checks fail ([raw](raw/test_betting_before_fix.txt)). After the fix, 0 fail ([raw](raw/test_betting_after_fix.txt)). The test has since grown to 239 checks with the seat-order tests (section 2a), all passing.

| policy mix (4 seats unless noted) | stacks | hands not finished in 2,000 steps, before fix |
|---|---|---:|
| random + 3 x call | reset each hand | 1,280 / 5,000 |
| random + check-fold + 2 x call | reset each hand | 2,480 / 5,000 |
| 3 x call + check-fold | tournament | 1,666 / 5,000 |
| heads-up random vs call | tournament | 881 / 5,000 |
| random + 3 x call | tournament | 38 / 5,000 |

Also before the fix: 15 + 7 hands reached a player with no legal action, 12 hands destroyed chips, one seat never acted after the flop, and preflop some seats acted twice.
The cause: a street closed only when the action returned to one marker seat with all bets matched. If that seat had folded or was all-in, the action never reached it. The 12 chip losses were side pots with no eligible winner, after a player who had put in more than anyone else folded out of turn. The fix tracks which live seats still have to act.

## 2a. Seat order (changed to standard hold'em)

Up to commit 212c4da the small blind was posted by the seat after the big blind and acted last on every street; heads-up, the big blind acted first on every street. Commit 2a3e3df uses the standard order, written out in the README (*Seats*, *Busted players*): a button that moves to the next seat still in the tournament each hand, small and big blind on the two seats after it (heads-up the button posts the small blind), the seat after the big blind first preflop with the big blind last, and the first seat after the button that can still act first on the flop, turn and river.

`test_betting` checks this through the `step()` API:
- the blinds and the order of action on every street, for 2, 3, 4 and 6 seats, over a full turn of the button and one more hand;
- the big blind's option when everyone calls (it may check or raise, and a raise sends the action round again), for 2, 3, 4 and 6 seats;
- two running tournaments with 20-big-blind stacks (4 and 6 seats, 5,000 hands each, random, all-in, calling and check-fold players), checked against the rule on every hand and every street. The test fails unless, in each tournament, the button and both blinds have busted, the table has gone heads-up, and the first seat after the button has folded and has been all-in at the start of a street, at least once each.

Compiled against the engine before the change (212c4da), 102 of 233 checks fail ([raw](raw/test_seat_order_before_fix.txt)); against `main` (2658c62) the order checks fail the same way and the binary then crashes (see 212c4da below). After the change, all 239 checks pass ([raw](raw/test_seat_order_after_fix.txt)). One existing check encoded the old order (`closingSeatFolds`: its third player checked, as the big blind) and now calls or checks.

Commit 212c4da fixes a memory fault found while writing these tests: between hands, `getObservation()` read the current-player iterator after `reset_tournament()` or a bust had freed its node (the training loop reads `getObservation(0)` at exactly that point), and on a new engine before it was ever set (a segfault). No caller in the repository uses the current player between hands.

## 2b. Training-log faults (found and fixed)

`python bench/check_training_log.py` runs 30 hands of `train_poker_dqn` (README models, seed 0) and checks the per-hand log ([raw](raw/check_training_log.txt)).
On the original training loop (commit 6727091: fixed engine, old `Poker.py`), no row has `folded=1`, and the four players' `bb_delta` sum to +1.5 BB on every hand.
On commit 2a3e3df (standard seat order), folded players are logged and the sum is 0.0 on every hand.
The causes:
- stacks were read after the blinds;
- pocket cards were read at the end of the hand, when a folded player's observation hides them;
- busted players were logged again with stale cards.

In the three 2,000-hand training logs below, 9.3–32.6% of rows are folds ([winrate_table.md](raw/winrate_table.md)), and the old logging dropped every one of them.
The EV-per-starting-hand figures in the README were therefore averaged only over hands that were not folded. They have not been regenerated.

## 2c. Training-loop and legality faults (found and fixed on `train/demo-agents`)

- **Reward for players who did not act last.** `python bench/check_terminal_rewards.py` trains four small networks for 300 hands and records every transition stored ([raw](raw/check_terminal_rewards.txt)). On PR #2's loop (876dd82), 299 of 656 seat-hands in which the seat acted ended with a terminal transition carrying the seat's result, one per hand (the player whose action ended it); 897 results were stored without a state and never trained. On this branch, 707 of 707 (tournament) and 1,198 of 1,198 (stacks reset every hand) do, and none is lost.
- **Exact legal actions.** `legalActions` now lists each raise size separately, marked legal exactly when `step()` accepts it from the player to act (`step()` always acts for the current player, and a seat that is not to act gets its actions listed as if it were). `test_betting`'s `legalActionsAreExact` plays random legal actions at 100 and 20 big blinds and, at every decision, tries every action marked illegal (each is rejected and changes nothing) and plays one marked legal (accepted; a raise raises to its listed `raise_to`) ([raw](raw/test_legal_actions.txt)). At 100 big blinds, in 9,412 of 19,697 decisions the old mask marked a raise size legal that the engine rejects, and the training loop then played another size or action instead.
- **Repeatability.** With the 2-second wall-clock limit gone, `sh bench/check_repeatable.sh` trains with one seed twice, in separate processes, and compares ([raw](raw/check_repeatable.txt)). The README set-up (four networks, tournament at 10,000 chips, seed 2, 2,000 hands) gave the same 5,129-row training log, apart from the per-process `run_id`, and the same weights hash both times, although the machine was heavily loaded and the two runs took 3,895 s and 3,444 s; with the timeout, a rerun of seed 2 (old seat order) left its first run's log at hand 114 (section 4). Short demo runs (`mlp-s` for 20,000 hands and `gru` for 4,000, seed 11) gave identical checkpoint logs, apart from timings, and identical kept weights. Not checked: repeatability on another machine or with another number of torch threads.
- **Other changes to the loop**, each its own commit with its reason: the state is encoded relative to the acting player, with its amount to call and the button's position (`docs/state_encoding.md`; `Observation.button` is new); one network can play every seat; stacks can be reset every hand; exploration decays over a set number of hands instead of stopping after 80; the bootstrap target maxes over legal actions only, optionally with Double DQN; the per-decision single-sample update is gone; the replay buffer is a list ring instead of a deque (`random.sample` indexes a deque in O(n) per item: `python bench/bench_replay_sample.py` drew 1,024 transitions from a million in 327 ms from a deque and 0.50 ms from a list, [raw](raw/bench_replay_sample.txt); the 51 ms quoted earlier came from an unrecorded run and is withdrawn); recurrent models can be given the hand so far. `python -m unittest discover -s tests` checks the pieces.

## 3. Engine throughput

4 seats, stacks reset to 10,000 every hand, blinds 1/2, fixed policies, single thread, 7 runs (seeds 1–7), standard seat order (commit 2a3e3df). Median, with the min–max over the runs in brackets.

| path | policy | hands/s | decisions/hand | raw |
|---|---|---:|---:|---|
| C++, `playOneHand` (controller API) | everyone calls/checks | **57,580** [56,702–58,091] | 16.0 | [bench_engine.txt](raw/bench_engine.txt) |
| C++, `step()` | everyone calls/checks | **43,757** [42,982–44,172] | 16.0 | same |
| C++, `step()` | uniform random legal action | **71,401** [70,492–71,674] | 5.6 | same |
| Python binding, `step()` | everyone calls/checks | **13,492** [13,221–13,765] | 16.0 | [bench_binding.txt](raw/bench_binding.txt) |
| Python binding, `step()` | uniform random legal action | **13,029** [12,818–13,199] | 5.7 | same |
| Python, training loop without the networks | random legal action index | **1,863** [1,850–1,888] | 7.3 | same |

`./build/bench_engine 200000 7` (200,000 hands per run) and `python bench/bench_binding.py --hands 20000 --runs 7` (20,000 hands per run; 5,000 for the last row).
The last row is the per-decision work `train_poker_dqn` does outside the networks: `getObservation`, `encode_observation`, `legal_mask_from_obs`, `step_with_fallback`, then `getObservation` and `encode_observation` again.
The load average at the start of the C++ run was 15, mostly from the exhaustive test that had just finished; within each row the slowest and fastest of the seven runs differ by at most 2.8% (C++) and 4.1% (binding).
The medians are 2–4% above those measured with the old seat order, when the machine was busier (load average 106 at the start); that difference is not attributed to the change.

## 4. Training cost

The README's four models (MLP 512-1024-2048-1024-512, LSTM and GRU 1024 x 3, Transformer d_model 256 x 3 layers), CPU, torch 8 threads, 2,000 self-play hands per seed ([winrate.log](raw/winrate.log)):

| seed | wall time | hands/s | CPU-s per hand | aborted hands | tournament resets |
|---|---:|---:|---:|---:|---:|
| 1 | 12.0 min | 2.79 | 0.87 | 15 | 324 |
| 2 | 10.8 min | 3.10 | 0.79 | 2 | 462 |
| 3 | 15.6 min | 2.14 | 1.22 | 41 | 792 |

Median 2.79 hands/s (0.36 s per hand). The engine, binding and state encoding take about 0.54 ms per hand at 7.3 decisions per hand (section 3, last row), under 0.2% of that. The rest is the networks: a single-sample update at every decision, plus a batch-1024 replay update per agent per hand.
At the measured rate, 30,000 hands would take about 3 hours of wall time and 7–10 CPU-hours. That is an extrapolation, not a measurement.
With the old seat order the same budget took 9.5–11.7 minutes and 0.62–0.82 CPU-seconds per hand per seed. The cause of the difference was not investigated.

**Aborted hands and repeatability.** After the betting fix, the hands the training loop aborts come from its 2-second wall-clock timeout (`HAND_TIMEOUT_SEC`), not from the engine. All 58 aborted hands above are `timeout` (recorded in the per-seed JSON). There were more than with the old seat order (0, 8 and 3), although the load average was lower (about 4 during this run, 26 at the start of the old one). Why was not investigated; long hands with many raises and calls, each decision carrying a network update, are one possible cause.
With the old seat order (before 2a3e3df), seed 2 was trained again from scratch with abort reasons recorded: `python bench/rerun_training.py --seed 2` ([raw](raw/rerun_seed2.txt), log [rerun_train_seed2.csv](raw/rerun_train_seed2.csv)). It aborted 7 hands (the original run aborted 8), all `timeout`.
Its training log matches that of the original old-order run (the `train_seed2.csv` in commit 2c79230) row for row, apart from the per-process `run_id`, up to row 285, hand 114, and differs from there on.
A timeout cuts a hand after however many decisions the machine managed in 2 seconds, and each decision has already applied a single-step update to the network. So the weights, and everything after, depend on machine load.
Seeded training therefore repeated only until the first timeout. The timeout is now removed (the 2,000-step cap remains) and seeded runs repeat exactly on the same machine, software and number of threads (section 2c). The rerun above was not repeated with the standard seat order on the old loop.

## 5. Win rate against fixed baselines (superseded: 2,000 hands, the training loop before section 2c)

Superseded by section 6. This table describes the training loop as it was before the fixes in section 2c, the README's four networks trained together for 2,000 hands at 5,000-big-blind stacks; it is kept as a record of that code. Its baselines chose from the old, inexact mask, so they did not behave exactly as section 6's do, and the two sets of win rates are not comparable.

`python bench/winrate.py --seeds 1 2 3 --train-hands 2000 --eval-deals 1000`, then `python bench/summarise_winrate.py` ([table](raw/winrate_table.md), [per-seed JSON](raw/), [training logs](raw/)).

**Training budget.** 2,000 self-play hands per seed with `train_poker_dqn` as in `main.ipynb`: four seats (MLP, LSTM, GRU, Transformer), stack 10,000, blinds 1/2, CPU. Seeds 1, 2 and 3 fix Python `random`, NumPy, torch and the engine's deals. 10.8–15.6 minutes of training per seed; training and evaluation together took 49 minutes (42 with the old seat order). The README's figures are labelled 50,000 and 100,000 hands, so this is 2–4% of that budget.

**Evaluation.** One trained agent (greedy: epsilon 0, `model.eval()`) sits at a 4-seat table with three copies of one baseline bot, with stacks reset to 10,000 (5,000 big blinds) every hand. Duplicate deals: each deal is played 4 times with the same cards, the agent taking each seat once, and the deal's score is the agent's mean result over the 4. That is 1,000 deals = 4,000 hands per cell. The 95% CI within a seed is mean ± 1.96 sd/√1000 over deals. Across seeds it is a t-interval (2 degrees of freedom) over the three per-seed values.
Baselines: **random** (uniform over the agent's legal action indices), **call** (calls a bet, otherwise checks), **tight** (preflop: pot-raise TT+, AK, AQ; call other pairs, two broadway cards or suited aces; otherwise check or fold. Postflop: call or check with a pocket pair or a hole card that pairs the board, otherwise check or fold).
**random-bot** is a control: the random bot in the agent's seat.
With duplicate deals, an agent that acts exactly like the baseline scores exactly 0. One that always folds scores −37.5 (the blinds).

bb/100, mean over the three seeds [95% t-CI over seeds]:

| agent | vs random | vs call | vs tight |
|---|---:|---:|---:|
| MLP | −2,756 [−36,070, 30,558] | −9 [−125, 107] | −27,892 [−111,421, 55,637] |
| LSTM | 8,669 [−18,055, 35,393] | −14 [−65, 37] | −9,681 [−30,793, 11,430] |
| GRU | 17,619 [−17,136, 52,374] | 46 [−163, 256] | −28,997 [−121,255, 63,261] |
| Transformer | 21,405 [−3,212, 46,023] | 9 [−40, 59] | −3,347 [−9,543, 2,849] |
| random-bot (control) | −374 [−15,841, 15,092] | 2,864 [−15,863, 21,591] | −31,271 [−37,252, −25,290] |

Per-seed values with their own CIs are in [winrate_table.md](raw/winrate_table.md).

**Reading it.**
- **No trained agent beats any baseline reliably at this budget.** Every across-seed interval, against all three bots, contains zero. (With the old seat order the LSTM's interval against the tight bot excluded zero, as a loss; it no longer does.)
- **Against the calling bot,** every agent is within 150 bb/100 of zero, and 4 of the 12 seed-agent pairs score exactly 0 with zero spread, i.e. they play exactly like a calling station at that table. Seed 2's LSTM scores exactly −37.5 against the calling bot, the score of folding every hand; against the tight bot it scores −18 [−19, −17].
- **Against the tight bot,** 9 of the 12 seed-agent pairs lose with a per-seed interval below zero. The other 3 (GRU seed 3, +2,389 [−4,823, 9,602]; Transformer seed 1, −2,357 [−9,856, 5,142]; Transformer seed 3, −1,500 [−8,206, 5,207]) have intervals that include zero; none wins clearly. The Transformer's across-seed mean is the closest to zero, −3,347 [−9,543, 2,849]. The random control loses −31,271 [−37,252, −25,290].
- **Against the random bot,** five single seeds win clearly: LSTM seed 1, +20,761 [7,858, 33,665]; GRU seed 2, +20,423 [7,160, 33,686]; GRU seed 3, +29,995 [16,863, 43,127]; Transformer seed 1, +24,115 [11,408, 36,821]; Transformer seed 3, +29,678 [16,363, 42,993]. The Transformer is positive in all three seeds (seed 2: +10,423 [−1,940, 22,787]), but its across-seed interval still includes zero. MLP seed 3 loses (−12,168 [−23,647, −689]). The random control itself ranges from −7,017 to +5,326 across seeds.
- **The sizes are in the tens of thousands of bb/100** because stacks are 5,000 big blinds deep and many policies collapse towards moving all-in. In the last quarter of training, 5 of the 10 seats still being dealt in opened with all-in in 91–100% of their hands; of the other 5, two mostly called (75% and 95% of their hands) and the three Transformers mostly raised by one big blind (45–80%) or folded (table in [winrate_table.md](raw/winrate_table.md); this is training-time play, with dropout on).
- **The training set-up limits learning, independent of budget.**
  - Only the player whose action ends a hand ever receives a reward (README, *Multi-player credit assignment*).
  - A busted agent sits out until the tournament resets, so in seed 2 the last 500 hands were heads-up between the GRU and the Transformer, while the MLP and LSTM kept replaying old transitions.

**Provenance.** The three training runs used commit 2a3e3df, the code described here. The per-seed JSON records each aborted hand's reason.

## 6. Demo agents: training, selection and win rate at 100 big blinds

The agents for the browser demo, chosen under the rules in [`SELECTION.md`](SELECTION.md), committed at 23:53:04 GMT on 17 November (abdf6f8), before the first selectable run started (23:53:16). Re-run: `JOBS=6 sh bench/run_benchmarks.sh demo`; the weights are in [`weights/`](../weights/README.md).

**Format.** 4 seats, blinds 1/2, stacks 200 (100 big blinds) reset before every hand, the engine after the standard seat order and the exact legal mask. One network plays all four seats in self-play; the training loop is the one in section 2c (every seat that acted gets its result, Double DQN over legal actions, decaying exploration, no wall-clock limit).

**Candidates.** `mlp-s`: MLP 151-128-128-7 (36,871 parameters, 0.14 MB). `mlp-l`: MLP 151-512-512-256-7 (473,607, 1.81 MB). `gru`: one GRU layer of 128 over the seat's own decisions in the hand so far, then a linear layer (108,807, 0.42 MB). Seeds 1–3 each. Budget 400,000 hands for the MLPs and 150,000 for the GRU (in the pilots, one process each, a GRU hand took about 15 times the wall time of an MLP hand; in these runs, under load, about 7 times an `mlp-s` hand in wall time and about 11 times in CPU time: 17 against 188 hands per CPU-second). Each run kept the checkpoint (one every tenth of the run) that scored best on the validation deals (1,000 duplicate deals from seed 5000).

**Evaluation.** Every kept network against three copies of each baseline, greedy, on 40,000 duplicate deals (160,000 hands) from seed 7000, which no training, validation or checkpoint choice used; before the evaluation, a smoke test of the scripts played pilot networks (not candidates) on the first 200–300 of them (`SELECTION.md`), and after it the parity records were taken from the selected agent's play on the first of them. bb/100 [95% interval over deals]. Score: the lowest, over the three baselines, of the interval's lower end. The baselines' code is unchanged, but they choose from the now exact mask, so they behave as written for an exact mask: against the selected agent the old mask would have differed in 46.1% of the random bot's decisions and changed the tight bot's action in 12 of 43,502 ([raw](raw/check_baseline_masks.txt)). Section 5's win rates (and PR #2's) are not comparable with these. Raw: [demo_table.md](raw/demo_table.md), [demo_eval.json](raw/demo_eval.json), one file per cell in [demo_cells/](raw/demo_cells/), log [demo_eval.log](raw/demo_eval.log).

| run | parameters | kept checkpoint (hands) | vs random | vs call | vs tight | score (worst lower bound) | rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp-s-seed3 | 36,871 | 320,000 | 359 [332, 386] | 341 [323, 359] | 16 [11, 21] | 11 | 1 |
| gru-seed3 | 108,807 | 150,000 | 809 [778, 841] | 293 [272, 314] | 3 [-5, 11] | -5 | 2 |
| mlp-s-seed2 | 36,871 | 280,000 | 338 [311, 365] | 203 [189, 218] | -2 [-9, 5] | -9 | 3 |
| mlp-s-seed1 | 36,871 | 400,000 | 378 [352, 404] | 99 [87, 111] | -14 [-19, -9] | -19 | 4 |
| gru-seed1 | 108,807 | 75,000 | 958 [924, 992] | 87 [70, 104] | -25 [-34, -17] | -34 | 5 |
| mlp-l-seed3 | 473,607 | 120,000 | 654 [620, 688] | 125 [109, 140] | -39 [-48, -30] | -48 | 6 |
| mlp-l-seed2 | 473,607 | 200,000 | 351 [325, 378] | -30 [-39, -21] | -42 [-50, -34] | -50 | 7 |
| mlp-l-seed1 | 473,607 | 120,000 | 497 [469, 525] | 223 [202, 244] | -55 [-64, -46] | -64 | 8 |
| gru-seed2 | 108,807 | 90,000 | 916 [880, 951] | 259 [236, 281] | -58 [-68, -48] | -68 | 9 |
| random-bot (control) | | | -0 [-40, 40] | -128 [-167, -88] | -669 [-692, -646] | | |

Across seeds, per architecture: mean of the per-seed bb/100 [95% t-interval over seeds].

| candidate | seeds | vs random | vs call | vs tight |
|---|---:|---:|---:|---:|
| mlp-s | 3 | 358 [309, 408] | 214 [-88, 516] | 0 [-37, 37] |
| mlp-l | 3 | 501 [125, 876] | 106 [-211, 422] | -45 [-66, -25] |
| gru | 3 | 894 [704, 1,084] | 213 [-61, 487] | -27 [-103, 50] |

Training (bench/train_demo.py; wall time with the other runs in parallel; decisions per hand of self-play over the last tenth of the run).

| run | hands | wall | CPU | hands/s | decisions/hand | kept checkpoint | its validation score |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru-seed1 | 150,000 | 287 min | 2.56 h | 9 | 27.7 | 75,000 | -68 |
| gru-seed2 | 150,000 | 262 min | 2.15 h | 10 | 16.1 | 90,000 | -77 |
| gru-seed3 | 150,000 | 283 min | 2.50 h | 9 | 20.2 | 150,000 | -74 |
| mlp-l-seed1 | 400,000 | 177 min | 1.31 h | 38 | 14.6 | 120,000 | -88 |
| mlp-l-seed2 | 400,000 | 179 min | 1.33 h | 37 | 13.6 | 200,000 | -93 |
| mlp-l-seed3 | 400,000 | 175 min | 1.31 h | 38 | 13.2 | 120,000 | -106 |
| mlp-s-seed1 | 400,000 | 96 min | 0.58 h | 69 | 18.1 | 400,000 | -38 |
| mlp-s-seed2 | 400,000 | 97 min | 0.59 h | 68 | 14.0 | 280,000 | -33 |
| mlp-s-seed3 | 400,000 | 99 min | 0.60 h | 67 | 16.4 | 320,000 | 4 |

Seats: mlp-s-seed3 in all three (clearly best).
Beats each baseline (lower end of the 95% interval above 0): mlp-s-seed3: random yes, call yes, tight yes.

**Reading it.**
- **The selected agent beats all three baselines, on fresh deals.** `mlp-s` seed 3 (the kept checkpoint at 320,000 hands) ranks first with a score of +11 bb/100 and is clearly best under the rule (+11 is above every other network's worst point estimate, the next being +3), so it plays **all three agent seats**. Its figures on the evaluation deals are the best of nine scored on those same deals, so they are a little optimistic; the figures to quote are from 40,000 fresh duplicate deals (seed 9000, used for nothing else; `python bench/confirm_demo.py`, [raw](raw/demo_confirm.log)): **random +372 [345, 398], call +332 [314, 350], tight +18 [14, 23] bb/100**. On the evaluation deals: +359 [332, 386], +341 [323, 359], +16 [11, 21]. The random bot in its seat scores 0, −128 and −669.
- **The tight bot is the hard one.** Every network beats the random bot, by 338–958 bb/100, and eight of nine beat the calling bot. Against the tight bot, two other networks have intervals that include zero (`gru` seed 3, +3 [−5, 11]; `mlp-s` seed 2, −2 [−9, 5]) and the other six lose to it with intervals below zero. Across seeds, `mlp-s` is 0 [−37, 37] against the tight bot, so the win depends on the seed: one of three `mlp-s` runs gives an agent that beats it.
- **Bigger was not better.** `mlp-l` (13 times the parameters) is worse than `mlp-s` against the tight bot in every seed, and its validation score peaked at 120,000–200,000 hands and then fell (`weights/runs/mlp-l-seed*/log.jsonl`); its kept checkpoints come from that peak. `gru` trained for 150,000 hands, not 400,000 (`SELECTION.md`), so the families are compared at unequal training. The GRU beats the random bot by the most (809–958) but does no better than `mlp-s` against the tight bot.
- **What the selected agent does** (actions chosen in evaluation): against the tight bot it folds 12% of its decisions, checks 34%, calls 20%, raises by one big blind 33% and rarely raises half the pot (0.2%), the pot (1.6%) or goes all-in (0.3%); against the calling bot it checks 41% and raises by one big blind 24%; against the random bot it folds 39% and calls 37%. Raising by the minimum is its main bet. **It often folds when it could check for free:** replaying the evaluation deals with a count (`python bench/confirm_demo.py --seed 7000 --out demo_free_folds`, which reproduces the three results above exactly; [raw](raw/demo_free_folds.log)), it folded with check legal in 53,190 of its 458,747 decisions against the calling bot (11.6%; 62% of its folds there), 19,323 of 362,082 against the tight bot (5.3%; 46% of its folds) and 2,532 of 325,675 against the random bot (0.8%; 2% of its folds). In the parity records, 47 of its 261 folds are free folds. Such a fold gives up the pot for nothing; a visitor will see it (see *After selection* below). In self-play, hands average 13–28 decisions late in training (7–8 early), because the networks re-raise each other by one big blind many times; a visitor will see this.
- **Seeds repeat** on the same machine, software and number of threads: `sh bench/check_repeatable.sh` (section 2c) trains the same seed twice and gets the same weights. No hand was aborted in any of the nine runs.

**After selection: a play-time rule against folding when checking is free.** Decided after the selection, not part of the pre-registered rule, and it does not change which network was selected. Because the selected agent folds when it could check for free (above), and a visitor would see that as broken, the play-time option `no_free_fold` was added: if Check is allowed, Fold is treated as not allowed (applied to the mask before the argmax; `docs/state_encoding.md`). It is off in training and in the evaluation above. `python bench/confirm_demo.py --seed 13000 --deals 40000 --no-free-fold both --out demo_no_free_fold` plays the selected agent with the rule off and on, on the same 40,000 new duplicate deals from seed 13000, which nothing else used ([raw](raw/demo_no_free_fold.log)):

| `mlp-s-seed3` | vs random | vs call | vs tight |
|---|---:|---:|---:|
| rule off | 396 [369, 422] | 315 [298, 333] | 17 [12, 21] |
| rule on | 365 [338, 392] | 357 [340, 375] | 27 [23, 32] |
| on minus off, deal by deal | −30.9 [−62.4, 0.6] | 42.1 [38.3, 45.9] | 10.7 [9.7, 11.7] |

With the rule the agent does better against the calling and tight bots, with paired intervals above zero. Against the random bot there is **no clear effect**: −31 [−62, 1] in this run, an interval that includes zero. The deal-by-deal pairing is also weak for that bot: it draws its choices from one random stream across all the deals, so as soon as the rule-on and rule-off games diverge it makes different choices for the rest of the run, and the two rows no longer face the same opponent play. With the rule on, the agent still beats all three baselines on these deals. The rule was not tuned or varied; this is its only measurement here.

**Machine and budget.** The nine runs started at 23:53 GMT on 17 November and ended at 04:40 on 18 November: 4.8 hours elapsed, 12.95 CPU-hours in all (`run.json` of each run), run in parallel (six at first, nine from 00:17). From about 00:10 to 04:15 another job on the machine (twelve training processes) kept the load average between about 100 and 450, and each run got about a third of a core; that is why the GRU runs took 4.4–4.8 hours. The pilots took 0.4 hours before that, so training used about 5.2 of the 6 hours; the 05:20 cut-off committed to `SELECTION.md` at 00:40:48 was not needed. The evaluation was not run after all training ended: the six MLP candidates and the control were evaluated from 02:53 to 03:14 and `gru` seed 2 from about 04:20 to 04:23, while GRU runs were still training (the last ended at 04:40), and the remaining six cells at 04:40–04:44. The partial tables were seen before the last two GRU runs ended; the training code, budgets, cut-off and rule were not changed after them (`SELECTION.md`, *Records*). The confirmation on fresh deals ran from 04:46 to 04:48. Each run's `run.json` records as `commit` the head of the branch when the run ended (fixed afterwards in 1b0446f to the commit it starts from); the training code is the same in abdf6f8, where the runs started, and in every recorded commit.

## 7. Not measured here

- The repository's own figures (50,000 and 100,000 hands): neither their run settings nor their seeds were recorded, and they were logged with the faults fixed on this branch. They were not regenerated.
- The demo agents against people, against each other, or against any opponent other than the three baselines; exploitability.
- The demo agents at any other format (stack depth, number of seats, a moving button with stacks reset: evaluation and training put the button on seat 0 every hand, and the encoding is relative, but this was not measured).
- The families at equal training: the GRU had 150,000 hands, the MLPs 400,000. More seeds per candidate, other hyperparameters, longer training.
- Whether training repeats on another machine, another torch version or another number of threads.
- Throughput on other machines. Every number above comes from one shared laptop, and the training times in section 6 were measured while another job loaded it heavily.
