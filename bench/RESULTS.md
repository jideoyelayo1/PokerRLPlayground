# Measurements

Every number here comes from a command in this repository. Raw output is in `bench/raw/`.
Re-run: `sh bench/run_benchmarks.sh` (tests and throughput) and `sh bench/run_benchmarks.sh winrate` (training and win rate).

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

Wall time: 62 s on 10 threads.

## 2. Betting rounds (bug found and fixed)

`PokerEngine/tests/test_betting.cpp` drives the `step()` API, the one the training loop uses. It checks that:
- a street closes after one action per live seat;
- every seat acts exactly once per street when everyone calls or checks (2 and 4 seats, every blind position);
- in a stress test with six policy mixes, each with stacks reset every hand and as a running tournament (5,000 hands each, engine seed 3, policy seed 4), every hand finishes within the training loop's 2,000-step cap, the player to act always has a legal action, and chips are conserved.

On the engine before the fix, 36 of 124 checks fail ([raw](raw/test_betting_before_fix.txt)). After the fix, 0 fail ([raw](raw/test_betting_after_fix.txt)).

| policy mix (4 seats unless noted) | stacks | hands not finished in 2,000 steps, before fix |
|---|---|---:|
| random + 3 x call | reset each hand | 1,280 / 5,000 |
| random + check-fold + 2 x call | reset each hand | 2,480 / 5,000 |
| 3 x call + check-fold | tournament | 1,666 / 5,000 |
| heads-up random vs call | tournament | 881 / 5,000 |
| random + 3 x call | tournament | 38 / 5,000 |

Also before the fix: 15 + 7 hands reached a player with no legal action, 12 hands destroyed chips, one seat never acted after the flop, and preflop some seats acted twice.
The cause: a street closed only when the action returned to one marker seat with all bets matched. If that seat had folded or was all-in, the action never reached it. The 12 chip losses were side pots with no eligible winner, after a player who had put in more than anyone else folded out of turn. The fix tracks which live seats still have to act.

## 2b. Training-log faults (found and fixed)

`python bench/check_training_log.py` runs 30 hands of `train_poker_dqn` (README models, seed 0) and checks the per-hand log ([raw](raw/check_training_log.txt)).
On the original training loop (commit 6727091: fixed engine, old `Poker.py`), no row has `folded=1`, and the four players' `bb_delta` sum to +1.5 BB on every hand.
On this branch, folded players are logged and the sum is 0.0 on every hand.
The causes:
- stacks were read after the blinds;
- pocket cards were read at the end of the hand, when a folded player's observation hides them;
- busted players were logged again with stale cards.

In the three 2,000-hand training logs below, 8.6–48.3% of rows are folds ([winrate_table.md](raw/winrate_table.md)), and the old logging dropped every one of them.
The EV-per-starting-hand figures in the README were therefore averaged only over hands that were not folded. They have not been regenerated.

## 3. Engine throughput

4 seats, stacks reset to 10,000 every hand, blinds 1/2, fixed policies, single thread, 7 runs (seeds 1–7), after the betting fix. Median, with the min–max over the runs in brackets.

| path | policy | hands/s | decisions/hand | raw |
|---|---|---:|---:|---|
| C++, `playOneHand` (controller API) | everyone calls/checks | **56,057** [50,211–56,778] | 16.0 | [bench_engine.txt](raw/bench_engine.txt) |
| C++, `step()` | everyone calls/checks | **42,374** [40,554–42,650] | 16.0 | same |
| C++, `step()` | uniform random legal action | **68,782** [67,944–69,159] | 5.6 | same |
| Python binding, `step()` | everyone calls/checks | **12,950** [12,883–13,194] | 16.0 | [bench_binding.txt](raw/bench_binding.txt) |
| Python binding, `step()` | uniform random legal action | **12,661** [12,496–12,778] | 5.7 | same |
| Python, training loop without the networks | random legal action index | **1,821** [1,792–1,862] | 7.3 | same |

`./build/bench_engine 200000 7` (200,000 hands per run) and `python bench/bench_binding.py --hands 20000 --runs 7` (20,000 hands per run; 5,000 for the last row).
The last row is the per-decision work `train_poker_dqn` does outside the networks: `getObservation`, `encode_observation`, `legal_mask_from_obs`, `step_with_fallback`, then `getObservation` and `encode_observation` again.
The load average at the start of the C++ run was 106, partly from the exhaustive test that had just finished; in one run of seven (run 2), `playOneHand` was 10% slower.

## 4. Training cost

The README's four models (MLP 512-1024-2048-1024-512, LSTM and GRU 1024 x 3, Transformer d_model 256 x 3 layers), CPU, torch 8 threads, 2,000 self-play hands per seed ([winrate.log](raw/winrate.log)):

| seed | wall time | hands/s | CPU-s per hand | aborted hands | tournament resets |
|---|---:|---:|---:|---:|---:|
| 1 | 9.5 min | 3.51 | 0.62 | 0 | 21 |
| 2 | 11.7 min | 2.84 | 0.82 | 8 | 1,263 |
| 3 | 10.0 min | 3.35 | 0.68 | 3 | 190 |

Median 3.35 hands/s (0.30 s per hand). The engine, binding and state encoding take about 0.55 ms per hand at 7.3 decisions per hand (section 3, last row), under 0.2% of that. The rest is the networks: a single-sample update at every decision, plus a batch-1024 replay update per agent per hand.
At the measured rate, 30,000 hands would take about 2.5 hours of wall time and 5–7 CPU-hours. That is an extrapolation, not a measurement.

**Aborted hands and repeatability.** After the betting fix, the hands the training loop aborts come from its 2-second wall-clock timeout (`HAND_TIMEOUT_SEC`), not from the engine.
Seed 2 was trained again from scratch with abort reasons recorded: `python bench/rerun_training.py --seed 2` ([raw](raw/rerun_seed2.txt), log [rerun_train_seed2.csv](raw/rerun_train_seed2.csv)). It aborted 7 hands (the original run aborted 8), all `timeout`.
Its training log matches the committed one row for row (apart from the per-process `run_id`) up to row 285, hand 114, and differs from there on.
A timeout cuts a hand after however many decisions the machine managed in 2 seconds, and each decision has already applied a single-step update to the network. So the weights, and everything after, depend on machine load.
Seeded training therefore repeats only until the first timeout. Removing the wall-clock timeout (the 2,000-step cap remains) would make runs repeat. It was left in so that the code here is the code that produced these results.

## 5. Win rate against fixed baselines

`python bench/winrate.py --seeds 1 2 3 --train-hands 2000 --eval-deals 1000`, then `python bench/summarise_winrate.py` ([table](raw/winrate_table.md), [per-seed JSON](raw/), [training logs](raw/)).

**Training budget.** 2,000 self-play hands per seed with `train_poker_dqn` as in `main.ipynb`: four seats (MLP, LSTM, GRU, Transformer), stack 10,000, blinds 1/2, CPU. Seeds 1, 2 and 3 fix Python `random`, NumPy, torch and the engine's deals. About 10 minutes per seed; the whole run, evaluation included, took 42 minutes. The README's figures are labelled 50,000 and 100,000 hands, so this is 2–4% of that budget.

**Evaluation.** One trained agent (greedy: epsilon 0, `model.eval()`) sits at a 4-seat table with three copies of one baseline bot, with stacks reset to 10,000 (5,000 big blinds) every hand. Duplicate deals: each deal is played 4 times with the same cards, the agent taking each seat once, and the deal's score is the agent's mean result over the 4. That is 1,000 deals = 4,000 hands per cell. The 95% CI within a seed is mean ± 1.96 sd/√1000 over deals. Across seeds it is a t-interval (2 degrees of freedom) over the three per-seed values.
Baselines: **random** (uniform over the agent's legal action indices), **call** (calls a bet, otherwise checks), **tight** (preflop: pot-raise TT+, AK, AQ; call other pairs, two broadway cards or suited aces; otherwise check or fold. Postflop: call or check with a pocket pair or a hole card that pairs the board, otherwise check or fold).
**random-bot** is a control: the random bot in the agent's seat.
With duplicate deals, an agent that acts exactly like the baseline scores exactly 0. One that always folds scores −37.5 (the blinds).

bb/100, mean over the three seeds [95% t-CI over seeds]:

| agent | vs random | vs call | vs tight |
|---|---:|---:|---:|
| MLP | −4,224 [−19,417, 10,969] | 0 [0, 0] | −46,064 [−140,905, 48,776] |
| LSTM | 12,695 [−2,094, 27,483] | 4 [−17, 25] | −17,873 [−26,974, −8,772] |
| GRU | 14,310 [−9,792, 38,412] | −21 [−68, 27] | −29,081 [−119,653, 61,491] |
| Transformer | 0 [−38,329, 38,329] | 5 [−15, 24] | −30,795 [−114,058, 52,468] |
| random-bot (control) | 3,150 [−13,441, 19,741] | 2,878 [−15,862, 21,618] | −31,501 [−37,351, −25,651] |

Per-seed values with their own CIs are in [winrate_table.md](raw/winrate_table.md).

**Reading it.**
- **No trained agent beats any baseline reliably at this budget.** Every across-seed interval against the random and calling bots contains zero.
- **Against the calling bot,** every agent is within 40 bb/100 of zero, and 7 of the 12 seed-agent pairs score exactly 0, i.e. they play exactly like a calling station at that table. Seed 3's GRU scores exactly −38 against both the calling and the tight bot: it folds every hand.
- **Against the tight bot,** every agent loses in every seed (each per-seed CI lies below zero). Only the LSTM's across-seed interval excludes zero: −17,873 [−26,974, −8,772]. The random control loses −31,501.
- **Against the random bot,** three single seeds win clearly: LSTM seed 2, +19,527 [7,287, 31,768]; GRU seed 2, +25,510 [12,632, 38,389]; Transformer seed 3, +17,811 [4,345, 31,277]. In the other seeds, the LSTM and GRU estimates are positive (about +8,500 to +9,900) but their intervals include zero, and Transformer seed 1 loses (−8,576 [−16,209, −942]). The random control itself ranges from −792 to +10,861 across seeds.
- **The sizes are in the tens of thousands of bb/100** because stacks are 5,000 big blinds deep and most policies collapse towards moving all-in. In the last quarter of training, 6 of the 10 seats still being dealt in opened with all-in in 53–100% of their hands; the other 4 mostly folded or raised (table in [winrate_table.md](raw/winrate_table.md); this is training-time play, with dropout on).
- **The training set-up limits learning, independent of budget.**
  - Only the player whose action ends a hand ever receives a reward (README, *Multi-player credit assignment*).
  - A busted agent sits out until the tournament resets, so in seed 1 the last 500 hands were heads-up between the LSTM and the Transformer, while the MLP and GRU kept replaying old transitions.

**Provenance.** The three training runs used commit dd03810. The later commit 12464d9 changes only the training loop's bookkeeping: abort reasons, and logging a player who busts in a hand that `start_hand` settles straight after the blinds. So the per-seed JSON has abort counts but no reasons, and `train_seed*.csv` can miss such rows.

## 6. Not measured here

- The repository's own figures (50,000 and 100,000 hands): neither their run settings nor their seeds were recorded, and they were logged with the faults fixed on this branch. They were not regenerated.
- Training beyond 2,000 hands per seed, on MPS, or with other stack depths; win rates of the four architectures trained separately rather than together.
- Throughput on other machines. Every number above comes from one shared laptop.
