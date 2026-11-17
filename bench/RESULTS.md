# Measurements

Every number here comes from a command in this repository. Raw output is in `bench/raw/`.
Re-run: `sh bench/run_benchmarks.sh` (tests and throughput) and `sh bench/run_benchmarks.sh winrate` (training and win rate).

**Code measured.** Sections 1, 3, 4 and 5 were measured on commit 2a3e3df, which gives the engine the standard hold'em seat order (section 2a), with one run of `sh bench/run_benchmarks.sh winrate`. The numbers in earlier versions of this file (in the git history, up to commit 2c79230) were measured with the engine's old, non-standard seat order, and are not comparable cell for cell. Sections 2 and 2b describe faults found and fixed on older code (2b's check was run again on 2a3e3df); the seed-2 rerun in section 4 was not repeated.

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
Seeded training therefore repeats only until the first timeout. Removing the wall-clock timeout (the 2,000-step cap remains) would make runs repeat. It was left in so that the code here is the code that produced these results. The rerun was not repeated with the standard seat order.

## 5. Win rate against fixed baselines

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

## 6. Not measured here

- The repository's own figures (50,000 and 100,000 hands): neither their run settings nor their seeds were recorded, and they were logged with the faults fixed on this branch. They were not regenerated.
- Training beyond 2,000 hands per seed, on MPS, or with other stack depths; win rates of the four architectures trained separately rather than together.
- Throughput on other machines. Every number above comes from one shared laptop.
