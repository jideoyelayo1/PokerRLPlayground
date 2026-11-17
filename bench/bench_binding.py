"""Engine throughput through the pybind11 binding, in hands per second.

  step/call    every seat calls (or checks); one getObservation + step per decision
  step/random  every seat picks a uniformly random legal action type
  loop/random  the per-decision engine work of train_poker_dqn without the networks:
               getObservation, encode_observation, legal_mask_from_obs, step,
               getObservation + encode_observation again; random legal action index

4 seats, stacks reset to 10,000 every hand, blinds 1/2. Seeds 1..runs (engine and policy).

    python bench/bench_binding.py --hands 20000 --runs 7
"""
import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import PokerEngine.poker_engine as pe
from ReinforcementLearningEnvironment.helper import action_index_to_engine_action, encode_observation, legal_mask_from_obs

SEATS, STACK = 4, 10000
SIZES = [pe.RaiseSize.BB, pe.RaiseSize.HalfPot, pe.RaiseSize.Pot]


def fallback_step(g, p, obs, act):
    try:
        g.step(act, p)
    except RuntimeError:  # rejected raise size: try the others, then check/call/all-in/fold
        legal = {s.type for s in obs.legal if s.allowed}
        for alt in [pe.Action(pe.ActionType.Raise, sz) for sz in SIZES] + [
                pe.Action(t) for t in (pe.ActionType.Check, pe.ActionType.Call, pe.ActionType.AllIn, pe.ActionType.Fold) if t in legal]:
            try:
                g.step(alt, p)
                return
            except RuntimeError:
                continue
        raise


def run_step(hands, seed, policy):
    g = pe.GameEngine(SEATS, STACK, 2, 1)
    g.seed(seed)
    rng = random.Random(seed + 1000)
    decisions = 0
    t0 = time.perf_counter()
    for _ in range(hands):
        g.reset_tournament(STACK)
        g.start_hand()
        while not g.hand_terminal():
            p = g.currentPlayerSafe()
            obs = g.getObservation(p)
            if policy == "call":
                act = pe.Action(pe.ActionType.Call if pe.GameEngine.isAllowed(obs.legal, pe.ActionType.Call) else pe.ActionType.Check)
            else:
                act = pe.Action(rng.choice(list(dict.fromkeys(s.type for s in obs.legal if s.allowed))), rng.choice(SIZES))
            fallback_step(g, p, obs, act)
            decisions += 1
    return hands / (time.perf_counter() - t0), decisions / hands


def run_loop(hands, seed):
    g = pe.GameEngine(SEATS, STACK, 2, 1)
    g.seed(seed)
    rng = random.Random(seed + 1000)
    decisions = 0
    t0 = time.perf_counter()
    for _ in range(hands):
        g.reset_tournament(STACK)
        g.start_hand()
        while not g.hand_terminal():
            p = g.currentPlayerSafe()
            obs = g.getObservation(p)
            encode_observation(obs, big_blind=2)
            mask = legal_mask_from_obs(obs)
            a_idx = int(rng.choice(np.where(mask == 0.0)[0]))
            g.step(action_index_to_engine_action(a_idx), p)
            encode_observation(g.getObservation(p), big_blind=2)
            decisions += 1
    return hands / (time.perf_counter() - t0), decisions / hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=7)
    a = ap.parse_args()
    print(f"hands per run {a.hands} (loop/random: {a.hands // 4}), runs {a.runs}, seats {SEATS}, stack {STACK}, "
          f"blinds 1/2, seeds 1..{a.runs}, Python {sys.version.split()[0]}")
    res = {"step/call": [], "step/random": [], "loop/random": []}
    dph = {}
    for r in range(1, a.runs + 1):
        for name, fn in (("step/call", lambda: run_step(a.hands, r, "call")),
                         ("step/random", lambda: run_step(a.hands, r, "random")),
                         ("loop/random", lambda: run_loop(a.hands // 4, r))):
            hps, d = fn()
            res[name].append(hps)
            dph[name] = d
        print(f"run {r}: " + "  ".join(f"{k} {v[-1]:.0f}" for k, v in res.items()) + " hands/s")
    for k, v in res.items():
        print(f"{k:12s} median {statistics.median(v):10.0f} hands/s  min {min(v):10.0f}  max {max(v):10.0f}  "
              f"(n={len(v)} runs, {dph[k]:.2f} decisions/hand)")


if __name__ == "__main__":
    main()
