"""How differently do the baseline bots behave with the exact legal mask than with the old one?

The bots' code is unchanged, but they choose from the mask they are given. The old mask marked all
three raise sizes legal whenever any raise was; the training loop then replaced a rejected size by
the first accepted one of BB, half pot, pot, or by check, call, all-in, fold. This plays the
selected demo agent against three copies of each baseline (bench/winrate.py's duplicate deals)
and, at every baseline decision, compares the old mask with the exact one, and the tight bot's
choice under each.

    python bench/check_baseline_masks.py [--deals 2000] [--seed 7000]
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

import PokerEngine.poker_engine as pe
from eval_demo import load
from train_demo import BB, SB, SEATS, STACK
from winrate import BOTS, greedy_policy, play_hand
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    A_ALLIN, A_CALL, A_CHECK, A_FOLD, A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT, ACTION_NAME)

RAISES = (A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT)


def old_mask(obs, mask):
    """The mask before the fix: all three sizes legal whenever the old legalActions allowed a raise,
    which was stack > 0, max raise-to > the current bet and max raise-to >= min raise-to. That can
    hold when no single size is legal (every size above the stack), so it is rebuilt from the
    Raise entries' bounds, not from the exact mask."""
    m = mask.copy()
    m[list(RAISES)] = -1e9
    spec = next(x for x in obs.legal if x.type == pe.ActionType.Raise)
    if obs.stack[obs.viewer_id] > 0 and spec.max_raise_to > obs.to_call and spec.max_raise_to >= spec.min_raise_to:
        m[list(RAISES)] = 0.0
    return m


def old_played(choice, mask):
    """What the old loop played for a choice made on the old mask (step_with_fallback): the choice
    if the engine accepts it, else, for a raise, the first accepted size of BB, half pot, pot, else
    the first accepted of check, call, all-in, fold."""
    if mask[choice] == 0.0:
        return choice
    for alt in (RAISES if choice in RAISES else ()) + (A_CHECK, A_CALL, A_ALLIN, A_FOLD):
        if mask[alt] == 0.0:
            return alt
    return A_FOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    seats = json.loads((ROOT / "bench" / "raw" / "demo_eval.json").read_text())["seats"]
    model, seq, _ = load(seats[0])
    out = {}
    for bname, bot in BOTS.items():
        stats = Counter()

        def spy(obs, mask, rng, bot=bot, stats=stats):
            stats["decisions"] += 1
            om = old_mask(obs, mask)
            stats["mask_differs"] += bool((om != mask).any())
            if bname == "tight":
                new, old = bot(obs, mask, rng), old_played(bot(obs, om, rng), mask)
                if new != old:
                    stats[f"tight now {ACTION_NAME[new]} where it played {ACTION_NAME[old]}"] += 1
            return bot(obs, mask, rng)

        pol = greedy_policy(model, seq)
        g = pe.GameEngine(SEATS, STACK, BB, SB)
        rng = random.Random(a.seed)
        for d in range(a.deals):
            for seat in range(SEATS):
                g.seed(a.seed * 1_000_003 + d)
                play_hand(g, [pol if i == seat else spy for i in range(SEATS)], rng, STACK)
        out[bname] = dict(stats)
        n = stats["decisions"]
        print(f"{bname}: {n:,} bot decisions; old mask differs from the exact one in {stats['mask_differs']:,} "
              f"({100 * stats['mask_differs'] / n:.1f}%)")
        for k, v in sorted(stats.items()):
            if k.startswith("tight now"):
                print(f"  {k}: {v:,} ({100 * v / n:.2f}% of its decisions)")


if __name__ == "__main__":
    main()
