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
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT, ACTION_NAME

RAISES = (A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT)


def old_mask(mask):
    m = mask.copy()
    if any(mask[i] == 0.0 for i in RAISES):
        m[list(RAISES)] = 0.0
    return m


def old_played(choice, mask):
    """What the old loop played for a choice made on the old mask (step_with_fallback)."""
    if mask[choice] == 0.0:
        return choice
    for alt in RAISES:
        if mask[alt] == 0.0:
            return alt
    return choice  # unreachable: the old mask only added raise sizes when some size was legal


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
            om = old_mask(mask)
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
