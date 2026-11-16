"""Repeat one seed's training from bench/winrate.py (no evaluation) and compare its training log
with the committed one, to check that a seeded run repeats and to see why hands were aborted.

    python bench/rerun_training.py --seed 2 --train-hands 2000
"""
import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

import ReinforcementLearningEnvironment.helper as helper
from ReinforcementLearningEnvironment.Poker import train_poker_dqn
from winrate import BB, SB, SEATS, STACK, readme_models

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--train-hands", type=int, default=2000)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)  # same order as winrate.py
helper.CSV_PATH = str(ROOT / "bench" / "raw" / f"rerun_train_seed{a.seed}.csv")
Path(helper.CSV_PATH).unlink(missing_ok=True)
t0 = time.time()
_, history = train_poker_dqn(readme_models(), SEATS, STACK, BB, SB, max_hands=a.train_hands,
                             device="cpu", ma_window=50, seed=a.seed)
print(f"seed {a.seed}: {a.train_hands} hands in {time.time() - t0:.0f} s, aborted {history['aborted']}")
old = (ROOT / "bench" / "raw" / f"train_seed{a.seed}.csv").read_text().splitlines()
new = Path(helper.CSV_PATH).read_text().splitlines()
strip = lambda line: line.split(",", 1)[1]  # drop run_id, a random uuid per process
first = next((i for i, (x, y) in enumerate(zip(old, new)) if strip(x) != strip(y)), None)
print(f"training log rows: committed {len(old)}, rerun {len(new)}; "
      + ("identical apart from run_id" if first is None and len(old) == len(new) else f"first difference at row {first}"))
