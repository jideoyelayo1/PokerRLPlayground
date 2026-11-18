"""Train one seed of bench/winrate.py's set-up (no evaluation), write its per-hand training log and
a hash of the trained weights, and optionally compare the log with another run's, row by row.
Run it twice, in two processes, to check that a seeded run repeats:

    python bench/rerun_training.py --seed 2 --train-hands 500 --log bench/raw/rerun_a.csv
    python bench/rerun_training.py --seed 2 --train-hands 500 --log bench/raw/rerun_b.csv --compare bench/raw/rerun_a.csv
"""
import argparse
import hashlib
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
ap.add_argument("--train-hands", type=int, default=500)
ap.add_argument("--log", required=True)
ap.add_argument("--compare")
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)  # same order as winrate.py
helper.CSV_PATH = a.log
Path(a.log).unlink(missing_ok=True)
t0 = time.time()
agents, history = train_poker_dqn(readme_models(), SEATS, STACK, BB, SB, max_hands=a.train_hands,
                                  device="cpu", ma_window=50, seed=a.seed)
h = hashlib.sha256()
for ag in agents:
    for t in ag.model.state_dict().values():
        h.update(t.numpy().tobytes())
print(f"seed {a.seed}: {a.train_hands} hands in {time.time() - t0:.0f} s, torch threads {torch.get_num_threads()}, "
      f"aborted {history['aborted']}, tournament resets {len(history['resets'])}")
print(f"sha256 of the four trained networks' weights: {h.hexdigest()}")
if a.compare:
    old, new = Path(a.compare).read_text().splitlines(), Path(a.log).read_text().splitlines()
    strip = lambda line: line.split(",", 1)[1]  # drop run_id, a random uuid per process
    first = next((i for i, (x, y) in enumerate(zip(old, new)) if strip(x) != strip(y)), None)
    print(f"training log rows: {a.compare} {len(old)}, {a.log} {len(new)}; "
          + ("identical apart from run_id" if first is None and len(old) == len(new) else f"first difference at row {first}"))
