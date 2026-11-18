"""Short train_poker_dqn run that checks the per-hand log: are folded players logged, and do a
hand's bb_delta values sum to zero?

    python bench/check_training_log.py [repo_root] [hands=30]

repo_root defaults to this checkout; pass another checkout (e.g. a worktree of an older commit)
to check its training loop with the same models and seeds.
"""
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
hands = int(sys.argv[2]) if len(sys.argv) > 2 else 30
sys.path.insert(0, str(root))
import csv

import numpy as np
import torch

import ReinforcementLearningEnvironment.helper as helper
from Models.GRUModel import GRUModel
from Models.LSTMModel import LSTMModel
from Models.MLPModel import MLPModel
from Models.TransformerQModel import TransformerQModel
from ReinforcementLearningEnvironment.Poker import train_poker_dqn
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import N_ACTIONS

random.seed(0); np.random.seed(0); torch.manual_seed(0)
helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
f = 151  # encode_observation length for 4 seats
models = [MLPModel(input_features=f, output_features=N_ACTIONS, hidden=(512, 1024, 2048, 1024, 512)),
          LSTMModel(input_dim=f, output_dim=N_ACTIONS), GRUModel(input_dim=f, output_dim=N_ACTIONS),
          TransformerQModel(input_size=f, num_actions=N_ACTIONS)]
_, hist = train_poker_dqn(models, 4, 10000, 2, 1, max_hands=hands, device="cpu", ma_window=5)
rows = list(csv.DictReader(open(helper.CSV_PATH)))
per_hand = Counter(r["hand_id"] for r in rows)
sums = [sum(hist["delta_bb"][i][h] for i in range(4)) for h in range(len(hist["hand"]))]
print(f"checkout {root.name}, {hands} hands, {len(rows)} rows")
print("rows per hand:", dict(sorted(Counter(per_hand.values()).items())))
print("rows with folded=1:", sum(r["folded"] == "1" for r in rows))
print(f"sum over the 4 players of a hand's bb_delta: min {min(sums)}, max {max(sums)}")
