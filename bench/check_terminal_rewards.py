"""Does every seat that acted in a hand receive its result for that hand on its last decision?

Runs train_poker_dqn (four small MLPs, seed 0, stack 200, blinds 1/2) as a tournament and, where
the training loop has the option, with the stacks reset every hand, and records every transition
the agents store. For each finished hand and each seat that made at least one decision, the
seat's last stored transition must be terminal and carry the seat's bb_delta for the hand, and
its earlier ones must carry 0. A transition without a state is counted as lost, because
train_long_memory cannot train on it.

    python bench/check_terminal_rewards.py [repo_root] [hands=300]

repo_root defaults to this checkout; pass another checkout to check an older training loop.
"""
import random
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
hands = int(sys.argv[2]) if len(sys.argv) > 2 else 300
sys.path.insert(0, str(root))
import numpy as np
import torch

import PokerEngine.poker_engine as pe
import ReinforcementLearningEnvironment.helper as helper
from Models.MLPModel import MLPModel
from ReinforcementLearningEnvironment.Agent import DQNAgent
from ReinforcementLearningEnvironment.helper import encode_observation
from ReinforcementLearningEnvironment.Poker import train_poker_dqn
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import N_ACTIONS

SEATS, STACK = 4, 200
log = []  # (seat, hand, has_state, reward, done) in the order the agents stored them
remember = DQNAgent.remember


def spy(self, s, a, r, s2, done, *rest):
    log.append((self.player_id, self.n_hands, s is not None, float(r), bool(done)))
    return remember(self, s, a, r, s2, done, *rest)


DQNAgent.remember = spy
g = pe.GameEngine(SEATS, STACK, 2, 1); g.reset_tournament(STACK); g.start_hand()
f = len(encode_observation(g.getObservation(0), big_blind=2))


def report(hist, mode):
    aborted = {h for h, _ in hist["aborted"]}  # an aborted hand shares its number with the next one
    acted = rewarded = lost = multi = bad = 0
    for k, h in enumerate(hist["hand"]):
        if h in aborted:
            continue
        actors = 0
        for i in range(SEATS):
            tr = [t for t in log if t[0] == i and t[1] == h]
            lost += sum(not t[2] for t in tr)
            tr = [t for t in tr if t[2]]
            if not tr:
                continue
            actors += 1
            acted += 1
            last, delta = tr[-1], hist["delta_bb"][i][k]
            if last[4] and last[3] == delta:
                rewarded += 1
            if not last[4] or last[3] != delta or any(t[4] or t[3] != 0.0 for t in tr[:-1]):
                bad += 1
                if bad <= 5:
                    print(f"hand {h} seat {i}: bb_delta {delta}, stored (reward, done) {[(t[3], t[4]) for t in tr]}")
        multi += actors >= 2
    checked = len(hist["hand"]) - len(aborted & set(hist["hand"]))
    print(f"checkout {root.name}, {mode}: {checked} hands checked ({len(hist['aborted'])} aborted, skipped)")
    print(f"  hands where two or more seats acted: {multi}")
    print(f"  seats that acted: {acted}; last decision terminal with the seat's bb_delta: {rewarded}")
    print(f"  seat-hands breaking the rule: {bad}; terminal rewards stored without a state (never trained): {lost}")
    return bad == 0 and lost == 0 and acted > 0


ok = True
for mode, kw in (("tournament", {}), ("stacks reset every hand", {"reset_stacks": True})):
    log.clear()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
    models = [MLPModel(input_features=f, output_features=N_ACTIONS, hidden=(64,)) for _ in range(SEATS)]
    try:
        _, hist = train_poker_dqn(models, SEATS, STACK, 2, 1, max_hands=hands, device="cpu", ma_window=5, seed=0, **kw)
    except TypeError:  # an older training loop without this option
        continue
    ok = report(hist, mode) and ok
sys.exit(0 if ok else 1)
