"""Unit checks for the training code. From the repository root (extension built):

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from Models.MLPModel import MLPModel
from ReinforcementLearningEnvironment.Agent import DQNAgent

F, A = 151, 7


class ReplayBuffer(unittest.TestCase):
    def test_ring_keeps_the_latest_transitions(self):
        ag = DQNAgent(0, MLPModel(F, A, hidden=(8,)), F, memory_size=5)
        for i in range(12):
            ag.remember(np.full(F, i, np.float32), 0, float(i), np.zeros(F, np.float32), True)
        self.assertEqual(len(ag.memory), 5)
        self.assertEqual(sorted(t[2] for t in ag.memory), [7.0, 8.0, 9.0, 10.0, 11.0])


class Exploration(unittest.TestCase):
    def test_epsilon_falls_linearly_then_holds(self):
        ag = DQNAgent(0, MLPModel(F, A, hidden=(8,)), F, eps_start=1.0, eps_end=0.1, eps_decay_hands=100)
        mask = np.zeros(A, np.float32)
        for n, want in [(0, 1.0), (50, 0.55), (100, 0.1), (10_000, 0.1)]:
            ag.n_hands = n
            ag.select_action(np.zeros(F, np.float32), mask)
            self.assertAlmostEqual(ag.epsilon, want)


class Target(unittest.TestCase):
    def test_bootstrap_maxes_over_legal_actions_only(self):
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        net = torch.nn.Linear(2, 3, bias=False)
        tgt = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            net.weight.zero_()
            tgt.weight.copy_(torch.tensor([[0.0, 0.0], [5.0, 0.0], [1.0, 0.0]]))  # Q(s') = [0, 5, 1]
        tr = QTrainer(net, tgt, lr=1.0, gamma=1.0)
        tr.optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # one exact step: w0 += target * x
        s, s2 = np.array([1.0, 0.0], np.float32), np.array([1.0, 0.0], np.float32)
        tr.train_step([s], [0], [0.0], [s2], [False], [np.array([0, -1e9, 0], np.float32)])
        # MSE over 3 outputs: grad of Q(s,0) is -2/3 * target; target = max(0, 1) = 1, not 5
        self.assertAlmostEqual(net.weight[0, 0].item(), 0.5 * 2 / 3 * 1.0, places=6)


class SharedSeats(unittest.TestCase):
    def test_one_model_plays_every_seat(self):
        import tempfile
        import ReinforcementLearningEnvironment.helper as helper
        from ReinforcementLearningEnvironment.Poker import train_poker_dqn
        helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
        torch.manual_seed(0)
        agents, hist = train_poker_dqn([MLPModel(F, A, hidden=(8,))], 4, 200, 2, 1, max_hands=20, seed=1,
                                       reset_stacks=True, agent_kwargs={"memory_size": 10_000})
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].n_hands, 20)
        # every hand stores one terminal transition per seat that acted, all four seats included
        self.assertGreaterEqual(sum(t[4] for t in agents[0].memory), 20 * 2)
        self.assertEqual(len(hist["delta_bb"]), 4)


if __name__ == "__main__":
    unittest.main()
