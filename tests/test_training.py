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


if __name__ == "__main__":
    unittest.main()
