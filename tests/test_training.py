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
        # loss on the chosen action: grad of Q(s,0) is -2 * target; target = max(0, 1) = 1, not 5
        self.assertAlmostEqual(net.weight[0, 0].item(), 0.5 * 2 * 1.0, places=6)

    def test_double_dqn_values_the_online_networks_choice(self):
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        net, tgt = torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]))  # online at s': [0, 1, 2]
            tgt.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 7.0], [0.0, 3.0]]))  # target at s': [0, 7, 3]
        tr = QTrainer(net, tgt, lr=1.0, gamma=1.0, double=True)
        tr.optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
        s, s2 = np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)
        tr.train_step([s], [0], [0.0], [s2], [False], [np.zeros(3, np.float32)])
        # online picks action 2 at s', target values it at 3 (plain max would use 7)
        self.assertAlmostEqual(net.weight[0, 0].item(), 0.5 * 2 * 3.0, places=6)


class Loss(unittest.TestCase):
    @staticmethod
    def batch(n=16, seed=0):
        rng = np.random.default_rng(seed)
        return ([rng.normal(size=F).astype(np.float32) for _ in range(n)], rng.integers(0, A, n).tolist(),
                rng.normal(size=n).tolist(), [rng.normal(size=F).astype(np.float32) for _ in range(n)],
                (rng.random(n) < 0.3).tolist(), [np.zeros(A, np.float32)] * n)

    def test_loss_is_the_mean_squared_td_error_of_the_chosen_actions(self):
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        torch.manual_seed(0)
        net, tgt = MLPModel(F, A, hidden=(16,), dropout=0.0), MLPModel(F, A, hidden=(16,), dropout=0.0)
        tr = QTrainer(net, tgt, lr=1e-3, gamma=0.9)
        s, a, r, s2, d, m = self.batch()
        with torch.no_grad():
            q = net(torch.tensor(np.array(s)))[torch.arange(16), torch.tensor(a)]
            nxt = tgt(torch.tensor(np.array(s2))).max(dim=1).values
            y = torch.tensor(r, dtype=torch.float32) + (~torch.tensor(d)).float() * 0.9 * nxt
        self.assertAlmostEqual(tr.train_step(s, a, r, s2, d, m), ((q - y) ** 2).mean().item(), places=5)

    def test_huber_loss_is_smooth_l1_of_the_chosen_actions(self):
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        torch.manual_seed(0)
        net, tgt = MLPModel(F, A, hidden=(16,), dropout=0.0), MLPModel(F, A, hidden=(16,), dropout=0.0)
        tr = QTrainer(net, tgt, lr=1e-3, gamma=0.9, loss="huber")
        s, a, r, s2, d, m = self.batch()
        r = (np.array(r) * 5).tolist()  # some residuals past 1, where Huber is linear
        with torch.no_grad():
            q = net(torch.tensor(np.array(s)))[torch.arange(16), torch.tensor(a)]
            nxt = tgt(torch.tensor(np.array(s2))).max(dim=1).values
            y = torch.tensor(r, dtype=torch.float32) + (~torch.tensor(d)).float() * 0.9 * nxt
        self.assertAlmostEqual(tr.train_step(s, a, r, s2, d, m), torch.nn.functional.smooth_l1_loss(q, y).item(), places=5)


class Stability(unittest.TestCase):
    @staticmethod
    def trainer(**kw):
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        torch.manual_seed(0)
        net = MLPModel(F, A, hidden=(32,), dropout=0.0)
        return net, QTrainer(net, MLPModel(F, A, hidden=(32,), dropout=0.0), lr=1e-3, gamma=1.0, **kw)

    @staticmethod
    def extreme(n=32):
        # a 200-chip pot at 2 chips a big blind, reward_scale 1: terminal rewards of +/-100
        rng = np.random.default_rng(1)
        return ([rng.normal(size=F).astype(np.float32) for _ in range(n)], rng.integers(0, A, n).tolist(),
                [100.0, -100.0] * (n // 2), [np.zeros(F, np.float32)] * n, [True] * n, [np.zeros(A, np.float32)] * n)

    def test_finite_gradients_and_stats_for_extreme_rewards(self):
        for kw in ({}, {"loss": "huber"}, {"max_grad_norm": 10.0}):
            net, tr = self.trainer(**kw)
            self.assertTrue(np.isfinite(tr.train_step(*self.extreme())))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))
            self.assertEqual(set(tr.last_stats), {"td_abs_mean", "td_abs_max", "q_abs_max", "grad_norm", "nonfinite"})
            self.assertTrue(all(np.isfinite(v) for v in tr.last_stats.values()))
            self.assertEqual(tr.last_stats["nonfinite"], 0)
            self.assertGreater(tr.last_stats["td_abs_max"], 50)

    def test_nonfinite_values_are_counted(self):
        net, tr = self.trainer()
        b = self.extreme()
        tr.train_step(b[0], b[1], [float("nan")] * len(b[2]), *b[3:])
        self.assertGreater(tr.last_stats["nonfinite"], 0)

    def test_clipping_bounds_the_gradient_norm(self):
        net, tr = self.trainer(max_grad_norm=1.0)
        tr.train_step(*self.extreme())
        norm = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())).item()
        self.assertGreater(tr.last_stats["grad_norm"], 1.0)  # the norm before clipping
        self.assertLessEqual(norm, 1.0 + 1e-5)
        net, tr = self.trainer()  # no clipping by default
        tr.train_step(*self.extreme())
        norm = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())).item()
        self.assertAlmostEqual(norm, tr.last_stats["grad_norm"], places=3)

    def test_agent_exposes_the_stats(self):
        ag = DQNAgent(0, MLPModel(F, A, hidden=(8,)), F, loss="huber", max_grad_norm=5.0)
        for _ in range(4):
            ag.remember(np.zeros(F, np.float32), 1, 1.0, np.zeros(F, np.float32), True)
        ag.train_long_memory()
        self.assertEqual(ag.last_stats, ag.trainer.last_stats)
        self.assertIsNotNone(ag.last_stats)


class Dropout(unittest.TestCase):
    def test_greedy_action_is_repeatable_with_dropout(self):
        torch.manual_seed(0)
        ag = DQNAgent(0, MLPModel(F, A, hidden=(64, 64), dropout=0.5), F, eps_start=0.0, eps_end=0.0)
        s, mask = np.random.default_rng(0).normal(size=F).astype(np.float32), np.zeros(A, np.float32)
        self.assertTrue(ag.model.training)
        self.assertEqual(len({ag.select_action(s, mask) for _ in range(50)}), 1)
        self.assertTrue(ag.model.training)  # the mode it found is restored
        ag.model.eval()
        ag.select_action(s, mask)
        self.assertFalse(ag.model.training)

    def test_training_step_updates_the_model_in_train_mode(self):
        torch.manual_seed(0)
        ag = DQNAgent(0, MLPModel(F, A, hidden=(16,), dropout=0.5), F, eps_start=0.0, eps_end=0.0, double=True)
        ag.model.eval()  # as select_action may leave a model that was in eval mode
        for _ in range(8):
            ag.remember(np.ones(F, np.float32), 2, 1.0, np.ones(F, np.float32), False)
        before = [p.clone() for p in ag.model.parameters()]
        ag.train_long_memory()
        self.assertTrue(ag.model.training)
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, ag.model.parameters())))


class TargetSync(unittest.TestCase):
    def test_target_is_copied_after_hands_3_6_9(self):
        import tempfile
        import ReinforcementLearningEnvironment.helper as helper
        from ReinforcementLearningEnvironment.Poker import train_poker_dqn
        helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
        synced = []

        def on_hand_end(n, agents):
            ag = agents[0]
            synced.append(all(torch.equal(a, b) for a, b in zip(ag.model.state_dict().values(),
                                                                 ag.target_model.state_dict().values())))

        torch.manual_seed(0)
        train_poker_dqn([MLPModel(F, A, hidden=(8,))], 4, 200, 2, 1, max_hands=10, seed=1, reset_stacks=True,
                        agent_kwargs={"target_update_freq": 3}, log_csv=False, on_hand_end=on_hand_end)
        self.assertEqual([n for n, same in enumerate(synced, 1) if same], [3, 6, 9])


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


class Sequences(unittest.TestCase):
    def test_padded_batch_gives_each_sequences_own_output(self):
        from Models.GRUModel import GRUModel
        from ReinforcementLearningEnvironment.QTrainer import QTrainer
        torch.manual_seed(0)
        net = GRUModel(F, A, hidden_dim=16, layer_dim=1, dropout=0.0)
        seqs = [np.random.rand(t, F).astype(np.float32) for t in (1, 4, 2)]
        x, lengths = QTrainer(net, net, 1e-3, 1.0)._batch(seqs)
        from Models.LSTMModel import LSTMModel
        for net in (net, LSTMModel(F, A, hidden_dim=16, layer_dim=2, dropout=0.0).eval()):
            with torch.no_grad():
                batched = net(x, lengths)
                for i, q in enumerate(seqs):
                    torch.testing.assert_close(batched[i], net(torch.tensor(q)[None])[0], rtol=0, atol=1e-6)

    def test_loop_stores_the_hand_so_far(self):
        import tempfile
        import ReinforcementLearningEnvironment.helper as helper
        from Models.GRUModel import GRUModel
        from ReinforcementLearningEnvironment.Poker import train_poker_dqn
        helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
        torch.manual_seed(0)
        agents, _ = train_poker_dqn([GRUModel(F, A, hidden_dim=8, layer_dim=1)], 4, 200, 2, 1, max_hands=30,
                                    seed=1, reset_stacks=True, agent_kwargs={"sequence": True, "memory_size": 10_000})
        mem = agents[0].memory
        self.assertTrue(all(t[0].ndim == 2 and t[0].shape[1] == F for t in mem))
        for st, _, _, s2, done, _ in mem:
            if not done:  # the next state is this hand's sequence one decision later
                self.assertEqual(len(s2), len(st) + 1)
                np.testing.assert_array_equal(s2[:-1], st)
        self.assertGreater(max(len(t[0]) for t in mem), 1)


if __name__ == "__main__":
    unittest.main()
