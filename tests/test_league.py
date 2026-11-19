"""Checks for the opponent pool in train_poker_dqn (league=...). From the repository root:

    python -m unittest discover -s tests -v
"""
import io
import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import torch

from Models.GRUModel import GRUModel
from Models.MLPModel import MLPModel
from ReinforcementLearningEnvironment import Poker
from ReinforcementLearningEnvironment.Agent import DQNAgent
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import A_CALL, A_CHECK

torch.set_num_threads(1)
F, A = 151, 7


def bot_call(obs, mask, rng):
    return A_CALL if mask[A_CALL] == 0.0 else A_CHECK


def run(league, hands=60, seed=3, model=None, on_hand_end=None, resume=None, **kw):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(99 if resume else seed)  # a resumed run must take everything from the snapshot
    agent_kw = dict(batch_size=16, eps_decay_hands=40, **kw)
    return Poker.train_poker_dqn([model or MLPModel(F, A, hidden=(16,))], 4, 200, 2, 1, max_hands=hands, seed=seed,
                                 reset_stacks=True, log_csv=False, agent_kwargs=agent_kw, reward_scale=0.01,
                                 league=league, on_hand_end=on_hand_end, resume=resume)


def weights(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


class League(unittest.TestCase):
    def test_snapshots_join_at_the_right_hands_and_the_pool_is_bounded(self):
        pool, seen = [], []
        run({"snapshot_every": 10, "pool_size": 3, "pool": pool}, hands=55,
            on_hand_end=lambda n, ags, snap: seen.append((n, len(pool))))
        size = dict(seen)
        self.assertEqual(size[9], 0)
        self.assertEqual(size[10], 1)
        self.assertEqual(size[20], 2)
        self.assertEqual(size[30], 3)
        self.assertEqual(size[50], 3)  # five snapshots made, the oldest two dropped
        self.assertEqual(len(pool), 3)

    def test_snapshots_are_frozen_copies_of_the_learner_at_that_hand(self):
        pool, at = [], {}
        run({"snapshot_every": 20, "pool_size": 5, "pool": pool}, hands=60,
                   on_hand_end=lambda n, ags, snap: at.__setitem__(n, weights(ags[0].model).clone()) if n % 20 == 0 else None)
        self.assertEqual(len(pool), 3)
        for snap, n in zip(pool, (20, 40, 60)):
            self.assertFalse(snap.training)
            self.assertFalse(any(p.requires_grad for p in snap.parameters()))
            self.assertTrue(torch.equal(weights(snap), at[n]))  # still what the learner had then
        self.assertFalse(torch.equal(at[20], at[60]))  # the learner kept moving

    def test_only_learner_seats_store_and_every_hand_has_a_learner(self):
        calls, seats, now = [], {}, [0]  # learner decisions; bot-played seats by hand
        orig = DQNAgent.select_action

        def counted(self, s, m):
            calls.append(now[0])
            return orig(self, s, m)

        def bot(obs, mask, rng):
            seats.setdefault(now[0], set()).add(obs.viewer_id)
            return A_CALL if mask[A_CALL] == 0.0 else A_CHECK

        stored = []
        with mock.patch.object(DQNAgent, "select_action", counted):
            agents, _ = run({"snapshot_every": 10, "p_snapshot": 0.3, "bots": [bot], "p_bot": 0.5}, hands=80,
                            on_hand_end=lambda n, ags, snap: (stored.append(ags[0].transitions), now.__setitem__(0, n)))
        self.assertTrue(seats)
        self.assertLessEqual(max(len(v) for v in seats.values()), 3)  # a learner seat in every hand
        per_hand = np.diff([0] + stored)
        self.assertEqual(per_hand.tolist(), [calls.count(h) for h in range(80)])  # one per learner decision, none for others
        self.assertEqual(len(set().union(*seats.values())), 4)  # the learner seat moves around

    def test_all_bots_leaves_exactly_one_learner_seat(self):
        seats, now = {}, [0]

        def bot(obs, mask, rng):
            seats.setdefault(now[0], set()).add(obs.viewer_id)
            return A_CALL if mask[A_CALL] == 0.0 else A_CHECK

        run({"snapshot_every": 1000, "p_snapshot": 0.0, "bots": [bot], "p_bot": 1.0}, hands=40,
            on_hand_end=lambda n, ags, snap: now.__setitem__(0, n))
        self.assertTrue(all(len(v) <= 3 for v in seats.values()))

    def test_bots_are_asked_only_when_seated(self):
        asked = {"a": 0, "b": 0}

        def make(k):
            def bot(obs, mask, rng):
                asked[k] += 1
                return A_CALL if mask[A_CALL] == 0.0 else A_CHECK
            return bot

        run({"snapshot_every": 10, "bots": [make("a"), make("b")], "p_bot": 0.0}, hands=30)
        self.assertEqual(asked, {"a": 0, "b": 0})
        run({"snapshot_every": 10, "bots": [make("a"), make("b")], "p_bot": 0.6}, hands=30)
        self.assertGreater(min(asked.values()), 0)

    def test_league_off_path_is_unchanged(self):
        # no snapshot or bot seat is ever drawn: the same play, replay and weights as league=None
        # (the league's own random.Random leaves the global streams alone)
        (a, _), (b, _) = run(None), run({"snapshot_every": 10, "p_snapshot": 0.0, "p_bot": 0.0})
        self.assertEqual(len(a[0].memory), len(b[0].memory))
        for t, u in zip(a[0].memory, b[0].memory):
            for x, y in zip(t, u):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(torch.equal(weights(a[0].model), weights(b[0].model)))

    def test_sequence_learner_plays_snapshots_with_its_states_stacked(self):
        shapes, orig = [], Poker.greedy_action

        def spy(net, s, m, sequence, device):
            shapes.append((s.ndim, sequence))
            return orig(net, s, m, sequence, device)

        gru, pool = GRUModel(input_dim=F, output_dim=A, hidden_dim=8, layer_dim=1, dropout=0.0), []
        with mock.patch.object(Poker, "greedy_action", spy):
            agents, _ = run({"snapshot_every": 10, "p_snapshot": 0.8, "pool": pool}, hands=40, model=gru, sequence=True)
        self.assertTrue(shapes)
        self.assertTrue(all(ndim == 2 and seq for ndim, seq in shapes))  # (decisions so far, F)
        self.assertEqual(len(pool), 4)
        self.assertTrue(all(torch.isfinite(weights(m)).all() for m in (agents[0].model, *pool)))

    def test_resume_is_bitwise_the_straight_run(self):
        def cfg():  # a fresh dict per run: the pool fills in place
            return {"snapshot_every": 10, "pool_size": 2, "p_snapshot": 0.4, "p_bot": 0.3, "bots": [bot_call], "pool": []}

        def snap_at_30(n, ags, snapshot):
            if n == 30:
                buf = io.BytesIO()
                torch.save(snapshot(), buf)  # through the file format, as resume.pt
                saved.append(buf.getvalue())

        saved = []
        straight_cfg = cfg()
        (a, ha) = run(straight_cfg, hands=60)
        run(cfg(), hands=30, on_hand_end=snap_at_30)
        resumed_cfg = cfg()
        snap = torch.load(io.BytesIO(saved[0]), weights_only=False)
        self.assertEqual(len(snap["league"]["pool"]), 2)  # snapshots at 10, 20 and 30, the oldest dropped
        (b, hb) = run(resumed_cfg, hands=60, resume=snap)
        self.assertTrue(torch.equal(weights(a[0].model), weights(b[0].model)))
        self.assertEqual(len(a[0].memory), len(b[0].memory))
        for t, u in zip(a[0].memory, b[0].memory):
            for x, y in zip(t, u):
                np.testing.assert_array_equal(x, y)
        self.assertEqual(len(straight_cfg["pool"]), 2)
        self.assertEqual(len(resumed_cfg["pool"]), 2)
        for p, q in zip(straight_cfg["pool"], resumed_cfg["pool"]):
            self.assertTrue(torch.equal(weights(p), weights(q)))
            self.assertFalse(q.training or any(r.requires_grad for r in q.parameters()))
        for i in range(4):
            self.assertEqual(list(ha["delta_bb"][i]), list(hb["delta_bb"][i]))
        self.assertEqual((ha["resets"], ha["aborted"]), (hb["resets"], hb["aborted"]))

    def test_snapshot_without_league_has_no_league_state_and_resume_checks_it(self):
        snaps = []
        run(None, hands=4, on_hand_end=lambda n, ags, snap: snaps.append(snap()) if n == 4 else None)
        self.assertNotIn("league", snaps[0])
        with self.assertRaisesRegex(ValueError, "resume and league go together"):
            run({"snapshot_every": 10}, hands=8, resume=snaps[0])

    def test_needs_exactly_one_model(self):
        with self.assertRaisesRegex(ValueError, "exactly one model"):
            Poker.train_poker_dqn([MLPModel(F, A, hidden=(8,)), MLPModel(F, A, hidden=(8,))], 4, 200, 2, 1, max_hands=2,
                                  log_csv=False, league={"snapshot_every": 1})


if __name__ == "__main__":
    unittest.main()
