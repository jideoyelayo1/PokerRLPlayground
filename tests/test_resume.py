"""Resumable training (#17): a run stopped at a snapshot and resumed equals the uninterrupted run.

    python -m unittest discover -s tests -v
"""
import io
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

import train_demo
import PokerEngine.poker_engine as pe
import ReinforcementLearningEnvironment.helper as helper
from Models.GRUModel import GRUModel
from Models.MLPModel import MLPModel
from ReinforcementLearningEnvironment.Poker import train_poker_dqn

torch.set_num_threads(1)
F, A = 151, 7


def same(a, b):
    """Exact equality of nested dicts, lists, tuples, tensors, arrays and scalars."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


class Resume(unittest.TestCase):
    def setUp(self):
        helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")

    def run_train(self, make, hands, kw, resume=None, stop_at=None, reset_stacks=True):
        """Seeded as train_demo does. Returns (agents, history, the snapshot taken at hand stop_at)."""
        random.seed(1); np.random.seed(1)
        torch.manual_seed(99 if resume else 1)  # a resumed run starts from other weights: all must come from the snapshot
        model = make()
        snaps = []
        agents, hist = train_poker_dqn(
            [model], 4, 200, 2, 1, max_hands=hands, seed=1, reset_stacks=reset_stacks, agent_kwargs=kw,
            reward_scale=0.01, log_csv=False, resume=resume,
            on_hand_end=lambda n, ags, snap: snaps.append(snap()) if n == stop_at else None)
        return agents, hist, (snaps[0] if snaps else None)

    def check(self, make, **extra):
        kw = dict(memory_size=50, batch_size=8, target_update_freq=7, eps_decay_hands=40, double=True, **extra)
        straight, h1, _ = self.run_train(make, 60, kw)
        _, _, snap = self.run_train(make, 30, kw, stop_at=30)
        buf = io.BytesIO(); torch.save(snap, buf); buf.seek(0)
        snap = torch.load(buf, weights_only=False)  # our own file: numpy arrays and RNG states are pickled
        resumed, h2, _ = self.run_train(make, 60, kw, resume=snap)
        a, b = straight[0], resumed[0]
        self.assertEqual(len(a.memory), 50)  # the ring wrapped, and memory_pos is part of the state
        for key in ("model", "target_model", "optimizer", "memory", "memory_pos", "transitions", "n_hands", "epsilon",
                    "last_loss", "last_stats"):
            self.assertTrue(same(a.state_dict()[key], b.state_dict()[key]), key)
        self.assertNotEqual(a.epsilon, 1.0)
        for i in range(4):
            self.assertEqual(list(h1["delta_bb"][i]), list(h2["delta_bb"][i]))
        self.assertEqual(list(h1["hand"]), list(h2["hand"]))
        self.assertEqual(h1["resets"], h2["resets"])

    def test_mlp_resumed_equals_uninterrupted(self):
        self.check(lambda: MLPModel(F, A, hidden=(16,)))

    def test_gru_sequences_resumed_equals_uninterrupted(self):
        self.check(lambda: GRUModel(input_dim=F, output_dim=A, hidden_dim=8, layer_dim=1, dropout=0.0), sequence=True)

    def test_resume_without_reset_stacks_raises(self):
        with self.assertRaisesRegex(ValueError, "reset_stacks"):
            self.run_train(lambda: MLPModel(F, A, hidden=(8,)), 5, {}, resume={"version": 1}, reset_stacks=False)

    def test_unknown_snapshot_version_raises(self):
        with self.assertRaisesRegex(ValueError, "version"):
            self.run_train(lambda: MLPModel(F, A, hidden=(8,)), 5, {}, resume={"version": 0})


class TrainDemo(unittest.TestCase):
    def test_interrupted_run_resumed_equals_uninterrupted(self):
        # the score depends on the weights, so the best checkpoint is decided by the training; real validation is slow
        score = lambda m, seq: ({}, float(sum(p.sum() for p in m.parameters())))
        lines = lambda out: (out / "log.jsonl").read_text().splitlines()
        rows = lambda out: [{k: v for k, v in json.loads(line).items() if k not in ("wall_s", "cpu_s")}
                            for line in lines(out)]
        real_replace = os.replace

        def interrupt_after_resume_file(*a):
            real_replace(*a)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as d, mock.patch.object(train_demo, "validate", score):
            full, part = Path(d) / "full", Path(d) / "part"
            train_demo.train("mlp-s", 1, 60, full, 3, 1)
            with mock.patch.object(train_demo.os, "replace", interrupt_after_resume_file), \
                    self.assertRaises(KeyboardInterrupt):
                train_demo.train("mlp-s", 1, 60, part, 3, 1)
            self.assertTrue((part / "resume.pt").exists())
            self.assertFalse((part / "run.json").exists())
            with (part / "log.jsonl").open("a") as f:
                f.write('{"hands": 99}\n')  # a row written after resume.pt: dropped on resume
            train_demo.train("mlp-s", 1, 60, part, 3, 1, resume=True)
            self.assertEqual(len(rows(part)), 3)
            self.assertEqual(rows(part), rows(full))
            self.assertTrue(same(torch.load(full / "model.pt"), torch.load(part / "model.pt")))
            self.assertFalse((part / "resume.pt").exists())
            self.assertTrue((part / "run.json").exists())
            walls = [json.loads(line)["wall_s"] for line in lines(part)]
            self.assertEqual(walls, sorted(walls))  # the resumed segment adds to the time before it


class EngineRng(unittest.TestCase):
    def test_restored_state_deals_the_same_hands(self):
        def deal(g, n):
            out = []
            for _ in range(n):
                g.reset_tournament(200); g.start_hand()
                out.append([repr(g.getObservation(i).hole) for i in range(4)])
            return out

        g = pe.GameEngine(4, 200, 2, 1); g.seed(5)
        deal(g, 2)
        state = g.rng_state()
        first = deal(g, 3)
        g.set_rng_state(state)
        self.assertEqual(deal(g, 3), first)
        other = pe.GameEngine(4, 200, 2, 1)  # another engine, other seed: the state alone decides the deals
        other.seed(77); other.set_rng_state(state)
        self.assertEqual(deal(other, 3), first)


if __name__ == "__main__":
    unittest.main()
