"""Unit checks for the models: checkpoints that keep the architecture (#21) and padded sequence
batches that match per-sequence inference (#15). From the repository root (extension built):

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from Models.GRUModel import GRUModel
from Models.LSTMModel import LSTMModel
from Models.MLPModel import MLPModel
from Models.TransformerQModel import TransformerQModel
from ReinforcementLearningEnvironment.QTrainer import QTrainer

F, A = 151, 7
# non-default sizes, so a load through cls() with the defaults cannot pass
MODELS = {
    "mlp": (MLPModel, dict(input_features=F, output_features=A, hidden=(32, 16), dropout=0.0)),
    "lstm": (LSTMModel, dict(input_dim=F, output_dim=A, hidden_dim=24, layer_dim=2, dropout=0.0)),
    "gru": (GRUModel, dict(input_dim=F, output_dim=A, hidden_dim=24, layer_dim=2, dropout=0.0)),
    "transformer": (TransformerQModel, dict(input_size=F, num_actions=A, d_model=16, nhead=2, num_layers=2,
                                            dim_feedforward=32, dropout=0.0, max_len=8)),
}
SEQUENCE_MODELS = ["gru", "lstm", "transformer"]


def build(name):
    torch.manual_seed(0)
    cls, kw = MODELS[name]
    return cls(**kw).eval()


class InTempDir(unittest.TestCase):
    """save() writes to ./model, so each test runs in its own directory."""
    def setUp(self):
        self._cwd, self._tmp = os.getcwd(), tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class SaveLoad(InTempDir):
    def test_round_trip_gives_identical_q_values(self):
        x = torch.randn(5, F)
        for name, (cls, _) in MODELS.items():
            with self.subTest(name):
                m = build(name)
                m.save("m.pth")
                loaded = cls.load("m.pth")
                self.assertFalse(loaded.training)
                with torch.no_grad():
                    self.assertTrue(torch.equal(m(x), loaded(x)))

    def test_kwargs_override_the_saved_config(self):
        build("transformer").save("m.pth")
        self.assertEqual(TransformerQModel.load("m.pth", max_len=8).max_len, 8)
        with self.assertRaises(RuntimeError):  # a different width no longer fits the weights
            TransformerQModel.load("m.pth", d_model=32)

    def test_legacy_state_dict_needs_kwargs(self):
        os.makedirs("model")
        for name, (cls, kw) in MODELS.items():
            with self.subTest(name):
                m = build(name)
                torch.save(m.state_dict(), "model/legacy.pth")
                with self.assertRaisesRegex(ValueError, "bare state_dict"):
                    cls.load("legacy.pth")
                loaded = cls.load("legacy.pth", **kw)
                x = torch.randn(3, F)
                with torch.no_grad():
                    self.assertTrue(torch.equal(m(x), loaded(x)))

    def test_demo_state_dict_still_loads(self):
        sd = torch.load(ROOT / "weights" / "runs" / "mlp-s-seed3" / "model.pt", weights_only=True)
        m = MLPModel(F, A, hidden=(128, 128), dropout=0.0)
        m.load_state_dict(sd)  # same keys as before the config was added
        self.assertEqual(list(m.state_dict()), list(sd))


class PaddedSequences(unittest.TestCase):
    @staticmethod
    def sequences():
        rng = np.random.default_rng(0)
        return [rng.standard_normal((t, F)).astype(np.float32) for t in (1, 4, 2, 8)]

    def test_padded_batch_matches_each_sequence_alone(self):
        trainer = QTrainer(MLPModel(F, A, hidden=(4,)), MLPModel(F, A, hidden=(4,)), lr=1e-3, gamma=1.0)
        seqs = self.sequences()
        x, lengths = trainer._batch(seqs)
        for name in SEQUENCE_MODELS:
            with self.subTest(name):
                net = build(name)
                with torch.no_grad():
                    batch = QTrainer._q(net, x, lengths)
                    alone = torch.cat([net(torch.from_numpy(s)[None]) for s in seqs])
                torch.testing.assert_close(batch, alone, atol=1e-5, rtol=0)

    def test_pad_values_do_not_change_the_output(self):
        trainer = QTrainer(MLPModel(F, A, hidden=(4,)), MLPModel(F, A, hidden=(4,)), lr=1e-3, gamma=1.0)
        x, lengths = trainer._batch(self.sequences())
        pad = torch.arange(x.shape[1])[None, :] >= lengths[:, None]
        junk = x.clone()
        junk[pad] = 100 * torch.randn(int(pad.sum()), F)
        for name in SEQUENCE_MODELS:
            with self.subTest(name):
                net = build(name)
                with torch.no_grad():
                    torch.testing.assert_close(net(x, lengths), net(junk, lengths), atol=1e-5, rtol=0)

    def test_transformer_reads_position_and_bounds_it(self):
        net = build("transformer")
        s = torch.randn(1, 3, F)
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(s), net(s[:, [1, 0, 2]]), atol=1e-5))  # order matters
        with self.assertRaisesRegex(ValueError, "max_len"):
            net(torch.randn(1, 9, F))


class TrainsTransformer(unittest.TestCase):
    def test_sequence_training_runs(self):
        import ReinforcementLearningEnvironment.helper as helper
        from ReinforcementLearningEnvironment.Poker import train_poker_dqn
        helper.CSV_PATH = str(Path(tempfile.mkdtemp()) / "log.csv")
        torch.manual_seed(0)
        agents, _ = train_poker_dqn([build("transformer")], 4, 200, 2, 1, max_hands=10, seed=1, reset_stacks=True,
                                    agent_kwargs={"sequence": True, "batch_size": 8, "memory_size": 1000})
        self.assertEqual(agents[0].n_hands, 10)
        self.assertGreater(agents[0].transitions, 10)


if __name__ == "__main__":
    unittest.main()
