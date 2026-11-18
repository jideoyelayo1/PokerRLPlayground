"""Checks on the exported demo agents in weights/demo/. From the repository root (extension built):

    python -m unittest discover -s tests -v

For each agent: the plain float32 export holds the same numbers as the PyTorch state_dict, its
sha256 matches the manifest, and it is at most 8 MB. For each parity decision: re-encoding the
recorded raw observation gives the recorded state vector and mask, the network gives the recorded
Q-values, and argmax(q + mask) is the recorded action (and, with the play-time option
no_free_fold, the recorded action_no_free_fold). For bench/figures.py: on a short seeded self-play
run every hand's results sum to zero, and the per-class aggregation and grid layout are right.
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

DEMO = ROOT / "weights" / "demo"
MANIFESTS = sorted(DEMO.glob("*-seed*.json"))


def observation(d):
    """Rebuild an engine Observation from a parity record's raw observation."""
    import PokerEngine.poker_engine as pe
    o = pe.Observation()
    for k in ("viewer_id", "street", "current_player_id", "button", "hand_in_progress", "terminal", "winner_id",
              "pot_total", "to_call", "min_raise", "alive", "in_hand", "can_act", "stack", "street_contrib",
              "total_contrib", "folded", "all_in", "viewer_to_call"):
        setattr(o, k, d[k])
    o.hole = [pe.Card(pe.Suits(s), pe.Ranks(r)) for r, s in d["hole"]]
    o.board = [pe.Card(pe.Suits(s), pe.Ranks(r)) for r, s in d["board"]]
    legal = []
    for x in d["legal"]:
        s = pe.ActionSpec()
        s.type, s.raise_size = pe.ActionType(x["type"]), pe.RaiseSize(x["raise_size"])
        for k in ("raise_to", "call_amount", "min_raise_to", "max_raise_to", "allowed"):
            setattr(s, k, x[k])
        legal.append(s)
    o.legal = legal
    return o


@unittest.skipUnless(MANIFESTS, "no exported agents in weights/demo")
class Export(unittest.TestCase):
    def test_binary_matches_state_dict(self):
        for path in MANIFESTS:
            m = json.loads(path.read_text())
            blob = (DEMO / m["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), m["sha256"], path.name)
            self.assertEqual(len(blob), m["bytes"])
            self.assertLessEqual(len(blob), 8 * 2**20)
            sd = torch.load(DEMO / f"{m['run']}.pt", weights_only=True)
            self.assertEqual([t["name"] for t in m["tensors"]], list(sd))
            for t in m["tensors"]:
                a = np.frombuffer(blob, dtype="<f4", count=t["nbytes"] // 4, offset=t["offset"]).reshape(t["shape"])
                self.assertTrue(np.array_equal(a, sd[t["name"]].numpy()), t["name"])

    def test_parity_vectors_reproduce(self):
        from eval_demo import load
        from ReinforcementLearningEnvironment.helper import encode_observation, legal_mask_from_obs, no_free_fold
        for path in MANIFESTS:
            run = json.loads(path.read_text())["run"]
            model, seq, _ = load(run)
            rows = [json.loads(line) for line in (DEMO / f"{run}.parity.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 1000)
            for r in rows:
                steps = r["history"] + [{"observation": r["observation"], "state": r["state"]}]
                xs = [encode_observation(observation(s["observation"]), big_blind=2) for s in steps]
                for x, s in zip(xs, steps):
                    np.testing.assert_array_equal(x, np.array(s["state"], np.float32))
                mask = legal_mask_from_obs(observation(r["observation"]))
                np.testing.assert_array_equal(mask, np.array(r["mask"], np.float32))
                x = np.stack(xs) if seq else xs[-1]
                with torch.no_grad():
                    q = model(torch.from_numpy(x)[None])[0].numpy()
                np.testing.assert_allclose(q, np.array(r["q"], np.float32), rtol=0, atol=1e-5)
                q = np.array(r["q"], np.float32)
                self.assertEqual(int(np.argmax(q + mask)), r["action"])
                self.assertEqual(int(np.argmax(q + no_free_fold(mask))), r["action_no_free_fold"])


class StartingHandFigure(unittest.TestCase):
    def test_hands_sum_to_zero_and_classes_aggregate(self):
        import figures
        rows, _ = figures.self_play((0, 30, 1))
        self.assertEqual(len(rows), 4 * 30)
        for h in range(30):
            self.assertEqual(sum(x for k, _, _, x in rows if k == h), 0)
        self.assertEqual(figures.self_play((10, 5, 1))[0], rows[40:60])  # a hand's deal depends only on its number
        agg = figures.by_class(rows)
        self.assertEqual(sum(n for n, _, _ in agg.values()), 120)
        for c, (n, m, se) in agg.items():
            v = [x for _, _, k, x in rows if k == c]
            self.assertEqual(n, len(v))
            self.assertAlmostEqual(m, np.mean(v))
        n, m, se = figures.by_class([(0, 0, "AA", 3.0), (0, 1, "72o", -3.0), (1, 2, "AA", -1.0), (1, 3, "AA", 1.0)])["AA"]
        self.assertEqual((n, m), (3, 1.0))
        self.assertAlmostEqual(se, 2 / np.sqrt(3))

    def test_grid_layout(self):
        import figures
        import PokerEngine.poker_engine as pe
        from ReinforcementLearningEnvironment.helper import pocket_169_key, pocket_key
        card = lambda r, s: pe.Card(pe.Suits(s), pe.Ranks(r))
        self.assertEqual(pocket_169_key(pocket_key(card(13, 2), card(14, 4))), "AKo")
        self.assertEqual(pocket_169_key(pocket_key(card(2, 3), card(7, 3))), "72s")
        cs = figures.classes()
        self.assertEqual(len(set(cs)), 169)
        self.assertEqual([figures.cell(c) for c in cs], [divmod(k, 13) for k in range(169)])
        self.assertEqual([figures.cell(c) for c in ("AA", "AKs", "AKo", "32s", "32o", "22")],
                         [(0, 0), (0, 1), (1, 0), (11, 12), (12, 11), (12, 12)])


if __name__ == "__main__":
    unittest.main()
