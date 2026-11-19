"""Checks for bench/league_eval.py on two tiny runs: every checkpoint is evaluated against the suite,
cells are reused, and the report and figure are written.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))

import league_eval
import train_demo


class LeagueEval(unittest.TestCase):
    def test_cells_cross_play_and_report(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(train_demo, "validate", lambda m, seq: ({}, 0.0)):
            runs_dir, out = Path(d) / "runs", Path(d) / "out"
            train_demo.train("mlp-s", 1, 40, runs_dir / "latest-seed1", 2, 1, keep=True)
            train_demo.train("mlp-s", 1, 40, runs_dir / "snap-seed1", 2, 1, keep=True, league_p_bot=0.0)
            runs = league_eval.runs_in(runs_dir)
            self.assertEqual(list(runs), ["latest-seed1", "snap-seed1"])
            jobs = league_eval.jobs_for(runs, out, 3)
            self.assertEqual(len(jobs), 2 * 2 * 7 + 2)  # 2 runs x 2 checkpoints x 7 bots, and the 2 ordered pairs
            first = [league_eval.cell(j) for j in jobs]
            with mock.patch.object(league_eval, "evaluate", side_effect=AssertionError("cell not reused")):
                again = [league_eval.cell(j) for j in jobs]
            self.assertEqual([c["bb100"] for c in first], [c["bb100"] for c in again])
            self.assertTrue(all(c["seed"] == league_eval.TEST_SEED for c in first))
            league_eval.report(runs, out)
            self.assertTrue((out / "progression.png").stat().st_size > 0)
            self.assertEqual(len((out / "progression.csv").read_text().splitlines()), 1 + 2 * 2 * 8)
            self.assertIn("unseen mean", (out / "tables.md").read_text())
            self.assertEqual(len(json.loads((out / "results.json").read_text())["cross"]), 2)


if __name__ == "__main__":
    unittest.main()
