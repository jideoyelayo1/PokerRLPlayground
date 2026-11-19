"""Checks for bench/train_demo.py: the --all runner reports failures, --keep-checkpoints saves snapshots.

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import torch

import train_demo

FAIL = [sys.executable, "-c", "raise SystemExit(3)"]
OK = [sys.executable, "-c", "pass"]


def touch(path):
    return [sys.executable, "-c", f"open({str(path)!r}, 'w')"]


class RunJobs(unittest.TestCase):
    def test_failed_job_is_reported_with_its_label(self):
        jobs = [("mlp-s seed 1", OK), ("gru seed 2", FAIL), ("mlp-l seed 3", OK)]
        self.assertEqual(train_demo.run_jobs(jobs, 2, poll=0.05), [("gru seed 2", 3)])

    def test_all_pass_returns_no_failures(self):
        self.assertEqual(train_demo.run_jobs([("a", OK), ("b", OK)], 2, poll=0.05), [])

    def test_fail_fast_stops_running_and_queued_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "started"
            jobs = [("bad", FAIL), ("slow", [sys.executable, "-c", "import time; time.sleep(60)"]), ("late", touch(marker))]
            t = time.time()
            self.assertEqual(train_demo.run_jobs(jobs, 2, fail_fast=True, poll=0.05), [("bad", 3)])
            self.assertLess(time.time() - t, 30)  # the 60 s sleeper was terminated
            self.assertFalse(marker.exists())  # the queued job never started

    def test_without_fail_fast_the_rest_still_run(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "started"
            self.assertEqual(train_demo.run_jobs([("bad", FAIL), ("late", touch(marker))], 1, poll=0.05), [("bad", 3)])
            self.assertTrue(marker.exists())


class MainAll(unittest.TestCase):
    def run_main(self, failures):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(train_demo, "RUNS", Path(d)), \
                mock.patch.object(train_demo, "run_jobs", return_value=failures) as rj, \
                mock.patch.object(sys, "argv", ["train_demo.py", "--all", "--keep-checkpoints", "--resume"]):
            try:
                train_demo.main()
            finally:
                self.jobs = rj.call_args[0][0]

    def test_exits_nonzero_naming_the_failed_run(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main([("gru seed 2", 3)])
        self.assertIn("gru seed 2 (code 3)", str(cm.exception.code))
        self.assertEqual(len(self.jobs), 9)
        self.assertTrue(all("--keep-checkpoints" in cmd and "--resume" in cmd for _, cmd in self.jobs))  # passed to the children

    def test_exits_zero_when_all_pass(self):
        self.run_main([])


class KeepCheckpoints(unittest.TestCase):
    def test_every_checkpoint_saved_and_model_pt_is_the_best(self):
        scores = iter([-5.0, -9.0])  # the first checkpoint is the best
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(train_demo, "validate", lambda m, seq: ({}, next(scores))):
            out = Path(d) / "run"
            train_demo.train("mlp-s", 1, 40, out, 2, 1, keep=True)
            self.assertEqual(sorted(p.name for p in out.glob("*.pt")), ["ckpt-20.pt", "ckpt-40.pt", "model.pt"])
            best, kept = (torch.load(out / f, map_location="cpu") for f in ("model.pt", "ckpt-20.pt"))
            self.assertTrue(all(torch.equal(best[k], kept[k]) for k in best))
            self.assertIn('"keep_checkpoints": true', (out / "run.json").read_text())

    def test_default_saves_only_model_pt(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(train_demo, "validate", lambda m, seq: ({}, 0.0)):
            out = Path(d) / "run"
            train_demo.train("mlp-s", 1, 40, out, 2, 1)
            self.assertEqual([p.name for p in out.glob("*.pt")], ["model.pt"])
            self.assertIn('"keep_checkpoints": false', (out / "run.json").read_text())


if __name__ == "__main__":
    unittest.main()
