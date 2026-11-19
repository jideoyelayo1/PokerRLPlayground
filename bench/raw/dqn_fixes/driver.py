"""python bench/raw/dqn_fixes/driver.py <huber|clip10> <seed> <out>: bench/train_demo.train for mlp-s,
40,000 hands, 4 checkpoints, 1 thread, with one more agent setting (the runs in RESULTS.md section 9
used a copy of this file that named this checkout by its path)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import train_demo

variant, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
train_demo.AGENT.update({"huber": {"loss": "huber"}, "clip10": {"max_grad_norm": 10.0}}[variant])
train_demo.train("mlp-s", seed, 40000, out, 4, 1)
