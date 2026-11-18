"""Close a training run that was stopped at the time budget (bench/SELECTION.md, amendment) before
it finished: write its run.json from its log.jsonl. The run's candidate stays the checkpoint it
kept (model.pt, the best validation score up to its last checkpoint); hands trained after that
last checkpoint are not used. Stop the run's process first.

    python bench/stop_run.py weights/runs/gru-seed1 [...]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

from train_demo import AGENT, BB, CANDIDATES, REWARD_SCALE, SB, SEATS, STACK, VAL_DEALS, VAL_SEED, build

for d in map(Path, sys.argv[1:]):
    if (d / "run.json").exists():
        sys.exit(f"{d} has a run.json already")
    name, seed = d.name.rsplit("-seed", 1)
    rows = [json.loads(line) for line in open(d / "log.jsonl")]
    best = max(rows, key=lambda r: (r["val_score"], -r["hands"]))  # the earliest best, as train_demo keeps
    hands = CANDIDATES[name]["hands"]
    run = {"candidate": name, "architecture": CANDIDATES[name], "seed": int(seed), "hands": hands,
           "stopped": True, "hands_trained": rows[-1]["hands"], "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "agent": dict(AGENT, eps_decay_hands=hands // 5, sequence=CANDIDATES[name]["sequence"]),
           "reward_scale": REWARD_SCALE, "format": {"seats": SEATS, "stack": STACK, "blinds": [SB, BB], "reset_stacks": True},
           "validation": {"seed": VAL_SEED, "deals": VAL_DEALS, "every_hands": rows[0]["hands"]},
           "kept_checkpoint_hands": best["hands"], "kept_val_score": best["val_score"],
           "wall_s": rows[-1]["wall_s"], "cpu_s": rows[-1]["cpu_s"], "torch_threads": 1, "aborted_hands": None,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "dirty": None, "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "parameters": sum(p.numel() for p in build(name).parameters())}
    (d / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"{d.name}: stopped after {run['hands_trained']:,} of {hands:,} hands; kept checkpoint {best['hands']:,}")
