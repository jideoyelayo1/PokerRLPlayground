"""Re-measure the selected demo agents on fresh deals.

bench/eval_demo.py ranks the nine candidates on the evaluation deals (seed 7000) and reports the
winner's results on those same deals, so they are a little optimistic (the best of nine noisy
estimates). This plays each selected network against each baseline again, same method, on
deals from another seed that nothing else uses. It does not change the selection.

    python bench/confirm_demo.py [--seed 9000] [--deals 40000] [--jobs 3] [--out demo_confirm]

Each cell also counts the agent's folds chosen when checking was legal ("free_folds").
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import torch

from eval_demo import RAW, fmt, load
from train_demo import STACK
from winrate import BOTS, evaluate, greedy_policy


def cell(job):
    run, bname, deals, seed = job
    torch.set_num_threads(1)
    model, seq, _ = load(run)
    pol = greedy_policy(model, seq)
    with torch.no_grad():
        r = evaluate(pol, BOTS[bname], deals, seed, stack=STACK)
    r.update({"agent": run, "baseline": bname, "seed": seed, "actions": pol.counts, "free_folds": pol.free_folds,
              "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    return {k: float(v) if hasattr(v, "item") else v for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--deals", type=int, default=40_000)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="demo_confirm", help="bench/raw/<out>.json")
    a = ap.parse_args()
    seats = json.loads((RAW / "demo_eval.json").read_text())["seats"]
    runs = list(dict.fromkeys(seats))
    t0 = time.time()
    with Pool(a.jobs) as pool:
        cells = pool.map(cell, [(r, b, a.deals, a.seed) for r in runs for b in BOTS], chunksize=1)
    res = {(c["agent"], c["baseline"]): c for c in cells}
    (RAW / f"{a.out}.json").write_text(json.dumps({"seed": a.seed, "deals": a.deals, "wall_s": round(time.time() - t0, 1),
                                                       "load_average": os.getloadavg(), "cells": cells}, indent=1) + "\n")
    print(f"{a.deals:,} duplicate deals per cell from seed {a.seed}. bb/100 [95% CI over deals].\n")
    print("| run | vs random | vs call | vs tight |\n|---|---:|---:|---:|")
    for r in runs:
        print(f"| {r} | " + " | ".join(fmt(res[(r, b)]) for b in BOTS) + " |")
    for c in cells:
        n, f = sum(c["actions"]), c["free_folds"]
        print(f"{c['agent']} vs {c['baseline']}: {n:,} decisions, {c['actions'][0]:,} folds, {f:,} of them with check legal "
              f"({100 * f / n:.1f}% of decisions, {100 * f / max(1, c['actions'][0]):.0f}% of folds)")


if __name__ == "__main__":
    main()
