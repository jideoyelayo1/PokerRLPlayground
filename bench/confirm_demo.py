"""Re-measure the selected demo agents on fresh deals.

bench/eval_demo.py ranks the nine candidates on the evaluation deals (seed 7000) and reports the
winner's results on those same deals, so they are a little optimistic (the best of nine noisy
estimates). This plays each selected network against each baseline again, same method, on
deals from another seed that nothing else uses. It does not change the selection.

    python bench/confirm_demo.py [--seed 9000] [--deals 40000] [--jobs 3] [--out demo_confirm]
                                 [--no-free-fold off|on|both]

Each cell also counts the agent's folds chosen when checking was legal ("free_folds", the
network's own choice before any rule). --no-free-fold on plays with the play-time option
no_free_fold (if Check is allowed, Fold is treated as not allowed), which was decided after the
selection and is not part of the pre-registered evaluation; both plays the same deals each way.
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
import numpy as np
import torch

from eval_demo import RAW, fmt, load
from train_demo import STACK
from winrate import BOTS, evaluate, greedy_policy


def cell(job):
    run, bname, deals, seed, rule = job
    torch.set_num_threads(1)
    model, seq, _ = load(run)
    pol = greedy_policy(model, seq, free_fold_rule=rule)
    with torch.no_grad():
        r = evaluate(pol, BOTS[bname], deals, seed, stack=STACK, keep_scores=True)
    r.update({"agent": run, "baseline": bname, "seed": seed, "no_free_fold": rule, "actions": pol.counts,
              "free_folds": pol.free_folds,
              "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    return {k: float(v) if isinstance(v, np.generic) else v for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--deals", type=int, default=40_000)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="demo_confirm", help="bench/raw/<out>.json")
    ap.add_argument("--no-free-fold", choices=("off", "on", "both"), default="off")
    a = ap.parse_args()
    rules = {"off": [False], "on": [True], "both": [False, True]}[a.no_free_fold]
    seats = json.loads((RAW / "demo_eval.json").read_text())["seats"]
    runs = list(dict.fromkeys(seats))
    t0 = time.time()
    with Pool(a.jobs) as pool:
        cells = pool.map(cell, [(r, b, a.deals, a.seed, rule) for rule in rules for r in runs for b in BOTS], chunksize=1)
    scores = {(c["agent"], c["baseline"], c["no_free_fold"]): c.pop("scores") for c in cells}
    res = {(c["agent"], c["baseline"], c["no_free_fold"]): c for c in cells}
    paired = []  # rule on minus off, deal by deal on the same deals
    if len(rules) == 2:
        for r in runs:
            for b in BOTS:
                d = scores[(r, b, True)] - scores[(r, b, False)]
                m, h = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                paired.append({"agent": r, "baseline": b, "on_minus_off_bb100": 100 * m,
                               "ci95_lo": 100 * (m - h), "ci95_hi": 100 * (m + h)})
    (RAW / f"{a.out}.json").write_text(json.dumps({"seed": a.seed, "deals": a.deals, "wall_s": round(time.time() - t0, 1),
                                                       "load_average": os.getloadavg(), "cells": cells,
                                                       "paired_on_minus_off": paired}, indent=1) + "\n")
    print(f"{a.deals:,} duplicate deals per cell from seed {a.seed}. bb/100 [95% CI over deals].\n")
    print("| run | no_free_fold | vs random | vs call | vs tight |\n|---|---|---:|---:|---:|")
    for rule in rules:
        for r in runs:
            print(f"| {r} | {'on' if rule else 'off'} | " + " | ".join(fmt(res[(r, b, rule)]) for b in BOTS) + " |")
    for p in paired:
        print(f"{p['agent']} vs {p['baseline']}: no_free_fold on minus off, deal by deal: "
              f"{p['on_minus_off_bb100']:,.1f} [{p['ci95_lo']:,.1f}, {p['ci95_hi']:,.1f}] bb/100")
    for c in cells:
        n, f = sum(c["actions"]), c["free_folds"]
        print(f"{c['agent']} vs {c['baseline']}, no_free_fold {'on' if c['no_free_fold'] else 'off'}: {n:,} decisions, "
              f"{c['actions'][0]:,} folds played; the network chose a fold with check legal {f:,} times "
              f"({100 * f / n:.1f}% of decisions)")


if __name__ == "__main__":
    main()
