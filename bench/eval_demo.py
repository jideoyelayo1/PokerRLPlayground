"""Evaluate every trained demo candidate on the evaluation deals and apply the selection rule
written in bench/SELECTION.md.

Each candidate (weights/runs/*/model.pt, the checkpoint its run kept on the validation deals)
plays greedily in the demo format (4 seats, stacks 200, blinds 1/2, reset every hand) against
three copies of each baseline, with bench/winrate.py's duplicate deals: EVAL_DEALS deals from
EVAL_SEED, each played 4 times with the agent in every seat. The random bot in the agent's seat
is the control. Writes bench/raw/demo_eval.json (every cell) and bench/raw/demo_table.md.

    python bench/eval_demo.py --jobs 5
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

from train_demo import CANDIDATES, RUNS, STACK, build
from winrate import BOTS, bot_random, evaluate, greedy_policy

EVAL_SEED, EVAL_DEALS = 7000, 40_000
RAW = ROOT / "bench" / "raw"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def load(run):
    meta = json.loads((RUNS / run / "run.json").read_text())
    model = build(meta["candidate"])
    model.load_state_dict(torch.load(RUNS / run / "model.pt", weights_only=True))
    return model.eval(), CANDIDATES[meta["candidate"]]["sequence"], meta


def cell(job):
    run, bname, deals = job
    torch.set_num_threads(1)
    t0 = time.time()
    if run == "random-bot":
        pol = bot_random
    else:
        model, seq, _ = load(run)
        pol = greedy_policy(model, seq)
    with torch.no_grad():
        r = evaluate(pol, BOTS[bname], deals, EVAL_SEED, stack=STACK)
    r.update({"agent": run, "baseline": bname, "seed": EVAL_SEED, "wall_s": round(time.time() - t0, 1),
              "actions": getattr(pol, "counts", None)})
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()}


def fmt(c):
    return f"{c['bb100']:,.0f} [{c['ci95_lo']:,.0f}, {c['ci95_hi']:,.0f}]"


def select(runs, res, params):
    """bench/SELECTION.md: rank by the worst, over the baselines, of the lower end of the 95% interval
    on bb/100; ties by the mean bb/100 over the baselines, then by fewer parameters. Seats get the top
    three networks, unless the top one is clearly best (its score is above every other network's
    worst-case point estimate), in which case it plays all three seats."""
    score = {r: min(res[(r, b)]["ci95_lo"] for b in BOTS) for r in runs}
    worst = {r: min(res[(r, b)]["bb100"] for b in BOTS) for r in runs}
    mean = {r: statistics.mean(res[(r, b)]["bb100"] for b in BOTS) for r in runs}
    ranked = sorted(runs, key=lambda r: (-score[r], -mean[r], params[r]))
    clearly = len(ranked) > 1 and all(score[ranked[0]] > worst[r] for r in ranked[1:])
    seats = [ranked[0]] * 3 if clearly else ranked[:3]
    return ranked, score, seats, clearly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--deals", type=int, default=EVAL_DEALS)
    a = ap.parse_args()
    runs = sorted(p.parent.name for p in RUNS.glob("*/run.json"))
    jobs = [(r, b, a.deals) for r in runs + ["random-bot"] for b in BOTS]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    res = {(c["agent"], c["baseline"]): c for c in cells}
    metas = {r: json.loads((RUNS / r / "run.json").read_text()) for r in runs}
    params = {r: metas[r]["parameters"] for r in runs}
    ranked, score, seats, clearly = select(runs, res, params)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"commit": commit, "eval_seed": EVAL_SEED, "deals": a.deals, "wall_s": round(time.time() - t0, 1),
           "jobs": a.jobs, "load_average": os.getloadavg(), "cells": cells, "ranking": ranked,
           "score": score, "seats": seats, "one_network_in_all_seats": clearly}
    (RAW / "demo_eval.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = [f"Evaluation: {a.deals:,} duplicate deals ({4 * a.deals:,} hands) per cell, deals from seed {EVAL_SEED}, "
             f"4 seats, stacks 200 (100 big blinds), blinds 1/2. bb/100 [95% CI over deals]. Commit {commit}.", "",
             "| run | parameters | kept checkpoint (hands) | vs random | vs call | vs tight | score (worst lower bound) | rank |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranked:
        m = metas[r]
        lines.append(f"| {r} | {params[r]:,} | {m['kept_checkpoint_hands']:,} | " + " | ".join(fmt(res[(r, b)]) for b in BOTS)
                     + f" | {score[r]:,.0f} | {ranked.index(r) + 1} |")
    lines.append("| random-bot (control) | | | " + " | ".join(fmt(res[("random-bot", b)]) for b in BOTS) + " | | |")
    lines += ["", "Across seeds, per architecture: mean of the per-seed bb/100 [95% t-interval over seeds].", "",
              "| candidate | seeds | vs random | vs call | vs tight |", "|---|---:|---:|---:|---:|"]
    for c in CANDIDATES:
        rs = [r for r in runs if metas[r]["candidate"] == c]
        if len(rs) < 2:
            continue
        cols = []
        for b in BOTS:
            v = [res[(r, b)]["bb100"] for r in rs]
            m, h = statistics.mean(v), T95[len(v) - 1] * statistics.stdev(v) / len(v) ** 0.5
            cols.append(f"{m:,.0f} [{m - h:,.0f}, {m + h:,.0f}]")
        lines.append(f"| {c} | {len(rs)} | " + " | ".join(cols) + " |")
    lines += ["", "Training (bench/train_demo.py; wall time with the other runs in parallel; decisions per hand of "
              "self-play over the last tenth of the run).", "",
              "| run | hands | wall | CPU | hands/s | decisions/hand | kept checkpoint | its validation score |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        m = metas[r]
        last = json.loads((RUNS / r / "log.jsonl").read_text().splitlines()[-1])
        lines.append(f"| {r} | {m['hands']:,} | {m['wall_s'] / 60:.0f} min | {m['cpu_s'] / 3600:.2f} h | "
                     f"{m['hands'] / m['wall_s']:.0f} | {last.get('decisions_per_hand', float('nan')):.1f} | "
                     f"{m['kept_checkpoint_hands']:,} | {m['kept_val_score']:,.0f} |")
    lines += ["", "Seats: " + (f"{seats[0]} in all three (clearly best)" if clearly else ", ".join(seats)) + ".",
              "Beats each baseline (lower end of the 95% interval above 0): "
              + "; ".join(f"{r}: " + ", ".join(f"{b} {'yes' if res[(r, b)]['ci95_lo'] > 0 else 'no'}" for b in BOTS)
                          for r in dict.fromkeys(seats)) + "."]
    (RAW / "demo_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
