"""Opponent-pool self-play against latest-only self-play, and checkpoint progression (#14, #13).

Reads runs trained by bench/train_demo.py --keep-checkpoints (one directory per run, named
<variant>-seed<k>: run.json, log.jsonl, ckpt-<hands>.pt) and evaluates, on test deals that nothing
else uses (TEST_SEED 34000; checkpoints are selected on the validation deals, 5000):

  progression  every ckpt-<hands>.pt against the seven bots of winrate.SUITE: greedy, 4 seats, stacks
               200 (100 bb) reset every hand, blinds 1/2, duplicate deals (each deal replayed with the
               agent in each of the 4 seats; bb/100 and its 95% CI over deals, as winrate.evaluate).
               The kept checkpoint (run.json kept_checkpoint_hands: best validation score) is one of them.
  cross        each run's kept checkpoint, one seat, against three copies of every other run's kept
               checkpoint (all played greedily), same deals.

Every cell is one file in OUT/cells (per-deal scores in OUT/scores, for paired comparisons), reused
while the weights, deals and seed match. `report` writes OUT/tables.md, OUT/results.json,
OUT/progression.csv and OUT/progression.png. The bots in UNSEEN were never in a league's bot pool
(train_demo.BOTS holds random, call and tight).

    python bench/league_eval.py cells bench/raw/league/runs --out bench/raw/league --deals 4000 [--jobs 2]
    python bench/league_eval.py report bench/raw/league/runs --out bench/raw/league
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_demo import STACK, build
from winrate import SUITE, evaluate, greedy_policy

TEST_SEED = 34000
UNSEEN = ("tag", "lag", "potodds", "bluff")
VARIANTS = {"latest": "latest-only self-play", "snap": "league: snapshots", "snapbot": "league: snapshots + bots"}
COLORS = {"latest": "#555555", "snap": "#1f77b4", "snapbot": "#d95f02"}
CANDIDATE = "mlp-s"
HASH_LEN = 12


def runs_in(runs_dir):
    """{run name: (directory, run.json, [checkpoint hands])}, ordered by variant then seed."""
    out = {}
    for f in Path(runs_dir).glob("*/run.json"):
        run = json.loads(f.read_text())
        ck = sorted(int(p.stem.split("-")[1]) for p in f.parent.glob("ckpt-*.pt"))
        out[f.parent.name] = (f.parent, run, ck)
    key = lambda n: (list(VARIANTS).index(n.rsplit("-seed", 1)[0]), n)
    return {n: out[n] for n in sorted(out, key=key)}


def policy(path):
    model = build(CANDIDATE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return greedy_policy(model.eval()), hashlib.sha256(Path(path).read_bytes()).hexdigest()[:HASH_LEN]


def cell(job):
    out, run, ckpt, path, opp, opp_path, deals = job
    name = f"{run}_{ckpt}_vs_{opp}"
    pol, sha = policy(path)
    if opp_path:
        bot, osha = policy(opp_path)
    else:
        bot, osha = SUITE[opp], None
    f = Path(out) / "cells" / f"{name}.json"
    if f.exists():
        c = json.loads(f.read_text())
        if (c["sha"], c["opp_sha"], c["deals"], c["seed"]) == (sha, osha, deals, TEST_SEED):
            return c
    torch.set_num_threads(1)
    t0, c0 = time.time(), time.process_time()
    with torch.no_grad():
        r = evaluate(pol, bot, deals, TEST_SEED, stack=STACK, keep_scores=True)
    scores = r.pop("scores")
    r.update({"run": run, "ckpt": ckpt, "opponent": opp, "deals": deals, "seed": TEST_SEED, "sha": sha, "opp_sha": osha,
              "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1), "load_average": os.getloadavg(),
              "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    for sub in ("cells", "scores"):
        (Path(out) / sub).mkdir(parents=True, exist_ok=True)
    np.save(Path(out) / "scores" / f"{name}.npy", scores.astype(np.float32))
    f.write_text(json.dumps(r, indent=1) + "\n")
    return r


def jobs_for(runs, out, deals):
    jobs = []
    for run, (d, info, cks) in runs.items():
        jobs += [(str(out), run, n, str(d / f"ckpt-{n}.pt"), o, None, deals) for n in cks for o in SUITE]
    for a, (da, ia, _) in runs.items():
        for b, (db, ib, _) in runs.items():
            if a != b:
                ka, kb = ia["kept_checkpoint_hands"], ib["kept_checkpoint_hands"]
                jobs.append((str(out), a, ka, str(da / f"ckpt-{ka}.pt"), f"{b}@{kb}", str(db / f"ckpt-{kb}.pt"), deals))
    return jobs


def stats(scores):
    """bb/100 and its 95% CI over deals from per-deal scores in bb."""
    m, h = scores.mean(), 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
    return 100 * m, 100 * (m - h), 100 * (m + h)


def fmt(t):
    return f"{t[0]:,.0f} [{t[1]:,.0f}, {t[2]:,.0f}]"


def report(runs, out):
    out = Path(out)
    sc = lambda run, ck, opp: np.load(out / "scores" / f"{run}_{ck}_vs_{opp}.npy").astype(np.float64)
    unseen = lambda run, ck: np.mean([sc(run, ck, o) for o in UNSEEN], axis=0)
    deals = len(sc(next(iter(runs)), next(iter(runs.values()))[2][0], "random"))
    seeds = sorted({int(n.rsplit("-seed", 1)[1]) for n in runs})
    bots = list(SUITE)
    L, res = [], {"deals": deals, "seed": TEST_SEED, "runs": {}}

    L += [f"Kept checkpoint (best validation score) of each run against the suite: bb/100 [95% CI over deals], {deals:,} duplicate deals per cell, "
          f"seed {TEST_SEED}. Last column: the mean over the four unseen bots ({', '.join(UNSEEN)}), per deal.", "",
          "| run | kept at hands | val score | " + " | ".join(bots) + " | unseen mean |", "|---|---:|---:|" + "---:|" * (len(bots) + 1)]
    for run, (_, info, cks) in runs.items():
        k = info["kept_checkpoint_hands"]
        row = {o: stats(sc(run, k, o)) for o in bots}
        un = stats(unseen(run, k))
        res["runs"][run] = {"kept": k, "val_score": info["kept_val_score"], "league": info.get("league"), "suite": row, "unseen_mean": un}
        L.append(f"| {run} | {k:,} | {info['kept_val_score']:,.0f} | " + " | ".join(fmt(row[o]) for o in bots) + f" | {fmt(un)} |")

    last = lambda run: runs[run][2][-1]
    L += ["", "The last checkpoint (end of training) instead of the kept one, same cells.", "",
          "| run | hands | " + " | ".join(bots) + " | unseen mean |", "|---|---:|" + "---:|" * (len(bots) + 1)]
    for run in runs:
        k = last(run)
        L.append(f"| {run} | {k:,} | " + " | ".join(fmt(stats(sc(run, k, o))) for o in bots) + f" | {fmt(stats(unseen(run, k)))} |")

    for label, ckpt in (("kept checkpoint", lambda r: runs[r][1]["kept_checkpoint_hands"]), ("last checkpoint", last)):
        L += ["", f"Paired difference from latest-only self-play at the same seed, {label}: mean over deals of (variant - latest) on the same deals, "
              "bb/100 [95% CI over deals]. The interval covers the deals, not the retraining: it does not include how much a run changes with its "
              "seed.", "", "| seed | variant | " + " | ".join(bots) + " | unseen mean |", "|---|---|" + "---:|" * (len(bots) + 1)]
        for s in seeds:
            base = f"latest-seed{s}"
            for v in ("snap", "snapbot"):
                run = f"{v}-seed{s}"
                if run in runs and base in runs:
                    d = lambda f: stats(f(run, ckpt(run)) - f(base, ckpt(base)))
                    L.append(f"| {s} | {v} | " + " | ".join(fmt(d(lambda r, k, o=o: sc(r, k, o))) for o in bots) + f" | {fmt(d(unseen))} |")

    L += ["", "Mean over the seeds of each run's own result (kept checkpoint), no interval: two seeds do not support one.", "",
          "| variant | " + " | ".join(bots) + " | unseen mean |", "|---|" + "---:|" * (len(bots) + 1)]
    for v in VARIANTS:
        rs = [r for r in runs if r.startswith(v + "-seed")]
        if rs:
            L.append(f"| {v} | " + " | ".join(f"{np.mean([res['runs'][r]['suite'][o][0] for r in rs]):,.0f}" for o in bots)
                     + f" | {np.mean([res['runs'][r]['unseen_mean'][0] for r in rs]):,.0f} |")

    names = [n for n in runs]
    key = lambda n: f"{n}@{runs[n][1]['kept_checkpoint_hands']}"
    cross = {(a, b): json.loads((out / "cells" / f"{a}_{runs[a][1]['kept_checkpoint_hands']}_vs_{key(b)}.json").read_text())
             for a in names for b in names if a != b}
    L += ["", "Cross-play, kept checkpoints: the run on the left in one seat against three copies of the run on the top, bb/100 [95% CI over deals], "
          f"{deals:,} duplicate deals, seed {TEST_SEED}. Positive: the row run wins.", "",
          "| run \\ copies | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for a in names:
        L.append(f"| {a} | " + " | ".join("-" if a == b else fmt((cross[a, b]['bb100'], cross[a, b]['ci95_lo'], cross[a, b]['ci95_hi'])) for b in names) + " |")
    vs = list(VARIANTS)
    L += ["", "Cross-play by variant: mean bb/100 of the row variant's runs against the column variant's runs (every pairing of different runs; "
          "for the same variant, the two runs of different seeds), with the lowest and highest pairing.", "",
          "| variant \\ copies | " + " | ".join(vs) + " |", "|---|" + "---:|" * len(vs)]
    for va in vs:
        cols = []
        for vb in vs:
            x = [cross[a, b]["bb100"] for a in names for b in names if a != b and a.startswith(va + "-") and b.startswith(vb + "-")]
            cols.append(f"{np.mean(x):,.0f} ({min(x):,.0f} to {max(x):,.0f})" if x else "-")
        L.append(f"| {va} | " + " | ".join(cols) + " |")
    res["cross"] = {f"{a} vs {b}": {k: c[k] for k in ("bb100", "ci95_lo", "ci95_hi", "deals")} for (a, b), c in cross.items()}

    prog = [{"run": run, "variant": run.rsplit("-seed", 1)[0], "seed": int(run.rsplit("-seed", 1)[1]), "hands": k, "opponent": o,
             **dict(zip(("bb100", "ci95_lo", "ci95_hi"), stats(unseen(run, k) if o == "unseen mean" else sc(run, k, o))))}
            for run, (_, _, cks) in runs.items() for k in cks for o in (*bots, "unseen mean")]
    with (out / "progression.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prog[0]))
        w.writeheader(); w.writerows({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in prog)
    figure(prog, out / "progression.png", deals)
    (out / "tables.md").write_text("\n".join(L) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    return L


def figure(prog, path, deals):
    panels = [*SUITE, "unseen mean"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, opp in zip(axes.flat, panels):
        for i, v in enumerate(VARIANTS):
            for s, ls in ((1, "-"), (2, "--")):
                rows = [r for r in prog if r["opponent"] == opp and r["variant"] == v and r["seed"] == s]
                if not rows:
                    continue
                x = np.array([r["hands"] for r in rows]) + (i - 1) * 600  # dodge so the bars do not overlap
                y = np.array([r["bb100"] for r in rows])
                err = np.array([[r["bb100"] - r["ci95_lo"] for r in rows], [r["ci95_hi"] - r["bb100"] for r in rows]])
                ax.errorbar(x, y, yerr=err, color=COLORS[v], ls=ls, marker="o", ms=3.5, lw=1.4, elinewidth=0.8, capsize=2, alpha=0.9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(opp + (" (mean of tag, lag, potodds, bluff)" if opp == "unseen mean" else (" (unseen)" if opp in UNSEEN else " (in the bot pool)")), fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("training hands")
    for ax in axes[:, 0]:
        ax.set_ylabel("bb/100")
    handles = [plt.Line2D([], [], color=COLORS[v], marker="o", label=VARIANTS[v]) for v in VARIANTS]
    handles += [plt.Line2D([], [], color="black", ls=ls, label=f"seed {s}") for s, ls in ((1, "-"), (2, "--"))]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.12), fontsize=9)
    fig.suptitle(f"Checkpoint progression against the opponent suite (greedy, 100 bb, {deals:,} duplicate deals per point, seed {TEST_SEED}, 95% CI over deals)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["cells", "report"])
    ap.add_argument("runs", help="directory of <variant>-seed<k> run directories")
    ap.add_argument("--out", default=str(ROOT / "bench" / "raw" / "league"))
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    runs = runs_in(a.runs)
    if a.part == "cells":
        t0 = time.time()
        jobs = jobs_for(runs, a.out, a.deals)
        with Pool(a.jobs) as pool:
            for i, c in enumerate(pool.imap_unordered(cell, jobs, chunksize=1), 1):
                print(i, len(jobs), c["run"], c["ckpt"], c["opponent"], round(c["bb100"]), c["cpu_s"], flush=True)
        print(f"cells: wall {time.time() - t0:.0f} s, load average {os.getloadavg()}")
    print("\n".join(report(runs, a.out)))


if __name__ == "__main__":
    main()
