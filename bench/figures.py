"""The README's two figures, from the current code (bench/RESULTS.md, section 7).

1. Result by starting hand (images/selfplay_result_by_starting_hand.*). The selected agent
   (mlp-s seed 3, weights/runs/mlp-s-seed3/model.pt, the same tensors as weights/demo) in all four
   seats, greedy, with the play-time option no_free_fold on, as the demo seats it. 4 seats, blinds
   1/2, stacks 200 reset before every hand. HANDS hands; hand h is dealt from deck seed
   SEED * 1,000,003 + h, so the result does not depend on --jobs. For every hand and seat: the hole
   cards, read straight after the deal, and the seat's result in big blinds, blinds and folded
   hands included (each hand's results sum to zero; asserted). Per starting-hand class (169): the
   count, the mean in big blinds per hand and its standard error (sd / sqrt(count)).
   Reference: each class's all-in equity against three random hands (its expected share of the
   pot with five board cards, ties split), EQ_TRIALS seeded deals per class through the engine's
   evaluator, and the Spearman rank correlation between the two: over all 169 classes, within the
   84 highest by equity and within the other 85.
   Replication: the same HANDS hands on a second deck seed, REP_SEED, reported in the log and written
   to bench/raw/ (Fig. 1 is drawn from SEED only).
   The button stays on seat 0: the Python binding has no reset_stacks, so the stacks are reset with
   reset_tournament, which puts the button back on seat 0 (the demo's table moves it). With the same
   network in every seat and an input relative to the player and the button, that changes nothing
   but which seat gets the odd chip of a split pot.
2. Validation over training (images/validation_over_training.*): the checkpoint rows already in
   weights/runs/*/log.jsonl (no retraining).

Writes each figure as .png, .svg and .csv (exactly what is plotted) in images/. --plot-only redraws
both from the committed CSVs.

    python bench/figures.py [--jobs 4] [--plot-only]
"""
import argparse
import csv
import json
import random
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

import PokerEngine.poker_engine as pe
from eval_demo import load
from ReinforcementLearningEnvironment.helper import pocket_169_key, pocket_key
from train_demo import RUNS, SEATS, STACK
from winrate import BOTS, greedy_policy, play_hand

AGENT = "mlp-s-seed3"
# Chosen before any result was seen, and used by nothing else (training 0-3, 11 and
# 101-104, validation 5000, evaluation 7000, confirmation 9000 and 13000, web replay 20261118).
SEED, HANDS = 17000, 250_000  # a suited class (4 of 1,326 combinations) gets about 3,000 seat-hands
# Replication seed, declared before it was run (after review of the first), used by nothing else.
REP_SEED = 23000
EQ_SEED, EQ_TRIALS = 17001, 50_000
QUOTED = ("AA", "KK", "QQ", "JJ", "AKs", "AKo", "22", "72o", "K7s", "K2o")  # hands whose ranks the text quotes
RANKS = "AKQJT98765432"
IMAGES = ROOT / "images"
FIG1, FIG2 = IMAGES / "selfplay_result_by_starting_hand", IMAGES / "validation_over_training"
REP_CSV = ROOT / "bench" / "raw" / f"selfplay_result_by_starting_hand_seed{REP_SEED}.csv"
FAMILY = {"mlp-s": "#2a78d6", "mlp-l": "#eb6834", "gru": "#1baf7a"}  # colour-blind-safe in this order
SELECTED = "#0b0b0b"
plt.rcParams.update({"font.size": 11, "svg.hashsalt": "figures"})


def classes():
    """The 169 starting-hand classes in grid order, row by row, as pocket_169_key writes them."""
    return [a + b if i == j else a + b + "s" if i < j else b + a + "o"
            for i, a in enumerate(RANKS) for j, b in enumerate(RANKS)]


def cell(c):
    """(row, column) in the 13 x 13 grid: pairs on the diagonal, suited above, offsuit below."""
    i, j = RANKS.index(c[0]), RANKS.index(c[1])
    return (j, i) if c.endswith("o") else (i, j)


def self_play(job):
    """Hands first .. first + n - 1: rows (hand, seat, class, result in bb), and the actions chosen."""
    first, n, seed = job
    torch.set_num_threads(1)
    model, seq, _ = load(AGENT)
    pol = greedy_policy(model, seq, free_fold_rule=True)
    g = pe.GameEngine(SEATS, STACK, 2, 1)
    rng = random.Random(seed)  # play_hand's argument; the greedy policy draws nothing from it
    rows, holes = [], []
    with torch.no_grad():
        for h in range(first, first + n):
            g.seed(seed * 1_000_003 + h)
            bb = play_hand(g, [pol] * SEATS, rng, STACK, holes=holes)
            assert sum(bb) == 0, (h, bb)
            rows += [(h, s, pocket_169_key(pocket_key(*holes[s])), bb[s]) for s in range(SEATS)]
    return rows, pol.counts


def by_class(rows):
    """{class: (count, mean bb, standard error)} over the rows (hand, seat, class, bb)."""
    vals = {}
    for _, _, c, x in rows:
        vals.setdefault(c, []).append(x)
    out = {}
    for c, v in vals.items():
        v = np.array(v)
        out[c] = (len(v), float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
    return out


def equity(job):
    """All-in equity of a class against three random hands: (mean pot share, standard error)."""
    k, c = job
    deck = [pe.Card(pe.Suits(s), pe.Ranks(r)) for r in range(2, 15) for s in range(1, 5)]
    rank = {r: 14 - i for i, r in enumerate(RANKS)}
    hero = [(rank[c[0]], 1), (rank[c[1]], 1 if c.endswith("s") else 2)]
    hole = [x for x in deck if (int(x.rank), int(x.suit)) in hero]
    rest = [x for x in deck if x not in hole]
    rng = random.Random(EQ_SEED * 1000 + k)
    share = np.empty(EQ_TRIALS)
    for t in range(EQ_TRIALS):
        d = rng.sample(rest, 11)
        board = d[6:]
        hands = [pe.best_hand_of_7(hole + board)] + [pe.best_hand_of_7(d[2 * i:2 * i + 2] + board) for i in range(3)]
        best = max(hands, key=lambda h: (int(h.cat), list(h.kickers)))
        tied = [not (h < best) for h in hands]
        share[t] = tied[0] / sum(tied)
    return float(share.mean()), float(share.std(ddof=1) / np.sqrt(EQ_TRIALS))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def compute(jobs):
    chunk = 10_000
    with Pool(jobs) as pool:
        t0 = time.time()
        eq = pool.map(equity, list(enumerate(classes())), chunksize=1)
        print(f"equity: {EQ_TRIALS:,} deals per class, seed {EQ_SEED} x 1,000 + class index, {time.time() - t0:.0f} s")
        for seed, path in ((SEED, FIG1.with_suffix(".csv")), (REP_SEED, REP_CSV)):
            t0 = time.time()
            parts = pool.map(self_play, [(f, min(chunk, HANDS - f), seed) for f in range(0, HANDS, chunk)], chunksize=1)
            rows = [r for p, _ in parts for r in p]
            counts = np.sum([c for _, c in parts], axis=0)
            hands = {}
            for h, _, _, x in rows:
                hands[h] = hands.get(h, 0) + x
            assert len(hands) == HANDS and all(v == 0 for v in hands.values())
            agg = by_class(rows)
            assert sorted(agg) == sorted(classes()) and sum(n for n, _, _ in agg.values()) == SEATS * HANDS
            print(f"self-play: {HANDS:,} hands ({SEATS * HANDS:,} seat-hands), deck seed {seed} x 1,000,003 + hand, "
                  f"{time.time() - t0:.0f} s with {jobs} processes; every hand's results sum to zero")
            print(f"decisions: {counts.sum():,} ({counts.sum() / HANDS:.1f} per hand); actions chosen (fold, check, call, "
                  f"raise 1 bb, raise half pot, raise pot, all-in): {', '.join(f'{x:,} ({100 * x / counts.sum():.1f}%)' for x in counts)}")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["class", "row", "col", "count", "mean_bb_per_hand", "se_bb_per_hand", "equity_vs_3_random", "equity_se"])
                for c, (e, ese) in zip(classes(), eq):
                    n, m, se = agg[c]
                    w.writerow([c, *cell(c), n, f"{m:.6f}", f"{se:.6f}", f"{e:.6f}", f"{ese:.6f}"])
    with open(FIG2.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "candidate", "seed", "selected", "kept", "hands", "baseline", "bb100", "ci95_lo", "ci95_hi"])
        for d in sorted(RUNS.glob("*/log.jsonl")):
            meta = json.loads((d.parent / "run.json").read_text())
            for row in map(json.loads, d.read_text().splitlines()):
                for b in BOTS:
                    v = row["val"][b]
                    w.writerow([d.parent.name, meta["candidate"], meta["seed"], int(d.parent.name == AGENT),
                                int(row["hands"] == meta["kept_checkpoint_hands"]), row["hands"],
                                b, f"{v['bb100']:.4f}", f"{v['lo']:.4f}", f"{v['hi']:.4f}"])


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def report_fig1(rows, seed):
    print(f"--- result by starting hand, deck seed {seed}")
    c = [r["class"] for r in rows]
    m, se, e = (np.array([float(r[k]) for r in rows]) for k in ("mean_bb_per_hand", "se_bb_per_hand", "equity_vs_3_random"))
    n = np.array([int(r["count"]) for r in rows])
    rank_m, rank_e = 1 + np.argsort(np.argsort(-m)), 1 + np.argsort(np.argsort(-e))
    order = np.argsort(-m)
    print(f"seat-hands per class: {n.min():,} to {n.max():,}; standard error {se.min():.3f} to {se.max():.3f} bb per hand")
    print("highest mean (bb per hand, standard error, equity rank of 169): "
          + "; ".join(f"{c[i]} {m[i]:+.2f} ({se[i]:.2f}, {rank_e[i]})" for i in order[:10]))
    print("lowest mean: " + "; ".join(f"{c[i]} {m[i]:+.2f} ({se[i]:.2f}, {rank_e[i]})" for i in order[::-1][:10]))
    kind = np.array([0 if x[0] == x[1] else 1 if x.endswith("s") else 2 for x in c])
    for k, name in enumerate(("pairs", "suited", "offsuit")):
        w = n * (kind == k)
        print(f"{name}: {int((kind == k).sum())} classes, {w.sum():,} seat-hands, mean {np.sum(w * m) / w.sum():+.3f} bb per hand")
    pos = [c[i] for i in np.flatnonzero(m > 0)]
    print(f"positive mean: {len(pos)} of {len(c)} classes ({', '.join(pos)}); the lowest mean is {m.min():+.2f}")
    print(f"negative mean: {int(np.sum(m < 0))} classes, {int(np.sum((m < 0) & (-m < se)))} of them within one standard error of zero")
    suited = {x[:2]: i for i, x in enumerate(c) if x.endswith("s")}
    d = np.array([m[suited[x[:2]]] - m[i] for i, x in enumerate(c) if x.endswith("o")])
    print(f"suited minus offsuit, same ranks (78 pairs of classes): mean {d.mean():+.3f} bb per hand, "
          f"suited higher in {int((d > 0).sum())} of 78")
    top, bottom = rank_e <= len(c) // 2, rank_e > len(c) // 2
    print(f"Spearman rank correlation, mean result against all-in equity vs three random hands: {spearman(m, e):.3f} "
          f"over all {len(c)}; {spearman(m[top], e[top]):.3f} within the {top.sum()} highest by equity; "
          f"{spearman(m[bottom], e[bottom]):.3f} within the other {bottom.sum()}")
    print(f"of the ten best hands by result, {int(np.sum(rank_e[order[:10]] <= 14))} are in equity's top 14")
    print("result rank of 169 (mean, equity rank): " + "; ".join(
        f"{x} {rank_m[c.index(x)]} ({m[c.index(x)]:+.2f}, {rank_e[c.index(x)]})" for x in QUOTED))


def report_fig2(rows):
    runs = sorted({r["run"] for r in rows})
    for b in BOTS:
        pts = {run: [r for r in rows if r["run"] == run and r["baseline"] == b] for run in runs}
        first, last = ([float(p[k]["bb100"]) for p in pts.values()] for k in (0, -1))
        sel = {int(r["hands"]): r for r in pts[AGENT]}
        print(f"vs {b}: first checkpoint {min(first):+,.0f} to {max(first):+,.0f} bb/100 over the {len(runs)} runs, "
              f"last {min(last):+,.0f} to {max(last):+,.0f}; {AGENT}: "
              + ", ".join(f"{float(r['bb100']):+,.0f} [{float(r['ci95_lo']):+,.0f}, {float(r['ci95_hi']):+,.0f}] at {h:,}"
                          + (" (kept)" if r["kept"] == "1" else "") for h, r in sorted(sel.items())))
    half = [(float(r["ci95_hi"]) - float(r["ci95_lo"])) / 2 for r in rows if r["run"] == AGENT]
    print(f"{AGENT}'s 95% intervals: +/-{min(half):.0f} to +/-{max(half):.0f} bb/100")
    score = {}
    for r in rows:
        k = (r["run"], int(r["hands"]))
        score[k] = min(score.get(k, float("inf")), float(r["ci95_lo"]))
    print("checkpoints whose 95% interval is above zero against all three baselines: "
          + ("; ".join(f"{run} at {h:,}" for (run, h), v in sorted(score.items()) if v > 0) or "none"))


def plot_fig1(rows):
    grid = np.full((13, 13), np.nan)
    for r in rows:
        grid[int(r["row"]), int(r["col"])] = float(r["mean_bb_per_hand"])
    lim = np.nanmax(np.abs(grid))
    se = [float(r["se_bb_per_hand"]) for r in rows]
    fig = plt.figure(figsize=(12, 11.4))
    ax = fig.add_axes([0.03, 0.15, 0.84, 0.77])
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim)  # linear, symmetric about zero
    for r in rows:
        i, j, v = int(r["row"]), int(r["col"]), float(r["mean_bb_per_hand"])
        ax.text(j, i, f"{r['class']}\n{v:+.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if abs(v) > 0.55 * lim else "#0b0b0b")
    ax.set_xticks(range(13), list(RANKS)); ax.set_yticks(range(13), list(RANKS))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_frame_on(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.875, 0.2, 0.018, 0.65]))
    cb.set_label("mean result, big blinds per hand (blinds and folded hands included)")
    fig.suptitle("Self-play result by starting hand: mlp-s seed 3 in all four seats", y=0.975, fontsize=14)
    fig.text(0.05, 0.015,
             "Pairs on the diagonal, suited hands above it, offsuit below. Four copies of the selected network, greedy, with the play-time\n"
             "rule no_free_fold on, as in the demo: 4 seats, blinds 1/2, stacks 200 (100 big blinds) reset every hand; button on seat 0\n"
             f"(the demo moves it). {HANDS:,} hands ({SEATS * HANDS:,} seat-hands), deck seed {SEED}; standard error "
             f"{min(se):.2f} to {max(se):.2f} bb per hand.\n"
             f"Colour is linear to \u00b1{lim:.0f}, so the pale cells' detail is in their printed numbers. Self-play against copies of itself: not a\n"
             "measure of strength against other opponents.",
             fontsize=10.5, color="#52514e", va="bottom")
    save(fig, FIG1)


def plot_fig2(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharex=True)
    runs = sorted({r["run"] for r in rows}, key=lambda x: (x == AGENT, x))  # a total order; the selected run last
    for ax, b in zip(axes, BOTS):
        for run in runs:
            pts = [r for r in rows if r["run"] == run and r["baseline"] == b]
            x = np.array([int(r["hands"]) for r in pts])
            y, lo, hi = (np.array([float(r[k]) for r in pts]) for k in ("bb100", "ci95_lo", "ci95_hi"))
            if run == AGENT:
                ax.fill_between(x, lo, hi, color=SELECTED, alpha=0.12, lw=0)
                ax.plot(x, y, color=SELECTED, lw=2.5, marker="o", ms=5, zorder=3)
                k = [r["kept"] == "1" for r in pts]
                ax.plot(x[k], y[k], "o", ms=12, mfc="none", mec=SELECTED, mew=2, zorder=4)
            else:
                ax.plot(x, y, color=FAMILY[pts[0]["candidate"]], lw=1.3, marker="o", ms=3, alpha=0.85)
        ax.axhline(0, color="#52514e", lw=0.9, ls="--", zorder=1)
        ax.set_title(f"vs {b} bots", fontsize=12)
        ax.set_xlabel("training hands")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("validation result, bb/100")
    handles = [matplotlib.lines.Line2D([], [], color=SELECTED, lw=2.5, marker="o", label="mlp-s seed 3 (selected)")]
    handles += [matplotlib.lines.Line2D([], [], color=v, lw=1.3, marker="o", ms=3, label=f"{k}, seeds 1-2" if k == "mlp-s" else f"{k}, seeds 1-3")
                for k, v in FAMILY.items()]
    handles += [matplotlib.lines.Line2D([], [], color=SELECTED, ls="", marker="o", ms=12, mfc="none", mew=2, label="kept checkpoint")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.text(0.06, 0.02,
             "One point per checkpoint (a tenth of each run): greedy play against three copies of each baseline on the "
             "validation deals (1,000 duplicate deals,\nseed 5000; no_free_fold off). Shaded: the selected run's 95% "
             "interval. The GRU runs trained for 150,000 hands, the MLPs for 400,000.",
             fontsize=10, color="#52514e", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.22)
    save(fig, FIG2)


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=100)
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the committed CSVs")
    a = ap.parse_args()
    if not a.plot_only:
        compute(a.jobs)
    rows1, rows2 = read(FIG1.with_suffix(".csv")), read(FIG2.with_suffix(".csv"))
    report_fig1(rows1, SEED)
    report_fig1(read(REP_CSV), REP_SEED)
    report_fig2(rows2)
    plot_fig1(rows1)
    plot_fig2(rows2)


if __name__ == "__main__":
    main()
