"""Markdown table from bench/raw/winrate_seed*.json, plus what the agents did late in training.

Per seed: bb/100 with its 95% CI over deals. Across seeds: mean of the per-seed bb/100 with a
95% t-interval over seeds (this one includes the variation between training runs).

    python bench/summarise_winrate.py
"""
import csv
import json
import statistics
from collections import Counter
import sys
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def main(raw=Path(__file__).resolve().parent / "raw"):
    runs = [json.loads(p.read_text()) for p in sorted(raw.glob("winrate_seed*.json"))]
    if not runs:
        sys.exit("no winrate_seed*.json files")
    seeds = [r["train"]["seed"] for r in runs]
    cell = {(x["agent"], x["baseline"], r["train"]["seed"]): x for r in runs for x in r["results"]}
    agents = list(dict.fromkeys(x["agent"] for r in runs for x in r["results"]))
    bots = list(dict.fromkeys(x["baseline"] for r in runs for x in r["results"]))
    print("| agent | vs | " + " | ".join(f"seed {s}: bb/100 [95% CI]" for s in seeds) + " | mean over seeds [95% t-CI] |")
    print("|---|---|" + "---|" * (len(seeds) + 1))
    for a in agents:
        for b in bots:
            vals = [cell[(a, b, s)]["bb100"] for s in seeds]
            cols = [f"{c['bb100']:,.0f} [{c['ci95_lo']:,.0f}, {c['ci95_hi']:,.0f}]" for c in (cell[(a, b, s)] for s in seeds)]
            m = statistics.mean(vals)
            h = T95[len(vals) - 1] * statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
            print(f"| {a} | {b} | " + " | ".join(cols) + f" | {m:,.0f} [{m - h:,.0f}, {m + h:,.0f}] |")
    for r in runs:
        t = r["train"]
        print(f"\nseed {t['seed']}: trained {t['train_hands']} hands in {t['wall_s'] / 60:.1f} min wall "
              f"({t['cpu_s'] / 3600:.2f} CPU-hours, {t['tournament_resets']} tournament resets, {t['aborted_hands']} aborted hands); "
              f"evaluation {t['eval_wall_s'] / 60:.1f} min, {r['results'][0]['deals']} deals x 4 seats per cell")
    # First action of each player, last quarter of training (epsilon is already 0 after hand 80)
    print("\n| seed | seat (model) | first actions in the last 25% of training hands |\n|---|---|---|")
    for r in runs:
        seed, n = r["train"]["seed"], r["train"]["train_hands"]
        log = list(csv.DictReader(open(raw / f"train_seed{seed}.csv")))
        folded = sum(x["folded"] == "1" for x in log)
        print(f"| {seed} | all | {folded} of {len(log)} logged rows are folds ({100 * folded / len(log):.1f}%); the old logging dropped every one of them |")
        rows = [x for x in log if int(x["hand_id"]) >= 0.75 * n]
        for pid in range(4):
            c = Counter(x["first_action_name"] for x in rows if int(x["player_id"]) == pid)
            tot = sum(c.values())
            desc = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common()) if tot else "not dealt in (busted)"
            print(f"| {seed} | {pid} ({agents[pid]}) | {desc} (n={tot}) |")


if __name__ == "__main__":
    main()
