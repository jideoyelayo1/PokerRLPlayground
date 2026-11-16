"""Markdown table from bench/raw/winrate_seed*.json.

Per seed: bb/100 with its 95% CI over deals. Across seeds: mean of the per-seed bb/100 with a
95% t-interval over seeds (this one includes the variation between training runs).

    python bench/summarise_winrate.py
"""
import json
import statistics
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


if __name__ == "__main__":
    main()
