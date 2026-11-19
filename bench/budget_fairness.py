"""Training-budget fairness (#18): compare the demo candidates at equal hands and at equal CPU time.

Reads weights/runs/*/log.jsonl and run.json (the committed runs) and writes
bench/raw/budget_fairness.json and bench/raw/budget_fairness/tables.md:
  sample efficiency   val_score at equal hands (the last checkpoint at or before each hand budget)
  compute efficiency  val_score at equal CPU seconds (the last checkpoint at or before each budget)
  throughput          hands per CPU second and per wall second, parameters

The MLPs checkpoint every 40,000 hands and the GRU every 15,000, so only 120,000 is a checkpoint all
three share; at other budgets a cell shows the checkpoint used.

    python bench/budget_fairness.py
    python bench/budget_fairness.py --matched SCRATCH --run   # also train all three at 15,000 hands, seeds 1-2
    python bench/budget_fairness.py --matched SCRATCH         # only summarise runs already in SCRATCH
"""
import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "weights" / "runs"
OUT = ROOT / "bench" / "raw"
ORDER = ("mlp-s", "mlp-l", "gru")
HAND_BUDGETS = (40_000, 80_000, 120_000, 150_000)
CPU_BUDGETS = (150, 300, 600, 1100, 2100, 4800, 9300)
MATCHED_HANDS, MATCHED_CHECKPOINTS, MATCHED_SEEDS = 15_000, 3, (1, 2)
# two at a time on the shared machine
WAVES = [(("gru", 1), ("mlp-l", 1)), (("gru", 2), ("mlp-l", 2)), (("mlp-s", 1), ("mlp-s", 2))]


def load(runs_dir):
    out = {}
    for f in sorted(Path(runs_dir).glob("*/run.json")):
        run = json.loads(f.read_text())
        log = [json.loads(l) for l in (f.parent / "log.jsonl").read_text().splitlines()]
        out[(run["candidate"], run["seed"])] = {"run": run, "log": log}
    return sorted(out.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1]))


def at_or_before(log, key, budget):
    rows = [r for r in log if r[key] <= budget]
    if not rows:
        return None
    r = rows[-1]
    return {k: r[k] for k in ("hands", "cpu_s", "wall_s", "epsilon", "val_score")}


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def budget_table(runs, key, budgets, label):
    """Per-run cells plus one mean-over-seeds row per candidate; returns (markdown, json)."""
    js = {str(b): {f"{c}-seed{s}": at_or_before(v["log"], key, b) for (c, s), v in runs} for b in budgets}

    def cell(r, b):
        if r is None:
            return "-"
        # sample table: say which checkpoint was used when it is not the budget; compute table: always
        return f"{r['val_score']:.0f}" + (f" @{r['hands'] // 1000}k" if key == "cpu_s" or r["hands"] != b else "")

    rows = [[f"{c}-seed{s}"] + [cell(js[str(b)][f"{c}-seed{s}"], b) for b in budgets] for (c, s), _ in runs]
    for c in ORDER:
        row = [f"**{c} mean**"]
        for b in budgets:
            rs = [r for k, r in js[str(b)].items() if k.startswith(c + "-seed") and r]
            row.append(f"{statistics.mean(r['val_score'] for r in rs):.0f}" + (f" (n={len(rs)})" if len(rs) < 3 else "") if rs else "-")
        rows.append(row)
    return table([label] + [f"{b:,}" for b in budgets], rows), js


def throughput(runs):
    rows, js = [], {}
    for c in ORDER:
        rs = [v for (cc, _), v in runs if cc == c]
        iv = [[(b["hands"] - a["hands"]) / (b["cpu_s"] - a["cpu_s"]) for a, b in zip([{"hands": 0, "cpu_s": 0}] + v["log"], v["log"])] for v in rs]
        m = lambda xs: statistics.mean(xs)
        js[c] = dict(parameters=rs[0]["run"]["parameters"], hands=rs[0]["run"]["hands"],
                     hands_per_cpu_s=m([v["run"]["hands"] / v["run"]["cpu_s"] for v in rs]),
                     hands_per_wall_s=m([v["run"]["hands"] / v["run"]["wall_s"] for v in rs]),
                     wall_over_cpu=m([v["run"]["wall_s"] / v["run"]["cpu_s"] for v in rs]),
                     first_interval_hands_per_cpu_s=m([i[0] for i in iv]), last_interval_hands_per_cpu_s=m([i[-1] for i in iv]),
                     mean_cpu_s=m([v["run"]["cpu_s"] for v in rs]), mean_wall_s=m([v["run"]["wall_s"] for v in rs]))
        j = js[c]
        rows.append([c, f"{j['parameters']:,}", f"{j['hands']:,}", f"{j['mean_cpu_s']:.0f}", f"{j['mean_wall_s']:.0f}", f"{j['hands_per_cpu_s']:.1f}",
                     f"{j['hands_per_wall_s']:.1f}", f"{j['wall_over_cpu']:.1f}", f"{j['first_interval_hands_per_cpu_s']:.0f}",
                     f"{j['last_interval_hands_per_cpu_s']:.0f}"])
    return table(["candidate", "parameters", "hands", "CPU s", "wall s", "hands/CPU-s", "hands/wall-s", "wall/CPU",
                  "hands/CPU-s, first checkpoint interval", "last interval"], rows), js


def run_matched(scratch):
    sys.path.insert(0, str(ROOT / "bench"))
    import train_demo
    waves = []
    for wave in WAVES:
        jobs = [(f"{c} seed {s}", [sys.executable, str(ROOT / "bench" / "train_demo.py"), "--candidate", c, "--seed", str(s),
                                   "--hands", str(MATCHED_HANDS), "--checkpoints", str(MATCHED_CHECKPOINTS), "--threads", "1",
                                   "--out", str(scratch / f"{c}-seed{s}")]) for c, s in wave]
        before = os.getloadavg()
        failed = train_demo.run_jobs(jobs, 2, poll=2)
        waves.append({"runs": [f"{c}-seed{s}" for c, s in wave], "loadavg_before": before, "loadavg_after": os.getloadavg()})
        if failed:
            sys.exit(f"matched runs failed: {failed}")
    (scratch / "loadavg.json").write_text(json.dumps(waves, indent=1) + "\n")


def matched(scratch):
    runs = load(scratch)
    load_of = {r: w for w in json.loads((scratch / "loadavg.json").read_text()) for r in w["runs"]} if (scratch / "loadavg.json").exists() else {}
    rows, js = [], {}
    for (c, s), v in runs:
        r, name = v["run"], f"{c}-seed{s}"
        w = load_of.get(name)
        js[name] = dict(parameters=r["parameters"], wall_s=r["wall_s"], cpu_s=r["cpu_s"], hands_per_cpu_s=r["hands"] / r["cpu_s"],
                        hands_per_wall_s=r["hands"] / r["wall_s"], val_scores=[x["val_score"] for x in v["log"]],
                        epsilon_at_end=v["log"][-1]["epsilon"], loadavg_before=w and w["loadavg_before"], loadavg_after=w and w["loadavg_after"])
        rows.append([name, f"{r['wall_s']:.0f}", f"{r['cpu_s']:.0f}", f"{js[name]['hands_per_cpu_s']:.0f}", f"{js[name]['hands_per_wall_s']:.0f}"]
                    + [f"{x['val_score']:.0f}" for x in v["log"]]
                    + ([f"{w['loadavg_before'][0]:.1f} -> {w['loadavg_after'][0]:.1f}"] if w else ["?"]))
    nck = len(runs[0][1]["log"])
    md = table(["run", "wall s", "CPU s", "hands/CPU-s", "hands/wall-s"] + [f"val @{x['hands'] // 1000}k" for x in runs[0][1]["log"]]
               + ["load avg (1 min) before -> after"], rows)
    mean = [[c, f"{statistics.mean(js[f'{c}-seed{s}']['cpu_s'] for s in MATCHED_SEEDS):.0f}",
             f"{statistics.mean(js[f'{c}-seed{s}']['val_scores'][-1] for s in MATCHED_SEEDS):.0f}"] for c in ORDER]
    return md + "\n\n" + table(["candidate", f"mean CPU s for {MATCHED_HANDS:,} hands", f"mean val_score at {MATCHED_HANDS:,}"], mean), js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS), help="directory of committed runs")
    ap.add_argument("--matched", help="scratch directory of the matched-hands runs")
    ap.add_argument("--run", action="store_true", help="train the matched-hands runs into --matched first")
    a = ap.parse_args()
    runs = load(a.runs)
    sample_md, sample_js = budget_table(runs, "hands", HAND_BUDGETS, "val_score at hands")
    cpu_md, cpu_js = budget_table(runs, "cpu_s", CPU_BUDGETS, "val_score by CPU s")
    thr_md, thr_js = throughput(runs)
    result = {"sample_efficiency": sample_js, "compute_efficiency": cpu_js, "throughput": thr_js}
    md = [f"### Sample efficiency (equal hands)\n\n{sample_md}", f"### Compute efficiency (equal CPU seconds)\n\n{cpu_md}",
          f"### Throughput and size\n\n{thr_md}"]
    if a.matched:
        scratch = Path(a.matched)
        if a.run:
            run_matched(scratch)
        m_md, result["matched"] = matched(scratch)
        md.append(f"### Matched hands ({MATCHED_HANDS:,} hands, {MATCHED_CHECKPOINTS} checkpoints)\n\n{m_md}")
        dst = OUT / "budget_fairness" / "matched"
        for d in scratch.iterdir():
            if d.is_dir():
                (dst / d.name).mkdir(parents=True, exist_ok=True)
                for f in ("run.json", "log.jsonl"):
                    shutil.copy(d / f, dst / d.name / f)
        shutil.copy(scratch / "loadavg.json", dst / "loadavg.json")
    (OUT / "budget_fairness").mkdir(parents=True, exist_ok=True)
    (OUT / "budget_fairness.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT / "budget_fairness" / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
