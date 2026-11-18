"""Opponent suite, play diagnostics and game-format sensitivity for the kept demo agents (#13, #20, #23).

Three parts, each a set of cells (one agent against one opponent), written one file per cell to
bench/raw/eval_suite/ and reused while the model.pt, deals, seed and format match:

  suite    every kept run (and the random bot as control) against the seven bots of winrate.SUITE,
           greedy, 4 seats, stacks 200 (100 bb), blinds 1/2 reset every hand, duplicate deals from
           seed 31000 (winrate.evaluate: each deal replayed with the agent in each of the 4 seats).
  cross    head-to-head: a kept run in one seat against three copies of mlp-s-seed3's greedy policy
           (the shipped agent), same format and duplicate deals, seed 33000.
  formats  the shipped agent and two other kept runs against winrate.BOTS at stacks of 50, 100, 200
           and 400 bb (duplicate deals, seed 32000), then in a tournament carry-over format
           (`tournament`: stacks carry across hands and the button moves; sessions of 1 agent and
           3 copies of a bot, until one player is left, the agent busts or MAX_HANDS hands, the
           agent rotating through the seats; deck seeds from 32000).

Every evaluate() cell also fills a Stats (opt-in diagnostics, #20): VPIP, PFR, 3-bet, fold-to-bet,
showdown rate, aggression factor, final pot sizes, result by position and by starting hand,
legality, free folds and all-ins, for the agent's seat. Flags (FLAGS below) mark degenerate play.
`report` writes bench/raw/diagnostics.json and diagnostics.md from all cells, and the tables
used in bench/RESULTS.md section 12 to bench/raw/eval_suite/tables.md.

    python bench/eval_suite.py suite|cross|formats|report [--jobs 2] [--deals N] [--sessions N]
"""
import argparse
import json
import os
import random
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

import PokerEngine.poker_engine as pe
import ReinforcementLearningEnvironment.helper as helper
from ReinforcementLearningEnvironment.helper import action_index_to_engine_action, legal_mask_from_obs
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    A_ALLIN, A_CALL, A_CHECK, A_FOLD, A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT)
from winrate import BB, BOTS, SB, SUITE, bot_random, evaluate, greedy_policy

SEATS = 4
RAISES = (A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT, A_ALLIN)  # all-in counts as a bet or raise
POSITIONS = ("BTN", "SB", "BB", "UTG")  # seat minus button, 4 seats
POT_EDGES = (3, 6, 12, 25, 50, 100, 200)  # final pot in bb: bins <=3, <=6, ... <=200, >200
SUITE_SEED, FORMAT_SEED, CROSS_SEED = 31000, 32000, 33000
STACK = 200  # the demo's: 100 bb at blinds 1/2
MAX_HANDS = 200  # tournament session cap
SHIPPED = "mlp-s-seed3"
FORMAT_AGENTS = [SHIPPED, "gru-seed3", "mlp-s-seed2", "random-bot"]
FORMAT_STACKS_BB = (50, 100, 200, 400)
# thresholds for the degenerate-play flags: name -> (test on the summary, description)
FLAGS = {
    "illegal": (lambda s: s["illegal"] > 0, "an action with mask != 0"),
    "free_folds": (lambda s: s["free_folds"] > 0, "folds when check was legal"),
    "all_in_rate": (lambda s: s["all_in_rate"] > 0.5, "all-in is more than 50% of decisions"),
    "repeated_all_in": (lambda s: s["hands_2plus_all_in"] > 0, "a hand with 2+ all-in actions"),
    "vpip_low": (lambda s: s["vpip"] < 0.05, "VPIP < 5%"),
    "vpip_high": (lambda s: s["vpip"] > 0.90, "VPIP > 90%"),
    "fold_to_bet": (lambda s: s["fold_to_bet"] > 0.90, "folds more than 90% of the time facing a bet"),
}
RAW = ROOT / "bench" / "raw"
CELLS = RAW / "eval_suite"


class Stats:
    """Play diagnostics for one seat (self.agent, set by evaluate() before each hand), collected
    through winrate.play_hand's hooks. decision(seat, obs, mask, action) per decision of any seat,
    hand(results, holes, final_obs) per hand. Definitions, for the agent's seat:
      VPIP  hands with a preflop call or raise (a big blind's free check is not voluntary)
      PFR   hands with a preflop raise or all-in;  3-bet: share of its preflop decisions that
            faced exactly one earlier raise (the blinds are the bet before it) and re-raised
      fold_to_bet  folds / decisions with viewer_to_call > 0 (any street)
      showdown     hands that reached the river (5 board cards) with the agent not having folded
      aggression   (raises + all-ins) / calls
      free_folds   folds when check was legal;  illegal: actions with mask[a] != 0
      all_in_rate  all-in actions / decisions; hands_2plus_all_in: hands with 2 or more
                   (impossible in this engine: an all-in player cannot act again; kept as a check)
      by_position  result in bb per hand by seat relative to the button (BTN, SB, BB, UTG)
      by_hand      result in bb per hand by 169-class starting hand
      pot          final pot in bb: count per bin of POT_EDGES, mean, median, 90th percentile"""

    def __init__(self, bb=BB):
        self.bb, self.agent, self.cur = bb, 0, []
        self.n = dict.fromkeys(("hands", "decisions", "facing", "folds_facing", "free_folds", "illegal", "calls", "raises",
                                "all_ins", "hands_all_in", "hands_2plus_all_in", "vpip", "pfr", "opp3", "bet3", "showdown"), 0)
        self.pos = {p: [0, 0.0] for p in POSITIONS}
        self.buckets = {}
        self.pots = []
        self.result_sum = 0.0

    def decision(self, seat, obs, mask, a):
        self.cur.append((seat, int(obs.street), int(a), obs.viewer_to_call > 0, bool(mask[A_CHECK] == 0.0)))
        if seat == self.agent:
            self.n["decisions"] += 1
            self.n["illegal"] += int(mask[a] != 0.0)

    def hand(self, results, holes, end):
        n, raises, mine = self.n, 0, []
        for seat, street, a, facing, can_check in self.cur:
            if seat == self.agent:
                n["facing"] += facing
                n["folds_facing"] += facing and a == A_FOLD
                n["free_folds"] += a == A_FOLD and can_check
                n["calls"] += a == A_CALL
                n["raises"] += a in RAISES
                if street == 0:
                    n["opp3"] += raises == 1
                    n["bet3"] += raises == 1 and a in RAISES
                mine.append((street, a))
            if street == 0 and a in RAISES:
                raises += 1
        pre = [a for street, a in mine if street == 0]
        shoves = sum(a == A_ALLIN for _, a in mine)
        board = sum(int(c.rank.value) != 0 for c in end.board)
        n["hands"] += 1
        n["all_ins"] += shoves
        n["hands_all_in"] += shoves >= 1
        n["hands_2plus_all_in"] += shoves >= 2
        n["vpip"] += any(a == A_CALL or a in RAISES for a in pre)
        n["pfr"] += any(a in RAISES for a in pre)
        n["showdown"] += board == 5 and all(a != A_FOLD for _, a in mine)
        r = results[self.agent]
        self.result_sum += r
        p = self.pos[POSITIONS[(self.agent - end.button) % SEATS]]
        p[0] += 1; p[1] += r
        key = helper.pocket_169_key(helper.pocket_key(*holes[self.agent]))
        h = self.buckets.setdefault(key, [0, 0.0])
        h[0] += 1; h[1] += r
        self.pots.append(end.pot_total / self.bb)
        self.cur = []

    def summary(self):
        n, d = self.n, lambda a, b: a / b if b else float("nan")
        pots = np.array(self.pots)
        return {
            "hands": n["hands"], "decisions": n["decisions"], "bb100": 100 * d(self.result_sum, n["hands"]),
            "vpip": d(n["vpip"], n["hands"]), "pfr": d(n["pfr"], n["hands"]), "three_bet": d(n["bet3"], n["opp3"]),
            "three_bet_chances": n["opp3"], "fold_to_bet": d(n["folds_facing"], n["facing"]), "facing_bet": n["facing"],
            "showdown": d(n["showdown"], n["hands"]), "aggression": d(n["raises"], n["calls"]),
            "illegal": n["illegal"], "free_folds": n["free_folds"], "free_fold_share": d(n["free_folds"], n["decisions"]),
            "all_in_rate": d(n["all_ins"], n["decisions"]), "hands_with_all_in": d(n["hands_all_in"], n["hands"]),
            "hands_2plus_all_in": n["hands_2plus_all_in"],
            "pot_bb": {"mean": float(pots.mean()), "median": float(np.median(pots)), "p90": float(np.percentile(pots, 90)),
                       "bins": np.bincount(np.searchsorted(POT_EDGES, pots), minlength=len(POT_EDGES) + 1).tolist()},
            "by_position": {p: {"hands": c, "bb100": 100 * d(s, c)} for p, (c, s) in self.pos.items()},
            "by_hand": {k: {"hands": c, "bb100": 100 * s / c} for k, (c, s) in sorted(self.buckets.items())},
        }


def flags(s):
    return [k for k, (test, _) in FLAGS.items() if test(s)]


def tournament(policy, bot, sessions, seed, stack=STACK, max_hands=MAX_HANDS):
    """Carry-over format: stacks carry across hands and the button moves (the engine's, after the
    standard order). A session is 1 agent (seat s mod 4 in session s) and 3 copies of bot, from a
    fresh tournament, until one player is left, the agent has no chips or max_hands hands have
    been played. A session's value is the agent's chip result in bb divided by its hands; the
    deck of session s comes from seed * 1_000_003 + s. Returns the mean of the values with a 95%
    interval over sessions, the same result pooled over hands, and how sessions ended."""
    g = pe.GameEngine(SEATS, stack, BB, SB)
    rng = random.Random(seed)
    vals, chips, hands, ended = [], 0.0, 0, {"agent_busted": 0, "agent_won_all": 0, "cap": 0}
    for s in range(sessions):
        seat = s % SEATS
        pols = [policy if i == seat else bot for i in range(SEATS)]
        g.seed(seed * 1_000_003 + s)
        g.reset_tournament(stack)
        n = 0
        alive = lambda: sum(x > 0 for x in g.getObservation(0).stack)
        while n < max_hands and alive() >= 2 and g.getObservation(0).stack[seat] > 0:
            g.start_hand()
            for pol in pols:
                if hasattr(pol, "new_hand"):
                    pol.new_hand()
            n += 1
            steps = 0
            while not g.hand_terminal():
                steps += 1
                if steps > 2000:
                    raise RuntimeError("hand did not finish")
                p = g.currentPlayerSafe()
                obs = g.getObservation(p)
                g.step(action_index_to_engine_action(pols[p](obs, legal_mask_from_obs(obs), rng)), p)
        res = (g.getObservation(0).stack[seat] - stack) / BB
        vals.append(res / n); chips += res; hands += n
        ended["agent_busted" if g.getObservation(0).stack[seat] <= 0 else "agent_won_all" if alive() < 2 else "cap"] += 1
    v = np.array(vals)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean_bb_per_hand": float(v.mean()), "ci95_lo": float(v.mean() - half), "ci95_hi": float(v.mean() + half),
            "pooled_bb_per_hand": chips / hands, "sessions": sessions, "hands": hands, "mean_session_hands": hands / sessions,
            "ended": ended, "max_hands": max_hands, "stack": stack}


def load_policy(run):
    """(policy, sha256 of its weights or None) for a kept run or "random-bot"."""
    if run == "random-bot":
        return bot_random, None
    import eval_demo
    import hashlib
    model, seq, _ = eval_demo.load(run)
    sha = hashlib.sha256((eval_demo.RUNS / run / "model.pt").read_bytes()).hexdigest()
    return greedy_policy(model, seq), sha


def opponent(name):
    if name in SUITE:
        return SUITE[name]
    return load_policy(name)[0]  # a kept run, played greedily in all three seats


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x


def cell(job):
    part, agent, opp, deals, seed, stack = job  # stack in chips; "tournament" cells: deals = sessions
    kind = "tournament" if part == "tournament" else "duplicate"
    path = CELLS / f"{part}_{agent}_vs_{opp}_{stack}.json"
    pol, sha = load_policy(agent)
    if path.exists():
        c = json.loads(path.read_text())
        if (c["model_sha256"], c["deals"], c["seed"]) == (sha, deals, seed):
            return c
    torch.set_num_threads(1)
    t0, c0 = time.time(), time.process_time()
    bot = opponent(opp)
    with torch.no_grad():
        if kind == "tournament":
            r = tournament(pol, bot, deals, seed, stack)
        else:
            st = Stats()
            r = evaluate(pol, bot, deals, seed, stack=stack, stats=st)
            r["stats"] = st.summary()
            r["flags"] = flags(r["stats"])
    r.update({"part": part, "agent": agent, "opponent": opp, "seed": seed, "deals": deals, "stack": stack, "format": kind,
              "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
              "load_average": os.getloadavg(), "model_sha256": sha,
              "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    r = plain(r)
    CELLS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1) + "\n")
    return r


def kept_runs():
    import eval_demo
    return sorted(p.parent.name for p in eval_demo.RUNS.glob("*/run.json"))


def jobs_for(part, deals, sessions):
    runs = kept_runs()
    if part == "suite":
        return [("suite", a, o, deals, SUITE_SEED, STACK) for a in runs + ["random-bot"] for o in SUITE]
    if part == "cross":
        return [("cross", a, SHIPPED, deals, CROSS_SEED, STACK) for a in runs if a != SHIPPED]
    jobs = [("formats", a, o, deals, FORMAT_SEED, bb * BB) for a in FORMAT_AGENTS for o in BOTS for bb in FORMAT_STACKS_BB]
    return jobs + [("tournament", a, o, sessions, FORMAT_SEED, STACK) for a in FORMAT_AGENTS for o in BOTS]


def ci(c, key="bb100"):
    return f"{c[key]:,.0f} [{c['ci95_lo']:,.0f}, {c['ci95_hi']:,.0f}]"


def report(cells):
    """diagnostics.json / diagnostics.md from every duplicate-deal cell, and eval_suite/tables.md."""
    dup = [c for c in cells if c["format"] == "duplicate"]
    (RAW / "diagnostics.json").write_text(json.dumps([{k: c[k] for k in ("part", "agent", "opponent", "seed", "deals", "stack", "bb100",
                                                       "ci95_lo", "ci95_hi", "flags", "stats", "commit")} for c in dup], indent=1) + "\n")
    pc = lambda x: "-" if x != x else f"{100 * x:.0f}%"
    md = ["Diagnostics for the agent's seat in every duplicate-deal cell of bench/eval_suite.py (definitions: `Stats` there; "
          f"thresholds: `FLAGS`). Hands = 4 x deals. 3-bet: share of its preflop decisions facing one raise; AF = (raises + all-ins) / calls; "
          "free folds = folds with check legal (count, share of decisions); all-in = share of decisions.", "",
          "| part | agent | opponent | stack bb | hands | bb/100 | VPIP | PFR | 3-bet | fold to bet | showdown | AF | free folds | all-in | illegal | flags |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in dup:
        s = c["stats"]
        md.append(f"| {c['part']} | {c['agent']} | {c['opponent']} | {c['stack'] // BB} | {s['hands']:,} | {s['bb100']:,.0f} | {pc(s['vpip'])} | {pc(s['pfr'])} | "
                  f"{pc(s['three_bet'])} | {pc(s['fold_to_bet'])} | {pc(s['showdown'])} | {s['aggression']:.2f} | {s['free_folds']:,} ({100 * s['free_fold_share']:.1f}%) | "
                  f"{pc(s['all_in_rate'])} | {s['illegal']} | {', '.join(c['flags']) or '-'} |")
    md += ["", "Result by position (bb/100; seat relative to the button, 4 seats) and final pot (bb: mean / median / 90th percentile; share of hands "
           "in bins <=3, <=6, <=12, <=25, <=50, <=100, <=200, >200).", "",
           "| part | agent | opponent | stack bb | BTN | SB | BB | UTG | pot mean / median / p90 | pot bins % |", "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for c in dup:
        s, p = c["stats"], c["stats"]["pot_bb"]
        md.append(f"| {c['part']} | {c['agent']} | {c['opponent']} | {c['stack'] // BB} | " + " | ".join(f"{s['by_position'][q]['bb100']:,.0f}" for q in POSITIONS)
                  + f" | {p['mean']:.1f} / {p['median']:.1f} / {p['p90']:.1f} | " + " ".join(f"{100 * b / s['hands']:.0f}" for b in p["bins"]) + " |")
    md += ["", "Result by starting hand (bb/100): pairs, suited and offsuit classes pooled, and the best and worst class with at least 30 hands "
           "(each class has few hands, so single classes are noisy: see `by_hand` in diagnostics.json).", "",
           "| part | agent | opponent | stack bb | pairs | suited | offsuit | best class | worst class |", "|---|---|---|---:|---:|---:|---:|---|---|"]
    for c in dup:
        h = c["stats"]["by_hand"]
        grp = lambda f: (lambda v: sum(x["bb100"] * x["hands"] for x in v) / sum(x["hands"] for x in v))([x for k, x in h.items() if f(k)])
        big = {k: v for k, v in h.items() if v["hands"] >= 30}
        hi, lo = max(big, key=lambda k: big[k]["bb100"]), min(big, key=lambda k: big[k]["bb100"])
        md.append(f"| {c['part']} | {c['agent']} | {c['opponent']} | {c['stack'] // BB} | {grp(lambda k: len(k) == 2):,.0f} | {grp(lambda k: k.endswith('s')):,.0f} | "
                  f"{grp(lambda k: k.endswith('o')):,.0f} | {hi} {big[hi]['bb100']:,.0f} | {lo} {big[lo]['bb100']:,.0f} |")
    (RAW / "diagnostics.md").write_text("\n".join(md) + "\n")

    by = {(c["part"], c["agent"], c["opponent"], c["stack"]): c for c in cells}
    runs = sorted({c["agent"] for c in cells if c["part"] == "suite"} - {"random-bot"})
    out = []
    if runs:
        opps = list(SUITE)
        out += [f"Suite: bb/100 [95% CI over deals], {by[('suite', runs[0], opps[0], STACK)]['deals']:,} duplicate deals per cell, seed {SUITE_SEED}.", "",
                "| run | " + " | ".join(opps) + " |", "|---|" + "---:|" * len(opps)]
        for a in runs + ["random-bot"]:
            out.append(f"| {a} | " + " | ".join(ci(by[("suite", a, o, STACK)]) for o in opps) + " |")
        out += ["", "Per family, mean of the per-seed bb/100 [95% t-interval over seeds].", "", "| family | " + " | ".join(opps) + " |", "|---|" + "---:|" * len(opps)]
        t95 = {2: 4.303, 3: 3.182}
        for fam in dict.fromkeys(a.rsplit("-seed", 1)[0] for a in runs):
            rs = [a for a in runs if a.rsplit("-seed", 1)[0] == fam]
            cols = []
            for o in opps:
                v = [by[("suite", a, o, STACK)]["bb100"] for a in rs]
                m, h = statistics.mean(v), t95[len(v)] * statistics.stdev(v) / len(v) ** 0.5
                cols.append(f"{m:,.0f} [{m - h:,.0f}, {m + h:,.0f}]")
            out.append(f"| {fam} ({len(rs)} seeds) | " + " | ".join(cols) + " |")
    cr = [c for c in cells if c["part"] == "cross"]
    if cr:
        out += ["", f"Cross-play: bb/100 of the run in the left column (one seat) against three copies of {SHIPPED}, {cr[0]['deals']:,} duplicate deals, seed {CROSS_SEED}.", "",
                "| run | bb/100 [95% CI] |", "|---|---:|"] + [f"| {c['agent']} | {ci(c)} |" for c in sorted(cr, key=lambda c: c["agent"])]
    fm = [c for c in cells if c["part"] == "formats"]
    if fm:
        out += ["", f"Stack depth: bb/100 [95% CI], {fm[0]['deals']:,} duplicate deals per cell, seed {FORMAT_SEED}, blinds 1/2, stacks reset every hand.", "",
                "| agent | opponent | " + " | ".join(f"{b} bb" for b in FORMAT_STACKS_BB) + " |", "|---|---|" + "---:|" * len(FORMAT_STACKS_BB)]
        for a in FORMAT_AGENTS:
            for o in BOTS:
                out.append(f"| {a} | {o} | " + " | ".join(ci(by[("formats", a, o, bb * BB)]) for bb in FORMAT_STACKS_BB) + " |")
    tn = [c for c in cells if c["part"] == "tournament"]
    if tn:
        out += ["", f"Tournament carry-over (stack {STACK // BB} bb, blinds 1/2, the button moves; session = 1 agent and 3 bots until one player is left, the agent busts or "
                f"{MAX_HANDS} hands; {tn[0]['deals']:,} sessions per cell, seed {FORMAT_SEED}): mean of the sessions' agent result in bb per hand [95% CI over sessions], "
                "the same pooled over hands, mean hands per session, how sessions ended (agent busted / agent won everything / hand cap).", "",
                "| agent | opponent | bb per hand | pooled | hands per session | ended |", "|---|---|---:|---:|---:|---|"]
        for a in FORMAT_AGENTS:
            for o in BOTS:
                c = by[("tournament", a, o, STACK)]
                e = c["ended"]
                out.append(f"| {a} | {o} | {c['mean_bb_per_hand']:.3f} [{c['ci95_lo']:.3f}, {c['ci95_hi']:.3f}] | {c['pooled_bb_per_hand']:.3f} | "
                           f"{c['mean_session_hands']:.0f} | {e['agent_busted']} / {e['agent_won_all']} / {e['cap']} |")
    (CELLS / "tables.md").write_text("\n".join(out) + "\n")
    return out, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["suite", "cross", "formats", "report"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--deals", type=int, default=4000)
    ap.add_argument("--sessions", type=int, default=300)
    a = ap.parse_args()
    if a.part != "report":
        t0 = time.time()
        with Pool(a.jobs) as pool:
            for c in pool.imap_unordered(cell, jobs_for(a.part, a.deals, a.sessions), chunksize=1):
                print(c["part"], c["agent"], c["opponent"], c["stack"], c.get("bb100", c.get("mean_bb_per_hand")), c["wall_s"], flush=True)
        print(f"{a.part}: wall {time.time() - t0:.0f} s, load average {os.getloadavg()}")
    cells = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    out, _ = report(cells)
    print("\n".join(out))


if __name__ == "__main__":
    main()
