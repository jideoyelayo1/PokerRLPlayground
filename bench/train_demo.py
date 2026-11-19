"""Train the demo agents (bench/SELECTION.md fixes the candidates, seeds and budget).

Format: 4 seats, blinds 1/2, stacks 200 (100 big blinds) reset before every hand. One network
plays all four seats by self-play (train_poker_dqn with a single model). Every hands/CHECKPOINTS
hands the network plays greedily against the three baselines on the validation deals (VAL_SEED;
none is a training or evaluation deal) and the checkpoint with the best validation score is kept.
Score = the worst, over the baselines, of the lower end of the 95% interval on bb/100.

Output per run, in weights/runs/<candidate>-seed<seed>/: model.pt (the kept checkpoint's
state_dict), log.jsonl (one line per checkpoint), run.json (settings, timings, kept checkpoint;
written last, so a run with run.json is complete and is skipped).

    python bench/train_demo.py --candidate mlp-s --seed 1    # one run
    python bench/train_demo.py --all --jobs 5                # every candidate x seed not yet done

--resume: at every checkpoint but the last the run also writes resume.pt: the whole training state
(networks, optimizer, replay, counters, every RNG, this script's bookkeeping and the time spent so far),
written to a temporary file and moved into place, so a kill leaves the previous one. With --resume, a
run that has resume.pt and no run.json continues from it. model.pt and the log.jsonl rows then come out
exactly as the uninterrupted run's (bitwise, on the same machine, with the same code, library versions
and torch threads), except wall_s and cpu_s, which add up the time of all the segments. resume.pt holds
the replay (about 100 MB at the full budget), so it is deleted once run.json is written.

--keep-checkpoints also saves ckpt-<hands>.pt (the state_dict) at every checkpoint; model.pt is
still the best-validation one. --all exits non-zero if any run failed (--fail-fast stops at the first).

--league trains against an opponent pool instead of the latest copy of itself (#14; league_config
below, and the run's "league" in run.json); --league-p-bot 0 leaves the fixed bots out.
"""
import argparse
import copy
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("TQDM_DISABLE", "1")  # progress goes to log.jsonl instead
import numpy as np
import torch

import ReinforcementLearningEnvironment.helper as helper
from Models.GRUModel import GRUModel
from Models.MLPModel import MLPModel
from ReinforcementLearningEnvironment.Poker import train_poker_dqn
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import N_ACTIONS
from winrate import BOTS, evaluate, greedy_policy

SEATS, STACK, BB, SB = 4, 200, 2, 1
STATE_DIM = 151  # encode_observation length for 4 seats
# hands: the training budget. A GRU hand costs about 15 times an MLP hand here (pilots: 17 against
# about 250 hands/s), so the GRU gets fewer hands to fit the time budget (bench/SELECTION.md).
CANDIDATES = {
    "mlp-s": dict(cls="MLPModel", kwargs=dict(hidden=(128, 128), dropout=0.0), sequence=False, hands=400_000),
    "mlp-l": dict(cls="MLPModel", kwargs=dict(hidden=(512, 512, 256), dropout=0.0), sequence=False, hands=400_000),
    "gru": dict(cls="GRUModel", kwargs=dict(hidden_dim=128, layer_dim=1, dropout=0.0), sequence=True, hands=150_000),
}
SEEDS = (1, 2, 3)
CHECKPOINTS = 10
# eps_decay_hands is set per run to a fifth of its hands
AGENT = dict(gamma=1.0, lr=1e-4, batch_size=256, memory_size=100_000, eps_start=1.0, eps_end=0.05,
             double=True, target_update_freq=200)
REWARD_SCALE = 0.01  # Q-values in units of 100 big blinds
VAL_SEED, VAL_DEALS = 5000, 1000
RUNS = ROOT / "weights" / "runs"
# --league: one snapshot every tenth of the run (10 in all) and a pool of 5, so the opponents are
# the learner of the later half of training; each seat but one (the learner's, drawn uniformly) is
# a snapshot with 40% and a baseline bot (BOTS, uniform) with 20%, else the learner. With 4 seats the
# learner has 55% of the seats (1/4 + 3/4 x 40%), snapshots 30% and bots 15%.
LEAGUE = dict(snapshots_per_run=10, pool_size=5, p_snapshot=0.4, p_bot=0.2)


def build(name):
    c = CANDIDATES[name]
    cls = {"MLPModel": MLPModel, "GRUModel": GRUModel}[c["cls"]]
    if cls is MLPModel:
        return cls(input_features=STATE_DIM, output_features=N_ACTIONS, **c["kwargs"])
    return cls(input_dim=STATE_DIM, output_dim=N_ACTIONS, **c["kwargs"])


def validate(model, sequence):
    net = copy.deepcopy(model).eval()
    out = {}
    for bname, bot in BOTS.items():
        pol = greedy_policy(net, sequence)
        with torch.no_grad():
            r = evaluate(pol, bot, VAL_DEALS, VAL_SEED, stack=STACK)
        out[bname] = {"bb100": r["bb100"], "lo": r["ci95_lo"], "hi": r["ci95_hi"], "actions": pol.counts}
    return out, min(v["lo"] for v in out.values())


def league_config(hands, p_bot):
    """The league dict for train_poker_dqn, and its record for run.json (bots by name)."""
    cfg = {"snapshot_every": hands // LEAGUE["snapshots_per_run"], "pool_size": LEAGUE["pool_size"],
           "p_snapshot": LEAGUE["p_snapshot"], "p_bot": p_bot}
    return {**cfg, "bots": list(BOTS.values())}, {**cfg, "bots": list(BOTS)}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def diagnostics(stats):
    """Mean of each QTrainer.last_stats entry over the train steps since the last checkpoint;
    nonfinite is the total count."""
    if not stats:
        return {}
    return {k: float(sum(s[k] for s in stats) if k == "nonfinite" else np.mean([s[k] for s in stats])) for k in stats[0]}


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def train(name, seed, hands, out, checkpoints, threads, keep=False, resume=False, league_p_bot=None):
    """league_p_bot: None = latest-only self-play, else the league of LEAGUE with this chance of a bot seat."""
    out.mkdir(parents=True, exist_ok=True)
    # the code the run uses, read before it starts (other commits may land while it trains)
    commit = git("rev-parse", "HEAD")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)weights"))
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build(name)
    seq = CANDIDATES[name]["sequence"]
    agent_kw = dict(AGENT, eps_decay_hands=hands // 5, sequence=seq)
    league, league_record = league_config(hands, league_p_bot) if league_p_bot is not None else (None, None)
    every = hands // checkpoints
    # wall_s and cpu_s count from the start of the first segment: t0 and c0 are shifted back by the time already spent
    state = {"best": None, "score": -float("inf"), "loss": [], "stats": [], "transitions": 0}
    snap, spent = None, {"wall_s": 0.0, "cpu_s": 0.0}
    rpath = out / "resume.pt"
    if resume and rpath.exists():
        # weights_only=False: the file holds numpy arrays and RNG states, and is our own
        saved = torch.load(rpath, map_location="cpu", weights_only=False)
        if saved["version"] != 1 or saved["run"] != [name, seed, hands, checkpoints, league_p_bot]:
            sys.exit(f"{rpath} is not from this run (version {saved['version']}, run {saved['run']})")
        snap, spent = saved["snapshot"], saved["spent"]
        state.update(saved["state"])
    t0, c0 = time.time() - spent["wall_s"], cpu_s() - spent["cpu_s"]
    rows = []
    if snap:
        rows = (out / "log.jsonl").read_text().splitlines()[:snap["hands_played"] // every]  # drop a row written after resume.pt
    log = (out / "log.jsonl").open("w")
    for row in rows:
        log.write(row + "\n")

    def on_hand_end(n, agents, snapshot):
        ag = agents[0]
        if ag.last_loss is not None:
            state["loss"].append(ag.last_loss)
            state["stats"].append(ag.last_stats)
        if n % every:
            return
        val, score = validate(ag.model, seq)
        row = {"hands": n, "wall_s": round(time.time() - t0, 1), "cpu_s": round(cpu_s() - c0, 1),
               "epsilon": round(ag.epsilon, 4), "mean_loss": float(np.mean(state["loss"])) if state["loss"] else None,
               **diagnostics(state["stats"]),
               "decisions_per_hand": round((ag.transitions - state["transitions"]) / every, 2),
               "val": val, "val_score": score}
        state["loss"], state["stats"], state["transitions"] = [], [], ag.transitions
        if score > state["score"]:
            state["score"], state["best"] = score, n
            torch.save(ag.model.state_dict(), out / "model.pt")
        if keep:
            torch.save(ag.model.state_dict(), out / f"ckpt-{n}.pt")
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"{name} seed {seed}: {n} hands, {row['wall_s']} s, val score {score:.0f} (best {state['score']:.0f} at {state['best']})", flush=True)
        if n < hands:
            torch.save({"version": 1, "run": [name, seed, hands, checkpoints, league_p_bot], "snapshot": snapshot(), "state": state,
                        "spent": {"wall_s": row["wall_s"], "cpu_s": row["cpu_s"]}}, f"{rpath}.tmp")
            os.replace(f"{rpath}.tmp", rpath)

    agents, history = train_poker_dqn([model], SEATS, STACK, BB, SB, max_hands=hands, device="cpu", seed=seed,
                                      reset_stacks=True, agent_kwargs=agent_kw, reward_scale=REWARD_SCALE,
                                      log_csv=False, on_hand_end=on_hand_end, ma_window=1, resume=snap, league=league)
    run = {"candidate": name, "architecture": CANDIDATES[name], "seed": seed, "hands": hands,
           "agent": agent_kw, "reward_scale": REWARD_SCALE, "format": {"seats": SEATS, "stack": STACK, "blinds": [SB, BB],
                                                                       "reset_stacks": True},
           "validation": {"seed": VAL_SEED, "deals": VAL_DEALS, "every_hands": every},
           "kept_checkpoint_hands": state["best"], "kept_val_score": state["score"],
           "wall_s": round(time.time() - t0, 1), "cpu_s": round(cpu_s() - c0, 1),
           "torch_threads": threads, "keep_checkpoints": keep, "league": league_record, "aborted_hands": len(history["aborted"]),
           "commit": commit, "dirty": dirty, "started_utc": started_utc,
           "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "parameters": sum(p.numel() for p in model.parameters())}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    rpath.unlink(missing_ok=True)


def run_jobs(jobs, n_parallel, fail_fast=False, poll=5):
    """Run jobs = [(label, cmd)], n_parallel at a time. Returns the failures [(label, returncode)];
    with fail_fast, the first failure terminates the running jobs and no more are started."""
    todo, running, failures = list(jobs), [], []
    while todo or running:
        while todo and len(running) < n_parallel:
            label, cmd = todo.pop(0)
            running.append((label, subprocess.Popen(cmd)))
        time.sleep(poll)
        for label, p in list(running):
            if p.poll() is None:
                continue
            running.remove((label, p))
            if p.returncode:
                print(f"FAILED: {label} exited with code {p.returncode}", flush=True)
                failures.append((label, p.returncode))
        if failures and fail_fast:
            for label, p in running:
                p.terminate(); p.wait(); print(f"stopped: {label}", flush=True)
            running, todo = [], []
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true", help="every candidate and seed in SELECTION.md")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel with --all")
    ap.add_argument("--hands", type=int, help="training hands (default: the candidate's budget)")
    ap.add_argument("--checkpoints", type=int, default=CHECKPOINTS)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per run")
    ap.add_argument("--keep-checkpoints", action="store_true", help="also save ckpt-<hands>.pt at every checkpoint")
    ap.add_argument("--resume", action="store_true", help="continue a run that has resume.pt and no run.json")
    ap.add_argument("--league", action="store_true", help="opponent pool of snapshots and bots instead of latest-only self-play")
    ap.add_argument("--league-p-bot", type=float, default=LEAGUE["p_bot"], help="with --league: chance a seat is a fixed bot (0: snapshots only)")
    ap.add_argument("--fail-fast", action="store_true", help="with --all, stop at the first failed run")
    ap.add_argument("--out", help="run directory (default weights/runs/<candidate>-seed<seed>)")
    a = ap.parse_args()
    if not a.all:
        out = Path(a.out) if a.out else RUNS / f"{a.candidate}-seed{a.seed}"
        if (out / "run.json").exists():
            print(f"{out} is complete, skipped"); return
        return train(a.candidate, a.seed, a.hands or CANDIDATES[a.candidate]["hands"], out, a.checkpoints, a.threads, a.keep_checkpoints, a.resume,
                     a.league_p_bot if a.league else None)
    todo = [(c, s) for c in reversed(CANDIDATES) for s in SEEDS  # slowest candidates first
            if not (RUNS / f"{c}-seed{s}" / "run.json").exists()]
    print(f"{len(todo)} runs to do: {todo}", flush=True)
    jobs = []
    for c, s_ in todo:
        cmd = [sys.executable, str(Path(__file__)), "--candidate", c, "--seed", str(s_),
               "--hands", str(a.hands or CANDIDATES[c]["hands"]), "--checkpoints", str(a.checkpoints),
               "--threads", str(a.threads)] + (["--keep-checkpoints"] if a.keep_checkpoints else []) + (["--resume"] if a.resume else []) \
            + (["--league", "--league-p-bot", str(a.league_p_bot)] if a.league else [])
        jobs.append((f"{c} seed {s_}", cmd))
    failed = run_jobs(jobs, a.jobs, a.fail_fast)
    if failed:
        sys.exit(f"{len(failed)} of {len(jobs)} runs failed: " + ", ".join(f"{l} (code {r})" for l, r in failed))


if __name__ == "__main__":
    main()
