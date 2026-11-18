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


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def train(name, seed, hands, out, checkpoints, threads):
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
    log = (out / "log.jsonl").open("w")
    every = hands // checkpoints
    state = {"best": None, "score": -float("inf"), "loss": [], "t0": time.time(),
             "r0": resource.getrusage(resource.RUSAGE_SELF), "transitions": 0}

    def on_hand_end(n, agents):
        ag = agents[0]
        if ag.last_loss is not None:
            state["loss"].append(ag.last_loss)
        if n % every:
            return
        val, score = validate(ag.model, seq)
        r = resource.getrusage(resource.RUSAGE_SELF)
        row = {"hands": n, "wall_s": round(time.time() - state["t0"], 1),
               "cpu_s": round(r.ru_utime + r.ru_stime - state["r0"].ru_utime - state["r0"].ru_stime, 1),
               "epsilon": round(ag.epsilon, 4), "mean_loss": float(np.mean(state["loss"])) if state["loss"] else None,
               "decisions_per_hand": round((ag.transitions - state["transitions"]) / every, 2),
               "val": val, "val_score": score}
        state["loss"], state["transitions"] = [], ag.transitions
        if score > state["score"]:
            state["score"], state["best"] = score, n
            torch.save(ag.model.state_dict(), out / "model.pt")
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"{name} seed {seed}: {n} hands, {row['wall_s']} s, val score {score:.0f} (best {state['score']:.0f} at {state['best']})", flush=True)

    t0 = time.time()
    agents, history = train_poker_dqn([model], SEATS, STACK, BB, SB, max_hands=hands, device="cpu", seed=seed,
                                      reset_stacks=True, agent_kwargs=agent_kw, reward_scale=REWARD_SCALE,
                                      log_csv=False, on_hand_end=on_hand_end, ma_window=1)
    r = resource.getrusage(resource.RUSAGE_SELF)
    run = {"candidate": name, "architecture": CANDIDATES[name], "seed": seed, "hands": hands,
           "agent": agent_kw, "reward_scale": REWARD_SCALE, "format": {"seats": SEATS, "stack": STACK, "blinds": [SB, BB],
                                                                       "reset_stacks": True},
           "validation": {"seed": VAL_SEED, "deals": VAL_DEALS, "every_hands": every},
           "kept_checkpoint_hands": state["best"], "kept_val_score": state["score"],
           "wall_s": round(time.time() - t0, 1), "cpu_s": round(r.ru_utime + r.ru_stime, 1),
           "torch_threads": threads, "aborted_hands": len(history["aborted"]),
           "commit": commit, "dirty": dirty, "started_utc": started_utc,
           "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "parameters": sum(p.numel() for p in model.parameters())}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true", help="every candidate and seed in SELECTION.md")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel with --all")
    ap.add_argument("--hands", type=int, help="training hands (default: the candidate's budget)")
    ap.add_argument("--checkpoints", type=int, default=CHECKPOINTS)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per run")
    ap.add_argument("--out", help="run directory (default weights/runs/<candidate>-seed<seed>)")
    a = ap.parse_args()
    if not a.all:
        out = Path(a.out) if a.out else RUNS / f"{a.candidate}-seed{a.seed}"
        if (out / "run.json").exists():
            print(f"{out} is complete, skipped"); return
        return train(a.candidate, a.seed, a.hands or CANDIDATES[a.candidate]["hands"], out, a.checkpoints, a.threads)
    todo = [(c, s) for c in reversed(CANDIDATES) for s in SEEDS  # slowest candidates first
            if not (RUNS / f"{c}-seed{s}" / "run.json").exists()]
    print(f"{len(todo)} runs to do: {todo}", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.jobs:
            c, s = todo.pop(0)
            cmd = [sys.executable, str(Path(__file__)), "--candidate", c, "--seed", str(s),
                   "--hands", str(a.hands or CANDIDATES[c]["hands"]), "--checkpoints", str(a.checkpoints),
                   "--threads", str(a.threads)]
            running.append(subprocess.Popen(cmd))
        time.sleep(5)
        running = [p for p in running if p.poll() is None]


if __name__ == "__main__":
    main()
