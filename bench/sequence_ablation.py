"""Does the betting history help? A short matched-budget comparison for issue #15.

Three candidates at the same number of hands, seeds and settings as bench/train_demo.py: `gru`
(the demo's GRU over the seat's decisions so far, sequence=True), `gru-noseq` (the same network
given one state at a time, sequence=False) and `transformer-seq` (a small Transformer over the
same sequences). They are added to train_demo.CANDIDATES here at run time, so `train_demo.py
--all` does not sweep them. After training, the kept checkpoint plays greedily against each
baseline on EVAL_DEALS duplicate deals from EVAL_SEED, a seed nothing else uses.

    python bench/sequence_ablation.py --candidate gru-noseq --seed 1 --out /tmp/ablation

Writes bench/raw/sequence_ablation/<candidate>-seed<seed>.json; the weights and the training log
go to --out (not weights/runs, which git tracks).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import torch

import train_demo
from Models.TransformerQModel import TransformerQModel
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import N_ACTIONS
from winrate import BOTS, evaluate, greedy_policy

GRU = dict(hidden_dim=128, layer_dim=1, dropout=0.0)
CANDIDATES = {
    "gru": train_demo.CANDIDATES["gru"],
    "gru-noseq": dict(cls="GRUModel", kwargs=GRU, sequence=False),
    "transformer-seq": dict(cls="TransformerQModel", sequence=True,
                            kwargs=dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0)),
}
EVAL_SEED, EVAL_DEALS = 31500, 1000
RAW = ROOT / "bench" / "raw" / "sequence_ablation"

# train_demo.train looks these up by name in its own module
train_demo.CANDIDATES.update({k: v for k, v in CANDIDATES.items() if k not in train_demo.CANDIDATES})
_build = train_demo.build


def build(name):
    c = train_demo.CANDIDATES[name]
    if c["cls"] == "TransformerQModel":
        return TransformerQModel(input_size=train_demo.STATE_DIM, num_actions=N_ACTIONS, **c["kwargs"])
    return _build(name)


train_demo.build = build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hands", type=int, default=10_000)
    ap.add_argument("--out", required=True, help="scratch directory for the weights and the training log")
    a = ap.parse_args()
    out = Path(a.out) / f"{a.candidate}-seed{a.seed}"
    load_before = os.getloadavg()
    train_demo.train(a.candidate, a.seed, a.hands, out, checkpoints=2, threads=1)
    run = json.loads((out / "run.json").read_text())

    model = build(a.candidate)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    model.eval()
    seq = CANDIDATES[a.candidate]["sequence"]
    torch.set_num_threads(1)
    vs, t0 = {}, time.time()
    for bname, bot in BOTS.items():
        with torch.no_grad():
            r = evaluate(greedy_policy(model, seq), bot, EVAL_DEALS, EVAL_SEED)
        vs[bname] = {k: r[k] for k in ("bb100", "ci95_lo", "ci95_hi")}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{a.candidate}-seed{a.seed}.json").write_text(json.dumps({
        "candidate": a.candidate, "architecture": CANDIDATES[a.candidate], "sequence": seq, "seed": a.seed,
        "hands": a.hands, "kept_checkpoint_hands": run["kept_checkpoint_hands"], "parameters": run["parameters"],
        "train_wall_s": run["wall_s"], "train_cpu_s": run["cpu_s"], "eval_wall_s": round(time.time() - t0, 1),
        "eval": {"seed": EVAL_SEED, "deals": EVAL_DEALS, "bb100": vs},
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "dirty": run["dirty"], "load_average_before": load_before, "load_average_after": os.getloadavg(),
        "torch": torch.__version__}, indent=1) + "\n")


if __name__ == "__main__":
    main()
