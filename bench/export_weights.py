"""Export the selected demo agents (bench/raw/demo_eval.json "seats") to weights/demo/.

For each selected run:
  <run>.pt            the PyTorch state_dict (torch.load(path, weights_only=True))
  <run>.bin           every tensor of the state_dict, in state_dict order, as little-endian float32,
                      C order (row-major), concatenated with no padding
  <run>.json          manifest: each tensor's name, shape and byte offset, the architecture and its
                      hyperparameters, training seed and commit, and the sha256 of <run>.bin
  <run>.parity.jsonl  decision points from evaluation play (see weights/README.md)
and seats.json, which run plays which of the three agent seats.

    python bench/export_weights.py [--decisions 1200]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np
import torch

import PokerEngine.poker_engine as pe
from eval_demo import EVAL_SEED, load
from train_demo import BB, SB, SEATS, STACK, STATE_DIM
from winrate import BOTS, greedy_policy, play_hand
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import ACTION_NAME

DEMO = ROOT / "weights" / "demo"


def card(c):
    return [int(c.rank.value), int(c.suit.value)]


def obs_dict(o):
    """Every field of the engine's Observation; enums as their integer values."""
    return {
        "viewer_id": o.viewer_id, "street": o.street, "current_player_id": o.current_player_id, "button": o.button,
        "hand_in_progress": o.hand_in_progress, "terminal": o.terminal, "winner_id": o.winner_id,
        "pot_total": o.pot_total, "to_call": o.to_call, "min_raise": o.min_raise,
        "hole": [card(c) for c in o.hole], "board": [card(c) for c in o.board],
        "alive": list(o.alive), "in_hand": list(o.in_hand), "can_act": list(o.can_act),
        "stack": list(o.stack), "street_contrib": list(o.street_contrib), "total_contrib": list(o.total_contrib),
        "folded": list(o.folded), "all_in": list(o.all_in), "viewer_to_call": o.viewer_to_call,
        "legal": [{"type": int(s.type.value), "raise_size": int(s.raise_size.value), "raise_to": s.raise_to,
                   "call_amount": s.call_amount, "min_raise_to": s.min_raise_to, "max_raise_to": s.max_raise_to,
                   "allowed": s.allowed} for s in o.legal],
    }


def f32(a):
    return np.asarray(a, dtype=np.float32).tolist()  # exact: each value is a float32, printed to round-trip


def export(run, decisions):
    model, seq, meta = load(run)
    sd = model.state_dict()
    torch.save(sd, DEMO / f"{run}.pt")
    tensors, blob, off = [], bytearray(), 0
    for name, t in sd.items():
        b = t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": off, "nbytes": len(b)})
        blob += b
        off += len(b)
    (DEMO / f"{run}.bin").write_bytes(bytes(blob))
    arch = meta["architecture"]
    manifest = {
        "run": run, "file": f"{run}.bin", "dtype": "float32", "byte_order": "little", "layout": "C order (row-major)",
        "bytes": len(blob), "parameters": sum(t.numel() for t in sd.values()), "sha256": hashlib.sha256(blob).hexdigest(),
        "tensors": tensors,
        "architecture": {"class": arch["cls"], **arch["kwargs"], "input_dim": STATE_DIM, "output_dim": len(ACTION_NAME),
                         "sequence_input": seq},
        "forward": forward_doc(arch["cls"]),
        "input": "docs/state_encoding.md (151 float32 values for 4 seats; a sequence model gets one per own decision this hand)",
        "output": "Q-value per action index, in units of 100 big blinds; play argmax(q + legal mask), ties to the lowest index",
        "actions": ACTION_NAME,
        "format": {"seats": SEATS, "stack": STACK, "blinds": [SB, BB]},
        "training": {k: meta[k] for k in ("seed", "hands", "agent", "reward_scale", "commit", "dirty",
                                          "kept_checkpoint_hands", "torch", "python")},
    }
    (DEMO / f"{run}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    parity(run, model, seq, decisions)
    return manifest


def forward_doc(cls):
    if cls == "MLPModel":
        return ("net.0 Linear, ReLU, (Dropout p=0: identity), net.3 Linear, ReLU, ... last Linear: "
                "y = x W^T + b for each Linear (weight shape [out, in]); ReLU after every Linear but the last")
    return ("PyTorch nn.GRU (batch_first, h0 = 0, gate order r, z, n in weight_ih_l<k> [3H, in] and weight_hh_l<k> [3H, H]): "
            "r = sigmoid(W_ir x + b_ir + W_hr h + b_hr); z = sigmoid(W_iz x + b_iz + W_hz h + b_hz); "
            "n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); h' = (1 - z) * n + z * h. Layer k > 0 takes layer k-1's h "
            "as its x. fc Linear on the last layer's h at the last step.")


def parity(run, model, seq, n):
    """Decision points from evaluation play: the first deals of the evaluation seed, against each
    baseline in turn, every decision of the agent, until n are recorded (n/3 per baseline)."""
    rows = []
    for bname, bot in BOTS.items():
        rec = []
        pol = greedy_policy(model, seq, record=rec)
        g = pe.GameEngine(SEATS, STACK, BB, SB)
        rng = random.Random(EVAL_SEED)
        d = 0
        while len(rec) < n // len(BOTS):
            for seat in range(SEATS):
                g.seed(EVAL_SEED * 1_000_003 + d)  # as winrate.evaluate
                play_hand(g, [pol if i == seat else bot for i in range(SEATS)], rng, STACK)
            d += 1
        for r in rec:
            *prev, cur = r["observations"]
            x = r["input"] if seq else r["input"][None]
            rows.append({"baseline": bname, "hand": r["hand"], "observation": obs_dict(cur), "state": f32(x[-1]),
                         "history": [{"observation": obs_dict(o), "state": f32(s)} for o, s in zip(prev, x[:-1])] if seq else [],
                         "mask": f32(r["mask"]), "q": f32(r["q"]), "action": r["action"]})
    with open(DEMO / f"{run}.parity.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1200)
    a = ap.parse_args()
    torch.set_num_threads(1)
    DEMO.mkdir(parents=True, exist_ok=True)
    ev = json.loads((ROOT / "bench" / "raw" / "demo_eval.json").read_text())
    out = {"seats": ev["seats"], "one_network_in_all_seats": ev["one_network_in_all_seats"], "agents": {}}
    for run in dict.fromkeys(ev["seats"]):
        m = export(run, a.decisions)
        out["agents"][run] = {"bytes": m["bytes"], "parameters": m["parameters"], "sha256": m["sha256"]}
        print(run, m["bytes"], "bytes", m["sha256"])
    (DEMO / "seats.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
