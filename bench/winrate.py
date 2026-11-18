"""Train the README's four agents by self-play, then measure each against fixed baseline bots.

Training: train_poker_dqn exactly as in main.ipynb (MLP, LSTM, GRU, Transformer; 4 seats;
stack 10,000; blinds 1/2), for --train-hands hands per seed. Seeds fix Python, NumPy and torch
RNGs and the engine's deals. The per-hand training log goes to bench/raw/train_seed<S>.csv.

Evaluation: one trained agent (greedy: epsilon 0, model.eval()) at a 4-seat table with three
copies of one baseline bot; stacks reset to 10,000 every hand. Duplicate deals: each deal is
replayed 4 times with the agent in each seat (so each seat's cards and position), and the deal's
score is the agent's mean result over the 4 replays. bb/100 = 100 x mean deal score; the 95% CI
is mean +/- 1.96 x sd / sqrt(deals). "random-bot" is a control: the random baseline in the
agent's seat, i.e. what an agent that learned nothing would score.

Baselines:
  random  uniform over the legal action indices the agent sees (3 raise sizes count separately)
  call    calls when facing a bet, otherwise checks
  tight   preflop: pot-raise TT+/AK/AQ, call other pairs, two broadway cards or suited aces,
          otherwise check/fold; postflop: call/check with a pocket pair or a hole card that
          pairs the board, otherwise check/fold

    python bench/winrate.py --seeds 1 2 3 --train-hands 2000 --eval-deals 1000
"""
import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

import PokerEngine.poker_engine as pe
import ReinforcementLearningEnvironment.helper as helper
from Models.GRUModel import GRUModel
from Models.LSTMModel import LSTMModel
from Models.MLPModel import MLPModel
from Models.TransformerQModel import TransformerQModel
from ReinforcementLearningEnvironment.helper import action_index_to_engine_action, encode_observation, legal_mask_from_obs, no_free_fold
from ReinforcementLearningEnvironment.Poker import train_poker_dqn
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    A_CALL, A_CHECK, A_FOLD, A_RAISE_POT, N_ACTIONS)

SEATS, STACK, BB, SB = 4, 10000, 2, 1
ARCH = ["MLP", "LSTM", "GRU", "Transformer"]


def readme_models():
    f = 151  # encode_observation length for 4 seats
    return [MLPModel(input_features=f, output_features=N_ACTIONS, hidden=(512, 1024, 2048, 1024, 512)),
            LSTMModel(input_dim=f, output_dim=N_ACTIONS),
            GRUModel(input_dim=f, output_dim=N_ACTIONS),
            TransformerQModel(input_size=f, num_actions=N_ACTIONS)]


def legal_idxs(mask):
    return np.where(mask == 0.0)[0]


def pick(mask, *prefs):
    for a in prefs:
        if mask[a] == 0.0:
            return a
    return A_FOLD


def rank(c):
    return int(c.rank.value)


def bot_random(obs, mask, rng):
    return int(rng.choice(legal_idxs(mask)))


def bot_call(obs, mask, rng):
    return pick(mask, A_CALL, A_CHECK)


def bot_tight(obs, mask, rng):
    h1, h2 = obs.hole
    hi, lo = max(rank(h1), rank(h2)), min(rank(h1), rank(h2))
    suited = h1.suit == h2.suit
    if obs.street == 0:
        if (hi == lo and hi >= 10) or (hi == 14 and lo >= 12):
            return pick(mask, A_RAISE_POT, A_CALL, A_CHECK)
        if hi == lo or lo >= 10 or (hi == 14 and suited):
            return pick(mask, A_CALL, A_CHECK)
        return pick(mask, A_CHECK, A_FOLD)
    board = {rank(c) for c in obs.board if rank(c) != 0}
    made = hi == lo or rank(h1) in board or rank(h2) in board
    return pick(mask, A_CALL, A_CHECK) if made else pick(mask, A_CHECK, A_FOLD)


BOTS = {"random": bot_random, "call": bot_call, "tight": bot_tight}


def greedy_policy(model, sequence=False, record=None, free_fold_rule=False):
    """A trained network's play: argmax of its Q-values plus the legal mask. A sequence model sees
    the player's encoded states at each of its decisions in this hand so far, the current one last
    (docs/state_encoding.md). act.counts tallies the actions chosen; act.free_folds counts the
    folds chosen when checking was legal (by the network, before any rule). free_fold_rule=True
    applies the play-time option no_free_fold (helper.no_free_fold) before the argmax; it is off in
    training and in the pre-registered evaluation. If record is a list, each decision is appended to
    it: the observations this hand (the current one last), the network input, the mask, the
    Q-values, the action without the rule and the action with it."""
    hist, seen = {}, {}

    def act(obs, mask, rng):
        s = encode_observation(obs, big_blind=BB)
        seen.setdefault(obs.viewer_id, []).append(obs)
        if sequence:
            hist.setdefault(obs.viewer_id, []).append(s)
            s = np.stack(hist[obs.viewer_id])
        with torch.no_grad():
            q = model(torch.from_numpy(s)[None])[0]
        a_raw = int(torch.argmax(q + torch.from_numpy(mask)))
        a_rule = int(torch.argmax(q + torch.from_numpy(no_free_fold(mask))))
        act.free_folds += int(a_raw == A_FOLD and mask[A_CHECK] == 0.0)
        a = a_rule if free_fold_rule else a_raw
        act.counts[a] += 1
        if record is not None:
            record.append({"observations": list(seen[obs.viewer_id]), "input": s, "mask": mask, "q": q.numpy().copy(),
                           "action": a_raw, "action_no_free_fold": a_rule, "hand": act.hands})
        return a

    def new_hand():
        hist.clear(); seen.clear()
        act.hands += 1
    act.new_hand = new_hand
    act.counts = [0] * N_ACTIONS
    act.free_folds = 0
    act.hands = 0
    return act


def agent_policy(agent):
    agent.model.eval()
    return greedy_policy(agent.model, agent.sequence)


def play_hand(g, policies, rng, stack=STACK):
    g.reset_tournament(stack)
    g.start_hand()
    for pol in policies:
        if hasattr(pol, "new_hand"):
            pol.new_hand()
    steps = 0
    while not g.hand_terminal():
        steps += 1
        if steps > 2000:
            raise RuntimeError("hand did not finish")
        p = g.currentPlayerSafe()
        obs = g.getObservation(p)
        mask = legal_mask_from_obs(obs)
        g.step(action_index_to_engine_action(policies[p](obs, mask, rng)), p)
    return [(s - stack) / BB for s in g.getObservation(0).stack]


def evaluate(policy, bot, deals, seed, stack=STACK, keep_scores=False):
    """Agent result in BB per hand for each deal, averaged over the 4 seat rotations. keep_scores
    adds the per-deal scores (for paired comparisons on the same deals)."""
    g = pe.GameEngine(SEATS, stack, BB, SB)
    rng = random.Random(seed)
    scores = []
    for d in range(deals):
        total = 0.0
        for seat in range(SEATS):
            g.seed(seed * 1_000_003 + d)  # same deck for all 4 rotations of this deal
            pols = [policy if i == seat else bot for i in range(SEATS)]
            total += play_hand(g, pols, rng, stack)[seat]
        scores.append(total / SEATS)
    s = np.array(scores)
    mean, half = s.mean(), 1.96 * s.std(ddof=1) / np.sqrt(len(s))
    out = {"bb100": 100 * mean, "ci95_lo": 100 * (mean - half), "ci95_hi": 100 * (mean + half),
           "sd_bb_per_deal": float(s.std(ddof=1)), "deals": deals, "hands": deals * SEATS}
    if keep_scores:
        out["scores"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--train-hands", type=int, default=2000)
    ap.add_argument("--eval-deals", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "bench" / "raw"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(), "device": "cpu", "train_hands": a.train_hands,
            "eval_deals": a.eval_deals, "seats": SEATS, "stack": STACK, "blinds": [SB, BB]}
    print(json.dumps(meta))
    for seed in a.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        helper.CSV_PATH = str(out / f"train_seed{seed}.csv")
        Path(helper.CSV_PATH).unlink(missing_ok=True)
        t0, r0 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        agents, history = train_poker_dqn(readme_models(), SEATS, STACK, BB, SB, max_hands=a.train_hands,
                                          device="cpu", ma_window=50, seed=seed)
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        train = {"seed": seed, "train_hands": a.train_hands, "wall_s": time.time() - t0,
                 "cpu_s": (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime),
                 "tournament_resets": len(history["resets"]), "aborted_hands": len(history["aborted"]),
                 "abort_reasons": [r for _, r in history["aborted"]]}
        print(json.dumps(train), flush=True)
        results = []
        players = [(ARCH[i], agent_policy(ag)) for i, ag in enumerate(agents)] + [("random-bot", bot_random)]
        t1 = time.time()
        for name, pol in players:
            for bname, bot in BOTS.items():
                with torch.no_grad():
                    r = evaluate(pol, bot, a.eval_deals, seed)
                r.update({"seed": seed, "agent": name, "baseline": bname})
                results.append(r)
                print(json.dumps(r), flush=True)
        train["eval_wall_s"] = time.time() - t1
        (out / f"winrate_seed{seed}.json").write_text(json.dumps({"meta": meta, "train": train, "results": results}, indent=1))


if __name__ == "__main__":
    main()
