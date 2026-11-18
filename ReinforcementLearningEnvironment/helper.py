import os
import numpy as np
import csv

import matplotlib.pyplot as plt


import PokerEngine.poker_engine as pe
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    ACTION_NAME, STREET_NAME, RUN_ID, A_FOLD,
    A_CHECK, A_CALL, A_RAISE_BB, A_RAISE_HALF,
    A_RAISE_POT, A_ALLIN, CSV_FIELDS, CSV_PATH,
    N_ACTIONS, RANK_ORDER, RANK_TO_I, RANKS, SUITS,
    rank_map
    )

def action_index_to_engine_action(a_idx: int) -> pe.Action:
    """Action index to engine action

    Args:
        a_idx (int): a_idx

    Raises:
        ValueError: _description_

    Returns:
        pe.Action: action
    """
    a = pe.Action()
    if a_idx == A_FOLD:
        a.type = pe.ActionType.Fold
    elif a_idx == A_CHECK:
        a.type = pe.ActionType.Check
    elif a_idx == A_CALL:
        a.type = pe.ActionType.Call
    elif a_idx == A_RAISE_BB:
        a.type = pe.ActionType.Raise
        a.raiseSize = pe.RaiseSize.BB
    elif a_idx == A_RAISE_HALF:
        a.type = pe.ActionType.Raise
        a.raiseSize = pe.RaiseSize.HalfPot
    elif a_idx == A_RAISE_POT:
        a.type = pe.ActionType.Raise
        a.raiseSize = pe.RaiseSize.Pot
    elif a_idx == A_ALLIN:
        a.type = pe.ActionType.AllIn
    else:
        raise ValueError(a_idx)
    return a

def card_to_str(c):
    """converts the playing card into a string

    Args:
        c (_type_):card

    Returns:
        _type_: string
    """
    rank_map = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",
                9:"9",10:"T",11:"J",12:"Q",13:"K",14:"A"}
    suit_map = {1:"h", 2:"d", 3:"c", 4:"s"}

    r = int(getattr(c.rank, "value", c.rank))
    s = int(getattr(c.suit, "value", c.suit))

    if r == 0 or s == 0:
        return "??"  # empty/unknown card

    return rank_map[r] + suit_map[s]


def pocket_key(c1, c2):
    """returns the pocket card key

    Args:
        c1 (_type_): card1
        c2 (_type_): card2

    Returns:
        _type_: string
    """
    a, b = card_to_str(c1), card_to_str(c2)
    if "?" in a or "?" in b:
        return None
    return "".join(sorted([a, b]))


def moving_average(x, window=200):
    """Generates the moving average"""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    w = int(max(1, window))
    if len(x) < w:
        return np.convolve(x, np.ones(len(x), dtype=np.float32)/len(x), mode="valid")
    return np.convolve(x, np.ones(w, dtype=np.float32)/w, mode="valid")

def append_hand_rows(rows):
    """Appends the hand rows"""
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def legal_mask_from_obs(obs) -> np.ndarray:
    """
    Returns mask of shape (N_ACTIONS,), with 0 for legal and -1e9 for illegal.
    """
    mask = np.full((N_ACTIONS,), -1e9, dtype=np.float32)
    # obs.legal has one ActionSpec per action, one per raise size; for the player to act, each is
    # marked exactly as step() judges it
    for spec in obs.legal:
        if spec.allowed:
            mask[spec_index(spec)] = 0.0
    return mask


def spec_index(spec) -> int:
    """Action index of an engine ActionSpec (the inverse of action_index_to_engine_action)."""
    if spec.type == pe.ActionType.Raise:
        return {pe.RaiseSize.BB: A_RAISE_BB, pe.RaiseSize.HalfPot: A_RAISE_HALF, pe.RaiseSize.Pot: A_RAISE_POT}[spec.raise_size]
    return {pe.ActionType.Fold: A_FOLD, pe.ActionType.Check: A_CHECK, pe.ActionType.Call: A_CALL,
            pe.ActionType.AllIn: A_ALLIN}[spec.type]

def encode_card(card) -> np.ndarray:
    """encodes the cards and outputs a numpy array"""
    # rank in [0..14], suit in [0..4]
    return np.array([int(card.rank), int(card.suit)], dtype=np.float32)


def parse_pocket(p):
    """Parses the pockets into ranks and is suited"""
    # p like "AhKd" or "7d7c" (sorted already)
    r1, s1, r2, s2 = p[0], p[1], p[2], p[3]
    suited = (s1 == s2)
    return r1, r2, suited

def pocket_169_key(p):
    """creates pockey key for storing and processing"""
    r1, r2, suited = parse_pocket(p)
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if RANK_TO_I[r1] > RANK_TO_I[r2] else (r2, r1)
    return f"{hi}{lo}{'s' if suited else 'o'}"


def card_one_hot(card):
    """
    Card → 17-dim vector
    [rank one-hot (13), suit one-hot (4)]
    Empty cards become all zeros.
    """
    vec = [0.0] * (RANKS + SUITS)

    if card.rank.value != 0:
        vec[card.rank.value - 2] = 1.0  # ranks 2..14 → 0..12
    if card.suit.value != 0:
        vec[RANKS + card.suit.value - 1] = 1.0  # suits 1..4 → 0..3

    return vec

def encode_observation(obs, big_blind=2):
    """State vector for the player viewing obs (obs.viewer_id), 127 + 6 * seats floats (151 for 4).

    docs/state_encoding.md specifies every element. Seats are listed relative to the viewer
    (viewer first, then the seats after it in table order), so the same network can play any seat,
    and chip amounts are in units of 100 big blinds.
    """
    n_players = len(obs.stack)
    v = obs.viewer_id
    chips = 1.0 / (100.0 * big_blind)
    parts = []

    # 7 cards -> 7 * 17 = 119 (hole, then board; cards not dealt yet are all zeros)
    for c in obs.hole:
        parts.append(card_one_hot(c))
    for c in obs.board:
        parts.append(card_one_hot(c))

    # scalars -> 4
    parts.append(np.array([
        obs.pot_total * chips,
        obs.to_call * chips,
        obs.min_raise * chips,
        obs.viewer_to_call * chips,
    ], dtype=np.float32))

    # street one-hot -> 4
    street_oh = np.zeros(4, dtype=np.float32)
    if 0 <= obs.street < 4:
        street_oh[obs.street] = 1.0
    parts.append(street_oh)

    # per-seat info, viewer first: 5 each -> 5*n_players
    per = []
    for k in range(n_players):
        i = (v + k) % n_players
        per.extend([
            obs.stack[i] * chips,
            float(obs.folded[i]),
            float(obs.all_in[i]),
            obs.street_contrib[i] * chips,
            obs.total_contrib[i] * chips,
        ])
    parts.append(np.array(per, dtype=np.float32))

    # button, relative to the viewer: one-hot over n_players (0 = the viewer has the button)
    button_oh = np.zeros(n_players, dtype=np.float32)
    if obs.button >= 0:
        button_oh[(obs.button - v) % n_players] = 1.0
    parts.append(button_oh)

    return np.concatenate(parts, axis=0, dtype=np.float32)  # the card lists would otherwise make it float64


def pocket_features(pocket: str) -> dict: 
    """Turns pocket into 4 variables for symbolic regression

    Args:
        pocket (str): pocket

    Returns:
        dict: dictionary of 4 variables: higher ranked cards number, low ranked cards number, is it a pair, is it suited
    """
    r1, s1 = pocket[0], pocket[1]
    r2, s2 = pocket[2], pocket[3]
    v1, v2 = rank_map[r1], rank_map[r2]
    return {
        "rank_high": max(v1, v2),
        "rank_low":  min(v1, v2),
        "is_pair":   int(v1 == v2),
        "is_suited": int(s1 == s2),
    }
    
def pocket_win_by_street(df, player_id, street_end):
    """Get the pocket win by street"""
    d = df[(df.player_id==player_id) & (df.street_end==street_end)].copy()
    d["p169"] = d["pocket"].apply(pocket_169_key)
    g = d.groupby("p169").agg(
        n=("bb_delta","size"),
        ev=("bb_delta","mean"),
        profit_rate=("profitable","mean"),
    ).reset_index().sort_values("ev", ascending=False)
    return g    

def bluff_proxy_table(df, player_id, min_samples=50):
    """bluff proxy score = aggressive + nonshowdown wins, penalize high showdown"""
    d = df[df["player_id"] == player_id].copy()
    d["p169"] = d["pocket"].apply(pocket_169_key)

    g = d.groupby("p169").agg(
        n=("bb_delta","size"),
        ev=("bb_delta","mean"),
        agg_rate=("aggressive","mean"),
        ns_win_rate=("non_showdown_win","mean"),
        showdown_rate=("went_showdown","mean"),
        fold_rate=("folded","mean"),
    ).reset_index()

    g = g[g["n"] >= min_samples]

    g["bluff_score"] = g["agg_rate"] * g["ns_win_rate"] * (1.0 - g["showdown_rate"])

    return g.sort_values("bluff_score", ascending=False)
def pocket_aggression_table(df, player_id, min_samples=50):
    """Generate aggression table"""
    d = df[df["player_id"] == player_id].copy()
    d["p169"] = d["pocket"].apply(pocket_169_key)
    g = d.groupby("p169").agg(
        n=("bb_delta","size"),
        ev=("bb_delta","mean"),
        agg_rate=("aggressive","mean"),
        fold_rate=("folded","mean"),
    ).reset_index()
    g = g[g["n"] >= min_samples].sort_values("ev", ascending=False)
    return g

def first_action_dist(df, player_id):
    """first action"""
    d = df[df["player_id"] == player_id]
    return d["first_action_name"].value_counts(normalize=True)


def plot_pocket_ev_vs_sharpe_scatter(
    df,
    min_hands=200,
    use_excess=False,
    annualize_like=False,
    clip_sharpe=None,
    label_top_k=15,
):
    """Plot ev vs sharpe ratio in a scatter plot"""
    dfx = df.copy()
    dfx["p169"] = dfx["pocket"].apply(pocket_169_key)

    g = dfx.groupby("p169")["bb_delta"]
    stats = g.agg(n="count", mean="mean", std="std").reset_index()

    overall_mean = float(dfx["bb_delta"].mean())

    if use_excess:
        stats["ev"] = stats["mean"] - overall_mean
        y_label = "EV (mean excess BB/hand)"
    else:
        stats["ev"] = stats["mean"]
        y_label = "EV (mean BB/hand)"

    stats["sharpe"] = stats["ev"] / stats["std"].replace(0, np.nan)

    if annualize_like:
        stats["sharpe"] = stats["sharpe"] * np.sqrt(stats["n"])

    # filter AFTER ev/sharpe computed, and use columns from stats only
    stats = stats[(stats["n"] >= min_hands) & np.isfinite(stats["sharpe"]) & np.isfinite(stats["ev"])]

    if clip_sharpe is not None:
        lo, hi = clip_sharpe
        stats = stats[(stats["sharpe"] >= lo) & (stats["sharpe"] <= hi)]

    plt.figure(figsize=(10, 6))
    plt.scatter(stats["sharpe"].to_numpy(), stats["ev"].to_numpy(), alpha=0.75)

    plt.axhline(0, linewidth=1, alpha=0.3)
    plt.axvline(0, linewidth=1, alpha=0.3)

    plt.xlabel("Sharpe (EV/std of BB_delta per hand)")
    plt.ylabel(y_label)
    plt.title("Pocket (169) scatter: EV vs Sharpe")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if label_top_k and label_top_k > 0 and len(stats) > 0:
        # score = high EV and high Sharpe
        stats2 = stats.copy()
        stats2["_score"] = stats2["ev"].rank(pct=True) + stats2["sharpe"].rank(pct=True)
        top = stats2.nlargest(min(label_top_k, len(stats2)), "_score")
        for _, r in top.iterrows():
            plt.annotate(
                r["p169"],
                (r["sharpe"], r["ev"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=9,
                alpha=0.9,
            )

    plt.show()

    return stats.sort_values(["ev", "sharpe"], ascending=False)

def plot_pocket_ev_curves(df, pockets=("AKs", "AQs", "72o"), max_points=2000):
    """plot pocket ev curves"""
    dfx = df.copy()
    dfx["p169"] = dfx["pocket"].apply(pocket_169_key)

    plt.figure(figsize=(12, 5))

    for pocket in pockets:
        d = dfx[dfx["p169"] == pocket].sort_values("hand_id")
        if len(d) == 0:
            print("No samples for", pocket)
            continue

        ev = d["bb_delta"].to_numpy(dtype=np.float32)
        cum_avg = np.cumsum(ev) / (np.arange(len(ev)) + 1)

        if len(cum_avg) > max_points:
            idx = np.linspace(0, len(cum_avg) - 1, max_points).astype(int)
            x = d["hand_id"].to_numpy()[idx]
            y = cum_avg[idx]
        else:
            x = d["hand_id"].to_numpy()
            y = cum_avg

        plt.plot(x, y, label=pocket)

    plt.xlabel("Hand")
    plt.ylabel("Cumulative avg BB/hand")
    plt.title("EV curves by pocket (cumulative average)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    
def plot_pocket_ev_curve(df, pocket_169="AKs", max_points=2000):
    """plot pocket ev curve"""
    df = df.copy()
    df["p169"] = df["pocket"].apply(pocket_169_key)
    d = df[df["p169"] == pocket_169].sort_values("hand_id")
    if len(d) == 0:
        print("No samples for", pocket_169)
        return

    # cumulative average EV
    ev = d["bb_delta"].to_numpy(dtype=np.float32)
    cum_avg = np.cumsum(ev) / (np.arange(len(ev)) + 1)


    # downsample for plotting
    if len(cum_avg) > max_points:
        idx = np.linspace(0, len(cum_avg)-1, max_points).astype(int)
        x = d["hand_id"].to_numpy()[idx]
        y = cum_avg[idx]
    else:
        x = d["hand_id"].to_numpy()
        y = cum_avg

    plt.figure(figsize=(10,4))
    plt.plot(x, y)
    plt.xlabel("Hand")
    plt.ylabel("Cumulative avg BB/hand")
    plt.title(f"EV curve: {pocket_169} (cumulative average)")
    plt.tight_layout()
    plt.show()


def make_13x13(df, value_col="bb_delta", agg="mean", min_samples=30):
    """Generate a 13x13 grid"""
    grid = np.full((13,13), np.nan, dtype=np.float32)
    counts = np.zeros((13,13), dtype=np.int32)

    for _, row in df.iterrows():
        p = row["pocket"]
        r1, r2, suited = parse_pocket(p)
        i1, i2 = RANK_TO_I[r1], RANK_TO_I[r2]
        v = row[value_col]

        if r1 == r2:
            i, j = i1, i1
        else:
            hi, lo = (i1, i2) if i1 > i2 else (i2, i1)
            if suited:
                i, j = hi, lo   # above diagonal
            else:
                i, j = lo, hi   # below diagonal

        counts[i,j] += 1
        # accumulate in temp list via groupby later is cleaner; but keep simple:
    # do proper aggregation:
    # rebuild via groupby
    buckets = {}
    for _, row in df.iterrows():
        p = row["pocket"]
        r1, r2, suited = parse_pocket(p)
        i1, i2 = RANK_TO_I[r1], RANK_TO_I[r2]
        if r1 == r2:
            i, j = i1, i1
        else:
            hi, lo = (i1, i2) if i1 > i2 else (i2, i1)
            i, j = (hi, lo) if suited else (lo, hi)
        buckets.setdefault((i,j), []).append(row[value_col])

    for (i,j), vals in buckets.items():
        if len(vals) < min_samples:
            continue
        if agg == "mean":
            grid[i,j] = float(np.mean(vals))
        elif agg == "winrate":
            grid[i,j] = float(np.mean(np.array(vals) > 0))
        else:
            raise ValueError("agg must be mean or winrate")

    return grid

def plot_heatmap(grid, title, vmin=None, vmax=None):
    """plot heatmap"""
    plt.figure(figsize=(8,7))
    plt.imshow(grid, origin="lower", vmin=vmin, vmax=vmax)
    plt.xticks(range(13), list(RANK_ORDER))
    plt.yticks(range(13), list(RANK_ORDER))
    plt.xlabel("Rank")
    plt.ylabel("Rank")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    