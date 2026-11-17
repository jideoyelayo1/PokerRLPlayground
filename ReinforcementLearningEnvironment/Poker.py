import PokerEngine.poker_engine as pe
from ReinforcementLearningEnvironment.Agent import DQNAgent
from ReinforcementLearningEnvironment.helper import (
    legal_mask_from_obs, action_index_to_engine_action, moving_average,
    encode_observation, pocket_key, append_hand_rows
    )
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    ACTION_NAME, STREET_NAME, RUN_ID, A_FOLD,
    A_CHECK, A_CALL, A_RAISE_BB, A_RAISE_HALF,
    A_RAISE_POT, A_ALLIN, CSV_FIELDS, CSV_PATH,
    N_ACTIONS, RANK_ORDER, RANK_TO_I, RANKS, SUITS
    )

from tqdm.auto import tqdm
import random
import numpy as np



def train_poker_dqn(
    models,
    n_players: int,
    starting_stack: int,
    big_blind: int,
    small_blind: int,
    max_hands: int = 2000,
    device="cpu",
    ma_window: int = 200,
    seed=None,
    reset_stacks: bool = False,
):
    """Self-play DQN training. reset_stacks=True resets every stack to starting_stack before each
    hand (a cash game, as the demo is played; the button then always starts on seat 0); otherwise the
    stacks carry over and the tournament is reset when fewer than two players have chips."""

    g = pe.GameEngine(n_players, starting_stack, big_blind, small_blind)
    if seed is not None:
        g.seed(seed)  # deals only; seed random/numpy/torch in the caller for exploration and weights
    bb = float(big_blind)

    # ---- Infer state_dim (probe hand), then reset so training starts clean ----
    g.reset_tournament(starting_stack)
    g.start_hand()
    tmp_obs = g.getObservation(0)
    state_dim = encode_observation(tmp_obs, big_blind=big_blind).shape[0]
    g.reset_tournament(starting_stack)  # discard probe hand

    agents = [DQNAgent(i, models[i], state_dim=state_dim, device=device)
              for i in range(n_players)]

    history = {
        "hand": [],
        "delta_bb": {i: [] for i in range(n_players)},
        "ma_bb":    {i: [] for i in range(n_players)},
        "resets":   [],
        "aborted":  [],
    }

    hands_played = 0
    pbar = tqdm(total=max_hands)

    MAX_STEPS_PER_HAND = 2000  # no wall-clock limit: it made seeded runs depend on machine load

    while hands_played < max_hands:

        # Reset tournament if needed
        if reset_stacks:
            g.reset_tournament(starting_stack)
        elif g.alivePlayers() < 2:
            history["resets"].append(hands_played)
            g.reset_tournament(starting_stack)

        # Stacks before the blinds, so the reward and bb_delta include them
        stacks_start = list(g.getObservation(0).stack[:n_players])
        # Dealt in = has chips (start_hand may also finish the hand and drop whoever busts in it)
        dealt_in = [st > 0 for st in stacks_start]

        g.start_hand()

        # Pocket cards now: a folded player's observation hides them by the end of the hand,
        # and a busted player still holds stale cards from the hand they busted in
        pockets = [pocket_key(*g.getObservation(i).hole) if dealt_in[i] else None for i in range(n_players)]

        # Determine BB player for this hand (heuristic)
        bb_player = None
        for pid in range(n_players):
            o = g.getObservation(pid)
            if float(o.viewer_to_call) == 0.0:  # BB posted fully
                bb_player = pid
                break

        hand_actions = {
            i: {
                "first_action": None,
                "raises_made": 0,
                "raises_faced": 0,
                "bb_raised": 0,
                "folded": False,
                "exit_street": None,
            }
            for i in range(n_players)
        }

        steps = 0
        hand_aborted = False
        # Each seat's last decision (state, action), stored once its outcome is known: at the
        # seat's next decision (reward 0, next state = that decision's state) or at the end of the
        # hand (reward = the seat's result for the hand, done). So every seat that acted learns
        # its result, not only the one whose action ends the hand.
        pending = {}

        def store(i, s_next, r, d, mask_next=None):
            s_prev, a_prev = pending.pop(i)
            agents[i].remember(s_prev, a_prev, r, s_next, d, mask_next)

        while not g.hand_terminal():
            steps += 1

            if steps > MAX_STEPS_PER_HAND:
                hand_aborted = "steps"
                break

            acting = g.currentPlayerSafe()
            if acting < 0:
                hand_aborted = "no current player"
                break

            obs = g.getObservation(acting)
            s = encode_observation(obs, big_blind=big_blind)
            mask = legal_mask_from_obs(obs)
            if acting in pending:
                store(acting, s, 0.0, False, mask)

            legal_idxs = np.where(mask == 0.0)[0]
            if len(legal_idxs) == 0:
                hand_aborted = "no legal action"
                break

            a_idx = agents[acting].select_action(s, mask)
            if mask[a_idx] != 0.0:
                a_idx = int(random.choice(legal_idxs))

            try:
                g.step(action_index_to_engine_action(a_idx), acting)  # the mask is exact: no fallback
            except Exception as e:
                hand_aborted = f"error: {e}"
                break

            if a_idx == A_FOLD and hand_actions[acting]["exit_street"] is None:
                hand_actions[acting]["exit_street"] = int(obs.street)

            pending[acting] = (s, a_idx)

            if hand_actions[acting]["first_action"] is None:
                hand_actions[acting]["first_action"] = a_idx

            if a_idx in (A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT, A_ALLIN):
                hand_actions[acting]["raises_made"] += 1
                for i in range(n_players):
                    if i != acting:
                        hand_actions[i]["raises_faced"] += 1
                if bb_player is not None and acting == bb_player and int(obs.street) == 0:
                    hand_actions[acting]["bb_raised"] = 1

            if a_idx == A_FOLD:
                hand_actions[acting]["folded"] = True

        if hand_aborted:
            history["aborted"].append((hands_played, hand_aborted))
            history["resets"].append(hands_played)
            g.reset_tournament(starting_stack)
            pbar.set_postfix_str("hand_aborted")
            pbar.refresh()
            continue

        obs_end = g.getObservation(0)

        # For anyone who didn't fold, set exit_street to final street of the hand
        for i in range(n_players):
            if hand_actions[i]["exit_street"] is None:
                hand_actions[i]["exit_street"] = int(obs_end.street)

        stacks_end = list(obs_end.stack[:n_players])
        assert sum(stacks_end) == sum(stacks_start), "chips not conserved"

        # Terminal transition for every seat that acted; the next state is unused when done.
        for i in list(pending):
            store(i, pending[i][0], float(stacks_end[i] - stacks_start[i]) / bb, True)

        rows = []
        for i in range(n_players):
            pkey = pockets[i]
            if pkey is None:
                continue

            street_end_i = int(hand_actions[i]["exit_street"])
            street_end_name_i = STREET_NAME.get(street_end_i, "unknown")

            bb_delta = (stacks_end[i] - stacks_start[i]) / bb
            went_showdown = int(street_end_i == 3 and not hand_actions[i]["folded"])
            first_a = hand_actions[i]["first_action"]

            rows.append({
                "run_id": RUN_ID,
                "hand_id": hands_played,
                "player_id": i,
                "pocket": pkey,

                "street_end": street_end_i,
                "street_end_name": street_end_name_i,

                "bb_delta": bb_delta,
                "profitable": int(bb_delta > 0),

                "folded": int(hand_actions[i]["folded"]),
                "went_showdown": went_showdown,
                "non_showdown_win": int((not went_showdown) and (bb_delta > 0)),

                "first_action": first_a if first_a is not None else -1,
                "first_action_name": ACTION_NAME.get(first_a, "unknown"),

                "bb_raised": hand_actions[i]["bb_raised"],
                "raises_made": hand_actions[i]["raises_made"],
                "raises_faced": hand_actions[i]["raises_faced"],
                "aggressive": int(hand_actions[i]["raises_made"] > 0),
            })

        append_hand_rows(rows)

        history["hand"].append(hands_played)
        for i in range(n_players):
            d = float(stacks_end[i] - stacks_start[i]) / bb
            history["delta_bb"][i].append(d)

        for ag in agents:
            ag.train_long_memory()
            ag.n_hands += 1

        hands_played += 1
        pbar.update(1)

    pbar.close()

    for i in range(n_players):
        history["ma_bb"][i] = moving_average(
            history["delta_bb"][i], window=ma_window
        ).tolist()

    return agents, history
