

CSV_PATH = "poker_hand_stats.csv"

CSV_FIELDS = [
    "run_id",
    "hand_id",
    "player_id",
    "pocket",                 # e.g. AhKd (sorted)
    "street_end",             # 0..3
    "street_end_name",        # preflop/flop/turn/river
    "bb_delta",
    "profitable",             # int(bb_delta > 0)
    "folded",
    "went_showdown",
    "non_showdown_win",       # int((not went_showdown) and bb_delta > 0)
    "first_action",
    "first_action_name",
    "bb_raised",
    "raises_made",
    "raises_faced",
    "aggressive",             # int(raises_made > 0)
]



MAX_MEMORY = 1_000_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3


# Discrete action space
A_FOLD=0; A_CHECK=1; A_CALL=2; A_RAISE_BB=3; A_RAISE_HALF=4; A_RAISE_POT=5; A_ALLIN=6
N_ACTIONS = 7

import uuid
RUN_ID = str(uuid.uuid4())[:8]

STREET_NAME = {0:"preflop", 1:"flop", 2:"turn", 3:"river"}
ACTION_NAME = {
    0:"fold", 1:"check", 2:"call",
    3:"raise_bb", 4:"raise_half", 5:"raise_pot", 6:"allin"
}


RANK_ORDER = "23456789TJQKA"
RANK_TO_I = {r:i for i,r in enumerate(RANK_ORDER)}

RANKS = 13
SUITS = 4

rank_map = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "T": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14,
}