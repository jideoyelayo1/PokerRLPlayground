"""Checks for the opponent suite and the diagnostics (bench/winrate.py, bench/eval_suite.py). From the
repository root (extension built):

    python -m unittest discover -s tests -v
"""
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import numpy as np

import eval_suite as es
import winrate as w
from ReinforcementLearningEnvironment.TrainingEnvironmentVariables import (
    A_ALLIN, A_CALL, A_CHECK, A_FOLD, A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT)


def card(rank, suit=1):
    return NS(rank=NS(value=rank), suit=NS(value=suit))


def obs(hole, board=(), street=0, to_call=2, pot=3):
    return NS(hole=[card(*h) for h in hole], board=[card(*b) for b in board], street=street, viewer_to_call=to_call, pot_total=pot)


ALL = np.zeros(7, np.float32)


class Unchanged(unittest.TestCase):
    """evaluate() without stats returns what it did before the suite was added (values taken from
    the code at 57b741f), and the pre-registered bot set is the same."""

    def test_bots_unchanged(self):
        self.assertEqual(list(w.BOTS), ["random", "call", "tight"])
        self.assertEqual(list(w.SUITE), ["random", "call", "tight", "tag", "lag", "potodds", "bluff"])

    def test_evaluate_matches_the_old_values(self):
        for a, b, stack, bb100, lo, hi, sd in [
                ("call", "tight", 200, -8.333333333333332, -51.414780745312314, 34.748114078645656, 0.8512945324342104),
                ("random", "tight", 10000, -33367.5, -102940.92571993035, 36205.92571993035, 1374.7791793463725),
                ("tight", "call", 200, 30.0, -15.226356622364944, 75.22635662236495, 0.8936781939186579)]:
            r = w.evaluate(w.BOTS[a], w.BOTS[b], 15, 7, stack=stack)
            self.assertEqual((r["bb100"], r["ci95_lo"], r["ci95_hi"], r["sd_bb_per_deal"], r["deals"], r["hands"]), (bb100, lo, hi, sd, 15, 60))
            self.assertNotIn("stats", r)

    def test_stats_do_not_change_the_play(self):
        for bot in ("tight", "bluff"):  # bluff draws from the shared rng
            a = w.evaluate(w.bot_tag, w.SUITE[bot], 30, 3, stack=200)
            b = w.evaluate(w.bot_tag, w.SUITE[bot], 30, 3, stack=200, stats=es.Stats())
            self.assertEqual(a, b)


class Counters(unittest.TestCase):
    def run_cell(self, policy, bot, deals=10):
        st = es.Stats()
        w.evaluate(policy, bot, deals, 11, stack=200, stats=st)
        return st.summary()

    def test_everyone_limps_to_showdown(self):
        s = self.run_cell(w.bot_call, w.bot_call)
        self.assertEqual((s["hands"], s["decisions"]), (40, 160))
        self.assertEqual((s["vpip"], s["pfr"], s["showdown"]), (0.75, 0.0, 1.0))  # the big blind's check is not voluntary
        self.assertEqual((s["aggression"], s["three_bet_chances"], s["illegal"], s["free_folds"], s["hands_with_all_in"]), (0.0, 0, 0, 0, 0.0))
        self.assertAlmostEqual(s["bb100"], 0.0)
        self.assertEqual(s["pot_bb"]["bins"][1], 40)  # a 4 bb pot, in the bin (3, 6]
        self.assertEqual([s["by_position"][p]["hands"] for p in es.POSITIONS], [10] * 4)
        self.assertEqual(es.flags(s), [])

    def test_always_fold(self):
        s = self.run_cell(lambda o, m, r: A_FOLD, w.bot_call)
        self.assertEqual((s["decisions"], s["vpip"], s["fold_to_bet"], s["facing_bet"], s["showdown"]), (40, 0.0, 1.0, 30, 0.0))
        self.assertEqual(s["free_folds"], 10)  # the big blind, when everyone limped
        self.assertAlmostEqual(s["bb100"], -37.5)  # -1 as the big blind, -0.5 as the small blind, 0 otherwise
        self.assertEqual(s["by_position"]["BB"]["bb100"], -100.0)
        self.assertEqual(es.flags(s), ["free_folds", "vpip_low", "fold_to_bet"])

    def test_always_all_in(self):
        s = self.run_cell(lambda o, m, r: A_ALLIN, w.bot_call)
        self.assertEqual((s["decisions"], s["all_in_rate"], s["hands_with_all_in"], s["hands_2plus_all_in"]), (40, 1.0, 1.0, 0))
        self.assertEqual((s["vpip"], s["pfr"], s["showdown"], s["illegal"]), (1.0, 1.0, 1.0, 0))
        self.assertEqual(es.flags(s), ["all_in_rate", "vpip_high"])

    def test_three_bet_aggression_position_and_bucket(self):
        st = es.Stats()
        st.agent = 0
        ok, bad = ALL, np.array([0, -1e9, 0, 0, 0, 0, 0], np.float32)
        pre = obs([(14, 1), (13, 1)], street=0)
        for seat, o, a, m in [(3, pre, A_RAISE_HALF, ok),              # open raise: not facing a raise yet
                              (0, pre, A_RAISE_POT, ok),               # the agent re-raises: a 3-bet
                              (1, pre, A_FOLD, ok), (2, pre, A_FOLD, ok),
                              (3, pre, A_CALL, ok),
                              (0, obs([(14, 1), (13, 1)], street=1, to_call=0), A_RAISE_BB, ok),   # flop bet
                              (3, obs([(2, 1), (2, 2)], street=1), A_CALL, ok),
                              (0, obs([(14, 1), (13, 1)], street=2, to_call=0), A_CHECK, ok),
                              (3, obs([(2, 1), (2, 2)], street=2, to_call=0), A_RAISE_BB, ok),
                              (0, obs([(14, 1), (13, 1)], street=2), A_CALL, ok),
                              (0, obs([(14, 1), (13, 1)], street=3), A_CHECK, bad)]:   # a check facing a bet: illegal
            st.decision(seat, o, m, a)
        hole = [[card(2, 1), card(7, 2)]] * 4
        hole[0] = [card(14, 1), card(13, 1)]
        st.hand([5.0, -1.0, -1.0, -3.0], hole, NS(button=1, pot_total=24, board=[card(r) for r in (2, 5, 9, 10, 12)]))
        s = st.summary()
        self.assertEqual((s["three_bet"], s["three_bet_chances"], s["pfr"], s["vpip"], s["showdown"]), (1.0, 1, 1.0, 1.0, 1.0))
        self.assertEqual((s["aggression"], s["illegal"], s["decisions"], s["free_folds"]), (2.0, 1, 5, 0))
        self.assertEqual(s["facing_bet"], 3)  # preflop raise, turn call, river (the faked bet)
        self.assertEqual(s["by_position"]["UTG"], {"hands": 1, "bb100": 500.0})  # seat 0 with the button on seat 1
        self.assertEqual(s["by_hand"], {"AKs": {"hands": 1, "bb100": 500.0}})
        self.assertEqual(s["pot_bb"]["mean"], 12.0 * 1)  # 24 chips at the default big blind of 2
        self.assertEqual(s["pot_bb"]["bins"][2], 1)

    def test_a_fold_ends_the_showdown(self):
        st = es.Stats()
        st.agent = 2
        st.decision(2, obs([(9, 1), (4, 1)], to_call=2), ALL, A_FOLD)
        st.hand([1.0, 1.0, -1.0, -1.0], [[card(9), card(4)]] * 4, NS(button=0, pot_total=6, board=[card(r) for r in (2, 5, 9, 10, 12)]))
        s = st.summary()
        self.assertEqual((s["showdown"], s["fold_to_bet"], s["by_position"]["BB"]["hands"]), (0.0, 1.0, 1))


class Suite(unittest.TestCase):
    def test_every_bot_plays_legal_actions(self):
        for name, bot in w.SUITE.items():
            for opp in (w.bot_random, bot):
                st = es.Stats()
                w.evaluate(bot, opp, 100, 17, stack=200, stats=st)  # 400 hands, the bot in every seat
                s = st.summary()
                self.assertEqual(s["illegal"], 0, name)
                self.assertGreater(s["decisions"], 300, name)

    def test_rules_on_chosen_hands(self):
        free, facing = ALL, np.array([0, -1e9, 0, 0, 0, 0, 0], np.float32)
        aa, k7 = [(14, 1), (14, 2)], [(13, 1), (7, 2)]
        self.assertEqual(w.bot_tag(obs(aa), free, None), A_RAISE_POT)
        self.assertEqual(w.bot_tag(obs(k7), facing, None), A_FOLD)
        self.assertEqual(w.bot_tag(obs(k7, to_call=0), free, None), A_CHECK)
        self.assertEqual(w.bot_tag(obs([(9, 1), (9, 2)], to_call=6), free, None), A_CALL)  # in range, but a raise to face
        # top pair: raises half the pot; a pair under the board: calls a small bet only
        self.assertEqual(w.bot_tag(obs([(14, 1), (7, 2)], [(14, 3), (9, 1), (3, 2)], street=1, to_call=0, pot=10), free, None), A_RAISE_HALF)
        self.assertEqual(w.bot_tag(obs([(7, 1), (4, 2)], [(14, 3), (7, 3), (3, 2)], street=1, to_call=4, pot=10), free, None), A_CALL)
        self.assertEqual(w.bot_tag(obs([(7, 1), (4, 2)], [(14, 3), (7, 3), (3, 2)], street=1, to_call=8, pot=10), facing, None), A_FOLD)
        self.assertEqual(w.bot_lag(obs(k7), free, None), A_CHECK)  # K7o is outside its range
        self.assertEqual(w.bot_lag(obs([(14, 1), (5, 2)]), free, None), A_RAISE_HALF)  # any ace is inside
        self.assertEqual(w.bot_lag(obs([(8, 1), (3, 2)], to_call=0), free, None), A_CHECK)
        self.assertEqual(w.bot_lag(obs([(8, 1), (3, 2)], [(2, 3), (9, 1), (12, 2)], street=1, to_call=0, pot=10), free, None), A_RAISE_HALF)  # c-bet with air
        # pot odds: the proxy beats 10/110 with 72o, not 100/200 (AA does)
        self.assertEqual(w.bot_potodds(obs([(7, 1), (2, 2)], to_call=10, pot=100), free, None), A_CALL)
        self.assertEqual(w.bot_potodds(obs([(7, 1), (2, 2)], to_call=100, pot=100), facing, None), A_FOLD)
        self.assertEqual(w.bot_potodds(obs(aa, to_call=100, pot=100), free, None), A_CALL)
        self.assertEqual(w.bot_potodds(obs([(7, 1), (2, 2)], to_call=0), free, None), A_CHECK)

    def test_bluff_frequency(self):
        rng = random.Random(1)
        picks = [w.bot_bluff(None, ALL, rng) for _ in range(4000)]
        raised = sum(a in (A_RAISE_BB, A_RAISE_HALF, A_RAISE_POT) for a in picks) / len(picks)
        self.assertAlmostEqual(raised, w.BLUFF_FREQ, delta=0.03)
        self.assertEqual(set(picks) - {A_RAISE_HALF, A_CALL}, set())  # raises half the pot, else calls (this mask: call legal)

    def test_hand_levels(self):
        lvl = lambda hole, board: w._level(obs(hole, board, street=1))
        self.assertEqual(lvl([(14, 1), (13, 2)], [(9, 1), (4, 2), (2, 3)]), 0)
        self.assertEqual(lvl([(9, 1), (13, 2)], [(10, 1), (9, 2), (2, 3)]), 1)   # a pair below the top board card
        self.assertEqual(lvl([(10, 1), (13, 2)], [(12, 3), (10, 2), (2, 3)]), 1)
        self.assertEqual(lvl([(14, 1), (13, 2)], [(14, 3), (9, 2), (2, 3)]), 2)  # top pair
        self.assertEqual(lvl([(12, 1), (12, 2)], [(10, 3), (9, 2), (2, 3)]), 2)  # an overpair
        self.assertEqual(lvl([(9, 1), (13, 2)], [(13, 3), (9, 2), (2, 3)]), 3)   # two pair
        self.assertEqual(lvl([(9, 1), (9, 2)], [(9, 3), (4, 2), (2, 3)]), 3)     # a set
        self.assertEqual(lvl([(14, 1), (13, 2)], [(9, 3), (9, 2), (2, 3)]), 0)   # the board's pair alone
        self.assertEqual(lvl([(14, 1), (4, 2)], [(9, 3), (9, 2), (4, 3)]), 3)    # a pair and the board's pair


class Tournament(unittest.TestCase):
    def test_sessions_carry_stacks_and_end(self):
        r = es.tournament(w.bot_call, w.bot_random, 4, 5, stack=200, max_hands=30)
        self.assertEqual(sum(r["ended"].values()), 4)
        self.assertLessEqual(r["hands"], 4 * 30)
        self.assertTrue(math.isfinite(r["mean_bb_per_hand"]))
        # calling stations against one another cannot bust: every session runs to the cap
        r = es.tournament(w.bot_call, w.bot_call, 2, 5, stack=200, max_hands=10)
        self.assertEqual((r["ended"]["cap"], r["hands"]), (2, 20))


if __name__ == "__main__":
    unittest.main()
