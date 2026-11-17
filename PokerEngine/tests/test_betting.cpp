// Betting-round tests for the step() API that the Python training loop uses.
#include "policies.hpp"
#include "test_util.hpp"

static constexpr int kCap = 2000;  // MAX_STEPS_PER_HAND in ReinforcementLearningEnvironment/Poker.py

static int sumStacks(const GameEngine& g) {
    auto o = g.getObservation(0);
    int s = 0;
    for (int x : o.stack) s += x;
    return s;
}

// The seat that closes the action folds preflop; the others then check the flop.
// The flop must close after one check each (it never closed before the fix).
static void closingSeatFolds() {
    GameEngine g(4, 200, 2, 1);
    g.seed(1);
    g.reset_tournament(200);
    g.start_hand();  // seat 0 big blind, seat 1 small blind, seat 2 first to act
    for (ActionType t : {ActionType::Call, ActionType::Call, ActionType::Check, ActionType::Fold})
        g.step({t}, g.currentPlayerSafe());
    CHECK(g.streetInt() == 1, "flop dealt after preflop, street %d", g.streetInt());
    int checks = 0;
    while (g.streetInt() == 1 && !g.hand_terminal() && checks < 20) { g.step({ActionType::Check}, g.currentPlayerSafe()); ++checks; }
    CHECK(checks == 3, "three live players close the flop in three checks, took %d", checks);
}

// Everyone calls or checks: each live seat acts exactly once per street.
static void everySeatActsOncePerStreet(int players) {
    GameEngine g(players, 200, 2, 1);
    g.seed(2);
    g.reset_tournament(200);
    for (int hand = 0; hand < players; ++hand) {  // blinds rotate, so every seat takes every position
        g.start_hand();
        std::vector<std::vector<int>> acts(4, std::vector<int>(players, 0));
        int steps = 0;
        while (!g.hand_terminal() && ++steps < kCap) {
            int p = g.currentPlayerSafe(), st = g.streetInt();
            auto o = g.getObservation(p);
            ++acts[st][p];
            g.step({allowed(o, ActionType::Call) ? ActionType::Call : ActionType::Check}, p);
        }
        CHECK(g.hand_terminal(), "%d players: hand ends", players);
        for (int st = 0; st < 4; ++st)
            for (int p = 0; p < players; ++p)
                CHECK(acts[st][p] == 1, "%d players: seat %d acted %d times on street %d", players, p, acts[st][p], st);
    }
}

// Random and fixed policy mixes, with stacks reset every hand and as a running tournament (as
// in training). Every hand must finish within the cap, the player to act must always have a
// legal action, and chips must be conserved.
static void stress(std::vector<Policy> seats, bool tournament, const char* name) {
    const int n = (int)seats.size(), stack = 10000, hands = 5000;
    GameEngine g(n, stack, 2, 1);
    g.seed(3);
    std::mt19937_64 rng(4);
    int stuck = 0, noLegal = 0, leaks = 0;
    for (int h = 0; h < hands; ++h) {
        if (!tournament || g.alivePlayers() < 2) g.reset_tournament(stack);
        g.start_hand();
        int steps = 0;
        while (!g.hand_terminal()) {
            if (++steps > kCap) { ++stuck; g.reset_tournament(stack); break; }
            int p = g.currentPlayerSafe();
            auto o = g.getObservation(p);
            bool any = false;
            for (auto& s : o.legal) any = any || s.allowed;
            if (!any) { ++noLegal; g.reset_tournament(stack); break; }
            stepWithFallback(g, p, o, choose(seats[p], o, rng));
        }
        if (sumStacks(g) != n * stack) { ++leaks; g.reset_tournament(stack); }
    }
    CHECK(stuck == 0, "%s%s: %d of %d hands did not finish within %d steps", name, tournament ? " (tournament)" : "", stuck, hands, kCap);
    CHECK(noLegal == 0, "%s%s: %d hands reached a player with no legal action", name, tournament ? " (tournament)" : "", noLegal);
    CHECK(leaks == 0, "%s%s: %d hands did not conserve chips", name, tournament ? " (tournament)" : "", leaks);
}

int main() {
    closingSeatFolds();
    everySeatActsOncePerStreet(4);
    everySeatActsOncePerStreet(2);
    using P = Policy;
    for (bool t : {false, true}) {
        stress({P::Random, P::Random, P::Random, P::Random}, t, "random x4");
        stress({P::Random, P::Call, P::Call, P::Call}, t, "random + 3 call");
        stress({P::Random, P::CheckFold, P::Call, P::Call}, t, "random + checkfold + 2 call");
        stress({P::Call, P::Call, P::Call, P::CheckFold}, t, "3 call + checkfold");
        stress({P::AllIn, P::Call, P::Random, P::CheckFold}, t, "allin + call + random + checkfold");
        stress({P::Random, P::Call}, t, "heads-up random vs call");
    }
    return finish("test_betting");
}
