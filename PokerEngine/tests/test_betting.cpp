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

static void callOrCheck(GameEngine& g) {
    int p = g.currentPlayerSafe();
    g.step({allowed(g.getObservation(p), ActionType::Call) ? ActionType::Call : ActionType::Check}, p);
}

static std::string str(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : " ") + std::to_string(x);
    return "[" + s + "]";
}

// The live seats in table order, starting with seat `from`.
static std::vector<int> roundFrom(const std::vector<int>& live, int from) {
    auto it = std::find(live.begin(), live.end(), from);
    std::vector<int> v(it, live.end());
    v.insert(v.end(), live.begin(), it);
    return v;
}

// The seat that closes the action (the big blind) folds preflop; the others then check the flop.
// The flop must close after one check each (it never closed before the fix).
static void closingSeatFolds() {
    GameEngine g(4, 200, 2, 1);
    g.seed(1);
    g.reset_tournament(200);
    g.start_hand();  // button seat 0, small blind 1, big blind 2, seat 3 first to act
    for (int i = 0; i < 3; ++i) callOrCheck(g);
    g.step({ActionType::Fold}, g.currentPlayerSafe());
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

// Standard hold'em order, everyone calling or checking, over a full turn of the button and one
// more hand: the button starts on seat 0 and moves one seat per hand; the two seats after it post
// the small and big blind (heads-up the button posts the small blind); preflop the seat after the
// big blind acts first and the big blind last; on later streets the first seat after the button
// acts first and the button last.
static void standardOrder(int n) {
    GameEngine g(n, 200, 2, 1);
    g.seed(5);
    g.reset_tournament(200);
    std::vector<int> live(n);
    for (int i = 0; i < n; ++i) live[i] = i;
    for (int hand = 0; hand <= n; ++hand) {
        int button = hand % n, sb = n == 2 ? button : (button + 1) % n, bb = (sb + 1) % n;
        g.start_hand();
        auto o = g.getObservation(0);
        std::vector<int> posted;
        for (int s = 0; s < n; ++s) posted.push_back(o.street_contrib[s]);
        std::vector<int> want(n, 0);
        want[sb] = 1; want[bb] = 2;
        CHECK(posted == want, "%d seats, hand %d: blinds posted %s, expected %s (button %d)", n, hand, str(posted).c_str(), str(want).c_str(), button);
        std::vector<std::vector<int>> acts(4);
        for (int steps = 0; !g.hand_terminal() && steps < kCap; ++steps) {
            acts[g.streetInt()].push_back(g.currentPlayerSafe());
            callOrCheck(g);
        }
        CHECK(acts[0] == roundFrom(live, (bb + 1) % n), "%d seats, hand %d: preflop order %s, expected %s",
              n, hand, str(acts[0]).c_str(), str(roundFrom(live, (bb + 1) % n)).c_str());
        for (int st = 1; st < 4; ++st)
            CHECK(acts[st] == roundFrom(live, (button + 1) % n), "%d seats, hand %d: street %d order %s, expected %s",
                  n, hand, st, str(acts[st]).c_str(), str(roundFrom(live, (button + 1) % n)).c_str());
    }
}

// When everyone just calls, the big blind still acts preflop and may check or raise; a raise
// sends the action round the others again before the flop.
static void bigBlindOption(int n) {
    GameEngine g(n, 200, 2, 1);
    g.seed(6);
    g.reset_tournament(200);
    g.start_hand();
    const int bb = n == 2 ? 1 : 2;  // first hand: button on seat 0
    for (int i = 0; i < n - 1; ++i) callOrCheck(g);
    auto o = g.getObservation(bb);
    bool option = g.streetInt() == 0 && g.currentPlayerSafe() == bb;
    CHECK(option, "%d seats: after %d calls the big blind (seat %d) should act preflop, got seat %d on street %d",
          n, n - 1, bb, g.currentPlayerSafe(), g.streetInt());
    if (!option) return;
    CHECK(allowed(o, ActionType::Check) && allowed(o, ActionType::Raise), "%d seats: big blind may check or raise", n);
    g.step({ActionType::Raise, RaiseSize::BB}, bb);
    std::vector<int> after;
    while (g.streetInt() == 0 && !g.hand_terminal() && (int)after.size() < n) {
        after.push_back(g.currentPlayerSafe());
        callOrCheck(g);
    }
    std::vector<int> live(n);
    for (int i = 0; i < n; ++i) live[i] = i;
    std::vector<int> want = roundFrom(live, (bb + 1) % n);
    want.pop_back();  // everyone but the raiser
    CHECK(after == want && g.streetInt() == 1, "%d seats: after the big blind raises, %s act (expected %s), then the flop (street %d)",
          n, str(after).c_str(), str(want).c_str(), g.streetInt());
}

// Running tournaments with small stacks, so players bust and fold or go all-in in front of the
// button. At every hand the button must be the next seat still in after the last hand's button;
// the blinds and the first player preflop follow from it; on every later street the first seat
// after the button that can still act acts first. The test counts the situations it exists for
// (a blind or the button busting, going heads-up, the first seat after the button having folded
// or gone all-in) and fails if one never happens, so a change of seed cannot hollow it out.
static void rotationAndSkips(std::vector<Policy> seats, const char* name) {
    const int n = (int)seats.size(), stack = 40, hands = 5000;
    GameEngine g(n, stack, 2, 1);
    g.seed(7);
    std::mt19937_64 rng(8);
    int button = -1, sb = -1, bb = -1, prevLive = 0;
    int badBlinds = 0, badPreflop = 0, badLater = 0, streets = 0;
    int buttonBust = 0, sbBust = 0, bbBust = 0, headsUp = 0, skipFold = 0, skipAllIn = 0;
    for (int h = 0; h < hands; ++h) {
        if (h == 0 || g.alivePlayers() < 2) { g.reset_tournament(stack); button = -1; }
        std::vector<int> live = g.aliveIds();
        auto has = [&](int s) { return std::find(live.begin(), live.end(), s) != live.end(); };
        if (button >= 0) {
            buttonBust += !has(button); sbBust += !has(sb); bbBust += !has(bb);
            headsUp += prevLive > 2 && live.size() == 2;
        }
        prevLive = (int)live.size();
        auto before = g.getObservation(0).stack;
        auto next = std::find_if(live.begin(), live.end(), [&](int s) { return s > button; });
        button = next == live.end() ? live[0] : *next;
        auto order = roundFrom(live, button);  // button first
        sb = live.size() == 2 ? button : order[1];
        bb = roundFrom(live, sb)[1];

        g.start_hand();
        auto o = g.getObservation(0);
        bool ok = true;
        for (int s : live)
            ok = ok && o.total_contrib[s] == (s == sb ? std::min(1, before[s]) : s == bb ? std::min(2, before[s]) : 0);
        badBlinds += !ok;

        auto firstLive = [&](const std::vector<int>& from, const Observation& ob) {
            for (int s : from) if (!ob.folded[s] && !ob.all_in[s]) return s;
            return -1;
        };
        if (!g.hand_terminal()) badPreflop += g.currentPlayerSafe() != firstLive(roundFrom(live, roundFrom(live, bb)[1]), o);

        int street = 0, steps = 0;
        while (!g.hand_terminal()) {
            if (++steps > kCap) { g.reset_tournament(stack); button = -1; break; }
            int p = g.currentPlayerSafe();
            auto ob = g.getObservation(p);
            if (g.streetInt() != street) {
                street = g.streetInt();
                ++streets;
                std::vector<int> after(order.begin() + 1, order.end());
                after.push_back(button);
                badLater += p != firstLive(after, ob);
                skipFold += ob.folded[after[0]];
                skipAllIn += ob.all_in[after[0]];
            }
            stepWithFallback(g, p, ob, choose(seats[p], ob, rng));
        }
    }
    CHECK(badBlinds == 0, "%s: %d of %d hands had the blinds on the wrong seats", name, badBlinds, hands);
    CHECK(badPreflop == 0, "%s: %d of %d hands started preflop with the wrong player", name, badPreflop, hands);
    CHECK(badLater == 0, "%s: %d of %d flop/turn/river rounds started with the wrong player", name, badLater, streets);
    CHECK(buttonBust > 0 && sbBust > 0 && bbBust > 0 && headsUp > 0 && skipFold > 0 && skipAllIn > 0,
          "%s: cases covered: button busted %d, small blind busted %d, big blind busted %d, went heads-up %d, "
          "first seat after the button folded %d / all-in %d (each must be > 0)", name, buttonBust, sbBust, bbBust, headsUp, skipFold, skipAllIn);
    std::printf("%s: button busted %d, small blind busted %d, big blind busted %d, went heads-up %d, "
                "first seat after the button folded %d / all-in %d times\n", name, buttonBust, sbBust, bbBust, headsUp, skipFold, skipAllIn);
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
    for (int n : {2, 3, 4, 6}) standardOrder(n);
    for (int n : {2, 3, 4, 6}) bigBlindOption(n);
    using P = Policy;
    rotationAndSkips({P::Random, P::AllIn, P::Random, P::Call}, "rotation, 4 seats");
    rotationAndSkips({P::Random, P::AllIn, P::Call, P::CheckFold, P::Random, P::Call}, "rotation, 6 seats");
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
