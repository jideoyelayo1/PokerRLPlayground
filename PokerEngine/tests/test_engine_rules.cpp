// Rule edge cases pinned with fixed deals and stacks, and randomized invariants against an
// independent payout reference (#19). Every fixture is played through step(), like the training
// loop. Each case says whether the engine follows standard no-limit hold'em or deliberately
// simplifies it (the README's rules text).
//
// The engine has no public way to set stacks or the deck (stacks are uniform, the deck is
// shuffled), so this file opens the class with `#define private public` after the standard
// headers, sets stacks and button_ directly and loads deck_ in dealing order. Only setup uses it;
// everything that follows goes through start_hand() and step().
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define private public
#include "policies.hpp"
#include "test_util.hpp"
#undef private

using Ints = std::vector<int>;

static std::string str(const Ints& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : " ") + std::to_string(x);
    return "[" + s + "]";
}

static int key(const Card& c) { return static_cast<int>(c.suit) * 16 + static_cast<int>(c.rank); }

// A table with these stacks, one per seat. The button for the first hand is `button`.
static std::unique_ptr<GameEngine> table(const Ints& stacks, int button = 0, int bb = 2, int sb = 1) {
    int n = (int)stacks.size();
    auto g = std::make_unique<GameEngine>(n, 100, bb, sb);
    for (int i = 0; i < n; ++i) g->players_[i].adjustMoney(stacks[i] - g->players_[i].getMoney());
    g->button_ = button - 1;  // postBlinds moves it to the next seat
    return g;
}

// The next hand's deal: `holes` are two cards for each live seat in seat order, then the board.
// Burn cards and the rest of the deck are the unused cards, in a fixed order.
static void rig(GameEngine& g, const std::string& holes, const std::string& board) {
    std::vector<Card> deal = cards(holes), b = cards(board), rest;
    std::set<int> used;
    for (auto& c : deal) used.insert(key(c));
    for (auto& c : b) used.insert(key(c));
    CHECK(used.size() == deal.size() + b.size() && b.size() == 5, "rig: cards must be distinct, board of 5");
    for (int s = 1; s <= 4; ++s)
        for (int r = 2; r <= 14; ++r) {
            Card c(static_cast<Suits>(s), static_cast<Ranks>(r));
            if (!used.count(key(c))) rest.push_back(c);
        }
    size_t at = 0;
    deal.push_back(rest[at++]); deal.insert(deal.end(), b.begin(), b.begin() + 3);  // burn, flop
    deal.push_back(rest[at++]); deal.push_back(b[3]);                                // burn, turn
    deal.push_back(rest[at++]); deal.push_back(b[4]);                                // burn, river
    deal.insert(deal.end(), rest.begin() + at, rest.end());
    g.deck_ = std::queue<Card>();
    for (auto& c : deal) g.deck_.push(c);
    g.reshuffleAlways_ = false;
}

static const char* kBoard = "2c 7d 9h Tc 3s";  // no straight or flush for any hand used below

// The player to act must be `seat`; then play the action.
static void act(GameEngine& g, int seat, ActionType t, RaiseSize sz = RaiseSize::BB) {
    CHECK(g.currentPlayerSafe() == seat, "seat %d should act, seat %d does", seat, g.currentPlayerSafe());
    g.step({t, sz}, g.currentPlayerSafe());
}

static void checkDown(GameEngine& g) {
    for (int i = 0; i < 100 && !g.hand_terminal(); ++i) {
        int p = g.currentPlayerSafe();
        g.step({allowed(g.getObservation(p), ActionType::Call) ? ActionType::Call : ActionType::Check}, p);
    }
    CHECK(g.hand_terminal(), "hand should have ended");
}

static void expectStacks(GameEngine& g, const Ints& want, const char* what) {
    Ints got = g.getObservation(0).stack;
    got.resize(want.size());
    CHECK(got == want, "%s: stacks %s, expected %s", what, str(got).c_str(), str(want).c_str());
}

static void expectPots(GameEngine& g, const std::vector<PotResult>& want, const char* what) {
    const auto& got = g.potResults();
    bool ok = got.size() == want.size();
    for (size_t i = 0; ok && i < got.size(); ++i)
        ok = got[i].amount == want[i].amount && got[i].eligible == want[i].eligible &&
             got[i].winners == want[i].winners && got[i].shares == want[i].shares;
    if (!ok) {
        std::string s;
        for (auto& p : got) s += " {" + std::to_string(p.amount) + " elig " + str(p.eligible) + " win " + str(p.winners) + " shares " + str(p.shares) + "}";
        CHECK(false, "%s: pots were%s", what, s.c_str());
    } else CHECK(true, "ok");
}

// ---- Nested side pots ------------------------------------------------------------------------
// Standard: a pot is built for each depth of contribution; a player can win only the pots they
// paid into at full depth. Seats 0-3 hold AA, KK, QQ, JJ, so each pot goes to the best hand
// among its eligible seats, and the deepest stack gets back what nobody could match.
static void nestedSidePots4() {
    auto g = table({20, 50, 90, 140});  // button 0, small blind 1, big blind 2, seat 3 first
    rig(*g, "As Ah Ks Kh Qs Qh Js Jh", kBoard);
    g->start_hand();
    for (int s : {3, 0, 1, 2}) act(*g, s, ActionType::AllIn);
    CHECK(g->hand_terminal(), "all-in hand over");
    expectPots(*g, {{80, {0, 1, 2, 3}, {0}, {80}}, {90, {1, 2, 3}, {1}, {90}}, {80, {2, 3}, {2}, {80}}, {50, {3}, {3}, {50}}},
               "4 all-ins at 20/50/90/140");
    expectStacks(*g, {80, 90, 80, 50}, "4 all-ins at 20/50/90/140");
}

// Three all-ins at different depths and a folded player whose blind is dead money in the first
// pot (folded players pay into the pots they reach but cannot win them).
static void nestedSidePots3() {
    auto g = table({30, 70, 100, 150});
    rig(*g, "As Ah Ks Kh Qs Qh Js Jh", kBoard);
    g->start_hand();
    for (int s : {3, 0, 1}) act(*g, s, ActionType::AllIn);
    act(*g, 2, ActionType::Fold);  // the big blind, 2 chips in
    CHECK(g->hand_terminal(), "hand over");
    expectPots(*g, {{8, {0, 1, 3}, {0}, {8}}, {84, {0, 1, 3}, {0}, {84}}, {80, {1, 3}, {1}, {80}}, {80, {3}, {3}, {80}}},
               "3 all-ins at 30/70/150 plus a folded blind");
    expectStacks(*g, {92, 80, 98, 80}, "3 all-ins at 30/70/150 plus a folded blind");
}

// ---- Ties and the odd chip -------------------------------------------------------------------
// Deliberate simplification (README, Pots): the whole remainder of a split pot goes to the
// winner in the lowest-numbered seat. Standard hold'em gives odd chips one at a time starting
// with the winner nearest the left of the button. These fixtures put the button on seat 0, so
// standard play would give the chip to seat 2 (two-way) and one chip each to seats 2 and 3
// (three-way); the engine gives them to seat 0.
static void tiedPotOddChip() {
    auto g = table({100, 100, 100});
    rig(*g, "As Kh Qs Qh Ad Kc", kBoard);  // seats 0 and 2 tie on A K T 9 7
    g->start_hand();
    act(*g, 0, ActionType::Call);
    act(*g, 1, ActionType::Fold);  // the small blind's chip is dead: pot 5
    act(*g, 2, ActionType::Check);
    checkDown(*g);
    // pots by depth: 3 chips (1 from each seat) and 2 (from seats 0 and 2); the odd chip is in the first
    expectPots(*g, {{3, {0, 2}, {0, 2}, {2, 1}}, {2, {0, 2}, {0, 2}, {1, 1}}}, "two-way tie, odd chip");
    expectStacks(*g, {101, 99, 100}, "two-way tie, odd chip");
}

static void tiedPotRemainderTwo() {
    auto g = table({100, 100, 100, 100});
    rig(*g, "As Kh Qs Qh Ad Kc Ah Kd", kBoard);  // seats 0, 2, 3 tie; seat 1 holds a better pair but folds
    g->start_hand();
    for (int s : {3, 0}) act(*g, s, ActionType::Call);
    act(*g, 1, ActionType::Call);
    act(*g, 2, ActionType::Check);
    act(*g, 1, ActionType::Fold);  // pot 8 among three: 2 each, remainder 2
    checkDown(*g);
    expectPots(*g, {{8, {0, 2, 3}, {0, 2, 3}, {4, 2, 2}}}, "three-way tie, remainder 2");
    expectStacks(*g, {102, 98, 100, 100}, "three-way tie, remainder 2");
}

// A tie inside a side pot while the main pot has its own winner.
static void tiedSidePot() {
    auto g = table({20, 51, 51, 51});
    rig(*g, "As Ah Ks Kh Kd Kc Qs Qh", kBoard);
    g->start_hand();
    for (int s : {3, 0, 1, 2}) act(*g, s, ActionType::AllIn);
    expectPots(*g, {{80, {0, 1, 2, 3}, {0}, {80}}, {93, {1, 2, 3}, {1, 2}, {47, 46}}}, "tie in a side pot");
    expectStacks(*g, {80, 47, 46, 0}, "tie in a side pot");
    CHECK(g->alivePlayers() == 3, "the loser with no chips is out");
}

// ---- A short all-in raise --------------------------------------------------------------------
// Deliberate simplification (README, Limitations): an all-in for less than a full raise reopens
// the betting for players who have already acted, raise included. Standard no-limit lets them
// only call or fold. What is standard here: the short all-in raises the amount to call but not
// the minimum raise, so the next raise-to is the new amount plus the last full raise.
static void shortAllInReopens() {
    auto g = table({100, 100, 5});  // seat 2, the big blind, has 3 chips left after posting
    g->start_hand();
    act(*g, 0, ActionType::Raise);  // to 4: a full raise of 2
    act(*g, 1, ActionType::Call);
    CHECK(g->toCall() == 4 && g->minRaise() == 2, "after the raise: to call %d, min raise %d", g->toCall(), g->minRaise());
    act(*g, 2, ActionType::AllIn);  // to 5: a raise of 1, short of the minimum 2
    CHECK(g->toCall() == 5, "the short all-in sets the amount to call, %d", g->toCall());
    CHECK(g->minRaise() == 2, "a short all-in leaves the minimum raise at the last full raise, %d", g->minRaise());
    CHECK(g->currentPlayerSafe() == 0, "seat 0, who already acted, acts again, got seat %d", g->currentPlayerSafe());
    auto o = g->getObservation(0);
    CHECK(allowed(o, ActionType::Call) && allowed(o, ActionType::Raise),
          "DELIBERATE: seat 0 may re-raise a short all-in (standard no-limit allows only call or fold)");
    for (auto& s : o.legal)
        if (s.type == ActionType::Raise && s.raise_size == RaiseSize::BB)
            CHECK(s.allowed && s.raise_to == 7, "raise to 5 + 2 = 7, got %d", s.raise_to);
    act(*g, 0, ActionType::Raise);
    CHECK(g->toCall() == 7, "the re-raise went through, to call %d", g->toCall());
    CHECK(g->currentPlayerSafe() == 1, "seat 1 acts again after the re-raise");
}

// A full raise by an all-in (standard) also reopens, and moves the minimum raise.
static void fullAllInRaiseMovesMinimum() {
    auto g = table({100, 100, 12});
    g->start_hand();
    act(*g, 0, ActionType::Call);
    act(*g, 1, ActionType::Call);
    act(*g, 2, ActionType::AllIn);  // to 12: a raise of 10
    CHECK(g->toCall() == 12 && g->minRaise() == 10, "full all-in raise: to call %d, min raise %d", g->toCall(), g->minRaise());
    CHECK(g->currentPlayerSafe() == 0, "betting reopens at seat 0");
}

// ---- All-in blinds ---------------------------------------------------------------------------
// Standard: a blind larger than the stack posts what is left and is all-in. Deliberate
// simplification (README, Seats): the amount to call preflop is the larger of the blinds
// actually posted, not the full big blind (many rule sets keep the full big blind as the bet to
// match when the big blind is short; here the pots come out the same because nobody can win
// more than they matched).
static void allInBlinds() {
    {   // both blinds short: small blind 3 of 5, big blind 7 of 10
        auto g = table({100, 3, 7}, 0, 10, 5);
        rig(*g, "Qs Qh Ks Kh As Ah", kBoard);
        g->start_hand();
        auto o = g->getObservation(0);
        CHECK(o.all_in[1] && o.all_in[2] && o.pot_total == 10, "blinds all-in: pot %d", o.pot_total);
        CHECK(o.street_contrib[1] == 3 && o.street_contrib[2] == 7, "blinds posted %d and %d", o.street_contrib[1], o.street_contrib[2]);
        CHECK(g->toCall() == 7 && g->currentPlayerSafe() == 0, "to call 7 (DELIBERATE: not the full big blind 10), seat 0 acts");
        act(*g, 0, ActionType::Call);
        CHECK(g->hand_terminal(), "the call ends the betting");
        expectPots(*g, {{9, {0, 1, 2}, {2}, {9}}, {8, {0, 2}, {2}, {8}}}, "short blinds");
        expectStacks(*g, {93, 0, 17}, "short blinds");
        CHECK(g->alivePlayers() == 2, "the busted small blind is out");
    }
    {   // heads-up, both all-in from the blinds: showdown at once
        auto g = table({3, 7}, 0, 10, 5);
        rig(*g, "As Ah Ks Kh", kBoard);
        g->start_hand();
        CHECK(g->hand_terminal(), "both blinds all-in: no betting, hand over inside start_hand");
        expectPots(*g, {{6, {0, 1}, {0}, {6}}, {4, {1}, {1}, {4}}}, "heads-up blinds all-in");
        expectStacks(*g, {6, 4}, "heads-up blinds all-in");
    }
    {   // big blind short of the small blind: 4 against 5
        auto g = table({100, 100, 4}, 0, 10, 5);
        g->start_hand();
        CHECK(g->toCall() == 5, "DELIBERATE: to call is the small blind's 5, got %d", g->toCall());
        act(*g, 0, ActionType::Call);
        CHECK(g->getObservation(1).viewer_to_call == 0 && allowed(g->getObservation(1), ActionType::Check),
              "the small blind has matched and may check");
        act(*g, 1, ActionType::Check);
        CHECK(g->streetInt() == 1, "flop dealt");
    }
}

// ---- Heads-up after a bust, and the button after eliminations ----------------------------------
// Standard heads-up: the button posts the small blind, acts first preflop and last afterwards.
// README (Busted players): the button goes to the first live seat after the previous button's
// seat, whether or not that player busted: no dead button, no dead blinds. Deliberate
// simplification: tournament rules move the button so nobody posts the big blind twice.
static void headsUpAfterBust() {
    auto g = table({100, 100, 20});
    rig(*g, "As Ah Qs Qh Ks Kh", kBoard);
    g->start_hand();
    act(*g, 0, ActionType::AllIn);
    act(*g, 1, ActionType::Fold);
    act(*g, 2, ActionType::Call);
    expectPots(*g, {{3, {0, 2}, {0}, {3}}, {38, {0, 2}, {0}, {38}}, {80, {0}, {0}, {80}}}, "bust: pots");
    expectStacks(*g, {121, 99, 0}, "bust");
    CHECK(g->alivePlayers() == 2, "seat 2 is out");

    rig(*g, "As Ah Ks Kh", kBoard);
    g->start_hand();
    auto o = g->getObservation(0);
    CHECK(o.button == 1 && o.street_contrib[1] == 1 && o.street_contrib[0] == 2,
          "heads-up: button %d (seat 1) posts the small blind 1, the other seat the big blind 2 (posted %d, %d)",
          o.button, o.street_contrib[1], o.street_contrib[0]);
    std::vector<std::pair<int, int>> order;  // (street, seat)
    while (!g->hand_terminal()) {
        int p = g->currentPlayerSafe(), st = g->streetInt();
        order.push_back({st, p});
        g->step({allowed(g->getObservation(p), ActionType::Call) ? ActionType::Call : ActionType::Check}, p);
    }
    std::vector<std::pair<int, int>> want = {{0, 1}, {0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}};
    CHECK(order == want, "heads-up order: the button acts first preflop and last on every later street (got %zu actions)", order.size());
    expectStacks(*g, {123, 97, 0}, "heads-up showdown");

    g->start_hand();  // the button moves on to seat 0: wraps past the last seat still in
    o = g->getObservation(0);
    CHECK(o.button == 0 && o.street_contrib[0] == 1 && o.street_contrib[1] == 2 && g->currentPlayerSafe() == 0,
          "next hand: button seat %d posts the small blind and acts first", o.button);
}

static void buttonAfterEliminations() {
    auto g = table({100, 10, 100, 100}, 1);
    rig(*g, "As Ah Ks Kh Qs Qh Js Jh", kBoard);
    g->start_hand();  // button 1 (10 chips), small blind 2, big blind 3, seat 0 first
    CHECK(g->getObservation(0).button == 1, "first hand: button 1");
    act(*g, 0, ActionType::AllIn);
    act(*g, 1, ActionType::Call);
    act(*g, 2, ActionType::Fold);
    act(*g, 3, ActionType::Fold);
    expectStacks(*g, {113, 0, 99, 98}, "the button busts");
    CHECK(g->alivePlayers() == 3, "the button is out");

    // The button was seat 1 and has busted: the next button is the first live seat after it, seat 2.
    g->players_[2].adjustMoney(10 - g->players_[2].getMoney());
    rig(*g, "As Ah Ks Kh Qs Qh", kBoard);  // seats 0, 2, 3
    g->start_hand();
    auto o = g->getObservation(0);
    CHECK(o.button == 2 && o.street_contrib[3] == 1 && o.street_contrib[0] == 2 && g->currentPlayerSafe() == 2,
          "button %d, expected 2; small blind seat 3, big blind seat 0, seat 2 first", o.button);
    act(*g, 2, ActionType::AllIn);
    act(*g, 3, ActionType::Fold);
    act(*g, 0, ActionType::Call);
    expectStacks(*g, {124, 0, 0, 97}, "the button busts again");

    // Two seats left, the button was seat 2: the first live seat after it is seat 3, which is
    // the small blind and acts first heads-up.
    rig(*g, "As Ah Ks Kh", kBoard);
    g->start_hand();
    o = g->getObservation(0);
    CHECK(o.button == 3 && o.street_contrib[3] == 1 && o.street_contrib[0] == 2 && g->currentPlayerSafe() == 3,
          "heads-up after two busts: button %d, expected 3, acting first", o.button);
    act(*g, 3, ActionType::Fold);
    CHECK(g->hand_terminal() && g->winner_id() == 0, "the big blind wins by the fold");

    // Past the last seat: the button wraps to the first live seat.
    g->start_hand();
    o = g->getObservation(0);
    CHECK(o.button == 0 && o.street_contrib[0] == 1 && o.street_contrib[3] == 2, "the button wraps to seat %d", o.button);
}

// ---- Randomized invariants ---------------------------------------------------------------------
// Reference payout, written from the rules and the public hand evaluator only: pots from total
// contributions, winners by best 7-card hand among the unfolded seats that paid in at that depth,
// ties split evenly with the whole remainder to the lowest seat.
struct RefPot { int amount; Ints eligible, winners, shares; };

static HandRank best7(const std::vector<Card>& seven) {
    HandRank best{};
    bool has = false;
    for (int a = 0; a < 7; ++a)
        for (int b = a + 1; b < 7; ++b) {
            std::array<Card, 5> h{};
            int k = 0;
            for (int i = 0; i < 7; ++i) if (i != a && i != b) h[k++] = seven[i];
            HandRank r = evaluate5(h);
            if (!has || best < r) { best = r; has = true; }
        }
    return best;
}

static std::vector<RefPot> referencePots(GameEngine& g, int* orphan) {
    auto o = g.getObservation(0);
    int n = (int)o.stack.size();
    auto board = g.boardCardsFixed();
    std::vector<HandRank> rank(n);
    for (int s = 0; s < n; ++s) {
        if (o.folded[s]) continue;
        auto hole = g.getObservation(s).hole;
        rank[s] = best7({hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]});
    }
    std::set<int> levels;
    for (int s = 0; s < n; ++s) if (o.total_contrib[s] > 0) levels.insert(o.total_contrib[s]);
    std::vector<RefPot> pots;
    int prev = 0;
    for (int lvl : levels) {
        RefPot p{0, {}, {}, {}};
        for (int s = 0; s < n; ++s) {
            if (o.total_contrib[s] < lvl) continue;
            p.amount += lvl - prev;
            if (!o.folded[s]) p.eligible.push_back(s);
        }
        prev = lvl;
        if (p.eligible.empty()) { ++*orphan; continue; }  // would mean chips nobody can win
        p.winners = {p.eligible[0]};
        for (size_t i = 1; i < p.eligible.size(); ++i) {
            const HandRank& r = rank[p.eligible[i]];
            if (rank[p.winners[0]] < r) p.winners = {p.eligible[i]};
            else if (!(r < rank[p.winners[0]])) p.winners.push_back(p.eligible[i]);
        }
        int k = (int)p.winners.size();
        p.shares.assign(k, p.amount / k);
        p.shares[0] += p.amount % k;
        pots.push_back(p);
    }
    return pots;
}

static void randomInvariants(int hands, uint64_t seed) {
    std::mt19937_64 rng(seed);
    int stepsTotal = 0, leaks = 0, negative = 0, potMismatch = 0, wrongStacks = 0, wrongPots = 0, orphans = 0, stuck = 0;
    int foldWins = 0, showdowns = 0, multiPot = 0, threePots = 0, ties = 0, oddChips = 0, blindAllIn = 0, headsUp = 0, shortStack = 0;
    const int blinds[3][2] = {{2, 1}, {10, 5}, {4, 2}};
    for (int h = 0; h < hands; ++h) {
        int n = 2 + (int)(rng() % 5);
        auto& b = blinds[rng() % 3];
        Ints stacks(n);
        int total = 0;
        for (int& s : stacks) { s = rng() % 3 == 0 ? 1 + (int)(rng() % 12) : 15 + (int)(rng() % 186); total += s; }
        auto g = table(stacks, (int)(rng() % n), b[0], b[1]);
        g->seed(rng());
        g->start_hand();
        headsUp += n == 2;
        {
            auto o = g->getObservation(0);
            for (int s = 0; s < n; ++s) blindAllIn += o.all_in[s] && o.street_contrib[s] > 0;
        }
        int steps = 0;
        auto invariant = [&](const Observation& o) {
            int sum = 0, contrib = 0;
            for (int s = 0; s < n; ++s) { sum += o.stack[s]; contrib += o.total_contrib[s]; negative += o.stack[s] < 0; }
            leaks += sum + (o.terminal ? 0 : o.pot_total) != total;  // pot_total keeps its value after the payout
            potMismatch += contrib != o.pot_total;
        };
        invariant(g->getObservation(0));
        while (!g->hand_terminal() && steps < 2000) {
            int p = g->currentPlayerSafe();
            auto o = g->getObservation(p);
            g->step(choose(Policy::Random, o, rng), p);
            invariant(g->getObservation(0));
            ++steps;
        }
        stepsTotal += steps;
        if (!g->hand_terminal()) { ++stuck; continue; }

        auto o = g->getObservation(0);
        int sum = 0;
        for (int s : o.stack) sum += s;
        leaks += sum != total;
        const auto& got = g->potResults();
        int live = 0;
        for (int s = 0; s < n; ++s) live += !o.folded[s];
        for (int s = 0; s < n; ++s) shortStack += stacks[s] <= 12;
        if (live == 1) {
            ++foldWins;
            // the last player in takes the pot (every other seat folded, so nothing else is eligible)
            int w = (int)(std::find(o.folded.begin(), o.folded.end(), 0) - o.folded.begin());
            wrongStacks += o.stack[w] != stacks[w] - o.total_contrib[w] + o.pot_total;
            for (int s = 0; s < n; ++s) if (s != w) wrongStacks += o.stack[s] != stacks[s] - o.total_contrib[s];
            continue;
        }
        ++showdowns;
        int orphanBefore = orphans;
        auto ref = referencePots(*g, &orphans);
        if (orphans != orphanBefore) continue;  // reported below; no reference to compare
        multiPot += ref.size() > 1;
        threePots += ref.size() > 2;
        Ints win(n, 0);
        bool same = ref.size() == got.size();
        for (size_t i = 0; i < ref.size(); ++i) {
            ties += ref[i].winners.size() > 1;
            oddChips += ref[i].amount % ref[i].winners.size() != 0;
            for (size_t k = 0; k < ref[i].winners.size(); ++k) win[ref[i].winners[k]] += ref[i].shares[k];
            same = same && i < got.size() && got[i].amount == ref[i].amount && got[i].eligible == ref[i].eligible &&
                   got[i].winners == ref[i].winners && got[i].shares == ref[i].shares;
        }
        wrongPots += !same;
        for (int s = 0; s < n; ++s) wrongStacks += o.stack[s] != stacks[s] - o.total_contrib[s] + win[s];
    }
    std::printf("random invariants, seed %llu: %d hands, %d actions; %d fold wins, %d showdowns (%d with 2+ pots, %d with 3+, "
                "%d split pots, %d with an odd chip); %d all-in blinds, %d heads-up hands, %d short stacks (<= 12)\n",
                (unsigned long long)seed, hands, stepsTotal, foldWins, showdowns, multiPot, threePots, ties, oddChips,
                blindAllIn, headsUp, shortStack);
    CHECK(stuck == 0, "%d hands did not finish in 2000 actions", stuck);
    CHECK(negative == 0, "%d observations with a negative stack", negative);
    CHECK(leaks == 0, "%d observations where stacks + pot differ from the chips at the start", leaks);
    CHECK(potMismatch == 0, "%d observations where the pot differs from the sum of contributions", potMismatch);
    CHECK(orphans == 0, "%d pots at showdown with no unfolded seat eligible", orphans);
    CHECK(wrongPots == 0, "%d showdowns where the engine's pots, winners or shares differ from the reference", wrongPots);
    CHECK(wrongStacks == 0, "%d hands where a final stack differs from the reference", wrongStacks);
    CHECK(foldWins > 0 && showdowns > 0 && multiPot > 0 && threePots > 0 && ties > 0 && oddChips > 0 && blindAllIn > 0 && headsUp > 0,
          "cases covered (each must be > 0)");
}

int main() {
    try {
        nestedSidePots4();
        nestedSidePots3();
        tiedPotOddChip();
        tiedPotRemainderTwo();
        tiedSidePot();
        shortAllInReopens();
        fullAllInRaiseMovesMinimum();
        allInBlinds();
        headsUpAfterBust();
        buttonAfterEliminations();
        randomInvariants(30000, 19);
    } catch (const std::exception& e) {
        CHECK(false, "exception: %s", e.what());
    }
    return finish("test_engine_rules");
}
