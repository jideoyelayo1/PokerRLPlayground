#pragma once
// Table: the demo's table, the class behind the JavaScript binding (web/poker_wasm.cpp).
//
// The format is fixed to the one the agents were measured in (bench/RESULTS.md section 6): 4 seats,
// blinds 1/2, every stack reset to 200 before each hand, the button moving one seat per hand. The
// visitor plays seat 0; seats 1-3 are the selected network with the play-time option no_free_fold
// on. Deals follow from the seed only (one shuffle per hand), and the agents are deterministic, so
// the same seed and the same visitor actions give the same hands.
//
// The view is the visitor's: it holds the visitor's own hole cards and no one else's, except, once a
// hand is over, those of the seats that reached showdown (never a seat that folded).
//
// Plain C++ (no Emscripten), so the native tests and web/replay_driver.cpp use the same class.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Agent.hpp"

struct TableSeat {
    int seat = 0, stack = 0, streetContribution = 0, totalContribution = 0;
    std::string status;  // in_hand, folded or all_in
    bool operator==(const TableSeat&) const = default;
};
struct TableAction {  // one of the visitor's legal actions
    std::string action;
    int chips = 0;  // chips it puts in
    int to = 0;     // the visitor's street contribution after it
    bool operator==(const TableAction&) const = default;
};
struct TableLogEntry {
    int seat = 0;
    std::string street, action;  // action: small_blind, big_blind or one of agent::kActionNames
    int chips = 0, to = 0;
    bool operator==(const TableLogEntry&) const = default;
};
struct TablePot {
    int amount = 0;
    std::vector<int> eligible, winners, shares;  // shares[k] chips to winners[k]
    bool operator==(const TablePot&) const = default;
};
struct TableShowdown {
    int seat = 0;
    std::vector<std::string> cards;
    std::string hand;  // high_card, pair, two_pair, ..., straight_flush
    bool operator==(const TableShowdown&) const = default;
};
struct TableResult {
    std::vector<TablePot> pots;
    std::vector<int> net;  // each seat's result in chips
    std::vector<TableShowdown> showdown;
    bool operator==(const TableResult&) const = default;
};
struct TableView {
    int hand = 0;  // hands started, 0 before the first
    bool over = true, aborted = false;
    std::string street;  // preflop, flop, turn or river: the round the hand is on, or ended on
    int visitor = 0, button = -1, smallBlind = -1, bigBlind = -1, pot = 0, toAct = -1;
    std::vector<std::string> board, cards;  // cards: the visitor's own
    std::vector<TableSeat> seats;
    std::vector<TableAction> legal;  // empty unless the visitor is to act
    std::vector<TableLogEntry> log;
    std::optional<TableResult> result;  // once the hand is over (not if aborted)
    bool operator==(const TableView&) const = default;
};
struct TableMove {
    bool ok = false;
    // hand_over, not_your_turn, unknown_action, illegal_action, visitor_to_act; hand_in_progress
    // from startHand(); internal_error, with message, if the engine threw (the hand is then aborted)
    std::string error, message;
    int seat = -1;  // -1 for startHand()
    std::string action;
    int chips = 0, to = 0;
    bool over = true;  // after this call
    int toAct = -1;
};

class Table {
public:
    static constexpr int kSeats = 4, kSmallBlind = 1, kBigBlind = 2, kStack = 200, kVisitor = 0;
    // MAX_STEPS_PER_HAND of the training loop: a hand that reaches it is aborted (none did in training)
    static constexpr int kMaxDecisions = 2000;

    Table(const agent::Mlp& net, uint64_t seed) : net_(net) { g_.seed(seed); }

    // Starts the next hand. Refused (hand_in_progress), with nothing changed, while a hand is in
    // progress: a page that wants a fresh start creates a new Table.
    TableMove startHand() {
        if (!over()) return refuse("hand_in_progress");
        g_.reset_stacks(kStack);
        g_.start_hand();
        ++hands_;
        decisions_ = 0;
        aborted_ = false;
        cards_ = g_.getObservation(kVisitor).hole;
        TableMove m;
        m.ok = true;
        m.over = over();
        m.toAct = toAct();
        return m;
    }

    bool over() const { return hands_ == 0 || aborted_ || g_.hand_terminal(); }
    int toAct() const { return over() ? -1 : g_.currentPlayerSafe(); }

    // The visitor's action, by name (agent::kActionNames). Refused, with nothing changed, unless a
    // hand is in progress, it is the visitor's turn and the action is legal.
    TableMove act(const std::string& action) {
        if (over()) return refuse("hand_over");
        if (toAct() != kVisitor) return refuse("not_your_turn");
        const int a = agent::actionIndex(action);
        if (a < 0) return refuse("unknown_action");
        if (agent::legalMask(g_.getObservation(kVisitor))[a] != 0.0f) return refuse("illegal_action");
        return play(kVisitor, a);
    }

    // Exactly one agent decision, when an agent is to act.
    TableMove advance() {
        if (over()) return refuse("hand_over");
        const int p = toAct();
        if (p == kVisitor) return refuse("visitor_to_act");
        return play(p, agent::decide(net_, g_.getObservation(p), true, kBigBlind).action);
    }

    TableView view() const {
        static const char* streets[] = {"preflop", "flop", "turn", "river"};
        TableView v;
        const Observation o = g_.getObservation(kVisitor);
        v.hand = hands_;
        v.over = over();
        v.aborted = aborted_;
        v.street = streets[o.street >= 0 && o.street < 4 ? o.street : 3];
        v.visitor = kVisitor;
        v.pot = o.pot_total;
        v.toAct = toAct();
        for (int s = 0; s < kSeats; ++s)
            v.seats.push_back({s, o.stack[s], o.street_contrib[s], o.total_contrib[s],
                               o.folded[s] ? "folded" : o.all_in[s] ? "all_in" : "in_hand"});
        if (hands_ == 0) return v;

        v.button = o.button;
        v.smallBlind = (o.button + 1) % kSeats;  // every seat is in every hand: stacks reset
        v.bigBlind = (o.button + 2) % kSeats;
        for (const Card& c : o.board) if (c.rank != Ranks::Empty) v.board.push_back(cardName(c));
        v.cards = {cardName(cards_[0]), cardName(cards_[1])};
        if (v.toAct == kVisitor) {
            const auto mask = agent::legalMask(o);
            const int mine = o.street_contrib[kVisitor], stack = o.stack[kVisitor];
            for (int a = 0; a < agent::kActions; ++a) {
                if (mask[a] != 0.0f) continue;
                int to = a == 2 ? mine + std::min(o.viewer_to_call, stack) : a == 6 ? mine + stack : mine;
                for (const auto& s : o.legal)
                    if (s.type == ActionType::Raise && agent::specIndex(s) == a) to = s.raise_to;
                v.legal.push_back({agent::kActionNames[a], to - mine, to});
            }
        }
        v.log.push_back({v.smallBlind, "preflop", "small_blind", kSmallBlind, kSmallBlind});
        v.log.push_back({v.bigBlind, "preflop", "big_blind", kBigBlind, kBigBlind});
        for (const auto& r : g_.handHistory())
            v.log.push_back({r.seat, streets[r.street], agent::kActionNames[actionOf(r.action)], r.chips, r.street_contrib});

        if (!g_.hand_terminal() || aborted_) return v;
        TableResult res;
        for (const auto& p : g_.potResults()) res.pots.push_back({p.amount, p.eligible, p.winners, p.shares});
        for (int s = 0; s < kSeats; ++s) res.net.push_back(o.stack[s] - kStack);
        static const char* hands[] = {"", "high_card", "pair", "two_pair", "three_of_a_kind", "straight",
                                      "flush", "full_house", "four_of_a_kind", "straight_flush"};
        const auto board = g_.boardCardsFixed();
        for (int s : g_.showdownSeats()) {
            const auto hole = g_.getObservation(s).hole;  // a seat at showdown has not folded, so the engine shows it
            const HandRank r = GameEngine::bestHandOf7({hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]});
            res.showdown.push_back({s, {cardName(hole[0]), cardName(hole[1])}, hands[static_cast<int>(r.cat)]});
        }
        v.result = res;
        return v;
    }

    const GameEngine& engine() const { return g_; }  // for the native tests; not bound

#ifdef POKER_TEST_HOOKS
    // Test seam, compiled only into the test builds (test_table, the test module): plays an action
    // index for the player to act without checking it, so that a test can make the engine throw
    // inside play().
    TableMove playUnchecked(int a) { return play(g_.currentPlayerSafe(), a); }
#endif

    static std::string cardName(const Card& c) {
        if (c.rank == Ranks::Empty || c.suit == Suits::Empty) return "";
        return {"23456789TJQKA"[static_cast<int>(c.rank) - 2], "hdcs"[static_cast<int>(c.suit) - 1]};
    }

private:
    const agent::Mlp& net_;
    GameEngine g_{kSeats, kStack, kBigBlind, kSmallBlind};
    int hands_ = 0, decisions_ = 0;
    bool aborted_ = false;
    std::array<Card, 2> cards_{};

    static int actionOf(const Action& a) {
        ActionSpec s;
        s.type = a.type;
        s.raise_size = a.raiseSize;
        return agent::specIndex(s);
    }

    TableMove refuse(const char* why) const {
        TableMove m;
        m.error = why;
        m.over = over();
        m.toAct = toAct();
        return m;
    }

    TableMove play(int seat, int a) {
        TableMove m;
        try {
            g_.step(agent::engineAction(a), seat);  // seat is the player to act: step() acts for that player
        } catch (const std::exception& e) {        // not expected: every action was checked legal
            aborted_ = true;
            m = refuse("internal_error");
            m.message = e.what();
            return m;
        }
        const ActionRecord& r = g_.handHistory().back();
        if (++decisions_ >= kMaxDecisions && !g_.hand_terminal()) aborted_ = true;
        m.ok = true;
        m.seat = seat;
        m.action = agent::kActionNames[a];
        m.chips = r.chips;
        m.to = r.street_contrib;
        m.over = over();
        m.toAct = toAct();
        return m;
    }
};
