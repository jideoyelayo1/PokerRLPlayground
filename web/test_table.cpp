// Native tests of Table (web/Table.hpp), the class the JavaScript binding exposes.
//   test_table <weights/demo>
#include <algorithm>
#include <fstream>
#include <random>
#include <regex>
#include <set>
#include <sstream>

#include "../PokerEngine/tests/test_util.hpp"
#include "Table.hpp"

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// Every string field of a view and of a move (keep this in step with the structs in Table.hpp).
static std::vector<std::string> strings(const TableView& v, const TableMove& m) {
    std::vector<std::string> out{v.street, m.error, m.action};
    out.insert(out.end(), v.board.begin(), v.board.end());
    out.insert(out.end(), v.cards.begin(), v.cards.end());
    for (const auto& s : v.seats) out.push_back(s.status);
    for (const auto& a : v.legal) out.push_back(a.action);
    for (const auto& e : v.log) { out.push_back(e.street); out.push_back(e.action); }
    if (v.result)
        for (const auto& s : v.result->showdown) { out.insert(out.end(), s.cards.begin(), s.cards.end()); out.push_back(s.hand); }
    return out;
}

// Every card-like text ([2-9TJQKA][hdcs]) inside those strings, wherever it is.
static std::set<std::string> cardsIn(const TableView& v, const TableMove& m = {}) {
    static const std::regex card("[2-9TJQKA][hdcs]");
    std::set<std::string> out;
    for (const auto& text : strings(v, m))
        for (std::sregex_iterator it(text.begin(), text.end(), card), end; it != end; ++it) out.insert(it->str());
    return out;
}

// A scripted visitor: a uniformly random legal action.
static std::string pick(const TableView& v, std::mt19937& rng) { return v.legal[rng() % v.legal.size()].action; }

// One hand to its end; returns a transcript of what the view showed.
static std::string playHand(Table& t, std::mt19937& rng) {
    std::ostringstream out;
    CHECK(t.startHand().ok, "startHand after the previous hand");
    for (int guard = 0; !t.over() && guard < 2 * Table::kMaxDecisions; ++guard) {
        TableMove m = t.toAct() == Table::kVisitor ? t.act(pick(t.view(), rng)) : t.advance();
        out << m.seat << ' ' << m.action << ' ' << m.chips << ' ' << m.to << ';';
    }
    auto v = t.view();
    for (auto& c : v.board) out << c;
    for (auto& c : v.cards) out << c;
    if (v.result) for (int n : v.result->net) out << ' ' << n;
    return out.str();
}

static void deterministic(const agent::Mlp& net) {
    Table a(net, 7), b(net, 7), c(net, 8);
    std::mt19937 ra(1), rb(1), rc(1);
    int same = 0, differ = 0;
    for (int h = 0; h < 300; ++h) {
        auto x = playHand(a, ra), y = playHand(b, rb), z = playHand(c, rc);
        same += x == y;
        differ += x != z;
    }
    CHECK(same == 300, "same seed and visitor actions: %d of 300 hands identical", same);
    CHECK(differ > 250, "another seed gives other hands (%d of 300 differ)", differ);
}

// The format: 4 seats, blinds 1/2 on the two seats after the button, stacks 200 every hand, the
// button moving one seat per hand.
static void format(const agent::Mlp& net) {
    Table t(net, 3);
    std::mt19937 rng(2);
    for (int h = 0; h < 12; ++h) {
        t.startHand();
        auto v = t.view();
        CHECK(v.button == h % 4, "hand %d: button %d, expected %d", v.hand, v.button, h % 4);
        CHECK(v.smallBlind == (v.button + 1) % 4 && v.bigBlind == (v.button + 2) % 4, "blinds after the button");
        for (auto& s : v.seats)
            CHECK(s.stack + s.streetContribution == 200, "seat %d starts with %d", s.seat, s.stack + s.streetContribution);
        CHECK(v.seats[v.smallBlind].streetContribution == 1 && v.seats[v.bigBlind].streetContribution == 2, "blinds 1/2 posted");
        CHECK(v.log.size() >= 2 && v.log[0].action == "small_blind" && v.log[1].action == "big_blind", "the log starts with the blinds");
        CHECK(v.toAct == (v.button + 3) % 4, "the seat after the big blind acts first");
        while (!t.over()) t.toAct() == Table::kVisitor ? t.act(pick(t.view(), rng)) : t.advance();
    }
}

// The view never holds an agent's hole cards before showdown, nor ever a folded seat's; at the end
// it holds exactly the showdown seats' real cards.
static void hiddenCards(const agent::Mlp& net) {
    Table t(net, 11);
    std::mt19937 rng(5);
    double smallestGap = 1e9;
    int agentDecisions = 0;
    int views = 0, leaks = 0, strays = 0, wrongShowdown = 0, showdowns = 0, foldedAgents = 0, resultsBad = 0;
    for (int h = 0; h < 2000; ++h) {
        t.startHand();
        std::vector<std::set<std::string>> hole(4);
        for (int s = 0; s < 4; ++s)
            for (const Card& c : t.engine().getObservation(s).hole) hole[s].insert(Table::cardName(c));
        auto check = [&](const TableMove& m) {
            auto v = t.view();
            ++views;
            auto seen = cardsIn(v, m);
            std::set<int> shown;  // the engine's own record, not the view's
            if (v.over) for (int s : t.engine().showdownSeats()) shown.insert(s);
            // anything card-like must be the visitor's, the board's or, at the end, a showdown seat's
            std::set<std::string> allowed = hole[0];
            for (const Card& c : t.engine().boardCardsFixed()) if (c.rank != Ranks::Empty) allowed.insert(Table::cardName(c));
            for (int s : shown) allowed.insert(hole[s].begin(), hole[s].end());
            for (const auto& c : seen) strays += !allowed.count(c);
            for (int s = 1; s < 4; ++s) {
                bool visible = false;
                for (auto& c : hole[s]) visible = visible || seen.count(c);
                bool allowed = v.over && shown.count(s) && v.seats[s].status != "folded";
                leaks += visible && !allowed;
            }
            if (v.over && v.result) {
                for (auto& s : v.result->showdown) {
                    wrongShowdown += v.seats[s.seat].status == "folded";
                    if (s.seat != 0) wrongShowdown += std::set<std::string>(s.cards.begin(), s.cards.end()) != hole[s.seat];
                }
                int sum = 0, paid = 0;
                for (int n : v.result->net) sum += n;
                for (auto& p : v.result->pots) for (int x : p.shares) paid += x;
                resultsBad += sum != 0 || paid != v.pot;
            }
        };
        check({});
        while (!t.over()) {
            if (t.toAct() != Table::kVisitor) {  // how close the agent's choice is: the two highest legal Q-values
                const auto d = agent::decide(net, t.engine().getObservation(t.toAct()), true);
                std::vector<float> legal;
                for (int a = 0; a < agent::kActions; ++a) if (d.mask[a] == 0.0f) legal.push_back(d.q[a]);
                std::sort(legal.rbegin(), legal.rend());
                if (legal.size() > 1) smallestGap = std::min(smallestGap, static_cast<double>(legal[0] - legal[1]));
                ++agentDecisions;
            }
            check(t.toAct() == Table::kVisitor ? t.act(pick(t.view(), rng)) : t.advance());
        }
        auto v = t.view();
        showdowns += v.result && !v.result->showdown.empty();
        for (int s = 1; s < 4; ++s) foldedAgents += v.seats[s].status == "folded";
    }
    std::printf("hidden cards: %d views over 2,000 hands, %d with a showdown, %d agent folds\n", views, showdowns, foldedAgents);
    // a gap below the C++-PyTorch difference (8.34e-07 over the parity records) is a choice PyTorch could make otherwise
    std::printf("agent decisions: %d; smallest gap between the two highest legal Q-values %.3g\n", agentDecisions, smallestGap);
    CHECK(leaks == 0, "%d views showed an agent's hole cards they should not", leaks);
    CHECK(strays == 0, "%d card-like texts in views or moves were not the visitor's, the board's or a showdown seat's", strays);
    CHECK(wrongShowdown == 0, "%d showdown entries were folded seats or the wrong cards", wrongShowdown);
    CHECK(resultsBad == 0, "%d results do not add up", resultsBad);
    CHECK(showdowns > 0 && foldedAgents > 0, "cases covered");
}

// act() and advance() refuse, with nothing changed, out of turn, illegal or unknown actions and
// when no hand is in progress.
static void refusals(const agent::Mlp& net) {
    Table t(net, 13);
    std::mt19937 rng(6);
    auto refused = [&](TableMove m, const char* why) {
        bool ok = !m.ok && m.error == why;
        CHECK(ok, "expected refusal %s, got ok=%d error=%s", why, m.ok, m.error.c_str());
        return ok;
    };
    TableView before = t.view();
    refused(t.act("call"), "hand_over");
    refused(t.advance(), "hand_over");
    CHECK(t.view() == before, "a refusal before the first hand changes nothing");

    int notTurn = 0, illegal = 0, unknown = 0, visitorTurn = 0, over = 0, inProgress = 0;
    for (int h = 0; h < 300; ++h) {
        t.startHand();
        while (!t.over()) {
            before = t.view();
            if (t.toAct() != Table::kVisitor) {
                for (const char* a : {"fold", "check", "call", "allin"}) notTurn += refused(t.act(a), "not_your_turn");
                inProgress += refused(t.startHand(), "hand_in_progress");
                CHECK(t.view() == before, "act out of turn changes nothing");
                t.advance();
                continue;
            }
            visitorTurn += refused(t.advance(), "visitor_to_act");
            inProgress += refused(t.startHand(), "hand_in_progress");
            for (const char* a : {"", "Call", "raise", "bet", "7"}) unknown += refused(t.act(a), "unknown_action");
            std::set<std::string> legal;
            for (auto& a : before.legal) legal.insert(a.action);
            for (const char* a : agent::kActionNames)
                if (!legal.count(a)) illegal += refused(t.act(a), "illegal_action");
            CHECK(t.view() == before, "refused visitor actions change nothing");
            t.act(pick(before, rng));
        }
        before = t.view();
        over += refused(t.act("fold"), "hand_over") && refused(t.advance(), "hand_over");
        CHECK(t.view() == before, "a refusal after the hand changes nothing");
    }
    std::printf("refusals: %d out of turn, %d illegal, %d unknown, %d advance on the visitor's turn, %d after the hand, "
                "%d startHand during a hand\n", notTurn, illegal, unknown, visitorTurn, over, inProgress);
    CHECK(notTurn > 0 && illegal > 0 && unknown > 0 && visitorTurn > 0 && over == 300 && inProgress > 0, "cases covered");
    CHECK(t.startHand().ok && t.view().hand == 301, "startHand after a finished hand starts the next");
}

// The visitor's legal actions carry the amounts the engine will apply.
static void legalAmounts(const agent::Mlp& net) {
    Table t(net, 17);
    std::mt19937 rng(7);
    int checked = 0, wrong = 0;
    for (int h = 0; h < 500; ++h) {
        t.startHand();
        while (!t.over()) {
            if (t.toAct() != Table::kVisitor) { t.advance(); continue; }
            auto v = t.view();
            const auto& a = v.legal[rng() % v.legal.size()];
            TableMove m = t.act(a.action);
            ++checked;
            wrong += !m.ok || m.chips != a.chips || m.to != a.to;
        }
    }
    CHECK(checked > 0 && wrong == 0, "%d of %d visitor actions put in other amounts than the view listed", wrong, checked);
}

// Through the test seam Table::playUnchecked, an illegal action makes the engine throw inside
// play(): the result is the documented refusal (internal_error with a message), the hand is
// aborted with nothing paid, and the table then plays full hands as usual.
static void internalError(const agent::Mlp& net) {
    Table t(net, 19);
    std::mt19937 rng(8);
    int forced = 0, badShape = 0, notAborted = 0, played = 0;
    for (int h = 0; h < 50; ++h) {
        CHECK(t.startHand().ok, "startHand after an aborted hand");
        while (!t.over()) {
            const auto mask = agent::legalMask(t.engine().getObservation(t.toAct()));
            int bad = -1;
            for (int a = 0; a < agent::kActions && bad < 0; ++a) if (mask[a] != 0.0f) bad = a;
            if (bad >= 0) {
                const TableMove m = t.playUnchecked(bad);
                ++forced;
                badShape += m.ok || m.error != "internal_error" || m.message.empty() || !m.over || m.toAct != -1;
                const auto v = t.view();
                notAborted += !v.over || !v.aborted || v.result.has_value() || !t.act("fold").error.size();
                break;
            }
            t.toAct() == Table::kVisitor ? t.act(pick(t.view(), rng)) : t.advance();
        }
    }
    for (int h = 0; h < 20; ++h) {
        CHECK(t.startHand().ok, "startHand after the forced errors");
        while (!t.over()) t.toAct() == Table::kVisitor ? t.act(pick(t.view(), rng)) : t.advance();
        played += t.view().result.has_value();
    }
    std::printf("internal errors: %d forced, then %d of 20 hands played to a result\n", forced, played);
    CHECK(forced > 0 && badShape == 0, "%d of %d forced errors had another shape than internal_error with a message", badShape, forced);
    CHECK(notAborted == 0, "%d forced errors did not abort the hand cleanly", notAborted);
    CHECK(played == 20, "the table plays on after the forced errors");
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "weights/demo";
    const agent::Mlp net = agent::Mlp::fromManifest(slurp(dir + "/mlp-s-seed3.json"), slurp(dir + "/mlp-s-seed3.bin"));
    deterministic(net);
    format(net);
    hiddenCards(net);
    refusals(net);
    legalAmounts(net);
    internalError(net);
    return finish("test_table");
}
