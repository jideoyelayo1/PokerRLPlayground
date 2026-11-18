// What a table view reads from the engine: reset_stacks (stacks reset, button keeps moving), the
// action history of the hand, each pot's winners and shares, and the seats that reached showdown.
#include <map>

#include "policies.hpp"
#include "test_util.hpp"

static constexpr int kCap = 2000;  // MAX_STEPS_PER_HAND in ReinforcementLearningEnvironment/Poker.py

// reset_stacks gives every seat the starting stack, busted seats included, and the button moves
// one seat per hand; reset_tournament puts it back on seat 0.
static void resetStacksKeepsButton() {
    GameEngine g(4, 200, 2, 1);
    g.seed(11);
    std::mt19937_64 rng(12);
    int busted = 0;
    for (int hand = 0; hand < 40; ++hand) {
        g.reset_stacks(200);
        CHECK(g.alivePlayers() == 4, "hand %d: every seat is back after reset_stacks", hand);
        g.start_hand();
        auto o = g.getObservation(0);
        CHECK(o.button == hand % 4, "hand %d: button %d, expected %d", hand, o.button, hand % 4);
        for (int s = 0; s < 4; ++s)
            CHECK(o.stack[s] + o.street_contrib[s] == 200, "hand %d: seat %d started with %d", hand, s, o.stack[s] + o.street_contrib[s]);
        for (int steps = 0; !g.hand_terminal() && steps < kCap; ++steps) {
            int p = g.currentPlayerSafe();
            g.step(choose(p == 1 ? Policy::AllIn : Policy::Random, g.getObservation(p), rng), p);
        }
        busted += g.alivePlayers() < 4;
    }
    CHECK(busted > 0, "some hand busted a seat, so reset_stacks brought one back (%d hands)", busted);
    g.reset_tournament(200);
    g.start_hand();
    CHECK(g.getObservation(0).button == 0, "reset_tournament still puts the button on seat 0");
}

// Random and fixed policies with stacks reset every hand (as the demo plays): side pots, splits and
// fold wins all occur, and every hand's records agree with what happened.
static void historyPotsAndShowdown() {
    const int hands = 20000;
    GameEngine g(4, 200, 2, 1);
    g.seed(13);
    std::mt19937_64 rng(14);
    const Policy mixes[2][4] = {{Policy::Random, Policy::AllIn, Policy::Call, Policy::Random},
                                {Policy::Random, Policy::CheckFold, Policy::Random, Policy::CheckFold}};
    int badHistory = 0, badContrib = 0, badShares = 0, badNet = 0, badShowdown = 0, badWinners = 0, rejectedLogged = 0;
    int foldWins = 0, showdowns = 0, sidePots = 0, splits = 0;
    for (int h = 0; h < hands; ++h) {
        g.reset_stacks(200);
        g.start_hand();
        std::vector<ActionRecord> mine;
        for (int steps = 0; !g.hand_terminal() && steps < kCap; ++steps) {
            int p = g.currentPlayerSafe(), st = g.streetInt();
            auto o = g.getObservation(p);
            if (h % 50 == 0) {  // an illegal action is refused and not logged
                for (auto& s : o.legal) {
                    if (s.allowed) continue;
                    size_t before = g.handHistory().size();
                    try { g.step({s.type, s.raise_size}, p); } catch (const std::runtime_error&) {}
                    rejectedLogged += g.handHistory().size() != before;
                    break;
                }
            }
            Action a = choose(mixes[h % 2][p], o, rng);
            g.step(a, p);
            mine.push_back({p, st, a, 0, 0});
            // while the street goes on, the logged street contribution is the seat's
            if (!g.hand_terminal() && g.streetInt() == st)
                badContrib += g.handHistory().back().street_contrib != g.getObservation(p).street_contrib[p];
        }
        const auto& hist = g.handHistory();
        bool ok = hist.size() == mine.size();
        for (size_t i = 0; ok && i < hist.size(); ++i)
            ok = hist[i].seat == mine[i].seat && hist[i].street == mine[i].street && hist[i].action.type == mine[i].action.type &&
                 (hist[i].action.type != ActionType::Raise || hist[i].action.raiseSize == mine[i].action.raiseSize);
        // chips: per seat, the logged chips plus the blinds are the seat's total contribution
        auto o = g.getObservation(0);
        std::vector<int> logged(4, 0);
        for (auto& r : hist) logged[r.seat] += r.chips;
        const int button = o.button, sb = (button + 1) % 4, bb = (button + 2) % 4;
        for (int s = 0; s < 4; ++s)
            ok = ok && logged[s] + (s == sb ? 1 : s == bb ? 2 : 0) == o.total_contrib[s];
        badHistory += !ok;

        const auto& pots = g.potResults();
        int paid = 0;
        std::map<int, int> won;
        for (auto& pr : pots) {
            int sum = 0;
            for (size_t k = 0; k < pr.winners.size(); ++k) { sum += pr.shares[k]; won[pr.winners[k]] += pr.shares[k]; }
            badShares += pr.winners.empty() || pr.shares.size() != pr.winners.size() || sum != pr.amount;
            for (int w : pr.winners)
                badWinners += std::find(pr.eligible.begin(), pr.eligible.end(), w) == pr.eligible.end() || o.folded[w];
            paid += pr.amount;
            splits += pr.winners.size() > 1;
        }
        badShares += paid != o.pot_total;
        for (int s = 0; s < 4; ++s) badNet += o.stack[s] - 200 != won[s] - o.total_contrib[s];

        std::vector<int> live;
        for (int s = 0; s < 4; ++s) if (!o.folded[s]) live.push_back(s);
        if (live.size() == 1) {
            ++foldWins;
            badShowdown += !g.showdownSeats().empty() || pots.size() != 1 || pots[0].winners != live || g.winner_id() != live[0];
        } else {
            ++showdowns;
            sidePots += pots.size() > 1;
            badShowdown += g.showdownSeats() != live;
            // the winners of each pot hold the best hand among its eligible seats
            auto board = g.boardCardsFixed();
            auto rank = [&](int s) {
                auto hole = g.getObservation(s).hole;
                return GameEngine::bestHandOf7({hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]});
            };
            for (auto& pr : pots) {
                HandRank best = rank(pr.winners[0]);
                for (int e : pr.eligible) {
                    bool isWinner = std::find(pr.winners.begin(), pr.winners.end(), e) != pr.winners.end();
                    badWinners += isWinner ? !sameRank(rank(e), best) : !(rank(e) < best);
                }
            }
        }
    }
    std::printf("hand results, %d hands: %d fold wins, %d showdowns, %d with side pots, %d split pots\n",
                hands, foldWins, showdowns, sidePots, splits);
    CHECK(badHistory == 0, "%d hands' history differs from the actions played", badHistory);
    CHECK(badContrib == 0, "%d logged street contributions differ", badContrib);
    CHECK(rejectedLogged == 0, "%d rejected actions were logged", rejectedLogged);
    CHECK(badShares == 0, "%d pots or hands whose shares do not add up", badShares);
    CHECK(badNet == 0, "%d seat results differ from shares minus contributions", badNet);
    CHECK(badShowdown == 0, "%d hands with the wrong showdown seats or fold-win pot", badShowdown);
    CHECK(badWinners == 0, "%d pot winners who are not eligible or not the best hand", badWinners);
    CHECK(foldWins > 0 && showdowns > 0 && sidePots > 0 && splits > 0, "cases covered (each must be > 0)");
}

int main() {
    resetStacksKeepsButton();
    historyPotsAndShowdown();
    return finish("test_hand_result");
}
