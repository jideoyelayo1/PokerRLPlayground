// Known-answer tests for the hand evaluator (evaluate5 / GameEngine::bestHandOf7):
// one or more hands per category, the kicker vector the engine stores, and tie-break order.
#include "test_util.hpp"

static HandRank best7(const std::string& s) { return GameEngine::bestHandOf7(cards(s)); }

static void expect(const std::string& hand, HandCat cat, std::array<int, 5> kickers) {
    HandRank r = best7(hand);
    CHECK(r.cat == cat && r.kickers == kickers, "%s -> cat %d kickers %d %d %d %d %d", hand.c_str(),
          static_cast<int>(r.cat), r.kickers[0], r.kickers[1], r.kickers[2], r.kickers[3], r.kickers[4]);
}

// board + two hole cards for each player; expected sign of (a vs b): 1 a wins, -1 b wins, 0 split
static void showdown(const std::string& board, const std::string& a, const std::string& b, int expected) {
    HandRank ra = best7(a + " " + board), rb = best7(b + " " + board);
    int got = (rb < ra) ? 1 : (ra < rb) ? -1 : 0;
    CHECK(got == expected, "board [%s] %s vs %s: got %d want %d", board.c_str(), a.c_str(), b.c_str(), got, expected);
}

int main() {
    using HC = HandCat;
    // Every category, best five picked out of seven.
    expect("As Kd 9h 7c 5s 3d 2h", HC::HighCard, {14, 13, 9, 7, 5});
    expect("Ah Ad Kc 9s 7d 4c 2h", HC::OnePair, {14, 13, 9, 7, 0});
    expect("Kh Kd 9c 9s Ad 4c 2h", HC::TwoPair, {13, 9, 14, 0, 0});
    expect("Kh Kd 9c 9s 4d 4c 2h", HC::TwoPair, {13, 9, 4, 0, 0});      // three pairs: third pair is the kicker
    expect("7h 7d 7c As Kd 4c 2h", HC::Trips, {7, 14, 13, 0, 0});
    expect("9h 8d 7c 6s 5d Kc 2h", HC::Straight, {9, 0, 0, 0, 0});
    expect("Ah 2d 3c 4s 5d Kc 9h", HC::Straight, {5, 0, 0, 0, 0});      // wheel
    expect("Ah 2d 3c 4s 5d 6c Kh", HC::Straight, {6, 0, 0, 0, 0});      // six-high beats the wheel inside one hand
    expect("Ah Kd Qc Js Td 3c 2h", HC::Straight, {14, 0, 0, 0, 0});     // broadway
    expect("9h 8d 7c 6s 5d 4c Kh", HC::Straight, {9, 0, 0, 0, 0});      // six-card run
    expect("Ah Jh 9h 6h 4h 2h Kd", HC::Flush, {14, 11, 9, 6, 4});       // six hearts: top five
    expect("9h 8d 7h 6h 5c 2h Kh", HC::Flush, {13, 9, 7, 6, 2});        // flush over a straight
    expect("Ah Kh 4h 3h 2c 5d 9h", HC::Flush, {14, 13, 9, 4, 3});       // flush over a wheel
    expect("Kh Kd Kc 9s 9d 9c 2h", HC::FullHouse, {13, 9, 0, 0, 0});    // two sets of trips
    expect("5h 5d 5c Qs Qd 9c 9h", HC::FullHouse, {5, 12, 0, 0, 0});    // trips plus two pairs
    expect("8h 8d 8c 8s Ad Kc 2h", HC::Quads, {8, 14, 0, 0, 0});
    expect("8h 8d 8c 8s 3d 3c 3h", HC::Quads, {8, 3, 0, 0, 0});
    expect("9h 8h 7h 6h 5h Ah Kd", HC::StraightFlush, {9, 0, 0, 0, 0});
    expect("Ah 2h 3h 4h 5h Kd Qc", HC::StraightFlush, {5, 0, 0, 0, 0}); // steel wheel
    expect("Ah Kh Qh Jh Th 9h 2c", HC::StraightFlush, {14, 0, 0, 0, 0});// royal over the king-high one
    expect("9h 8h 7h 6h 5h Tc Jd", HC::StraightFlush, {9, 0, 0, 0, 0}); // straight flush over a higher straight

    // evaluate5 directly.
    std::array<Card, 5> five{card("Td"), card("Jd"), card("Qd"), card("Kd"), card("Ad")};
    CHECK(evaluate5(five).cat == HC::StraightFlush && evaluate5(five).kickers[0] == 14, "royal via evaluate5");

    // Category order: each hand beats the one before it.
    const char* ladder[] = {"As Kd 9h 7c 5s 3d 2h", "2h 2d Kc 9s 7d 4c 3h", "3h 3d 2c 2s Ad 9c 7h",
                            "2h 2d 2c As Kd 9c 7h", "Ah 2d 3c 4s 5d 9c Jh", "2h 4h 6h 8h Th Ac Kd",
                            "2h 2d 2c 3s 3d 9c 7h", "2h 2d 2c 2s 3d 9c 7h", "Ah 2h 3h 4h 5h 9c 7d"};
    for (int i = 1; i < 9; ++i) CHECK(best7(ladder[i - 1]) < best7(ladder[i]), "ladder %d < %d", i - 1, i);

    // Tie-breaks.
    showdown("Ah Ad 9c 7s 4d", "Kc 2h", "Qc 3h", 1);    // pair: first kicker
    showdown("Ah Ad Kc Qs Jd", "3c 2h", "4c 2s", 0);    // pair: board plays, sixth card ignored
    showdown("Kh Kd 9c 9s 4d", "Ac 2h", "Qc 2s", 1);    // two pair: kicker
    showdown("Kh Kd 9c 9s 5d", "6h 6s", "Ac 2h", -1);   // two pair: pocket pair counterfeited
    showdown("Kh Kd 7c 7s 4d", "Qh Qs", "Ac 2h", 1);    // two pair: KKQQ over KK77
    showdown("7h 7d 2c 9s Jd", "7c 3h", "7s 4h", 0);    // trips: kickers from board, split
    showdown("8h 8d 2c 9s Jd", "8c Ah", "8s Kh", 1);    // trips: kicker
    showdown("2d 3c 4s 5h Kd", "Ac 9h", "6c 9d", -1);   // straight: six-high over wheel
    showdown("5h 6d 7c 8s 9h", "2c 3d", "2d 3s", 0);    // straight on board, split
    showdown("5h 6d 7c 8s 9h", "Tc 3d", "2d 3s", 1);    // straight: ten-high over board
    showdown("Ah Jh 9h 6h Kd", "4h 2c", "3h 2d", 1);    // flush: fifth card decides
    showdown("Ah Kh 9h 6h 5h", "4h 2c", "3h 2d", 0);    // flush on board: both play it
    showdown("Kh Kd Kc 7s 2d", "7h 3c", "2h 3d", 1);    // full house: pair decides
    showdown("Kh Kd Qc Qs 2d", "Kc 3c", "Qh Ad", 1);    // full house: trips rank first (KKKQQ vs QQQKK)
    showdown("8h 8d 8c 8s 2d", "Ac 3c", "Kc Qc", 1);    // quads: kicker
    showdown("8h 8d 8c 8s Ad", "Kc 3c", "Qc 2h", 0);    // quads: ace kicker on board, split
    showdown("2h 3h 4h 5h Kd", "Ah 9c", "6h 9d", -1);   // straight flush: six-high over steel wheel
    showdown("9s 8s 7s 6s 2d", "5s Ad", "Ts Ad", -1);   // straight flush: higher wins

    return finish("test_hand_eval");
}
