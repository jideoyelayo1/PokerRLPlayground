// Exhaustive check of the hand evaluator.
//  1. All 2,598,960 five-card hands through evaluate5.
//  2. All 133,784,560 seven-card hands through GameEngine::bestHandOf7.
// Each hand is compared, field by field (category and kicker vector), with refEval below: an
// independent evaluator written for this test that works from rank counts and suit bitmasks
// instead of trying the 21 five-card subsets. Category counts and the number of distinct hand
// values per category are compared with the published tables (e.g. Wikipedia, "Poker probability").
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

#include "test_util.hpp"

static constexpr int kCats = 9;
static const char* kCatName[kCats] = {"High card", "One pair", "Two pair", "Three of a kind", "Straight",
                                      "Flush", "Full house", "Four of a kind", "Straight flush"};
static const uint64_t kCount5[kCats] = {1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
static const uint64_t kDistinct5[kCats] = {1277, 2860, 858, 858, 10, 1277, 156, 156, 10};
static const uint64_t kCount7[kCats] = {23294460, 58627800, 31433400, 6461620, 6180020, 4047644, 3473184, 224848, 41584};
static const uint64_t kDistinct7[kCats] = {407, 1470, 763, 575, 10, 1277, 156, 156, 10};

// Highest straight in a rank bitmask (bit r set for rank r, 2..14), 5 for the wheel, 0 for none.
static int topStraight(unsigned m) {
    if (m & (1u << 14)) m |= 1u << 1;  // ace also plays low
    for (int hi = 14; hi >= 5; --hi)
        if (((m >> (hi - 4)) & 0x1Fu) == 0x1Fu) return hi;
    return 0;
}

// Ranks present in mask, high to low, skipping up to two excluded ranks, at most n of them.
static void topRanks(unsigned m, int n, int ex1, int ex2, int* out) {
    for (int r = 14, k = 0; r >= 2 && k < n; --r)
        if ((m & (1u << r)) && r != ex1 && r != ex2) out[k++] = r;
}

static HandRank refEval(const Card* c, int n) {
    int cnt[15] = {0}, suitCnt[5] = {0};
    unsigned suitMask[5] = {0}, all = 0;
    for (int i = 0; i < n; ++i) {
        int r = static_cast<int>(c[i].rank), s = static_cast<int>(c[i].suit);
        ++cnt[r]; ++suitCnt[s]; suitMask[s] |= 1u << r; all |= 1u << r;
    }
    int flushSuit = 0;
    for (int s = 1; s <= 4; ++s) if (suitCnt[s] >= 5) flushSuit = s;
    int quad = 0, trips[2] = {0, 0}, pairs[3] = {0, 0, 0}, nt = 0, np = 0;
    for (int r = 14; r >= 2; --r) {
        if (cnt[r] == 4) quad = r;
        else if (cnt[r] == 3) trips[nt++] = r;
        else if (cnt[r] == 2) pairs[np++] = r;
    }
    HandRank h;
    int k[5] = {0, 0, 0, 0, 0};
    auto set = [&](HandCat cat) { h.cat = cat; for (int i = 0; i < 5; ++i) h.kickers[i] = k[i]; return h; };

    if (flushSuit && (k[0] = topStraight(suitMask[flushSuit]))) return set(HandCat::StraightFlush);
    if (quad) { k[0] = quad; topRanks(all, 1, quad, 0, k + 1); return set(HandCat::Quads); }
    if (nt && (nt > 1 || np)) { k[0] = trips[0]; k[1] = std::max(trips[1], pairs[0]); return set(HandCat::FullHouse); }
    if (flushSuit) { topRanks(suitMask[flushSuit], 5, 0, 0, k); return set(HandCat::Flush); }
    if ((k[0] = topStraight(all))) return set(HandCat::Straight);
    if (nt) { k[0] = trips[0]; topRanks(all, 2, trips[0], 0, k + 1); return set(HandCat::Trips); }
    if (np >= 2) { k[0] = pairs[0]; k[1] = pairs[1]; topRanks(all, 1, pairs[0], pairs[1], k + 2); return set(HandCat::TwoPair); }
    if (np) { k[0] = pairs[0]; topRanks(all, 3, pairs[0], 0, k + 1); return set(HandCat::OnePair); }
    topRanks(all, 5, 0, 0, k);
    return set(HandCat::HighCard);
}

static uint32_t encode(const HandRank& h) {
    uint32_t v = static_cast<uint32_t>(h.cat) - 1;
    for (int x : h.kickers) v = v * 15 + x;
    return v;  // < 9 * 15^5
}
static constexpr uint32_t kCodes = 9 * 759375;

struct Tally {
    uint64_t count[kCats] = {0};
    std::vector<uint8_t> seen = std::vector<uint8_t>(kCodes, 0);
    uint64_t hands = 0, mismatches = 0;
    std::vector<std::string> examples;

    void add(const HandRank& engine, const HandRank& ref, const Card* c, int n) {
        ++hands;
        ++count[static_cast<int>(engine.cat) - 1];
        seen[encode(engine)] = 1;
        if (engine.cat != ref.cat || engine.kickers != ref.kickers) {
            if (++mismatches <= 10) {
                std::string s;
                for (int i = 0; i < n; ++i) s += std::to_string(static_cast<int>(c[i].rank)) + "/" + std::to_string(static_cast<int>(c[i].suit)) + " ";
                examples.push_back(s + " engine cat " + std::to_string(static_cast<int>(engine.cat)) + " ref cat " + std::to_string(static_cast<int>(ref.cat)));
            }
        }
    }
    void merge(const Tally& o) {
        hands += o.hands; mismatches += o.mismatches;
        for (int i = 0; i < kCats; ++i) count[i] += o.count[i];
        for (uint32_t i = 0; i < kCodes; ++i) seen[i] |= o.seen[i];
        for (auto& e : o.examples) if (examples.size() < 10) examples.push_back(e);
    }
};

static bool report(const char* title, const Tally& t, const uint64_t* wantCount, const uint64_t* wantDistinct) {
    uint64_t distinct[kCats] = {0};
    for (uint32_t i = 0; i < kCodes; ++i) if (t.seen[i]) ++distinct[i / 759375];
    bool ok = t.mismatches == 0;
    std::printf("\n%s: %llu hands, %llu disagreements with the reference evaluator\n", title,
                (unsigned long long)t.hands, (unsigned long long)t.mismatches);
    for (auto& e : t.examples) std::printf("  e.g. %s\n", e.c_str());
    std::printf("  %-16s %12s %12s %9s %9s\n", "category", "count", "expected", "distinct", "expected");
    uint64_t totalDistinct = 0;
    for (int i = kCats - 1; i >= 0; --i) {
        bool row = t.count[i] == wantCount[i] && distinct[i] == wantDistinct[i];
        ok = ok && row;
        totalDistinct += distinct[i];
        std::printf("  %-16s %12llu %12llu %9llu %9llu %s\n", kCatName[i], (unsigned long long)t.count[i],
                    (unsigned long long)wantCount[i], (unsigned long long)distinct[i],
                    (unsigned long long)wantDistinct[i], row ? "ok" : "MISMATCH");
    }
    std::printf("  distinct hand values: %llu\n", (unsigned long long)totalDistinct);
    return ok;
}

int main() {
    std::vector<Card> deck;
    for (int s = 1; s <= 4; ++s)
        for (int r = 2; r <= 14; ++r) deck.emplace_back(static_cast<Suits>(s), static_cast<Ranks>(r));

    auto t0 = std::chrono::steady_clock::now();
    Tally five;
    for (int a = 0; a < 52; ++a) for (int b = a + 1; b < 52; ++b) for (int c = b + 1; c < 52; ++c)
    for (int d = c + 1; d < 52; ++d) for (int e = d + 1; e < 52; ++e) {
        std::array<Card, 5> h{deck[a], deck[b], deck[c], deck[d], deck[e]};
        five.add(evaluate5(h), refEval(h.data(), 5), h.data(), 5);
    }
    bool ok5 = report("Five-card hands (evaluate5)", five, kCount5, kDistinct5);

    // Seven-card hands, split across threads by the first card.
    std::atomic<int> nextFirst{0};
    unsigned nThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<Tally> tallies(nThreads);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nThreads; ++t) pool.emplace_back([&, t] {
        std::vector<Card> h(7);
        for (int a; (a = nextFirst++) < 46;) {
            h[0] = deck[a];
            for (int b = a + 1; b < 52; ++b) { h[1] = deck[b];
            for (int c = b + 1; c < 52; ++c) { h[2] = deck[c];
            for (int d = c + 1; d < 52; ++d) { h[3] = deck[d];
            for (int e = d + 1; e < 52; ++e) { h[4] = deck[e];
            for (int f = e + 1; f < 52; ++f) { h[5] = deck[f];
            for (int g = f + 1; g < 52; ++g) { h[6] = deck[g];
                tallies[t].add(GameEngine::bestHandOf7(h), refEval(h.data(), 7), h.data(), 7);
            }}}}}}
        }
    });
    for (auto& th : pool) th.join();
    Tally seven;
    for (auto& t : tallies) seven.merge(t);
    bool ok7 = report("Seven-card hands (GameEngine::bestHandOf7)", seven, kCount7, kDistinct7);

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nthreads %u, wall time %.1f s\n%s\n", nThreads, secs, ok5 && ok7 ? "PASS" : "FAIL");
    return ok5 && ok7 ? 0 : 1;
}
