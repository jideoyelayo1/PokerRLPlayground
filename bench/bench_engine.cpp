// Engine throughput in hands per second, measured in C++ with fixed policies.
//   playOneHand: the C++ controller API, every seat calls (or checks).
//   step/call:   the step() API the Python loop uses, every seat calls (or checks).
//   step/random: the step() API, every seat picks a uniformly random legal action.
// 4 seats, stacks reset to 10,000 every hand, blinds 1/2 (the README's example settings).
// Usage: bench_engine [hands_per_run=200000] [runs=7]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../PokerEngine/tests/policies.hpp"

static constexpr int kSeats = 4, kStack = 10000;

static double runPlayOneHand(int hands, uint64_t seed) {
    GameEngine g(kSeats, kStack, 2, 1);
    g.seed(seed);
    for (int p = 0; p < kSeats; ++p)
        g.setController(p, [](const GameEngine& e, int id) {
            return Action{GameEngine::isAllowed(e.legalActions(id), ActionType::Call) ? ActionType::Call : ActionType::Check};
        });
    auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < hands; ++h) { g.reset_tournament(kStack); g.playOneHand(); }
    return hands / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double runStep(int hands, uint64_t seed, Policy pol, long& decisions) {
    GameEngine g(kSeats, kStack, 2, 1);
    g.seed(seed);
    std::mt19937_64 rng(seed + 1000);
    decisions = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < hands; ++h) {
        g.reset_tournament(kStack);
        g.start_hand();
        while (!g.hand_terminal()) {
            int p = g.currentPlayerSafe();
            auto o = g.getObservation(p);
            stepWithFallback(g, p, o, choose(pol, o, rng));
            ++decisions;
        }
    }
    return hands / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void summary(const char* name, std::vector<double> v, double decisionsPerHand) {
    std::sort(v.begin(), v.end());
    std::printf("%-12s median %10.0f hands/s  min %10.0f  max %10.0f  (n=%zu runs, %.2f decisions/hand)\n", name,
                v[v.size() / 2], v.front(), v.back(), v.size(), decisionsPerHand);
}

int main(int argc, char** argv) {
    int hands = argc > 1 ? std::atoi(argv[1]) : 200000, runs = argc > 2 ? std::atoi(argv[2]) : 7;
    std::printf("hands per run %d, runs %d, seats %d, stack %d, blinds 1/2, seeds 1..%d\n", hands, runs, kSeats, kStack, runs);
    std::vector<double> a, b, c;
    double dCall = 0, dRand = 0;
    for (int r = 1; r <= runs; ++r) {
        long d1 = 0, d2 = 0;
        a.push_back(runPlayOneHand(hands, r));
        b.push_back(runStep(hands, r, Policy::Call, d1));
        c.push_back(runStep(hands, r, Policy::Random, d2));
        dCall = double(d1) / hands; dRand = double(d2) / hands;
        std::printf("run %d: playOneHand %.0f  step/call %.0f  step/random %.0f hands/s\n", r, a.back(), b.back(), c.back());
    }
    summary("playOneHand", a, dCall);
    summary("step/call", b, dCall);
    summary("step/random", c, dRand);
}
