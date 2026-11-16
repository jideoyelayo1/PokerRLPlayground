#pragma once
// Fixed policies for the step() API, shared by the betting tests and the throughput benchmark.
#include <random>
#include <stdexcept>

#include "../GameEngine.hpp"

enum class Policy { Random, Call, CheckFold, AllIn };

inline bool allowed(const Observation& o, ActionType t) { return GameEngine::isAllowed(o.legal, t); }

inline Action choose(Policy p, const Observation& o, std::mt19937_64& rng) {
    switch (p) {
        case Policy::Call:      return {allowed(o, ActionType::Call) ? ActionType::Call : ActionType::Check};
        case Policy::CheckFold: return {allowed(o, ActionType::Check) ? ActionType::Check : ActionType::Fold};
        case Policy::AllIn:     return {ActionType::AllIn};
        case Policy::Random: {
            std::vector<ActionType> legal;
            for (auto& s : o.legal) if (s.allowed) legal.push_back(s.type);
            if (legal.empty()) throw std::logic_error("no legal action for the current player");
            ActionType t = legal[rng() % legal.size()];
            return {t, static_cast<RaiseSize>(rng() % 3)};
        }
    }
    return {ActionType::Fold};
}

// Apply a for the current player; if a raise size is rejected, fall back the way the Python
// training loop's step_with_fallback does (other raise sizes, then check/call/all-in/fold).
inline void stepWithFallback(GameEngine& g, int pid, const Observation& o, Action a) {
    try { g.step(a, pid); return; } catch (const std::runtime_error&) {}
    if (a.type == ActionType::Raise)
        for (RaiseSize sz : {RaiseSize::BB, RaiseSize::HalfPot, RaiseSize::Pot}) {
            if (sz == a.raiseSize) continue;
            try { g.step({ActionType::Raise, sz}, pid); return; } catch (const std::runtime_error&) {}
        }
    for (ActionType t : {ActionType::Check, ActionType::Call, ActionType::AllIn, ActionType::Fold})
        if (allowed(o, t)) { g.step({t}, pid); return; }
    throw std::logic_error("no fallback action");
}
