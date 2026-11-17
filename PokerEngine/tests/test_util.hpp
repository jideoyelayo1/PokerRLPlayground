#pragma once
// Minimal test helpers shared by the engine tests (no test framework dependency).
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "../GameEngine.hpp"

inline int g_checks = 0, g_failures = 0;

#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_failures; \
    std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

inline int finish(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

// "As" -> Card(Spades, Ace). Ranks 2-9,T,J,Q,K,A; suits h,d,c,s.
inline Card card(const std::string& s) {
    static const char* ranks = "23456789TJQKA";
    const char* r = std::strchr(ranks, s.at(0));
    if (!r || s.size() != 2) throw std::invalid_argument("bad card " + s);
    Suits suit = s[1] == 'h' ? Suits::Hearts : s[1] == 'd' ? Suits::Diamonds
               : s[1] == 'c' ? Suits::Clubs : s[1] == 's' ? Suits::Spades : Suits::Empty;
    if (suit == Suits::Empty) throw std::invalid_argument("bad suit " + s);
    return Card(suit, static_cast<Ranks>(2 + (r - ranks)));
}

inline std::vector<Card> cards(const std::string& list) {
    std::istringstream in(list);
    std::vector<Card> out;
    for (std::string t; in >> t;) out.push_back(card(t));
    return out;
}

inline bool sameRank(const HandRank& a, const HandRank& b) { return !(a < b) && !(b < a); }
