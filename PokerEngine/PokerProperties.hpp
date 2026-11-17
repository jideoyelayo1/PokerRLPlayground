#pragma once

#include <stdio.h>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <list>
#include <queue>
#include <random>  
#include <chrono>  
#include <unordered_map>
#include <functional>

enum class ActionType { Fold, Check, Call, Raise, AllIn };
enum class RaiseSize { BB, HalfPot, Pot };

struct Pot {
    int amount = 0;
    std::vector<int> eligible; // ids who can win this pot
};

struct Action {
    ActionType type{};
    RaiseSize raiseSize{RaiseSize::BB}; // only used if type==Raise
};

enum class Suits {
    Empty = 0,
    Hearts,
    Diamonds,
    Clubs,
    Spades
};

enum class Ranks {
    Empty = 0,
    Two = 2, Three = 3, Four = 4, Five = 5, Six = 6, Seven = 7,
    Eight = 8, Nine = 9, Ten = 10, Jack = 11, Queen = 12, King = 13, Ace = 14
};

class Card{
    public:
        Suits suit;
        Ranks rank;

        Card() : suit(Suits::Empty), rank(Ranks::Empty) {}
        Card(Suits s, Ranks r) : suit(s), rank(r) {}
};

// legalActions lists Fold, Check, Call, AllIn, then one Raise per RaiseSize (BB, HalfPot, Pot).
// `allowed` is exactly whether step() accepts that action, raise size included.
struct ActionSpec {
    ActionType type{};
    int call_amount = 0;      // chips needed to call (0 => check)
    int min_raise_to = 0;     // for Raise: minimum "raise-to" street contribution
    int max_raise_to = 0;     // for Raise: maximum "raise-to" (all-in)
    bool allowed = false;
    RaiseSize raise_size{RaiseSize::BB};  // for Raise: the size this entry is for
    int raise_to = 0;         // for Raise: the street contribution this size raises to
};

struct Observation {
    int viewer_id = -1;

    int street = 0;
    int current_player_id = -1;
    int button = -1;                // seat with the dealer button; -1 before the first hand after a reset

    // Hand / episode flags
    bool hand_in_progress = false;
    bool terminal = false;          // true at end of hand (showdown done or everyone folded)
    int winner_id = -1;             // optional (only if terminal and single winner)

    int pot_total = 0;
    int to_call = 0;
    int min_raise = 0;

    std::array<Card, 2> hole{};
    std::array<Card, 5> board{};

    // Seat masks (size == players_.size())
    std::vector<uint8_t> alive;     // in activate_player_ids_ (has chips and still in game)
    std::vector<uint8_t> in_hand;   // not folded (still contesting pots)
    std::vector<uint8_t> can_act;   // not folded and not all-in

    std::vector<int> stack;
    std::vector<int> street_contrib;
    std::vector<int> total_contrib;
    std::vector<uint8_t> folded;
    std::vector<uint8_t> all_in;

    int viewer_to_call = 0;
    std::vector<ActionSpec> legal;
};

enum class HandCat : int {
    HighCard = 1,
    OnePair,
    TwoPair,
    Trips,
    Straight,
    Flush,
    FullHouse,
    Quads,
    StraightFlush
};

struct HandRank {
    HandCat cat{};
    // kicker vector, high-to-low, used for tie-breaks within same category
    std::array<int, 5> kickers{};
    // the actual 5 cards of this best hand (sorted by rank desc, mostly for readability)
    std::array<Card, 5> cards{};

    bool operator<(const HandRank& o) const {
        if (cat != o.cat) return static_cast<int>(cat) < static_cast<int>(o.cat);
        return kickers < o.kickers; // lexicographic
    }
};

static int rint(Ranks r) { return static_cast<int>(r); }

static bool cardRankDesc(const Card& a, const Card& b) {
    return rint(a.rank) > rint(b.rank);
}

// Evaluate exactly 5 cards -> HandRank
static HandRank evaluate5(std::array<Card,5> hand) {
    std::sort(hand.begin(), hand.end(), cardRankDesc);

    // counts by rank (2..14)
    std::array<int, 15> cnt{}; cnt.fill(0);
    for (auto& c : hand) cnt[rint(c.rank)]++;

    bool flush = true;
    for (int i = 1; i < 5; ++i) flush = flush && (hand[i].suit == hand[0].suit);

    // collect distinct ranks (desc)
    std::vector<int> uniq;
    uniq.reserve(5);
    for (auto& c : hand) {
        int rr = rint(c.rank);
        if (uniq.empty() || uniq.back() != rr) uniq.push_back(rr);
    }
    // uniq currently in hand order, may have duplicates removed but still desc

    auto isStraight = [&](int& straightHigh) -> bool {
        // For a straight you need 5 distinct ranks
        std::vector<int> u;
        u.reserve(5);
        for (int rr = 14; rr >= 2; --rr) if (cnt[rr]) u.push_back(rr);

        if (u.size() != 5) return false;
        // wheel A-2-3-4-5
        if (u[0] == 14 && u[1] == 5 && u[2] == 4 && u[3] == 3 && u[4] == 2) {
            straightHigh = 5;
            return true;
        }
        // normal
        if (u[0] - u[4] == 4) {
            straightHigh = u[0];
            return true;
        }
        return false;
    };

    int straightHigh = 0;
    bool straight = isStraight(straightHigh);

    // Build groups (count, rank) sorted by count desc, rank desc
    std::vector<std::pair<int,int>> groups;
    groups.reserve(5);
    for (int rr = 14; rr >= 2; --rr) if (cnt[rr]) groups.push_back({cnt[rr], rr});
    std::sort(groups.begin(), groups.end(), [](auto a, auto b){
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    });

    HandRank out;
    out.cards = hand;

    // Straight flush
    if (straight && flush) {
        out.cat = HandCat::StraightFlush;
        out.kickers = {straightHigh, 0,0,0,0};
        return out;
    }

    // Quads
    if (groups[0].first == 4) {
        int quad = groups[0].second;
        int kicker = groups[1].second;
        out.cat = HandCat::Quads;
        out.kickers = {quad, kicker, 0,0,0};
        return out;
    }

    // Full house
    if (groups[0].first == 3 && groups[1].first == 2) {
        int trips = groups[0].second;
        int pair  = groups[1].second;
        out.cat = HandCat::FullHouse;
        out.kickers = {trips, pair, 0,0,0};
        return out;
    }

    // Flush
    if (flush) {
        out.cat = HandCat::Flush;
        out.kickers = {rint(hand[0].rank), rint(hand[1].rank), rint(hand[2].rank), rint(hand[3].rank), rint(hand[4].rank)};
        return out;
    }

    // Straight
    if (straight) {
        out.cat = HandCat::Straight;
        out.kickers = {straightHigh, 0,0,0,0};
        return out;
    }

    // Trips
    if (groups[0].first == 3) {
        int trips = groups[0].second;
        // remaining kickers (high-to-low)
        std::vector<int> ks;
        for (size_t i = 1; i < groups.size(); ++i) ks.push_back(groups[i].second);
        out.cat = HandCat::Trips;
        out.kickers = {trips, ks[0], ks[1], 0,0};
        return out;
    }

    // Two pair
    if (groups[0].first == 2 && groups[1].first == 2) {
        int hiPair = std::max(groups[0].second, groups[1].second);
        int loPair = std::min(groups[0].second, groups[1].second);
        int kicker = groups[2].second;
        out.cat = HandCat::TwoPair;
        out.kickers = {hiPair, loPair, kicker, 0,0};
        return out;
    }

    // One pair
    if (groups[0].first == 2) {
        int pair = groups[0].second;
        std::vector<int> ks;
        for (size_t i = 1; i < groups.size(); ++i) ks.push_back(groups[i].second);
        out.cat = HandCat::OnePair;
        out.kickers = {pair, ks[0], ks[1], ks[2], 0};
        return out;
    }

    // High card
    out.cat = HandCat::HighCard;
    out.kickers = {rint(hand[0].rank), rint(hand[1].rank), rint(hand[2].rank), rint(hand[3].rank), rint(hand[4].rank)};
    return out;
}


