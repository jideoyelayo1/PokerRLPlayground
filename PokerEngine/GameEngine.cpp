#include "GameEngine.hpp"


void GameEngine::start_hand() {
    eliminateBustedPlayers();
    if (!canStartHand()) {
        hand_terminal_ = true;
        hand_winner_id_ = (activate_player_ids_.size() == 1 ? *activate_player_ids_.begin() : -1);
        hand_started_ = false;
        return;
    }

    if (reshuffleAlways_) initialiseDeck();

    sharedCards_.clear();
    pots_.clear();
    pot_ = 0;
    to_call_ = 0;
    street_ = Street::Preflop;

    hand_terminal_ = false;
    hand_winner_id_ = -1;
    hand_started_ = true;

    resetPlayersForNewHandAliveOnly();
    dealOutCardsToPlayersAliveOnly();

    currently_betting_player_ids_ = activate_player_ids_;

    if (big_blind_position_ == activate_player_ids_.end())
        big_blind_position_ = activate_player_ids_.begin();

    // BB posts
    current_betting_player_ = big_blind_position_;
    int bbId = *current_betting_player_;
    int bbPay = std::min(big_blind_, players_[bbId].getMoney());
    players_[bbId].betChips(bbPay);
    pot_ += bbPay;
    to_call_ = std::max(to_call_, players_[bbId].streetContrib());

    // SB posts (next)
    ++current_betting_player_;
    if (current_betting_player_ == activate_player_ids_.end())
        current_betting_player_ = activate_player_ids_.begin();

    int sbId = *current_betting_player_;
    int sbPay = std::min(small_blind_, players_[sbId].getMoney());
    players_[sbId].betChips(sbPay);
    pot_ += sbPay;

    // Preflop min raise = BB
    min_raise_ = big_blind_;

    // Next to act (UTG)
    ++current_betting_player_;
    if (current_betting_player_ == activate_player_ids_.end())
        current_betting_player_ = activate_player_ids_.begin();

    // Action closes on BB if no full raise
    end_action_it_ = iterOf(bbId);
    end_action_valid_ = (end_action_it_ != activate_player_ids_.end());

    // Rotate BB position for next hand
    ++big_blind_position_;
    if (big_blind_position_ == activate_player_ids_.end())
        big_blind_position_ = activate_player_ids_.begin();
}

Observation GameEngine::step(const Action& a, int viewerId) {
    if (!hand_started_) {
        // If not started, start automatically
        start_hand();
    }
    if (hand_terminal_ || activate_player_ids_.empty()) {
        return getObservation(viewerId);
    }

    // If current player cannot act, advance until someone can (or no one can)
    if (!canActNow(*current_betting_player_)) {
        advanceToNextPlayer();
        if (countCanActNow() == 0) {
            // everyone all-in or folded -> run out + showdown
            if (!handEndedByFolds()) runOutBoardToFive();
            showdownAndPayout();
            eliminateBustedPlayers();
            hand_terminal_ = true;
            // winner_id_ is ambiguous on split pots; keep -1 unless fold-win
            return getObservation(viewerId);
        }
    }

    // Apply single action by current player
    int acting_id = *current_betting_player_;
    applyAction(acting_id, a);

    // If hand ended by folds immediately
    if (handEndedByFolds()) {
        int w = lastActivePlayerId();
        showdownAndPayout();          // this pays pot to last active
        eliminateBustedPlayers();
        hand_terminal_ = true;
        hand_winner_id_ = w;
        return getObservation(viewerId);
    }

    // If everyone left is all-in (or folded), run out and showdown
    if (everyoneActiveIsAllInOrFolded()) {
        runOutBoardToFive();
        showdownAndPayout();
        eliminateBustedPlayers();
        hand_terminal_ = true;
        hand_winner_id_ = -1; // could be split; leave unknown
        return getObservation(viewerId);
    }

    // Move to next player for action
    advanceToNextPlayer();

    // If betting round is complete, advance street automatically (no extra actions here)
    // Completion condition mirrors runBettingRound's break:
    if (allNonAllInMatched() && end_action_valid_ && current_betting_player_ == end_action_it_) {
        // Advance street / or showdown if river complete
        if (street_ == Street::Preflop) {
            dealFlop();
            street_ = Street::Flop;
            startNewStreet();
        } else if (street_ == Street::Flop) {
            dealTurn();
            street_ = Street::Turn;
            startNewStreet();
        } else if (street_ == Street::Turn) {
            dealRiver();
            street_ = Street::River;
            startNewStreet();
        } else if (street_ == Street::River) {
            // showdown
            showdownAndPayout();
            eliminateBustedPlayers();
            hand_terminal_ = true;
            hand_winner_id_ = -1;
            return getObservation(viewerId);
        }
    }

    return getObservation(viewerId);
}

void GameEngine::reset_tournament(int startingMoney) {
    // reset all players to startingMoney
    for (auto& p : players_) {
        p.resetForNewHand();
        // set money directly via adjustMoney if you have it, otherwise implement a setter
        int cur = p.getMoney();
        p.adjustMoney(startingMoney - cur);
    }

    // everyone alive again
    activate_player_ids_.clear();
    for (int i = 0; i < numPlayers_; ++i) activate_player_ids_.push_back(i);

    // reset positions
    big_blind_position_ = activate_player_ids_.begin();

    // reset hand state
    sharedCards_.clear();
    pots_.clear();
    pot_ = 0;
    to_call_ = 0;
    street_ = Street::Preflop;
    hand_terminal_ = false;
    hand_winner_id_ = -1;
    hand_started_ = false;
}



std::array<Card,5> GameEngine::boardFixed() const {
    std::array<Card,5> b{ Card(), Card(), Card(), Card(), Card() };
    for (size_t i = 0; i < sharedCards_.size() && i < 5; ++i) b[i] = sharedCards_[i];
    return b;
}


// Utility: iterator to a seat id
std::list<int>::iterator GameEngine::iterOf(int pid) {
    for (auto it = activate_player_ids_.begin(); it != activate_player_ids_.end(); ++it)
        if (*it == pid) return it;
    return activate_player_ids_.end();
}

// Utility: in-hand / can-act
bool GameEngine::canActNow(int id) const {
    const auto& p = players_[id];
    return !p.hasFolded() && !p.isAllIn();
}

int GameEngine::countCanActNow() const {
    int n = 0;
    for (int id : activate_player_ids_) if (canActNow(id)) ++n;
    return n;
}

// Everyone who is not folded/all-in has matched to_call_
bool GameEngine::allNonAllInMatched() const {
    for (int id : activate_player_ids_) {
        const auto& p = players_[id];
        if (p.hasFolded()) continue;
        if (p.isAllIn())  continue;
        if (p.streetContrib() != to_call_) return false;
    }
    return true;
}



std::list<int>::iterator GameEngine::nextSeat(std::list<int>::iterator it) {
    if (activate_player_ids_.empty()) return activate_player_ids_.end();
    ++it;
    if (it == activate_player_ids_.end()) it = activate_player_ids_.begin();
    return it;
}


void GameEngine::initialiseDeck(){
    // Initialize deck
    deck_ = std::queue<Card>();
    std::vector<Card> intermeddiateDeck_;
    for(int s = 1; s <= 4; ++s) {
        for(int r = 2; r <= 14; ++r) {
            intermeddiateDeck_.emplace_back(static_cast<Suits>(s), static_cast<Ranks>(r));
        }
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // Shuffle deck
    std::shuffle(intermeddiateDeck_.begin(), intermeddiateDeck_.end(), std::default_random_engine(seed));

    for(const auto& card : intermeddiateDeck_) {
        deck_.push(card);
    }
}

void GameEngine::eliminateBustedPlayers() {
    if (activate_player_ids_.empty()) return;

    // If bb iterator invalid, reset it.
    if (big_blind_position_ == activate_player_ids_.end())
        big_blind_position_ = activate_player_ids_.begin();

    auto it = activate_player_ids_.begin();
    while (it != activate_player_ids_.end()) {
        int id = *it;
        if (players_[id].getMoney() <= 0) {
            // If we are about to erase the node big_blind_position_ points to,
            // advance big_blind_position_ first.
            if (it == big_blind_position_) {
                auto nxt = std::next(it);
                if (nxt == activate_player_ids_.end()) nxt = activate_player_ids_.begin();
                big_blind_position_ = nxt;
            }

            // Erase busted player
            it = activate_player_ids_.erase(it);
        } else {
            ++it;
        }
    }

    // If list became empty, keep iterator consistent
    if (activate_player_ids_.empty()) {
        big_blind_position_ = activate_player_ids_.end();
    } else if (big_blind_position_ == activate_player_ids_.end()) {
        big_blind_position_ = activate_player_ids_.begin();
    }
}

// Use only alive players when dealing
void GameEngine::dealOutCardsToPlayersAliveOnly() {
    if(deck_.size() < 2 * activate_player_ids_.size()) initialiseDeck();
    for (int id : activate_player_ids_) {
        std::vector<Card> pocketCards;
        // deck_ is a queue and always has enough cards here
        pocketCards.push_back(deck_.front()); deck_.pop();
        pocketCards.push_back(deck_.front()); deck_.pop();
        players_[id].giveCards(pocketCards);
    }
}

void GameEngine::resetPlayersForNewHandAliveOnly() {
    for (int id : activate_player_ids_) players_[id].resetForNewHand();
}

void GameEngine::burnCard() {
    if (deck_.empty()) throw std::runtime_error("Deck empty on burn");
    deck_.pop();
}

void GameEngine::dealFlop() {
    burnCard();
    for (int i = 0; i < 3; ++i) {
        if (deck_.empty()) throw std::runtime_error("Deck empty on flop");
        sharedCards_.push_back(deck_.front());
        deck_.pop();
    }
}

void GameEngine::dealTurn() {
    burnCard();
    if (deck_.empty()) throw std::runtime_error("Deck empty on turn");
    sharedCards_.push_back(deck_.front());
    deck_.pop();
}

void GameEngine::dealRiver() {
    burnCard();
    if (deck_.empty()) throw std::runtime_error("Deck empty on river");
    sharedCards_.push_back(deck_.front());
    deck_.pop();
}

// ---- Count who is still in hand (not folded)
int GameEngine::countActiveInHand() const {
    int n = 0;
    for (int id : activate_player_ids_) {
        if (!players_[id].hasFolded()) ++n;
    }
    return n;
}

int GameEngine::lastActivePlayerId() const {
    for (int id : activate_player_ids_) {
        if (!players_[id].hasFolded()) return id;
    }
    return -1;
}


// ---- If all remaining players are all-in, no more betting possible.
bool GameEngine::everyoneActiveIsAllInOrFolded() const {
    for (int id : activate_player_ids_) {
        const auto& p = players_[id];
        if (p.hasFolded()) continue;
        if (!p.isAllIn()) return false;
    }
    return true; // everyone not folded is all-in (or nobody)
}
    
void GameEngine::startNewStreet() {
    to_call_ = 0;
    for (int id : activate_player_ids_) players_[id].resetStreetBet();

    // NLH min-raise resets each street to BB
    min_raise_ = big_blind_;

    // Your simplified "end action" seat: big blind position (postflop)
    end_action_it_ = big_blind_position_;
    end_action_valid_ = (end_action_it_ != activate_player_ids_.end());

    current_betting_player_ = big_blind_position_;
    advanceToNextPlayer(); // first to act
}

void GameEngine::startNewRound() {
    eliminateBustedPlayers();
    if (!canStartHand()) return;

    if (reshuffleAlways_) initialiseDeck();

    sharedCards_.clear();
    pots_.clear();
    pot_ = 0;
    to_call_ = 0;
    street_ = Street::Preflop;

    resetPlayersForNewHandAliveOnly();
    dealOutCardsToPlayersAliveOnly();

    currently_betting_player_ids_ = activate_player_ids_;

    if (big_blind_position_ == activate_player_ids_.end())
        big_blind_position_ = activate_player_ids_.begin();

    // BB posts
    current_betting_player_ = big_blind_position_;
    int bbId = *current_betting_player_;
    int bbPay = std::min(big_blind_, players_[bbId].getMoney());
    players_[bbId].betChips(bbPay);
    pot_ += bbPay;
    to_call_ = std::max(to_call_, players_[bbId].streetContrib());

    // SB posts (next)
    ++current_betting_player_;
    if (current_betting_player_ == activate_player_ids_.end())
        current_betting_player_ = activate_player_ids_.begin();

    int sbId = *current_betting_player_;
    int sbPay = std::min(small_blind_, players_[sbId].getMoney());
    players_[sbId].betChips(sbPay);
    pot_ += sbPay;

    // Preflop: minimum raise is BB
    min_raise_ = big_blind_;

    // Next to act (UTG)
    ++current_betting_player_;
    if (current_betting_player_ == activate_player_ids_.end())
        current_betting_player_ = activate_player_ids_.begin();

    // Preflop action closes on BB if no raises
    end_action_it_ = iterOf(bbId);
    end_action_valid_ = (end_action_it_ != activate_player_ids_.end());

    // Rotate BB position for next hand (your simplified logic)
    ++big_blind_position_;
    if (big_blind_position_ == activate_player_ids_.end())
        big_blind_position_ = activate_player_ids_.begin();
}



void GameEngine::runOutBoardToFive() {
    while ((int)sharedCards_.size() < 5) {
        if (sharedCards_.size() == 0) dealFlop();
        else if (sharedCards_.size() == 3) dealTurn();
        else if (sharedCards_.size() == 4) dealRiver();
        else throw std::runtime_error("Invalid board size in runOutBoardToFive");
    }
}


void GameEngine::removeFromBetting(int playerId) {
    currently_betting_player_ids_.remove(playerId);
}


int GameEngine::computeRaiseTo(int playerId, RaiseSize sz) const {
    // raise-to amount in terms of total street contribution (not delta)
    // base is current to_call_ (i.e., raising above call)
    int callTarget = to_call_;
    int add = 0;

    if (sz == RaiseSize::BB) {
        add = big_blind_;
    } else if (sz == RaiseSize::HalfPot) {
        add = std::max(1, pot_ / 2);
    } else { // Pot
        add = std::max(1, pot_);
    }

    return callTarget + add;
}

void GameEngine::applyAction(int id, const Action& a) {
    if (!isActionLegal(id, a)) {
        if(DEBUG){
            std::cout << "Illegal action attempted by player " << id << std::endl;
            std::cout << "Action type: " << static_cast<int>(a.type) << std::endl;
        }

        throw std::runtime_error("Illegal action for player " + std::to_string(id));
    }

    auto& p = players_[id];
    int need = playerToCall(id);

    switch (a.type) {
        case ActionType::Fold:
            p.fold();
            return;

        case ActionType::Check:
            // legal only when need==0
            return;

        case ActionType::Call: {
            int pay = std::min(need, p.getMoney());
            p.betChips(pay);
            pot_ += pay;
            return;
        }

        case ActionType::Raise: {
            int prevToCall = to_call_;
            int raiseTo = computeRaiseTo(id, a.raiseSize);
            int wantDelta = raiseTo - p.streetContrib();

            int pay = std::min(wantDelta, p.getMoney());
            p.betChips(pay);
            pot_ += pay;

            int newToCall = std::max(to_call_, p.streetContrib());
            if (newToCall > to_call_) {
                int raiseInc = newToCall - prevToCall;
                bool fullRaise = (raiseInc >= min_raise_);
                to_call_ = newToCall;

                if (fullRaise) {
                    min_raise_ = raiseInc;
                    auto it = iterOf(id);
                    if (it != activate_player_ids_.end()) {
                        end_action_it_ = it;
                        end_action_valid_ = true;
                    }
                }
            }
            return;
        }

        case ActionType::AllIn: {
            int prevToCall = to_call_;

            int pay = p.getMoney();
            p.betChips(pay);
            pot_ += pay;

            int newToCall = std::max(to_call_, p.streetContrib());
            if (newToCall > to_call_) {
                int raiseInc = newToCall - prevToCall;
                bool fullRaise = (raiseInc >= min_raise_);
                to_call_ = newToCall;

                if (fullRaise) {
                    min_raise_ = raiseInc;
                    auto it = iterOf(id);
                    if (it != activate_player_ids_.end()) {
                        end_action_it_ = it;
                        end_action_valid_ = true;
                    }
                }
            }
            return;
        }
    }
}


bool GameEngine::isSettled() const {
    // settled if every player who can still act has matched to_call_
    for (int id : activate_player_ids_) {
        const auto& p = players_[id];
        if (p.hasFolded()) continue;
        if (p.isAllIn())  continue; // all-in players don't need to match
        if (p.streetContrib() != to_call_) return false;
    }
    return true;
}

int GameEngine::countCanAct() const {
    int n = 0;
    for (int id : activate_player_ids_) {
        const auto& p = players_[id];
        if (!p.hasFolded() && !p.isAllIn()) ++n;
    }
    return n;
}

void GameEngine::advanceToNextPlayer() {
    do {
        ++current_betting_player_;
        if (current_betting_player_ == activate_player_ids_.end())
            current_betting_player_ = activate_player_ids_.begin();

        int id = *current_betting_player_;
        if (!players_[id].hasFolded() && !players_[id].isAllIn()) break;

        // If nobody can act, break to avoid infinite loop
        if (countCanActNow() == 0) break;

    } while (true);
}

void GameEngine::runBettingRound() {
    if (countCanActNow() == 0) return;
    if (!end_action_valid_) return;

    if (!canActNow(*current_betting_player_)) advanceToNextPlayer();

    int safety = 0;
    const int SAFETY_MAX = 100000;

    while (true) {
        if (++safety > SAFETY_MAX) throw std::runtime_error("Betting loop safety break");

        // End condition: everyone matched and we're at the end-of-action seat
        if (allNonAllInMatched() && current_betting_player_ == end_action_it_) break;

        int id = *current_betting_player_;
        if (!canActNow(id)) {
            advanceToNextPlayer();
            if (countCanActNow() == 0) break;
            continue;
        }

        Action act{ ActionType::Check, RaiseSize::BB };
        auto it = controllers_.find(id);
        if (it != controllers_.end()) act = it->second(*this, id);
        else {
            act.type = (playerToCall(id) > 0 ? ActionType::Call : ActionType::Check);
        }

        // applyAction will throw if illegal
        applyAction(id, act);

        advanceToNextPlayer();
        if (countCanActNow() == 0) break;
    }
}


std::vector<Pot> GameEngine::buildSidePotsFromTotal() const {
    struct Entry { int id; int contrib; };

    // All contributors (including folded) define slicing levels
    std::vector<Entry> all;
    all.reserve(players_.size());
    for (int id : activate_player_ids_) {
        int c = players_[id].totalContrib();
        if (c > 0) all.push_back({id, c});
    }
    if (all.empty()) return {};

    std::sort(all.begin(), all.end(), [](const Entry& a, const Entry& b){
        return a.contrib < b.contrib;
    });

    std::vector<Pot> pots;
    int prev = 0;

    // unique levels ascending
    for (size_t i = 0; i < all.size(); ) {
        int level = all[i].contrib;
        int slice = level - prev;
        if (slice <= 0) { ++i; continue; }

        // contributors: everyone with totalContrib >= level (including folded)
        int contributors = 0;
        for (int id : activate_player_ids_) {
            if (players_[id].totalContrib() >= level) ++contributors;
        }

        Pot pot;
        pot.amount = slice * contributors;

        // eligible: NOT folded + totalContrib >= level
        for (int id : activate_player_ids_) {
            const auto& p = players_[id];
            if (p.hasFolded()) continue;
            if (p.totalContrib() >= level) pot.eligible.push_back(id);
        }

        // It's possible pot.eligible is empty if everyone at/above level folded (rare but possible in malformed states).
        // In real poker, if everyone folded you'd have ended earlier.
        pots.push_back(std::move(pot));

        prev = level;
        while (i < all.size() && all[i].contrib == level) ++i;
    }

    return pots;
}


void GameEngine::awardPotToWinners(const Pot& pot, const std::vector<int>& winners) {
    if (winners.empty()) return;
    int share = pot.amount / (int)winners.size();
    int rem   = pot.amount % (int)winners.size();

    for (int id : winners) players_[id].awardChips(share);
    if (rem > 0) players_[winners[0]].awardChips(rem);
}

void GameEngine::showdownAndPayout() {
    // If hand ended by folds, pay whole pot to last active player
    if (handEndedByFolds()) {
        int winner = lastActivePlayerId();
        if (winner >= 0) players_[winner].awardChips(pot_);
        return;
    }

    // Ensure board complete
    if ((int)sharedCards_.size() != 5) throw std::runtime_error("Showdown requires 5 board cards");

    // Build pots now (recommended: only once at showdown)
    pots_ = buildSidePotsFromTotal();

    // Precompute each player's HandRank (only if not folded)
    std::unordered_map<int, HandRank> ranks;
    ranks.reserve(players_.size());
    for (int id : activate_player_ids_) {
        if (players_[id].hasFolded()) continue;
        std::vector<Card> seven;
        seven.reserve(7);
        auto ph = players_[id].getPocketHand();
        if (ph.size() != 2) throw std::runtime_error("Pocket hand missing at showdown");
        seven.push_back(ph[0]);
        seven.push_back(ph[1]);
        for (auto& c : sharedCards_) seven.push_back(c);
        ranks[id] = bestHandOf7(seven);
    }

    // For each pot: among eligible, choose best rank (ties split)
    for (const auto& pot : pots_) {
        if (pot.amount <= 0) continue;

        bool hasBest = false;
        HandRank best{};
        std::vector<int> winners;

        for (int id : pot.eligible) {
            auto it = ranks.find(id);
            if (it == ranks.end()) continue; // folded or missing

            const HandRank& r = it->second;
            if (!hasBest) {
                best = r;
                hasBest = true;
                winners = {id};
            } else if (best < r) {
                best = r;
                winners = {id};
            } else if (equalRank(best, r)) {
                winners.push_back(id);
            }
        }

        awardPotToWinners(pot, winners);
    }
}

bool GameEngine::isActionLegal(int playerId, const Action& a) const {
    if (!canActNow(playerId)) return false;
    int need = playerToCall(playerId);
    const auto& p = players_[playerId];

    switch (a.type) {
        case ActionType::Fold:  return true;
        case ActionType::Check: return (need == 0);
        case ActionType::Call:  return (need > 0) && (p.getMoney() > 0);
        case ActionType::AllIn: return (p.getMoney() > 0);
        case ActionType::Raise: {
            if (p.getMoney() <= 0) return false;
            int max_to = maxRaiseTo(playerId);
            int min_to = to_call_ + min_raise_;
            if (max_to < min_to) return false;

            // If controller uses RaiseSize, computed raiseTo must be within [min_to, max_to]
            int raiseTo = computeRaiseTo(playerId, a.raiseSize);
            return (raiseTo >= min_to && raiseTo <= max_to);
        }
    }
    return false;
}

GameEngine::GameEngine(int numPlayers, int startingMoney, int big_blind, int small_blind) {
    // Initialize players
    numPlayers_ = numPlayers;
    for(int i = 0; i < numPlayers; ++i) {
        players_.emplace_back(i, startingMoney);
        activate_player_ids_.push_back(i);
    }
    // Set big blind position
    big_blind_position_ = activate_player_ids_.begin();
    big_blind_ = big_blind;
    small_blind_ = small_blind;

}


void GameEngine::playOneHand() {
    startNewRound();
    if (!canStartHand()) return;

    runBettingRound();
    if (handEndedByFolds()) { showdownAndPayout(); eliminateBustedPlayers(); return; }

    if (everyoneActiveIsAllInOrFolded()) {
        runOutBoardToFive();
        showdownAndPayout();
        eliminateBustedPlayers();
        return;
    }

    dealFlop();
    startNewStreet();
    runBettingRound();
    if (handEndedByFolds()) { showdownAndPayout(); eliminateBustedPlayers(); return; }
    if (everyoneActiveIsAllInOrFolded()) {
        runOutBoardToFive();
        showdownAndPayout();
        eliminateBustedPlayers();
        return;
    }

    dealTurn();
    startNewStreet();
    runBettingRound();
    if (handEndedByFolds()) { showdownAndPayout(); eliminateBustedPlayers(); return; }
    if (everyoneActiveIsAllInOrFolded()) {
        runOutBoardToFive();
        showdownAndPayout();
        eliminateBustedPlayers();
        return;
    }

    dealRiver();
    startNewStreet();
    runBettingRound();

    showdownAndPayout();
    eliminateBustedPlayers();
}

void GameEngine::printAllPlayersStatus() const {
    auto suitToString = [](Suits s) {
        switch (s) {
            case Suits::Hearts:   return "H";
            case Suits::Diamonds: return "D";
            case Suits::Clubs:    return "C";
            case Suits::Spades:   return "S";
            default:              return "E";
        }
    };
    for (const auto& player : players_) {
        std::cout << "Player " << player.getId() 
                    << " | Active: " << (player.isActiveInHand() ? "Yes" : "No")
                    << " | Game Over: " << (player.gameOver() ? "Yes" : "No")
                    << " | Money: " << player.getMoney() 
                    << " | Folded: " << (player.hasFolded() ? "Yes" : "No") 
                    << " | AllIn: " << (player.isAllIn() ? "Yes" : "No") 
                    << " | Street Contribution: " << player.getStreetContribution() 
                    << " | Total Contribution: " << player.totalContrib();
        std::cout << " | Pocket Hand: ";
        for (const auto& card : player.getPocketHand()) {
            if(card.rank == Ranks::Empty) {
                std::cout << "Empty ";
                continue;
            }
            std::cout << suitToString(card.suit) << rint(card.rank) << " ";
        }
        std::cout << std::endl;
    }
}


std::vector<ActionSpec> GameEngine::legalActions(int playerId) const {
    std::vector<ActionSpec> out;
    out.reserve(5);

    // If can't act, nothing is legal
    if (!canActNow(playerId)) return out;

    const auto& p = players_[playerId];
    int need = playerToCall(playerId);
    int max_to = maxRaiseTo(playerId);

    // Fold always allowed if you can act
    out.push_back(ActionSpec{ActionType::Fold, need, 0, 0, true});

    // Check only if need==0
    out.push_back(ActionSpec{ActionType::Check, need, 0, 0, (need == 0)});

    // Call allowed if need>0 and you have >0 chips (if you have 0 you'd be all-in already)
    // We still allow "call" when need>0; amount is min(need, stack) at execution, but legality is "you can call all-in".
    bool can_call = (need > 0) && (p.getMoney() > 0);
    out.push_back(ActionSpec{ActionType::Call, need, 0, 0, can_call});

    // All-in allowed if you have >0 chips
    bool can_allin = (p.getMoney() > 0);
    out.push_back(ActionSpec{ActionType::AllIn, need, 0, 0, can_allin});

    // Raise: only if you have chips and your max_to > to_call_ (i.e., can increase price)
    // Minimum "raise-to" is to_call_ + min_raise_ (NLH)
    int min_to = to_call_ + min_raise_;
    bool can_raise = (p.getMoney() > 0) && (max_to > to_call_) && (max_to >= min_to);
    out.push_back(ActionSpec{ActionType::Raise, need, min_to, max_to, can_raise});

    return out;
}

Observation GameEngine::getObservation(int viewerId) const {
    Observation o;
    o.viewer_id = viewerId;

    o.street = static_cast<int>(street_);
    o.pot_total = pot_;
    o.to_call = to_call_;
    o.min_raise = min_raise_;
    o.board = boardFixed();
    o.terminal = hand_terminal_;
    o.winner_id = hand_winner_id_;


    // Safe current player
    o.hand_in_progress = !activate_player_ids_.empty();
    o.current_player_id = (o.hand_in_progress ? *current_betting_player_ : -1);

    // Viewer hole cards only
    o.hole = (!players_[viewerId].hasFolded() ? players_[viewerId].holeFixed()
                                            : std::array<Card,2>{Card(), Card()});

    int n = (int)players_.size();
    o.alive.assign(n, 0);
    o.in_hand.assign(n, 0);
    o.can_act.assign(n, 0);

    o.stack.resize(n);
    o.street_contrib.resize(n);
    o.total_contrib.resize(n);
    o.folded.resize(n);
    o.all_in.resize(n);

    // Mark alive seats from activate_player_ids_
    for (int id : activate_player_ids_) o.alive[id] = 1;

    for (int i = 0; i < n; ++i) {
        o.stack[i] = players_[i].getMoney();
        o.street_contrib[i] = players_[i].streetContrib();
        o.total_contrib[i]  = players_[i].totalContrib();
        o.folded[i] = (uint8_t)players_[i].hasFolded();
        o.all_in[i] = (uint8_t)players_[i].isAllIn();

        o.in_hand[i] = (uint8_t)(o.alive[i] && !players_[i].hasFolded());
        o.can_act[i] = (uint8_t)(o.alive[i] && canActNow(i));
    }

    o.viewer_to_call = std::max(0, to_call_ - players_[viewerId].streetContrib());
    o.legal = legalActions(viewerId);
    return o;
}

bool GameEngine::isAllowed(const std::vector<ActionSpec>& legal, ActionType t) {
    for (const auto& a : legal) if (a.type == t) return a.allowed;
    return false;
}

HandRank GameEngine::bestHandOf7(const std::vector<Card>& seven) {
    if (seven.size() != 7) throw std::invalid_argument("bestHandOf7 needs 7 cards");
    HandRank best{};
    bool has = false;

    for (int a = 0; a < 7; ++a) {
        for (int b = a + 1; b < 7; ++b) {
            std::array<Card,5> hand{};
            int k = 0;
            for (int i = 0; i < 7; ++i) {
                if (i == a || i == b) continue;
                hand[k++] = seven[i];
            }
            HandRank cur = evaluate5(hand);
            if (!has || best < cur) { best = cur; has = true; }
        }
    }
    return best;
}

