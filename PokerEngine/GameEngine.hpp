#pragma once

#include "PokerProperties.hpp"
#include "Player.hpp"

class GameEngine{
    public:
        using Controller = std::function<Action(const GameEngine&, int playerId)>;
    private:
        std::vector<Player> players_;
        std::vector<Card> sharedCards_;
        std::queue<Card> deck_;
        int numPlayers_;
        std::list<int> activate_player_ids_, currently_betting_player_ids_;
        std::list<int>::iterator current_betting_player_;
        std::list<int>::iterator big_blind_position_;
        std::list<int>::iterator button_position_;
        bool reshuffleAlways_ = true;
        // Deck RNG: clock-seeded by default (as before); call seed() for repeatable deals.
        std::mt19937_64 rng_{static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count())};
        int big_blind_, small_blind_;
        std::vector<Pot> pots_;

        // Hand termination state (for Python RL)
        bool hand_terminal_ = false;
        int  hand_winner_id_ = -1;
        bool hand_started_ = false;


        Card emptyCard() const { return Card(); } // suit/rank Empty by default

        std::array<Card,5> boardFixed() const;

        // Betting-round termination (end of action)
        std::list<int>::iterator end_action_it_{};
        bool end_action_valid_ = false;

        // Minimum raise tracking (NLH-style)
        int min_raise_ = 0;         // minimum raise increment for this street

        // Utility: iterator to a seat id
        std::list<int>::iterator iterOf(int pid);

        // Utility: in-hand / can-act
        bool canActNow(int id) const;

        int countCanActNow() const;

        // Everyone who is not folded/all-in has matched to_call_
        bool allNonAllInMatched() const;


        enum class Street { Preflop, Flop, Turn, River, Showdown };
        Street street_ = Street::Preflop;   

        const bool DEBUG = false;

        std::list<int>::iterator nextSeat(std::list<int>::iterator it);


        void initialiseDeck();

        void eliminateBustedPlayers();

        bool canStartHand() const {
            return activate_player_ids_.size() >= 2;
        }

        // Use only alive players when dealing
        void dealOutCardsToPlayersAliveOnly();

        void resetPlayersForNewHandAliveOnly();

        void burnCard();

        void dealFlop();

        void dealTurn();

        void dealRiver();

        // ---- Count who is still in hand (not folded)
        int countActiveInHand() const;

        int lastActivePlayerId() const;

        bool handEndedByFolds() const {
            return countActiveInHand() <= 1;
        }

        // ---- If all remaining players are all-in, no more betting possible.
        bool everyoneActiveIsAllInOrFolded() const;
            
        void startNewStreet();

        void startNewRound();



        void runOutBoardToFive();

        

        std::unordered_map<int, Controller> controllers_;

        int pot_ = 0;              // track pot
        int to_call_ = 0;          // highest streetContribution among active players

        Player& P(int id) { return players_.at(id); }
        const Player& P(int id) const { return players_.at(id); }

        int activeCount() const { return (int)currently_betting_player_ids_.size(); }

        void removeFromBetting(int playerId);

        int playerToCall(int playerId) const {
            return std::max(0, to_call_ - P(playerId).getStreetContribution());
        }

        int maxRaiseTo(int playerId) const {
            // raise-to maximum = current street contribution + remaining stack
            return players_[playerId].streetContrib() + players_[playerId].getMoney();
        }

        int computeRaiseTo(int playerId, RaiseSize sz) const;

        void applyAction(int id, const Action& a);

        int getCurrentPlayerId() const {
            return *current_betting_player_;
        }

        bool isSettled() const;

        int countCanAct() const ;

        void advanceToNextPlayer();

        void runBettingRound();

        const std::vector<Pot>& getPots() const { return pots_; }

        std::vector<Pot> buildSidePotsFromTotal() const;

        
        // Compare two HandRank (your operator< already exists)
        static bool equalRank(const HandRank& a, const HandRank& b) {
            return (a.cat == b.cat) && (a.kickers == b.kickers);
        }

        void awardPotToWinners(const Pot& pot, const std::vector<int>& winners);

        void showdownAndPayout();

        bool isActionLegal(int playerId, const Action& a) const;

        public:
        GameEngine(int numPlayers, int startingMoney = 200, int big_blind = 2, int small_blind = 1);

        void setController(int playerId, Controller c) {
            controllers_[playerId] = std::move(c);
        }

        int potTotal() const { return pot_; }
        int toCall() const { return to_call_; }
        int minRaise() const { return min_raise_; }
        int currentPlayer() const { return *current_betting_player_; }
        int streetInt() const { return static_cast<int>(street_); }
        std::array<Card,5> boardCardsFixed() const { return boardFixed(); }

        bool handInProgress() const { return !activate_player_ids_.empty(); }

        int currentPlayerSafe() const {
            if (activate_player_ids_.empty()) return -1;
            return *current_betting_player_;
        }

        int alivePlayers() const { return (int)activate_player_ids_.size(); }

        std::vector<int> aliveIds() const {
            return std::vector<int>(activate_player_ids_.begin(), activate_player_ids_.end());
        }
        
        void playOneHand();

        void printAllPlayersStatus() const;

        int getPlayerCount () const { return numPlayers_; }

        std::vector<ActionSpec> legalActions(int playerId) const;

        Observation getObservation(int viewerId) const;

        static bool isAllowed(const std::vector<ActionSpec>& legal, ActionType t);

        static HandRank bestHandOf7(const std::vector<Card>& seven);

        void reset_tournament(int startingMoney);   // resets stacks + alive list
        void start_hand();                          // starts a new hand (posts blinds, deals, sets current player)
        bool hand_terminal() const { return hand_terminal_; }
        int  winner_id() const { return hand_winner_id_; }

        // Single-step advance: apply exactly ONE action for current player.
        // Returns Observation for `viewerId` (usually the same as the next player who will act).
        Observation step(const Action& a, int viewerId);

        // Re-seed the deck RNG; every later shuffle follows from this seed.
        void seed(uint64_t s) { rng_.seed(s); }

        // Convenience
        int bigBlind() const { return big_blind_; }
        int smallBlind() const { return small_blind_; }

};

