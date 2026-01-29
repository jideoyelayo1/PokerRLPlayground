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

#include "PokerProperties.hpp"

class Player {
    private:
        int playerId_;
        std::vector<Card> pocketHand_, bestCardCombination_;
        bool folded = false;
        int money;

        bool allIn_ = false;

        int streetContribution_ = 0;
        int totalContribution_  = 0;


        std::vector<Card> findHighestCombination(const std::vector<Card>& allCards);

    public:
        Player(int playerId, int startingMoney = 1000) :  playerId_(playerId), money(startingMoney) {}

        int  streetContrib() const { return streetContribution_; }
        int  totalContrib()  const { return totalContribution_; }
        bool isActiveInHand() const { return !folded; } // folded cannot win pots

        bool gameOver() const {
            return money <= 0;
        }

        int stack() const { return money; }
        bool isFolded() const { return folded; }
        bool isAllIn() const { return allIn_; }

        std::array<Card,2> holeFixed() const;

        void fold();

        void betChips(int delta);

        void resetForNewHand();

        void awardChips(int amount);
        
        bool canAct() const { return !folded && !allIn_; }

        void roundReset();
        void giveCards(const std::vector<Card>& cards);
        void tossCards();

        bool hasFolded() const {
            return folded;
        }
        int getId() const {
            return playerId_;
        }
        int getMoney() const {
            return money;
        }
        void adjustMoney(int amount);

        void resetStreetBet() { streetContribution_ = 0; }

        int getStreetContribution() const { return streetContribution_; }

        std::vector<Card> getBestCombination(const std::vector<Card>& sharedCards);

        void printPocketHand() const;

        std::vector<Card> getPocketHand() const {
            return pocketHand_;
        }

};



