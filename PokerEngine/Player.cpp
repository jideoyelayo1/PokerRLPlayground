#include "Player.hpp"

std::vector<Card> Player::findHighestCombination(const std::vector<Card>& allCards) {
    if (allCards.size() != 7) {
        throw std::invalid_argument("findHighestCombination expects exactly 7 cards");
    }

    HandRank best;
    bool hasBest = false;

    // Iterate all 5-card combos from 7 by choosing which 2 to exclude
    for (int a = 0; a < 7; ++a) {
        for (int b = a + 1; b < 7; ++b) {
            std::array<Card,5> hand{};
            int k = 0;
            for (int i = 0; i < 7; ++i) {
                if (i == a || i == b) continue;
                hand[k++] = allCards[i];
            }

            HandRank cur = evaluate5(hand);
            if (!hasBest || best < cur) {
                best = cur;
                hasBest = true;
            }
        }
    }

    return std::vector<Card>(best.cards.begin(), best.cards.end());
}


std::array<Card,2> Player::holeFixed() const {
    std::array<Card,2> out{ Card(), Card() };
    if (pocketHand_.size() == 2) { out[0] = pocketHand_[0]; out[1] = pocketHand_[1]; }
    return out;
}

void Player::fold(){
    folded = true;
    // DO NOT toss cards here. It breaks showdown + debugging + observation design.
    // pocketHand_ stays as-is, but Observation will hide them from others.
    bestCardCombination_.clear();
}

void Player::betChips(int delta) {
    if (delta < 0) throw std::invalid_argument("delta < 0");
    if (delta > money) throw std::invalid_argument("delta > stack");
    money -= delta;
    streetContribution_ += delta;
    totalContribution_  += delta;
    if (money == 0) allIn_ = true;
}

void Player::resetForNewHand() {
    allIn_ = false;
    folded = false;
    streetContribution_ = 0;
    totalContribution_  = 0;
    bestCardCombination_.clear();
    pocketHand_.clear();
}

void Player::awardChips(int amount) {
    if (amount < 0) throw std::invalid_argument("awardChips: amount < 0");
    money += amount;
}
        

void Player::roundReset() {
    allIn_ = false;
    folded = false;
    streetContribution_ = 0;
    totalContribution_  = 0;
    bestCardCombination_.clear();
}

void Player::giveCards(const std::vector<Card>& cards) {
    if(cards.size() != 2) {
        throw std::invalid_argument("A player must be given exactly two cards.");
    }
    bestCardCombination_.clear();
    pocketHand_= cards;
}
        
void Player::tossCards(){
    pocketHand_.clear();
    pocketHand_ = {Card(Suits::Hearts, Ranks::Empty), Card(Suits::Hearts, Ranks::Empty)};
    bestCardCombination_.clear();
}



std::vector<Card> Player::getBestCombination(const std::vector<Card>& sharedCards) {
    if(!bestCardCombination_.empty()){
        return bestCardCombination_;
    }
    if(pocketHand_.size() != 2) {
        throw std::runtime_error("Player does not have a valid pocket hand.");
    }
    if(sharedCards.size() != 5) {
        throw std::invalid_argument("There must be exactly five shared cards.");
    }
    if(folded) {
        // throw std::runtime_error("Folded players cannot have a best combination.");
        return {Card(Suits::Hearts, Ranks::Empty),
                Card(Suits::Hearts, Ranks::Empty), 
                Card(Suits::Hearts, Ranks::Empty), 
                Card(Suits::Hearts, Ranks::Empty),  
                Card(Suits::Hearts, Ranks::Empty)};
        }
    
    std::vector<Card> allCards;
    allCards.insert(allCards.end(), pocketHand_.begin(), pocketHand_.end());
    allCards.insert(allCards.end(), sharedCards.begin(), sharedCards.end());

    bestCardCombination_ = findHighestCombination(allCards);

    return bestCardCombination_;
}

void Player::printPocketHand() const {
    std::cout << "Player " << playerId_ << " Pocket Hand: ";
    for (const auto& card : pocketHand_) {
        std::cout << "(" << rint(card.rank) << ", " << static_cast<int>(card.suit) << ") ";
    }
    std::cout << std::endl;
}


void Player::adjustMoney(int amount) {
    money += amount;
    if (money < 0) money = 0;
}

