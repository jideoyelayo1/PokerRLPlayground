
#include "poker.hpp"



#ifndef POKER_PYBIND
int main() {
    GameEngine engine(4);

    auto allowed = [](const std::vector<ActionSpec>& legal, ActionType t) {
        for (const auto& a : legal) if (a.type == t && a.allowed) return true;
        return false;
    };

    engine.setController(0, [allowed](const GameEngine& g, int id) {
        auto legal = g.legalActions(id);

        if (allowed(legal, ActionType::Raise)) return Action{ActionType::Raise, RaiseSize::Pot};
        if (allowed(legal, ActionType::Call))  return Action{ActionType::Call,  RaiseSize::BB};
        if (allowed(legal, ActionType::Check)) return Action{ActionType::Check, RaiseSize::BB};
        return Action{ActionType::Fold, RaiseSize::BB};
    });

    engine.setController(1, [allowed](const GameEngine& g, int id) {
        auto legal = g.legalActions(id);

        // Player 1 wants to CALL when possible, otherwise CHECK.
        if (allowed(legal, ActionType::Call))  return Action{ActionType::Call,  RaiseSize::BB};
        if (allowed(legal, ActionType::Check)) return Action{ActionType::Check, RaiseSize::BB};
        return Action{ActionType::Fold, RaiseSize::BB};
    });

    engine.setController(2, [allowed](const GameEngine& g, int id) {
        // Always fold (Fold is always allowed if canActNow==true)
        return Action{ActionType::Fold, RaiseSize::BB};
    });

    engine.setController(3, [allowed](const GameEngine& g, int id) {
        auto legal = g.legalActions(id);

        if (allowed(legal, ActionType::AllIn)) return Action{ActionType::AllIn, RaiseSize::BB};
        if (allowed(legal, ActionType::Call))  return Action{ActionType::Call,  RaiseSize::BB};
        if (allowed(legal, ActionType::Check)) return Action{ActionType::Check, RaiseSize::BB};
        return Action{ActionType::Fold, RaiseSize::BB};
    });




    // for(int i = 0; i < 10; ++i)
    // {
    //     std::cout << "=== Hand " << (i+1) << " ===" << std::endl;
    //     engine.playOneHand();
    //     engine.printAllPlayersStatus();
    // }
    for (int i = 0; i < 200; ++i) {
        if (engine.alivePlayers() < 2) break;
        std::cout << engine.alivePlayers() << " players remaining." << std::endl;
        std::cout << "=== Hand " << (i+1) << " ===" << std::endl;
        engine.playOneHand();
        engine.printAllPlayersStatus();
        std::cout << "======" << std::endl;
    }
    
    std::cout << engine.alivePlayers() << " players remaining." << std::endl;

    return 0;
}
#endif
