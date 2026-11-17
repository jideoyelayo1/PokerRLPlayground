// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

#include "poker.hpp"
#include "PokerProperties.hpp"
#include "Player.hpp"
#include "GameEngine.hpp"

static std::string card_repr(const Card& c) {
    return "Card(suit=" + std::to_string(static_cast<int>(c.suit)) +
           ", rank=" + std::to_string(static_cast<int>(c.rank)) + ")";
}

PYBIND11_MODULE(poker_engine, m) {
    m.doc() = "Poker engine bindings (pybind11)";

    // -------------------------
    // Enums
    // -------------------------
    py::enum_<ActionType>(m, "ActionType")
        .value("Fold",  ActionType::Fold)
        .value("Check", ActionType::Check)
        .value("Call",  ActionType::Call)
        .value("Raise", ActionType::Raise)
        .value("AllIn", ActionType::AllIn)
        .export_values();

    py::enum_<RaiseSize>(m, "RaiseSize")
        .value("BB",      RaiseSize::BB)
        .value("HalfPot", RaiseSize::HalfPot)
        .value("Pot",     RaiseSize::Pot);
        //.export_values();

    py::enum_<Suits>(m, "Suits")
        .value("Empty",    Suits::Empty)
        .value("Hearts",   Suits::Hearts)
        .value("Diamonds", Suits::Diamonds)
        .value("Clubs",    Suits::Clubs)
        .value("Spades",   Suits::Spades)
        .export_values();

    py::enum_<Ranks>(m, "Ranks")
        .value("Empty", Ranks::Empty)
        .value("Two",   Ranks::Two)
        .value("Three", Ranks::Three)
        .value("Four",  Ranks::Four)
        .value("Five",  Ranks::Five)
        .value("Six",   Ranks::Six)
        .value("Seven", Ranks::Seven)
        .value("Eight", Ranks::Eight)
        .value("Nine",  Ranks::Nine)
        .value("Ten",   Ranks::Ten)
        .value("Jack",  Ranks::Jack)
        .value("Queen", Ranks::Queen)
        .value("King",  Ranks::King)
        .value("Ace",   Ranks::Ace)
        .export_values();

    py::enum_<HandCat>(m, "HandCat")
        .value("HighCard",      HandCat::HighCard)
        .value("OnePair",       HandCat::OnePair)
        .value("TwoPair",       HandCat::TwoPair)
        .value("Trips",         HandCat::Trips)
        .value("Straight",      HandCat::Straight)
        .value("Flush",         HandCat::Flush)
        .value("FullHouse",     HandCat::FullHouse)
        .value("Quads",         HandCat::Quads)
        .value("StraightFlush", HandCat::StraightFlush)
        .export_values();

    // -------------------------
    // Plain structs / classes
    // -------------------------
    py::class_<Card>(m, "Card")
        .def(py::init<>())
        .def(py::init<Suits, Ranks>(), py::arg("suit"), py::arg("rank"))
        .def_readwrite("suit", &Card::suit)
        .def_readwrite("rank", &Card::rank)
        .def("__repr__", [](const Card& c) { return card_repr(c); });

    py::class_<Pot>(m, "Pot")
        .def(py::init<>())
        .def_readwrite("amount", &Pot::amount)
        .def_readwrite("eligible", &Pot::eligible);

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def(py::init<ActionType, RaiseSize>(), py::arg("type"), py::arg("raise_size") = RaiseSize::BB)
        .def_readwrite("type", &Action::type)
        .def_readwrite("raiseSize", &Action::raiseSize);

    py::class_<ActionSpec>(m, "ActionSpec")
        .def(py::init<>())
        .def_readwrite("type", &ActionSpec::type)
        .def_readwrite("call_amount", &ActionSpec::call_amount)
        .def_readwrite("min_raise_to", &ActionSpec::min_raise_to)
        .def_readwrite("max_raise_to", &ActionSpec::max_raise_to)
        .def_readwrite("allowed", &ActionSpec::allowed)
        .def_readwrite("raise_size", &ActionSpec::raise_size)
        .def_readwrite("raise_to", &ActionSpec::raise_to);

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("viewer_id", &Observation::viewer_id)
        .def_readwrite("street", &Observation::street)
        .def_readwrite("current_player_id", &Observation::current_player_id)
        .def_readwrite("hand_in_progress", &Observation::hand_in_progress)
        .def_readwrite("terminal", &Observation::terminal)
        .def_readwrite("winner_id", &Observation::winner_id)
        .def_readwrite("pot_total", &Observation::pot_total)
        .def_readwrite("to_call", &Observation::to_call)
        .def_readwrite("min_raise", &Observation::min_raise)
        .def_readwrite("hole", &Observation::hole)
        .def_readwrite("board", &Observation::board)
        .def_readwrite("alive", &Observation::alive)
        .def_readwrite("in_hand", &Observation::in_hand)
        .def_readwrite("can_act", &Observation::can_act)
        .def_readwrite("stack", &Observation::stack)
        .def_readwrite("street_contrib", &Observation::street_contrib)
        .def_readwrite("total_contrib", &Observation::total_contrib)
        .def_readwrite("folded", &Observation::folded)
        .def_readwrite("all_in", &Observation::all_in)
        .def_readwrite("viewer_to_call", &Observation::viewer_to_call)
        .def_readwrite("legal", &Observation::legal);

    py::class_<HandRank>(m, "HandRank")
        .def(py::init<>())
        .def_readwrite("cat", &HandRank::cat)
        .def_readwrite("kickers", &HandRank::kickers)
        .def_readwrite("cards", &HandRank::cards)
        .def("__lt__", [](const HandRank& a, const HandRank& b) { return a < b; });

    py::class_<Player>(m, "Player")
        .def(py::init<int,int>(), py::arg("player_id"), py::arg("starting_money") = 1000)
        .def("streetContrib", &Player::streetContrib)
        .def("totalContrib", &Player::totalContrib)
        .def("isActiveInHand", &Player::isActiveInHand)
        .def("gameOver", &Player::gameOver)
        .def("stack", &Player::stack)
        .def("isFolded", &Player::isFolded)
        .def("isAllIn", &Player::isAllIn)
        .def("holeFixed", &Player::holeFixed)
        .def("fold", &Player::fold)
        .def("betChips", &Player::betChips, py::arg("delta"))
        .def("resetForNewHand", &Player::resetForNewHand)
        .def("awardChips", &Player::awardChips, py::arg("amount"))
        .def("canAct", &Player::canAct)
        .def("roundReset", &Player::roundReset)
        .def("giveCards", &Player::giveCards, py::arg("cards"))
        .def("tossCards", &Player::tossCards)
        .def("hasFolded", &Player::hasFolded)
        .def("getId", &Player::getId)
        .def("getMoney", &Player::getMoney)
        .def("adjustMoney", &Player::adjustMoney, py::arg("amount"))
        .def("resetStreetBet", &Player::resetStreetBet)
        .def("getStreetContribution", &Player::getStreetContribution)
        .def("getBestCombination", &Player::getBestCombination, py::arg("sharedCards"))
        .def("printPocketHand", &Player::printPocketHand)
        .def("getPocketHand", &Player::getPocketHand);

    // -------------------------
    // Free functions
    // -------------------------
    m.def("evaluate5",
          [](const std::array<Card,5>& hand) { return evaluate5(hand); },
          py::arg("hand"));

    m.def("best_hand_of_7",
          [](const std::vector<Card>& seven) { return GameEngine::bestHandOf7(seven); },
          py::arg("seven"));

    // -------------------------
    // GameEngine bindings
    // -------------------------
    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<int,int,int,int>(),
             py::arg("numPlayers"),
             py::arg("startingMoney") = 200,
             py::arg("big_blind") = 2,
             py::arg("small_blind") = 1)

        // Existing controller API (kept)
        .def("set_controller",
             [](GameEngine& eng, int playerId, py::function py_cb) {
                 GameEngine::Controller c = [py_cb](const GameEngine& g, int id) -> Action {
                     py::gil_scoped_acquire gil;
                     py::object ret = py_cb(std::cref(g), id);
                     return ret.cast<Action>();
                 };
                 eng.setController(playerId, std::move(c));
             },
             py::arg("player_id"),
             py::arg("callback"))

        // Existing getters
        .def("potTotal", &GameEngine::potTotal)
        .def("toCall", &GameEngine::toCall)
        .def("minRaise", &GameEngine::minRaise)
        .def("currentPlayer", &GameEngine::currentPlayer)
        .def("streetInt", &GameEngine::streetInt)
        .def("boardCardsFixed", &GameEngine::boardCardsFixed)
        .def("handInProgress", &GameEngine::handInProgress)
        .def("currentPlayerSafe", &GameEngine::currentPlayerSafe)
        .def("alivePlayers", &GameEngine::alivePlayers)
        .def("aliveIds", &GameEngine::aliveIds)
        .def("playOneHand", &GameEngine::playOneHand)
        .def("printAllPlayersStatus", &GameEngine::printAllPlayersStatus)
        .def("getPlayerCount", &GameEngine::getPlayerCount)
        .def("legalActions", &GameEngine::legalActions, py::arg("player_id"))
        .def("getObservation", &GameEngine::getObservation, py::arg("viewer_id"))

        // NEW: Step-based API you need for DQN (you must implement these in C++)
        .def("reset_tournament", &GameEngine::reset_tournament, py::arg("startingMoney"))
        .def("start_hand", &GameEngine::start_hand)
        .def("step", &GameEngine::step, py::arg("action"), py::arg("viewer_id"))
        .def("hand_terminal", &GameEngine::hand_terminal)
        .def("winner_id", &GameEngine::winner_id)
        .def("seed", &GameEngine::seed, py::arg("seed"))
        .def("bigBlind", &GameEngine::bigBlind)
        .def("smallBlind", &GameEngine::smallBlind)

        // FIX: do NOT bind &GameEngine::isAllowed (your C++ doesn't define it as a real class static right now)
        .def_static("isAllowed",
            [](const std::vector<ActionSpec>& legal, ActionType t) {
                return GameEngine::isAllowed(legal, t);  // any allowed entry of type t
            },
            py::arg("legal"),
            py::arg("type"));
}
