// Embind binding: exposes Table (web/Table.hpp) and one agent decision to JavaScript, with plain
// objects and arrays out.
//
// The network is compiled into the module: web/build.sh embeds weights/demo/<run>.json and <run>.bin
// (POKER_MANIFEST, POKER_WEIGHTS) after checking the binary's sha256 against the manifest, so
// nothing is fetched at run time.
//
// Numbers cross as doubles. Arguments arrive as val and are checked for their JavaScript type, not
// coerced: act() takes a string, and anything else (a number, null, a String object) is an unknown
// action; decideObservation() takes a string and a boolean. The only call that throws is the Table
// constructor, with a TypeError, when the seed is not a whole number from 0 to 2^53 - 1. Built with
// -fexceptions, so an unexpected C++ exception reaches JavaScript as an exception, not an abort.
#include "Table.hpp"

#include <emscripten/bind.h>

#include <cmath>
#include <string>

using emscripten::val;

namespace {

const unsigned char kManifest[] = {
#embed POKER_MANIFEST
};
const unsigned char kWeights[] = {
#embed POKER_WEIGHTS
};

const agent::Mlp& network() {
    static const agent::Mlp net = agent::Mlp::fromManifest(
        std::string(reinterpret_cast<const char*>(kManifest), sizeof kManifest),
        std::string(reinterpret_cast<const char*>(kWeights), sizeof kWeights));
    return net;
}

template <class T>
val array(const std::vector<T>& xs) {
    val out = val::array();
    for (const auto& x : xs) out.call<void>("push", x);
    return out;
}

template <class T, class F>
val array(const std::vector<T>& xs, F toJs) {
    val out = val::array();
    for (const auto& x : xs) out.call<void>("push", toJs(x));
    return out;
}

val viewToJs(const TableView& v) {
    val o = val::object();
    o.set("hand", v.hand);
    o.set("over", v.over);
    o.set("aborted", v.aborted);
    o.set("street", v.street);
    o.set("visitor", v.visitor);
    o.set("button", v.button);
    o.set("smallBlind", v.smallBlind);
    o.set("bigBlind", v.bigBlind);
    o.set("pot", v.pot);
    o.set("toAct", v.toAct);
    o.set("board", array(v.board));
    o.set("cards", array(v.cards));
    o.set("seats", array(v.seats, [](const TableSeat& s) {
        val x = val::object();
        x.set("seat", s.seat);
        x.set("stack", s.stack);
        x.set("streetContribution", s.streetContribution);
        x.set("totalContribution", s.totalContribution);
        x.set("status", s.status);
        return x;
    }));
    o.set("legal", array(v.legal, [](const TableAction& a) {
        val x = val::object();
        x.set("action", a.action);
        x.set("chips", a.chips);
        x.set("to", a.to);
        return x;
    }));
    o.set("log", array(v.log, [](const TableLogEntry& e) {
        val x = val::object();
        x.set("seat", e.seat);
        x.set("street", e.street);
        x.set("action", e.action);
        x.set("chips", e.chips);
        x.set("to", e.to);
        return x;
    }));
    if (!v.result) {
        o.set("result", val::null());
        return o;
    }
    val r = val::object();
    r.set("pots", array(v.result->pots, [](const TablePot& p) {
        val x = val::object();
        x.set("amount", p.amount);
        x.set("eligible", array(p.eligible));
        x.set("winners", array(p.winners));
        x.set("shares", array(p.shares));
        return x;
    }));
    r.set("net", array(v.result->net));
    r.set("showdown", array(v.result->showdown, [](const TableShowdown& s) {
        val x = val::object();
        x.set("seat", s.seat);
        x.set("cards", array(s.cards));
        x.set("hand", s.hand);
        return x;
    }));
    o.set("result", r);
    return o;
}

val moveToJs(const TableMove& m) {
    val o = val::object();
    o.set("ok", m.ok);
    if (m.ok) {
        o.set("seat", m.seat);
        o.set("action", m.action);
        o.set("chips", m.chips);
        o.set("to", m.to);
    } else {
        o.set("error", m.error);
    }
    o.set("over", m.over);
    o.set("toAct", m.toAct);
    return o;
}

uint64_t seedOf(const val& seed) {
    const double x = seed.isNumber() ? seed.as<double>() : std::nan("");
    if (!(x >= 0 && x <= 9007199254740991.0 && std::floor(x) == x))
        val::global("TypeError").new_(std::string("Table: the seed must be a whole number from 0 to 2^53 - 1")).throw_();
    return static_cast<uint64_t>(x);
}

val failure(const std::string& error) {
    val o = val::object();
    o.set("ok", false);
    o.set("error", error);
    return o;
}

// One decision of the network for an observation in the parity records' format (weights/README.md):
// the state vector, the mask (after no_free_fold if asked), the Q-values and the action index.
val decideObservation(val json, val noFreeFold) {
    if (!json.isString()) return failure("the observation must be a JSON string");
    if (noFreeFold.typeOf().as<std::string>() != "boolean") return failure("noFreeFold must be a boolean");
    try {
        const auto d = agent::decide(network(), agent::observationFromJson(Json::parse(json.as<std::string>())), noFreeFold.as<bool>());
        val o = val::object();
        o.set("ok", true);
        o.set("state", array(d.state));
        o.set("mask", array(std::vector<float>(d.mask.begin(), d.mask.end())));
        o.set("q", array(d.q));
        o.set("action", d.action);
        o.set("actionName", std::string(agent::kActionNames[d.action]));
        return o;
    } catch (const std::exception& e) {
        return failure(e.what());
    }
}

val format() {
    val o = val::object();
    o.set("seats", Table::kSeats);
    o.set("smallBlind", Table::kSmallBlind);
    o.set("bigBlind", Table::kBigBlind);
    o.set("stack", Table::kStack);
    o.set("visitor", Table::kVisitor);
    o.set("maxDecisions", Table::kMaxDecisions);
    o.set("actions", array(std::vector<std::string>(std::begin(agent::kActionNames), std::end(agent::kActionNames))));
    return o;
}

}  // namespace

EMSCRIPTEN_BINDINGS(poker) {
    using emscripten::optional_override;

    emscripten::class_<Table>("Table")
        .constructor(optional_override([](val seed) { return new Table(network(), seedOf(seed)); }), emscripten::allow_raw_pointers())
        .class_function("format", &format)
        .function("startHand", &Table::startHand)
        .function("act", optional_override([](Table& t, val action) {
            return moveToJs(t.act(action.isString() ? action.as<std::string>() : std::string{}));
        }))
        .function("advance", optional_override([](Table& t) { return moveToJs(t.advance()); }))
        .function("view", optional_override([](const Table& t) { return viewToJs(t.view()); }));

    emscripten::function("decideObservation", &decideObservation);
}
