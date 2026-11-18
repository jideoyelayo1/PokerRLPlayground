// Embind binding: exposes Table (web/Table.hpp) to JavaScript, with plain objects and arrays out.
// The test build adds web/test_hooks.cpp; the release module has nothing else.
//
// The network is compiled into the module: web/build.sh embeds weights/demo/<run>.json and <run>.bin
// (POKER_MANIFEST, POKER_WEIGHTS) after checking the binary's sha256 against the manifest, so
// nothing is fetched at run time.
//
// Numbers cross as doubles. Arguments arrive as val and are checked for their JavaScript type, not
// coerced: act() takes a string, and anything else (a number, null, a String object) is an unknown
// action. Missing arguments arrive as undefined and extra ones are ignored (embind).
//
// Nothing is thrown from C++ into JavaScript. Such an exception unwinds past WebAssembly without
// restoring its stack pointer, so each one leaks stack until the module fails: 4,499 refused seeds
// did it when the seed was checked here. Now web/post.js checks the seed in JavaScript and throws its
// TypeError before WebAssembly is entered, and every bound call below catches C++ exceptions
// (built with -fexceptions) and returns a refusal, error "internal_error", instead. Embind's own
// BindingError (a deleted Table, a wrong number of constructor arguments) is thrown by its
// JavaScript before WebAssembly is entered. Not covered: a page that replaces built-ins the binding
// calls back into (Array.prototype.push, property setters on Object.prototype) with ones that throw.
#include "Table.hpp"

#include <emscripten/bind.h>

#include <optional>
#include <string>

using emscripten::val;

namespace {

const unsigned char kManifest[] = {
#embed POKER_MANIFEST
};
const unsigned char kWeights[] = {
#embed POKER_WEIGHTS
};

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
    if (!m.ok) {
        o.set("error", m.error);
    } else if (m.seat >= 0) {  // a move; startHand() reports only ok, over and toAct
        o.set("seat", m.seat);
        o.set("action", m.action);
        o.set("chips", m.chips);
        o.set("to", m.to);
    }
    o.set("over", m.over);
    o.set("toAct", m.toAct);
    return o;
}

val internalError(const char* what) {
    val o = val::object();
    o.set("ok", false);
    o.set("error", std::string("internal_error"));
    o.set("message", std::string(what));
    return o;
}

// Runs a bound call; a C++ exception becomes a returned refusal rather than leaving WebAssembly.
template <class F>
val guarded(F call) {
    try {
        return call();
    } catch (const std::exception& e) {
        return internalError(e.what());
    } catch (...) {
        return internalError("unknown exception");
    }
}

// The network, parsed on first use from the embedded manifest and weights. That cannot fail for the
// files web/build.sh embeds, but if it did the error is kept, not thrown: web/post.js reads it
// (_networkError) and refuses to construct a Table.
struct Network {
    std::optional<agent::Mlp> net;
    std::string error;
};
const Network& network() {
    static const Network n = [] {
        Network x;
        try {
            x.net.emplace(agent::Mlp::fromManifest(std::string(reinterpret_cast<const char*>(kManifest), sizeof kManifest),
                                                   std::string(reinterpret_cast<const char*>(kWeights), sizeof kWeights)));
        } catch (const std::exception& e) {
            x.error = e.what();
        }
        return x;
    }();
    return n;
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

// Only once _networkError() is empty, which web/post.js checks before any Table is made.
const agent::Mlp& demoNetwork() { return *network().net; }

EMSCRIPTEN_BINDINGS(poker) {
    using emscripten::optional_override;

    // Reached only through web/post.js's Table, which passes a whole number from 0 to 2^53 - 1 (the
    // range check here only keeps the cast defined).
    emscripten::class_<Table>("Table")
        .constructor(optional_override([](double seed) -> Table* {
            try {
                return new Table(demoNetwork(), seed >= 0 && seed <= 9007199254740991.0 ? static_cast<uint64_t>(seed) : 0);
            } catch (...) {  // out of memory, the only way here: a null pointer rather than a throw (untested)
                return nullptr;
            }
        }), emscripten::allow_raw_pointers())
        .class_function("format", optional_override([] { return guarded(format); }))
        .function("startHand", optional_override([](Table& t) { return guarded([&] { return moveToJs(t.startHand()); }); }))
        .function("act", optional_override([](Table& t, val action) {
            return guarded([&] { return moveToJs(t.act(action.isString() ? action.as<std::string>() : std::string{})); });
        }))
        .function("advance", optional_override([](Table& t) { return guarded([&] { return moveToJs(t.advance()); }); }))
        .function("view", optional_override([](const Table& t) { return guarded([&] { return viewToJs(t.view()); }); }));
    emscripten::function("_networkError", optional_override([] { return network().error; }));

}
