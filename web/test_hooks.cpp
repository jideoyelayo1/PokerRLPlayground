// Test hooks, linked only into the test build of the module (web/build.sh writes it to
// build/wasm-test/poker_test.mjs), never into the release module in web/dist:
// decideObservation(json, noFreeFold), the network's decision for one observation in the parity
// records' format (weights/README.md), which web/test.mjs runs over every record.
#include <emscripten/bind.h>

#include <string>

#include "Agent.hpp"

using emscripten::val;

const agent::Mlp& demoNetwork();  // web/poker_wasm.cpp

namespace {

template <class T>
val array(const std::vector<T>& xs) {
    val out = val::array();
    for (const auto& x : xs) out.call<void>("push", x);
    return out;
}

val failure(const std::string& error) {
    val o = val::object();
    o.set("ok", false);
    o.set("error", error);
    return o;
}

// The state vector, the mask (after no_free_fold if asked), the Q-values and the action index; a
// refusal for anything that is not an observation string and a boolean.
val decideObservation(val json, val noFreeFold) {
    if (!json.isString()) return failure("the observation must be a JSON string");
    if (noFreeFold.typeOf().as<std::string>() != "boolean") return failure("noFreeFold must be a boolean");
    try {
        const auto d = agent::decide(demoNetwork(), agent::observationFromJson(Json::parse(json.as<std::string>())), noFreeFold.as<bool>());
        val o = val::object();
        o.set("ok", true);
        o.set("state", array(d.state));
        o.set("mask", array(std::vector<float>(d.mask.begin(), d.mask.end())));
        o.set("q", array(d.q));
        o.set("action", d.action);
        o.set("actionName", std::string(agent::kActionNames[d.action]));
        return o;
    } catch (const std::exception& e) {  // caught here: nothing is thrown into JavaScript
        return failure(e.what());
    }
}

}  // namespace

EMSCRIPTEN_BINDINGS(poker_test) {
    emscripten::function("decideObservation", &decideObservation);
}
