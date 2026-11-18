// The JSON reader (PokerEngine/Json.hpp): what the manifest and the parity records need, and
// refusals that must not crash: nesting deeper than Json::kMaxDepth, malformed text.
#include "../Json.hpp"
#include "test_util.hpp"

static bool refused(const std::string& text) {
    try { Json::parse(text); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main() {
    const Json j = Json::parse(R"({"a": [1, -2.5, true, null, "x\"y"], "b": {"c": []}})");
    CHECK(j["a"].size() == 5 && j["a"][0].integer() == 1 && j["a"][1].number == -2.5, "numbers");
    CHECK(j["a"][2].boolean && j["a"][3].type == Json::Type::Null && j["a"][4].string == "x\"y", "literals and escapes");
    CHECK(j["b"]["c"].type == Json::Type::Array && j["b"]["c"].size() == 0, "empty array");

    // nesting: kMaxDepth levels are read, one more is refused, and 100,000 do not exhaust the stack
    auto nested = [](int n) { return std::string(n, '[') + std::string(n, ']'); };
    CHECK(!refused(nested(Json::kMaxDepth)), "%d nested arrays are read", Json::kMaxDepth);
    CHECK(refused(nested(Json::kMaxDepth + 1)), "%d nested arrays are refused", Json::kMaxDepth + 1);
    CHECK(refused(std::string(100000, '[')), "100,000 open brackets are refused");
    std::string objects;
    for (int k = 0; k <= Json::kMaxDepth; ++k) objects += "{\"k\":";
    CHECK(refused(objects + "1" + std::string(Json::kMaxDepth + 1, '}')), "objects nested too deeply are refused");
    for (const char* bad : {"", "[", "{", "[1,", "{\"a\"}", "\"open", "tru", "[1] x", "-", "{1: 2}"})
        CHECK(refused(bad), "malformed text refused: %s", bad);
    return finish("test_json");
}
