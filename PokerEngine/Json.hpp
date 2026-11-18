#pragma once
// Minimal JSON reader (standard library only) for the agent's manifest and the parity records
// (weights/README.md). Parses the whole text into a tree; throws std::runtime_error on bad input,
// a missing key, or arrays and objects nested deeper than kMaxDepth (the parser recurses once per
// level, and in WebAssembly the stack is small).
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct Json {
    static constexpr int kMaxDepth = 64;
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json& operator[](const std::string& key) const {
        for (const auto& [k, v] : object) if (k == key) return v;
        throw std::runtime_error("JSON: no key " + key);
    }
    const Json& operator[](size_t i) const { return array.at(i); }
    size_t size() const { return type == Type::Object ? object.size() : array.size(); }
    // A whole number that fits an int (true and false read as 1 and 0); anything else (1.5, "1",
    // NaN, infinity, 1e300) throws. The range is checked before the cast, which is undefined outside it.
    int integer() const {
        if (type == Type::Bool) return boolean;
        if (type != Type::Number || !(number >= -2147483648.0 && number <= 2147483647.0) ||
            number != static_cast<double>(static_cast<int>(number)))
            throw std::runtime_error("JSON: not an integer");
        return static_cast<int>(number);
    }

    static Json parse(const std::string& text) {
        size_t i = 0;
        Json v = value(text, i, 0);
        skip(text, i);
        if (i != text.size()) throw std::runtime_error("JSON: trailing characters");
        return v;
    }

private:
    static void skip(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    static void expect(const std::string& s, size_t& i, char c) {
        skip(s, i);
        if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("JSON: expected ") + c);
        ++i;
    }
    static std::string str(const std::string& s, size_t& i) {
        expect(s, i, '"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                if (++i >= s.size()) break;
                char e = s[i];
                // shortcut: \uXXXX is kept as written; the manifest and records are ASCII
                out += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
            } else {
                out += s[i];
            }
            ++i;
        }
        if (i >= s.size()) throw std::runtime_error("JSON: unterminated string");
        ++i;
        return out;
    }
    static Json value(const std::string& s, size_t& i, int depth) {
        skip(s, i);
        if (i >= s.size()) throw std::runtime_error("JSON: unexpected end");
        Json v;
        char c = s[i];
        if ((c == '{' || c == '[') && depth >= kMaxDepth) throw std::runtime_error("JSON: nested too deeply");
        if (c == '{') {
            v.type = Type::Object;
            ++i;
            skip(s, i);
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            for (;;) {
                std::string k = str(s, i);
                expect(s, i, ':');
                v.object.emplace_back(std::move(k), value(s, i, depth + 1));
                skip(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                expect(s, i, '}');
                return v;
            }
        }
        if (c == '[') {
            v.type = Type::Array;
            ++i;
            skip(s, i);
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            for (;;) {
                v.array.push_back(value(s, i, depth + 1));
                skip(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                expect(s, i, ']');
                return v;
            }
        }
        if (c == '"') { v.type = Type::String; v.string = str(s, i); return v; }
        for (auto [word, t, b] : {std::tuple{"true", Type::Bool, true}, std::tuple{"false", Type::Bool, false},
                                  std::tuple{"null", Type::Null, false}}) {
            if (s.compare(i, std::char_traits<char>::length(word), word) == 0) {
                i += std::char_traits<char>::length(word);
                v.type = t;
                v.boolean = b;
                return v;
            }
        }
        const char* start = s.c_str() + i;
        char* end = nullptr;
        v.number = std::strtod(start, &end);  // the "C" locale: nothing here calls setlocale
        if (end == start) throw std::runtime_error("JSON: bad value");
        i += static_cast<size_t>(end - start);
        v.type = Type::Number;
        return v;
    }
};
