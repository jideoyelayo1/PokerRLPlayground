// Native side of web/test.mjs's replay. Plays <hands> hands at a Table seeded with <seed>; the
// visitor chooses with mulberry32(<visitorSeed>): "random" takes a uniformly random legal action,
// "call" checks, or else calls. Prints one line per hand start, move and hand end, in the format
// test.mjs builds from the WebAssembly module, so the two outputs can be compared as text.
//   replay_driver <weights/demo> <seed> <visitorSeed> <hands> <random|call>
// With "q" instead, it prints the network's Q-values for every parity record, as float32 bits in hex.
//   replay_driver <weights/demo> q
#include <bit>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Table.hpp"

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// mulberry32, as in test.mjs
struct Mulberry32 {
    uint32_t s;
    uint32_t next() {
        s += 0x6d2b79f5u;
        uint32_t t = (s ^ (s >> 15)) * (1u | s);
        t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
        return t ^ (t >> 14);
    }
};

template <class T, class F>
std::string join(const std::vector<T>& xs, const char* sep, F show) {
    std::string out;
    for (size_t i = 0; i < xs.size(); ++i) out += (i ? sep : "") + show(xs[i]);
    return out;
}
std::string str(const std::string& s) { return s; }
std::string num(int x) { return std::to_string(x); }

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[2]) == "q") {
        const std::string dir = argv[1];
        const agent::Mlp net = agent::Mlp::fromManifest(slurp(dir + "/mlp-s-seed3.json"), slurp(dir + "/mlp-s-seed3.bin"));
        std::istringstream lines(slurp(dir + "/mlp-s-seed3.parity.jsonl"));
        for (std::string line; std::getline(lines, line);) {
            if (line.empty()) continue;
            const auto q = agent::decide(net, agent::observationFromJson(Json::parse(line)["observation"]), false).q;
            std::cout << join(q, " ", [](float x) { char b[9]; std::snprintf(b, sizeof b, "%08x", std::bit_cast<uint32_t>(x)); return std::string(b); }) << "\n";
        }
        return 0;
    }
    if (argc != 6) {
        std::cerr << "usage: replay_driver <weights/demo> <seed> <visitorSeed> <hands> <random|call>\n";
        return 2;
    }
    const std::string dir = argv[1], policy = argv[5];
    const agent::Mlp net = agent::Mlp::fromManifest(slurp(dir + "/mlp-s-seed3.json"), slurp(dir + "/mlp-s-seed3.bin"));
    Table t(net, std::stoull(argv[2]));
    Mulberry32 rng{static_cast<uint32_t>(std::stoul(argv[3]))};
    const int hands = std::stoi(argv[4]);
    std::string out;
    for (int h = 0; h < hands; ++h) {
        t.startHand();
        TableView v = t.view();
        out += "hand " + num(v.hand) + " button " + num(v.button) + " cards " + join(v.cards, " ", str) + "\n";
        while (!v.over) {
            TableMove m;
            if (v.toAct == Table::kVisitor) {
                std::string a;
                if (policy == "call") {
                    for (const char* want : {"call", "check"})
                        for (const auto& l : v.legal) if (l.action == want) a = want;
                } else {
                    a = v.legal[rng.next() % v.legal.size()].action;
                }
                m = t.act(a);
            } else {
                m = t.advance();
            }
            if (!m.ok) { std::cerr << "refused: " << m.error << "\n"; return 1; }
            out += num(m.seat) + " " + m.action + " " + num(m.chips) + " " + num(m.to) + "\n";
            v = t.view();
        }
        if (!v.result) { out += "end aborted\n"; continue; }
        const TableResult& r = *v.result;
        out += "end board " + join(v.board, " ", str) + " pot " + num(v.pot) + " pots " +
               join(r.pots, " ", [](const TablePot& p) { return num(p.amount) + ":" + join(p.winners, ",", num) + ":" + join(p.shares, ",", num); }) +
               " net " + join(r.net, " ", num) + " showdown " +
               join(r.showdown, " ", [](const TableShowdown& s) { return num(s.seat) + ":" + join(s.cards, "", str) + ":" + s.hand; }) + "\n";
    }
    std::cout << out;
    return 0;
}
