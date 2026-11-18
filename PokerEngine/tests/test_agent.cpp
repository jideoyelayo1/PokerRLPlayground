// Parity of the C++ agent (Agent.hpp) with PyTorch, over every record of the parity file:
// the encoded state equal element for element, the mask equal, the Q-values within kTolerance of
// PyTorch's, and the same chosen action with the play-time option no_free_fold off and on (the
// smallest gap between the two highest legal Q-values is about 1.5e-5, so close values alone
// would not prove the same play).
//   test_agent <weights/demo> [run]
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "../Agent.hpp"
#include "test_util.hpp"

static constexpr double kTolerance = 1e-5;  // as tests/test_weights.py allows PyTorch against itself

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "weights/demo", run = argc > 2 ? argv[2] : "mlp-s-seed3";
    const agent::Mlp net = agent::Mlp::fromManifest(slurp(dir + "/" + run + ".json"), slurp(dir + "/" + run + ".bin"));
    CHECK(net.inputSize() == 151 && net.outputSize() == 7, "network %d -> %d", net.inputSize(), net.outputSize());

    std::istringstream lines(slurp(dir + "/" + run + ".parity.jsonl"));
    int records = 0, stateBad = 0, maskBad = 0, qBad = 0, actionBad = 0, ruleBad = 0, ruleChanges = 0;
    double worst = 0, smallestGap = 1e9;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        ++records;
        const Json r = Json::parse(line);
        CHECK(r["history"].size() == 0, "record %d: an MLP record has no history", records);
        const Observation o = agent::observationFromJson(r["observation"]);

        const auto x = agent::encode(o, 2);
        bool same = x.size() == r["state"].size();
        for (size_t i = 0; same && i < x.size(); ++i) same = x[i] == static_cast<float>(r["state"][i].number);
        stateBad += !same;

        const auto mask = agent::legalMask(o);
        bool maskSame = r["mask"].size() == 7;
        for (int i = 0; maskSame && i < 7; ++i) maskSame = mask[i] == static_cast<float>(r["mask"][i].number);
        maskBad += !maskSame;

        const auto q = net.forward(x);
        double diff = 0;
        for (int i = 0; i < 7; ++i) diff = std::max(diff, std::fabs(q[i] - static_cast<double>(static_cast<float>(r["q"][i].number))));
        worst = std::max(worst, diff);
        qBad += diff > kTolerance;

        std::vector<float> legal;  // the gap the argmax depends on, from our own Q-values
        for (int i = 0; i < 7; ++i) if (mask[i] == 0.0f) legal.push_back(q[i]);
        std::sort(legal.rbegin(), legal.rend());
        if (legal.size() > 1) smallestGap = std::min(smallestGap, static_cast<double>(legal[0] - legal[1]));

        const int a = agent::argmax(q, mask), b = agent::argmax(q, agent::noFreeFold(mask));
        actionBad += a != r["action"].integer();
        ruleBad += b != r["action_no_free_fold"].integer();
        ruleChanges += a != b;
        const auto d = agent::decide(net, o, true);
        CHECK(d.action == b && d.q == q && d.state == x, "record %d: decide() differs from its parts", records);
    }
    std::printf("parity: %d records; state differs in %d, mask in %d; largest |q - q_torch| %.3g (tolerance %g, over it in %d); "
                "action differs in %d (no_free_fold off) and %d (on); the rule changes the action in %d; "
                "smallest gap between the two highest legal Q-values %.3g\n",
                records, stateBad, maskBad, worst, kTolerance, qBad, actionBad, ruleBad, ruleChanges, smallestGap);
    CHECK(records == 2674, "expected the 2,674 records of weights/README.md, read %d", records);
    CHECK(stateBad == 0 && maskBad == 0, "%d states and %d masks differ", stateBad, maskBad);
    CHECK(qBad == 0, "%d records with a Q-value more than %g from PyTorch's", qBad, kTolerance);
    CHECK(actionBad == 0 && ruleBad == 0, "%d actions differ with the rule off, %d with it on", actionBad, ruleBad);
    CHECK(ruleChanges == 47 + 18, "the rule changes %d actions; weights/README.md says 47 + 18", ruleChanges);

    // The manifest is checked, not trusted: a binary of another size is refused.
    bool refused = false;
    try { agent::Mlp::fromManifest(slurp(dir + "/" + run + ".json"), slurp(dir + "/" + run + ".bin").substr(4)); }
    catch (const std::runtime_error&) { refused = true; }
    CHECK(refused, "a truncated binary is refused");
    return finish("test_agent");
}
