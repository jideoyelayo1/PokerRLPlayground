#include "Agent.hpp"

#include <bit>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace agent {

namespace {

constexpr int kRanks = 13, kSuits = 4, kCard = kRanks + kSuits;

void cardOneHot(const Card& c, float* out) {
    int r = static_cast<int>(c.rank), s = static_cast<int>(c.suit);
    if (r != 0) out[r - 2] = 1.0f;
    if (s != 0) out[kRanks + s - 1] = 1.0f;
}

[[noreturn]] void bad(const std::string& what) { throw std::runtime_error("agent: " + what); }

}  // namespace

int actionIndex(const std::string& name) {
    for (int i = 0; i < kActions; ++i)
        if (name == kActionNames[i]) return i;
    return -1;
}

int specIndex(const ActionSpec& spec) {
    switch (spec.type) {
        case ActionType::Fold: return 0;
        case ActionType::Check: return 1;
        case ActionType::Call: return 2;
        case ActionType::AllIn: return 6;
        case ActionType::Raise: return 3 + static_cast<int>(spec.raise_size);
    }
    bad("unknown action type");
}

Action engineAction(int index) {
    switch (index) {
        case 0: return {ActionType::Fold};
        case 1: return {ActionType::Check};
        case 2: return {ActionType::Call};
        case 3: return {ActionType::Raise, RaiseSize::BB};
        case 4: return {ActionType::Raise, RaiseSize::HalfPot};
        case 5: return {ActionType::Raise, RaiseSize::Pot};
        case 6: return {ActionType::AllIn};
    }
    bad("action index out of range");
}

std::vector<float> encode(const Observation& o, int bigBlind) {
    const int n = static_cast<int>(o.stack.size()), v = o.viewer_id;
    // chip amounts in units of 100 big blinds: the product in double, then rounded to float32
    const double c = 1.0 / (100.0 * bigBlind);
    auto chips = [c](int x) { return static_cast<float>(x * c); };
    std::vector<float> x(127 + 6 * n, 0.0f);
    float* p = x.data();
    for (const Card& h : o.hole) { cardOneHot(h, p); p += kCard; }
    for (const Card& b : o.board) { cardOneHot(b, p); p += kCard; }
    *p++ = chips(o.pot_total);
    *p++ = chips(o.to_call);
    *p++ = chips(o.min_raise);
    *p++ = chips(o.viewer_to_call);
    if (o.street >= 0 && o.street < 4) p[o.street] = 1.0f;
    p += 4;
    for (int k = 0; k < n; ++k) {
        int i = (v + k) % n;
        *p++ = chips(o.stack[i]);
        *p++ = o.folded[i] ? 1.0f : 0.0f;
        *p++ = o.all_in[i] ? 1.0f : 0.0f;
        *p++ = chips(o.street_contrib[i]);
        *p++ = chips(o.total_contrib[i]);
    }
    if (o.button >= 0) p[((o.button - v) % n + n) % n] = 1.0f;
    return x;
}

Mask legalMask(const Observation& o) {
    Mask m;
    m.fill(kIllegal);
    for (const auto& s : o.legal)
        if (s.allowed) m[specIndex(s)] = 0.0f;
    return m;
}

Mask noFreeFold(Mask m) {
    if (m[1] == 0.0f) m[0] = kIllegal;
    return m;
}

int argmax(const std::vector<float>& q, const Mask& mask) {
    if (q.size() != mask.size()) bad("argmax: sizes differ");
    int best = 0;
    float top = q[0] + mask[0];
    for (int i = 1; i < kActions; ++i) {
        float v = q[i] + mask[i];
        if (v > top) { top = v; best = i; }
    }
    return best;
}

Mlp Mlp::fromManifest(const std::string& manifest, const std::string& bin) {
    static_assert(std::endian::native == std::endian::little, "the .bin is little-endian float32");
    const Json m = Json::parse(manifest);
    if (m["dtype"].string != "float32" || m["byte_order"].string != "little") bad("not little-endian float32");
    const Json& arch = m["architecture"];
    if (arch["class"].string != "MLPModel" || arch["sequence_input"].boolean) bad("not an MLP on one state");
    if (m["bytes"].integer() != static_cast<int>(bin.size())) bad("binary size differs from the manifest");

    auto read = [&](const Json& t, size_t count) {
        size_t off = static_cast<size_t>(t["offset"].integer()), nbytes = static_cast<size_t>(t["nbytes"].integer());
        if (nbytes != count * sizeof(float) || off + nbytes > bin.size()) bad("tensor " + t["name"].string + " out of range");
        std::vector<float> out(count);
        std::memcpy(out.data(), bin.data() + off, nbytes);
        return out;
    };
    Mlp net;
    const Json& ts = m["tensors"];
    if (ts.size() == 0 || ts.size() % 2) bad("expected weight and bias pairs");
    for (size_t k = 0; k < ts.size(); k += 2) {
        const Json &w = ts[k], &b = ts[k + 1];
        const std::string prefix = w["name"].string.substr(0, w["name"].string.rfind('.'));
        if (w["name"].string != prefix + ".weight" || b["name"].string != prefix + ".bias" ||
            w["shape"].size() != 2 || b["shape"].size() != 1 || b["shape"][0].integer() != w["shape"][0].integer())
            bad("tensor pair " + w["name"].string + " is not a Linear layer");
        Layer L;
        L.out = w["shape"][0].integer();
        L.in = w["shape"][1].integer();
        if (L.out <= 0 || L.in <= 0 || (!net.layers_.empty() && L.in != net.layers_.back().out)) bad("layers do not chain");
        L.w = read(w, static_cast<size_t>(L.out) * L.in);
        L.b = read(b, L.out);
        net.layers_.push_back(std::move(L));
    }
    if (net.inputSize() != arch["input_dim"].integer() || net.outputSize() != arch["output_dim"].integer())
        bad("input or output size differs from the architecture");
    if (net.outputSize() != kActions) bad("expected 7 outputs");
    return net;
}

std::vector<float> Mlp::forward(const std::vector<float>& x) const {
    if (static_cast<int>(x.size()) != inputSize()) bad("input size " + std::to_string(x.size()));
    std::vector<float> h = x;
    for (size_t k = 0; k < layers_.size(); ++k) {
        const Layer& L = layers_[k];
        const bool relu = k + 1 < layers_.size();
        std::vector<float> y(L.out);
        for (int o = 0; o < L.out; ++o) {
            const float* w = L.w.data() + static_cast<size_t>(o) * L.in;
            double acc = 0;  // in double, in index order: the same sums on every platform
            for (int i = 0; i < L.in; ++i) acc += static_cast<double>(w[i]) * h[i];
            float v = static_cast<float>(acc + L.b[o]);
            y[o] = relu && v < 0.0f ? 0.0f : v;
        }
        h = std::move(y);
    }
    return h;
}

Decision decide(const Mlp& net, const Observation& o, bool noFreeFoldRule, int bigBlind) {
    Decision d;
    d.state = encode(o, bigBlind);
    d.mask = legalMask(o);
    if (noFreeFoldRule) d.mask = noFreeFold(d.mask);
    d.q = net.forward(d.state);
    d.action = argmax(d.q, d.mask);
    return d;
}

Observation observationFromJson(const Json& j) {
    Observation o;
    auto ints = [&](const char* key, auto& out) {
        const Json& a = j[key];
        out.clear();
        for (size_t i = 0; i < a.size(); ++i) out.push_back(static_cast<typename std::decay_t<decltype(out)>::value_type>(a[i].integer()));
    };
    auto card = [](const Json& c) {
        if (c.size() != 2) bad("bad card");
        int r = c[0].integer(), s = c[1].integer();
        if ((r != 0 && (r < 2 || r > 14)) || s < 0 || s > 4) bad("bad card");
        return Card(static_cast<Suits>(s), static_cast<Ranks>(r));
    };
    o.viewer_id = j["viewer_id"].integer();
    o.street = j["street"].integer();
    o.current_player_id = j["current_player_id"].integer();
    o.button = j["button"].integer();
    o.hand_in_progress = j["hand_in_progress"].integer();
    o.terminal = j["terminal"].integer();
    o.winner_id = j["winner_id"].integer();
    o.pot_total = j["pot_total"].integer();
    o.to_call = j["to_call"].integer();
    o.min_raise = j["min_raise"].integer();
    o.viewer_to_call = j["viewer_to_call"].integer();
    if (j["hole"].size() != 2 || j["board"].size() != 5) bad("expected 2 hole and 5 board cards");
    for (int i = 0; i < 2; ++i) o.hole[i] = card(j["hole"][i]);
    for (int i = 0; i < 5; ++i) o.board[i] = card(j["board"][i]);
    ints("alive", o.alive);
    ints("in_hand", o.in_hand);
    ints("can_act", o.can_act);
    ints("stack", o.stack);
    ints("street_contrib", o.street_contrib);
    ints("total_contrib", o.total_contrib);
    ints("folded", o.folded);
    ints("all_in", o.all_in);
    const size_t n = o.stack.size();
    if (n == 0 || o.street_contrib.size() != n || o.total_contrib.size() != n || o.folded.size() != n ||
        o.all_in.size() != n || o.viewer_id < 0 || o.viewer_id >= static_cast<int>(n) || o.button >= static_cast<int>(n))
        bad("seat arrays or seats inconsistent");
    const Json& legal = j["legal"];
    for (size_t i = 0; i < legal.size(); ++i) {
        const Json& x = legal[i];
        int type = x["type"].integer(), size = x["raise_size"].integer();
        if (type < 0 || type > 4 || size < 0 || size > 2) bad("bad legal entry");
        ActionSpec s;
        s.type = static_cast<ActionType>(type);
        s.raise_size = static_cast<RaiseSize>(size);
        s.raise_to = x["raise_to"].integer();
        s.call_amount = x["call_amount"].integer();
        s.min_raise_to = x["min_raise_to"].integer();
        s.max_raise_to = x["max_raise_to"].integer();
        s.allowed = x["allowed"].integer();
        o.legal.push_back(s);
    }
    return o;
}

}  // namespace agent
