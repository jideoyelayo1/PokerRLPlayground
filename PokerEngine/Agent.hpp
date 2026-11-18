#pragma once
// The trained agent's play in C++: the state encoding of docs/state_encoding.md, the legal mask,
// the play-time option no_free_fold, the MLP forward pass read from an exported manifest and its
// float32 binary (weights/README.md), and the argmax. Standard library only.
//
// Build Agent.cpp with -ffp-contract=off (CMakeLists.txt and web/build.sh do), so that no multiply
// and add are fused: the native and WebAssembly builds then compute the same Q-values bit for bit.
#include <array>
#include <string>
#include <vector>

#include "GameEngine.hpp"
#include "Json.hpp"

namespace agent {

constexpr int kActions = 7;
using Mask = std::array<float, kActions>;
constexpr float kIllegal = -1e9f;

// Action indices 0..6, as in docs/state_encoding.md and the manifest's "actions".
inline constexpr const char* kActionNames[kActions] = {"fold", "check", "call", "raise_bb", "raise_half", "raise_pot", "allin"};
int actionIndex(const std::string& name);  // -1 if not one of kActionNames
int specIndex(const ActionSpec& spec);     // the action index of a legalActions entry
Action engineAction(int index);

// 127 + 6 n floats (151 for 4 seats) for the player viewing o (o.viewer_id).
std::vector<float> encode(const Observation& o, int bigBlind);
// 0 where o.legal marks the action allowed, kIllegal elsewhere. Read it only for the player to act.
Mask legalMask(const Observation& o);
// no_free_fold: if Check is allowed, Fold is treated as not allowed.
Mask noFreeFold(Mask m);
// argmax(q + mask) in float32, the first index on a tie (as torch.argmax).
int argmax(const std::vector<float>& q, const Mask& mask);

// An MLP exported by bench/export_weights.py: Linear layers with ReLU after every one but the last.
class Mlp {
public:
    // manifest: the <run>.json text; bin: the <run>.bin bytes. Throws std::runtime_error if they
    // do not describe an MLP of chained Linear layers, or the sizes disagree.
    static Mlp fromManifest(const std::string& manifest, const std::string& bin);
    std::vector<float> forward(const std::vector<float>& x) const;
    int inputSize() const { return layers_.front().in; }
    int outputSize() const { return layers_.back().out; }

private:
    struct Layer { int in = 0, out = 0; std::vector<float> w, b; };  // w is [out][in], row-major
    std::vector<Layer> layers_;
};

struct Decision {
    std::vector<float> state;
    Mask mask{};       // after no_free_fold, if it was applied
    std::vector<float> q;
    int action = 0;
};
Decision decide(const Mlp& net, const Observation& o, bool noFreeFoldRule, int bigBlind = 2);

// A parity record's raw "observation" (weights/README.md, "Parity records") as an Observation.
Observation observationFromJson(const Json& j);

}  // namespace agent
