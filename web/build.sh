#!/usr/bin/env bash
# Builds the WebAssembly module into web/dist/: poker.mjs (ES module factory) and poker.wasm, with
# the selected network (weights/demo/seats.json) compiled in, plus BUILD.txt and SHA256SUMS. Also
# builds the test module, build/wasm-test/poker_test.mjs: the same plus web/test_hooks.cpp, for
# web/test.mjs only; it is not released.
# Needs em++ on PATH (emsdk: source emsdk_env.sh) and node. It calls em++ directly, as OrderbookSim's
# web/build.sh does. Fails if the weights' sha256 differs from their manifest's.
# SOURCE_COMMIT, if set, is the commit BUILD.txt names (CI passes a pull request's head: GitHub builds
# a temporary merge of it into the base branch).
set -euo pipefail
cd "$(dirname "$0")/.."

# the previous build goes first, so that a failed check below cannot leave stale files
rm -rf web/dist build/wasm-test

# one network in all three agent seats (Table has one)
run=$(node -e "const s = JSON.parse(require('fs').readFileSync('weights/demo/seats.json', 'utf8')).seats;
if (s.length !== 3 || new Set(s).size !== 1) { console.error('seats.json: expected one network in all three seats'); process.exit(1); }
console.log(s[0]);")
manifest=weights/demo/$run.json
bin=weights/demo/$run.bin
want=$(node -p "JSON.parse(require('fs').readFileSync('$manifest', 'utf8')).sha256")
got=$(shasum -a 256 "$bin" | cut -d ' ' -f 1)
if [ "$got" != "$want" ]; then
    echo "error: $bin has sha256 $got; $manifest says $want" >&2
    exit 1
fi

mkdir -p web/dist build/wasm-test
version=$(em++ --version | head -n 1)
echo "$version"

# -ffp-contract=off: no fused multiply-add, as in the native build (CMakeLists.txt), so both compute
# the same Q-values. The #embed paths are relative to web/.
flags=(-std=c++20 -O2 -fexceptions -ffp-contract=off -Wno-c23-extensions -I PokerEngine -I web
       "-DPOKER_MANIFEST=\"../$manifest\"" "-DPOKER_WEIGHTS=\"../$bin\""
       -lembind -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH -sFILESYSTEM=0
       --post-js web/post.js)
sources=(web/poker_wasm.cpp PokerEngine/Agent.cpp PokerEngine/GameEngine.cpp PokerEngine/Player.cpp)
em++ "${flags[@]}" "${sources[@]}" -o web/dist/poker.mjs
em++ "${flags[@]}" "${sources[@]}" web/test_hooks.cpp -o build/wasm-test/poker_test.mjs

built=$(git rev-parse HEAD)
commit=${SOURCE_COMMIT:-$built}
[ "$commit" != "$built" ] && commit="$commit (built as $built: for a pull request, GitHub's merge of it into the base branch)"
git status --porcelain -- PokerEngine web weights | grep -q . && commit="$commit (dirty: the sources or weights differ from it)"
size() { printf '%s: %s bytes (gzip -9: %s bytes)\n' "$1" "$(wc -c < "web/dist/$1" | tr -d ' ')" "$(gzip -9 -c "web/dist/$1" | wc -c | tr -d ' ')"; }
{
    echo "$version"
    echo "source commit: $commit"
    echo "weights: $bin, sha256 $got (as in $manifest)"
    echo "command: em++ ${flags[*]} ${sources[*]} -o web/dist/poker.mjs"
    size poker.mjs
    size poker.wasm
} > web/dist/BUILD.txt
(cd web/dist && shasum -a 256 poker.mjs poker.wasm BUILD.txt > SHA256SUMS)
cat web/dist/BUILD.txt
