#!/bin/sh
# The native tests (Release, the exhaustive evaluator included), the WebAssembly build and its tests,
# into bench/raw/web_tests.txt. From the repository root, with em++ on PATH (emsdk_env.sh) and node.
# Paths in the output are shown relative to the repository: the repository's absolute path is stripped.
set -e
root=$(pwd)
out=bench/raw/web_tests.txt
# Both builds first, outside the pipe below (which hides exit status): a failed build stops the
# script here, before any test runs, rather than leaving the tests to an earlier build's binaries.
fail() { echo "bench/web_tests.sh: $1 failed; nothing was tested and $out is unchanged" >&2; exit 1; }
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null || fail 'cmake -S . -B build'
cmake --build build >/dev/null || fail 'the native build (cmake --build build)'
wasm=$(web/build.sh 2>&1) || { printf '%s\n' "$wasm" >&2; fail 'web/build.sh'; }
{
  echo "# commit $(git rev-parse --short HEAD)$(git status --porcelain --untracked-files=no -- . ':(exclude)bench/raw' | grep -q . && echo ' (dirty)'), $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(uname -sr), $(c++ --version | head -1), cmake $(cmake --version | head -1 | cut -d ' ' -f 3), node $(node --version)"
  echo "# sh bench/web_tests.sh; paths are shown relative to the repository"
  echo "# load average before: $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
  echo "# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure"
  ctest --test-dir build --output-on-failure 2>&1 | sed -n '/tests passed/,$p;/tests failed/,$p;/Test #/p'
  for t in test_json test_hand_result; do echo "# ./build/$t"; ./build/$t; done
  for t in test_agent test_table; do echo "# ./build/$t weights/demo"; ./build/$t weights/demo; done
  echo "# web/build.sh"
  printf '%s\n' "$wasm" | tail -6
  echo "# node web/test.mjs"
  node web/test.mjs 2>&1
  echo "# load average after: $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
  echo "# web/dist/SHA256SUMS"
  cat web/dist/SHA256SUMS
} | sed "s|$root/||g" > "$out"
grep -q "^PASS" "$out" && grep -q "100% tests passed" "$out"  # the pipe hides the commands' exit status
