#!/bin/sh
# Every native test under AddressSanitizer and UndefinedBehaviorSanitizer (any undefined behaviour
# fails the test), into bench/raw/ctest_sanitizers.txt. From the repository root; about an hour on an
# M1 Pro, nearly all of it the exhaustive evaluator. Paths in the output are shown relative to the
# repository: the repository's absolute path is stripped.
set -e
root=$(pwd)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined" >/dev/null
cmake --build build-asan >/dev/null
{
  echo "# commit $(git rev-parse --short HEAD)$(git status --porcelain --untracked-files=no -- . ':(exclude)bench/raw' | grep -q . && echo ' (dirty)'), $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(uname -sr), $(c++ --version | head -1)"
  echo "# sh bench/ctest_sanitizers.sh (Debug, -fsanitize=address,undefined -fno-sanitize-recover=undefined; ctest --test-dir build-asan --output-on-failure)"
  echo "# paths are shown relative to the repository"
  ctest --test-dir build-asan --output-on-failure 2>&1
} | sed "s|$root/||g" > bench/raw/ctest_sanitizers.txt
grep -q "100% tests passed" bench/raw/ctest_sanitizers.txt  # the pipe hides ctest's exit status
