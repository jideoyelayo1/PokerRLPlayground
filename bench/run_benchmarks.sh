#!/bin/sh
# Re-run every measurement in bench/RESULTS.md. From the repository root, with the venv active:
#   sh bench/run_benchmarks.sh            # tests + throughput (about 5 minutes)
#   sh bench/run_benchmarks.sh winrate    # also train and evaluate the agents (about an hour)
set -e
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build >/dev/null
(cd PokerEngine && python setup.py build_ext --inplace >/dev/null 2>&1)

# Commit measured, and "dirty" if tracked files outside bench/raw differ from it.
rev() { echo "# commit $(git rev-parse --short HEAD)$(git status --porcelain --untracked-files=no -- . ':(exclude)bench/raw' | grep -q . && echo ' (dirty)')"; }
machine() {
  rev
  echo "# $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(uname -sr), $(c++ --version | head -1)"
  echo "# load average before: $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
}
after() { echo "# load average after: $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"; }

ctest --test-dir build --output-on-failure -E eval_exhaustive
{ rev; ./build/eval_exhaustive; } > bench/raw/eval_exhaustive.txt
{ machine; echo "# ./build/bench_engine 200000 7, CMake Release (-O3 -DNDEBUG), single thread"; ./build/bench_engine 200000 7; after; } > bench/raw/bench_engine.txt
{ machine; echo "# python bench/bench_binding.py --hands 20000 --runs 7 (extension built by setup.py: -O3)"; python bench/bench_binding.py --hands 20000 --runs 7; after; } > bench/raw/bench_binding.txt

if [ "$1" = winrate ]; then
  { machine; python bench/winrate.py --seeds 1 2 3 --train-hands 2000 --eval-deals 1000; after; } 2>&1 | grep -v "it/s]" > bench/raw/winrate.log
  python bench/summarise_winrate.py > bench/raw/winrate_table.md
fi
