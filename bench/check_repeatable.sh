#!/bin/sh
# Train with the same seed twice, in separate processes, and compare the results row for row:
#  1. the README set-up (bench/rerun_training.py: four networks, tournament, per-hand log);
#  2. short demo runs (bench/train_demo.py, mlp-s and gru, seed 11): checkpoint logs without the
#     timing fields, and the kept weights.
# From the repository root, with the venv active:  sh bench/check_repeatable.sh
set -e
d=$(mktemp -d)
echo "# commit $(git rev-parse --short HEAD), $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), load average $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
for r in a b; do
  echo "# python bench/rerun_training.py --seed 2 --train-hands 2000 --log $r.csv$([ $r = b ] && echo ' --compare a.csv')"
  python bench/rerun_training.py --seed 2 --train-hands 2000 --log "$d/$r.csv" $([ $r = b ] && echo "--compare $d/a.csv") 2>&1 \
    | grep -v -e "it/s" -e "Warning" -e "warnings.warn" | sed "s|$d/||g"
done
for c in "mlp-s 20000" "gru 4000"; do
  set -- $c
  for r in a b; do
    echo "# python bench/train_demo.py --candidate $1 --seed 11 --hands $2 --checkpoints 2 --out $1_$r"
    python bench/train_demo.py --candidate "$1" --seed 11 --hands "$2" --checkpoints 2 --out "$d/$1_$r" | sed "s|$d/||g"
  done
  python - "$d/$1_a" "$d/$1_b" <<'EOF'
import json, sys
from pathlib import Path
import torch
a, b = (Path(p) for p in sys.argv[1:])
strip = lambda p: [{k: v for k, v in json.loads(l).items() if k not in ("wall_s", "cpu_s")} for l in open(p / "log.jsonl")]
same_log = strip(a) == strip(b)
wa, wb = (torch.load(p / "model.pt", weights_only=True) for p in (a, b))
same_w = list(wa) == list(wb) and all(torch.equal(wa[k], wb[k]) for k in wa)
print(f"{a.name[:-2]}: checkpoint logs {'identical' if same_log else 'DIFFER'} apart from timings; "
      f"kept weights {'identical' if same_w else 'DIFFER'}")
sys.exit(0 if same_log and same_w else 1)
EOF
done
rm -rf "$d"
