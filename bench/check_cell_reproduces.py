"""Recompute one committed evaluation cell with bench/eval_demo.py's own cell() (no cache), writing
to a scratch folder instead of bench/raw/demo_cells, and compare it with the committed cell. Run it
after an engine change (with the extension rebuilt) to check that the measured numbers still
describe the code. From the repository root, with the venv active:

    python bench/check_cell_reproduces.py mlp-s-seed3 tight /tmp/cells
"""
import json, sys, time
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import eval_demo
run, base = sys.argv[1], sys.argv[2]
committed = json.loads((eval_demo.CELLS / f"{run}_{base}.json").read_text())
eval_demo.CELLS = Path(sys.argv[3])
t0 = time.time()
c = eval_demo.cell((run, base, eval_demo.EVAL_DEALS, True))
keys = ("bb100", "ci95_lo", "ci95_hi", "sd_bb_per_deal", "deals", "hands", "actions", "model_sha256")
for k in keys:
    print(f"{k}: committed {committed[k]!r} recomputed {c[k]!r} {'same' if committed[k] == c[k] else 'DIFFERENT'}")
print("all equal:", all(committed[k] == c[k] for k in keys), f"({time.time() - t0:.0f} s, commit {c['commit']})")
