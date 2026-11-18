"""Time random.sample(buffer, 1024), the replay draw, from a deque and from a list of n items.

    python bench/bench_replay_sample.py [n ...]
"""
import random
import sys
import time
from collections import deque

for n in map(int, sys.argv[1:] or ["100000", "1000000"]):
    d = deque((i, i) for i in range(n))
    lst = list(d)
    for name, buf in (("deque", d), ("list", lst)):
        random.seed(0)
        t = time.perf_counter()
        for _ in range(20):
            random.sample(buf, 1024)
        print(f"n={n:,}: {name} {(time.perf_counter() - t) / 20 * 1e3:.2f} ms per 1,024-sample draw (mean of 20)")
