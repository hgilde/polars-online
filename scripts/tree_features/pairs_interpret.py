"""InterpretML's FAST (measure_interactions) on the same planted data, on y and
with init_score = y - residual (the same online additive model's prediction)."""

import json
import sys
from pathlib import Path

import numpy as np
from interpret.utils import measure_interactions

TRUE = {(4, 5), (6, 7), (8, 9), (10, 11)}


def ap(ranked):
    hits = 0
    s = 0.0
    for r, k in enumerate(ranked, 1):
        if tuple(sorted(k)) in TRUE:
            hits += 1
            s += hits / r
    return s / len(TRUE)


d = sys.argv[1]
out = {}
for n in (20000, 200000):
    for seed in (0, 1, 2):
        z = np.load(f"{d}/pairs_{n}_{seed}.npz")
        x, y = z["x"], z["y"]
        r = np.load(f"{d}/pairs_{n}_{seed}_resid.npy")
        row = {}
        for bins in (8, 32):
            a = measure_interactions(x, y, max_interaction_bins=bins)
            b = measure_interactions(x, y, init_score=y - r, max_interaction_bins=bins)
            row[f"bins{bins} on y"] = ap([k for k, _ in a])
            row[f"bins{bins} init_score"] = ap([k for k, _ in b])
            if seed == 0:
                print(
                    n,
                    bins,
                    "top on y",
                    [k for k, _ in a[:6]],
                    "| with init_score",
                    [k for k, _ in b[:6]],
                )
        out[f"{n}/{seed}"] = row
        print(n, seed, row, flush=True)
Path(f"{d}/pairs_interpret.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
