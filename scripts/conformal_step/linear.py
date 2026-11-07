"""The derivation's check: the tracker on iid Gaussian residuals (sigma = 1,
known), each row's feedback exactly D releases late. The theory says the
radius's spread is set by k*D, k = rate * g(alpha): flat below 1/e, rising
towards pi/2, a limit cycle above (docs/records/CONFORMAL-STEP.md)."""

import json
import math
import sys
from pathlib import Path

import numba as nb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold  # noqa: E402
import streams  # noqa: E402


@nb.njit(cache=True)
def run(z, D, rate, alpha, q0):
    n = len(z)
    q = q0
    shown = np.empty(n)
    for i in range(n):
        shown[i] = q
        if i >= D:
            j = i - D
            miss = 1.0 if abs(z[j]) > shown[j] else 0.0
            q = max(q + rate * (miss - alpha), 0.0)
    return shown


alpha = 0.1
g = fold.gain(alpha)
qstar = fold.norm_ppf(1 - alpha / 2)
z = np.random.default_rng(7).standard_normal(400_000)
res = []
print("| D | k*D | rate | coverage | mean half-width / q* | sd of the radius / q* |")
print("|---:|---:|---:|---:|---:|---:|")
for D in (10, 100, 1000):
    for kd in (0.1, 1 / math.e, 0.5, 1.0, 1.5, 2.0, 3.0):
        rate = kd / (g * D)
        s = run(z, D, rate, alpha, qstar)[100_000:]
        cov = float(np.mean(np.abs(z[100_000:]) <= s))
        res.append(
            {
                "D": D,
                "kD": kd,
                "rate": rate,
                "coverage": cov,
                "mean": float(s.mean() / qstar),
                "sd": float(s.std() / qstar),
            }
        )
        print(
            f"| {D} | {kd:.2f} | {rate:.5f} | {cov:.4f} "
            f"| {s.mean() / qstar:.3f} | {s.std() / qstar:.3f} |"
        )
(streams.CACHE / "linear.json").write_text(json.dumps(res), encoding="utf-8")
