# The capture tables of MODELS.md's `huber` and `sgd` sections and HUBER.md §9
# (task 277): one online pass of Huber regression can be captured by the
# outliers -- the first fits are close to least squares, the scale learns the
# pulled fit's residuals, the cut is drawn wide -- and `huber_refit_rows`
# refits each target's first rows to convergence against it.
#
# The stream is HUBER.md §2's: y = 1000 + 2x + 0.5e, +50 on a share of the
# rows with x > 0 (the share given is of all rows). Over seeds 100.. it counts
# the streams whose final slope is more than 0.05, and more than 0.25, from
# statsmodels' converged RLM (HuberT(1.345), the MAD scale), and gives the
# median gap, the worst and the worst seed. Under a half-life the oracle is
# RLM's loop with the decayed weights as prior weights (statsmodels' WLS and
# HuberT weights, the scale the weighted median of |r| over 0.6745).
#
#   uv run python scripts/huber/m6_refit_capture.py --model huber --rows 20000
#   uv run python scripts/huber/m6_refit_capture.py --model sgd --rows 2000 --refit 0,200,500
#   uv run python scripts/huber/m6_refit_capture.py --scale rms --shares 0.05,0.125
import argparse
import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

import polars_online as po

warnings.simplefilter("ignore")
DELTA = 1.345
Q3 = 0.6744897501960817


def stream(n, share, seed, level=1000.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
    y[(rng.random(n) < 2 * share) & (x > 0)] += 50.0
    return pl.DataFrame({"x": x, "y": y})


def wmedian(a, v):
    o = np.argsort(a)
    a, v = a[o], v[o]
    c = np.cumsum(v)
    return float(a[int(np.searchsorted(c, 0.5 * c[-1]))])


def oracle(x, y, half_life):
    X = sm.add_constant(x, has_constant="add")
    if np.isinf(half_life):
        fit = sm.RLM(y, X, M=HuberT(DELTA)).fit(scale_est="mad", conv="coefs", tol=1e-11)
        return float(fit.params[1])
    v = 0.5 ** ((len(y) - 1 - np.arange(len(y))) / half_life)
    keep = v > 1e-300
    X, y, v = X[keep], y[keep], v[keep]
    b = sm.WLS(y, X, weights=v).fit().params
    for _ in range(200):
        r = y - X @ b
        s = wmedian(np.abs(r), v) / Q3
        b_new = sm.WLS(y, X, weights=v * HuberT(DELTA).weights(r / s)).fit().params
        if np.max(np.abs(b_new - b)) <= 1e-11 * (1 + np.max(np.abs(b_new))):
            return float(b_new[1])
        b = b_new
    return float(b[1])


def slope(model, df, half_life, rows, scale):
    c = dict(targets=["y"], features=["x"], half_life=half_life, huber_refit_rows=rows)
    if scale != "mad":
        c["huber_scale"] = scale
    huber = model == "huber"
    spec = po.spec.huber("m", **c) if huber else po.spec.sgd("m", loss="huber", **c)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.coef("m")["coef"].to_list()[1]


p = argparse.ArgumentParser()
p.add_argument("--model", default="huber", choices=["huber", "sgd"])
p.add_argument("--rows", type=int, default=20_000)
p.add_argument("--seeds", type=int, default=40)
p.add_argument("--half-life", type=float, default=float("inf"))
p.add_argument("--refit", default="0,500,1000")
p.add_argument("--scale", default="mad", choices=["mad", "rms"])
p.add_argument("--shares", default="0.05,0.10,0.125,0.15,0.20")
a = p.parse_args()
refits = [int(v) for v in a.refit.split(",")]

print(
    f"{a.model}, huber_scale={a.scale!r}, {a.rows:,} rows, half-life {a.half_life:g}, "
    f"seeds 100-{99 + a.seeds}: per huber_refit_rows, the streams more than 0.05 / more than "
    "0.25 in slope from statsmodels' RLM; the median gap; the worst, and its seed"
)
print("| outliers | " + " | ".join(f"huber_refit_rows={r}" for r in refits) + " |")
print("|---|" + "---|" * len(refits))
for share in [float(v) for v in a.shares.split(",")]:
    gaps = {r: [] for r in refits}
    seeds = range(100, 100 + a.seeds)
    for seed in seeds:
        df = stream(a.rows, share, seed)
        want = oracle(df["x"].to_numpy(), df["y"].to_numpy(), a.half_life)
        for r in refits:
            gaps[r].append(abs(slope(a.model, df, a.half_life, r, a.scale) - want))
    cells = []
    for r in refits:
        g = np.array(gaps[r])
        worst = int(np.argmax(g))
        cells.append(
            f"{(g > 0.05).sum()} / {(g > 0.25).sum()} of {a.seeds}; "
            f"median {np.median(g):.4f}; worst {g[worst]:.3f} (seed {100 + worst})"
        )
    print(f"| {share:.1%} | " + " | ".join(cells) + " |", flush=True)
