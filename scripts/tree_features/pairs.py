"""Candidate 3: a pairwise interaction screen from B x B tables of
(sum w, sum w r, sum w r^2) per feature pair, r = y or an online model's
residual. Scores: FAST (Lou et al. 2013, sec. 4.1: best one-cut-per-feature
2x2 quadrant-means RSS on the residual) and a two-way ANOVA interaction F
(additive fit to the weighted cell means by backfitting). Planted truth."""

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import polars_online as po  # noqa: E402

P = 20
TRUE = {(4, 5), (6, 7), (8, 9), (10, 11)}


def make(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, P))
    x[:, 3] = 0.8 * x[:, 2] + 0.6 * x[:, 3]  # correlated with a nonlinear main effect
    x[:, 11] = 0.5 * x[:, 10] + np.sqrt(0.75) * x[:, 11]  # a correlated interacting pair
    x[:, 2] = x[:, 14] + x[:, 15] + 0.3 * x[:, 2]  # x2 is (nearly) x14 + x15
    y = (
        x[:, 0]
        + 0.5 * x[:, 2] ** 2  # main effects, one nonlinear
        + 0.5 * x[:, 4] * x[:, 5]  # product
        + 0.7 * (np.sign(x[:, 6]) * np.sign(x[:, 7]))  # XOR: no main effect at all
        + 1.0 * ((x[:, 8] > 1) & (x[:, 9] > 1))  # AND of thresholds
        + 0.3 * x[:, 10] * x[:, 11]  # weaker product, correlated pair
        + rng.standard_normal(n)
    )
    return x, y


def edges_of(x, warm, B):
    q = np.quantile(x[:warm], np.linspace(0, 1, B + 1)[1:-1], axis=0)
    return q  # (B-1, P)


def binned(x, e):
    return np.column_stack(
        [np.searchsorted(e[:, j], x[:, j], side="right") for j in range(x.shape[1])]
    )


def tables(b, r, B, w=None):
    w = np.ones(len(r)) if w is None else w
    T = {}
    for i, j in itertools.combinations(range(b.shape[1]), 2):
        c = b[:, i] * B + b[:, j]
        T[(i, j)] = np.stack(
            [
                np.bincount(c, w, B * B),
                np.bincount(c, w * r, B * B),
                np.bincount(c, w * r * r, B * B),
            ]
        ).reshape(3, B, B)
    return T


def fast(T):
    """Lou et al. 4.1: min over (ci, cj) of the 2x2 quadrant-means RSS; the score
    is the reduction from one mean, so larger is a stronger interaction."""
    W, S, Q = T
    cW = W.cumsum(0).cumsum(1)
    cS = S.cumsum(0).cumsum(1)
    tW, tS = cW[-1, -1], cS[-1, -1]
    best = 0.0
    B = W.shape[0]
    for ci in range(B - 1):
        for cj in range(B - 1):
            a_w, a_s = cW[ci, cj], cS[ci, cj]
            bw, bs = cW[ci, -1] - a_w, cS[ci, -1] - a_s
            cw, cs = cW[-1, cj] - a_w, cS[-1, cj] - a_s
            dw, ds = tW - a_w - bw - cw, tS - a_s - bs - cs
            g = (
                sum(s * s / w for w, s in ((a_w, a_s), (bw, bs), (cw, cs), (dw, ds)) if w > 0)
                - tS * tS / tW
            )
            best = max(best, g)
    return best


def anova(T, iters=50):
    """Two-way interaction F on the table: cell means against the best additive
    fit mu + a_i + b_j (weighted, by backfitting); within-cell variance from Q."""
    W, S, Q = T
    nz = W > 0
    m = np.where(nz, S / np.where(nz, W, 1), 0.0)
    mu = S.sum() / W.sum()
    a = np.zeros(W.shape[0])
    bb = np.zeros(W.shape[1])
    for _ in range(iters):
        ra = (W * (m - mu - bb[None, :])).sum(1)
        wa = W.sum(1)
        a = np.where(wa > 0, ra / np.where(wa > 0, wa, 1), 0)
        rb = (W * (m - mu - a[:, None])).sum(0)
        wb = W.sum(0)
        bb = np.where(wb > 0, rb / np.where(wb > 0, wb, 1), 0)
        mu = (W * (m - a[:, None] - bb[None, :])).sum() / W.sum()
    ss_int = (W * (m - mu - a[:, None] - bb[None, :]) ** 2).sum()
    ss_within = (Q - np.where(nz, S * S / np.where(nz, W, 1), 0)).sum()
    n = W.sum()
    cells = nz.sum()
    df_int = max((W.shape[0] - 1) * (W.shape[1] - 1), 1)
    return (ss_int / df_int) / (ss_within / max(n - cells, 1))


def avg_precision(scores):
    ranked = sorted(scores, key=lambda k: -scores[k])
    hits = 0
    ap = 0.0
    for r, k in enumerate(ranked, 1):
        if k in TRUE:
            hits += 1
            ap += hits / r
    return ap / len(TRUE), ranked[:6]


def residual_online(x, y, e, B):
    """An online additive model: ewridge on one-hot bins of every feature (B-1
    columns each), out-of-sample residual."""
    b = binned(x, e)
    cols = {f"b{j}_{k}": (b[:, j] == k).astype(float) for j in range(P) for k in range(1, B)}
    d = pl.DataFrame({"c": np.arange(len(y), dtype=float), "y": y, **cols})
    s = po.spec.ewridge(
        "o", targets=["y"], features=list(cols), clock="c", gap_cap=10.0, half_life=1e9, ridge=1e-3
    )
    p = po.ModelBank([s]).fit_predict(d)["o"].struct.field("pred_y").to_numpy()
    return y - np.nan_to_num(p, nan=0.0)


if __name__ == "__main__":
    print(C.load_line())
    out = {}
    for n in (20_000, 200_000):
        for seed in (0, 1, 2):
            x, y = make(seed, n)
            np.savez(str(C.OUT / f"pairs_{n}_{seed}.npz"), x=x, y=y)
            row = {}
            for B in (8, 16):
                e = edges_of(x, 2000, B)
                b = binned(x, e)
                t0 = time.perf_counter()
                T = tables(b, y, B)
                tt = time.perf_counter() - t0
                r = residual_online(x, y, e, 8 if B == 8 else 8)
                np.save(str(C.OUT / f"pairs_{n}_{seed}_resid.npy"), r)
                Tr = tables(b, r, B)
                for tag, TT in (("y", T), ("resid", Tr)):
                    for sname, f in (("fast", fast), ("anova", anova)):
                        ap, top = avg_precision({k: f(v) for k, v in TT.items()})
                        row[f"B{B} {sname} on {tag}"] = ap
                        if seed == 0:
                            print(
                                f"   n={n} B={B} {sname:5s} on {tag:5s}: AP {ap:.3f} top {top}",
                                flush=True,
                            )
                row[f"B{B} table build s (190 pairs)"] = tt
            out[f"{n}/{seed}"] = row
            print(
                f"n={n} seed={seed}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()),
                flush=True,
            )
    Path(str(C.OUT / "pairs.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(C.load_line())
