"""The experiments behind ``docs/BOOSTED-TREES.md``, on the ``ogbt_proto`` model.

Prequential (predict-then-learn) MSE on Friedman-1 with an irregular clock,
optionally drifting, against a noise floor of 1.0. Baselines: the EW mean,
``ewridge`` from this package, XGBoost refit on a rolling window (the honest
out-of-sample use of a batch learner), and XGBoost's structure with the leaf
values refreshed online (what its ``refresh`` updater would give with decay).

    uv run python scripts/ogbt_experiments.py all           # without xgboost
    uv run --with xgboost --with scikit-learn python scripts/ogbt_experiments.py all

Experiments: ``baselines``, ``knobs``, ``pool``, ``negatives``,
``invariance``; ``all`` runs them in that order. Each prints a table whose
rows are quoted in the document. XGBoost is optional -- its rows are skipped
with a note when it cannot be imported. On macOS the xgboost wheel needs
libomp, which scikit-learn's wheel bundles:

    DYLD_LIBRARY_PATH=$(uv run --with scikit-learn python -c \\
        "import os,sklearn;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \\
        uv run --with xgboost --with scikit-learn python scripts/ogbt_experiments.py all
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ogbt_proto import Cfg, LeafRefresh, OnlineGBT, friedman, mse  # noqa: E402

try:
    import xgboost as xgb
except Exception:  # noqa: BLE001 -- any import failure (missing wheel, missing libomp) is "not available"
    xgb = None

N, P = 24000, 10
XGB_PARAMS = {
    "tree_method": "hist",
    "max_depth": 4,
    "eta": 0.3,
    "lambda": 1.0,
    "nthread": 4,
    "max_bin": 32,
    "verbosity": 0,
}
# The warm-started design that the document recommends.
WARM = dict(
    n_trees=20,
    max_depth=4,
    n_bins=32,
    eta=0.3,
    grace=100,
    grow_every=50,
    bin_rows=2000,
    warm_start=True,
    halflife=3000.0,
)


def segments(n: int, fine: bool) -> list[tuple[int, int]]:
    h = n // 2
    if fine:
        return [(2000, 4000), (4000, 8000), (8000, h), (h, h + n // 8), (h + n // 8, n)]
    return [(2000, 8000), (8000, h), (h, h + n // 8), (h + n // 8, n)]


def header(drift: str, seed: int, y, f, fine: bool) -> None:
    cols = (
        "[2k,4k) [4k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n)"
        if fine
        else "[2k,8k) [8k,n/2) [n/2,5n/8) [5n/8,n)"
    )
    print(f"=== drift={drift} seed={seed} n={N}: MSE on {cols}; noise floor {np.var(y - f):.3f}")


def row(label: str, pred, y, segs, extra: str = "") -> None:
    s = "  ".join(f"{mse(pred[a:b], y[a:b], 0):6.3f}" for a, b in segs)
    print(f"{label:<52} {s}  {extra}")


def run_ours(cfg: Cfg, X, y, d_clock, w, label, segs):
    t0 = time.time()
    m = OnlineGBT(cfg, X.shape[1])
    pred, _ = m.fit_chunk(X, y, d_clock, w)
    hist = sum(1 for t in m.trees for nd in t.nodes() if nd.hist_G is not None)
    nodes = sum(t.n_nodes for t in m.trees)
    kd = m.state_doubles() / 1e3
    extra = f"{time.time() - t0:4.1f}s nodes={nodes} hist_leaves={hist} state={kd:.0f}k"
    row(label, pred, y, segs, extra)
    return m, pred


def run_ew_mean(y, d_clock, halflife):
    pred = np.full(len(y), np.nan)
    G = H = 0.0
    for i in range(len(y)):
        lam = 0.5 ** (d_clock[i] / halflife)
        G, H = lam * G, lam * H
        pred[i] = G / H if H > 0 else np.nan
        G, H = G + y[i], H + 1.0
    return pred


def run_ewridge(X, y, d_clock, halflife):
    import polars as pl

    import polars_online as po

    t = np.cumsum(d_clock)
    df = pl.DataFrame({"t": t, "y": y, **{f"x{j}": X[:, j] for j in range(X.shape[1])}})
    spec = po.spec.ewridge(
        name="r",
        targets=["y"],
        features=[f"x{j}" for j in range(X.shape[1])],
        clock="t",
        halflife=halflife,
        max_dclock=100.0,
        min_periods=X.shape[1] + 1,
    )
    out = po.fit_predict(df, [spec])
    return out.select(pl.col("r").struct.field("pred_y")).to_numpy().ravel()


def run_xgb_window(X, y, W, R, rounds, start):
    """Refit on the last W rows every R rows; predict the next R rows with it."""
    pred = np.full(len(y), np.nan)
    for s in range(start, len(y), R):
        lo = max(0, s - W)
        b = xgb.train(XGB_PARAMS, xgb.DMatrix(X[lo:s], y[lo:s]), num_boost_round=rounds)
        pred[s : s + R] = b.predict(xgb.DMatrix(X[s : s + R]))
    return pred


def run_xgb_leaf_refresh(X, y, d_clock, warm, rounds, halflife):
    """XGBoost structure from the first ``warm`` rows; leaf (G, H) refreshed online."""
    b = xgb.train(XGB_PARAMS, xgb.DMatrix(X[:warm], y[:warm]), num_boost_round=rounds)

    def leaf_ids(x):
        return b.predict(xgb.DMatrix(x), pred_leaf=True).astype(int).reshape(len(x), -1)

    m = LeafRefresh(
        leaf_ids,
        rounds,
        XGB_PARAMS["eta"],
        XGB_PARAMS["lambda"],
        halflife,
        base=float(np.mean(y[:warm])),
    )
    pred = np.empty(len(y))
    for i in range(len(y)):
        pred[i], _ = m.step(X[i], y[i], d_clock[i])
    return pred


# ---------------------------------------------------------------------------------
def exp_baselines(seed: int = 4) -> None:
    """The recommended design against the baselines, on three drift regimes."""
    for drift in ("none", "abrupt", "walk"):
        X, y, f, d_clock, w = friedman(N, P, seed=seed, drift=drift, noise=1.0)
        segs = segments(N, fine=True)
        header(drift, seed, y, f, fine=True)
        row("EW mean hl=3000", run_ew_mean(y, d_clock, 3000.0), y, segs)
        try:
            row("ewridge hl=3000 (this package)", run_ewridge(X, y, d_clock, 3000.0), y, segs)
        except ImportError as e:
            print(f"ewridge: skipped ({e})")
        if xgb is None:
            print("xgboost rows: skipped (xgboost not importable)")
        else:
            row(
                "xgb refit, window W=2000 every R=500 rows",
                run_xgb_window(X, y, 2000, 500, 20, 2000),
                y,
                segs,
            )
            row(
                "xgb refit, window W=8000 every R=500 rows",
                run_xgb_window(X, y, 8000, 500, 20, 2000),
                y,
                segs,
            )
            for hl in (math.inf, 3000.0):
                row(
                    f"xgb structure (rows<2000) + leaf refresh hl={hl}",
                    run_xgb_leaf_refresh(X, y, d_clock, 2000, 20, hl),
                    y,
                    segs,
                )
        scratch = dict(WARM, warm_start=False, bin_rows=500, stagger_rows=100)
        run_ours(
            Cfg(**scratch), X, y, d_clock, w, "ours: from scratch, staggered births, hl=3000", segs
        )
        for hl in (math.inf, 3000.0):
            run_ours(
                Cfg(**dict(WARM, halflife=hl, freeze_after_warm=True)),
                X,
                y,
                d_clock,
                w,
                f"ours: warm 2000, leaf refresh only, hl={hl}",
                segs,
            )
            run_ours(
                Cfg(**dict(WARM, halflife=hl)),
                X,
                y,
                d_clock,
                w,
                f"ours: warm 2000, grow+prune, hl={hl}  <- recommended"
                if hl == 3000.0
                else f"ours: warm 2000, grow+prune, hl={hl}",
                segs,
            )
        run_ours(
            Cfg(**dict(WARM, halflife=8000.0)),
            X,
            y,
            d_clock,
            w,
            "ours: warm 2000, grow+prune, hl=8000",
            segs,
        )
        print()


def exp_knobs(seed: int = 5) -> None:
    """What each memory / parallelism knob costs, static data."""
    X, y, f, d_clock, w = friedman(N, P, seed=seed, drift="none", noise=1.0)
    segs = segments(N, fine=False)
    header("none", seed, y, f, fine=False)
    run_ours(Cfg(**WARM), X, y, d_clock, w, "reference: M=20 d=4 B=32 grow_every=50", segs)
    for nb in (8, 16, 64):
        run_ours(Cfg(**dict(WARM, n_bins=nb)), X, y, d_clock, w, f"n_bins={nb}", segs)
    for cs in (0.5, 0.7):
        run_ours(Cfg(**dict(WARM, colsample=cs)), X, y, d_clock, w, f"colsample={cs}", segs)
    for ge in (1, 10, 500, 2000):
        run_ours(Cfg(**dict(WARM, grow_every=ge)), X, y, d_clock, w, f"grow_every={ge}", segs)
    for d in (3, 6):
        run_ours(Cfg(**dict(WARM, max_depth=d)), X, y, d_clock, w, f"max_depth={d}", segs)
    run_ours(Cfg(**dict(WARM, n_trees=50, eta=0.15)), X, y, d_clock, w, "M=50 eta=0.15", segs)
    run_ours(Cfg(**dict(WARM, n_trees=10, eta=0.5)), X, y, d_clock, w, "M=10 eta=0.5", segs)
    run_ours(Cfg(**dict(WARM, bin_rows=4000)), X, y, d_clock, w, "warm-up buffer 4000", segs)
    run_ours(Cfg(**dict(WARM, bin_rows=1000)), X, y, d_clock, w, "warm-up buffer 1000", segs)
    print()


def exp_pool(seed: int = 6) -> None:
    """A bounded histogram pool: only the P heaviest splittable leaves keep histograms."""
    for drift in ("none", "abrupt"):
        X, y, f, d_clock, w = friedman(N, P, seed=seed, drift=drift, noise=1.0)
        segs = segments(N, fine=False)
        header(drift, seed, y, f, fine=False)
        run_ours(Cfg(**WARM), X, y, d_clock, w, "unbounded histograms", segs)
        for pool in (32, 16, 8, 4):
            run_ours(Cfg(**dict(WARM, hist_pool=pool)), X, y, d_clock, w, f"hist_pool={pool}", segs)
        print()


def structures(m: OnlineGBT) -> None:
    """How many distinct trees an ensemble holds, and how many distinct root splits."""
    sig = {tuple((nd.feat, nd.cut) for nd in t.nodes()) for t in m.trees}
    roots = {(t.root.feat, t.root.cut) for t in m.trees}
    print(f"    distinct trees: {len(sig)} of {len(m.trees)}; distinct root splits: {len(roots)}")


def exp_negatives(seed: int = 4) -> None:
    """Ideas that measured worse, kept so nobody re-tries them on the same argument."""
    for drift in ("none", "abrupt"):
        X, y, f, d_clock, w = friedman(N, P, seed=seed, drift=drift, noise=1.0)
        segs = segments(N, fine=False)
        header(drift, seed, y, f, fine=False)
        scratch = dict(WARM, warm_start=False, bin_rows=500)
        stagger = dict(scratch, stagger_rows=100, recycle=False)
        m, _ = run_ours(
            Cfg(**scratch), X, y, d_clock, w, "from scratch, all 20 trees born together", segs
        )
        structures(m)
        m, _ = run_ours(
            Cfg(**stagger), X, y, d_clock, w, "from scratch, one birth per 100 rows (stagger)", segs
        )
        structures(m)
        run_ours(
            Cfg(**dict(stagger, bin_rows=2000)),
            X,
            y,
            d_clock,
            w,
            "  + bins from 2000 rows (the warm start's buffer)",
            segs,
        )
        run_ours(
            Cfg(**dict(stagger, recycle=True)),
            X,
            y,
            d_clock,
            w,
            "  + recycle: retire the oldest tree every 100 rows",
            segs,
        )
        run_ours(
            Cfg(**dict(stagger, gamma_rel=0.05)),
            X,
            y,
            d_clock,
            w,
            "  + gamma_rel=0.05 (gain must beat 5% of var g)",
            segs,
        )
        run_ours(
            Cfg(**dict(stagger, hoeffding_delta=1e-3)),
            X,
            y,
            d_clock,
            w,
            "  + hoeffding_delta=1e-3 (best-vs-second margin)",
            segs,
        )
        run_ours(
            Cfg(**dict(scratch, stage_rows=500, recycle=False)),
            X,
            y,
            d_clock,
            w,
            "stagewise: one tree grows at a time, 500 rows each",
            segs,
        )
        run_ours(
            Cfg(**dict(scratch, stagger_rows=100, recycle=True, grow_trees=5)),
            X,
            y,
            d_clock,
            w,
            "stagger + only the 5 youngest trees may grow",
            segs,
        )
        run_ours(
            Cfg(**dict(WARM, halflife=math.inf)),
            X,
            y,
            d_clock,
            w,
            "warm start, hl=inf (no forgetting)",
            segs,
        )
        run_ours(
            Cfg(**dict(WARM, prune=False)),
            X,
            y,
            d_clock,
            w,
            "warm start, grow but never collapse",
            segs,
        )
        run_ours(
            Cfg(**WARM), X, y, d_clock, w, "warm start, grow+prune, hl=3000 (recommended)", segs
        )
        print()


def exp_invariance(seed: int = 6) -> None:
    """The guarantees, checked on the recommended design with a histogram pool."""
    X, y, f, d_clock, w = friedman(N, P, seed=seed, drift="abrupt", noise=1.0)
    cfg = Cfg(**dict(WARM, hist_pool=16))
    print(f"=== invariance checks: drift=abrupt seed={seed}, recommended design + hist_pool=16")
    m1 = OnlineGBT(cfg, P)
    p1, e1 = m1.fit_chunk(X, y, d_clock, w)
    m2 = OnlineGBT(cfg, P)
    cuts = [0, 1, 777, 1999, 2000, 2001, 5000, 12345, 12346, 20000, N]
    p2, e2 = np.empty(N), np.empty(N)
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        p2[a:b], e2[a:b] = m2.fit_chunk(X[a:b], y[a:b], d_clock[a:b], w[a:b])
    same = np.array_equal(p1, p2, equal_nan=True) and np.array_equal(e1, e2)
    print(f"chunk invariance, 1 chunk vs {len(cuts) - 1} uneven chunks: identical={same}")
    t = 15000
    y3 = y.copy()
    y3[t] += 1e3
    p3, _ = OnlineGBT(cfg, P).fit_chunk(X, y3, d_clock, w)
    later = not np.allclose(p3[t + 1 : t + 200], p1[t + 1 : t + 200])
    print(
        f"out-of-sample: pred[t] unchanged when y[t] is perturbed={p3[t] == p1[t]}; "
        f"later preds move={later}"
    )
    w4 = w.copy()
    w4[13000:13500] = 0.0
    y4 = y.copy()
    y4[13000:13500] = 1e6
    p4, _ = OnlineGBT(cfg, P).fit_chunk(X, y4, d_clock, w4)
    p5, e5 = OnlineGBT(cfg, P).fit_chunk(X, y, d_clock, w4)
    print(
        "zero-weight rows: wild y on them changes nothing="
        f"{np.array_equal(p4, p5, equal_nan=True)}; all finite={np.isfinite(p4[2000:]).all()}"
    )
    print(
        f"n_eff before/after the zero-weight block: {e5[12999]:.3f} -> {e5[13500]:.3f}, "
        f"ratio {e5[13500] / e5[12999]:.4f} (decay only; pure decay over that span = "
        f"{math.exp(-math.log(2) * d_clock[13000:13501].sum() / 3000.0):.4f})"
    )
    w6 = w.copy()
    w6[0] = 0.0
    p6, e6 = OnlineGBT(cfg, P).fit_chunk(X, y, d_clock, w6)
    print(
        "zero-weight first row: not a learned row, so warm-up ends one row later "
        f"(pred[2000] nan={np.isnan(p6[2000])}); finite after={np.isfinite(p6[2001:]).all()}; "
        f"n_eff[0]={e6[0]}, n_eff[1]={e6[1]}"
    )
    # Parallel additivity: a segment's histogram contribution is the sum of its blocks'.
    m = OnlineGBT(Cfg(**dict(WARM, grow_every=10**9)), P)
    m.fit_chunk(X[:4000], y[:4000], d_clock[:4000], w[:4000])  # warm start, 2000 pending rows
    seg = m._seg
    Xb = m._bin(np.stack([r[0] for r in seg]))
    ys = np.array([r[1] for r in seg])
    cs = np.array([r[2] * math.exp(m.L - r[3]) for r in seg])
    _, partials = m._predict_frozen(Xb)
    g = cs * (partials[0] - ys)
    one, four = copy.deepcopy(m.trees[0]), copy.deepcopy(m.trees[0])
    one.accumulate(Xb, g, cs, m.L)
    for k in range(4):
        sl = slice(k * len(g) // 4, (k + 1) * len(g) // 4)
        four.accumulate(Xb[sl], g[sl], cs[sl], m.L)
    worst = 0.0
    for na, nb in zip(one.nodes(), four.nodes(), strict=True):
        worst = max(worst, abs(na.G - nb.G), abs(na.H - nb.H))
        if na.hist_G is not None:
            worst = max(worst, float(np.abs(na.hist_G - nb.hist_G).max()))
            worst = max(worst, float(np.abs(na.hist_H - nb.hist_H).max()))
    print(
        "parallel additivity: one tree, a 2000-row segment accumulated as 1 block vs 4 blocks, "
        f"max |diff| = {worst:.1e} (fp reassociation only; sums are ~1e3)"
    )
    print()


EXPERIMENTS = {
    "baselines": exp_baselines,
    "knobs": exp_knobs,
    "pool": exp_pool,
    "negatives": exp_negatives,
    "invariance": exp_invariance,
}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(EXPERIMENTS)
    for name in names:
        EXPERIMENTS[name]()
