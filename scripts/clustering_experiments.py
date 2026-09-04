"""The experiments behind ``docs/CLUSTERING.md``, on the ``clustering_proto`` models.

A synthetic stream from a Gaussian mixture whose components drift, die and are
born, with injected outliers and an irregular clock; every model is fed row by
row (predict-then-learn) and scored per segment against the generating
labels: ARI, purity, the tracking error of the true centres, and the outlier
flags. Batch references: Lloyd's k-means on the full history (in-sample) and
refit on a rolling window (the honest out-of-sample use of a batch learner);
scikit-learn's MiniBatchKMeans / GaussianMixture when importable.

    uv run python scripts/clustering_experiments.py all
    uv run --with scikit-learn python scripts/clustering_experiments.py all

Experiments: ``guarantees`` (chunk invariance, determinism, zero-weight and
null rows, NaN-free state), ``baselines``, ``seeding``, ``decay``,
``outliers``, ``regime``, ``knobs``, ``cost``, ``hard``; ``all`` runs them in that
order. Each prints a table whose rows are quoted in the document. The
scikit-learn rows are skipped with a note when it cannot be imported.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clustering_proto import (  # noqa: E402
    GNG,
    NONE,
    ODAC,
    SOM,
    DPCfg,
    DPMeans,
    EWKMeans,
    GMMCfg,
    GNGCfg,
    KMeansCfg,
    MicroCfg,
    MicroClusters,
    ODACCfg,
    OnlineGMM,
    SOMCfg,
    ari,
    purity,
)

try:
    import sklearn.cluster
    import sklearn.metrics
    import sklearn.mixture
except Exception:  # noqa: BLE001 -- any import failure is "not available"
    sklearn = None

N, P, K = 20000, 4, 5
SIGMA = 1.0


# --------------------------------------------------------------------------- data
def mixture(
    seed: int,
    n: int = N,
    p: int = P,
    k: int = K,
    sep: float = 6.0,
    sigma: float = SIGMA,
    drift: float = 0.0,
    outlier_frac: float = 0.0,
    regime_at: int | None = None,
    unequal: bool = False,
) -> dict:
    """A stream from k Gaussian components with pairwise centre distance >= sep * sigma.

    ``drift``: every component's centre takes a random-walk step of ``drift *
    sigma`` per row (so it moves ~ ``drift * sigma * sqrt(n)`` overall).
    ``outlier_frac``: that fraction of rows is uniform over a box three times
    the mixture's extent, labelled -1. ``regime_at``: at that row component 0
    dies and a new component (label k) is born at a fresh location.
    ``unequal``: mixing weights 1:2:...:k instead of equal, and component
    spreads 0.5 .. 1.5 sigma. Irregular clock (exponential gaps, mean 1) and
    unit row weights. Returns X, labels, dt, w and the centres per row.
    """
    rng = np.random.default_rng(seed)
    while True:  # rejection: well separated centres
        mu = rng.uniform(-sep * 1.5, sep * 1.5, (k + 1, p)) * sigma
        D = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(2)) + np.eye(k + 1) * 1e9
        if D.min() >= sep * sigma:
            break
    spread = np.linspace(0.5, 1.5, k + 1) * sigma if unequal else np.full(k + 1, sigma)
    mix = np.arange(1, k + 1, dtype=float) if unequal else np.ones(k)
    mix /= mix.sum()
    lab = rng.choice(k, size=n, p=mix)
    if regime_at is not None:
        late = np.arange(n) >= regime_at
        lab[late & (lab == 0)] = k
    steps = rng.normal(0.0, drift * sigma, (n, k + 1, p)) if drift > 0 else np.zeros((n, k + 1, p))
    centres = mu[None] + np.cumsum(steps, axis=0)
    X = centres[np.arange(n), lab] + rng.normal(0.0, 1.0, (n, p)) * spread[lab, None]
    is_out = rng.random(n) < outlier_frac
    if is_out.any():
        lo, hi = X[~is_out].min(0), X[~is_out].max(0)
        span = hi - lo
        X[is_out] = rng.uniform(lo - span, hi + span, (int(is_out.sum()), p))
        lab[is_out] = -1
    dt = rng.exponential(1.0, n)
    return {
        "X": X,
        "lab": lab,
        "dt": dt,
        "w": np.ones(n),
        "centres": centres,
        "k": k,
        "sigma": sigma,
    }


def segments(n: int) -> list[tuple[int, int]]:
    return [(n // 8, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]


def seg_ari(truth: np.ndarray, pred: np.ndarray, seg: tuple[int, int]) -> float:
    a, b = seg
    t, q = truth[a:b], pred[a:b]
    m = (t >= 0) & (q != NONE)
    return ari(t[m], q[m]) if m.sum() > 10 else math.nan


def seg_purity(truth: np.ndarray, pred: np.ndarray, seg: tuple[int, int]) -> float:
    a, b = seg
    t, q = truth[a:b], pred[a:b]
    m = (t >= 0) & (q != NONE)
    return purity(t[m], q[m]) if m.sum() > 10 else math.nan


def tracking(model, data: dict, row: int) -> float:
    """Mean distance from each live true centre at `row` to the nearest model centre."""
    C = model.centres()
    if C is None or len(C) == 0 or data.get("centres") is None:
        return math.nan
    live = np.unique(data["lab"][max(0, row - 500) : row + 1])
    live = live[live >= 0]
    tc = data["centres"][row, live]
    return float(np.mean(np.sqrt(((tc[:, None] - C[None]) ** 2).sum(2)).min(1)))


def fmt(v: float, w: int = 6) -> str:
    return (
        f"{'nan':>{w}}" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{w}.3f}"
    )


# --------------------------------------------------------------------------- batch references
def lloyd(X: np.ndarray, k: int, seed: int = 0, iters: int = 50, restarts: int = 5) -> np.ndarray:
    """Weighted-equal Lloyd's k-means with k-means++ seeds, best of `restarts` by SSQ."""
    best, best_ssq = None, math.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        C = X[[rng.integers(len(X))]]
        d2 = ((X - C[0]) ** 2).sum(1)
        while len(C) < k:
            i = rng.choice(len(X), p=d2 / d2.sum()) if d2.sum() > 0 else rng.integers(len(X))
            C = np.vstack([C, X[i]])
            d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)
            newC = C.copy()
            for j in range(k):
                if (lab == j).any():
                    newC[j] = X[lab == j].mean(0)
            if np.allclose(newC, C):
                break
            C = newC
        ssq = ((X - C[lab]) ** 2).sum()
        if ssq < best_ssq:
            best, best_ssq = C, ssq
    return best


def batch_full(data: dict, k: int, standardize: bool = False) -> np.ndarray:
    X = data["X"]
    if standardize:  # the online models standardize in the metric; match them
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    C = lloyd(X, k)
    return np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)


def batch_rolling(data: dict, k: int, window: int = 2000, refit: int = 500) -> np.ndarray:
    """Refit on the last `window` rows every `refit` rows; label the next rows out-of-sample."""
    X = data["X"]
    n = len(X)
    out = np.full(n, NONE)
    C = None
    for start in range(0, n, refit):
        if start >= window:
            C = lloyd(X[start - window : start], k, restarts=2)
        if C is not None:
            blk = X[start : start + refit]
            out[start : start + refit] = np.argmin(((blk[:, None] - C[None]) ** 2).sum(2), axis=1)
    return out


def sk_minibatch(data: dict, k: int, batch: int = 256) -> np.ndarray | None:
    if sklearn is None:
        return None
    X = data["X"]
    m = sklearn.cluster.MiniBatchKMeans(n_clusters=k, batch_size=batch, random_state=0, n_init=1)
    out = np.full(len(X), NONE)
    for s in range(0, len(X), batch):
        blk = X[s : s + batch]
        if s > 0:
            out[s : s + batch] = m.predict(blk)
        m.partial_fit(blk)
    return out


# --------------------------------------------------------------------------- model zoo
def zoo(k: int, p: int, sigma: float, halflife: float = 3000.0, warm: int = 500) -> dict:
    return {
        "kmeans first": EWKMeans(KMeansCfg(k=k, halflife=halflife), p),
        "kmeans k++ warm": EWKMeans(
            KMeansCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="kmeanspp"), p
        ),
        "kmeans lloyd warm": EWKMeans(
            KMeansCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd"), p
        ),
        "kmeans reseed": EWKMeans(
            KMeansCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd", reseed=True), p
        ),
        "kmeans split-merge": EWKMeans(
            KMeansCfg(
                k=k,
                halflife=halflife,
                warm_rows=warm,
                seed_rule="lloyd",
                reseed=True,
                split_merge=0.5,
                sm_every=100,
            ),
            p,
        ),
        "kmeans huber c=2": EWKMeans(
            KMeansCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd", huber_c=2.0), p
        ),
        "fuzzy m=2": EWKMeans(
            KMeansCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd", fuzzifier=2.0), p
        ),
        "gmm diag": OnlineGMM(
            GMMCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd", cov="diag"), p
        ),
        "gmm full": OnlineGMM(
            GMMCfg(k=k, halflife=halflife, warm_rows=warm, seed_rule="lloyd", cov="full"), p
        ),
        "dpmeans r=4s": DPMeans(
            DPCfg(radius=4.0 * sigma, max_clusters=50, halflife=halflife, prune_weight=2.0), p
        ),
        "micro eps=1.5s": MicroClusters(
            MicroCfg(
                eps=1.5 * sigma, beta_mu=5.0, max_micro=200, halflife=halflife, macro_link=2.0
            ),
            p,
        ),
        "som 3x3": SOM(
            SOMCfg(
                rows=3, cols=3, halflife=halflife, sigma=0.5, warm_rows=warm, seed_rule="kmeanspp"
            ),
            p,
        ),
        "gng 30": GNG(GNGCfg(max_nodes=30, insert_every=100, halflife=halflife, a_max=30), p),
    }


def run(model, data: dict) -> dict:
    return model.fit_chunk(data["X"], data["dt"], data["w"])


# --------------------------------------------------------------------------- experiments
def _states(model) -> list[np.ndarray]:
    """The model's canonical state (derived caches excluded), for a bitwise comparison."""
    return model.canonical_state()


def _same_state(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(
        x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
        for x, y in zip(a, b, strict=True)
    )


def exp_guarantees() -> None:
    print(
        "=== guarantees (docs/PLAN.md §2, CLAUDE.md rules 2/3/8/9)\n"
        "    chunks   1 vs 37 vs 1000 vs one chunk: outputs bit-identical\n"
        "    rerun    the same stream twice on a fresh model: bit-identical\n"
        "    w=0      a zero-weight row at d_clock = 0 leaves the state\n"
        "             bit-identical (and still emits)\n"
        "    null     a null feature at d_clock = 0 leaves the state\n"
        "             bit-identical (and emits nothing)\n"
        "    drop     w=0 rows vs deleting them and folding the clock gap:\n"
        "             max |diff| over the float outputs, and the share of rows\n"
        "             given a different label. `n_eff` is excluded: it is read\n"
        "             before the row's own decay (rule 8), so folding a gap into\n"
        "             a row moves that row's `n_eff` -- by design, not a defect\n"
        "    lead     100 leading zero-weight and null rows: no NaN in the state\n"
        "    finite   state finite after outliers, drift, a regime change, w=0 rows"
    )
    import copy

    data = mixture(1, n=4000, drift=0.01, outlier_frac=0.02, regime_at=2500)
    X, dt, w = data["X"], data["dt"], data["w"]
    n = len(X)
    models = zoo(K, P, SIGMA, halflife=800.0, warm=200)
    models["odac"] = ODAC(ODACCfg(halflife=800.0, n_min=100), P)
    models["kmeans batch 50"] = EWKMeans(
        KMeansCfg(
            k=K, halflife=800.0, warm_rows=200, seed_rule="lloyd", update_every=50, reseed=True
        ),
        P,
    )
    models["kmeans std"] = EWKMeans(
        KMeansCfg(k=K, halflife=800.0, warm_rows=200, seed_rule="lloyd", standardize=True), P
    )
    head = f"{'model':22s} {'chunks':>7s} {'rerun':>6s} {'w=0':>5s} {'null':>5s}"
    print(f"{head} {'drop':>9s} {'relabel':>8s} {'lead':>5s} {'finite':>7s}")
    ok = lambda v: "PASS" if v else "FAIL"  # noqa: E731
    for name, m0 in models.items():
        outs = []
        for chunk in (n, 1, 37, 1000):
            m = copy.deepcopy(m0)
            parts = [
                m.fit_chunk(X[s : s + chunk], dt[s : s + chunk], w[s : s + chunk])
                for s in range(0, n, chunk)
            ]
            keys = list(parts[0])
            outs.append({key: np.concatenate([q[key] for q in parts]) for key in keys})
        same_chunks = all(
            np.array_equal(outs[0][key], o[key], equal_nan=True) for o in outs[1:] for key in keys
        )
        rerun = copy.deepcopy(m0).fit_chunk(X, dt, w)
        same_rerun = all(np.array_equal(outs[0][key], rerun[key], equal_nan=True) for key in keys)

        # w=0 and null rows at d_clock = 0 freeze the state exactly
        base = copy.deepcopy(m0)
        base.fit_chunk(X[:1500], dt[:1500], w[:1500])
        st = _states(base)
        zero = copy.deepcopy(base)
        oz = zero.fit_chunk(X[1500:1600], np.zeros(100), np.zeros(100))
        emitted = any(
            np.isfinite(v).any() for k, v in oz.items() if v.dtype.kind == "f" and k != "n_eff"
        )
        froze_zero = _same_state(st, _states(zero)) and (emitted or isinstance(base, ODAC))
        Xn = X[1500:1600].copy()
        Xn[:, 0] = np.nan
        nul = copy.deepcopy(base)
        on = nul.fit_chunk(Xn, np.zeros(100), np.ones(100))
        blank = all((v == NONE).all() for k, v in on.items() if k == "cluster")
        froze_null = _same_state(st, _states(nul)) and (blank or isinstance(base, ODAC))

        # w=0 rows vs deleting them and folding the clock gap forward: equal up to
        # the floating-point associativity of 0.5**(dt/halflife)
        rng = np.random.default_rng(0)
        Z = np.sort(rng.choice(np.arange(1, n), size=n // 20, replace=False))
        keep = np.ones(n, bool)
        keep[Z] = False
        wz = w.copy()
        wz[Z] = 0.0
        dt2 = dt.copy()
        carry = 0.0
        for i in range(n):  # a dropped row's gap goes to the next kept row
            if keep[i]:
                dt2[i] += carry
                carry = 0.0
            else:
                carry += dt[i]
        a = copy.deepcopy(m0).fit_chunk(X, dt, wz)
        b = copy.deepcopy(m0).fit_chunk(X[keep], dt2[keep], w[keep])
        diffs = [
            np.nanmax(np.abs(a[key][keep].astype(float) - b[key].astype(float)))
            for key in keys
            if key != "n_eff"
            and a[key].dtype.kind == "f"
            and np.isfinite(a[key][keep].astype(float)).any()
        ]
        drop = max((d for d in diffs if np.isfinite(d)), default=0.0)
        lab_a, lab_b = a.get("cluster"), b.get("cluster")
        relab = float(np.mean(lab_a[keep] != lab_b)) if lab_a is not None else 0.0

        # 100 leading zero-weight rows, then 100 leading nulls, then the stream
        lead_w = np.concatenate([np.zeros(100), np.ones(100), w])
        lead_X = np.vstack([X[:100], np.full((100, P), np.nan), X])
        lead_dt = np.concatenate([dt[:100], dt[:100], dt])
        ml = copy.deepcopy(m0)
        ol = ml.fit_chunk(lead_X, lead_dt, lead_w)
        lead = all(np.isfinite(v).all() for v in _states(ml)) and not np.isnan(ol["n_eff"]).any()

        mf = copy.deepcopy(m0)
        mf.fit_chunk(X, dt, wz)
        C = mf.centres() if hasattr(mf, "centres") else None
        finite = all(np.isfinite(v).all() for v in _states(mf)) and (
            C is None or np.isfinite(C).all()
        )
        row = f"{name:22s} {ok(same_chunks):>7s} {ok(same_rerun):>6s}"
        row += f" {ok(froze_zero):>5s} {ok(froze_null):>5s} {drop:9.2e} {relab:8.1%}"
        print(f"{row} {ok(lead):>5s} {ok(finite):>7s}")


def table_header(title: str, n: int) -> None:
    segs = " ".join(f"[{a // 1000}k,{b // 1000}k)".rjust(11) for a, b in segments(n))
    print(f"=== {title}")
    tail = f"{'purity':>6s} {'track':>6s} {'live':>4s} {'seen':>5s} {'ms/row':>6s}"
    print(f"{'model':22s} {segs}  {tail}")


def n_live(model, pred: np.ndarray) -> int:
    if model is None:
        return len(np.unique(pred[pred != NONE]))
    C = model.centres() if hasattr(model, "centres") else None
    return 0 if C is None else len(C)


def report(
    name: str, pred: np.ndarray, data: dict, model=None, elapsed: float = math.nan
) -> list[float]:
    n = len(pred)
    vals = [seg_ari(data["lab"], pred, s) for s in segments(n)]
    pur = seg_purity(data["lab"], pred, segments(n)[-1])
    trk = tracking(model, data, n - 1) if model is not None else math.nan
    cells = " ".join(fmt(v, 11) for v in vals)
    seen = len(np.unique(pred[(pred != NONE) & (np.arange(n) >= n // 2)]))
    ms = elapsed * 1e3 / n if math.isfinite(elapsed) else math.nan
    print(f"{name:22s} {cells}  {fmt(pur)} {fmt(trk)} {n_live(model, pred):4d} {seen:5d} {fmt(ms)}")
    return vals


def exp_baselines() -> None:
    for title, kw in (
        ("baselines: static mixture (k=5, p=4, sep 6 sigma), halflife 3000", {}),
        ("baselines: drifting mixture (random walk 0.02 sigma / row)", {"drift": 0.02}),
        (
            "baselines: unequal mixture (weights 1:2:3:4:5, spreads 0.5-1.5 sigma), drifting",
            {"drift": 0.02, "unequal": True},
        ),
    ):
        data = mixture(7, **kw)
        table_header(title, N)
        for name, m in zoo(K, P, SIGMA).items():
            t0 = time.time()
            out = run(m, data)
            report(name, out["cluster"], data, m, time.time() - t0)
        report("batch lloyd (in-sample)", batch_full(data, K), data)
        roll = batch_rolling(data, K)
        report("batch lloyd rolling", roll, data)
        blocks = [seg_ari(data["lab"], roll, (b, b + 500)) for b in range(2000, N, 500)]
        print(
            f"{'  ...within a 500-row block':22s} ARI {np.nanmean(blocks):.3f}"
            " (the same labelling only holds between refits)"
        )
        mb = sk_minibatch(data, K)
        if mb is None:
            print("sklearn MiniBatchKMeans     (scikit-learn not importable; skipped)")
        else:
            report("sklearn minibatch 256", mb, data)


def exp_seeding() -> None:
    print(
        "=== seeding: EWKMeans ARI on the last quarter, mean +- sd over 10 data\n"
        "    seeds; 'miss' = runs with purity < 0.9"
    )
    rules = [
        ("first k rows", dict(seed_rule="first")),
        ("farthest-first 500", dict(warm_rows=500, seed_rule="farthest")),
        ("k-means++ 500 s0", dict(warm_rows=500, seed_rule="kmeanspp", seed=0)),
        ("k-means++ 500 s1", dict(warm_rows=500, seed_rule="kmeanspp", seed=1)),
        ("lloyd 500", dict(warm_rows=500, seed_rule="lloyd")),
        ("lloyd 2000", dict(warm_rows=2000, seed_rule="lloyd")),
        ("lloyd 500 + reseed", dict(warm_rows=500, seed_rule="lloyd", reseed=True)),
        ("first + reseed", dict(seed_rule="first", reseed=True)),
    ]
    for title, kw in (
        ("clean", {}),
        ("5% outliers", {"outlier_frac": 0.05}),
        ("sep 3 sigma (overlapping)", {"sep": 3.0}),
    ):
        print(f"--- {title}")
        for name, cfg in rules:
            aris, miss = [], 0
            for seed in range(10):
                data = mixture(100 + seed, n=8000, **kw)
                m = EWKMeans(KMeansCfg(k=K, halflife=3000.0, **cfg), P)
                out = run(m, data)
                aris.append(seg_ari(data["lab"], out["cluster"], (6000, 8000)))
                miss += seg_purity(data["lab"], out["cluster"], (6000, 8000)) < 0.9
            print(
                f"{name:22s} ARI {np.mean(aris):.3f} +- {np.std(aris):.3f}"
                f"   miss {miss}/10   reseeds {m.n_reseeds}"
            )


def exp_decay() -> None:
    print("=== decay: the halflife sweep, mean over 5 data seeds")
    print("    ARI  = last quarter, clean rows;  track = mean distance from each live true centre")
    print("    to the nearest model centre at the last row;  miss = seeds with purity < 0.9")
    for title, kw in (
        ("drifting (random walk 0.02 sigma / row)", {"drift": 0.02}),
        ("static", {}),
        ("regime change at n/2 + slow drift", {"regime_at": N // 2, "drift": 0.01}),
    ):
        print(f"--- {title}")
        cols = f"{'kmeans':>25s}   {'kmeans split-merge':>25s}   {'gmm diag':>25s}"
        print(f"{'halflife':>9s}  {cols}")
        for hl in (math.inf, 20000.0, 5000.0, 3000.0, 1000.0, 300.0, 100.0):
            km = KMeansCfg(k=K, halflife=hl, warm_rows=500, seed_rule="lloyd", reseed=True)
            sm = KMeansCfg(
                k=K,
                halflife=hl,
                warm_rows=500,
                seed_rule="lloyd",
                reseed=True,
                split_merge=0.5,
                sm_every=100,
            )
            gm = GMMCfg(k=K, halflife=hl, warm_rows=500, seed_rule="lloyd", cov="diag")
            cells = []
            for cls, cfg in ((EWKMeans, km), (EWKMeans, sm), (OnlineGMM, gm)):
                aris, trks, miss = [], [], 0
                for seed in range(5):
                    data = mixture(200 + seed, **kw)
                    m = cls(cfg, P)
                    out = run(m, data)
                    aris.append(seg_ari(data["lab"], out["cluster"], segments(N)[-1]))
                    trks.append(tracking(m, data, N - 1))
                    miss += seg_purity(data["lab"], out["cluster"], segments(N)[-1]) < 0.9
                cells.append(f"ARI {np.mean(aris):.3f} track {np.mean(trks):5.2f} miss {miss}")
            print(f"{hl:>9g}  {cells[0]:>25s}   {cells[1]:>25s}   {cells[2]:>25s}")


def exp_outliers() -> None:
    print(
        "=== outliers: 5% uniform outliers, drifting; ARI on clean rows;\n"
        "    flag = the model's own outlier signal"
    )
    data = mixture(13, drift=0.02, outlier_frac=0.05)
    is_out = data["lab"] < 0
    table_header("5% outliers", N)
    for name, m in zoo(K, P, SIGMA).items():
        out = run(m, data)
        report(name, out["cluster"], data, m)
        flag = None
        if "outlier" in out:
            flag = out["outlier"].astype(bool)
        elif "new" in out:
            flag = out["new"].astype(bool)
        elif isinstance(m, EWKMeans) and m.C is not None:
            # a row farther than 3 radii from its centre
            rad = np.sqrt(np.maximum(m.R, 1e-12))
            cl = out["cluster"]
            ok = cl >= 0
            flag = np.zeros(N, bool)
            flag[ok] = out["dist"][ok] > 3.0 * rad[cl[ok]]
        elif isinstance(m, OnlineGMM):
            flag = out["dist"] > 4.0  # Mahalanobis
        if flag is not None:
            seg = slice(N // 2, N)
            tp = (flag[seg] & is_out[seg]).sum()
            prec = tp / max(flag[seg].sum(), 1)
            rec = tp / max(is_out[seg].sum(), 1)
            print(f"{'':22s} outlier flag: precision {prec:.2f} recall {rec:.2f} (second half)")


def exp_regime() -> None:
    print(
        "=== regime change at n/2: component 0 dies, a new one is born elsewhere;\n"
        "    ARI per segment, and the rows to ARI > 0.9 after the change"
    )
    data = mixture(17, regime_at=N // 2, drift=0.01)
    table_header("regime change", N)
    for name, m in zoo(K, P, SIGMA).items():
        out = run(m, data)
        report(name, out["cluster"], data, m)
        # recovery: first 500-row block after n/2 with ARI > 0.9
        rec = None
        for s in range(N // 2, N, 500):
            if seg_ari(data["lab"], out["cluster"], (s, s + 500)) > 0.9:
                rec = s - N // 2
                break
        extra = f" reseeds {m.n_reseeds}" if isinstance(m, EWKMeans) else ""
        print(f"{'':22s} recovered after {rec if rec is not None else '> n/2'} rows{extra}")


def exp_knobs() -> None:
    print("=== knobs")
    data = mixture(19, drift=0.02)
    print("--- split-merge: threshold x cadence, over 20 drifting data seeds")
    print("    the plain model collapses two centres onto one component on 5 of the 20;")
    print(
        "    'miss' counts those (purity < 0.9 on the last quarter), 'moves' is the mean per stream"
    )
    print(
        f"{'threshold':>9s} {'cadence':>8s}   {'ARI':>5s} {'track':>6s} {'miss':>4s} {'moves':>7s}"
    )
    for sm in (0.0, 0.3, 0.5, 0.8, 1.0):
        for sme in (1, 100, 500, 2000):
            if sm == 0.0 and sme != 1:
                continue
            aris, trks, moves, miss = [], [], [], 0
            for seed in range(20):
                d = mixture(400 + seed, drift=0.02)
                m = EWKMeans(
                    KMeansCfg(
                        k=K,
                        halflife=3000.0,
                        warm_rows=500,
                        seed_rule="lloyd",
                        reseed=True,
                        split_merge=sm,
                        sm_every=sme,
                    ),
                    P,
                )
                out = run(m, d)
                aris.append(seg_ari(d["lab"], out["cluster"], segments(N)[-1]))
                trks.append(tracking(m, d, N - 1))
                moves.append(m.n_merges)
                miss += seg_purity(d["lab"], out["cluster"], segments(N)[-1]) < 0.9
            lbl = "off" if sm == 0.0 else f"{sm:g}"
            print(
                f"{lbl:>9s} {sme:>8d}   {np.mean(aris):.3f} {np.mean(trks):6.2f}"
                f" {miss:4d} {np.mean(moves):7.1f}"
            )
    print("--- update_every (data-parallel form): kmeans lloyd 500 + reseed, ARI per segment")
    table_header("update_every", N)
    for ue in (1, 10, 100, 1000):
        m = EWKMeans(
            KMeansCfg(
                k=K, halflife=3000.0, warm_rows=500, seed_rule="lloyd", reseed=True, update_every=ue
            ),
            P,
        )
        out = run(m, data)
        report(f"update_every {ue}", out["cluster"], data, m)
    print("--- DP-means radius (in sigma): clusters found (true k = 5), ARI last quarter")
    for r in (1.5, 2.0, 3.0, 4.0, 6.0):
        m = DPMeans(DPCfg(radius=r * SIGMA, max_clusters=50, halflife=3000.0, prune_weight=2.0), P)
        out = run(m, data)
        a = seg_ari(data["lab"], out["cluster"], segments(N)[-1])
        pur = seg_purity(data["lab"], out["cluster"], segments(N)[-1])
        print(
            f"radius {r:<4g} clusters {out['n_clusters'][-1]:3d}"
            f" (evicted {m.n_evicted:4d})  ARI {fmt(a)}  purity {fmt(pur)}"
        )
    print(
        "--- micro-clusters eps (in sigma) x beta_mu: potential micro-clusters,\n"
        "    macro clusters, ARI last quarter"
    )
    for eps in (1.0, 1.5, 2.0):
        for bm in (3.0, 10.0):
            m = MicroClusters(
                MicroCfg(
                    eps=eps * SIGMA, beta_mu=bm, max_micro=200, halflife=3000.0, macro_link=2.0
                ),
                P,
            )
            out = run(m, data)
            nmac = len(set(m.macro.values()))
            a = seg_ari(data["lab"], out["cluster"], segments(N)[-1])
            pur = seg_purity(data["lab"], out["cluster"], segments(N)[-1])
            print(
                f"eps {eps:<4g} beta_mu {bm:<4g} potential {out['n_potential'][-1]:4d}"
                f" macro {nmac:3d}  ARI {fmt(a)}  purity {fmt(pur)}"
                f"  evicted {m.n_evicted} pruned {m.n_pruned}"
            )
    print("--- standardize: feature 0 scaled by 100 (kmeans lloyd 500), ARI last quarter")
    data2 = mixture(19, drift=0.02)
    data2["X"] = data2["X"].copy()
    data2["X"][:, 0] *= 100.0
    for std in (False, True):
        m = EWKMeans(
            KMeansCfg(k=K, halflife=3000.0, warm_rows=500, seed_rule="lloyd", standardize=std), P
        )
        out = run(m, data2)
        a = seg_ari(data2["lab"], out["cluster"], segments(N)[-1])
        print(f"standardize {str(std):5s} ARI {fmt(a)}")
    print(
        "--- ODAC on 8 variables in 3 correlated blocks (+ a block switch at n/2):\n"
        "    leaves and labels"
    )
    rng = np.random.default_rng(3)
    n = 6000
    f = rng.normal(size=(n, 3))
    blocks = [0, 0, 0, 1, 1, 2, 2, 2]
    Xv = np.stack([f[:, b] + 0.3 * rng.normal(size=n) for b in blocks], 1)
    Xv[n // 2 :, 7] = f[n // 2 :, 0] + 0.3 * rng.normal(
        size=n - n // 2
    )  # variable 7 switches to block 0
    m = ODAC(ODACCfg(halflife=1500.0, n_min=200, confidence=0.9, tau=0.1), 8)
    out = m.fit_chunk(Xv, np.ones(n))
    for row in (n // 4, n // 2, 3 * n // 4, n - 1):
        print(f"row {row:5d} leaves {out['n_leaves'][row]}  labels {out['labels'][row]}")
    print(f"splits {m.n_splits} merges {m.n_merges}")


def exp_cost() -> None:
    print(
        "=== cost: state (f64s) and prototype time per row\n"
        "    (numpy, one row at a time; relative only)"
    )
    data = mixture(23, n=4000)
    print(f"{'model':22s} {'state f64':>9s} {'us/row':>7s}  per-row work")
    work = {
        "kmeans": "O(k p) distances",
        "fuzzy": "O(k p)",
        "gmm diag": "O(k p)",
        "gmm full": "O(k p^2) (solve per component)",
        "dpmeans": "O(M p), M <= max_clusters",
        "micro": "O(M p), M <= max_micro; macro step O(M^2) per checkpoint",
        "som": "O(K p + K) (K neurons)",
        "gng": "O(M p + E) (E edges)",
        "odac": "O(p^2) moments; tests O(p^2) per checkpoint",
    }
    models = zoo(K, P, SIGMA, halflife=800.0, warm=200)
    models["odac"] = ODAC(ODACCfg(halflife=800.0, n_min=100), P)
    for name, m in models.items():
        t0 = time.time()
        run(m, data)
        el = (time.time() - t0) * 1e6 / len(data["X"])
        key = next((k for k in work if name.startswith(k)), "")
        print(f"{name:22s} {m.state_doubles():9d} {el:7.0f}  {work.get(key, '')}")


# --------------------------------------------------------------------------- hard geometries
def hard_dataset(name: str, seed: int, n: int = 6000) -> dict:
    """A stream whose clusters are *not* isotropic Gaussian blobs.

    ``mixture`` generates exactly the shape k-means is optimal for, so it can
    only say whether these models are broken, never whether they cluster well.
    These seven can: two are non-convex, two defeat a spherical distance, one
    has densities an order of magnitude apart, and two ask what happens as ``p``
    grows. Rows are drawn i.i.d. and shuffled, the clock is regular and nothing
    drifts, so the only question asked is clustering quality. Same dict shape as
    ``mixture`` with ``centres = None`` -- a tracking error is meaningless for a
    shape.
    """
    rng = np.random.default_rng(seed)
    if name == "moons":
        m = n // 2
        t = rng.uniform(0, math.pi, m)
        a = np.stack([np.cos(t), np.sin(t)], 1)
        b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
        X = np.vstack([a, b]) + rng.normal(0, 0.07, (2 * m, 2))
        lab = np.repeat([0, 1], m)
    elif name == "rings":
        per = n // 3
        Xs, ls = [], []
        for j, r in enumerate((1.0, 2.2, 3.4)):
            t = rng.uniform(0, 2 * math.pi, per)
            Xs.append(np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, 0.10, (per, 2)))
            ls.append(np.full(per, j))
        X, lab = np.vstack(Xs), np.concatenate(ls)
    elif name == "aniso":
        per = n // 3
        mu = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
        T = np.array([[0.6, -0.63], [-0.4, 0.85]])  # a shear: clusters become long and tilted
        X = np.vstack([rng.normal(0, 1.0, (per, 2)) @ T + mu[j] for j in range(3)])
        lab = np.repeat(np.arange(3), per)
    elif name == "elongated":
        per = n // 3
        Xs = []
        for j in range(3):
            u = rng.uniform(-6, 6, per)
            Xs.append(np.stack([u, np.full(per, 2.0 * j)], 1) + rng.normal(0, 0.25, (per, 2)))
        X = np.vstack(Xs)
        lab = np.repeat(np.arange(3), per)
    elif name == "varied":
        per = n // 3
        mu = np.array([[0.0, 0.0], [7.0, 0.0], [3.5, 6.0]])
        sd = np.array([0.4, 2.5, 0.4])
        X = np.vstack([rng.normal(0, 1.0, (per, 2)) * sd[j] + mu[j] for j in range(3)])
        lab = np.repeat(np.arange(3), per)
    elif name.startswith("highdim"):
        p, k = int(name.removeprefix("highdim")), 5
        per = n // k
        mu = rng.normal(0, 1.0, (k, p)) * 4.0
        X = np.vstack([rng.normal(0, 1.0, (per, p)) + mu[j] for j in range(k)])
        lab = np.repeat(np.arange(k), per)
    else:
        raise ValueError(name)
    order = rng.permutation(len(X))
    X, lab = X[order], lab[order]
    return {
        "X": X,
        "lab": lab,
        "dt": np.ones(len(X)),
        "w": np.ones(len(X)),
        "centres": None,
        "k": int(lab.max()) + 1,
        "sigma": 1.0,
    }


HARD = ("moons", "rings", "aniso", "elongated", "varied", "highdim20", "highdim50")


def sk_batch(data: dict, kind: str) -> np.ndarray | None:
    """Batch, in-sample, best of a sweep -- the ceiling a shape-aware method reaches."""
    if sklearn is None:
        return None
    X, lab, k = data["X"], data["lab"], data["k"]
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    if kind == "dbscan":
        best, best_s = None, -2.0
        rp = math.sqrt(X.shape[1])
        for c in (0.03, 0.07, 0.14, 0.2, 0.35, 0.6, 1.0):
            for mp in (5, 15):
                y = sklearn.cluster.DBSCAN(eps=c * rp, min_samples=mp).fit_predict(Z)
                sc = sklearn.metrics.adjusted_rand_score(lab, y)
                if sc > best_s:
                    best, best_s = y, sc
        return best
    if kind == "single":
        return sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(
            Z
        )
    return sklearn.mixture.GaussianMixture(
        n_components=k, covariance_type="full", random_state=0
    ).fit_predict(Z)


def hard_models(k: int, p: int) -> dict:
    """Every prototype, standardizing in the metric, with `k` given as an oracle.

    Every distance threshold is quoted as ``c * sqrt(p)``: two standardized points
    are ``sqrt(2p)`` apart on average, so a threshold fixed in absolute units means
    something different at every ``p`` -- and a 2-d default emits *no clusters at
    all* at ``p = 20``, rather than bad ones.
    """
    rp = math.sqrt(p)
    b: dict = {
        "kmeans": lambda: EWKMeans(
            KMeansCfg(k=k, warm_rows=500, seed_rule="lloyd", standardize=True), p
        ),
        "kmeans split-merge": lambda: EWKMeans(
            KMeansCfg(
                k=k,
                warm_rows=500,
                seed_rule="lloyd",
                standardize=True,
                reseed=True,
                split_merge=0.5,
                sm_every=100,
            ),
            p,
        ),
        "fuzzy m=2": lambda: EWKMeans(
            KMeansCfg(k=k, warm_rows=500, seed_rule="lloyd", standardize=True, fuzzifier=2.0), p
        ),
        "gmm diag": lambda: OnlineGMM(
            GMMCfg(k=k, warm_rows=500, seed_rule="lloyd", cov="diag", standardize=True), p
        ),
        "gmm full": lambda: OnlineGMM(
            GMMCfg(k=k, warm_rows=500, seed_rule="lloyd", cov="full", standardize=True), p
        ),
        "som 3x3": lambda: SOM(
            SOMCfg(
                rows=3, cols=3, sigma=0.5, warm_rows=500, seed_rule="kmeanspp", standardize=True
            ),
            p,
        ),
        "gng 30": lambda: GNG(GNGCfg(max_nodes=30, insert_every=100, standardize=True), p),
    }
    for c in (0.2, 0.4, 0.7, 1.4):
        b[f"dpmeans r={c}"] = lambda c=c: DPMeans(
            DPCfg(radius=c * rp, max_clusters=100, prune_weight=5.0, standardize=True), p
        )
    # `eps` sets the micro-cluster spacing; `macro_link` decides whether a chain of
    # them follows a shape or bridges the gap between two shapes
    for c, ml in ((0.1, 2.0), (0.2, 2.0), (0.4, 2.0), (0.1, 3.0), (0.2, 3.0)):
        b[f"micro eps={c} link={ml}"] = lambda c=c, ml=ml: MicroClusters(
            MicroCfg(
                eps=c * rp,
                beta_mu=5.0,
                max_micro=300,
                macro_link=ml,
                prune_every=200,
                standardize=True,
            ),
            p,
        )
    return b


def _score(
    data: list[dict], ys: list[np.ndarray], half: tuple[int, int]
) -> tuple[float, float, float, float]:
    pairs = list(zip(data, ys, strict=True))
    a = [seg_ari(d["lab"], y, half) for d, y in pairs]
    q = [seg_purity(d["lab"], y, half) for d, y in pairs]
    seen = [len(np.unique(y[half[0] :][y[half[0] :] != NONE])) for y in ys]
    if all(math.isnan(v) for v in a):
        return math.nan, math.nan, math.nan, float(np.mean(seen))
    return (
        float(np.nanmean(a)),
        float(np.nanstd(a)),
        float(np.nanmean(q)),
        float(np.mean(seen)),
    )


def exp_hard() -> None:
    print(
        "=== hard geometries: clusters that are not isotropic blobs\n"
        "    ARI over the second half (the first half is warm-up), mean +- sd\n"
        "    over 3 seeds. Every model standardizes in the metric and is given\n"
        "    the true k as an oracle where it takes one. The batch rows are\n"
        "    in-sample and best-of-a-sweep: a ceiling, not a competitor."
    )
    seeds = (5, 6, 7)
    for name in HARD:
        probe = hard_dataset(name, seeds[0])
        n, k, p = len(probe["X"]), probe["k"], probe["X"].shape[1]
        half = (n // 2, n)
        print(f"--- {name}: n={n} p={p} true k={k}")
        print(f"{'model':22s} {'ARI':>6s} {'+-':>5s} {'purity':>7s} {'labels':>7s}")
        data = [hard_dataset(name, s) for s in seeds]
        for mname, build in hard_models(k, p).items():
            ys = [run(build(), d)["cluster"] for d in data]
            a, sd, q, seen = _score(data, ys, half)
            print(f"{mname:22s} {fmt(a)} {fmt(sd, 5)} {fmt(q, 7)} {seen:7.0f}")
        for bname, kind in (
            ("batch lloyd (k given)", None),
            ("batch dbscan (swept)", "dbscan"),
            ("batch single linkage", "single"),
            ("batch gmm full", "gmm-full"),
        ):
            ys = [
                batch_full(d, k, standardize=True) if kind is None else sk_batch(d, kind)
                for d in data
            ]
            if any(y is None for y in ys):
                print(f"{bname:22s}  (scikit-learn not importable; skipped)")
                continue
            a, sd, q, seen = _score(data, ys, half)
            print(f"{bname:22s} {fmt(a)} {fmt(sd, 5)} {fmt(q, 7)} {seen:7.0f}")

    print(
        "--- why macro_link matters: nearest-neighbour spacing between potential\n"
        "    micro-clusters on `moons`, in units of eps. A link threshold below\n"
        "    that spacing severs the chain and a shape fragments."
    )
    d = hard_dataset("moons", 5)
    for c in (0.1, 0.2):
        eps = c * math.sqrt(2)
        m = MicroClusters(
            MicroCfg(
                eps=eps,
                beta_mu=5.0,
                max_micro=300,
                macro_link=2.0,
                prune_every=200,
                standardize=True,
            ),
            2,
        )
        run(m, d)
        cs = m.centres()
        dist = np.sqrt(np.stack([((cs - c0) ** 2 * m.mw).sum(1) for c0 in cs]))
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(1)
        links = {}
        for t in (2.0, 2.5, 3.0):
            links[t] = int(np.count_nonzero(dist < t * eps) // 2)
        print(
            f"eps={c}*sqrt(p): {len(cs):3d} potential MCs, spacing/eps"
            f" median {np.median(nn) / eps:.2f} p90 {np.percentile(nn, 90) / eps:.2f}"
            f"; pairs linked at 2.0/2.5/3.0 = {links[2.0]}/{links[2.5]}/{links[3.0]}"
        )


EXPERIMENTS = {
    "guarantees": exp_guarantees,
    "baselines": exp_baselines,
    "seeding": exp_seeding,
    "decay": exp_decay,
    "outliers": exp_outliers,
    "regime": exp_regime,
    "knobs": exp_knobs,
    "cost": exp_cost,
    "hard": exp_hard,
}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if which == ["all"]:
        which = list(EXPERIMENTS)
    for name in which:
        t0 = time.time()
        EXPERIMENTS[name]()
        print(f"({name}: {time.time() - t0:.0f}s)\n")
