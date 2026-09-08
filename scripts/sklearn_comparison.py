"""`sklearn.linear_model.SGDRegressor` against `po.spec.sgd` and `po.spec.ewridge`.

Usage: uv run python scripts/sklearn_comparison.py [accuracy|short|wide|grid|all]

scikit-learn is **not** a dependency of this project; install it into the
environment first (`uv pip install scikit-learn`). Everything here is
generated, so the script needs no data.

Every contender is held to the protocol this library guarantees: a row is
scored from the state as it stands, and only then learned from. sklearn is
given the setup its own documentation prescribes for out-of-core work --
features standardised by a `StandardScaler` that is itself fitted online, and
`partial_fit` -- and is run two ways, row by row (our semantics exactly) and
in mini-batches (what people write, and what costs its predictions their
freshness inside a batch). Each contender is then swept over a small grid of
its own settings and reported at its best, so the comparison is between
designs rather than between defaults. The grids are printed with the results,
and so is each stream's noise ceiling -- the R^2 of the generating signal
itself -- because "0.99" means nothing until you know what the best possible
number was. The first version of `accuracy` swept `ewridge` over halflives no
shorter than 500 rows and reported it 0.003 behind `sgd` on the drifting
stream; at `halflife=100` the two tie. A sweep whose best point is at its
edge is not a result, it is a grid.

docs/PERFORMANCE.md section 19 is this script's output, read.
"""

from __future__ import annotations

import itertools
import pickle
import sys
import time

import numpy as np
import polars as pl

import polars_online as po

ROWS = 100_000
K = 20
#: The batch mini-batch sklearn is given. Its predictions are then up to
#: `BATCH - 1` rows stale, which is the trade this script is measuring.
BATCH = 1_000
ALPHAS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]


def _sklearn():
    try:
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover - the message is the point
        sys.exit(
            "scikit-learn is not installed, and is not a dependency of this "
            "project. `uv pip install scikit-learn` and run this again."
        )
    return SGDRegressor, StandardScaler


def stream(n: int, k: int, drift: float, seed: int = 0):
    """A linear stream; with `drift > 0` the coefficients take a random walk."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    b = rng.standard_normal(k) / np.sqrt(k)
    if drift:
        beta = b + np.cumsum(drift * rng.standard_normal((n, k)) / np.sqrt(k), axis=0)
    else:
        beta = np.tile(b, (n, 1))
    signal = (x * beta).sum(axis=1)
    y = signal + 0.1 * rng.standard_normal(n)
    return x, y, signal


def frame(x, y):
    return pl.DataFrame({**{f"x{j}": x[:, j] for j in range(x.shape[1])}, "y": y})


def run_sklearn(x, y, batch, **kw):
    """Predict-then-fit in batches of `batch` rows; `batch=1` is row by row."""
    sgd_regressor, standard_scaler = _sklearn()
    n = len(x)
    model, scaler = sgd_regressor(random_state=0, **kw), standard_scaler()
    pred = np.full(n, np.nan)
    t0 = time.perf_counter()
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        if lo:
            pred[lo:hi] = model.predict(scaler.transform(x[lo:hi]))
        scaler.partial_fit(x[lo:hi])
        model.partial_fit(scaler.transform(x[lo:hi]), y[lo:hi])
    dt = time.perf_counter() - t0
    return pred, dt, model, scaler


def run_bank(spec, df):
    """A grid suffixes the output fields, so the first `pred_y*` is the one
    to read: with one setting it is the fit, with a grid it is the grid's
    first point, and either way the timing is the whole bank's."""
    bank = po.ModelBank([spec])
    t0 = time.perf_counter()
    out = bank.fit_predict(df).unnest("m")
    dt = time.perf_counter() - t0
    field = next(c for c in out.columns if c.startswith("pred_y"))
    return out[field].to_numpy(), dt, bank


def r2(pred, y, ok):
    err = y[ok] - pred[ok]
    return 1.0 - float(err @ err) / float(((y[ok] - y[ok].mean()) ** 2).sum())


def accuracy():
    """Out-of-sample R^2 and rows/sec, each contender at its best setting."""
    print(
        "Sweeps: SGDRegressor over learning_rate x eta0 x average; sgd over "
        "learning_rate x halflife; ewridge over halflife.\n"
    )
    for drift, label in ((0.0, "stationary"), (0.004, "drifting coefficients")):
        x, y, signal = stream(ROWS, K, drift)
        df, feats = frame(x, y), [f"x{j}" for j in range(K)]
        common = dict(targets=["y"], features=feats, min_periods=50.0)
        ok = np.zeros(ROWS, dtype=bool)
        ok[BATCH:] = True  # every contender has an opinion from here on

        runs: dict[str, list] = {}
        for lr, eta, avg in [
            ("invscaling", 0.01, False),
            ("constant", 0.001, False),
            ("constant", 0.003, False),
            ("constant", 0.01, False),
            ("constant", 0.03, False),
            ("adaptive", 0.01, False),
            ("constant", 0.01, True),
            ("constant", 0.03, True),
        ]:
            pred, dt, _, _ = run_sklearn(x, y, BATCH, learning_rate=lr, eta0=eta, average=avg)
            runs.setdefault(f"SGDRegressor, batches of {BATCH}", []).append(
                (r2(pred, y, ok), f"{lr}, eta0={eta}, average={avg}", dt)
            )
        for lr, eta in [("invscaling", 0.01), ("constant", 0.003)]:
            pred, dt, _, _ = run_sklearn(x, y, 1, learning_rate=lr, eta0=eta)
            runs.setdefault("SGDRegressor, row by row", []).append(
                (r2(pred, y, ok), f"{lr}, eta0={eta}", dt)
            )
        for rate, hl in itertools.product(
            [0.001, 0.003, 0.01, 0.03], [5e2, 2e3, 2e4, float("inf")]
        ):
            spec = po.spec.sgd("m", halflife=hl, learning_rate=rate, scale_features=True, **common)
            pred, dt, _ = run_bank(spec, df)
            runs.setdefault("po.spec.sgd", []).append(
                (r2(pred, y, ok), f"learning_rate={rate}, halflife={hl:g}", dt)
            )
        for hl in [50.0, 100.0, 200.0, 500.0, 2e3, 2e4, float("inf")]:
            spec = po.spec.ewridge(
                "m", halflife=hl, ridge=1e-6, max_rows_between_solves=1, **common
            )
            pred, dt, _ = run_bank(spec, df)
            runs.setdefault("po.spec.ewridge, refit every row", []).append(
                (r2(pred, y, ok), f"halflife={hl:g}", dt)
            )

        print(f"--- {label}: {ROWS:,} rows, k={K}, noise ceiling R2 = {r2(signal, y, ok):.4f}")
        print(f"{'contender':<36} {'best R2':>8}  {'at':<40} {'rows/sec':>10}")
        for name, sweep in runs.items():
            best, setting, dt = max(sweep)
            print(f"{name:<36} {best:8.4f}  {setting:<40} {ROWS / dt:10,.0f}")
        print("the sweeps, best to worst:")
        for name, sweep in runs.items():
            cells = ", ".join(f"{s}: {r:.4f}" for r, s, _ in sorted(sweep, reverse=True))
            print(f"  {name}: {cells}")
        print()


def short():
    """Where the Gram wins: many groups, each with a short history.

    500 groups of 200 rows, each group its own coefficients, `k = 20`, scored
    by position in the group. sklearn is one estimator and one scaler per
    group in a dict, row by row (there is no other way to give it a group);
    the bank is `group="g"`."""
    n_groups, rows_per = 500, 200
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_groups * rows_per, K))
    beta = np.repeat(rng.standard_normal((n_groups, K)) / np.sqrt(K), rows_per, axis=0)
    signal = (x * beta).sum(axis=1)
    y = signal + 0.1 * rng.standard_normal(len(x))
    pos = np.tile(np.arange(rows_per), n_groups)
    df = frame(x, y).with_columns(g=pl.Series(np.repeat(np.arange(n_groups), rows_per)))
    feats = [f"x{j}" for j in range(K)]
    windows = {"rows 25-50": (25, 50), "rows 50-100": (50, 100), "rows 100-200": (100, 200)}
    sgd_regressor, standard_scaler = _sklearn()

    def sklearn_groups(**kw):
        pred = np.full(len(x), np.nan)
        t0 = time.perf_counter()
        for gi in range(n_groups):
            lo = gi * rows_per
            model, scaler = sgd_regressor(random_state=0, **kw), standard_scaler()
            for i in range(lo, lo + rows_per):
                xi = x[i : i + 1]
                if i > lo:
                    pred[i] = model.predict(scaler.transform(xi))[0]
                scaler.partial_fit(xi)
                model.partial_fit(scaler.transform(xi), y[i : i + 1])
        return pred, time.perf_counter() - t0

    rows = []
    for lr, eta in [("invscaling", 0.01), ("constant", 0.01), ("constant", 0.03)]:
        pred, dt = sklearn_groups(learning_rate=lr, eta0=eta)
        rows.append((f"SGDRegressor per group, {lr}, eta0={eta}", pred, dt))
    common = dict(targets=["y"], features=feats, group="g", halflife=float("inf"), min_periods=25.0)
    for rate in (0.01, 0.03):
        spec = po.spec.sgd("m", learning_rate=rate, scale_features=True, **common)
        pred, dt, _ = run_bank(spec, df)
        rows.append((f"po.spec.sgd, learning_rate={rate}", pred, dt))
    spec = po.spec.ewridge("m", ridge=1e-6, max_rows_between_solves=1, **common)
    pred, dt, _ = run_bank(spec, df)
    rows.append(("po.spec.ewridge, refit every row", pred, dt))

    print(
        f"--- {n_groups} groups x {rows_per} rows, k={K}: out-of-sample R2 by position in the group"
    )
    print(f"{'contender':<46} " + " ".join(f"{w:>13}" for w in windows) + f" {'rows/sec':>10}")
    cells = [r2(signal, y, (pos >= lo) & (pos < hi)) for lo, hi in windows.values()]
    print(f"{'noise ceiling':<46} " + " ".join(f"{c:13.4f}" for c in cells))
    for name, pred, dt in rows:
        cells = [r2(pred, y, (pos >= lo) & (pos < hi)) for lo, hi in windows.values()]
        print(f"{name:<46} " + " ".join(f"{c:13.4f}" for c in cells) + f" {len(x) / dt:10,.0f}")
    print()


def wide():
    """Where sklearn wins: the Gram is quadratic in the feature count."""
    print("--- a wide row: rows/sec, and the state you have to keep")
    print(f"{'contender':<38} {'k':>6} {'rows':>7} {'rows/sec':>10} {'state':>10}")
    for k, n in ((1_000, 20_000), (10_000, 2_000)):
        x, y, _ = stream(n, k, 0.0)
        df, feats = frame(x, y), [f"x{j}" for j in range(k)]
        common = dict(targets=["y"], features=feats, halflife=float("inf"), min_periods=50.0)

        _, dt, model, scaler = run_sklearn(x, y, BATCH)
        size = len(pickle.dumps(model)) + len(pickle.dumps(scaler))
        rows = [(f"SGDRegressor, batches of {BATCH}", n / dt, size)]

        for name, spec in (
            ("po.spec.sgd", po.spec.sgd("m", learning_rate=0.01, scale_features=True, **common)),
            (
                "po.spec.ewridge, solve every 1000 rows",
                po.spec.ewridge(
                    "m", ridge=1e-6, solve_every=1e9, max_rows_between_solves=1000, **common
                ),
            ),
            (
                "po.spec.ewridge, gram_block_rows=256",
                po.spec.ewridge(
                    "m",
                    ridge=1e-6,
                    solve_every=1e9,
                    max_rows_between_solves=1000,
                    gram_block_rows=256,
                    **common,
                ),
            ),
        ):
            _, dt, bank = run_bank(spec, df)
            rows.append((name, n / dt, len(bank.save_bytes())))

        for name, rps, size in rows:
            print(f"{name:<38} {k:>6,} {n:>7,} {rps:>10,.0f} {size / 1024 / 1024:>8.2f} MB")
    print()


def grid():
    """Six penalties over one stream: N estimators against one accumulator."""
    x, y, _ = stream(ROWS, K, 0.0)
    df, feats = frame(x, y), [f"x{j}" for j in range(K)]
    sgd_regressor, standard_scaler = _sklearn()

    def sklearn_grid(alphas):
        models = [sgd_regressor(random_state=0, alpha=a) for a in alphas]
        scaler = standard_scaler()
        t0 = time.perf_counter()
        for lo in range(0, ROWS, BATCH):
            xb, yb = x[lo : lo + BATCH], y[lo : lo + BATCH]
            if lo:
                for model in models:
                    model.predict(scaler.transform(xb))
            scaler.partial_fit(xb)
            xs = scaler.transform(xb)
            for model in models:
                model.partial_fit(xs, yb)
        return time.perf_counter() - t0

    def bank_grid(ridges):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=feats,
            halflife=float("inf"),
            ridge=ridges if len(ridges) > 1 else ridges[0],
            solve_every=1e9,
            max_rows_between_solves=100,
            min_periods=50.0,
        )
        return run_bank(spec, df)[1]

    print(f"--- a grid of {len(ALPHAS)} penalties, {ROWS:,} rows, k={K}")
    print(f"{'contender':<20} {'one':>12} {'six':>12} {'six / one':>10}")
    for name, one, six in (
        ("SGDRegressor", sklearn_grid(ALPHAS[:1]), sklearn_grid(ALPHAS)),
        ("po.spec.ewridge", bank_grid(ALPHAS[:1]), bank_grid(ALPHAS)),
    ):
        print(f"{name:<20} {ROWS / one:>12,.0f} {ROWS / six:>12,.0f} {six / one:>9.2f}x")
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("accuracy", "all"):
        accuracy()
    if which in ("short", "all"):
        short()
    if which in ("wide", "all"):
        wide()
    if which in ("grid", "all"):
        grid()
