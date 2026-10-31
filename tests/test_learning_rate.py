"""Task 278 (c), class 10: the learning-rate models, `sgd`, `pa` and `ftrl`.

Each case streams a well-specified problem through the bank, with no decay,
and reads the coefficients at checkpoints. The measure is the **excess
loss**: the case's loss of those coefficients on the rows seen so far, less
the loss of the batch optimum on the same rows, which scikit-learn's batch
solvers compute (``LinearRegression``, ``LogisticRegression`` without a
penalty, ``PoissonRegressor``, and statsmodels' ``QuantReg`` and ``RLM``); it is
at least 0 by the optimum's definition. A schedule that anneals (``inv_scaling``,
AdaGrad, FTRL's per-coordinate rates) must keep lowering it, judged on the
worst of 50 seeds; a constant rate lowers it to the least-mean-squares
plateau, about ``lr (k + 1) sigma^2 / 2`` in standardized coordinates, and
holds it there. `pa` has no rate: on a realizable target it reaches the
truth, and on a noisy one its excess holds at a level its ``c`` sets.

The datasets: synthetic streams with a known optimum, and two LIBSVM sets
(``tests/public_benchmarks.py``), ``abalone`` (regression) and ``a9a``
(binary), over several passes in shuffled orders. river's online learners
give a second opinion on the plateau of a constant rate.

A learning rate too large must fail loudly (a refusal, a notice, a
``withheld_reason``, or bounded output): ``sgd``'s rate past its stability
bound is withheld as ``unstable_rate``, with a notice.

`sgd(loss="huber")` joined once task 277's refit of its first rows merged.
Its loss is Huber's at ``delta`` times a scale the model learns, the
residuals' median absolute deviation, which on these streams settles on
the noise's standard deviation, 1; the loss here and statsmodels' ``RLM``
with ``HuberT(delta)`` hold the scale at that 1, so both measure the same
loss on the same rows.

Every bound is the measurement with headroom, 2026-10-25, macOS arm64.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

import ledger
import polars_online as po
from public_benchmarks import libsvm_or_skip

TIER = "mixed"

INF = float("inf")
K = 5
FEATURES = [f"x{j}" for j in range(K)]
BETA = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
SCALE = np.array([1.0, 2.0, 0.5, 1.0, 3.0])
SHIFT = np.array([0.0, 1.0, -1.0, 2.0, 0.0])
MARKS = [1_000, 10_000, 100_000]
SEEDS = range(50)
#: `sgd`'s default Huber ``delta``, statsmodels' ``HuberT`` default too.
HUBER = sm.robust.norms.HuberT(t=1.345)


def stream(seed: int, n: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Five features of unequal scale and level; the target well specified
    for ``kind``: Gaussian noise of 1 (``squared``, ``quantile``), Bernoulli
    (``logistic``), Poisson (``poisson``)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K)) * SCALE + SHIFT
    if kind == "logistic":
        eta = 0.3 + x @ (0.4 * BETA)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif kind == "poisson":
        eta = 0.5 + (x - SHIFT) @ (0.1 * BETA / SCALE)
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = 0.3 + x @ BETA + rng.standard_normal(n)
    return x, y


def _frame(x: np.ndarray, y: np.ndarray, names: list[str]) -> pl.DataFrame:
    return pl.DataFrame({**{f: x[:, j] for j, f in enumerate(names)}, "y": y})


def coefficients(spec: dict, x: np.ndarray, y: np.ndarray, marks: list[int]) -> list[np.ndarray]:
    """The intercept and slopes after each of ``marks`` rows."""
    names = spec["features"]
    bank = po.ModelBank([spec])
    df = _frame(x, y, names)
    out, prev = [], 0
    for m in marks:
        bank.fit(df.slice(prev, m - prev))
        prev = m
        out.append(bank.coef()["coef"].to_numpy())
    return out


def _loss(kind: str, c: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float = 0.5) -> float:
    eta = c[0] + x @ c[1:]
    if kind == "squared":
        return float(np.mean((y - eta) ** 2))
    if kind == "logistic":
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    if kind == "poisson":
        return float(np.mean(np.exp(eta) - y * eta))
    if kind == "huber":
        # At the noise's scale, 1 (the module's docstring).
        return float(np.mean(HUBER.rho(y - eta)))
    r = y - eta
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def optimum(kind: str, x: np.ndarray, y: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """scikit-learn's batch optimum of the same loss on the same rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "squared":
            m = LinearRegression().fit(x, y)
            return np.r_[m.intercept_, m.coef_]
        if kind == "logistic":
            m = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-12).fit(x, y)
            return np.r_[m.intercept_, m.coef_[0]]
        if kind == "poisson":
            m = PoissonRegressor(alpha=0.0, max_iter=10_000, tol=1e-12).fit(x, y)
            return np.r_[m.intercept_, m.coef_]
        if kind == "huber":
            # statsmodels' RLM, its scale held at the noise's, 1.
            fit = sm.RLM(y, sm.add_constant(x), M=HUBER).fit(
                update_scale=False, start_scale=1.0, conv="coefs", tol=1e-12, maxiter=1_000
            )
            return np.asarray(fit.params)
        # statsmodels' QuantReg (iteratively reweighted least squares): a
        # second at 1e5 rows, where scikit-learn's linear program took eight
        # at 1e4.
        fit = sm.QuantReg(y, sm.add_constant(x)).fit(q=tau, max_iter=5_000, p_tol=1e-10)
        return np.asarray(fit.params)


def excess(kind: str, coefs: list[np.ndarray], x, y, marks, tau: float = 0.5) -> list[float]:
    out = []
    for m, c in zip(marks, coefs, strict=True):
        xs, ys = x[:m], y[:m]
        out.append(
            _loss(kind, c, xs, ys, tau) - _loss(kind, optimum(kind, xs, ys, tau), xs, ys, tau)
        )
    return out


def _sgd(**kw) -> dict:
    return po.spec.sgd("m", targets=["y"], features=FEATURES, half_life=INF, **kw)


def _ftrl(**kw) -> dict:
    return po.spec.ftrl("m", targets=["y"], features=FEATURES, half_life=INF, l1=0.0, l2=0.0, **kw)


#: The annealing schedules: ``name -> (spec, loss kind, quantile level,
#: (fall 1e3 -> 1e4, fall 1e4 -> 1e5))``: each decade's fall of the worst
#: seed's excess, at least, set at about half of what 50 seeds measured
#: (the comment: the worst seed's excess at 1e3, 1e4 and 1e5 rows).
SCHEDULES: dict[str, tuple[dict, str, float, tuple[float, float]]] = {
    # 1.10e-1, 3.44e-2, 1.12e-2: falls 3.2x, 3.1x
    "sgd inv_scaling": (
        _sgd(learning_rate=0.5, schedule="inv_scaling"),
        "squared",
        0.5,
        (1.6, 1.5),
    ),
    # 7.40e-2, 3.21e-2, 1.12e-2: 2.3x, 2.9x
    "sgd adagrad": (_sgd(learning_rate=0.5, schedule="adagrad"), "squared", 0.5, (1.25, 1.5)),
    # 5.37e-1, 1.97e-2, 6.39e-3: 27x, 3.1x
    "ftrl squared": (_ftrl(loss="squared"), "squared", 0.5, (8.0, 1.5)),
    # 1.83e-2, 2.58e-3, 1.03e-3: 7.1x, 2.5x
    "ftrl logistic": (_ftrl(), "logistic", 0.5, (3.5, 1.25)),
    # 5.04e-3, 1.88e-3, 9.42e-4: 2.7x, 2.0x
    "sgd logistic inv_scaling": (
        _sgd(loss="logistic", learning_rate=0.5, schedule="inv_scaling"),
        "logistic",
        0.5,
        (1.35, 1.25),
    ),
    # 1.35e-2, 4.10e-3, 2.52e-3: 3.3x, 1.6x
    "sgd quantile inv_scaling": (
        _sgd(loss="quantile", quantile=0.5, learning_rate=0.5, schedule="inv_scaling"),
        "quantile",
        0.5,
        (1.6, 1.25),
    ),
    # 4.65e-2, 1.25e-2, 4.78e-3: 3.7x, 2.6x (2026-10-26)
    "sgd huber inv_scaling": (
        _sgd(loss="huber", learning_rate=0.5, schedule="inv_scaling"),
        "huber",
        0.5,
        (1.85, 1.3),
    ),
    # 1.57e-1, 1.34e-3, 4.62e-4: 117x, 2.9x
    "sgd quantile 0.9 inv_scaling": (
        _sgd(loss="quantile", quantile=0.9, learning_rate=0.5, schedule="inv_scaling"),
        "quantile",
        0.9,
        (10.0, 1.5),
    ),
    # 8.20e-3, 5.14e-3, 1.83e-3: 1.6x, 2.8x
    "sgd poisson adagrad": (
        _sgd(loss="poisson", learning_rate=0.1, schedule="adagrad"),
        "poisson",
        0.5,
        (1.25, 1.4),
    ),
}


def slug(name: str) -> str:
    """A schedule's or a case's name as a ledger case's: ``sgd quantile 0.9
    inv_scaling`` is ``sgd_quantile_0p9_inv_scaling``."""
    return name.replace(" ", "_").replace(".", "p")


#: The ledger's inputs (tests/ledger.py): the coefficients at every mark on
#: the first two seeds of each schedule, where the grid reads fifty.
ledger.guard(
    "schedules",
    lambda: {
        name: [coefficients(spec, *stream(s, MARKS[-1], kind), MARKS) for s in range(2)]
        for name, (spec, kind, _, _) in sorted(SCHEDULES.items())
    },
    oracles=["scikit-learn", "statsmodels"],
)


def worst_excess(spec: dict, kind: str, tau: float, seeds, marks=MARKS) -> np.ndarray:
    """Each mark's excess, a row a seed."""
    rows = []
    for s in seeds:
        x, y = stream(s, marks[-1], kind)
        rows.append(excess(kind, coefficients(spec, x, y, marks), x, y, marks, tau))
    return np.array(rows)


def test_an_annealed_rate_keeps_lowering_the_excess_loss():
    """The essential case: one seed, ``inv_scaling``, two decades."""
    spec, kind, tau, (f1, _) = SCHEDULES["sgd inv_scaling"]
    ex = worst_excess(spec, kind, tau, [0], MARKS[:2])[0]
    assert ex[0] > 0 and ex[1] > 0, ex  # the optimum is below the online fit
    assert ex[1] * f1 <= ex[0], ex


@pytest.mark.oracle_grid(reason="50 seeds of 100,000 rows per schedule, each with batch optima")
@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_each_schedule_keeps_lowering_the_worst_seeds_excess_loss(name):
    spec, kind, tau, (f1, f2) = SCHEDULES[name]
    ex = worst_excess(spec, kind, tau, SEEDS)
    worst = ex.max(axis=0)
    p95 = np.quantile(ex, 0.95, axis=0)
    print(f"{name}: worst {worst}, p95 {p95}, worst seeds {ex.argmax(axis=0)}")
    for i, mark in enumerate(MARKS):
        ledger.record("schedules", f"{slug(name)}_at_{mark}", ex[:, i])
    assert (ex > -1e-9).all(), "an online fit below the batch optimum: the oracle is wrong"
    assert worst[1] * f1 <= worst[0], (worst, f1)
    assert worst[2] * f2 <= worst[1], (worst, f2)


CONSTANT_RATE = 0.01
CONSTANT_MARKS = [100, 3_000, 10_000, 30_000, 100_000]

ledger.guard(
    "constant_rate",
    lambda: [
        coefficients(
            _sgd(learning_rate=CONSTANT_RATE),
            *stream(s, CONSTANT_MARKS[-1], "squared"),
            CONSTANT_MARKS,
        )
        for s in range(3)
    ],
    oracles=["scikit-learn"],
)


@pytest.mark.oracle_grid(reason="50 seeds of 100,000 rows, each with batch optima")
def test_a_constant_rate_falls_to_its_plateau_and_holds_it():
    """Least mean squares at a constant rate does not converge: it settles
    where each row's step balances the noise, an excess of about ``lr (k +
    1) sigma^2 / 2``, 0.03 here (standardized coordinates, five features and
    the intercept, unit noise). Measured over 50 seeds: the worst 7.9e-2,
    7.5e-2, 6.8e-2 and 9.0e-2 at 3e3 to 1e5 rows, the median 2.4e-2 to
    2.9e-2; at 100 rows the worst 7.0. Bounds: the worst four times the
    plateau, the median at 1e5 twice it."""
    lr = CONSTANT_RATE
    plateau = lr * (K + 1) * 1.0 / 2
    marks = CONSTANT_MARKS
    ex = worst_excess(_sgd(learning_rate=lr), "squared", 0.5, SEEDS, marks)
    worst = ex.max(axis=0)
    print(f"constant: worst {worst}, median {np.median(ex, axis=0)}")
    for i, mark in enumerate(marks):
        ledger.record("constant_rate", f"excess_at_{mark}", ex[:, i])
    ledger.record("constant_rate", "median_excess_at_100000", np.median(ex[:, -1]))
    assert worst[0] > 10 * plateau, "the first rows should be far from the plateau"
    assert (worst[1:] < 4 * plateau).all(), worst
    # It holds the plateau rather than drifting off it.
    assert np.median(ex[:, -1]) < 2 * plateau, np.median(ex[:, -1])


@pytest.mark.extended(reason="river's learners, row by row in Python, over 20,000 rows")
def test_rivers_constant_rate_learner_settles_on_the_same_plateau():
    """A second opinion on the plateau: river's ``StandardScaler |
    LinearRegression`` with plain SGD and no L2 is ``sgd``'s standardized
    constant-rate fit, at half the rate, since river's squared loss is
    ``(p - y)^2`` with the gradient ``2 (p - y)``. Their excess at 20,000
    rows, the median over 5 seeds, agree within 10%. Measured: ours 0.0636,
    river's 0.0638; at the same nominal rate, before the halving, river's
    was 0.153."""
    from river import linear_model, optim, preprocessing

    lr, n = 0.01, 20_000
    ours, theirs = [], []
    for s in range(5):
        x, y = stream(s, n, "squared")
        ours.append(
            excess("squared", coefficients(_sgd(learning_rate=lr), x, y, [n]), x, y, [n])[0]
        )
        model = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(lr / 2), intercept_lr=lr / 2, l2=0.0
        )
        for row, target in zip(x, y, strict=True):
            model.learn_one(dict(zip(FEATURES, row, strict=True)), target)
        pred = np.array([model.predict_one(dict(zip(FEATURES, r, strict=True))) for r in x])
        b = optimum("squared", x, y)
        theirs.append(np.mean((y - pred) ** 2) - _loss("squared", b, x, y))
    print(f"excess at {n}: ours {np.median(ours):.4f}, river's {np.median(theirs):.4f}")
    assert 0.9 < np.median(ours) / np.median(theirs) < 1.1, (ours, theirs)


# --- pa ------------------------------------------------------------------------


def test_pa_reaches_a_realizable_target():
    """Without noise every mode's excess falls toward 0 (Crammer et al.'s
    mistake bound: the cumulative loss is finite on a realizable stream)."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.standard_normal((n, K)) * SCALE + SHIFT
    y = 0.3 + x @ BETA
    for mode in ("pa", "pa1", "pa2"):
        spec = po.spec.pa("m", targets=["y"], features=FEATURES, half_life=INF, mode=mode)
        c = coefficients(spec, x, y, [1_000, n])
        ex = [
            np.mean((y[:m] - ci[0] - x[:m] @ ci[1:]) ** 2)
            for m, ci in zip([1_000, n], c, strict=True)
        ]
        assert ex[1] < 1e-2 * np.var(y), (mode, ex)


PA_MARKS = [10_000, 100_000]
PA_SPECS = {
    "default_c": po.spec.pa("m", targets=["y"], features=FEATURES, half_life=INF),
    "c_0p01": po.spec.pa("m", targets=["y"], features=FEATURES, half_life=INF, c=0.01),
}

ledger.guard(
    "pa_noisy",
    lambda: {
        name: [coefficients(spec, *stream(s, PA_MARKS[-1], "squared"), PA_MARKS) for s in range(3)]
        for name, spec in sorted(PA_SPECS.items())
    },
    oracles=["scikit-learn"],
)


@pytest.mark.oracle_grid(reason="50 seeds of 100,000 rows, each with batch optima")
def test_pas_excess_on_a_noisy_target_holds_where_c_sets_it():
    """On a noisy target PA has no rate to anneal, so its excess does not fall
    to 0: every row outside the tube moves the fit by up to ``c``. A smaller
    ``c`` holds it lower. Measured worst over 50 seeds at 1e4 and 1e5 rows:
    ``pa1`` at the default ``c`` 3.1 and 3.9 noise variances, flat with the
    rows; at ``c = 0.01``, 0.083 and 0.092. Bounds 8, and a quarter of the
    default's."""
    marks = PA_MARKS
    big = worst_excess(PA_SPECS["default_c"], "squared", 0.5, SEEDS, marks)
    small = worst_excess(PA_SPECS["c_0p01"], "squared", 0.5, SEEDS, marks)
    print(f"pa1 default c: worst {big.max(0)}; c = 0.01: worst {small.max(0)}")
    for name, ex in (("default_c", big), ("c_0p01", small)):
        for i, mark in enumerate(marks):
            ledger.record("pa_noisy", f"{name}_excess_at_{mark}", ex[:, i])
    assert (big.max(0) < 8.0).all(), big.max(0)
    assert small.max(0)[-1] < big.max(0)[-1] / 4, (small.max(0), big.max(0))


# --- LIBSVM -----------------------------------------------------------------------


def _passes(spec: dict, x, y, order, epochs: int) -> list[np.ndarray]:
    names = spec["features"]
    bank = po.ModelBank([spec])
    df = _frame(x[order], y[order], names)
    out = []
    for _ in range(epochs):
        bank.fit(df)
        out.append(bank.coef()["coef"].to_numpy())
    return out


LIBSVM_CASES = [
    "abalone sgd inv_scaling",
    "abalone sgd adagrad",
    "abalone ftrl",
    "a9a ftrl",
    "a9a sgd",
]


def libsvm_case(case: str) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """A case's set, its loss and its spec."""
    data, model = case.split(" ", 1)
    if data == "abalone":
        x, y = libsvm_or_skip("regression/abalone", 8)
        kind = "squared"
    else:
        x, y = libsvm_or_skip("binary/a9a", 123)
        y = (y > 0).astype(float)
        x = x[:, x.std(axis=0) > 0]
        kind = "logistic"
    names = [f"x{j}" for j in range(x.shape[1])]
    specs = {
        "sgd inv_scaling": dict(learning_rate=0.5, schedule="inv_scaling"),
        "sgd adagrad": dict(learning_rate=0.5, schedule="adagrad"),
    }
    if model == "ftrl":
        loss = "squared" if kind == "squared" else "logistic"
        spec = po.spec.ftrl(
            "m", targets=["y"], features=names, half_life=INF, l1=0.0, l2=0.0, loss=loss
        )
    elif model == "sgd":
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=names,
            half_life=INF,
            loss="logistic",
            learning_rate=0.5,
            schedule="inv_scaling",
        )
    else:
        spec = po.spec.sgd("m", targets=["y"], features=names, half_life=INF, **specs[model])
    return x, y, kind, spec


def libsvm_order(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


#: The first of the ten orders of each case: 0.4 s.
ledger.guard(
    "libsvm_passes",
    lambda: {
        case: [_passes(spec, x, y, libsvm_order(len(y), s), 4) for s in range(1)]
        for case in LIBSVM_CASES
        for x, y, _, spec in [libsvm_case(case)]
    },
    oracles=["scikit-learn"],
    dataset="libsvm:regression/abalone,binary/a9a",
)


@pytest.mark.oracle_grid(reason="the network, and four passes over a LIBSVM set in ten orders")
@pytest.mark.parametrize("case", LIBSVM_CASES)
def test_passes_over_a_libsvm_set_keep_lowering_the_worst_orders_excess(case):
    """Four passes over the whole set, in each of ten shuffled orders; after
    each pass the excess over scikit-learn's batch optimum on the whole set.
    The worst order's excess falls with every pass. Measured worst over the
    ten orders: abalone ``inv_scaling`` 0.45, 0.37, 0.33, 0.30; AdaGrad
    0.38, 0.26, 0.21, 0.18; FTRL 3.42, 3.06, 2.84, 2.67 (its default rate
    takes an unscaled target at a level of 10 slowly: after four passes its
    mean squared error is 1.55 times the optimum's 4.83); a9a FTRL 4.0e-3,
    2.4e-3, 1.8e-3, 1.5e-3; ``sgd`` logistic ``inv_scaling`` 9.6e-3,
    5.8e-3, 4.2e-3, 3.4e-3."""
    x, y, kind, spec = libsvm_case(case)
    best = _loss(kind, optimum(kind, x, y), x, y)
    ex = np.array(
        [
            [_loss(kind, c, x, y) - best for c in _passes(spec, x, y, libsvm_order(len(y), s), 4)]
            for s in range(10)
        ]
    )
    worst = ex.max(axis=0)
    print(f"{case}: worst by pass {worst}, median {np.median(ex, axis=0)}")
    for i in range(ex.shape[1]):
        ledger.record("libsvm_passes", f"{slug(case)}_pass_{i + 1}", ex[:, i])
    assert (ex > -1e-9).all(), ex.min()
    assert (np.diff(worst) < 0).all(), worst


# --- a rate too large ---------------------------------------------------------------


def _diverged(spec: dict, n: int = 20_000) -> tuple[float, float, list, bool]:
    """The last 1,000 rows' out-of-sample mean squared error over the
    target's variance, the largest prediction, the withheld reasons and
    whether any prediction was NaN."""
    x, y = stream(0, n, "squared")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = po.ModelBank([spec]).fit_predict(_frame(x, y, FEATURES)).unnest("m")
    pred = out["pred_y"].to_numpy().astype(float)
    tail = slice(n - 1_000, n)
    # Every row withheld (a gate) leaves no error to read: NaN.
    seen = np.isfinite(pred[tail])
    ratio = float(np.mean((pred[tail] - y[tail])[seen] ** 2) / np.var(y)) if seen.any() else np.nan
    withheld = out["withheld_reason"].is_not_null().to_numpy()
    reasons = out["withheld_reason"].drop_nulls().unique().to_list()
    notices = [str(w.message) for w in caught]
    # A NaN where no gate withheld the row; the warm-up's rows are null.
    nan = bool(np.isnan(pred[~withheld]).any())
    largest = float(np.nanmax(np.abs(pred))) if np.isfinite(pred).any() else np.nan
    return ratio, largest, [*reasons, *notices], nan


@pytest.mark.parametrize(
    "rate",
    # Least mean squares is stable in the mean square below 2 / tr(R): in
    # standardized coordinates tr(R) = k + 1 = 6, so about 0.33. Its excess
    # error is about lr tr(R) / (2 - lr tr(R)) noise variances: 0.43 at
    # 0.1 and 1.5 at 0.2, so with the noise itself (1.43 and 2.5) against
    # the target's variance of 28.3, 0.05 and 0.09 of it. Measured on the
    # last 1,000 rows of one stream: 0.052 and 0.137; 0.78 at 0.25, near
    # the edge. Bound 0.25.
    [0.1, 0.2],
)
def test_a_stable_constant_rate_is_quiet_and_right(rate):
    ratio, _, reasons, nan = _diverged(_sgd(learning_rate=rate))
    print(f"rate {rate}: error over the target's variance {ratio:.4f}")
    assert not nan
    assert ratio < 0.25, ratio
    assert reasons == ["below_min_weight"], reasons


@pytest.mark.parametrize("rate", [0.5, 2.0, 100.0])
def test_a_constant_rate_too_large_fails_loudly(rate):
    """What failing loudly means here: a refusal at the spec, a warning, a
    ``withheld_reason`` other than the warm-up's on the diverged rows, or an
    error that stays within ten times the target's variance."""
    try:
        spec = _sgd(learning_rate=rate)
    except ValueError:
        return  # refused at the spec: loud
    ratio, _, reasons, nan = _diverged(spec)
    loud = nan or ratio < 10 or any(r != "below_min_weight" for r in reasons)
    assert loud, (ratio, reasons)


@pytest.mark.parametrize("rate", [0.5, 2.0, 100.0])
def test_a_constant_rate_past_its_bound_is_withheld_with_a_notice(rate):
    """What ``sgd`` does past the bound (task 278, D8; it was quiet: at 0.5
    the last 1,000 rows' error was 3.6e4 times the target's variance, the clip
    holding the fit finite and wrong). A step multiplies its own row's
    residual by ``1 - q``, ``q = lr * w * |z|^2`` here, so the fit diverges in
    mean square once ``E[q^2] > 2 E[q]``: on five standardized normal
    features and the intercept's 1 that is a rate past 0.261. From the first
    rows on every prediction is withheld as ``unstable_rate``, and the bank
    says so once."""
    x, y = stream(0, 2_000, "squared")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = po.ModelBank([_sgd(learning_rate=rate)]).fit_predict(_frame(x, y, FEATURES))
    out = out.unnest("m")
    reasons = out["withheld_reason"].cast(pl.String)
    counts = reasons.value_counts().sort("withheld_reason").to_dicts()
    print(f"rate {rate}: {counts}")
    assert out["pred_y"].is_not_null().sum() == 0
    assert (reasons == "unstable_rate").sum() > 1_990
    said = [str(w.message) for w in caught if "stability bound" in str(w.message)]
    assert len(said) == 1, said
    assert "bound at 0.261" in said[0]


@pytest.mark.parametrize(
    "spec",
    [
        # Bounded by construction: a probability, and FTRL's per-coordinate
        # rates, which shrink as the squared gradients add up.
        po.spec.sgd(
            "m", targets=["y"], features=FEATURES, half_life=INF, loss="logistic", learning_rate=1e3
        ),
        po.spec.ftrl(
            "m", targets=["y"], features=FEATURES, half_life=INF, loss="squared", alpha=1e6
        ),
        po.spec.sgd(
            "m",
            targets=["y"],
            features=FEATURES,
            half_life=INF,
            learning_rate=1e3,
            schedule="adagrad",
        ),
    ],
    ids=["sgd logistic 1e3", "ftrl alpha 1e6", "sgd adagrad 1e3"],
)
def test_a_large_rate_that_stays_bounded(spec):
    """Measured (2026-10-25): logistic predictions stay probabilities (and
    at 1e3 ``sgd``'s gate withholds most of them, its steps past their
    stability bound, ``p (1 - p) |z|^2`` the curvature); FTRL at ``alpha =
    1e6`` ends at 2.4 noise variances; AdaGrad at 1e3 at 19, its own rate's
    normalisation holding it. NaN-free and bounded, so loud enough: the
    error is the caller's to see, not hidden behind a number that grows."""
    if spec["model"]["loss"] == "logistic":
        x, y = stream(0, 20_000, "logistic")
        out = po.ModelBank([spec]).fit_predict(_frame(x, y, FEATURES)).unnest("m")
        p = out["pred_y"].drop_nulls().to_numpy()
        assert ((p >= 0) & (p <= 1)).all()
        return
    ratio, _, _, nan = _diverged(spec)
    assert not nan
    assert ratio < 10, ratio
