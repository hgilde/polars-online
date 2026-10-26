"""The convergence harness (docs/PLAN.md task 278).

Task 277's capture -- one online Huber pass pulled for good toward least
squares on some streams -- was invisible to a median over seeds (0.001 at
12.5% outliers) while 13 of 40 seeds were captured, the worst by 0.43. So
every convergence class here follows four rules:

1. **The tail, not the middle.** A class runs a grid of 50 or more seeds and
   asserts on the worst seed, or a high quantile where one seed's luck is
   the statistic (and says so); it reports the seed that was worst.
2. **A third-party oracle on the same weighted rows.** statsmodels or
   scikit-learn fits the batch problem the model is meant to solve, on the
   rows the model saw, each at its prior weight ``v_i = 2^-(age_i / h)``
   under a half-life ``h`` (:func:`decay_weights`). The model's own
   iteration is never the oracle. The oracle's answer is taken as given:
   these tests check this library, not the oracle.
3. **The rate, not only the endpoint.** Error against the row count (no
   decay) or against the half-lives since a break (decay), fitted as a
   slope on log-log axes (:func:`rate`).
4. **Evidence the hard path ran.** Each test prints what it measured --
   the tail, the worst seed, and the quantity that shows the adversity
   activated (outliers present, the break taken, the start far away) --
   before it asserts, so a green result can be read.

Each regression model **registers** here (:data:`REGISTRY`): a generator of
known-truth streams, the oracle, the classes that apply to it, and one
sentence saying what it converges to. ``tests/test_convergence.py`` holds
every model ``po.spec`` builds that predicts a target -- and each of its
losses, where it takes ``loss=`` -- to have a registration, or an entry in
:data:`PENDING` or :data:`NOT_REGISTERED` that says why not
(``docs/EXTENDING.md``, step 12).

The classes (task 278 numbers them; this module builds 1 to 5):

==  ===============  ==========================================================
1   consistency      stationary known truth, no decay: the worst seed's error
                     falls with ``n`` toward the oracle at the theory's rate
2   start            a wrong start (a state from another regime, an extreme
                     prior, a large level) reaches the same fit
3   adversity        outliers, a level shift or heavy tails in the first rows
4   breakdown        contamination below the documented breakdown point
                     captures no seed
5   reconvergence    after a break in level, slope or variance, the fit comes
                     within a stated distance of the decayed oracle within a
                     stated number of half-lives
==  ===============  ==========================================================
"""

from __future__ import annotations

import inspect
import typing
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge

import polars_online as po

INF = float("inf")
FEATURES = ["x0", "x1"]
#: The known truth of the linear streams: intercept, then a slope per feature.
TRUTH = np.array([1.0, 2.0, -1.0])
#: The noise's standard deviation (or scale, for a heavy tail) on the linear
#: streams.
NOISE = 0.5

CLASSES = {
    1: "consistency",
    2: "start",
    3: "adversity",
    4: "breakdown",
    5: "reconvergence",
}


# ---------------------------------------------------------------------------
# Streams with a known truth


@dataclass(frozen=True)
class Stream:
    """A frame and the coefficients it was drawn from, in ``coef``'s layout
    (intercept first)."""

    frame: pl.DataFrame
    truth: np.ndarray

    @property
    def X(self) -> np.ndarray:
        """The design the oracle fits: a 1, then the features."""
        cols = [c for c in FEATURES if c in self.frame.columns]
        x = self.frame.select(cols).to_numpy() if cols else np.empty((self.frame.height, 0))
        return np.column_stack([np.ones(self.frame.height), x])

    @property
    def y(self) -> np.ndarray:
        return self.frame["y"].to_numpy()


def noise(rng: np.random.Generator, n: int, kind: str = "normal") -> np.ndarray:
    """Unit-scale noise: ``normal``, ``laplace`` (variance 2), ``t3``, ``t2``
    (infinite variance) or ``cauchy`` (no mean)."""
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "laplace":
        return rng.laplace(size=n)
    if kind == "t3":
        return rng.standard_t(3, n)
    if kind == "t2":
        return rng.standard_t(2, n)
    if kind == "cauchy":
        return rng.standard_cauchy(n)
    raise ValueError(kind)


def features(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two standard normal features correlated at 0.5, so a fit has a
    direction it learns slower than the others."""
    x0 = rng.standard_normal(n)
    x1 = 0.5 * x0 + np.sqrt(0.75) * rng.standard_normal(n)
    return x0, x1


def linear(
    n: int,
    seed: int,
    *,
    kind: str = "normal",
    level: float = 0.0,
    coef: np.ndarray = TRUTH,
    scale: float = NOISE,
) -> Stream:
    """``y = level + coef[0] + coef[1:] . x + scale * e``, ``e`` of ``kind``."""
    rng = np.random.default_rng(seed)
    x0, x1 = features(rng, n)
    y = level + coef[0] + coef[1] * x0 + coef[2] * x1 + scale * noise(rng, n, kind)
    truth = coef + np.array([level, 0.0, 0.0])
    return Stream(pl.DataFrame({"x0": x0, "x1": x1, "y": y}), truth)


#: The Poisson and logistic streams' truth: a log-rate and a log-odds.
POISSON_TRUTH = np.array([0.5, 0.3, -0.2])
LOGISTIC_TRUTH = np.array([0.3, 1.0, -0.7])


def poisson(n: int, seed: int, *, coef: np.ndarray = POISSON_TRUTH) -> Stream:
    """A count, Poisson with log-rate ``coef[0] + coef[1:] . x``."""
    rng = np.random.default_rng(seed)
    x0, x1 = features(rng, n)
    y = rng.poisson(np.exp(coef[0] + coef[1] * x0 + coef[2] * x1)).astype(float)
    return Stream(pl.DataFrame({"x0": x0, "x1": x1, "y": y}), coef.copy())


def logistic(n: int, seed: int, *, coef: np.ndarray = LOGISTIC_TRUTH) -> Stream:
    """A 0/1 label with log-odds ``coef[0] + coef[1:] . x``."""
    rng = np.random.default_rng(seed)
    x0, x1 = features(rng, n)
    p = 1.0 / (1.0 + np.exp(-(coef[0] + coef[1] * x0 + coef[2] * x1)))
    y = (rng.random(n) < p).astype(float)
    return Stream(pl.DataFrame({"x0": x0, "x1": x1, "y": y}), coef.copy())


#: The trend stream's level at its first row and its slope a clock unit.
TREND_TRUTH = np.array([3.0, 0.01])


def trend(
    n: int,
    seed: int,
    *,
    coef: np.ndarray = TREND_TRUTH,
    scale: float = NOISE,
    level: float = 0.0,
    kind: str = "normal",
) -> Stream:
    """``y = level + coef[0] + coef[1] * t + scale * e`` on a row-count clock,
    no features; the truth is ``holt``'s ``coef``: the level at the last row,
    then the slope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = level + coef[0] + coef[1] * t + scale * noise(rng, n, kind)
    end = level + coef[0] + coef[1] * (n - 1)
    return Stream(pl.DataFrame({"y": y}), np.array([end, coef[1]]))


def contaminate(
    s: Stream, share: float, *, seed: int, size: float = 50.0, rows: slice | None = None
) -> tuple[Stream, int]:
    """Add ``size`` to ``y`` on rows with ``x0 > 0``, at rate ``2 * share``,
    so ``share`` is the outliers' share of every row (``docs/HUBER.md``'s
    convention): one-sided wild targets concentrated on one side of a
    feature, the case that tilts a slope. ``rows`` keeps them to a slice.
    Returns the stream and how many rows were made wild."""
    rng = np.random.default_rng(10_000 + seed)
    n = s.frame.height
    wild = (rng.random(n) < 2 * share) & (s.frame["x0"].to_numpy() > 0)
    if rows is not None:
        keep = np.zeros(n, bool)
        keep[rows] = True
        wild &= keep
    y = s.y + size * wild
    return Stream(s.frame.with_columns(y=pl.Series(y)), s.truth), int(wild.sum())


# ---------------------------------------------------------------------------
# Oracles: third-party batch fits on the same weighted rows


def decay_weights(n: int, half_life: float) -> np.ndarray:
    """Each row's weight at the last row, on a row-count clock:
    ``2^-((n - 1 - i) / h)``, all 1 without decay."""
    if np.isinf(half_life):
        return np.ones(n)
    return 0.5 ** ((n - 1 - np.arange(n)) / half_life)


def ols(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Weighted least squares, statsmodels' ``WLS``."""
    return np.asarray(sm.WLS(y, X, weights=v).fit().params)


def ridge_on_means(penalty: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """``ewridge``'s default system, ``(S + r D) b = eta`` with ``S`` the
    weighted *mean* of ``z z'`` and the intercept unpenalised: scikit-learn's
    ``Ridge`` at ``alpha = r * sum(v)``, which centres by the weighted means."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        m = Ridge(alpha=penalty * v.sum(), fit_intercept=True, solver="cholesky")
        m.fit(X[:, 1:], y, sample_weight=v)
        return np.concatenate([[m.intercept_], m.coef_])

    return fit


def ridge_on_sums(delta: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """RLS's prior, ``(A0 + sum v z z') b = sum v z y`` with ``A0 = delta I``,
    the intercept penalised too: ``Ridge`` through the origin on ``[1, x]``.
    The prior decays with the sums, ``lam`` a row from the first, so at the
    last row it weighs ``delta`` times the first row's weight, ``v[0]``."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        m = Ridge(alpha=delta * v[0], fit_intercept=False, solver="cholesky")
        return np.asarray(m.fit(X, y, sample_weight=v).coef_)

    return fit


def lasso_oracle(penalty: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """``lasso``'s problem: the features standardised by their weighted mean
    and population standard deviation, the target centred, ``(1 / 2W) sum v
    r^2 + l |b|_1``; scikit-learn's ``Lasso`` solves it, and the slopes are
    unscaled and the intercept recovered as ``ybar - m . beta``."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        x = X[:, 1:]
        w = v / v.sum()
        m = w @ x
        s = np.sqrt(w @ (x - m) ** 2)
        model = Lasso(alpha=penalty, fit_intercept=True, tol=1e-12, max_iter=100_000)
        model.fit((x - m) / s, y, sample_weight=v)
        beta = model.coef_ / s
        return np.concatenate([[w @ y - m @ beta], beta])

    return fit


def quantreg(q: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """statsmodels' ``QuantReg`` at level ``q``. It takes no weights; the
    check loss is positively homogeneous, ``v rho(y - x b) = rho(v y - v x b)``,
    so the weighted fit is ``QuantReg`` on the rows scaled by their weights."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sm.QuantReg(v * y, v[:, None] * X).fit(q=q, max_iter=5000, p_tol=1e-10)
        return np.asarray(r.params)

    return fit


def insensitive(eps: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """The epsilon-insensitive fit, ``min sum v max(0, |y - x b| - e)``, as
    the linear programme it is, solved by scipy's HiGHS. ``e`` is ``eps``
    times the target's weighted standard deviation, the units ``sgd`` draws
    its tube in (its running value converges to this one without decay)."""
    from scipy.optimize import linprog
    from scipy.sparse import eye, hstack

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        n, k = X.shape
        w = v / v.sum()
        e = eps * np.sqrt(w @ (y - w @ y) ** 2)
        # variables [b (free), u >= 0, d >= 0]: y - X b <= e + u, X b - y <= e + d
        cost = np.concatenate([np.zeros(k), v, v])
        a = hstack([np.vstack([-X, X]), -eye(2 * n)]).tocsr()
        rhs = np.concatenate([e - y, e + y])
        bounds = [(None, None)] * k + [(0, None)] * (2 * n)
        r = linprog(cost, A_ub=a, b_ub=rhs, bounds=bounds, method="highs-ipm")
        assert r.status == 0, r.message
        return np.asarray(r.x[:k])

    return fit


def huber_oracle(
    delta: float = 1.345,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """The converged batch Huber fit with a MAD scale, statsmodels' ``RLM``
    loop with prior weights ``v``: weighted least squares (``WLS``) at ``v``
    times ``HuberT``'s weights of the residuals over the scale, the scale the
    ``v``-weighted median of ``|r|`` over ``Phi^-1(3/4)``, from ``WLS`` at
    ``v``, until the coefficients settle. At equal weights it is ``RLM``'s
    ``HuberT(delta)`` with ``scale_est="mad"``."""
    from statsmodels.robust.norms import HuberT

    q3 = 0.6744897501960817

    def wmedian(a: np.ndarray, w: np.ndarray) -> float:
        o = np.argsort(a)
        a, w = a[o], w[o]
        c = np.cumsum(w)
        half = 0.5 * c[-1]
        i = int(np.searchsorted(c, half, side="left"))
        if np.isclose(c[i], half, rtol=1e-13, atol=0) and i + 1 < len(a):
            return 0.5 * (a[i] + a[i + 1])
        return float(a[i])

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        keep = v > 1e-300
        X, y, v = X[keep], y[keep], v[keep]
        b = sm.WLS(y, X, weights=v).fit().params
        norm = HuberT(delta)
        for _ in range(200):
            r = y - X @ b
            s = wmedian(np.abs(r), v) / q3
            new = sm.WLS(y, X, weights=v * norm.weights(r / s)).fit().params
            if np.max(np.abs(new - b)) <= 1e-11 * (1 + np.max(np.abs(new))):
                return np.asarray(new)
            b = new
        return np.asarray(b)

    return fit


def glm(family: sm.families.Family) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """statsmodels' ``GLM`` maximum likelihood, the rows at ``var_weights``."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        r = sm.GLM(y, X, family=family, var_weights=v).fit(tol=1e-12, maxiter=200)
        return np.asarray(r.params)

    return fit


def trend_oracle(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A linear trend fitted to the history on its row clock by weighted
    least squares (statsmodels), read as ``holt`` reads its state: the
    fitted level at the last row, then the slope."""
    t = np.arange(len(y), dtype=float)
    b = ols(np.column_stack([np.ones(len(y)), t]), y, v)
    return np.array([b[0] + b[1] * t[-1], b[1]])


# ---------------------------------------------------------------------------
# Registrations


@dataclass(frozen=True)
class Registration:
    """What a model brings to the harness.

    ``build(half_life, **kw)`` makes its spec, named ``"m"``, on target
    ``"y"`` and :data:`FEATURES`. ``stream(n, seed, **kw)`` draws a stream
    with a known truth. ``oracle(X, y, v)`` is the third-party batch fit the
    model is meant to reach on those rows at weights ``v``. ``classes`` are
    the classes that apply (:data:`CLASSES`); ``limit`` says in one sentence
    what the model converges to."""

    key: str
    builder: str
    build: Callable[..., dict]
    stream: Callable[..., Stream]
    oracle: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    classes: frozenset[int]
    limit: str
    #: Whether the model keeps a robust fit: the classes that contaminate a
    #: stream hold it to the robust oracle, and the others to theirs.
    robust: bool = False
    #: The stream's keywords for *another regime*: a start a long way from
    #: this stream's truth (class 2) and the far side of a break (class 5).
    other: dict = field(default_factory=dict)
    #: The level class 2 adds to the target, or ``None`` where the stream has
    #: no level to add (a count, a label); with the reason in ``limit``.
    level: float | None = 1e6


def _spec(builder: str, **fixed: object) -> Callable[..., dict]:
    def build(half_life: float, **kw: object) -> dict:
        args: dict[str, object] = {"targets": ["y"], "half_life": half_life}
        if builder != "holt":
            args["features"] = FEATURES
        args.update(fixed)
        args.update(kw)
        return getattr(po.spec, builder)("m", **args)

    return build


def _kalman(half_life: float, **kw: object) -> dict:
    """``kalman`` forgets through its coefficients' drift, not the spec's
    half-life (which drives its scaler and its noise): no drift without
    decay, and ``coef_half_life`` at the half-life under one."""
    drift = {"q": [0.0, 0.0, 0.0]} if np.isinf(half_life) else {"coef_half_life": half_life}
    return _spec("kalman", **drift)(half_life, **kw)


def _laplace(n: int, seed: int, **kw: object) -> Stream:
    return linear(n, seed, **{"kind": "laplace", **kw})


#: Another regime for each kind of stream: every coefficient moved and the
#: level too, the start class 2 loads and the far side of class 5's break.
OTHER_LINEAR = {"coef": np.array([-3.0, -2.0, 4.0]), "level": 50.0}
OTHER_POISSON = {"coef": np.array([-1.0, -0.5, 0.8])}
OTHER_LOGISTIC = {"coef": np.array([-1.0, -2.0, 1.5])}
OTHER_TREND = {"coef": np.array([100.0, -0.05])}


def _registrations() -> list[Registration]:
    lin = frozenset({1, 2, 3, 5})
    return [
        Registration(
            key="ewridge",
            builder="ewridge",
            build=_spec("ewridge", solve_every=0.0),
            stream=linear,
            oracle=ridge_on_means(1e-6),
            classes=lin,
            limit="the batch ridge fit of every row at its weight, exactly: its sums are the "
            "problem",
            other=OTHER_LINEAR,
        ),
        Registration(
            key="rls",
            builder="rls",
            build=_spec("rls"),
            stream=linear,
            oracle=ridge_on_sums(1.0),
            classes=lin,
            limit="the batch fit with RLS's prior delta I, which fades as 1/n against the rows",
            other=OTHER_LINEAR,
        ),
        Registration(
            key="lasso",
            builder="lasso",
            build=_spec("lasso", penalties=[0.05], solve_every=0.0),
            stream=linear,
            oracle=lasso_oracle(0.05),
            classes=lin,
            limit="scikit-learn's Lasso at the same penalty on the standardised rows, solved "
            "on every row",
            other=OTHER_LINEAR,
        ),
        Registration(
            key="kalman",
            builder="kalman",
            build=_kalman,
            stream=linear,
            oracle=ols,
            classes=lin,
            limit="without decay, no process noise: Bayesian linear regression from a prior "
            "sized by the first rows' noise, least squares with the prior fading as 1/n; "
            "under a half-life h, coefficients that drift at coef_half_life = h, whose "
            "steady-state gain is EW least squares' at h",
            other=OTHER_LINEAR,
        ),
        Registration(
            key="quantile",
            builder="quantile",
            build=_spec("quantile", quantile=0.5, solve_every=0.0),
            stream=_laplace,
            oracle=quantreg(0.5),
            classes=frozenset({1, 2, 3, 4, 5}),
            limit="statsmodels' QuantReg on the same rows, its smoothing band narrowing as the "
            "effective sample grows",
            robust=True,
            other=OTHER_LINEAR,
        ),
        Registration(
            key="sgd:squared",
            builder="sgd",
            # Robbins-Monro: lr / (1 + n) reaches the 1/sqrt(n) rate only where
            # lr times the smallest eigenvalue of E[z z'] passes 1/2. The
            # standardised features correlate at 0.5, so that eigenvalue is
            # 0.5 and the rate 2.0. At 0.5 the error fell as n^-0.25 (measured
            # 2026-10-25: rate -0.24 over 500..32,000 rows), as theory says.
            build=_spec(
                "sgd", loss="squared", schedule="inv_scaling", power=1.0, learning_rate=2.0
            ),
            stream=linear,
            oracle=ols,
            classes=lin,
            limit="least squares, at 1/sqrt(n) under inv_scaling with power 1 and a rate past "
            "the smallest curvature's 1/2; at a constant rate a neighbourhood of it that "
            "shrinks with the rate and not with n. clip_gradient (1e3) caps a step at the rate "
            "times 1e3, so a level of L is learned at that pace: at 1e6 under lr / (1 + n) "
            "never (measured 9.8e5 off after 8,000 rows), and class 2 adds a level of 1e3",
            other=OTHER_LINEAR,
            level=1e3,
        ),
        Registration(
            key="sgd:quantile",
            builder="sgd",
            # A sign-valued gradient's curvature is twice the noise's density at
            # the quantile times E[z z']: 2 f(0) = 2 for this Laplace noise
            # (scale 0.5), 1 at the smallest eigenvalue, so a rate of 2 under
            # power 1 passes the 1/2 the 1/sqrt(n) rate needs. The default
            # power, 0.5, gives n^-1/4 (tested beside the constant rate).
            build=_spec(
                "sgd",
                loss="quantile",
                quantile=0.5,
                schedule="inv_scaling",
                power=1.0,
                learning_rate=2.0,
            ),
            stream=_laplace,
            oracle=quantreg(0.5),
            classes=frozenset({1, 2, 5}),
            limit="QuantReg's fit, at 1/sqrt(n) under inv_scaling with power 1; a step moves "
            "the level by at most half the rate a row (the gradient is 1/2 in size), and the "
            "rates sum to lr ln(n), so a level of L is out of reach until lr ln(n) / 2 passes "
            "it (measured 2.87 off a level of 10 at 8,000 rows, where it is 9): class 2 adds "
            "a level of 3",
            other={**OTHER_LINEAR, "level": 5.0},
            level=3.0,
        ),
        Registration(
            key="sgd:epsilon_insensitive",
            builder="sgd",
            # Inside a tube of 1% of the target's spread the loss is the
            # absolute error's: the curvature and the rate are the quantile's.
            build=_spec(
                "sgd",
                loss="epsilon_insensitive",
                schedule="inv_scaling",
                power=1.0,
                learning_rate=2.0,
            ),
            stream=_laplace,
            oracle=insensitive(0.01),
            classes=frozenset({1, 2, 5}),
            limit="the batch epsilon-insensitive fit (scipy's HiGHS on its linear programme), "
            "at 1/sqrt(n) under inv_scaling with power 1; a step moves the level by at most "
            "the rate a row, so a level of L waits for lr ln(n) to pass it: class 2 adds 3",
            other={**OTHER_LINEAR, "level": 5.0},
            level=3.0,
        ),
        Registration(
            key="sgd:poisson",
            builder="sgd",
            # The Poisson curvature is exp(eta) E[z z'], unbounded, so a rate
            # must be small for the first steps: at 2.0 (and at a constant 0.5)
            # the first rows threw the coefficients to about 1e3 (reported,
            # 2026-10-25). At 0.5 the rate is lr times the smallest curvature,
            # about 0.4 (measured -0.40).
            build=_spec(
                "sgd", loss="poisson", schedule="inv_scaling", power=1.0, learning_rate=0.5
            ),
            stream=poisson,
            oracle=glm(sm.families.Poisson()),
            classes=frozenset({1, 2, 5}),
            limit="the Poisson maximum likelihood fit (statsmodels GLM), at n^-0.4: lr / (1 + "
            "n) at 0.5 times the smallest curvature; a count has no level to add",
            other=OTHER_POISSON,
            level=None,
        ),
        Registration(
            key="sgd:logistic",
            builder="sgd",
            # The logistic curvature is p (1 - p) E[z z'], about 0.1 at its
            # smallest here, so the rate that passes 1/2 is about 10. At 0.5
            # the error fell as n^-0.06 (measured).
            build=_spec(
                "sgd", loss="logistic", schedule="inv_scaling", power=1.0, learning_rate=10.0
            ),
            stream=logistic,
            oracle=glm(sm.families.Binomial()),
            classes=frozenset({1, 2, 5}),
            limit="the logistic maximum likelihood fit (statsmodels GLM), at 1/sqrt(n) with a "
            "rate past the smallest curvature's 1/2; a label has no level to add",
            other=OTHER_LOGISTIC,
            level=None,
        ),
        Registration(
            key="pa",
            builder="pa",
            build=_spec("pa"),
            stream=linear,
            oracle=ols,
            classes=frozenset({1, 2, 3, 5}),
            limit="not a batch fit: a row outside the tube is projected onto it, the step "
            "capped by C, so the coefficients wander in a neighbourhood of the truth whose "
            "size is set by C and the noise, and which does not shrink with n; C caps a step "
            "in the target's units, so a level of L takes about L / C rows (1e6 rows at 1e6 "
            "and C = 1: measured 9.9e5 off after 8,000), and class 2 adds a level of 1e3",
            other=OTHER_LINEAR,
            level=1e3,
        ),
        Registration(
            key="ftrl:logistic",
            builder="ftrl",
            build=_spec("ftrl", loss="logistic"),
            stream=logistic,
            oracle=glm(sm.families.Binomial()),
            classes=frozenset({1, 2, 5}),
            limit="the logistic maximum likelihood fit: without decay FTRL's fixed regulariser "
            "fades against the growing sum of losses; a label has no level to add",
            other=OTHER_LOGISTIC,
            level=None,
        ),
        Registration(
            key="ftrl:squared",
            builder="ftrl",
            build=_spec("ftrl", loss="squared"),
            stream=linear,
            oracle=ols,
            classes=frozenset({1, 2, 5}),
            limit="least squares, the fixed regulariser fading against the growing sum of "
            "losses. A coefficient moves about alpha sqrt(n) over n rows of one-signed "
            "gradient, so a level of L needs about (L / alpha)^2 rows (measured 1e6 off at a "
            "level of 1e6 after 8,000), and class 2 adds a level of 5",
            other=OTHER_LINEAR,
            level=5.0,
        ),
        Registration(
            key="holt",
            builder="holt",
            build=_spec("holt"),
            stream=trend,
            oracle=trend_oracle,
            classes=frozenset({1, 2, 5}),
            limit="under a finite half-life a tracking filter, Holt's textbook recursion, with "
            "no bias on a straight line; without decay meant to be the whole history's trend",
            other=OTHER_TREND,
        ),
        Registration(
            key="huber",
            builder="huber",
            build=_spec("huber", solve_every=0.0),
            stream=linear,
            oracle=huber_oracle(),
            classes=frozenset({1, 2, 3, 4, 5}),
            limit="the converged batch Huber fit with the MAD scale on the same weighted rows "
            "(statsmodels' RLM loop): the refit of the first 500 learned rows lands on it, and "
            "each row after it is weighed once, by the fit before it, so the one pass converges "
            "to it on a stationary stream",
            robust=True,
            other=OTHER_LINEAR,
        ),
        Registration(
            key="sgd:huber",
            builder="sgd",
            build=_spec("sgd", loss="huber", schedule="inv_scaling", power=1.0, learning_rate=2.0),
            stream=linear,
            oracle=huber_oracle(),
            classes=frozenset({1, 2, 3, 4, 5}),
            limit="the converged batch Huber fit with the MAD scale, at 1/sqrt(n) under "
            "inv_scaling with power 1: the refit of the first 500 learned rows sets the "
            "coefficients to that fit of those rows, and the steps go on from there",
            robust=True,
            other=OTHER_LINEAR,
            level=1e3,
        ),
    ]


REGISTRY: dict[str, Registration] = {r.key: r for r in _registrations()}

#: Registered, but with no assertions yet, and why. Each name leaves this
#: table when its reason is gone. Empty since task 277 gave ``huber`` and
#: ``sgd(loss="huber")`` the refit of their first rows, and both registered.
PENDING: dict[str, str] = {}

#: Models (or losses) that predict a target and have no registration, and why.
NOT_REGISTERED: dict[str, str] = {}


def losses(builder: str) -> tuple[str, ...]:
    """The values a builder's ``loss=`` takes, read from its signature, or
    ``()`` for a builder without one."""
    p = inspect.signature(getattr(po.spec, builder)).parameters.get("loss")
    if p is None:
        return ()
    hints = typing.get_type_hints(getattr(po.spec, builder))
    return typing.get_args(hints["loss"])


def keys_for(builder: str) -> list[str]:
    """The names a builder must answer for: itself, or one per loss."""
    ls = losses(builder)
    return [f"{builder}:{loss}" for loss in ls] if ls else [builder]


# ---------------------------------------------------------------------------
# Running a model, and reading the tail


def fit(spec: dict, frame: pl.DataFrame, cuts: Iterable[int]) -> dict[int, np.ndarray]:
    """The model's ``coef`` after each row count in ``cuts``, fed as slices
    of one stream (chunk invariance makes that the same as one pass)."""
    bank = po.ModelBank([spec])
    out: dict[int, np.ndarray] = {}
    start = 0
    for cut in sorted(cuts):
        bank.fit_predict(frame.slice(start, cut - start))
        start = cut
        out[cut] = coef(bank)
    return out


def coef(bank: po.ModelBank) -> np.ndarray:
    return np.array(bank.coef("m")["coef"].to_list(), dtype=float)


def oracle_at(reg: Registration, s: Stream, cut: int, half_life: float) -> np.ndarray:
    """The oracle on the stream's first ``cut`` rows, decayed to the last."""
    X, y = s.X[:cut], s.y[:cut]
    return reg.oracle(X, y, decay_weights(cut, half_life))


def gap(a: np.ndarray, b: np.ndarray) -> float:
    """The largest coefficient difference."""
    return float(np.max(np.abs(a - b)))


@dataclass(frozen=True)
class Tail:
    """The tail of a measurement over seeds."""

    worst: float
    worst_seed: int
    q95: float
    median: float
    n: int

    def __str__(self) -> str:
        return (
            f"worst {self.worst:.3g} (seed {self.worst_seed}), 95th {self.q95:.3g}, "
            f"median {self.median:.3g}, over {self.n} seeds"
        )


def tail(by_seed: dict[int, float]) -> Tail:
    """The worst seed, the 95th percentile and the median of a measurement."""
    order = list(by_seed)
    v = np.array([by_seed[s] for s in order])
    i = int(np.argmax(v))
    return Tail(float(v[i]), order[i], float(np.quantile(v, 0.95)), float(np.median(v)), len(v))


def rate(xs: Sequence[float], errors: Sequence[float]) -> float:
    """The slope of ``log(error)`` on ``log(x)`` by least squares: ``-0.5``
    for an error falling as ``1/sqrt(n)``, ``-1`` for ``1/n``."""
    lx, le = np.log(np.asarray(xs, float)), np.log(np.asarray(errors, float))
    return float(np.polyfit(lx, le, 1)[0])


def seeds(count: int, start: int = 0) -> range:
    return range(start, start + count)


# ---------------------------------------------------------------------------
# Scenarios the classes share


def join(first: Stream, then: Stream) -> Stream:
    """One stream, ``first``'s rows then ``then``'s; the truth is ``then``'s."""
    return Stream(pl.concat([first.frame, then.frame]), then.truth)


def is_trend(reg: Registration) -> bool:
    return reg.stream is trend


def break_kwargs(reg: Registration, kind: str, before: Stream) -> dict:
    """The stream keywords for the far side of a break of ``kind``:
    ``level`` moves the intercept (the trend's level) by 10 noise deviations,
    ``slope`` moves the first slope (the trend's slope), ``variance``
    multiplies the noise by 4, and ``regime`` is :attr:`Registration.other`.
    A trend's far side starts where its near side ended, so only the kind
    of break named moves."""
    if is_trend(reg):
        end, slope = before.truth
        coef = np.array([end + slope, slope])
        if kind == "level":
            coef[0] += 10 * NOISE
        elif kind == "slope":
            coef[1] += 0.02
        elif kind == "regime":
            return dict(reg.other)
        return {"coef": coef, **({"scale": 4 * NOISE} if kind == "variance" else {})}
    base = reg.stream(1, 0).truth if reg.level is None else TRUTH
    if kind == "level":
        return {"coef": base + np.array([10 * NOISE if reg.level is not None else 1.0, 0, 0])}
    if kind == "slope":
        return {"coef": base + np.array([0.0, 1.0, 0.0])}
    if kind == "variance":
        return {"scale": 4 * NOISE}
    if kind == "regime":
        return dict(reg.other)
    raise ValueError(kind)


BREAKS = ("level", "slope", "variance", "regime")


def breaks_for(reg: Registration) -> tuple[str, ...]:
    """A count or a label has no noise scale to multiply."""
    return BREAKS if reg.level is not None else ("level", "slope", "regime")


def broken(reg: Registration, kind: str, before: int, after: int, seed: int) -> Stream:
    """``before`` rows of the registration's stream, then ``after`` rows past
    a break of ``kind``, drawn from another seed."""
    near = reg.stream(before, seed)
    far = reg.stream(after, seed + 50_000, **break_kwargs(reg, kind, near))
    return join(near, far)


def from_state(
    reg: Registration, spec: dict, start: pl.DataFrame, frame: pl.DataFrame, cuts: Iterable[int]
) -> dict[int, np.ndarray]:
    """``fit`` from a bank that learned ``start``, was saved and was loaded:
    the start is a state from elsewhere, not the rows of this stream."""
    bank = po.ModelBank([spec])
    bank.fit_predict(start)
    bank = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
    out: dict[int, np.ndarray] = {}
    begin = 0
    for cut in sorted(cuts):
        bank.fit_predict(frame.slice(begin, cut - begin))
        begin = cut
        out[cut] = coef(bank)
    return out


def adverse(reg: Registration, kind: str, n: int, seed: int, rows: int = 500) -> tuple[Stream, int]:
    """The registration's stream with ``kind`` in its first ``rows`` rows,
    and how many rows the adversity touched:

    ``outliers``  one-sided wild targets, +50 (100 noise deviations), on 20%
                  of the first rows, all where ``x0 > 0``
    ``wild``      the first three targets 1e4 off, the rows a prior or a
                  scale is sized from
    ``shift``     the first rows at a level 20 above the rest
    ``t2``        the first rows' noise Student's t with 2 degrees of freedom
    ``cauchy``    the first rows' noise Cauchy
    """
    s = reg.stream(n, seed)
    if kind == "outliers":
        return contaminate(s, 0.2, seed=seed, rows=slice(0, rows))
    y = s.y.copy()
    if kind == "wild":
        y[:3] += 1e4
        touched = 3
    elif kind == "shift":
        y[:rows] += 20.0
        touched = rows
    elif kind in ("t2", "cauchy"):
        rng = np.random.default_rng(20_000 + seed)
        clean = reg.stream(n, seed, scale=0.0).y
        y[:rows] = clean[:rows] + NOISE * noise(rng, rows, kind)
        touched = rows
    else:
        raise ValueError(kind)
    return Stream(s.frame.with_columns(y=pl.Series(y)), s.truth), touched


ADVERSITIES = ("outliers", "wild", "shift", "t2", "cauchy")


# ---------------------------------------------------------------------------
# The measurements each class asserts on. Each returns the tails over seeds
# and prints them, with the evidence that the hard path ran.


@dataclass(frozen=True)
class Path:
    """Tails over seeds at each point of a run: ``to_oracle[x]`` and, where
    the stream has one, ``to_truth[x]``, ``x`` a row count or a number of
    half-lives."""

    xs: list[float]
    to_oracle: dict[float, Tail]
    to_truth: dict[float, Tail]
    #: The same model on the same rows from a clean start, where the
    #: measurement has one: the floor a hard start is held against.
    clean: dict[float, Tail] = field(default_factory=dict)

    def worst(self, which: str = "oracle") -> list[float]:
        d = {"oracle": self.to_oracle, "truth": self.to_truth, "clean": self.clean}[which]
        return [d[x].worst for x in self.xs]

    def show(self, label: str) -> str:
        lines = [label]
        for name, d in (
            ("to oracle", self.to_oracle),
            ("to truth", self.to_truth),
            ("clean", self.clean),
        ):
            if d:
                lines.append(f"  {name}: " + " | ".join(f"{x:g}: {d[x]}" for x in self.xs))
        text = "\n".join(lines)
        print(text)
        return text


def _tails(d: dict[float, dict[int, float]]) -> dict[float, Tail]:
    return {x: tail(v) for x, v in d.items()}


def consistency(reg: Registration, n_seeds: int, cuts: Sequence[int], **kw: object) -> Path:
    """Class 1: no decay, a stationary stream; the gap to the oracle on the
    first ``n`` rows, and to the truth, at each ``n`` in ``cuts``."""
    to_o: dict[float, dict[int, float]] = {c: {} for c in cuts}
    to_t: dict[float, dict[int, float]] = {c: {} for c in cuts}
    for seed in seeds(n_seeds):
        s = reg.stream(cuts[-1], seed)
        path = fit(reg.build(INF, **kw), s.frame, cuts)
        for c in cuts:
            to_o[c][seed] = gap(path[c], oracle_at(reg, s, c, INF))
            to_t[c][seed] = gap(path[c], reg.stream(c, seed).truth)
    return Path(list(cuts), _tails(to_o), _tails(to_t))


def start_from_regime(
    reg: Registration, n_seeds: int, half_life: float, cuts: Sequence[int], start_rows: int = 2000
) -> Path:
    """Class 2, a loaded state from another regime: a bank learns
    ``start_rows`` of :attr:`Registration.other`, is saved and loaded, and
    goes on with this stream. Its gap to the oracle on every row it saw
    (the other regime's decayed by their age), against the same model's gap
    from a fresh start on this stream alone (``clean``)."""
    to_o: dict[float, dict[int, float]] = {c: {} for c in cuts}
    to_t: dict[float, dict[int, float]] = {c: {} for c in cuts}
    clean: dict[float, dict[int, float]] = {c: {} for c in cuts}
    for seed in seeds(n_seeds):
        a = reg.stream(start_rows, seed + 7919, **reg.other)
        b = reg.stream(cuts[-1], seed)
        both = join(a, b)
        spec = reg.build(half_life)
        cont = from_state(reg, spec, a.frame, b.frame, cuts)
        fresh = fit(spec, b.frame, cuts)
        for c in cuts:
            to_o[c][seed] = gap(cont[c], oracle_at(reg, both, start_rows + c, half_life))
            to_t[c][seed] = gap(cont[c], oracle_at(reg, b, c, half_life))
            clean[c][seed] = gap(fresh[c], oracle_at(reg, b, c, half_life))
    return Path(list(cuts), _tails(to_o), _tails(to_t), _tails(clean))


def start_at_level(reg: Registration, n_seeds: int, n: int) -> tuple[Tail, Tail]:
    """Class 2, a large level: the gap to the oracle at :attr:`Registration.level`
    and at level 0, same seeds, no decay."""
    assert reg.level is not None
    out: list[dict[int, float]] = [{}, {}]
    for seed in seeds(n_seeds):
        for i, level in enumerate((reg.level, 0.0)):
            s = reg.stream(n, seed, level=level)
            out[i][seed] = gap(fit(reg.build(INF), s.frame, [n])[n], oracle_at(reg, s, n, INF))
    return tail(out[0]), tail(out[1])


def front_loaded(
    reg: Registration,
    kind: str,
    n_seeds: int,
    half_life: float,
    cuts: Sequence[int],
    rows: int = 500,
) -> tuple[Path, int]:
    """Class 3: :func:`adverse` puts ``kind`` in the first ``rows`` rows; the
    gap to the oracle on the same rows (decayed under a half-life), against
    the same model's gap on the clean stream. Returns the path and the
    fewest rows the adversity touched in any seed (the evidence it ran)."""
    to_o: dict[float, dict[int, float]] = {c: {} for c in cuts}
    to_t: dict[float, dict[int, float]] = {c: {} for c in cuts}
    clean: dict[float, dict[int, float]] = {c: {} for c in cuts}
    touched = []
    for seed in seeds(n_seeds):
        s, k = adverse(reg, kind, cuts[-1], seed, rows)
        touched.append(k)
        c0 = reg.stream(cuts[-1], seed)
        spec = reg.build(half_life)
        hard, easy = fit(spec, s.frame, cuts), fit(spec, c0.frame, cuts)
        for c in cuts:
            to_o[c][seed] = gap(hard[c], oracle_at(reg, s, c, half_life))
            to_t[c][seed] = gap(hard[c], c0.truth if c == cuts[-1] else reg.stream(c, seed).truth)
            clean[c][seed] = gap(easy[c], oracle_at(reg, c0, c, half_life))
    return Path(list(cuts), _tails(to_o), _tails(to_t), _tails(clean)), min(touched)


def reconvergence(
    reg: Registration,
    kind: str,
    n_seeds: int,
    half_life: float,
    before: int,
    half_lives: Sequence[float],
) -> Path:
    """Class 5: ``before`` rows, then a break of ``kind``; the gap to the
    decayed oracle on every row so far, and to the far side's truth, at each
    number of half-lives past the break."""
    cuts = [before + int(round(k * half_life)) for k in half_lives]
    to_o: dict[float, dict[int, float]] = {k: {} for k in half_lives}
    to_t: dict[float, dict[int, float]] = {k: {} for k in half_lives}
    for seed in seeds(n_seeds):
        s = broken(reg, kind, before, cuts[-1] - before, seed)
        path = fit(reg.build(half_life), s.frame, cuts)
        far = reg.stream(
            cuts[-1] - before,
            seed + 50_000,
            **break_kwargs(reg, kind, reg.stream(before, seed)),
        )
        for k, c in zip(half_lives, cuts, strict=True):
            to_o[k][seed] = gap(path[c], oracle_at(reg, s, c, half_life))
            truth = far.truth if not is_trend(reg) else reg_trend_truth(far, c - before)
            to_t[k][seed] = gap(path[c], truth)
    return Path(list(half_lives), _tails(to_o), _tails(to_t))


def reg_trend_truth(far: Stream, rows: int) -> np.ndarray:
    """A trend stream's truth after its first ``rows`` rows."""
    end, slope = far.truth
    n = far.frame.height
    return np.array([end - slope * (n - rows), slope])


def captures(
    reg: Registration, share: float, n_seeds: int, n: int, half_life: float, threshold: float = 0.05
) -> tuple[int, Tail, int]:
    """Class 4: one-sided contamination at ``share`` of every row
    (:func:`contaminate`); how many seeds end with a slope more than
    ``threshold`` from the robust oracle on the same rows, the tail of that
    gap, and the fewest wild rows in any seed."""
    gaps: dict[int, float] = {}
    wild = []
    for seed in seeds(n_seeds):
        s, k = contaminate(reg.stream(n, seed), share, seed=seed)
        wild.append(k)
        b = fit(reg.build(half_life), s.frame, [n])[n]
        o = oracle_at(reg, s, n, half_life)
        gaps[seed] = float(abs(b[1] - o[1]))
    t = tail(gaps)
    return sum(g > threshold for g in gaps.values()), t, min(wild)


#: A prior far from every stream's truth, in the coefficients' units.
FAR_PRIOR = np.array([1e6, -1e6, 1e6])

#: The models that take a ``coef_prior`` that fades: ``rls``, and
#: ``ewridge`` under ``ridge_scale="sum"``, which is the same system.
PRIOR_SPECS: dict[str, Callable[..., dict]] = {
    "rls": _spec("rls", delta=1.0),
    "ewridge": _spec("ewridge", ridge=1.0, ridge_scale="sum", solve_every=0.0),
}


def toward(
    prior: np.ndarray, delta: float
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """RLS's fit from a prior: ``Ridge`` through the origin on the residuals
    from the prior, plus the prior, at ``delta`` decayed as the sums are."""

    def fit(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        return prior + ridge_on_sums(delta)(X, y - X @ prior, v)

    return fit


def start_from_prior(builder: str, n_seeds: int, half_life: float, cuts: Sequence[int]) -> Path:
    """Class 2, an extreme prior: :data:`FAR_PRIOR` as ``coef_prior``. The gap
    to the batch fit from the same prior (``to_oracle``), and to the batch
    fit with no prior (``to_truth``), which the prior's pull must leave."""
    to_o: dict[float, dict[int, float]] = {c: {} for c in cuts}
    to_t: dict[float, dict[int, float]] = {c: {} for c in cuts}
    for seed in seeds(n_seeds):
        s = linear(cuts[-1], seed)
        spec = PRIOR_SPECS[builder](half_life, coef_prior=[FAR_PRIOR.tolist()])
        path = fit(spec, s.frame, cuts)
        for c in cuts:
            v = decay_weights(c, half_life)
            to_o[c][seed] = gap(path[c], toward(FAR_PRIOR, 1.0)(s.X[:c], s.y[:c], v))
            to_t[c][seed] = gap(path[c], ols(s.X[:c], s.y[:c], v))
    return Path(list(cuts), _tails(to_o), _tails(to_t))


# ---------------------------------------------------------------------------
# Rousseeuw and Leroy's robust-regression datasets, as an online stream

#: Rdatasets package, the target column, and the rows Rousseeuw and Leroy
#: (1987) name as outliers, 0-based: stackloss's observations 1, 3, 4 and 21;
#: starsCYG's four giants, 11, 20, 30 and 34, leverage points that pull a
#: residual-based fit; hbk's first 14, ten bad leverage points and four good,
#: which mask one another.
ROBUST_SETS: dict[str, tuple[str, str, list[int]]] = {
    "stackloss": ("datasets", "stack.loss", [0, 2, 3, 20]),
    "starsCYG": ("robustbase", "log.light", [10, 19, 29, 33]),
    "hbk": ("robustbase", "Y", list(range(14))),
}


def check_loss(r: np.ndarray, q: float = 0.5) -> float:
    """The quantile regression objective, ``sum r (q - 1{r < 0})``."""
    return float(np.sum(r * (q - (r < 0))))


def front_loaded_dataset(
    table: pl.DataFrame, target: str, outliers: list[int], cycles: int, seed: int
) -> tuple[pl.DataFrame, list[str]]:
    """The named outliers first, then the whole table ``cycles`` times, each
    pass in a seeded order: a stream whose first rows are its worst, and
    whose batch problem is the table's own, every row repeated. Columns are
    renamed ``f0, f1, ...`` and ``y``."""
    xs = [c for c in table.columns if c != target]
    named = table.rename({c: f"f{i}" for i, c in enumerate(xs)} | {target: "y"})
    rng = np.random.default_rng(seed)
    order = [*outliers] + [int(i) for _ in range(cycles) for i in rng.permutation(table.height)]
    return named[order], [f"f{i}" for i in range(len(xs))]
