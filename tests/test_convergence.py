"""Convergence, judged by the worst seed (docs/PLAN.md task 278, classes 1-5).

The harness is ``tests/convergence.py``: each regression model registers a
known-truth stream, a third-party oracle and the classes that apply to it,
and the measurements there return the tail over seeds. This file holds the
tails to bounds. Every bound below was set from a measurement on this
machine (2026-10-25, an Apple M4 Pro under a load of 100 to 150 from other
work, which moves no number here), written beside it with its headroom; a
bound derived from theory says so.

Each test prints its measurement before it asserts (``pytest -s`` shows it),
with the evidence that the hard path ran: how many rows were made wild, how
far the start was, the gap at the break.

Tiers: the full grids of 50 seeds are extended; one cheap representative
case per class runs in the essentials; the heaviest grid, class 4's sweep
over 200 seeds and longer streams, is in ``tests/test_convergence_soak.py``,
which the weekly leak-check workflow runs.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

import convergence as cv
import polars_online as po
from test_model_registry import REGRESSIONS

TIER = "mixed"

SEEDS = 50
CUTS = [500, 2000, 8000, 32000]

_GRID = pytest.mark.extended(reason="a grid of 50 seeds against a third-party oracle")


# ---------------------------------------------------------------------------
# Registration


def test_every_regression_registers_its_convergence():
    """A model that predicts a target -- and each of its losses, where it
    takes ``loss=`` -- has a registration, or a stated reason why not
    (docs/EXTENDING.md, step 12). The models come from the registry the
    sweeps are held to, the losses from each builder's signature."""
    missing, unreasoned = [], []
    for builder in sorted(REGRESSIONS):
        for key in cv.keys_for(builder):
            if key in cv.REGISTRY:
                assert cv.REGISTRY[key].builder == builder, key
                continue
            reason = cv.PENDING.get(key) or cv.NOT_REGISTERED.get(key)
            if reason is None:
                missing.append(key)
            elif len(reason.split()) < 8:
                unreasoned.append(key)
    for key, reason in {**cv.PENDING, **cv.NOT_REGISTERED}.items():
        print(f"{key}: not registered -- {reason}")
    assert not missing, f"no convergence registration and no reason: {missing}"
    assert not unreasoned, f"a reason of a few words is no reason: {unreasoned}"


def test_nothing_registered_is_also_excused():
    both = set(cv.REGISTRY) & (set(cv.PENDING) | set(cv.NOT_REGISTERED))
    assert not both, both
    known = {k for b in REGRESSIONS for k in cv.keys_for(b)}
    stale = (set(cv.REGISTRY) | set(cv.PENDING) | set(cv.NOT_REGISTERED)) - known
    assert not stale, f"registered under a name no builder answers to: {stale}"


def test_the_pending_reasons_still_hold():
    """``huber`` and ``sgd(loss="huber")`` wait on task 277's warm-up. Its
    arrival is a parameter neither builder had before it: then this fails,
    and their registrations and assertions take the place of the entries in
    ``PENDING``."""
    assert set(cv.PENDING) == {"huber", "sgd:huber"} == set(cv.AFTER_277)
    for builder, before in cv.BEFORE_277.items():
        now = set(inspect.signature(getattr(po.spec, builder)).parameters)
        new = sorted(now - before)
        assert not new, (
            f"po.spec.{builder} gained {new}: if this is task 277's warm-up, move "
            f"convergence.AFTER_277's entries into REGISTRY and assert their classes"
        )


def test_the_waiting_registrations_are_ready():
    """``AFTER_277``'s entries build a ``Registration``, their specs build,
    and their oracle runs, so registering them is a move, not a build."""
    for key, kw in cv.AFTER_277.items():
        reg = cv.Registration(key=key, limit="pending", **kw)
        s = reg.stream(400, 0)
        b = cv.fit(reg.build(cv.INF), s.frame, [400])[400]
        o = cv.oracle_at(reg, s, 400, cv.INF)
        assert b.shape == o.shape == s.truth.shape


@pytest.mark.parametrize("key", sorted(cv.REGISTRY))
def test_each_registration_runs(key):
    """The spec builds, the stream carries the truth's layout, and the model
    and the oracle agree on it, on a short stream (the classes below measure
    how well)."""
    reg = cv.REGISTRY[key]
    assert reg.classes <= set(cv.CLASSES), reg.classes
    s = reg.stream(400, 0)
    coef = cv.fit(reg.build(cv.INF), s.frame, [400])[400]
    oracle = cv.oracle_at(reg, s, 400, cv.INF)
    assert coef.shape == oracle.shape == s.truth.shape
    assert np.all(np.isfinite(coef)) and np.all(np.isfinite(oracle))


# ---------------------------------------------------------------------------
# Class 1: consistency. No decay, a stationary stream with a known truth.

#: Per registration: the worst seed's gap to the oracle allowed at the last
#: cut, and the steepest slope of the worst seed's gap against ``n`` (log-log)
#: allowed over ``CUTS``, ``None`` where the model *is* the batch fit and the
#: gap is rounding at every ``n``. Measured over 50 seeds on 2026-10-25, the
#: worst seed's gap at 32,000 rows and its rate, then the bound:
CONSISTENCY: dict[str, tuple[float, float | None]] = {
    # The same normal equations, solved twice in double precision: the gap is
    # rounding, a few n u kappa with kappa about 3 (the features correlate at
    # 0.5), 1e-11 at 32,000 rows. Measured 5.8e-14 and 9.5e-14.
    "ewridge": (1e-11, None),
    "rls": (1e-11, None),
    # The descent stops when no standardised coefficient moves by its tol,
    # 1e-10; scikit-learn's stops at 1e-12. Measured 4.1e-11.
    "lasso": (1e-9, None),
    # The prior sized from the first rows fades as 1/n (rate -1). Measured
    # 0.0051 (seed 22, whose prior came out small) and -0.93.
    "kalman": (0.01, -0.8),
    # Measured 0.00275 and -0.70: the band narrows with the effective sample.
    "quantile": (0.006, -0.5),
    # Robbins-Monro past the curvature's 1/2: the gap to least squares falls
    # faster than the estimate's own 1/sqrt(n). Measured 0.0038 and -0.78.
    "sgd:squared": (0.008, -0.6),
    # The sign-valued losses at power 1. Measured 0.0341 (seed 26, thrown by
    # its first steps; the 95th percentile 0.0111) and -0.75; 0.00766 and
    # -0.62.
    "sgd:quantile": (0.07, -0.5),
    "sgd:epsilon_insensitive": (0.015, -0.5),
    # lr 0.5 times the smallest curvature, about 0.8, so n^-0.4 (theory).
    # Measured 0.096 (seed 23, thrown far by its second row) and -0.36.
    "sgd:poisson": (0.2, -0.28),
    # Measured 0.317 (seed 23) and -0.51: a large first step (lr 10) throws
    # a seed, which then returns at the theory's rate.
    "sgd:logistic": (0.6, -0.4),
    # FTRL's fixed regulariser fades against the growing sum of losses.
    # Measured 0.028 and -0.77; 0.051 and -0.65.
    "ftrl:squared": (0.05, -0.6),
    "ftrl:logistic": (0.1, -0.5),
}

#: Registrations whose class 1 is a test of its own below.
CONSISTENCY_ELSEWHERE = {"pa": "test_pa_wanders_in_a_neighbourhood_set_by_c", "holt": "test_holt"}


def test_every_consistent_registration_has_a_bound():
    with1 = {k for k, r in cv.REGISTRY.items() if 1 in r.classes}
    assert with1 == set(CONSISTENCY) | set(CONSISTENCY_ELSEWHERE)


@pytest.mark.parametrize("key", [pytest.param(k, marks=_GRID) for k in CONSISTENCY])
def test_consistency(key):
    reg = cv.REGISTRY[key]
    end, steepest = CONSISTENCY[key]
    p = cv.consistency(reg, SEEDS, CUTS)
    p.show(f"{key}: {reg.limit}")
    worst = p.worst()
    assert worst[-1] <= end, f"worst seed {p.to_oracle[CUTS[-1]]}"
    if steepest is not None:
        r = cv.rate(CUTS, worst)
        print(f"  rate of the worst seed's gap: {r:.2f} (bound {steepest})")
        assert r <= steepest
    else:
        assert max(worst) <= end, "a model that is the batch fit is it at every n"
    if key != "lasso":
        # The estimate approaches the truth too, at no worse than n^-0.3
        # (theory -0.5; measured -0.35 for sgd's Poisson loss at the
        # slowest). The lasso's penalty holds it a fixed distance away.
        t = cv.rate(CUTS, p.worst("truth"))
        print(f"  rate of the worst seed's gap to the truth: {t:.2f}")
        assert t <= -0.3


def test_consistency_essential():
    """The representative cell: ``quantile``, whose fit only approaches the
    oracle, on ten seeds. Measured 2026-10-25: worst 0.0159 at 2,000 rows and
    0.00746 at 8,000 over 50 seeds; on these ten, see the print."""
    p = cv.consistency(cv.REGISTRY["quantile"], 10, [2000, 8000])
    p.show("quantile, 10 seeds")
    a, b = p.worst()
    assert b <= 0.012
    assert b <= 0.75 * a, "the worst seed's gap falls with n"


@pytest.mark.extended(reason="a grid of 50 seeds at three settings of C")
def test_pa_wanders_in_a_neighbourhood_set_by_c():
    """``pa`` is not consistent by design: a row outside the tube is
    projected onto it, its step capped by C, so the coefficients wander about
    the truth in a neighbourhood C and the noise set, which does not shrink
    with n. Held: the neighbourhood is bounded, stationary (neither shrinks
    nor grows from 2,000 rows to 32,000) and smaller for a smaller C.
    Measured 2026-10-25 (50 seeds, worst over 2,000..32,000 rows; 95th at
    32,000): C = 1, 0.949 and 0.723; C = 0.1, 0.582 and 0.346; C = 0.01,
    0.176 and 0.129."""
    bound = {1.0: 1.25, 0.1: 0.75, 0.01: 0.25}
    q95 = {}
    for c, b in bound.items():
        p = cv.consistency(cv.REGISTRY["pa"], SEEDS, CUTS, c=c)
        p.show(f"pa, C = {c}")
        assert max(p.worst("truth")[1:]) <= b
        q = [p.to_truth[x].q95 for x in CUTS[1:]]
        assert 0.5 <= q[-1] / q[0] <= 2.0, "the neighbourhood is stationary"
        q95[c] = q[-1]
    assert q95[0.01] < q95[0.1] < q95[1.0]


@pytest.mark.extended(reason="a grid of 50 seeds at two constant rates and a power")
def test_a_constant_rate_settles_in_a_neighbourhood_and_a_power_sets_the_rate():
    """``sgd`` at a constant rate is not consistent by design: the iterate's
    stationary spread is about ``sqrt(lr / 2)`` times the noise per
    standardised coefficient, so four times the rate doubles it, and it does
    not shrink with n. Under ``inv_scaling`` with a power ``p`` below 1 the
    rate is ``n^(-p/2)``: -0.25 at the default 0.5. Measured 2026-10-25 (50
    seeds, the 95th percentile at 2,000 and 32,000 rows): lr 0.01, 0.0796
    and 0.0751; lr 0.04, 0.182 and 0.162 (ratio 2.16 at 32,000);
    ``loss="quantile"`` at power 0.5, the worst seed's rate -0.27."""
    reg = cv.REGISTRY["sgd:squared"]
    q = {}
    for lr in (0.01, 0.04):
        p = cv.consistency(reg, SEEDS, CUTS, schedule="constant", power=None, learning_rate=lr)
        p.show(f"sgd, constant rate {lr}")
        q[lr] = [p.to_oracle[x].q95 for x in CUTS[1:]]
        assert 0.6 <= q[lr][-1] / q[lr][0] <= 1.6, "settled: neither shrinking nor growing"
    ratio = q[0.04][-1] / q[0.01][-1]
    print(f"  spread ratio at four times the rate: {ratio:.2f} (theory 2)")
    assert 1.5 <= ratio <= 2.8
    p = cv.consistency(cv.REGISTRY["sgd:quantile"], SEEDS, CUTS, power=0.5, learning_rate=0.5)
    p.show("sgd quantile, inv_scaling at power 0.5")
    r = cv.rate(CUTS, p.worst())
    print(f"  rate {r:.2f} (theory -0.25)")
    assert -0.4 <= r <= -0.15


@pytest.mark.extended(reason="a grid of 50 seeds against a weighted least-squares trend")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "defect (task 278 (a)): holt without decay gains 1/n on both the level and the "
        "trend, and the trend's early error, which falls only as 1/n, sums into the level "
        "as log n: the worst level gap grows from 7.3 at 500 rows to 11.5 at 32,000 (seed "
        "3, noise 0.5), level-invariant and present with no trend at all"
    ),
)
def test_holt():
    """Without decay ``holt`` is meant to be the whole history's trend, so
    its level should approach the weighted least-squares line's at the last
    row. Held to the theory's rate for a consistent estimate of a level,
    -0.5, loosely: the worst seed's gap must fall at all."""
    reg = cv.REGISTRY["holt"]
    p = cv.consistency(reg, SEEDS, CUTS)
    p.show(f"holt: {reg.limit}")
    r = cv.rate(CUTS, p.worst())
    print(f"  rate of the worst seed's level gap: {r:.2f}")
    assert r <= -0.25


# ---------------------------------------------------------------------------
# Class 2: start independence.

START_CUTS = {200.0: [200, 400, 800, 1600, 2400, 3200], cv.INF: CUTS}

#: Per registration and half-life: the worst seed's gap allowed at the last
#: cut, after a state learned on 2,000 rows of another regime was saved,
#: loaded and run on this stream, to the oracle on every row it saw (the
#: other regime's rows decayed by their age). 3,200 rows are 16 half-lives
#: of 200. Measured over 50 seeds on 2026-10-25 (the worst seed, then the same
#: model's worst from a fresh start), and the bound:
START: dict[str, dict[float, float]] = {
    # The batch fit on every row, the other regime's included: rounding, to
    # 8.6e-14 (ewridge), 2.3e-13 (rls) and 4.5e-11 (lasso).
    "ewridge": {200.0: 1e-11, cv.INF: 1e-11},
    "rls": {200.0: 1e-11, cv.INF: 1e-11},
    "lasso": {200.0: 1e-9, cv.INF: 1e-9},
    # 0.0119 (fresh 0.0105).
    "kalman": {200.0: 0.02},
    # 0.0291 (fresh 0.0277).
    "quantile": {200.0: 0.045},
    # 0.0465 (fresh 0.0472).
    "sgd:squared": {200.0: 0.07},
    # A sign-valued step: half the settled rate, 2 / 290 / 2, a row against a
    # distance of about 5, so 0.618 at 16 half-lives (fresh 0.049).
    "sgd:quantile": {200.0: 0.95},
    # 0.0754 (fresh 0.0799).
    "sgd:epsilon_insensitive": {200.0: 0.11},
    # 0.0545 (fresh 0.0446).
    "sgd:poisson": {200.0: 0.08},
    # 0.17 (fresh 0.17).
    "sgd:logistic": {200.0: 0.25},
    # 0.827 (fresh 0.827).
    "pa": {200.0: 1.25},
    # 0.45 and 0.243 (fresh 0.45 and 0.198).
    "ftrl:logistic": {200.0: 0.68},
    "ftrl:squared": {200.0: 0.37},
    # holt's trend forgets at four half-lives: 16 half-lives are four of its
    # trend's, so the far regime's slope (0.06 off) is 0.004 off still, and the
    # level carries it: 0.532 (fresh 0.061).
    "holt": {200.0: 0.8},
}

#: Without decay a model that is not the batch fit is held to leave the
#: loaded state for this stream's own fit at a rate: the slope of the worst
#: seed's gap to this stream's oracle alone over 500..32,000 rows. Measured
#: 2026-10-25, and the bound:
START_RATE: dict[str, float] = {
    # lr / (1 + n) with n counting the loaded state's 2,000 rows: -1.06.
    "sgd:squared": -0.7,
    # The rates from a state 2,000 rows old sum to lr ln((2000 + t) / 2000),
    # and a sign-valued step moves a coefficient by at most that (half that
    # for the quantile's): 5.7 by 32,000 rows against a distance of about 5,
    # so the gap barely moves. Measured -0.07 and -0.16 (4.73 to 3.50, 4.83
    # to 2.47): the schedule's limit, not a failure to converge.
    "sgd:quantile": -0.04,
    "sgd:epsilon_insensitive": -0.1,
    # -0.28 (1.38 to 0.433), the settled curvature's rate.
    "sgd:poisson": -0.2,
    # -0.61 (4.32 to 0.338).
    "sgd:logistic": -0.45,
    # AdaGrad's sums carry the loaded state's 2,000 rows of gradients, so the
    # steps start small: -0.43 (7.31 to 1.48) and -1.25 (1.96 to 0.053).
    "ftrl:squared": -0.3,
    "ftrl:logistic": -0.8,
}

#: pa is at its neighbourhood (class 1) from the first cut: its gap to this
#: stream's own fit, 0.68 at 32,000 rows, is held to 1.5 times the fresh
#: start's, 0.793.
START_AS_FRESH = {"pa"}

#: Cells that fail on a defect: the reason, with its measurement.
START_DEFECTS: dict[tuple[str, float], str] = {
    ("holt", cv.INF): (
        "defect (task 278 (a)): holt without decay keeps a loaded state's trend at a gain "
        "of 1/n, and the trend's error sums into the level: from another regime's state "
        "the worst level gap grows from 37.6 at 500 rows to 346 at 32,000 (seed 34)"
    ),
    ("quantile", cv.INF): (
        "defect (task 278 (a)): quantile's band is drawn in the root mean square of its "
        "residuals, which without decay keeps the other regime's residuals for good: 0.137 "
        "off the batch fit of every row at 32,000 rows (seed 41), against 0.00275 from a "
        "fresh start"
    ),
    ("kalman", cv.INF): (
        "defect (task 278 (a)): kalman with no process noise and the learned noise R "
        "stays at the other regime's fit: the old fit's residuals inflate R, which divides "
        "the gain, so the fit never moves (gap 45.7 of 46 after 32,000 rows; with obs_var "
        "fixed it reaches the pooled least squares to 1e-4)"
    ),
}


def _start_params() -> list:
    out = []
    for k, r in cv.REGISTRY.items():
        if 2 not in r.classes:
            continue
        for h in START_CUTS:
            marks = [_GRID]
            if (k, h) in START_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=START_DEFECTS[(k, h)]))
            out.append(pytest.param(k, h, marks=marks, id=f"{k}-{h:g}"))
    return out


def test_every_start_cell_has_a_bound_or_a_defect():
    cells = {(k, h) for k, r in cv.REGISTRY.items() if 2 in r.classes for h in START_CUTS}
    bounded = (
        {(k, h) for k, d in START.items() for h in d}
        | {(k, cv.INF) for k in START_RATE}
        | {(k, cv.INF) for k in START_AS_FRESH}
        | set(START_DEFECTS)
    )
    assert cells == bounded, cells ^ bounded


@pytest.mark.parametrize("key, half_life", _start_params())
def test_start_from_another_regime(key, half_life):
    reg = cv.REGISTRY[key]
    cuts = START_CUTS[half_life]
    p = cv.start_from_regime(reg, SEEDS, half_life, cuts)
    p.show(f"{key} from another regime's loaded state, half-life {half_life:g}")
    # The evidence: at the first cut the fit is still far from this stream's own.
    assert p.to_truth[cuts[0]].median > 0.3, "the loaded state was a long way off"
    if np.isinf(half_life) and key in START_AS_FRESH:
        assert p.to_truth[cuts[-1]].worst <= 1.5 * p.clean[cuts[-1]].worst
    elif np.isinf(half_life) and key in START_RATE:
        # Not the batch fit, which keeps the other regime's rows for good:
        # what is checked is that the fit leaves the loaded state for this
        # stream's own, at the rate given, from the gap to this stream's
        # oracle alone.
        r = cv.rate(cuts, p.worst("truth"))
        print(f"  rate of the worst seed's gap to this stream's own fit: {r:.2f}")
        assert r <= START_RATE[key]
    else:
        assert p.to_oracle[cuts[-1]].worst <= START[key][half_life]


def test_start_essential():
    """The representative cell: ``sgd``'s squared loss from another regime's
    state, at a half-life of 200, ten seeds. Measured 2026-10-25 (50 seeds):
    the worst gap 11.8 at one half-life, 0.0275 at 12 and 0.033 at 16, its
    fresh start's 0.033."""
    p = cv.start_from_regime(cv.REGISTRY["sgd:squared"], 10, 200.0, [200, 3200])
    p.show("sgd squared from another regime, 10 seeds")
    assert p.to_truth[200].median > 3.0
    assert p.to_oracle[3200].worst <= 0.06


#: A large level must cost no more than rounding: the worst seed's gap to the
#: oracle at the registration's level is held to three times its gap at level
#: 0, plus 1e-13 of the level, the rounding of a fit that carries it (measured
#: 2026-10-25: rls at 1e6, 1.6e-8 against 3e-14 at level 0, about 130 ulps of
#: the intercept; ewridge 1.2e-10, one ulp). 8,000 rows, no decay, 50 seeds.
_SGD_LEVEL = (
    "defect (task 278 (a)): sgd does not centre its target, so at a level L its first "
    "steps move every coefficient by about the rate times L, and under inv_scaling the "
    "slopes come back only at the schedule's pace: {}"
)
LEVEL_DEFECTS: dict[str, str] = {
    "sgd:squared": _SGD_LEVEL.format(
        "2.49 off least squares at 8,000 rows at a level of 1e3 (seed 26), against 0.0096 "
        "at level 0"
    ),
    "sgd:quantile": _SGD_LEVEL.format(
        "0.206 off QuantReg at 8,000 rows at a level of 3 (seed 32), against 0.042 at level 0"
    ),
    "kalman": (
        "defect (task 278 (a)): kalman's learned noise R keeps the level's first residuals "
        "(the intercept starts at 0) when nothing decays, and the rows are then weighed "
        "unevenly against the prior: at a level of 1e6 the worst seed is 0.402 off least "
        "squares at 8,000 rows (seed 40) against 0.0049 at level 0; it converges, as 1/n "
        "(0.0024 at 128,000 rows over 20 seeds)"
    ),
}


def _level_params() -> list:
    out = []
    for k, r in cv.REGISTRY.items():
        if 2 not in r.classes or r.level is None:
            continue
        marks = [_GRID]
        if k in LEVEL_DEFECTS:
            marks.append(pytest.mark.xfail(strict=True, reason=LEVEL_DEFECTS[k]))
        out.append(pytest.param(k, marks=marks))
    return out


@pytest.mark.parametrize("key", _level_params())
def test_start_at_a_large_level(key):
    reg = cv.REGISTRY[key]
    at, base = cv.start_at_level(reg, SEEDS, 8000)
    print(f"{key}: level {reg.level:g}: {at}\n  level 0: {base}\n  {reg.limit}")
    assert at.worst <= 3 * base.worst + 1e-13 * abs(reg.level)


@pytest.mark.extended(reason="a grid of seeds, with and without decay, for two models")
@pytest.mark.parametrize("builder", sorted(cv.PRIOR_SPECS))
def test_start_from_an_extreme_prior(builder):
    """A ``coef_prior`` of 1e6 against coefficients of about 1. The fit is
    the batch fit from that prior at every row (``to oracle``); the prior's
    pull, ``delta`` against the rows' weight, fades as 1/n without decay and
    as ``2^-k`` after ``k`` half-lives under one (``to truth`` is the gap to
    the batch fit with no prior). Measured 2026-10-25 over 20 seeds, the same
    for ``rls`` and ``ewridge(ridge_scale="sum")``: from the prior's batch
    fit at most 3.1e-9, a relative 3e-15 of the prior; without decay 4,750 at
    500 rows to 64 at 32,000, rate -1.00; at a half-life of 200, 7.8 at 10
    half-lives, 7.8e-6 at 30 and 7.5e-9 at 40."""
    for half_life, cuts in ((cv.INF, CUTS), (200.0, [200, 2000, 6000, 8000])):
        p = cv.start_from_prior(builder, 20, half_life, cuts)
        p.show(f"{builder} from a prior of 1e6, half-life {half_life:g}")
        assert max(p.worst()) <= 1e-8, "the batch fit from the prior, to rounding"
        if np.isinf(half_life):
            r = cv.rate(cuts, p.worst("truth"))
            assert -1.1 <= r <= -0.95, r
        else:
            assert p.to_truth[200].median > 1e3, "the prior held the first rows"
            assert p.to_truth[8000].worst <= 1e-7


# ---------------------------------------------------------------------------
# Class 3: adversity in the first rows.


@pytest.mark.extended(reason="downloads three public datasets and fits three orders of each")
@pytest.mark.parametrize("name", sorted(cv.ROBUST_SETS))
def test_quantile_on_the_classic_robust_datasets_outliers_first(name):
    """Rousseeuw and Leroy's datasets as a stream whose first rows are the
    named outliers, then the table 300 times in seeded orders: ``quantile``
    must reach QuantReg's objective on those rows. The oracle is taken as
    given; what is checked is that the online fit gets there from a start
    that is all outliers.

    The bound is on the check loss, not the coefficients: on a small table
    repeated, the median regression's minimiser can be a flat set, and the
    model's smoothed loss lands somewhere on it or near it. ``quantile``
    smooths the check loss over a band of ``quantile_eps = 0.2`` residual
    deviations, so it converges to the smoothed fit: measured 2026-10-25 the
    excess was 1.52%, 0.11% and 0.44% of QuantReg's loss on stackloss,
    starsCYG and hbk, the same from 30 passes to 3,000 (so a floor, not a
    rate). The bound is 1.5 times that.

    starsCYG's four giants are leverage points: a fit that reads only
    residuals is pulled by them, and the median regression's slope on all
    47 stars is -0.69 where the 43 others give 2.50. The stated expectation:
    ``quantile`` is pulled the same way, its slope below 0."""
    import data

    package, target, outliers = cv.ROBUST_SETS[name]
    table = data.rdataset_or_skip(package, name)
    bound = {"stackloss": 0.023, "starsCYG": 0.0017, "hbk": 0.0066}[name]
    for seed in range(3):
        stream, fs = cv.front_loaded_dataset(table, target, outliers, 300, seed)
        spec = po.spec.quantile(
            "m", targets=["y"], features=fs, half_life=cv.INF, quantile=0.5, solve_every=0.0
        )
        coef = cv.fit(spec, stream, [stream.height])[stream.height]
        X = np.column_stack([np.ones(stream.height), stream.select(fs).to_numpy()])
        y = stream["y"].to_numpy()
        oracle = cv.quantreg(0.5)(X, y, np.ones(len(y)))
        best = cv.check_loss(y - X @ oracle)
        excess = (cv.check_loss(y - X @ coef) - best) / best
        print(
            f"{name}, order {seed}: coef {np.round(coef, 3)}, "
            f"QuantReg {np.round(oracle, 3)}, excess {excess:.2e}"
        )
        assert excess <= bound
        if name == "starsCYG":
            assert coef[1] < 0, "pulled by the giants, as every residual-based fit is"


ADVERSE_CUTS = [2000, 8000]
ADVERSE_HALF_LIVES = (cv.INF, 500.0)


#: Per registration, adversity and half-life: the worst seed's gap to the
#: oracle on the same rows allowed at 8,000 rows (16 half-lives of 500).
#: Measured over 50 seeds on 2026-10-25, the worst seed beside the same
#: model's worst on the clean stream, and the bound:
def _every(bound: float) -> dict[tuple[str, float], float]:
    return {(kind, h): bound for kind in cv.ADVERSITIES for h in ADVERSE_HALF_LIVES}


INF = cv.INF
ADVERSE: dict[str, dict[tuple[str, float], float]] = {
    # The batch fit on the same rows, wild ones and all: rounding. Measured at
    # most 1.1e-13 (ewridge, three targets 1e4 off), 4.2e-11 (lasso's tol).
    "ewridge": _every(1e-11),
    "rls": _every(1e-11),
    "lasso": _every(1e-9),
    # Under a half-life kalman forgets through its drift (coef_half_life =
    # 500). Measured 0.0060, 0.0118, 0.0049, 0.0072 and 0.0157, beside
    # 0.0072 from a clean start. Without decay, t2 noise alone: 0.049
    # (clean 0.0049); the rest are defects (below).
    "kalman": {
        ("outliers", 500.0): 0.01,
        ("wild", 500.0): 0.018,
        ("shift", 500.0): 0.008,
        ("t2", 500.0): 0.011,
        ("cauchy", 500.0): 0.024,
        ("t2", INF): 0.075,
    },
    # A robust fit: without decay it is held to three times its clean worst
    # (0.0079): t2 noise measured 0.0076. Under a half-life of 500, measured
    # 0.0131, 0.0367 (three targets 1e4 off, forgotten as 2^-16), 0.0130,
    # 0.0129 and 0.0258, beside 0.0129 clean.
    "quantile": {
        ("t2", INF): 0.024,
        ("outliers", 500.0): 0.02,
        ("wild", 500.0): 0.055,
        ("shift", 500.0): 0.02,
        ("t2", 500.0): 0.02,
        ("cauchy", 500.0): 0.04,
    },
    # A gradient step weighs the first rows as least squares does only
    # roughly, so without decay the gap to the same rows' least squares falls
    # at a rate (held below) from a level the first steps set: measured 0.734
    # (outliers), 18.1 (wild: a clipped first step of lr times 1e3), 1.18,
    # 0.0236 and 20.0 at 8,000 rows, each falling as 1/n from 2,000. Under a
    # half-life, 0.0199 to 0.0206 for every kind, beside 0.0200 clean.
    "sgd:squared": {
        ("outliers", INF): 1.1,
        ("wild", INF): 27.0,
        ("shift", INF): 1.8,
        ("t2", INF): 0.036,
        ("cauchy", INF): 30.0,
        **{(kind, 500.0): 0.03 for kind in cv.ADVERSITIES},
    },
    # pa wanders in its neighbourhood (class 1): under a half-life every kind
    # measured 0.880 to 0.890, its clean 0.889; without decay 1.58, 5.59,
    # 2.09, 0.889 and 21.3, falling from 2,000 rows as fast as 1/n.
    "pa": {
        ("outliers", INF): 2.4,
        ("wild", INF): 8.4,
        ("shift", INF): 3.2,
        ("t2", INF): 1.35,
        ("cauchy", INF): 32.0,
        **{(kind, 500.0): 1.35 for kind in cv.ADVERSITIES},
    },
}

_KALMAN_CAPTURE = (
    "defect (task 278 (a)): kalman with no process noise and the learned noise R is "
    "captured by a burst of large residuals in its first rows: R grows with them and "
    "divides the gain, and the fit stays far from least squares on the same rows "
    "(measured {}); with obs_var fixed it is least squares to rounding"
)
_QUANTILE_RMS = (
    "defect (task 278 (a)): quantile's band is drawn in the root mean square of its "
    "residuals, which without decay keeps the first rows' wild residuals for good, so the "
    "band stays wide, the rows inside it are least-squares rows, and a wild row's step "
    "grows with the scale: measured {} at 8,000 rows, three times its clean worst 0.024"
)
ADVERSE_DEFECTS: dict[tuple[str, str, float], str] = {
    ("kalman", "outliers", INF): _KALMAN_CAPTURE.format("19.3 at 8,000 rows, seed 31"),
    ("kalman", "wild", INF): _KALMAN_CAPTURE.format("5.76, seed 22"),
    ("kalman", "shift", INF): _KALMAN_CAPTURE.format("17.8 and growing, seed 22"),
    ("kalman", "cauchy", INF): _KALMAN_CAPTURE.format("658 and growing, seed 39"),
    ("quantile", "outliers", INF): _QUANTILE_RMS.format("0.165 (seed 1)"),
    ("quantile", "wild", INF): _QUANTILE_RMS.format("4.08 (seed 47)"),
    ("quantile", "shift", INF): _QUANTILE_RMS.format("0.097 (seed 18)"),
    ("quantile", "cauchy", INF): _QUANTILE_RMS.format("0.143 (seed 36)"),
}


def _adverse_params() -> list:
    out = []
    for k, r in cv.REGISTRY.items():
        if 3 not in r.classes:
            continue
        for kind in cv.ADVERSITIES:
            for h in ADVERSE_HALF_LIVES:
                marks = [_GRID]
                if (k, kind, h) in ADVERSE_DEFECTS:
                    reason = ADVERSE_DEFECTS[(k, kind, h)]
                    marks.append(pytest.mark.xfail(strict=True, reason=reason))
                out.append(pytest.param(k, kind, h, marks=marks, id=f"{k}-{kind}-{h:g}"))
    return out


def test_every_adverse_cell_has_a_bound_or_a_defect():
    cells = {
        (k, kind, h)
        for k, r in cv.REGISTRY.items()
        if 3 in r.classes
        for kind in cv.ADVERSITIES
        for h in ADVERSE_HALF_LIVES
    }
    bounded = {(k, kind, h) for k, d in ADVERSE.items() for kind, h in d} | set(ADVERSE_DEFECTS)
    assert cells == bounded, cells ^ bounded


@pytest.mark.parametrize("key, kind, half_life", _adverse_params())
def test_front_loaded_adversity(key, kind, half_life):
    reg = cv.REGISTRY[key]
    p, touched = cv.front_loaded(reg, kind, SEEDS, half_life, ADVERSE_CUTS)
    p.show(f"{key}, {kind} in the first 500 rows, half-life {half_life:g}: {touched}+ rows touched")
    assert touched >= 3, "the adversity is there"
    first, last = (p.to_oracle[c].worst for c in ADVERSE_CUTS)
    assert last <= ADVERSE[key][(kind, half_life)]
    clean = p.clean[ADVERSE_CUTS[-1]].worst
    if np.isinf(half_life) and key in ("sgd:squared", "pa") and last > 2 * clean:
        # Not the batch fit, so what it keeps of the first rows must wash
        # out: at least as fast as 1/sqrt(n).
        r = cv.rate(ADVERSE_CUTS, [first, last])
        print(f"  rate from 2,000 to 8,000 rows: {r:.2f}")
        assert r <= -0.5


def test_adversity_essential():
    """The representative cell: ``quantile`` with wild targets in its first
    500 rows, at a half-life of 500, ten seeds. Measured 2026-10-25 over 50
    seeds: 0.0131 at 8,000 rows, beside 0.0129 clean."""
    p, touched = cv.front_loaded(cv.REGISTRY["quantile"], "outliers", 10, 500.0, [8000])
    p.show(f"quantile, outliers first, 10 seeds, {touched}+ rows wild")
    assert touched >= 50
    assert p.to_oracle[8000].worst <= ADVERSE["quantile"][("outliers", 500.0)]


# ---------------------------------------------------------------------------
# Class 5: re-convergence after a break.

BREAK_HALF_LIFE = 200.0
#: 4,000 rows, 20 half-lives, settle the fit before the break.
BREAK_AT = 4000
BREAK_KS = [0, 1, 2, 4, 6, 8, 12]


#: Per registration and kind of break: the number of half-lives past the
#: break, and the worst seed's gap to the decayed oracle allowed there.
#: Measured over 50 seeds on 2026-10-25, the worst seed at each number of
#: half-lives, and the bound:
def _exact(bound: float) -> dict[str, tuple[float, float]]:
    return {kind: (1, bound) for kind in cv.BREAKS}


REBREAK: dict[str, dict[str, tuple[float, float]]] = {
    # The decayed batch fit at every row, before the break and after: rounding.
    # Measured at most 2.4e-13 (ewridge), 2.2e-13 (rls), 4.4e-11 (lasso).
    "ewridge": _exact(1e-11),
    "rls": _exact(1e-11),
    "lasso": _exact(1e-9),
    # Drift at coef_half_life = 200 matches EW least squares' gain in steady
    # state only. Measured 0.0206 at 6, 0.0196 at 8, 0.0474 and 0.0246 at 12.
    "kalman": {
        "level": (6, 0.04),
        "slope": (8, 0.03),
        "variance": (12, 0.075),
        "regime": (12, 0.04),
    },
    # Measured 0.0249 and 0.0247 at 6, 0.0674 at 12, 0.035 at 8.
    "quantile": {
        "level": (6, 0.04),
        "slope": (6, 0.04),
        "variance": (12, 0.1),
        "regime": (8, 0.055),
    },
    # Under a half-life inv_scaling's weight saturates at h / ln 2 = 289 rows,
    # so the rate settles at lr / 290: a constant-rate tracker whose noise is
    # the variance break's 0.159. Measured 0.0384, 0.0333 (at 8), 0.159 and
    # 0.0378 at 12.
    "sgd:squared": {
        "level": (12, 0.06),
        "slope": (8, 0.05),
        "variance": (12, 0.24),
        "regime": (12, 0.06),
    },
    # A sign-valued step moves the level half the settled rate, 2 / 290 / 2, a
    # row, so the regime's distance of about 5 takes some 1,500 rows at the
    # very least: 1.53 at 12 half-lives. The rest measured 0.0465, 0.0429 and
    # 0.202 at 12.
    "sgd:quantile": {
        "level": (12, 0.07),
        "slope": (12, 0.065),
        "variance": (12, 0.3),
        "regime": (12, 2.3),
    },
    # Measured 0.069 and 0.0732 at 8, 0.141 and 0.0826 at 12.
    "sgd:epsilon_insensitive": {
        "level": (8, 0.1),
        "slope": (8, 0.11),
        "variance": (12, 0.21),
        "regime": (12, 0.125),
    },
    # The settled rate 0.5 / 290 against a curvature of exp(eta) E[z z'],
    # about 0.1 on the far regime's log-rate of -1: a memory of some 30
    # half-lives, so 0.51 at 12. The rest measured 0.0332 at 8 and 0.108 at 6.
    "sgd:poisson": {
        "level": (8, 0.05),
        "slope": (6, 0.16),
        "regime": (12, 0.77),
    },
    # A large settled rate (10 / 290) on a weak curvature: a noisy tracker.
    # Measured 0.19, 0.221 and 0.334 at 12.
    "sgd:logistic": {
        "level": (12, 0.29),
        "slope": (12, 0.33),
        "regime": (12, 0.5),
    },
    # pa wanders in its neighbourhood (class 1), which the variance break
    # quadruples: measured 0.88, 0.87, 2.13 and 0.862 at 12.
    "pa": {
        "level": (12, 1.35),
        "slope": (12, 1.35),
        "variance": (12, 3.2),
        "regime": (12, 1.3),
    },
    # ftrl is not the decayed batch fit: it minimises linearised losses at
    # AdaGrad's per-coordinate rate against a fixed regulariser, a noisy
    # tracker. Measured 0.314, 0.244 and 0.203 at 12; 0.56 for the logistic
    # level break.
    "ftrl:squared": {
        "level": (12, 0.47),
        "slope": (12, 0.37),
        "variance": (12, 0.3),
    },
    "ftrl:logistic": {"level": (12, 0.84)},
    # holt's trend forgets at trend_half_life, four times the half-life, so
    # 12 half-lives are three of its trend's: measured 0.267, 0.744, 0.127 and
    # 4.88 (a level 100 off and a slope 0.06 off at the break) at 12.
    "holt": {
        "level": (12, 0.4),
        "slope": (12, 1.12),
        "variance": (12, 0.19),
        "regime": (12, 7.3),
    },
}

_FTRL_SLOW = (
    "defect (task 278 (a)): ftrl under a half-life follows a break at AdaGrad's pace, "
    "and the break's own large gradients inflate the sum that divides every step, so the "
    "fit lags the decayed batch fit for many half-lives: {}"
)
REBREAK_DEFECTS: dict[tuple[str, str], str] = {
    ("ftrl:squared", "regime"): _FTRL_SLOW.format(
        "20.2 off one half-life past a change of regime, 34.4 at four, 19.9 at twelve (seed 44)"
    ),
    ("ftrl:logistic", "slope"): _FTRL_SLOW.format(
        "after a slope break the gap grows from 0.53 at one half-life to 1.0 at twelve "
        "(seed 5), the slope still 0.93 from the new truth"
    ),
    ("ftrl:logistic", "regime"): _FTRL_SLOW.format(
        "after a change of regime the gap grows from 0.33 to 1.16 at twelve half-lives (seed 39)"
    ),
}


def _break_params() -> list:
    out = []
    for k, r in cv.REGISTRY.items():
        if 5 not in r.classes:
            continue
        for kind in cv.breaks_for(r):
            marks = [_GRID]
            if (k, kind) in REBREAK_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=REBREAK_DEFECTS[(k, kind)]))
            out.append(pytest.param(k, kind, marks=marks, id=f"{k}-{kind}"))
    return out


def test_every_break_cell_has_a_bound_or_a_defect():
    cells = {
        (k, kind) for k, r in cv.REGISTRY.items() if 5 in r.classes for kind in cv.breaks_for(r)
    }
    bounded = {(k, kind) for k, d in REBREAK.items() for kind in d} | set(REBREAK_DEFECTS)
    assert cells == bounded, cells ^ bounded


@pytest.mark.parametrize("key, kind", _break_params())
def test_reconvergence_after_a_break(key, kind):
    reg = cv.REGISTRY[key]
    p = cv.reconvergence(reg, kind, SEEDS, BREAK_HALF_LIFE, BREAK_AT, BREAK_KS)
    p.show(f"{key}, a break in {kind} after {BREAK_AT} rows, half-life {BREAK_HALF_LIFE:g}")
    k, bound = REBREAK[key][kind]
    if kind != "variance":
        # The evidence: at some point past the break (at 0 half-lives, the
        # fit before it) the fit was far from the new truth, farther than at
        # the end. A trend's slope break shows later, as the level drifts.
        most = max(p.to_truth[k].median for k in BREAK_KS)
        assert most > 1.5 * p.to_truth[BREAK_KS[-1]].median, "the break was taken"
    assert p.to_oracle[k].worst <= bound


def test_reconvergence_essential():
    """The representative cell: ``quantile`` after a break in level, ten
    seeds. Measured 2026-10-25 over 50 seeds: the worst gap to the decayed
    oracle 1.26 one half-life past the break, 0.0389 at four and 0.0173 at
    twelve."""
    p = cv.reconvergence(cv.REGISTRY["quantile"], "level", 10, BREAK_HALF_LIFE, BREAK_AT, [1, 6])
    p.show("quantile after a break in level, 10 seeds")
    assert p.to_truth[1].median > 1.0
    assert p.to_oracle[6].worst <= 0.05


# ---------------------------------------------------------------------------
# Class 4: contamination short of the breakdown point.
#
# One-sided wild targets (+50, a hundred noise deviations) on a share of all
# rows, every one where x0 > 0 (cv.contaminate): a share s is 2 s of the rows
# on that side, where the conditional median breaks at s = 25%. A seed is
# captured when its slope ends more than 0.05 from QuantReg's on the same
# rows (task 277's threshold). The full sweep, 200 seeds at 100,000 rows, is
# in tests/test_convergence_soak.py.

BREAKDOWN_SHARES = (0.05, 0.10, 0.15, 0.20)
BREAKDOWN_HALF_LIVES = (cv.INF, 5000.0)
#: Measured 2026-10-25, 50 seeds at 20,000 rows: every seed captured at
#: every share, without decay and at a half-life of 5,000 alike, the worst
#: slope gap below; least squares, for scale, is pulled by about 2 at 5%.
_MEASURED = {
    (0.05, cv.INF): "0.098 (seed 39)",
    (0.10, cv.INF): "0.285 (seed 41)",
    (0.15, cv.INF): "0.567 (seed 47)",
    (0.20, cv.INF): "0.980 (seed 31)",
    (0.05, 5000.0): "0.094 (seed 39)",
    (0.10, 5000.0): "0.279 (seed 29)",
    (0.15, 5000.0): "0.573 (seed 45)",
    (0.20, 5000.0): "0.987 (seed 8)",
}
BREAKDOWN_DEFECTS: dict[tuple[float, float], str] = {
    cell: (
        "defect (task 278 (a)): quantile draws its band in the root mean square of its "
        "residuals, which the wild rows inflate (about 11 at 5% against a noise of 0.5), so "
        "the band of 0.2 of it holds every clean row as a least-squares row and gives each "
        "wild row a pull of the band's width: all 50 seeds captured, the worst slope "
        f"{worst} from QuantReg's, the same at 80,000 rows as at 20,000"
    )
    for cell, worst in _MEASURED.items()
}


def _breakdown_params() -> list:
    out = []
    for share in BREAKDOWN_SHARES:
        for h in BREAKDOWN_HALF_LIVES:
            marks = [_GRID]
            if (share, h) in BREAKDOWN_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=BREAKDOWN_DEFECTS[(share, h)]))
            out.append(pytest.param(share, h, marks=marks, id=f"{share:.0%}-{h:g}"))
    return out


@pytest.mark.parametrize("share, half_life", _breakdown_params())
def test_quantile_is_never_captured_short_of_breakdown(share, half_life):
    captured, t, wild = cv.captures(cv.REGISTRY["quantile"], share, SEEDS, 20_000, half_life)
    print(f"quantile, {share:.0%} wild, half-life {half_life:g}: {captured}/{SEEDS}; {t}")
    assert wild > 0.5 * share * 20_000, "the contamination is there"
    assert captured == 0, t


def test_breakdown_essential():
    """The representative cell, where the class holds today: a band ten times
    narrower than the default (``quantile_eps = 0.02``) at 5% wild rows, ten
    seeds. Measured 2026-10-25 over 50 seeds: none captured, the worst slope
    gap 0.026 (seed 39); at 10% 12 of 50 were, and at 15% all, since the
    root mean square scale still widens the band (the defect above)."""
    narrow = dataclasses.replace(
        cv.REGISTRY["quantile"],
        build=cv._spec("quantile", quantile=0.5, solve_every=0.0, quantile_eps=0.02),
    )
    captured, t, wild = cv.captures(narrow, 0.05, 10, 20_000, cv.INF)
    print(f"quantile, quantile_eps 0.02, 5% wild: {captured}/10; {t}")
    assert wild > 500
    assert captured == 0, t
