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

Tiers: the full grids of 50 seeds against a third-party oracle carry the
``oracle_grid`` mark, which no local run takes (``pyproject.toml``'s
``addopts`` and ``scripts/gate.sh`` leave it out) and
``.github/workflows/oracles.yml`` runs on every push, sharded across jobs;
``pytest -m oracle_grid tests/test_convergence.py`` runs them here. One cheap
representative case per class runs in the essentials. The heaviest grid,
class 4's sweep over 200 seeds and longer streams, is in
``tests/test_convergence_soak.py``, which the weekly leak-check workflow
runs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import convergence as cv
import ledger
import polars_online as po
from test_model_registry import REGRESSIONS

TIER = "mixed"

SEEDS = 50
CUTS = [500, 2000, 8000, 32000]

_GRID = pytest.mark.oracle_grid(reason="a grid of 50 seeds against a third-party oracle")


# ---------------------------------------------------------------------------
# The oracle ledger (tests/ledger.py; docs/TESTING.md, "The oracle ledger").
#
# Each registration is a group named by its key (``sgd:huber`` is
# ``convergence.sgd_huber``), so a change to one model's source re-runs that
# model's grids alone. Each grid records what it bounds, one value a seed:
# the gap at the cut, or the half-lives, its assertion reads.


def slug(key: str) -> str:
    """A registration's key as a ledger group's name."""
    return key.replace(":", "_")


def tag(x: float) -> str:
    """A number in a case's name: ``200``, ``inf``, ``0_05``."""
    return "inf" if np.isinf(x) else f"{x:g}".replace(".", "_")


def _record(key: str, case: str, values, **kw) -> None:
    """Record a case of ``key``'s group, with its round-off floor if it has one."""
    floor = ROUND_OFF.get(f"{slug(key)}.{case}", 0.0)
    ledger.record(slug(key), case, values, floor=floor, **kw)


#: The rows of each scenario's stream the guard runs.
GUARD_ROWS = 2_000


def _fields(spec: dict, frame) -> object:
    """Every field ``fit_predict`` emits for a spec named ``m`` on a frame."""
    return po.ModelBank([spec]).fit_predict(frame).unnest("m")


def guard_outputs(key: str) -> dict:
    """This library's outputs on the first seed of each scenario the grids
    run a registration on, at 2,000 rows, every field: the grids' 50 seeds
    at up to 32,000 rows take minutes and their oracles longer, while these
    take milliseconds, and a change to the model's arithmetic moves the
    first seed as it moves the rest."""
    reg = cv.REGISTRY[key]
    n = GUARD_ROWS
    s = reg.stream(n, 0)
    out: dict = {"no_decay": _fields(reg.build(cv.INF), s.frame)}
    out["decayed"] = _fields(reg.build(200.0), s.frame)
    if 2 in reg.classes:
        other = reg.stream(500, 7919, **reg.other).frame
        out["from_state"] = cv.from_state(reg, reg.build(200.0), other, s.frame, [500, n])
        if reg.level is not None:
            out["level"] = _fields(reg.build(cv.INF), reg.stream(n, 0, level=reg.level).frame)
    if 3 in reg.classes:
        for kind in cv.ADVERSITIES:
            out[kind] = _fields(reg.build(cv.INF), cv.adverse(reg, kind, n, 0)[0].frame)
    if 4 in reg.classes:
        wild = cv.contaminate(s, 0.2, seed=0)[0].frame
        out["contaminated"] = _fields(reg.build(5000.0), wild)
    if 5 in reg.classes:
        for kind in cv.breaks_for(reg):
            out[f"break_{kind}"] = _fields(
                reg.build(BREAK_HALF_LIFE), cv.broken(reg, kind, n // 2, n // 2, 0).frame
            )
    return out


def guard_prior(builder: str) -> dict:
    """``guard_outputs`` and the extreme prior's fits, with and without decay."""
    frame = cv.linear(GUARD_ROWS, 0).frame
    prior = {"coef_prior": [cv.FAR_PRIOR.tolist()]}
    out = guard_outputs(builder)
    for h in (cv.INF, 200.0):
        out[f"prior_{tag(h)}"] = _fields(cv.PRIOR_SPECS[builder](h, **prior), frame)
    return out


def guard_kalman() -> dict:
    """``guard_outputs`` and K1's spec at its level."""
    out = guard_outputs("kalman")
    out["k1"] = _fields(
        po.spec.kalman("m", **K1_SPEC), cv.linear(GUARD_ROWS, 0, level=K1_LEVEL).frame
    )
    return out


def guard_pa() -> dict:
    """``guard_outputs`` and the two smaller settings of C."""
    out = guard_outputs("pa")
    s = cv.REGISTRY["pa"].stream(GUARD_ROWS, 0)
    for c in (0.1, 0.01):
        out[f"c_{tag(c)}"] = _fields(cv.REGISTRY["pa"].build(cv.INF, c=c), s.frame)
    return out


def guard_ftrl(key: str, **settings: float) -> dict:
    """``guard_outputs`` and the pace test's other settings after a change
    of regime."""
    reg = cv.REGISTRY[key]
    out = guard_outputs(key)
    frame = cv.broken(reg, "regime", GUARD_ROWS // 2, GUARD_ROWS // 2, 0).frame
    spec = cv._spec("ftrl", loss=key.split(":")[1], **settings)(BREAK_HALF_LIFE)
    out["settings"] = _fields(spec, frame)
    return out


def guard_rates() -> dict:
    """``sgd`` at two constant rates, and its quantile loss at power 0.5."""
    squared, quantile = cv.REGISTRY["sgd:squared"], cv.REGISTRY["sgd:quantile"]
    s = squared.stream(GUARD_ROWS, 0)
    out = {
        f"constant_{tag(lr)}": _fields(
            squared.build(cv.INF, schedule="constant", power=None, learning_rate=lr), s.frame
        )
        for lr in (0.01, 0.04)
    }
    spec = quantile.build(cv.INF, power=0.5, learning_rate=0.5)
    out["quantile_power"] = _fields(spec, quantile.stream(GUARD_ROWS, 0).frame)
    return out


def guard_robust_sets() -> dict:
    """``quantile`` on the first order of each of Rousseeuw and Leroy's sets."""
    import data

    out = {}
    for name in sorted(cv.ROBUST_SETS):
        package, target, outliers = cv.ROBUST_SETS[name]
        table = data.rdataset_or_skip(package, name)
        stream, fs = cv.front_loaded_dataset(table, target, outliers, 300, 0)
        out[name] = _fields(robust_spec(fs), stream)
    return out


def robust_spec(features: list[str]) -> dict:
    """The spec the robust datasets run."""
    return po.spec.quantile(
        "m", targets=["y"], features=features, half_life=cv.INF, quantile=0.5, solve_every=0.0
    )


#: One group a registration; ewridge, rls, lasso, holt and kalman hold
#: cases at rounding, so their hashes keep every bit (``bits=52``).
ledger.guard(
    "ewridge", lambda: guard_prior("ewridge"), oracles=["scikit-learn", "statsmodels"], bits=52
)
ledger.guard("rls", lambda: guard_prior("rls"), oracles=["scikit-learn", "statsmodels"], bits=52)
ledger.guard("lasso", lambda: guard_outputs("lasso"), oracles=["scikit-learn"], bits=52)
ledger.guard("kalman", guard_kalman, oracles=["statsmodels"], bits=52)
ledger.guard("holt", lambda: guard_outputs("holt"), oracles=["statsmodels"], bits=52)
ledger.guard("quantile", lambda: guard_outputs("quantile"), oracles=["statsmodels"])
ledger.guard("huber", lambda: guard_outputs("huber"), oracles=["statsmodels"])
ledger.guard("pa", guard_pa, oracles=["statsmodels"])
ledger.guard("sgd_squared", lambda: guard_outputs("sgd:squared"), oracles=["statsmodels"])
ledger.guard("sgd_quantile", lambda: guard_outputs("sgd:quantile"), oracles=["statsmodels"])
ledger.guard(
    "sgd_epsilon_insensitive", lambda: guard_outputs("sgd:epsilon_insensitive"), oracles=["scipy"]
)
ledger.guard("sgd_poisson", lambda: guard_outputs("sgd:poisson"), oracles=["statsmodels"])
ledger.guard("sgd_logistic", lambda: guard_outputs("sgd:logistic"), oracles=["statsmodels"])
ledger.guard("sgd_huber", lambda: guard_outputs("sgd:huber"), oracles=["statsmodels"])
ledger.guard("sgd_rates", guard_rates, oracles=["statsmodels"])
ledger.guard("ftrl_squared", lambda: guard_ftrl("ftrl:squared", alpha=1.0), oracles=["statsmodels"])
ledger.guard(
    "ftrl_logistic", lambda: guard_ftrl("ftrl:logistic", beta=0.1, l2=0.0), oracles=["statsmodels"]
)
ledger.guard(
    "quantile_robust_sets",
    guard_robust_sets,
    oracles=["statsmodels"],
    dataset="rdatasets:datasets/stackloss,robustbase/starsCYG,robustbase/hbk",
)


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
    # Without decay holt is the weighted line fit through every observation
    # (task 278, D-H): the gap is rounding in the running mean age and
    # spread, growing with n and the level's size (a line rising to 323).
    # Measured 2026-10-26, 50 seeds: 1.78e-14 at 500 rows, 3.41e-12 at
    # 32,000 (seed 27); before, the level's gap grew like ln n, 7.3 to 11.5.
    "holt": (2e-11, None),
    # The prior, `p0` on each standardised slope with the intercept's
    # diffuse, fades as 1/n (rate -1). Measured 0.00051 (seed 3) and -1.01;
    # before task 278's fix the prior was sized from the first rows' noise,
    # 0.0051 (seed 22, whose prior came out small) and -0.93.
    "kalman": (0.001, -0.9),
    # The band narrows with the effective sample. Measured 0.00275 and -0.70
    # before the refit of the first rows (task 278, D-Q2); with it at the
    # 500-row default (2026-10-26), 0.00224 (seed 25) and, from 2,000 rows,
    # -0.58: the refit lands the first 500 rows on their own converged fit
    # (0.054 -> 0.022 at 500 rows), so the rate is read after it.
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
    # The refit of the first 500 rows lands on the batch fit of those rows,
    # and each row after is weighed by the fit before it: the one pass's gap
    # to the batch fit of every row falls as 1/sqrt(n), read from the cut
    # after the refit's (RATE_FROM). Measured at the 500-row default
    # (2026-10-25): 0.00171 and -0.40 over 2,000..32,000 (huber); 0.00413
    # and -0.43 (sgd at power 1, rate 2).
    "huber": (0.0025, -0.35),
    "sgd:huber": (0.006, -0.35),
}

#: The cut a registration's rate is read from, where the first cut is not
#: yet the one pass's: the Huber fits refit their first 500 rows, so at 500
#: rows the gap is the refit's tolerance (3e-4 for both, measured), not a
#: point on the 1/sqrt(n) the rows after it follow (task 277).
RATE_FROM = {"huber": 2000, "sgd:huber": 2000, "quantile": 2000}

#: Registrations whose class 1 is a test of its own below.
CONSISTENCY_ELSEWHERE = {"pa": "test_pa_wanders_in_a_neighbourhood_set_by_c"}


def test_every_consistent_registration_has_a_bound():
    with1 = {k for k, r in cv.REGISTRY.items() if 1 in r.classes}
    assert with1 == set(CONSISTENCY) | set(CONSISTENCY_ELSEWHERE)


@pytest.mark.parametrize("key", [pytest.param(k, marks=_GRID) for k in CONSISTENCY])
def test_consistency(key):
    reg = cv.REGISTRY[key]
    end, steepest = CONSISTENCY[key]
    p = cv.consistency(reg, SEEDS, CUTS)
    p.show(f"{key}: {reg.limit}")
    for c in CUTS:
        _record(key, f"consistency_at_{c}", p.to_oracle[c].values)
    worst = p.worst()
    assert worst[-1] <= end, f"worst seed {p.to_oracle[CUTS[-1]]}"
    if steepest is not None:
        first = CUTS.index(RATE_FROM.get(key, CUTS[0]))
        r = cv.rate(CUTS[first:], worst[first:])
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


@pytest.mark.oracle_grid(reason="a grid of 50 seeds at three settings of C")
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
        for x in CUTS[1:]:
            _record("pa", f"c_{tag(c)}_truth_at_{x}", p.to_truth[x].values)
        assert max(p.worst("truth")[1:]) <= b
        q = [p.to_truth[x].q95 for x in CUTS[1:]]
        assert 0.5 <= q[-1] / q[0] <= 2.0, "the neighbourhood is stationary"
        q95[c] = q[-1]
    assert q95[0.01] < q95[0.1] < q95[1.0]


@pytest.mark.oracle_grid(reason="a grid of 50 seeds at two constant rates and a power")
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
        for x in CUTS[1:]:
            _record("sgd_rates", f"constant_{tag(lr)}_at_{x}", p.to_oracle[x].values)
        q[lr] = [p.to_oracle[x].q95 for x in CUTS[1:]]
        assert 0.6 <= q[lr][-1] / q[lr][0] <= 1.6, "settled: neither shrinking nor growing"
    ratio = q[0.04][-1] / q[0.01][-1]
    print(f"  spread ratio at four times the rate: {ratio:.2f} (theory 2)")
    assert 1.5 <= ratio <= 2.8
    p = cv.consistency(cv.REGISTRY["sgd:quantile"], SEEDS, CUTS, power=0.5, learning_rate=0.5)
    p.show("sgd quantile, inv_scaling at power 0.5")
    for c in CUTS:
        _record("sgd_rates", f"quantile_power_0_5_at_{c}", p.to_oracle[c].values)
    r = cv.rate(CUTS, p.worst())
    print(f"  rate {r:.2f} (theory -0.25)")
    assert -0.4 <= r <= -0.15


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
    # 0.0129 (fresh 0.0106). Without decay, no process noise: Bayesian
    # regression on every row, the other regime's included, its prior fading
    # as 1/n: 0.00104 (seed 10). Before task 278's fix the other regime's
    # residuals inflated the noise in the gain and held the fit there, 45.7
    # of a 46 gap after 32,000 rows.
    "kalman": {200.0: 0.02, cv.INF: 0.002},
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
    # Without decay the line fit through every row, the other regime's
    # included (task 278, D-H): rounding, 3.3e-12 (seed 42); before, 346.
    "holt": {200.0: 0.8, cv.INF: 2e-11},
    # 0.00892 (fresh 0.0101) and 0.0328 (fresh 0.0328).
    "huber": {200.0: 0.0135},
    "sgd:huber": {200.0: 0.05},
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
    # quantile without decay, as the Huber fits below (task 278, D-Q2): the
    # loaded state's refit ran on the other regime's rows, and the one pass
    # keeps the rows inside its bands in its sums as least-squares rows at
    # full weight, where the batch fit of every row reads them as outliers;
    # they leave as the rows after them outweigh them. Measured 2026-10-26
    # at the 500-row default: -1.54 (0.123 off the oracle on every row at
    # 32,000 rows, seed 20; 0.142 before the refit, seed 36).
    "quantile": -1.0,
    # A loaded state's refit has run, on the other regime's rows: the one
    # pass keeps those rows in its sums at the weights they were learned at,
    # where the batch fit of every row reads them as outliers, so the gap to
    # this stream's own fit falls as the rows after it outweigh them: -0.71
    # (2.72 at 32,000 rows) and, for sgd's steps, -1.05 (0.651), measured at
    # the 500-row default (2026-10-25; -0.67 and -1.02 at 200).
    "huber": -0.5,
    "sgd:huber": -0.7,
}

#: pa is at its neighbourhood (class 1) from the first cut: its gap to this
#: stream's own fit, 0.68 at 32,000 rows, is held to 1.5 times the fresh
#: start's, 0.793.
START_AS_FRESH = {"pa"}

#: Cells that fail on a defect: the reason, with its measurement.
START_DEFECTS: dict[tuple[str, float], str] = {}


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
    h, last = tag(half_life), cuts[-1]
    if np.isinf(half_life) and key in START_AS_FRESH:
        _record(key, f"start_{h}_truth_at_{last}", p.to_truth[last].values)
        _record(key, f"start_{h}_clean_at_{last}", p.clean[last].values)
    elif np.isinf(half_life) and key in START_RATE:
        for c in cuts:
            _record(key, f"start_{h}_truth_at_{c}", p.to_truth[c].values)
    else:
        _record(key, f"start_{h}_at_{last}", p.to_oracle[last].values)
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
#: ``sgd``'s intercept takes the running mean's step while it is the larger
#: (task 278, D-S), so a level never passes through the slopes' steps: it
#: was 2.49 off least squares at a level of 1e3 (seed 26, against 0.0096 at
#: level 0) and, under the quantile loss, 0.206 at a level of 3 (seed 32,
#: against 0.042); after it 0.0077 (seed 27) and 0.043 (seed 3) at either
#: level (measured 2026-10-26, 50 seeds).
LEVEL_DEFECTS: dict[str, str] = {}


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
    _record(key, "large_level_at_8000", at.values)
    _record(key, "level_zero_at_8000", base.values)
    print(f"{key}: level {reg.level:g}: {at}\n  level 0: {base}\n  {reg.limit}")
    assert at.worst <= 3 * base.worst + 1e-13 * abs(reg.level)


@pytest.mark.oracle_grid(reason="a grid of seeds, with and without decay, for two models")
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
        h = tag(half_life)
        for c in cuts:
            _record(builder, f"prior_{h}_at_{c}", p.to_oracle[c].values)
        for c in cuts if np.isinf(half_life) else cuts[-1:]:
            _record(builder, f"prior_{h}_truth_at_{c}", p.to_truth[c].values)
        assert max(p.worst()) <= 1e-8, "the batch fit from the prior, to rounding"
        if np.isinf(half_life):
            r = cv.rate(cuts, p.worst("truth"))
            assert -1.1 <= r <= -0.95, r
        else:
            assert p.to_truth[200].median > 1e3, "the prior held the first rows"
            assert p.to_truth[8000].worst <= 1e-7


# ---------------------------------------------------------------------------
# Class 3: adversity in the first rows.


@pytest.mark.oracle_grid(reason="downloads three public datasets and fits three orders of each")
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
    runs = []
    for seed in range(3):
        stream, fs = cv.front_loaded_dataset(table, target, outliers, 300, seed)
        coef = cv.fit(robust_spec(fs), stream, [stream.height])[stream.height]
        X = np.column_stack([np.ones(stream.height), stream.select(fs).to_numpy()])
        y = stream["y"].to_numpy()
        oracle = cv.quantreg(0.5)(X, y, np.ones(len(y)))
        best = cv.check_loss(y - X @ oracle)
        excess = (cv.check_loss(y - X @ coef) - best) / best
        print(
            f"{name}, order {seed}: coef {np.round(coef, 3)}, "
            f"QuantReg {np.round(oracle, 3)}, excess {excess:.2e}"
        )
        runs.append((excess, coef))
    _record("quantile_robust_sets", f"{name.lower()}_excess", [e for e, _ in runs])
    for excess, coef in runs:
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
    # most 1.1e-13 (ewridge, three targets 1e4 off).
    "ewridge": _every(1e-11),
    "rls": _every(1e-11),
    # In units of the gap the descent's stopping rule allows each seed
    # (`cv.lasso_stop_bound`): its tolerance is relative to the target's
    # spread (task 278, D3), which the wild and Cauchy rows inflate to 194
    # and 545 against 2.3 clean, so a fixed 1e-9 failed them at 6.6e-9 and
    # 1.4e-8 while the gap stood where the rule puts it. Measured at most
    # 0.34 of the bound in every cell (2026-10-26).
    "lasso": _every(1.0),
    # Without decay, no process noise, the gain reads no noise estimate
    # (task 278): Bayesian regression on the same rows, the adversity
    # weighed as least squares weighs it, its prior fading as 1/n. Measured
    # 0.0103, 0.00248, 0.00185, 0.0019 and 0.0284 (Cauchy: least squares'
    # own fit moves by hundreds), beside 0.00188 from a clean start. Under a
    # half-life kalman forgets through its drift (coef_half_life = 500),
    # whose start-up keeps its first rows about 2.7 times longer than EW
    # least squares' weights do (the filter weighs them evenly until its `P`
    # meets the drift, some h / ln 2 rows): measured 0.00881, 0.028, 0.00744,
    # 0.00744 and 0.0682, beside 0.00744 clean. Before task 278 the noise
    # estimate in the gain down-weighed the rows after a burst (0.0060 to
    # 0.0157 here) and captured the fit without decay (19.3, 5.76, 17.8 and
    # 658 at 8,000 rows).
    "kalman": {
        ("outliers", INF): 0.015,
        ("wild", INF): 0.004,
        ("shift", INF): 0.003,
        ("t2", INF): 0.003,
        ("cauchy", INF): 0.045,
        ("outliers", 500.0): 0.013,
        ("wild", 500.0): 0.04,
        ("shift", 500.0): 0.011,
        ("t2", 500.0): 0.011,
        ("cauchy", 500.0): 0.1,
    },
    # A robust fit: without decay it is held to three times its clean worst
    # (0.0079): t2 noise measured 0.0076. Under a half-life of 500, measured
    # 0.0131, 0.0367 (three targets 1e4 off, forgotten as 2^-16), 0.0130,
    # 0.0129 and 0.0258, beside 0.0129 clean.
    # Without decay, with the refit of the first 500 rows (task 278, D-Q2),
    # measured 2026-10-26: 0.00846 (outliers), 0.00528 (wild), 0.00575 (t2)
    # and 0.00584 (cauchy), beside 0.00519 clean; before the refit 0.038,
    # 0.22, 0.0078 and 0.0175, the rows the warm-up took as least squares
    # and the first bands held kept in the sums at full weight. A shift of
    # every one of the first 500 rows is no outlier among them, so the refit
    # keeps them as the shifted fit's rows and the one pass lets them go
    # faster than 1/n: 0.535 at 2,000 rows, 0.0608 at 8,000 (seed 5; 0.093
    # before the refit), as the Huber fits' shift below.
    "quantile": {
        ("outliers", INF): 0.024,
        ("wild", INF): 0.024,
        ("shift", INF): 0.09,
        ("t2", INF): 0.024,
        # Measured 0.0178 (seed 45) on 2026-10-26, the band in the MAD (task
        # 278, D-Q): in the RMS it was 0.143, a defect.
        ("cauchy", INF): 0.024,
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
    # A robust fit with the refit of its first 500 rows. Measured at that
    # default (2026-10-25) without decay 0.00367 (wild), 0.00361 (t2) and
    # 0.00509 (cauchy), beside 0.0034 clean; a shift of every one of the
    # first 500 rows is no outlier among them, so the refit keeps them and
    # the one pass lets them go as 1/n (5.97 at 2,000 rows, 1.24 at 8,000).
    # Under a half-life every kind 0.00497 to 0.00501, beside 0.00501 clean.
    # sgd, at power 1: 0.00693, 0.00792, 0.00827 and the shift 0.239 (4.17
    # at 2,000); under a half-life 0.0163 to 0.0164, its clean 0.0163.
    # One-sided wild rows, 20% of the first 500, 40% of those with x0 above
    # 0: the refit of the first 500 rows starts from the one-pass fit they
    # pulled, and in some seeds ends there; without decay the one pass then
    # keeps the captured rows. A bound at the edge of Huber's breakdown, not
    # a fault (2026-10-28): measured 0.158 (huber) and 0.149 (sgd) at 8,000
    # rows, seed 30, on 2026-10-26; 0.0576 and 0.0698, seed 21, on
    # 2026-10-31, against 0.0034 and 0.0067 clean. The class 4 cells at 20%
    # below are the same regime.
    "huber": {
        ("outliers", INF): 0.1,
        ("wild", INF): 0.0055,
        ("shift", INF): 1.9,
        ("t2", INF): 0.007,
        ("cauchy", INF): 0.01,
        **{(kind, 500.0): 0.0075 for kind in cv.ADVERSITIES},
    },
    "sgd:huber": {
        ("outliers", INF): 0.1,
        ("wild", INF): 0.011,
        ("shift", INF): 0.36,
        ("t2", INF): 0.014,
        ("cauchy", INF): 0.015,
        **{(kind, 500.0): 0.025 for kind in cv.ADVERSITIES},
    },
}

#: A cell that fails on a defect, by its reason: none since the Huber fits'
#: 20% cells were bounded at the breakdown they show (2026-10-28).
ADVERSE_DEFECTS: dict[tuple[str, str, float], str] = {}


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
    unit = cv.lasso_stop_bound if key == "lasso" else None
    p, touched = cv.front_loaded(reg, kind, SEEDS, half_life, ADVERSE_CUTS, unit=unit)
    p.show(f"{key}, {kind} in the first 500 rows, half-life {half_life:g}: {touched}+ rows touched")
    h, (first_cut, last_cut) = tag(half_life), ADVERSE_CUTS
    values = p.to_oracle[last_cut].values
    _record(key, f"{kind}_{h}_at_{last_cut}", values)
    if np.isinf(half_life) and key in ("sgd:squared", "pa"):
        _record(key, f"{kind}_{h}_at_{first_cut}", p.to_oracle[first_cut].values)
    if unit is not None:
        print("  (the gap to the oracle in units of the stopping rule's bound)")
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


def test_kalman_is_not_captured_by_a_burst_essential():
    """Task 278 (a)'s D-K, the representative cell: ``kalman`` with no
    process noise and the noise learned, one-sided outliers on a fifth of its
    first 500 rows, ten seeds. Its gain reads no noise estimate, so it is
    Bayesian regression on the same rows and least squares' fit to them.
    Measured 2026-10-25 over 50 seeds: 0.0103 at 8,000 rows (seed 46); with
    the estimate in the gain, 19.3 (seed 31)."""
    p, touched = cv.front_loaded(cv.REGISTRY["kalman"], "outliers", 10, cv.INF, [2000, 8000])
    p.show(f"kalman, outliers first, no decay, 10 seeds, {touched}+ rows wild")
    assert touched >= 50
    assert p.to_oracle[8000].worst <= ADVERSE["kalman"][("outliers", cv.INF)]


#: K1 (task 278 (b)): ``kalman`` at a drift of 200 and a half-life of 300 on
#: a target 1e4 above the truth's. Its residual variance learned from the
#: warm-up's innovations, the level among them, and the filter ran on that
#: noise for some log2(L^2 / sigma^2) half-lives: predictions 0.019 to 0.032
#: of the target's spread off the fit at level 0 over rows 3,000 to 6,000.
K1_SPEC = dict(targets=["y"], features=cv.FEATURES, coef_half_life=200.0, half_life=300.0)
K1_LEVEL = 1e4


def _k1_level_deviation(seed: int, n: int, level: float) -> tuple[float, float]:
    """The largest |pred at level L - L - pred at level 0| over a stream's
    rows, in units of the target's spread and in ``u (|L| + max|y|)``."""
    spec = po.spec.kalman("m", **K1_SPEC)
    base = cv.linear(n, seed)
    preds = []
    for s in (base, cv.linear(n, seed, level=level)):
        out = po.ModelBank([spec]).fit_predict(s.frame).unnest("m")
        preds.append(out["pred_y"].to_numpy().astype(float))
    p0, p1 = preds
    assert np.array_equal(np.isnan(p0), np.isnan(p1)), "the level withholds no other row"
    assert np.isfinite(p0[100:]).all()
    off = float(np.nanmax(np.abs(p1 - level - p0)))
    u = 2.0**-53 * (abs(level) + float(np.max(np.abs(base.y))))
    return off / float(np.std(base.y)), off / u


def test_kalman_moves_every_prediction_by_a_level_essential():
    """K1, the representative cell: ten seeds, 6,000 rows. The diffuse
    intercept takes the level on the first row and nothing reads the noise,
    so every prediction moves by the level, to the rounding of a number of
    its size: 64 ``u (|L| + max|y|)``, as the equivariance class holds the
    models that absorb a level exactly. Measured 2026-10-25: at most 24.6 of
    those units on these ten seeds, 22.8 over 50 seeds of 12,000 rows (1.4e-11
    of the spread); before task 278's fix, 0.0307 of the spread at rows 3,000
    to 6,000 (seed 25)."""
    worst = max(_k1_level_deviation(seed, 6000, K1_LEVEL)[1] for seed in range(10))
    print(f"kalman, level {K1_LEVEL:g}: at most {worst:.1f} units of u (|L| + max|y|)")
    assert worst <= 64.0


@pytest.mark.oracle_grid(reason="a grid of 50 seeds against a decayed least-squares oracle")
def test_kalman_reaches_the_oracle_at_a_level_as_at_zero():
    """K1 against the oracle: at the level and at 0 the worst seed's gap to
    EW least squares at the drift's half-life is the same at every cut to
    3e-13 of the level -- the level costs nothing at any row -- and every
    prediction moves by the level (above). Without decay, at the
    registration's level of 1e6, the worst seed's gap to least squares falls
    as 1/n, its prior fading: measured 2026-10-25, 0.0343 at 500 rows to
    0.00051 at 32,000 (seed 3), rate -1.01, the same as at level 0; before
    task 278's fix 0.402 at 8,000 (seed 40)."""
    cuts = [1000, 2000, 4000, 8000, 12000]
    spec = po.spec.kalman("m", **K1_SPEC)
    gaps: dict[float, dict[int, dict[int, float]]] = {
        L: {c: {} for c in cuts} for L in (0.0, K1_LEVEL)
    }
    units = []
    for seed in range(SEEDS):
        for level in gaps:
            s = cv.linear(cuts[-1], seed, level=level)
            path = cv.fit(spec, s.frame, cuts)
            for c in cuts:
                gaps[level][c][seed] = cv.gap(
                    path[c], cv.oracle_at(cv.REGISTRY["kalman"], s, c, 200.0)
                )
        units.append(_k1_level_deviation(seed, cuts[-1], K1_LEVEL)[1])
    worst_units = max(units)
    print(f"  every prediction within {worst_units:.1f} units of u (|L| + max|y|) of level 0's")
    _record("kalman", "k1_prediction_units", units)
    for c in cuts:
        _record("kalman", f"k1_level_at_{c}", list(gaps[K1_LEVEL][c].values()))
    assert worst_units <= 64.0
    for c in cuts:
        at, base = cv.tail(gaps[K1_LEVEL][c]), cv.tail(gaps[0.0][c])
        print(f"  {c} rows: level {K1_LEVEL:g} {at}; level 0 {base}")
        assert abs(at.worst - base.worst) <= 3e-13 * K1_LEVEL
    reg = cv.REGISTRY["kalman"]
    by_cut: dict[int, dict[int, float]] = {c: {} for c in CUTS}
    for seed in range(SEEDS):
        s = reg.stream(CUTS[-1], seed, level=reg.level)
        path = cv.fit(reg.build(cv.INF), s.frame, CUTS)
        for c in CUTS:
            by_cut[c][seed] = cv.gap(path[c], cv.oracle_at(reg, s, c, cv.INF))
    for c in CUTS:
        _record("kalman", f"level_inf_at_{c}", list(by_cut[c].values()))
    worst = [cv.tail(by_cut[c]).worst for c in CUTS]
    r = cv.rate(CUTS, worst)
    print(f"  no decay, level {reg.level:g}: worst {worst}, rate {r:.2f}")
    assert worst[-1] <= CONSISTENCY["kalman"][0]
    assert r <= CONSISTENCY["kalman"][1]


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
    # state only. Measured 0.0221 at 4 (0.0122 at 6), 0.021 at 8, 0.0472 and
    # 0.0274 at 12. A level break is followed faster than before task 278,
    # whose noise estimate the break inflated and the gain read: 0.241 at 4
    # half-lives then, 0.0206 at 6.
    "kalman": {
        "level": (4, 0.035),
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
    #
    # Under a half-life ftrl follows a break at the pace of its steps (task
    # 278, D-F: all three sums already decay; AdaGrad's normalised step
    # moves a settled coefficient at most about 2 alpha sqrt(1 - lam) a row,
    # 0.0118 here). A change of regime moves the intercept by 46, so it is
    # still 19.9 off at 12 half-lives (seed 44); the pace itself is held in
    # test_ftrl_follows_a_break_at_the_pace_of_its_steps. A logistic fit's
    # gap after a slope break or a change of regime, 1.07 and 1.36 at 12, is
    # the fixed prior's steady-state ridge, (1 - lam)(beta/alpha + l2), on a
    # log-odds the evidence pins only weakly (0.30 and 0.58 at beta=0.1,
    # l2=0, held in the same test). Measured 2026-10-25, 50 seeds.
    "ftrl:squared": {
        "level": (12, 0.47),
        "slope": (12, 0.37),
        "variance": (12, 0.3),
        "regime": (12, 30.0),
    },
    "ftrl:logistic": {"level": (12, 0.84), "slope": (12, 1.6), "regime": (12, 2.0)},
    # holt's trend forgets at trend_half_life, four times the half-life, so
    # 12 half-lives are three of its trend's: measured 0.267, 0.744, 0.127 and
    # 4.88 (a level 100 off and a slope 0.06 off at the break) at 12.
    "holt": {
        "level": (12, 0.4),
        "slope": (12, 1.12),
        "variance": (12, 0.19),
        "regime": (12, 7.3),
    },
    # The MAD scale holds the fit back after a level or a regime break until
    # the median has crossed (MODELS.md, a robust scale after a change of
    # level). Measured 0.0266 and 0.0115 at 8, 0.0309 (flat from one
    # half-life: the variance changes the scale, not the fit) and 0.0183 at
    # 12; sgd at power 1, 0.0251 and 0.0247 at 8, 0.13 and 0.0326 at 12.
    "huber": {
        "level": (8, 0.04),
        "slope": (8, 0.018),
        "variance": (12, 0.047),
        "regime": (12, 0.028),
    },
    "sgd:huber": {
        "level": (8, 0.038),
        "slope": (8, 0.037),
        "variance": (12, 0.2),
        "regime": (12, 0.05),
    },
}

#: Cells that fail on a defect: the reason, with its measurement. Empty since
#: task 278's D-F was found a limit of the method, not a defect (above).
REBREAK_DEFECTS: dict[tuple[str, str], str] = {}


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
    _record(key, f"break_{kind}_at_{tag(k)}", p.to_oracle[k].values)
    if kind != "variance":
        # The evidence: at some point past the break (at 0 half-lives, the
        # fit before it) the fit was far from the new truth, farther than at
        # the end. A trend's slope break shows later, as the level drifts.
        most = max(p.to_truth[k].median for k in BREAK_KS)
        assert most > 1.5 * p.to_truth[BREAK_KS[-1]].median, "the break was taken"
    assert p.to_oracle[k].worst <= bound


@pytest.mark.oracle_grid(reason="a grid of 50 seeds at three settings of ftrl")
def test_ftrl_follows_a_break_at_the_pace_of_its_steps():
    """Under a half-life ``ftrl``'s sums all decay, and what bounds how fast
    it follows a break is AdaGrad's normalised step: in steady state a
    coefficient moves at most about ``2 alpha sqrt(1 - lam)`` a row however
    large its error (``ftrl.rs``, "Following a break"). Held: (a) after a
    change of regime the intercept's median distance to the new truth closes
    at that pace from 4 to 12 half-lives; (b) a larger ``alpha`` follows the
    same break sooner; (c) a logistic fit's lasting gap is the prior's: a
    weaker one closes it. Measured 2026-10-25 over 50 seeds (task 278, D-F):
    (a) 0.0108 a row against 0.0118; (b) at ``alpha = 1`` the worst gap 0.70
    at 8 half-lives, against 28.5 at the default; (c) at ``beta = 0.1, l2 =
    0`` the worst gap 0.30 after a slope break and 0.58 after a change of
    regime at 12 half-lives, against 1.07 and 1.36 at the defaults."""
    lam = 0.5 ** (1.0 / BREAK_HALF_LIFE)
    pace = 2 * 0.1 * np.sqrt(1 - lam)
    reg = cv.REGISTRY["ftrl:squared"]
    p = cv.reconvergence(reg, "regime", SEEDS, BREAK_HALF_LIFE, BREAK_AT, [4, 12])
    p.show("ftrl, a change of regime at the default alpha")
    closed = (p.to_truth[4].median - p.to_truth[12].median) / (8 * BREAK_HALF_LIFE)
    print(f"  closed {closed:.4f} a row against 2 alpha sqrt(1 - lam) = {pace:.4f}")
    _record("ftrl:squared", "regime_closed_over_pace", closed / pace, ideal=1.0)
    assert 0.7 * pace <= closed <= 1.1 * pace
    fast = dataclasses.replace(reg, build=cv._spec("ftrl", loss="squared", alpha=1.0))
    q = cv.reconvergence(fast, "regime", SEEDS, BREAK_HALF_LIFE, BREAK_AT, [8])
    q.show("ftrl, a change of regime at alpha = 1")
    _record("ftrl:squared", "alpha_1_regime_at_8", q.to_oracle[8].values)
    assert q.to_oracle[8].worst <= 1.1
    weak = dataclasses.replace(
        cv.REGISTRY["ftrl:logistic"], build=cv._spec("ftrl", loss="logistic", beta=0.1, l2=0.0)
    )
    for kind, bound in (("slope", 0.45), ("regime", 0.85)):
        r = cv.reconvergence(weak, kind, SEEDS, BREAK_HALF_LIFE, BREAK_AT, [12])
        r.show(f"ftrl logistic, a break in {kind}, beta = 0.1 and l2 = 0")
        _record("ftrl:logistic", f"weak_prior_{kind}_at_12", r.to_oracle[12].values)
        assert r.to_oracle[12].worst <= bound


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
#: Measured 2026-10-26, 50 seeds at 20,000 rows, the band in the MAD (task
#: 278, D-Q): no seed captured at any share, without decay or at a
#: half-life of 5,000, the worst slope gap 0.0059 / 0.0068 / 0.0095 / 0.026
#: at 5 / 10 / 15 / 20% without decay (seeds 31, 25, 36, 14) and 0.0041 /
#: 0.0050 / 0.0069 / 0.012 at the half-life. In the RMS every seed was
#: captured at every share (the worst 0.098 at 5%, 0.98 at 20%): the wild
#: rows inflated the scale to about 11 against a noise of 0.5, the band held
#: every clean row as a least-squares row and gave each wild row a pull of
#: its width. Least squares, for scale, is pulled by about 2 at 5%.
#: Re-measured with the sketch anchored (task 278, 2026-10-26): still none
#: captured, the worst 0.0062 / 0.0075 / 0.0095 / 0.026 without decay and
#: 0.0041 / 0.0049 / 0.0074 / 0.012 at the half-life; the Huber fits' grid
#: below moved by at most 1.5e-4 in its worst gaps, its capture counts not
#: at all.
BREAKDOWN_DEFECTS: dict[tuple[float, float], str] = {}


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
    _record("quantile", f"breakdown_{tag(share)}_{tag(half_life)}_slope_gap", t.values)
    assert wild > 0.5 * share * 20_000, "the contamination is there"
    assert captured == 0, t


def test_breakdown_essential():
    """The representative cell: ``quantile`` at its defaults at 10% wild
    rows, ten seeds. Measured 2026-10-26 over 50 seeds: none captured, the
    worst slope gap 0.0068 (seed 25); in the root mean square scale, before
    the band was drawn in the MAD, all 50 were (the worst 0.285)."""
    captured, t, wild = cv.captures(cv.REGISTRY["quantile"], 0.10, 10, 20_000, cv.INF)
    print(f"quantile, 10% wild: {captured}/10; {t}")
    assert wild > 1000
    assert captured == 0, t


# Class 4 for the Huber fits, on the same contamination (cv.contaminate): a
# seed is captured when its slope ends more than 0.05 from the batch Huber
# fit with the MAD scale on the same rows. Measured 2026-10-25, 50 seeds at
# 20,000 rows: no seed captured at 5%, 10% or 15%, with or without decay
# (the worst 0.0175 for huber, 0.0208 for sgd, at 15% without decay); at
# 20%, 40% of the rows with x0 above 0, the refit of the first rows is
# itself captured in some seeds (below). Re-measured at the 500-row default
# (2026-10-25): still none at 5% to 15% (the worst 0.0128 for huber, 0.0144
# for sgd, at 15% without decay); at 20% 8 and 5 of 50 for huber, 6 and 5
# for sgd, where 200 rows left 16, 12, 15 and 10. Re-measured 2026-10-26 on
# task 278's wave-2 build: the same counts, the worst 0.853 and 0.587 for
# huber, 0.811 and 0.532 for sgd, all seed 21.
#
# The 20% cells are the edge of Huber's breakdown, a bound rather than a
# fault (the user, 2026-10-28; MODELS.md, "huber"): the refit weighs each
# row by its residual from the one-pass fit the wild rows pulled, so some
# seeds stay captured. Each is held to at most 10 of 50 seeds captured and
# the worst slope at most 1.0 off, the counts and gaps above with headroom;
# every other cell captures none.
HUBER_BREAKDOWN_BOUND: dict[tuple[str, float, float], tuple[int, float]] = {
    (key, 0.20, h): (10, 1.0) for key in ("huber", "sgd:huber") for h in BREAKDOWN_HALF_LIVES
}
#: A cell that fails on a defect, by its reason: none since the 20% cells
#: were bounded (2026-10-28).
HUBER_BREAKDOWN_DEFECTS: dict[tuple[str, float, float], str] = {}


def _huber_breakdown_params() -> list:
    out = []
    for key in ("huber", "sgd:huber"):
        for share in BREAKDOWN_SHARES:
            for h in BREAKDOWN_HALF_LIVES:
                marks = [_GRID]
                if (key, share, h) in HUBER_BREAKDOWN_DEFECTS:
                    reason = HUBER_BREAKDOWN_DEFECTS[(key, share, h)]
                    marks.append(pytest.mark.xfail(strict=True, reason=reason))
                out.append(pytest.param(key, share, h, marks=marks, id=f"{key}-{share:.0%}-{h:g}"))
    return out


@pytest.mark.parametrize("key, share, half_life", _huber_breakdown_params())
def test_huber_is_never_captured_short_of_breakdown(key, share, half_life):
    captured, t, wild = cv.captures(cv.REGISTRY[key], share, SEEDS, 20_000, half_life)
    print(f"{key}, {share:.0%} wild, half-life {half_life:g}: {captured}/{SEEDS}; {t}")
    _record(key, f"breakdown_{tag(share)}_{tag(half_life)}_slope_gap", t.values)
    assert wild > 0.5 * share * 20_000, "the contamination is there"
    most, worst = HUBER_BREAKDOWN_BOUND.get((key, share, half_life), (0, np.inf))
    assert captured <= most, t
    assert t.worst <= worst, t


#: The cases whose gap to the oracle is rounding, by ``<group>.<case>``:
#: the batch fits (ewridge, rls, lasso to its stopping rule, holt without
#: decay) and K1's predictions in units of ``u``. Each is the case's
#: ``floor``, the gap below which a difference is another processor's last
#: bits: about four times the worst measured over the grid's seeds on macOS
#: arm64 and on two of CI's Linux x86-64 runs, whichever is largest
#: (2026-10-28; GitHub's runners vary in CPU and faer picks its SIMD kernel
#: by the CPU, so two Linux runs read some of these up to 7 times apart;
#: docs/TESTING.md, "The oracle ledger").
ROUND_OFF: dict[str, float] = {
    "ewridge.break_level_at_1": 1.1e-13,
    "ewridge.break_regime_at_1": 9.4e-13,
    "ewridge.break_slope_at_1": 7.2e-14,
    "ewridge.break_variance_at_1": 4.5e-14,
    "ewridge.cauchy_500_at_8000": 5.2e-14,
    "ewridge.cauchy_inf_at_8000": 2.8e-13,
    "ewridge.consistency_at_2000": 8.2e-14,
    "ewridge.consistency_at_32000": 2.4e-13,
    "ewridge.consistency_at_500": 4e-14,
    "ewridge.consistency_at_8000": 1.2e-13,
    "ewridge.large_level_at_8000": 4.7e-10,
    "ewridge.level_zero_at_8000": 1.5e-13,
    "ewridge.outliers_500_at_8000": 5.4e-14,
    "ewridge.outliers_inf_at_8000": 1.9e-13,
    "ewridge.prior_200_at_200": 8.9e-09,
    "ewridge.prior_200_at_2000": 5.5e-09,
    "ewridge.prior_200_at_6000": 7.2e-09,
    "ewridge.prior_200_at_8000": 4.5e-09,
    "ewridge.prior_inf_at_2000": 6.5e-09,
    "ewridge.prior_inf_at_32000": 1.3e-08,
    "ewridge.prior_inf_at_500": 1.1e-08,
    "ewridge.prior_inf_at_8000": 4.7e-09,
    "ewridge.shift_500_at_8000": 5.7e-14,
    "ewridge.shift_inf_at_8000": 1.2e-13,
    "ewridge.start_200_at_3200": 2.9e-14,
    "ewridge.start_inf_at_32000": 3.5e-13,
    "ewridge.t2_500_at_8000": 5e-14,
    "ewridge.t2_inf_at_8000": 1.1e-13,
    "ewridge.wild_500_at_8000": 4.3e-14,
    "ewridge.wild_inf_at_8000": 4.4e-13,
    "holt.consistency_at_2000": 2.2e-13,
    "holt.consistency_at_32000": 1.4e-11,
    "holt.consistency_at_500": 7.2e-14,
    "holt.consistency_at_8000": 1.5e-12,
    "holt.large_level_at_8000": 2.1e-08,
    "holt.level_zero_at_8000": 1.5e-12,
    "holt.start_inf_at_32000": 1.4e-11,
    "lasso.break_level_at_1": 5.3e-10,
    "lasso.break_regime_at_1": 3.4e-09,
    "lasso.break_slope_at_1": 3.7e-10,
    "lasso.break_variance_at_1": 3.8e-10,
    "lasso.consistency_at_2000": 2.5e-10,
    "lasso.consistency_at_32000": 2.4e-10,
    "lasso.consistency_at_500": 2.8e-10,
    "lasso.consistency_at_8000": 2.5e-10,
    "lasso.large_level_at_8000": 1.9e-09,
    "lasso.level_zero_at_8000": 2.4e-10,
    "lasso.start_200_at_3200": 2.8e-10,
    "lasso.start_inf_at_32000": 1.5e-09,
    "rls.break_level_at_1": 4e-14,
    "rls.break_regime_at_1": 2.3e-13,
    "rls.break_slope_at_1": 5.2e-14,
    "rls.break_variance_at_1": 4.9e-14,
    "rls.cauchy_500_at_8000": 5.2e-14,
    "rls.cauchy_inf_at_8000": 1.2e-12,
    "rls.consistency_at_2000": 5.6e-14,
    "rls.consistency_at_32000": 4.1e-13,
    "rls.consistency_at_500": 3.6e-14,
    "rls.consistency_at_8000": 1.6e-13,
    "rls.large_level_at_8000": 6.4e-08,
    "rls.level_zero_at_8000": 1.2e-13,
    "rls.outliers_500_at_8000": 4.8e-14,
    "rls.outliers_inf_at_8000": 1.6e-13,
    "rls.prior_200_at_200": 8.7e-09,
    "rls.prior_200_at_2000": 5.5e-09,
    "rls.prior_200_at_6000": 7.2e-09,
    "rls.prior_200_at_8000": 4.6e-09,
    "rls.prior_inf_at_2000": 8e-09,
    "rls.prior_inf_at_32000": 1.3e-08,
    "rls.prior_inf_at_500": 1.1e-08,
    "rls.prior_inf_at_8000": 4.7e-09,
    "rls.shift_500_at_8000": 5.4e-14,
    "rls.shift_inf_at_8000": 1.7e-13,
    "rls.start_200_at_3200": 4.3e-14,
    "rls.start_inf_at_32000": 3.4e-13,
    "rls.t2_500_at_8000": 6.8e-14,
    "rls.t2_inf_at_8000": 1.3e-13,
    "rls.wild_500_at_8000": 6.1e-14,
    "rls.wild_inf_at_8000": 3.5e-13,
    "kalman.k1_prediction_units": 46.0,
}
