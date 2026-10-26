"""A diagnostic never moves the fit (docs/PLAN.md task 278, batch 6).

Every per-spec diagnostic switch, and every knob that changes how a
diagnostic works, writes fields of its own and nothing else. With it on
and with it off, the fit's fields are the same to the bit: ``pred_*``,
``resid_*``, ``coef``, ``support_coef``, ``weight_sum``, ``settled_frac``,
``withheld_reason`` and ``lasso``'s ``penalty_selected_*`` -- every field
the spec writes with no diagnostic on, which is what ``_fit_fields`` holds
the base run to. A diagnostic that read a model's state through a mutable
path, consumed a row the model then missed, or advanced a clock the model
shares, would show here as a moved bit.

The knobs: ``conformal``'s level and ``conformal_rate``, ``resid_quantiles``'
levels, ``resid_autocorr_lag``, the ``drift_*`` tolerances, every
``*_half_life`` memory (a finite one and ``inf`` where it is legal),
``robust_se_lags``, ``ljung_box_lags``, ``horizon_rows`` under each
diagnostic that reads it, ``average_eta``, ``emit_error_inflation`` (the
readiness gates' own diagnostic), ``emit_clocks``, and every switch at once.

``drift_action="reset"`` is left out: it restarts the model by design, and
``tests/test_drift_reset.py`` holds it to a fresh bank.

The readiness gates (``min_weight``, ``min_settled_frac``,
``max_error_inflation``) are here too, held to the narrower promise a gate
makes. A gate's job is to withhold predictions, so on the rows it withholds
it nulls ``pred`` and ``resid`` and names itself in ``withheld_reason``:
those two moves are the gate working, not a leak, so ``withheld_reason`` is
not compared, and ``pred``/``resid`` are compared on the rows both runs
score, a null allowed only where that run's gate withheld the row.
Everything a gate could only move by touching the fit -- ``coef``,
``support_coef``, ``weight_sum``, ``settled_frac``, ``penalty_selected_*``
-- is compared on every row: a gate never moves learning, ``min_weight``
in every kind (``kalman``'s noise and ``lasso``'s selection learned only
from the rows it let through, documented exceptions until task 278, D10);
``quantile``'s is a defect, a strict xfail (``DEFECTS``).

Each case prints and asserts evidence that it took effect, so no case
compares two identical runs: a switch writes at least one non-null value
of its own; a knob's diagnostic fields differ from its switch's at the
default; a gate withholds a different set of rows; the drift detector
fires (the stream has a level shift of 8 noise sd at row 1,300).

Two streams: ``rows``, with no clock, uneven weights with zeros and a
second target missing on one row in twenty; and ``clocked``, the same
rows on an irregular clock with steps past ``gap_cap``, two session
changes and an ``embargo``, so the diagnostics' clock, run and delay
paths are on.

Which kind takes which switch is a table (``TAKES``), not a try: a pair
the table leaves out must be refused by the builder
(``test_every_pair_left_out_is_refused``), so the table cannot hide a pair
that runs.

Essentials: ``ewridge``, ``sgd`` and ``holt``. Extended: the other linear
kinds, ``huber`` and ``sgd(loss="huber")`` among them, with the refit of
their first rows on (task 277). The Rust twin, which ``cargo mutants`` sees, is
``crates/online-polars/tests/diagnostic_isolation.rs``.
"""

from __future__ import annotations

import functools
import json
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_model_registry import REGRESSIONS

TIER = "mixed"

N = 2000
SHIFT_AT = 1300

Builder = Callable[..., dict]

#: The linear kinds, each with the keywords its plainest well-behaved fit
#: needs on these streams.
KINDS: dict[str, Builder] = {
    "ewridge": lambda **kw: po.spec.ewridge("m", **kw),
    "rls": lambda **kw: po.spec.rls("m", **kw),
    "kalman": lambda **kw: po.spec.kalman("m", coef_half_life=100.0, **kw),
    "lasso": lambda **kw: po.spec.lasso("m", penalties=[0.1, 0.01], **kw),
    "quantile": lambda **kw: po.spec.quantile("m", quantile=0.5, **kw),
    "sgd": lambda **kw: po.spec.sgd("m", learning_rate=0.02, **kw),
    "pa": lambda **kw: po.spec.pa("m", **kw),
    "ftrl": lambda **kw: po.spec.ftrl("m", loss="squared", **kw),
    # The trend at the level's half-life: at its default, four times it, the
    # trend settles past the stream's end, and every diagnostic that waits
    # for the fit to settle (calibration, breaks, specification, tails)
    # would be null throughout.
    "holt": lambda **kw: po.spec.holt("m", trend_half_life=120.0, **kw),
    # With the refit of their first rows, on by default (task 277).
    "huber": lambda **kw: po.spec.huber("m", **kw),
    "sgd_huber": lambda **kw: po.spec.sgd("m", loss="huber", learning_rate=0.02, **kw),
}

ESSENTIAL_KINDS = ("ewridge", "sgd", "holt")
#: The kinds with no features: they take ``targets`` only.
NO_FEATURES = frozenset({"holt"})

#: Where a switch is taken by fewer than every linear kind (the builders'
#: refusals; docs/DIAGNOSTICS.md, "What each costs, and which models take it").
LEVERAGE = frozenset({"ewridge", "rls", "kalman"})
LEAST_SQUARES = frozenset({"ewridge", "rls"})
NOISE_GATE = frozenset({"ewridge", "rls", "kalman", "lasso"})
ALL = frozenset(KINDS)
WITH_FEATURES = ALL - NO_FEATURES

#: Each switch's kinds. A knob is taken where its switch is.
TAKES: dict[str, frozenset[str]] = {
    "emit_error_inflation": LEVERAGE,
    "emit_se_coef": LEVERAGE,
    "emit_influence": LEVERAGE,
    "emit_robust_se": LEAST_SQUARES,
    "emit_feature_health": WITH_FEATURES,
    "max_error_inflation": NOISE_GATE,
}


@dataclass(frozen=True)
class Case:
    """One switch or knob: the keywords it adds to the base spec.

    ``parent`` names the case this one varies a knob of, whose diagnostic
    fields this one's must differ from (the knob took effect); ``grid``
    runs it on a half-life grid, for the switches that choose across
    slots; ``gate`` holds it to a gate's promise instead of a
    diagnostic's."""

    name: str
    kw: dict[str, Any]
    parent: str | None = None
    grid: bool = False
    gate: bool = False
    needs: str | None = field(default=None)
    #: The kinds where the case is not a default the base already has.
    only: frozenset[str] | None = None

    def taken_by(self, kind: str) -> bool:
        key = self.needs or next(iter(self.kw))
        return kind in TAKES.get(key, ALL)

    def swept_on(self, kind: str) -> bool:
        return self.taken_by(kind) and (self.only is None or kind in self.only)


def _cases() -> list[Case]:
    c: list[Case] = []

    def switch(name: str, needs: str | None = None, **kw: Any) -> None:
        c.append(Case(name, kw or {name: True}, needs=needs))

    def knob(name: str, parent: str, **kw: Any) -> None:
        base = next(x for x in c if x.name == parent)
        c.append(Case(name, base.kw | kw, parent=parent, grid=base.grid, needs=base.needs))

    for s in (
        "emit_sigma",
        "emit_zscore",
        "emit_metrics",
        "emit_autocorr",
        "emit_drift",
        "emit_calibration",
        "emit_breaks",
        "emit_specification",
        "emit_tails",
        "emit_clocks",
        "emit_error_inflation",
        "emit_se_coef",
        "emit_influence",
        "emit_robust_se",
        "emit_feature_health",
    ):
        switch(s)
    switch("conformal", conformal=0.9)
    knob("conformal=0.5", "conformal", conformal=0.5)
    knob("conformal=0.99", "conformal", conformal=0.99)
    knob("conformal_rate=0.3", "conformal", conformal_rate=0.3)
    knob("conformal_rate=0.005", "conformal", conformal_rate=0.005)
    switch("resid_quantiles", resid_quantiles=[0.5])
    knob("resid_quantiles=[0.05,0.5,0.99]", "resid_quantiles", resid_quantiles=[0.05, 0.5, 0.99])
    knob("resid_autocorr_lag=2", "emit_autocorr", resid_autocorr_lag=2)
    knob("resid_autocorr_lag=5", "emit_autocorr", resid_autocorr_lag=5)
    knob("drift_delta=0", "emit_drift", drift_delta=0.0)
    knob("drift_delta=2", "emit_drift", drift_delta=2.0)
    knob("drift_threshold=5", "emit_drift", drift_threshold=5.0)
    knob("drift_threshold=200", "emit_drift", drift_threshold=200.0)
    for parent, memory, legal_inf in (
        ("emit_calibration", "calibration_half_life", True),
        ("emit_breaks", "breaks_half_life", True),
        ("emit_specification", "specification_half_life", True),
        ("emit_tails", "tails_half_life", True),
        ("emit_influence", "influence_half_life", True),
        ("emit_robust_se", "robust_se_half_life", True),
        # A run-once feature health would be null on every row: refused.
        ("emit_feature_health", "feature_health_half_life", False),
    ):
        knob(f"{memory}=30", parent, **{memory: 30.0})
        if legal_inf:
            knob(f"{memory}=inf", parent, **{memory: math.inf})
    # 0, which writes no `se_coef_hac`, is the default with no horizon.
    knob("robust_se_lags=1", "emit_robust_se", robust_se_lags=1)
    knob("robust_se_lags=3", "emit_robust_se", robust_se_lags=3)
    knob("ljung_box_lags=1", "emit_specification", ljung_box_lags=1)
    knob("ljung_box_lags=3", "emit_specification", ljung_box_lags=3)
    for reader in ("emit_calibration", "emit_breaks", "emit_specification", "emit_robust_se"):
        knob(f"horizon_rows=3 ({reader})", reader, horizon_rows=3)
    c.append(Case("emit_selected", {"emit_selected": True}, grid=True))
    c.append(Case("emit_averaged", {"emit_averaged": True}, grid=True))
    for eta in (0.1, 10.0, math.inf):
        knob(f"average_eta={eta}", "emit_averaged", average_eta=eta)
    # The gates: held to what a gate promises (the module's docstring).
    c.append(Case("min_weight=50", {"min_weight": 50.0}, gate=True))
    c.append(Case("min_settled_frac=0.5", {"min_settled_frac": 0.5}, gate=True))
    c.append(Case("max_error_inflation=1.05", {"max_error_inflation": 1.05}, gate=True))
    # `ewridge`'s gate is on at sqrt(2) by default; elsewhere inf is the default.
    c.append(
        Case(
            "max_error_inflation=inf",
            {"max_error_inflation": math.inf},
            gate=True,
            only=frozenset({"ewridge"}),
        )
    )
    return c


CASES = {c.name: c for c in _cases()}


def _everything(kind: str) -> dict[str, Any]:
    """Every switch the kind takes, at once, with a knob off its default."""
    kw: dict[str, Any] = {}
    for c in CASES.values():
        if c.gate or c.grid or c.parent is not None or not c.taken_by(kind):
            continue
        kw |= c.kw
    return kw | dict(conformal_rate=0.3, resid_autocorr_lag=2, horizon_rows=3)


# ---------------------------------------------------------------------------
# The streams


@functools.cache
def _frame() -> pl.DataFrame:
    r = np.random.default_rng(278_06)
    x = r.normal(size=(N, 2))
    # An AR(2) residual, heteroskedastic in x0, so the autocorrelation lags,
    # the sandwich and the specification tests read something.
    e = r.standard_t(5, size=N)
    u = np.zeros(N)
    for i in range(2, N):
        u[i] = 0.1 * u[i - 1] + 0.5 * u[i - 2] + e[i]
    y = 0.3 + x @ np.array([1.2, -0.7]) + u * (1.0 + 0.3 * np.abs(x[:, 0]))
    # A level shift of 8 noise sd, so the drift detector fires on every kind.
    y[SHIFT_AT:] += 8.0 * u.std()
    y2 = -0.5 + x @ np.array([0.4, 0.9]) + r.normal(size=N)
    y2[r.random(N) < 0.05] = np.nan
    w = r.uniform(0.5, 1.5, size=N)
    w[r.random(N) < 0.03] = 0.0
    # An irregular clock: steps on [0.5, 1.5], and a step of 12 (past the
    # cap of 5) every 400 rows.
    steps = r.uniform(0.5, 1.5, size=N)
    steps[::400] = 12.0
    session = np.repeat([0, 1, 2], [700, 700, N - 1400])
    return pl.DataFrame(
        {
            "t": np.cumsum(steps),
            "s": session,
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": y,
            "y2": pl.Series(y2).fill_nan(None),
            "w": w,
        }
    )


#: Each stream's keywords on the base spec.
VARIANTS: dict[str, dict[str, Any]] = {
    "rows": dict(weight="w"),
    "clocked": dict(weight="w", clock="t", gap_cap=5.0, session="s", session_gap=10.0, embargo=2.0),
}


def _spec(build: Builder, kind: str, variant: str, kw: dict[str, Any], grid: bool) -> dict:
    full: dict[str, Any] = dict(
        targets=["y", "y2"],
        half_life=[60.0, 240.0] if grid else 120.0,
        coef_every=0,
        **VARIANTS[variant],
    )
    if kind not in NO_FEATURES:
        full["features"] = ["x0", "x1"]
    full |= kw
    if variant == "clocked" and full.get("emit_drift") and "drift_threshold" not in full:
        # On a clock, the threshold is in sigma x clock units and has no
        # default; 20 is the row default's value.
        full["drift_threshold"] = 20.0
    return build(**full)


def _builder(kind: str) -> Builder:
    return KINDS[kind]


@functools.cache
def _run_cached(kind: str, variant: str, key: str, grid: bool) -> pl.DataFrame:
    kw = json.loads(key, parse_constant=lambda c: {"Infinity": math.inf}[c])
    spec = _spec(_builder(kind), kind, variant, kw, grid)
    with warnings.catch_warnings():
        # A clocked embargo without `horizon_rows` says so once per reader.
        warnings.simplefilter("ignore", po.ReadinessWarning)
        out = po.ModelBank([spec], threads=1).fit_predict(_frame())
    return out.select("m").unnest("m")


def run(kind: str, variant: str, kw: dict[str, Any], grid: bool = False) -> pl.DataFrame:
    return _run_cached(kind, variant, json.dumps(kw, sort_keys=True), grid)


# ---------------------------------------------------------------------------
# Comparing to the bit

#: The field kinds a spec writes with no diagnostic on.
FIT_FIELDS = ("weight_sum", "settled_frac", "withheld_reason", "coef", "support_coef")
FIT_PREFIXES = ("pred_", "resid_", "penalty_selected_")
#: The fields only the fit moves, which a gate must leave alone on every row.
LEARNED = ("weight_sum", "settled_frac", "coef", "support_coef")


def _stem(name: str) -> str:
    return name.split("@", 1)[0]


def _fit_fields(base: pl.DataFrame) -> list[str]:
    names = base.columns
    for n in names:
        assert _stem(n) in FIT_FIELDS or n.startswith(FIT_PREFIXES), (
            f"{n}: a field the plain spec writes that is not a fit field; "
            f"classify it in FIT_FIELDS or FIT_PREFIXES"
        )
    return names


def _bits(s: pl.Series) -> tuple:
    """A series as its exact contents: floats by their bits (so -0.0 and
    0.0, and two NaNs of different payloads, differ), lists by their
    lengths and their values' bits, anything else by value."""
    if s.dtype == pl.Float64:
        a = s.to_numpy(allow_copy=True)
        mask = s.is_null().to_numpy()
        return ("f", mask.tobytes(), np.where(mask, 0.0, a).view(np.uint64).tobytes())
    if isinstance(s.dtype, pl.List):
        return ("l", _bits(s.list.len()), _bits(s.explode()))
    return ("v", tuple(s.to_list()))


def _first_difference(a: pl.Series, b: pl.Series) -> str:
    av, bv = a.to_list(), b.to_list()
    for i, (p, q) in enumerate(zip(av, bv, strict=True)):
        if p != q or repr(p) != repr(q):
            return f"row {i}: {p!r} vs {q!r}"
    return "the values compare equal but their bits differ"


def assert_same(base: pl.DataFrame, other: pl.DataFrame, names: list[str], what: str) -> None:
    for n in names:
        assert n in other.columns, f"{what}: {n} is gone"
        if _bits(base[n]) != _bits(other[n]):
            raise AssertionError(
                f"{what}: {n} moved, first at {_first_difference(base[n], other[n])}"
            )


def _withheld(out: pl.DataFrame) -> pl.Series:
    """Per row, whether any instance's gate withheld it."""
    cols = [n for n in out.columns if _stem(n) == "withheld_reason"]
    return out.select(pl.any_horizontal(pl.col(c).is_not_null() for c in cols)).to_series()


def _diagnostic_fields(out: pl.DataFrame, fit: list[str]) -> dict[str, tuple]:
    return {n: _bits(out[n]) for n in out.columns if n not in fit}


#: The pairs where a switch moves a fit field and nothing documents it: each
#: a defect, a strict xfail running the real assertion until it is fixed.
DEFECTS: dict[tuple[str, str], str] = {
    ("quantile", "min_weight=50"): (
        "quantile: min_weight moves coef (first at row 13 on `rows`, 0.0038 against "
        "-0.2214 on x0's slope). A row min_weight withholds teaches the residual "
        "scale s nothing (robust.rs: s learns only from a row with a prediction), and "
        "the band h = quantile_eps * s and the first solve (on the row the weight "
        "reaches min_weight) follow it. min_settled_frac and max_error_inflation "
        "withhold rows without moving the fit; the builder does not document the "
        "difference, as kalman's and lasso's do. Measured 2026-10-25."
    ),
    # The Huber legs share the cause (a residual scale learned only from
    # rows with a prediction); measured on main at b45fa81, 2026-10-25, both
    # streams, first at rows 5-9, and still so with the refit of the first
    # rows on (task 277, re-measured when they joined KINDS).
    ("huber", "min_weight=50"): "huber: min_weight moves coef, as quantile's does",
    ("sgd_huber", "min_weight=50"): "sgd(loss='huber'): min_weight moves coef, as quantile's does",
}


# ---------------------------------------------------------------------------
# The sweep


def _pairs(kinds: Callable[[str], bool]) -> list[Any]:
    params = []
    for kind in KINDS:
        if not kinds(kind):
            continue
        marks = (
            ()
            if kind in ESSENTIAL_KINDS
            else (pytest.mark.extended(reason="a grid: the other six linear kinds"),)
        )
        for variant in VARIANTS:
            for case in CASES.values():
                if case.swept_on(kind):
                    defect = DEFECTS.get((kind, case.name))
                    more = (
                        (pytest.mark.xfail(strict=True, raises=AssertionError, reason=defect),)
                        if defect
                        else ()
                    )
                    params.append(
                        pytest.param(
                            kind,
                            variant,
                            case.name,
                            marks=marks + more,
                            id=f"{kind}-{variant}-{case.name}",
                        )
                    )
    return params


@pytest.mark.parametrize(("kind", "variant", "name"), _pairs(lambda k: True))
def test_a_diagnostic_never_moves_the_fit(kind: str, variant: str, name: str) -> None:
    case = CASES[name]
    base = run(kind, variant, {}, case.grid)
    fit = _fit_fields(base)
    out = run(kind, variant, case.kw, case.grid)
    what = f"{kind} on {variant}, {name}"
    if case.gate:
        _assert_a_gate_withholds_only(kind, case, base, out, fit, what)
        return
    assert_same(base, out, fit, what)
    # Evidence it took effect: fields of its own with a value in them.
    own = [n for n in out.columns if n not in fit]
    values = sum(out[n].drop_nulls().len() for n in own)
    print(f"{what}: {len(own)} fields of its own, {values} non-null values")
    assert own and values, f"{what}: wrote nothing of its own, so the comparison proves nothing"
    if case.parent is not None:
        parent = CASES[case.parent]
        mine = _diagnostic_fields(out, fit)
        theirs = _diagnostic_fields(run(kind, variant, parent.kw, case.grid), fit)
        moved = sorted(n for n in mine.keys() | theirs.keys() if mine.get(n) != theirs.get(n))
        print(f"{what}: against {case.parent} at the default, moved {moved}")
        assert moved, f"{what}: the knob changed nothing against {case.parent}"
    if "emit_drift" in case.kw:
        flags = sum(int(out[n].fill_null(False).sum()) for n in own if n.startswith("drift_"))
        print(f"{what}: the detector fired {flags} times")
        # The knobs' evidence is the move against this case, which fires.
        if case.name == "emit_drift":
            assert flags, f"{what}: the detector never fired"


def _assert_a_gate_withholds_only(
    kind: str, case: Case, base: pl.DataFrame, out: pl.DataFrame, fit: list[str], what: str
) -> None:
    learned = [n for n in fit if _stem(n) in LEARNED or n.startswith("penalty_selected_")]
    assert_same(base, out, learned, what)
    a, b = _withheld(base), _withheld(out)
    print(f"{what}: withheld {int(a.sum())} rows without the gate, {int(b.sum())} with it")
    assert not a.equals(b), f"{what}: the gate withheld the same rows, so it proves nothing"
    # A prediction and its residual are, on each row, the same bits in both
    # runs or null in one; a null only where that run's gate withheld the
    # row. A row can be withheld for one target and not another (each
    # target's own `min_weight`), so a value beside a withheld row is legal.
    for n in (m for m in fit if m.startswith(("pred_", "resid_"))):
        x, y = base[n], out[n]
        both = x.is_not_null() & y.is_not_null()
        assert _bits(x.filter(both)) == _bits(y.filter(both)), (
            f"{what}: {n} moved on a row both runs score, first at "
            f"{_first_difference(x.filter(both), y.filter(both))}"
        )
        for nulls, held, which in (
            (x.is_null() & ~both, a, "without"),
            (y.is_null() & ~both, b, "with"),
        ):
            stray = (nulls & ~held & (x.is_not_null() | y.is_not_null())).arg_true()
            assert stray.len() == 0, (
                f"{what}: {n} null {which} the gate on row {stray[0]}, which no gate withheld"
            )


@pytest.mark.parametrize(
    ("kind", "variant"),
    [
        pytest.param(
            k,
            v,
            marks=()
            if k in ESSENTIAL_KINDS
            else (pytest.mark.extended(reason="a grid: the other six linear kinds"),),
            id=f"{k}-{v}",
        )
        for k in KINDS
        for v in VARIANTS
    ],
)
def test_every_switch_at_once_never_moves_the_fit(kind: str, variant: str) -> None:
    """Every switch the kind takes, together: one diagnostic's state cannot
    leak through another's into the fit."""
    base = run(kind, variant, {})
    fit = _fit_fields(base)
    kw = _everything(kind)
    out = run(kind, variant, kw)
    assert_same(base, out, fit, f"{kind} on {variant}, every switch")
    own = [n for n in out.columns if n not in fit]
    print(f"{kind} on {variant}: every switch writes {len(own)} fields")
    # `holt` has no feature for a spread to move with: its Breusch-Pagan
    # is null by design.
    empty = [
        n
        for n in own
        if out[n].drop_nulls().len() == 0
        and not (kind in NO_FEATURES and n.startswith("breusch_pagan_"))
    ]
    assert not empty, f"{kind} on {variant}: fields with no value at all: {empty}"


# ---------------------------------------------------------------------------
# Keeping the tables honest


def test_every_linear_kind_is_swept():
    """A new regression joins this sweep (``sgd_huber`` is ``sgd``'s Huber
    loss, swept beside it)."""
    swept = set(KINDS) - {"sgd_huber"}
    assert swept == set(REGRESSIONS), set(REGRESSIONS) ^ swept


def test_every_common_diagnostic_parameter_is_a_case():
    """Every common parameter is either swept here or named with the reason
    it is not, so a new diagnostic switch or knob cannot miss the sweep."""
    swept = {k for c in CASES.values() for k in c.kw}
    not_diagnostics = {
        # What the fit is: the stream, its clock, its weights and memory.
        "fit_intercept",
        "clock",
        "half_life",
        "lam",
        "gap_cap",
        "restart_after_step_back",
        "session",
        "session_gap",
        "weight",
        "embargo",
        "group",
        "group_close",
        # When `coef` is written: tests/test_coef_cadence.py.
        "coef_every",
        "max_rows_between_coefs",
        # Restarts the model by design: tests/test_drift_reset.py.
        "drift_action",
    }
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0)
    common = set(spec) - {"name", "model", "targets", "features"}
    assert common - swept - not_diagnostics == set(), "classify these"
    assert (swept | not_diagnostics) - common == set(), "names no spec takes"


@pytest.mark.parametrize("kind", sorted(ALL))
def test_every_pair_left_out_is_refused(kind: str):
    """``TAKES`` leaves a pair out only where the builder refuses it."""
    for case in CASES.values():
        if case.taken_by(kind):
            continue
        with pytest.raises(ValueError, match=case.needs or next(iter(case.kw))):
            _spec(_builder(kind), kind, "rows", case.kw, case.grid)
