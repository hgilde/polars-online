"""The readiness gates on every kind: they nest, they switch off, and they
never move learning (docs/PLAN.md task 278, batch 17).

Three gates withhold what a model reports until it is ready:
``min_weight`` (the weight the model has seen, hard rule 8),
``min_settled_frac`` (how far the decay window has filled) and
``max_error_inflation`` (the estimation error over the noise, on the four
ridge-system kinds). Each withholds output and nothing else -- the user's
decision of 2026-10-26: ``min_weight`` withholds output only in every
model. So, per kind and per gate:

- **Nesting.** A stricter gate withholds a superset of rows: the rows
  ``min_weight = 3`` withholds are among those ``10`` withholds, and those
  among ``30``'s; likewise ``min_settled_frac`` at 0.25, 0.5 and 0.75, and
  ``max_error_inflation`` at 1.5, 1.2 and 1.05.
- **The boundary.** ``min_weight`` withholds exactly the rows whose
  ``weight_sum`` is below it -- the same weight the gate reads, before the
  row's update -- naming itself, and names itself on no row at or above
  it; ``min_settled_frac`` likewise against ``settled_frac``.
- **Off.** ``min_weight = 0`` names itself on no row but under the kind's
  own floor (``ew_cov``'s two rows: a variance needs two), and on the kinds
  whose default is 0 it is the default to the bit; ``max_error_inflation =
  inf`` names itself on no row, and on ``rls``, ``lasso`` and ``kalman``,
  where the gate is off unless set, it is the default to the bit.
- **Never moves learning.** With and without each gate, the state is the
  same to the bit (its JSON export, which round-trips its msgpack exactly,
  with the gate's own settings left out), and on the rows both runs report
  every field is the same; the fields only the fit moves (``weight_sum``,
  ``settled_frac``, ``coef``, ``penalty_selected_*``) are the same on every
  row. ``tests/test_diagnostic_isolation.py`` sweeps the row fields of the
  eleven linear kinds; here the state of every kind, and the row fields of
  the twelve others.
- **Activation on ``deco`` and ``corrchange``,** which no test showed
  gating before.
- **The residual spread.** The bank's per-slot residual spread -- the
  statistic ``sigma``, ``zscore``, the drift detector's scale,
  ``emit_selected``'s ranking and ``emit_averaged``'s weights all read --
  folds each slot's own prediction on the rows a gate withholds, so no gate
  moves it, nor which slot ``pred_<t>__selected`` takes, nor
  ``pred_<t>__averaged`` on a row the gate withheld no slot of. Under an
  ``embargo`` too, where the release folds what the score kept. (Until task
  278's batch 17 it folded only the residuals the bank reports; a state
  written before schema 64 goes on that way.)

Known defects, each a strict xfail running the real assertion
(``DEFECTS``, ``DRIFT_DEFECT``): ``ewridge`` keeps a residual variance of
its own in its state that learns only from the predictions ``min_weight``
lets through; and the drift detector folds ``|resid| / sigma`` only on the
rows the bank reports, so under ``drift_action="reset"`` a gate that
withholds the rows after a restart hides the next break, and the restart it
would have made moves the fit.

Each case prints and asserts evidence that the gate took effect: it
withholds rows the ungated run reports (or, on ``marginal`` and ``rcov``,
nulls a statistic in the table it reports through).

Two streams: ``rows``, uneven weights with zeros on the row clock; and
``clocked``, the same rows on an irregular clock with steps past
``gap_cap``. Essentials: each kind on ``rows`` with ``min_weight``, the
nesting on ``ewridge``, the spread and the selection on ``ewridge``, the
defects on ``ewridge``. Extended: the rest of the grid.
"""

from __future__ import annotations

import functools
import json
import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_every_kind import frame_for
from test_model_registry import MINIMAL, REGRESSIONS

TIER = "mixed"

N = 300
HALF_LIFE = 40.0

#: Every kind but ``audit``, which refuses every gate (it gates nothing).
KINDS = sorted(set(MINIMAL) - {"audit"})
#: Per kind, keywords beyond ``MINIMAL`` so a gate shows past the model's
#: own warm-up: ``hmm`` and ``kmeans`` seed from 1,000 and 500 rows by
#: default, longer than the stream.
EXTRA: dict[str, dict[str, Any]] = {
    "hmm": {"warm_rows": 20},
    "kmeans": {"warm_rows": 20},
}
#: The kinds that write no row of their own: they report through the state
#: (``marginal``'s table, ``rcov``'s closed blocks).
TABLE_KINDS = frozenset({"marginal", "rcov"})
ROW_KINDS = [k for k in KINDS if k not in TABLE_KINDS]
#: The kinds with no decay: ``min_settled_frac`` has nothing to settle.
NO_DECAY = frozenset({"seqtest", "bocpd", "corrchange", "rcov"})
SETTLED_KINDS = [k for k in KINDS if k not in NO_DECAY]
#: The kinds whose ridge system reads ``max_error_inflation``.
INFLATION_KINDS = ["ewridge", "rls", "lasso", "kalman"]
#: ``max_error_inflation`` is off unless set on these; ``ewridge``'s is
#: ``sqrt(2)``.
INFLATION_OFF_BY_DEFAULT = ("rls", "lasso", "kalman")
#: A floor of the kind's own that ``min_weight`` cannot lower: a variance
#: needs two rows (``tests/test_readiness.py``,
#: ``test_a_models_own_floor_and_warm_up_say_why``).
OWN_FLOOR: dict[str, float] = {"ew_cov": 2.0}
#: The kinds whose ``min_weight`` defaults to 0 (``spec.rs``,
#: ``default_min_periods``), where 0 is the default to the bit.
DEFAULT_ZERO = frozenset({"ewridge", "seqtest", "rcov", "hmm", "corrchange"})

#: The fields only the fit moves: a gate leaves them alone on every row.
LEARNED = ("weight_sum", "settled_frac", "coef", "support_coef")
LEARNED_PREFIXES = ("penalty_selected_",)
#: The gates' own settings, in the spec and in each model's configuration:
#: left out of the state comparison, which is about what was learned.
GATE_KEYS = frozenset(
    {"min_weight", "target_min_weight", "min_settled_frac", "max_error_inflation"}
)

#: The values each grid steps through, loosest first.
MIN_WEIGHTS = (3.0, 10.0, 30.0)
SETTLED_FRACS = (0.25, 0.5, 0.75)
INFLATIONS = (1.5, 1.2, 1.05)

VARIANTS: dict[str, dict[str, Any]] = {
    "rows": {},
    "clocked": dict(clock="t", gap_cap=5.0),
}


# ---------------------------------------------------------------------------
# The stream and the runs


@functools.cache
def _frame(kind: str) -> pl.DataFrame:
    df = frame_for(kind, n=N)
    r = np.random.default_rng(278_17)
    w = r.uniform(0.5, 1.5, size=N)
    w[r.random(N) < 0.05] = 0.0
    # An irregular clock: steps on [0.5, 1.5], and a step of 12 (past the
    # cap of 5) every 100 rows.
    steps = r.uniform(0.5, 1.5, size=N)
    steps[::100] = 12.0
    # `rcov` weighs a row 0 or 1: a return or none.
    return df.with_columns(w=w, w01=(w > 0).astype(float), t=np.cumsum(steps))


#: The kinds that report no coefficients, so take no ``coef_every``; the
#: rest write ``coef`` on every row, so the fit is compared on every row.
NO_COEF = frozenset({"seqtest", "bocpd", "corrchange", "ew_cov", "marginal", "rcov"})


def _weight_column(kind: str) -> str | None:
    if kind == "seqtest":
        return None  # every learned row is one trial
    return "w01" if kind == "rcov" else "w"


def _spec(kind: str, variant: str, gate: dict[str, Any]) -> dict:
    kw: dict[str, Any] = dict(targets=["y"], features=["x0", "x1"], half_life=HALF_LIFE)
    kw |= MINIMAL[kind]
    kw |= EXTRA.get(kind, {})
    kw["weight"] = _weight_column(kind)
    kw["coef_every"] = None if kind in NO_COEF else 0
    kw |= VARIANTS[variant]
    kw |= gate
    return getattr(po.spec, kind)("m", **{k: v for k, v in kw.items() if v is not None})


def _key(gate: dict[str, Any]) -> str:
    return json.dumps(gate, sort_keys=True)


@functools.cache
def _run_cached(kind: str, variant: str, key: str) -> tuple[pl.DataFrame, dict, pl.DataFrame]:
    gate = json.loads(key, parse_constant=lambda c: {"Infinity": math.inf}[c])
    bank = po.ModelBank([_spec(kind, variant, gate)], threads=1)
    with warnings.catch_warnings():
        # A floor the stream never meets says so once.
        warnings.simplefilter("ignore", po.ReadinessWarning)
        out = bank.fit_predict(_frame(kind))
    table = _table(kind, bank)
    return out.select("m").unnest("m"), json.loads(bank.write_json()), table


def _table(kind: str, bank: po.ModelBank) -> pl.DataFrame:
    """What a kind with no row of its own reports: ``marginal``'s table,
    ``rcov``'s closed blocks; an empty frame elsewhere."""
    if kind == "marginal":
        return bank.marginal("m")
    if kind == "rcov":
        return bank.closed_groups(drop=False)
    return pl.DataFrame()


def run(kind: str, variant: str, gate: dict[str, Any]) -> pl.DataFrame:
    return _run_cached(kind, variant, _key(gate))[0]


def state(kind: str, variant: str, gate: dict[str, Any]) -> dict:
    return _run_cached(kind, variant, _key(gate))[1]


def table(kind: str, variant: str, gate: dict[str, Any]) -> pl.DataFrame:
    return _run_cached(kind, variant, _key(gate))[2]


def _reasons(out: pl.DataFrame) -> list[str | None]:
    return out["withheld_reason"].cast(pl.String).to_list()


def _withheld(out: pl.DataFrame) -> set[int]:
    return {i for i, r in enumerate(_reasons(out)) if r is not None}


def _named(out: pl.DataFrame, reason: str) -> set[int]:
    return {i for i, r in enumerate(_reasons(out)) if r == reason}


def _tier(essential: bool) -> tuple:
    return () if essential else (pytest.mark.extended(reason="a grid: every kind and gate"),)


# ---------------------------------------------------------------------------
# Comparing to the bit


def _bits(s: pl.Series) -> tuple:
    """A series as its exact contents: floats by their bits, lists by their
    lengths and their values' bits, anything else by value."""
    if s.dtype == pl.Float64:
        a = s.to_numpy(allow_copy=True)
        mask = s.is_null().to_numpy()
        return ("f", mask.tobytes(), np.where(mask, 0.0, a).view(np.uint64).tobytes())
    if isinstance(s.dtype, pl.List | pl.Array):
        return (
            "l",
            _bits(s.list.len() if isinstance(s.dtype, pl.List) else s.arr.len()),
            _bits(s.explode()),
        )
    return ("v", tuple(s.to_list()))


def _first_difference(a: pl.Series, b: pl.Series) -> str:
    for i, (p, q) in enumerate(zip(a.to_list(), b.to_list(), strict=True)):
        if p != q or repr(p) != repr(q):
            return f"row {i}: {p!r} vs {q!r}"
    return "the values compare equal but their bits differ"


def _learned(name: str) -> bool:
    return name.split("@", 1)[0] in LEARNED or name.startswith(LEARNED_PREFIXES)


def _strip(node: Any, drop: frozenset[str]) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip(v, drop) for v in node]
    return node


def _differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Every path where two JSON trees differ. The export writes each float
    as the shortest text that reads back to its bits, so equal text is
    equal bits."""
    if type(a) is not type(b):
        return [path or "/"]
    if isinstance(a, dict):
        return [
            d
            for k in sorted(a.keys() | b.keys())
            for d in _differences(a.get(k), b.get(k), f"{path}/{k}")
        ]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}[len]"]
        return [
            d
            for i, (p, q) in enumerate(zip(a, b, strict=True))
            for d in _differences(p, q, f"{path}[{i}]")
        ]
    return [] if a == b else [path]


#: The bank's per-slot residual spread, compared in the state with the rest
#: and on its own (``test_the_residual_spread_never_moves_with_a_gate``).
SPREAD = frozenset({"resid_var", "resid_w"})
#: What the state keeps of the output: the last row's fields (``last_row``,
#: so a loaded bank reports them) and the closed groups not yet read
#: (``closed``). They are what a gate withholds, so they are compared as
#: output -- the rows and the tables -- not as what was learned.
OUTPUT_RECORDS = frozenset({"last_row", "closed"})


def assert_the_state_is_the_same(base: dict, gated: dict, what: str, drop: frozenset[str]) -> None:
    moved = _differences(_strip(base, GATE_KEYS | drop), _strip(gated, GATE_KEYS | drop))
    assert not moved, f"{what}: the gate moved the state at {moved[:6]}"


def assert_rows_agree(base: pl.DataFrame, gated: pl.DataFrame, what: str) -> None:
    """The fit's fields on every row; every other field on the rows both
    runs report, a null only where that run withheld the row."""
    assert base.columns == gated.columns, what
    held = {"base": _withheld(base), "gated": _withheld(gated)}
    for n in base.columns:
        if n == "withheld_reason":
            continue
        x, y = base[n], gated[n]
        if _learned(n):
            assert _bits(x) == _bits(y), f"{what}: {n} moved, first at {_first_difference(x, y)}"
            continue
        both = x.is_not_null() & y.is_not_null()
        assert _bits(x.filter(both)) == _bits(y.filter(both)), (
            f"{what}: {n} moved on a row both report, first at "
            f"{_first_difference(x.filter(both), y.filter(both))}"
        )
        for nulls, run_, other in ((x.is_null(), "base", y), (y.is_null(), "gated", x)):
            stray = [
                i for i in (nulls & other.is_not_null()).arg_true().to_list() if i not in held[run_]
            ]
            assert not stray, (
                f"{what}: {n} null in the {run_} run on row {stray[0]}, which it did not withhold"
            )


# ---------------------------------------------------------------------------
# min_weight: nesting, the boundary, and off


@pytest.mark.parametrize(
    ("kind", "variant"),
    [
        pytest.param(k, v, marks=_tier(v == "rows"), id=f"{k}-{v}")
        for k in ROW_KINDS
        for v in VARIANTS
    ],
)
def test_min_weight_withholds_exactly_the_rows_below_it_and_they_nest(kind: str, variant: str):
    """At each ``min_weight`` the rows named ``below_min_weight`` are among
    the rows whose ``weight_sum`` is below it, every such row is withheld (a
    gate before it in precedence may name it), and its value fields are
    null; the withheld rows grow with the floor. At 0 the gate names no row
    but under the kind's own floor."""
    floor = OWN_FLOOR.get(kind, 0.0)
    previous: set[int] = set()
    for m in (0.0, *MIN_WEIGHTS):
        out = run(kind, variant, {"min_weight": m})
        ws = out["weight_sum"].to_list()
        bar = max(m, floor)
        below = {i for i, w in enumerate(ws) if w < bar}
        named, held = _named(out, "below_min_weight"), _withheld(out)
        what = f"{kind} on {variant}, min_weight={m}"
        print(f"{what}: {len(below)} rows below, {len(named)} named, {len(held)} withheld")
        assert named <= below, (
            f"{what}: named at or above the floor on rows {sorted(named - below)[:5]}"
        )
        assert below <= held, f"{what}: rows below the floor reported: {sorted(below - held)[:5]}"
        values = [c for c in out.columns if c not in ("withheld_reason",) and not _learned(c)]
        for c in values:
            leak = [i for i in named if out[c][i] is not None]
            assert not leak, f"{what}: {c} reported on withheld row {leak[0]}"
        assert previous <= held, (
            f"{what}: rows withheld at a lower floor reported: {sorted(previous - held)[:5]}"
        )
        previous = held
        if m == MIN_WEIGHTS[-1]:
            # Evidence the grid reached past the model's own warm-up.
            assert len(named) > len(
                _named(run(kind, variant, {"min_weight": 0.0}), "below_min_weight")
            ), what


@pytest.mark.parametrize(
    "kind", [pytest.param(k, marks=_tier(True)) for k in sorted(DEFAULT_ZERO - {"rcov"})]
)
def test_min_weight_zero_is_the_default_where_the_default_is_zero(kind: str):
    base, zero = run(kind, "rows", {}), run(kind, "rows", {"min_weight": 0.0})
    for n in base.columns:
        assert _bits(base[n]) == _bits(zero[n]), f"{kind}: {n} moved"


# ---------------------------------------------------------------------------
# A gate never moves learning


def _gates(kind: str) -> list[tuple[str, dict[str, Any]]]:
    if kind in TABLE_KINDS:
        # Read once, after the stream: a floor the end of the stream has
        # not met, so the table shows the gate acting.
        g = [("min_weight=1e9", {"min_weight": 1e9})]
        if kind in SETTLED_KINDS:
            g.append(("min_settled_frac=0.999", {"min_settled_frac": 0.999}))
        return g
    g = [("min_weight=30", {"min_weight": 30.0})]
    if kind in SETTLED_KINDS:
        g.append(("min_settled_frac=0.5", {"min_settled_frac": 0.5}))
    if kind in INFLATION_KINDS:
        g.append(("max_error_inflation=1.05", {"max_error_inflation": 1.05}))
    return g


#: The (kind, gate) pairs where a gate moves the model's state: each a
#: defect, a strict xfail running the real assertion until it is fixed.
DEFECTS: dict[tuple[str, str], str] = {
    # `ewridge` keeps a per-target residual variance in its state (`sig2`,
    # weighed by `wsig`), folded from its own prediction, which it withholds
    # below `min_weight`: the floor moves it (first at the first row the
    # floor withholds and the default reports). No output reads it today
    # (`EwRidge::sigma2` has only test callers), so no field moves.
    ("ewridge", "min_weight=30"): "ewridge's state residual variance learns only past min_weight",
}

#: The defect behind a drift reset under a gate.
DRIFT_DEFECT = (
    "the drift detector folds |resid| / sigma only on the rows the bank reports, so a gate "
    "that withholds the rows after a restart hides the next break, and drift_action='reset''s "
    "restart with it"
)


def _gate_params(kinds: list[str]) -> list[Any]:
    params = []
    for kind in kinds:
        for variant in VARIANTS:
            for name, _gate in _gates(kind):
                essential = variant == "rows" and name.startswith("min_weight")
                defect = DEFECTS.get((kind, name))
                more = (
                    (pytest.mark.xfail(strict=True, raises=AssertionError, reason=defect),)
                    if defect
                    else ()
                )
                params.append(
                    pytest.param(
                        kind,
                        variant,
                        name,
                        marks=_tier(essential) + more,
                        id=f"{kind}-{variant}-{name}",
                    )
                )
    return params


def _gate_of(kind: str, name: str) -> dict[str, Any]:
    return dict(_gates(kind))[name]


def _evidence(kind: str, variant: str, gate: dict[str, Any], what: str) -> None:
    """The gate withheld something the ungated run reports."""
    if kind in TABLE_KINDS:
        a, b = table(kind, variant, {}), table(kind, variant, gate)
        moved = [c for c in a.columns if _bits(a[c]) != _bits(b[c])]
        print(f"{what}: the table's columns the gate nulled: {moved}")
        assert moved, f"{what}: the gate nulled nothing, so the comparison proves nothing"
        for c in moved:
            assert b[c].null_count() > a[c].null_count(), f"{what}: {c} moved, and not to null"
        return
    a, b = _withheld(run(kind, variant, {})), _withheld(run(kind, variant, gate))
    print(f"{what}: withheld {len(a)} rows without the gate, {len(b)} with it")
    assert a != b, f"{what}: the gate withheld the same rows, so the comparison proves nothing"


@pytest.mark.parametrize(("kind", "variant", "name"), _gate_params(KINDS))
def test_a_gate_never_moves_the_state(kind: str, variant: str, name: str):
    """The state after the stream, with and without the gate, is the same
    to the bit: every model's and the bank's -- the residual spread with
    the rest -- but the gate's own settings and the output the state keeps
    (``OUTPUT_RECORDS``)."""
    gate = _gate_of(kind, name)
    what = f"{kind} on {variant}, {name}"
    _evidence(kind, variant, gate, what)
    assert_the_state_is_the_same(
        state(kind, variant, {}), state(kind, variant, gate), what, OUTPUT_RECORDS
    )


@pytest.mark.parametrize(
    ("kind", "variant", "name"),
    _gate_params([k for k in ROW_KINDS if k not in REGRESSIONS]),
)
def test_a_gate_never_moves_the_rows(kind: str, variant: str, name: str):
    """Every row field of the kinds ``tests/test_diagnostic_isolation.py``
    does not sweep: the fit's on every row, the rest where both report."""
    gate = _gate_of(kind, name)
    what = f"{kind} on {variant}, {name}"
    _evidence(kind, variant, gate, what)
    assert_rows_agree(run(kind, variant, {}), run(kind, variant, gate), what)


# ---------------------------------------------------------------------------
# min_settled_frac


@pytest.mark.parametrize(
    ("kind", "variant"),
    [
        pytest.param(k, v, marks=_tier(k == "ewridge" and v == "rows"), id=f"{k}-{v}")
        for k in SETTLED_KINDS
        if k not in TABLE_KINDS
        for v in VARIANTS
    ],
)
def test_min_settled_frac_withholds_exactly_the_rows_below_it_and_they_nest(
    kind: str, variant: str
):
    """The rows named ``below_min_settled_frac`` are exactly those whose
    ``settled_frac`` is below the fraction (the gate is first in
    precedence), and they grow with it."""
    previous: set[int] = set()
    for f in SETTLED_FRACS:
        out = run(kind, variant, {"min_settled_frac": f})
        below = {i for i, s in enumerate(out["settled_frac"].to_list()) if s < f}
        named, held = _named(out, "below_min_settled_frac"), _withheld(out)
        what = f"{kind} on {variant}, min_settled_frac={f}"
        print(f"{what}: {len(below)} rows below, {len(held)} withheld")
        assert below, f"{what}: no row below the fraction, so it proves nothing"
        assert named == below, f"{what}: named {sorted(named ^ below)[:5]} against the fraction"
        assert previous <= held, f"{what}: rows withheld at a lower fraction reported"
        previous = held


# ---------------------------------------------------------------------------
# max_error_inflation


@pytest.mark.parametrize(
    ("kind", "variant"),
    [
        pytest.param(k, v, marks=_tier(k == "ewridge" and v == "rows"), id=f"{k}-{v}")
        for k in INFLATION_KINDS
        for v in VARIANTS
    ],
)
def test_max_error_inflation_nests_and_switches_off_at_inf(kind: str, variant: str):
    """A tighter ratio withholds a superset of rows; ``inf`` names the gate
    on no row, and where the gate is off unless set it is the default to
    the bit."""
    previous: set[int] = set()
    counts = []
    for r in INFLATIONS:
        out = run(kind, variant, {"max_error_inflation": r})
        held = _withheld(out)
        counts.append(len(_named(out, "above_max_error_inflation")))
        assert previous <= held, (
            f"{kind} on {variant}, {r}: rows withheld at a looser ratio reported"
        )
        previous = held
    print(f"{kind} on {variant}: named above the ratio at {INFLATIONS}: {counts}")
    assert counts[-1] > 0, f"{kind} on {variant}: 1.05 withheld nothing, so it proves nothing"
    off = run(kind, variant, {"max_error_inflation": math.inf})
    assert not _named(off, "above_max_error_inflation"), f"{kind} on {variant}: inf withheld"
    if kind in INFLATION_OFF_BY_DEFAULT:
        base = run(kind, variant, {})
        for n in base.columns:
            assert _bits(base[n]) == _bits(off[n]), f"{kind} on {variant}: inf moved {n}"
    else:
        assert _withheld(off) < _withheld(run(kind, variant, {})), "sqrt(2) is the default"


# ---------------------------------------------------------------------------
# Activation where no test showed it


@pytest.mark.parametrize("kind", ["deco", "corrchange"])
def test_min_weight_gates_deco_and_corrchange(kind: str):
    """``min_weight`` names itself on exactly the rows whose weight is below
    it, every value field null there; and it withholds rows the default
    reports: the first row with a value moves later as the floor rises
    (``corrchange``'s statistic waits for its windows to fill as well, so
    a floor below that changes nothing it reports)."""
    firsts = []
    for m in (None, 10.0, 30.0, 45.0):
        out = run(kind, "rows", {} if m is None else {"min_weight": m})
        values = [c for c in out.columns if c != "withheld_reason" and not _learned(c)]
        reported = out.select(
            pl.any_horizontal(pl.col(c).is_not_null() for c in values)
        ).to_series()
        firsts.append(reported.arg_true()[0])
        if m is not None:
            ws = out["weight_sum"].to_list()
            named = _named(out, "below_min_weight")
            assert named == {i for i, w in enumerate(ws) if w < m}, f"{kind}, {m}"
            assert not any(reported[i] for i in named), f"{kind}, {m}: a value on a withheld row"
    print(f"{kind}: first row with a value at the default, 10, 30 and 45: {firsts}")
    assert firsts == sorted(firsts) and firsts[0] < firsts[2] < firsts[3], firsts


# ---------------------------------------------------------------------------
# The residual spread and what reads it


def _grid_spec(kind: str, gate: dict[str, Any], **kw: Any) -> dict:
    base: dict[str, Any] = dict(targets=["y"], features=["x0", "x1"], half_life=[20.0, 200.0])
    base |= {k: v for k, v in MINIMAL[kind].items() if k != "half_life"}
    base |= kw | gate
    return getattr(po.spec, kind)("m", **{k: v for k, v in base.items() if v is not None})


@functools.cache
def _grid_frame() -> pl.DataFrame:
    r = np.random.default_rng(17)
    x = r.normal(size=(400, 2))
    y = x @ np.array([1.0, -0.5]) + r.normal(size=400)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


SELECTION_KINDS = ("ewridge", "rls", "kalman", "sgd", "holt")


def _selection_params() -> list[Any]:
    params = []
    for kind in SELECTION_KINDS:
        gates = [
            ("min_weight=10", {"min_weight": 10.0}),
            ("min_settled_frac=0.5", {"min_settled_frac": 0.5}),
        ]
        if kind in INFLATION_KINDS:
            gates.append(("max_error_inflation=1.05", {"max_error_inflation": 1.05}))
        for name, gate in gates:
            essential = kind == "ewridge" and name == "min_weight=10"
            params.append(pytest.param(kind, gate, marks=_tier(essential), id=f"{kind}-{name}"))
    return params


@pytest.mark.parametrize(("kind", "gate"), _selection_params())
def test_a_gate_never_moves_which_slot_is_selected(kind: str, gate: dict[str, Any]):
    """``pred_<t>__selected`` and ``selected_<t>`` rank the half-life grid's
    slots by each slot's EW out-of-sample squared error, its ``sigma``, and
    ``pred_<t>__averaged`` weighs them by it. With and without a gate every
    slot's ``sigma`` is the same on every row, so on every row the gate
    withheld no slot of, the three are the same. (A row it withheld a slot
    of selects and averages among the slots it reports: that is the gate
    withholding output.) Every slot reaches each gate here (the weight at a
    half-life of 20 tops out near 29), so a withheld row is the gate's, not
    a slot that never fits."""
    kw: dict[str, Any] = dict(emit_selected=True, emit_averaged=True, emit_sigma=True)
    if kind == "holt":
        kw["features"] = None
    if kind == "sgd":
        kw["learning_rate"] = 0.02
    df = _grid_frame()
    base = (
        po.ModelBank([_grid_spec(kind, {}, **kw)], threads=1).fit_predict(df)["m"].struct.unnest()
    )
    gated = (
        po.ModelBank([_grid_spec(kind, gate, **kw)], threads=1).fit_predict(df)["m"].struct.unnest()
    )
    # Evidence: the gate withheld rows the base reports, and the slots'
    # own predictions and the fit are the same (the diagnostic isolation
    # sweep), so only the ranking can move the selection.
    slots = [c for c in base.columns if c.startswith("pred_y@")]
    sigmas = [c for c in base.columns if c.startswith("sigma_y@")]
    assert slots and all(base[c].null_count() < gated[c].null_count() for c in slots), slots
    for c in sigmas:
        assert _bits(base[c]) == _bits(gated[c]), (
            f"{kind}, {gate}: {c} moved, first at {_first_difference(base[c], gated[c])}"
        )
    every = gated.select(pl.all_horizontal(pl.col(c).is_not_null() for c in slots)).to_series()
    for n in ("pred_y__selected", "selected_y", "pred_y__averaged"):
        x, y = base[n], gated[n]
        both = every & x.is_not_null() & y.is_not_null()
        # `min_settled_frac = 0.5` holds the half-life of 200 back for its
        # first 200 rows of the 400; every other gate, a few dozen.
        assert int(both.sum()) > 150, n
        assert _bits(x.filter(both)) == _bits(y.filter(both)), (
            f"{kind}, {gate}: {n} moved on a row the gate withheld no slot of, first at "
            f"{_first_difference(x.filter(both), y.filter(both))}"
        )


#: An `embargo` on the clocked stream: a held row is scored, its record
#: kept, and the spread folds it at the release.
EMBARGO: dict[str, Any] = dict(clock="t", gap_cap=5.0, embargo=3.0)


@pytest.mark.parametrize(
    ("kind", "name", "held"),
    [
        pytest.param(
            k,
            name,
            held,
            marks=_tier(k == "ewridge" and name == "min_weight=30"),
            id=f"{k}-{name}" + ("-embargo" if held else ""),
        )
        for k in sorted(REGRESSIONS)
        for name, _ in _gates(k)
        for held in (False, True)
    ],
)
def test_the_residual_spread_never_moves_with_a_gate(kind: str, name: str, held: bool):
    """The bank keeps each slot's EW squared residual in its state
    (``resid_var``, weighed by ``resid_w``), on with no diagnostic switch:
    with and without a gate it is the same to the bit, and so is the
    ``sigma`` read from it on every row. Under an ``embargo`` the release
    folds what the score kept: the record of a score a gate withheld
    carries each slot's own prediction."""
    extra = EMBARGO if held else {}
    gate = _gate_of(kind, name)
    what = f"{kind}, {name}" + (" under an embargo" if held else "")
    a, b = state(kind, "rows", extra), state(kind, "rows", extra | gate)
    keep = lambda s: [st["persisted"][k] for k in sorted(SPREAD) for _key, st in s["states"][0]]  # noqa: E731
    x, y = _withheld(run(kind, "rows", extra)), _withheld(run(kind, "rows", extra | gate))
    print(f"{what}: withheld {len(x)} rows without the gate, {len(y)} with it")
    assert x != y, f"{what}: the gate withheld the same rows, so the comparison proves nothing"
    assert _differences(keep(a), keep(b)) == [], f"{what}: the residual spread moved"


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param({"min_weight": 40.0}, marks=_tier(True), id="min_weight=40"),
        pytest.param(
            {"max_error_inflation": 1.02}, marks=_tier(False), id="max_error_inflation=1.02"
        ),
    ],
)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=DRIFT_DEFECT)
def test_a_gate_never_moves_a_drift_reset(gate: dict[str, Any]):
    """Under ``drift_action="reset"`` the detector restarts the model, so
    it is part of the fit: it reads ``|resid| / sigma`` on the rows the
    bank reports. Breaks at rows 300, 450, 520 and 600: after the restart
    at 450 the gate withholds the rows that would show the break at 520,
    the detector misses it, and the fit moves from row 521 (measured
    2026-10-26: resets at 301, 450, 520, 600 without the gate, 301, 450,
    600 with it)."""
    r = np.random.default_rng(5)
    n = 1200
    x = r.normal(size=(n, 2))
    y = x @ np.array([1.0, -0.5]) + r.normal(size=n)
    for at, shift in ((300, 4.0), (450, -6.0), (520, 5.0), (600, 8.0), (640, -3.0)):
        y[at:] += shift
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})

    def fit(g: dict[str, Any]) -> pl.DataFrame:
        s = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=30.0,
            emit_drift=True,
            drift_action="reset",
            drift_threshold=3.0,
            coef_every=0,
            **g,
        )
        return po.ModelBank([s], threads=1).fit_predict(df)["m"].struct.unnest()

    base, gated = fit({}), fit(gate)
    fired = base["drift_y"].fill_null(False).arg_true().to_list()
    moved = gated["drift_y"].fill_null(False).arg_true().to_list()
    print(f"{gate}: resets without the gate {fired}, with it {moved}")
    assert len(fired) >= 3, "the detector must fire for the case to mean anything"
    for n in ("weight_sum", "coef", "drift_y"):
        assert _bits(base[n]) == _bits(gated[n]), (
            f"{gate}: {n} moved, first at {_first_difference(base[n], gated[n])}"
        )
