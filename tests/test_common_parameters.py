"""The common parameters' equivalences, on every kind (docs/PLAN.md task 278, batch 16).

Each common stream parameter has spellings and limits that must give the
same numbers, and none of them has a third-party oracle: the oracle is the
library itself under the other spelling. Each relation runs on every kind
it applies to, so a kind whose plumbing reads a parameter its own way
shows here, and a kind left out is a named exception with its reason.

The relations:

- ``gap_cap=c`` is the stream with every clock step cut to ``c`` (under a
  cap no step reaches). Exceptions: ``corrchange`` and ``rcov``, for which
  a capped step is a break by design (the span or the block stretch ends
  there); the test holds each exception to differing, so the list cannot
  hide a kind the relation holds for.
- a cap past every step is the same as any larger cap: no step is cut.
- ``half_life=inf`` is ``lam=1``.
- ``lam=0.5 ** (1 / h)`` is ``half_life=h``: ``weight_sum`` to the bit, the
  rest within a stated tolerance (``lam`` reaches the decay through a
  power where ``half_life`` uses an exponent, so the last bits differ).
  The solving kinds run ``solve_every=0`` here: their default cadence
  reads the half-life, and ``lam`` has none to read.
- a half-life grid's instance is the lone spec at that half-life, to the
  bit, field for field.
- a weight column of ones is no weight column.
- weight ``k`` is the row repeated ``k`` times at one clock instant, where
  the model sums weight as a count: ``rls``, ``marginal``, ``ewridge`` with
  its Kish-reading gate off, ``kalman`` with a fixed ``obs_var``. The other
  kinds read a weight differently by design (a Kish correction, a count of
  rows, a step size: ``tests/test_weight_scale.py``'s table).
- the order the features are listed in moves no number: each field is the
  same once the names are mapped.
- a temporal clock with duration parameters is a seconds clock with the
  same parameters as numbers.
- no clock is the row-index clock ``0, 1, 2, ...``.
- ``ModelBank.fit(chunk_size=k)`` leaves the same ``save_bytes()`` at every
  ``k``, and the same as one ``fit_predict`` over the frame.

Each relation prints, and asserts, evidence that the parameter took effect
on the stream it compares (the cap binds, the half-life decays, the grid's
instances differ, the weights move the fit, the chunks are many), so no
case compares two runs that would agree with the parameter ignored.

Essentials: three kinds per relation (``ewridge``, ``ew_cov`` and
``kmeans`` where they apply). Extended: every kind the relation applies
to. ``huber`` and ``sgd(loss="huber")`` wait in ``AFTER_TASK_277``: task
277 is changing their warm-up. Their legs build and run today
(``test_the_waiting_legs_run``), and ``test_the_huber_legs_still_wait``
fails when a new parameter arrives on ``po.spec.huber`` or
``po.spec.sgd``: then the two entries move into ``KINDS``.
"""

from __future__ import annotations

import datetime as dt
import functools
import inspect
import math
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_diagnostic_isolation import BEFORE_277
from test_model_registry import MINIMAL

TIER = "mixed"

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")

N = 400
#: The clock's long steps, and a cap that cuts them.
LONG_STEP = 40.0
CAP = 5.0
HALF_LIFE = 50.0

#: Every kind, each under a name, built from ``MINIMAL`` with the keywords
#: its plainest well-behaved fit needs on this stream. ``corrchange_scalar``
#: is the one ``corrchange`` that decays (its standardiser).
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    kind: (kind, {}) for kind in sorted(MINIMAL) if kind != "huber"
}
# Seed early, so the clustering and the hidden states learn on these rows
# (``hmm``'s default warm-up is longer than the stream).
VARIANTS["kmeans"] = ("kmeans", {"warm_rows": 20})
VARIANTS["hmm"] = ("hmm", {"warm_rows": 20})
VARIANTS["rcov"] = ("rcov", {"block_rows": 50})
VARIANTS["corrchange_scalar"] = ("corrchange", {"scalar": True, "half_life": HALF_LIFE})

#: Task 277 changes these two's warm-up; their legs join ``VARIANTS`` when
#: it merges (``test_the_huber_legs_still_wait`` says when).
AFTER_TASK_277: dict[str, tuple[str, dict[str, Any]]] = {
    "huber": ("huber", {}),
    "sgd_huber": ("sgd", {"loss": "huber"}),
}

KINDS = tuple(VARIANTS)
ESSENTIAL = ("ewridge", "ew_cov", "kmeans")

#: The kinds whose fit decays: a half-life applies.
DECAYING = tuple(
    k
    for k in KINDS
    if k == "corrchange_scalar" or MINIMAL[VARIANTS[k][0]].get("half_life", 1.0) is not None
)
#: The kinds that refuse a weight column.
UNWEIGHTED = frozenset({"seqtest", "audit"})
#: The kinds that read no features.
NO_FEATURES = frozenset({"holt", "seqtest"})
#: A capped step is a break by design: ``corrchange`` abandons its span,
#: ``rcov`` splits the block into stretches.
CAP_IS_A_BREAK = frozenset({"corrchange", "corrchange_scalar", "rcov"})
#: The kinds that sum weight as a count, with what makes it exact: the
#: ewridge readiness gate reads Kish's effective count, which copies change,
#: and its default solve cadence counts rows; kalman's learned noise reads a
#: weight as precision.
COPIES: dict[str, dict[str, Any]] = {
    # Every row solved: the default cadence's row cap counts copies as rows.
    "ewridge": {"max_error_inflation": math.inf, "solve_every": 0},
    "rls": {},
    "marginal": {},
    "kalman": {"obs_var": 1.0},
}
#: The columns of a state's table that report how the decay was spelled
#: (``marginal``'s table names its ``half_life`` or its ``lam``).
SPELLING = ("half_life", "lam")
#: The fields that read Kish's effective count ``(sum w)^2 / sum w^2``,
#: which copies change by design: one row of weight 2 is not two rows.
KISH = {"marginal": ("n_kish", "t_stat")}
#: A state table's clock columns are in the clock's own type.
CLOCK_COLUMNS = ("clock_min", "clock_max")
#: The solving kinds, whose default cadence reads the half-life.
SOLVES = frozenset({"ewridge", "lasso", "quantile", "huber"})


def _params(kinds, essential=ESSENTIAL):
    """Each kind a parameter; extended unless it is one of ``essential``."""
    return [
        k if k in essential else pytest.param(k, marks=pytest.mark.extended(reason="every kind"))
        for k in kinds
    ]


#: The streams a toleranced relation's bound is held over: the essentials'
#: one, and 50 seeds in the extended tier, so the bound is the worst seed's.
SEEDS = 50


def _seeded(kinds, essential=ESSENTIAL):
    """Each (kind, seed): seed 5 on the essential kinds is essential; every
    other pair is extended."""
    ext = pytest.mark.extended(reason="every kind, the worst of 50 seeds")
    return [
        (k, seed) if (k in essential and seed == 5) else pytest.param(k, seed, marks=ext)
        for k in kinds
        for seed in range(SEEDS)
    ]


# --- the stream ------------------------------------------------------------


@functools.cache
def _stream(seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random(N) < 0.05, LONG_STEP, 1.0)
    steps[0] = 0.0
    t = np.cumsum(steps)
    x0, x1 = rng.normal(size=N), rng.normal(size=N)
    shift = np.where(np.arange(N) > N // 2, 3.0, 0.0)
    y = 1.0 + 2.0 * x0 - x1 + 0.3 * rng.normal(size=N) + shift
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": x1,
            "y": y,
            "g": np.arange(N) // 100,
            "w": rng.uniform(0.2, 3.0, N),
        }
    )


STREAM = _stream()


def frame(variant: str, df: pl.DataFrame | None = None) -> pl.DataFrame:
    """The stream as ``variant`` reads it: a label for ``ftrl`` (logistic)
    and ``ew_class``."""
    df = STREAM if df is None else df
    kind = VARIANTS.get(variant, AFTER_TASK_277.get(variant, (variant, {})))[0]
    if kind == "ftrl":
        df = df.with_columns(y=(pl.col("y") > 1).cast(pl.Float64))
    if kind == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 1).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


def spec(variant: str, **kw: Any) -> dict:
    """``variant``'s spec with ``kw`` over it; a ``None`` removes a key."""
    kind, extra = VARIANTS.get(variant) or AFTER_TASK_277[variant]
    a: dict[str, Any] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": HALF_LIFE}
    a.update(MINIMAL[kind])
    a.update(extra)
    a.update(kw)
    return getattr(po.spec, kind)("m", **{k: v for k, v in a.items() if v is not None})


def _state(bank: po.ModelBank, kind: str) -> pl.DataFrame | None:
    """What a kind holds outside its rows: the tables it is read from."""
    if kind == "marginal":
        return bank.marginal("m")
    if kind == "audit":
        return bank.audit("m")
    if kind == "rcov":
        return bank.closed_groups("m", drop=False)
    return None


def run(variant: str, df: pl.DataFrame | None = None, **kw: Any):
    """The rows' fields and the state's table, one chunk, one thread."""
    kind = (VARIANTS.get(variant) or AFTER_TASK_277[variant])[0]
    bank = po.ModelBank([spec(variant, **kw)], threads=1)
    out = bank.fit_predict(frame(variant, df)).select("m").unnest("m")
    return out, _state(bank, kind)


# --- comparing -------------------------------------------------------------


def _leaves(s: pl.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """A float column (or a list of floats) as its values and null mask."""
    if s.dtype == pl.List(pl.Float64):
        lengths = s.list.len().fill_null(-1).to_numpy()
        flat = s.explode()
        v = flat.fill_null(np.nan).to_numpy().astype(np.float64)
        return np.concatenate([lengths.astype(np.float64), v]), np.concatenate(
            [np.zeros(len(lengths), bool), flat.is_null().to_numpy()]
        )
    if s.dtype.is_float():
        return s.fill_null(np.nan).to_numpy().astype(np.float64), s.is_null().to_numpy()
    return None


def worst(a: pl.DataFrame, b: pl.DataFrame, skip: tuple[str, ...] = ()) -> dict[str, float]:
    """Each column's largest relative difference: 0 for the same bits, inf
    for a different null, NaN or non-float value; ``skip`` names prefixes
    left out."""
    assert a.columns == b.columns, (a.columns, b.columns)
    out = {}
    for c in a.columns:
        if c.startswith(skip):
            continue
        la, lb = _leaves(a[c]), _leaves(b[c])
        if la is None or lb is None:
            out[c] = 0.0 if a[c].equals(b[c], null_equal=True) else math.inf
            continue
        (x, nx), (y, ny) = la, lb
        if x.shape != y.shape or (nx != ny).any() or (np.isnan(x) != np.isnan(y)).any():
            out[c] = math.inf
            continue
        ok = ~np.isnan(x)
        x, y = x[ok], y[ok]
        moved = x.view(np.uint64) != y.view(np.uint64)
        if not moved.any():
            out[c] = 0.0
            continue
        x, y = x[moved], y[moved]
        if not (np.isfinite(x) & np.isfinite(y)).all():
            out[c] = math.inf
            continue
        out[c] = float((np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.0)).max())
    return out


def assert_same(a, b, rtol: float = 0.0, skip: tuple[str, ...] = (), label: str = ""):
    """``a`` and ``b`` (each the rows and the state's table) agree: to the
    bit at ``rtol=0``, else each float within ``rtol`` (relative, floored at
    1)."""
    (oa, sa), (ob, sb) = a, b
    w = worst(oa, ob, skip)
    if sa is not None or sb is not None:
        w |= {f"state.{k}": v for k, v in worst(sa, sb, skip).items()}
    top = max(w.values(), default=0.0)
    print(f"{label}: worst relative difference {top:.1e}")
    bad = {k: v for k, v in w.items() if v > rtol}
    assert not bad, f"{label}: differ beyond {rtol:g}: {bad}"


def assert_live(a, label: str = ""):
    """The run produced numbers to compare: a float field other than
    ``weight_sum`` and ``settled_frac`` with a finite non-zero value."""
    out, state = a
    frames = [out] + ([state] if state is not None else [])
    for f in frames:
        for c in f.columns:
            if c.startswith(("weight_sum", "settled_frac")):
                continue
            leaves = _leaves(f[c])
            if leaves is None:
                continue
            v = (
                leaves[0][~leaves[1]]
                if f[c].dtype.is_float()
                else f[c].explode().drop_nulls().to_numpy()
            )
            v = v[np.isfinite(v)]
            if (v != 0).any():
                return
    raise AssertionError(f"{label}: no field carries a number; the comparison would be empty")


def assert_differs(a, b, label: str):
    """Evidence the parameter took effect: the two runs differ somewhere."""
    (oa, sa), (ob, sb) = a, b
    w = worst(oa, ob)
    if sa is not None and sb is not None and sa.columns == sb.columns:
        w |= {f"state.{k}": v for k, v in worst(sa, sb).items()}
    moved = sorted(k for k, v in w.items() if v > 0)
    print(f"{label}: moves {moved[:6]}")
    assert moved, f"{label}: nothing moved; the relation would hold with the parameter ignored"


# --- the relations ---------------------------------------------------------


def _clocked(**kw: Any) -> dict[str, Any]:
    return {"clock": "t", "gap_cap": 1e9} | kw


def _cut(df: pl.DataFrame, cap: float) -> pl.DataFrame:
    """The stream with every clock step cut to ``cap``."""
    t = df["t"].to_numpy()
    steps = np.minimum(np.diff(t, prepend=t[0]), cap)
    return df.with_columns(t=pl.Series(np.cumsum(steps) + t[0]))


@pytest.mark.parametrize("variant", _params([k for k in KINDS if k not in CAP_IS_A_BREAK]))
def test_a_cap_is_every_step_cut_to_it(variant: str):
    capped = run(variant, clock="t", gap_cap=CAP)
    cut = run(variant, _cut(STREAM, CAP), clock="t", gap_cap=1e9)
    assert (np.diff(STREAM["t"].to_numpy()) > CAP).sum() >= 10  # the cap binds
    assert_live(capped, variant)
    if variant in DECAYING:
        assert_differs(capped, run(variant, **_clocked()), f"{variant}: the cap")
    assert_same(capped, cut, label=f"{variant}: cap {CAP} vs steps cut to it")


@pytest.mark.parametrize("variant", sorted(CAP_IS_A_BREAK))
def test_a_cap_is_a_break_where_the_docs_say_so(variant: str):
    """The exceptions differ, so the list holds only kinds the relation
    fails for."""
    capped = run(variant, clock="t", gap_cap=CAP)
    cut = run(variant, _cut(STREAM, CAP), clock="t", gap_cap=1e9)
    assert_differs(capped, cut, f"{variant}: a capped step is a break")


@pytest.mark.parametrize("variant", _params(KINDS))
def test_a_cap_past_every_step_cuts_nothing(variant: str):
    longest = float(np.diff(STREAM["t"].to_numpy()).max())
    assert longest == LONG_STEP
    near = run(variant, clock="t", gap_cap=longest * (1 + 2**-40))
    far = run(variant, clock="t", gap_cap=1e300)
    assert_live(near, variant)
    assert_same(near, far, label=f"{variant}: cap just past the longest step vs 1e300")


@pytest.mark.parametrize("variant", _params(DECAYING))
def test_an_infinite_half_life_is_lam_one(variant: str):
    a = run(variant, half_life=math.inf)
    b = run(variant, half_life=None, lam=1.0)
    assert_live(a, variant)
    assert_differs(a, run(variant), f"{variant}: half_life inf vs {HALF_LIFE}")
    assert_same(a, b, skip=SPELLING, label=f"{variant}: half_life=inf vs lam=1")


#: How far ``lam = 0.5 ** (1 / h)`` may move a field from ``half_life=h``:
#: the two reach a step's decay by different arithmetic (a power of
#: ``lam`` against ``exp2(-d / h)``), so the last bits differ. Measured
#: 2026-10-25 on this stream: TBD.
LAM_RTOL = 1e-11


@pytest.mark.parametrize(("variant", "seed"), _seeded(DECAYING))
def test_lam_is_the_half_life_it_stands_for(variant: str, seed: int):
    kind = VARIANTS[variant][0]
    solve = {"solve_every": 0} if kind in SOLVES else {}
    df = _stream(seed)
    a = run(variant, df, **solve)
    b = run(variant, df, half_life=None, lam=0.5 ** (1 / HALF_LIFE), **solve)
    assert_live(a, variant)
    assert_differs(a, run(variant, df, half_life=math.inf, **solve), f"{variant}: the decay")
    assert_same(
        (a[0].select("weight_sum"), None),
        (b[0].select("weight_sum"), None),
        label=f"{variant}: weight_sum under lam vs half_life",
    )
    assert_same(a, b, rtol=LAM_RTOL, skip=SPELLING, label=f"{variant}: lam vs half_life")


@pytest.mark.parametrize("variant", _params(DECAYING))
def test_a_grid_instance_is_the_lone_spec(variant: str):
    grid = run(variant, half_life=[HALF_LIFE, 10.0])
    lone = run(variant, half_life=10.0)
    suffix = "@h10"
    cols = [c for c in grid[0].columns if c.endswith(suffix)]
    assert cols, grid[0].columns
    picked = grid[0].select(cols).rename({c: c[: -len(suffix)] for c in cols})
    other = grid[0].select(c.replace(suffix, "@h50") for c in cols)
    assert_differs(
        (picked, None),
        (other.rename(dict(zip(other.columns, picked.columns, strict=True))), None),
        f"{variant}: the grid's two instances",
    )
    state = None
    if grid[1] is not None:
        state = grid[1].filter(pl.col("half_life") == 10.0).drop("instance")
        lone_state = lone[1].drop("instance")
        assert state.height == lone_state.height > 0
        lone = (lone[0], lone_state)
    assert_live((picked, state), variant)
    assert_same(
        (picked, state),
        (lone[0].select(picked.columns), lone[1]),
        label=f"{variant}: grid instance h10 vs half_life=10",
    )


@pytest.mark.parametrize("variant", _params([k for k in KINDS if k not in UNWEIGHTED]))
def test_a_weight_of_ones_is_no_weight(variant: str):
    plain = run(variant)
    ones = run(variant, STREAM.with_columns(one=pl.lit(1.0)), weight="one")
    assert_live(plain, variant)
    if variant != "rcov":  # rcov takes a weight of 0 or 1 only
        assert_differs(plain, run(variant, weight="w"), f"{variant}: a weight")
    assert_same(plain, ones, label=f"{variant}: weight 1 vs none")


#: Weight ``k`` against ``k`` copies: the copies' sums add in another order
#: than one multiplication, so the last bits may differ. Measured
#: 2026-10-25, worst of 50 seeds: 1.3e-15 (``marginal``), 1.4e-14
#: (``rls``), 1.6e-14 (``kalman``), 2.8e-14 (``ewridge``, seed 15).
COPIES_RTOL = 1e-12


@pytest.mark.parametrize(
    ("variant", "seed"), _seeded(sorted(COPIES), essential=("ewridge", "rls", "marginal"))
)
def test_a_weight_k_is_k_copies_of_the_row(variant: str, seed: int):
    df = _stream(seed)
    k = np.where(np.arange(N) % 3 == 0, 2, 1)
    weighted = df.with_columns(k=pl.Series(k.astype(np.float64)))
    copies = (
        df.with_row_index("i")
        .select(pl.all().repeat_by(pl.Series(k)).explode())
        .with_columns(k=pl.lit(1.0))
    )
    ends = (
        copies.with_row_index("r")
        .group_by("i", maintain_order=True)
        .agg(first=pl.col("r").first(), last=pl.col("r").last())
    )
    every = {} if variant == "marginal" else {"coef_every": 0}
    kw = _clocked(weight="k", **every, **COPIES[variant])
    a = run(variant, weighted, **kw)
    b = run(variant, copies.drop("i"), **kw)
    assert_live(a, variant)
    assert_differs(
        a, run(variant, df, **_clocked(**every, **COPIES[variant])), f"{variant}: the weights"
    )
    # A row's output is read before its update and its coefficients after:
    # the first copy's and the last copy's.
    b_rows = b[0][ends["first"].to_numpy()]
    post = [c for c in ("coef", "support_coef") if c in b_rows.columns]
    b_rows = b_rows.with_columns(b[0][ends["last"].to_numpy()].select(post))
    assert_same(
        a,
        (b_rows, b[1]),
        rtol=COPIES_RTOL,
        skip=KISH.get(variant, ()),
        label=f"{variant}: weight k vs k copies",
    )


def _reordered(
    variant: str, out: pl.DataFrame, state: pl.DataFrame | None, forward: dict, backward: dict
):
    """The reversed run's fields in the forward run's order. A field named
    for a feature keeps its name (a pair's name lists its two in the order
    given, so it is turned round); a list indexed by position is gathered
    back by ``coef_fields``' names."""
    out = out.rename({c: c.replace("x1_x0", "x0_x1") for c in out.columns})
    fields = po.spec.coef_fields(forward)
    rfields = po.spec.coef_fields(backward)
    for field in fields["field"].unique().to_list():
        names = fields.filter(pl.col("field") == field)["name"].to_list()
        rnames = rfields.filter(pl.col("field") == field)["name"].to_list()
        order = [rnames.index(n) for n in names]
        cols = [field] + (
            ["support_coef"] if field == "coef" and "support_coef" in out.columns else []
        )
        out = out.with_columns(pl.col(c).list.gather(order) for c in cols)
    if variant == "micro":
        # A row of six a summary (id, label, weight, radius, then the
        # centre's features in order); the count of summaries varies, so
        # ``coef_fields`` has no fixed index for it.
        out = out.with_columns(
            pl.Series(
                "coef",
                [
                    None
                    if c is None
                    else np.asarray(c).reshape(-1, 6)[:, [0, 1, 2, 3, 5, 4]].ravel().tolist()
                    for c in out["coef"].to_list()
                ],
                dtype=pl.List(pl.Float64),
            )
        )
    if state is not None:
        key = STATE_KEY.get(variant)
        if key:
            state = state.sort(key, maintain_order=True)
        else:  # rcov: two features, so each vector and flattened matrix reverses
            state = state.with_columns(
                pl.col(c).list.reverse()
                for c in state.columns
                if state[c].dtype == pl.List(pl.Float64)
            )
    return out, state


#: The state tables keyed by a feature, one row each.
STATE_KEY = {"marginal": "feature", "audit": "column"}


#: Reversing the features permutes every sum a fit takes over them: the
#: same numbers in another order, so the last bits may move. Measured
#: 2026-10-25, worst of 50 seeds: 1.6e-13 (``kalman``), 2.9e-14 (``rls``),
#: at most 1.1e-14 elsewhere; ``lasso`` (run converged, below) 1.1e-12.
ORDER_RTOL = 1e-12
ORDER_RTOL_BY_KIND = {"lasso": 1e-10}
#: ``lasso``'s descent visits the features in the order given and stops at
#: its sweep cap (100) or its tolerance, so at its defaults the order picks
#: which unconverged point a solve returns: 4.6e-4 relative on ``pred`` at
#: the worst of 50 seeds (seed 15), 2.0e-5 at the 95th percentile, the cap
#: counted in ``solve_failures``; at ``max_iter=10_000`` 1.1e-8. Run
#: converged, it is order-free to 1.1e-12.
CONVERGED = {"lasso": {"max_iter": 100_000, "tol": 1e-14}}


@pytest.mark.parametrize(("variant", "seed"), _seeded([k for k in KINDS if k not in NO_FEATURES]))
def test_the_feature_order_moves_no_number(variant: str, seed: int):
    key = "columns" if variant == "audit" else "features"
    extra = CONVERGED.get(variant, {})
    fw, bw = {key: ["x0", "x1"], **extra}, {key: ["x1", "x0"], **extra}
    df = _stream(seed)
    a = run(variant, df, **fw)
    b = run(variant, df, **bw)
    b = _reordered(variant, *b, spec(variant, **fw), spec(variant, **bw))
    assert_live(a, variant)
    if variant in STATE_KEY:
        a = (a[0], a[1].sort(STATE_KEY[variant], maintain_order=True))
    assert_same(
        a,
        (b[0].select(a[0].columns), b[1]),
        rtol=ORDER_RTOL_BY_KIND.get(variant, ORDER_RTOL),
        label=f"{variant}: features reversed",
    )


#: The clock parameters each kind takes, as numbers of seconds.
def _clock_parameters(variant: str) -> dict[str, Any]:
    kw: dict[str, Any] = {"gap_cap": 20.0}
    if variant in DECAYING:
        kw["half_life"] = HALF_LIFE
    if variant == "kalman":
        kw["coef_half_life"] = MINIMAL["kalman"]["coef_half_life"]
    return kw


def _as_duration(seconds: float) -> dt.timedelta:
    return dt.timedelta(seconds=seconds)


@pytest.mark.parametrize("variant", _params(KINDS))
def test_durations_are_numbers_of_seconds(variant: str):
    origin = dt.datetime(2024, 3, 1, 9, 30)
    when = STREAM.with_columns(
        t=pl.lit(origin) + pl.duration(milliseconds=(pl.col("t") * 1000).cast(pl.Int64))
    ).with_columns(pl.col("t").dt.cast_time_unit("ms"))
    numbers = _clock_parameters(variant)
    durations = {k: _as_duration(v) for k, v in numbers.items()}
    a = run(variant, clock="t", **numbers)
    b = run(variant, when, clock="t", **durations)
    assert_live(a, variant)
    if variant in DECAYING or variant in CAP_IS_A_BREAK:
        uncapped = run(variant, clock="t", **(numbers | {"gap_cap": 1e9}))
        assert_differs(a, uncapped, f"{variant}: the cap")
    assert_same(a, b, skip=CLOCK_COLUMNS, label=f"{variant}: durations vs seconds")


@pytest.mark.parametrize("variant", _params(KINDS))
def test_no_clock_is_the_row_index(variant: str):
    a = run(variant)
    b = run(
        variant,
        STREAM.with_columns(t=pl.Series(np.arange(N, dtype=np.float64))),
        clock="t",
        gap_cap=2.0,
    )
    assert_live(a, variant)
    assert_same(a, b, skip=CLOCK_COLUMNS, label=f"{variant}: no clock vs the row index")


@pytest.mark.parametrize("variant", _params(KINDS))
def test_fit_leaves_the_same_state_at_every_chunk_size(variant: str):
    def saved(chunk_size: int | None, predict: bool = False) -> bytes:
        bank = po.ModelBank([spec(variant, clock="t", gap_cap=CAP)], threads=1)
        if predict:
            bank.fit_predict(frame(variant))
        else:
            bank.fit(frame(variant), chunk_size=chunk_size)
        return bank.save_bytes()

    whole = saved(None, predict=True)
    sizes = (1, 7, 600) if variant in ESSENTIAL else (7, 600)
    for k in (None, *sizes):
        assert saved(k) == whole, f"{variant}: fit(chunk_size={k}) left another state"
    # The state is the stream's: a different stream leaves different bytes.
    other = po.ModelBank([spec(variant, clock="t", gap_cap=CAP)], threads=1)
    other.fit(frame(variant).head(N - 37))
    assert other.save_bytes() != whole


# --- the legs that wait for task 277 ---------------------------------------


@pytest.mark.parametrize("variant", sorted(AFTER_TASK_277))
def test_the_waiting_legs_run(variant: str):
    """The ``huber`` legs build and run, so adding them is a move into
    ``VARIANTS``, not a build."""
    assert_live(run(variant, clock="t", gap_cap=CAP), variant)
    assert_live(run(variant, half_life=[HALF_LIFE, 10.0]), variant)


def test_the_huber_legs_still_wait():
    """Task 277's warm-up is a parameter ``po.spec.huber`` did not take
    before it. When one arrives, move ``AFTER_TASK_277``'s entries into
    ``VARIANTS`` (``SOLVES`` already names ``huber``)."""
    for builder, before in BEFORE_277.items():
        now = set(inspect.signature(getattr(po.spec, builder)).parameters)
        new = sorted(now - before)
        assert not new, (
            f"po.spec.{builder} gained {new}: if this is task 277's warm-up, move "
            f"AFTER_TASK_277's entries into VARIANTS"
        )


def test_every_kind_is_swept_or_waits():
    assert set(MINIMAL) == {VARIANTS[v][0] for v in VARIANTS} | {
        AFTER_TASK_277[v][0] for v in AFTER_TASK_277
    }
