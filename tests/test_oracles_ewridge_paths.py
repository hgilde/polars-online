"""``ewridge`` on the paths its numpy oracle (``tests/reference.py``
``ewridge_ref``, which solves after every row) does not reach, held row by
row to ``tests/reference_paths.py::ewridge_paths_ref``: the solve schedule
(``solve_every``, its ``halflife / 50`` default, ``max_rows_between_solves``,
``gram_block_rows``), the grids (a ridge list, ``feature_sets``, a halflife
list), both ``target_gaps`` with a penalty and ``standardize``,
``add_intercept=False``, a ``window`` over weights and skipped rows, and
sessions -- a gap, a reset and a ``session_shrink`` blend.

The reference recomputes each solve from the raw rows at their effective
weights, so it shares neither the core's mean-form recursion, its Cholesky
factor nor its schedule code. Every stream is ``tests/test_oracles_lasso_paths.py``'s:
an irregular dyadic clock with two capped gaps, zero weights (the first row
among them), skipped rows and three targets with different gaps.

``max_error_inflation`` is off in every spec. It is a readiness gate, held
by ``tests/test_readiness.py``, and not part of the fit the reference is
written from.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from reference_paths import ewridge_paths_ref
from test_oracles_lasso_paths import FEATURES, TARGETS, _stream

MAX_DCLOCK = 6.0

# Measured over the cases below as |got - expected| / (1 + |expected|): pred
# 1.1e-14, resid 2.4e-14, n_eff 1.5e-15, coef 9.7e-14. Each tolerance is 100x the
# largest it covers, rounded up to a power of ten.
#
# Seeded into a copy of the reference, each of these fails every case it
# touches by far more (pred, same measure; coef where pred barely moves): the
# ridge on the intercept too, pred 1e-5 to 4.5 and coef 2.0-7.4; the ridge on
# the raw scale under `standardize`, 1.2e-6 to 0.78; centred without an
# intercept, 3.7-5.0; own_rows and pairwise swapped, 0.22-3.4; each feature
# set solved on every feature, 1.0-1.3; the window's boundary strict (`<`),
# 0.28; the blend after the row's own decay, 6.4e-4 to 2.8e-3 (n_eff
# 6.8e-3 to 2.7e-2), or at 1 - f, 3.3e-2 to 4.3e-2; a solve after every row
# in place of the schedule, 6.4e-2 to 0.38. So do the in-sample and row-late
# probes below.
PRED_TOL = 1e-11
COEF_TOL = 1e-11


def _session_stream(seed: int) -> pl.DataFrame:
    """The stream, in four sessions that change at rows 60, 130 and 200, no
    skipped row beside a boundary."""
    df = _stream(seed)
    s = np.zeros(df.height, dtype=np.int64)
    s[60:], s[130:], s[200:] = 1, 2, 3
    return df.with_columns(pl.Series("s", s))


def _check(df, halflife, *, sets=None, add_intercept=True, **kw):
    """Fit one ``ewridge`` spec that ``kw`` completes and hold every
    instance, target and grid slot to the reference: ``pred``, ``resid``,
    ``n_eff``, every held ``coef`` and the rows where each is null. ``sets``
    is ``{name: columns}``; ``halflife`` may be a list, one instance each."""
    spec_only = {k: kw.pop(k) for k in ("gram_block_rows",) if k in kw}
    spec = po.spec.ewridge(
        "m",
        targets=TARGETS,
        features=FEATURES,
        clock="t",
        max_dclock=MAX_DCLOCK,
        weight="w",
        halflife=halflife,
        feature_sets=sets,
        add_intercept=add_intercept,
        coef_every=1,
        max_error_inflation=float("inf"),
        **spec_only,
        **kw,
    )
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    index = po.spec.output_index(spec)
    n = df.height
    dc = np.zeros(n)
    dc[1:] = np.diff(df["t"].to_numpy())
    x = df.select(FEATURES).to_numpy()
    y = df.select(TARGETS).to_numpy().astype(float)
    ref_kw = dict(kw)
    if "session" in ref_kw:
        ref_kw["session"] = df[ref_kw["session"]].to_numpy()
    names = list(sets) if sets else None
    idx_sets = [[FEATURES.index(c) for c in sets[name]] for name in names] if sets else None
    halflives = halflife if isinstance(halflife, list) else [halflife]
    z = np.column_stack([np.ones(n), x]) if add_intercept else x
    for h in halflives:
        ref = ewridge_paths_ref(
            x,
            y,
            dc,
            df["w"].to_numpy(),
            halflife=h,
            feature_sets=idx_sets,
            add_intercept=add_intercept,
            max_dclock=MAX_DCLOCK,
            **ref_kw,
        )
        assert ref["solved"].sum() >= 20, "the stream should span many solves"
        mine = index.filter(pl.col("halflife") == h) if len(halflives) > 1 else index
        suffix = f"@h{h:g}" if len(halflives) > 1 else ""
        _held(out, mine, ref, y, z, suffix, names, idx_sets)


def _field(index, kind, target, cols, r, names, idx_sets):
    """The one field of ``kind`` for ``target`` and the (set, ridge) slot."""
    q = index.filter((pl.col("kind") == kind) & (pl.col("target") == target))
    if q["ridge"].n_unique() > 1:
        q = q.filter(pl.col("ridge") == r)
    if names:
        q = q.filter(pl.col("feature_set") == names[idx_sets.index(cols)])
    assert q.height == 1, q
    return q["field"][0]


def _held(out, index, ref, y, z, suffix, names, idx_sets):
    got = np.full_like(ref["pred"], np.nan)
    for j, t in enumerate(TARGETS):
        for c, (cols, r) in enumerate(ref["combos"]):
            f = _field(index, "pred", t, cols, r, names, idx_sets)
            got[:, j, c] = out.struct.field(f).to_numpy().astype(float)
            _close(got[:, j, c], ref["pred"][:, j, c], PRED_TOL, f)
            f = _field(index, "resid", t, cols, r, names, idx_sets)
            resid = out.struct.field(f).to_numpy().astype(float)
            _close(resid, y[:, j] - ref["pred"][:, j, c], PRED_TOL, f)
        assert np.isfinite(ref["pred"][:, j, 0]).sum() > 100, f"{t} is scored too little"
    n_eff = out.struct.field(f"n_eff{suffix}").to_numpy().astype(float)
    _close(n_eff, ref["n_eff"], PRED_TOL, f"n_eff{suffix}")

    # `coef` is each row's last solve: null on a skipped row and wherever the
    # model has none yet -- before its first, and after a reset until the
    # next.
    rows = out.struct.field(f"coef{suffix}").to_list()
    empty = [np.nan] * ref["coef"][0].size
    coef = np.array([empty if r is None else r for r in rows], float).reshape(ref["coef"].shape)
    wrong = np.flatnonzero(np.array([r is None for r in rows]) != ~ref["has_fit"])
    assert wrong.size == 0, f"coef{suffix} null on the wrong rows, first {wrong[:5]}"
    held = ~np.isnan(ref["coef"])
    assert held.mean() > 0.8, "most of the problems should be held"
    _close(coef[held], ref["coef"][held], COEF_TOL, f"coef{suffix}")

    # The comparison can fail: the reference read in sample (each row scored
    # with the solve its own row fed), or with each solve a row late.
    in_sample = np.einsum("ijck,ik->ijc", ref["coef"], z)
    late = np.full_like(in_sample, np.nan)
    late[2:] = np.einsum("ijck,ik->ijc", ref["coef"][:-2], z[2:])
    for what, probe in (("in sample", in_sample), ("a row late", late)):
        both = np.isfinite(got) & np.isfinite(probe)
        assert np.max(np.abs(got[both] - probe[both])) > 1e-2, f"pred cannot tell {what} apart"


def _close(got, exp, tol, what):
    assert (np.isnan(got) == np.isnan(exp)).all(), (
        f"{what}: null patterns differ at rows {np.flatnonzero(np.isnan(got) != np.isnan(exp))[:8]}"
    )
    ok = ~np.isnan(exp)
    err = np.abs(got[ok] - exp[ok]) / (1.0 + np.abs(exp[ok]))
    assert err.size == 0 or err.max() <= tol, f"{what}: max rel diff {err.max():.3e}"


SETS = {"all": FEATURES, "sub": ["x0", "x1"]}


class TestTheSchedule:
    """The coefficients are the sums' as of the last solve, and the solve
    runs on the documented schedule."""

    # The 0.5 slot is ridge-dominated on purpose, which ReadinessWarning says.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_the_default_cadence_over_a_grid_of_ridges_and_sets(self):
        """No ``solve_every``: a solve every ``halflife / 50`` = 0.8 clock
        units, over two ridges times two feature sets, each target gated on
        its own threshold."""
        _check(
            _stream(41),
            40.0,
            ridge=[1e-6, 0.5],
            sets=SETS,
            min_periods=[15.0, 10.0, 20.0],
        )

    def test_a_long_halflife_solves_every_four_clock_units(self):
        """``halflife = 200``: the default cadence is 4, so a row is scored
        with coefficients up to four clock units stale."""
        _check(_stream(46), 200.0, min_periods=12.0)

    # The 0.5 slot is ridge-dominated on purpose, which ReadinessWarning says.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_blocked_grams_keep_the_schedule(self):
        """``gram_block_rows = 8`` beside a row cap of 8 and a clock cadence
        of 5: the blocked sums solve to the same fits."""
        _check(
            _stream(45),
            40.0,
            ridge=[1e-6, 0.5],
            gram_block_rows=8,
            max_rows_between_solves=8,
            solve_every=5.0,
            min_periods=12.0,
        )


class TestGrids:
    # The 0.5 slot is ridge-dominated on purpose, which ReadinessWarning says.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_standardized_ridges_read_pairwise(self):
        """Two ridges on the correlation scale, ``pairwise`` gaps and a row
        cap of 4 in place of a clock cadence."""
        _check(
            _stream(41),
            40.0,
            ridge=[1e-6, 0.5],
            standardize=True,
            target_gaps="pairwise",
            max_rows_between_solves=4,
            solve_every=1e9,
            min_periods=15.0,
        )

    def test_a_halflife_grid_is_one_model_per_halflife(self):
        """Two halflives: two accumulators, two cadences (``20 / 50`` and
        ``80 / 50``), two ``n_eff``, each instance held on its own."""
        _check(_stream(47), [20.0, 80.0], sets=SETS, min_periods=10.0)


class TestWithoutAnIntercept:
    # The 0.3 slot is ridge-dominated on purpose, which ReadinessWarning says.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize("standardize", [False, True])
    def test_nothing_is_centred(self, standardize):
        """Features at levels an intercept would absorb; the ridge is on
        every slope, and ``standardize`` scales by the root mean square."""
        _check(
            _stream(42, level=3.0),
            60.0,
            ridge=[1e-6, 0.3],
            add_intercept=False,
            standardize=standardize,
            min_periods=15.0,
        )


class TestWindow:
    @pytest.mark.parametrize(
        ("target_gaps", "standardize"), [("own_rows", True), ("pairwise", False)]
    )
    def test_the_rows_at_most_window_old(self, target_gaps, standardize):
        """A window of 24 clock units under a halflife of 40, over weights,
        zero weights, skipped rows and capped gaps."""
        _check(
            _stream(43),
            40.0,
            window=24.0,
            standardize=standardize,
            target_gaps=target_gaps,
            min_periods=10.0,
            solve_every=1.0,
        )


class TestSessions:
    def test_a_session_gap(self):
        _check(_session_stream(44), 40.0, session="s", session_gap=3.0, min_periods=10.0)

    def test_a_reset_starts_over(self):
        _check(_session_stream(44), 40.0, session="s", session_gap="reset", min_periods=10.0)

    @pytest.mark.parametrize(
        ("shrink", "long_halflife", "schedule"),
        [
            (0.3, 200.0, {"solve_every": 1e-9}),
            (0.7, float("inf"), {"standardize": True, "max_rows_between_solves": 3}),
        ],
    )
    def test_a_blend_with_the_slow_twin(self, shrink, long_halflife, schedule):
        """``session_shrink``: every row's weight becomes ``(1 - f)`` of its
        own plus ``f`` of its weight in the twin, the new session's first row
        is scored from the blend, and the rows age from there."""
        _check(
            _session_stream(44),
            40.0,
            session="s",
            session_gap=2.0,
            session_shrink=shrink,
            long_halflife=long_halflife,
            min_periods=10.0,
            **schedule,
        )
