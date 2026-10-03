"""The lasso's paths ``tests/test_oracles.py::TestLassoPredPath`` does not
reach, held to ``tests/reference_paths.py::lasso_paths_ref`` row by row:
several targets, null targets, ``target_gaps``, ``fit_intercept=False``, a
``window`` and ``lam_selected`` at a ``select_half_life`` of its own.

The reference recomputes every statistic from the raw rows at each solve,
each row at ``w * 0.5 ** (age / half_life)``, and descends from zero to 1e-14,
so neither the core's mean-form recursion nor its warm start can agree with
it by construction. Every stream here has an irregular dyadic clock with two
gaps past ``gap_cap``, zero-weight rows (the first among them), rows
skipped for a null feature, and three targets: one with a few random nulls,
one present only where ``x0 > -0.5`` (gaps tied to a feature, so
``own_rows`` and ``pairwise`` part), and one with a block of nulls that the
window outlives.

``lam_selected`` is held everywhere, which it was not on 2026-09-24 when
this file was written:

- under a ``window`` (docs/PLAN.md task 95), where the core truncated the
  selection error against a snapshot taken after the row's own error, with
  its weight aged twice, chose on the window as it stood a row earlier and
  not at all on a row it did not score, and aged the snapshot by the model's
  half-life where ``select_half_life`` differs;
- with a ``min_weight`` list (task 96). The model predicts once ``weight_sum``
  reaches the smallest threshold, and a target's selection error folds its
  own predictions from then on, on rows its own larger threshold still
  withholds: a gate on the output, not on the model (review S2). The user
  kept that reading on 2026-09-24, and docs/ENHANCEMENTS.md E7, which said
  otherwise, was corrected.

A window that holds one row of a target drops every feature of that
target's problem, which then fits its intercept alone (task 94); under a
lowered gate those rows are scored, and held.
"""

import numpy as np
import polars as pl

import polars_online as po
from reference_paths import lasso_paths_ref

FEATURES = ["x0", "x1", "x2", "x3"]
TARGETS = ["ya", "yb", "yc"]
PATH = [0.2, 0.05, 0.0]
MAX_DCLOCK = 6.0

# Measured over the cases below as |got - expected| / (1 + |expected|): pred
# 4.5e-14, resid 2.3e-13 (pred's absolute error over a residual far smaller
# than a target at a level of 5), weight_sum 2.3e-15, coef 1.7e-12. Each
# tolerance is 100x the largest it covers, rounded up to a power of ten.
#
# Seeded into a copy of the reference, each of these fails a case here by
# far more (pred, same measure): own_rows and pairwise swapped 0.23-4.6;
# pairwise cross-moments centred at every row's means (review N3) 0.04-0.10;
# centred without an intercept (C8) 4.4; the window's boundary strict (`<`)
# 0.30-0.39, or no window 0.80-1.1; the decay at twice the half-life
# 0.04-0.12; each target gated on the shared weight_sum (S2) moves the null
# pattern on 18-162 rows; the selection at the model's half-life instead of
# its own moves lam_selected on 133 rows, and from emitted rows only on
# 15-39.
PRED_TOL = 1e-10
COEF_TOL = 1e-9


def _stream(seed: int, n: int = 240, level: float = 2.0) -> pl.DataFrame:
    """Four features and three targets on an irregular clock.

    ``x2`` is correlated with ``x0`` and out of every target's truth, so the
    penalty has a coefficient to zero; ``x1`` and ``x3`` sit at levels only an
    intercept (or, without one, the raw moments) can absorb. The clock steps
    are dyadic, so the clock since a solve and a row's age sum exactly and a
    row exactly ``window`` old is decided by the rule, not by rounding."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], size=n)
    dt[0] = 0.0
    dt[[70, 160]] = 40.0
    x0 = rng.standard_normal(n)
    x1 = level + rng.standard_normal(n)
    x2 = 0.6 * x0 + 0.8 * rng.standard_normal(n)
    x3 = -1.0 + 0.7 * rng.standard_normal(n)
    ya = 0.3 + x0 - 0.6 * x1 + 0.25 * x3 + 0.5 * rng.standard_normal(n)
    yb = 5.0 + 0.8 * x0 + 0.5 * x1 + 0.4 * rng.standard_normal(n)
    yc = -2.0 + 1.5 * x3 - 0.4 * x1 + 0.3 * rng.standard_normal(n)
    ya[rng.choice(n, 10, replace=False)] = np.nan
    yb[x0 <= -0.5] = np.nan
    yc[100:140] = np.nan
    yc[::7] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[0] = 0.0
    w[rng.choice(np.arange(1, n), size=n // 20 - 1, replace=False)] = 0.0
    x1[[30, 31, 75, 120]] = np.nan
    frame = {"t": np.cumsum(dt), "x0": x0, "x1": x1, "x2": x2, "x3": x3}
    frame |= {"ya": ya, "yb": yb, "yc": yc, "w": w}
    return pl.DataFrame(frame).with_columns(pl.col(c).fill_nan(None) for c in ["x1", *TARGETS])


def _close(got, exp, tol, what):
    assert (np.isnan(got) == np.isnan(exp)).all(), (
        f"{what}: null patterns differ at rows {np.flatnonzero(np.isnan(got) != np.isnan(exp))[:8]}"
    )
    ok = ~np.isnan(exp)
    err = np.abs(got[ok] - exp[ok]) / (1.0 + np.abs(exp[ok]))
    assert err.size == 0 or err.max() <= tol, f"{what}: max rel diff {err.max():.3e}"


def _fit(df: pl.DataFrame, **kw):
    """The bank's output and the reference's, for one spec that ``kw``
    completes, with the descent run to where its start cannot matter."""
    spec = po.spec.lasso(
        "m",
        targets=TARGETS,
        features=FEATURES,
        lasso_path=PATH,
        clock="t",
        gap_cap=MAX_DCLOCK,
        weight="w",
        coef_every=1,
        tol=1e-14,
        max_iter=100_000,
        **kw,
    )
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    n = df.height
    dc = np.zeros(n)
    dc[1:] = np.diff(df["t"].to_numpy())
    ref = lasso_paths_ref(
        df.select(FEATURES).to_numpy(),
        df.select(TARGETS).to_numpy().astype(float),
        dc,
        df["w"].to_numpy(),
        PATH,
        gap_cap=MAX_DCLOCK,
        **kw,
    )
    return spec, out, ref


def _held_to_the_reference(df, spec, out, ref, lam_selected=True, min_selected=150):
    """Every path point's ``pred`` and ``resid`` for every target, ``weight_sum``,
    every held ``coef`` and, where asked, ``lam_selected`` on at least
    ``min_selected`` rows of each target; then two probes that the
    comparison can fail."""
    n, kt = df.height, len(ref["coef"][0, 0, 0])
    index = po.spec.output_index(spec)
    y = df.select(TARGETS).to_numpy().astype(float)
    got = np.full_like(ref["pred"], np.nan)
    for j, t in enumerate(TARGETS):
        for p, lam in enumerate(PATH):
            rows = index.filter(
                (pl.col("target") == t)
                & (pl.col("penalty") == lam)
                & pl.col("kind").is_in(["pred"])
            )
            got[:, j, p] = out.struct.field(rows["field"][0]).to_numpy().astype(float)
            _close(got[:, j, p], ref["pred"][:, j, p], PRED_TOL, f"pred_{t} at {lam}")
            resid_field = rows["field"][0].replace("pred_", "resid_", 1)
            resid = out.struct.field(resid_field).to_numpy().astype(float)
            _close(resid, y[:, j] - ref["pred"][:, j, p], PRED_TOL, f"resid_{t} at {lam}")
        assert np.isfinite(ref["pred"][:, j, 0]).sum() > 100, f"{t} is scored too little"
        if lam_selected:
            sel = out.struct.field(f"penalty_selected_{t}").to_numpy().astype(float)
            held = ~np.isnan(ref["lam_selected"][:, j])
            assert held.sum() > min_selected, f"{t}: too few rows with a clear selection"
            wrong = np.flatnonzero(held & (sel != ref["lam_selected"][:, j]))
            assert wrong.size == 0, f"penalty_selected_{t} differs at rows {wrong[:8]}"
            assert len(set(sel[held])) > 1, f"penalty_selected_{t} never moves"
    _close(
        out.struct.field("weight_sum").to_numpy().astype(float),
        ref["weight_sum"],
        PRED_TOL,
        "weight_sum",
    )

    rows = out.struct.field("coef").to_list()
    empty = [np.nan] * (len(TARGETS) * len(PATH) * kt)
    coef = np.array([empty if r is None else r for r in rows], float).reshape(ref["coef"].shape)
    first = np.argmax(ref["solved"])
    expect_null = (np.arange(n) < first) | np.isnan(ref["weight_sum"])
    assert (np.array([r is None for r in rows]) == expect_null).all(), "coef null on wrong rows"
    held = ~np.isnan(ref["coef"])
    assert held.sum() > 0.8 * held.size, "most of the problems should be held"
    _close(coef[held], ref["coef"][held], COEF_TOL, "coef")
    # Where the L1 acts, it zeroes exactly what the reference zeroes. (At a
    # zero penalty an exact zero is incidental: a cross-moment of one row.)
    ratio = spec["model"]["l1_ratio"]
    l1 = np.asarray(PATH) * (1.0 if ratio is None else ratio) > 0.0
    zero, zero_ref = coef[:, :, l1] == 0.0, ref["coef"][:, :, l1] == 0.0
    assert zero_ref[held[:, :, l1]].any(), "the L1 should zero something"
    assert (zero == zero_ref)[held[:, :, l1]].all(), "the L1 zeroed other coefficients"

    # The comparison can fail: the same reference read in sample (each row
    # scored with the solve its own row fed), or with every solve reaching
    # the predictions a row late, is far outside the tolerance.
    off = 1 if kt == len(FEATURES) + 1 else 0
    x = df.select(FEATURES).to_numpy()
    z = np.column_stack([np.ones(n), x]) if off else x
    in_sample = np.einsum("ijpk,ik->ijp", ref["coef"], z)
    late = np.full_like(in_sample, np.nan)
    late[2:] = np.einsum("ijpk,ik->ijp", ref["coef"][:-2], z[2:])
    for what, probe in (("in sample", in_sample), ("a row late", late)):
        both = np.isfinite(got) & np.isfinite(probe)
        assert np.max(np.abs(got[both] - probe[both])) > 1e-2, f"pred cannot tell {what} apart"


class TestSeveralTargetsWithGaps:
    """Three targets under ``own_rows``: each fitted on the rows it is
    present on, gated on its own weight against its own threshold, with a
    solve every 2.5 clock units."""

    def test_each_target_is_the_path_of_its_own_rows(self):
        df = _stream(31)
        spec, out, ref = _fit(
            df, half_life=40.0, min_weight=[20.0, 12.0, 25.0], solve_every=2.5, l1_ratio=1.0
        )
        # A list of thresholds: a target's selection folds the model's own
        # predictions for it, whatever its own threshold withholds (task 96).
        _held_to_the_reference(df, spec, out, ref)
        # Each target reports from its own weight, not the shared one: the
        # gappy target waits for its own rows to reach its threshold, after
        # the shared weight_sum has.
        first_own = int(np.argmax(np.isfinite(ref["pred"][:, 1, 0])))
        first_shared = int(np.argmax(ref["weight_sum"] >= 12.0))
        assert first_own > first_shared, (first_own, first_shared)

    def test_pairwise_reads_the_features_over_every_row(self):
        """``pairwise``: the feature correlations and scales over every row,
        each target's cross-moments over its own rows and centred at its own
        means. An elastic net, with a row cap beside the clock."""
        df = _stream(32)
        spec, out, ref = _fit(
            df,
            half_life=40.0,
            min_weight=20.0,
            solve_every=6.0,
            max_rows_between_solves=3,
            l1_ratio=0.5,
            target_gaps="pairwise",
        )
        _held_to_the_reference(df, spec, out, ref)


class TestSelection:
    """``lam_selected`` ranks each path point on its own decay: a
    ``select_half_life`` of 10 against a model half-life of 40."""

    def test_the_selection_decays_at_its_own_halflife(self):
        df = _stream(33)
        spec, out, ref = _fit(
            df, half_life=40.0, select_half_life=10.0, min_weight=15.0, solve_every=2.0
        )
        _held_to_the_reference(df, spec, out, ref)


class TestWithoutAnIntercept:
    """``fit_intercept=False``: nothing centred, each column scaled by its
    root mean square, and the features at levels a centred fit would have
    absorbed into an intercept it does not have."""

    def test_the_path_reads_the_raw_moments(self):
        df = _stream(34, level=3.0)
        spec, out, ref = _fit(
            df, half_life=60.0, min_weight=15.0, solve_every=2.0, fit_intercept=False
        )
        _held_to_the_reference(df, spec, out, ref)


class TestWindow:
    """A ``window`` of 24 clock units under a half-life of 40: the rows whose
    age on the capped clock is at most 24, each at its decayed weight --
    ``weight_sum``, each target's gate and every fit. The block of nulls in the
    third target outlives the window, so that target empties and comes back
    while the others stay windowed."""

    def _check(self, target_gaps, **kw):
        df = _stream(35)
        spec, out, ref = _fit(
            df,
            half_life=40.0,
            min_weight=10.0,
            solve_every=1.0,
            window_size=24.0,
            target_gaps=target_gaps,
            **kw,
        )
        # The third target's block of nulls empties its window for a while,
        # so it has fewer rows with a selection than an unwindowed one.
        _held_to_the_reference(df, spec, out, ref, min_selected=100)
        # The window really cut: weight_sum sits below the unwindowed weight.
        weight_sum = out.struct.field("weight_sum").to_numpy().astype(float)
        assert np.nanmax(weight_sum) < 0.8 * (1.0 / (1.0 - 0.5 ** (1.0 / 40.0))), weight_sum.max()
        # The emptied target had no fit to report.
        assert np.isnan(ref["pred"][130:140, 2, 0]).all()

    def test_own_rows(self):
        self._check("own_rows")

    def test_pairwise(self):
        self._check("pairwise")

    def test_the_selection_inside_the_window_decays_at_its_own_halflife(self):
        """Task 95's third departure: the window's selection aged its
        boundary by the model's half-life, not ``select_half_life``."""
        self._check("own_rows", select_half_life=10.0)

    def test_a_window_down_to_one_row_of_a_target_fits_its_intercept_alone(self):
        """Task 94. The third target is present on the first row of every
        32 clock units, so the window of 24 holds at most one of its rows.
        Every feature holds one value on that row, so every one is dropped
        and the fit is the target's value, at every penalty. The core kept
        each feature as the remainder its subtraction left and, at a penalty
        of zero, predicted values like 8.6e33 and 2.8e55 where the fit was
        -1.0. The third target's gate is lowered to half a row, so those
        rows are scored; the others' stay at 10, and a solve on every row
        keeps the reference's first-solve rule out of it."""
        df = _stream(35)
        # On the model's clock, whose steps ``gap_cap`` caps and a row
        # skipped for a null feature folds into the next: 32 units a bucket,
        # so no two of the target's rows are within 24 of each other.
        dt = np.diff(df["t"].to_numpy(), prepend=0.0)
        skipped = df.select(pl.any_horizontal(pl.col(FEATURES).is_null())).to_series().to_numpy()
        clock, pending, last = np.zeros(df.height), 0.0, 0.0
        for i in range(df.height):
            if skipped[i]:
                clock[i], pending = last, pending + dt[i]
                continue
            last = last + min(dt[i] + pending, MAX_DCLOCK) if i else 0.0
            clock[i], pending = last, 0.0
        bucket = np.floor(clock / 32.0)
        first = np.r_[True, bucket[1:] != bucket[:-1]] & ~skipped
        df = df.with_columns(
            yc=pl.when(pl.Series(first)).then(-2.0 + 1.5 * pl.col("x3") - 0.4 * pl.col("x0"))
        )
        spec, out, ref = _fit(
            df,
            half_life=40.0,
            min_weight=[10.0, 10.0, 0.5],
            max_rows_between_solves=1,
            window_size=24.0,
        )
        index = po.spec.output_index(spec)
        for j, t in enumerate(TARGETS):
            for p, lam in enumerate(PATH):
                rows = index.filter(
                    (pl.col("target") == t)
                    & (pl.col("penalty") == lam)
                    & (pl.col("kind") == "pred")
                )
                got = out.struct.field(rows["field"][0]).to_numpy().astype(float)
                _close(got, ref["pred"][:, j, p], PRED_TOL, f"pred_{t} at {lam}")
        _close(
            out.struct.field("weight_sum").to_numpy().astype(float),
            ref["weight_sum"],
            PRED_TOL,
            "weight_sum",
        )
        scored = np.isfinite(ref["pred"][:, 2, -1])
        assert scored.sum() >= 20, f"only {scored.sum()} rows scored from a window of one row"
        assert (ref["w_target"][scored, 2] <= 1.5).all(), "one row of the target in the window"
