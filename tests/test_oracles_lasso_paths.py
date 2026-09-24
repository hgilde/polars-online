"""The lasso's paths ``tests/test_oracles.py::TestLassoPredPath`` does not
reach, held to ``tests/reference_paths.py::lasso_paths_ref`` row by row:
several targets, null targets, ``target_gaps``, ``add_intercept=False``, a
``window`` and ``lam_selected`` at a ``select_halflife`` of its own.

The reference recomputes every statistic from the raw rows at each solve,
each row at ``w * 0.5 ** (age / halflife)``, and descends from zero to 1e-14,
so neither the core's mean-form recursion nor its warm start can agree with
it by construction. Every stream here has an irregular dyadic clock with two
gaps past ``max_dclock``, zero-weight rows (the first among them), rows
skipped for a null feature, and three targets: one with a few random nulls,
one present only where ``x0 > -0.5`` (gaps tied to a feature, so
``own_rows`` and ``pairwise`` part), and one with a block of nulls that the
window outlives.

Two things are left out on purpose, each a disagreement with the documented
reading, reported rather than held (2026-09-24):

- ``lam_selected`` under a ``window``. The core truncates the selection error
  against a snapshot taken after the row's own error has been folded in and
  its weight aged twice, reads the boundary and clock of the previous row,
  and uses the model's decay factor where ``select_halflife`` differs. A
  scratch emulation of that arithmetic reproduces the library on every row;
  with the three corrected it lands on this reference on every row.
- ``lam_selected`` with a ``min_periods`` list. The model is ready once
  ``n_eff`` reaches the smallest threshold (docs/ENHANCEMENTS.md E7), and a
  target's selection error then folds its own predictions on rows its own,
  larger threshold still withholds. That matches review S2's "a gate on the
  output, not on the model", but E7 also says a not-yet-ready target's slots
  are withheld before they reach selection. The scalar case, where the two
  readings agree, is held.
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
# than a target at a level of 5), n_eff 2.3e-15, coef 1.7e-12. Each
# tolerance is 100x the largest it covers, rounded up to a power of ten.
#
# Seeded into a copy of the reference, each of these fails a case here by
# far more (pred, same measure): own_rows and pairwise swapped 0.23-4.6;
# pairwise cross-moments centred at every row's means (review N3) 0.04-0.10;
# centred without an intercept (C8) 4.4; the window's boundary strict (`<`)
# 0.30-0.39, or no window 0.80-1.1; the decay at twice the halflife
# 0.04-0.12; each target gated on the shared n_eff (S2) moves the null
# pattern on 18-162 rows; the selection at the model's halflife instead of
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
        max_dclock=MAX_DCLOCK,
        weight="w",
        coef_every=1,
        cd_tol=1e-14,
        max_cd_iters=100_000,
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
        max_dclock=MAX_DCLOCK,
        **kw,
    )
    return spec, out, ref


def _held_to_the_reference(df, spec, out, ref, lam_selected=True):
    """Every path point's ``pred`` and ``resid`` for every target, ``n_eff``,
    every held ``coef`` and, where asked, ``lam_selected``; then two probes
    that the comparison can fail."""
    n, kt = df.height, len(ref["coef"][0, 0, 0])
    index = po.spec.output_index(spec)
    y = df.select(TARGETS).to_numpy().astype(float)
    got = np.full_like(ref["pred"], np.nan)
    for j, t in enumerate(TARGETS):
        for p, lam in enumerate(PATH):
            rows = index.filter(
                (pl.col("target") == t) & (pl.col("lambda") == lam) & pl.col("kind").is_in(["pred"])
            )
            got[:, j, p] = out.struct.field(rows["field"][0]).to_numpy().astype(float)
            _close(got[:, j, p], ref["pred"][:, j, p], PRED_TOL, f"pred_{t} at {lam}")
            resid_field = rows["field"][0].replace("pred_", "resid_", 1)
            resid = out.struct.field(resid_field).to_numpy().astype(float)
            _close(resid, y[:, j] - ref["pred"][:, j, p], PRED_TOL, f"resid_{t} at {lam}")
        assert np.isfinite(ref["pred"][:, j, 0]).sum() > 100, f"{t} is scored too little"
        if lam_selected:
            sel = out.struct.field(f"lam_selected_{t}").to_numpy().astype(float)
            held = ~np.isnan(ref["lam_selected"][:, j])
            assert held.sum() > 150, f"{t}: too few rows with a clear selection"
            wrong = np.flatnonzero(held & (sel != ref["lam_selected"][:, j]))
            assert wrong.size == 0, f"lam_selected_{t} differs at rows {wrong[:8]}"
            assert len(set(sel[held])) > 1, f"lam_selected_{t} never moves"
    _close(out.struct.field("n_eff").to_numpy().astype(float), ref["n_eff"], PRED_TOL, "n_eff")

    rows = out.struct.field("coef").to_list()
    empty = [np.nan] * (len(TARGETS) * len(PATH) * kt)
    coef = np.array([empty if r is None else r for r in rows], float).reshape(ref["coef"].shape)
    first = np.argmax(ref["solved"])
    expect_null = (np.arange(n) < first) | np.isnan(ref["n_eff"])
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
            df, halflife=40.0, min_periods=[20.0, 12.0, 25.0], solve_every=2.5, l1_ratio=1.0
        )
        # A list of thresholds: lam_selected is left out (the module docstring).
        _held_to_the_reference(df, spec, out, ref, lam_selected=False)
        # Each target reports from its own weight, not the shared one: the
        # gappy target waits for its own rows to reach its threshold, after
        # the shared n_eff has.
        first_own = int(np.argmax(np.isfinite(ref["pred"][:, 1, 0])))
        first_shared = int(np.argmax(ref["n_eff"] >= 12.0))
        assert first_own > first_shared, (first_own, first_shared)

    def test_pairwise_reads_the_features_over_every_row(self):
        """``pairwise``: the feature correlations and scales over every row,
        each target's cross-moments over its own rows and centred at its own
        means. An elastic net, with a row cap beside the clock."""
        df = _stream(32)
        spec, out, ref = _fit(
            df,
            halflife=40.0,
            min_periods=20.0,
            solve_every=6.0,
            max_rows_between_solves=3,
            l1_ratio=0.5,
            target_gaps="pairwise",
        )
        _held_to_the_reference(df, spec, out, ref)


class TestSelection:
    """``lam_selected`` ranks each path point on its own decay: a
    ``select_halflife`` of 10 against a model halflife of 40."""

    def test_the_selection_decays_at_its_own_halflife(self):
        df = _stream(33)
        spec, out, ref = _fit(
            df, halflife=40.0, select_halflife=10.0, min_periods=15.0, solve_every=2.0
        )
        _held_to_the_reference(df, spec, out, ref)


class TestWithoutAnIntercept:
    """``add_intercept=False``: nothing centred, each column scaled by its
    root mean square, and the features at levels a centred fit would have
    absorbed into an intercept it does not have."""

    def test_the_path_reads_the_raw_moments(self):
        df = _stream(34, level=3.0)
        spec, out, ref = _fit(
            df, halflife=60.0, min_periods=15.0, solve_every=2.0, add_intercept=False
        )
        _held_to_the_reference(df, spec, out, ref)


class TestWindow:
    """A ``window`` of 24 clock units under a halflife of 40: the rows whose
    age on the capped clock is at most 24, each at its decayed weight --
    ``n_eff``, each target's gate and every fit. The block of nulls in the
    third target outlives the window, so that target empties and comes back
    while the others stay windowed."""

    def _check(self, target_gaps):
        df = _stream(35)
        spec, out, ref = _fit(
            df,
            halflife=40.0,
            min_periods=10.0,
            solve_every=1.0,
            window=24.0,
            target_gaps=target_gaps,
        )
        # lam_selected under a window is left out (the module docstring).
        _held_to_the_reference(df, spec, out, ref, lam_selected=False)
        # The window really cut: n_eff sits below the unwindowed weight.
        n_eff = out.struct.field("n_eff").to_numpy().astype(float)
        assert np.nanmax(n_eff) < 0.8 * (1.0 / (1.0 - 0.5 ** (1.0 / 40.0))), n_eff.max()
        # The emptied target had no fit to report.
        assert np.isnan(ref["pred"][130:140, 2, 0]).all()

    def test_own_rows(self):
        self._check("own_rows")

    def test_pairwise(self):
        self._check("pairwise")
