"""The robust scale a Huber cut is drawn in: ``huber_scale`` and
``huber_scale_half_life`` on ``huber`` and on ``sgd(loss="huber")`` (task
271; the analysis is ``docs/HUBER.md``).

Each test is one of the record's tables, through the library (the MAD
scale's slope against statsmodels' converged ``RLM``, §2 and §3, is
``tests/test_second_opinion.py``'s ``TestHuberMadAgainstStatsmodels``):
``sgd``'s Huber loss at a level of 1,000 with no decay, which
under the RMS is the squared loss to the bit (C-1 of review round 9), the
level shift a scale with a shorter memory follows (§4), and chunk
invariance and a save and resume with the sketches in the state.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.robust.norms import HuberT

import polars_online as po

TIER = "mixed"

ROBUST_RS = Path(__file__).resolve().parents[1] / "crates/online-core/src/robust.rs"


def stream(n: int, frac: float, seed: int, level: float = 1000.0) -> pl.DataFrame:
    """``docs/HUBER.md`` §2: ``y = level + 2x + 0.5ε``, +50 on a share
    ``frac`` of the rows with ``x > 0``, which tilts the slope."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
    y[(rng.random(n) < frac) & (x > 0)] += 50.0
    return pl.DataFrame({"x": x, "y": y})


NO_DECAY = dict(targets=["y"], features=["x"], half_life=float("inf"))


def slope(spec: dict, df: pl.DataFrame) -> float:
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return float(bank.coef(spec["name"])["coef"].to_list()[1])


def test_the_mad_constant_is_the_normals_third_quartile():
    """``NORMAL_Q3``, a literal so that no libm result enters the state, is
    scipy's ``norm.ppf(0.75)`` to the last digit written."""
    literal = re.search(
        r"pub const NORMAL_Q3: f64 = ([0-9._]+);", ROBUST_RS.read_text(encoding="utf-8")
    )
    assert literal, "robust.rs defines NORMAL_Q3"
    assert float(literal.group(1).replace("_", "")) == pytest.approx(norm.ppf(0.75), abs=1e-16)


@pytest.mark.parametrize("frac", [0.02, 0.10], ids=["1% of rows", "5% of rows"])
def test_sgd_huber_at_a_level_is_robust_under_the_mad(frac):
    """C-1 (review round 9): with no decay the first residuals, the level's
    size, held the RMS scale wide for good, so ``sgd(loss="huber")`` was the
    squared loss to the bit. The intercept's running-mean step takes the
    level from the first row (task 278, D-S), so the RMS scale no longer
    keeps it and its slope is nearer the truth than the squared loss's too,
    the outliers alone inflating it; under the MAD nearer still. Measured at
    5% (seeds 0-4, median): 2.013 under the MAD, 2.557 under the RMS, 3.751
    for the squared loss; at 1%: 1.990, 2.067 and 2.322 (2026-10-26)."""
    df = stream(20_000, frac, 3)
    squared = slope(po.spec.sgd("s", **NO_DECAY), df)
    # One pass under the RMS, as 1.2.0 ran it: the refit of the first rows
    # (task 277) would move it off the squared loss.
    rms = slope(
        po.spec.sgd("s", loss="huber", huber_scale="rms", huber_refit_rows=0, **NO_DECAY), df
    )
    mad = slope(po.spec.sgd("s", loss="huber", **NO_DECAY), df)
    assert abs(rms - 2.0) < abs(squared - 2.0), (rms, squared)
    assert abs(mad - 2.0) < abs(rms - 2.0), (mad, rms)
    assert abs(mad - 2.0) < 0.25 * abs(squared - 2.0), (mad, squared)
    assert abs(mad - 2.0) < 0.05, mad


def test_a_scale_with_a_shorter_memory_follows_a_level_shift():
    """``docs/HUBER.md`` §4 through the model: a fit half-life of 500 rows,
    a +20 level shift at row 10,000 and no outliers. The MAD at the fit's
    memory holds every post-shift row back as an outlier until its median
    has crossed to the new level; at a fifth of it, it adapts as least
    squares does. Measured on seeds 1-3, the mean absolute error over rows
    200-400 after the shift: 13.33-13.39 for least squares, 19.52-19.57 for
    the MAD at 500, 14.55-14.61 at 100 (1.09 of least squares')."""
    rng = np.random.default_rng(1)
    n = 14_000
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.5 * rng.standard_normal(n)
    y[10_000:] += 20.0
    df = pl.DataFrame({"x": x, "y": y, "t": np.arange(n, dtype=float)})
    c = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0)

    def late_error(spec):
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        return float(np.abs(y - pred)[10_200:10_400].mean())

    least_squares = late_error(po.spec.huber("m", huber_delta=float("inf"), **c))
    fit_memory = late_error(po.spec.huber("m", **c))
    short = late_error(po.spec.huber("m", huber_scale_half_life=100.0, **c))
    assert short < 1.10 * least_squares, (short, least_squares)
    assert fit_memory > 1.3 * least_squares, (fit_memory, least_squares)
    # The same lever on sgd's Huber loss.
    sgd_fit = late_error(po.spec.sgd("m", loss="huber", **c))
    sgd_short = late_error(po.spec.sgd("m", loss="huber", huber_scale_half_life=100.0, **c))
    assert sgd_short < 0.5 * sgd_fit, (sgd_short, sgd_fit)


def test_sgd_follows_a_level_shift_with_a_fifth_of_the_fits_memory():
    """``sgd``'s Huber gradient is clipped at ``delta * s``, so under the
    MAD at the fit's memory it follows a level shift seven times worse than
    under the RMS, the default before the MAD; a scale half-life of a fifth
    of the fit's brings it within 1.3x of the RMS (MODELS.md). The stream of
    the test above, seed 1; the mean absolute error over rows 200-400 after
    the shift, measured: 1.17 for the squared loss, 2.45 for the RMS, 17.33
    for the MAD at the fit's 500 rows, 2.99 at 100 and 1.41 at 20 (seeds 2
    and 3 within 0.06 of each)."""
    rng = np.random.default_rng(1)
    n = 14_000
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.5 * rng.standard_normal(n)
    y[10_000:] += 20.0
    df = pl.DataFrame({"x": x, "y": y, "t": np.arange(n, dtype=float)})
    c = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0)

    def late_error(spec):
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        return float(np.abs(y - pred)[10_200:10_400].mean())

    rms = late_error(po.spec.sgd("m", loss="huber", huber_scale="rms", **c))
    fit_memory = late_error(po.spec.sgd("m", loss="huber", **c))
    fifth = late_error(po.spec.sgd("m", loss="huber", huber_scale_half_life=100.0, **c))
    assert fifth < 1.3 * rms, (fifth, rms)
    assert fit_memory > 5.0 * rms, (fit_memory, rms)
    assert fit_memory == pytest.approx(17.33, abs=0.01)


def _specs():
    c = dict(
        targets=["y0", "y1"],
        features=["x"],
        clock="t",
        gap_cap=50.0,
        half_life=300.0,
        coef_every=1,
    )
    return [
        po.spec.huber("m", huber_scale_half_life=60.0, **c),
        po.spec.huber("m", huber_scale="rms", **c),
        po.spec.sgd("m", loss="huber", huber_scale_half_life=60.0, **c),
        po.spec.sgd("m", loss="huber", huber_scale="rms", **c),
    ]


SPEC_IDS = ["huber mad", "huber rms", "sgd mad", "sgd rms"]


def _frame(n: int = 1400) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n)
    y0 = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
    y0[(rng.random(n) < 0.1) & (x > 0)] += 50.0
    y1 = -3.0 * x + rng.standard_t(3, n)
    y1[rng.random(n) < 0.1] = None
    w = np.where(np.arange(n) % 11 == 5, 0.0, rng.uniform(0.5, 1.5, n))
    t = np.cumsum(rng.integers(1, 4, n)).astype(float)
    return pl.DataFrame({"x": x, "y0": y0, "y1": y1, "w": w, "t": t})


@pytest.mark.parametrize("spec", _specs(), ids=SPEC_IDS)
def test_one_chunk_or_many_give_one_run(spec):
    """Hard rule 3 with the sketches in the state: chunks of 1, 7 and 600
    rows give the one chunk's output to the bit."""
    df = _frame()
    spec = {**spec, "weight": "w"}
    whole = po.ModelBank([spec]).fit_predict(df)
    for rows in (1, 7, 600):
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
        assert pl.concat(parts).equals(whole, null_equal=True), rows


@pytest.mark.parametrize("spec", _specs(), ids=SPEC_IDS)
def test_a_bank_saved_anywhere_resumes_to_the_bit(spec):
    """Saved and loaded at rows across the stream, a bank goes on as the
    one that never stopped, every field to the bit: the sketches are
    state."""
    df = _frame()
    spec = {**spec, "weight": "w"}
    whole = po.ModelBank([spec]).fit_predict(df)
    for cut in (1, 2, 37, 500, 1001, 1399):
        bank = po.ModelBank([spec])
        head = bank.fit_predict(df.slice(0, cut))
        loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
        tail = loaded.fit_predict(df.slice(cut))
        assert pl.concat([head, tail]).equals(whole, null_equal=True), cut


def test_the_scale_is_refused_beside_another_loss():
    with pytest.raises(ValueError, match='huber_scale is for loss "huber"'):
        po.spec.sgd("s", huber_scale="mad", **NO_DECAY)
    with pytest.raises(ValueError, match='huber_scale_half_life is for loss "huber"'):
        po.spec.sgd("s", loss="quantile", quantile=0.5, huber_scale_half_life=5.0, **NO_DECAY)
    with pytest.raises(ValueError, match='huber_scale must be one of "mad", "rms"'):
        po.spec.huber("h", huber_scale="median", **NO_DECAY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="huber_scale_half_life must be > 0"):
        po.spec.huber("h", huber_scale_half_life=0.0, **NO_DECAY)
    with pytest.raises(ValueError, match="huber_scale_half_life must be > 0"):
        po.spec.sgd("s", loss="huber", huber_scale_half_life=0.0, **NO_DECAY)
    with pytest.raises(ValueError, match="huber_scale_half_life must be > 0"):
        po.spec.sgd("s", loss="huber", huber_scale_half_life=-1.0, **NO_DECAY)


#: Four rows on which `huber`'s MAD scale panicked during 1.3.0's
#: development (docs/PLAN.md task 272 (p)): the residuals' median sat at 0
#: while a residual arrived more than the scale's band below every other,
#: and the sketch's fold took its end bucket's place as a negative index.
PANIC_ROWS = pl.DataFrame(
    {
        "t": [50003.5, 60004.5, 70013.0, 80027.0],
        "x0": [-1e100, -1e100, -1e8, -0.0],
        "x1": [-0.0, 1e-12, -0.0, 1e100],
        "y0": [1e-12, 1e8, 1e-12, 0.0],
        "w": [1e100, 7.7322, 1e100, 1e100],
    }
)


def test_a_residual_far_below_a_median_at_zero_does_not_panic():
    """A residual more than the band below the rest, while the median is 0,
    folds every other bucket into the band's end; every output is finite or
    null, and one chunk or many give the same numbers (`coef` is emitted on
    each chunk's last row, so it is left out)."""
    spec = po.spec.huber(
        "m",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        half_life=20.0,
        gap_cap=100.0,
        weight="w",
        min_weight=2.0,
        max_rows_between_solves=1,
    )
    whole = po.ModelBank([spec]).fit_predict(PANIC_ROWS)
    fields = whole.select("m").unnest("m")
    for name, dtype in fields.schema.items():
        if dtype.is_float():
            vals = fields[name].drop_nulls().to_numpy()
            assert np.isfinite(vals).all(), name
    bank = po.ModelBank([spec])
    parts = pl.concat([bank.fit_predict(PANIC_ROWS.slice(i, 1)) for i in range(4)])
    numbers = pl.exclude("coef", "support_coef")
    got = parts.select("m").unnest("m").select(numbers)
    assert got.equals(fields.select(numbers), null_equal=True)


# ---- task 278, batch 19: the scale's memory, the infinite cut, powers of
# two and the cut's contamination bias, on both Huber models ----

INF = float("inf")
KINDS = ["huber", "sgd"]


def _kind(kind: str, **kw) -> dict:
    """A spec of ``kind`` on ``_frame``'s two targets: ``huber``, or ``sgd``
    under the Huber loss, at a half-life of 300 on the clock ``t``, every
    row's ``coef`` emitted."""
    c = dict(
        targets=["y0", "y1"],
        features=["x"],
        clock="t",
        gap_cap=50.0,
        half_life=300.0,
        weight="w",
        coef_every=1,
    )
    c.update(kw)
    if kind == "huber":
        return po.spec.huber("m", **c)
    return po.spec.sgd("m", loss="huber", **c)


def _run(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([spec]).fit_predict(df)


class TestTheScalesMemory:
    """``huber_scale_half_life`` left out is the model's half-life: given as
    that, as the same span in a duration, or as ``inf`` beside an infinite
    half-life, every field is the default run's to the bit, the refit of
    the first rows on (it runs here: each target passes its 500th learned
    row, and the refit moves the run)."""

    @pytest.mark.parametrize("scale", ["mad", "rms"])
    @pytest.mark.parametrize("kind", KINDS)
    def test_given_as_the_models_half_life_it_is_the_default(self, kind, scale):
        df = _frame()
        default = _run(_kind(kind, huber_scale=scale), df)
        given = _run(_kind(kind, huber_scale=scale, huber_scale_half_life=300.0), df)
        assert given.equals(default, null_equal=True)
        # A memory a row's clock unit shorter moves it, and the refit ran.
        other = _run(_kind(kind, huber_scale=scale, huber_scale_half_life=299.0), df)
        assert not other.equals(default, null_equal=True)
        unrefit = _run(_kind(kind, huber_scale=scale, huber_refit_rows=0), df)
        assert not unrefit.equals(default, null_equal=True)

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_duration_is_its_seconds(self, kind):
        """On a ``Datetime`` clock a duration is the span it names: ``"1m"``
        is ``60`` on the same clock in seconds, every number field the
        same."""
        df = _frame()
        start = dt.datetime(2024, 1, 1)
        dated = df.with_columns(
            (pl.lit(start) + pl.duration(seconds=pl.col("t").cast(pl.Int64))).alias("t")
        )
        clock = dict(gap_cap="50s", half_life="300s", coef_every="1s")
        got = _run(_kind(kind, huber_scale_half_life="1m", **clock), dated)
        want = _run(_kind(kind, huber_scale_half_life=60.0), df)
        assert got.drop("t").equals(want.drop("t"), null_equal=True)
        default = _run(_kind(kind), df)
        assert not want.equals(default, null_equal=True)

    @pytest.mark.parametrize("kind", KINDS)
    def test_inf_beside_an_infinite_half_life_is_the_default(self, kind):
        df = _frame()
        default = _run(_kind(kind, half_life=INF), df)
        assert _run(_kind(kind, half_life=INF, huber_scale_half_life=INF), df).equals(
            default, null_equal=True
        )


class TestTheSgdInfiniteCut:
    """``sgd(loss="huber")`` at ``huber_delta=inf`` is ``loss="squared"``, to
    the bit, on two targets with gaps past ``gap_cap``, rows of weight 0 and
    a target null on a tenth of its rows, under both scales and the default
    ``standardize``. The refit of the first rows is off there by default and
    refused when asked for (an infinite cut has no outliers to refit for),
    so the default, which refits a finite cut's first 500 rows, is the
    squared loss's run."""

    @pytest.mark.parametrize(
        "kw",
        [{}, {"huber_scale": "rms"}, {"schedule": "adagrad", "learning_rate": 0.5}],
        ids=["mad", "rms", "adagrad"],
    )
    def test_it_is_the_squared_loss_to_the_bit(self, kw):
        df = _frame()
        squared = {k: v for k, v in kw.items() if k != "huber_scale"}
        c = dict(
            targets=["y0", "y1"],
            features=["x"],
            clock="t",
            gap_cap=50.0,
            half_life=300.0,
            weight="w",
            coef_every=1,
        )
        inf = _run(po.spec.sgd("m", loss="huber", huber_delta=INF, **c, **kw), df)
        assert inf.equals(_run(po.spec.sgd("m", **c, **squared), df), null_equal=True)
        # A finite cut moves it, its refit included.
        assert not inf.equals(_run(po.spec.sgd("m", loss="huber", **c, **kw), df), null_equal=True)

    def test_a_refit_is_refused_beside_it(self):
        with pytest.raises(ValueError, match="huber_refit_rows needs a finite huber_delta"):
            po.ModelBank(
                [po.spec.sgd("s", loss="huber", huber_delta=INF, huber_refit_rows=10, **NO_DECAY)]
            )


#: Task 278 (b19), a defect: `sgd`'s refit solves its buffered rows in the
#: caller's units with a fixed ridge of 1e-6 on the mean-form system
#: (`sgd.rs`, `REFIT_RIDGE`), whatever `standardize` says, so the ridge is
#: in the features' squared units. A feature whose variance is near 1e-6 (a
#: return in fractions with a spread of 1e-3) has its slope shrunk to about
#: half in the refit, and the steps after it start from there: with `x`
#: times 2^-10 the predictions part from the unscaled run's from the row
#: after each target's refit, by up to 1.8 on `y0` (at 1,000, slope 2) and
#: 2.9 on `y1` (slope -3), measured 2026-10-26, where the one pass is exact.
SGD_REFIT_RIDGE = pytest.mark.xfail(
    strict=True,
    reason="task 278 (b19): sgd's refit ridge is in the features' units under standardize",
)


class TestSgdPowersOfTwo:
    """``sgd`` standardizes its features by default, so scaling ``x`` by
    ``2^k`` changes nothing but its coefficient, divided by ``2^k``: the
    scaler's moments scale exactly, and so does every step mapped through
    them, the clip and the level rule read the target, not the feature."""

    @staticmethod
    def check(df, **kw):
        base = _run(_kind("sgd", **kw), df)
        width = 4  # [intercept, x] for each of the two targets
        coef = np.array(
            [[np.nan] * width if r is None else r for r in base["m"].struct.field("coef")],
            dtype=float,
        )
        for k in (10, -10, 30):
            out = _run(_kind("sgd", **kw), df.with_columns(pl.col("x") * 2.0**k))
            got = np.array(
                [[np.nan] * width if r is None else r for r in out["m"].struct.field("coef")],
                dtype=float,
            )
            m = pl.col("m").struct.field
            fields = [f for f in base["m"].struct.fields if f != "coef"]
            assert out.select(m(*fields)).equals(base.select(m(*fields)), null_equal=True), k
            np.testing.assert_array_equal(got[:, [1, 3]] * 2.0**k, coef[:, [1, 3]])
            np.testing.assert_array_equal(got[:, [0, 2]], coef[:, [0, 2]])

    def test_a_feature_scaled_by_a_power_of_two_changes_only_its_coefficient(self):
        """The one pass, the refit of the first rows off."""
        self.check(_frame(), huber_refit_rows=0)

    def test_with_the_refit_of_the_first_rows_it_holds_until_the_refit(self):
        """The default, the refit on: ``y0`` passes its 500th learned row
        at row 549, and on every row before it the relation holds."""
        df = _frame()
        assert int(np.argmax(np.cumsum(df["w"].to_numpy() > 0) >= 500)) == 549
        self.check(df.head(549))

    @SGD_REFIT_RIDGE
    def test_and_so_after_the_refit(self):
        self.check(_frame())


#: The cuts the contamination grid walks, from least squares inwards.
DELTAS = [INF, 8.0, 4.0, 2.0, 1.345, 1.0]


def _biases(kind: str, share: float, seed: int, n: int = 5000) -> np.ndarray:
    """``|slope - 2|`` at the end of ``stream(n, share, seed)`` (the slope's
    bias, the +50 rows on ``x > 0`` tilting it) for each of ``DELTAS``,
    with no decay and every other keyword the default, the refit of the
    first 500 rows included."""
    df = stream(n, share, seed)
    out = []
    for delta in DELTAS:
        c = dict(NO_DECAY, huber_delta=delta)
        spec = po.spec.huber("m", **c) if kind == "huber" else po.spec.sgd("m", loss="huber", **c)
        out.append(abs(slope(spec, df) - 2.0))
    return np.array(out)


@pytest.mark.parametrize(("share", "seed"), [(0.1, 7), (0.25, 0)])
def test_a_narrower_cut_never_tilts_huber_further(share, seed):
    """The contamination grid's representative (``test_the_cuts_bias_over_
    fifty_seeds`` has the rest): on the seed whose bias came nearest to
    rising as the cut narrowed, it falls or stays at each step, from least
    squares' (above 1) to under 0.1 at the default cut."""
    b = _biases("huber", share, seed)
    print(f"share {share}, seed {seed}: {np.round(b, 4)}")
    assert (np.diff(b) <= 0.0).all(), b
    assert b[0] > 1.0 and b[DELTAS.index(1.345)] < 0.2, b


@pytest.mark.oracle_grid(reason="a grid of 50 seeds at two shares, with statsmodels beside it")
def test_the_cuts_bias_over_fifty_seeds():
    """``huber_delta`` against contamination (task 278, class 4): ``y = 1000
    + 2x + 0.5ε`` with +50 (100 noise standard deviations) on a share of
    the rows with ``x > 0`` (5% and 12.5% of all rows), 5,000 rows, no
    decay, seeds 0-49, the cut walked from ``inf`` to 1. Judged by the
    worst seed:

    - ``huber``: on every seed the slope's bias falls or stays at each
      narrower cut. Measured 2026-10-26: the nearest any came to rising was
      a fall of 1.1e-3 (seed 7, 1.345 to 1, at 5%); at 12.5% the cut of 8
      never binds and is least squares to the bit on every seed.
    - ``sgd``: a constant rate keeps the slope wandering, by 0.05 to 0.11
      about statsmodels' fit at the narrow cuts (95th percentile and worst),
      so a seed's bias can rise by that much between two cuts (0.031,
      seed 30, 8 to 4 at 5%): the worst seed's and the 95th percentile's
      biases fall or stay at each narrower cut.
    - Both against statsmodels' converged ``RLM`` (``HuberT(delta)``, the
      MAD scale) on the same rows: at the cuts of 2 and under, ``huber``
      ends within 0.05 of its slope on every seed (worst 2.4e-2, at 2 and
      12.5%). At the wide cuts the two can end on different fixed points of
      the joint scale and fit: at 8 and 5% the default refit of the first
      500 rows lands, as ``RLM`` on those rows does, on the fit the
      outliers hold, and seeds 9 and 15 end 1.8 and 1.4 from the whole
      stream's ``RLM`` (0.14 and 0.27 with the refit off). Printed, not
      held."""
    for share in (0.1, 0.25):
        b = {k: np.array([_biases(k, share, s) for s in range(50)]) for k in KINDS}
        rlm = []
        for seed in range(50):
            df = stream(5000, share, seed)
            X = sm.add_constant(df["x"].to_numpy())
            y = df["y"].to_numpy()
            row = [abs(sm.OLS(y, X).fit().params[1] - 2.0)]
            for delta in DELTAS[1:]:
                fit = sm.RLM(y, X, M=HuberT(delta)).fit(
                    scale_est="mad", conv="coefs", tol=1e-10, maxiter=500
                )
                row.append(abs(fit.params[1] - 2.0))
            rlm.append(row)
        rlm = np.array(rlm)
        for kind in KINDS:
            worst, p95 = b[kind].max(axis=0), np.quantile(b[kind], 0.95, axis=0)
            rise = np.diff(b[kind], axis=1).max()
            print(f"{kind} at {share}: worst {np.round(worst, 4)}, 95th {np.round(p95, 4)}")
            print(f"  RLM's worst {np.round(rlm.max(axis=0), 4)}; largest rise {rise:.2e}")
            assert (np.diff(worst) <= 0.0).all(), (kind, share, worst)
            assert (np.diff(p95) <= 0.0).all(), (kind, share, p95)
        assert (np.diff(b["huber"], axis=1) <= 0.0).all(), share
    for share in (0.1, 0.25):
        narrow = [DELTAS.index(d) for d in (2.0, 1.345, 1.0)]
        for seed in range(50):
            df = stream(5000, share, seed)
            X = sm.add_constant(df["x"].to_numpy())
            y = df["y"].to_numpy()
            for i in narrow:
                want = sm.RLM(y, X, M=HuberT(DELTAS[i])).fit(
                    scale_est="mad", conv="coefs", tol=1e-10, maxiter=500
                )
                got = slope(po.spec.huber("m", huber_delta=DELTAS[i], **NO_DECAY), df)
                assert abs(got - want.params[1]) < 0.05, (share, seed, DELTAS[i], got)
