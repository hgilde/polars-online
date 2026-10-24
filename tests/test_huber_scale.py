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

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

import polars_online as po

TIER = "essential"

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
    squared loss to the bit. Under the MAD its slope is nearer the truth
    than the squared loss's. Measured at 5% (seeds 0-4, median): 2.016
    against 3.751 for both the squared loss and the RMS scale."""
    df = stream(20_000, frac, 3)
    squared = slope(po.spec.sgd("s", **NO_DECAY), df)
    rms = slope(po.spec.sgd("s", loss="huber", huber_scale="rms", **NO_DECAY), df)
    mad = slope(po.spec.sgd("s", loss="huber", **NO_DECAY), df)
    assert rms == squared, "the RMS scale at a level is the squared loss, kept as it was"
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
    ]


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


@pytest.mark.parametrize("spec", _specs(), ids=["huber mad", "huber rms", "sgd mad"])
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


@pytest.mark.parametrize("spec", _specs(), ids=["huber mad", "huber rms", "sgd mad"])
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
