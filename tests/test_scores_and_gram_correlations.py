"""Five outputs the user decided (2026-11-07, task 359): a null clock is in
no window of ``window_metrics``; ``from_sums`` is null, not NaN, where no
row weighs anything; ``merge_sums`` keeps a lone part's sums to the bit;
every ``po.corr`` function reads a Gram without its intercept and refuses
a constant feature by name; ``block_means`` counts only the pairs it
averages."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import corr

TIER = "essential"

N = 400


def _frame(seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=N), rng.normal(size=N)
    return pl.DataFrame(
        {
            "t": np.arange(N, dtype=float),
            "x0": x0,
            "x1": x1,
            "y": 2.0 * x0 + x1 + rng.normal(size=N),
        }
    )


SPEC = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=100.0)


@pytest.fixture(scope="module")
def scored() -> pl.DataFrame:
    return po.ModelBank([SPEC]).fit_predict(_frame())


# --- window_metrics -----------------------------------------------------------


@pytest.mark.parametrize("every", [100.0, 30.0])
def test_a_null_clock_is_in_no_window(scored: pl.DataFrame, every: float) -> None:
    holed = scored.with_columns(
        t=pl.when(pl.int_range(pl.len()) % 10 == 0).then(None).otherwise(pl.col("t"))
    )
    got = po.eval.window_metrics(holed, "m", clock="t", every=every, min_samples=5)
    assert got["window_start"].null_count() == 0
    # The windows are those of the rows that have a clock, scored alone.
    kept = holed.filter(pl.col("t").is_not_null())
    want = po.eval.window_metrics(kept, "m", clock="t", every=every, min_samples=5)
    assert got.equals(want)
    predicted = kept.filter(pl.col("m").struct.field("pred_y").is_not_null()).height
    assert (
        got["n"].sum()
        == predicted
        < holed.filter(pl.col("m").struct.field("pred_y").is_not_null()).height
    )


def test_a_temporal_null_clock_is_in_no_window(scored: pl.DataFrame) -> None:
    stamped = scored.with_columns(
        t=pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col("t").cast(pl.Int64).cast(pl.Datetime("ms")))
    )
    got = po.eval.window_metrics(stamped, "m", clock="t", every="100ms", min_samples=5)
    assert got["window_start"].null_count() == 0
    assert got.height == 4


# --- from_sums ------------------------------------------------------------------


def test_from_sums_is_null_where_no_row_weighs_anything(scored: pl.DataFrame) -> None:
    s = po.eval.sums(scored.with_columns(w=pl.lit(0.0)), "m", weight="w")
    assert s["w"][0] == 0.0
    got = po.eval.from_sums(s).row(0, named=True)
    for score in ("r2", "ic", "hit_rate", "mse", "rmse"):
        assert got[score] is None, score
    # One weighted row brings the scores back.
    some = po.eval.from_sums(po.eval.sums(scored.with_columns(w=pl.lit(1.0)), "m", weight="w")).row(
        0, named=True
    )
    assert all(some[c] is not None and np.isfinite(some[c]) for c in ("r2", "ic", "mse"))


# --- merge_sums -----------------------------------------------------------------

LONE = {
    "slot": ["pred_y"],
    "target": ["y"],
    "n": [400],
    "w": [637.3247256341328],
    "mean_y": [-0.8448275328993269],
    "mean_pred": [0.25],
    "m2_y": [1e-30],
    "m2_pred": [1.0],
    "cov": [0.5],
    "sse": [1.0],
    "hits": [1.0],
    "signed": [2.0],
}


def test_merge_sums_of_one_frame_returns_it_to_the_bit() -> None:
    s = pl.DataFrame(LONE)
    # The pair this needs: recomputed, `w * mean / w` is not `mean`.
    assert LONE["w"][0] * LONE["mean_y"][0] / LONE["w"][0] != LONE["mean_y"][0]
    assert po.eval.merge_sums(s).equals(s)


def test_merge_sums_keeps_a_key_in_one_part_alone_to_the_bit() -> None:
    lone = pl.DataFrame(LONE)
    other = lone.with_columns(target=pl.lit("z"), mean_y=pl.lit(0.5))
    got = po.eval.merge_sums(other, lone)
    assert got.filter(pl.col("target") == "y").equals(lone)
    assert got.filter(pl.col("target") == "z").equals(other)


def test_merge_sums_of_one_frame_sorts_it(scored: pl.DataFrame) -> None:
    grouped = scored.with_columns(g=pl.int_range(pl.len()) % 3)
    s = po.eval.sums(grouped, "m", group="g").sort("g", descending=True)
    assert po.eval.merge_sums(s).equals(s.sort(["slot", "target", "g"]))


# --- po.corr on a Gram -------------------------------------------------------------


@pytest.fixture(scope="module")
def gram() -> dict:
    spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=float("inf"))
    bank = po.ModelBank([spec])
    bank.fit_predict(_frame())
    g = bank.gram("m")[0]
    assert g["columns"] == ["intercept", "x0", "x1"]
    return g


def test_every_function_reads_a_gram_without_its_intercept(gram: dict) -> None:
    bare = po.gram.subset(gram, ["x0", "x1"])
    r = po.gram.correlation(bare)
    x = _frame().select("x0", "x1").to_numpy()
    xs = (x - x.mean(0)) / x.std(0)
    for name, call in {
        "shrink": lambda a: corr.shrink(a, x=xs),
        "shrink identity": lambda a: corr.shrink(a, target="identity", x=xs),
        "equicorr": corr.equicorr,
        "absorption": lambda a: corr.absorption(a, 1),
        "spectral": lambda a: corr.spectral(a, 2),
        "block_means": lambda a: corr.block_means(a, ["a", "b"]),
        "loss": lambda a: corr.loss(a, a),
        "nearest": corr.nearest,
    }.items():
        got, want = call(gram), call(bare)
        assert repr(got) == repr(want), name
    shrunk, alpha = corr.shrink(gram, x=xs)
    assert np.all(np.isfinite(shrunk)) and shrunk.shape == (2, 2)
    assert alpha == corr.shrink(r, x=xs)[1] > 0.0


def test_epps_invert_drops_the_intercept_too(gram: dict) -> None:
    assert corr.epps_invert(gram, L=1).shape == (2, 2)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: corr.shrink(g, alpha=0.5),
        corr.equicorr,
        lambda g: corr.spectral(g, 1),
        lambda g: corr.absorption(g, 1),
        lambda g: corr.block_means(g, ["a", "b"]),
        corr.nearest,
    ],
)
def test_a_constant_feature_is_refused_by_name(call: object) -> None:
    flat = _frame().with_columns(x1=pl.lit(3.0))
    bank = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=100.0)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bank.fit_predict(flat)
    g = bank.gram("m")[0]
    with pytest.raises(ValueError, match=r"Gram but the intercept must vary.*'x1' has no variance"):
        call(g)  # type: ignore[operator]


# --- block_means ------------------------------------------------------------------


def test_block_means_counts_only_the_pairs_it_averages() -> None:
    r = np.array(
        [
            [1.0, np.nan, 0.2, 0.4],
            [np.nan, 1.0, 0.3, np.nan],
            [0.2, 0.3, 1.0, 0.5],
            [0.4, np.nan, 0.5, 1.0],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "Mean of empty slice" either
        means, counts = corr.block_means(r, ["a", "a", "b", "b"])
    assert counts.tolist() == [[0, 3], [3, 2]]
    assert np.isnan(means[0, 0])
    assert means[0, 1] == pytest.approx((0.2 + 0.4 + 0.3) / 3)
    assert means[1, 1] == 0.5
