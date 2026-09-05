"""E49: `po.eval.sums` / `merge_sums` / `from_sums`.

Two claims, and both are held against the thing that already computes the
answer rather than against a second copy of the formula:

1. `from_sums(sums(df))` is `metrics(df)` -- the same numbers, from ten
   doubles per key instead of the rows.
2. `merge_sums` over any split of the rows is `sums` over all of them, and
   stays that way as the split gets finer.

The third thing worth a test is the reason the sums are centred rather than
raw: a target sitting on a large offset destroys the "sum of squares minus
the square of the sum" form entirely, and does not touch this one.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def fitted(n=3000, seed=0, offset=0.0, groups=1, weight=False):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "y": offset + 2.0 * x0 - x1 + 0.5 * rng.standard_normal(n),
            "venue": [f"v{i % groups}" for i in range(n)],
        }
    )
    if weight:
        df = df.with_columns(w=pl.Series(0.5 + rng.random(n)))
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], halflife=300.0, min_periods=5.0
    )
    return po.ModelBank([spec]).fit_predict(df)


def close(a: pl.DataFrame, b: pl.DataFrame, *, rel=1e-9):
    assert a.columns == b.columns, (a.columns, b.columns)
    assert a.height == b.height
    for c in a.columns:
        if a.schema[c] == pl.String:
            assert a[c].to_list() == b[c].to_list(), c
        else:
            assert a[c].to_numpy() == pytest.approx(b[c].to_numpy(), rel=rel), c


class TestItIsTheSameAnswer:
    @pytest.mark.parametrize("groups", [1, 3])
    def test_from_sums_equals_metrics(self, groups):
        out = fitted(groups=groups)
        by = ["venue"] if groups > 1 else []
        want = po.eval.metrics(out, "m", by=by)
        got = po.eval.from_sums(po.eval.sums(out, "m", by=by))
        close(want, got.drop("rmse"))

    def test_rmse_is_the_root_of_mse(self):
        got = po.eval.from_sums(po.eval.sums(fitted(), "m"))
        assert got["rmse"].to_numpy() == pytest.approx(np.sqrt(got["mse"].to_numpy()))

    def test_min_obs_drops_a_thin_key_as_metrics_does(self):
        out = fitted(n=200, groups=2)
        # One venue gets 100 rows; ask for more than that.
        assert po.eval.from_sums(po.eval.sums(out, "m", by=["venue"]), min_obs=1000).height == 0
        assert po.eval.metrics(out, "m", by=["venue"], min_obs=1000).height == 0

    def test_weights_are_respected(self):
        """A weighted mean squared error is not the unweighted one, and the
        weighted form is what a weighted stream should be scored by."""
        out = fitted(weight=True)
        plain = po.eval.from_sums(po.eval.sums(out, "m"))
        weighted = po.eval.from_sums(po.eval.sums(out, "m", weight="w"))
        assert weighted["mse"][0] != plain["mse"][0]
        # And it is the weighted average of the squared residuals.
        long = po.eval.unpack(out, "m").drop_nulls(["pred", "y"])
        w = long["w"].to_numpy()
        r = (long["y"] - long["pred"]).to_numpy()
        assert weighted["mse"][0] == pytest.approx((w * r * r).sum() / w.sum(), rel=1e-9)

    def test_a_constant_target_has_no_r2_or_ic(self):
        n = 200
        df = pl.DataFrame({"x0": np.arange(float(n)), "y": np.full(n, 3.0)})
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        got = po.eval.from_sums(po.eval.sums(out, "m"), min_obs=1)
        assert got["r2"][0] is None, "no variance to explain"
        assert got["ic"][0] is None


class TestMerging:
    @pytest.mark.parametrize("parts", [2, 5, 97])
    def test_merging_a_split_gives_the_whole(self, parts):
        out = fitted(n=3000, seed=1, groups=3)
        whole = po.eval.sums(out, "m", by=["venue"])
        pieces = [
            po.eval.sums(out[chunk.tolist()], "m", by=["venue"])
            for chunk in np.array_split(np.arange(out.height), parts)
        ]
        close(whole, po.eval.merge_sums(*pieces), rel=1e-9)

    def test_the_metrics_survive_the_merge(self):
        out = fitted(n=2400, seed=2)
        whole = po.eval.from_sums(po.eval.sums(out, "m"))
        running = None
        for chunk in out.iter_slices(97):
            s = po.eval.sums(chunk, "m")
            running = s if running is None else po.eval.merge_sums(running, s)
        close(whole, po.eval.from_sums(running), rel=1e-9)

    def test_merging_one_is_the_identity(self):
        s = po.eval.sums(fitted(n=400), "m")
        close(s, po.eval.merge_sums(s), rel=0)

    def test_keys_present_in_only_one_part_are_carried_through(self):
        out = fitted(n=1200, seed=3, groups=3)
        a = po.eval.sums(out.filter(pl.col("venue") != "v2"), "m", by=["venue"])
        b = po.eval.sums(out.filter(pl.col("venue") == "v2"), "m", by=["venue"])
        merged = po.eval.merge_sums(a, b)
        assert sorted(merged["venue"].to_list()) == ["v0", "v1", "v2"]
        close(merged, po.eval.sums(out, "m", by=["venue"]), rel=1e-9)

    def test_mismatched_keys_are_refused(self):
        out = fitted(n=400)
        with pytest.raises(ValueError, match="same keys in every part"):
            po.eval.merge_sums(po.eval.sums(out, "m"), po.eval.sums(out, "m", by=["venue"]))


class TestWhyTheSumsAreCentred:
    def test_a_large_offset_does_not_destroy_the_variance(self):
        """A target around 1e8 with unit spread has `var / E[y**2]` of about
        1e-16: the raw form (`sum(y**2) - sum(y)**2 / n`) has nothing left
        after the subtraction. The centred form is unaffected, which is the
        whole reason for the parallel-axis term in `merge_sums`."""
        out = fitted(n=4000, seed=4, offset=1e8)
        got = po.eval.from_sums(po.eval.sums(out, "m"))
        assert got["r2"][0] == pytest.approx(po.eval.metrics(out, "m")["r2"][0], rel=1e-9)
        assert 0.9 < got["r2"][0] < 1.0, got["r2"][0]

        # What the raw form would have given, from the same rows.
        long = po.eval.unpack(out, "m").drop_nulls(["pred", "y"])
        y = long["y"].to_numpy()
        raw_m2 = float((y * y).sum() - y.sum() ** 2 / len(y))
        centred = float(po.eval.sums(out, "m")["m2_y"][0])
        assert abs(raw_m2 / centred - 1.0) > 0.01, (
            "the fixture no longer shows the cancellation; raise the offset"
        )

    def test_and_it_still_merges_at_that_offset(self):
        out = fitted(n=4000, seed=5, offset=1e8)
        whole = po.eval.sums(out, "m")
        parts = [
            po.eval.sums(out[c.tolist()], "m") for c in np.array_split(np.arange(out.height), 40)
        ]
        close(whole, po.eval.merge_sums(*parts), rel=1e-9)


class TestTheShape:
    def test_the_columns_are_the_documented_ones(self):
        s = po.eval.sums(fitted(n=500, groups=2), "m", by=["venue"])
        assert s.columns == ["slot", "target", "venue", *po.eval.SUM_FIELDS]
        assert len(po.eval.SUM_FIELDS) == 10, "ten doubles per key is the claim"

    def test_one_row_per_slot(self):
        n = 800
        rng = np.random.default_rng(6)
        df = pl.DataFrame({"x0": rng.standard_normal(n)})
        df = df.with_columns(y=2 * pl.col("x0"), z=-pl.col("x0"))
        spec = po.spec.ewridge(
            "m", targets=["y", "z"], features=["x0"], halflife=[50.0, 500.0], min_periods=3.0
        )
        out = po.ModelBank([spec]).fit_predict(df)
        s = po.eval.sums(out, "m", spec=spec)
        assert s.height == 4, "two targets x two halflives"
        close(po.eval.from_sums(s).drop("rmse"), po.eval.metrics(out, "m"))

    def test_it_raises_where_unpack_does(self):
        out = fitted(n=300)
        with pytest.raises(KeyError):
            po.eval.sums(out, "nope")
        with pytest.raises(TypeError, match="not a model-output struct"):
            po.eval.sums(out, "x0")
