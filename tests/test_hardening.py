"""Post-refactor hardening (docs/PERFORMANCE.md P1-P8 changed the whole hot path).

The rewrite moved every output through flat slot-major buffers written by
parallel per-instance tasks. Stride bugs in that layout are invisible at the
row counts most tests use (40-400 rows, often one group, one instance, few
outputs): a transposed index can land in a buffer region that is never read,
or alias a slot that happens to hold the same value. These tests exist to make
that class of bug loud:

* a kitchen-sink stream at 30k rows with every output enabled at once, across
  grids, groups, sessions, weights and nulls -- compared across chunkings,
  across a save/load boundary, and across thread counts;
* the one code path the refactor added that nothing exercised: a multi-instance
  grid with `drift_action="reset"`, where instances are coupled within a row;
* parameter ranges at their edges rather than their comfortable middles.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import child
import polars_online as po
from conftest import run_online

TIER = "mixed"

REPO = Path(__file__).resolve().parent.parent


def kitchen_sink_frame(n=30_000, groups=5, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    beta = np.array([[1.5, -0.7, 0.3, 0.0], [0.2, 0.9, -1.1, 0.5]])
    y = x @ beta.T + 0.3 * rng.standard_normal((n, 2))
    # Irregular clock per group with occasional large gaps.
    dt = np.where(rng.random(n) < 0.01, 50.0, rng.uniform(0.5, 2.0, n))
    g = np.arange(n) % groups
    t = np.zeros(n)
    for gi in range(groups):
        mask = g == gi
        t[mask] = np.cumsum(dt[mask])
    w = rng.uniform(0.2, 3.0, n)
    w[rng.random(n) < 0.02] = 0.0  # legal pure-decay rows
    df = pl.DataFrame(
        {
            "t": t,
            "x0": x[:, 0],
            "x1": x[:, 1],
            "x2": x[:, 2],
            "x3": x[:, 3],
            "y0": y[:, 0],
            "y1": y[:, 1],
            "g": g,
            "session": (t // 500).astype(np.int64),
            "w": w,
        }
    )
    # Nulls in a feature and a target, away from row 0 so warmup still happens.
    return df.with_columns(
        pl.when(pl.int_range(0, n) % 41 == 13).then(None).otherwise(pl.col("x2")).alias("x2"),
        pl.when(pl.int_range(0, n) % 37 == 11).then(None).otherwise(pl.col("y0")).alias("y0"),
    )


def kitchen_sink_spec(**overrides):
    d = dict(
        targets=["y0", "y1"],
        features=["x0", "x1", "x2", "x3"],
        feature_sets={"pair": ["x0", "x1"], "all": ["x0", "x1", "x2", "x3"]},
        ridge=[1e-6, 0.3],
        half_life=[150.0, 900.0],
        clock="t",
        gap_cap=25.0,
        session="session",
        session_gap=10.0,
        weight="w",
        group="g",
        min_weight=[10.0, 20.0],
        max_rows_between_solves=16,
        max_rows_between_coefs=997,
        emit_sigma=True,
        emit_zscore=True,
        emit_drift=True,
        drift_threshold=20.0,
        emit_metrics=True,
        emit_autocorr=True,
        resid_quantiles=[0.05, 0.95],
        conformal=0.9,
        emit_selected=True,
        emit_averaged=True,
        emit_calibration=True,
        calibration_half_life=600.0,
        emit_breaks=True,
        emit_robust_se=True,
        robust_se_lags=2,
        emit_specification=True,
    )
    d.update(overrides)
    return po.spec.ewridge("m", **d)


def run_chunked(df, spec, chunk):
    bank = po.ModelBank([spec])
    if chunk is None:
        return bank.fit_predict(df).select("m").unnest("m")
    parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
    return pl.concat(parts).select("m").unnest("m")


def drop_coef(out):
    # coef is a reporting cadence: by default emitted on every chunk's LAST
    # row, so it legitimately differs across chunkings. Under a cadence, as
    # the kitchen sink's, it does not (docs/PLAN.md task 178), which
    # `test_chunk_invariance_with_everything_on` checks apart.
    return out.drop([c for c in out.columns if c.startswith(("coef", "support_coef"))])


@pytest.fixture(scope="module")
def frame():
    return kitchen_sink_frame()


class TestKitchenSinkAtScale:
    """Every output at once, 30k rows, 16 slots x 2 instances, 5 groups."""

    def test_the_spec_is_as_wide_as_it_claims(self, frame):
        fields = po.spec.output_fields(kitchen_sink_spec())
        # 2 instances x (2 targets x 4 combos x (pred,resid,sigma,zscore,
        # drift,ic,r2,hit_rate,autocorr,2 quantiles) + weight_sum + coef) + selection.
        assert len(fields) > 150, f"only {len(fields)} fields -- the stress is gone"
        out = run_chunked(frame, kitchen_sink_spec(), None)
        assert out.columns == fields

    def test_chunk_invariance_with_everything_on(self, frame):
        whole = run_chunked(frame, kitchen_sink_spec(), None)
        chunked = run_chunked(frame, kitchen_sink_spec(), 1013)
        one, odd = drop_coef(whole), drop_coef(chunked)
        assert one.equals(odd, null_equal=True), "1 chunk vs 1013-row chunks"
        # Under a cadence the rows carrying `coef` do not move either (task
        # 178): every 997th row of each group, whatever the chunks.
        coef = [c for c in whole.columns if c.startswith(("coef", "support_coef"))]
        assert coef and whole.select(coef).equals(chunked.select(coef), null_equal=True)
        assert 0 < whole[coef[0]].drop_nulls().len() < whole.height // 900
        tiny = drop_coef(run_chunked(frame.head(2000), kitchen_sink_spec(), 1))
        ref = drop_coef(run_chunked(frame.head(2000), kitchen_sink_spec(), None))
        assert ref.equals(tiny, null_equal=True), "row-at-a-time must match too"

    def test_save_load_transparency_mid_stream(self, frame, tmp_path):
        spec = kitchen_sink_spec()
        whole = drop_coef(run_chunked(frame, spec, None))

        a = po.ModelBank([spec])
        first = a.fit_predict(frame.slice(0, 17_777))
        p = tmp_path / "ks.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = b.fit_predict(frame.slice(17_777, frame.height))
        joined = drop_coef(pl.concat([first, rest]).select("m").unnest("m"))
        assert whole.equals(joined, null_equal=True)

    @pytest.mark.extended(reason="threads: one thread against eight on a large stream (3.4 s)")
    def test_identical_across_thread_counts(self, frame, tmp_path):
        """The parallel fan-out (spec x group x instance) must be a pure
        scheduling choice. 400 rows -- the old determinism test's size -- could
        not distinguish one task per core from one task total."""
        script = tmp_path / "run.py"
        data = tmp_path / "ks.parquet"
        frame.write_parquet(data)
        script.write_text(
            "import sys, hashlib, polars as pl, polars_online as po\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from test_hardening import kitchen_sink_spec, run_chunked, drop_coef\n"
            "df = pl.read_parquet(sys.argv[1])\n"
            "out = drop_coef(run_chunked(df, kitchen_sink_spec(), 2048))\n"
            "h = hashlib.sha256()\n"
            "for s in out.get_columns():\n"
            "    h.update(str(s.to_list()).encode())\n"
            "print(h.hexdigest())\n"
        )
        digests = set()
        for threads in ("1", "8"):
            res = subprocess.run(
                [sys.executable, str(script), str(data), str(REPO / "tests")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                # Inherit the environment and override only the thread count.
                # A hardcoded POSIX PATH left the child with no resolvable
                # interpreter on Windows.
                env=child.env(POLARS_ONLINE_MAX_THREADS=threads),
                cwd=str(REPO),
                check=True,
            )
            digests.add(res.stdout.strip())
        assert len(digests) == 1, "thread count changed the numbers"


class TestCoupledDriftPath:
    """A multi-instance grid with `drift_action="reset"` couples the instances
    within a row (a break in any resets all), which forces the row-major path
    -- the one branch of the refactor nothing else executes."""

    def _df(self, n=6000, flip_at=3000, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        sign = np.where(np.arange(n) < flip_at, 1.0, -1.0)
        return pl.DataFrame({"x0": x, "y0": sign * 2 * x + 0.2 * rng.standard_normal(n)})

    def _spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            half_life=[1e5, 2e5],  # two instances -> the coupled branch
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_drift=True,
            drift_action="reset",
        )
        d.update(kw)
        return po.spec.ewridge("m", **d)

    def _run(self, df, spec, chunk=None):
        bank = po.ModelBank([spec])
        if chunk is None:
            return bank.fit_predict(df).select("m").unnest("m")
        return (
            pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
            .select("m")
            .unnest("m")
        )

    def test_a_break_in_either_instance_resets_both(self):
        out = self._run(self._df(), self._spec())
        flags = np.zeros(out.height, dtype=bool)
        for c in out.columns:
            if c.startswith("drift_"):
                flags |= out[c].fill_null(False).to_numpy()
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1, "the sign flip was not detected"
        first = hits[0]
        for c in out.columns:
            if c.startswith("weight_sum"):
                weight_sum = out[c].to_numpy().astype(float)
                assert weight_sum[first + 1] < weight_sum[first], (
                    f"{c}: instance not reset at the break "
                    f"({weight_sum[first]} -> {weight_sum[first + 1]})"
                )

    def test_the_coupled_path_is_chunk_invariant(self):
        df = self._df()
        one = drop_coef(self._run(df, self._spec()))
        many = drop_coef(self._run(df, self._spec(), chunk=577))
        assert one.equals(many, null_equal=True)

    def test_coupled_equals_uncoupled_when_nothing_fires(self):
        """On a stream with no break, the row-major coupled path and the
        parallel path must produce identical numbers -- they are the same
        arithmetic in a different order."""
        df = self._df(flip_at=10**9)  # never flips
        coupled = drop_coef(self._run(df, self._spec()))
        parallel = drop_coef(self._run(df, self._spec(drift_action="flag")))
        assert coupled.equals(parallel, null_equal=True)


class TestParameterRanges:
    """Edges of ranges the suite only ever exercised in the middle."""

    def _df(self, n=5000, seed=1):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        return pl.DataFrame(
            {
                "t": np.arange(float(n)),
                "x0": x,
                "y0": 2.0 * x + 0.1 * rng.standard_normal(n),
            }
        )

    def _spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            clock="t",
            gap_cap=10.0,
            half_life=200.0,
            min_weight=3.0,
            max_rows_between_solves=1,
        )
        d.update(kw)
        return po.spec.ewridge("m", **d)

    def test_halflife_extremes_give_the_exact_limits(self):
        n = 5000
        df = self._df(n)
        # Infinite half-life: no forgetting, weight_sum counts every accepted row.
        out = po.ModelBank([self._spec(half_life=float("inf"))]).fit_predict(df)
        weight_sum = out["m"].struct.field("weight_sum").to_list()
        assert weight_sum[-1] == pytest.approx(n - 1, abs=1e-9)
        # Tiny half-life: everything before this row has decayed to nothing.
        out = po.ModelBank([self._spec(half_life=1e-3)]).fit_predict(df)
        weight_sum = out["m"].struct.field("weight_sum").to_list()
        assert weight_sum[-1] == pytest.approx(1.0, abs=1e-6), "only the previous row survives"
        preds = out["m"].struct.field("pred_y0").to_list()
        assert all(v is None or np.isfinite(v) for v in preds), "no NaN under extreme decay"

    def test_weight_scale_invariance(self):
        """Mean-form accumulators divide weight by accumulated weight, so
        multiplying every weight by c must change nothing but `weight_sum` (which
        scales by c) provided `min_weight` scales with it. Exercised at 1e-6
        and 1e6, where a sum-form implementation would lose precision or
        overflow -- this is the test that the mean form is real."""
        base = self._df().with_columns(w=pl.lit(1.0))
        for c in (1e-6, 1e6):
            scaled = base.with_columns(w=pl.lit(float(c)))
            a = po.ModelBank([self._spec(weight="w", min_weight=3.0)]).fit_predict(base)
            b = po.ModelBank([self._spec(weight="w", min_weight=3.0 * c)]).fit_predict(scaled)
            pa_ = a["m"].struct.field("pred_y0").to_numpy().astype(float)
            pb = b["m"].struct.field("pred_y0").to_numpy().astype(float)
            mask = np.isfinite(pa_) | np.isfinite(pb)
            np.testing.assert_allclose(pa_[mask], pb[mask], rtol=1e-9, err_msg=f"c={c}")
            na = a["m"].struct.field("weight_sum").to_numpy().astype(float)
            nb = b["m"].struct.field("weight_sum").to_numpy().astype(float)
            np.testing.assert_allclose(nb, na * c, rtol=1e-9, err_msg=f"weight_sum c={c}")

    def test_sixty_four_features(self):
        """Wide: k=64 exercises the solve and the extraction fan-out well past
        the k<=5 of most tests."""
        n, k = 4000, 64
        rng = np.random.default_rng(3)
        x = rng.standard_normal((n, k))
        beta = rng.standard_normal(k)
        cols = {f"x{i}": x[:, i] for i in range(k)}
        cols["y0"] = x @ beta + 0.01 * rng.standard_normal(n)
        df = pl.DataFrame(cols)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=[f"x{i}" for i in range(k)],
            half_life=1e6,
            min_weight=float(k + 5),
            max_rows_between_solves=64,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        coef = out["m"].struct.field("coef").to_list()[-1]
        got = np.array(coef[1:])  # drop intercept
        np.testing.assert_allclose(got, beta, atol=0.05)

    def test_twelve_targets_with_per_target_warmup(self):
        n, m = 3000, 12
        rng = np.random.default_rng(4)
        x = rng.standard_normal(n)
        cols = {"x0": x}
        for j in range(m):
            cols[f"y{j}"] = (j + 1) * x
        df = pl.DataFrame(cols)
        thresholds = [5.0 * (j + 1) for j in range(m)]
        spec = po.spec.ewridge(
            "m",
            targets=[f"y{j}" for j in range(m)],
            features=["x0"],
            half_life=float("inf"),
            min_weight=thresholds,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for j, thr in enumerate(thresholds):
            preds = out["m"].struct.field(f"pred_y{j}").to_list()
            first = next(i for i, v in enumerate(preds) if v is not None)
            # weight_sum before row i is i (unit weights, no decay), so the first
            # emitted row is exactly ceil(thr).
            assert first == int(np.ceil(thr)), f"target {j}: first pred at {first}"

    def test_extreme_quantile_levels(self):
        df = self._df(8000)
        spec = self._spec(resid_quantiles=[0.001, 0.999])
        out = po.ModelBank([spec]).fit_predict(df)
        lo = out["m"].struct.field("abs_resid_q0.001_y0").to_list()[-1]
        hi = out["m"].struct.field("abs_resid_q0.999_y0").to_list()[-1]
        assert 0.0 <= lo <= hi, f"{lo} vs {hi}"
        assert hi < 2.0, "q0.999 of |resid| should be near the noise scale"

    @pytest.mark.parametrize(
        ("kw", "want"),
        [
            ({}, None),
            ({"coef_every": 0}, 3000),
            # The 997th, 1,994th and 2,991st accepted rows.
            ({"max_rows_between_coefs": 997}, 3),
            # The clock steps by 1 from 0: t = 997, 1,994 and 2,991.
            ({"coef_every": 997.0}, 3),
        ],
        ids=["unset", "every-row", "rows", "clock"],
    )
    def test_coef_cadence_across_chunks(self, kw, want):
        """Unset, ``coef`` is on each chunk's last row; under a cadence the
        same rows whatever the chunking (docs/PLAN.md task 178)."""
        n = 3000
        df = self._df(n)
        spec = self._spec(**kw)
        for chunk in (None, 700):
            bank = po.ModelBank([spec])
            if chunk is None:
                out = bank.fit_predict(df)
                boundaries = 1
            else:
                out = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, n, chunk)])
                boundaries = -(-n // chunk)
            got = sum(v is not None for v in out["m"].struct.field("coef").to_list())
            assert got == (boundaries if want is None else want), (kw, chunk, got)


class TestRunnerErrorPaths:
    """P6 put the parquet reader on its own thread behind a sync_channel.
    Every error path must end in a clean exception, never a deadlock: a
    blocked `send` with no receiver, or a consumer waiting on a reader that
    already died, would hang the whole process."""

    def _spec(self):
        return po.spec.ewridge(
            "m", targets=["y0"], features=["x0"], half_life=100.0, min_weight=3.0
        )

    def test_corrupt_input_errors_cleanly(self, tmp_path, online_cli):
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"PAR1 this is not really a parquet file")
        res = run_online(
            online_cli,
            tmp_path,
            [self._spec()],
            input=bad,
            output=tmp_path / "out.parquet",
            check=False,
        )
        assert res.returncode != 0
        assert re.search("(?i)parquet|deserialize|read", res.stderr), res.stderr

    def test_a_bank_error_mid_stream_errors_cleanly(self, tmp_path, online_cli):
        """The consumer errors while the reader is a chunk ahead: the
        rejection (a negative weight, row named) must propagate, and the
        reader must be shut down rather than left blocked on `send`."""
        n = 5000
        rng = np.random.default_rng(6)
        w = np.ones(n)
        w[3777] = -2.0
        df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n), "w": w})
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=100.0,
            min_weight=3.0,
            weight="w",
        )
        res = run_online(
            online_cli,
            tmp_path,
            [spec],
            input=src,
            output=tmp_path / "out.parquet",
            chunk_size=500,
            check=False,
        )
        assert res.returncode != 0
        assert "negative" in res.stderr, res.stderr

    def test_a_failed_run_leaves_the_previous_output_intact(self, tmp_path, online_cli):
        """The output is written to a temporary and renamed into place, so a
        run that dies on chunk eight does not replace yesterday's output with
        seven chunks and no footer (IMPROVEMENTS C6)."""
        n = 5000
        rng = np.random.default_rng(7)
        good = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
        src, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
        good.write_parquet(src)
        run_online(online_cli, tmp_path, [self._spec()], input=src, output=out, chunk_size=500)
        before = out.read_bytes()

        w = np.ones(n)
        w[3777] = -2.0
        bad = good.with_columns(pl.Series("w", w))
        bad.write_parquet(src)
        spec = po.spec.ewridge(
            "m", targets=["y0"], features=["x0"], half_life=100.0, min_weight=3.0, weight="w"
        )
        res = run_online(
            online_cli, tmp_path, [spec], input=src, output=out, chunk_size=500, check=False
        )
        assert res.returncode != 0
        assert "negative" in res.stderr, res.stderr

        assert out.read_bytes() == before, "the failed run overwrote the good output"
        assert pl.read_parquet(out).height == n
        assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name], "temporary left behind"


class TestExpressionSpecCache:
    """P5 added a thread-local parsed-spec cache keyed by the kwargs JSON. Two
    different specs evaluated on the same thread must not bleed into each
    other, and the cache must not survive incorrectly across .over groups."""

    def test_two_specs_in_one_query_stay_distinct(self):
        """Two specs differing only in half-life must not share a state: the
        faster one accumulates less weight at every row. Equality here would
        mean one spec's model served both."""
        n = 4000
        rng = np.random.default_rng(5)
        x = rng.standard_normal(n)
        df = pl.DataFrame(
            {"x0": x, "y": 2 * x + 0.1 * rng.standard_normal(n), "g": np.arange(n) % 8}
        )
        fast = po.spec.ewridge(
            "fast", targets=["y"], features=["x0"], half_life=50.0, min_weight=5.0, group="g"
        )
        slow = po.spec.ewridge(
            "slow", targets=["y"], features=["x0"], half_life=5000.0, min_weight=5.0, group="g"
        )
        out = po.ModelBank([fast, slow]).fit_predict(df)
        pf = out["fast"].struct.field("weight_sum").to_numpy().astype(float)
        ps = out["slow"].struct.field("weight_sum").to_numpy().astype(float)
        mask = np.isfinite(pf) & np.isfinite(ps) & (np.arange(n) > 800)
        assert (pf[mask] < ps[mask]).all(), "the fast half_life must accumulate less weight"


def test_an_infinity_in_a_later_chunk_of_a_column_is_skipped():
    """Every row passes ``usable`` before a model steps, however its column
    is laid out in memory: here a column in two chunks, with an infinity in
    the second. The review asked whether the extraction could hand a value
    past that check (2026-09-12, V11); it cannot, and this pins it."""
    x = np.arange(40, dtype=float) % 7
    y = 2 * x + 1.0
    x_tail = x[20:].copy()
    x_tail[5] = np.inf
    head = pl.DataFrame({"x0": x[:20], "y": y[:20]})
    tail = pl.DataFrame({"x0": x_tail, "y": y[20:]})
    df = pl.concat([head, tail], rechunk=False)
    assert df["x0"].n_chunks() == 2
    bank = po.ModelBank([po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)])
    out = bank.fit_predict(df).unnest("m")
    assert bank.summary()["rows_skipped"].to_list() == [1]
    last = out["pred_y"][-1]
    assert last is not None and np.isfinite(last), out.tail(3)
