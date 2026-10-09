"""``po.spec.audit`` (docs/PLAN.md task 223 (b)): what a stream's columns hold,
read in one pass.

The statistics are held to scipy, statsmodels and numpy in
``tests/test_second_opinion.py`` (``TestAuditIsScipyAndStatsmodels``) and to
their definitions in ``crates/online-core/src/audit/tests.rs``. Here: the
builder's refusals, every row read where a model skips one, a null counted
apart from a NaN, the same frames from 1, 7 and 600 chunks (hard rule 3) and
from a saved state, a restart that keeps the counts, the groups merged, and
the reader's narrowing. ``check()``'s findings from an audit are
``tests/test_check.py``'s.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

N = 1500


def _frame(seed: int = 0, n: int = N) -> pl.DataFrame:
    """Three columns over two interleaved groups on a clock with duplicate
    stamps and gaps: ``a`` null on every 10th row and NaN on every 25th, ``b``
    of five values with a run stuck in the middle, ``c`` a random walk with an
    infinity every 100th row."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.integers(0, 5, size=n).astype(float)
    b[600:640] = 2.0
    c = np.cumsum(rng.normal(size=n))
    c[np.arange(n) % 100 == 50] = np.inf
    steps = np.where(np.arange(n) % 40 == 0, 0.0, 1.0)
    steps[np.arange(n) % 500 == 250] = 30.0
    frame = pl.DataFrame(
        {
            "t": np.cumsum(steps),
            "a": np.where(np.arange(n) % 25 == 7, np.nan, a),
            "b": b,
            "c": c,
            "g": np.where(np.arange(n) % 2 == 0, "p", "q"),
            "s": np.arange(n) // 500,
        }
    )
    return frame.with_columns(
        a=pl.when(pl.int_range(pl.len()) % 10 == 3).then(None).otherwise(pl.col("a"))
    )


def _spec(**kw) -> dict:
    kw.setdefault("clock", "t")
    kw.setdefault("gap_cap", 10.0)
    return po.spec.audit("audit", columns=["a", "b", "c"], pairs=True, **kw)


def _fed(spec: dict, frame: pl.DataFrame, chunks: int = 1) -> po.ModelBank:
    bank = po.ModelBank([spec])
    size = -(-frame.height // chunks)
    for part in frame.iter_slices(size):
        bank.fit_predict(part)
    return bank


def _frames(bank: po.ModelBank) -> list[pl.DataFrame]:
    return [bank.audit(table=t) for t in ("columns", "pairs", "clock")] + [bank.audit(pooled=True)]


class TestTheBuilder:
    def test_it_writes_a_spec_a_toml_file_would(self):
        spec = po.spec.audit("a", columns=["x0", "x1"])
        assert spec["model"] == {"type": "audit"}
        assert spec["features"] == ["x0", "x1"] and spec["targets"] == ["x0"]
        assert po.spec.audit("a", columns=["x0"], pairs=True)["model"]["pairs"] is True

    def test_targets_and_features_are_refused_by_name(self):
        with pytest.raises(TypeError, match="audit\\(\\) takes no targets"):
            po.spec.audit("a", columns=["x0"], targets=["y"])
        with pytest.raises(TypeError, match="takes columns=, not features="):
            po.spec.audit("a", columns=["x0"], features=["x1"])
        with pytest.raises(ValueError, match="columns must be non-empty"):
            po.spec.audit("a", columns=[])

    @pytest.mark.parametrize(
        ("kw", "says"),
        [
            ({"half_life": 10.0}, "half_life/lam do not apply to audit"),
            ({"lam": 0.9}, "half_life/lam do not apply to audit"),
            ({"weight": "w"}, "list it in columns"),
            ({"embargo": 3.0, "clock": "t", "gap_cap": 5.0}, "embargo does not apply to audit"),
            # Review 6, F-10: taken and ignored, the output, `audit()` and
            # `check()` the same at 0, 5 and 1e9.
            ({"min_weight": 5.0}, "min_weight does not apply to audit"),
            ({"min_weight": 0.0}, "min_weight does not apply to audit"),
            ({"distinct_cap": 0}, "distinct_cap must be in 1..=65536"),
            ({"distinct_cap": 70_000}, "distinct_cap must be in 1..=65536"),
        ],
    )
    def test_what_does_not_apply_is_refused(self, kw, says):
        with pytest.raises(ValueError, match=says):
            po.spec.audit("a", columns=["x0"], **kw)


class TestEveryRowIsRead:
    def test_a_null_a_nan_and_an_infinity_are_counted_apart(self):
        frame = _frame()
        bank = _fed(_spec(), frame)
        cols = bank.audit(pooled=True)
        a, c = cols.row(0, named=True), cols.row(2, named=True)
        idx = np.arange(N)
        nulls = int((idx % 10 == 3).sum())
        nans = int(((idx % 10 != 3) & (idx % 25 == 7)).sum())
        assert (a["rows"], a["null"], a["nan"], a["count"]) == (N, nulls, nans, N - nulls - nans)
        assert frame["a"].null_count() == nulls, "Polars counts the nulls alone"
        assert (c["pos_inf"], c["null"], c["nan"]) == (int((idx % 100 == 50).sum()), 0, 0)
        # Every row reached the audit, where a model with `a` as a feature
        # skips each null and NaN.
        model = po.spec.ewridge("m", targets=["b"], features=["a"], half_life=50.0)
        both = po.ModelBank([_spec(), model])
        both.fit_predict(frame)
        summary = both.summary()
        processed = dict(zip(summary["spec"], summary["rows_processed"], strict=True))
        assert processed == {"audit": N, "m": N - nulls - nans}
        # Its summary in its own terms: every row read and none skipped, and
        # nothing learned and no coefficient, so those are null (review 6,
        # C-7: a row with its first column usable counted as learned, and
        # the columns plus one as its coefficients).
        row = summary.filter(spec="audit").row(0, named=True)
        assert (row["rows_fed"], row["rows_processed"], row["rows_skipped"]) == (N, N, 0)
        assert row["rows_learned"] is None and row["n_coef"] is None, row
        m = summary.filter(spec="m").row(0, named=True)
        assert m["rows_learned"] == N - nulls - nans and m["n_coef"] == 2, m

    def test_predict_counts_nothing(self):
        frame = _frame()
        bank = _fed(_spec(), frame.head(1000))
        before = _frames(bank)
        bank.predict(frame.tail(500))
        assert all(a.equals(b) for a, b in zip(_frames(bank), before, strict=True))

    def test_each_row_writes_the_rows_before_it(self):
        out = po.ModelBank([_spec()]).fit_predict(_frame())
        assert out["audit"].struct.fields == ["weight_sum"]
        assert out["audit"].struct.field("weight_sum").to_list() == [float(i) for i in range(N)]


class TestTheChunkingAndTheState:
    @pytest.mark.parametrize("chunks", [7, 600])
    def test_the_frames_are_the_same_from_one_chunk_or_many(self, chunks):
        frame = _frame()
        spec = _spec(group="g")
        one = _frames(_fed(spec, frame))
        many = _frames(_fed(spec, frame, chunks))
        assert all(a.equals(b) for a, b in zip(many, one, strict=True))

    def test_a_saved_audit_goes_on_where_it_stopped(self, tmp_path):
        frame = _frame()
        spec = _spec(group="g")
        whole = _frames(_fed(spec, frame))
        first = _fed(spec, frame.head(700))
        first.save(tmp_path / "audit.state")
        loaded = po.ModelBank.load(tmp_path / "audit.state")
        loaded.fit_predict(frame.tail(N - 700))
        assert all(a.equals(b) for a, b in zip(_frames(loaded), whole, strict=True))

    def test_a_restart_keeps_the_counts(self):
        """A session set to reset restarts a model; an audit starts its runs
        and its clock steps over and keeps everything it counted."""
        frame = _frame()
        plain = _fed(_spec(), frame)
        reset = _fed(_spec(session="s", session_gap="reset"), frame)
        assert reset.summary()["resets"][0] == 2
        p, r = plain.audit(), reset.audit()
        assert p.select("rows", "null", "nan", "count", "mean").equals(
            r.select("rows", "null", "nan", "count", "mean")
        )
        steps = plain.audit(table="clock")["steps"][0], reset.audit(table="clock")["steps"][0]
        assert steps == (N - 1, N - 3)


class TestGroupsMerge:
    def test_pooled_counts_and_moments_are_one_streams(self):
        """The groups merged are the stream read whole, for what does not
        depend on which row came before: the counts exactly, the moments to
        rounding, the exact counts of a column of five values exactly."""
        frame = _frame()
        pooled = _fed(_spec(group="g"), frame).audit(pooled=True)
        whole = _fed(_spec(), frame).audit()
        exact = ["column", "rows", "null", "nan", "pos_inf", "count", "min", "max", "distinct"]
        assert pooled.select(exact).equals(whole.select(exact))
        for col in ("mean", "std", "skew", "kurtosis"):
            np.testing.assert_allclose(pooled[col], whole[col], rtol=1e-10)
        b = pooled.row(1, named=True), whole.row(1, named=True)
        assert (b[0]["top_count"], b[0]["median"], b[0]["mad"]) == (
            b[1]["top_count"],
            b[1]["median"],
            b[1]["mad"],
        )
        assert "group" not in pooled.columns

    def test_the_groups_frame_is_narrowed_as_summary_is(self):
        bank = _fed(_spec(group="g"), _frame())
        assert set(bank.audit()["group"]) == {"p", "q"}
        assert set(bank.audit(group="q")["group"]) == {"q"}
        assert bank.audit(group="never").is_empty()
        assert bank.audit("audit").equals(bank.audit(0))


class TestTheReader:
    def test_it_finds_the_one_audit_and_refuses_otherwise(self):
        model = po.spec.ewridge("m", targets=["b"], features=["a"], half_life=50.0)
        frame = _frame()
        bank = po.ModelBank([model, _spec()])
        bank.fit_predict(frame)
        assert bank.audit().equals(bank.audit("audit"))
        with pytest.raises(ValueError, match='not "audit"'):
            bank.audit("m")
        with pytest.raises(ValueError, match="no audit spec"):
            po.ModelBank([model]).audit()
        two = po.ModelBank([_spec(), po.spec.audit("other", columns=["a"])])
        with pytest.raises(ValueError, match="2 audit specs"):
            two.audit()
        with pytest.raises(ValueError, match="table must be"):
            bank.audit(table="rows")
        no_pairs = po.ModelBank([po.spec.audit("plain", columns=["a", "b"])])
        no_pairs.fit_predict(frame)
        with pytest.raises(ValueError, match="pairs=True"):
            no_pairs.audit(table="pairs")
        assert no_pairs.audit(table="clock").is_empty(), "no clock column, no clock row"

    def test_a_bank_that_has_seen_nothing_has_the_schemas_and_no_rows(self):
        bank = po.ModelBank([_spec()])
        fed = _fed(_spec(), _frame())
        for table in ("columns", "pairs", "clock"):
            empty, full = bank.audit(table=table), fed.audit(table=table)
            assert empty.is_empty() and empty.schema == full.schema
