"""``ModelBank.check`` (docs/PLAN.md task 223 (a)): findings about the data a
bank was fed, read from what it keeps.

Each check fires on a stream with its problem planted and stays silent on the
clean shapes (``tests/check_streams.py``); a bank loaded from its file gives
the findings the live bank gave; and the findings do not depend on how the
stream was chunked. ``TestTenSeeds`` is the sweep the docstring's rates come
from.
"""

from __future__ import annotations

import builtins

import numpy as np
import polars as pl
import pytest

import check_streams as cs
import polars_online as po
from polars_online import _check
from polars_online import _polars_online as _native

TIER = "essential"


def _codes(f: pl.DataFrame, code: str) -> set[str]:
    return set(f.filter(pl.col("code") == code)["spec"].to_list())


def test_the_frame_has_its_schema_and_a_clean_stream_raises_nothing_worse_than_info():
    df, specs = cs.clean_independent(0)
    f = cs.findings(df, specs)
    assert f.schema == _check.SCHEMA
    assert f.schema["severity"] == pl.Enum(["error", "warning", "info"])
    assert f.filter(pl.col("severity") != "info").is_empty(), f


@pytest.mark.parametrize("shape", sorted(cs.CLEAN))
def test_a_clean_shape_raises_no_error_or_warning(shape):
    df, specs = cs.CLEAN[shape](0)
    f = cs.findings(df, specs)
    assert f.filter(pl.col("severity") != "info").is_empty(), f


@pytest.mark.parametrize("name", sorted(cs.PLANTED))
def test_each_planted_problem_is_found_where_it_harms_and_nowhere_else(name):
    make, code, must, silent = cs.PLANTED[name]
    df, specs = make(0)
    f = cs.findings(df, specs)
    got = _codes(f, code)
    names = {s["name"] for s in specs}
    expected = must if must is not None else names - silent
    assert expected <= got, (expected - got, f)
    assert not (got & silent), (got & silent, f)


class TestTenSeeds:
    """The measurement the docstring quotes: no error or warning on any of the
    five clean shapes over ten seeds, for any of the fifteen specs (750
    spec-runs), and every planted problem found on every seed."""

    @pytest.mark.parametrize("shape", sorted(cs.CLEAN))
    def test_no_false_alarm(self, shape):
        for seed in range(10):
            df, specs = cs.CLEAN[shape](seed)
            f = cs.findings(df, specs)
            assert f.filter(pl.col("severity") != "info").is_empty(), (seed, f)

    @pytest.mark.parametrize("name", sorted(cs.PLANTED))
    def test_every_seed_finds_it(self, name):
        make, code, must, silent = cs.PLANTED[name]
        for seed in range(1, 10):
            df, specs = make(seed)
            got = _codes(cs.findings(df, specs), code)
            names = {s["name"] for s in specs}
            expected = must if must is not None else names - silent
            assert expected <= got and not (got & silent), (seed, expected, got)


def _troubled() -> tuple[pl.DataFrame, list[dict]]:
    """One stream with most problems at once, grouped: a missing column, a
    constant one, a level, a near-duplicate, a leak, a small group, steps
    back and resets."""
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.0, 0.5, 0.5]) + 0.3 * rng.normal(size=n)
    x[:, 1][rng.random(n) < 0.3] = np.nan
    df = pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x0": x[:, 0] + 50.0,
            "x1": x[:, 1],
            "x2": np.full(n, 4.0),
            "dup": x[:, 0] + 1e-3 * rng.normal(size=n),
            "leak": y + 1e-4 * rng.normal(size=n),
            "y": y,
            "g": np.where(np.arange(n) % 500 == 0, "tiny", np.where(np.arange(n) % 2, "a", "b")),
            "s": np.arange(n) // 1000,
        }
    )
    feats = ["x0", "x1", "x2", "dup", "leak"]
    specs = cs.every_model(
        feats, half_life=100.0, group="g", clock="t", gap_cap=10.0, session="s", session_gap="reset"
    )
    return df, specs


class TestStateAndChunks:
    def test_a_loaded_bank_finds_what_the_live_bank_found(self, tmp_path):
        df, specs = _troubled()
        bank = po.ModelBank(specs)
        with pytest.warns(po.ReadinessWarning):
            bank.fit_predict(df)
        live = bank.check()
        assert {"missing", "constant", "level_over_spread", "collinear", "leakage"} <= set(
            live["code"]
        )
        bank.save(tmp_path / "bank.state")
        loaded = po.ModelBank.load(tmp_path / "bank.state").check()
        assert loaded.equals(live)

    @pytest.mark.parametrize("chunks", [7, 600])
    def test_the_findings_do_not_depend_on_the_chunking(self, chunks):
        df, specs = _troubled()
        one = cs.findings(df, specs, chunks=1)
        many = cs.findings(df, specs, chunks=chunks)
        assert not one.is_empty()
        assert many.equals(one)

    def test_predict_moves_nothing(self):
        df, specs = _troubled()
        bank = po.ModelBank(specs)
        with pytest.warns(po.ReadinessWarning):
            bank.fit_predict(df.head(2000))
        before = bank.check()
        bank.predict(df.tail(1000))
        assert bank.check().equals(before)


class TestNarrowing:
    def test_spec_and_group_narrow_the_frame(self):
        df, specs = _troubled()
        bank = po.ModelBank(specs)
        with pytest.warns(po.ReadinessWarning):
            bank.fit_predict(df)
        every = bank.check()
        one = bank.check("rls")
        assert set(one["spec"]) == {"rls"}
        assert one.equals(every.filter(pl.col("spec") == "rls"))
        assert bank.check(7).equals(bank.check(specs[7]["name"]))
        tiny = bank.check("ewridge", group="tiny")
        assert set(tiny["group"]) == {"tiny"}
        assert "few_rows" in set(tiny["code"])
        assert bank.check("ewridge", group="never").is_empty()

    def test_an_unknown_spec_is_refused_as_the_other_readers_refuse_it(self):
        bank = po.ModelBank(cs.every_model(cs.FEATURES))
        with pytest.raises(KeyError, match="no spec named"):
            bank.check("nope")
        with pytest.raises(IndexError):
            bank.check(99)

    def test_a_bank_that_has_seen_nothing_has_nothing_to_say(self):
        assert po.ModelBank(cs.every_model(cs.FEATURES)).check().is_empty()


class TestValues:
    """The measured value is the number the bank's own tables give."""

    def test_missing_is_describes_null_share(self):
        df, specs = cs.planted_missing(0)
        bank = po.ModelBank(specs[:1])
        bank.fit_predict(df)
        row = bank.check().filter(code="missing").row(0, named=True)
        d = bank.describe().filter(column="x1").row(0, named=True)
        assert row["value"] == d["null_count"] / (d["count"] + d["null_count"])
        assert row["column"] == "x1"
        assert row["threshold"] == _check.MISSING_SHARE

    def test_leakage_is_the_grams_correlation(self):
        df, specs = cs.planted_leakage(0)
        spec = po.spec.ewridge("m", targets=["y"], features=["x0", "leak"], lam=1.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        row = bank.check().filter(code="leakage").row(0, named=True)
        # lam = 1 weighs every row alike, so numpy's correlation is the oracle.
        want = np.corrcoef(df["leak"].to_numpy(), df["y"].to_numpy())[0, 1]
        assert row["value"] == pytest.approx(abs(want), abs=1e-12)
        assert row["severity"] == "warning"

    def test_leakage_is_a_warning_as_a_long_random_walk_reads_as_it(self):
        """A random walk against its own previous row passes the line once
        the stream is long (review 6, G-7: on five seeds of ten at 30,000
        rows, nine at 100,000), and so does a relation measured with little
        noise. Neither stops a model learning, so ``leakage`` is a warning,
        from a Gram and from a ``marginal`` alike."""
        rng = np.random.default_rng(0)
        walk = np.cumsum(rng.standard_normal(100_001))
        df = pl.DataFrame({"y": walk[1:], "lag": walk[:-1]})
        specs = [
            po.spec.ewridge("ridge", targets=["y"], features=["lag"], half_life=float("inf")),
            po.spec.marginal("pairs", targets=["y"], features=["lag"], half_life=float("inf")),
        ]
        bank = po.ModelBank(specs)
        bank.fit(df)
        f = bank.check().filter(code="leakage")
        assert set(f["spec"]) == {"ridge", "pairs"}, f
        assert set(f["severity"]) == {"warning"}, f
        assert "random walk" in f["message"][0]

    def test_a_target_missing_by_design_is_information_only(self):
        df, specs = cs.clean_sparse_target(0)
        f = cs.findings(df, specs[:1])
        row = f.filter(code="missing").row(0, named=True)
        assert (row["severity"], row["column"]) == ("info", "y")
        assert row["value"] == pytest.approx(0.6, abs=0.03)

    def test_a_feature_null_on_every_row_is_an_error(self):
        df, specs = cs.planted_missing(0)
        df = df.with_columns(x1=pl.lit(None, pl.Float64))
        f = cs.findings(df, specs[:1])
        got = {(r["code"], r["severity"]) for r in f.iter_rows(named=True)}
        assert ("missing", "error") in got
        assert ("nothing_learned", "error") in got

    def test_the_ridge_read_is_the_one_the_fit_uses(self):
        """The resolved ridge, its smallest value under a grid; a ridge on the
        decaying sum scale fades, and a standardized one is scale-free."""
        df, _ = cs.clean_independent(0)
        c = dict(targets=["y"], features=cs.FEATURES, half_life=200.0)
        specs = [
            po.spec.ewridge("big", ridge=1.0, **c),
            po.spec.ewridge("grid", ridge=[1e-6, 1.0], **c),
            po.spec.ewridge("sum", ridge=1.0, ridge_scale="sum", **c),
            po.spec.ewridge("std", ridge=1.0, standardize=True, **c),
            po.spec.huber("huber", ridge=1.0, **c),
        ]
        f = cs.findings(df, specs)
        assert _codes(f, "ridge_shrinks") == {"big", "huber"}
        row = f.filter(code="ridge_shrinks", spec="big", column="x0").row(0, named=True)
        var = df["x0"].var()
        assert row["value"] == pytest.approx(1.0 / var, rel=1e-12)

    def test_low_support_is_not_repeated_for_a_constant_feature(self):
        df, specs = cs.planted_constant_feature(0)
        bank = po.ModelBank(specs[:1])
        with pytest.warns(po.ReadinessWarning):
            bank.fit_predict(df)
        assert bank.summary()["min_support_coef"][0] < _check.SUPPORT
        f = bank.check()
        assert set(f["code"]) == {"constant"}


class TestAudit:
    """An ``audit`` spec's findings (task 223 (b)): silent, but for
    information, on every clean shape of ``cs.AUDIT_CLEAN``, and each problem
    of ``cs.AUDIT_PLANTED`` found on its column with no other error or
    warning. ``TestAuditTenSeeds`` is the sweep the docstring's rates come
    from."""

    @pytest.mark.parametrize("shape", sorted(cs.AUDIT_CLEAN))
    def test_a_clean_shape_raises_no_error_or_warning(self, shape):
        f = cs.audit_findings(cs.AUDIT_CLEAN[shape](0))
        assert f.filter(pl.col("severity") != "info").is_empty(), f

    @pytest.mark.parametrize("name", sorted(cs.AUDIT_PLANTED))
    def test_each_planted_problem_is_found_on_its_column(self, name):
        make, code, column = cs.AUDIT_PLANTED[name]
        f = cs.audit_findings(make(0))
        assert not f.filter(code=code, column=column).is_empty(), f
        others = f.filter(pl.col("severity") != "info", pl.col("code") != code)
        assert others.is_empty(), others

    def test_the_value_is_the_audits_own_number(self):
        df = cs.AUDIT_PLANTED["minus_999_on_5pct"][0](0)
        bank = po.ModelBank([cs.audit_spec(df)])
        bank.fit_predict(df)
        row = bank.check().filter(code="sentinel", column="x1").row(0, named=True)
        a = bank.audit().filter(column="x1").row(0, named=True)
        assert row["value"] == a["top_count"] / a["count"]
        assert row["threshold"] == _check.SENTINEL_SHARE
        assert "-999" in row["message"]
        df = cs.AUDIT_PLANTED["random_walk"][0](0)
        bank = po.ModelBank([cs.audit_spec(df)])
        bank.fit_predict(df)
        row = bank.check().filter(code="random_walk").row(0, named=True)
        assert row["value"] == bank.audit().filter(column="x1")["unit_root_t"][0]

    def test_heavy_tails_reports_the_statistic_that_fired(self):
        """``heavy_tails`` on a kurtosis alone reports the kurtosis beside
        its own threshold, and on a robust z the robust z (review 6, G-11:
        the robust z was reported with its threshold of 10 whichever fired,
        a value below its own line)."""
        rng = np.random.default_rng(0)
        n = cs.N
        # A unit normal with 3% of its rows at +-8: a kurtosis near 12, and
        # no value 10 robust standard deviations out.
        kurt = np.where(rng.random(n) < 0.03, 8.0 * rng.choice([-1.0, 1.0], n), rng.normal(size=n))
        far = rng.normal(size=n)
        far[n // 2] = 30.0
        df = pl.DataFrame({"t": np.arange(n, dtype=float), "kurt": kurt, "far": far})
        bank = po.ModelBank([cs.audit_spec(df)])
        bank.fit_predict(df)
        a = {r["column"]: r for r in bank.audit().iter_rows(named=True)}
        assert a["kurt"]["kurtosis"] >= _check.HEAVY_KURTOSIS > 0
        assert a["kurt"]["robust_z"] < _check.HEAVY_ROBUST_Z
        assert a["far"]["robust_z"] >= _check.HEAVY_ROBUST_Z
        f = {r["column"]: r for r in bank.check().filter(code="heavy_tails").iter_rows(named=True)}
        assert (f["kurt"]["value"], f["kurt"]["threshold"]) == (
            a["kurt"]["kurtosis"],
            _check.HEAVY_KURTOSIS,
        )
        assert (f["far"]["value"], f["far"]["threshold"]) == (
            a["far"]["robust_z"],
            _check.HEAVY_ROBUST_Z,
        )

    def test_an_audit_alone_reports_steps_back_and_restarts(self):
        """An audit-only bank reports the summary's steps back and restarts
        as a model's spec does (review 6, F-6 first half: the audit's
        findings returned before the clock's were read)."""
        t = np.arange(cs.N, dtype=float)
        t[1200:] -= 600.0  # a restart
        df = cs.AUDIT_CLEAN["model_independent"](0).with_columns(t=pl.Series(t))
        audit = cs.audit_spec(df, restart_after_step_back=100.0)
        model = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=10.0,
            half_life=50.0,
            restart_after_step_back=100.0,
        )
        alone = cs.findings(df, [audit])
        beside = cs.findings(df, [audit, model])
        for spec, f in (("audit", alone), ("audit", beside), ("m", beside)):
            got = {r["code"]: r["value"] for r in f.filter(spec=spec).iter_rows(named=True)}
            assert got.get("step_back") == 1.0, (spec, f)
            assert got.get("resets") == 1.0, (spec, f)
        msg = alone.filter(code="resets")["message"][0]
        assert "keeps its counts" in msg, msg

    def test_a_column_missing_everywhere_is_an_error_and_partly_information(self):
        df = cs.AUDIT_CLEAN["model_independent"](0).with_columns(
            x1=pl.lit(None, pl.Float64), x2=pl.when(pl.col("t") % 4 == 0).then(None).otherwise("x2")
        )
        f = cs.audit_findings(df).filter(code="missing")
        got = {r["column"]: r["severity"] for r in f.iter_rows(named=True)}
        assert got == {"x1": "error", "x2": "info"}

    @pytest.mark.parametrize("chunks", [7, 600])
    def test_the_findings_do_not_depend_on_the_chunking(self, chunks):
        df = cs.AUDIT_PLANTED["stuck_50_rows"][0](0).with_columns(
            x2=pl.Series(np.where(np.arange(cs.N) % 9 == 0, np.inf, np.arange(cs.N) % 7))
        )
        one = cs.audit_findings(df)
        assert {"frozen", "sentinel", "few_values"} <= set(one["code"])
        assert cs.audit_findings(df, chunks=chunks).equals(one)

    def test_an_audit_beside_the_models_moves_none_of_their_findings(self):
        df, specs = cs.planted_collinear(0)
        alone = cs.findings(df, specs)
        beside = cs.findings(df, [*specs, cs.audit_spec(df)])
        assert beside.filter(pl.col("spec") != "audit").equals(alone)
        assert set(beside.filter(spec="audit")["code"]) <= {"duplicate"}

    def test_every_code_is_in_the_docstring(self):
        doc = po.ModelBank.check.__doc__ or ""
        codes = {code for _, code, _ in cs.AUDIT_PLANTED.values()}
        assert {f"``{c}``" for c in codes | {"missing", "constant"}} <= {
            w for w in doc.split() if w.startswith("``")
        }


class TestAuditTenSeeds:
    """The measurement the docstring quotes: no error or warning on any of the
    eleven clean shapes over ten seeds, and every planted problem found on
    every seed, on its column. Fast enough for the essentials, as
    ``TestTenSeeds`` is."""

    @pytest.mark.parametrize("shape", sorted(cs.AUDIT_CLEAN))
    def test_no_false_alarm(self, shape):
        for seed in range(10):
            f = cs.audit_findings(cs.AUDIT_CLEAN[shape](seed))
            assert f.filter(pl.col("severity") != "info").is_empty(), (seed, f)

    @pytest.mark.parametrize("name", sorted(cs.AUDIT_PLANTED))
    def test_every_seed_finds_it(self, name):
        make, code, column = cs.AUDIT_PLANTED[name]
        for seed in range(1, 10):
            f = cs.audit_findings(make(seed))
            assert not f.filter(code=code, column=column).is_empty(), (seed, f)


def test_check_places_every_regression():
    """A new regression joins ``_check._REGRESSIONS`` (``few_rows``), and is
    measured for ``_UNCENTRED_LIMITS`` if it does not centre its features.
    ``holt`` sits out: it fits a level and a trend to the target alone."""
    from test_model_registry import REGRESSIONS

    assert REGRESSIONS - {"holt"} == _check._REGRESSIONS
    assert set(_check._UNCENTRED_LIMITS) <= _check._REGRESSIONS
    assert _check._RIDGED <= _check._REGRESSIONS
    assert set(_native.model_kinds()) >= _check._GRAMMED


def test_without_numpy_the_gram_checks_say_they_did_not_run(monkeypatch):
    df, specs = cs.planted_collinear(0)
    bank = po.ModelBank(specs[:1])
    bank.fit_predict(df)
    assert "collinear" in set(bank.check()["code"])
    real_import = builtins.__import__

    def no_numpy(name, *a, **kw):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    f = bank.check()
    assert "collinear" not in set(f["code"])
    row = f.filter(code="not_checked").row(0, named=True)
    assert row["severity"] == "info"
    assert "polars-online[numpy]" in row["message"]
