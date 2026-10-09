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
        assert row["severity"] == "error"

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

    def test_low_support_is_not_repeated_for_a_constant_feature(self):
        df, specs = cs.planted_constant_feature(0)
        bank = po.ModelBank(specs[:1])
        with pytest.warns(po.ReadinessWarning):
            bank.fit_predict(df)
        assert bank.summary()["min_support_coef"][0] < _check.SUPPORT
        f = bank.check()
        assert set(f["code"]) == {"constant"}


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
