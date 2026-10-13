"""``ModelBank.check`` on the paths no other test took (task 252): a model
other than ``lasso`` whose solve needed jitter, a finding about the group
whose key is null, and an audit's clock counts for a group with a single
row, which has no step to judge."""

from __future__ import annotations

import numpy as np
import polars as pl

import polars_online as po

TIER = "essential"


def test_a_solve_that_needed_jitter_is_a_warning_with_its_count():
    """``ewridge`` with no ridge on two copies of one feature cannot solve
    its normal equations, and each row it jitters or keeps the last fit is
    counted: ``check`` reports that count, as ``solve_failures`` gives it,
    as a warning (``lasso`` reports its sweeps as information instead)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    df = pl.DataFrame({"x": x, "x2": x, "y": 2 * x + rng.normal(size=400) * 0.1})
    bank = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x", "x2"], half_life=50.0, ridge=0.0)]
    )
    bank.fit_predict(df)
    failed = bank.solve_failures()["solve_failures"].item()
    assert failed > 0, "the duplicated feature made the solve fail"
    found = bank.check().filter(pl.col("code") == "solve_failures")
    assert found.height == 1
    row = found.row(0, named=True)
    assert row["severity"] == "warning"
    assert row["value"] == float(failed) and row["threshold"] == 0.0
    assert row["message"].startswith(
        f"The stream needed jitter or kept the previous fit {failed} times: the features are "
        "constant, collinear or too few rows for their number"
    )


def test_a_finding_about_the_null_key_names_the_null_group():
    """Rows whose group key is null form a group of their own, which a
    message calls "the null group" (an empty key is "the stream")."""
    df = pl.DataFrame(
        {
            "g": [None] * 200 + ["a"] * 200,
            "x": [None if i % 3 else float(i) for i in range(200)] + [float(i) for i in range(200)],
            "y": [float(i % 7) for i in range(400)],
        },
        schema={"g": pl.String, "x": pl.Float64, "y": pl.Float64},
    )
    bank = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x"], half_life=50.0, group="g")]
    )
    bank.fit_predict(df)
    missing = bank.check().filter(pl.col("code") == "missing")
    assert missing["group"].to_list() == [None]
    assert "fed to the null group" in missing["message"].item()


def test_an_audit_says_nothing_of_the_clock_of_a_group_with_one_row():
    """A group fed one row has taken no step, so the audit's clock counts
    give it no finding, while a group whose rows share a stamp is told so."""
    audit = po.spec.audit("a", columns=["x"], group="g", clock="t", gap_cap=10.0)
    df = pl.DataFrame(
        {"g": ["a", "b", "a", "a"], "t": [1.0, 1.0, 2.0, 2.0], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    bank = po.ModelBank([audit])
    bank.fit_predict(df)
    clock = bank.audit(table="clock")
    assert dict(zip(clock["group"], clock["steps"], strict=True)) == {"a": 2, "b": 0}
    found = bank.check()
    assert found.filter(pl.col("group") == "b").height == 0
    dup = found.filter(pl.col("code") == "duplicate_stamps")
    assert dup["group"].to_list() == ["a"] and dup["value"].item() == 0.5
