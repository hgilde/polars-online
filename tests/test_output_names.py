"""Task 244: the output surface standardized before 1.0.

A renamed parameter can be forwarded after 1.0; a renamed output field or
frame column cannot, since a struct field and a column have no alias. So the
output names are settled here, before the release that freezes them, by one
rule each:

- the frames say what Polars says for the same count (``null_count``,
  ``n_unique``), and ``unusable_count`` where the count is not Polars'
  (``describe()``'s counts every value a model cannot learn from);
- a count is fed rows wherever it counts them: ``rows_fed``;
- ``<quantity>_<stat>``: ``clock_last``, ``step_max``,
  ``support_coef_min``; a t-statistic follows the quantity it tests
  (``split_gain_t``, ``unit_root_t``), and the correlation's own keeps the
  conventional ``t_stat``, its serial-count twin ``t_stat_serial``;
- a reader leads with ``spec``, then the group and the instance, the
  instance with its decay (``half_life``, ``lam``), as ``coef()`` does;
- one spelling, ``cross_centered``; one name per diagnostic,
  ``ramsey_reset`` beside the clock rules' ``resets``;
- ``solve_failures`` is a column of ``summary()`` and a frame of its own.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def _rows(n: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    return pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "y": 2.0 * x0 - x1 + 0.1 * rng.normal(size=n),
            "g": np.arange(n) // 40,
            "t": np.arange(n, dtype=float),
        }
    )


def test_audit_names_its_counts_as_polars_does_and_leads_with_spec():
    audit = po.spec.audit("a", columns=["x0", "x1"], group="g", clock="t", gap_cap=5.0, pairs=True)
    bank = po.ModelBank([audit])
    bank.fit_predict(_rows().with_columns(x0=pl.when(pl.col("t") < 3).then(None).otherwise("x0")))
    columns = bank.audit()
    assert columns.columns[:6] == ["spec", "group", "column", "rows_fed", "null_count", "nan_count"]
    assert "n_unique" in columns.columns
    for gone in ("rows", "null", "nan", "distinct"):
        assert gone not in columns.columns
    assert columns.filter(pl.col("column") == "x0")["null_count"].sum() == 3
    assert set(columns["spec"].to_list()) == {"a"}
    assert bank.audit(pooled=True).columns[:3] == ["spec", "column", "rows_fed"]
    assert bank.audit(table="pairs").columns[:4] == ["spec", "group", "column_a", "column_b"]
    clock = bank.audit(table="clock")
    assert clock.columns[:2] == ["spec", "group"]
    assert "step_max" in clock.columns and "max_step" not in clock.columns


def test_describe_counts_what_a_model_cannot_learn_from_as_unusable():
    """``null_count`` counted NaN, infinities and out-of-bound values too,
    where Polars' ``null_count`` counts nulls alone."""
    bank = po.ModelBank([po.spec.ewridge("r", targets=["y"], features=["x0"], half_life=50.0)])
    df = _rows().with_columns(
        x0=pl.when(pl.col("t") < 3).then(float("nan")).otherwise(pl.col("x0"))
    )
    bank.fit_predict(df)
    row = bank.describe().filter(column="x0").row(0, named=True)
    assert (row["count"], row["unusable_count"]) == (117, 3)
    assert "null_count" not in bank.describe().columns
    assert df["x0"].null_count() == 0


def test_summary_and_groups_name_their_quantity_first():
    spec = po.spec.ewridge(
        "r", targets=["y"], features=["x0"], half_life=50.0, clock="t", gap_cap=10.0, group="g"
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(_rows())
    summary = bank.summary()
    for new, old in (
        ("clock_last", "last_clock"),
        ("support_coef_min", "min_support_coef"),
        ("support_coef_min_feature", "min_support_coef_feature"),
    ):
        assert new in summary.columns and old not in summary.columns, new
    assert summary["clock_last"].to_list() == [39.0, 79.0, 119.0]
    groups = bank.groups()
    assert groups.columns == ["spec", "group", "rows_fed", "clock_last"]


def test_solve_failures_is_a_summary_column_and_a_frame():
    """It was the one reader returning a dict, ``{spec: {group: count}}``."""
    spec = po.spec.ewridge(
        "r", targets=["y"], features=["x0", "x1"], half_life=50.0, group="g", ridge=0.0
    )
    # Two copies of one column and no ridge: every solve needs jitter.
    df = _rows().with_columns(x1=pl.col("x0"))
    bank = po.ModelBank([spec, po.spec.rls("s", targets=["y"], features=["x0"], half_life=50.0)])
    bank.fit_predict(df)
    failures = bank.solve_failures()
    assert isinstance(failures, pl.DataFrame)
    assert failures.schema == {"spec": pl.String, "group": pl.String, "solve_failures": pl.Int64}
    summary = bank.summary().select("spec", "group", "solve_failures")
    assert failures.equals(summary)
    assert failures.filter(spec="r")["solve_failures"].min() > 0
    assert failures.filter(spec="s")["solve_failures"].to_list() == [0]
    assert bank.solve_failures("s").equals(failures.filter(spec="s"))
    assert bank.solve_failures("r", group="0").height == 1


def test_coef_marginal_and_closed_groups_carry_the_instance_decay():
    """``output_index`` carried ``half_life`` and ``lam`` and no instance;
    ``coef()``, ``marginal()`` and ``closed_groups()`` the instance and no
    decay. Each now carries both, so a frame joins to another on either."""
    common = dict(targets=["y"], features=["x0", "x1"], half_life=[50.0, 200.0], group="g")
    ridge = po.spec.ewridge("r", group_close="monotone", **{k: v for k, v in common.items()})
    pairs = po.spec.marginal("p", lags=[1], **common)
    bank = po.ModelBank([ridge, pairs])
    bank.fit_predict(_rows())
    coef = bank.coef()
    assert coef.columns[:6] == ["spec", "group", "instance", "half_life", "lam", "weight_sum"]
    assert set(zip(coef["instance"], coef["half_life"], strict=True)) == {
        ("@h50", 50.0),
        ("@h200", 200.0),
    }
    marginal = bank.marginal("p")
    assert marginal.columns[:5] == ["spec", "group", "instance", "half_life", "lam"]
    assert set(marginal["spec"].to_list()) == {"p"}
    assert "t_stat_serial" in marginal.columns and "t_serial" not in marginal.columns
    closed = bank.closed_groups(drop=False)
    assert closed.columns[:5] == ["spec", "group", "instance", "half_life", "lam"]
    assert closed.filter(spec="r")["half_life"].unique().sort().to_list() == [50.0, 200.0]


def test_cross_centered_is_the_one_spelling():
    """``cross_centred`` was the library's one British spelling: the gram
    key, the closed column, and what ``po.gram.from_row`` reads, which takes
    the old key for one release with a deprecation warning."""
    spec = po.spec.ewridge(
        "r", targets=["y"], features=["x0"], half_life=50.0, group="g", group_close="monotone"
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(_rows())
    g = bank.gram("r")[0]
    assert "cross_centered" in g and "cross_centred" not in g
    closed = bank.closed_groups()
    assert "cross_centered" in closed.columns and "cross_centred" not in closed.columns
    row = closed.head(1)
    want = po.gram.from_row(row)
    old = row.rename({"cross_centered": "cross_centred"})
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="cross_centered"):
        got = po.gram.from_row(old)
    np.testing.assert_array_equal(got["cross_centered"], want["cross_centered"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        po.gram.from_row(row)


def test_ramsey_reset_is_named_apart_from_the_clock_rules_resets():
    spec = po.spec.ewridge(
        "r", targets=["y"], features=["x0"], half_life=50.0, emit_specification=True
    )
    fields = po.spec.output_fields(spec)
    assert "ramsey_reset_y" in fields and "reset_y" not in fields
    idx = po.spec.output_index(spec)
    assert idx.filter(field="ramsey_reset_y")["kind"].item() == "ramsey_reset"


def test_a_word_from_a_fixed_set_is_an_enum():
    """A field or column whose values come from a set the spec fixes is an
    ``Enum`` of that set, in its order, as ``withheld_reason`` is: a
    class, a selected slot, a column's role, an estimator. ``check()``'s
    ``code`` stays a ``String``: a later release may add a finding, and an
    ``Enum``'s categories are part of its dtype."""
    cls = po.spec.ew_class(
        "c",
        features=["x0", "x1"],
        label="lab",
        classes=["up", "down"],
        precision_prior=1.0,
        half_life=50.0,
    )
    sel = po.spec.ewridge(
        "s", targets=["y"], features=["x0", "x1"], half_life=[50.0, 200.0], emit_selected=True
    )
    rcov = po.spec.rcov(
        "r", features=["x0", "x1"], block_rows=20, group="g", group_close="monotone"
    )
    df = _rows().with_columns(
        lab=pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down"))
    )
    bank = po.ModelBank([cls, sel, rcov])
    out = bank.fit_predict(df)
    assert out["c"].struct.field("class").dtype == pl.Enum(["up", "down"])
    assert set(out["c"].struct.field("class").drop_nulls().to_list()) == {"up", "down"}
    selected = out["s"].struct.field("selected_y")
    assert selected.dtype == pl.Enum(["@h50", "@h200"])
    assert set(selected.drop_nulls().to_list()) <= {"@h50", "@h200"}
    for spec, field in ((cls, "class"), (sel, "selected_y")):
        idx = po.spec.output_index(spec)
        assert idx.filter(field=field)["dtype"].item() == "enum"
    assert bank.describe().schema["role"] == pl.Enum(["feature", "target", "weight"])
    assert bank.closed_groups(drop=False).schema["rcov_kind"] == pl.Enum(
        ["plain", "kernel", "preavg"]
    )
    assert bank.check().schema["code"] == pl.String
    # The declared struct is the realized one, the IO plugin's schema too.
    plan = po.fit_predict(df.lazy(), [cls, sel])
    assert plan.collect_schema()["c"] == out.schema["c"]
