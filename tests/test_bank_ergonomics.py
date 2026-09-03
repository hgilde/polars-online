"""What a ModelBank can say about itself (docs/IMPROVEMENTS.md U3).

A bank used to be opaque: no ``repr``, ``specs == []`` after ``load``, and no
way to see which groups it held or to forget the stale ones, so a long-running
bank's memory grew with every group ever seen. These pin ``repr``,
``rows_seen()``, ``groups()``, ``drop_groups()``, and that the specs survive
the state file -- inf and all.
"""

from __future__ import annotations

import pickle

import polars as pl
import polars.testing as plt
import pytest

import polars_online as po

INF = float("inf")
BASE = dict(targets=["y"], features=["x0"], halflife=10.0)


def _df(n: int = 60, groups: tuple[str, ...] = ("a", "b", "c")) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": [float(i % 7) for i in range(n)],
            "y": [float((i % 7) * 2 + 1) for i in range(n)],
            "g": [groups[i % len(groups)] for i in range(n)],
        }
    )


def _grouped_bank() -> po.ModelBank:
    return po.ModelBank(
        [
            po.spec.ewridge("m", group="g", clock="t", max_dclock=INF, **BASE),
            po.spec.rls("r", **BASE),
        ]
    )


# --- repr and rows_seen -----------------------------------------------------


def test_repr_names_the_specs_groups_and_rows():
    bank = _grouped_bank()
    assert repr(bank) == "ModelBank(['m', 'r'], groups=0, rows_seen=0)"
    bank.fit_predict(_df(60))
    assert repr(bank) == "ModelBank(['m', 'r'], groups=3, rows_seen=60)"
    bank.fit_predict(_df(30))
    assert bank.rows_seen() == 90
    assert repr(bank) == "ModelBank(['m', 'r'], groups=3, rows_seen=90)"


def test_rows_seen_counts_fed_rows_and_groups_count_processed_ones():
    # A null feature skips the row (README, null policy): the bank still saw
    # it, but the group's model did not process it.
    bank = po.ModelBank([po.spec.ewridge("m", group="g", **BASE)])
    df = _df(60).with_columns(
        pl.when(pl.col("t") < 6).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    bank.fit_predict(df)
    assert bank.rows_seen() == 60
    assert bank.groups()["rows_processed"].to_list() == [18, 18, 18]


# --- groups() ----------------------------------------------------------------


def test_groups_is_one_row_per_spec_and_group():
    bank = _grouped_bank()
    assert bank.groups().height == 0
    assert bank.groups().schema == {
        "spec": pl.String,
        "group": pl.String,
        "rows_processed": pl.UInt64,
        "last_clock": pl.Float64,
    }
    bank.fit_predict(_df(60))
    expected = pl.DataFrame(
        {
            "spec": ["m", "m", "m", "r"],
            # An ungrouped spec has the one key "", as in solve_failures().
            "group": ["a", "b", "c", ""],
            "rows_processed": [20, 20, 20, 60],
            # The last clock value each group saw; null on a row-count clock.
            "last_clock": [57.0, 58.0, 59.0, None],
        },
        schema=bank.groups().schema,
    )
    plt.assert_frame_equal(bank.groups(), expected)
    plt.assert_frame_equal(bank.groups("m"), expected.head(3))
    plt.assert_frame_equal(bank.groups(1), expected.tail(1))


def test_groups_keeps_a_null_key_apart_from_the_empty_string():
    bank = po.ModelBank([po.spec.ewridge("m", group="g", **BASE)])
    df = _df(30).with_columns(
        pl.when(pl.col("g") == "a").then(None).otherwise(pl.col("g")).alias("g")
    )
    bank.fit_predict(df)
    assert bank.groups()["group"].to_list() == [None, "b", "c"]
    assert bank.drop_groups([None]) == 1
    assert bank.groups()["group"].to_list() == ["b", "c"]


# --- drop_groups() -----------------------------------------------------------


def test_drop_groups_counts_streams_and_can_be_scoped_to_one_spec():
    bank = _grouped_bank()
    bank.fit_predict(_df(60))
    assert bank.drop_groups(["zzz"]) == 0
    assert bank.drop_groups(["b", "c"], spec="m") == 2
    assert bank.groups()["group"].to_list() == ["a", ""]
    # "r" is ungrouped: its one stream has the key "", the same string
    # groups() reports, so the stale-group idiom works for it too.
    assert bank.drop_groups([""]) == 1
    assert bank.groups()["spec"].to_list() == ["m"]
    with pytest.raises(KeyError, match="no spec named 'nope'; the bank has \\['m', 'r'\\]"):
        bank.drop_groups(["a"], spec="nope")
    with pytest.raises(IndexError, match="spec index 7 out of range"):
        bank.drop_groups(["a"], spec=7)


def test_a_dropped_group_starts_cold_and_the_others_are_untouched():
    first, second = _df(60), _df(60).with_columns(pl.col("t") + 60.0)
    bank = _grouped_bank()
    bank.fit_predict(first)
    assert bank.drop_groups(["b"], spec="m") == 1
    out = bank.fit_predict(second)

    # Untouched groups continue exactly as if nothing had been dropped.
    control = _grouped_bank()
    control.fit_predict(first)
    expected = control.fit_predict(second)
    plt.assert_frame_equal(out.filter(pl.col("g") != "b"), expected.filter(pl.col("g") != "b"))
    # The dropped group is a never-seen one: it matches a fresh bank fed only
    # its own second-chunk rows.
    fresh = _grouped_bank().fit_predict(second.filter(pl.col("g") == "b"))
    plt.assert_series_equal(out.filter(pl.col("g") == "b")["m"], fresh["m"])
    # rows_seen counts what was fed, not what is still held.
    assert bank.rows_seen() == 120
    assert bank.groups("m").filter(pl.col("group") == "b")["rows_processed"].item() == 20


# --- specs after load --------------------------------------------------------

ROUNDTRIP_SPECS = [
    po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        halflife=[INF, 10.0],
        ridge=[1e-6, 0.1],
        feature_sets={"a": ["x0"]},
        group="g",
        clock="t",
        max_dclock=INF,
        session="g",
        session_gap=INF,
    ),
    po.spec.ewridge("reset", clock="t", max_dclock=5.0, session="g", session_gap="reset", **BASE),
    po.spec.kalman("k", coef_halflife=[INF, 10.0], q=[0.0, 1.0], **BASE),
    po.spec.holt("h", targets=["y"], halflife=10.0, trend_halflife=INF),
    po.spec.sgd("s", clip_gradient=INF, **BASE),
    po.spec.lasso("l", lasso_path=[0.1, 0.01], **BASE),
]


def test_specs_survive_the_state_file_and_pickle():
    bank = po.ModelBank(ROUNDTRIP_SPECS)
    bank.fit_predict(_df(60))
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    assert loaded.specs == bank.specs
    assert loaded.rows_seen() == 60
    assert repr(loaded) == repr(bank)
    plt.assert_frame_equal(loaded.groups(), bank.groups())
    assert pickle.loads(pickle.dumps(bank)).specs == bank.specs


def test_specs_survive_save_to_a_path(tmp_path):
    bank = po.ModelBank(ROUNDTRIP_SPECS)
    bank.fit_predict(_df(60))
    bank.save(tmp_path / "bank.msgpack")
    assert po.ModelBank.load(tmp_path / "bank.msgpack").specs == bank.specs


def test_a_column_literally_named_inf_is_still_a_name():
    df = _df(30).rename({"x0": "inf", "y": "-inf"})
    spec = po.spec.ewridge("m", targets=["-inf"], features=["inf"], halflife=10.0)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    assert loaded.specs == [spec]
    assert loaded.specs[0]["features"] == ["inf"]
    plt.assert_frame_equal(loaded.fit_predict(df), bank.fit_predict(df))
