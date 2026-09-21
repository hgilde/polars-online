"""What a ModelBank can say about itself (docs/IMPROVEMENTS.md U3).

A bank used to be opaque: no ``repr``, ``specs == []`` after ``load``, and no
way to see which groups it held or to forget the stale ones, so a long-running
bank's memory grew with every group ever seen. These pin ``repr``,
``rows_seen()``, ``groups()``, ``drop_groups()``, and that the specs survive
the state file -- inf and all.
"""

from __future__ import annotations

import pickle

import numpy as np
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


def test_the_specs_are_the_banks_before_a_round_trip_as_after_it():
    """A dict the builders did not write -- a ``bocpd`` with no ``targets``
    and no ``drift_action`` -- was the caller's own before a save and the
    bank's filled form after it, and ``coef_index``, ``gram`` and ``coef``
    read the unfilled one (review 2026-09-12, S21)."""
    hand = {"name": "b", "model": {"type": "bocpd", "prior_scale": [1.0]}, "features": ["x0"]}
    bank = po.ModelBank([hand])
    assert bank.specs == po.ModelBank.load_bytes(bank.save_bytes()).specs
    assert bank.specs[0]["targets"] == ["x0"]
    assert bank.specs[0]["drift_action"] == "flag"


def _stream(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )


def _one(**kw):
    return po.spec.ewridge(
        "m", targets=["y"], features=["x0"], halflife=float("inf"), min_periods=2.0, **kw
    )


def test_fit_predict_batches_takes_a_plan_and_chunks_it():
    """A LazyFrame in, and the method does the chunking: the same rows and the
    same numbers as feeding the chunks by hand, whatever `chunk_rows` is."""
    df = _stream()
    want = pl.concat(
        po.ModelBank([_one()]).fit_predict_batches(df.slice(i, 7) for i in range(0, 40, 7))
    )
    for rows in (7, 13, 1000):
        got = pl.concat(po.ModelBank([_one()]).fit_predict_batches(df.lazy(), chunk_rows=rows))
        # `coef` rides the chunk cadence; every other field is the same.
        drop = lambda f: f.with_columns(  # noqa: E731
            pl.col("m").struct.with_fields(
                pl.lit(None).alias("coef"), pl.lit(None).alias("support_coef")
            )
        )
        assert drop(got).equals(drop(want), null_equal=True), rows
    with pytest.raises(ValueError, match="chunk_rows must be at least 1"):
        list(po.ModelBank([_one()]).fit_predict_batches(df.lazy(), chunk_rows=0))


def test_a_frame_is_one_chunk():
    df = _stream()
    outs = list(po.ModelBank([_one()]).fit_predict_batches(df))
    assert len(outs) == 1 and outs[0].height == df.height


def test_fit_leaves_the_state_fit_predict_batches_leaves():
    """The learn-only form is the same run with the output dropped: the state
    it saves is byte-identical, so nothing about the fit depends on who reads
    the frames."""
    df = _stream(seed=1)
    kept = po.ModelBank([_one()])
    for _ in kept.fit_predict_batches(df.lazy(), chunk_rows=9):
        pass
    quiet = po.ModelBank([_one()])
    assert quiet.fit(df.lazy(), chunk_rows=9) is None
    assert quiet.save_bytes() == kept.save_bytes()
    assert quiet.rows_seen() == kept.rows_seen() == df.height


def test_a_frame_with_chunk_rows_is_fed_in_slices():
    """``chunk_rows`` on a ``DataFrame`` slices it, rather than being checked and
    then ignored (review 2026-09-17, B5): the same rows, in bounded pieces, and
    the state the plan route leaves, byte for byte."""
    df = _stream(seed=1)
    outs = list(po.ModelBank([_one()]).fit_predict_batches(df, chunk_rows=9))
    assert len(outs) == -(-df.height // 9)
    assert pl.concat(outs).height == df.height
    sliced = po.ModelBank([_one()])
    for _ in sliced.fit_predict_batches(df, chunk_rows=9):
        pass
    planned = po.ModelBank([_one()])
    for _ in planned.fit_predict_batches(df.lazy(), chunk_rows=9):
        pass
    assert sliced.save_bytes() == planned.save_bytes()
    # Without a size, the frame is still one chunk.
    assert len(list(po.ModelBank([_one()]).fit_predict_batches(df))) == 1


def test_fit_over_an_empty_plan_is_not_an_error():
    """Nothing to learn from is a no-op that still leaves a loadable state."""
    bank = po.ModelBank([_one()])
    bank.fit(_stream().clear().lazy())
    assert bank.rows_seen() == 0
    assert po.ModelBank.load_bytes(bank.save_bytes()).rows_seen() == 0


def test_fit_predict_batches_drains_the_closed_groups_as_it_goes(tmp_path):
    """A ``ModelBank``'s closed groups wait in its queue until something
    drains it, and ``save`` writes them all. ``fit_predict_batches`` with a
    ``closed_groups`` path drains after every batch, so the queue is empty
    whenever a batch is handed on, and writes what it drained when the
    batches run out (review 2026-09-12, P5; the user's decision of
    2026-09-15)."""
    spec, df, batches = _closing_batches()
    bank = po.ModelBank([spec])
    path = tmp_path / "closed.parquet"
    for _ in bank.fit_predict_batches(batches, closed_groups=path):
        assert bank.closed_groups(drop=False).height == 0, "drained as it goes"
    want = po.ModelBank([spec])
    want.fit_predict(df)
    assert pl.read_parquet(path).equals(want.closed_groups())


@pytest.mark.parametrize("stop", ["break", "error"])
def test_fit_predict_batches_writes_what_it_drained_however_it_stops(tmp_path, stop):
    """A drained row has left the bank, so the file is the only place it is.
    A caller that stops after two batches, or a source that fails there,
    still gets the rows those two closed written, and the file and the
    bank's queue together hold every close."""
    spec, _, batches = _closing_batches()

    def source():
        yield batches[0]
        yield batches[1]
        if stop == "error":
            raise RuntimeError("the source failed")
        yield from batches[2:]

    bank = po.ModelBank([spec])
    path = tmp_path / "closed.parquet"
    gen = iter(bank.fit_predict_batches(source(), closed_groups=path))
    next(gen)
    next(gen)
    if stop == "break":
        gen.close()  # what a `break` out of a `for` over it does
    else:
        with pytest.raises(RuntimeError, match="the source failed"):
            next(gen)
    want = po.ModelBank([spec])
    want.fit_predict(pl.concat(batches[:2]))
    assert pl.read_parquet(path).equals(want.closed_groups())
    assert bank.closed_groups(drop=False).height == 0


def _closing_batches():
    """Twelve groups of five rows, in key order, as six batches of ten: under
    ``monotone`` a group closes when the next key arrives."""
    spec = po.spec.ew_cov(
        "c", features=["x0", "y"], halflife=40.0, group="g", group_close="monotone"
    )
    keys = [f"k{i:02d}" for i in range(12)]
    df = pl.DataFrame(
        {
            "g": [k for k in keys for _ in range(5)],
            "x0": [float((i * 7) % 11) for i in range(60)],
            "y": [float((i * 5) % 13) for i in range(60)],
        }
    )
    return spec, df, [df.slice(i, 10) for i in range(0, df.height, 10)]


# --- the specs are a read-only view ------------------------------------------


def test_specs_cannot_be_assigned():
    """``bank.specs`` is a property with no setter.

    The bank's behaviour comes from the Rust state built at construction, so
    a Python-side list can only ever disagree with it. Assignment used to
    succeed and desynchronise the two.
    """
    bank = po.ModelBank([po.spec.ewridge("m", **BASE)])
    with pytest.raises(AttributeError):
        bank.specs = []


def test_mutating_what_specs_returns_changes_nothing():
    """It hands back a copy, so an edit in place cannot reach the bank.

    Before, ``bank.specs[0]["features"] = [...]`` left ``coef()`` reading a
    layout the bank was not running -- an ``AssertionError`` about the number
    of coefficients, blamed on the model rather than on the edit.
    """
    bank = po.ModelBank([po.spec.ewridge("m", **BASE)])
    bank.fit_predict(_df(40))
    before = bank.coef("m")["term"].to_list()

    got = bank.specs
    got[0]["features"] = ["x0", "ghost", "phantom"]
    got.append({"nonsense": True})

    assert bank.specs[0]["features"] == ["x0"], "the bank kept its own copy"
    assert len(bank.specs) == 1
    assert bank.coef("m")["term"].to_list() == before


def test_no_public_attribute_escapes_the_api_snapshot():
    """Every public name must be on the class, or ``tests/api_surface.txt``
    never sees it.

    The snapshot walks ``dir(po.ModelBank)``, so an attribute set in
    ``__init__`` is invisible to it -- which is what happened to ``specs``:
    public, documented nowhere, and free to change without the reviewable
    diff the snapshot exists to produce.
    """
    bank = po.ModelBank([po.spec.ewridge("m", **BASE)])
    bank.fit_predict(_df(20))
    on_class = {n for n in dir(po.ModelBank) if not n.startswith("_")}
    on_instance = {n for n in dir(bank) if not n.startswith("_")}
    assert on_instance - on_class == set()


# --- walking a state file that nothing has described -------------------------


def test_a_state_file_describes_itself(tmp_path):
    """Everything needed to navigate a bank, from the file and nothing else.

    No ``specs=``, no knowledge of what was run: the specs come back as the
    dicts the builders made, and every accessor agrees with them.
    """
    specs = [
        po.spec.ewridge("ridge", targets=["y"], features=["x0"], halflife=50.0, group="g"),
        po.spec.ew_cov("cov", features=["x0", "y"], stats=["corr"], halflife=INF, group="g"),
    ]
    bank = po.ModelBank(specs)
    bank.fit_predict(_df(60).with_columns(g=pl.Series(["a"] * 30 + ["b"] * 30)))
    bank.save(tmp_path / "bank.state")

    loaded = po.ModelBank.load(tmp_path / "bank.state")
    assert loaded.specs == specs, "the file carries the specs, defaults included"

    names = [s["name"] for s in loaded.specs]
    assert names == ["ridge", "cov"]
    assert set(loaded.output_fields()) == set(names)
    assert loaded.groups()["spec"].unique().sort().to_list() == sorted(names)
    for frame in (loaded.last_row(), loaded.summary(), loaded.describe()):
        assert frame.columns[0] == "spec", "every table leads with the spec"
        assert set(frame["spec"].unique()) == set(names), "and defaults to all of them"
    assert loaded.rows_seen() == 60
    # the model kind is in there too, so a caller can branch on it
    assert loaded.specs[0]["model"]["type"] == "ew_ridge"
    assert loaded.specs[1]["model"]["type"] == "ew_cov"
