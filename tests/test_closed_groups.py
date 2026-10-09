"""E54: a group's accumulators emitted when the bank can prove it is finished.

The promise is exact: the closed row **is** the ``gram()`` a driver would
have read at that point, and the stream is dropped, so a bank over an
unbounded key space stays bounded. Both halves are tested here -- the row
against ``gram()`` bit for bit, and the schedule against the two rules
(a key smaller than the largest seen, or a session that has ended) under
every chunking.
"""

import struct
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online

TIER = "essential"

HALFLIFE = 40.0


def cov_spec(**kw):
    kw.setdefault("group", "g")
    return po.spec.ew_cov("c", features=["x0", "x1"], half_life=HALFLIFE, **kw)


def frame(keys, n_per=5, seed=0, session=None):
    """One run of `n_per` rows per key, in key order."""
    rng = np.random.default_rng(seed)
    g, x0, x1, t = [], [], [], []
    for i, k in enumerate(keys):
        for j in range(n_per):
            g.append(k)
            x0.append(float(rng.normal()))
            x1.append(float(rng.normal()))
            t.append(float(i * n_per + j))
    df = pl.DataFrame({"g": g, "x0": x0, "x1": x1, "t": t})
    if session is not None:
        df = df.with_columns(pl.Series("s", session))
    return df


# --- the row is the gram -----------------------------------------------------


def test_a_closed_row_is_the_gram_read_at_the_same_point():
    """The acceptance: field for field, and bit for bit on the half the row
    carries. One builder makes both (`gram_of`), so this is a tripwire."""
    df = frame(["a", "b"])
    closing = po.ModelBank([cov_spec(group_close="monotone")])
    closing.fit_predict(df)
    row = closing.closed_groups()
    assert row.height == 1, "only 'a' is finished; 'b' is still open"

    # The same rows, to a bank that does not close: `gram()` for group 'a'.
    plain = po.ModelBank([cov_spec()])
    plain.fit_predict(df.filter(pl.col("g") == "a"))
    g = plain.gram("c")[0]

    got = po.gram.from_row(row)
    assert got["weight_sum"] == g["weight_sum"]
    assert got["n_kish"] == g["n_kish"]
    assert got["columns"] == g["columns"] == ["x0", "x1"]
    assert np.array_equal(got["means"], g["means"])
    # The row packs the upper triangle; that half is bit-identical, and the
    # lower one is its mirror (the two differ in the last bit -- the products
    # are added in the opposite order, docs/PERFORMANCE.md §14).
    k = len(g["columns"])
    iu = np.triu_indices(k)
    assert np.array_equal(got["comoments"][iu], g["comoments"][iu])
    assert np.allclose(got["comoments"], g["comoments"], rtol=0, atol=1e-15)
    assert row["rows_fed"][0] == 5 and row["rows_learned"][0] == 5
    assert row["clock_min"][0] is None, "no clock column: no clock range"


def test_a_closed_ridge_row_carries_the_target_moments_and_coef():
    df = frame(["a", "b"]).with_columns(y=pl.col("x0") * 2.0 + pl.col("x1"))
    spec = po.spec.ewridge(
        "r",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALFLIFE,
        group="g",
        group_close="monotone",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    row = bank.closed_groups()
    got = po.gram.from_row(row)
    assert got["columns"] == ["intercept", "x0", "x1"]
    assert got["targets"] == ["y"]
    assert got["cross_moments"].shape == (1, 3)
    beta = po.gram.solve(got, target=0)
    assert np.allclose(beta[1:], [2.0, 1.0], atol=1e-6)
    # `coef` is as of the last solve, and the exact solve is `from_row`'s.
    assert np.allclose(row["coef"][0].to_list()[1:], [2.0, 1.0], atol=1e-6)


def test_the_clock_range_and_the_weight_counts_come_from_the_summary():
    df = frame(["a", "b"]).with_columns(w=pl.lit(2.0))
    bank = po.ModelBank([cov_spec(group_close="monotone", clock="t", gap_cap=1e9, weight="w")])
    bank.fit_predict(df)
    row = bank.closed_groups()
    assert row["clock_min"][0] == 0.0 and row["clock_max"][0] == 4.0
    assert row["rows_fed"][0] == 5 and row["rows_learned"][0] == 5


def test_eig_is_the_eigendecomposition_of_the_rows_own_comoments():
    df = frame(["a", "b"], n_per=40)
    spec = po.spec.ew_cov(
        "c",
        features=["x0", "x1"],
        half_life=HALFLIFE,
        group="g",
        group_close="monotone",
        pca=2,
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    row = bank.closed_groups()
    g = po.gram.from_row(row)
    vals, vecs = np.linalg.eigh(g["comoments"])
    order = np.argsort(vals)[::-1]
    assert np.allclose(row["eig_vals"][0].to_list(), vals[order], rtol=1e-12)
    got = np.asarray(row["eig_vecs"][0].to_list()).reshape(2, 2)
    for j in range(2):
        want = vecs[:, order[j]]
        # An eigenvector's sign is arbitrary; the row signs for continuity.
        assert np.allclose(np.abs(got[j] @ want), 1.0, atol=1e-10)


def test_a_marginal_spec_closes_with_its_pairs():
    df = frame(["a", "b"]).with_columns(y=pl.col("x0") * 3.0)
    spec = po.spec.marginal(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALFLIFE,
        group="g",
        group_close="monotone",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    row = bank.closed_groups()
    assert row["pair_feature"][0].to_list() == ["x0", "x1"]
    assert row["pair_target"][0].to_list() == ["y", "y"]
    assert row["pair_beta"][0].to_list()[0] == pytest.approx(3.0, abs=1e-9)


def test_one_bank_gives_one_schema_whatever_closed():
    """A driver concatenating a run's drains needs every frame to have the
    same columns, so the blocks follow the bank's specs, not the rows."""
    specs = [
        cov_spec(group_close="monotone"),
        po.spec.marginal(
            "m",
            targets=["x0"],
            features=["x1"],
            half_life=HALFLIFE,
            group="g",
            group_close="monotone",
        ),
    ]
    bank = po.ModelBank(specs)
    empty = bank.closed_groups()
    bank.fit_predict(frame(["a", "b"]))
    got = bank.closed_groups()
    assert dict(empty.schema) == dict(got.schema)
    assert "pair_corr" in got.columns and "comoments" in got.columns
    # The marginal row has no Gram block and the ew_cov row has no pairs.
    by_spec = {r["spec"]: r for r in got.iter_rows(named=True)}
    assert by_spec["c"]["pair_corr"] is None
    assert by_spec["m"]["comoments"] is None


def test_a_closed_marginal_row_carries_its_lags_and_bins():
    """E66 and E67 ride into the closed row as lists of lists, one inner
    list per pair, and are -- pair for pair, value for value -- what
    ``marginal()`` reports on the same rows. The closed row is the only
    readout for a closed group, so a column it dropped would be lost."""
    df = frame(["a", "b"], n_per=60, seed=3).with_columns(y=pl.col("x0") * 3.0 + pl.col("x1"))
    kw = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALFLIFE,
        lags=[1, 2],
        serial_rule="truncated",
        bin_edges=[[0.0], [-0.5, 0.5]],
    )
    closing = po.ModelBank([po.spec.marginal("m", group="g", group_close="monotone", **kw)])
    closing.fit_predict(df)
    closed = closing.closed_groups()
    assert closed.schema["pair_lag_corr_xx"] == pl.List(pl.List(pl.Float64))
    assert closed.schema["pair_bin_n"] == pl.List(pl.List(pl.Float64))
    row = closed.row(0, named=True)
    plain = po.ModelBank([po.spec.marginal("m", **kw)])
    plain.fit_predict(df.filter(pl.col("g") == "a"))
    want = plain.marginal("m")
    assert row["pair_feature"] == want["feature"].to_list() == ["x0", "x1"]
    for col in (
        "lag_corr_xx",
        "lag_corr_yy",
        "lag_corr_xy",
        "lag_corr_yx",
        "n_serial",
        "t_serial",
        "phi_x",
        "phi_y",
        "bin_edges",
        "bin_n",
        "bin_mean_y",
        "bin_var_y",
        "split_gain",
        "split_at",
        "split_gain_t",
    ):
        assert row[f"pair_{col}"] == want[col].to_list(), col
    assert row["pair_bin_edges"] == [[0.0], [-0.5, 0.5]], "ragged, as given"
    assert row["pair_phi_x"] == [None, None], "truncated fits no decay"
    assert row["pair_split_gain"][0] > 0.5, "x0 carries the target, and a cut at 0 sees it"


def test_the_lag_and_bin_blocks_follow_the_specs_that_asked():
    """Two closing marginals, one with lags and bins and one without: one
    schema for the bank, and on the row of the one that did not ask the
    columns are null -- the columns ``marginal()`` would not have."""
    common = dict(
        targets=["y"], features=["x0", "x1"], half_life=HALFLIFE, group="g", group_close="monotone"
    )
    specs = [
        po.spec.marginal("plain", **common),
        po.spec.marginal("asked", lags=[1], bin_edges=[[0.0], [0.0]], **common),
    ]
    bank = po.ModelBank(specs)
    empty = bank.closed_groups()
    bank.fit_predict(frame(["a", "b"], n_per=20).with_columns(y=pl.col("x0")))
    got = bank.closed_groups()
    assert dict(empty.schema) == dict(got.schema)
    by_spec = {r["spec"]: r for r in got.iter_rows(named=True)}
    assert by_spec["plain"]["pair_corr"] is not None
    for col in ("pair_lag_corr_xx", "pair_n_serial", "pair_bin_n", "pair_split_gain"):
        assert by_spec["plain"][col] is None, col
        assert by_spec["asked"][col] is not None, col
    # And a bank whose closing marginals asked for neither has neither
    # block, as `marginal()` has neither column.
    bare = po.ModelBank([po.spec.marginal("plain", **common)]).closed_groups()
    assert not any(c.startswith(("pair_lag_corr", "pair_bin", "pair_split")) for c in bare.columns)


def test_the_cross_terms_follow_the_specs_that_keep_them():
    """``cross_lags`` (E70) under the same rule: ``pair_lag_corr_xy`` and
    ``pair_lag_corr_yx`` are there when any closing spec keeps a cross term,
    null on the row of one that keeps none, and absent when none does, as
    ``marginal()`` leaves them out under ``cross_lags=[]``."""
    common = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALFLIFE,
        group="g",
        group_close="monotone",
        lags=[1, 2],
    )
    bank = po.ModelBank(
        [
            po.spec.marginal("none", cross_lags=[], **common),
            po.spec.marginal("one", cross_lags=[2], **common),
        ]
    )
    bank.fit_predict(frame(["a", "b"], n_per=20).with_columns(y=pl.col("x0")))
    by_spec = {r["spec"]: r for r in bank.closed_groups().iter_rows(named=True)}
    assert by_spec["none"]["pair_lag_corr_xy"] is None
    assert by_spec["none"]["pair_lag_corr_xx"] is not None
    assert [len(v) for v in by_spec["one"]["pair_lag_corr_xy"]] == [1, 1], "one per cross lag"
    assert [len(v) for v in by_spec["one"]["pair_lag_corr_xx"]] == [2, 2], "one per lag"
    bare = po.ModelBank([po.spec.marginal("none", cross_lags=[], **common)]).closed_groups()
    assert "pair_lag_corr_xx" in bare.columns
    assert not {"pair_lag_corr_xy", "pair_lag_corr_yx"} & set(bare.columns)


def test_the_sidecar_carries_the_nested_lists(tmp_path):
    """Parquet has a form for a list of lists, so the sidecar is
    the driver's frames with the blocks in them, as it is without."""
    df = frame(["a", "b", "c"], n_per=30, seed=5).with_columns(y=pl.col("x1") - pl.col("x0"))
    df.write_parquet(tmp_path / "in.parquet")
    spec = po.spec.marginal(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALFLIFE,
        lags=[1, 3],
        bins=4,
        bin_warm_rows=10,
        group="g",
        group_close="monotone",
    )
    side = tmp_path / "closed.parquet"
    df.lazy().online.fit_predict([spec], closed_groups=side, chunk_size=25).collect()
    driver = po.ModelBank([spec])
    frames = []
    for i in range(0, df.height, 25):
        driver.fit_predict(df[i : i + 25])
        frames.append(driver.closed_groups())
    want = pl.concat(frames)
    assert want.height == 2 and want["pair_bin_n"].null_count() == 0
    assert pl.read_parquet(side).equals(want)


# --- the schedule ------------------------------------------------------------


def test_the_last_group_never_closes_and_stays_readable():
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(frame(["a", "b", "c"]))
    closed = bank.closed_groups()
    assert closed["group"].to_list() == ["a", "b"]
    assert [g["group"] for g in bank.gram("c")] == ["c"]
    assert bank.groups()["group"].to_list() == ["c"]


def test_an_integer_key_closes_in_numeric_order_not_lexicographic():
    """`"10"` sorts before `"9"` as a string; a monotone close on that order
    would emit group 10 before group 9 had arrived."""
    df = frame([9, 10, 11]).with_columns(pl.col("g").cast(pl.Int64))
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(df)
    assert bank.closed_groups()["group"].to_list() == ["9", "10"]


def test_a_categorical_key_closes_bytewise():
    df = frame(["a", "b", "c"]).with_columns(pl.col("g").cast(pl.Categorical))
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(df)
    assert bank.closed_groups()["group"].to_list() == ["a", "b"]


def test_a_categorical_key_out_of_order_is_refused_and_a_sort_fixes_it():
    """Bytewise is the order for anything but an integer column. A
    ``Categorical`` once sorted by its *physical* order, the order its
    categories were first seen, so the refusal told the reader to cast it to
    String before sorting (docs/REVIEW-E54-E64.md G2). Polars sorts it as
    text from 1.34.0 on, the floor, so the refusal names the sort alone and
    sorting the Categorical column is enough (task 154)."""
    df = frame(["c", "a", "b"]).with_columns(pl.col("g").cast(pl.Categorical))
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="sort the input by") as e:
        bank.fit_predict(df)
    assert "Categorical" not in str(e.value)
    fixed = df.sort("g", maintain_order=True)
    assert fixed.schema["g"] == pl.Categorical
    po.ModelBank([cov_spec(group_close="monotone")]).fit_predict(fixed)


def test_a_high_water_mark_read_under_the_wrong_dtype_is_refused_both_ways():
    """A mark and the keys have to be ordered the same way, or the mark lets
    through exactly the groups it exists to refuse: ``"10"`` comes after
    ``"9"`` as an integer and before it as text
    (docs/REVIEW-E54-E64.md G1). Since review round 4 (N22) the bank keeps
    every group column's form, and refuses a column of another form before
    the mark is read, for every spec and not for ``"monotone"`` alone."""
    spec = cov_spec(group_close="monotone")

    # Saved under an integer column, resumed under a text one.
    bank = po.ModelBank([spec])
    bank.fit_predict(frame([9, 10, 11]))
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    with pytest.raises(ValueError, match='group column "g" was i64 and is now str'):
        again.fit_predict(frame(["12", "13"]))

    # And the other way.
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(["a", "b", "c"]))
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    with pytest.raises(ValueError, match='group column "g" was str and is now i64'):
        again.fit_predict(frame([9, 10]))


def test_a_group_split_across_chunks_closes_once_and_at_the_same_numbers():
    df = frame(["a", "b", "c"], n_per=7)
    whole = po.ModelBank([cov_spec(group_close="monotone")])
    whole.fit_predict(df)
    piecemeal = po.ModelBank([cov_spec(group_close="monotone")])
    for i in range(0, df.height, 3):
        piecemeal.fit_predict(df[i : i + 3])
    assert whole.closed_groups().equals(piecemeal.closed_groups())


def test_a_session_close_splits_the_run_and_restarts_the_stream():
    sess = ["m"] * 5 + ["t"] * 5 + ["w"] * 5
    df = frame(["a"], n_per=15, session=sess)
    bank = po.ModelBank([cov_spec(group_close="session", session="s", clock="t", gap_cap=1e9)])
    bank.fit_predict(df)
    closed = bank.closed_groups()
    assert closed["session"].to_list() == ["m", "t"]
    assert closed["rows_fed"].to_list() == [5, 5]
    # The new session's first row is a first row: its clock starts over.
    assert closed["clock_min"].to_list() == [0.0, 5.0]
    # The live gram is the current five-row span. The old assertion compared
    # `weight_sum` with `height and weight_sum`, and `height` (5) is truthy, so it
    # compared the value with itself (review 2026-09-18, minor). Five rows at
    # half-life 40 sum to 4.83.
    assert 4.5 < bank.gram("c")[0]["weight_sum"] < 5.0, bank.gram("c")[0]["weight_sum"]


@pytest.mark.parametrize("size", [1, 2, 4, 15])
def test_a_session_close_is_chunk_invariant(size):
    sess = ["m"] * 4 + ["t"] * 6 + ["w"] * 5
    df = frame(["a", "b"], n_per=15, session=sess * 2)
    want = po.ModelBank([cov_spec(group_close="session", session="s")])
    want.fit_predict(df)
    got = po.ModelBank([cov_spec(group_close="session", session="s")])
    for i in range(0, df.height, size):
        got.fit_predict(df[i : i + size])
    assert want.closed_groups().equals(got.closed_groups())


def test_a_closing_streams_live_summary_is_the_current_span():
    """The other half of `test_summary.py`'s skip: a session close restarts
    the stream, so `summary()` counts the span it is in, and the spans
    before it are in the closed rows."""
    sess = ["m"] * 5 + ["t"] * 7
    df = frame(["a"], n_per=12, session=sess)
    bank = po.ModelBank([cov_spec(group_close="session", session="s")])
    bank.fit_predict(df)
    live = bank.summary()
    assert live["rows_fed"].to_list() == [7], "the second span"
    assert bank.closed_groups()["rows_fed"].to_list() == [5], "and the first"


def test_a_zero_row_chunk_and_an_unseen_group_close_nothing():
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(frame(["a"]).clear())
    assert bank.closed_groups().height == 0


# --- reading the queue -------------------------------------------------------


def test_peeking_twice_gives_the_same_frame_and_draining_empties_it():
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(frame(["a", "b", "c"]))
    first = bank.closed_groups(drop=False)
    assert first.equals(bank.closed_groups(drop=False))
    assert bank.closed_groups().equals(first)
    assert bank.closed_groups().height == 0


def test_narrowing_to_one_spec_leaves_the_others_queued():
    specs = [cov_spec(group_close="monotone"), cov_spec(group_close="monotone")]
    specs[1]["name"] = "d"
    bank = po.ModelBank(specs)
    bank.fit_predict(frame(["a", "b"]))
    assert bank.closed_groups("c")["spec"].to_list() == ["c"]
    assert bank.closed_groups()["spec"].to_list() == ["d"]


def test_predict_closes_nothing():
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    df = frame(["a", "b"])
    bank.fit_predict(df)
    bank.closed_groups()
    bank.predict(df)
    assert bank.closed_groups().height == 0


def test_building_a_predict_plan_leaves_the_queue_alone():
    """``lf.online.predict(bank)`` read its closed-row schema by draining the
    caller's bank, so the rows fitted and not yet read were gone before the
    plan ran -- from a call documented to leave the bank as it was (review
    2026-09-12, C4)."""
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    df = frame(["a", "b", "c"])
    bank.fit_predict(df)
    before = bank.closed_groups(drop=False).height
    assert before > 0
    plan = df.lazy().online.predict(bank)
    assert bank.closed_groups(drop=False).height == before, "building the plan"
    plan.collect()
    assert bank.closed_groups(drop=False).height == before, "running it"


def test_predict_scores_a_new_session_as_the_row_after_the_close():
    """Under ``group_close = "session"`` a new session's rows are a fresh
    stream's first rows to ``fit_predict``, which restarts at the change.
    ``predict`` scored them with the closed session's fit and ``weight_sum``
    (review 2026-09-12, C19)."""
    sess = ["m"] * 20 + ["t"] * 10
    df = frame(["a"], n_per=30, session=sess)
    spec = cov_spec(group_close="session", session="s", min_weight=5.0)
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(20))
    scored = bank.predict(df.tail(10))["c"].struct.unnest()
    assert scored["weight_sum"].to_list() == [0.0] * 10
    # `withheld_reason` says why the rest is null, so it is not null itself,
    # and a fresh stream's `settled_frac` is 0, a value.
    readiness = ("weight_sum", "settled_frac", "withheld_reason")
    stats = [c for c in scored.columns if c not in readiness]
    assert all(scored[c].null_count() == 10 for c in stats), scored
    fresh = po.ModelBank([spec]).fit_predict(df)["c"].struct.unnest().tail(10)
    assert fresh["weight_sum"][0] == 0.0, "fit_predict restarts at the change"
    assert scored.head(1).equals(fresh.head(1))


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"group_close": "nope"}, 'must be "monotone" or "session"'),
        ({"group_close": "monotone", "group": None}, "needs a group column"),
        ({"group_close": "session"}, "needs a session column"),
        (
            {"group_close": "session", "session": "s", "session_gap": 5.0},
            "two prescriptions for one event",
        ),
        (
            {"group_close": "monotone", "embargo": 5.0},
            "does not work with embargo",
        ),
    ],
)
def test_a_spec_that_cannot_close_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        cov_spec(**kw)


def test_interleaved_keys_are_refused_naming_the_row():
    df = pl.DataFrame({"g": ["a", "b", "a"], "x0": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.0]})
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="row 2: group a after group b"):
        bank.fit_predict(df)
    # A refused chunk leaves the bank as it was.
    assert bank.groups().height == 0
    assert bank.closed_groups().height == 0


def test_a_null_key_is_refused_under_monotone():
    df = pl.DataFrame({"g": ["a", None], "x0": [1.0, 2.0], "x1": [1.0, 2.0]})
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="has a null group"):
        bank.fit_predict(df)


def test_a_null_key_is_an_ordinary_group_under_session():
    df = pl.DataFrame(
        {
            "g": ["a", None, None],
            "s": ["m", "m", "t"],
            "x0": [1.0, 2.0, 3.0],
            "x1": [1.0, 2.0, 3.0],
        }
    )
    bank = po.ModelBank([cov_spec(group_close="session", session="s")])
    bank.fit_predict(df)
    assert bank.closed_groups()["group"].to_list() == [None]


def test_a_key_dtype_that_cannot_be_ordered_is_refused():
    df = pl.DataFrame({"g": [1.5, 2.5], "x0": [1.0, 2.0], "x1": [1.0, 2.0]})
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="needs a group column it can order"):
        bank.fit_predict(df)


def test_predict_refuses_a_key_dtype_that_cannot_be_ordered_too():
    """The check moved into the chunk adapter with task 86, so ``predict`` meets
    it as ``fit_predict`` does. Until 0.7.0 ``predict`` alone ran on a key the
    spec could never have ordered; a spec invalid for the column is refused
    whichever call reads it (review 2026-09-17, B4)."""
    df = pl.DataFrame({"g": [1.5, 2.5], "x0": [1.0, 2.0], "x1": [1.0, 2.0]})
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="needs a group column it can order"):
        bank.predict(df)


def test_a_key_below_the_high_water_mark_is_refused():
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(frame(["a", "b"]))
    with pytest.raises(ValueError, match="is below b, which this bank has already closed"):
        bank.fit_predict(frame(["a"]))


def test_the_high_water_mark_and_the_queue_survive_a_save_and_load(tmp_path):
    spec = cov_spec(group_close="monotone")
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(["a", "b", "c"]))
    path = tmp_path / "bank.state"
    bank.save(path)
    again = po.ModelBank.load(path, [spec])
    assert again.closed_groups(drop=False).equals(bank.closed_groups(drop=False))
    assert again.groups()["group"].to_list() == ["c"], "the open group came back"
    with pytest.raises(ValueError, match="already closed past"):
        again.fit_predict(frame(["a"]))
    # And it continues where it stopped.
    again.closed_groups()
    again.fit_predict(frame(["d"]))
    assert again.closed_groups()["group"].to_list() == ["c"]


# --- the sidecar -------------------------------------------------------------


def toml_config(tmp_path, **kw):
    """Paths written POSIX-style, as `run_online` writes them: a Windows path's
    backslashes are escapes in a TOML basic string (docs/TESTING.md T-W3b)."""
    lines = [
        f'input = "{(tmp_path / "in.parquet").as_posix()}"',
        f'output = "{(tmp_path / "out.parquet").as_posix()}"',
        *[f'{k} = "{v.as_posix() if hasattr(v, "as_posix") else v}"' for k, v in kw.items()],
        "[[specs]]",
        'name = "c"',
        'features = ["x0", "x1"]',
        'group = "g"',
        'group_close = "monotone"',
        "half_life = 40.0",
        "[specs.model]",
        'type = "ew_cov"',
    ]
    p = tmp_path / "bank.toml"
    p.write_text("\n".join(lines))
    return p


def test_the_sidecar_is_the_drained_frames_and_is_chunk_invariant(tmp_path):
    df = frame(["a", "b", "c", "d"], n_per=9)
    df.write_parquet(tmp_path / "in.parquet")
    side = tmp_path / "closed.parquet"
    df.lazy().online.fit_predict(
        [cov_spec(group_close="monotone")], closed_groups=side, chunk_size=7
    ).collect()
    driver = po.ModelBank([cov_spec(group_close="monotone")])
    frames = []
    for i in range(0, df.height, 7):
        driver.fit_predict(df[i : i + 7])
        frames.append(driver.closed_groups())
    assert pl.read_parquet(side).equals(pl.concat(frames))

    other = tmp_path / "closed2.parquet"
    df.lazy().online.fit_predict(
        [cov_spec(group_close="monotone")], closed_groups=other, chunk_size=2
    ).collect()
    assert pl.read_parquet(other).equals(pl.read_parquet(side))


def _ipc_record_batches(path) -> int:
    """The record batches an Arrow IPC file holds, read from its footer:
    polars' reader merges them, and pyarrow is not a dependency. The footer
    is a flatbuffer ``Footer`` whose field 3, ``recordBatches``, is a vector
    of one ``Block`` per batch."""
    buf = path.read_bytes()
    assert buf[:6] == b"ARROW1" and buf[-6:] == b"ARROW1", "not an Arrow IPC file"
    (size,) = struct.unpack_from("<i", buf, len(buf) - 10)
    footer = buf[len(buf) - 10 - size : len(buf) - 10]
    (table,) = struct.unpack_from("<I", footer, 0)
    vtable = table - struct.unpack_from("<i", footer, table)[0]
    (vtable_size,) = struct.unpack_from("<H", footer, vtable)
    slot = 4 + 2 * 3
    field = struct.unpack_from("<H", footer, vtable + slot)[0] if slot < vtable_size else 0
    if field == 0:
        return 0
    (rel,) = struct.unpack_from("<I", footer, table + field)
    return struct.unpack_from("<I", footer, table + field + rel)[0]


@pytest.mark.parametrize(("chunk_size", "batches"), [(7, 3), (100, 1)])
def test_the_runner_writes_the_sidecar_as_it_goes(tmp_path, chunk_size, batches, online_cli):
    """The runner drains the bank after every chunk and hands each drain to
    the sidecar's writer, so a closed row reaches the file while the run is
    still going instead of waiting in the bank for its end (review
    2026-09-12, P5). The IPC writer keeps each drain as a record batch of its
    own, which shows it: at 7 rows a chunk ``a``, ``b`` and ``c`` close in
    three different chunks and the file holds three batches; in one chunk,
    one. A single drain at the end wrote one batch either way.

    The query path holds its drains to the end instead, so this is the
    command line's behaviour and is tested there."""
    df = frame(["a", "b", "c", "d"], n_per=9)
    df.write_parquet(tmp_path / "in.parquet")
    side = tmp_path / "closed.arrow"
    run_online(
        online_cli,
        tmp_path,
        [cov_spec(group_close="monotone")],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        closed_groups=side,
        chunk_size=chunk_size,
    )
    assert _ipc_record_batches(side) == batches
    driver = po.ModelBank([cov_spec(group_close="monotone")])
    driver.fit_predict(df)
    # No `memory_map=`: polars 2.0 removed the keyword from `read_ipc`, and on
    # 1.x it already defaults to `False` -- so this was passing the default
    # explicitly and dropping it changes nothing on either version. Passing it
    # made the advisory next-major leg of `release.yml` red on 2.0.0rc1 from
    # 0.6.0 onward, the only two failures in that leg.
    assert pl.read_ipc(side).equals(driver.closed_groups())


def test_the_io_plugin_writes_the_same_sidecar(tmp_path):
    df = frame(["a", "b", "c"], n_per=8)
    side = tmp_path / "closed.parquet"
    out = (
        df.lazy()
        .online.fit_predict([cov_spec(group_close="monotone")], closed_groups=side, chunk_size=5)
        .collect()
    )
    assert out.height == df.height
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(df)
    assert pl.read_parquet(side).equals(bank.closed_groups())


def test_a_run_in_which_nothing_closed_writes_the_empty_schema(tmp_path):
    frame(["a"]).write_parquet(tmp_path / "in.parquet")
    side = tmp_path / "closed.parquet"
    frame(["a"]).lazy().online.fit_predict(
        [cov_spec(group_close="monotone")], closed_groups=side
    ).collect()
    got = pl.read_parquet(side)
    assert got.height == 0
    empty = po.ModelBank([cov_spec(group_close="monotone")]).closed_groups()
    assert dict(got.schema) == dict(empty.schema)


def test_closed_groups_is_refused_where_nothing_closes_and_with_predict(tmp_path, online_cli):
    frame(["a", "b"]).write_parquet(tmp_path / "in.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [cov_spec()],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        closed_groups=tmp_path / "closed.parquet",
        check=False,
    )
    assert res.returncode != 0 and "no spec closes groups" in res.stderr, res.stderr
    with pytest.raises(ValueError, match="no spec closes groups"):
        frame(["a"]).lazy().online.fit_predict([cov_spec()], closed_groups=tmp_path / "c2.parquet")
    state = tmp_path / "b.state"
    po.ModelBank([cov_spec(group_close="monotone")]).save(state)
    res = run_online(
        online_cli,
        tmp_path,
        [cov_spec(group_close="monotone")],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        load_state=state,
        predict=True,
        closed_groups=tmp_path / "closed.parquet",
        check=False,
    )
    assert res.returncode != 0 and "no group ever closes" in res.stderr, res.stderr


def test_the_cli_writes_the_sidecar(tmp_path, online_cli):
    frame(["a", "b", "c"], n_per=6).write_parquet(tmp_path / "in.parquet")
    cfg = toml_config(tmp_path)
    side = tmp_path / "closed.parquet"
    r = subprocess.run(
        [
            str(online_cli),
            "--config",
            str(cfg),
            "--closed-groups",
            str(side),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "wrote closed groups to" in r.stdout
    assert pl.read_parquet(side)["group"].to_list() == ["a", "b"]


def test_from_row_refuses_a_row_with_no_accumulators():
    spec = po.spec.holt("h", targets=["x0"], half_life=HALFLIFE, group="g", group_close="monotone")
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(["a", "b"]))
    row = bank.closed_groups()
    assert row.height == 1 and row["weight_sum"][0] > 0
    with pytest.raises(ValueError, match="no accumulators"):
        po.gram.from_row(row)


def test_schema_version_is_current():
    """The version a bank file names, held to the library's: 54 since
    2026-10-09 (task 223 (b): a spec may be an `audit`; the bank refuses 53
    and older), after 52 since 2026-10-08 (task 217: `ewridge`'s and
    `lasso`'s cross-moments keep their rows of positive weight, per target
    and over every row; the bank refuses 51 and older), after 51 the same
    day (task 215: `ftrl` keeps no penalty scale, and `kmeans`' and
    `micro`'s centres are pairs), after 50 the same day (task 214: a standardizing `kalman`
    keeps the first three rows' squared innovations, which its prior is the
    median of; the bank refuses 49 and older), after 49 the same day (task 211: `kalman` keeps
    an anchor, its clocks since an observation and its prior's squared
    innovations in place of its warm-up; the bank refuses 48 and older),
    after 48 the same day (task 212: a
    windows state is version 9, a variance's queue six wide and each
    operator's `bias`; the bank refuses 47 and older), after 47 the same day
    (task 206: a standardizing `sgd`, `pa` or `kalman` keeps its scaler's
    warm-up, and past it holds its fit so that the scaler's moving moves no
    prediction; the bank refuses 46 and older), after 46 the same
    day (task 208: a stream keeps the waits its "has not been met" notices
    count; the bank refuses 45 and older), after 45 since
    2026-10-07 (task 116: `rls` keeps the squared-weight sum its noise
    statistic reads Kish's size from; the bank refuses 44 and older), after
    44 the same day (task 202: an insensitivity band in the target's own spread,
    which `pa` and `sgd` keep; the bank refuses 43 and older), after 43 the
    same day (task 201: relative targets were removed, and every target's
    ``hit_rate`` is taken about 0; the bank refuses 42 and older), after 42
    the same day (task 200: an integer clock is held as an integer, in the
    clock state, its stamps, the bank's clock dtypes and the windows state,
    version 8; the bank refuses 41 and older), after 41 the same day (task
    196: a model window's ring keeps its edge, Polars' ``closed``, and the
    specs say ``type = "ewridge"`` and task 196's other names; the bank
    refuses 40 and older), after 40 the same day (task 195: `sgd`'s and
    `pa`'s residual scales, `pa`'s scaler, `bocpd`'s warm-up, the solving
    models' per-target thresholds and `rls`'s `delta`; the bank refuses 39
    and older), after 39 the same day (task 194: a stream's state nests what
    it keeps beside its models, a clock range is kept as clock values, a
    closed pair's statistic is ``t_stat``, and the bank keeps each key
    column's form; the bank refuses 38), after 38 since
    2026-10-06 (task 186: ``ewridge``'s kept systems lost a Gram index
    nothing read, and a closed ``rcov`` row's ``psd_repaired`` can be null;
    the bank refuses 37), after 37 the same day (task 180: the solve,
    component and checkpoint cadences keep
    the stamp of their last event, the decayed clock held exactly, where a
    36 file keeps a summed clock; the bank refuses 36), after 36 the same
    day (task 179: ``bocpd``'s configuration says whether its hazard
    is a duration, which a 35 build would read as a number of rows; the bank
    still loads 35), after 35 the same day (task 178: ``coef_every`` reads
    the clock, its old default ``0`` is every row, and a stream keeps where
    its cadence stands), after 34 the same day (task 176: an ``embargo`` is
    decided on the elapsed clock held exactly, which a stream's clock keeps
    and each held row keeps its place on), after 33 the same day (task 170:
    a quantile fit's state keeps its band systems, the factors its nudges
    read, which a band row at ``ridge = 0`` moves in
    place), after 32 the same day (task 175: a model window's snapshots are
    keyed by each row's stamp, its decayed clock held exactly), after 31 the
    same day (task 174:
    `lasso` keeps each target's own `min_weight` for its selection), after 30
    the same day (task 162: a model window's snapshots are spaced on the
    clock, `window_every` in clock units beside `max_rows_between_snapshots`),
    after 29 the same day (task 163: `micro`'s pruning on the clock) and 28
    (task 161: `ew_cov`'s configuration and state carry its PCA cadence on
    the clock); 27 since
    2026-10-05 (task 159: the windows state carries the group and session
    columns' dtypes, windows state 7, after 6 the same day for a number
    clock's raw value in its rows); 25 since
    2026-10-03 (review R6: the windows state a bank file embeds per formula
    target went to version 5, and the bank's schema moves with it; 24 in
    review R4 the same day, for the same reason); 21 since
    2026-10-02, when the stream's diagnostics moved onto the clock and a
    bank stopped loading any file from before it (task 146); 20 since
    2026-09-28, when the solve cadence went by weight (task 115 (b)); 19 the
    same day, when `on_clock_reset` lost `"max"` and `"zero"` and a bank
    stopped loading any file from before it (task 120); 13 since
    2026-09-21, for the readiness statistics -- `ewridge`'s per-slot
    degrees of freedom and data shares, the stream's decay time and its
    notices (docs/WARMUP-AND-CONVERGENCE.md); 12 since 2026-09-20, when the
    clock state dropped the three fields the two 0.8.x disorder rules kept;
    11 for `robust`'s centred cross-moments (review of
    2026-09-18, S2); 10 since the second review of 2026-09-15, for
    `robust`'s per-target observation weights (F1), after 9 the same day for
    `holt`'s weighted means and `ftrl`'s proximal sum. Pre-1.0, an older
    file is refused by its version."""
    assert po.schema_version() == 54


def test_an_integer_key_used_as_both_session_and_group_orders_numerically():
    """When one integer column is both `session` (cast to text) and `group`
    (a key), `group_close = "monotone"` must still order it numerically, so a
    numerically sorted key spanning single and double digits is not refused as
    out of order at "10 after 9" (review 2026-09-18, V12)."""
    n = 40
    g = np.repeat(np.arange(1, 21), 2).astype(np.int64)  # ..., 9, 9, 10, 10, ...
    df = pl.DataFrame(
        {
            "x0": np.random.default_rng(0).standard_normal(n),
            "y": np.random.default_rng(1).standard_normal(n),
            "k": g,
            "t": np.arange(float(n)),
        }
    )
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        half_life=10.0,
        group="k",
        session="k",
        session_gap=1.0,
        clock="t",
        gap_cap=100.0,
        group_close="monotone",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)  # must not raise "group 10 after group 9"
    keys = [int(k) for k in bank.closed_groups(drop=False)["group"].to_list()]
    assert keys == sorted(keys), keys  # numeric order, not lexical
    assert 9 in keys and 10 in keys and keys.index(9) < keys.index(10), keys
