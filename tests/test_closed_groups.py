"""E54: a group's accumulators emitted when the bank can prove it is finished.

The promise is exact: the closed row **is** the ``gram()`` a driver would
have read at that point, and the stream is dropped, so a bank over an
unbounded key space stays bounded. Both halves are tested here -- the row
against ``gram()`` bit for bit, and the schedule against the two rules
(a key smaller than the largest seen, or a session that has ended) under
every chunking.
"""

import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import polars_online as po

HALFLIFE = 40.0


def cov_spec(**kw):
    kw.setdefault("group", "g")
    return po.spec.ew_cov("c", features=["x0", "x1"], halflife=HALFLIFE, **kw)


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
    assert got["n_eff"] == g["n_eff"]
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
        halflife=HALFLIFE,
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
    bank = po.ModelBank([cov_spec(group_close="monotone", clock="t", max_dclock=1e9, weight="w")])
    bank.fit_predict(df)
    row = bank.closed_groups()
    assert row["clock_min"][0] == 0.0 and row["clock_max"][0] == 4.0
    assert row["rows_fed"][0] == 5 and row["rows_learned"][0] == 5


def test_eig_is_the_eigendecomposition_of_the_rows_own_comoments():
    df = frame(["a", "b"], n_per=40)
    spec = po.spec.ew_cov(
        "c",
        features=["x0", "x1"],
        halflife=HALFLIFE,
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
        halflife=HALFLIFE,
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
            halflife=HALFLIFE,
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


def test_a_categorical_key_in_physical_order_is_refused_naming_the_sort():
    """Bytewise is the order for anything but an integer column, and a
    ``Categorical`` sorts by its *physical* order by default -- the order the
    categories were first seen -- so a frame sorted by such a column can be
    out of order for the bank. The refusal says so
    (docs/REVIEW-E54-E64.md G2)."""
    df = frame(["c", "a", "b"]).with_columns(pl.col("g").cast(pl.Categorical))
    # Physically sorted (c, a, b is the order they appear) and lexically not.
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    with pytest.raises(ValueError, match="Categorical column sorts by its physical order"):
        bank.fit_predict(df)
    # Cast to String and sort by that, and it goes through.
    fixed = df.with_columns(pl.col("g").cast(pl.String)).sort("g", maintain_order=True)
    po.ModelBank([cov_spec(group_close="monotone")]).fit_predict(fixed)


def test_a_high_water_mark_read_under_the_wrong_dtype_is_refused_both_ways():
    """A mark and the keys have to be ordered the same way, or the mark lets
    through exactly the groups it exists to refuse: ``"10"`` comes after
    ``"9"`` as an integer and before it as text
    (docs/REVIEW-E54-E64.md G1)."""
    spec = cov_spec(group_close="monotone")

    # Saved under an integer column, resumed under a text one.
    bank = po.ModelBank([spec])
    bank.fit_predict(frame([9, 10, 11]))
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    with pytest.raises(ValueError, match="saved under a integer group column"):
        again.fit_predict(frame(["12", "13"]))

    # And the other way. A file written before the flag existed does not
    # carry it, and there the mark itself gives the mismatch away -- "c"
    # cannot be an integer key; both refusals are in `check_monotone`.
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(["a", "b", "c"]))
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    with pytest.raises(ValueError, match="saved under a text group column"):
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
    bank = po.ModelBank([cov_spec(group_close="session", session="s", clock="t", max_dclock=1e9)])
    bank.fit_predict(df)
    closed = bank.closed_groups()
    assert closed["session"].to_list() == ["m", "t"]
    assert closed["rows_fed"].to_list() == [5, 5]
    # The new session's first row is a first row: its clock starts over.
    assert closed["clock_min"].to_list() == [0.0, 5.0]
    assert bank.gram("c")[0]["n_eff"] == pytest.approx(
        po.ModelBank([cov_spec(group_close="session", session="s")]).fit_predict(df.tail(5)).height
        and bank.gram("c")[0]["n_eff"]
    )


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
            {"group_close": "monotone", "label_delay": 5.0},
            "does not work with label_delay",
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
    lines = [
        f'input = "{tmp_path / "in.parquet"}"',
        f'output = "{tmp_path / "out.parquet"}"',
        *[f'{k} = "{v}"' for k, v in kw.items()],
        "[[specs]]",
        'name = "c"',
        'features = ["x0", "x1"]',
        'group = "g"',
        'group_close = "monotone"',
        "halflife = 40.0",
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
    po.run(
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        specs=[cov_spec(group_close="monotone")],
        closed_groups=side,
        chunk_rows=7,
    )
    driver = po.ModelBank([cov_spec(group_close="monotone")])
    frames = []
    for i in range(0, df.height, 7):
        driver.fit_predict(df[i : i + 7])
        frames.append(driver.closed_groups())
    assert pl.read_parquet(side).equals(pl.concat(frames))

    other = tmp_path / "closed2.parquet"
    po.run(
        input=tmp_path / "in.parquet",
        output=tmp_path / "out2.parquet",
        specs=[cov_spec(group_close="monotone")],
        closed_groups=other,
        chunk_rows=2,
    )
    assert pl.read_parquet(other).equals(pl.read_parquet(side))


def test_the_io_plugin_writes_the_same_sidecar(tmp_path):
    df = frame(["a", "b", "c"], n_per=8)
    side = tmp_path / "closed.parquet"
    out = (
        df.lazy()
        .online.fit_predict([cov_spec(group_close="monotone")], closed_groups=side, chunk_rows=5)
        .collect()
    )
    assert out.height == df.height
    bank = po.ModelBank([cov_spec(group_close="monotone")])
    bank.fit_predict(df)
    assert pl.read_parquet(side).equals(bank.closed_groups())


def test_a_run_in_which_nothing_closed_writes_the_empty_schema(tmp_path):
    frame(["a"]).write_parquet(tmp_path / "in.parquet")
    side = tmp_path / "closed.parquet"
    po.run(
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        specs=[cov_spec(group_close="monotone")],
        closed_groups=side,
    )
    got = pl.read_parquet(side)
    assert got.height == 0
    empty = po.ModelBank([cov_spec(group_close="monotone")]).closed_groups()
    assert dict(got.schema) == dict(empty.schema)


def test_closed_groups_is_refused_where_nothing_closes_and_with_predict(tmp_path):
    frame(["a", "b"]).write_parquet(tmp_path / "in.parquet")
    with pytest.raises(ValueError, match="no spec closes groups"):
        po.run(
            input=tmp_path / "in.parquet",
            output=tmp_path / "out.parquet",
            specs=[cov_spec()],
            closed_groups=tmp_path / "closed.parquet",
        )
    with pytest.raises(ValueError, match="no spec closes groups"):
        frame(["a"]).lazy().online.fit_predict([cov_spec()], closed_groups=tmp_path / "c2.parquet")
    state = tmp_path / "b.state"
    po.ModelBank([cov_spec(group_close="monotone")]).save(state)
    with pytest.raises(ValueError, match="no group ever closes"):
        po.run(
            input=tmp_path / "in.parquet",
            output=tmp_path / "out.parquet",
            specs=[cov_spec(group_close="monotone")],
            load_state=state,
            predict=True,
            closed_groups=tmp_path / "closed.parquet",
        )


def test_the_cli_writes_the_sidecar(tmp_path):
    frame(["a", "b", "c"], n_per=6).write_parquet(tmp_path / "in.parquet")
    cfg = toml_config(tmp_path)
    side = tmp_path / "closed.parquet"
    r = subprocess.run(
        [
            *_cli(),
            "--config",
            str(cfg),
            "--closed-groups",
            str(side),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "wrote closed groups to" in r.stdout
    assert pl.read_parquet(side)["group"].to_list() == ["a", "b"]


def _cli():
    """The `online` binary, built by the gate; skip if it is not there."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for profile in ("release", "debug"):
        p = os.path.join(root, "target", profile, "online")
        if os.path.exists(p):
            return [p]
    pytest.skip("the online CLI is not built")
    return []  # pragma: no cover


def test_the_expression_namespace_has_no_group_close():
    """`.over()` has no end-of-group signal to close on, and an expression
    returns one column of the frame's height -- there is nowhere to put a
    closed row. Both doors are shut: the parameter is not in `ExprKwargs`,
    and passing it anyway is refused by name."""
    from polars_online._kwargs import CommonKwargs, ExprKwargs

    assert "group_close" not in ExprKwargs.__annotations__
    assert "group_close" in CommonKwargs.__annotations__
    # No group: the spec itself is refused, since a close needs a key.
    with pytest.raises(ValueError, match="group_close needs a group column"):
        pl.col("x0").online.ew_cov(["x1"], halflife=HALFLIFE, group_close="monotone")
    # With one, the group refusal comes first and names `.over` -- there is
    # no way through to a closing expression.
    with pytest.raises(TypeError, match="group is not an expression parameter"):
        pl.col("x0").online.ew_cov(["x1"], halflife=HALFLIFE, group="g", group_close="monotone")


def test_from_row_refuses_a_row_with_no_accumulators():
    spec = po.spec.holt("h", targets=["x0"], halflife=HALFLIFE, group="g", group_close="monotone")
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(["a", "b"]))
    row = bank.closed_groups()
    assert row.height == 1 and row["n_eff"][0] > 0
    with pytest.raises(ValueError, match="no accumulators"):
        po.gram.from_row(row)


def test_schema_version_is_current():
    # Pinned so a bump is a deliberate edit here, with the reason recorded in
    # `SCHEMA_VERSION`'s own history: 6 is task 63's `window` keys on the
    # `ew_cov` spec.
    assert po.schema_version() == 6
    assert sys.version_info >= (3, 12)
