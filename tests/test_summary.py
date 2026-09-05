"""``ModelBank.summary`` and ``ModelBank.describe`` (docs/PLAN.md task 35):
what each stream has been fed, carried in the state file.

The contract is that the numbers are the frame's -- counted and averaged
over the rows routed to each group with the models' own notion of a usable
value -- for every model in the golden set; that they survive the file
unchanged and a re-save writes the same bytes; that chunking cannot move a
bit of them; and that ``predict`` leaves them alone.
"""

import math

import polars as pl
import pytest

import polars_online as po
from test_golden_pipeline import specs, stream

BOUND = 1e100

SUMMARY_SCHEMA = {
    "spec": pl.String,
    "group": pl.String,
    "rows_fed": pl.UInt64,
    "rows_processed": pl.UInt64,
    "rows_skipped": pl.UInt64,
    "rows_learned": pl.UInt64,
    "rows_zero_weight": pl.UInt64,
    "weight_sum": pl.Float64,
    "clock_min": pl.Float64,
    "clock_max": pl.Float64,
    "last_clock": pl.Float64,
    "session_changes": pl.UInt64,
    "clock_backwards": pl.UInt64,
    "resets": pl.UInt64,
}
DESCRIBE_SCHEMA = {
    "spec": pl.String,
    "group": pl.String,
    "column": pl.String,
    "role": pl.String,
    "count": pl.UInt64,
    "null_count": pl.UInt64,
    "mean": pl.Float64,
    "std": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
}


def feed(bank: po.ModelBank, df: pl.DataFrame, n_chunks: int) -> pl.DataFrame:
    step = -(-df.height // n_chunks)
    return pl.concat([bank.fit_predict(c) for c in df.iter_slices(step)])


def usable(name: str) -> pl.Expr:
    """A value the models take: finite and within the input bound."""
    c = pl.col(name).cast(pl.Float64)
    return c.is_finite() & (c.abs() <= BOUND)


def columns_of(spec: dict) -> list[tuple[str, str]]:
    """The input columns ``describe`` lists for a spec, with their roles."""
    cols = [(f, "feature") for f in spec.get("features", [])]
    model = spec["model"]["type"]
    if model not in ("ew_cov", "kmeans", "micro"):
        cols += [(t, "target") for t in spec.get("targets", [])]
    if spec.get("weight"):
        cols.append((spec["weight"], "weight"))
    return cols


@pytest.fixture(scope="module")
def fitted() -> tuple[pl.DataFrame, po.ModelBank, pl.DataFrame]:
    df = stream(150)
    bank = po.ModelBank(specs())
    out = feed(bank, df, 5)
    return df, bank, out


def test_schema_and_one_row_per_group_for_every_model(fitted):
    _, bank, _ = fitted
    s = bank.summary()
    assert s.schema == SUMMARY_SCHEMA
    assert s.height == 2 * len(specs())
    d = bank.describe()
    assert d.schema == DESCRIBE_SCHEMA
    for spec in specs():
        rows = d.filter(pl.col("spec") == spec["name"])
        want = columns_of(spec)
        assert rows.height == 2 * len(want), spec["name"]
        for g in ("a", "b"):
            got = rows.filter(pl.col("group") == g).select("column", "role").rows()
            assert got == want, f"{spec['name']} / {g}"


def test_summary_is_the_frame_s_count_per_group(fitted):
    df, bank, out = fitted
    s = bank.summary()
    for spec in specs():
        name = spec["name"]
        features = spec.get("features", [])
        weight = spec.get("weight")
        needed = [usable(c) for c in features] + ([usable(weight)] if weight else [])
        accept = pl.all_horizontal(needed) if needed else pl.lit(True)
        if spec["model"]["type"] == "ew_class":
            has_target = pl.col(spec["targets"][0]).is_not_null()
        elif spec["model"]["type"] == "seqtest" and spec["model"].get("a"):
            # The comparison's target is the difference of the two sides'
            # residuals, null where either side did not predict.
            a = out["ridge"].struct.field("resid_y0__r0.5")
            b = out["kalman"].struct.field("resid_y0")
            has_target = pl.lit(a.is_not_null() & b.is_not_null())
        elif spec.get("targets"):
            has_target = pl.any_horizontal([usable(t) for t in spec["targets"]])
        else:
            has_target = pl.lit(True)
        w = pl.col(weight) if weight else pl.lit(1.0)
        want = (
            df.with_columns(accept=accept, has_target=has_target, w=w)
            .group_by("g", maintain_order=True)
            .agg(
                rows_fed=pl.len(),
                rows_processed=pl.col("accept").sum(),
                rows_learned=(pl.col("accept") & (pl.col("w") > 0) & pl.col("has_target")).sum(),
                rows_zero_weight=(pl.col("accept") & (pl.col("w") == 0)).sum(),
                weight_sum=pl.when(pl.col("accept")).then(pl.col("w")).otherwise(0.0).sum(),
                clock_min=pl.col("t").min(),
                clock_max=pl.col("t").max(),
                last_clock=pl.col("t").last(),
            )
            .sort("g")
        )
        got = s.filter(pl.col("spec") == name).sort("group")
        assert got["group"].to_list() == want["g"].to_list()
        for c in ("rows_fed", "rows_processed", "rows_learned", "rows_zero_weight"):
            assert got[c].to_list() == want[c].to_list(), f"{name}: {c}"
        assert (
            got["rows_skipped"].to_list() == (want["rows_fed"] - want["rows_processed"]).to_list()
        )
        for c in ("clock_min", "clock_max", "last_clock"):
            assert got[c].to_list() == want[c].to_list(), f"{name}: {c}"
        for a, b in zip(got["weight_sum"], want["weight_sum"], strict=True):
            assert math.isclose(a, b, rel_tol=1e-12), f"{name}: weight_sum"
        # No session in these specs, and the clock only runs forward.
        assert got["session_changes"].to_list() == [0, 0]
        assert got["clock_backwards"].to_list() == [0, 0]
        assert got["resets"].to_list() == [0, 0]
    # `rows_processed` and `last_clock` are what `groups()` says.
    g = bank.groups().sort("spec", "group")
    assert (
        s.sort("spec", "group")
        .select("rows_processed", "last_clock")
        .equals(g.select("rows_processed", "last_clock"))
    )


def test_describe_is_the_frame_s_statistics_per_column(fitted):
    df, bank, out = fitted
    d = bank.describe()
    for spec in specs():
        name = spec["name"]
        for column, role in columns_of(spec):
            if name == "seqtest_compare" and role == "target":
                a = out["ridge"].struct.field("resid_y0__r0.5")
                b = out["kalman"].struct.field("resid_y0")
                frame = df.with_columns((b.abs() - a.abs()).alias(column))
            elif name == "ew_class" and role == "target":
                # A label is its class index to the model, and NaN otherwise:
                # any finite stand-in gives the same counts.
                frame = df.with_columns(
                    pl.when(pl.col(column).is_in(["lo", "hi"])).then(0.0).alias(column)
                )
            else:
                frame = df
            v = pl.when(usable(column)).then(pl.col(column).cast(pl.Float64))
            want = (
                frame.group_by("g", maintain_order=True)
                .agg(
                    count=v.count(),
                    null_count=pl.len() - v.count(),
                    mean=v.mean(),
                    std=v.std(ddof=1),
                    min=v.min(),
                    max=v.max(),
                )
                .sort("g")
            )
            got = d.filter((pl.col("spec") == name) & (pl.col("column") == column)).sort("group")
            assert got["role"].to_list() == [role, role]
            assert got["count"].to_list() == want["count"].to_list(), f"{name}/{column}"
            assert got["null_count"].to_list() == want["null_count"].to_list(), f"{name}/{column}"
            if name == "ew_class" and role == "target":
                # A label column: counts only.
                assert got.select("mean", "std", "min", "max").null_count().sum_horizontal()[0] == 8
                continue
            for c in ("mean", "std", "min", "max"):
                for a, b in zip(got[c], want[c], strict=True):
                    if a is None or b is None:
                        assert a is None and b is None, f"{name}/{column}: {c}"
                    else:
                        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (
                            f"{name}/{column}: {c}"
                        )


def test_events_zero_weights_sessions_and_backwards_clocks():
    n = 90
    t = [float(i) for i in range(n)]
    for i in range(20, n, 20):
        t[i] = t[i - 1] - 0.5  # a step back, within the session
    sess = ["s0"] * 30 + ["s1"] * 30 + ["s2"] * 30
    w = [0.0 if i % 9 == 4 else 1.0 for i in range(n)]
    df = pl.DataFrame(
        {
            "t": t,
            "sess": sess,
            "x": [math.sin(i) for i in range(n)],
            "y": [math.cos(i) for i in range(n)],
            "w": w,
        }
    )
    reset_on_session = po.spec.ewridge(
        "rs",
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=10.0,
        max_dclock=5.0,
        session="sess",
        session_gap="reset",
        weight="w",
    )
    reset_on_backwards = po.spec.ewridge(
        "rb",
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=10.0,
        max_dclock=5.0,
        on_clock_reset="reset_state",
    )
    bank = po.ModelBank([reset_on_session, reset_on_backwards])
    feed(bank, df, 3)
    s = bank.summary().sort("spec")
    rb, rs = s.rows(named=True)
    assert rs["session_changes"] == 2 and rs["resets"] == 2
    # The step back at row 60 is also the s1 -> s2 boundary: a new session
    # owns its clock, so that one is a session change, not a step back.
    assert rs["clock_backwards"] == 3 and rs["rows_zero_weight"] == 10
    assert rs["rows_learned"] == n - 10 and rs["weight_sum"] == n - 10
    assert rb["session_changes"] == 0 and rb["clock_backwards"] == 4 and rb["resets"] == 4
    assert rb["rows_zero_weight"] == 0 and rb["weight_sum"] == n
    for r in (rs, rb):
        assert r["rows_fed"] == n and r["rows_processed"] == n and r["rows_skipped"] == 0
        assert r["clock_min"] == 0.0 and r["clock_max"] == n - 1 and r["last_clock"] == n - 1


def test_a_row_count_clock_has_no_range():
    df = stream(40)
    spec = po.spec.ewridge("m", targets=["y0"], features=["x0"], halflife=10.0, group="g")
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    s = bank.summary()
    assert s["clock_min"].null_count() == 2
    assert s["clock_max"].null_count() == 2
    assert s["last_clock"].null_count() == 2
    assert s["rows_fed"].to_list() == [20, 20]


def test_it_travels_with_the_state_file_and_a_re_save_is_the_same_bytes(fitted, tmp_path):
    _, bank, _ = fitted
    s, d = bank.summary(), bank.describe()
    raw = bank.save_bytes()
    loaded = po.ModelBank.load_bytes(raw)
    assert loaded.summary().equals(s)
    assert loaded.describe().equals(d)
    assert loaded.save_bytes() == raw, "a loaded bank saves to the same bytes"
    bank.save(tmp_path / "bank.bin")
    from_file = po.ModelBank.load(tmp_path / "bank.bin")
    assert from_file.summary().equals(s)
    assert from_file.describe().equals(d)
    assert (tmp_path / "bank.bin").read_bytes() == raw


def test_chunking_cannot_move_a_bit(fitted):
    _, bank, _ = fitted
    df = stream(150)
    for n_chunks in (1, 150):
        other = po.ModelBank(specs())
        feed(other, df, n_chunks)
        # `equals` compares values; the bits too, for the floats.
        assert other.summary().equals(bank.summary()), n_chunks
        assert other.describe().equals(bank.describe()), n_chunks
        for frame, mine in ((other.summary(), bank.summary()), (other.describe(), bank.describe())):
            for c in frame.columns:
                if frame[c].dtype == pl.Float64:
                    a = frame[c].fill_null(float("nan")).to_numpy().view("uint64").tolist()
                    b = mine[c].fill_null(float("nan")).to_numpy().view("uint64").tolist()
                    assert a == b, f"{c} moved a bit between chunkings"


def test_predict_and_scoring_leave_it_alone(fitted):
    df, bank, _ = fitted
    s, d = bank.summary(), bank.describe()
    bank.predict(df.head(50))
    assert bank.summary().equals(s)
    assert bank.describe().equals(d)


def test_narrowing_errors_and_the_empty_cases(fitted):
    _, bank, _ = fitted
    ridge = bank.summary("ridge")
    assert ridge.equals(bank.summary().filter(pl.col("spec") == "ridge"))
    assert bank.summary(0).equals(ridge)
    assert bank.summary("ridge", group="b").equals(ridge.filter(pl.col("group") == "b"))
    assert bank.describe("ridge", group="b").equals(
        bank.describe().filter((pl.col("spec") == "ridge") & (pl.col("group") == "b"))
    )
    with pytest.raises(KeyError, match="no spec named 'nope'"):
        bank.summary("nope")
    with pytest.raises(IndexError, match="out of range"):
        bank.describe(len(specs()))
    none = bank.summary("kalman", group="zzz")
    assert none.height == 0 and none.schema == SUMMARY_SCHEMA
    none = bank.describe("kalman", group="zzz")
    assert none.height == 0 and none.schema == DESCRIBE_SCHEMA
    fresh = po.ModelBank(specs())
    assert fresh.summary().height == 0 and fresh.summary().schema == SUMMARY_SCHEMA
    assert fresh.describe().height == 0 and fresh.describe().schema == DESCRIBE_SCHEMA


def test_dropped_groups_start_their_count_over():
    df = stream(60)
    bank = po.ModelBank(specs()[:1])
    bank.fit_predict(df)
    before = bank.summary()
    assert before["rows_fed"].to_list() == [30, 30]
    bank.drop_groups(["a"])
    assert bank.summary()["group"].to_list() == ["b"]
    bank.fit_predict(df.filter(pl.col("g") == "a").head(7))
    after = bank.summary().sort("group")
    assert after["rows_fed"].to_list() == [7, 30]
    assert after.filter(pl.col("group") == "b").equals(before.filter(pl.col("group") == "b"))
