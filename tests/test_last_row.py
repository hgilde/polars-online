"""``ModelBank.last_row`` (docs/PLAN.md task 34): the output struct on the last
row each stream learned from, carried in the state file so that fitted models
compare from their files alone.

The contract is that the row *is* the output frame's row for that stream --
every field, the bit included, for every model -- that it survives the file,
that ``predict`` leaves it alone, and that a chunk ending in skipped rows
leaves the row before them.
"""

import polars as pl
import pytest

import polars_online as po
from test_golden_pipeline import specs, stream


def feed(bank: po.ModelBank, df: pl.DataFrame, n_chunks: int) -> pl.DataFrame:
    step = -(-df.height // n_chunks)
    return pl.concat([bank.fit_predict(c) for c in df.iter_slices(step)])


def last_learned(df: pl.DataFrame, out: pl.DataFrame, name: str, group: str) -> int | None:
    """The last row of `group` that spec `name` learned from, if any: a
    skipped row has every field of its struct null, a learned one has not."""
    fields = out[name].struct.unnest()
    learned = fields.select(pl.any_horizontal(pl.all().is_not_null())).to_series()
    rows = [i for i in range(df.height) if df["g"][i] == group and learned[i]]
    return rows[-1] if rows else None


def frame_row(out: pl.DataFrame, name: str, i: int) -> pl.DataFrame:
    return out[name].slice(i, 1).struct.unnest()


def bank_row(table: pl.DataFrame, name: str, group: str, columns: list[str]) -> pl.DataFrame:
    return (
        table.filter((pl.col("spec") == name) & (pl.col("group") == group))
        .drop("spec", "group")
        .select(columns)
    )


def without_coef(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if not c.startswith("coef")])


@pytest.fixture(scope="module")
def fitted() -> tuple[pl.DataFrame, po.ModelBank, pl.DataFrame]:
    df = stream(120)
    bank = po.ModelBank(specs())
    out = feed(bank, df, 4)
    return df, bank, out


def test_it_is_the_frame_s_last_learned_row_for_every_model(fitted):
    df, bank, out = fitted
    table = bank.last_row()
    assert table.columns[:2] == ["spec", "group"]
    assert table.height == 2 * len(specs()), "one row per (spec, group)"
    for spec in specs():
        name = spec["name"]
        for g in ("a", "b"):
            i = last_learned(df, out, name, g)
            assert i is not None
            want = frame_row(out, name, i)
            got = bank_row(table, name, g, want.columns)
            # `equals` is exact: same values, same nulls, same dtypes.
            assert want.equals(got), f"{name} / {g}:\n{want}\n{got}"
    # Narrowed to one spec, by name or position, and to one group.
    ridge = bank.last_row("ridge")
    assert ridge.equals(table.filter(pl.col("spec") == "ridge").select(ridge.columns))
    assert ridge.columns == ["spec", "group", *po.spec.output_fields(specs()[0])]
    assert bank.last_row(0).equals(ridge)
    assert bank.last_row("ridge", group="b").equals(ridge.filter(pl.col("group") == "b"))


def test_it_travels_with_the_state_file(fitted, tmp_path):
    _, bank, _ = fitted
    table = bank.last_row()
    assert po.ModelBank.load_bytes(bank.save_bytes()).last_row().equals(table)
    bank.save(tmp_path / "bank.bin")
    assert po.ModelBank.load(tmp_path / "bank.bin").last_row().equals(table)


def test_chunking_changes_nothing_but_coef(fitted):
    """`coef` rides on a chunk's last row (docs/PLAN.md §3), so whether a
    group's last learned row carries it is the chunking's to decide -- in the
    frame and here alike. Everything else is chunk-invariant."""
    df, bank, _ = fitted
    one = po.ModelBank(specs())
    feed(one, df, 1)
    a, b = without_coef(one.last_row()), without_coef(bank.last_row())
    assert a.equals(b)


def test_predict_leaves_it_and_skipped_rows_step_back():
    df = stream(120)
    bank = po.ModelBank(specs())
    out = feed(bank, df.head(90), 3)
    before = bank.last_row()

    bank.predict(df.slice(90, 30))
    assert bank.last_row().equals(before)

    # A chunk in which every row of group "a" has a null feature and a null
    # weight: a spec that reads either skips them all, so "a" keeps its row,
    # while "b" moves on to the chunk's last learned row. A spec that reads
    # neither (`seqtest`: a target, no weight) learns from them, and its "a"
    # moves on too -- the frame says which, and the row is the frame's.
    tail = df.slice(90, 30).with_columns(
        pl.when(pl.col("g") == "a").then(None).otherwise(pl.col(c)).alias(c) for c in ("x1", "w")
    )
    out2 = bank.fit_predict(tail)
    after = bank.last_row()
    kept = 0
    for spec in specs():
        name = spec["name"]
        for g in ("a", "b"):
            i = last_learned(tail, out2, name, g)
            if i is None:
                assert g == "a", f"{name}: group b has learnable rows"
                cols = po.spec.output_fields(spec)
                want = bank_row(before, name, g, cols)
                kept += 1
            else:
                want = frame_row(out2, name, i)
            assert want.equals(bank_row(after, name, g, want.columns)), f"{name} / {g}"
    assert kept == len(specs()) - 2, "every spec but the two seqtests skipped a's rows"
    # The first frame is untouched by any of this, as it should be.
    assert out.height == 90


def test_a_group_with_no_learned_row_is_a_row_of_nulls():
    spec = specs()[0]
    bank = po.ModelBank([spec])
    df = stream(20).with_columns(pl.lit(None, pl.Float64).alias("x1"))
    bank.fit_predict(df)
    table = bank.last_row()
    assert table["group"].to_list() == ["a", "b"]
    assert table.drop("spec", "group").null_count().sum_horizontal().item() == 2 * (table.width - 2)
    # And a bank that has seen nothing has the columns and no rows.
    empty = po.ModelBank([spec]).last_row()
    assert empty.height == 0
    assert empty.columns == ["spec", "group", *po.spec.output_fields(spec)]


def test_a_spec_the_bank_has_not_got_and_a_group_it_has_not_seen(fitted):
    _, bank, _ = fitted
    with pytest.raises(KeyError, match="no spec named 'nope'"):
        bank.last_row("nope")
    with pytest.raises(IndexError, match="out of range"):
        bank.last_row(len(specs()))
    none = bank.last_row("kalman", group="zzz")
    assert none.height == 0
    assert none.columns == ["spec", "group", *po.spec.output_fields(specs()[2])]
