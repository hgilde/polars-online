"""Relative targets were removed (docs/PLAN.md task 201): a target derived
from its own row's columns is a column, computed by Polars.

Task 107a's ``po.target("p", relative_to="mid", relative=...)`` learned ``p``
taken against ``mid`` at the same row. Polars' ``with_columns`` computes the
same target, in the columns' own type, and before the removal the bank
learned it to the bit: every field of ten models, under each of the three
forms, with ``coef`` on every row, the metrics and a conformal interval, the
ratio's ``hit_rate`` aside, whose centre was 1 (task 201's evidence, run on
the commit before it). What remains true is held here: the refusals on every
surface, each naming what replaces it, and the ``with_columns`` form through
each way in -- a bank, a lazy plan over a file, a saved and resumed bank, and
the command line reading a column made upstream. A window looking ahead less
a column of the row stays a formula target (``test_formula_targets.py``).
"""

from __future__ import annotations

import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: The return a target table took three ways, as the columns Polars makes.
HOW = {
    "difference": pl.col("p") - pl.col("mid"),
    "ratio": pl.col("p") / pl.col("mid"),
    "log_ratio": (pl.col("p") / pl.col("mid")).log(),
}

#: What every surface's refusal says.
REMOVED = "relative targets were removed"


def frame(n: int = 600, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    mid = 100.0 + np.cumsum(0.2 * rng.standard_normal(n))
    p = mid * np.exp(0.01 * (0.5 * x[:, 0] - 0.3 * x[:, 1]) + 0.002 * rng.standard_normal(n))
    return pl.DataFrame(
        {"t": np.arange(n, dtype=float), "x0": x[:, 0], "x1": x[:, 1], "mid": mid, "p": p}
    )


def common(**kw):
    d = dict(features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=80.0, emit_sigma=True)
    d.update(kw)
    return d


# --- The refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"relative_to": "mid"},
        {"relative_to": "mid", "relative": "log_ratio"},
        {"relative": "ratio"},
        {"relative_to": "mid", "name": "ret"},
    ],
    ids=["relative_to", "both", "relative", "named"],
)
def test_po_target_refuses_a_relative_target_by_name(kw):
    """``po.target`` refuses either keyword, naming ``with_columns`` and the
    log ratio a return should be."""
    with pytest.raises(TypeError, match=REMOVED) as err:
        po.target("p", **kw)
    msg = str(err.value)
    assert 'lf.with_columns(ret=pl.col("p") - pl.col("mid"))' in msg, msg
    assert '(pl.col("p") / pl.col("mid")).log()' in msg, msg
    assert "targets" in msg, msg


def test_a_builder_refuses_a_relative_table_by_name():
    """A table written by hand into a builder's ``targets``: refused by name,
    where the table's shape check would have said only that ``targets`` is
    not a list of targets."""
    for table in [
        {"column": "p", "relative_to": "mid"},
        {"column": "p", "relative": "ratio", "name": "r"},
    ]:
        with pytest.raises(TypeError, match=f'spec "m": target "p": {REMOVED}'):
            po.spec.ewridge("m", targets=["x1", table], features=["x0"], half_life=10.0)


@pytest.mark.parametrize(
    "table",
    [
        {"column": "p", "relative_to": "mid"},
        {"column": "p", "relative_to": "mid", "relative": "log_ratio", "name": "ret"},
        {"column": "p", "relative": "ratio"},
    ],
)
def test_a_spec_dict_refuses_a_relative_table_by_name(table):
    """A spec dict, read by the Rust side as the command line's TOML and a
    saved state are: refused by name, saying to derive the column upstream.
    ``po.eval`` reads its ``spec=`` through the same parser, so a spec
    holding one is refused there too, rather than scored against the raw
    column."""
    base = po.spec.ewridge("m", targets=["p"], **common())
    spec = {**base, "targets": [table]}
    with pytest.raises(ValueError, match=f"{REMOVED}; derive the target column upstream"):
        po.ModelBank([spec])
    out = po.ModelBank([base]).fit_predict(frame())
    with pytest.raises(ValueError, match=REMOVED):
        po.eval.metrics(out, "m", spec=spec)


def test_the_cli_refuses_a_relative_table_by_name(tmp_path, online_cli):
    src, dst, cfg = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "bank.toml"
    frame().write_parquet(src)
    cfg.write_text(
        "\n".join(
            [
                f'input = "{src.as_posix()}"',
                f'output = "{dst.as_posix()}"',
                "[[specs]]",
                'name = "m"',
                'features = ["x0"]',
                'targets = ["x1", { column = "p", relative_to = "mid", relative = "log_ratio" }]',
                'clock = "t"',
                "gap_cap = 5.0",
                "half_life = 80.0",
                "[specs.model]",
                'type = "ewridge"',
            ]
        )
    )
    res = subprocess.run(
        [str(online_cli), "--config", str(cfg)], capture_output=True, text=True, check=False
    )
    assert res.returncode != 0
    assert f"{REMOVED}; derive the target column upstream" in res.stderr, res.stderr
    assert not dst.exists()


# --- What replaces them: the target as a column ------------------------------


@pytest.mark.parametrize("model", ["ewridge", "kalman", "holt"])
def test_a_target_made_by_with_columns_runs_the_same_on_every_way_in(tmp_path, model):
    """The log ratio as a column, made in the query over a file: the lazy plan
    is the bank fed the frame Polars computed, to the bit (the README's
    form)."""
    path = tmp_path / "quotes.parquet"
    frame().write_parquet(path)
    kw = {"coef_half_life": 200.0} if model == "kalman" else {}
    if model == "holt":
        kw["features"] = []
    spec = getattr(po.spec, model)("m", targets=["ret"], **common(**kw))
    lazy = pl.scan_parquet(path).with_columns(ret=HOW["log_ratio"])
    got = lazy.online.fit_predict([spec]).collect()
    want = po.ModelBank([spec]).fit_predict(
        pl.read_parquet(path).with_columns(ret=HOW["log_ratio"])
    )
    assert got.equals(want, null_equal=True)
    assert got["m"].struct.field("pred_ret").drop_nulls().len() > 500


def test_hit_rate_is_about_zero_so_a_return_is_a_difference_or_a_log_ratio():
    """Every target's hit test is about 0, a column made by Polars as much as
    any. A log ratio and a difference sit about 0, so their ``hit_rate`` is a
    rate; a plain ratio sits about 1, where two positive numbers always agree,
    so its ``hit_rate`` reads 1.0 whatever the fit -- which is why the README
    recommends the first two for a return."""
    df = frame().with_columns(**HOW)
    spec = po.spec.ewridge("m", targets=list(HOW), **common(emit_metrics=True))
    out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    for t in ("difference", "log_ratio"):
        assert 0.3 < out[f"hit_rate_{t}"][-1] < 0.95, (t, out[f"hit_rate_{t}"][-1])
    assert out["hit_rate_ratio"].drop_nulls().unique().to_list() == [1.0]


def test_a_saved_bank_resumes_on_a_column_made_upstream(tmp_path):
    """Across a save and load, the same rows as one run, every field, with
    ``coef`` on every row so a chunk boundary adds none."""
    df = frame().with_columns(ret=HOW["log_ratio"])
    spec = po.spec.ewridge("m", targets=["ret", "x1"], **common(features=["x0"], coef_every=0))
    whole = po.ModelBank([spec]).fit_predict(df)
    bank = po.ModelBank([spec])
    first = bank.fit_predict(df[:250])
    bank.save(tmp_path / "r.state")
    second = po.ModelBank.load(tmp_path / "r.state").fit_predict(df[250:])
    assert pl.concat([first, second]).equals(whole, null_equal=True)


def test_the_cli_reads_a_target_made_upstream(tmp_path, online_cli):
    """The command line has no ``with_columns``: the column is made before
    the file it reads, and its output is the Python bank's, to the bit."""
    df = frame().with_columns(ret=HOW["log_ratio"])
    src, dst, cfg = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "bank.toml"
    df.write_parquet(src)
    cfg.write_text(
        "\n".join(
            [
                f'input = "{src.as_posix()}"',
                f'output = "{dst.as_posix()}"',
                "[[specs]]",
                'name = "m"',
                'features = ["x0"]',
                'targets = ["x1", "ret"]',
                'clock = "t"',
                "gap_cap = 5.0",
                "half_life = 80.0",
                "[specs.model]",
                'type = "ewridge"',
            ]
        )
    )
    subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
    spec = po.spec.ewridge("m", targets=["x1", "ret"], **common(emit_sigma=False, features=["x0"]))
    want = po.ModelBank([spec]).fit_predict(df)
    got = pl.read_parquet(dst)
    assert got["m"].struct.fields == want["m"].struct.fields
    assert got["m"].equals(want["m"], null_equal=True)


# --- What po.target keeps: a column under a name of its own -------------------


def test_po_target_names_a_column():
    """``po.target(column, name=)`` stays: the column read as it is, its
    output fields named apart from it -- the same numbers as the column under
    its own name."""
    df = frame()
    renamed = po.ModelBank(
        [po.spec.ewridge("m", targets=[po.target("p", name="price")], **common())]
    )
    plain = po.ModelBank([po.spec.ewridge("m", targets=["p"], **common())])
    a = renamed.fit_predict(df)["m"].struct.unnest()
    b = plain.fit_predict(df)["m"].struct.unnest()
    assert "pred_price" in a.columns
    assert a.rename({c: c.replace("_price", "_p") for c in a.columns}).equals(b, null_equal=True)
    assert po.target("p") == {"column": "p"}
    assert po.target("p", name="price") == {"column": "p", "name": "price"}


def test_a_renamed_column_is_the_name_everywhere():
    """The output fields, the Gram's targets, ``describe``'s rows and
    ``summary``'s counts carry the name; a table naming its own column is the
    column, written back as the string (review 2026-09-26, F6)."""
    i = pl.int_range(pl.len())
    df = frame().with_columns(p=pl.when(i < 5).then(None).otherwise(pl.col("p")))
    bank = po.ModelBank([po.spec.ewridge("m", targets=[po.target("p", name="ret")], **common())])
    out = bank.fit_predict(df)
    assert "pred_ret" in out["m"].struct.fields
    assert bank.gram("m")[0]["targets"] == ["ret"]
    desc = bank.describe("m").filter(pl.col("role") == "target")
    assert desc["column"].to_list() == ["ret"]
    assert desc["null_count"].to_list() == [5]
    assert bank.summary("m")["rows_learned"].item() == len(df) - 5
    assert bank.specs[0]["targets"] == [{"column": "p", "name": "ret"}]
    plain = po.ModelBank([po.spec.ewridge("m", targets=[po.target("p", name="p")], **common())])
    assert plain.specs[0]["targets"] == ["p"]


def test_the_lazy_plan_reads_a_renamed_columns_column():
    """The IO plugin's projection keeps a table target's column (review
    2026-09-26, D1/F1: every LazyFrame path raised ``unhashable type:
    'dict'``)."""
    df = frame()
    spec = po.spec.ewridge("m", targets=[po.target("p", name="price")], **common())
    want = po.ModelBank([spec]).fit_predict(df)["m"]
    got = df.lazy().online.fit_predict([spec]).select("m").collect()["m"]
    assert got.equals(want, null_equal=True)
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:400])
    want_scores = bank.predict(df[400:])["m"]
    scored = df[400:].lazy().online.predict(bank).select("m").collect()["m"]
    assert scored.equals(want_scores, null_equal=True)


def test_a_target_cannot_be_named_nothing():
    """An empty name would give fields called ``pred_``; an empty column names
    no column. Refused by the builder and by the bank (review 2026-09-26, D
    missing 5)."""
    with pytest.raises(ValueError, match="column must not be empty"):
        po.target("")
    with pytest.raises(ValueError, match="name must not be empty"):
        po.target("p", name="")
    with pytest.raises(TypeError, match="column must be a str"):
        po.target(3)  # type: ignore[arg-type]
    base = po.spec.ewridge("m", targets=["p"], **common())
    for table, what in [
        ({"column": "p", "name": ""}, "name"),
        ({"column": "", "name": "p"}, "column"),
    ]:
        with pytest.raises(ValueError, match=f"{what} must not be empty"):
            po.ModelBank([{**base, "targets": [table]}])


def test_a_target_named_like_a_feature_is_not_a_leak():
    """The leak is a feature that *is* a target's column; a target merely
    named like a feature reads another column and builds (review 2026-09-26,
    D7), and a renamed target's column used as a feature is still refused."""
    df = frame()
    spec = po.spec.ewridge("m", targets=[po.target("p", name="x0")], **common())
    out = po.ModelBank([spec]).fit_predict(df)
    assert "pred_x0" in out["m"].struct.fields
    with pytest.raises(ValueError, match='"p" is both a target and a feature'):
        po.ModelBank(
            [po.spec.ewridge("m", targets=[po.target("p", name="ret")], features=["x0", "p"])]
        )
