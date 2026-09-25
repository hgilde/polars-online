"""Relative targets for plain columns (docs/PLAN.md task 107a).

``po.target("p", relative_to="mid")`` has a model learn and predict ``p``
taken against ``mid``, read at the target's own row. The oracle is the plain
column computed in polars, fed to the same spec: every output field must be
the same number. Around it, what a relative target must not do -- learn from
a value either side cannot use, leak its column into the features, or apply
where the targets slot holds something other than a regression target.
"""

from __future__ import annotations

import re
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po

HOW = {
    "difference": pl.col("p") - pl.col("mid"),
    "ratio": pl.col("p") / pl.col("mid"),
    "log_ratio": (pl.col("p") / pl.col("mid")).log(),
}


def frame(n: int = 600, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    mid = 100.0 + np.cumsum(0.2 * rng.standard_normal(n))
    p = mid * np.exp(0.01 * (0.5 * x[:, 0] - 0.3 * x[:, 1]) + 0.002 * rng.standard_normal(n))
    return pl.DataFrame(
        {"t": np.arange(n, dtype=float), "x0": x[:, 0], "x1": x[:, 1], "mid": mid, "p": p}
    )


def common(**kw):
    d = dict(features=["x0", "x1"], clock="t", max_dclock=5.0, halflife=80.0, emit_sigma=True)
    d.update(kw)
    return d


def fields(out: pl.DataFrame, name: str, target: str) -> pl.DataFrame:
    """The spec's struct, its per-target fields renamed to one name."""
    df = out[name].struct.unnest()
    return df.rename({c: c.replace(f"_{target}", "_T") for c in df.columns if c.endswith(target)})


@pytest.mark.parametrize("relative", list(HOW))
@pytest.mark.parametrize("model", ["ewridge", "kalman", "huber", "holt"])
def test_a_relative_target_is_the_column_computed_first(model, relative):
    """The relative target against the same spec given the computed column:
    every field, to the bit."""
    df = frame()
    kw = {} if model != "kalman" else {"coef_halflife": 200.0}
    if model == "holt":
        kw["features"] = []
    target = po.target("p", relative_to="mid", relative=relative)
    got = po.ModelBank([getattr(po.spec, model)("m", targets=[target], **common(**kw))])
    want = po.ModelBank([getattr(po.spec, model)("m", targets=["d"], **common(**kw))])
    a = fields(got.fit_predict(df), "m", "p")
    b = fields(want.fit_predict(df.with_columns(d=HOW[relative])), "m", "d")
    assert a.columns == b.columns
    assert a.equals(b, null_equal=True), relative


def test_marginal_screens_against_a_relative_target():
    """``marginal``'s pairs against a return, which is what a screen wants."""
    df = frame()
    target = po.target("p", relative_to="mid", relative="log_ratio", name="ret")
    got = po.ModelBank([po.spec.marginal("m", targets=[target], **common(emit_sigma=False))])
    got.fit_predict(df)
    want = po.ModelBank([po.spec.marginal("m", targets=["ret"], **common(emit_sigma=False))])
    want.fit_predict(df.with_columns(ret=HOW["log_ratio"]))
    assert got.marginal("m").equals(want.marginal("m"), null_equal=True)


def test_a_value_either_side_cannot_use_makes_the_target_null():
    """A null ``p`` or ``mid``, one past the input bound, or for a ratio one
    that is not positive: the row is scored, not learned from -- the output of
    the same spec given the computed column with nulls there, to the bit. Two
    values past the bound must not subtract into a usable zero."""
    df = frame().with_row_index("i")
    i = pl.col("i")
    df = df.with_columns(
        p=pl.when(i % 17 == 3).then(None).when(i == 50).then(1e200).otherwise(pl.col("p")),
        mid=pl.when(i % 23 == 5)
        .then(None)
        .when(i == 50)
        .then(1e200)
        .when(i % 29 == 7)
        .then(-1.0)
        .otherwise(pl.col("mid")),
    ).drop("i")
    for relative in HOW:
        target = po.target("p", relative_to="mid", relative=relative)
        got = po.ModelBank([po.spec.ewridge("m", targets=[target], **common())])
        a = fields(got.fit_predict(df), "m", "p")
        usable = pl.col("p").is_not_null() & (pl.col("p").abs() <= 1e100)
        usable = usable & pl.col("mid").is_not_null() & (pl.col("mid").abs() <= 1e100)
        if relative != "difference":
            usable = usable & (pl.col("p") > 0) & (pl.col("mid") > 0)
        computed = df.with_columns(d=pl.when(usable).then(HOW[relative]).otherwise(None))
        want = po.ModelBank([po.spec.ewridge("m", targets=["d"], **common())])
        b = fields(want.fit_predict(computed), "m", "d")
        assert a.equals(b, null_equal=True), relative
        assert a["resid_T"][50] is None, "past the bound on both sides is no target"


def test_a_relative_target_is_named_after_its_column_or_its_name():
    df = frame()
    for target, want in [
        (po.target("p", relative_to="mid"), "pred_p"),
        (po.target("p", relative_to="mid", name="ret"), "pred_ret"),
        (po.target("p", name="price"), "pred_price"),
    ]:
        out = po.ModelBank([po.spec.ewridge("m", targets=[target], **common())]).fit_predict(df)
        assert want in out["m"].struct.fields
    # A renamed plain column is the column.
    renamed = po.ModelBank(
        [po.spec.ewridge("m", targets=[po.target("p", name="price")], **common())]
    )
    plain = po.ModelBank([po.spec.ewridge("m", targets=["p"], **common())])
    a = fields(renamed.fit_predict(df), "m", "price")
    b = fields(plain.fit_predict(df), "m", "p")
    assert a.equals(b, null_equal=True)


def test_a_saved_bank_resumes_with_its_relative_target(tmp_path):
    """Across a save and load, the same rows as one run: every field, with
    ``coef`` on every row, since by default it is emitted on each chunk's last
    row and a chunk boundary would add one."""
    df = frame()
    target = po.target("p", relative_to="mid", relative="ratio")
    spec = po.spec.ewridge("m", targets=[target, "x1"], **common(features=["x0"], coef_every=1))
    whole = po.ModelBank([spec]).fit_predict(df)
    bank = po.ModelBank([spec])
    first = bank.fit_predict(df[:250])
    bank.save(tmp_path / "r.state")
    resumed = po.ModelBank.load(tmp_path / "r.state")
    second = resumed.fit_predict(df[250:])
    assert pl.concat([first, second]).equals(whole, null_equal=True)
    assert resumed.gram("m")[0]["targets"] == ["p", "x1"], "the Gram names its targets"


def test_scoring_leaves_out_what_the_frame_does_not_have():
    """``predict`` scores without targets: a relative target is absent when
    either of its columns is, as a plain one is when its column is."""
    df = frame()
    spec = po.spec.ewridge("m", targets=[po.target("p", relative_to="mid")], **common())
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:400])
    full = bank.predict(df[400:])
    without = bank.predict(df[400:].drop("mid"))
    np.testing.assert_array_equal(
        full["m"].struct.field("pred_p").to_numpy(), without["m"].struct.field("pred_p").to_numpy()
    )
    assert without["m"].struct.field("resid_p").null_count() == len(without)


def with_targets(spec: dict, *targets) -> dict:
    """``spec`` with its targets replaced, for a builder whose own parameter
    would refuse a table before the bank could say why."""
    return {**spec, "targets": list(targets)}


@pytest.mark.parametrize(
    ("build", "msg"),
    [
        (
            lambda t: with_targets(
                po.spec.ew_class(
                    "m",
                    label="p",
                    classes=["a", "b"],
                    features=["x0"],
                    precision_prior=1.0,
                    halflife=50.0,
                ),
                t,
            ),
            "classifies its target as a label",
        ),
        (lambda t: po.spec.seqtest("m", targets=[t]), "tests the signs of its targets"),
        (
            lambda t: po.spec.ftrl("m", targets=[t], features=["x0"]),
            'fits a probability to a 0/1 target (loss = "logistic")',
        ),
        (
            # Named apart from its column, so only the column can give it away.
            lambda t: po.spec.ewridge(
                "m", targets=[po.target("p", relative_to="mid", name="ret")], features=["x0", "p"]
            ),
            '"p" is both a target and a feature',
        ),
        (
            lambda t: po.spec.ewridge(
                "m", targets=[po.target("p", relative_to="p")], features=["x0"]
            ),
            'target "p" is taken against its own column',
        ),
    ],
)
def test_a_relative_target_where_it_does_not_apply_is_refused_by_name(build, msg):
    with pytest.raises((ValueError, TypeError), match=re.escape(msg)):
        po.ModelBank([build(po.target("p", relative_to="mid"))])


def test_a_relative_target_may_be_taken_against_a_feature():
    """The reference is read at the row, as a feature is: no leak."""
    df = frame()
    spec = po.spec.ewridge(
        "m", targets=[po.target("p", relative_to="mid")], **common(features=["x0", "mid"])
    )
    out = po.ModelBank([spec]).fit_predict(df)
    assert out["m"].struct.field("pred_p").drop_nulls().len() > 500


def test_po_target_refuses_what_it_cannot_mean():
    with pytest.raises(ValueError, match="relative must be one of"):
        po.target("p", relative_to="mid", relative="sum")
    with pytest.raises(ValueError, match="needs relative_to"):
        po.target("p", relative="ratio")
    with pytest.raises(TypeError, match="column must be a str"):
        po.target(3)  # type: ignore[arg-type]
    assert po.target("p") == {"column": "p"}


def test_the_cli_reads_the_toml_table_form(tmp_path, online_cli):
    """``targets = [{ column = "p", relative_to = "mid" }]`` in the CLI's
    TOML is the Python spec: the same output, to the bit."""
    df = frame()
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
                'targets = ["x1", { column = "p", relative_to = "mid", relative = "log_ratio" }]',
                'clock = "t"',
                "max_dclock = 5.0",
                "halflife = 80.0",
                "[specs.model]",
                'type = "ew_ridge"',
            ]
        )
    )
    subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
    target = po.target("p", relative_to="mid", relative="log_ratio")
    spec = po.spec.ewridge("m", targets=["x1", target], **common(emit_sigma=False, features=["x0"]))
    want = po.ModelBank([spec]).fit_predict(df)
    got = pl.read_parquet(dst)
    assert got["m"].struct.fields == want["m"].struct.fields
    assert got["m"].equals(want["m"], null_equal=True)
