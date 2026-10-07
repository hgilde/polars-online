"""The JSON export of a bank state (docs/ENHANCEMENTS.md E68).

``ModelBank.to_json`` writes everything ``save`` writes, in a form something
that is not this library can read. It is an **export**, not a second state
format: ``load`` reads msgpack and only msgpack.

The whole risk is silent loss. JSON has no literal for ``NaN`` or ``±inf``
and ``serde_json`` writes all three as ``null`` without a word -- and a state
reaches that on the most ordinary setting there is, ``half_life=inf`` (no
decay), which puts an infinity in every stream's ``decay``. So the crate
tags them as ``"inf"`` / ``"-inf"`` / ``"nan"`` (``online_core::humanfloat``,
the spelling specs already use) and the exporter re-reads its own output and
refuses if it does not match the state.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import polars_online as po

INF = float("inf")


def _df(n: int = 150) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "g": ["a"] * (n // 2) + ["b"] * (n - n // 2),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
        }
    ).with_columns(y=2 * pl.col("x0") - pl.col("x1"))


B = dict(targets=["y"], features=["x0", "x1"])

#: One per model family, each with a non-finite value wherever the spec allows
#: one -- which is what makes this a test of the tagging and not of JSON.
EVERY_MODEL = {
    "ewridge": po.spec.ewridge("s", half_life=INF, **B),
    "ewridge_grid": po.spec.ewridge(
        "s",
        half_life=[INF, 10.0],
        clock="t",
        # Finite, as a cap and a session gap must be since task 120.
        gap_cap=1e9,
        session="g",
        session_gap=1e9,
        group="g",
        **B,
    ),
    "rls": po.spec.rls("s", half_life=INF, **B),
    "lasso": po.spec.lasso("s", half_life=INF, lasso_path=[0.1, 0.01], **B),
    "kalman": po.spec.kalman("s", half_life=50.0, coef_half_life=INF, **B),
    "sgd": po.spec.sgd("s", half_life=INF, clip_gradient=INF, **B),
    "pa": po.spec.pa("s", half_life=INF, **B),
    "huber": po.spec.huber("s", half_life=INF, **B),
    "quantile": po.spec.quantile("s", half_life=INF, quantile=0.5, **B),
    "ftrl": po.spec.ftrl("s", half_life=INF, **B),
    "holt": po.spec.holt("s", targets=["y"], half_life=INF, trend_half_life=INF),
    "ew_cov": po.spec.ew_cov("s", features=["x0", "x1"], stats=["corr"], half_life=INF),
    "marginal": po.spec.marginal(
        "s",
        half_life=INF,
        lags=[1, 2],
        serial_rule="geometric",
        bins=4,
        bin_warm_rows=20,
        **B,
    ),
    "kmeans": po.spec.kmeans("s", features=["x0", "x1"], k=2, half_life=INF),
    "micro": po.spec.micro("s", features=["x0", "x1"], eps=0.5, half_life=INF),
    "deco": po.spec.deco("s", features=["x0", "x1"], half_life=INF),
    "seqtest": po.spec.seqtest("s", targets=["y"]),
    "corrchange": po.spec.corrchange("s", features=["x0", "x1"], span_rows=40),
    "bocpd": po.spec.bocpd("s", features=["x0"]),
}


@pytest.mark.parametrize("name", sorted(EVERY_MODEL))
def test_every_model_exports_without_losing_a_value(name):
    """The export refuses rather than dropping anything, so reaching the end
    of this is the assertion. It is also the audit: a new model with an
    inf-capable config float fails here until its field is annotated."""
    bank = po.ModelBank([EVERY_MODEL[name]])
    bank.fit_predict(_df())
    text = bank.to_json()
    json.loads(text)  # and it is strict JSON, not a relaxed dialect


def test_a_non_finite_is_tagged_rather_than_nulled():
    """``half_life=inf`` is the case that made this necessary: it is a
    documented setting, and it lands in the state's own ``decay``."""
    bank = po.ModelBank([po.spec.ewridge("s", half_life=INF, **B)])
    bank.fit_predict(_df())
    doc = json.loads(bank.to_json())

    assert doc["specs"][0]["half_life"] == "inf", "the spec's own field"

    def decays(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "decay":
                    yield v
                else:
                    yield from decays(v)
        elif isinstance(node, list):
            for v in node:
                yield from decays(v)

    found = list(decays(doc))
    assert found, "the state carries a decay"
    assert all(d == {"Halflife": "inf"} for d in found), found


def _column_stats(node):
    """Every column's summary in a state's JSON: the dicts with a count, a
    null count and the range."""
    if isinstance(node, dict):
        if {"count", "nulls", "min", "max"} <= node.keys():
            yield node
        for v in node.values():
            yield from _column_stats(v)
    elif isinstance(node, list):
        for v in node:
            yield from _column_stats(v)


@pytest.mark.parametrize("how", ["a target null on every row", "every row skipped"])
def test_a_column_that_never_held_a_value_exports(how):
    """The data summary starts each column's ``min`` and ``max`` at ``inf``
    and ``-inf``, and a column no row gave a usable value keeps them. The
    export refused such a bank as a dropped value, on every Python; it
    writes them tagged, as every other non-finite float (found running the
    suite on Python 3.15, 2026-09-29)."""
    df = _df()
    if how == "a target null on every row":
        spec = po.spec.ewridge("s", half_life=10.0, **B)
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    else:
        spec = po.spec.ewridge("s", half_life=10.0, weight="w", **B)
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("w"))
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    doc = json.loads(bank.to_json())
    empty = [c for c in _column_stats(doc) if c["count"] == 0]
    assert empty, "a column with no value is in the summary"
    assert all(c["min"] == "inf" and c["max"] == "-inf" for c in empty), empty


def test_a_coefficient_no_solve_gave_exports_tagged():
    """A target no solve has fit has NaN coefficients (review round 4, CC1):
    in the stream's last row and in a closed row the bank still holds, which
    a bank file carries until it is drained. Both export tagged, as
    ``"nan"``, where an untagged NaN made the export refuse the bank."""
    df = _df().with_columns(
        z=pl.when(pl.col("g") == "a").then(None).otherwise(pl.col("y")).cast(pl.Float64)
    )
    spec = po.spec.ewridge(
        "s",
        targets=["y", "z"],
        features=["x0", "x1"],
        half_life=10.0,
        group="g",
        group_close="monotone",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    doc = json.loads(bank.to_json())
    # Group a closed, undrained: one row per Gram, `z`'s on its own.
    coefs = [row["coef"] for row in doc["closed"]]
    assert ["nan"] * 3 in coefs, coefs
    assert any(all(isinstance(v, float) for v in c) for c in coefs), coefs


def test_the_export_is_the_state_and_not_a_summary():
    """Same envelope, same specs, one entry per (spec, group) as the bank
    holds -- not a digest of them."""
    specs = [
        po.spec.ewridge("r", group="g", half_life=20.0, **B),
        po.spec.ew_cov("c", features=["x0", "x1"], stats=["corr"], half_life=20.0, group="g"),
    ]
    bank = po.ModelBank(specs)
    bank.fit_predict(_df())
    doc = json.loads(bank.to_json())

    assert doc["schema_version"] == po.schema_version()
    assert doc["specs"] == json.loads(json.dumps(bank.specs))
    assert doc["rows_fed"] == bank.rows_seen()
    assert len(doc["states"]) == len(specs), "one block per spec"
    for block, spec in zip(doc["states"], specs, strict=True):
        held = bank.groups(spec["name"]).height
        assert len(block) == held, f"{spec['name']}: one entry per group held"


def test_pretty_and_compact_carry_the_same_thing():
    bank = po.ModelBank([po.spec.ewridge("s", half_life=INF, **B)])
    bank.fit_predict(_df())
    pretty, compact = bank.to_json(), bank.to_json(pretty=False)
    assert len(compact) < len(pretty)
    assert json.loads(pretty) == json.loads(compact)


def test_save_json_writes_what_to_json_returns(tmp_path):
    bank = po.ModelBank([po.spec.ewridge("s", half_life=INF, **B)])
    bank.fit_predict(_df())
    p = tmp_path / "state.json"
    bank.save_json(p)
    assert json.loads(p.read_text(encoding="utf-8")) == json.loads(bank.to_json())


def test_the_export_does_not_disturb_the_state(tmp_path):
    """Reading a bank must not change it: the msgpack before and after an
    export is the same bytes, and the bank keeps predicting identically."""
    bank = po.ModelBank([po.spec.ewridge("s", half_life=INF, group="g", **B)])
    df = _df()
    bank.fit_predict(df)
    before = bank.save_bytes()
    bank.to_json()
    assert bank.save_bytes() == before


def test_json_is_an_export_and_not_a_load_format(tmp_path):
    """``load`` takes msgpack. Handing it JSON is refused as what it is --
    not a bank file -- rather than half-read."""
    bank = po.ModelBank([po.spec.ewridge("s", half_life=20.0, **B)])
    bank.fit_predict(_df())
    p = tmp_path / "state.json"
    bank.save_json(p)
    with pytest.raises(ValueError):
        po.ModelBank.load(p)
