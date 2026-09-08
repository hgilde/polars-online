"""The JSON export of a bank state (docs/ENHANCEMENTS.md E68).

``ModelBank.to_json`` writes everything ``save`` writes, in a form something
that is not this library can read. It is an **export**, not a second state
format: ``load`` reads msgpack and only msgpack.

The whole risk is silent loss. JSON has no literal for ``NaN`` or ``±inf``
and ``serde_json`` writes all three as ``null`` without a word -- and a state
reaches that on the most ordinary setting there is, ``halflife=inf`` (no
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
    "ewridge": po.spec.ewridge("s", halflife=INF, **B),
    "ewridge_grid": po.spec.ewridge(
        "s",
        halflife=[INF, 10.0],
        clock="t",
        max_dclock=INF,
        session="g",
        session_gap=INF,
        group="g",
        **B,
    ),
    "rls": po.spec.rls("s", halflife=INF, **B),
    "lasso": po.spec.lasso("s", halflife=INF, lasso_path=[0.1, 0.01], **B),
    "kalman": po.spec.kalman("s", halflife=50.0, coef_halflife=INF, **B),
    "sgd": po.spec.sgd("s", halflife=INF, clip_gradient=INF, **B),
    "pa": po.spec.pa("s", halflife=INF, **B),
    "huber": po.spec.huber("s", halflife=INF, **B),
    "quantile": po.spec.quantile("s", halflife=INF, quantile=0.5, **B),
    "ftrl": po.spec.ftrl("s", halflife=INF, **B),
    "holt": po.spec.holt("s", targets=["y"], halflife=INF, trend_halflife=INF),
    "ew_cov": po.spec.ew_cov("s", features=["x0", "x1"], stats=["corr"], halflife=INF),
    "marginal": po.spec.marginal(
        "s",
        halflife=INF,
        lags=[1, 2],
        serial_rule="geometric",
        bins=4,
        bin_warm_rows=20,
        **B,
    ),
    "kmeans": po.spec.kmeans("s", features=["x0", "x1"], k=2, halflife=INF),
    "micro": po.spec.micro("s", features=["x0", "x1"], eps=0.5, halflife=INF),
    "deco": po.spec.deco("s", features=["x0", "x1"], halflife=INF),
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
    """``halflife=inf`` is the case that made this necessary: it is a
    documented setting, and it lands in the state's own ``decay``."""
    bank = po.ModelBank([po.spec.ewridge("s", halflife=INF, **B)])
    bank.fit_predict(_df())
    doc = json.loads(bank.to_json())

    assert doc["specs"][0]["halflife"] == "inf", "the spec's own field"

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


def test_the_export_is_the_state_and_not_a_summary():
    """Same envelope, same specs, one entry per (spec, group) as the bank
    holds -- not a digest of them."""
    specs = [
        po.spec.ewridge("r", group="g", halflife=20.0, **B),
        po.spec.ew_cov("c", features=["x0", "x1"], stats=["corr"], halflife=20.0, group="g"),
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
    bank = po.ModelBank([po.spec.ewridge("s", halflife=INF, **B)])
    bank.fit_predict(_df())
    pretty, compact = bank.to_json(), bank.to_json(pretty=False)
    assert len(compact) < len(pretty)
    assert json.loads(pretty) == json.loads(compact)


def test_save_json_writes_what_to_json_returns(tmp_path):
    bank = po.ModelBank([po.spec.ewridge("s", halflife=INF, **B)])
    bank.fit_predict(_df())
    p = tmp_path / "state.json"
    bank.save_json(p)
    assert json.loads(p.read_text(encoding="utf-8")) == json.loads(bank.to_json())


def test_the_export_does_not_disturb_the_state(tmp_path):
    """Reading a bank must not change it: the msgpack before and after an
    export is the same bytes, and the bank keeps predicting identically."""
    bank = po.ModelBank([po.spec.ewridge("s", halflife=INF, group="g", **B)])
    df = _df()
    bank.fit_predict(df)
    before = bank.save_bytes()
    bank.to_json()
    assert bank.save_bytes() == before


def test_json_is_an_export_and_not_a_load_format(tmp_path):
    """``load`` takes msgpack. Handing it JSON is refused as what it is --
    not a bank file -- rather than half-read."""
    bank = po.ModelBank([po.spec.ewridge("s", halflife=20.0, **B)])
    bank.fit_predict(_df())
    p = tmp_path / "state.json"
    bank.save_json(p)
    with pytest.raises(ValueError):
        po.ModelBank.load(p)
