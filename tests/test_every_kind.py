"""Two sweeps over every model kind, where they ran over the regression models
alone (docs/PLAN.md task 111): the output struct a bank writes is the one
``po.spec.output_fields`` declares, and a bank saved mid-stream and loaded goes
on exactly as the one that was not. Parametrised over the registry's
``MINIMAL`` itself, so a new kind is in both the day it is registered, and no
list here can fall behind it. ``rcov``, ``hmm``, ``corrchange`` and ``bocpd``
had neither check anywhere.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_model_registry import MINIMAL, _build

TIER = "essential"


def frame_for(name: str, n: int = 240) -> pl.DataFrame:
    """Rows every kind can run on: two features, a target -- a 0/1 one for
    ``ftrl``'s logistic loss, a text label for ``ew_class`` -- and a group key
    that only grows, for ``rcov``'s ``monotone`` close."""
    rng = np.random.default_rng(11)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * x0 - x1 + 0.1 * rng.normal(size=n)
    df = pl.DataFrame({"x0": x0, "x1": x1, "y": y, "g": np.arange(n) // 60})
    if name == "ftrl":
        df = df.with_columns(y=(pl.col("y") > 0).cast(pl.Float64))
    if name == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 0).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


@pytest.mark.parametrize("name", sorted(MINIMAL))
def test_every_kind_writes_the_struct_it_declares(name):
    spec = _build(name)
    out = po.ModelBank([spec]).fit_predict(frame_for(name))
    assert [f.name for f in out.schema["m"].fields] == po.spec.output_fields(spec)


@pytest.mark.parametrize("name", sorted(MINIMAL))
def test_every_kind_resumes_mid_stream(name, tmp_path):
    spec = _build(name)
    df = frame_for(name)
    a = po.ModelBank([spec])
    a.fit_predict(df[:120])
    a.save(tmp_path / f"{name}.state")
    b = po.ModelBank.load(tmp_path / f"{name}.state", specs=[spec])
    ra, rb = a.fit_predict(df[120:]), b.fit_predict(df[120:])
    assert ra.equals(rb, null_equal=True)
    assert b.specs == a.specs
