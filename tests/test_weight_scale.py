"""A row weight scales evidence, not counts (docs/PLAN.md task 147).

Multiplying every row's weight by one constant changes no output -- the EW
models hold to it by construction, since their sums are means -- unless the
model's docs say a weight counts on the sum scale. This fits every spec of
the release probe's workload, every kind the bank builds, at the stream's
weights and at a hundred times them, ``min_weight`` scaled with them, and
holds the set of specs whose outputs move to exactly the exceptions named
below, each with the reason its docs give. ``weight_sum`` is the accumulated
weight, so it scales with the weights where the model counts weight, and
stays where it counts rows.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_probe", REPO / "scripts/release_probe.py")
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")

#: The specs whose outputs a weight's scale reaches, and why: each is in its
#: builder's docs.
EXCEPTIONS = {
    "rls": "A <- lam A + w z z' starts from delta * I: a sum-scale prior a heavier "
    "stream outweighs sooner (classic RLS regularization)",
    "kalman": "a row's weight scales its observation's precision: its variance is obs_var / w",
    "sgd": "the gradient is d * z * w: a weight is a step size",
    "ftrl": "an importance weight, as Vowpal Wabbit's: the gradient carries it "
    "against l1, l2 and beta in absolute weight (task 147 kept VW's semantics)",
    "pa": "a weight below 1 scales tau, one above 1 counts as 1",
    "hmm": "A <- decay A + w xi against the Dirichlet pseudo-count tau",
    "seqtest": "it scores ridge against kalman, which moves",
}

SCALE = 100.0


def _fit(c: float) -> pl.DataFrame:
    specs = []
    for name, builder, kw, _ in probe.WORKLOAD:
        kw = dict(kw)
        if builder != "seqtest":
            kw["min_weight"] = kw.get("min_weight", 4.0) * c
        specs.append(getattr(po.spec, builder)(name, **kw))
    return po.ModelBank(specs).fit_predict(probe.stream().with_columns(pl.col("w") * c))


def _moved(a: pl.DataFrame, b: pl.DataFrame, name: str) -> list[str]:
    out = []
    for f in a.schema[name].fields:
        if f.name.startswith(("coef", "support_coef")) or not f.dtype.is_numeric():
            continue
        x = a[name].struct.field(f.name).cast(pl.Float64).to_numpy()
        y = b[name].struct.field(f.name).cast(pl.Float64).to_numpy()
        if f.name.startswith("weight_sum") and not np.allclose(x, y, equal_nan=True):
            y = y / SCALE
        if (np.isnan(x) != np.isnan(y)).any():
            out.append(f.name)
            continue
        ok = ~np.isnan(x)
        if ok.any() and np.max(np.abs(x[ok] - y[ok]) / (1.0 + np.abs(x[ok]))) > 1e-8:
            out.append(f.name)
    return out


def test_the_docs_say_what_each_unit_is():
    """Task 151: ``ftrl``'s penalties are a prior of fixed mass, with
    ``lasso`` named for the mean scale, and ``micro``'s ``beta_mu`` is a point
    density with the rule to set it from the arrival rate."""
    ftrl = inspect.cleandoc(po.spec.ftrl.__doc__ or "")
    assert "a prior of fixed\nmass against evidence that grows with weight and density" in ftrl
    assert ":func:`lasso`" in ftrl
    micro = inspect.cleandoc(po.spec.micro.__doc__ or "")
    assert "``1.44 * s * v * h``" in micro
    assert "DBSCAN's\n    ``MinPts``" in micro or "DBSCAN's ``MinPts``" in micro


def test_only_the_documented_exceptions_move_with_the_weights_scale():
    a, b = _fit(1.0), _fit(SCALE)
    moved = {name: f for name, *_ in probe.WORKLOAD if (f := _moved(a, b, name))}
    assert set(moved) == set(EXCEPTIONS), {k: v[:4] for k, v in moved.items()}
