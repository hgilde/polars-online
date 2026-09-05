"""The public API, rendered to text and pinned (docs/RELEASE-READINESS.md).

The public surface is much larger than `__all__`: it includes every spec
constructor's keyword names **and default values** (changing a default silently
changes users' numbers), the expression namespace, and — largest and least
obvious — the **output field names**. Users index the result struct by strings
like `pred_y__r0.5@h100`; those strings are produced by `format!` over floats,
and rustc's float formatting is an implementation detail that has changed
between compiler versions before. Nothing but this file pins the whole grammar.

`tests/api_surface.txt` is the contract. This test regenerates the same text
and diffs it, so **every API change becomes a reviewable diff in the PR that
makes it** — deliberate changes are visible and versioned, accidental ones
fail. Regenerate after an intended change with:

    UPDATE_API_SURFACE=1 uv run pytest tests/test_api_surface.py

and treat the diff to `api_surface.txt` as part of the change under review.
Pre-1.0, a diff here needs at least a minor version bump; a changed *default*
does too, because it changes results without an error.
"""

import difflib
import inspect
import os
from pathlib import Path

import polars as pl

import polars_online as po
from polars_online import spec as spec_mod

SNAPSHOT = Path(__file__).parent / "api_surface.txt"


def describe_api() -> str:
    out: list[str] = []
    w = out.append

    w("# polars-online public API surface. Regenerate: UPDATE_API_SURFACE=1 pytest")
    w("")

    w("[package]")
    for name in sorted(po.__all__):
        w(f"  {name}")
    w(f"  schema_version = {po.schema_version()}")
    w("")

    w("[common parameters]  # every constructor accepts these via **common")
    from polars_online import _spec as _impl

    for p in inspect.signature(_impl._common).parameters.values():
        if p.name in ("name", "model") or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        w(f"  {p.name}{default}")
    w("")

    w("[spec constructors]  # keyword names AND defaults are API")
    for name in sorted(dir(spec_mod)):
        if name.startswith("_"):
            continue
        fn = getattr(spec_mod, name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        w(f"  {name}:")
        for p in sig.parameters.values():
            default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
            w(f"    {p.name}{default}")
    w("")

    w("[ModelBank]")
    for name in sorted(dir(po.ModelBank)):
        if not name.startswith("_") or name in ("__reduce__",):
            w(f"  {name}")
    w("")

    w("[expression namespace]  # pl.col(...).online.<method>")
    ns = pl.col("x").online
    for name in sorted(dir(ns)):
        if not name.startswith("_"):
            w(f"  {name}")
    w("")

    w("[frame namespaces]  # lf.online.<method> -> LazyFrame; df.online.<method> -> DataFrame")
    for label, frame in (("LazyFrame", pl.LazyFrame()), ("DataFrame", pl.DataFrame())):
        for name in sorted(dir(frame.online)):
            if name.startswith("_"):
                continue
            sig = inspect.signature(getattr(frame.online, name))
            w(f"  {label}.online.{name}{sig}")
    w("")

    w("[output field grammar]  # the strings users index the output struct by")
    cases: list[tuple[str, dict]] = [
        ("ewridge minimal", dict(targets=["y"], features=["x0"], halflife=100.0)),
        (
            "ewridge full grid, every output",
            dict(
                targets=["y", "z"],
                features=["x0", "x1"],
                feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
                ridge=[1e-6, 0.5],
                halflife=[100.0, 500.0],
                emit_sigma=True,
                emit_resid_z=True,
                emit_drift=True,
                emit_metrics=True,
                emit_autocorr=True,
                resid_quantiles=[0.05, 0.95],
                emit_selected=True,
                emit_averaged=True,
            ),
        ),
        (
            "float rendering at the extremes",
            dict(targets=["y"], features=["x0"], ridge=[1e-300, 0.5], halflife=[100.0, 1e9]),
        ),
        ("lasso path", dict(targets=["y"], features=["x0"], lasso_path=[0.1, 0.0], halflife=100.0)),
    ]
    for label, kw in cases:
        model = (
            kw.pop("_model", "ewridge")
            if "_model" in kw
            else ("lasso" if "lasso_path" in kw else "ewridge")
        )
        s = getattr(po.spec, model)("m", min_periods=2.0, **kw)
        w(f"  {label}:")
        for f in po.spec.output_fields(s):
            w(f"    {f}")
    for model, kw in [
        ("rls", dict(targets=["y"], features=["x0"], halflife=100.0)),
        ("lasso", dict(targets=["y"], features=["x0"], halflife=100.0, lasso_path=[0.1, 0.0])),
        ("kalman", dict(targets=["y"], features=["x0"], halflife=100.0, coef_halflife=50.0)),
        ("huber", dict(targets=["y"], features=["x0"], halflife=100.0)),
        ("quantile", dict(targets=["y"], features=["x0"], halflife=100.0, quantile=0.5)),
        ("sgd", dict(targets=["y"], features=["x0"], halflife=100.0, learning_rate=0.01)),
        ("pa", dict(targets=["y"], features=["x0"], halflife=100.0)),
        ("ftrl", dict(targets=["y"], features=["x0"], halflife=100.0)),
        ("holt", dict(targets=["y"], halflife=100.0)),
        (
            "ew_cov",
            dict(
                features=["x0", "x1"], stats=["mean", "var", "std", "cov", "corr"], halflife=100.0
            ),
        ),
        ("kmeans", dict(features=["x0", "x1"], k=2, halflife=100.0)),
    ]:
        s = getattr(po.spec, model)("m", min_periods=2.0, **kw)
        w(f"  {model} minimal:")
        for f in po.spec.output_fields(s):
            w(f"    {f}")
    return "\n".join(out) + "\n"


def test_api_surface_matches_the_snapshot():
    got = describe_api()
    if os.environ.get("UPDATE_API_SURFACE"):
        SNAPSHOT.write_text(got, encoding="utf-8")
        return
    assert SNAPSHOT.exists(), (
        "no snapshot yet — run UPDATE_API_SURFACE=1 pytest tests/test_api_surface.py"
    )
    want = SNAPSHOT.read_text(encoding="utf-8")
    if got != want:
        diff = "\n".join(
            difflib.unified_diff(
                want.splitlines(), got.splitlines(), "api_surface.txt", "current", lineterm=""
            )
        )
        raise AssertionError(
            "The public API changed. If intended, regenerate the snapshot with\n"
            "UPDATE_API_SURFACE=1 and include the diff in the PR — a changed\n"
            "field name or default needs a version bump.\n\n" + diff
        )
