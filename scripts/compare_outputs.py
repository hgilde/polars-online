"""Two builds' outputs on the same cases, column by column: how far they move, and what flips.

    uv run python scripts/compare_outputs.py --a PYTHON_A --b PYTHON_B --cases DIR_OR_FILE --out DIR
    uv run python scripts/compare_outputs.py --a release:1.5.0 --b .venv --builtin --time-rounds 0

`scripts/compare_release.py` says whether two builds' outputs are the same,
to the bit. This tool says, when they are not, *how* they are not. A change
that moves output bits on purpose (a different order of summation, a fused
operation, a faster recursion) is judged on how far each emitted column moves
and whether any decision made from it changes, over more than the simplest
case: several features, targets, groups and chunkings, a stream long enough
for drift to accumulate, a save and a load in the middle. Per case, for every
emitted column:

- floats: the largest, the median and the 99.9th-percentile relative movement,
  ``|a - b| / max(|a|, |b|, floor)`` over the rows where both are finite (the
  floor is printed, default 1e-12: a value near zero is noise, and a ratio to
  it would call the noise a movement), the rows that differ at all (by bits),
  nulls that appear or vanish, a NaN or an infinity that does;
- a list of floats (``coef``): the same, over its elements, and the rows
  whose length changed;
- every other column (a Boolean, an integer, a string, an Enum such as
  ``withheld_reason``, a selected slot or label): the rows where it differs,
  which is a flipped decision;
- the decisions a case declares: a test statistic crossing a threshold, an
  argmax over several columns, with the rows where the decision flips;
- the saved state's bytes, equal or not (a version number alone moves them);
- ABBA time, ``--time-rounds`` rounds (A B, B A, A B, ...), the minimum and
  the median of each build and their ratio.

A *build* is an interpreter whose environment has the package installed: a
venv's path or its ``python`` (``--a`` and ``--b``), or ``release:X.Y.Z``,
a released wheel in the cache `compare_release.py` keeps. ``--a-path DIR``
puts a directory holding a ``rheon`` first on the path of A's runs,
for a build that is a directory and not an environment. Each run reports
which package it imported (path, version, native library's hash), and a
comparison of a build with itself is refused.

**Cases** are plain dictionaries in Python files, so that a model's cases are
added without editing this tool (``--cases`` names files or folders of
``*.py``; each defines ``cases()`` or ``CASES``). The fields, all optional
but ``name``:

``name`` ``family`` ``item``
    the id; the file's stem; the table a case is reported in (default the family).
``rows`` ``k`` ``targets`` ``groups`` ``chunk``
    the stream: rows, features ``x0..``, targets ``y0..``, groups in column
    ``g`` (0: no group), rows per chunk (``None``: one chunk).
``seed`` ``null_features`` ``null_targets`` ``extras`` ``gaps`` ``shift``
    the data: the seed; the share of rows with ``x0`` or ``y0`` null; extra
    columns (``hit`` a 0/1, ``lab`` a two-class and ``lab3`` a three-class label, ``s`` a session,
    ``block`` a growing key); clock jumps past ``ctx["gap_cap"]``; a change of
    the true coefficients at 60% of the stream.
``save_at``
    a share of the stream after which the state is saved, loaded and the run continues.
``specs``
    ``specs(po, ctx) -> list of spec dicts``, the bank's specs. ``ctx`` names
    ``features``, ``targets``, ``clock``, ``group``, ``weight``, ``half_life``
    and ``gap_cap`` (in clock units, with the group count: a half-life is
    ``max(20, 5k)`` of a group's own rows, and ``gap_cap`` is ten of them), ``k``, ``m``,
    ``groups``, ``rows``.
``windows``
    instead of ``specs``: ``windows(po, ctx) -> dict`` of the keyword
    arguments of ``po.stream.with_windows`` (the named formulas, ``clock``, ...).
``decisions``
    ``{"name", "kind": "threshold", "column", "value", "op", "abs"}`` or
    ``{"name", "kind": "argmax", "columns": [...]}``; names may use ``*``.
``timed``
    ``False`` leaves the case out of the timing rounds.

Inputs are generated once, by this tool, and read by both builds, so the
streams are the same bits. Everything is written to ``--out``: a markdown table
per item, ``summary.md`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
DEFAULT_FLOOR = 1e-12

#: Defaults of a case's fields.
CASE_DEFAULTS: dict[str, Any] = {
    "rows": 20_000,
    "k": 4,
    "targets": 1,
    "groups": 1,
    "chunk": 1000,
    "seed": 0,
    "null_features": 0.0,
    "null_targets": 0.0,
    "extras": [],
    "gaps": True,
    "shift": True,
    "save_at": None,
    "decisions": [],
    "timed": True,
}
DATA_KEYS = (
    "rows",
    "k",
    "targets",
    "groups",
    "seed",
    "null_features",
    "null_targets",
    "extras",
    "gaps",
    "shift",
)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _smoke_specs(po: Any, ctx: dict) -> list[dict]:
    common = dict(
        targets=ctx["targets"],
        features=ctx["features"],
        group=ctx["group"],
        clock="t",
        gap_cap=ctx["gap_cap"],
        half_life=ctx["half_life"],
        weight="w",
    )
    return [
        po.spec.ewridge("smoke_ridge", min_weight=float(ctx["k"]), emit_sigma=True, **common),
        po.spec.rls("smoke_rls", **common),
    ]


#: Two small cases, so the tool runs with nothing else (``--builtin``).
BUILTIN: list[dict] = [
    {
        "name": "smoke/k4-g3-c500",
        "family": "smoke",
        "rows": 6_000,
        "k": 4,
        "groups": 3,
        "chunk": 500,
        "specs": _smoke_specs,
        "decisions": [
            {
                "name": "ridge flagged",
                "kind": "threshold",
                "column": "smoke_ridge.sigma_*",
                "value": 1.0,
                "op": ">",
            },
        ],
    },
    {
        "name": "smoke/k4-g3-save",
        "family": "smoke",
        "rows": 6_000,
        "k": 4,
        "groups": 3,
        "chunk": 500,
        "save_at": 0.5,
        "specs": _smoke_specs,
    },
]


def load_cases(paths: list[Path], builtin: bool) -> list[dict]:
    """Every case in the given files and folders (and the built-in ones), defaults filled."""
    files: list[Path] = []
    for p in paths:
        files += (
            sorted(q for q in p.glob("*.py") if not q.name.startswith("_")) if p.is_dir() else [p]
        )
    found: list[tuple[str, dict]] = [("smoke", c) for c in BUILTIN] if builtin else []
    for f in files:
        spec = importlib.util.spec_from_file_location(f"_output_cases_{f.stem}", f)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load case file {f}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        got = module.cases() if hasattr(module, "cases") else module.CASES
        found += [(f.stem, c) for c in got]
    cases: list[dict] = []
    seen: set[str] = set()
    for stem, raw in found:
        case = {**CASE_DEFAULTS, **raw}
        case.setdefault("family", stem)
        case.setdefault("item", case["family"])
        if "specs" not in case and "windows" not in case:
            raise SystemExit(f"case {case['name']!r} has neither `specs` nor `windows`")
        if case["name"] in seen:
            raise SystemExit(f"two cases are named {case['name']!r}")
        seen.add(case["name"])
        cases.append(case)
    return cases


def select(cases: list[dict], only: list[str], items: list[str]) -> list[dict]:
    """The cases whose name matches a glob of ``only`` (or contains it) and whose item is listed."""

    def wanted(c: dict) -> bool:
        by_name = not only or any(fnmatch.fnmatch(c["name"], o) or o in c["name"] for o in only)
        return by_name and (not items or c["item"] in items)

    return [c for c in cases if wanted(c)]


def scaled(case: dict, scale: float) -> dict:
    return {**case, "rows": max(int(case["rows"] * scale), 50)} if scale != 1.0 else case


def half_life(case: dict) -> float:
    """A half-life of about ``max(20, 5k)`` of one group's own rows, in clock units."""
    return float(max(int(case["groups"]), 1) * max(20, 5 * case["k"]))


def context(case: dict) -> dict:
    """What a case's ``specs`` and ``windows`` read; the same in the parent and the worker."""
    g = int(case["groups"])
    return {
        "features": [f"x{j}" for j in range(case["k"])],
        "targets": [f"y{j}" for j in range(case["targets"])],
        "clock": "t",
        "group": "g" if g else None,
        "weight": "w",
        "half_life": half_life(case),
        "gap_cap": 10.0 * half_life(case),
        "k": case["k"],
        "m": case["targets"],
        "groups": g,
        "rows": case["rows"],
    }


def data_key(case: dict) -> str:
    blob = json.dumps({k: case[k] for k in DATA_KEYS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_input(case: dict) -> Any:
    """The stream a case runs on: correlated features, targets from a linear
    rule whose coefficients change at 60% of the stream, a few outliers, an
    irregular clock with a few jumps, zero weights, and the extras asked for."""
    import numpy as np
    import polars as pl

    n, k, m, g = case["rows"], case["k"], case["targets"], int(case["groups"])
    rng = np.random.default_rng(case["seed"])
    z = rng.standard_normal((n, k))
    x = z.copy()
    x[:, 1:] += 0.5 * z[:, :-1]
    beta = rng.standard_normal((k, m)) / np.sqrt(k)
    beta2 = beta + (0.5 * rng.standard_normal((k, m)) / np.sqrt(k) if case["shift"] else 0.0)
    cut = int(0.6 * n)
    signal = np.empty((n, m))
    signal[:cut] = x[:cut] @ beta
    signal[cut:] = x[cut:] @ beta2
    noise = 0.5 * rng.standard_normal((n, m))
    outliers = rng.random(n) < 0.005
    noise[outliers] *= 12.0
    y = signal + noise
    steps = rng.choice([0.5, 1.0, 1.5, 2.0], size=n)
    if case["gaps"] and n >= 1000:
        for share in (0.3, 0.7):
            steps[int(share * n)] += 15.0 * half_life(case)
    t = np.cumsum(steps)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.random(n) < 0.005] = 0.0
    cols: dict[str, Any] = {"t": t, "w": w}
    if g:
        keys = rng.integers(0, g, n)
        cols["g"] = pl.Series([f"g{v:05d}" for v in keys])
    for j in range(k):
        cols[f"x{j}"] = x[:, j]
    for j in range(m):
        cols[f"y{j}"] = y[:, j]
    extras = set(case["extras"])
    if "hit" in extras:
        cols["hit"] = (signal[:, 0] + noise[:, 0] > 0).astype(float)
    if "lab" in extras:
        cols["lab"] = np.where(signal[:, 0] + noise[:, 0] > 0, "up", "down")
    if "lab3" in extras:
        v = signal[:, 0] + noise[:, 0]
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        cols["lab3"] = np.where(v < lo, "down", np.where(v > hi, "up", "flat"))
    if "s" in extras:
        cols["s"] = np.where(np.arange(n) < n // 2, "am", "pm")
    if "block" in extras:
        cols["block"] = np.arange(n) // 1000
    df = pl.DataFrame(cols)
    if case["null_features"]:
        df = df.with_columns(
            pl.when(pl.Series(rng.random(n) < case["null_features"]))
            .then(None)
            .otherwise(pl.col("x0"))
            .alias("x0")
        )
    if case["null_targets"]:
        df = df.with_columns(
            pl.when(pl.Series(rng.random(n) < case["null_targets"]))
            .then(None)
            .otherwise(pl.col("y0"))
            .alias("y0")
        )
    return df


# ---------------------------------------------------------------------------
# The worker: one case under one build
# ---------------------------------------------------------------------------


def _flatten(frame: Any, drop: set[str]) -> Any:
    """The columns a run added, every struct unnested into ``outer.inner`` columns."""
    import polars as pl

    frame = frame.select([c for c in frame.columns if c not in drop])
    while any(isinstance(d, pl.Struct) for d in frame.schema.values()):
        parts = []
        for name, dtype in frame.schema.items():
            if isinstance(dtype, pl.Struct):
                parts.append(frame.select(pl.col(name).struct.unnest().name.prefix(f"{name}.")))
            else:
                parts.append(frame.select(name))
        frame = pl.concat(parts, how="horizontal")
    return frame


def _run_case(po: Any, case: dict, df: Any, tmp: Path, collect: bool) -> tuple[Any, bytes | None]:
    """Feed the stream through the case's bank (or windows); the output columns
    added (None if not collecting) and the final state's bytes (a bank's)."""
    import polars as pl

    ctx = context(case)
    n = df.height
    split = None
    if case["save_at"] is not None:
        chunk = case["chunk"] or n
        split = min(max(round(n * case["save_at"] / chunk) * chunk, chunk), n - 1)
    parts = [df] if split is None else [df.slice(0, split), df.slice(split)]
    outs: list[Any] = []
    drop = set(df.columns)
    state: bytes | None = None
    if "specs" in case:
        bank = po.ModelBank(case["specs"](po, ctx))
        for i, part in enumerate(parts):
            if i:
                bank = po.ModelBank.load_bytes(bank.save_bytes())
            for out in bank.fit_predict_batches(part, chunk_size=case["chunk"]):
                if collect:
                    outs.append(_flatten(out, drop))
        state = bank.save_bytes()
    else:
        kw = case["windows"](po, ctx)
        path = tmp / "windows.state"
        for i, part in enumerate(parts):
            extra: dict[str, Any] = {}
            if split is not None:
                extra = {"save_state": path} if i == 0 else {"load_state": path}
            out = po.stream.with_windows(part, chunk_size=case["chunk"], **kw, **extra)
            if collect:
                outs.append(_flatten(out, drop))
    return (pl.concat(outs, how="vertical_relaxed") if collect else None), state


def worker(args: argparse.Namespace) -> None:
    """``--worker run|time``: one case, read from ``--input``; the outputs to
    ``--out`` and a manifest to ``--meta``, or the seconds of the feed."""
    import tempfile

    warnings.simplefilter("ignore")
    if args.pythonpath:
        sys.path.insert(0, args.pythonpath)
    import polars as pl

    try:
        from rheon import polars as po
    except ImportError:  # a build from before the split: the flat package
        import rheon as po

    native = ""
    try:
        from rheon import _rheon

        native = hashlib.sha256(Path(_rheon.__file__).read_bytes()).hexdigest()[:12]
    except Exception:  # a layout with no native file of its own
        pass
    meta: dict[str, Any] = {
        "file": po.__file__,
        "version": po.__version__,
        "native": native,
        "polars": pl.__version__,
    }
    case = next(
        c for c in load_cases([Path(p) for p in args.cases], args.builtin) if c["name"] == args.case
    )
    df = pl.read_parquet(args.input)
    case = {**case, "rows": df.height}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if args.worker == "run":
                out, state = _run_case(po, case, df, Path(tmp), collect=True)
            else:
                # A run too short to time is repeated, each on a fresh bank; the fastest counts.
                out, state, spent, took = None, None, 0.0, []
                while not took or (spent < args.min_time and len(took) < 20):
                    t0 = time.perf_counter()
                    _run_case(po, case, df, Path(tmp), collect=False)
                    took.append(time.perf_counter() - t0)
                    spent += took[-1]
                meta["seconds"], meta["reps"] = min(took), len(took)
        meta["status"] = "ok"
        if out is not None:
            out.write_parquet(args.out)
            meta["columns"] = out.columns
        if state is not None:
            meta["state_sha"] = hashlib.sha256(state).hexdigest()[:16]
            meta["state_bytes"] = len(state)
    except Exception as e:  # a case this build refuses: not comparable, not a crash
        meta["status"] = "refused"
        meta["error"] = f"{type(e).__name__}: {e}"[:500]
    Path(args.meta).write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _is_floatish(dtype: Any) -> bool:
    import polars as pl

    return dtype in (pl.Float64, pl.Float32)


def _list_inner(dtype: Any) -> Any:
    import polars as pl

    return dtype.inner if isinstance(dtype, (pl.List, pl.Array)) else None


def _float_arrays(a: Any, b: Any) -> tuple[Any, ...]:
    import numpy as np
    import polars as pl

    na, nb = a.is_null().to_numpy(), b.is_null().to_numpy()
    x = np.ascontiguousarray(a.cast(pl.Float64).fill_null(0.0).to_numpy())
    y = np.ascontiguousarray(b.cast(pl.Float64).fill_null(0.0).to_numpy())
    return na, nb, x, y


def _float_stats(a: Any, b: Any, floor: float, rowid: Any, nrows: int) -> dict[str, Any]:
    """Movement of one float column pair of equal length, element by element;
    ``rowid`` maps an element to its row (the identity for a plain column)."""
    import numpy as np

    na, nb, x, y = _float_arrays(a, b)
    nan_x, nan_y = np.isnan(x) & ~na, np.isnan(y) & ~nb
    valid = ~na & ~nb
    bitdiff = (na != nb) | (valid & (x.view(np.uint64) != y.view(np.uint64)) & ~(nan_x & nan_y))
    finite = valid & np.isfinite(x) & np.isfinite(y)
    inf_change = valid & ~nan_x & ~nan_y & (np.isinf(x) | np.isinf(y)) & (x != y)
    idx = np.flatnonzero(finite)
    rel = np.abs(x[idx] - y[idx]) / np.maximum(np.maximum(np.abs(x[idx]), np.abs(y[idx])), floor)
    moved = rel[rel > 0]
    rows = np.zeros(nrows, dtype=bool)
    rows[rowid[bitdiff]] = True
    worst = int(rowid[idx[int(np.argmax(rel))]]) if rel.size and rel.max() > 0 else None
    return {
        "compared": int(finite.sum()),
        "non_null_a": int((~na).sum()),
        "elements": int(na.size),
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "p999_rel": float(np.quantile(rel, 0.999)) if rel.size else 0.0,
        "median_rel": float(np.median(rel)) if rel.size else 0.0,
        "median_rel_moved": float(np.median(moved)) if moved.size else 0.0,
        "max_abs": float(np.max(np.abs(x[idx] - y[idx]))) if idx.size else 0.0,
        "worst_row": worst,
        "nulls_vanish": int((~na & nb).sum()),
        "nulls_appear": int((na & ~nb).sum()),
        "nan_vanish": int((nan_x & ~nan_y).sum()),
        "nan_appear": int((~nan_x & nan_y).sum()),
        "inf_changes": int(inf_change.sum()),
        "_rows": rows,
    }


def column_stats(a: Any, b: Any, floor: float) -> dict[str, Any]:
    """How one column moved between two builds: a float column's relative
    movement, a list of floats' (per element), or, for any other type, the rows
    that differ, which is a decision that flipped. ``_rows`` is the mask of
    rows that differ at all."""
    import numpy as np
    import polars as pl

    n = a.len()
    if a.len() != b.len():
        return {
            "kind": "length",
            "rows_differing": abs(a.len() - b.len()),
            "_rows": np.zeros(0, bool),
        }
    out: dict[str, Any] = {"rows": n}
    ia, ib = _list_inner(a.dtype), _list_inner(b.dtype)
    if _is_floatish(a.dtype) and _is_floatish(b.dtype):
        out.update(kind="float", **_float_stats(a, b, floor, np.arange(n), n))
    elif ia is not None and ib is not None and _is_floatish(ia) and _is_floatish(ib):
        la, lb = a.cast(pl.List(pl.Float64)), b.cast(pl.List(pl.Float64))
        lens_a, lens_b = la.list.len(), lb.list.len()
        shape = ~lens_a.eq_missing(lens_b).to_numpy()
        keep = np.flatnonzero(~shape)
        sub_a, sub_b = la.gather(keep), lb.gather(keep)
        counts = np.maximum(sub_a.list.len().fill_null(0).to_numpy(), 1)
        rowid = np.repeat(keep, counts)
        out.update(kind="list", **_float_stats(_explode(sub_a), _explode(sub_b), floor, rowid, n))
        out["shape_rows"] = int(shape.sum())
        out["_rows"] |= shape
    else:
        out["kind"] = "exact"
        if isinstance(a.dtype, (pl.Enum, pl.Categorical)) or isinstance(
            b.dtype, (pl.Enum, pl.Categorical)
        ):
            a, b = a.cast(pl.String), b.cast(pl.String)
        elif a.dtype != b.dtype:
            try:
                b = b.cast(a.dtype)
            except Exception:
                a, b = a.cast(pl.String), b.cast(pl.String)
        diff = ~a.eq_missing(b).to_numpy()
        out["_rows"] = diff
    out["rows_differing"] = int(out["_rows"].sum())
    return out


def _explode(s: Any) -> Any:
    """A list column's elements, an empty list as one null (the form `compare_release` uses)."""
    try:
        return s.explode(empty_as_null=True)
    except TypeError:
        return s.explode()


def _decision_flips(spec: dict, cols_a: Any, cols_b: Any, names: list[str]) -> list[dict]:
    """Rows where a declared decision differs between the builds."""
    import numpy as np
    import polars as pl

    ops = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
    out = []
    if spec["kind"] == "threshold":
        for name in [c for c in names if fnmatch.fnmatch(c, spec["column"])]:
            sides = []
            for src in (cols_a, cols_b):
                s = src(name).cast(pl.Float64)
                v = np.where(s.is_null().to_numpy(), np.nan, s.fill_null(0.0).to_numpy())
                v = np.abs(v) if spec.get("abs") else v
                with np.errstate(invalid="ignore"):
                    sides.append((ops[spec.get("op", ">")](v, spec["value"]), np.isnan(v)))
            (da, na), (db, nb) = sides
            flips = int(((da != db) | (na != nb)).sum())
            out.append({"name": f"{spec['name']} [{name}]", "flips": flips, "rows": int(da.size)})
    else:
        groups = [c for pat in spec["columns"] for c in names if fnmatch.fnmatch(c, pat)]
        if groups:
            picks = []
            for src in (cols_a, cols_b):
                m = np.column_stack(
                    [src(c).cast(pl.Float64).fill_null(-np.inf).to_numpy() for c in groups]
                )
                picks.append(np.argmax(m, axis=1))
            out.append(
                {
                    "name": spec["name"],
                    "flips": int((picks[0] != picks[1]).sum()),
                    "rows": int(picks[0].size),
                }
            )
    return out


def compare_files(a_path: Path, b_path: Path, case: dict, floor: float) -> dict[str, Any]:
    """The two builds' output files of one case, compared."""
    import numpy as np
    import polars as pl

    sa, sb = pl.read_parquet_schema(a_path), pl.read_parquet_schema(b_path)
    cols = [c for c in sa if c in sb]
    result: dict[str, Any] = {
        "only_in_a": [c for c in sa if c not in sb],
        "only_in_b": [c for c in sb if c not in sa],
        "columns": {},
    }
    row_any: Any = None
    for c in cols:
        a = pl.read_parquet(a_path, columns=[c])[c]
        b = pl.read_parquet(b_path, columns=[c])[c]
        st = column_stats(a, b, floor)
        rows = st.pop("_rows")
        if row_any is None:
            row_any = np.zeros(len(rows), dtype=bool)
        if len(rows) == len(row_any):
            row_any |= rows
        result["columns"][c] = st
    result["rows_any_differ"] = int(row_any.sum()) if row_any is not None else 0

    def reader(path: Path):
        return lambda name: pl.read_parquet(path, columns=[name])[name]

    result["decisions"] = []
    for spec in case["decisions"]:
        result["decisions"] += _decision_flips(spec, reader(a_path), reader(b_path), cols)
    return result


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """A case's row of the table, from its columns."""
    cols = result["columns"]
    floaty = {c: s for c, s in cols.items() if s["kind"] in ("float", "list")}
    exact = {c: s for c, s in cols.items() if s["kind"] == "exact"}
    moved = [c for c, s in floaty.items() if s["rows_differing"]]
    worst = max(floaty, key=lambda c: floaty[c]["max_rel"], default=None)
    flips = {c: s["rows_differing"] for c, s in exact.items() if s["rows_differing"]}
    return {
        "float_columns": len(floaty),
        # Evidence the case reaches the model: the share of values in A that are not null.
        "live_share": (
            statistics.median(s["non_null_a"] / max(s["elements"], 1) for s in floaty.values())
            if floaty
            else 0.0
        ),
        "float_columns_moved": len(moved),
        "worst_column": worst if worst and floaty[worst]["max_rel"] > 0 else None,
        "max_rel": max((s["max_rel"] for s in floaty.values()), default=0.0),
        "p999_rel": max((s["p999_rel"] for s in floaty.values()), default=0.0),
        "median_rel": max((s["median_rel"] for s in floaty.values()), default=0.0),
        "rows_any_differ": result["rows_any_differ"],
        "nulls_vanish": sum(s.get("nulls_vanish", 0) for s in floaty.values()),
        "nulls_appear": sum(s.get("nulls_appear", 0) for s in floaty.values()),
        "nonfinite_changes": sum(
            s.get("nan_vanish", 0) + s.get("nan_appear", 0) + s.get("inf_changes", 0)
            for s in floaty.values()
        ),
        "shape_changes": sum(s.get("shape_rows", 0) for s in floaty.values()),
        "exact_columns": len(exact),
        "flipped_columns": flips,
        "decision_flips": {d["name"]: d["flips"] for d in result["decisions"] if d["flips"]},
        "added_columns": len(result["only_in_b"]),
        "removed_columns": len(result["only_in_a"]),
    }


# ---------------------------------------------------------------------------
# Running builds
# ---------------------------------------------------------------------------


def resolve_python(spec: str) -> Path:
    """An interpreter from a venv's folder, a ``python`` path, or ``release:X.Y.Z``."""
    if spec.startswith("release:"):
        sys.path.insert(0, str(SELF.parent))
        import compare_release

        return compare_release.released_python(spec.split(":", 1)[1])
    p = Path(spec).expanduser()
    if p.is_dir():
        for rel in ("bin/python", "Scripts/python.exe"):
            if (p / rel).exists():
                return p / rel
        raise SystemExit(f"{spec}: no interpreter inside")
    return p


class Build:
    def __init__(self, label: str, python: Path, pythonpath: str | None, threads: int) -> None:
        self.label, self.python, self.pythonpath, self.threads = label, python, pythonpath, threads
        self.meta: dict[str, Any] = {}
        self.min_time = 1.5

    def run(
        self,
        mode: str,
        case: dict,
        files: list[Path],
        builtin: bool,
        inp: Path,
        meta: Path,
        out: Path | None,
        timeout: float,
    ) -> dict[str, Any]:
        cmd = [
            str(self.python),
            "-I",
            str(SELF),
            "--worker",
            mode,
            "--case",
            case["name"],
            "--input",
            str(inp),
            "--meta",
            str(meta),
        ]
        if out:
            cmd += ["--out", str(out)]
        for f in files:
            cmd += ["--cases", str(f)]
        if builtin:
            cmd += ["--builtin"]
        if self.pythonpath:
            cmd += ["--pythonpath", self.pythonpath]
        cmd += ["--min-time", str(self.min_time)]
        env = {
            **os.environ,
            "POLARS_MAX_THREADS": str(self.threads),
            "RAYON_NUM_THREADS": str(self.threads),
        }
        meta.unlink(missing_ok=True)
        try:
            done = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if done.returncode != 0 or not meta.exists():
            return {"status": "crashed", "error": (done.stderr or done.stdout)[-600:]}
        return json.loads(meta.read_text(encoding="utf-8"))


def _load() -> float:
    """The 1-minute load average, or NaN where the platform has none (Windows)."""
    getloadavg = getattr(os, "getloadavg", None)
    return getloadavg()[0] if getloadavg is not None else float("nan")


def wait_for_load(limit: float, patience: float) -> float:
    """The 1-minute load once it is under ``limit`` (or after ``patience`` seconds)."""
    start = time.monotonic()
    load = _load()
    while load > limit and time.monotonic() - start < patience:
        time.sleep(15)
        load = _load()
    return load


def abba(
    builds: tuple[Build, Build], rounds: int, once: Any, limit: float, patience: float
) -> dict[str, Any]:
    """``rounds`` rounds in the order A B, B A, A B, ...; the seconds each build took."""
    secs: dict[str, list[float]] = {b.label: [] for b in builds}
    loads: list[float] = []
    for r in range(rounds):
        for build in builds if r % 2 == 0 else builds[::-1]:
            loads.append(wait_for_load(limit, patience))
            took = once(build)
            if took is None:
                return {"error": f"{build.label} failed in the timing run"}
            secs[build.label].append(took)
    a, b = (secs[x.label] for x in builds)
    loads.append(_load())
    return {
        "rounds": rounds,
        "seconds": secs,
        "min": [min(a), min(b)],
        "median": [statistics.median(a), statistics.median(b)],
        "ratio_min": min(b) / min(a) if min(a) else None,
        "ratio_median": statistics.median(b) / statistics.median(a)
        if statistics.median(a)
        else None,
        "load_max": max(loads),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return "0" if x == 0 else f"{x:.2e}"


def table_for(item: str, results: list[dict], floor: float, builds: tuple[Build, Build]) -> str:
    lines = [
        f"## {item}",
        "",
        "Relative movement = |a-b| / max(|a|, |b|, "
        f"{floor:g}); max, 99.9th percentile and median are the worst column's, over the "
        "rows where both are finite. `live` is the median over float columns of the share of "
        "A's values that are not null: a case that never leaves warm-up shows it.",
        "",
        "| case | rows | float cols (moved) | live | max | p99.9 | median | rows differing "
        "| nulls -/+ "
        "| flips | state | time A min/med | time B min/med | B/A min, med |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail: list[str] = []
    for r in results:
        name = r["name"]
        if r["status"] != "ok":
            lines.append(
                f"| {name} | | {r['status']}: {r.get('why', '')[:80]} | | | | | | | | | | |"
            )
            continue
        s, t = r["summary"], r.get("timing") or {}
        flips = (
            sum(s["flipped_columns"].values())
            + sum(s["decision_flips"].values())
            + s["shape_changes"]
        )
        state = r.get("state", "-")
        if t and "min" in t:
            ta = f"{t['min'][0]:.3f} / {t['median'][0]:.3f}"
            tb = f"{t['min'][1]:.3f} / {t['median'][1]:.3f}"
            ratio = f"{t['ratio_min']:.3f}, {t['ratio_median']:.3f}"
        else:
            ta = tb = ratio = "-"
        lines.append(
            f"| {name} | {r['rows']:,} | {s['float_columns']} ({s['float_columns_moved']}) | "
            f"{s['live_share']:.0%} | "
            f"{fmt(s['max_rel'])} | {fmt(s['p999_rel'])} | {fmt(s['median_rel'])} | "
            f"{s['rows_any_differ']:,} | {s['nulls_vanish']}/{s['nulls_appear']} | {flips} | "
            f"{state} | {ta} | {tb} | {ratio} |"
        )
        notes = []
        if s["added_columns"] or s["removed_columns"]:
            notes.append(
                f"columns only in B: {s['added_columns']}, only in A: {s['removed_columns']}"
            )
        if s["flipped_columns"]:
            notes.append(
                "flipped (exact) columns: "
                + ", ".join(f"`{c}` {n}" for c, n in s["flipped_columns"].items())
            )
        if s["decision_flips"]:
            notes.append(
                "flipped decisions: "
                + ", ".join(f"{c} {n}" for c, n in s["decision_flips"].items())
            )
        if s["nonfinite_changes"]:
            notes.append(f"NaN/infinity changes: {s['nonfinite_changes']}")
        moved = sorted(
            (
                (c, st)
                for c, st in r["columns"].items()
                if st["kind"] in ("float", "list") and st["rows_differing"]
            ),
            key=lambda cs: -cs[1]["max_rel"],
        )
        if notes or moved:
            detail.append(f"**{name}**")
            detail += [f"- {n}" for n in notes]
            if moved:
                detail += [
                    "",
                    "| column | max | p99.9 | median | median (moved) | rows differing "
                    "| nulls -/+ | worst row |",
                    "|---|---|---|---|---|---|---|---|",
                ]
                for c, st in moved[:25]:
                    detail.append(
                        f"| `{c}` | {fmt(st['max_rel'])} | {fmt(st['p999_rel'])} | "
                        f"{fmt(st['median_rel'])} | "
                        f"{fmt(st['median_rel_moved'])} | {st['rows_differing']:,} | "
                        f"{st['nulls_vanish']}/{st['nulls_appear']} | {st['worst_row']} |"
                    )
                if len(moved) > 25:
                    detail.append(
                        f"| ... {len(moved) - 25} more columns in report.json | | | | | | | |"
                    )
            detail.append("")
    if any(
        "load" in (r.get("timing") or {}) or (r.get("timing") or {}).get("load_max")
        for r in results
    ):
        loads = [
            r["timing"]["load_max"]
            for r in results
            if r.get("timing") and "load_max" in r["timing"]
        ]
        lines += ["", f"1-minute load while timing: up to {max(loads):.1f}."]
    return "\n".join(lines + [""] + detail)


def write_reports(
    out: Path, results: list[dict], floor: float, builds: tuple[Build, Build], header: dict
) -> str:
    items: dict[str, list[dict]] = {}
    for r in results:
        items.setdefault(r["item"], []).append(r)
    head = [
        "# Output comparison",
        "",
        *(
            f"- {b.label}: `{b.meta.get('file', '?')}` version {b.meta.get('version', '?')}, "
            f"native {b.meta.get('native') or '?'}, polars {b.meta.get('polars', '?')}"
            for b in builds
        ),
        f"- floor {floor:g}, threads {builds[0].threads}, {header['when']}",
        "",
    ]
    text = []
    for item, rs in items.items():
        md = table_for(item, rs, floor, builds)
        (out / f"item-{item}.md").write_text("\n".join(head) + "\n" + md, encoding="utf-8")
        text.append(md)
    summary = "\n".join(head) + "\n" + "\n".join(text)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps({"header": header, "floor": floor, "cases": results}, indent=1), encoding="utf-8"
    )
    return summary


def moved_anything(results: list[dict]) -> bool:
    for r in results:
        if r["status"] != "ok":
            continue
        s = r["summary"]
        if (
            s["rows_any_differ"]
            or s["decision_flips"]
            or s["flipped_columns"]
            or s["added_columns"]
            or s["removed_columns"]
            or r.get("state") == "differs"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", help="build A: a venv, its python, or release:X.Y.Z")
    ap.add_argument("--b", help="build B, as --a")
    ap.add_argument("--a-path", help="a directory holding rheon, put first on A's path")
    ap.add_argument("--b-path", help="the same for B")
    ap.add_argument("--a-label")
    ap.add_argument("--b-label")
    ap.add_argument("--cases", action="append", default=[], help="a case file or a folder of them")
    ap.add_argument("--builtin", action="store_true", help="include the built-in smoke cases")
    ap.add_argument("--only", action="append", default=[], help="a glob or substring of case names")
    ap.add_argument("--item", action="append", default=[], help="an item to run")
    ap.add_argument("--list", action="store_true", help="list the selected cases and stop")
    ap.add_argument("--rows-scale", type=float, default=1.0, help="scale every case's rows")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--time-rounds", type=int, default=5, help="ABBA rounds; 0 skips timing")
    ap.add_argument("--threads", type=int, default=2, help="POLARS_MAX_THREADS of every run")
    ap.add_argument(
        "--max-load", type=float, default=12.0, help="wait before a timed run above this load"
    )
    ap.add_argument("--load-patience", type=float, default=600.0)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds a run may take")
    ap.add_argument("--out", default=str(REPO / ".cache" / "compare-outputs" / "report"))
    ap.add_argument("--work", default=str(REPO / ".cache" / "compare-outputs" / "work"))
    ap.add_argument(
        "--resume", action="store_true", help="keep the finished cases already in --out"
    )
    ap.add_argument("--keep-outputs", action="store_true", help="keep each build's output files")
    ap.add_argument(
        "--require-identical", action="store_true", help="exit 1 if anything moved or flipped"
    )
    # The worker's own flags.
    ap.add_argument("--worker", choices=["run", "time"])
    ap.add_argument("--case")
    ap.add_argument("--input")
    ap.add_argument("--meta")
    ap.add_argument("--pythonpath")
    ap.add_argument("--min-time", type=float, default=1.5, help="seconds of repeats in a timed run")
    return ap.parse_args(argv)


def run_case_pair(
    case: dict,
    files: list[Path],
    args: argparse.Namespace,
    builds: tuple[Build, Build],
    work: Path,
    out: Path,
) -> dict[str, Any]:
    """One case under both builds: the comparison, then the timing."""
    import polars as pl

    safe = case["name"].replace("/", "__")
    result: dict[str, Any] = {"name": case["name"], "item": case["item"], "rows": case["rows"]}
    inp = work / "inputs" / f"{data_key(case)}.parquet"
    inp.parent.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        tmp = inp.with_suffix(f".{os.getpid()}.tmp")
        make_input(case).write_parquet(tmp)
        tmp.rename(inp)
    runs = work / "runs" / safe
    runs.mkdir(parents=True, exist_ok=True)
    metas, outs = {}, {}
    for build in builds:
        outs[build.label] = runs / f"{build.label}.parquet"
        metas[build.label] = build.run(
            "run",
            case,
            files,
            args.builtin,
            inp,
            runs / f"{build.label}.json",
            outs[build.label],
            args.timeout,
        )
        for key in ("file", "version", "native", "polars"):
            if key in metas[build.label]:
                build.meta[key] = metas[build.label][key]
    bad = [(b.label, metas[b.label]) for b in builds if metas[b.label].get("status") != "ok"]
    if bad:
        result["status"] = "not comparable"
        result["why"] = "; ".join(
            f"{label}: {m.get('status')} {m.get('error', '')}" for label, m in bad
        )
        return result
    if builds[0].meta["file"] == builds[1].meta["file"]:
        raise SystemExit(
            f"both builds imported {builds[0].meta['file']}: nothing would be compared"
        )
    cmp = compare_files(outs[builds[0].label], outs[builds[1].label], case, args.floor)
    result.update(status="ok", summary=summarize(cmp), **cmp)
    sa, sb = (metas[b.label].get("state_sha") for b in builds)
    result["state"] = "-" if sa is None and sb is None else ("same" if sa == sb else "differs")
    if not args.keep_outputs:
        for p in outs.values():
            p.unlink(missing_ok=True)
    if args.time_rounds and case["timed"]:

        def once(build: Build) -> float | None:
            m = build.run(
                "time",
                case,
                files,
                args.builtin,
                inp,
                runs / f"{build.label}.time.json",
                None,
                args.timeout,
            )
            return m.get("seconds") if m.get("status") == "ok" else None

        result["timing"] = abba(builds, args.time_rounds, once, args.max_load, args.load_patience)
    del pl
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.worker:
        worker(args)
        return 0
    files = [Path(c).resolve() for c in args.cases]
    cases = select(load_cases(files, args.builtin or not files), args.only, args.item)
    cases = [scaled(c, args.rows_scale) for c in cases]
    if args.list:
        for c in cases:
            print(
                f"{c['item']:12s} {c['name']:50s} rows {c['rows']:>9,} k {c['k']:>3} "
                f"targets {c['targets']} "
                f"groups {c['groups']:>6} chunk {c['chunk']}"
            )
        return 0
    if not args.a or not args.b:
        raise SystemExit("--a and --b are required")
    if not cases:
        raise SystemExit("no case selected")
    args.builtin = args.builtin or not files
    builds = (
        Build(args.a_label or "A", resolve_python(args.a), args.a_path, args.threads),
        Build(args.b_label or "B", resolve_python(args.b), args.b_path, args.threads),
    )
    for build in builds:
        build.min_time = args.min_time
    out, work = Path(args.out), Path(args.work)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    load_start = _load()
    for case in cases:
        done = out / "cases" / (case["name"].replace("/", "__") + ".json")
        if args.resume and done.exists():
            results.append(json.loads(done.read_text(encoding="utf-8")))
            print(f"{case['name']}: kept", flush=True)
            continue
        t0 = time.monotonic()
        res = run_case_pair(case, files, args, builds, work, out)
        done.write_text(json.dumps(res), encoding="utf-8")
        results.append(res)
        s = res.get("summary")
        print(
            f"{case['name']}: {res['status']}"
            + (
                f", max movement {fmt(s['max_rel'])}, {s['rows_any_differ']} rows differ"
                if s
                else ""
            )
            + f" ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
    header = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "load_at_start": load_start,
        "load_at_end": _load(),
        "argv": sys.argv[1:],
    }
    print(write_reports(out, results, args.floor, builds, header))
    return 1 if args.require_identical and moved_anything(results) else 0


if __name__ == "__main__":
    sys.exit(main())
