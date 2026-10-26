"""The oracle ledger's two calls (docs/TESTING.md, "The oracle ledger").

The heavy grids against a third-party oracle (``oracle_grid``) run on CI,
not in the gate. Two things keep them from drifting:

- ``oracles.toml``, the ledger, holds each case's error against its oracle
  -- the worst seed and the 95th percentile -- and the oracle's version.
  It only tightens, as ``budgets.toml`` does (``scripts/oracles.py``).
- ``oracle_hashes.toml`` holds a hash of this library's outputs on each
  group's inputs, which the gate recomputes in seconds
  (``tests/test_oracle_ledger.py``). While it is unchanged, the comparison
  still holds: the oracle's answer is fixed by its pinned version.

A grid registers in two lines. At module level, the group's inputs -- a
callable returning this library's outputs on them, with no oracle in it --
and the libraries that compute the oracle's answer::

    ledger.guard("kmeans_blobs", lambda: fit_kmeans(...), oracles=["scikit-learn"])

and in the test, each quantity it bounds, one value a seed::

    ledger.record("kmeans_blobs", "raw_ari", aris, ideal=1.0)

A group's id is the module's name without ``test_``, a dot, and the name
given: ``truth_recovery.kmeans_blobs``. A case is judged on its *gap*,
``|value - ideal|``: ``ideal`` is 0 for an error and 1 for a score such as
an ARI, a share or an F1. With ``POLARS_ONLINE_ORACLES_OUT`` naming a
directory, ``record`` appends what it measured there as JSON lines, one
file a process, for ``scripts/oracles.py check`` and ``apply``; otherwise
it only checks its arguments.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

#: The directory ``record`` writes to, when set.
OUT_VARIABLE = "POLARS_ONLINE_ORACLES_OUT"

#: The significant bits a float keeps in a hash (``canonical``).
BITS = 24
#: Values under this share of their array's largest are hashed as 0.
ZERO_BELOW = 2.0**-40

#: The modules with an ``oracle_grid`` test that do not register yet, each
#: with why (``tests/test_oracle_ledger.py`` holds the list to the modules).
#: Empty: every module with a grid registers.
PENDING: dict[str, str] = {}


@dataclass
class Group:
    """One group's inputs: ``outputs`` computes this library's outputs on
    them, and ``oracles`` names the libraries whose answer the grid takes
    as given (distribution names, as ``uv.lock`` spells them)."""

    gid: str
    module: str
    outputs: Callable[[], Any]
    oracles: tuple[str, ...] = ()
    dataset: str = ""
    bits: int = BITS
    cases: set[str] = field(default_factory=set)


#: Every group registered so far, by id.
GROUPS: dict[str, Group] = {}


def _module_name(depth: int = 2) -> str:
    return sys._getframe(depth).f_globals["__name__"]


def group_id(module: str, name: str) -> str:
    """``test_truth_recovery`` and ``kmeans_blobs`` make
    ``truth_recovery.kmeans_blobs``."""
    short = module.rsplit(".", 1)[-1].removeprefix("test_")
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"a group's name is lower case, digits and _: {name!r}")
    return f"{short}.{name}"


def guard(
    name: str,
    outputs: Callable[[], Any],
    *,
    oracles: Iterable[str] = (),
    dataset: str = "",
    bits: int = BITS,
) -> None:
    """Register a group: ``outputs`` returns this library's outputs on the
    group's inputs -- arrays, frames, numbers, nested in lists, tuples and
    dicts -- and must not call the oracle. ``bits`` is the precision each
    float keeps in the hash: the default holds across platforms; 52, the
    double itself, is for a group whose oracle agrees to round-off, where a
    change in the last bits is a change the ledger would see."""
    module = _module_name()
    gid = group_id(module, name)
    if gid in GROUPS and GROUPS[gid].module != module:
        raise ValueError(f"{gid} is registered twice")
    if not 8 <= bits <= 52:
        raise ValueError(f"bits = {bits}: from 8 to 52")
    GROUPS[gid] = Group(gid, module, outputs, tuple(oracles), dataset, bits)


def versions(oracles: Iterable[str]) -> dict[str, str]:
    """The installed version of each library."""
    return {name: importlib.metadata.version(name) for name in oracles}


def summarize(values: Any, ideal: float = 0.0) -> tuple[float, float, int]:
    """The value of the seed with the largest gap from ``ideal``, the value
    of the seed at the nearest-rank 95th percentile of the gaps, and how
    many there are."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("no values to record")
    if not np.isfinite(v).all():
        raise ValueError(f"a value is not finite: {v[~np.isfinite(v)][:3]}")
    gap = np.abs(v - ideal)
    order = np.argsort(gap, kind="stable")
    rank = max(1, math.ceil(0.95 * v.size))
    return float(v[order[-1]]), float(v[order[rank - 1]]), int(v.size)


def record(
    name: str,
    case: str,
    values: Any,
    *,
    ideal: float = 0.0,
    floor: float = 0.0,
    exact: bool = False,
) -> tuple[float, float]:
    """Record one quantity a grid bounds, one value a seed (or one value):
    its worst and its 95th percentile by the gap ``|value - ideal|``, which
    it returns. ``floor`` is a gap below which differences are round-off:
    a gap under it counts as it. ``exact`` marks a count -- rows of delay,
    cap hits, alarms -- which the ledger holds with no tolerance: one more
    on the worst seed is a regression, however large the count."""
    gid = group_id(_module_name(), name)
    group = GROUPS.get(gid)
    if group is None:
        raise ValueError(f"{gid}: no ledger.guard registers its inputs in this module")
    if not re.fullmatch(r"[a-z0-9_]+", case):
        raise ValueError(f"a case's name is lower case, digits and _: {case!r}")
    if exact:
        v = np.asarray(values, dtype=float)
        if floor or not np.array_equal(v, np.round(v)):
            raise ValueError(f"{gid}.{case}: an exact case is a count, whole and with no floor")
    worst, p95, n = summarize(values, ideal)
    out = os.environ.get(OUT_VARIABLE)
    if out:
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        nodeid = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        line = {
            "case": f"{gid}.{case}",
            "group": gid,
            "worst": worst,
            "p95": p95,
            "n": n,
            "ideal": float(ideal),
            "floor": float(floor),
            **({"tolerance": 0.0} if exact else {}),
            "test": nodeid,
            "oracles": versions(group.oracles),
            "dataset": group.dataset,
            "platform": platform_tag(),
        }
        with open(path / f"records-{os.getpid()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    return worst, p95


# --- the hash ------------------------------------------------------------------


def platform_tag() -> str:
    """``darwin-arm64``, ``linux-x86_64``: the system and the processor,
    whose arithmetic the numbers and the hashes are. Not Python's version:
    ``ast.unparse`` writes the tests' source into the hash, and on each of
    CI's systems Python 3.12, 3.13 and 3.14 gave every group the same hash
    (2026-10-26)."""
    return f"{sys.platform}-{platform.machine().lower()}"


def _floats(a: np.ndarray, bits: int) -> bytes:
    """A float array as bytes that survive the last bits' differences
    between platforms: values under ``ZERO_BELOW`` of the largest set to 0,
    the rest rounded to ``bits`` significant bits, NaN and the infinities
    kept as themselves."""
    a = np.asarray(a, dtype=np.float64).ravel()
    finite = np.isfinite(a)
    out = np.where(finite, a, 0.0)
    big = float(np.abs(out).max()) if out.size else 0.0
    out = np.where(np.abs(out) < big * ZERO_BELOW, 0.0, out)
    if bits < 52:
        m, e = np.frexp(out)
        out = np.ldexp(np.round(np.ldexp(m, bits)), e - bits)
    out = out + 0.0  # -0.0 is 0.0
    tags = np.where(np.isnan(a), 1, np.where(np.isposinf(a), 2, np.where(np.isneginf(a), 3, 0)))
    return out.astype("<f8").tobytes() + tags.astype(np.uint8).tobytes()


def canonical(obj: Any, bits: int = BITS, h: Any = None) -> Any:
    """Feed ``obj`` into the hash ``h`` (a new sha256 when not given) in a
    form that names each part's kind and shape, every float through
    ``_floats``; returns ``h``."""
    import polars as pl

    h = hashlib.sha256() if h is None else h

    def tag(t: str) -> None:
        h.update(t.encode() + b"\0")

    if isinstance(obj, pl.DataFrame):
        tag(f"frame{obj.columns}")
        for name in obj.columns:
            canonical(obj[name], bits, h)
    elif isinstance(obj, pl.Series):
        tag(f"series:{obj.name}:{obj.dtype}")
        if obj.dtype == pl.Struct:
            canonical(obj.struct.unnest(), bits, h)
        elif obj.dtype.is_numeric() or obj.dtype == pl.Boolean:
            tag(f"nulls{obj.is_null().to_list()}")
            canonical(obj.cast(pl.Float64).fill_null(0.0).to_numpy(), bits, h)
        else:
            canonical(obj.to_list(), bits, h)
    elif isinstance(obj, dict):
        tag(f"dict{len(obj)}")
        for k in sorted(obj, key=str):
            tag(str(k))
            canonical(obj[k], bits, h)
    elif isinstance(obj, (list, tuple)) and not all(
        isinstance(x, (int, float, bool, np.number)) and not isinstance(x, str) for x in obj
    ):
        tag(f"list{len(obj)}")
        for x in obj:
            canonical(x, bits, h)
    elif obj is None or isinstance(obj, str):
        tag(f"{type(obj).__name__}:{obj}")
    else:
        a = np.asarray(obj)
        if a.dtype.kind in "biu":
            tag(f"int{a.shape}")
            h.update(a.astype("<i8").tobytes())
        elif a.dtype.kind == "f":
            tag(f"float{a.shape}")
            h.update(_floats(a, bits))
        else:
            raise TypeError(f"no canonical form for {type(obj).__name__} of {a.dtype}")
    return h


def _drop_docstrings(node: ast.AST) -> ast.AST:
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            n.body = body[1:] or [ast.Pass()]
    return node


def _defines(node: ast.stmt) -> set[str]:
    """The module-level names a statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def source_of(test: str) -> str:
    """The source of the test a node id names -- its function, or its whole
    class for a method -- with every module-level function, class and
    constant it reaches by name, transitively, each with its docstrings
    dropped and in ``ast.unparse``'s normal form: a change to what a grid
    computes or asserts, or to a helper it calls, moves its group's hash; a
    change to its prose, or to a neighbour it does not call, does not."""
    path, *names = test.split("::")
    names = [re.sub(r"\[.*\]$", "", n) for n in names]
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    top = {}
    for stmt in tree.body:
        for name in _defines(stmt):
            top.setdefault(name, []).append(stmt)
    root = top[names[0]][-1]
    seen: list[ast.stmt] = []
    todo = [root]
    while todo:
        stmt = todo.pop()
        if any(stmt is s for s in seen):
            continue
        seen.append(stmt)
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id in top:
                todo += top[n.id]
    order = {id(s): i for i, s in enumerate(tree.body)}
    seen.sort(key=lambda s: order[id(s)])
    return "\n".join(ast.unparse(_drop_docstrings(s)) for s in seen)


def digest(group: Group, tests: Iterable[str] = ()) -> str:
    """The group's hash: its outputs, quantized, and the source of the tests
    that record its cases."""
    h = hashlib.sha256()
    h.update(f"bits={group.bits}\0".encode())
    with warnings.catch_warnings():
        # A notice the inputs draw is the grid's to read, not the hash's.
        warnings.simplefilter("ignore")
        outputs = group.outputs()
    canonical(outputs, group.bits, h)
    for test in sorted(set(tests)):
        h.update(test.encode() + b"\0" + source_of(test).encode() + b"\0")
    return h.hexdigest()


# --- finding the groups ----------------------------------------------------------

_GUARD = re.compile(r"""^ledger\.guard\(\s*["']([a-z0-9_]+)["']""", re.M)


def registered_modules() -> dict[str, list[str]]:
    """The test modules that register groups, and the names each registers,
    read from the source without importing it."""
    out = {}
    for path in sorted(TESTS.glob("test_*.py")):
        names = _GUARD.findall(path.read_text(encoding="utf-8"))
        if names:
            out[path.stem] = names
    return out


def load(module: str) -> dict[str, Group]:
    """Import ``module`` and return the groups it registers."""
    importlib.import_module(module)
    return {g: grp for g, grp in GROUPS.items() if grp.module == module}
