"""A state written by a released wheel loads in this build and goes on as this
build's own would (docs/PLAN.md task 109). The CHANGELOG promises that files
from earlier schemas load, and hard rule 5 keeps a loader for each; until
this test, only states this build wrote itself were ever loaded here.

Each release is installed from PyPI into the cache
``scripts/compare_release.py`` keeps (``.cache/release-compare/<version>``),
and ``scripts/release_probe.py --states`` runs under it: every spec of the
release comparison's workload -- all twenty-one kinds -- fitted on the first
half of its stream and saved. This build loads each file with its own spec
and fits the second half, beside a bank of its own that fitted both halves.

Measured 2026-09-27: 0.11.1's files (schema 17) go on bit for bit; 0.10.0's
(schema 14) within 2.2e-13, relative to ``1 + |v|``, where the means' low
parts, which schema 14 does not carry, restart at zero. A numeric change a
CHANGELOG declares moves this test as it moves the release comparison.
Downloading needs the network, so offline the test is skipped (hard rule 1).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po

REPO = Path(__file__).resolve().parents[1]

#: The oldest release whose files the loaders promise, and the latest.
RELEASES = ["0.10.0", "0.11.1"]

# The workload's specs are built to reach every path, not to be ready: a
# readiness warning about one of them is expected and says nothing here.
pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_release")
probe = _load("release_probe")


def _online() -> bool:
    try:
        urllib.request.urlopen("https://pypi.org/simple/polars-online/", timeout=15).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    return True


@pytest.fixture(scope="module", params=RELEASES)
def released(request, tmp_path_factory):
    """The version, its manifest, and the directory its states are in."""
    if not _online():
        pytest.skip("offline: the released wheel cannot be downloaded")
    version = request.param
    python = compare.released_python(version)
    out = tmp_path_factory.mktemp(f"states-{version}")
    manifest = out / "manifest.json"
    # `-I`: the released build imports its own package, not this checkout's.
    subprocess.run(
        [
            str(python),
            "-I",
            str(REPO / "scripts" / "release_probe.py"),
            "--states",
            str(out),
            str(manifest),
        ],
        check=True,
    )
    return version, json.loads(manifest.read_text(encoding="utf-8")), out


def _floats(frame: pl.DataFrame, prefix: str = "") -> dict[str, np.ndarray]:
    """Every float in a frame, by column, with structs and lists opened."""
    out: dict[str, np.ndarray] = {}
    for name in frame.columns:
        s = frame[name]
        if s.dtype == pl.Struct:
            out.update(_floats(s.struct.unnest(), f"{prefix}{name}."))
            continue
        if isinstance(s.dtype, pl.List):
            # Flattened by hand, not with `explode`, whose default polars is
            # changing: nulls stay NaN, empty lists add nothing.
            flat = _flatten(s.to_list())
            if flat and all(isinstance(v, float) for v in flat):
                out[f"{prefix}{name}"] = np.array(flat, dtype=float)
            continue
        if s.dtype == pl.Float64:
            out[f"{prefix}{name}"] = s.to_numpy()
    return out


def _flatten(values: list) -> list:
    out: list = []
    for v in values:
        if isinstance(v, list):
            out.extend(_flatten(v))
        else:
            out.append(float("nan") if v is None else v)
    return out


def _worst_gap(got: pl.DataFrame, want: pl.DataFrame) -> float:
    a, b = _floats(got), _floats(want)
    assert a.keys() == b.keys()
    worst = 0.0
    for name in a:
        u, v = a[name], b[name]
        assert u.shape == v.shape, name
        np.testing.assert_array_equal(np.isnan(u), np.isnan(v), err_msg=name)
        ok = np.isfinite(v)
        if ok.any():
            worst = max(worst, float((np.abs(u[ok] - v[ok]) / (1 + np.abs(v[ok]))).max()))
    return worst


def _tables(bank: po.ModelBank, name: str) -> list[pl.DataFrame]:
    """What a model keeps and a row does not show: the rows a closed group
    emits, the coefficients where the kind has them (the rest refuse by
    name), and `marginal`'s pairs table."""
    out = [bank.closed_groups()]
    try:
        out.append(bank.coef(name))
    except ValueError as e:
        assert "coef" in str(e) or "coefficients" in str(e), e
    if name == "marginal":
        out.append(bank.marginal(name))
    return out


def test_every_spec_was_saved_by_the_release_at_an_older_schema(released):
    version, manifest, _ = released
    assert manifest["version"] == version
    assert manifest["refused"] == {}
    assert manifest["saved"] == [name for name, *_ in probe.WORKLOAD]
    # An older schema, so a loader, not the current reader, is what runs.
    assert manifest["schema"] < po.schema_version()


def test_a_released_state_loads_and_goes_on_as_this_builds_own(released):
    version, manifest, states = released
    df = probe.stream()
    first, second = df.head(probe.HALF), df.slice(probe.HALF)
    built = {name: getattr(po.spec, b)(name, **kw) for name, b, kw, _ in probe.WORKLOAD}
    gaps = {}
    for name, _, _, reads in probe.WORKLOAD:
        specs = [built[r] for r in reads] + [built[name]]
        loaded = po.ModelBank.load(states / f"{name}.state", specs=specs)
        own = po.ModelBank(specs)
        own.fit_predict(first)
        gap = _worst_gap(
            loaded.fit_predict(second).select(name), own.fit_predict(second).select(name)
        )
        for got, want in zip(_tables(loaded, name), _tables(own, name), strict=True):
            assert got.height == want.height, name
            if got.height:
                gap = max(gap, _worst_gap(got, want))
        gaps[name] = gap
    assert max(gaps.values()) <= 1e-12, {k: f"{v:.1e}" for k, v in gaps.items() if v > 1e-12}
    print(f"{version}: worst {max(gaps.values()):.1e}", file=sys.stderr)
