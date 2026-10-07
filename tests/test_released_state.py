"""A state written by a released wheel meets this build as the release policy
says it will (docs/PLAN.md tasks 109 and 198).

Each release is installed from PyPI into the cache
``scripts/compare_release.py`` keeps (``.cache/release-compare/<version>``),
and ``scripts/release_probe.py --states`` runs under it: every spec of the
release comparison's workload -- all twenty-one kinds -- fitted on the first
half of its stream and saved.

**Before 1.0, a released file is refused by its version.** The bank's minimum
schema (``MIN_BANK_SCHEMA_VERSION``, crates/online-polars/src/bank.rs) is the
schema this build writes, above every release's: 0.10.0 wrote 14, 0.11.1 17,
0.12.0 19 and 0.13.0 20. Each such file is refused naming the way out, and
``test_a_pre_1_0_state_is_refused_by_its_version`` holds each to that.

**From 1.0, a file a 1.x release wrote loads in every later 1.x and goes on as
this build's own** (review round 4, D1, CI3 and TB3): this build loads it with
its own spec and fits the second half, beside a bank of its own that fitted
both halves, and every float, the closed groups, the coefficients and
``marginal``'s table agree. Until 1.0.0 is released and listed, that test runs
on the states this build writes itself, so the harness exists, and has run,
before the first release that must honour it. A numeric change a CHANGELOG
declares moves it as it moves the release comparison.

Until 2026-09-28 the loading test held 0.11.1's files (schema 17) bit for
bit and 0.10.0's (schema 14) within 2.2e-13; the pre-1.0 waiver (the user,
"Do not worry about old specs") made them refusals. Downloading needs the
network, so a release is skipped offline (hard rule 1); this build's own
states are not.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import is_transient

REPO = Path(__file__).resolve().parents[1]

#: Every release from 0.10.0, the oldest whose files a loader once promised,
#: to the latest. Each wrote a schema of its own, so a release joins the list
#: when it ships (docs/RELEASE-READINESS.md, "The steps of a release").
RELEASES = ["0.10.0", "0.11.1", "0.12.0", "0.13.0"]


def _major(version: str) -> int:
    return int(version.split(".")[0])


#: The releases before 1.0, whose files are refused by their version.
REFUSED = [v for v in RELEASES if _major(v) < 1]

#: The releases from 1.0, whose files load and go on; and this build, whose
#: own files stand in until 1.0.0 is listed and run beside them after.
THIS_BUILD = "this build"
LOADED = [v for v in RELEASES if _major(v) >= 1] + [THIS_BUILD]

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
    """Whether PyPI answers. An HTTP error is an ``OSError``, and a 404 here
    is PyPI saying the package is gone, so only what a retry could fix reads
    as offline (`data.is_transient`, review 2026-10-05, TA3)."""
    try:
        urllib.request.urlopen("https://pypi.org/simple/polars-online/", timeout=15).close()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if not is_transient(e):
            raise
        return False
    return True


def _states(version: str, out: Path) -> tuple[dict, Path]:
    """The workload's states as ``version`` writes them, and its manifest."""
    manifest = out / "manifest.json"
    if version == THIS_BUILD:
        probe.states(str(out), str(manifest))
    else:
        if not _online():
            pytest.skip("offline: the released wheel cannot be downloaded")
        python = compare.released_python(version)
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
    return json.loads(manifest.read_text(encoding="utf-8")), out


@pytest.fixture(scope="module", params=REFUSED)
def refused(request, tmp_path_factory):
    """A release before 1.0: its version, its manifest, its states."""
    version = request.param
    manifest, out = _states(version, tmp_path_factory.mktemp(f"states-{version}"))
    return version, manifest, out


@pytest.fixture(scope="module", params=LOADED)
def loaded(request, tmp_path_factory):
    """A release from 1.0, or this build: its version, manifest, states."""
    version = request.param
    manifest, out = _states(version, tmp_path_factory.mktemp("states-loaded"))
    return version, manifest, out


def test_every_spec_was_saved_by_the_release_at_an_older_schema(refused):
    version, manifest, _ = refused
    assert manifest["version"] == version
    assert manifest["refused"] == {}
    assert manifest["saved"] == [name for name, *_ in probe.WORKLOAD]
    # An older schema: the refusal below is by the version, not the layout.
    assert manifest["schema"] < po.schema_version()


def test_a_pre_1_0_state_is_refused_by_its_version(refused):
    version, manifest, states = refused
    built = {name: getattr(po.spec, b)(name, **kw) for name, b, kw, _ in probe.WORKLOAD}
    for name, _, _, reads in probe.WORKLOAD:
        specs = [built[r] for r in reads] + [built[name]]
        with pytest.raises(ValueError) as e:
            po.ModelBank.load(states / f"{name}.state", specs=specs)
        message = str(e.value)
        assert f"state schema version {manifest['schema']} not supported" in message, message
        assert "refit it from its input" in message, message


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
    """The largest gap between two frames' floats, relative to ``1 + |v|``,
    after their nulls are held to the same rows."""
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
    name), and ``marginal``'s pairs table."""
    out = [bank.closed_groups()]
    try:
        out.append(bank.coef(name))
    except ValueError as e:
        assert "coef" in str(e) or "coefficients" in str(e), e
    if name == "marginal":
        out.append(bank.marginal(name))
    return out


def test_a_released_state_loads_and_goes_on_as_this_builds_own(loaded):
    version, manifest, states = loaded
    assert manifest["refused"] == {}
    assert manifest["saved"] == [name for name, *_ in probe.WORKLOAD]
    df = probe.stream()
    first, second = df.head(probe.HALF), df.slice(probe.HALF)
    built = {name: getattr(po.spec, b)(name, **kw) for name, b, kw, _ in probe.WORKLOAD}
    gaps = {}
    for name, _, _, reads in probe.WORKLOAD:
        specs = [built[r] for r in reads] + [built[name]]
        file_bank = po.ModelBank.load(states / f"{name}.state", specs=specs)
        own = po.ModelBank(specs)
        own.fit_predict(first)
        gap = _worst_gap(
            file_bank.fit_predict(second).select(name), own.fit_predict(second).select(name)
        )
        for got, want in zip(_tables(file_bank, name), _tables(own, name), strict=True):
            assert got.height == want.height, name
            if got.height:
                gap = max(gap, _worst_gap(got, want))
        gaps[name] = gap
    # This build's own files go on to the bit; a release's within what the
    # release comparison allows (a declared numeric change moves both).
    bound = 0.0 if version == THIS_BUILD else 1e-12
    assert max(gaps.values()) <= bound, {k: f"{v:.1e}" for k, v in gaps.items() if v > bound}
