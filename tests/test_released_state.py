"""A state written by a released wheel meets this build as the CHANGELOG says
it will (docs/PLAN.md task 109).

Each release is installed from PyPI into the cache
``scripts/compare_release.py`` keeps (``.cache/release-compare/<version>``),
and ``scripts/release_probe.py --states`` runs under it: every spec of the
release comparison's workload -- all twenty-one kinds -- fitted on the first
half of its stream and saved.

Until 2026-09-28 this build loaded each file and fitted the second half
beside a bank of its own that fitted both halves: 0.11.1's files (schema 17)
went on bit for bit, 0.10.0's (schema 14) within 2.2e-13. Task 120 then
removed the ``on_clock_reset`` every one of those files names (``"max"``,
the old default, written whether or not a spec had a clock), and the user
said "Do not worry about old specs": a bank file before schema 19 is
refused by its version, naming the way out, and this test holds every
release's files to that. Downloading needs the network, so offline the test
is skipped (hard rule 1).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

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


def test_every_spec_was_saved_by_the_release_at_an_older_schema(released):
    version, manifest, _ = released
    assert manifest["version"] == version
    assert manifest["refused"] == {}
    assert manifest["saved"] == [name for name, *_ in probe.WORKLOAD]
    # An older schema, so a loader, not the current reader, is what runs.
    assert manifest["schema"] < po.schema_version()


def test_a_released_state_is_refused_by_its_version(released):
    version, manifest, states = released
    built = {name: getattr(po.spec, b)(name, **kw) for name, b, kw, _ in probe.WORKLOAD}
    for name, _, _, reads in probe.WORKLOAD:
        specs = [built[r] for r in reads] + [built[name]]
        with pytest.raises(ValueError) as e:
            po.ModelBank.load(states / f"{name}.state", specs=specs)
        message = str(e.value)
        assert f"state schema version {manifest['schema']} not supported" in message, message
        assert "refit it from its input" in message, message
