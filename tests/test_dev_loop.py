"""The tools of the development loop (docs/TESTING.md, "Iterating"): the
compiler cache, the extension's dev build, the parallel gate and
`scripts/quick.sh`. Each is a dev tool on a contributor's machine, so these
hold that none of them reaches what ships or what CI runs, and that each
does what the guide says.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((REPO / ".github" / "actions").glob("*/action.yml"))
CARGO = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
GATE = (REPO / "scripts" / "gate.sh").read_text(encoding="utf-8")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


# --- sccache ---------------------------------------------------------------


def _env_sh(tmp_path: Path, **env: str) -> str:
    """What `source scripts/env.sh` leaves in RUSTC_WRAPPER and
    SCCACHE_CACHE_SIZE, with a stand-in `sccache` first on the PATH."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "sccache").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake / "sccache").chmod(0o755)
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CI", "RUSTC_WRAPPER", "SCCACHE_CACHE_SIZE", "POLARS_ONLINE_SCCACHE"}
    }
    base["PATH"] = f"{fake}{os.pathsep}{base['PATH']}"
    script = (
        "source scripts/env.sh >/dev/null 2>&1; "
        'printf "%s|%s" "${RUSTC_WRAPPER-unset}" "${SCCACHE_CACHE_SIZE-unset}"'
    )
    res = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO,
        env={**base, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


posix_shell = pytest.mark.skipif(
    sys.platform == "win32",
    reason="scripts/env.sh is the POSIX shells' file; Windows sources env.ps1, with no wrapper",
)


@posix_shell
def test_env_sh_turns_sccache_on_when_asked(tmp_path):
    assert _env_sh(tmp_path, POLARS_ONLINE_SCCACHE="1") == "sccache|10G"
    capped = _env_sh(tmp_path, POLARS_ONLINE_SCCACHE="1", SCCACHE_CACHE_SIZE="3G")
    assert capped == "sccache|3G"


@posix_shell
@pytest.mark.parametrize(
    ("env", "why"),
    [
        ({}, "off unless asked: from a new worktree's path it is slower than no cache"),
        ({"POLARS_ONLINE_SCCACHE": "0"}, "the switch, set off"),
        ({"POLARS_ONLINE_SCCACHE": "1", "CI": "true"}, "every CI service sets CI"),
        ({"POLARS_ONLINE_SCCACHE": "1", "RUSTC_WRAPPER": ""}, "a wrapper already chosen"),
    ],
)
def test_env_sh_leaves_sccache_off(tmp_path, env, why):
    wrapper, _ = _env_sh(tmp_path, **env).split("|")
    assert wrapper == env.get("RUSTC_WRAPPER", "unset"), why


def _envs(doc: dict):
    """Every `env:` mapping in a workflow or action: top, jobs and steps."""
    yield doc.get("env") or {}
    for job in (doc.get("jobs") or {}).values():
        yield job.get("env") or {}
        for step in job.get("steps") or []:
            yield step.get("env") or {}
    for step in (doc.get("runs") or {}).get("steps") or []:
        yield step.get("env") or {}


def _steps(doc: dict):
    for job in (doc.get("jobs") or {}).values():
        yield from job.get("steps") or []
    yield from (doc.get("runs") or {}).get("steps") or []


#: The one place a workflow names sccache, and it is not this repository's:
#: maturin-action's own cache of the release wheel's build, on GitHub's
#: runners, which it exports as RUSTC_WRAPPER for that job only (the CLI's
#: step clears it). It predates the local cache and does not read it.
MATURIN_ACTION_CACHE = ("release.yml", "build the wheel")


def test_no_workflow_turns_the_local_compiler_cache_on():
    """The cache is for a contributor's machine. A workflow never sets
    RUSTC_WRAPPER to anything, sets no SCCACHE_ variable, installs no
    sccache and runs no script that would turn it on: the scripts CI runs
    source env.sh, which stays off under CI (above)."""
    found = []
    for path in [*WORKFLOWS, *ACTIONS]:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for env in _envs(doc):
            for key, value in env.items():
                if key == "RUSTC_WRAPPER" and value != "":
                    found.append(f"{path.name}: RUSTC_WRAPPER={value!r}")
                if key.startswith("SCCACHE"):
                    found.append(f"{path.name}: {key}")
        for step in _steps(doc):
            text = f"{step.get('uses', '')} {step.get('run', '')}".lower()
            if "sccache" in text:
                found.append(f"{path.name}: {step.get('name')!r} runs or uses sccache")
            where = (path.name, step.get("name"))
            if "sccache" in step.get("with", {}) and where != MATURIN_ACTION_CACHE:
                found.append(f"{path.name}: {step.get('name')!r} asks for sccache")
    assert not found, found


# --- the dev-wheel profile ---------------------------------------------------


def test_the_dev_wheel_profile_is_release_without_its_slow_parts():
    dev = CARGO["profile"]["dev-wheel"]
    assert dev == {
        "inherits": "release",
        "lto": False,
        "codegen-units": 256,
        "incremental": True,
        "strip": "none",
    }
    # What the shipped wheel is built with does not move.
    assert CARGO["profile"]["release"] == {"lto": "thin", "codegen-units": 1, "strip": "symbols"}


#: A build with the dev profile, as a command line or an argument list.
BUILDS_DEV = re.compile(r"""--profile["',\s]+dev-wheel""")


def test_what_ships_and_what_gates_a_push_build_release():
    """The dev build is for the loop inside a task alone: the gate's own
    build, every workflow and the release build the extension `--release`,
    and nothing but `scripts/quick.py` builds with the dev profile."""
    assert "maturin develop --release" in GATE
    for path in [*WORKFLOWS, *ACTIONS]:
        assert "dev-wheel" not in path.read_text(encoding="utf-8"), path.name
    named = sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "scripts").glob("*")
        if p.is_file() and BUILDS_DEV.search(p.read_text(encoding="utf-8", errors="replace"))
    )
    assert set(named) <= {"scripts/quick.py"}, named


# --- the parallel gate -------------------------------------------------------


@pytest.mark.parametrize(
    ("cores", "load", "expected"),
    [
        (14, 0.3, 8),  # an idle machine: at most eight
        (14, 11.4, 3),  # the other user's jobs on eleven cores
        (14, 13.9, 2),  # a full machine: still two
        (14, 30.0, 2),
        (8, None, 4),  # no load average (Windows): half the cores
        (2, None, 2),
    ],
)
def test_the_worker_count_follows_the_free_cores(cores, load, expected):
    assert _load("workers").workers(cores, load) == expected


def test_the_gate_runs_in_parallel_and_keeps_a_serial_switch():
    assert "--serial) serial=1" in GATE
    assert "python3 scripts/workers.py" in GATE
    # The Python tests: the parallel run leaves the `serial` ones out, and
    # they run afterwards in one process, so between them every test runs.
    assert '-n "$workers" --dist worksteal -m "$marks and not serial"' in GATE
    assert 'pytest_or_none -q -m "$marks and serial"' in GATE
    # nextest where installed, with cargo's doctests beside it, which
    # nextest does not run; `cargo test` otherwise.
    assert "cargo nextest run --workspace --exclude online-py" in GATE
    assert "cargo test --workspace --exclude online-py --doc" in GATE


def test_the_memory_tests_run_alone():
    """The tests that read their process's memory carry `serial`, so the
    gate runs them after the parallel run, in one process."""
    for path, needle in [
        ("tests/test_ffi_memory.py", "pytest.mark.serial,"),
        ("tests/test_gbdt.py", "@pytest.mark.serial\ndef test_many_builds_do_not_leak"),
    ]:
        assert needle in (REPO / path).read_text(encoding="utf-8"), path


def test_a_cached_download_is_written_whole_or_not_at_all(tmp_path):
    """Several processes share `.cache/`: a reader must never see a file
    another is still writing. The write goes to a file of the writer's own,
    and a failure leaves neither that file nor the cached one behind."""
    import data

    target = tmp_path / "day.parquet"
    seen = []

    def write(tmp: Path) -> None:
        tmp.write_bytes(b"half")
        seen.append(target.exists())
        tmp.write_bytes(b"whole")

    data.write_atomically(target, write)
    assert seen == [False] and target.read_bytes() == b"whole"

    def fails(tmp: Path) -> None:
        tmp.write_bytes(b"half")
        raise OSError("disk full")

    other = tmp_path / "other.parquet"
    with pytest.raises(OSError, match="disk full"):
        data.write_atomically(other, fails)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["day.parquet"]
