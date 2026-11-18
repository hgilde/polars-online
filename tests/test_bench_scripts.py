"""The benchmark and experiment scripts say what they ran on, and import on
every platform (task 160, SC3 and SC4)."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

TIER = "essential"

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Every script whose output is a measurement the documents cite.
BENCHMARKS = [
    "benchmark",
    "diagnostics_rates",
    "memory_bench",
    "parallel_bench",
    "plan_inspection_bench",
    "regime_experiments",
    "scaling_bench",
    "sklearn_comparison",
    "windows_bench",
]


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_the_header_names_the_build_and_the_machine():
    """The README's throughput sat stale for weeks and a parallel run read
    576 ms where the machine's load had gone to 524, with nothing in the
    output to say so."""
    text = _load("bench_header").header()
    for field in ("commit ", "polars-online ", "polars ", "Python ", "cpus", "load "):
        assert field in text, (field, text)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_every_benchmark_prints_the_header_first(name):
    source = (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
    assert "from bench_header import header" in source, name
    assert "print(header()" in source, name


@pytest.mark.parametrize("name", ["parallel_bench", "windows_bench"])
def test_the_platform_specific_benchmarks_import_everywhere(name):
    """`parallel_bench.py` ran BSD `time -l`, which GNU time lacks, and
    `windows_bench.py` imported `resource` and called `os.getloadavg` at the
    top, both absent on Windows. They import on every platform now, and
    measure memory where the platform says how."""
    assert callable(_load(name).main)


def test_the_memory_measure_runs_on_this_platform():
    out, peak_gb = _load("parallel_bench").timed("print(41 + 1)", {})
    assert out == "42"
    # As `timed` decides: BSD `time` on macOS, GNU `time` on Linux only where
    # `/usr/bin/time` is installed (a slim image lacks it), and nothing else.
    measured = sys.platform == "darwin" or (
        sys.platform.startswith("linux") and Path("/usr/bin/time").exists()
    )
    if measured:
        assert peak_gb > 0, peak_gb
    else:
        assert math.isnan(peak_gb), peak_gb


def test_memory_bench_measures_a_childs_peak_memory_wall_and_cpu_on_this_platform():
    """`scripts/memory_bench.py`'s `measure` reads each platform's own
    high-water mark (macOS footprint, Linux resident set, Windows working
    set) for the command line's cases on CI's Windows and Linux runners."""
    mb = _load("memory_bench")
    code = "import time; b = bytearray(b'x' * 300_000_000); time.sleep(0.3)"
    # In a venv on Windows `sys.executable` is the venv's `python.exe`, a
    # launcher that starts the base interpreter as a child and waits: the
    # process `measure` reads is the launcher (3 MB), not the one holding the
    # bytearray. The base interpreter is the process that does the work. The
    # command line's cases run a real .exe, which has no launcher in front.
    python = (
        getattr(sys, "_base_executable", sys.executable)
        if sys.platform == "win32"
        else sys.executable
    )
    m = mb.measure([python, "-c", code], dict(os.environ))
    assert 0.3 < m["peak_gb"] < 1.5, m
    assert m["wall_s"] >= 0.3 and m["cpu_s"] >= 0, m
    assert m["metric"] in {"footprint", "peak resident set", "peak working set"}
    if sys.platform.startswith("linux"):
        assert 0.3 < m["peak_anon_gb"] <= m["peak_gb"] * 1.05, m
    if sys.platform == "win32":
        assert m["peak_commit_gb"] >= 0.3, m


def test_memory_bench_a_failing_command_raises_and_sizes_are_multiples_of_3m():
    import subprocess

    mb = _load("memory_bench")
    with pytest.raises(subprocess.CalledProcessError):
        mb.measure([sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ))
    assert [mb._parts(s) for s in ("3m", "12m", "24m")] == [1, 4, 8]
    for bad in ("4m", "12", "0m", "m"):
        with pytest.raises(ValueError):
            mb._parts(bad)


def test_memory_bench_config_paths_survive_toml_on_windows(tmp_path):
    import tomllib

    mb = _load("memory_bench")
    cfg = tomllib.loads(mb._config(tmp_path, "3m").read_text(encoding="utf-8"))
    assert cfg["input"] == (tmp_path / "r3m.parquet").as_posix()
    assert "\\" not in cfg["output"]
