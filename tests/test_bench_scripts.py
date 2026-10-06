"""The benchmark and experiment scripts say what they ran on, and import on
every platform (task 160, SC3 and SC4)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Every script whose output is a measurement the documents cite.
BENCHMARKS = [
    "benchmark",
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
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        assert peak_gb > 0, peak_gb
    else:
        assert math.isnan(peak_gb), peak_gb
