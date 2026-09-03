"""Task 1 smoke tests: the extension builds, imports, and agrees with the Python package."""

import re
from pathlib import Path

import polars as pl
import pytest

import polars_online


def test_native_extension_loads() -> None:
    assert polars_online.native_version() == polars_online.__version__


def test_schema_version_is_positive() -> None:
    assert polars_online.schema_version() >= 1


# The version the wheel is built and tested against: Cargo.toml pins rust
# polars =0.55.2, which is what py-polars 1.44.1 is built from. `uv.lock` holds
# the dev environment to it.
BUILT_AGAINST = "1.44.1"
# Measured floor -- see the note in pyproject.toml and the matrix in
# docs/RELEASE-READINESS.md. `LazyFrame.collect_batches`, which `po.run` and
# `lf.online.fit_predict` read with, arrived in 1.34.0.
SUPPORTED_FLOOR = "1.34.0"


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split(".")[:3])


# These two are about *our* pins, not about polars: the canary unpins polars
# on purpose and runs `-m "not pins"`, since a red canary must mean "Polars
# broke us" and nothing else. Its first run failed on the range assertion.
@pytest.mark.pins
def test_the_dev_environment_is_on_the_version_we_build_against() -> None:
    """The golden files and `docs/VALIDATION.md` were produced here, so a
    silent local upgrade would move numbers without anyone deciding to."""
    assert pl.__version__ == BUILT_AGAINST


@pytest.mark.pins
def test_the_declared_range_brackets_what_we_build_against() -> None:
    """The runtime requirement is a range now, so the invariant worth pinning
    is that it actually contains the tested version -- and that the floor is
    not silently raised above it."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    req = re.search(r'"polars(>=[^"]+)"', pyproject)
    assert req, "pyproject no longer declares a polars range"
    floor, ceiling = re.match(r">=([\d.]+),<(\d+)", req.group(1)).groups()
    assert floor == SUPPORTED_FLOOR
    assert _ver(SUPPORTED_FLOOR) <= _ver(BUILT_AGAINST) < (int(ceiling), 0, 0)


def test_the_allocator_capsule_pyo3_polars_imports_still_resolves() -> None:
    """A tripwire for a silent 43% throughput loss.

    `pyo3_polars::PolarsAllocator` routes this extension's allocations through
    py-polars' allocator by importing the capsule `polars.polars._allocator`,
    and **falls back to the system allocator without erroring** if that name
    stops resolving. Measured A/B/A, having it is worth +43% at k=5, so losing
    it would be invisible and expensive.

    The name is fragile in a non-obvious way: `polars.polars` is not an
    importable submodule (`import_module` raises), it is an *attribute* of the
    `polars` package aliasing the real runtime module -- currently
    `_polars_runtime_32._polars_runtime`. `PyCapsule_Import` walks dotted names
    by import-then-getattr, which is why it works at all. This asserts the
    resolution path rather than the module's name, since the alias target is
    polars' business and has already changed once.
    """
    native = getattr(pl, "polars", None)
    assert native is not None, "polars no longer exposes `polars.polars`"
    assert hasattr(native, "_allocator"), (
        "the `polars.polars._allocator` capsule is gone; pyo3-polars' "
        "PolarsAllocator is now silently falling back to the system allocator"
    )
