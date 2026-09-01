"""Task 1 smoke tests: the extension builds, imports, and agrees with the Python package."""

import re
from pathlib import Path

import polars as pl

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
# docs/RELEASE-READINESS.md.
SUPPORTED_FLOOR = "1.28.1"


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split(".")[:3])


def test_the_dev_environment_is_on_the_version_we_build_against() -> None:
    """The golden files and `docs/VALIDATION.md` were produced here, so a
    silent local upgrade would move numbers without anyone deciding to."""
    assert pl.__version__ == BUILT_AGAINST


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
