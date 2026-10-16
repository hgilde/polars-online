"""Task 1 smoke tests: the extension builds, imports, and agrees with the Python package."""

import re
from pathlib import Path

import polars as pl
import pytest

import polars_online

TIER = "essential"


def test_native_extension_loads() -> None:
    assert polars_online.native_version() == polars_online.__version__


# The version the wheel is built and tested against. Cargo.toml pins rust
# polars =0.55.2, the newest crate on crates.io (2026-10-08): py-polars 2.0
# is a Python-side major, and the wheel's own copy never meets the user's
# (docs/RELEASE-READINESS.md). `uv.lock` holds the dev environment to it.
# Moved 1.44.1 -> 1.44.2 for 0.5.1: every golden file passed on it
# unmodified, and `docs/VALIDATION.md`, regenerated on it, changed only in
# its header and one timing. Moved 1.44.2 -> 2.0.0 on 2026-10-08 (review
# round 4, D9), once the canary had passed on 2.0.0 final (run 37718689527
# of 2026-10-08): the goldens passed unmodified again, and VALIDATION.md
# changed only in its header and one timing.
BUILT_AGAINST = "2.0.0"
# Measured floor -- see the note in pyproject.toml and the matrix in
# docs/RELEASE-READINESS.md. `LazyFrame.collect_batches`, which
# `lf.online.fit_predict` reads with, arrived in 1.34.0 and first waits for
# a slower reader in 1.37.0 (before it, a 1 GB source was held whole).
SUPPORTED_FLOOR = "1.37.0"


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


#: A guide's sentence that states the py-polars floor by its version: "the
#: floor, 1.37.0", "The floor is 1.37.0", "That floor is py-polars 1.37.0".
#: A dated paragraph names an earlier floor as "the floor rose from" or
#: "the floor leg ran on", which this does not read.
_FLOOR_STATED = re.compile(r"\bfloor(?: is|,) (?:py-polars )?(\d+\.\d+\.\d+)", re.I)


def test_every_guide_that_states_the_floor_states_the_declared_one() -> None:
    """Task 269 (review 8, F-3): RUNNER.md's versioning section gave the
    floor as `collect_batches`, "added in 1.34.0", after the floor rose to
    1.37.0. Each guide that states the floor names its version, and every
    version so stated is the one `pyproject.toml` declares."""
    from test_no_plan_references import FILES, GUIDE_PARTS

    repo = Path(__file__).resolve().parents[1]
    stated: dict[str, list[str]] = {}
    for path in [*FILES, *GUIDE_PARTS]:
        text = (repo / path).read_text(encoding="utf-8")
        if path in GUIDE_PARTS:
            text = text[: text.index(GUIDE_PARTS[path])]
        found = _FLOOR_STATED.findall(" ".join(text.split()))
        if found:
            stated[path] = found
    for path in ("README.md", "docs/RUNNER.md", "docs/RELEASE-READINESS.md"):
        assert path in stated, f"{path} no longer states the floor by its version"
    wrong = {p: v for p, vs in stated.items() for v in vs if v != SUPPORTED_FLOOR}
    assert not wrong, f"a floor other than {SUPPORTED_FLOOR}: {wrong}"


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
