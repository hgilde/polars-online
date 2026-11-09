"""Tests that need a py-polars newer than the floor of the declared range.

Every release, and the canary each month, run the suite on the floor that
``pyproject.toml`` declares (``polars>=1.37.0``; review 2026-10-06, AP7
and CI2). A test that exercises a Polars feature added after the floor is
marked with :func:`needs_polars`, which skips it there, naming the version
it needs and why, and runs it everywhere else. A skip on the floor is a
version guard, as a platform guard is on another OS: the feature does not
exist there, and the package does not use it on that path.

The version a mark names must lie above the floor, where it can skip
something, and at most the version the development environment pins
(``tests/test_scaffold.py``), so that no mark hides a test from CI's own
runs. A raised floor that passes a mark's version fails here, at
collection, so the mark goes with it.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

from test_scaffold import BUILT_AGAINST, SUPPORTED_FLOOR


def _version(text: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    assert m, text
    major, minor, patch = (int(part) for part in m.groups())
    return major, minor, patch


INSTALLED = _version(pl.__version__)


def needs_polars(version: str, why: str) -> pytest.MarkDecorator:
    """Skip the test on a py-polars older than ``version``, saying ``why``."""
    need = _version(version)
    assert _version(SUPPORTED_FLOOR) < need <= _version(BUILT_AGAINST), (
        f"needs_polars({version!r}) must lie above the floor {SUPPORTED_FLOOR} "
        f"and at most the pinned {BUILT_AGAINST}"
    )
    return pytest.mark.skipif(
        need > INSTALLED,
        reason=f"needs py-polars >= {version}, and this is {pl.__version__}: {why}",
    )


#: The Polars calls a documented block may make that the floor lacks, by
#: the text that makes them, with the py-polars that has each.
_CALLS = (
    ("scan_arrow_c_stream(", "1.43.0", "pl.scan_arrow_c_stream, which py-polars added in 1.43.0"),
    (".ewm_sum_by(", "1.43.0", "Expr.ewm_sum_by, which py-polars added in 1.43.0"),
)

#: The Polars calls a documented block may print whose output the floor
#: gives otherwise, by the pattern that makes them. Polars' ``ewm_var`` and
#: ``ewm_std`` under ``bias=False`` give a window of one value null from
#: 1.44, where 1.42.0 and before give 0.0 (``tests/test_windows.py``).
_OUTPUTS = (
    (
        re.compile(r"pl\.col\([^)]*\)\.ewm_(?:var|std)\("),
        "1.44.1",
        "Polars' ewm_var(bias=False), the reference, gives the first row null from 1.44",
    ),
)


def block_marks(code: str, *, output: bool) -> list[pytest.MarkDecorator]:
    """The version skips a documented block's code needs: for each Polars
    call it makes that the floor lacks, and, where the test holds it to
    what it prints (``output``), for each whose output the floor gives
    otherwise."""
    marks = [needs_polars(version, why) for text, version, why in _CALLS if text in code]
    if output:
        marks += [needs_polars(version, why) for at, version, why in _OUTPUTS if at.search(code)]
    return marks
