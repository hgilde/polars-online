"""A test skipped because its public dataset is offline fails instead, in a
job that restored its datasets before the tests ran.

A test whose data cannot be downloaded skips (hard rule 1): on a desk with
no network the suite still runs, and says what it left out. CI's test job
and the oracle grids, though, put every dataset the suite reads in place
before the tests run, from three copies: the job's cache, the newest
artifact of the oracle runs, and a download with retries
(docs/TESTING.md, "Downloaded data in CI"). There, a test skipped as
offline means all three failed, and a skip among hundreds is easy to miss:
the two guards that read Fränti's sets skipped on nearly every leg of the
25 CI runs to 2026-11-05, while every run passed. So in those jobs the
skip fails the test, naming the dataset.

The jobs say so by setting :data:`RESTORED` (``1``). ``CI`` alone does not
turn it on: the canary, the release's Polars and NumPy legs and the CPU
paths also run the suite under ``CI`` and restore nothing, so there an
outage at a source is still a skip, as on a desk. The skips this reads are
the dataset loaders' (:data:`data.OFFLINE_SKIP`); a skip for a released
wheel or PyPI's list, which no job keeps a copy of, stays a skip.

Registered by ``tests/conftest.py`` (which imports the hook), and loadable
alone with ``-p offline_skips`` for a test of it.
"""

from __future__ import annotations

import os

import pytest

from data import OFFLINE_SKIP

#: Set (to ``1``) by a job that restored the datasets before its tests.
RESTORED = "POLARS_ONLINE_DATASETS_RESTORED"


def restored() -> bool:
    """Whether this run's datasets were put in place before it started."""
    return os.environ.get(RESTORED) == "1"


def offline_dataset(report: pytest.TestReport) -> str | None:
    """The dataset a skipped test's reason names, if it skipped as offline."""
    if not report.skipped or not isinstance(report.longrepr, tuple):
        return None
    reason = str(report.longrepr[2]).removeprefix("Skipped: ")
    return reason.removeprefix(OFFLINE_SKIP) if reason.startswith(OFFLINE_SKIP) else None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    dataset = offline_dataset(report) if restored() else None
    if dataset is not None:
        report.outcome = "failed"
        report.longrepr = (
            f"{dataset}: offline, in a job that restored its datasets before the tests "
            f"({RESTORED}=1), so the cache, the last oracle run's artifact and the "
            "download all failed to provide it. A skip here would hide that; see the "
            "job's dataset steps for which copy was missing."
        )
    return report
