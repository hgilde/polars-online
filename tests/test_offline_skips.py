"""A test skipped as offline fails in a job that restored its datasets
(``tests/offline_skips.py``), and skips everywhere else.

Both branches run an inner session in this process (pytest's ``pytester``)
with the hook loaded alone: a dataset's offline skip, in a fixture and in
the test body, fails where :data:`offline_skips.RESTORED` is ``1`` and
names the set, and skips where it is unset, ``CI`` set or not; a skip for
anything else -- a released wheel offline, a platform guard -- is a skip
either way. Then every loader's skip is held to the prefix the hook reads,
and the two workflows that restore their datasets to setting the variable
on the step that runs the tests, and no other workflow to setting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import data
import offline_skips
import public_benchmarks

TIER = "essential"
REPO = Path(__file__).resolve().parent.parent

pytest_plugins = ["pytester"]

INNER = f"""
import pytest

OFFLINE = {data.OFFLINE_SKIP!r}


@pytest.fixture
def rdata():
    pytest.skip(OFFLINE + "Rdatasets robustbase/stackloss (TimeoutError('timed out'))")


def test_in_a_fixture(rdata):
    pass


def test_in_the_body():
    pytest.skip(OFFLINE + "NIST StRD Longley (URLError('no route'))")


def test_a_released_wheel():
    pytest.skip("offline: the released wheel cannot be downloaded")


def test_a_platform_guard():
    pytest.skip("Unix only: a directory's mode bits")


def test_runs():
    pass
"""


def _outcomes(pytester: pytest.Pytester) -> tuple[dict[str, int], str]:
    pytester.makepyfile(test_inner=INNER)
    # The plugin is imported already (by tests/conftest.py), so pytest
    # cannot rewrite its asserts, and says so: it has none.
    result = pytester.runpytest_inprocess(
        "-p",
        "offline_skips",
        "-p",
        "no:cacheprovider",
        "-W",
        "ignore::pytest.PytestAssertRewriteWarning",
        "-rA",
    )
    return result.parseoutcomes(), result.stdout.str()


def test_restored_a_datasets_offline_skip_fails_naming_it(pytester, monkeypatch):
    monkeypatch.setenv(offline_skips.RESTORED, "1")
    outcomes, out = _outcomes(pytester)
    # The fixture's skip is in setup: a failure there is an error.
    assert outcomes == {"passed": 1, "failed": 1, "errors": 1, "skipped": 2}, out
    assert "FAILED test_inner.py::test_in_the_body" in out
    assert "ERROR test_inner.py::test_in_a_fixture" in out
    for named in ("NIST StRD Longley (URLError", "Rdatasets robustbase/stackloss (Timeout"):
        (line,) = [x for x in out.splitlines() if x.startswith(named)]
        assert "offline, in a job that restored its datasets before the tests" in line
        assert f"({offline_skips.RESTORED}=1)" in line
    assert "SKIPPED [1] test_inner.py:20: offline: the released wheel" in out, out
    assert "SKIPPED [1] test_inner.py:24: Unix only" in out, out


@pytest.mark.parametrize("ci", [None, "true"])
def test_not_restored_every_skip_is_a_skip(pytester, monkeypatch, ci):
    """On a desk, and on a CI job that restores nothing (the canary, the
    release's Polars and NumPy legs, the CPU paths), an outage is a skip."""
    monkeypatch.delenv(offline_skips.RESTORED, raising=False)
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    outcomes, out = _outcomes(pytester)
    assert outcomes == {"passed": 1, "skipped": 4}, out
    assert f"{data.OFFLINE_SKIP}NIST StRD Longley" in out


@pytest.mark.parametrize("value", ["", "0", "true"])
def test_only_1_turns_it_on(monkeypatch, value):
    monkeypatch.setenv(offline_skips.RESTORED, value)
    assert not offline_skips.restored()
    monkeypatch.setenv(offline_skips.RESTORED, "1")
    assert offline_skips.restored()


def _offline(*_args, **_kwargs):
    raise data.Offline("offline") from TimeoutError("timed out")


@pytest.mark.parametrize(
    ("load", "named"),
    [
        (lambda: data.public_intraday_or_skip(("2024-01-02",)), "BTCUSDT 1-minute rows of 2024"),
        (lambda: data.public_quotes_and_trades_or_skip("ETCUSDT", "2024-01-02"), "ETCUSDT quotes"),
        (lambda: data.nist_strd_linear_or_skip("Longley"), "NIST StRD Longley"),
        (lambda: data.rdataset_or_skip("robustbase", "stackloss"), "Rdatasets robustbase/"),
        (lambda: public_benchmarks._or_skip(_offline, "LIBSVM a1a"), "LIBSVM a1a"),
    ],
)
def test_every_loaders_skip_begins_with_the_prefix_the_hook_reads(
    tmp_path, monkeypatch, load, named
):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download", _offline)
    with pytest.raises(pytest.skip.Exception) as skipped:
        load()
    reason = str(skipped.value)
    assert reason.startswith(data.OFFLINE_SKIP) and named in reason, reason
    assert "timed out" in reason, reason


def test_the_jobs_that_restore_their_datasets_say_so_and_no_other_does():
    workflows = REPO / ".github" / "workflows"
    ci = yaml.safe_load((workflows / "ci.yml").read_text(encoding="utf-8"))
    (pytest_step,) = [s for s in ci["jobs"]["test"]["steps"] if s.get("name") == "pytest"]
    assert pytest_step["env"][offline_skips.RESTORED] == "1"
    oracles = yaml.safe_load((workflows / "oracles.yml").read_text(encoding="utf-8"))
    grid_steps = oracles["jobs"]["grid"]["steps"]
    (shard,) = [s for s in grid_steps if s.get("name") == "pytest, this shard"]
    assert shard["env"][offline_skips.RESTORED] == "1"
    # Each restores before it: the test job from its cache, the artifact and
    # a download; a grid from the `datasets` job's artifact.
    steps = [s.get("name") or s.get("uses", "") for s in ci["jobs"]["test"]["steps"]]
    assert steps.index("fetch every dataset the suite reads") < steps.index("pytest")
    grid = [s.get("name") or s.get("uses", "") for s in oracles["jobs"]["grid"]["steps"]]
    assert grid.index("the datasets") < grid.index("pytest, this shard")
    assert oracles["jobs"]["grid"]["needs"] == "datasets"
    for path in workflows.glob("*.yml"):
        if path.name not in {"ci.yml", "oracles.yml"}:
            assert offline_skips.RESTORED not in path.read_text(encoding="utf-8"), path
