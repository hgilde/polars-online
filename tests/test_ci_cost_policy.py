"""The CI cost policy, pinned.

Actions minutes are metered while this repository is private, unevenly: macOS
bills at 10x and Windows at 2x Linux. A single day of pushes spent 2,270
billed minutes. The largest single line item was not a slow build -- it was a
matrix expression that read

    github.event_name == 'push' && [ubuntu, windows] || [ubuntu, macos, windows]

whose *fallback* branch, taken by every pull request, was the expensive one.
One dependabot PR then spent 830 minutes, 37% of the month, on four macOS jobs,
while the comment above it said macOS ran "weekly, on demand, and on release
tags".

A comment cannot enforce that. These tests can: they read the workflows and
assert the properties that keep the bill bounded. They are cheap, they run on
every commit, and they fail loudly when someone widens the matrix or removes a
timeout without meaning to.
"""

import pathlib

import pytest
import yaml

WORKFLOWS = sorted(
    (pathlib.Path(__file__).resolve().parents[1] / ".github/workflows").glob("*.yml")
)
assert WORKFLOWS, "no workflows found"


def load(path):
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML parses the `on:` key as the boolean True.
    d["on"] = d.pop(True, d.get("on"))
    return d


ALL = {p.name: load(p) for p in WORKFLOWS}
CI = ALL["ci.yml"]


class TestEveryJobIsBounded:
    """A job with no timeout runs for GitHub's default six hours. On Windows
    that is 720 billed minutes for one hung step; on macOS, 3,600."""

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_every_job_has_a_timeout(self, name):
        for job, spec in ALL[name].get("jobs", {}).items():
            assert "timeout-minutes" in spec, f"{name}:{job} has no timeout-minutes"
            assert 0 < spec["timeout-minutes"] <= 120, f"{name}:{job} timeout is not sane"

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_every_workflow_has_a_concurrency_group(self, name):
        """Without one, pushing twice in a minute pays for both runs."""
        assert "concurrency" in ALL[name], f"{name} has no concurrency group"

    @pytest.mark.parametrize("name", sorted(set(ALL) - {"release.yml"}))
    def test_superseded_runs_are_cancelled(self, name):
        assert ALL[name]["concurrency"].get("cancel-in-progress") is True

    def test_releases_queue_rather_than_cancel(self):
        """The one place where cancelling costs more than it saves: a
        superseded release run may be midway through publishing to PyPI."""
        assert ALL["release.yml"]["concurrency"].get("cancel-in-progress") is False


class TestTheMatrixDefaultsToCheap:
    def test_lint_never_leaves_linux(self):
        """fmt/clippy/ruff read source, not platform behaviour. Windows lint
        cost 35 minutes a run (70 billed) to re-derive an identical answer."""
        lint = CI["jobs"]["lint"]
        assert lint["runs-on"] == "ubuntu-latest"
        assert "strategy" not in lint, "lint must not be a matrix job"

    def test_lint_does_not_build_the_extension(self):
        """The maturin build is the most expensive step in the job and lint
        never imports the package."""
        steps = CI["jobs"]["lint"]["steps"]
        syncs = [s for s in steps if "uv sync" in str(s.get("run", ""))]
        assert syncs, "lint no longer syncs at all -- check this test"
        for s in syncs:
            assert "--no-install-project" in s["run"], "lint is building the extension again"
        # ...and `uv run` must not silently put it back.
        for s in steps:
            run = str(s.get("run", ""))
            if run.startswith("uv run"):
                assert s.get("env", {}).get("UV_NO_SYNC") == "1", f"{run!r} may re-sync"

    def test_the_expensive_runners_are_opt_in_not_opt_out(self):
        """The regression that cost 830 minutes: the *fallback* branch of the
        expression must be the cheap one, since fallback is what unforeseen
        event types get."""
        expr = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
        fallback = expr.rsplit("||", 1)[-1]
        assert "macos" not in fallback, f"macOS is the fallback branch: {fallback}"
        assert "windows" not in fallback, f"Windows is the fallback branch: {fallback}"
        assert "ubuntu" in fallback

    def test_visibility_test_fails_safe_on_a_missing_field(self):
        """`private == false` is deliberate: an event payload with no
        repository yields null, and null == false is false, so the cheap
        branch is taken. `private` alone, or `!private`, would invert that."""
        expr = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
        assert "github.event.repository.private == false" in expr

    def test_macos_is_reachable_only_by_dispatch_schedule_or_going_public(self):
        expr = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
        for clause in expr.split("||"):
            if "macos" in clause:
                assert "private == false" in clause or "workflow_dispatch" in clause, (
                    f"macOS reachable from an unguarded clause: {clause.strip()}"
                )


class TestStepOrderingThatHasAlreadyBrokenCI:
    """Two ordering bugs cost real runs; both are invisible to YAML linting."""

    @staticmethod
    def _index(job, predicate):
        for i, step in enumerate(CI["jobs"][job]["steps"]):
            if predicate(f"{step.get('name', '')} {step.get('uses', '')}"):
                return i
        return None

    def test_disk_is_freed_before_the_cache_is_restored(self):
        """The Ubuntu image starts with 9.3 GB free and the restored target/
        does not fit: the job died with "No space left on device" inside
        rust-cache, too hard to even write its own log."""
        free = self._index("test", lambda t: "free disk" in t)
        cache = self._index("test", lambda t: "rust-cache" in t)
        assert free is not None and cache is not None
        assert free < cache, "disk must be freed before the cache is restored"

    def test_rustflags_are_set_before_anything_compiles(self):
        """Rustflags are part of cargo's fingerprint. Written after `uv sync`
        they invalidated its build and made `maturin develop --release`
        recompile the workspace: 18 minutes, every Linux run."""
        free = self._index("test", lambda t: "free disk" in t)
        sync = self._index("test", lambda t: False)
        for i, step in enumerate(CI["jobs"]["test"]["steps"]):
            if "uv sync" in str(step.get("run", "")):
                sync = i
                break
        assert free is not None and sync is not None
        assert free < sync, "the linker config must be written before any build"

    def test_the_cache_survives_a_failing_job(self):
        """rust-cache skips its save step on failure by default. The Windows
        job never passed, so ~70 minutes of compilation was discarded on every
        run and the next one started cold -- a cycle that could not break
        itself."""
        for job in ("lint", "test"):
            for step in CI["jobs"][job]["steps"]:
                if "rust-cache" in str(step.get("uses", "")):
                    assert step.get("with", {}).get("cache-on-failure") is True, job


class TestDocOnlyPushesAreFree:
    @pytest.mark.parametrize("trigger", ["push", "pull_request"])
    def test_ci_has_no_paths_filter(self, trigger):
        """It saved metered minutes while the repo was private and it cost the
        first CI run on the public one: the push ended in two doc commits and
        the filter swallowed all 163. A required status check fails the same
        way -- a doc-only PR never runs it, so it can never merge. Minutes are
        free on a public repo; CI runs on everything."""
        assert CI["on"][trigger] is None or "paths-ignore" not in CI["on"][trigger]

    def test_benchmark_skips_doc_only_pushes(self):
        """Reported, never gating (E11). It runs on main now that minutes are
        free, but a prose commit cannot change a throughput number, and a
        summary nobody reads is still noise in the run list."""
        on = ALL["benchmark.yml"]["on"]
        assert "pull_request" not in on, "a fork's runner is not a comparable number"
        assert "paths-ignore" in on["push"]
