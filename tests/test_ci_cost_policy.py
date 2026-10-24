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

import json
import pathlib
import re
import subprocess
import tomllib
from types import SimpleNamespace

import pytest
import yaml

TIER = "essential"

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

    #: Jobs allowed past two hours, with their cap. The weekly mutation
    #: shards run on Linux, only while the repository is public or by hand,
    #: and stop cargo-mutants well inside the limit (task 155); 350 minutes
    #: since 2026-10-11, inside GitHub's 360, as room for a slow week.
    LONGER = {("mutants.yml", "weekly"): 350}

    #: Jobs allowed past two hours on every push, on Linux: the changed
    #: lines' mutation shards, whose stop is 300 minutes since 2026-10-11
    #: (the user: "Ensure that the GitHub mutants run has plenty of headroom
    #: not to fail its next run"; the slowest shard of the push of 5cb8f7e
    #: took 98). The repository is public, so a longer job costs nothing; a
    #: shard stopped short leaves mutants untested and fails its report.
    LONGER_ON_PUSH = {("mutants.yml", "changed"): 330}

    #: Jobs allowed past two hours on any OS while the repository is public,
    #: and held to two hours if it is private again: CI's test matrix, whose
    #: Windows legs' Rust build took 33 to 78 minutes from a warm cache on
    #: 2026-10-13, and one was cancelled at 120 after its tests had passed.
    LONGER_WHILE_PUBLIC = {("ci.yml", "test"): 180}

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_every_job_has_a_timeout(self, name):
        for job, spec in ALL[name].get("jobs", {}).items():
            # A job that calls a reusable workflow takes no timeout; the
            # called workflow's jobs carry theirs, and are checked here too.
            if "uses" in spec:
                continue
            assert "timeout-minutes" in spec, f"{name}:{job} has no timeout-minutes"
            timeout = spec["timeout-minutes"]
            if (name, job) in self.LONGER_WHILE_PUBLIC:
                # Longer only while public: the expression gives 120 or less
                # to a private repository.
                m = re.fullmatch(
                    r"\$\{\{ github\.event\.repository\.private && (\d+) \|\| (\d+) \}\}",
                    str(timeout),
                )
                assert m, f"{name}:{job} timeout {timeout!r} is not the public/private form"
                private, public = int(m.group(1)), int(m.group(2))
                assert 0 < private <= 120, (name, job, private)
                assert 0 < public <= self.LONGER_WHILE_PUBLIC[(name, job)], (name, job, public)
                continue
            cap = {**self.LONGER, **self.LONGER_ON_PUSH}.get((name, job), 120)
            assert 0 < timeout <= cap, f"{name}:{job} timeout is not sane"

    def test_a_longer_job_on_every_push_runs_on_linux(self):
        for name, job in self.LONGER_ON_PUSH:
            assert ALL[name]["jobs"][job]["runs-on"] == "ubuntu-latest", (name, job)

    def test_a_longer_job_runs_only_on_linux_while_public_or_by_hand(self):
        for (name, job), _ in self.LONGER.items():
            spec = ALL[name]["jobs"][job]
            assert spec["runs-on"] == "ubuntu-latest", (name, job)
            cond = " ".join(spec["if"].split())
            assert "github.event.repository.private == false" in cond, (name, job)
            assert cond.replace("(", "").replace(")", "").count("||") == 1, (name, job)
            assert "github.event_name == 'workflow_dispatch'" in cond, (name, job)

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_every_workflow_has_a_concurrency_group(self, name):
        """Without one, pushing twice in a minute pays for both runs."""
        assert "concurrency" in ALL[name], f"{name} has no concurrency group"

    @pytest.mark.parametrize("name", sorted(set(ALL) - {"release.yml"}))
    def test_superseded_runs_are_cancelled(self, name):
        assert ALL[name]["concurrency"].get("cancel-in-progress") is True

    def test_a_scheduled_run_cannot_cancel_a_push(self):
        """ci.yml runs on a schedule as well as on push; with the event out of
        its group, the Monday run and a push to main shared one group under
        `cancel-in-progress`, so one cancelled the other, and a cancelled push
        run publishes no reference (task 160, CI8). mutants.yml carries the
        event for the same reason."""
        group = ALL["ci.yml"]["concurrency"]["group"]
        assert "github.event_name" in group, group

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

    def test_macos_is_reachable_only_by_dispatch_or_going_public(self):
        expr = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
        for clause in expr.split("||"):
            if "macos" in clause:
                assert "private == false" in clause or "workflow_dispatch" in clause, (
                    f"macOS reachable from an unguarded clause: {clause.strip()}"
                )


class TestWindowsIsTwoImagesAndEveryJobNamesItsMachine:
    """On Windows `exp`, `log` and `pow` come from `ucrtbase.dll`, part of the
    OS, so each Windows build has its own last bits; and a runner label is a
    pool of CPUs, AVX-512 or not and any L3. Task 273 (a): the test matrix
    runs Server 2022 beside the newest wherever it runs Windows, and every
    job that can land on Windows prints its machine first, so a failure
    names it (docs/TESTING.md, "The machines and the CPU paths")."""

    REPORT = "./.github/actions/machine-report"

    def test_windows_2022_runs_wherever_windows_latest_does(self):
        expr = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
        clauses = [c for c in expr.split("||") if "windows" in c]
        assert len(clauses) == 3, clauses
        for clause in clauses:
            assert '"windows-latest", "windows-2022"' in clause, clause

    def test_windows_peak_memory_is_held_on_one_image(self):
        """Both Windows images at the floor Python wrote the one
        `rss.*.windows` budget and read the streaming case 143.9 and 148.7
        MiB on one build, so each failed it as one to tighten (task 273
        (a2)). `windows-2022` reports its peak and `windows-latest` holds
        it (tests/test_budgets_memory.py, `_rss_held_here`)."""
        (step,) = [s for s in CI["jobs"]["test"]["steps"] if s.get("name") == "pytest"]
        assert step["env"]["POLARS_ONLINE_RSS_HELD"] == "${{ matrix.os != 'windows-2022' }}"

    @staticmethod
    def _can_run_on_windows(spec: dict) -> bool:
        return "windows" in json.dumps({k: spec.get(k) for k in ("runs-on", "strategy")})

    def test_every_job_that_can_run_on_windows_reports_its_machine_first(self):
        found = []
        for name, job, spec, steps in _every_step():
            if not self._can_run_on_windows(spec):
                continue
            found.append(f"{name}:{job}")
            uses = [str(s.get("uses", "")) for s in steps]
            assert uses[0].startswith("actions/checkout@"), (name, job, uses[:2])
            assert uses[1] == self.REPORT, (name, job, uses[:2])
            assert "if" not in steps[1], (name, job)
        assert sorted(found) == [
            "ci.yml:test",
            "cpu-paths.yml:oracles",
            "cpu-paths.yml:sde",
            "release.yml:build",
            "release.yml:read-state",
        ], found

    def test_the_report_covers_every_os_and_never_fails_a_job(self):
        action = yaml.safe_load(REPORT_ACTION.read_text(encoding="utf-8"))
        assert action["runs"]["using"] == "composite"
        steps = action["runs"]["steps"]
        assert [s["if"] for s in steps] == ["runner.os == 'Windows'", "runner.os != 'Windows'"]
        assert all(s.get("continue-on-error") is True for s in steps)
        windows = steps[0]["run"]
        for needed in (
            "Win32_Processor",
            "L2CacheSize",
            "L3CacheSize",
            "Avx2]::IsSupported",
            "Avx512F]::IsSupported",
            "Fma]::IsSupported",
            "ProcessorCount",
            "Win32_OperatingSystem",
            "ucrtbase.dll",
            "GITHUB_STEP_SUMMARY",
        ):
            assert needed in windows, needed
        assert "GITHUB_STEP_SUMMARY" in steps[1]["run"]


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


class TestPythonVersions:
    """Every Python the package declares is one CI runs (2026-09-24). Until
    then CI never asked for a version: each runner's `uv sync` took whatever
    interpreter it had, so macOS ran 3.14 while the classifiers stopped at
    3.13, and a change 3.13 made to docstrings surfaced on that leg alone."""

    @staticmethod
    def _declared() -> tuple[str, list[str]]:
        root = pathlib.Path(__file__).resolve().parents[1]
        meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        floor = meta["requires-python"]
        assert floor.startswith(">=") and "," not in floor, floor
        prefix = "Programming Language :: Python :: 3."
        minors = sorted(
            int(c[len(prefix) :])
            for c in meta["classifiers"]
            if c.startswith(prefix) and c[len(prefix) :].isdigit()
        )
        return floor[2:], [f"3.{m}" for m in minors]

    @staticmethod
    def _matrix() -> dict:
        return CI["jobs"]["test"]["strategy"]["matrix"]

    def test_the_classifiers_run_from_the_floor_without_a_gap(self):
        floor, declared = self._declared()
        assert declared[0] == floor, (floor, declared)
        minors = [int(v.split(".")[1]) for v in declared]
        assert minors == list(range(minors[0], minors[-1] + 1)), declared

    def test_linux_runs_every_version_and_the_others_run_both_ends(self):
        floor, declared = self._declared()
        matrix = self._matrix()
        # Strings: a bare 3.10 in YAML is the float 3.1.
        assert all(isinstance(v, str) for v in matrix["python"]), matrix["python"]
        assert matrix["python"] == [floor, declared[-1]], matrix["python"]
        extra = [e["python"] for e in matrix.get("include", [])]
        assert all(e.get("os") == "ubuntu-latest" for e in matrix.get("include", []))
        assert sorted(matrix["python"] + extra, key=lambda v: int(v.split(".")[1])) == declared

    def test_each_leg_runs_the_python_it_is_named_for(self):
        job = CI["jobs"]["test"]
        assert job.get("env", {}).get("UV_PYTHON") == "${{ matrix.python }}"
        assert "matrix.python" in job["name"]

    def test_the_release_comparison_reports_once_and_never_gates(self):
        """scripts/compare_release.py on one leg: it installs the newest
        release from PyPI, and a difference between releases may be intended."""
        steps = [
            s for s in CI["jobs"]["test"]["steps"] if "compare_release.py" in str(s.get("run", ""))
        ]
        assert len(steps) == 1, steps
        step = steps[0]
        assert "--report" in step["run"] and step.get("continue-on-error") is True
        floor, _ = self._declared()
        assert f"matrix.python == '{floor}'" in step["if"] and "runner.os == 'Linux'" in step["if"]

    def test_coverage_rides_on_one_legs_own_run(self):
        """`coverage (python)` ran the whole suite a second time on every OS,
        `continue-on-error`, for a report nobody read (review 2026-10-06,
        CI17). The figure now comes from one leg's own pytest run, Linux at
        the floor Python, so it costs that run's instrumentation and no
        second run, and that run gates as every leg's does."""
        floor, _ = self._declared()
        steps = CI["jobs"]["test"]["steps"]
        suites = [s for s in steps if str(s.get("run", "")).startswith("uv run pytest")]
        assert len(suites) == 1, [s.get("name") for s in suites]
        (suite,) = suites
        assert "continue-on-error" not in suite and "if" not in suite
        run = " ".join(suite["run"].split())
        assert "--cov=polars_online" in run, run
        assert "runner.os == 'Linux'" in run and f"matrix.python == '{floor}'" in run, run

    def test_the_canary_varies_polars_alone(self):
        """The canary holds that only one thing may vary in it, yet its Python
        followed the runner image, as `uv sync` takes whatever interpreter it
        finds (review 2026-10-06, CI19). It runs the floor Python, as
        leakcheck.yml pins one."""
        floor, _ = self._declared()
        canary = ALL["polars-canary.yml"]
        for job, spec in canary["jobs"].items():
            python = spec.get("env", {}).get("UV_PYTHON", canary.get("env", {}).get("UV_PYTHON"))
            assert python == floor, (job, python)

    def test_the_reference_is_built_and_published_once(self):
        """The Pages artifact can be uploaded only once a run; one Linux leg
        builds it, the one at the floor."""
        floor, _ = self._declared()
        for step in CI["jobs"]["test"]["steps"]:
            label = f"{step.get('name', '')} {step.get('uses', '')}"
            if "sphinx" in label or "upload-pages-artifact" in label:
                cond = step.get("if", "")
                assert "runner.os == 'Linux'" in cond, (label, cond)
                assert f"matrix.python == '{floor}'" in cond, (label, cond)


class TestMutationTesting:
    """mutants.yml (docs/TESTING.md T-D4): the changed lines on every push
    and pull request, gating; all of online-core weekly, reporting."""

    MUT = ALL["mutants.yml"]

    #: Each sharded job, the report job that reads its shards, and the prefix
    #: of the artifact each shard uploads.
    SHARDED = {
        "changed": ("changed-report", "mutants-changed-"),
        "weekly": ("weekly-report", "mutants-shard-"),
    }

    #: A GitHub runner's memory (16 GiB) and what its own agent and system
    #: keep outside the mutants' scope: a mutant that allocates without end
    #: is killed alone instead of shutting the runner down. The push of
    #: 5cb8f7e lost 8 of 111 shards that way.
    RUNNER_MEMORY = 16 * 2**30
    RUNNER_KEEPS = 3 * 2**30
    #: The step that runs the mutants, and the scope around it.
    SCOPE = 'bash scripts/memory_scope.sh "$MUTANTS_MEMORY_MAX" -- \\'

    @staticmethod
    def _bytes(text: str) -> int:
        units = {"K": 2**10, "M": 2**20, "G": 2**30}
        return int(float(text[:-1]) * units[text[-1]]) if text[-1] in units else int(text)

    def _mutants_step(self, job: str) -> str:
        return next(
            s["run"]
            for s in self.MUT["jobs"][job]["steps"]
            if "cargo mutants" in s.get("run", "") and "timeout" in s["run"]
        )

    @pytest.mark.parametrize("job", ["changed", "weekly"])
    def test_every_mutants_run_is_capped_in_one_memory_scope(self, job):
        """The whole `cargo mutants` run is inside one cgroup scope whose
        `MemoryMax` leaves the runner its share (scripts/memory_scope.sh).
        The cap it replaced, `prlimit --data` on each test binary, counted
        virtual writable memory, which glibc keeps after a free: a 6 GiB cap
        sized from a resident peak of 497 MiB failed the baseline of 4 of
        the 32 shards of run 38205650365. No job keeps that runner."""
        script = self._mutants_step(job)
        lines = [line.strip() for line in script.splitlines()]
        at = lines.index(self.SCOPE)
        assert lines[at + 1].startswith("timeout --signal=INT"), lines[at + 1]
        assert "prlimit" not in json.dumps(self.MUT["jobs"][job])
        cap = self._bytes(self.MUT["env"]["MUTANTS_MEMORY_MAX"])
        assert 8 * 2**30 <= cap <= self.RUNNER_MEMORY - self.RUNNER_KEEPS, cap

    def test_the_scope_caps_resident_memory_and_outlives_a_kill(self):
        """`MemoryMax` is resident memory, page cache included, which the
        kernel reclaims before it kills; no swap, so a runaway is killed
        rather than paged; `OOMPolicy=continue`, so the kernel's kill of a
        test leaves cargo-mutants running (systemd's default stops the whole
        unit); and the command runs as the calling user again."""
        root = pathlib.Path(__file__).resolve().parents[1]
        text = (root / "scripts" / "memory_scope.sh").read_text(encoding="utf-8")
        for needed in (
            "systemd-run --scope",
            '-p MemoryMax="$max"',
            "-p MemorySwapMax=0",
            "-p OOMPolicy=continue",
            'setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups',
            'env PATH="$PATH" HOME="$HOME"',
        ):
            assert needed in text, needed

    @pytest.mark.parametrize("job", ["changed", "weekly"])
    def test_every_mutants_run_builds_its_copies_on_the_runners_disk(self, job):
        """cargo-mutants copies the tree into `TMPDIR` (else `/tmp`) and
        builds each copy there, a cold target per build directory. On
        ubuntu-26.04 `/tmp` is a RAM-backed tmpfs with a per-user quota,
        charged to the scope's memory: the push of 85da91e lost 23 of its 24
        shards to `Disk quota exceeded (os error 122)` after one to six
        mutants each (run 40076792577), where the root disk had 109 GB free.
        So `TMPDIR` is a directory under the job's `RUNNER_TEMP`, on that
        disk, made before the run and set before the scope is entered (the
        scope carries the caller's environment through), and the step
        prints the most of that disk the run used, so a shard that fills it
        says how near it came (task 270 (m))."""
        script = self._mutants_step(job)
        lines = [line.strip() for line in script.splitlines()]
        assert 'export TMPDIR="$RUNNER_TEMP/mutants-tmp"' in lines, script
        assert 'mkdir -p "$TMPDIR"' in lines, script
        at = lines.index('export TMPDIR="$RUNNER_TEMP/mutants-tmp"')
        assert at < lines.index(self.SCOPE), script
        assert "disk_peak" in script, script

    def test_the_memory_scope_is_proven_on_every_run(self):
        """A job with no condition plants a runaway in a scope of the
        shards' own cap and holds it killed near the cap with its parent
        going on; a step after it shows the job goes on too."""
        job = self.MUT["jobs"]["memory-cap"]
        assert "if" not in job and "needs" not in job
        runs = [str(s.get("run", "")) for s in job["steps"]]
        at = runs.index('python3 scripts/memory_scope_check.py "$MUTANTS_MEMORY_MAX"')
        assert runs[at + 1 :], "a step after the check, which runs only if the runner lived"

    @pytest.mark.parametrize("job", ["changed", "weekly"])
    def test_every_mutants_run_tests_under_nextest_with_the_extended_tier(self, job):
        """Each mutant's tests run under nextest, a process per test (84 s
        for the full Rust suite against `cargo test`'s 182 s). cargo-mutants
        hands each `--cargo-test-arg` to `cargo nextest run` in the test
        phase: `--run-ignored=all` keeps the extended tier's
        `#[ignore = "extended: ..."]` tests in every mutant's run, as
        `-- --include-ignored` did under `cargo test`, and `--tests` the test
        targets; nextest runs no doctests in any case."""
        script = self._mutants_step(job)
        for flag in (
            "--test-tool nextest",
            "--cargo-test-arg=--tests",
            "--cargo-test-arg=--run-ignored=all",
        ):
            assert flag in script, (job, flag)
        assert "--include-ignored" not in script, job

    def test_nextest_is_one_pinned_version_where_mutants_are_tested(self):
        """The two jobs that test mutants install the same cargo-nextest, by
        exact version; the listing tests nothing and installs none."""
        installs = {
            job: [
                line.strip()
                for step in spec["steps"]
                for line in str(step.get("run", "")).splitlines()
                if "cargo-nextest" in line
            ]
            for job, spec in self.MUT["jobs"].items()
        }
        assert installs["changed"] == installs["weekly"], installs
        assert {j for j, found in installs.items() if found} == {"changed", "weekly"}
        (install,) = installs["changed"]
        assert re.fullmatch(
            r"cargo install --locked cargo-nextest --version \d+\.\d+\.\d+", install
        )

    @pytest.mark.parametrize("job", ["changed", "weekly"])
    def test_every_shard_stops_with_room_before_its_job_is_killed(self, job):
        """cargo-mutants is stopped (SIGINT, then the kill two minutes on)
        at least 25 minutes before the job's own limit, which leaves the
        setup (about 3 minutes and a 3-minute baseline) and the upload the
        room they need; and the job's limit stays inside GitHub's six hours."""
        spec = self.MUT["jobs"][job]
        script = next(
            s["run"]
            for s in spec["steps"]
            if "cargo mutants" in s.get("run", "") and "timeout" in s["run"]
        )
        stop = int(re.search(r"--kill-after=2m (\d+)m cargo mutants", script).group(1))
        assert stop + 2 + 23 <= spec["timeout-minutes"] <= 360, (job, stop, spec["timeout-minutes"])
        # The push of 5cb8f7e's slowest changed-lines shard took 98 minutes.
        if job == "changed":
            assert stop >= 3 * 98, stop

    #: The job that lists the changed lines' mutants and sizes their pass
    #: (task 219), and the shard count and indices it hands the shards.
    LISTING = "changed-list"
    COUNT = "${{ needs.changed-list.outputs.shards }}"
    MATRIX = "${{ fromJSON(needs.changed-list.outputs.matrix) }}"

    #: The slowest shard of run 37834489097, the push of 103d721: 51 mutants
    #: tested in its 100-minute stop, 114 s each at `-j 2` after a 208 s
    #: baseline (the fastest tested 99, the mean 65).
    SLOWEST_SHARD = 51

    @staticmethod
    def _runs(job: dict) -> str:
        return " ".join(str(step.get("run", "")) for step in job["steps"])

    def _count(self, job: str) -> str:
        """The shard count a sharded job deals over, as the workflow writes it."""
        if job == "changed":
            return self.COUNT
        return str(len(self.MUT["jobs"][job]["strategy"]["matrix"]["shard"]))

    def test_the_changed_lines_run_on_every_push_and_pull_request_and_gate(self):
        assert {"push", "pull_request"} <= set(self.MUT["on"])
        cond = " ".join(self.MUT["jobs"][self.LISTING]["if"].split())
        assert cond == (
            "github.event_name == 'push' || github.event_name == 'pull_request'"
            " || (github.event_name == 'workflow_dispatch' && inputs.base != '')"
        )
        assert self.MUT["jobs"]["changed"]["needs"] == self.LISTING
        assert "--in-diff changed.diff" in self.MUT["env"]["CHANGED_SCOPE"]
        report = self._runs(self.MUT["jobs"]["changed-report"])
        assert "mutants_report.py" in report and "--fail-on-missed" in report

    def test_a_dispatch_with_a_base_reruns_the_changed_lines_not_the_week(self):
        """A failed push's pass cannot be rerun under a fixed workflow (a
        rerun keeps its own workflow file) and the next push lists only its
        own diff, so a dispatch given `base` runs the changed-lines pass
        over base..HEAD, and the weekly pass only when it is given none:
        the push of 85da91e lost 23 of its 24 shards to a full /tmp (run
        40076792577, task 270 (m))."""
        base = self.MUT["on"]["workflow_dispatch"]["inputs"]["base"]
        assert base["required"] is False and base["default"] == ""
        weekly = " ".join(self.MUT["jobs"]["weekly"]["if"].split())
        assert "(github.event_name == 'workflow_dispatch' && inputs.base == '')" in weekly
        for job in (self.LISTING, "changed"):
            step = next(
                s for s in self.MUT["jobs"][job]["steps"] if "changed.diff" in str(s.get("run", ""))
            )
            assert step["env"]["BASE"].endswith("|| inputs.base }}"), job
            assert 'git diff "$BASE" HEAD -- crates/ > changed.diff' in step["run"], job

    def test_the_changed_lines_are_sharded_as_the_weekly_pass_is(self):
        """A shard deals round-robin, as the weekly pass does, so no shard
        takes one slow file whole, and one job reports them all."""
        for job, (report_job, prefix) in self.SHARDED.items():
            spec = self.MUT["jobs"][job]
            assert spec["strategy"]["fail-fast"] is False, job
            run = self._runs(spec)
            shard = f"--shard ${{{{ matrix.shard }}}}/{self._count(job)} --sharding round-robin"
            assert shard in run, job
            upload = next(s for s in spec["steps"] if "upload-artifact" in str(s.get("uses", "")))
            assert upload["with"]["name"] == f"{prefix}${{{{ matrix.shard }}}}", job
            report = self.MUT["jobs"][report_job]
            needs = [self.LISTING, job] if job == "changed" else job
            assert report["needs"] == needs, report_job
            cond = " ".join(report["if"].split())
            assert cond == f"always() && needs.{job}.result != 'skipped'", cond
            download = next(
                s for s in report["steps"] if "download-artifact" in str(s.get("uses", ""))
            )
            assert download["with"]["pattern"] == f"{prefix}*", report_job

    def test_the_changed_lines_take_as_many_shards_as_their_mutants_need(self):
        """The user, 2026-10-08: "We don't pay for minutes being open
        source." So the pass tests every mutant it lists, however large the
        push (task 219). Ten fixed shards could not: the push of 103d721
        listed 1,346, each shard stopped at its 100 minutes having tested 51
        to 99 of its 135, and 698 went untested (run 37834489097). A first
        job lists the pass's mutants, which builds nothing, and sizes the
        pass from the count, at most 40 a shard: four-fifths of the slowest
        shard's 51. The shards take their indices and the count from it, and
        the report expects that many runs, so a shard that sent nothing
        counts as untested."""
        listing = self.MUT["jobs"][self.LISTING]
        run = self._runs(listing)
        assert 'cargo mutants --list "${scope[@]}" > listed.txt' in run
        assert "--shard" not in run, "the listing lists the whole pass"
        size = re.search(
            r"python3 scripts/mutants_shards\.py listed\.txt --per-shard (\d+)"
            r' >> "\$GITHUB_OUTPUT"',
            run,
        )
        assert size, run
        assert 0 < int(size.group(1)) <= 0.8 * self.SLOWEST_SHARD, size.group(1)
        step = next(s for s in listing["steps"] if "mutants_shards.py" in str(s.get("run", "")))
        assert listing["outputs"] == {
            "shards": f"${{{{ steps.{step['id']}.outputs.shards }}}}",
            "matrix": f"${{{{ steps.{step['id']}.outputs.matrix }}}}",
        }
        changed = self.MUT["jobs"]["changed"]
        assert changed["strategy"]["matrix"] == {"shard": self.MATRIX}
        assert changed["name"].endswith(f"shard ${{{{ matrix.shard }}}} of {self.COUNT}")
        report = self._runs(self.MUT["jobs"]["changed-report"])
        assert f"--expect-runs {self.COUNT} " in report

    def test_the_listing_and_its_shards_list_the_same_mutants(self):
        """The listing's count sizes the pass and the shards deal it, so the
        two must list the same mutants: the same diff, from the same step in
        both jobs, and one scope, the workflow's `CHANGED_SCOPE`, which each
        job reads with `read -ra` (it splits on spaces and expands no glob,
        so `**` reaches cargo-mutants as written)."""
        listing, changed = self.MUT["jobs"][self.LISTING], self.MUT["jobs"]["changed"]

        def step(job: dict, prefix: str) -> dict:
            return next(s for s in job["steps"] if str(s.get("uses", "")).startswith(prefix))

        assert step(listing, "actions/checkout@") == step(changed, "actions/checkout@")
        assert step(listing, "actions/checkout@")["with"]["fetch-depth"] == 0
        diffs = [
            [s for s in job["steps"] if s.get("name") == "the diff this run examines"]
            for job in (listing, changed)
        ]
        assert diffs[0] == diffs[1] and len(diffs[0]) == 1, diffs
        scope = self.MUT["env"]["CHANGED_SCOPE"]
        assert "--shard" not in scope and "--list" not in scope, scope
        for job in (listing, changed):
            run = self._runs(job)
            assert 'read -ra scope <<< "$CHANGED_SCOPE"' in run
            for flag in ("--in-diff", "--package", "--file"):
                assert flag not in run, f"{flag} is spelled in a job, not in CHANGED_SCOPE"

    def test_cargo_mutants_is_one_pinned_version(self):
        """Every job installs the same cargo-mutants, by exact version, so
        the listing that sizes the pass lists what the shards deal."""
        installs = re.findall(r"cargo install .*cargo-mutants.*", self._all_runs())
        assert len(installs) == 3, installs
        assert len(set(installs)) == 1, installs
        assert re.fullmatch(
            r"cargo install --locked cargo-mutants --version \d+\.\d+\.\d+", installs[0]
        )

    def _all_runs(self) -> str:
        return "\n".join(
            str(step.get("run", "")) for job in self.MUT["jobs"].values() for step in job["steps"]
        )

    def test_the_weekly_pass_runs_while_public_or_by_hand(self):
        """COST POLICY: ninety-six shards of up to four hours is not for a
        private repo's metered minutes."""
        cond = " ".join(self.MUT["jobs"]["weekly"]["if"].split())
        assert "github.event_name == 'schedule' && github.event.repository.private == false" in cond
        assert "workflow_dispatch" in cond

    def test_the_weekly_pass_reports_and_never_gates(self):
        run = self._runs(self.MUT["jobs"]["weekly-report"])
        assert "mutants_report.py" in run and "--fail-on-missed" not in run

    def test_every_shard_runs(self):
        """The weekly matrix is 0 to 95 written out; the changed lines' is
        the listing's, 0 to the count less one, which
        `tests/test_mutants_report.py` holds `scripts/mutants_shards.py` to."""
        shards = self.MUT["jobs"]["weekly"]["strategy"]["matrix"]["shard"]
        assert shards == list(range(len(shards)))
        assert self.MUT["jobs"]["changed"]["strategy"]["matrix"]["shard"] == self.MATRIX
        for job in self.SHARDED:
            run = self._runs(self.MUT["jobs"][job])
            assert f"--shard ${{{{ matrix.shard }}}}/{self._count(job)}" in run, job

    def test_every_run_skips_the_doctests_and_leaves_a_survivor_room(self):
        """Task 155: a test run of online-core took 67 seconds, 39 of them its
        three doctests, and a mutant's time limit is a multiple of that run.
        Every run skips the doctests, which CI and the gate still run. The
        multiplier is 10, not the 3 first tried: a survivor runs the whole
        suite while the other jobs share the runner's cores, and nineteen
        survivors took 1.7 to 6.3 times the solo baseline at fourteen jobs on
        fourteen cores. At 3 the first full pass counted 1,588 timeouts, and
        19 of 25 sampled were survivors."""
        for job in ("changed", "weekly"):
            run = self._runs(self.MUT["jobs"][job])
            assert "--cargo-test-arg=--tests" in run, job
            assert "--timeout-multiplier 10" in run, job

    def test_a_run_stops_before_its_job_limit_and_counts_what_it_missed(self):
        """A job killed at its limit uploads and reports nothing. So each run
        stops cargo-mutants first, with SIGINT, which keeps the outcomes it
        finished; it writes the list of the mutants it was given; and the
        report fails on a run that tested fewer, or a shard that sent
        nothing (task 155)."""
        for job, (report_job, _) in self.SHARDED.items():
            j = self.MUT["jobs"][job]
            run = self._runs(j)
            m = re.search(r"timeout --signal=INT (?:--kill-after=\S+ )?(\d+)m cargo mutants", run)
            assert m, job
            assert int(m.group(1)) + 15 <= j["timeout-minutes"], job
            assert "cargo mutants --list" in run and "listed.txt" in run, job
            report = self._runs(self.MUT["jobs"][report_job])
            assert f"--expect-runs {self._count(job)}" in report, report_job
            assert "--fail-on-incomplete" in report, report_job

    def test_a_push_cannot_cancel_the_weekly_pass(self):
        assert "github.event_name" in self.MUT["concurrency"]["group"]

    def _group(self, event: str, ref: str, sha: str) -> str:
        """The concurrency group mutants.yml gives a run, its expressions
        evaluated. They use only contexts, quoted strings, `==`, `!=`, `&&`
        and `||`, which mean what Python's `==`, `!=`, `and` and `or` mean,
        values and precedence alike."""
        github = SimpleNamespace(event_name=event, ref=ref, sha=sha)

        def value(m: re.Match[str]) -> str:
            expr = m.group(1).replace("&&", " and ").replace("||", " or ")
            assert re.fullmatch(r"[\w.' =!()-]+", expr), expr
            return str(eval(expr, {"__builtins__": {}}, {"github": github}))

        return re.sub(r"\$\{\{(.*?)\}\}", value, self.MUT["concurrency"]["group"])

    def test_no_push_to_main_cancels_or_replaces_another_pushs_pass(self):
        """A push's changed-lines pass tests the lines it moved, and no
        later pass tests them again: the next push lists only its own diff.
        So no later push may cancel it, or replace it while it waits, and
        minutes are free (the user, 2026-10-08). `cancel-in-progress: false`
        would not do: GitHub keeps one pending run a group and cancels an
        older pending one, so of three quick pushes the middle one's pass
        would be lost. Each push has a group of its own, by its commit. A
        pull request's runs share one group and the newer commit cancels the
        older, since it lists the whole pull request's diff again; the
        weekly schedule and a dispatch keep one group each (task 219)."""
        main = "refs/heads/main"
        shas = ("a1" * 20, "b2" * 20, "c3" * 20)
        pushes = [self._group("push", main, sha) for sha in shas]
        assert len(set(pushes)) == 3, pushes
        assert all(sha in group for sha, group in zip(shas, pushes, strict=True)), pushes
        pr = "refs/pull/7/merge"
        prs = {self._group("pull_request", pr, sha) for sha in ("d4" * 20, "e5" * 20)}
        assert len(prs) == 1, prs
        assert self.MUT["concurrency"]["cancel-in-progress"] is True
        others = {
            self._group(event, main, "f6" * 20)
            for event in ("schedule", "workflow_dispatch", "pull_request")
        }
        assert not others & set(pushes), (others, pushes)
        assert self._group("schedule", main, "f6" * 20) == f"mutants-schedule-{main}"
        dispatch = self._group("workflow_dispatch", main, "f6" * 20)
        assert dispatch == f"mutants-workflow_dispatch-{main}"


class TestTheRustTestsLinkNoPython:
    """`cargo test --workspace` leaves online-py out, in every workflow and in
    the gate (2026-09-24). With it in the build, pyo3-polars turns on
    polars-error's `python` feature, and every test binary that links polars
    links libpython too. uv's own interpreters keep libpython off the Linux
    loader's path, so on the 3.13 and 3.14 legs the CLI's tests could not
    start. online-py has no Rust tests to lose; pytest covers it."""

    EXCLUDE = "--exclude online-py"
    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def test_every_workflow_leaves_the_extension_out(self):
        found = [
            (f"{name}:{job_name}", str(step.get("run", "")))
            for name, wf in ALL.items()
            for job_name, job in wf.get("jobs", {}).items()
            for step in job.get("steps", [])
            if "cargo test --workspace" in str(step.get("run", ""))
        ]
        assert {where.split(":")[0] for where, _ in found} >= {"ci.yml", "polars-canary.yml"}
        for where, run in found:
            assert self.EXCLUDE in run, where

    def test_the_gate_leaves_it_out_too(self):
        gate = (self.ROOT / "scripts/gate.sh").read_text(encoding="utf-8")
        runs = [
            line
            for line in gate.splitlines()
            if "cargo test --workspace" in line and not line.lstrip().startswith("#")
        ]
        assert runs and all(self.EXCLUDE in line for line in runs), runs

    def test_the_coverage_script_leaves_it_out_too(self):
        """`scripts/coverage.sh` runs the Rust tests under llvm-cov; without
        the exclusion every test binary links libpython, as above, and on
        Linux with a uv-managed Python the step cannot start (task 160, SC5)."""
        script = (self.ROOT / "scripts/coverage.sh").read_text(encoding="utf-8")
        runs = [
            line
            for line in script.splitlines()
            if "cargo llvm-cov" in line and not line.lstrip().startswith("#")
        ]
        assert runs and all(self.EXCLUDE in line for line in runs), runs

    def _packages(self, *selection: str) -> set[str]:
        """Every package cargo would build for `selection`, dev and build
        dependencies included, from this checkout's lock. `--locked`, not
        `--frozen`: release.yml runs pytest where only the extension has
        been built, so cargo may have crates to fetch, as the CLI fixture does."""
        cmd = ["cargo", "tree", *selection, "--locked", "--prefix", "none", "--format", "{p}"]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", cwd=self.ROOT, check=False
            )
        except FileNotFoundError:
            pytest.fail("cargo is not on PATH; `source scripts/env.sh` first")
        assert res.returncode == 0, res.stderr
        return {line.split()[0] for line in res.stdout.splitlines() if line.strip()}

    def test_without_it_nothing_in_the_build_links_python(self):
        """The mechanism, asked of cargo: pyo3-ffi is the package whose build
        script links libpython. The control shows the check can fail: with
        online-py in, it is there."""
        assert "pyo3-ffi" in self._packages("--workspace")
        left = self._packages("--workspace", *self.EXCLUDE.split())
        assert "online-cli" in left and "online-polars" in left
        assert not {"pyo3", "pyo3-ffi"} & left, sorted({"pyo3", "pyo3-ffi"} & left)


PREP = "./.github/actions/linux-build-prep"
PREP_ACTION = WORKFLOWS[0].parents[1] / "actions/linux-build-prep/action.yml"
REPORT_ACTION = WORKFLOWS[0].parents[1] / "actions/machine-report/action.yml"


def _every_step():
    for name, workflow in sorted(ALL.items()):
        for job, spec in workflow.get("jobs", {}).items():
            yield name, job, spec, spec.get("steps", [])


class TestTheLinuxPrepIsOneAction:
    """The step that frees the runner's disk and swaps in lld was five copies
    of the same ten lines (docs/SIMPLIFICATION.md, S7). One composite action
    now, so a sixth workflow calls it rather than copying whichever version
    it finds first."""

    def test_every_prep_step_calls_the_action_and_none_copies_it(self):
        callers = []
        for name, job, _, steps in _every_step():
            for step in steps:
                if "free disk" in step.get("name", ""):
                    assert step.get("uses") == PREP, f"{name}:{job} does not call {PREP}"
                    callers.append(f"{name}:{job}")
                run = str(step.get("run", ""))
                assert "fuse-ld=lld" not in run and "/usr/share/dotnet" not in run, (
                    f"{name}:{job} copies the prep again"
                )
        # Every caller by name, so a new one is a decision: the two NumPy
        # jobs were added on 2026-09-30, the two floor legs on 2026-10-07,
        # the instruction counts (Valgrind) on 2026-10-12.
        assert sorted(callers) == [
            "budgets.yml:instructions",
            "ci.yml:test",
            "leakcheck.yml:native",
            "mutants.yml:changed",
            "polars-canary.yml:floor-polars",
            "polars-canary.yml:latest-polars",
            "polars-canary.yml:next-numpy",
            "release.yml:floor-polars",
            "release.yml:next-numpy",
            "release.yml:next-polars",
        ], callers

    def test_a_local_action_follows_the_checkout(self):
        """A local action is read from the checked-out tree."""
        for name, job, _, steps in _every_step():
            uses = [str(s.get("uses", "")) for s in steps]
            for i, u in enumerate(uses):
                if u.startswith("./"):
                    assert any(x.startswith("actions/checkout@") for x in uses[:i]), (
                        f"{name}:{job} calls {u} before the checkout"
                    )

    def test_it_runs_on_linux_alone(self):
        """The action refuses another OS, so a matrix job guards the call."""
        for name, job, spec, steps in _every_step():
            for step in steps:
                if step.get("uses") != PREP:
                    continue
                if str(spec.get("runs-on", "")).startswith("ubuntu-"):
                    continue
                assert step.get("if") == "runner.os == 'Linux'", f"{name}:{job} is unguarded"

    def test_the_action_frees_the_disk_then_installs_then_writes_the_linker(self):
        action = yaml.safe_load(PREP_ACTION.read_text(encoding="utf-8"))
        assert action["runs"]["using"] == "composite"
        (step,) = action["runs"]["steps"]
        run = step["run"]
        at = [run.index(x) for x in ("RUNNER_OS", "rm -rf", "apt-get install", "fuse-ld=lld")]
        assert at == sorted(at), "out of order"
        # Inputs reach the script through its environment, never pasted in.
        assert "${{" not in run
        assert set(action["inputs"]) == {"packages", "update-lists"}


class TestActionsArePinnedToCommits:
    """`.github/dependabot.yml` says actions are pinned to commit SHAs, since a
    tag or a branch can be repointed by whoever owns the action; Dependabot
    then proposes the bumps. `dtolnay/rust-toolchain@stable` was a branch, at
    13 sites, release.yml's wheel build among them (task 160, CI1). Read from
    the text, since YAML drops the comment that names the version."""

    FILES = [*WORKFLOWS, PREP_ACTION, REPORT_ACTION]
    USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+?)@(\S+)(.*)$")

    def _external(self):
        for path in self.FILES:
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = self.USES.match(line)
                if m and not m.group(1).startswith("./"):
                    yield f"{path.name}:{n}", m.group(1), m.group(2), m.group(3)

    def test_every_external_action_is_a_commit_with_its_version_named(self):
        found = list(self._external())
        assert len(found) > 40, len(found)
        for where, action, ref, rest in found:
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{where}: {action}@{ref} is not a commit"
            assert re.match(r"\s+#\s+\S", rest), f"{where}: {action} names no version"


class TestTheDeclaredRustVersionBuildsTheLock:
    """`rust-version` in the workspace's Cargo.toml is the oldest Rust a
    source build supports. It said 1.85 while the locked tree held `sysinfo`
    0.39.6 (1.95) and `simd-json` 0.17.3 (1.88), so an sdist install on Rust
    1.85-1.94 failed in cargo (task 160, CI4). The declaration must cover
    every resolved package's own."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    @staticmethod
    def _version(text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in text.split("."))

    def test_the_workspace_declares_at_least_what_its_dependencies_need(self):
        cmd = ["cargo", "metadata", "--locked", "--format-version", "1"]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", cwd=self.ROOT, check=False
            )
        except FileNotFoundError:
            pytest.fail("cargo is not on PATH; `source scripts/env.sh` first")
        assert res.returncode == 0, res.stderr
        import json

        meta = json.loads(res.stdout)
        members = set(meta["workspace_members"])
        declared = {p["name"]: p["rust_version"] for p in meta["packages"] if p["id"] in members}
        needed = max(
            (
                (p["rust_version"], p["name"])
                for p in meta["packages"]
                if p.get("rust_version") and p["id"] not in members
            ),
            key=lambda t: self._version(t[0]),
        )
        assert len(set(declared.values())) == 1, declared
        floor = next(iter(declared.values()))
        assert self._version(floor) >= self._version(needed[0]), (floor, needed)


class TestTheDeclaredRustVersionCompiles:
    """The check above reads what the dependencies declare, and no job
    compiled the workspace on the Rust it declares: every build used the
    newest stable, so a source build on an older Rust the declaration admits
    was never tried (review 2026-10-06, CI11). A weekly job checks the whole
    workspace on exactly that Rust, from the lock as committed. The version
    is read from Cargo.toml, so a change to `rust-version` moves the job
    with it or fails here."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def _declared(self) -> str:
        cargo = tomllib.loads((self.ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        return cargo["workspace"]["package"]["rust-version"]

    def test_a_weekly_linux_job_checks_the_workspace_on_the_declared_rust(self):
        declared = self._declared()
        workflow = ALL["msrv.yml"]
        assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}, workflow["on"]
        (cron,) = (entry["cron"] for entry in workflow["on"]["schedule"])
        minute, hour, day, month, weekday = cron.split()
        assert (day, month) == ("*", "*") and weekday.isdigit(), f"not weekly: {cron}"
        (job,) = workflow["jobs"].values()
        assert job["runs-on"] == "ubuntu-latest"
        toolchains = [
            s.get("with", {}).get("toolchain")
            for s in job["steps"]
            if "dtolnay/rust-toolchain@" in str(s.get("uses", ""))
        ]
        assert toolchains == [declared], toolchains
        runs = [str(s.get("run", "")) for s in job["steps"]]
        # `+<version>`: the repository's rust-toolchain.toml names `stable`,
        # which a bare `cargo` inside the checkout would take instead.
        assert any(f"cargo +{declared} check --workspace --locked" in r for r in runs), runs


class TestEveryDeselectedMarkerRunsSomewhere:
    """pyproject's addopts leaves the soak out of every run that does not ask
    for it, and no workflow asked: its resume test failed unseen from task 120
    to task 160 (TC11). Each marker the default run deselects must be the
    `-m` of some workflow's pytest step."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def test_each_marker_the_default_run_leaves_out_runs_in_a_workflow(self):
        pyproject = tomllib.loads((self.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
        deselected = re.findall(r"\bnot (\w+)", addopts)
        assert deselected, addopts
        runs = [
            str(step.get("run", ""))
            for wf in ALL.values()
            for job in wf["jobs"].values()
            for step in job.get("steps", [])
        ]
        for marker in deselected:
            asks = re.compile(rf"pytest\b.*\s-m\s+(['\"]?){marker}\1(\s|$)")
            assert any(asks.search(run) for run in runs), f"no workflow runs `-m {marker}`"
