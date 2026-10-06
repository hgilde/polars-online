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
import re
import subprocess
import tomllib

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

    #: Jobs allowed past two hours, with their cap. The weekly mutation
    #: shards run on Linux, only while the repository is public or by hand,
    #: and stop cargo-mutants well inside the limit; four hours is the
    #: backstop for a slow week (task 155).
    LONGER = {("mutants.yml", "weekly"): 240}

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_every_job_has_a_timeout(self, name):
        for job, spec in ALL[name].get("jobs", {}).items():
            # A job that calls a reusable workflow takes no timeout; the
            # called workflow's jobs carry theirs, and are checked here too.
            if "uses" in spec:
                continue
            assert "timeout-minutes" in spec, f"{name}:{job} has no timeout-minutes"
            cap = self.LONGER.get((name, job), 120)
            assert 0 < spec["timeout-minutes"] <= cap, f"{name}:{job} timeout is not sane"

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

    @staticmethod
    def _runs(job: dict) -> str:
        return " ".join(str(step.get("run", "")) for step in job["steps"])

    def test_the_changed_lines_run_on_every_push_and_pull_request_and_gate(self):
        assert {"push", "pull_request"} <= set(self.MUT["on"])
        run = self._runs(self.MUT["jobs"]["changed"])
        assert "--in-diff" in run and "mutants_report.py" in run and "--fail-on-missed" in run

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
        shards = self.MUT["jobs"]["weekly"]["strategy"]["matrix"]["shard"]
        assert shards == list(range(len(shards)))
        assert f"--shard ${{{{ matrix.shard }}}}/{len(shards)}" in self._runs(
            self.MUT["jobs"]["weekly"]
        )

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
        for job in ("changed", "weekly"):
            j = self.MUT["jobs"][job]
            run = self._runs(j)
            m = re.search(r"timeout --signal=INT (?:--kill-after=\S+ )?(\d+)m cargo mutants", run)
            assert m, job
            assert int(m.group(1)) + 15 <= j["timeout-minutes"], job
            assert "cargo mutants --list" in run and "listed.txt" in run, job
        shards = len(self.MUT["jobs"]["weekly"]["strategy"]["matrix"]["shard"])
        report = self._runs(self.MUT["jobs"]["weekly-report"])
        assert f"--expect-runs {shards}" in report and "--fail-on-incomplete" in report
        assert "--fail-on-incomplete" in self._runs(self.MUT["jobs"]["changed"])

    def test_a_push_cannot_cancel_the_weekly_pass(self):
        assert "github.event_name" in self.MUT["concurrency"]["group"]


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
        # jobs were added on 2026-09-30.
        assert sorted(callers) == [
            "ci.yml:test",
            "leakcheck.yml:native",
            "mutants.yml:changed",
            "polars-canary.yml:latest-polars",
            "polars-canary.yml:next-numpy",
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

    FILES = [*WORKFLOWS, PREP_ACTION]
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
