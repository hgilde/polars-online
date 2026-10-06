"""The release workflow tags what it published, after everything ran.

The user, 2026-09-27: the 0.11.0 tag was pushed by hand, CI met the branch
there for the first time and failed, and a release tag cannot be moved. So a
release is dispatched on ``main``: with ``publish`` off it is the rehearsal;
on, it runs CI and every build and suite, waits for approval, uploads to
PyPI, and only then creates the tag and the GitHub release. These tests hold
the workflow's shape to that, and ``scripts/release_version.py`` -- the check
that runs before anything is built -- to its rules.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


RELEASE = _load("release.yml")
CI = _load("ci.yml")
# PyYAML reads the key `on` as the boolean True.
TRIGGERS = RELEASE.get("on", RELEASE.get(True))
JOBS = RELEASE["jobs"]


def _needs(job: str) -> set[str]:
    n = JOBS[job].get("needs", [])
    return {n} if isinstance(n, str) else set(n)


def test_a_release_is_dispatched_never_tagged_by_hand():
    """No tag trigger: a tag pushed by hand starts nothing, and the only
    trigger is a dispatch whose ``publish`` input is off unless asked."""
    assert set(TRIGGERS) == {"workflow_dispatch"}
    publish = TRIGGERS["workflow_dispatch"]["inputs"]["publish"]
    assert publish["type"] == "boolean"
    assert publish["default"] is False


def test_the_version_is_checked_before_anything_is_built():
    for job in ("sdist", "build"):
        assert "version" in _needs(job), job
    steps = " ".join(str(s.get("run", "")) for s in JOBS["version"]["steps"])
    assert "scripts/release_version.py" in steps
    assert "--publish" in steps
    main_only = next(s for s in JOBS["version"]["steps"] if "main only" in s.get("name", ""))
    assert "refs/heads/main" in main_only["if"]


def test_ci_runs_inside_the_release_on_every_os():
    """The 0.11.0 tag failed in CI, which the release did not run."""
    assert JOBS["ci"]["uses"] == "./.github/workflows/ci.yml"
    ci_triggers = CI.get("on", CI.get(True))
    assert "workflow_call" in ci_triggers
    # A call takes the caller's event: a dispatch, whose clause of the
    # matrix brings macOS and Windows.
    matrix = " ".join(str(CI["jobs"]["test"]["strategy"]["matrix"]["os"]).split())
    clause = next(c for c in matrix.split("||") if "workflow_dispatch" in c)
    assert "macos-latest" in clause and "windows-latest" in clause


def test_the_pages_grant_to_ci_is_inert():
    """A called workflow may ask for no more than its caller grants, so the
    release grants ``docs``'s scopes; ``docs`` runs on a push to main alone,
    never under the release's dispatch, so the grant deploys nothing."""
    grant = JOBS["ci"]["permissions"]
    assert set(grant) == {"contents", "pages", "id-token"}
    assert grant["contents"] == "read"
    docs_if = CI["jobs"]["docs"]["if"]
    assert "github.event_name == 'push'" in docs_if
    for job, spec in CI["jobs"].items():
        if job != "docs":
            assert spec["permissions"] == {"contents": "read"}, job


def test_ci_groups_per_caller_so_a_release_and_a_push_do_not_cancel():
    assert "github.workflow" in CI["concurrency"]["group"]


def _transitive_needs(job: str) -> set[str]:
    seen: set[str] = set()
    todo = [job]
    while todo:
        for need in _needs(todo.pop()):
            if need not in seen:
                seen.add(need)
                todo.append(need)
    return seen


def test_publishing_waits_for_every_job_and_the_tag_waits_for_publishing():
    """Every job but the three that follow the upload is upstream of it,
    directly or through another job. A literal list of ``publish``'s needs
    passed with a new job left out of it (task 160, TC8)."""
    assert set(JOBS) - {"publish", "tag", "release"} <= _transitive_needs("publish")
    assert JOBS["publish"]["environment"] == "pypi"
    assert _needs("tag") == {"version", "publish"}
    assert _needs("release") == {"version", "tag"}
    for job in ("publish", "tag", "release"):
        assert JOBS[job]["if"] == "inputs.publish", job


def test_numpy_is_tested_at_its_newest_and_its_next_release_candidate():
    """NumPy is the one optional dependency, `numpy>=1.24` with no ceiling,
    and every other job runs on the locked NumPy (the user, 2026-09-30:
    "test on the next release candidate of that too as we do with polars").
    Its newest stable is inside what the extra promises, so that leg holds
    back the upload; its next release candidate is early warning. Only NumPy
    moves, so a red leg names it."""
    job = JOBS["next-numpy"]
    legs = job["strategy"]["matrix"]["include"]
    blocking = [leg for leg in legs if leg["blocking"]]
    advisory = [leg for leg in legs if not leg["blocking"]]
    assert len(blocking) == 1 and blocking[0]["prerelease"] == ""
    assert len(advisory) == 1 and advisory[0]["prerelease"] == "--prerelease=allow"
    assert job["continue-on-error"] == "${{ !matrix.blocking }}"
    runs = [str(s.get("run", "")) for s in job["steps"]]
    assert "uv sync ${{ matrix.prerelease }} --upgrade-package numpy" in runs
    assert not any("--upgrade-package polars" in r for r in runs)
    assert any("pytest" in r for r in runs)
    canary = _load("polars-canary.yml")["jobs"]["next-numpy"]
    canary_runs = [str(s.get("run", "")) for s in canary["steps"]]
    assert "uv sync --prerelease=allow --upgrade-package numpy" in canary_runs
    assert any("pytest" in r for r in canary_runs)


def test_every_wheel_is_installed_and_run_where_it_belongs():
    """Three of the six wheels were built and uploaded without ever being
    imported, and none was run as the file that ships (task 160, CI3). Each
    leg installs its own wheel into a fresh environment and runs
    ``scripts/wheel_smoke.py`` on it with the version being released: on
    the runner, or in Alpine for the musl wheel, which the runner's glibc
    cannot load."""
    steps = JOBS["build"]["steps"]
    smoke = [s for s in steps if "scripts/wheel_smoke.py" in str(s.get("run", ""))]
    assert [s["if"] for s in smoke] == [
        "matrix.manylinux != 'musllinux_1_2'",
        "matrix.manylinux == 'musllinux_1_2'",
    ]
    for leg in JOBS["build"]["strategy"]["matrix"]["include"]:
        assert leg["target"].endswith("-musl") == (leg.get("manylinux") == "musllinux_1_2"), leg
    native, alpine = smoke
    assert "uv venv" in native["run"]
    assert "uv pip install --python smoke dist/*.whl" in native["run"]
    assert "python:3.12-alpine" in alpine["run"] and "pip install" in alpine["run"]
    for step in smoke:
        assert step["env"]["VERSION"] == "${{ needs.version.outputs.version }}"
        assert '--version "$VERSION"' in step["run"]
    # Last, after both uploads: a failed run leaves the artifacts to look at.
    assert steps.index(native) > max(
        i for i, s in enumerate(steps) if "upload-artifact" in str(s.get("uses", ""))
    )


def _wheel_smoke():
    spec = importlib.util.spec_from_file_location(
        "wheel_smoke", REPO / "scripts" / "wheel_smoke.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wheel_smoke_passes_on_this_build_and_refuses_the_wrong_version():
    import polars_online as po

    smoke = _wheel_smoke()
    assert smoke.check(po.__version__, installed=False).endswith(": ok")
    with pytest.raises(SystemExit, match="not 9.9.9"):
        smoke.check("9.9.9", installed=False)


def test_the_wheel_smoke_refuses_a_package_imported_from_the_checkout():
    import polars_online as po

    smoke = _wheel_smoke()
    if Path(po.__file__).resolve().is_relative_to(REPO / "python"):
        with pytest.raises(SystemExit, match="came from the checkout"):
            smoke.check(po.__version__)
    else:
        assert smoke.check(po.__version__).endswith(": ok")


def test_the_tag_is_on_the_tested_sha():
    run = next(s["run"] for s in JOBS["tag"]["steps"] if "create and push" in s.get("name", ""))
    assert 'git tag -a "$TAG" -F notes.md "$GITHUB_SHA"' in run
    assert JOBS["tag"]["permissions"] == {"contents": "write"}
    gh_release = next(
        s for s in JOBS["release"]["steps"] if "action-gh-release" in s.get("uses", "")
    )
    assert gh_release["with"]["tag_name"] == "${{ needs.version.outputs.tag }}"


def _build_steps() -> list[dict]:
    return JOBS["build"]["steps"]


def _step(name_part: str) -> tuple[int, dict]:
    return next((i, s) for i, s in enumerate(_build_steps()) if name_part in s.get("name", ""))


def test_the_linux_cli_is_built_in_the_wheels_manylinux2014_image():
    """docs/PLAN.md task 115 (i): built on the runner, 0.11.1's Linux CLI took
    the runner's glibc, 2.39. In the image the wheels come from, it is 2.17;
    the image is chosen by the runner's architecture, so arm64 stays native."""
    _, linux = _step("build the CLI (Linux")
    assert linux["if"] == "matrix.cli && runner.os == 'Linux'"
    run = linux["run"]
    assert '"quay.io/pypa/manylinux2014_$(uname -m)"' in run
    assert "cargo build --release --locked -p online-cli" in run
    assert "CARGO_TARGET_DIR=/io/target-cli" in run
    # The runner's own build is for macOS and Windows alone.
    host = next(s for s in _build_steps() if s.get("name") == "build the CLI")
    assert host["if"] == "matrix.cli && runner.os != 'Linux'"


def test_the_linux_cli_is_held_to_glibc_2_17_before_it_is_uploaded():
    """The floor is read from the binary and refused above 2.17 in the build
    job, which the publish job needs: a regression stops the release before
    anything reaches PyPI."""
    at_build, _ = _step("build the CLI (Linux")
    at_check, check = _step("needs glibc 2.17")
    at_upload = next(
        i for i, s in enumerate(_build_steps()) if "cli-" in str(s.get("with", {}).get("name", ""))
    )
    assert at_build < at_check < at_upload
    assert check["run"].split() == [
        "python3",
        "scripts/glibc_floor.py",
        "target-cli/release/${{",
        "matrix.bin",
        "}}",
        "--max",
        "2.17",
    ]
    upload = _build_steps()[at_upload]
    assert (
        "target-cli" in upload["with"]["path"] and "runner.os == 'Linux'" in upload["with"]["path"]
    )
    assert "build" in _needs("publish")


_glibc = importlib.util.spec_from_file_location("glibc_floor", REPO / "scripts" / "glibc_floor.py")
assert _glibc and _glibc.loader
glibc_floor = importlib.util.module_from_spec(_glibc)
_glibc.loader.exec_module(glibc_floor)


def test_glibc_versions_are_ordered_as_numbers():
    """2.9 is older than 2.17, and 2.2.5 older than both: a sort of the
    strings says otherwise. The text is `objdump -p`'s, as 0.11.1's x86_64
    CLI printed it, trimmed."""
    text = """Version References:
  required from libgcc_s.so.1:
    0x0b792650 0x00 05 GCC_3.0
  required from libc.so.6:
    0x06969195 0x00 04 GLIBC_2.17
    0x06969199 0x00 03 GLIBC_2.9
    0x09691a75 0x00 02 GLIBC_2.2.5
    0x069691b9 0x02 09 GLIBC_2.39
    0x069691b9 0x00 07 GLIBC_PRIVATE
"""
    got = glibc_floor.versions(text)
    assert got == [(2, 2, 5), (2, 9), (2, 17), (2, 39)]
    assert glibc_floor.dotted(got[-1]) == "2.39"
    assert glibc_floor.versions("GCC_3.0 only") == []


def test_only_the_release_jobs_write():
    for job, spec in JOBS.items():
        perms = spec.get("permissions", {})
        if job in ("tag", "release"):
            assert perms == {"contents": "write"}, job
        elif job == "publish":
            assert perms == {"id-token": "write"}, job
        elif job != "ci":
            assert perms == {"contents": "read"}, job


# --- scripts/release_version.py --------------------------------------------

_spec = importlib.util.spec_from_file_location(
    "release_version", REPO / "scripts/release_version.py"
)
release_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_version)

PLACES = [
    "pyproject.toml",
    "Cargo.toml",
    "python/polars_online/__init__.py",
    "docs/VALIDATION.md",
    "CHANGELOG.md",
]


@pytest.fixture
def copy(tmp_path):
    for rel in PLACES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    return tmp_path


def test_the_repository_agrees_with_itself():
    found = release_version.versions()
    assert len(set(found.values())) == 1, found
    assert release_version.problems() == []


def test_a_place_that_disagrees_is_named(copy):
    init = copy / "python/polars_online/__init__.py"
    v = release_version.versions(copy)["pyproject.toml"]
    init.write_text(init.read_text(encoding="utf-8").replace(f'"{v}"', '"9.9.9"'), encoding="utf-8")
    (bad,) = release_version.problems(copy)
    assert "python/polars_online/__init__.py = 9.9.9" in bad


def test_a_version_without_a_changelog_section_is_refused(copy):
    log = copy / "CHANGELOG.md"
    v = release_version.versions(copy)["pyproject.toml"]
    log.write_text(
        log.read_text(encoding="utf-8").replace(f"## [{v}]", "## [0.0.0]"), encoding="utf-8"
    )
    assert release_version.problems(copy) == [f"CHANGELOG.md has no `## [{v}]` section"]


def test_publishing_a_version_whose_tag_exists_is_refused(copy):
    v = release_version.versions(copy)["pyproject.toml"]
    asked = []

    def exists(tag):
        asked.append(tag)
        return True

    (bad,) = release_version.problems(copy, publish=True, tag_exists=exists)
    assert asked == [f"v{v}"]
    assert "immutable" in bad
    assert release_version.problems(copy, publish=True, tag_exists=lambda t: False) == []
    # Rehearsing never asks.
    assert release_version.problems(copy, tag_exists=exists) == []
