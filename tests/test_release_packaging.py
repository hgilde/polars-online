"""T-W5: the release workflow's artifact-collection step, run for real.

`release.yml` renames each CLI binary to `polars-online-<target>` and has to keep the
`.exe` suffix on the Windows one. That step is bash, it runs only on the ubuntu
job, and it has never processed a real `polars-online.exe` -- the Windows build has
never run, because nothing has ever been pushed.

The shell is extracted from the workflow rather than copied here, so editing
the workflow changes what this test runs. That is the whole point: a copy would
drift and keep passing.
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

TIER = "essential"

# The step under test is a `run:` block that executes only on the workflow's
# ubuntu job. Running it needs a POSIX shell with `find`, `basename` and
# `${f##*.}`; Git Bash on a Windows runner has a different `find` on PATH and
# fails for reasons that say nothing about the workflow. Skipping here loses no
# coverage — Linux and macOS both run it, and it is Linux that executes it for
# real.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the release step is bash-on-ubuntu; a Windows shell tests nothing real",
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "release.yml"

TARGETS = {
    "x86_64-pc-windows-msvc": "polars-online.exe",
    "aarch64-apple-darwin": "polars-online",
    "x86_64-unknown-linux-gnu": "polars-online",
}
WHEEL = "polars_online-0.1.0-cp312-abi3-macosx_11_0_arm64.whl"


def _step_script(name: str) -> str:
    """The `run:` block of the named step, dedented to a runnable script."""
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(
        rf"^(\s*)- name: {re.escape(name)}\n\1  run: \|\n(?P<body>(?:\1    .*\n|\n)+)",
        text,
        re.M,
    )
    assert m, f"no step named {name!r} with a `run: |` block in release.yml"
    return textwrap.dedent(m.group("body"))


@pytest.fixture
def staged(tmp_path):
    """What `download-artifact` leaves behind: one directory per matrix job."""
    for target, binary in TARGETS.items():
        d = tmp_path / "artifacts" / f"cli-{target}"
        d.mkdir(parents=True)
        (d / binary).write_bytes(b"\x7fELF fake binary")
    wheels = tmp_path / "artifacts" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / WHEEL).write_bytes(b"PK fake wheel")
    # The budgets baseline is a release asset too; the wheels' sizes go
    # into the notes, not the assets.
    baseline = tmp_path / "artifacts" / "budgets-baseline"
    baseline.mkdir(parents=True)
    (baseline / "budgets-baseline.json").write_text("{}", encoding="utf-8")
    sizes = tmp_path / "artifacts" / "wheel-sizes"
    sizes.mkdir(parents=True)
    (sizes / "sizes.md").write_text("### Wheel sizes\n", encoding="utf-8")
    return tmp_path


def _collect(cwd) -> list[str]:
    res = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step_script("collect the files")],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return sorted(p.name for p in (Path(cwd) / "release").iterdir())


def test_every_artifact_is_collected_and_named_by_target(staged):
    assert _collect(staged) == [
        "budgets-baseline.json",
        "polars-online-aarch64-apple-darwin",
        "polars-online-x86_64-pc-windows-msvc.exe",
        "polars-online-x86_64-unknown-linux-gnu",
        WHEEL,
    ]


def test_the_windows_binary_keeps_its_suffix_and_the_others_gain_none(staged):
    names = _collect(staged)
    exes = [n for n in names if n.endswith(".exe")]
    assert exes == ["polars-online-x86_64-pc-windows-msvc.exe"], (
        "exactly the Windows artifact keeps .exe; a rename that drops it "
        "produces a file Windows will not execute, and one that adds it "
        "everywhere breaks the unix downloads"
    )


def test_the_binaries_are_copied_not_just_renamed(staged):
    names = _collect(staged)
    for name in ["polars-online-aarch64-apple-darwin", "polars-online-x86_64-pc-windows-msvc.exe"]:
        assert name in names, f"{name} was not produced; got {names}"
        assert (staged / "release" / name).read_bytes() == b"\x7fELF fake binary"


def test_it_does_not_collapse_two_targets_onto_one_name(staged):
    """Both unix binaries are called `polars-online` in their own directories, so a
    rename that forgot the target would silently ship one of them twice."""
    names = _collect(staged)
    assert len(names) == len(set(names))
    assert len([n for n in names if n.startswith("polars-online-")]) == len(TARGETS)


def test_the_publish_job_collects_only_wheels(staged):
    res = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step_script("collect the wheels")],
        cwd=str(staged),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert sorted(p.name for p in (staged / "dist").iterdir()) == [WHEEL], (
        "the PyPI job must upload wheels only -- a CLI binary in dist/ would "
        "be rejected by the index"
    )


# --- the tag and the release, retried (task 317) ---------------------------------

RELEASE = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _run(job: str, step: str) -> str:
    """The shell of a step of a job, as the workflow holds it."""
    return next(s["run"] for s in RELEASE["jobs"][job]["steps"] if s.get("name") == step)


GIT = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "tag.gpgSign=false"]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *GIT, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()


@pytest.fixture
def remote(tmp_path):
    """A bare remote and a clone of it with two commits, the release's
    checkout: the tag goes on the first, and the CHANGELOG's notes are
    written as the tag job writes them."""
    bare = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(bare), str(work))
    for message in ("the release", "after it"):
        _git(work, "commit", "-q", "--allow-empty", "-m", message)
    _git(work, "push", "-q", "origin", "HEAD:refs/heads/main")
    (work / "notes.md").write_text("polars-online 9.9.9\n\n### Added\n", encoding="utf-8")
    return bare, work, _git(work, "rev-parse", "HEAD~1"), _git(work, "rev-parse", "HEAD")


def _push_tag(work: Path, sha: str, origin: str | None = None) -> subprocess.CompletedProcess:
    if origin is not None:
        _git(work, "remote", "set-url", origin.split("=")[0], origin.split("=")[1])
    env = {"PATH": os.environ["PATH"], "HOME": str(work), "TAG": "v9.9.9", "GITHUB_SHA": sha}
    env |= {"RETRY_PAUSE": "0", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["bash", "-e", "-c", _run("tag", "create and push the tag")],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _remote_tag(bare: Path) -> str:
    return _git(bare, "rev-parse", "refs/tags/v9.9.9^{commit}")


def test_the_tag_push_lands_once(remote):
    bare, work, sha, _ = remote
    res = _push_tag(work, sha)
    assert res.returncode == 0, res.stderr
    assert _remote_tag(bare) == sha


def test_a_tag_that_landed_already_on_this_sha_is_success(remote):
    """A push whose answer was lost, retried, or the job re-run: the tag is
    on the remote at the tested sha, the push is refused, and the job asks
    the remote and succeeds."""
    bare, work, sha, _ = remote
    assert _push_tag(work, sha).returncode == 0
    _git(work, "tag", "-d", "v9.9.9")
    # The re-run's annotation is another object (another time, here other
    # text), so the remote refuses the push rather than finding it equal.
    (work / "notes.md").write_text("polars-online 9.9.9, again\n", encoding="utf-8")
    res = _push_tag(work, sha)
    assert res.returncode == 0, res.stderr
    assert f"on the remote at {sha} already" in res.stdout
    assert _remote_tag(bare) == sha


def test_a_tag_on_another_sha_fails_without_retrying(remote):
    bare, work, sha, other = remote
    _git(work, "tag", "v9.9.9", other)
    _git(work, "push", "-q", "origin", "refs/tags/v9.9.9")
    _git(work, "tag", "-d", "v9.9.9")
    res = _push_tag(work, sha)
    assert res.returncode == 1
    assert f"on the remote at {other}, not {sha}" in res.stdout
    assert res.stderr.count("git push") <= 1 and res.stdout.count("::error::") == 1
    assert _remote_tag(bare) == other


def test_a_remote_that_never_answers_is_tried_three_times(remote):
    _, work, sha, _ = remote
    _git(work, "remote", "set-url", "origin", str(work.parent / "nowhere.git"))
    res = _push_tag(work, sha)
    assert res.returncode == 1
    assert "could not be pushed in three attempts" in res.stdout
    assert res.stderr.count("does not appear to be a git repository") >= 3, res.stderr


#: A stand-in for `gh`: each call is logged; `view` finds the release once
#: a `create` has run (or when told it exists), and the first FAILS calls
#: of `create` or `upload` fail, as a lost answer would.
FAKE_GH = """#!/usr/bin/env bash
echo "$*" >> "$GH_LOG"
case "$1 $2" in
  "release view") [ -e "$GH_STATE/exists" ] ;;
  "release create")
    touch "$GH_STATE/exists"
    n=$(cat "$GH_STATE/fails" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$GH_STATE/fails"; exit 1; fi ;;
  *) exit 0 ;;
esac
"""


def _release(tmp_path: Path, exists: bool = False, fails: int = 0) -> tuple[int, list[str]]:
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    if exists:
        (state / "exists").touch()
    (state / "fails").write_text(str(fails), encoding="utf-8")
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / WHEEL).write_bytes(b"PK")
    log = tmp_path / "gh.log"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_LOG": str(log),
        "GH_STATE": str(state),
        "TAG": "v9.9.9",
        "RETRY_PAUSE": "0",
    }
    res = subprocess.run(
        ["bash", "-e", "-c", _run("release", "create the release")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return res.returncode, log.read_text(encoding="utf-8").splitlines()


def test_the_release_is_created_once_from_the_tag(tmp_path):
    code, calls = _release(tmp_path)
    assert code == 0
    assert calls == [
        "release view v9.9.9",
        f"release create v9.9.9 release/{WHEEL} --verify-tag --title v9.9.9 "
        "--notes-file release-notes.md",
    ]


def test_a_creation_whose_answer_was_lost_is_finished_not_repeated(tmp_path):
    """The first `create` made the release and then failed: the retry finds
    it and uploads the assets over whatever landed, and sets the notes."""
    code, calls = _release(tmp_path, fails=1)
    assert code == 0
    assert [c.split()[1] for c in calls] == ["view", "create", "view", "upload", "edit"]
    assert calls[3] == f"release upload v9.9.9 release/{WHEEL} --clobber"
    assert "--draft=false" in calls[4] and "--notes-file release-notes.md" in calls[4]


def test_a_re_run_finds_the_release_and_finishes_it(tmp_path):
    code, calls = _release(tmp_path, exists=True)
    assert code == 0
    assert [c.split()[1] for c in calls] == ["view", "upload", "edit"]


def test_the_release_notes_are_the_changelogs_section_then_the_wheel_sizes(staged):
    (staged / "artifacts" / "notes").mkdir()
    (staged / "artifacts" / "notes" / "notes.md").write_text(
        "polars-online 9.9.9\n\n### Added\n", encoding="utf-8"
    )
    res = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _run("release", "the notes")],
        cwd=staged,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    notes = (staged / "release-notes.md").read_text(encoding="utf-8")
    assert notes == "polars-online 9.9.9\n\n### Added\n\n### Wheel sizes\n"
