"""T-W5: the release workflow's artifact-collection step, run for real.

`release.yml` renames each CLI binary to `online-<target>` and has to keep the
`.exe` suffix on the Windows one. That step is bash, it runs only on the ubuntu
job, and it has never processed a real `online.exe` -- the Windows build has
never run, because nothing has ever been pushed.

The shell is extracted from the workflow rather than copied here, so editing
the workflow changes what this test runs. That is the whole point: a copy would
drift and keep passing.
"""

import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "release.yml"

TARGETS = {
    "x86_64-pc-windows-msvc": "online.exe",
    "aarch64-apple-darwin": "online",
    "x86_64-unknown-linux-gnu": "online",
}
WHEEL = "polars_online-0.1.0-cp312-abi3-macosx_11_0_arm64.whl"


def _step_script(name: str) -> str:
    """The `run:` block of the named step, dedented to a runnable script."""
    text = WORKFLOW.read_text()
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
    return tmp_path


def _collect(cwd) -> list[str]:
    res = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step_script("collect the files")],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return sorted(p.name for p in (Path(cwd) / "release").iterdir())


def test_every_artifact_is_collected_and_named_by_target(staged):
    assert _collect(staged) == [
        "online-aarch64-apple-darwin",
        "online-x86_64-pc-windows-msvc.exe",
        "online-x86_64-unknown-linux-gnu",
        WHEEL,
    ]


def test_the_windows_binary_keeps_its_suffix_and_the_others_gain_none(staged):
    names = _collect(staged)
    exes = [n for n in names if n.endswith(".exe")]
    assert exes == ["online-x86_64-pc-windows-msvc.exe"], (
        "exactly the Windows artifact keeps .exe; a rename that drops it "
        "produces a file Windows will not execute, and one that adds it "
        "everywhere breaks the unix downloads"
    )


def test_the_binaries_are_copied_not_just_renamed(staged):
    names = _collect(staged)
    for name in ["online-aarch64-apple-darwin", "online-x86_64-pc-windows-msvc.exe"]:
        assert name in names, f"{name} was not produced; got {names}"
        assert (staged / "release" / name).read_bytes() == b"\x7fELF fake binary"


def test_it_does_not_collapse_two_targets_onto_one_name(staged):
    """Both unix binaries are called `online` in their own directories, so a
    rename that forgot the target would silently ship one of them twice."""
    names = _collect(staged)
    assert len(names) == len(set(names))
    assert len([n for n in names if n.startswith("online-")]) == len(TARGETS)


def test_the_publish_job_collects_only_wheels(staged):
    res = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step_script("collect the wheels")],
        cwd=str(staged),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert sorted(p.name for p in (staged / "dist").iterdir()) == [WHEEL], (
        "the PyPI job must upload wheels only -- a CLI binary in dist/ would "
        "be rejected by the index"
    )
