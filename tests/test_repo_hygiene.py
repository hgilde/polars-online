"""CLAUDE.md hard rule 1: no data files in the repo, ever.

Tests generate or download what they need and cache it under the gitignored
`.cache/`. Nothing checked in.

This is asserted rather than remembered because it is easy to break by
accident and invisible once broken: 136 files of `cargo mutants` output were
tracked for several commits, swept in by a `git add -A`, and nothing complained.
"""

import subprocess
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: Extensions that are data, not source, wherever they appear.
DATA_SUFFIXES = {
    ".parquet",
    ".arrow",
    ".feather",
    ".ipc",
    ".csv",
    ".tsv",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".zip",
    ".gz",
    ".tar",
    ".xz",
    ".parq",
    ".orc",
    ".avro",
    # The suite's own state files, which tests write by the dozen: a frozen
    # one checked in as a fixture would pass every check above (task 160, TC7).
    ".state",
    ".msgpack",
    ".mpk",
    ".bin",
}

#: Generated tool output that belongs in .gitignore, matched on any path part.
TOOL_OUTPUT_DIRS = {"mutants.out", "mutants.out.old", "target", ".venv", "htmlcov", "dist"}

#: The largest a source file has any business being, in bytes. The frozen v1
#: state fixture was the biggest legitimate one, a hex constant, until
#: `crates/online-polars/src/bank.rs` passed 200 KB with the readiness fields
#: (task 87, 2026-09-21); a data file would still be caught, only later.
MAX_SOURCE_BYTES = 250_000

#: Prose gets more room: `docs/PLAN.md` is the design log and passed 200 KB
#: when tasks 45-56 were prepared (2026-09-05). A data file renamed `.md`
#: would still be caught by the cap below, only later.
MAX_DOC_BYTES = 1_000_000

#: Lockfiles are generated but must be tracked, and grow with every wheel a
#: dependency publishes (mypy alone added 65 KB); their size says nothing
#: about data.
LOCKFILES = {"uv.lock", "Cargo.lock"}


def _tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [Path(p) for p in out.stdout.split("\0") if p]


@pytest.fixture(scope="module")
def tracked():
    files = _tracked()
    assert files, "git ls-files returned nothing -- is this a checkout?"
    return files


def test_no_data_files_are_tracked(tracked):
    offenders = [p for p in tracked if p.suffix.lower() in DATA_SUFFIXES]
    assert not offenders, (
        f"hard rule 1: data files are tracked: {offenders}. "
        "Generate or download them in the test instead, cached under .cache/."
    )


def _binary(path: Path) -> bool:
    """Git's own test: a NUL byte in the first 8,000 bytes."""
    with path.open("rb") as fh:
        return b"\0" in fh.read(8000)


def test_no_binary_file_is_tracked(tracked):
    """A data file by any other name: an image, a state blob, a model dump.
    None is tracked today, and the source tree has no need of one."""
    offenders = [str(p) for p in tracked if (REPO / p).is_file() and _binary(REPO / p)]
    assert not offenders, f"binary files are tracked: {offenders}"


def test_the_checks_know_the_suites_own_formats(tmp_path):
    """The state formats the suite writes itself are data, and a binary file
    is caught whatever its name (task 160, TC7: a 240 KB `bank.state`,
    `.msgpack`, `.bin` or `.png` passed every check)."""
    for name in ("bank.state", "frame.msgpack", "blob.bin"):
        assert Path(name).suffix in DATA_SUFFIXES, name
    blob = tmp_path / "picture.png"
    blob.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    assert _binary(blob)
    text = tmp_path / "notes.txt"
    text.write_text("plain text\n", encoding="utf-8")
    assert not _binary(text)


def test_no_tool_output_is_tracked(tracked):
    offenders = [p for p in tracked if TOOL_OUTPUT_DIRS & set(p.parts)]
    assert not offenders, f"generated tool output is tracked: {offenders[:10]}"


def test_no_tracked_file_is_data_sized(tracked):
    big = []
    for p in tracked:
        f = REPO / p
        if p.name in LOCKFILES:
            continue
        cap = MAX_DOC_BYTES if p.suffix == ".md" else MAX_SOURCE_BYTES
        if f.is_file() and f.stat().st_size > cap:
            big.append((str(p), f.stat().st_size))
    assert not big, f"suspiciously large tracked files (data in disguise?): {big}"


def test_the_cache_directory_is_ignored():
    """Downloads land in `.cache/`; if that stops being ignored, the next
    `git add -A` commits a dataset."""
    res = subprocess.run(
        ["git", "check-ignore", "-q", ".cache/anything.parquet"],
        cwd=REPO,
        check=False,
    )
    assert res.returncode == 0, ".cache/ is not gitignored"


@pytest.mark.parametrize("path", ["mutants.out/x.txt", "target/debug/x", ".venv/bin/python"])
def test_generated_directories_are_ignored(path):
    res = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO, check=False)
    assert res.returncode == 0, f"{path} is not gitignored"


def test_a_clean_checkout_has_what_the_build_needs(tracked):
    """`git archive` is what a fresh clone or an sdist sees. Anything the build
    reads must be in it -- a file that only exists in this working tree would
    make CI and every other machine fail in a way that is invisible here."""
    # `as_posix()`, not `str()`: git always reports forward slashes, while
    # `str(WindowsPath(...))` gives backslashes, so this compared
    # "python\\polars_online\\__init__.py" against a forward-slash literal and
    # declared a tracked file missing. Caught by the first Windows CI run.
    names = {p.as_posix() for p in tracked}
    for needed in [
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "python/polars_online/__init__.py",
        "crates/online-py/Cargo.toml",
        "crates/online-py/src/lib.rs",
        "scripts/env.sh",
        "scripts/env.ps1",
        "scripts/gate.sh",
        "examples/bank.toml",
        "tests/data.py",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        ".gitattributes",
    ]:
        assert needed in names, f"{needed} is not tracked; a fresh clone would not have it"
