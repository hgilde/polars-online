"""The tools of the development loop (docs/TESTING.md, "Iterating"): the
compiler cache, the extension's dev build, the parallel gate and
`scripts/quick.sh`. Each is a dev tool on a contributor's machine, so these
hold that none of them reaches what ships or what CI runs, and that each
does what the guide says.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((REPO / ".github" / "actions").glob("*/action.yml"))
CARGO = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
GATE = (REPO / "scripts" / "gate.sh").read_text(encoding="utf-8")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


# --- sccache ---------------------------------------------------------------


def _env_sh(tmp_path: Path, **env: str) -> str:
    """What `source scripts/env.sh` leaves in RUSTC_WRAPPER and
    SCCACHE_CACHE_SIZE, with a stand-in `sccache` first on the PATH."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "sccache").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake / "sccache").chmod(0o755)
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CI", "RUSTC_WRAPPER", "SCCACHE_CACHE_SIZE", "POLARS_ONLINE_SCCACHE"}
    }
    base["PATH"] = f"{fake}{os.pathsep}{base['PATH']}"
    script = (
        "source scripts/env.sh >/dev/null 2>&1; "
        'printf "%s|%s" "${RUSTC_WRAPPER-unset}" "${SCCACHE_CACHE_SIZE-unset}"'
    )
    res = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO,
        env={**base, **env},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return res.stdout


posix_shell = pytest.mark.skipif(
    sys.platform == "win32",
    reason="scripts/env.sh is the POSIX shells' file; Windows sources env.ps1, with no wrapper",
)


@posix_shell
def test_env_sh_turns_sccache_on_when_asked(tmp_path):
    assert _env_sh(tmp_path, POLARS_ONLINE_SCCACHE="1") == "sccache|10G"
    capped = _env_sh(tmp_path, POLARS_ONLINE_SCCACHE="1", SCCACHE_CACHE_SIZE="3G")
    assert capped == "sccache|3G"


@posix_shell
@pytest.mark.parametrize(
    ("env", "why"),
    [
        ({}, "off unless asked: from a new worktree's path it is slower than no cache"),
        ({"POLARS_ONLINE_SCCACHE": "0"}, "the switch, set off"),
        ({"POLARS_ONLINE_SCCACHE": "1", "CI": "true"}, "every CI service sets CI"),
        ({"POLARS_ONLINE_SCCACHE": "1", "RUSTC_WRAPPER": ""}, "a wrapper already chosen"),
    ],
)
def test_env_sh_leaves_sccache_off(tmp_path, env, why):
    wrapper, _ = _env_sh(tmp_path, **env).split("|")
    assert wrapper == env.get("RUSTC_WRAPPER", "unset"), why


def _envs(doc: dict):
    """Every `env:` mapping in a workflow or action: top, jobs and steps."""
    yield doc.get("env") or {}
    for job in (doc.get("jobs") or {}).values():
        yield job.get("env") or {}
        for step in job.get("steps") or []:
            yield step.get("env") or {}
    for step in (doc.get("runs") or {}).get("steps") or []:
        yield step.get("env") or {}


def _steps(doc: dict):
    for job in (doc.get("jobs") or {}).values():
        yield from job.get("steps") or []
    yield from (doc.get("runs") or {}).get("steps") or []


#: The one place a workflow names sccache, and it is not this repository's:
#: maturin-action's own cache of the release wheel's build, on GitHub's
#: runners, which it exports as RUSTC_WRAPPER for that job only (the CLI's
#: step clears it). It predates the local cache and does not read it.
MATURIN_ACTION_CACHE = ("release.yml", "build the wheel")


def test_no_workflow_turns_the_local_compiler_cache_on():
    """The cache is for a contributor's machine. A workflow never sets
    RUSTC_WRAPPER to anything, sets no SCCACHE_ variable, installs no
    sccache and runs no script that would turn it on: the scripts CI runs
    source env.sh, which stays off under CI (above)."""
    found = []
    for path in [*WORKFLOWS, *ACTIONS]:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for env in _envs(doc):
            for key, value in env.items():
                if key == "RUSTC_WRAPPER" and value != "":
                    found.append(f"{path.name}: RUSTC_WRAPPER={value!r}")
                if key.startswith("SCCACHE"):
                    found.append(f"{path.name}: {key}")
        for step in _steps(doc):
            text = f"{step.get('uses', '')} {step.get('run', '')}".lower()
            if "sccache" in text:
                found.append(f"{path.name}: {step.get('name')!r} runs or uses sccache")
            where = (path.name, step.get("name"))
            if "sccache" in step.get("with", {}) and where != MATURIN_ACTION_CACHE:
                found.append(f"{path.name}: {step.get('name')!r} asks for sccache")
    assert not found, found


# --- the dev-wheel profile ---------------------------------------------------


def test_the_dev_wheel_profile_is_release_without_its_slow_parts():
    dev = CARGO["profile"]["dev-wheel"]
    assert dev == {
        "inherits": "release",
        "lto": False,
        "codegen-units": 256,
        "incremental": True,
        "strip": "none",
    }
    # What the shipped wheel is built with does not move.
    assert CARGO["profile"]["release"] == {"lto": "thin", "codegen-units": 1, "strip": "symbols"}


#: A build with the dev profile, as a command line or an argument list.
BUILDS_DEV = re.compile(r"""--profile["',\s]+dev-wheel""")


def test_what_ships_and_what_gates_a_push_build_release():
    """The dev build is for the loop inside a task alone: the gate's own
    build, every workflow and the release build the extension `--release`,
    and nothing but `scripts/quick.py` builds with the dev profile."""
    assert "maturin develop --release" in GATE
    for path in [*WORKFLOWS, *ACTIONS]:
        assert "dev-wheel" not in path.read_text(encoding="utf-8"), path.name
    named = sorted(
        p.relative_to(REPO).as_posix()
        for p in (REPO / "scripts").glob("*")
        if p.is_file() and BUILDS_DEV.search(p.read_text(encoding="utf-8", errors="replace"))
    )
    assert set(named) <= {"scripts/quick.py"}, named


# --- the parallel gate -------------------------------------------------------


@pytest.mark.parametrize(
    ("cores", "load", "expected"),
    [
        (14, 0.3, 8),  # an idle machine: at most eight
        (14, 11.4, 3),  # the other user's jobs on eleven cores
        (14, 13.9, 2),  # a full machine: still two
        (14, 30.0, 2),
        (8, None, 4),  # no load average (Windows): half the cores
        (2, None, 2),
    ],
)
def test_the_worker_count_follows_the_free_cores(cores, load, expected):
    assert _load("workers").workers(cores, load) == expected


def test_the_gate_runs_in_parallel_and_keeps_a_serial_switch():
    assert "--serial) serial=1" in GATE
    assert "python3 scripts/workers.py" in GATE
    # The Python tests: the parallel run leaves the `serial` ones out, and
    # they run afterwards in one process, so between them every test runs.
    assert '-n "$workers" --dist worksteal -m "$marks and not serial"' in GATE
    assert 'pytest_or_none -q -m "$marks and serial"' in GATE
    # nextest where installed, with cargo's doctests beside it, which
    # nextest does not run; `cargo test` otherwise.
    assert "cargo nextest run --workspace --exclude online-py" in GATE
    assert "cargo test --workspace --exclude online-py --doc" in GATE


def test_nextest_starts_the_slowest_tests_first():
    """.config/nextest.toml parses and puts the slowest Rust tests first:
    the run lasts at least as long as its slowest test."""
    config = tomllib.loads((REPO / ".config" / "nextest.toml").read_text(encoding="utf-8"))
    overrides = config["profile"]["default"]["overrides"]
    assert [o["priority"] for o in overrides] == [100, 50]
    assert "pool_tests" in overrides[0]["filter"]


def test_the_memory_tests_run_alone():
    """The tests that read their process's memory carry `serial`, so the
    gate runs them after the parallel run, in one process."""
    for path, needle in [
        ("tests/test_ffi_memory.py", "pytest.mark.serial,"),
        ("tests/test_gbdt.py", "@pytest.mark.serial\ndef test_many_builds_do_not_leak"),
    ]:
        assert needle in (REPO / path).read_text(encoding="utf-8"), path


def test_a_cached_download_is_written_whole_or_not_at_all(tmp_path):
    """Several processes share `.cache/`: a reader must never see a file
    another is still writing. The write goes to a file of the writer's own,
    and a failure leaves neither that file nor the cached one behind."""
    import data

    target = tmp_path / "day.parquet"
    seen = []

    def write(tmp: Path) -> None:
        tmp.write_bytes(b"half")
        seen.append(target.exists())
        tmp.write_bytes(b"whole")

    data.write_atomically(target, write)
    assert seen == [False] and target.read_bytes() == b"whole"

    def fails(tmp: Path) -> None:
        tmp.write_bytes(b"half")
        raise OSError("disk full")

    other = tmp_path / "other.parquet"
    with pytest.raises(OSError, match="disk full"):
        data.write_atomically(other, fails)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["day.parquet"]


# --- scripts/quick.sh ----------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "crates", "rebuild", "must_run"),
    [
        (
            "crates/online-core/src/kalman.rs",
            {"online-core"},
            True,
            {"tests/test_kalman.py", "tests/test_kalman_revert.py"},
        ),
        (
            "crates/online-core/src/ewcov/par.rs",
            {"online-core"},
            True,
            {"tests/test_ew_cov.py", "tests/test_ew_cov_bias.py"},
        ),
        (
            "crates/online-core/src/margbins.rs",
            {"online-core"},
            True,
            {"tests/test_marginal_bins.py"},
        ),
        ("crates/online-polars/src/bank.rs", {"online-polars"}, True, {"tests/test_bank.py"}),
        ("crates/online-core/tests/golden.rs", {"online-core"}, False, set()),
        ("crates/online-cli/src/main.rs", {"online-cli"}, False, {"tests/test_cli_terminal.py"}),
        (
            "python/polars_online/gram.py",
            set(),
            False,
            {"tests/test_gram.py", "tests/test_gram_page.py"},
        ),
        ("tests/test_holt.py", set(), False, {"tests/test_holt.py"}),
        ("tests/data.py", set(), False, {"tests/test_data_and_reference.py"}),
        ("docs/TESTING.md", set(), False, {"tests/test_tiers.py"}),
    ],
)
def test_quick_maps_a_changed_path_to_its_tests(path, crates, rebuild, must_run):
    plan = _load("quick").plan([path])
    assert plan.crates == crates
    assert plan.rebuild == rebuild
    assert must_run <= plan.tests, sorted(plan.tests)
    assert not plan.unmapped


def test_quick_runs_everything_for_a_helper_every_test_runs_under():
    quick = _load("quick")
    every = set(quick._test_files())
    assert quick.plan(["tests/conftest.py"]).tests == every
    assert len(every) > 100


def test_quick_maps_each_path_on_its_own():
    """A path whose tests another path already brought in is mapped, not
    reported as unmapped (the first version compared the plan's size)."""
    plan = _load("quick").plan(["tests/conftest.py", "tests/test_holt.py", "scripts/env.sh"])
    assert plan.unmapped == []
    assert "tests/test_dev_loop.py" in plan.tests  # it names scripts/env.sh


def test_quick_names_what_it_cannot_map():
    # A name made here, so not even this file names it.
    page = f"docs/{uuid.uuid4().hex}.md"
    plan = _load("quick").plan([page])
    assert plan.unmapped == [page] and not plan.tests


def test_quick_runs_the_oracle_hash_guard_for_the_groups_a_change_can_move():
    """A model's source runs the guard for the groups whose id names it; a
    source module that names none, every group; a test file, its own; a
    page, none."""
    quick = _load("quick")
    groups = set(quick.ledger_groups())
    kmeans = quick.plan(["crates/online-core/src/kmeans.rs"]).guards
    assert kmeans and all("kmeans" in g for g in kmeans), kmeans
    assert quick.plan(["python/polars_online/spec_regimes.py"]).guards >= kmeans
    assert quick.plan(["crates/online-core/src/solve.rs"]).guards == groups
    own = quick.plan(["tests/test_truth_recovery.py"]).guards
    assert own and all(g.startswith("truth_recovery.") for g in own), own
    assert not quick.plan(["docs/TESTING.md"]).guards
    args = quick.plan(["crates/online-core/src/kmeans.rs"]).pytest_args()
    assert f"{quick.GUARD}[truth_recovery.kmeans_blobs]" in args, args
    whole = quick.plan(["crates/online-core/src/kmeans.rs", "tests/test_oracle_ledger.py"])
    assert not [a for a in whole.pytest_args() if "::" in a], "its file runs whole"


def test_quick_is_not_the_gate():
    """It says so when it passes, so a green loop is not read as a check."""
    text = (REPO / "scripts" / "quick.py").read_text(encoding="utf-8")
    assert re.search(
        r"PASS \(the mapped tests only; the essentials gate runs on the merged wave\)", text
    )


# --- scripts/clear_caches.sh ----------------------------------------------------


def _sandbox(tmp_path: Path) -> tuple[Path, dict[str, str], list[Path], list[Path]]:
    """A copy of the script in a repository of its own, a HOME of its own
    and a stand-in `sccache` that records its calls: the caches it may
    clear, and the files beside them it must not touch."""
    repo, home, bin_ = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    script = (REPO / "scripts" / "clear_caches.sh").read_text(encoding="utf-8")
    (repo / "scripts" / "clear_caches.sh").write_text(script, encoding="utf-8")
    bin_.mkdir()
    calls = tmp_path / "sccache-calls"
    (bin_ / "sccache").write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n', encoding="utf-8")
    (bin_ / "sccache").chmod(0o755)
    sccache_dir = home / "sccache-cache"
    caches = [
        repo / "target" / "debug" / "a.rlib",
        repo / "target" / "dev-wheel" / "b.so",
        sccache_dir / "0" / "entry",
    ]
    kept = [
        repo / ".cache" / "BTCUSDT.parquet",
        repo / ".hypothesis" / "examples" / "x",
        repo / "src.rs",
        home / "Library" / "Caches" / "other" / "y",
        tmp_path / "outside",
    ]
    for f in [*caches, *kept]:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 2048)
    env = {
        "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "SCCACHE_DIR": str(sccache_dir),
    }
    return repo, env, caches, kept


def _clear(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "clear_caches.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@posix_shell
def test_clear_caches_dry_run_removes_nothing(tmp_path):
    repo, env, caches, kept = _sandbox(tmp_path)
    res = _clear(repo, env, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert all(f.exists() for f in [*caches, *kept])
    assert "dry run: nothing removed" in res.stdout
    assert "would free" in res.stdout and str(repo / "target") in res.stdout
    assert not (tmp_path / "sccache-calls").exists(), "a dry run stops no server"


@posix_shell
def test_clear_caches_removes_its_list_and_nothing_else(tmp_path):
    repo, env, caches, kept = _sandbox(tmp_path)
    res = _clear(repo, env)
    assert res.returncode == 0, res.stderr
    assert not any(f.exists() for f in caches), res.stdout
    assert all(f.exists() for f in kept), res.stdout
    assert (tmp_path / "sccache-calls").read_text(encoding="utf-8").split() == ["--stop-server"]
    assert "clear_caches: freed" in res.stdout


@posix_shell
def test_clear_caches_takes_the_test_data_only_when_asked(tmp_path):
    repo, env, _, kept = _sandbox(tmp_path)
    res = _clear(repo, env, "--test-data")
    assert res.returncode == 0, res.stderr
    assert not (repo / ".cache").exists() and not (repo / ".hypothesis").exists()
    rest = [f for f in kept if ".cache" not in f.parts and ".hypothesis" not in f.parts]
    assert all(f.exists() for f in rest)


@posix_shell
def test_clear_caches_refuses_a_cache_directory_that_is_not_sccaches(tmp_path):
    repo, env, caches, kept = _sandbox(tmp_path)
    res = _clear(repo, {**env, "SCCACHE_DIR": env["HOME"]})
    assert res.returncode == 2 and "refusing" in res.stderr
    assert all(f.exists() for f in [*caches, *kept])


# --- scripts/slot.sh ------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()


def _slot_repo(tmp_path: Path) -> Path:
    """A repository of its own, two commits on `main`, with the slot and
    cache scripts copied in."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("slot.sh", "clear_caches.sh"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        (repo / "scripts" / name).write_text(text, encoding="utf-8")
    (repo / ".gitignore").write_text("/target/\n/.claude/slots/\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "one")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two")
    return repo


def _slot(repo: Path, *args: str, slots: int = 4) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "slot.sh"), *args],
        cwd=repo,
        env={**os.environ, "POLARS_ONLINE_SLOTS": str(slots)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@posix_shell
def test_a_slot_is_taken_on_its_branch_given_back_and_reused_warm(tmp_path):
    """`take` makes a worktree on a new branch from the base and prints it;
    a second task takes the next slot; past the cap it refuses; `give`
    detaches the slot, and the next `take` reuses it with its target/ --
    the warm build a slot is for -- and the task's branch free to delete."""
    repo = _slot_repo(tmp_path)
    first = _slot(repo, "take", "task-a", "main~1", slots=2)
    assert first.returncode == 0, first.stderr
    one = Path(first.stdout.strip())
    assert one == repo / ".claude" / "slots" / "1"
    assert _git(one, "rev-parse", "--abbrev-ref", "HEAD") == "task-a"
    assert _git(one, "rev-parse", "HEAD") == _git(repo, "rev-parse", "main~1")
    assert _git(repo, "status", "--porcelain") == "", "the slots never show in main"
    (one / "target" / "debug").mkdir(parents=True)
    (one / "target" / "debug" / "warm.rlib").write_bytes(b"x")

    second = _slot(repo, "take", "task-b", slots=2)
    assert Path(second.stdout.strip()) == repo / ".claude" / "slots" / "2"
    full = _slot(repo, "take", "task-c", slots=2)
    assert full.returncode == 1 and "all 2 slots are taken" in full.stderr
    listed = _slot(repo, "list", slots=2).stdout
    assert "1 taken by task-a" in listed and "2 taken by task-b" in listed

    gave = _slot(repo, "give", "1", slots=2)
    assert gave.returncode == 0, gave.stderr
    _git(repo, "branch", "-D", "task-a")  # detached, so the branch can go
    again = _slot(repo, "take", "task-c", slots=2)
    assert Path(again.stdout.strip()) == one
    assert _git(one, "rev-parse", "--abbrev-ref", "HEAD") == "task-c"
    assert _git(one, "rev-parse", "HEAD") == _git(repo, "rev-parse", "main")
    assert (one / "target" / "debug" / "warm.rlib").exists(), "the build stays warm"


@posix_shell
def test_a_slot_with_uncommitted_work_is_neither_given_back_nor_taken_over(tmp_path):
    repo = _slot_repo(tmp_path)
    one = Path(_slot(repo, "take", "task-a").stdout.strip())
    (one / "a.txt").write_text("edited\n", encoding="utf-8")
    res = _slot(repo, "give", "1")
    assert res.returncode == 1 and "uncommitted work" in res.stderr
    assert "1 taken by task-a" in _slot(repo, "list").stdout
    # Freed by hand with the edit still there: a new task does not take it over.
    lock = Path(_git(repo, "rev-parse", "--git-common-dir"))
    lock = (repo / lock if not lock.is_absolute() else lock) / "slots" / "1"
    lock.unlink()
    res = _slot(repo, "take", "task-b", slots=1)
    assert res.returncode == 1 and "uncommitted work" in res.stderr
    assert (one / "a.txt").read_text(encoding="utf-8") == "edited\n"


@posix_shell
def test_clear_caches_keeps_the_slots_warm_unless_asked(tmp_path):
    repo = _slot_repo(tmp_path)
    one = Path(_slot(repo, "take", "task-a").stdout.strip())
    other = tmp_path / "other"
    _git(repo, "worktree", "add", "-q", "--detach", str(other), "main")
    slot_build, other_build = one / "target" / "x.rlib", other / "target" / "y.rlib"
    for f in (slot_build, other_build):
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * 2048)
    env = {**os.environ, "SCCACHE_DIR": str(tmp_path / "sccache")}
    run = lambda *a: subprocess.run(  # noqa: E731
        ["bash", str(repo / "scripts" / "clear_caches.sh"), *a],
        cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", check=False,
    )  # fmt: skip
    res = run("--worktrees")
    assert res.returncode == 0, res.stderr
    assert not other_build.exists() and slot_build.exists(), res.stdout
    res = run("--slots")
    assert res.returncode == 0, res.stderr
    assert not slot_build.exists(), res.stdout


@posix_shell
def test_a_slot_is_never_taken_on_a_branch_another_worktree_holds(tmp_path):
    """`git checkout -B` does not refuse a branch checked out elsewhere, as a
    plain checkout does, so two worktrees could share one branch and a
    commit in either leave the other's tree behind its HEAD (review 8, E-1).
    `take` refuses the branch, names the worktree that holds it and leaves
    no lock behind."""
    repo = _slot_repo(tmp_path)
    held = tmp_path / "wt-x"
    _git(repo, "worktree", "add", "-q", "-b", "task-x", str(held), "main")
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    locks = (common if common.is_absolute() else repo / common) / "slots"
    for branch, holder in (("task-x", held), ("main", repo)):
        res = _slot(repo, "take", branch)
        assert res.returncode == 1, (branch, res.stdout, res.stderr)
        assert f"{branch} is checked out at {holder.resolve()}" in res.stderr, res.stderr
        assert not any(locks.glob("*")), f"{branch}: a lock was left behind"
        assert not (repo / ".claude" / "slots" / "1" / ".git").exists(), branch
    # A slot's own branch: given back without detaching (the lock removed by
    # hand), the slot may be taken again on the branch it holds.
    one = Path(_slot(repo, "take", "task-a").stdout.strip())
    (locks / "1").unlink()
    again = _slot(repo, "take", "task-a")
    assert again.returncode == 0, again.stderr
    assert Path(again.stdout.strip()) == one
    # And another slot's branch is refused, naming that slot.
    res = _slot(repo, "take", "task-a")
    assert res.returncode == 1 and f"is checked out at {one.resolve()}" in res.stderr


@posix_shell
def test_clear_caches_run_from_a_slot_keeps_the_other_slots_warm(tmp_path):
    """Run from inside a slot, `--worktrees` once took every other slot for
    an ordinary worktree, since it matched the slots against its own
    checkout (review 8, E-3: 186 GB of warm builds on this machine). The
    slots belong to the main checkout wherever the script runs from."""
    repo = _slot_repo(tmp_path)
    one = Path(_slot(repo, "take", "task-a").stdout.strip())
    two = Path(_slot(repo, "take", "task-b").stdout.strip())
    other = tmp_path / "other"
    _git(repo, "worktree", "add", "-q", "--detach", str(other), "main")
    for wt in (repo, one, two, other):
        (wt / "target").mkdir()
        (wt / "target" / "x.rlib").write_bytes(b"x" * 2048)
    env = {**os.environ, "SCCACHE_DIR": str(tmp_path / "sccache")}

    def listed(*args: str) -> set[Path]:
        res = subprocess.run(
            ["bash", str(one / "scripts" / "clear_caches.sh"), "--dry-run", *args],
            cwd=one, env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )  # fmt: skip
        assert res.returncode == 0, res.stderr
        return {
            Path(line.split()[-1]).parent
            for line in res.stdout.splitlines()
            if line.startswith("would free") and line.split()[-1].endswith("/target")
        }

    # Nor the main checkout's, where the coordinator's gate builds (task 269,
    # the second look): only a run from the main checkout clears it.
    assert listed("--worktrees") == {one, other}
    assert listed("--slots") == {one, two}
    assert listed("--worktrees", "--slots") == {one, two, other}
    assert all((wt / "target" / "x.rlib").exists() for wt in (repo, one, two, other))


@posix_shell
@pytest.mark.parametrize(("tier", "rc"), [("essentials", 0), ("extended", 5)])
def test_an_empty_serial_selection_passes_only_in_the_essentials(tmp_path, tier, rc):
    """pytest exits 5 when no test is selected. The essentials hold no
    `serial` test, so there that is no failure; the extended tier holds the
    memory tests, where an empty selection is a mark gone wrong (review 8,
    E-8). The gate's own function, run with a stand-in `uv` that selects
    nothing."""
    m = re.search(r"^pytest_or_none\(\) \{\n.*?^\}\n", GATE, re.M | re.S)
    assert m, "the gate defines pytest_or_none"
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uv").write_text("#!/bin/sh\nexit 5\n", encoding="utf-8")
    (fake / "uv").chmod(0o755)
    res = subprocess.run(
        ["bash", "-c", f'tier={tier}\n{m.group(0)}pytest_or_none -q -m serial; echo "rc=$?"'],
        env={**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True, text=True, encoding="utf-8", check=False,
    )  # fmt: skip
    assert res.stdout.strip().splitlines()[-1] == f"rc={rc}", (res.stdout, res.stderr)
