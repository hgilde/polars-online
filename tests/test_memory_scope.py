"""The memory scope the mutation runs live in (``scripts/memory_scope.sh``,
``scripts/memory_scope_check.py``; mutants.yml).

The scope itself needs systemd, cgroup v2 and root, which only the CI runner
has, and mutants.yml's ``memory-cap`` job proves it there on every run: a
runaway is killed near the cap and its parent goes on. What can be held
anywhere is held here: the script hands the command's own exit status back
whatever ``systemd-run`` returns, refuses to read a scope that never ran the
command as success, and asks for the properties it needs; and the check
passes only a runaway killed near the cap with its parent alive.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TIER = "essential"
REPO = Path(__file__).resolve().parent.parent
SCOPE = REPO / "scripts" / "memory_scope.sh"

#: Stands in for `sudo systemd-run --scope -p ... -- setpriv ... CMD`:
#: logs systemd-run's arguments, drops setpriv's options, runs CMD.
STUB_SUDO = """#!/usr/bin/env bash
[ "$1" = "-E" ] && shift
[ "$1" = "systemd-run" ] || exit 90
echo "$@" >> "$STUB_LOG"
while [ "$1" != "--" ]; do shift; done
shift
[ "$1" = "setpriv" ] || exit 91
shift
while [ "${1#--}" != "$1" ]; do shift; done
exec "$@"
"""

#: The same, but as the runners' sudo now is: ubuntu-26.04's sudo-rs
#: ignores `-E` ("preserving the entire environment is not supported") and
#: hands the command a reset environment, with only what the `env` after
#: setpriv names. On 2026-10-24 that lost the workflow's CARGO_INCREMENTAL=0
#: inside every mutants scope (run 40076792577, task 270 (m)).
RESETTING_SUDO = STUB_SUDO.replace('exec "$@"', 'exec env -i "$@"')

#: A `sudo` whose scope never runs the command, and says nothing.
SILENT_SUDO = "#!/usr/bin/env bash\nexit 0\n"

needs_posix = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the scope script is a Linux runner's; its stubs here need a POSIX shell",
)


def _run(
    tmp_path: Path, sudo: str, *command: str, env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "sudo"
    stub.write_text(sudo, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "systemd-run.log"
    extra = env or {}
    env = {**os.environ, **extra, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env["STUB_LOG"] = str(log)
    r = subprocess.run(
        ["bash", str(SCOPE), "12G", "--", *command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    return r, log.read_text(encoding="utf-8") if log.exists() else ""


@needs_posix
@pytest.mark.parametrize("status", [0, 2, 3, 124, 137])
def test_the_commands_own_status_comes_back(tmp_path, status):
    """cargo-mutants' 0, 2 (survivors), 3 (timeouts), and the stop's 124 or
    137: the workflow decides on each, so each must come back as it was."""
    r, log = _run(tmp_path, STUB_SUDO, "bash", "-c", f"exit {status}")
    assert r.returncode == status, r.stderr
    assert "--scope" in log and "-p MemoryMax=12G" in log, log
    assert "-p MemorySwapMax=0" in log and "-p OOMPolicy=continue" in log, log


@needs_posix
def test_the_command_runs_with_the_callers_path_and_home(tmp_path):
    r, _ = _run(tmp_path, STUB_SUDO, "bash", "-c", 'echo "$PATH|$HOME"')
    path, home = r.stdout.strip().split("|")
    assert path.split(os.pathsep)[0] == str(tmp_path / "bin")
    assert home == os.environ["HOME"]


@needs_posix
@pytest.mark.parametrize("sudo", [STUB_SUDO, RESETTING_SUDO], ids=["sudo", "sudo-rs"])
def test_the_command_runs_with_the_callers_whole_environment(tmp_path, sudo):
    """Everything the job sets reaches the command in the scope, whatever
    the runner's sudo keeps: the workflow's CARGO_INCREMENTAL=0 and the
    TMPDIR that puts cargo-mutants' build copies on the runner's disk, a
    value with a space, a newline and an `=` in it among them. The scope's
    own bookkeeping does not leak into the command."""
    odd = "a b\nc=d"
    r, _ = _run(
        tmp_path,
        sudo,
        "bash",
        "-c",
        'printf "%s|%s|%s|%s|%s" "$CARGO_INCREMENTAL" "$TMPDIR" "$SCOPE_ODD" '
        '"${MEMORY_SCOPE_ENV-unset}" "$name"',
        env={"CARGO_INCREMENTAL": "0", "TMPDIR": str(tmp_path), "SCOPE_ODD": odd, "name": "n"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("|") == ["0", str(tmp_path), odd, "unset", "n"], r.stdout


@needs_posix
def test_a_scope_that_never_ran_the_command_fails(tmp_path):
    r, _ = _run(tmp_path, SILENT_SUDO, "true")
    assert r.returncode == 1
    assert "never finished inside the scope" in r.stderr


@needs_posix
def test_it_refuses_a_call_without_its_separator():
    r = subprocess.run(
        ["bash", str(SCOPE), "12G", "true"], capture_output=True, text=True, encoding="utf-8"
    )
    assert r.returncode == 2 and "usage" in r.stderr


def _check():
    spec = importlib.util.spec_from_file_location("memory_scope_check", SCOPE.with_name(
        "memory_scope_check.py"
    ))  # fmt: skip
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: What the check's sentinel reads inside a scope the environment reached.
OK = "reached the scope = yes"


@pytest.mark.parametrize(
    ("result", "passes"),
    [
        ({"runaway_exit": -9, "reached_mib": 11_900, "survivor": 64 << 20, "env": OK}, True),
        # Not killed: it stopped on its own, or something else stopped it.
        ({"runaway_exit": 0, "reached_mib": 11_900, "survivor": 64 << 20, "env": OK}, False),
        # Killed far below the cap: some other, smaller limit is in force.
        ({"runaway_exit": -9, "reached_mib": 512, "survivor": 64 << 20, "env": OK}, False),
        # Killed past the cap: the cap is not the scope's.
        ({"runaway_exit": -9, "reached_mib": 20_000, "survivor": 64 << 20, "env": OK}, False),
        # Killed near the cap, but the command ran without the caller's
        # environment: sudo-rs reset it.
        ({"runaway_exit": -9, "reached_mib": 11_900, "survivor": 64 << 20, "env": None}, False),
    ],
)
def test_the_check_passes_only_a_runaway_killed_near_the_cap(monkeypatch, result, passes):
    check = _check()
    done = SimpleNamespace(returncode=0, stdout=json.dumps(result) + "\n", stderr="")
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: done)
    assert (check.main("12G") == 0) is passes


def test_the_check_fails_when_the_scope_command_does(monkeypatch):
    check = _check()
    done = SimpleNamespace(returncode=1, stdout="", stderr="no scope")
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: done)
    assert check.main("12G") == 1


def test_the_check_reads_the_cap_as_systemd_does():
    check = _check()
    assert check.mib("12G") == 12 * 1024
    assert check.mib("512M") == 512
    assert check.mib(str(3 << 30)) == 3 * 1024


def test_the_checks_sentinel_is_the_one_it_reads():
    check = _check()
    assert check.SENTINEL_VALUE == OK
