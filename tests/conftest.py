import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make tests/data.py and tests/reference.py importable as plain modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def online_cli() -> Path:
    """The CLI executable, built once per session (docs/IMPROVEMENTS.md T1).

    The CLI tests used to `cargo run` on every call, and each call cost 2.9 s
    with nothing to build. Not the freshness check -- that is 0.15 s -- but
    the launch: on macOS cargo re-clones the binary into `target/debug` on
    every fresh build, and the first exec of a new file of a 418 MB debug
    executable spends ~2.7 s before `main` validating its ad-hoc code
    signature; the second exec of the same file takes 10 ms. So: build once,
    run the executable directly, and pay the launch once per session.

    The path comes from cargo's own artifact message rather than being
    guessed, so `CARGO_TARGET_DIR` and the `.exe` suffix are cargo's problem.
    """
    cmd = ["cargo", "build", "-q", "-p", "online-cli", "--message-format=json-render-diagnostics"]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO),
            check=False,
        )
    except FileNotFoundError:
        pytest.fail("cargo is not on PATH; `source scripts/env.sh` first")
    assert res.returncode == 0, res.stderr
    exes = [
        msg["executable"]
        for line in res.stdout.splitlines()
        if line.startswith("{")
        and (msg := json.loads(line)).get("reason") == "compiler-artifact"
        and msg.get("executable")
        and "bin" in msg["target"]["kind"]
        and msg["target"]["name"] == "online"
    ]
    assert len(exes) == 1, f"expected one CLI artifact, got {exes}:\n{res.stdout}"
    return Path(exes[0])
