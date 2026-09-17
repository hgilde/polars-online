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


def _toml_value(v: object) -> str:
    """One TOML scalar, list or inline table.

    Infinities go as the strings the spec layer already uses for them
    (`online_core::humanfloat`), so a config round-trips a `halflife = inf`
    the way `po.spec` writes it.
    """
    if isinstance(v, bool):  # before int: bool is an int in Python
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == float("inf"):
            return '"inf"'
        if v == float("-inf"):
            return '"-inf"'
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"no TOML form for {type(v).__name__}")


def spec_toml(spec: dict) -> str:
    """One `[[specs]]` block, with the model under `[specs.model]`.

    `None` is dropped rather than written: a spec dict carries the keys its
    builder did not fill, and TOML has no null.
    """
    lines = ["[[specs]]"]
    model = None
    for k, v in spec.items():
        if k == "model":
            model = v
            continue
        if v is None:
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    if model:
        lines.append("[specs.model]")
        for k, v in model.items():
            if v is None:
                continue
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines)


def run_online(exe, tmp_path, specs, *, args=(), check=True, **top):
    """Run the `online` binary over `specs`, from a TOML config written here.

    `top` are the config's top-level keys (`input`, `output`, `save_state`,
    `load_state`, `closed_groups`, `chunk_rows`, `predict`); paths are taken
    as they come and written POSIX-style, which is what TOML wants on every
    platform. `args` are extra command-line flags. Returns the
    `CompletedProcess`, so a caller can assert on `returncode` and `stderr`;
    `check=False` is for the refusals.
    """
    lines = []
    for k, v in top.items():
        if v is None:
            continue
        lines.append(f"{k} = {_toml_value(Path(v).as_posix() if isinstance(v, Path) else v)}")
    lines.extend(spec_toml(s) for s in specs)
    cfg = Path(tmp_path) / "bank.toml"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = subprocess.run(
        [str(exe), "--config", str(cfg), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check:
        assert res.returncode == 0, res.stderr
    return res
