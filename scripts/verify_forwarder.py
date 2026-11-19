#!/usr/bin/env python3
"""Check a built polars-online (or polars-online-compat) 2.0.0 forwarder.

    python scripts/verify_forwarder.py --wheel forwarders/polars-online/dist/*.whl
    python scripts/verify_forwarder.py            # the current environment

With ``--wheel`` the wheel is installed, with its dependencies from PyPI, into
a fresh virtual environment and checked there; without it the interpreter that
runs this script is checked, and must already have the forwarder installed.
Standard library only (the checks themselves import polars and rheon, in the
environment under test). One line per check; the exit status is non-zero if
any fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

# Runs in the environment under test and prints {check: [ok, reason]} as JSON.
# `polars_online` is imported first, before rheon, so that nothing it warns is missed.
PROBE = r'''
import json, warnings
from importlib import metadata

out = {}

def check(name, fn):
    try:
        reason = fn()
        out[name] = [reason is None or reason is True, "" if reason in (None, True) else str(reason)]
    except Exception as e:  # a failed check, with its reason
        out[name] = [False, f"{type(e).__name__}: {e}"]

def installed():
    for n in ("polars-online", "polars-online-compat"):
        try:
            metadata.version(n)
            return None
        except metadata.PackageNotFoundError:
            pass
    return "neither polars-online nor polars-online-compat is installed"

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import polars_online
ours = [w for w in caught if issubclass(w.category, DeprecationWarning) and "polars_online" in str(w.message)]
check("installed", installed)
check("one DeprecationWarning", lambda: None if len(ours) == 1 else f"{len(ours)} DeprecationWarnings on import")

import polars as pl
import rheon.polars as rp
check("ModelBank identity", lambda: None if polars_online.ModelBank is rp.ModelBank else "polars_online.ModelBank is not rheon.polars.ModelBank")

def same_fit():
    from polars.testing import assert_frame_equal
    spec = polars_online.spec.ewridge("m", targets=["y"], features=["x"], half_life=float("inf"))
    df = pl.DataFrame({"x": [float(i % 7) for i in range(60)], "y": [0.5 * (i % 7) + (i % 3) for i in range(60)]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = df.online.fit_predict([spec])
    new = df.rheon.fit_predict([spec])
    assert_frame_equal(old, new)
    if old.columns == df.columns:
        return "fit_predict added no column"

check(".online fit equals .rheon fit", same_fit)
print(json.dumps(out))
'''


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else "no output"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel", type=Path, help="a built forwarder wheel to install into a fresh environment")
    args = ap.parse_args()

    results: dict[str, list] = {}
    python = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        if args.wheel is not None:
            if not args.wheel.is_file():
                print(f"FAIL install: no such file {args.wheel}")
                return 1
            venv.create(Path(tmp) / "env", with_pip=True)
            python = str(Path(tmp) / "env" / ("Scripts" if sys.platform == "win32" else "bin") / "python")
            done = run([python, "-m", "pip", "install", "--quiet", str(args.wheel)])
            results["install"] = [done.returncode == 0, last_line(done.stderr + done.stdout) if done.returncode else ""]
            if done.returncode:  # nothing to import
                return report(results)
        probe = run([python, "-c", PROBE])
        try:
            results.update(json.loads(last_line(probe.stdout)))
        except json.JSONDecodeError:
            results["import"] = [False, last_line(probe.stderr)]
    return report(results)


def report(results: dict[str, list]) -> int:
    for name, (ok, reason) in results.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {reason}" if reason else ""))
    return 0 if results and all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
