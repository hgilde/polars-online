"""The line every benchmark and experiment prints first: what it ran on.

A timing means nothing without the build and the machine behind it. The
README's throughput sat 40-50% stale for three weeks, and a parallel run once
read 576 ms where the README said 155, because the machine's load had gone
from 3 to 524 mid-run and nothing in the output said so (task 160, SC3). So
each script prints the commit (and whether the tree was modified), the
package and polars versions, the Python, the platform, the core count and the
load, before its first number:

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)

Standard library only, besides the two packages it names.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return res.stdout.strip() if res.returncode == 0 else ""


def load() -> str:
    """The one-minute load average, or "n/a" where the platform has none."""
    getloadavg = getattr(os, "getloadavg", None)
    return f"{getloadavg()[0]:.1f}" if getloadavg is not None else "n/a"


def header() -> str:
    import polars as pl

    import polars_online as po

    commit = _git("rev-parse", "--short=12", "HEAD") or "unknown"
    if _git("status", "--porcelain", "--untracked-files=no"):
        commit += " (modified)"
    return (
        f"ran on: commit {commit}; polars-online {po.__version__}; polars {pl.__version__}; "
        f"Python {platform.python_version()}; {platform.system()} {platform.machine()}; "
        f"{os.cpu_count()} cpus; load {load()}"
    )
