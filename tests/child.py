"""The environment for a child interpreter a test spawns: it runs as a user
without pyarrow does, as the pytest process itself does (tests/conftest.py),
by putting tests/_site on its path, whose sitecustomize.py installs the same
finder; and it writes UTF-8 on both ends of the pipe, which a piped Python
on Windows otherwise does not. tests/test_pyarrow_interop.py's children are
the one exception, and build their own environment."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path(__file__).resolve().parent / "_site"


def env(**extra: str) -> dict[str, str]:
    path = os.pathsep.join(p for p in (str(SITE), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, **extra, "PYTHONPATH": path, "PYTHONIOENCODING": "utf-8"}
