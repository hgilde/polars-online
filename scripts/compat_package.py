"""Turn this checkout into the source of ``polars-online-compat``, in place.

Usage: python3 scripts/compat_package.py [--root DIR]

``polars-online`` and ``polars-online-compat`` are two PyPI projects built
from one source tree and one version. They are imported as ``rheon``
alike; they differ in the CPU they need. The x86-64 wheels of
``polars-online`` are built for CPUs with AVX2 and FMA, and
``polars-online-compat`` is built for every x86-64 CPU, with the same
instructions the package had before the split. It ships for the ARM
platforms too, where it is the same build, so that
``pip install polars-online-compat`` works wherever ``pip install
polars-online`` does, as Polars' own ``polars-runtime-compat`` does.

The release workflow runs this on a runner's checkout before it builds the
compat wheels and sdist: it renames the project in ``pyproject.toml`` (the
one place maturin reads the distribution's name from), says in the
description what the package is, and puts a notice at the head of the README
that PyPI shows as the project page. Nothing is committed. The module name,
the crate and everything else stay: only the distribution is renamed.
The script refuses a tree it already changed, so a job cannot run it twice.
Standard library only, so it runs before any environment is built.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NAME = "polars-online"
COMPAT_NAME = "polars-online-compat"

#: What PyPI shows beside the project's name.
COMPAT_DESCRIPTION = (
    "Streaming / online regression models for Polars: the build for x86-64 CPUs "
    "without AVX2, with the same import name as polars-online"
)

#: The notice at the head of the compat package's PyPI page. After the README's title.
NOTICE = f"""> **This is `{COMPAT_NAME}`, the compatibility build of
> [`{NAME}`](https://pypi.org/project/{NAME}/).** It is the same source and the
> same version, built for every x86-64 CPU, including those from before about
> 2013 without AVX2 and FMA, and some virtual machines. It is imported as
> `rheon`, like `{NAME}`, so install one or the other: run
> `pip uninstall {NAME}` before `pip install {COMPAT_NAME}`. Where your CPU has
> AVX2 and FMA, `{NAME}` is the better choice: several models run faster on it.

"""

_NAME_LINE = re.compile(r'^name = "rheon"$', re.M)
_DESCRIPTION_LINE = re.compile(r'^description = "[^"\n]*"$', re.M)


def rename_pyproject(text: str) -> str:
    """``pyproject.toml`` with its project named ``polars-online-compat``."""
    if len(_NAME_LINE.findall(text)) != 1:
        raise ValueError(
            'pyproject.toml has no single `name = "rheon"` line: '
            "it is already renamed, or its project table moved"
        )
    if len(_DESCRIPTION_LINE.findall(text)) != 1:
        raise ValueError("pyproject.toml has no single one-line `description = ...`")
    text = _NAME_LINE.sub(f'name = "{COMPAT_NAME}"', text)
    return _DESCRIPTION_LINE.sub(f'description = "{COMPAT_DESCRIPTION}"', text)


def prepend_notice(readme: str) -> str:
    """The README with the notice after its title, the first ``# `` line."""
    lines = readme.split("\n")
    at = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if at is None:
        raise ValueError("README.md has no `# ` title")
    after = "\n".join(lines[at + 1 :]).lstrip("\n")
    return "\n".join(lines[: at + 1]) + f"\n\n{NOTICE}{after}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help="the checkout (default: this one)")
    args = ap.parse_args(argv)
    pyproject = args.root / "pyproject.toml"
    readme = args.root / "README.md"
    try:
        new_pyproject = rename_pyproject(pyproject.read_text(encoding="utf-8"))
        new_readme = prepend_notice(readme.read_text(encoding="utf-8"))
    except ValueError as e:
        print(f"::error::{e}")
        return 1
    pyproject.write_text(new_pyproject, encoding="utf-8")
    readme.write_text(new_readme, encoding="utf-8")
    print(f"{args.root}: the project is now {COMPAT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
