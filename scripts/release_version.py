"""The version a release would publish, checked before anything is built.

Usage: uv run --no-project --python 3.12 python scripts/release_version.py [--publish]

The version lives in six places (docs/RELEASE-READINESS.md, "The steps of a
release"): pyproject.toml, the workspace Cargo.toml's version and its two
path-dependency pins, python/polars_online/__init__.py and the generated
docs/VALIDATION.md. They must agree, and CHANGELOG.md must have a section for
it. With ``--publish`` the tag ``v<version>`` must not exist yet: release tags
are immutable (the ruleset "release tags are immutable"), so a version whose
tag exists can never be released again, and finding that out after the
builds wastes their hour and a half.

Prints ``version=<X.Y.Z>`` and ``tag=v<X.Y.Z>``, the lines ``release.yml``
appends to ``$GITHUB_OUTPUT``; exits 1 naming every disagreement. Standard
library only, so it runs before any environment is built.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def versions(root: Path = REPO) -> dict[str, str | None]:
    """Each place the version is written, and what it says there."""
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    deps = cargo.get("workspace", {}).get("dependencies", {})
    init = (root / "python/polars_online/__init__.py").read_text(encoding="utf-8")
    validation = (root / "docs/VALIDATION.md").read_text(encoding="utf-8")

    def match(pattern: str, text: str) -> str | None:
        m = re.search(pattern, text, re.M)
        return m.group(1) if m else None

    return {
        "pyproject.toml": tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"],
        "Cargo.toml [workspace.package]": cargo.get("workspace", {})
        .get("package", {})
        .get("version"),
        "Cargo.toml online-core pin": deps.get("online-core", {}).get("version"),
        "Cargo.toml online-polars pin": deps.get("online-polars", {}).get("version"),
        "python/polars_online/__init__.py": match(r'^__version__ = "([^"]+)"', init),
        "docs/VALIDATION.md": match(r"polars-online (\d+\.\d+\.\d+\S*)", validation),
    }


def problems(root: Path = REPO, *, publish: bool = False, tag_exists=None) -> list[str]:
    """Every reason the version cannot be released; empty when it can."""
    found = versions(root)
    out = []
    distinct = {v for v in found.values()}
    if len(distinct) != 1 or None in distinct:
        out.append(
            "the version disagrees across its places: "
            + ", ".join(f"{k} = {v}" for k, v in found.items())
        )
        return out
    (version,) = distinct
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(version)}\] ", changelog, re.M):
        out.append(f"CHANGELOG.md has no `## [{version}]` section")
    if publish:
        exists = tag_exists if tag_exists is not None else _remote_tag_exists
        if exists(f"v{version}"):
            out.append(
                f"tag v{version} exists already, and release tags are immutable: raise the version"
            )
    return out


def _remote_tag_exists(tag: str) -> bool:
    out = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(out.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--publish", action="store_true", help="also require a new tag")
    args = parser.parse_args()
    bad = problems(publish=args.publish)
    if bad:
        for b in bad:
            print(f"error: {b}", file=sys.stderr)
        return 1
    version = versions()["pyproject.toml"]
    print(f"version={version}")
    print(f"tag=v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
