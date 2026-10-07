"""The version a release would publish, checked before anything is built.

Usage: uv run --no-project --python 3.12 python scripts/release_version.py [--publish]

The version lives in six places (docs/RELEASE-READINESS.md, "The steps of a
release"): pyproject.toml, the workspace Cargo.toml's version and its two
path-dependency pins, python/polars_online/__init__.py and the generated
docs/VALIDATION.md. They must agree, and CHANGELOG.md must have a section for
it. The README's example pin, ``polars-online~=X.Y.0`` under *This package's
own versioning*, must name the version's minor, so a minor release cannot
leave it telling readers to stay on the last one.

With ``--publish``, nothing may be left under the CHANGELOG's
``[Unreleased]``: the tag's notes are the version's section, so an entry left
behind would ship without its line. And the tag ``v<version>`` must not exist
yet: release tags are immutable (the ruleset "release tags are immutable"),
so a version whose tag exists can never be released again, and finding that
out after the builds wastes their hour and a half.

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


#: The README's example pin: ``polars-online~=0.13.0`` names the 0.13 series.
README_PIN = re.compile(r"polars-online~=(\d+\.\d+\.\d+)")


def unreleased(changelog: str) -> str:
    """What the CHANGELOG's ``[Unreleased]`` section holds; empty when the
    section is empty or absent, as release day leaves it."""
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog, re.M | re.S)
    return m.group(1).strip() if m else ""


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
    major, minor = version.split(".")[:2]
    want = f"{major}.{minor}.0"
    pins = README_PIN.findall((root / "README.md").read_text(encoding="utf-8"))
    wrong = sorted({p for p in pins if p != want})
    if not pins or wrong:
        out.append(
            f"README.md's example pin must be `polars-online~={want}`, the minor of "
            f"{version}; it names {', '.join(f'`~={p}`' for p in wrong) or 'none'} "
            "(update the example under *This package's own versioning*)"
        )
    if publish and unreleased(changelog):
        out.append(
            "CHANGELOG.md has entries under `## [Unreleased]`: move them to "
            f"`## [{version}]` before publishing, since the tag's notes are that section"
        )
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
