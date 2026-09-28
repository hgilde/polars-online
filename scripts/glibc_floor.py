"""The newest glibc a Linux binary requires, refused above a floor.

Usage: python3 scripts/glibc_floor.py BINARY [--max 2.17]

Reads the binary's version requirements with ``objdump -p`` (its "Version
References") and prints the newest ``GLIBC_x.y`` among them. Exits 1 when it
is newer than ``--max``: the loader refuses a binary whose requirement the
system's glibc lacks, even one that only a weak symbol reads, which is how
0.11.1's Linux CLI needed 2.39 for Rust's ``pidfd_*`` (docs/PLAN.md task 115
(i)). ``release.yml`` runs it on the Linux CLI before anything is uploaded.
Standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

GLIBC = re.compile(r"\bGLIBC_(\d+(?:\.\d+)+)\b")


def versions(objdump_text: str) -> list[tuple[int, ...]]:
    """Every ``GLIBC_x.y[.z]`` the text names, as numbers: 2.9 is older than
    2.17, which a sort of the strings would not say."""
    return sorted(
        {tuple(int(p) for p in m.group(1).split(".")) for m in GLIBC.finditer(objdump_text)}
    )


def dotted(v: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary")
    parser.add_argument("--max", default="2.17", help="the newest glibc allowed (default 2.17)")
    args = parser.parse_args()
    text = subprocess.run(
        ["objdump", "-p", args.binary], capture_output=True, text=True, check=True
    ).stdout
    found = versions(text)
    if not found:
        print(
            f"error: {args.binary} names no GLIBC version; is it a glibc binary?", file=sys.stderr
        )
        return 1
    need, most = found[-1], tuple(int(p) for p in args.max.split("."))
    print(f"{args.binary} requires glibc {dotted(need)}")
    if need > most:
        print(
            f"error: {args.binary} needs glibc {dotted(need)}, above {args.max}; "
            f"it requires {', '.join(dotted(v) for v in found)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
