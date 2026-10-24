"""Prove `scripts/memory_scope.sh` on the runner it is used on (mutants.yml's
`memory-cap` job): inside a scope with the mutants jobs' own `MemoryMax`, a
process that allocates without end is killed by the kernel once the scope
holds about that much, and the command that started it goes on, and so does
the job.

    python3 scripts/memory_scope_check.py 13G

Standard library only. Exits non-zero, saying why, if the runaway was not
killed, was killed long before the cap, or took its parent with it, or if
the caller's environment did not reach the command in the scope (sudo-rs,
on ubuntu-26.04, ignores `sudo -E`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CHUNK_MIB = 64
#: Set here, read inside the scope: the caller's environment got through.
SENTINEL = "MEMORY_SCOPE_CHECK_SENTINEL"
SENTINEL_VALUE = "reached the scope = yes"

#: The runaway: allocate 64 MiB at a time, write a byte to every page so
#: the memory is resident rather than reserved, and say how far it got.
RUNAWAY = f"""
import sys
held, n = [], {CHUNK_MIB} << 20
while True:
    buf = bytearray(n)
    buf[::4096] = b"\\x01" * len(range(0, n, 4096))
    held.append(buf)
    print(len(held) * {CHUNK_MIB}, flush=True)
"""


def mib(text: str) -> int:
    """`13G`, `512M` or a number of bytes, in MiB."""
    units = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    if text[-1].upper() in units:
        return int(float(text[:-1]) * units[text[-1].upper()])
    return int(text) >> 20


def inside() -> None:
    """In the scope: start the runaway, wait for it, then show this process
    still runs by allocating and touching memory of its own."""
    child = subprocess.Popen(
        [sys.executable, "-c", RUNAWAY], stdout=subprocess.PIPE, text=True, encoding="utf-8"
    )
    assert child.stdout is not None
    reached = 0
    for line in child.stdout:
        reached = int(line)
    code = child.wait()
    mine = bytearray(CHUNK_MIB << 20)
    mine[::4096] = b"\x01" * len(range(0, len(mine), 4096))
    print(
        json.dumps(
            {
                "runaway_exit": code,
                "reached_mib": reached,
                "survivor": len(mine),
                "env": os.environ.get(SENTINEL),
            }
        )
    )


def main(cap: str) -> int:
    r = subprocess.run(
        ["bash", str(HERE.parent / "memory_scope.sh"), cap, "--", sys.executable, str(HERE)]
        + ["--inside"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, SENTINEL: SENTINEL_VALUE},
    )
    print(r.stdout, r.stderr, sep="\n")
    if r.returncode != 0:
        print(f"::error::the command inside the scope exited {r.returncode}")
        return 1
    got = json.loads(r.stdout.strip().splitlines()[-1])
    problems = []
    if got["runaway_exit"] != -9:
        problems.append(f"the runaway exited {got['runaway_exit']}, not killed (SIGKILL, -9)")
    # Killed near the cap, not at some smaller limit: within the last
    # quarter of it. The scope's other processes hold the rest.
    if not 0.75 * mib(cap) <= got["reached_mib"] <= mib(cap):
        problems.append(f"the runaway reached {got['reached_mib']} MiB against a cap of {cap}")
    if got["survivor"] != CHUNK_MIB << 20:
        problems.append("the command that started the runaway did not go on")
    if got.get("env") != SENTINEL_VALUE:
        seen = got.get("env")
        problems.append(f"the caller's environment did not reach the scope ({SENTINEL}={seen!r})")
    for p in problems:
        print(f"::error::{p}")
    if not problems:
        reached = got["reached_mib"]
        print(f"the runaway was killed at {reached} MiB of {cap}, and its parent went on")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--inside"]:
        inside()
    else:
        sys.exit(main(sys.argv[1]))
