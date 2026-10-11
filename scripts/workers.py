"""How many processes the test runners start on this machine: one number.

    python3 scripts/workers.py        # prints it

The cores the machine has free, read as its core count less the last
minute's load average, and at least 2 and at most 8. The machine this
project is developed on is shared, and another user's jobs take up to 11
of its 14 cores, so a fixed count would either leave cores idle or run
three times the work the cores can take. At least 2: a parallel run still
finishes sooner than one process when the load is high. At most 8: past
it, more processes buy little and crowd the machine, since every worker
imports the package and builds its own thread pools. Measured on
2026-10-11, the Python essentials took 118 s in one process, 39 s in 4,
30 s in 8 and 27 s in 12, and 12 drove the load average from 23 to 51.

`GATE_WORKERS` overrides it: `GATE_WORKERS=4` runs four, and 0 or 1 means
one process (`scripts/gate.sh --serial` does the same). Where the load
average cannot be read (Windows), half the cores are taken as free.

Standard library only, so the gate runs it before any environment exists.
"""

from __future__ import annotations

import os

#: The fewest and the most processes a parallel run starts.
LEAST, MOST = 2, 8


def workers(cores: int, load: float | None) -> int:
    """The processes to start for `cores` cores at a one-minute `load`
    average, or half the cores free when the load is unknown."""
    free = cores // 2 if load is None else cores - round(load)
    return max(LEAST, min(MOST, free))


def main() -> int:
    override = os.environ.get("GATE_WORKERS")
    if override is not None:
        print(max(1, int(override)))
        return 0
    try:
        load: float | None = os.getloadavg()[0]
    except (AttributeError, OSError):
        load = None
    print(workers(os.cpu_count() or 1, load))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
