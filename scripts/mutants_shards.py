"""How many shards the changed lines' mutation pass deals its mutants into.

    python3 scripts/mutants_shards.py listed.txt --per-shard 40 >> "$GITHUB_OUTPUT"

Reads the pass's listing, `cargo mutants --list` with the arguments every
shard runs, one mutant a line, and prints the three lines `mutants.yml`'s
listing job appends to `$GITHUB_OUTPUT`: `mutants`, the count; `shards`, how
many shards; and `matrix`, the shards' indices as the JSON list the shards'
matrix takes. It also says the count and the shards on stderr, for the log.

A shard is dealt at most `--per-shard` mutants, so every mutant the change
listed is tested however large the push (task 219). There is at least one
shard, so a change with nothing to mutate still runs one, which is dealt
nothing, stops before it builds anything, and reports. There are at most
256, GitHub's limit on the jobs a matrix makes, so past 256 times
`--per-shard` mutants each shard is dealt more.

Standard library only, so a CI job runs it with the runner's own Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: GitHub's limit on the jobs one matrix makes in a workflow run.
MOST_SHARDS = 256


def shards(mutants: int, per_shard: int, most: int = MOST_SHARDS) -> int:
    """The fewest shards that deal no shard more than `per_shard` mutants,
    at least one and at most `most`."""
    return min(max(1, -(-mutants // per_shard)), most)


def listed(path: Path) -> int:
    """How many mutants a listing names: one a line, blank lines aside."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("listing", type=Path, help="`cargo mutants --list` output, one mutant a line")
    ap.add_argument(
        "--per-shard", type=int, required=True, help="the most mutants a shard is dealt"
    )
    args = ap.parse_args()
    if args.per_shard < 1:
        ap.error("--per-shard must be at least 1")
    n = listed(args.listing)
    k = shards(n, args.per_shard)
    print(f"mutants={n}")
    print(f"shards={k}")
    print(f"matrix={json.dumps(list(range(k)), separators=(',', ':'))}")
    print(f"mutants: {n}, shards: {k}, the most a shard is dealt: {-(-n // k)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
