"""Generate the parquet input that examples/bank.toml expects.

uv run python scripts/make_example_parquet.py [--rows N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from data import synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=Path("example-input.parquet"))
    args = ap.parse_args()

    df, _ = synthetic(seed=0, n_groups=3, n_rows=args.rows, k=3, session_every=5_000)
    df = df.rename({"y0": "y"}).with_columns(session=df["session"].cast(str))
    df.write_parquet(args.out)
    print(f"wrote {df.height} rows to {args.out}")


if __name__ == "__main__":
    main()
