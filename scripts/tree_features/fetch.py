"""Fetch the public data the prototypes use, through tests/data.py's helpers.
Run: .venv/bin/python -I fetch.py <slot>"""

import io
import sys
from datetime import date, timedelta
from pathlib import Path

slot = Path(sys.argv[1])
sys.path.insert(0, str(slot / "tests"))
import polars as pl  # noqa: E402

import data  # noqa: E402

d0 = date(2024, 1, 2)
days = tuple((d0 + timedelta(days=i)).isoformat() for i in range(60))
df = data.public_intraday(days)
print("btc rows", df.height)
out = data.CACHE_DIR / "tree-features"
out.mkdir(parents=True, exist_ok=True)
f = out / "nycflights13-flights.parquet"
if not f.exists():
    raw = data._download(
        "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv"
    )
    (out / "nycflights13-flights.csv").write_bytes(raw)
    fl = pl.read_csv(io.BytesIO(raw), null_values=["NA"], infer_schema_length=100000)
    data.write_atomically(f, fl.write_parquet)
print(pl.read_parquet(f).schema, pl.read_parquet(f).height)
