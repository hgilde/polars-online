<!-- kind: how-to -->
# Feed a bank from a database

This how-to is for anyone whose rows live in a database, read through
DuckDB or an ADBC driver, who wants to fit a model bank on an SQL
statement's result without first writing it to a file.

## Prerequisites

These steps need the library, a database client and a table to read:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  the file `ticks.parquet`.
- py-polars 1.43.0 or later, whose `pl.scan_arrow_c_stream` reads the
  statement's result.
- DuckDB or an ADBC driver, which you install yourself: the package
  depends on neither.

## Read an SQL statement's result into a bank

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) reads its
rows in [clock](../GLOSSARY.md#time) order, so the database sorts them,
and Polars reads its result as a query. The rows cross from the database
to Polars through the Arrow PyCapsule interface, so no pyarrow is needed.

1. **Sort the rows by the clock with `ORDER BY` in the SQL statement.**
2. **Run the statement on a cursor of its own,** from `con.cursor()`.
3. **Read its result with Polars' `pl.scan_arrow_c_stream`,** which reads
   it as a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) of
   Polars'.
4. **Chain `.online.fit_predict(specs)` on that query.**
5. **Run the query with `collect()`,** which gives one frame with a
   column for each spec.

This code uses the file `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
import duckdb

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0,
                       half_life=100.0)
con = duckdb.connect()                                    # an in-memory database
sql = "SELECT * FROM 'ticks.parquet' ORDER BY t"          # clock order, sorted by the database
fitted = (
    pl.scan_arrow_c_stream(con.cursor().sql(sql))         # a DuckDB relation, on a cursor of its own
    .online.fit_predict([spec])
    .collect()
)
# With ADBC, on a cursor of its own: cur.execute(sql), then pl.scan_arrow_c_stream(cur.fetch_arrow())
```

## Check the result

**Compare the predictions with those of the same spec over the file
read directly.** This code uses `spec` and `fitted` from [Read an SQL statement's
result into a bank](#read-an-sql-statements-result-into-a-bank):

```python
direct = pl.scan_parquet("ticks.parquet").online.fit_predict([spec]).collect()   # the file, already in clock order
print("the same predictions as the file's:",
      fitted["ridge"].struct.field("pred_y").equals(direct["ridge"].struct.field("pred_y")))
```

```text
the same predictions as the file's: True
```

**To check against real databases, run the two examples.**
[examples/duckdb_cursors.py](../../examples/duckdb_cursors.py) and
[examples/adbc_cursors.py](../../examples/adbc_cursors.py) run these steps
against real databases. Each shows the rule of [Give every SQL statement its own
cursor](#give-every-sql-statement-its-own-cursor) and the trap of breaking it, and
exits non-zero if either stops holding.

## Give every SQL statement its own cursor

**Give every SQL statement its own cursor, from `con.cursor()`, because a
cursor holds one open result.** A second statement run on the same cursor
or connection springs a trap:

| database | what springs the trap | what comes back | what says so |
|---|---|---|---|
| DuckDB | a second statement run on the same connection | the first query yields no rows | `ConsumedSourceWarning` |
| ADBC | the cursor executed again before its first statement's result is read | the newer statement's result one block of rows short, in clock order, and the older one with that block added out of order: on SQLite, 198,976 and 201,024 of 200,000 rows | nothing, for the newer statement; for the older one, a spec with a `clock` refuses the step back |

## Wait before closing an ADBC cursor

**Do not close an ADBC cursor just after a read of its stream stopped
early.** A read stops early on an error, such as a spec refusing a clock
that steps back, or when the consumer stops. Wait about 3 s before closing
the cursor, or leave it open until the process exits. Closing it at once
crashed the process with a segmentation fault, measured on SQLite,
`adbc_driver_manager` 1.12.0 and polars 1.44.2 at 200,000 rows. Either
remedy avoided it.
