<!-- kind: how-to -->
# Hand the output to Arrow

This how-to is for anyone who wants a model bank's output in a library
that is not Polars, such as pyarrow or DuckDB, through the Arrow
PyCapsule interface.

To hand a [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)'s
output to a consumer that is not Polars, call `bank.fit_predict_arrow(df)`
or `bank.predict_arrow(df)` in place of `fit_predict` or `predict`. Each
returns one `ArrowStruct` per spec, which holds that spec's output through
the Arrow PyCapsule interface, an Arrow specification, so any reader of
`__arrow_c_array__` takes it directly. Its values are `fit_predict`'s,
field for field and null for null. The input still goes in as a Polars
`DataFrame`. `fit_predict` itself hands its output over as a Polars
`Series`, on py-polars' private methods, so this package measures a Polars
range and promises none ([Which interfaces carry a
promise](../explanation/polars-interfaces.md)). This export is unstable:
its form may change in a minor release ([What is
stable](../VERSIONS.md#what-is-stable)).

## Prerequisites

These steps need the library and a frame to fit:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `df`, a frame of 400 rows on the clock `t`, and `today`, the 400 rows
  that follow it.
- A consumer that reads `__arrow_c_array__`, such as pyarrow, or one that
  takes a Polars `Series`, such as DuckDB ([Pass a struct to
  pyarrow or DuckDB](#pass-a-struct-to-pyarrow-or-duckdb)).

## Export the output as Arrow

**Read each struct once.** Exporting hands its buffers to the consumer, so
a second read raises `ValueError`.

1. **Fit the bank with `bank.fit_predict_arrow(df)`,** which returns a
   list of one `ArrowStruct` per spec, in the specs' order.
2. **Read each struct once with your consumer,** here `pl.Series`, and
   keep what that read built.
3. **Serve new rows the same way with `bank.predict_arrow(df)`.**

This code uses `df` and `today` from [Example
data](../../README.md#example-data):

```python
# One spec's output as Arrow, read once as a Series.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
structs = bank.fit_predict_arrow(df)                    # 1. one per spec, as Arrow
ridge = pl.Series(structs[0])                           # 2. Polars as the reader here: any reader of __arrow_c_array__ takes it
print(f"{len(structs)} {type(structs[0]).__name__}; read as a Series: {ridge.name}, {ridge.len()} rows")
print(f"the same as fit_predict's column: {ridge.equals(po.ModelBank([spec]).fit_predict(df)['ridge'])}")
# A second read of the same struct is refused.
try:
    pl.Series(structs[0])                               # the second read
except ValueError as e:
    print(e)
scored_structs = bank.predict_arrow(today)              # 3. the same for predict, on today: the rows after df's
```

```text
1 ArrowStruct; read as a Series: ridge, 400 rows
the same as fit_predict's column: True
ArrowStruct.__arrow_c_array__(): this struct has already been exported, and the Arrow PyCapsule interface hands its buffers to the consumer, so it can be read once. Keep what the first read built, such as its pl.Series or pyarrow array, and read that.
```

## Check the result

**Compare what the consumer read with `fit_predict`'s column.** In the
output of [Export the output as Arrow](#export-the-output-as-arrow), the
`Series` read from the struct equals `fit_predict`'s column `ridge`, and a
second read of the same struct is refused.

## Pass a struct to pyarrow or DuckDB

pyarrow takes a struct as it is, and DuckDB takes it through a Polars
`Series`, as measured on the versions shown:

| consumer | measured on | takes |
|---|---|---|
| pyarrow | pyarrow 25.0.1 | a struct as it is: `pa.array(s)` and `pa.table(s)` each take one, values and types unchanged |
| DuckDB | duckdb 1.5.5 | `pl.Series(s)`. It refuses an `ArrowStruct`, because `__arrow_c_stream__` is the method it looks for. A spec's output arrives table-shaped, one column per field |
