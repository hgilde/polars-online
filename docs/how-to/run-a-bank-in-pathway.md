<!-- kind: how-to -->
# Run a bank in Pathway

This how-to is for anyone with a [Pathway](https://pathway.com) pipeline,
a Python framework for streaming pipelines, who wants a model bank to
predict and learn inside it.

## Prerequisites

These steps need the library and, for a real pipeline, Pathway:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The example reads
  `df`, 400 rows in clock order.
- For a pipeline, Pathway, installed under its own license, and an input
  connector. Pathway is not a dependency of this package, and the example
  runs without it.

## Wrap the bank in an operator

**Wrap a `ModelBank` in a stateful operator of your own.** The operator
calls `fit_predict` on each [chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads)
of rows it is handed, and saves and restores the bank as bytes. This
package ships no Pathway hookup. Pathway's ingestion and windowing go
before the operator, and connecting the operator's saving and restoring
to Pathway's persistence is yours to wire. A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) goes on from its
[state](../GLOSSARY.md#saving-and-serving) after a restore, so once that
is wired, a pipeline stopped and started again predicts as one run would
have.

1. **Hold a `ModelBank` as the operator's state,** built from your specs.
2. **On each chunk, call `bank.fit_predict(chunk)`,** and return its
   output. Hand the operator its chunks in clock order.
3. **To save the bank, keep `bank.save_bytes()`** wherever your pipeline's
   persistence keeps state.
4. **To restore, load those bytes with
   `po.ModelBank.load_bytes(blob, specs=specs)`.**

This code feeds the operator two plain frames with a save and a
restore between them, and compares its predictions with one run's over the
same rows. It uses `df` from [Example data](../EXAMPLE-DATA.md):

```python
# The operator: a bank fed a chunk at a time, saved and restored as bytes.
class BankOperator:                                  # the operator's state is the bank
    def __init__(self, specs):
        self.specs, self.bank = specs, po.ModelBank(specs)

    def __call__(self, chunk: pl.DataFrame) -> pl.DataFrame:
        return self.bank.fit_predict(chunk)          # each chunk, in clock order

    # Save the state, and restore it: for your pipeline's persistence to call.
    def snapshot(self) -> bytes:
        return self.bank.save_bytes()                # the bytes to store

    def restore(self, blob: bytes) -> None:
        self.bank = po.ModelBank.load_bytes(blob, specs=self.specs)


# Feed it two chunks, with a save and a restore between them.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0,
                       half_life=100.0)
op = BankOperator([spec])
first = op(df.head(200))                             # plain frames stand in for a pipeline's chunks
op.restore(op.snapshot())                            # a save and a restore between two chunks ...
rest = op(df.tail(200))                              # ... and the bank goes on where it was

# Compare with one run over the same rows.
one_run = po.ModelBank([spec]).fit_predict(df)       # the same rows, in one run
print("rows predicted:", first.height + rest.height)
restored = pl.concat([first, rest])
print("the same predictions as one run:",
      restored["ridge"].struct.field("pred_y").equals(one_run["ridge"].struct.field("pred_y")))
```

```text
rows predicted: 400
the same predictions as one run: True
```

## Check the result

**Compare the operator's predictions across a restore with one run's over
the same rows,** as the last lines of the example do: the two are equal.
[examples/pathway_integration.py](../../examples/pathway_integration.py)
runs the same operator over plain frames of 500 rows, without Pathway. It
also holds a sketch of a Pathway pipeline, which needs Pathway and a
directory of CSV files to read. The sketch calls the operator through
`pw.udf` one row at a time, as a one-row frame, and saves no state:
it shows where the operator sits, not a pipeline that resumes.
