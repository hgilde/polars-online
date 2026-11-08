<!-- kind: how-to -->
# Run a bank in Pathway

This how-to is for anyone with a [Pathway](https://pathway.com) pipeline,
a Python framework for streaming pipelines, who wants a model bank to
predict and learn inside it.

## Prerequisites

These steps need the library and, for a real pipeline, Pathway:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `df`, 400 rows in clock order.
- For a pipeline, Pathway, installed under its own license, and an input
  connector. Pathway is not a dependency of this package, and the example
  runs without it.

## Wrap the bank in an operator

**Wrap a `ModelBank` in a stateful operator.** The operator calls
`fit_predict` on each [chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads)
of rows Pathway hands it (Pathway calls it a batch), and Pathway
checkpoints it with `save_bytes` and `load_bytes`. Pathway does the
ingestion, the event-time alignment and the windowing before the
operator. A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
goes on from its [state](../GLOSSARY.md#saving-and-serving) after a
restore, so a pipeline stopped and started again predicts as one run
would have.

1. **Hold a `ModelBank` as the operator's state,** built from your specs.
2. **On each chunk, call `bank.fit_predict(batch)`,** and return its
   output. Pathway hands the operator each chunk in order.
3. **For Pathway's checkpoint, return `bank.save_bytes()`.**
4. **For Pathway's restore, load the bytes with
   `po.ModelBank.load_bytes(blob, specs=specs)`.**

This code feeds the operator two plain frames with a checkpoint and a
restore between them, and compares its predictions with one run's over the
same rows. It uses `df` from [Example data](../../README.md#example-data):

```python
# The operator: a bank that Pathway feeds and checkpoints.
class BankOperator:                                  # the operator's state is the bank
    def __init__(self, specs):
        self.specs, self.bank = specs, po.ModelBank(specs)

    def __call__(self, batch: pl.DataFrame) -> pl.DataFrame:
        return self.bank.fit_predict(batch)          # each ordered chunk Pathway hands it

    # Pathway's checkpoint, and its restore.
    def snapshot(self) -> bytes:
        return self.bank.save_bytes()                # what Pathway's persistence stores

    def restore(self, blob: bytes) -> None:
        self.bank = po.ModelBank.load_bytes(blob, specs=self.specs)


# Feed it two chunks, with a checkpoint and a restore between them.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0,
                       half_life=100.0)
op = BankOperator([spec])
first = op(df.head(200))                             # plain frames stand in for Pathway's chunks
op.restore(op.snapshot())                            # a checkpoint and a restore between two chunks ...
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
runs this operator over plain frames, so it runs without Pathway. Its
sketch of the pipeline around the operator needs Pathway and an input
connector.
