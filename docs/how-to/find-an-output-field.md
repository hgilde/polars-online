<!-- kind: how-to -->
# Find an output field by what it holds

This how-to is for anyone whose code reads a model bank's output by field
name, and who wants the name of the field that holds a given prediction or
coefficient without building the name by hand.

Each [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) adds one
[output column](../GLOSSARY.md#what-a-bank-writes) to the rows, named
after the spec, whose value in each row is a record of named
[fields](../GLOSSARY.md#what-a-bank-writes).
[OUTPUTS.md](../OUTPUTS.md) lists every field of every model. Give a spec
several values of `ridge` or `half_life`, several `feature_sets` or
`penalties`, and it fits a [grid](../GLOSSARY.md#grids): one fit per
combination of values, each a [grid point](../GLOSSARY.md#grids). A
field's name says which grid point it holds.

**The names are part of the stable API, so no release renames your
columns silently** ([What is stable](../VERSIONS.md#what-is-stable)).

## Prerequisites

These steps need the library and a spec:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The example reads
  `df`, a frame of 400 rows whose target `y` is `1 + 2·x0 − x1` plus
  noise.
- The spec whose output you read. `po.spec.output_index` and
  `po.spec.coef_fields` build their tables from the spec alone, before
  any row is fed.

## Find a field's name from the values it holds

Filter one of two tables, each built from the spec by the same code that
renders the names, so they always agree with the output:

| table | one row per | use it for |
|---|---|---|
| `po.spec.output_index(spec)` | field: its name, its kind, the target, ridge and half-life its name encodes, and its dtype | any field, such as a prediction |
| `po.spec.coef_fields(spec)` | coefficient: the list field it sits in, its position there, and its column once unnested | one coefficient |

1. **Filter `po.spec.output_index(spec)` on the values you want,** such as
   `kind`, `target`, `ridge` and `half_life`, and take its `field`.
2. **Read that field from the spec's column** with Polars'
   `struct.field(name)`.
3. **For a coefficient, filter `po.spec.coef_fields(spec)` on its `term`
   and grid values,** and read its `position` in the list its `field`
   names.

Here `grid` has two half-lives and two ridge values, and its coefficient
on `x1` at the smaller ridge comes back as `df`'s −1. This code uses `df`
from [Example data](../EXAMPLE-DATA.md):

```python
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       gap_cap=300.0, half_life=[100.0, 500.0], ridge=[1e-6, 0.5])
out = df.online.fit_predict([grid])  # df's columns, plus the spec's column "m"

# 1 and 2. A field's name, found from the values it holds.
idx = po.spec.output_index(grid)     # every field, with the values its name encodes, and its dtype
name = idx.filter((pl.col("kind") == "pred") & (pl.col("target") == "y")
                  & (pl.col("ridge") == 1e-6) & (pl.col("half_life") == 500.0))["field"].item()
print(name, out["m"].struct.field(name).dtype)

# 3. A coefficient's place in its list, found the same way.
row = po.spec.coef_fields(grid).filter(                        # one row per coefficient: its list, position, and unnest column
    (pl.col("term") == "x1") & (pl.col("ridge") == 1e-6) & (pl.col("half_life") == 500.0)
).row(0, named=True)
x1 = out["m"].struct.field(row["field"]).drop_nulls().list.get(row["position"])
print(row["field"], row["position"], f"{x1[-1]:.2f}")          # x1's coefficient on the last row
```

```text
pred_y__r0.000001@h500 Float64
coef@h500 2 -1.00
```

**If your own code parses the field names after the bank, keep `__` and
`@` out of target names and feature-set labels.** A target named `y__r0.5`
renders like a ridge grid on `y`. Filtering `po.spec.output_index` needs no
such care.

## Read a name by eye

**Each name follows a grammar.** The kind and the target come first, then
the grid point after `__`, then the instance (the part of the model that
fits one half-life of a grid) after `@`. [Field
names](../OUTPUTS.md#field-names) gives what each part stands for. In
the block below, `{target}` is its `<t>`,
`{combo}` one of its `__` forms, such as `__r<ridge>`, and `{instance}`
its `@h<half_life>`:

```
pred_{target}{combo}{instance}
resid_{target}{combo}{instance}
sigma_{target}{combo}{instance}
abs_resid_q{level}_{target}...
weight_sum{instance}
settled_frac{instance}
withheld_reason{instance}
coef{instance}
support_coef{instance}
scored_clock, learned_clock        under emit_clocks, once per spec
```

A number in a name renders as a plain decimal in `[1e-6, 1e7)`, as
`pred_y__r0.000001@h500`'s ridge, and in compact scientific notation outside it, as
`1e-300`.

## Check that it worked

**List the spec's fields with `po.spec.output_fields(spec)`, or the
output's with `out["m"].struct.fields`, and find the name there.** A name
that `output_index` gives is always among them, as `pred_y__r0.000001@h500`
is in the example. [Read the coefficients](read-the-coefficients.md)
spreads a grid's `coef` lists into columns named the same way.
