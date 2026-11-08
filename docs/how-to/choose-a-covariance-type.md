<!-- kind: how-to -->
# Choose a covariance type

This how-to is for anyone who classifies rows with `ew_class` and must choose
what each class keeps: `covariance_type`, scikit-learn's word for the shape
of each class's covariance. It also sets `window_size` when the classes
move, and `embargo` when each row's class arrives late.

## Prerequisites

These steps need the library, a stream with its classes, and a spec:

- polars-online installed, and numpy to make the example's data.
- A stream of numeric
  [feature](../GLOSSARY.md#the-model-bank-and-what-it-reads) columns and
  a column of classes, saved as a file in clock order, to read back with
  `pl.scan_parquet`.
- An [`ew_class`](../MODELS.md#ew_class--gaussian-classification-on-ew_cov-moments)
  spec: the class column in `label=`, its values in `classes=`, and a
  `precision_prior`, a small ridge on each class's covariance that lets a
  class be predicted from its first row.

## Choose the shape of each class's covariance

Each class keeps a weight, a mean and a covariance, and a row is put in
the class that makes it most likely. `covariance_type` chooses how much of
each covariance the model learns:

| `covariance_type` | the right choice when |
|---|---|
| `"full"`, the default (quadratic discriminant analysis) | in general |
| `"tied"`, one covariance for every class (linear discriminant analysis) | the classes differ in location but not in spread. It has fewer parameters to learn |
| `"diag"`, each class's variances alone (Gaussian naive Bayes) | speed matters most: at 20 features and three classes it runs about four times as fast as the other two ([Throughput](../../README.md#throughput)), and the gap grows with the number of features. It cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it |

1. **Start from the default, `"full"`.** It is the one shape that sees
   each class's correlation between features.
2. **Fit `"tied"` or `"diag"` beside it, as other specs in the same bank,**
   each spec named after its shape.
3. **Score each spec's `class` against the label on rows after the
   [warm-up](../GLOSSARY.md#warm-up-and-readiness),** and keep the
   cheapest shape that classifies as many rows right as `"full"`.

This code makes two kinds of class. In the first, two classes share their
marginals and differ only in the sign of their correlation. In the second,
two classes share one covariance and differ in their means. It scores the
three shapes on the last 5,000 rows of each:

```python
import numpy as np
import polars as pl
import polars_online as po

rng = np.random.default_rng(8)
n = 6_000
label = rng.integers(0, 2, n)
z = rng.standard_normal((n, 2))
kind = np.array(["a", "b"])[label]
# Two classes with the same marginals and opposite correlations, 0.8 and -0.8.
r = np.where(label == 0, 0.8, -0.8)
pl.DataFrame({"x": z[:, 0], "y": r * z[:, 0] + np.sqrt(1 - r**2) * z[:, 1], "kind": kind}).write_parquet("crossed.parquet")
# Two classes with one covariance and different means.
xy = np.array([[0.0, 0.0], [1.5, 1.0]])[label] + z @ np.linalg.cholesky([[1.0, 0.6], [0.6, 1.0]]).T
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "kind": kind}).write_parquet("shifted.parquet")

# 1 and 2. The three shapes, one spec each, in one bank.
shapes = [po.spec.ew_class(shape, features=["x", "y"], label="kind", classes=["a", "b"],
                           covariance_type=shape, precision_prior=0.1, half_life=float("inf"))
          for shape in ("full", "tied", "diag")]
# 3. The share of the last rows each shape classifies right.
for data in ("crossed", "shifted"):
    out = pl.scan_parquet(f"{data}.parquet").online.fit_predict(shapes).collect().tail(5_000)
    right = {shape: (out[shape].struct.field("class") == out["kind"]).mean() for shape in ("full", "tied", "diag")}
    print(f"{data}: " + ", ".join(f"{shape} {share:.3f}" for shape, share in right.items()))
```

```text
crossed: full 0.799, tied 0.467, diag 0.490
shifted: full 0.783, tied 0.784, diag 0.775
```

Where the classes differ only in their correlation, `"tied"` and `"diag"`
do no better than a coin, and `"full"` classifies four rows in five right.
Where they differ only in location, the three shapes classify about as many
rows right, so `"tied"` or `"diag"` does the same work with less.

## Give `window_size` when the classes move

Use a hard window when a class's mean moves. Over a long history two
regimes average together, and the predicted classes go to chance.

1. **Give `window_size`, in the clock's units.** Each class's moments then
   cover only the rows inside the [hard
   window](../../README.md#a-hard-window), and a row is predicted against
   them. Without a clock, the units are rows.
2. **Run the query, and score each stretch of rows after a move.**

This code makes two classes that trade places at row 3,000, and scores a
spec that weighs every row against one that keeps the last 500:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two classes on either side of 0 in x, which trade places at row 3,000.
rng = np.random.default_rng(8)
n = 6_000
label = rng.integers(0, 2, n)
side = np.where(label == 0, -1.0, 1.0) * np.where(np.arange(n) < 3_000, 1.0, -1.0)
xy = np.column_stack([side, np.zeros(n)]) + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "kind": np.array(["a", "b"])[label]}).write_parquet("swapped.parquet")

# 1. One spec over every row, and one over the last 500.
common = dict(features=["x", "y"], label="kind", classes=["a", "b"], precision_prior=0.1, half_life=float("inf"))
specs = [po.spec.ew_class("whole", **common), po.spec.ew_class("window", **common, window_size=500.0)]
out = pl.scan_parquet("swapped.parquet").online.fit_predict(specs).collect()
# 2. The share of rows each classifies right, before the move, just after it, and from then on.
for lo, hi in ((2_000, 3_000), (3_000, 3_500), (3_500, 6_000)):
    rows = out.slice(lo, hi - lo)
    right = {k: (rows[k].struct.field("class") == rows["kind"]).mean() for k in ("whole", "window")}
    print(f"rows {lo}-{hi - 1}: whole {right['whole']:.3f}, window {right['window']:.3f}")
```

```text
rows 2000-2999: whole 0.848, window 0.847
rows 3000-3499: whole 0.136, window 0.486
rows 3500-5999: whole 0.182, window 0.829
```

The spec over every row keeps the old places, and classifies most rows
wrong. The windowed spec has learned the new places by the time its
window holds only rows after the move.

## Give `embargo` when the classes arrive late

Use an embargo when a row's class is known only some time after its
features. [Labels that arrive
late](../../README.md#labels-that-arrive-late) shows the parameter on a
regression.

1. **Give `embargo=` the delay, in the clock's units,** such as `"5m"` on
   a temporal clock. Without a clock, the units are rows.
2. **Run the query.** Each row is predicted when it arrives, and its class
   is learned once that much clock has passed, so `class` stays null until
   the first classes have been learned.

This code reads `swapped.parquet`, which the recipe in [Give `window_size`
when the classes move](#give-window_size-when-the-classes-move) wrote, with
that recipe's imports, and learns each row's class 50 rows after the row:

```python
# 1. Each class learned 50 rows after its row.
late = po.spec.ew_class("late", features=["x", "y"], label="kind", classes=["a", "b"],
                        precision_prior=0.1, half_life=float("inf"), embargo=50)
# 2. The first row with a class.
out = pl.scan_parquet("swapped.parquet").online.fit_predict([late]).collect().unnest("late").with_row_index("row")
print(f"first row with a class: {out.filter(pl.col('class').is_not_null())['row'][0]}")
```

```text
first row with a class: 52
```

The first class is learned at row 50, and `class` is withheld until
`weight_sum`, which counts only the rows learned, reaches `min_weight`,
two rows later.

## Check that it worked

Score each spec's `class` against the label on rows after the warm-up, as
the recipes do. To see how sure the model is, read each class's
probability, `p_<class>`.

| what you see | what it means | what to do |
|---|---|---|
| `"tied"` or `"diag"` far below `"full"` | the classes differ in their spread or their correlation, which those shapes do not keep | keep `"full"` |
| `class` near a coin's rate on rows long after a change | the class means moved, and the history averages the old places with the new | give `window_size` |
| `class` and every `p_<class>` null on the first rows | `weight_sum` is below `min_weight`, or no class has been seen | read the rows after the warm-up |
| `class` null for longer than the warm-up | under an `embargo`, no class is learned until that much clock has passed | read the rows after the embargo and the warm-up |
| a `ValueError` naming a row | its `label` column holds a value `classes` does not list | list in `classes` every value the column can hold |

The fields are defined in [OUTPUTS.md](../OUTPUTS.md#ew_class), and every
parameter, with the work each shape takes a row, in `ew_class`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class).
