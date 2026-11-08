<!-- kind: explanation -->
# Grids, instances and slots

This page explains, for anyone who gives a spec several values of a
setting, how one pass fits them all, what an instance and a slot are, and
why the best error over a grid is optimistic.

**A spec given several values for a setting fits every combination of
them side by side, from the same rows, in one pass.** A
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) is the
description of one model that a [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits. The values
it lists form a [grid](../GLOSSARY.md#grids), and each combination is a
[grid point](../GLOSSARY.md#grids), such as `half_life=100` with
`ridge=0.1` and the feature set `one`. A grid shares its work according to what each setting
changes. A [half-life](../GLOSSARY.md#time) changes what the model
remembers, so each half-life keeps [state](../GLOSSARY.md#saving-and-serving) of
its own, in an [instance](../GLOSSARY.md#grids) (the part of the model that fits
that half-life, and not a Python object). In `ewridge` that state is a [Gram](../GLOSSARY.md#fitting-a-model): the
weight, means and co-moments a least-squares fit is
[solved](../GLOSSARY.md#fitting-a-model) from. A ridge value or a
feature set changes only how the fit is solved from it, so each grid
point it makes adds a solve and nothing to learning a row.
Here is a spec with two half-lives, two ridge values and two feature
sets, eight grid points in all, with the two [switches](../GLOSSARY.md#fitting-a-model) that pick or
blend among them as the stream runs:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = x0 - x1 plus noise, a row each unit of the clock column t.
rng = np.random.default_rng(1)
n = 2_000
x0, x1, e = rng.standard_normal((3, n))
frame = pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": x0 - x1 + e})
frame.write_parquet("grid.parquet")

# Two half-lives, by two ridge values, by two feature sets: eight grid points in one spec.
grid = po.spec.ewridge(
    "m", targets=["y"], features=["x0", "x1"],
    clock="t", gap_cap=5.0,                              # the clock, and the largest step a model decays by
    half_life=[100.0, 1000.0],                           # an instance each, with a Gram of its own
    ridge=[1e-6, 0.1],                                   # a solve each, from an instance's Gram
    feature_sets={"one": ["x0"], "both": ["x0", "x1"]},  # likewise
    emit_selected=True,                                  # the grid point with the lowest error so far
    emit_averaged=True,                                  # every grid point, weighted by its error
)
bank = po.ModelBank([grid])
bank.fit(pl.scan_parquet("grid.parquet"))
# The state: one Gram per instance.
sums = bank.gram("m")
print(f"{len(sums)} Grams: " + ", ".join(f"{g['instance']} (weight_sum {g['weight_sum']:.1f})" for g in sums))
# The predictions: one slot per grid point, and the two that choose among them.
slots = po.spec.output_index(grid).filter(pl.col("field").str.starts_with("pred_"))
print(f"{slots.height} slots:")
for field, instance in slots.select("field", "instance").iter_rows():
    print(f"  {field:30} instance {instance}")
```

```text
2 Grams: @h100 (weight_sum 144.8), @h1000 (weight_sum 1082.4)
10 slots:
  pred_y__one_r0.000001@h100     instance @h100
  pred_y__one_r0.1@h100          instance @h100
  pred_y__both_r0.000001@h100    instance @h100
  pred_y__both_r0.1@h100         instance @h100
  pred_y__one_r0.000001@h1000    instance @h1000
  pred_y__one_r0.1@h1000         instance @h1000
  pred_y__both_r0.000001@h1000   instance @h1000
  pred_y__both_r0.1@h1000        instance @h1000
  pred_y__selected               instance None
  pred_y__averaged               instance None
```

The bank keeps two Grams for the eight grid points, and writes ten
predictions on every row, each a [slot](../GLOSSARY.md#grids).
[`weight_sum`](../GLOSSARY.md#warm-up-and-readiness) is the decayed
weight of the rows an instance has learned, which a longer half-life
forgets more slowly. How
each field's name encodes its grid point is in the README's [Output field
names](../../README.md#output-field-names).

**Some settings take a list that is not a grid, and some take no list.**
`kalman`'s `coef_half_life`, its drift speed, takes one value per
coefficient. `sgd`'s `learning_rate`, `holt`'s `trend_half_life` and
`bocpd`'s `hazard` take a single value. For each of them, give the bank
one spec per value, which `po.eval.grid_search` builds from its
`param_grid`.
The bank still reads the stream once for all of them. [Choose a setting that takes no list](../CALIBRATION.md#choose-a-setting-that-takes-no-list)
shows how.

## An instance: one half-life, with state of its own

**Each half-life of a spec has an instance: the part of the model that
fits that half-life, with state of its own.** In each
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the rows that
share one value of the column `group=` names), the bank keeps one model
per spec, and that model keeps one instance for each value of the spec's
`half_life`. The decay is inside the state: each row enters it at a
weight that the half-life shrinks as the [clock](../GLOSSARY.md#time)
moves. So two half-lives
cannot share a state. A spec with one half-life, or one that sets its decay by
[`lam`](../GLOSSARY.md#time) (a decay factor per clock unit), has one
instance.

**An instance is named by a suffix of its half-life.** The name is
`@h100` for a half-life of 100 (`@h10m` for a duration), and `""` where
the spec has only one instance. It is the `instance` column of
`bank.coef()` and `po.spec.output_index()`, and the `instance` key of
each Gram `bank.gram()` returns. A slot that chooses across
instances, such as `pred_y__selected`, has
none.

## How a grid shares its work

**In a model that keeps a Gram, each grid point within an instance is one
solve from the instance's Gram.** `ewridge` keeps the means and
co-moments of its columns, its [Gram](../GLOSSARY.md#fitting-a-model). A ridge value
is added to the Gram's diagonal when the coefficients are solved, and a
feature set reads the rows and columns of its features out of the same
Gram. So every combination of ridge values and feature sets is a solve of
its own, and learning a row updates the Gram once, whatever the grid.
Only the named feature sets are fitted, so a fit on every feature exists
only where a set names them all. `lasso` solves each of its `penalties` from the same Gram, by coordinate descent, where `ewridge`
factorizes.

**A grid of ridge values and feature sets adds solves, and no work to
learning a row.** The bank reads each row once and updates each
instance's state once.
`ewridge` and `lasso` solve each grid point on the schedule `solve_every`
sets. In `ewridge`, learning a row updates an instance's Gram in O(k²)
time for `k` features. Because the grid's solves read the same Gram, a
grid of ridge values and feature sets adds nothing to that. Each grid point still makes its own prediction
on every row, in O(k) time. In `ewridge` a solve factorizes a `k × k` system, in O(k³) time. A
`lasso` solve runs coordinate descent, O(k²) a sweep. Under a finite half-life the default schedule solves
about every `half_life / 50` clock units once its weight has neared its
steady level, a few half-lives in, and more often before that. So where a half-life
spans many rows, a grid of solves adds little to the time a row takes.
Under a half-life of a few tens of rows, or under
`half_life=float("inf")` or `lam`, where the default solves on every
row, each grid point adds a solve every row or two.

**The instances of a group run in parallel on the bank's threads.** The
one setting that ties them together is `drift_action="reset"`, which resets the model when its drift detector
([Has it broken?](../DIAGNOSTICS.md#has-it-broken)) flags a
[break](../GLOSSARY.md#fitting-a-model) in its residuals. A break flagged in any instance then resets them all, so they
run one after another. On
`ewridge` over 20 features, a grid of five half-lives, 500 to 2,500, took
1.45 times as long as the half-life of 500 alone, as the README's
[Throughput](../../README.md#throughput) section measures.

**On one thread, a half-life grid inside one spec is faster than one spec
per value.** Each spec is separate work for the bank: it reads its columns
out of each [chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads),
steps its own clock and builds an output column of its own. Five half-lives (100 to
1,600) in one spec took less time than five specs of one half-life each.
Each was timed with `ModelBank.fit_predict` over 400,000 synthetic rows
of five features, on one thread (`threads=1`, so neither the instances
nor the specs ran in parallel), each the best of five runs. They ran on an
Apple M4 Pro at a load average of about 4, on 2026-11-07. The grid varies
`half_life` in each:

| model | five specs | one spec with a grid | time saved |
|---|---:|---:|---:|
| `ewridge` | 0.512 s | 0.445 s | 13% |
| `kalman` (`coef_half_life=1000` fixed, so the grid moves its standardization and residual variance) | 0.389 s | 0.325 s | 16% |
| `sgd` | 0.270 s | 0.209 s | 23% |
| `holt` (the target's level, with no features) | 0.185 s | 0.134 s | 28% |

## A slot: one prediction a spec writes

**On each row, a spec writes one [slot](../GLOSSARY.md#grids) for each
target at each grid point.** A grid point's slot holds a prediction and
its residual. A spec with three targets over the example's eight grid
points writes 24 such slots. `emit_selected` adds one slot per target,
the prediction of the grid point with the lowest
[out-of-sample](out-of-sample.md) error so
far. `emit_averaged` adds one more, a blend of every grid point's
prediction, each weighted down as its error rises above the best one's.
Those two slots hold a prediction and no residual. So the example's
eight grid points, in two instances, write eight slots of their own for
its one target, and the two switches add two more. `po.eval.metrics` and
`po.eval.grid_search` score each slot, and name it in their `slot`
column.



## Why the best error over a grid is optimistic

**The smallest of several errors measured on the same rows understates
what choosing its grid point will earn on the next rows.** Every slot's
error is out-of-sample: each prediction was made
before its row was learned. But each slot's error over a stretch of rows
is its true error plus noise, and the smallest of them belongs to the
grid point whose noise happened to run lowest. The fewer the rows and the
more grid points, the larger that gap. This code makes twenty feature
sets that differ only in which noise column they read, so none is better
than another, and chooses one on the first half of the rows:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = x0 + noise; z0 ... z19 are noise that y never reads.
rng = np.random.default_rng(3)
n = 20_000
x0, e = rng.standard_normal((2, n))
z = rng.standard_normal((20, n))
pl.DataFrame({"x0": x0, **{f"z{j}": z[j] for j in range(20)}, "y": x0 + e}).write_parquet("noise.parquet")

# Twenty feature sets, each x0 and one z: alike but for the noise each z adds.
sets = {f"z{j}": ["x0", f"z{j}"] for j in range(20)}
features = ["x0", *(f"z{j}" for j in range(20))]
spec = po.spec.ewridge("sets", targets=["y"], features=features, half_life=50.0, feature_sets=sets)
out = pl.scan_parquet("noise.parquet").online.fit_predict([spec]).collect()

# The set with the lowest error on the first half, against the average of the twenty, on each half.
first, second = po.eval.metrics(out.head(n // 2), spec), po.eval.metrics(out.tail(n // 2), spec)
pick = first.sort("mse").row(0, named=True)
later = second.filter(slot=pick["slot"])["mse"].item()
print(f"{pick['slot']} on the rows it was chosen on: mse {pick['mse']:.4f}, the average {first['mse'].mean():.4f}")
print(f"{pick['slot']} on the rows after: mse {later:.4f}, the average {second['mse'].mean():.4f}")
print(f"rank on the rows after: {(second['mse'] < later).sum() + 1} of 20")
```

```text
pred_y__z18 on the rows it was chosen on: mse 0.9933, the average 0.9953
pred_y__z18 on the rows after: mse 1.0298, the average 1.0302
rank on the rows after: 5 of 20
```

On the rows it was chosen on, `z18` beat the average of the twenty by
0.0020. On the rows after, it beat it by 0.0004, which is noise. The
first number is the optimism, and on real data it is mixed in with any
real difference between the grid points.

**So choose a grid point on one stretch of rows, and report what it
earned on the rows after that stretch.** `po.eval.grid_search` does both:
it cuts the rows into spans, chooses the best grid point from the rows
before each span, and scores the choice on the span. [Calibrating a
model](../CALIBRATION.md) gives the steps for each setting.

**Choosing as the stream runs has the same bias, and one more.**
`emit_selected` ranks every grid point by its exponentially weighted
out-of-sample error so far, each at its own half-life. So a grid point
with a short half-life is ranked on few rows, and wins its share of rows
partly by luck. On a stream whose coefficient wanders, a selection among
five half-lives held the shortest on 41% of the rows, though a longer one
predicted best over the whole stream. Use `emit_selected` where the best
grid point itself moves. To choose one grid point to run on the rows to
come, use `po.eval.grid_search`. [Choose among a grid's settings as the
stream runs](../how-to/choose-among-a-grid-online.md) shows both switches
on that stream.
