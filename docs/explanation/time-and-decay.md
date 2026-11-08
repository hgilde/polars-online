<!-- kind: explanation -->
# Time and decay

This page explains, for anyone who sets `half_life` on a spec, how a model
forgets old rows along a clock, why a huge half-life is not the same as
none, and how a hard window forgets differently.

**A model forgets old rows by its decay: each row's weight halves every
`half_life` along a clock.** A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
is the description of one model, and its
[clock](../GLOSSARY.md#time) is the column, named by `clock=`, that says
how far apart two rows are. At each row, the model multiplies the weight
of everything it has learned by the decay factor

```
0.5 ** (Δclock / half_life)
```

where `Δclock` is the clock's step from the last row of the group the
model did not [skip](../GLOSSARY.md#the-model-bank-and-what-it-reads). So a
row one half-life back counts half as much as the newest, and a row three
half-lives back counts 12.5%. `lam`, the weight kept over one clock unit,
sets the same decay by its factor: `lam = 0.5 ** (1 / half_life)`. Without
a clock column, each row is one clock unit after the row before it in its
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads).

## Decay along the clock

**Every parameter that measures time acts on the clock, so it means the
same span of time however densely the rows arrive.** A burst of a hundred
rows in one second ages the fit by one second, as a single row does,
though it adds a hundred rows' weight. So `half_life`, `gap_cap`, an
[`embargo`](../GLOSSARY.md#time) (the delay before a row is learned), a
[hard window](../GLOSSARY.md#fitting-a-model)'s `window_size` and a
model's cadences, such as `solve_every`
(how often a regression solves for its coefficients), are all in [clock
units](../GLOSSARY.md#time). Only a few settings count rows, such as
`warm_rows`, the rows some models wait for before they predict. The
README's [Clock types and units](../../README.md#clock-types-and-units)
lists every clock parameter, and the value each kind of clock takes, such
as a duration like `"10m"` for a clock of timestamps.

**On a temporal or an integer clock, a model takes each step in integers,
so the clock is read exactly.** A temporal clock (a `Datetime`, `Date` or `Duration` column) is
read in its own integer nanoseconds, and its step is taken in integers
before it becomes seconds. So a nanosecond timestamp keeps its nanoseconds
however old the stream is, and the same instants stored in milliseconds,
microseconds or nanoseconds give the same numbers. A time zone changes
nothing, since Polars stores a `Datetime` in UTC, so a change of clocks
for summer time neither stretches nor folds the clock. Integer nanoseconds
limit the range: a `Datetime` or `Date` clock must lie between the years
1677 and 2262, and a `Duration` clock must stay below about 292 years.

**An integer clock is read as its integers, for the same reason.** Any
column from `Int8` to `Int64` or `UInt8` to `UInt64` is a clock of
integers. The step between two rows is taken in integers before it becomes
a double, and so is every decision on the clock: a step back, a gap past
`gap_cap`, a cadence, a hard window's edge and an embargo's release. An
`Int64` column of epoch nanoseconds therefore keeps its nanoseconds, where
a double near 1.8e18 resolves only 256 of them. A fractional parameter on
such a clock, such as `gap_cap=0.5`, is compared as the number it is.

**A clock value a [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
cannot read exactly is refused, never rounded,** since the step to every
later row is measured from it. A feature may be null and its row skipped,
but a clock value cannot be skipped. The README's [Clock types and
units](../../README.md#clock-types-and-units) lists what is refused, by row
or by name.

## The running sums that decay

**Most models keep [running sums](../GLOSSARY.md#fitting-a-model) in place
of their rows, and the decay lowers the weight behind those sums.** A regression such as
`ewridge` keeps, for example, the weighted means of its features and its
target and their centered co-moments, and solves its coefficients from
them. The decay factor multiplies a decayed sum, and the weight a mean is
taken over, which lowers the weight of every row inside it at once. So
forgetting a row needs no copy of it. The
sums come in three kinds:

| running sum | where | why |
|---|---|---|
| an exponentially weighted **mean** | most models | it stays bounded over a stream of any length |
| a decayed **sum** | the matrix `A` and vector `b` of `rls` (recursive least squares), and the gradient sums of `ftrl` (online logistic regression) and of `sgd`'s AdaGrad (a step size per coefficient, from the sum of its squared gradients) | the update rule of each model is written in sums |
| a **centered** second moment, by a weighted Welford update | wherever a model keeps second moments as means | a variance is right even when a feature sits far from zero |

**`sgd` (stochastic gradient descent) and `pa` (passive-aggressive
regression) do not decay their coefficients.** Each takes a step
toward each row from its current coefficients, so there is no sum of
coefficients to decay. Their `half_life` decays `weight_sum`, the moments
they scale the features by, and the spreads they keep, and for `sgd` under
AdaGrad its gradient sums, but not the coefficients. [What `half_life`
decays](../MODELS.md#linear-models) lists it for each linear model.

## Local fits

**Under a finite half-life, the fit of a model that keeps running sums
describes the recent past, and moves from row to row.** Under
`half_life="10m"`, the fit weighs the last few tens of minutes on the
clock most. Its final [state](../GLOSSARY.md#saving-and-serving) (what it
has learned) describes mostly the recent end of the stream, so saving that
state may be of limited value as a summary of the whole stream. In a
regression such as `ewridge`, `coef_every=0` writes the coefficients on
every row instead: a time series of what the fit found, row by row, which
step 4 of the README's [A first fit](../../README.md#a-first-fit) shows.

**The weight behind a fit levels off at a ceiling, which says how much
past a half-life keeps.** For rows one clock unit apart, each of weight 1,
`weight_sum` (the weight a row's prediction rests on) approaches
`1 / (1 − 0.5 ** (1 / half_life))`, about `1.44 · half_life` rows' worth
for a long half-life. Under `half_life=50`, that is about 72.6. The field
`settled_frac`, `1 − 0.5 ** (T / half_life)` with `T` the clock time the
model has decayed over, is the share of that ceiling such a stream has
reached: 0.5 after one half-life, 0.75 after two.
[Warm-up](../../README.md#warm-up) says how the readiness gates read it.

## Gaps, sessions and steps back

A stream from a market has quiet hours, weekends, day boundaries and
replays, and the clock's step across each is not the time the fit should
age by. The spec has a parameter for each kind. The README's
[Sessions, gaps and steps back](../../README.md#sessions-gaps-and-steps-back)
fits one regression under four ways of treating a weekend: two caps and
two treatments of the session change.

**`gap_cap` caps each step a model decays by, so one long gap does not
forget everything at once.** A clock requires it. With a half-life of
thirty minutes, a weekend of two days is about a hundred half-lives, which
leaves the rows learned before it almost no weight: the model starts over
in all but name. Under `gap_cap="5m"` the weekend ages the fit as five
minutes do. After a run of skipped rows, the step to the next row is
capped too, however long the run.

**A session column marks the boundaries where the clock's jump is not
elapsed time.** At a change of the [session](../GLOSSARY.md#time) column,
the model steps by `session_gap`, a step you choose, in place of the jump,
capped at `gap_cap` like any step. Under `session_gap="reset"` it
[resets](../GLOSSARY.md#time) there. Under `group_close="session"`, the
group closes at the change, and no step is taken.

**A model that keeps past rows by position empties them at a [clock
break](../GLOSSARY.md#time):** a row where `gap_cap` shortened the step, or
where the session changes. A lag, a feature's value some rows back, is read
from those kept rows. At a clock break the row before no longer stands for the moment
just before, so a lag read across it would pair two moments far apart.

**A model cannot learn a row earlier on the clock than the row before it
in the same group and session, so the bank refuses a step back or starts
the model over.** The running sums hold no rows to place the late row
among. And a decay over a negative step, `0.5 ** (Δclock / half_life)`
with `Δclock` below 0, is above 1: it would increase the weight of every
row learned instead of lowering it. So by default the bank refuses a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the rows it
takes in at once) in which a group's clock [steps
back](../GLOSSARY.md#time) within a session. `restart_after_step_back`
makes a step back larger than its value a reset instead. A clock that
moves back where the session changes is no step back, since
`session_gap` takes the step's place. [Why you set the row
order](chunk-invariance-and-streaming.md#why-you-set-the-row-order)
says how the bank checks the order.

## Without a decay

**With decay off (`half_life=float("inf")` or `lam=1.0`) and no hard
window, the fit of a model that solves or accumulates converges to the
batch fit over every row it has seen.** Such a model keeps running sums
and computes its fit from them, and the README's [model
table](../../README.md#models) gives each model's kind. Its running sums weigh each row by
its weight alone, as a batch method weighs the rows it is given.

**With decay off, row order does not change the fit of a model that
solves or accumulates,** because the sums it keeps are the same in any
order. A model that reweights, steps, filters or tests instead depends on
the row order whether decay is on or off. Each model's kind is in the
README's [model table](../../README.md#models), with the outputs that
depend on the order even so, such as a lag or `lasso`'s
`penalty_selected`. The settings that read the rows in sequence, such as a
hard window, a session, a weight or an embargo, are named in [Why you set the row
order](chunk-invariance-and-streaming.md#why-you-set-the-row-order), and
`ModelBank.fit`'s docstring lists them all.
Without them and with no clock column, rows fed forwards, backwards or
shuffled give the same coefficients, to rounding. With a clock column, the
bank refuses a step back, unless `restart_after_step_back` makes it a
reset.

**For no forgetting, give `half_life=float("inf")`: a huge finite
half-life still forgets, a little.** Under `half_life=1e12`, `settled_frac`
stays near zero until the stream has run for about 10¹² clock units, so a
`min_settled_frac` gate ([Warm-up](../../README.md#warm-up)) stays closed.

This code fits `ewridge` with no decay to `df`'s rows in their order,
reversed and shuffled, and solves the same least squares with numpy. The
last line reads `settled_frac` on the last row under `half_life=1e12`.
It reads `df` from the README's [Example
data](../../README.md#example-data):

```python
# Plain least squares in three orders, then numpy's.
batch = po.spec.ewridge("batch", targets=["y"], features=["x0", "x1", "x2"],
                        half_life=float("inf"),   # no clock, no decay
                        ridge=0.0)                 # plain least squares, in place of the default ridge of 1e-6
for order, rows in [("in order", df), ("reversed", df.reverse()), ("shuffled", df.sample(fraction=1.0, shuffle=True, seed=1))]:
    coef = po.ModelBank([batch]).fit_predict(rows)["batch"].struct.field("coef")[-1]   # the fit after the last row
    print(f"{order:<9}", " ".join(f"{c:9.6f}" for c in coef))
design = np.column_stack([np.ones(df.height), df["x0"], df["x1"], df["x2"]])
print(f"{'numpy':<9}", " ".join(f"{c:9.6f}" for c in np.linalg.lstsq(design, df["y"], rcond=None)[0]))

# A huge finite half-life, which is not inf.
huge = po.spec.ewridge("huge", targets=["y"], features=["x0"], clock="t", half_life=1e12, gap_cap=300.0)
print(f"settled_frac after 399 units of t under half_life=1e12: {po.ModelBank([huge]).fit_predict(df)['huge'].struct.field('settled_frac')[-1]:.1e}")
```

```text
in order   0.992088  1.994121 -1.007945 -0.026134
reversed   0.992088  1.994121 -1.007945 -0.026134
shuffled   0.992088  1.994121 -1.007945 -0.026134
numpy      0.992088  1.994121 -1.007945 -0.026134
settled_frac after 399 units of t under half_life=1e12: 2.8e-10
```

## A hard window

**A half-life never forgets a row entirely, and a [hard
window](../GLOSSARY.md#fitting-a-model) does.** Three half-lives back, a
row still carries 12.5% of its weight. Under `window_size=w`, a row `w`
clock units old or older counts for exactly nothing, under the default
`closed="right"`. Under `closed="both"`, a row exactly `w` old keeps its
weight. So a fit
forgets a change in the process completely once the hard window has
passed it. A row's age is measured on the clock the model decays by, each
step after `gap_cap` and any `session_gap`, so a capped weekend ages the
rows before it by the cap. Five models take a hard window: `ewridge`,
`lasso`, `ew_cov`, `ew_class` and `marginal`, each described in the
README's [model table](../../README.md#models). Inside it the weights still
halve every `half_life`, and `half_life=float("inf")` makes them equal.
The README's [A hard window](../../README.md#a-hard-window) gives its
settings and what each model truncates.

**A hard window forgets a break that a half-life still holds.** In `df`,
the coefficient of `ret` on `signal_a` flips from 0.8 to −0.8 at row 200
(counting from 0). This code follows it with a half-life of 100 rows, and
with a flat hard window of 100 rows. A hundred rows after the break, the
hard window holds only rows after it and reads the new coefficient, while
the half-life still gives the old rows half their weight. It reads `df`
from the README's [Example data](../../README.md#example-data):

```python
common = dict(targets=["ret"], features=["signal_a"], clock="t", gap_cap=300.0, coef_every=0)   # the coefficients on every row
decayed = po.spec.ewridge("decayed", half_life=100.0, **common)                              # a half-life of 100 rows
windowed = po.spec.ewridge("windowed", half_life=float("inf"), window_size=100.0, **common)  # the last 100 rows, equally weighted
fitted = po.ModelBank([decayed, windowed]).fit_predict(df)
print("row  decayed  windowed")
for row in [199, 250, 300, 399]:
    beta = [fitted[name].struct.field("coef")[row][1] for name in ("decayed", "windowed")]   # [intercept, signal_a]
    print(f"{row:<4} {beta[0]:7.2f}  {beta[1]:8.2f}")
```

```text
row  decayed  windowed
199     0.77      0.83
250     0.29      0.01
300     0.00     -0.78
399    -0.39     -0.78
```

**The cut is exact, because a decayed sum can be subtracted.** The rows
learned at or before a time `u` still contribute the running sum as it
stood at `u`, multiplied by every decay factor since `u`. Subtracting that
leaves exactly the rows after `u`. So the model keeps a ring of snapshots
of its running sums, one every row by default, and its hard window's edge
is the oldest snapshot inside it. A coarser spacing, set by
`snapshot_every`, discards a little more than asked, never less. A
subtraction loses precision as the share of the weight it discards grows.
The loss is negligible at a hard window of three half-lives, and grows as
the hard window shortens toward the half-life.

**Under a hard window, `weight_sum` is the weight inside it, so a
`min_weight` gate (the weight a model must reach before it predicts) waits
on a weight that levels off once the hard window is full, below the
ceiling a half-life alone reaches.** This code fits `ew_cov`
over a hard window of 150 units of `t` with a half-life of 50, and a
snapshot every 10 units. Without a hard window, the weight levels off at
its ceiling, about 72.6. A hard window keeps the share of it younger than
its edge, `1 − 0.5 ** (edge / 50)`, with the edge between 140 and 150
units back. It reads `df` from the README's [Example
data](../../README.md#example-data):

```python
# The same model with a hard window and without.
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=50.0,
    window_size=150.0,             # a row this many clock units old or older contributes exactly nothing
    snapshot_every=10,             # a snapshot every 10 units of t
    window_budget={"refuse": 64},  # at most 64 MiB of snapshots per ring
)
whole = po.spec.ew_cov("whole", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=50.0)
# Each weight_sum on the last row, against the ceiling and the hard window's share of it.
last = po.ModelBank([cut, whole]).fit_predict(df).tail(1)
ceiling = 1 / (1 - 0.5 ** (1 / 50))
print(f"no hard window: weight_sum {last['whole'].struct.field('weight_sum')[0]:.2f}, the ceiling {ceiling:.2f}")
print(f"inside the hard window: weight_sum {last['cut'].struct.field('weight_sum')[0]:.2f}, "
      f"between {ceiling * (1 - 0.5 ** (140 / 50)):.2f} and {ceiling * (1 - 0.5 ** (150 / 50)):.2f}")
```

```text
no hard window: weight_sum 72.35, the ceiling 72.64
inside the hard window: weight_sum 63.43, between 62.21 and 63.56
```
