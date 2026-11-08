<!-- kind: reference -->
# What each model writes

The fields each model writes into its output column, one section per model,
for a reader who wants to know what a field holds, its type, and when it is
null. Every spec adds one column to the output, named after the spec, and its
value in each row is a record of these fields. `po.spec.output_fields(spec)` answers
the same question at runtime, for the exact spec you built.

| family | models |
|---|---|
| [Linear models](MODELS.md#linear-models) | [`ewridge`](#ewridge) · [`rls`](#rls) · [`lasso`](#lasso) · [`kalman`](#kalman) · [`huber`](#huber) · [`quantile`](#quantile) · [`sgd`](#sgd) · [`pa`](#pa) · [`ftrl`](#ftrl) · [`holt`](#holt) |
| [Moments and correlation](MODELS.md#moments-and-correlation) | [`ew_cov`](#ew_cov) · [`marginal`](#marginal) · [`deco`](#deco) · [`rcov`](#rcov) · [`audit`](#audit) |
| [Clustering and classification](MODELS.md#clustering-and-classification) | [`kmeans`](#kmeans) · [`micro`](#micro) · [`ew_class`](#ew_class) |
| [Sequential tests and regimes](MODELS.md#sequential-tests-and-regimes) | [`seqtest`](#seqtest) · [`corrchange`](#corrchange) · [`bocpd`](#bocpd) · [`hmm`](#hmm) |

## Reading this page

Each model's table lists the fields of its plainest spec, each with its
dtype, what it holds, and the rows on which it is null.

### The fields of your own spec

A spec of your own can carry a grid, several targets or another name, and
each changes its fields' names. Ask the spec itself:
`po.spec.output_fields` lists every field it writes, and
`po.spec.output_index` gives each field with the target, ridge and
half-life its name encodes, and its dtype. Here a spec with two half-lives
and two ridge values has four predictions of `y`, one per pair:

```python
import polars_online as po

grid = po.spec.ewridge(
    "ridge", targets=["y"], features=["x0", "x1"],
    half_life=[100.0, 500.0], ridge=[1e-6, 0.5],  # a grid: one fit per pair of values
)
print(po.spec.output_fields(grid))  # every field, by name
index = po.spec.output_index(grid)  # every field, with the values its name encodes
for row in index.filter(kind="pred").select("field", "target", "ridge", "half_life").iter_rows():
    print(*row)
```

```text
['pred_y__r0.000001@h100', 'resid_y__r0.000001@h100', 'pred_y__r0.5@h100', 'resid_y__r0.5@h100', 'weight_sum@h100', 'settled_frac@h100', 'withheld_reason@h100', 'coef@h100', 'support_coef@h100', 'pred_y__r0.000001@h500', 'resid_y__r0.000001@h500', 'pred_y__r0.5@h500', 'resid_y__r0.5@h500', 'weight_sum@h500', 'settled_frac@h500', 'withheld_reason@h500', 'coef@h500', 'support_coef@h500']
pred_y__r0.000001@h100 y 1e-06 100.0
pred_y__r0.5@h100 y 0.5 100.0
pred_y__r0.000001@h500 y 1e-06 500.0
pred_y__r0.5@h500 y 0.5 500.0
```

### Field names

Each table lists the fields of the model's plainest spec, whose target is
`y` and whose features are `x0`, and `x1` where the model needs two. Your
own spec's fields carry your column names in their place.

**Every name follows one grammar, which is stable.** Read left to right, a
name is the kind (`pred`, `abs_resid`, `lag_corr`), then its qualifier, then
the target or the columns it is about (`_<t>`, `_<f>`, `_<a>_<b>`), then the
grid point after `__` and the instance after `@`. An instance is the part of
the model that fits one half-life of a grid. A qualifier is a number
glued to the letter that says what it is: `q` a quantile level, `l` a lag
(`abs_resid_q0.05_y`, `mahal_q0.99`, `lag_corr_l1_x0_x1`). The grid point
uses letters too, `r` a ridge and `p` a lasso penalty, and the instance `h`,
a half-life. A thing the spec has several of is numbered where it is named,
glued to its word: a principal component `pc0` (`pc0_var`), an `hmm` state
`state0` (`filtered_state0`), a `kmeans` center `cluster0`
(`coef_cluster0_x1`). The README's [Output field
names](../README.md#output-field-names) builds a grid and reads its fields.

| in a name | stands for | in this page's tables |
|---|---|---|
| `<t>` | a target: its column, its `name`, or a window target's alias | `y` |
| `<f>` | a feature | `x0`, `x1` |
| `<a>`, `<b>` | the two columns of a pair | `x0`, `x1` |
| `<j>` | a state's index | `0`, `1` |
| `<label>` | a class label | `a`, `b` |
| `__r<ridge>` | one value of a `ridge` grid | |
| `__<set>` | one of the `feature_sets`, with one ridge value | |
| `__<set>_r<ridge>` | one of the `feature_sets`, with one value of a `ridge` grid | |
| `__p<penalty>` | one value of `penalties` | `__p0.1`, `__p0` |
| `@h<half_life>` | one half-life of a grid, at the end of every field's name: `weight_sum@h500`, or `weight_sum@h10m` for a duration | |

EW, in a meaning below, is short for exponentially weighted.

### Field types

A field's dtype is the type the bank declares to Polars before it reads a
row, as `po.spec.output_index(spec)` lists it:

| dtype | the Polars type |
|---|---|
| `f64` | `Float64` |
| `i64` | `Int64` |
| `bool` | `Boolean` |
| `list[f64]` | `List(Float64)`: `coef` and the fields laid out like it. A list stores nothing for a null row, and `coef` is null on most rows. For an `(n, k)` matrix, `k` the height of `po.spec.coef_index(spec)`, keep the rows that carry it and fix their width. For the output `out` of a spec named `m`: `out.select(pl.col("m").struct.field("coef")).filter(pl.col("coef").is_not_null()).select(pl.col("coef").list.to_array(k)).to_series().to_numpy()` |
| `enum` | an `Enum` of its values, in their order: `withheld_reason`'s six, an `ew_class` spec's `classes` for `class`, a grid's slots for `selected_<t>` |
| `clock` | the clock column's own type, an integer clock in its own width; with no clock, `Int64`, the group's row index |

**Every integer is an `Int64`**, in a field and in every frame the bank and
`po.spec` return: a count, a state, a cluster label, a position, a row's
index. It is Polars' own default integer, so a join, a concatenation or a
comparison between two frames never meets two widths. Only the clock values
differ, in a field or in a frame such as `groups()`' `clock_last`: an
integer clock keeps its own width.

### When a field is null

**Every field is null on a skipped row**: a row whose features or weight
hold a null, or a value that counts as one (NaN, ±inf, or a magnitude above
`1e100`). The model learns nothing from such a row. `audit` skips no row:
it reads every row and counts what it cannot use. A row of weight 0 is
not skipped: it is predicted, and the model learns nothing from it.

**Most of a model's own fields are also null on a withheld row**: a row
whose outputs a gate held back, or whose fit cannot be trusted there.
`withheld_reason` names which ([Warm-up](../README.md#warm-up)). Each
table's last column, *also null*, says where a field is null besides a
skipped row, and *withheld* there means such a row.

### Fields most models write

Four fields appear in nearly every table below, and are defined here once:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | the accumulated weight before the model learns this row, and before the row's own decay | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row: `1 - 2^(-T/half_life)`, with `T` the decay time seen so far, so 0.5 at one half-life and 0.75 at two. `min_settled_frac` gates on it | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null: `below_min_settled_frac`, `below_min_weight`, `above_max_error_inflation`, `below_warm_rows`, `ill_conditioned` or `unstable_rate`. That order is their precedence, so the first that applies is the one named. `below_warm_rows` is the model's own warm-up, the rows `kmeans`, `hmm` and `bocpd` hold to seed themselves from, and the rows after a target's first while `kalman` sizes its prior (its second and third observed rows under the default `standardize` without `obs_var`). `ill_conditioned` is an `ewridge` fit whose normal equations are too nearly singular for a solve to keep a digit. `unstable_rate` is an `sgd` target whose learning rate is past its stability bound: its recent steps' gains have a mean square above twice their mean, so each step grows its own row's residual and the fit diverges | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit, as one flat list written after the row is learned: a regression's coefficients, or what a model that is not a regression keeps in their place, such as its centers or state means. Its builder's docstring lays the list out. A model that solves on a schedule (`solve_every`) shows its latest solve. The entries of a target or class with no fit yet are null: a target no solve has fit, a `holt` target not yet observed, an `ew_class` class not yet seen. `hmm` writes zeros, not nulls, on its seeding rows before the last. `ewridge` does not run a solve that comes due while its weight is too far below the floor for any row to read it before the next one, so there it shows the last fit solved. Under an `embargo`, learning the row waits for the delay, so `coef` is written after the rows this row releases, and is the fit the next row is predicted with only when the next row releases none | on every row but those `coef_every` or `max_rows_between_coefs` fills, which with neither are each group's last row not skipped in each chunk; on an `ill_conditioned` row; and before the model has anything to report, such as a first solve |

### What is not listed

The optional outputs are left out of the tables: the `emit_*` switches (all
but `emit_clocks` and `huber`'s `emit_huber_scale`, whose fields are listed),
`conformal`, `resid_quantiles`, and the options that add or rename a model's
fields, such as `ew_cov`'s `stats`. Each optional output is shown where
the docs set it:

| the switch | where the docs show it |
|---|---|
| the diagnostics: `emit_sigma`, `emit_zscore`, `emit_selected`, `emit_averaged`, `emit_drift`, `emit_metrics`, `emit_autocorr`, `resid_quantiles`, `conformal`, `emit_calibration`, `emit_breaks`, `emit_specification`, `emit_tails`, `emit_influence` (`ewridge`'s, `rls`'s and `kalman`'s), `emit_feature_health` (not `holt`'s, which has no features) | [Is the model working?](DIAGNOSTICS.md), for the models that predict a target. `emit_metrics`' `hit_rate_<t>` is null throughout on an `sgd` fit with `loss="poisson"`, whose rate and count have no sign to hit |
| `emit_error_inflation` and `emit_se_coef`, `ewridge`'s, `rls`'s and `kalman`'s | [Warm-up](../README.md#warm-up) |
| `emit_robust_se`, `ewridge`'s and `rls`'s | [Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted) |
| `emit_clocks`, every model's | [Learn targets known late](how-to/learn-targets-known-late.md#check-that-it-worked), and the table below |
| `emit_huber_scale`, `huber`'s: `huber_scale_<t>`, the scale each row's cut is drawn in | [`huber`'s table](#huber), and its [entry](MODELS.md#huber--robust-regression) |
| `ew_cov`'s extra `stats`, its `lags` (`lag_corr_l<l>_<a>_<b>`), `pca` (`pc<j>_*`) and `mahal_quantiles` (`mahal_q<p>`) | its own [section](MODELS.md#ew_cov--exponentially-weighted-moments) and its [docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) |
| `seqtest`'s `a` and `b`, which replace its four fields with `log_e_a_<t>`, `log_e_b_<t>`, `wins_a_<t>` and `wins_b_<t>` | its own [section](MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) and its [docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) |
| `deco`'s named blocks, which turn `u` and `rho` into one field per block and pair of blocks | its [docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) |

`huber`'s `emit_huber_scale` adds a field listed under `huber`'s table.
`emit_clocks` is the one switch
every model takes, and it adds two fields:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `scored_clock` | `clock` | the clock the row was predicted at, in the clock column's own type; with no clock, the row's index in its group | never |
| `learned_clock` | `clock` | the clock of the newest row the model had learned from at a positive weight when this row was predicted, in the same type: under an `embargo`, at least the delay behind `scored_clock` | before the first row learned, and after a reset |

### How this page is made

`scripts/outputs_doc.py` writes this page from each model's plainest spec,
the one `tests/test_model_registry.py` builds. `tests/test_outputs_doc.py`
regenerates it and fails on any difference, so a new field cannot ship
undocumented and a removed one cannot linger. Do not edit this page by
hand: change the generator, then run
`uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.


## `ewridge`

The fields of `po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), on an `ill_conditioned` row, and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, with `S` the system the solve inverts, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

## `rls`

The fields of `po.spec.rls("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `lasso`

The fields of `po.spec.lasso("m", targets=["y"], features=["x0"], half_life=50.0, penalties=[0.1, 0.0])`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y__p0.1` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y__p0.1` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `pred_y__p0` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y__p0` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |
| `penalty_selected_y` | `f64` | the penalty of the grid point in force for `<t>` before this row, the one with the lowest EW out-of-sample error | never: it is written while the predictions are withheld too |

## `kalman`

The fields of `po.spec.kalman("m", targets=["y"], features=["x0"], half_life=50.0, coef_half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld, and on a target's first row, which has nothing to predict from |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `huber`

The fields of `po.spec.huber("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, with `S` the system the solve inverts, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

Under `emit_huber_scale=True`, it also writes:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `huber_scale_y` | `f64` | the scale the row's cut for `<t>` is drawn in | until the target has a scale above 0 |

## `quantile`

The fields of `po.spec.quantile("m", targets=["y"], features=["x0"], half_life=50.0, quantile=0.5)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, with `S` the system the solve inverts, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

## `sgd`

The fields of `po.spec.sgd("m", targets=["y"], features=["x0"], half_life=50.0, learning_rate=0.01)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `pa`

The fields of `po.spec.pa("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `ftrl`

The fields of `po.spec.ftrl("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `holt`

The fields of `po.spec.holt("m", targets=["y"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld, and on a target's first observation, which has nothing to predict from |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a formula target, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `ew_cov`

The fields of `po.spec.ew_cov("m", features=["x0", "x1"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `mean_x0` | `f64` | EW mean of `<f>` | while withheld |
| `mean_x1` | `f64` | EW mean of `<f>` | while withheld |
| `std_x0` | `f64` | EW standard deviation of `<f>`, the square root of its EW variance | while withheld |
| `std_x1` | `f64` | EW standard deviation of `<f>`, the square root of its EW variance | while withheld |
| `corr_x0_x1` | `f64` | EW correlation of the pair | while withheld, and where either column's EW variance is 0 |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `marginal`

`marginal` writes nothing per row but `weight_sum`. Its product is the state, and `ModelBank.marginal()` reads the pairs from it.

The fields of `po.spec.marginal("m", targets=["y"], features=["x0"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |

## `deco`

The fields of `po.spec.deco("m", features=["x0", "x1"], half_life=50.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `u` | `f64` | the equicorrelation this row alone implies (Engle and Kelly's Lemma 2.3, from the standardized row) | while withheld, where fewer than two of the block's columns have a positive variance before the row, and where every standardized value is 0 |
| `rho` | `f64` | the block's equicorrelation level, the smoothed value `u` is folded into | while withheld, and until the row after the first `u` |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld, where any column has no positive variance before the row, and where the levels hold no correlation matrix the density can be solved for |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `rcov`

`rcov` writes nothing per row but `weight_sum`. Its product is the closed block, in the row `ModelBank.closed_groups()` gives when a group closes (`group_close`). That row's `rcov_psd_repaired` is null where the repair could not run: on an estimate with an entry that is not finite, or where its eigendecomposition failed.

The fields of `po.spec.rcov("m", features=["x0", "x1"], group="g", group_close="monotone", block_rows=100)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |

## `audit`

`audit` writes nothing per row but `weight_sum`, the rows before this one. Its product is the state, and `ModelBank.audit()` reads what it counted from it.

The fields of `po.spec.audit("m", columns=["x0", "x1"])`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |

## `kmeans`

The fields of `po.spec.kmeans("m", features=["x0", "x1"], half_life=50.0, n_clusters=2)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `cluster` | `i64` | the nearest cluster's label, read before the row is learned from | while withheld, which includes the rows before seeding: seeding waits for `warm_rows` learned rows (at least `n_clusters`; by default 500, or `n_clusters` where that is more), and under `init="first"` past them until `n_clusters` distinct rows have come, holding at most `max(warm_rows, 1000)` rows |
| `dist` | `f64` | distance from the row to the center `cluster` was read from, in EW standard deviations under `standardize` and in the features' own units otherwise | where `cluster` is |
| `dist_second` | `f64` | distance to the second-nearest center, so `dist_second - dist` is the margin | where `cluster` is, and throughout when `n_clusters` is 1 |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `micro`

The fields of `po.spec.micro("m", features=["x0", "x1"], half_life=50.0, eps=0.3)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `cluster` | `i64` | the nearest cluster's label, read before the row is learned from | while withheld, and while no summary is established |
| `dist` | `f64` | distance from the row to the center `cluster` was read from, in EW standard deviations under `standardize` and in the features' own units otherwise | where `cluster` is |
| `micro_id` | `i64` | id of the summary the row joins, or the one it opens when none can take it | while withheld |
| `outlier` | `bool` | true when no established summary takes the row | while withheld |
| `n_clusters` | `i64` | clusters currently linked | while withheld |
| `n_micro` | `i64` | live summaries | while withheld |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `ew_class`

The fields of `po.spec.ew_class("m", features=["x0"], half_life=50.0, label="y", classes=["a", "b"], precision_prior=1.0)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `class` | `enum` | the most likely class label | while withheld, before any row has carried a class, and on a row whose class covariance cannot be factorized |
| `p_a` | `f64` | posterior probability of class `<label>` | while withheld, before any row has carried a class, and on a row whose class covariance cannot be factorized |
| `p_b` | `f64` | posterior probability of class `<label>` | while withheld, before any row has carried a class, and on a row whose class covariance cannot be factorized |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |

## `seqtest`

The fields of `po.spec.seqtest("m", targets=["y"])`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `log_e_pos_y` | `f64` | log of the e-process betting the sign of `<t>` is positive: 0 is no evidence, and `log(20)`, about 3.0, is evidence at level 0.05 | while withheld |
| `log_e_neg_y` | `f64` | log of the e-process betting it is negative | while withheld |
| `n_pos_y` | `i64` | learned rows whose `<t>` was positive | while withheld |
| `n_neg_y` | `i64` | learned rows whose `<t>` was negative | while withheld |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `corrchange`

The fields of `po.spec.corrchange("m", features=["x0", "x1"], window_rows=20)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `stat` | `f64` | the test statistic, this row included, for the span, the pair of windows or the monitored row | while withheld, and on every row but those a test is due on |
| `crit` | `f64` | the critical value `stat` is compared against | where `stat` is |
| `flag` | `bool` | true on the row where `stat` crossed `crit` | where `stat` is |
| `since_flag` | `i64` | learned rows since the last flag | where `stat` is |
| `since_change` | `i64` | on a flag, the rows since the change it dates, through the flag's row from the first changed one | on every row but a flag's; and on a `"sequential"` flag of the test between pairs (without `scalar`) that has fewer than two monitored rows before it, or where either column of the flagged pair is constant over those rows |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `bocpd`

The fields of `po.spec.bocpd("m", features=["x0", "x1"])`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `p_change` | `f64` | `P(run length <= 1)` once this row's evidence is in: the mass sitting on a change at or just before this row, the one `bocpd` field read after the row | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `run_mode` | `i64` | most likely run length *before* this row, so the current run began `run_mode` rows before this one | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `run_mean` | `f64` | posterior mean run length | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `pred_x0` | `f64` | the predictive mean for feature `<f>` under the fitted model | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `pred_x1` | `f64` | the predictive mean for feature `<f>` under the fitted model | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld, which includes the first `warm_rows` learned rows (by default the feature count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior left out; and on a row whose predictive cannot be evaluated (counted by `ModelBank.solve_failures()`) |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `hmm`

The fields of `po.spec.hmm("m", features=["x0", "x1"], half_life=50.0, n_components=2, precision_prior=0.1)`:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `filtered_state0` | `f64` | posterior probability of state `<j>` before this row | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `filtered_state1` | `f64` | posterior probability of state `<j>` before this row | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `predicted_state0` | `f64` | one-step-ahead probability of state `<j>` | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `predicted_state1` | `f64` | one-step-ahead probability of state `<j>` | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `state` | `i64` | the most likely state for this row: the one with the largest `predicted_state<j>` | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld, which includes the `warm_rows` learned rows (by default 1,000, or `n_components` where that is more) that seed the states |
| `weight_sum` | `f64` | accumulated weight before the model learns this row, and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the model had come toward its steady state under decay before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each group's last row not skipped in each chunk), and before the model has anything to report |
