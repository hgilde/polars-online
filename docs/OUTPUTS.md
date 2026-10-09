# What each model writes

Every spec adds one column to the output, named after the spec. Its value in
each row is a record of named fields, and this page lists the fields each
model writes, one section per model. `po.spec.output_fields(spec)` answers
the same question at runtime, for the exact spec you built.

| family | models |
|---|---|
| [Linear models](../README.md#linear-models) | [`ewridge`](#ewridge) · [`rls`](#rls) · [`lasso`](#lasso) · [`kalman`](#kalman) · [`huber`](#huber) · [`quantile`](#quantile) · [`sgd`](#sgd) · [`pa`](#pa) · [`ftrl`](#ftrl) · [`holt`](#holt) |
| [Moments and correlation](../README.md#moments-and-correlation) | [`ew_cov`](#ew_cov) · [`marginal`](#marginal) · [`deco`](#deco) · [`rcov`](#rcov) · [`audit`](#audit) |
| [Clustering and classification](../README.md#clustering-and-classification) | [`kmeans`](#kmeans) · [`micro`](#micro) · [`ew_class`](#ew_class) |
| [Sequential tests and regimes](../README.md#sequential-tests-and-regimes) | [`seqtest`](#seqtest) · [`corrchange`](#corrchange) · [`bocpd`](#bocpd) · [`hmm`](#hmm) |

Each field below comes with its dtype, what it holds, and the rows on which
it is null.

## Reading this page

### Field names

Each table lists the fields of the model's plainest spec, whose target is
`y` and whose features are `x0`, and `x1` where the model needs two. Your
own spec's fields carry your column names in their place. The names follow
the grammar in the README's [Output field
names](../README.md#output-field-names): `<stat>_<column>` for a per-column
value, `_<a>_<b>` for a pair, and a suffix where a grid makes several of the
same field.

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
| `__l<lambda>` | one value of `lasso_path` | `__l0.1`, `__l0` |
| `@h<half_life>` | one half-life of a grid, at the end of every field's name: `weight_sum@h500`, or `weight_sum@h10m` for a duration | |

EW, in a meaning below, is short for exponentially weighted.

### Field types

A field's dtype is the type the bank declares to Polars before it reads a
row, as `po.spec.output_index(spec)` lists it:

| dtype | the Polars type |
|---|---|
| `f64` | `Float64` |
| `i32` | `Int32` |
| `i64` | `Int64` |
| `bool` | `Boolean` |
| `str` | `String` |
| `list[f64]` | `List(Float64)` |
| `enum` | an `Enum` of `withheld_reason`'s three values |
| `clock` | the clock column's own type |

### When a field is null

**Every field is null on a skipped row**: a row whose features or weight
hold a null, or a value that counts as one (NaN, ±inf, or a magnitude above
`1e100`). The model learns nothing from such a row. A row of weight 0 is
not skipped: it is scored, and teaches the model nothing.

**Most of a model's own fields are also null on a withheld row**: a row
whose outputs a warm-up gate held back, which `withheld_reason` names
([Warm-up](../README.md#warm-up)). Each table's last column, *also null*,
says where a field is null besides a skipped row, and *withheld* there
means such a row.

### Fields most models write

Four fields appear in nearly every table below, and are defined here once:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | the accumulated weight before this row's update and before its own decay | never |
| `settled_frac` | `f64` | how far the decay window had filled toward steady state before this row: `1 - 2^(-T/half_life)`, with `T` the decay time seen so far, so 0.5 at one half-life and 0.75 at two. `min_settled_frac` gates on it | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null: `below_min_settled_frac`, `below_min_weight` or `above_max_error_inflation`. That order is their precedence, so the first that applies is the one named | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit, as one flat list written after the row's update: a regression's coefficients, or what a model that is not a regression keeps in their place, such as its centres or state means. Its builder's docstring lays the list out. A model that solves on a schedule (`solve_every`) shows its latest solve, and the entries of a target no solve has fit yet are null. Under an `embargo` the row's own update waits for the delay, so `coef` is written after the rows this row releases, and is the fit the next row is predicted with only when the next row releases none | on every row but those `coef_every` or `max_rows_between_coefs` fills, which with neither are each group's last accepted row in each chunk; and before the model has anything to report, such as a first solve |

### What is not listed

The optional outputs are left out: the `emit_*` switches, `conformal`,
`resid_quantiles` and extra `stats`. Each is shown where the README sets it:

| the switch | where the README shows it |
|---|---|
| the residual diagnostics: `emit_sigma`, `emit_zscore`, `emit_selected`, `emit_averaged`, `emit_drift`, `emit_metrics`, `emit_autocorr`, `resid_quantiles`, `conformal`, `emit_calibration`, `emit_breaks`, `emit_specification` | [Per-row diagnostics](../README.md#per-row-diagnostics), for the models that predict a target. `emit_metrics`' `hit_rate_<t>` is null throughout on an `sgd` fit with `loss="poisson"`, whose rate and count have no sign to hit |
| `emit_error_inflation` and `emit_se_coef`, `ewridge`'s, `rls`'s and `kalman`'s; `emit_robust_se`, `ewridge`'s and `rls`'s | [Warm-up](../README.md#warm-up) |
| `emit_clocks`, every model's | [Labels that arrive late](../README.md#labels-that-arrive-late), and the table below |
| `ew_cov`'s extra `stats` | its own [section](../README.md#ew_cov--exponentially-weighted-moments) |

`emit_clocks` is the one switch every model takes, and it adds two fields:

| field | dtype | what it holds | also null |
|---|---|---|---|
| `scored_clock` | `clock` | the clock the row was scored at, in the clock column's own type; with no clock, the row's index in its group | never |
| `learned_clock` | `clock` | the clock of the newest row the model had learned from when this row was scored, in the same type: under an `embargo`, at least the delay behind `scored_clock` | before the first row learned, and after a reset |

### How this page is made

`scripts/outputs_doc.py` writes this page from each model's plainest spec,
the one `tests/test_model_registry.py` builds. `tests/test_outputs_doc.py`
regenerates it and fails on any difference, so a new field cannot ship
undocumented and a removed one cannot linger. Do not edit this page by
hand: change the generator, then run
`uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.


## `ewridge`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

## `rls`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `lasso`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y__l0.1` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y__l0.1` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `pred_y__l0` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y__l0` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |
| `penalty_selected_y` | `f64` | the path point in force for `<t>`, by lowest EW out-of-sample error | never: it is written while the predictions are withheld too |

## `kalman`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `huber`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

## `quantile`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |
| `support_coef` | `list[f64]` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid out like `coef` | where `coef` is, and in the intercept's place, which is not a share |

## `sgd`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `pa`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `ftrl`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `holt`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `pred_y` | `f64` | the prediction for `<t>`, computed from the state **before** this row | while withheld |
| `resid_y` | `f64` | `y - pred` for `<t>` | while withheld; where the target is null; and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `ew_cov`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `mean_x0` | `f64` | EW mean of `<f>` | while withheld |
| `mean_x1` | `f64` | EW mean of `<f>` | while withheld |
| `std_x0` | `f64` | EW standard deviation of `<f>`, the square root of the population variance | while withheld |
| `std_x1` | `f64` | EW standard deviation of `<f>`, the square root of the population variance | while withheld |
| `corr_x0_x1` | `f64` | EW correlation of the pair | while withheld |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `marginal`

`marginal` writes nothing per row but `weight_sum`. Its product is the state, and `ModelBank.marginal()` reads the pairs from it.

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |

## `deco`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `u` | `f64` | the equicorrelation this row alone implies (Lemma 2.3, from the standardized row) | while withheld |
| `rho` | `f64` | the block's equicorrelation level, the smoothed value `u` is folded into | while withheld |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `rcov`

`rcov` writes nothing per row but `weight_sum`. Its product is the closed block, in the row `ModelBank.closed_groups()` gives when a group closes (`group_close`). That row's `rcov_psd_repaired` is null where the repair could not run, on an estimate with an entry that is not finite.

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |

## `audit`

`audit` writes nothing per row but `weight_sum`, the rows before this one. Its product is the state, and `ModelBank.audit()` reads what it counted from it.

| field | dtype | what it holds | also null |
|---|---|---|---|
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |

## `kmeans`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `cluster` | `i32` | the nearest cluster's label, read before the row is learned from | while withheld, and before seeding, which waits for `warm_rows` learned rows (at least `k`; by default 500, or `k` where that is more) |
| `dist` | `f64` | distance from the row to the centre `cluster` was read from | where `cluster` is |
| `dist_second` | `f64` | distance to the second-nearest centre, so `dist_second - dist` is the margin | where `cluster` is |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `micro`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `cluster` | `i64` | the nearest cluster's label, read before the row is learned from | while withheld, and while no micro-cluster is established |
| `dist` | `f64` | distance from the row to the centre `cluster` was read from | where `cluster` is |
| `micro_id` | `i64` | id of the micro-cluster the row joins, or opens when none can take it | while withheld |
| `outlier` | `bool` | true when no established micro-cluster takes the row | while withheld |
| `n_clusters` | `i32` | macro-clusters currently linked | while withheld |
| `n_micro` | `i32` | live micro-clusters | while withheld |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `ew_class`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `class` | `str` | the most likely class label | while withheld |
| `p_a` | `f64` | posterior probability of class `<label>` | while withheld |
| `p_b` | `f64` | posterior probability of class `<label>` | while withheld |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |

## `seqtest`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `log_e_pos_y` | `f64` | log of the e-process betting the sign of `<t>` is positive: 0 is no evidence, and `log(20)`, 3.0, is evidence at level 0.05 | while withheld |
| `log_e_neg_y` | `f64` | log of the e-process betting it is negative | while withheld |
| `n_pos_y` | `i64` | learned rows whose `<t>` was positive | while withheld |
| `n_neg_y` | `i64` | learned rows whose `<t>` was negative | while withheld |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `corrchange`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `stat` | `f64` | the test statistic for the span, the pair of windows or the monitored row | while withheld, and on every row but those a test is due on |
| `crit` | `f64` | the critical value `stat` is compared against | where `stat` is |
| `flag` | `bool` | true on the row where `stat` crossed `crit` | where `stat` is |
| `since_flag` | `i64` | learned rows since the last flag | where `stat` is |
| `since_change` | `i64` | on a flag, the rows since the change it dates, through the flag's row from the first changed one | on every row but a flag's |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `bocpd`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `p_change` | `f64` | `P(run length <= 1)`: the mass sitting on a change at or just before this row | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `run_mode` | `i64` | most likely run length *before* this row, so `t - run_mode` dates the regime | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `run_mean` | `f64` | posterior mean run length | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `pred_x0` | `f64` | the predictive mean for feature `<f>` under the fitted model | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `pred_x1` | `f64` | the predictive mean for feature `<f>` under the fitted model | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld, and on the first `warm_rows` learned rows (by default the feature count plus 2) where `prior_mean` or `prior_scale` is left out, which set it |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |

## `hmm`

| field | dtype | what it holds | also null |
|---|---|---|---|
| `filtered_0` | `f64` | posterior probability of state `<j>` before this row | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `filtered_1` | `f64` | posterior probability of state `<j>` before this row | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `predicted_0` | `f64` | one-step-ahead probability of state `<j>` | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `predicted_1` | `f64` | one-step-ahead probability of state `<j>` | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `state` | `i32` | the most likely state for this row: the one with the largest `predicted_<j>` | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `loglik` | `f64` | log predictive density of the row, under the model as it stood before the row | while withheld, and until `warm_rows` learned rows (default 50) have seeded the states |
| `weight_sum` | `f64` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) | never |
| `settled_frac` | `f64` | how far the decay window had filled before this row ([shared field](#fields-most-models-write)) | where nothing decays |
| `withheld_reason` | `enum` | why the row's predictions are null ([shared field](#fields-most-models-write)) | where nothing was withheld |
| `coef` | `list[f64]` | the numbers behind the fit as one list ([shared field](#fields-most-models-write)) | on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the model has anything to report |
