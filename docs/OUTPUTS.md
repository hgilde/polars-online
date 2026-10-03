# What each model writes

Every spec adds one column to the output, named after the spec. Its value in
each row is a record of named fields, and this page lists the fields each
model writes, one section per model. `po.spec.output_fields(spec)` answers
the same question at runtime, for the exact spec you built.

| family | models |
|---|---|
| [Linear models](../README.md#linear-models) | [`ewridge`](#ewridge) · [`rls`](#rls) · [`lasso`](#lasso) · [`kalman`](#kalman) · [`huber`](#huber) · [`quantile`](#quantile) · [`sgd`](#sgd) · [`pa`](#pa) · [`ftrl`](#ftrl) · [`holt`](#holt) |
| [Moments and correlation](../README.md#moments-and-correlation) | [`ew_cov`](#ew_cov) · [`marginal`](#marginal) · [`deco`](#deco) · [`rcov`](#rcov) |
| [Clustering and classification](../README.md#clustering-and-classification) | [`kmeans`](#kmeans) · [`micro`](#micro) · [`ew_class`](#ew_class) |
| [Sequential tests and regimes](../README.md#sequential-tests-and-regimes) | [`seqtest`](#seqtest) · [`corrchange`](#corrchange) · [`hmm`](#hmm) · [`bocpd`](#bocpd) |

[Reading this page](#reading-this-page) says how a field's name is built,
defines the four fields most models write, and says what is left out.

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

### Fields most models write

Four fields appear in nearly every table below, and are defined here once:

| field | what it holds | null |
|---|---|---|
| `weight_sum` | the accumulated weight before this row's update and before its own decay | |
| `settled_frac` | how far the decay window had filled toward steady state before this row: `1 - 2^(-T/half_life)`, with `T` the decay time seen so far, so 0.5 at one half-life and 0.75 at two. `min_settled_frac` gates on it | where nothing decays |
| `withheld_reason` | why the row's predictions are null: `below_min_settled_frac`, `below_min_weight` or `above_max_error_inflation`. That order is their precedence, so the first that applies is the one named | where nothing was withheld |
| `coef` | the numbers behind the fit, as one flat list written after the row's update: a regression's coefficients, or what a model that is not a regression keeps in their place, such as its centres or state means. Its builder's docstring lays the list out. A model that solves on a schedule (`solve_every`) shows its latest solve | on every row but those `coef_every` fills, which by default are each group's last row in each chunk; and before the model has anything to report, such as a first solve |

### What is not listed

The optional outputs are left out: the `emit_*` switches, `conformal`,
`resid_quantiles` and extra `stats`. Each is shown where the README sets it:

| the switch | where the README shows it |
|---|---|
| the residual diagnostics: `emit_sigma`, `emit_zscore`, `emit_selected`, `emit_averaged`, `emit_drift`, `emit_metrics`, `emit_autocorr`, `resid_quantiles`, `conformal` | [Per-row diagnostics](../README.md#per-row-diagnostics), for the models that predict a target |
| `emit_error_inflation`, `ewridge`'s | [Warm-up](../README.md#warm-up) |
| `emit_clocks`, every model's | [Labels that arrive late](../README.md#labels-that-arrive-late), and the table below |
| `ew_cov`'s extra `stats` | its own [section](../README.md#ew_cov--exponentially-weighted-moments) |

`emit_clocks` is the one switch every model takes, and it adds two fields:

| field | what it holds | null |
|---|---|---|
| `scored_clock` | the clock the row was scored at, in the clock column's own type; with no clock, the row's index in its group | on a skipped row |
| `learned_clock` | the clock of the newest row the model had learned from when this row was scored, in the same type: under an `embargo`, at least the delay behind `scored_clock` | on a skipped row, before the first row learned, and after a reset |

### How this page is made

`scripts/outputs_doc.py` writes this page from each model's plainest spec,
the one `tests/test_model_registry.py` builds. `tests/test_outputs_doc.py`
regenerates it and fails on any difference, so a new field cannot ship
undocumented and a removed one cannot linger. Do not edit this page by
hand: change the generator, then run
`uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.


## `ewridge`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |
| `support_coef` | on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid out like `coef`; the intercept is not a share (null) |

## `rls`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `lasso`

| field | meaning |
|---|---|
| `pred_y__l0.1` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y__l0.1` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `pred_y__l0` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y__l0` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |
| `penalty_selected_y` | the path point in force for `<t>`, by lowest EW out-of-sample error |

## `kalman`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `huber`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `quantile`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `sgd`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `pa`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `ftrl`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `holt`

| field | meaning |
|---|---|
| `pred_y` | the prediction for `<t>`, computed from the state **before** this row |
| `resid_y` | `y - pred` for `<t>`; null where the target is null, and on every row of a target that is a window expression, whose value is not known at its row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `ew_cov`

| field | meaning |
|---|---|
| `mean_x0` | EW mean of `<f>` |
| `mean_x1` | EW mean of `<f>` |
| `std_x0` | EW standard deviation of `<f>` |
| `std_x1` | EW standard deviation of `<f>` |
| `corr_x0_x1` | EW correlation of the pair |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |

## `marginal`

`marginal` writes nothing per row but `weight_sum`. Its product is the state, and `ModelBank.marginal()` reads the pairs from it.

| field | meaning |
|---|---|
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |

## `deco`

| field | meaning |
|---|---|
| `u` | the equicorrelation this row alone implies (Lemma 2.3, from the standardized row) |
| `rho` | the block's equicorrelation level, the smoothed value `u` is folded into |
| `loglik` | log predictive density of the row, under the model as it stood before the row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `rcov`

`rcov` writes nothing per row but `weight_sum`. Its product is the closed block, in the row `ModelBank.closed_groups()` gives when a group closes (`group_close`).

| field | meaning |
|---|---|
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |

## `kmeans`

| field | meaning |
|---|---|
| `cluster` | the nearest cluster's label, read before the row is learned from; null while there is none: for `kmeans`, before seeding, which waits for `max(warm_rows, k)` learned rows, and for `micro`, while no micro-cluster is established |
| `dist` | distance from the row to the centre `cluster` was read from |
| `dist2` | distance to the second-nearest centre, so `dist2 - dist` is the margin |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `micro`

| field | meaning |
|---|---|
| `cluster` | the nearest cluster's label, read before the row is learned from; null while there is none: for `kmeans`, before seeding, which waits for `max(warm_rows, k)` learned rows, and for `micro`, while no micro-cluster is established |
| `dist` | distance from the row to the centre `cluster` was read from |
| `micro_id` | id of the micro-cluster the row joins, or opens when none can take it |
| `outlier` | true when no established micro-cluster takes the row |
| `n_clusters` | macro-clusters currently linked |
| `n_micro` | live micro-clusters |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `ew_class`

| field | meaning |
|---|---|
| `class` | the most likely class label |
| `p_a` | posterior probability of class `<label>` |
| `p_b` | posterior probability of class `<label>` |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `seqtest`

| field | meaning |
|---|---|
| `log_e_pos_y` | log of the e-process betting the sign of `<t>` is positive: 0 is no evidence, and `log(20)`, 3.0, is evidence at level 0.05 |
| `log_e_neg_y` | log of the e-process betting it is negative |
| `n_pos_y` | learned rows whose `<t>` was positive |
| `n_neg_y` | learned rows whose `<t>` was negative |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |

## `corrchange`

| field | meaning |
|---|---|
| `stat` | the test statistic for the span, the pair of windows or the monitored row; null except on a row one is due |
| `crit` | the critical value `stat` is compared against |
| `flag` | true on the row where `stat` crossed `crit` |
| `since_flag` | learned rows since the last flag |
| `since_change` | on a flag, the rows since the change it dates, through the flag's row from the first changed one; null otherwise |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |

## `hmm`

| field | meaning |
|---|---|
| `filtered_0` | posterior probability of state `<j>` before this row |
| `filtered_1` | posterior probability of state `<j>` before this row |
| `predicted_0` | one-step-ahead probability of state `<j>` |
| `predicted_1` | one-step-ahead probability of state `<j>` |
| `state` | the most likely state for this row: the one with the largest `predicted_<j>` |
| `loglik` | log predictive density of the row, under the model as it stood before the row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
| `coef` | the numbers behind the fit as one list, on the rows `coef_every` fills ([shared field](#fields-most-models-write)) |

## `bocpd`

| field | meaning |
|---|---|
| `p_change` | `P(run length <= 1)`: the mass sitting on a change at or just before this row |
| `run_mode` | most likely run length *before* this row, so `t - run_mode` dates the regime |
| `run_mean` | posterior mean run length |
| `pred_x0` | the predictive mean for feature `<f>` under the fitted model |
| `pred_x1` | the predictive mean for feature `<f>` under the fitted model |
| `loglik` | log predictive density of the row, under the model as it stood before the row |
| `weight_sum` | accumulated weight before this row's update and before its own decay ([shared field](#fields-most-models-write)) |
| `settled_frac` | how far the decay window had filled before this row; null where nothing decays ([shared field](#fields-most-models-write)) |
| `withheld_reason` | why the row's predictions are null, and null where nothing was withheld ([shared field](#fields-most-models-write)) |
