<!-- kind: reference -->
# Glossary

This reference page gives the one word polars-online uses for each concept,
for anyone who reads or writes the library's docs, docstrings and messages
(the errors it raises and the warnings it issues). When a page uses a word
without defining it, look the word up here. When you write a sentence, a
docstring or a message about the library, choose the word here.

In the examples, `po` is the package, imported as
`import polars_online as po`, and `pl` is Polars, imported as
`import polars as pl`.

**Use the words of the topic tables on every page**, from [The model bank
and what it reads](#the-model-bank-and-what-it-reads) to [Saving and
serving](#saving-and-serving), since a reader takes a word varied for style
for a different thing. Every page means the README, `llms.txt`,
`CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, every guide and
map under `docs/`, the API reference's own pages under `docs/reference/`,
the docstrings and the messages. The guides include the guides for users,
which [docs/README.md](README.md#guides) lists under *Using the library*,
and the contributor guides, which it lists under *Changing the library* and
*Writing the documentation*, with `CONTRIBUTING.md`, `SECURITY.md` and
`CODE_OF_CONDUCT.md`. Two kinds of document keep their own words and are
not pages in this sense: `CHANGELOG.md`, and the records, the documents
[docs/README.md](README.md#records) lists under *Records*.

**Keep each word of a *replaces* column off every page, in the sense its
row retires.** A reader who meets both words looks for two concepts. The
uses its row keeps stay.

Each number in a *replaces* column counts a word's uses in the sense its
row retires, in the README, the guides for users, the public docstrings and
the messages, in the prose and the comments of code examples. The numbers
under [Why the contested words were
chosen](#why-the-contested-words-were-chosen) count the same texts. Search
the docs for a word with no number before you take it for unused.

**A retired word, or a word [kept out of user-facing
text](#words-kept-out-of-user-facing-text), may stay in a few places.** A
name in the code may appear in backticks, spelled as the code spells it: a
parameter, field, column, class, method, function, module, crate or file,
as `lam`, `settled_frac`, `fit_predict_batches` and `RUNNER.md` do. A code
example calls Polars' own functions and methods, such as `pl.scan_parquet`,
`collect` and `collect_batches`. A sentence that tells the reader to call
one names it in backticks, with what it does beside it: *end the query with
`sink_parquet`, which writes the result to a file without holding it all in
memory*. Where a word's row says it is a word of the code, a contributor
guide may call that part of the code by it, such as the *runner* or an
*accumulator*. A quotation keeps its source's words. A guide that names a
retired word as an example of what to replace may print it. A heading keeps
a retired word until every link to it is updated, since a heading's text is
its anchor.

**In a message, use only the words of the topic tables, as their *in a
message* column allows.** Say what any other word of the library means
instead, since a reader meets a message with no documentation open. A name
in the code that the reader gave, such as a parameter they set and the
value it got, may always appear, spelled as the code spells it. A message may use a word of the topic tables
in one of four ways:

| value | what it allows |
|---|---|
| yes | the word, in a sentence |
| yes, glossed | the word, with the gloss its row gives, since a message cannot assume the reader has met it |
| as `name` | only the parameter, field or column of that name, spelled as the code spells it, and never the word in its everyday sense |
| as `name`, if set | as `name`, in a message about a [spec](#the-model-bank-and-what-it-reads) on which the reader set that parameter; otherwise, the parameter the reader did set, such as `half_life` |

## The model bank and what it reads

A model bank reads its rows in order, and each spec names the columns its model reads.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **spec** | the description of a model, which a bank fits for each group: which model it is, the columns it reads and how it treats time. `po.spec.ewridge("m", ...)` builds one, and its name, `m`, names its output column | *specification* (4), which is kept for the specification test (`emit_specification`), the [diagnostic](#fitting-a-model) that tests for structure the model's form misses; *model spec*; *configuration* of a model | yes |
| **model bank**, then **bank** | a set of models, one for each spec and [group](#the-model-bank-and-what-it-reads), fitted together over the same rows, and held by the class `ModelBank`. Say *model bank* at a page's first use and *bank* after it. Say *a bank that starts with nothing learned* for a new bank | *fresh bank* (8), since *fresh* does not say what the bank lacks | yes |
| **stream** | the rows a bank reads, in the order it reads them, however many there are | *source* (128) for the rows themselves. `source` stays where the API uses it, as a parameter of `po.eval.grid_search` and other calls, and in `ConsumedSourceWarning`. There it names what a call reads from, such as a query, a frame, files or a saved state. *Stream* for the rows of one model: say *the model's rows* | yes |
| **query** | a Polars `LazyFrame`: a chain of steps that runs only when you ask for its result | *plan* (110), a word of Polars' own internals; *LazyFrame* in a sentence, where it is not naming the type | yes |
| **expression** | a Polars expression, a column computation such as `pl.col("x") + 1`. A query's steps take expressions, and a [window operator](#fitting-a-model) returns one. Gloss the word at its first use on a page or in a message: *an expression (a column computation, such as `pl.col("x") + 1`)* | | yes, glossed |
| **frame** | a Polars `DataFrame`, whose rows are already in memory. A parameter named `frame`, such as `po.fit_predict`'s, takes a query or a frame | *DataFrame* in a sentence, where it is not naming the type | yes |
| **chunk** | the rows a bank, or a run of `po.stream`, takes in at once: `chunk_size` rows of a query or a frame (100,000 by default), or each frame of an iterator. A bank takes a frame given without `chunk_size` whole | *batch* (109), kept only in the name `fit_predict_batches`, in the parameter `batches` of `fit` and `fit_predict_batches`, and in code that calls Polars' `collect_batches` | yes |
| **target** | a column a model predicts, or a [formula target](#fitting-a-model), named in `targets=` | *outcome* (21), *response* (17), *label* where it means a target (12, as in *labels that arrive late*). *Label* stays in the API's names: `ew_class`'s `label=`, the column of its categories; `label_clock`, the column of the time each row's target becomes known; `bank.summary()`'s `labels_missing` and `labels_dropped`; and the `label=` of `po.gbdt.xgboost_matrix` and `po.gbdt.lightgbm_dataset`, XGBoost's and LightGBM's word | yes |
| **feature** | a column a model reads to predict a target, named in `features=` | *signal* (10), *regressor* (7), *predictor* (1), *covariate* | yes |
| **weight** | how much a row counts in a fit: the value in the column `weight=` names, or 1 without one. In a model that keeps [running sums](#fitting-a-model), a row of weight 2 moves the fit as two rows of weight 1 at the same clock value do. Kish's effective sample size (`n_kish`), which the `max_error_inflation` gate of `ewridge`, `rls` and `lasso` reads, counts it as one row. A row of weight 0 is predicted and moves the [clock](#time), but the model learns nothing from it. Once learned, a row's weight decays along the clock (see [decay](#time)) | | yes |
| **group**, **group key** | the rows that share one value of the column `group=` names. The bank fits a separate model for each group, and the value is its group key, as Polars calls it | *partition* (3), since the API says `group=`; *key* alone, for a group's value | yes |
| **null** | Polars' marker for an empty entry in a column, the one `is_null()` finds. NaN, ±inf and any magnitude above `1e100` count as null in a feature, a target or a weight. A null target leaves the row predicted and the clock moving, but the model learns nothing about that target from the row. A model that still learns something from such a row, such as its features' means, says so in its docstring | *missing value* | yes |
| **skipped row** | a row with a null in any feature or in its weight: the model neither predicts it nor learns from it, so every field of its output is null. The model takes no clock step at it: the step at the next row it learns is measured from the last row it did not skip. `bank.summary()` counts these rows in `rows_skipped` | | yes |

## Time

Every parameter that measures time does so in clock units. A parameter that counts rows, such as `warm_rows`, counts rows whatever the clock. A model's decay runs on the clock.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **clock** | the column, named by `clock=`, that says how far apart two rows are. Without one, each row is one clock unit after the previous row of its group | *timestamp column*; *time column* (1), as names for the clock, since the API says `clock=`. *Timestamp column* stays as a description of a clock that holds times, as in *a clock (timestamp column `ts`)* | yes |
| **clock units** | the unit of every parameter measured on the clock: the clock column's own numbers, a duration such as `"5m"` when the clock holds times, or rows of the group without a clock | | yes |
| **decay** | how a model forgets old rows. A model that keeps [running sums](#fitting-a-model) multiplies, at each row, the weight of every row it has learned by `0.5 ** (Δclock / half_life)`. `Δclock` is the clock's step since the last row of its group the model did not skip, capped at `gap_cap`. At a [session](#time) change, `session_gap` takes the step's place, capped the same way. A model whose coefficients move by a step at each row, such as `sgd` and `pa`, decays only its weight and the scales it keeps, such as its feature scaler, not its coefficients | *forgetting factor* (1), since the API sets decay by `half_life` or `lam`. *Forget* stays an ordinary verb in an explanation, not a second name | yes |
| **`gap_cap`** | the largest clock step a model decays by, required with a clock. A longer step, such as a weekend, decays the learned rows as a step of `gap_cap` does, so one gap does not forget everything at once | | as `gap_cap` |
| **half-life** | the distance on the clock over which a learned row's weight halves, set by `half_life=`. `half_life=float("inf")` turns decay off | *halflife* (54), an old spelling of the parameter | yes |
| **`lam`** | the parameter that sets decay by its factor, in place of `half_life`: over each clock unit, a learned row's weight is multiplied by `lam`, and `lam = 0.5 ** (1 / half_life)` | *λ* outside a docstring's update equation; *lam* as a bare word in a sentence (write `lam`); *lambda* for the decay (55), since *lambda* is also the lasso's penalty | as `lam`, if set |
| **session** | a column, named by `session=`, whose value changes where the stream crosses a boundary such as the start of a trading day. Unless `group_close="session"` closes the span there, the clock moves at the change by `session_gap` in place of its real jump, or, under `session_gap="reset"`, the model resets. A group's rows between two changes of its session, or before the first change or after the last, are a *session span*. Under `group_close="session"`, each span ends as a [closed group](#saving-and-serving) | | yes |
| **step back** | a row whose clock is earlier than the previous row's in the same group and session. If `restart_after_step_back` is set and the step is larger than it, the model resets. `bank.summary()` counts the step in `clock_backwards`. If `restart_after_step_back` is not set, or the step is no larger than it, the bank refuses the chunk before learning any of it, with an error that names the row. A step back never stops `predict`, which learns nothing | *goes backwards* (24) in a sentence; *out of order* (12) for a clock step. *Out of order* stays for a query whose row order Polars does not promise, the case `OrderNotGuaranteedWarning` reports | yes |
| **reset** | a model starting over with nothing learned. It resets at a session change under `session_gap="reset"`, and at a step back larger than `restart_after_step_back` where that is set, and `bank.summary()` counts these in `resets`. A model may also reset where a detector of its own flags a change under a parameter such as `drift_action="reset"`, as its docstring says | *restart* (50), kept only in the name `restart_after_step_back` | yes |
| **clock break** | a row where `gap_cap` shortened the clock step, or a row at a session change. A model that keeps past rows by position, as a lag does, empties them there, since the row before a clock break does not stand for the moment immediately before it | *break* alone for this, since the API's `emit_breaks` and `break_wald` give *break* to a change in the process a model fits | yes |
| **embargo** | the delay on the clock between predicting a row and learning from it. It is for a target whose value is known only some time after the row. Under `embargo="5m"`, the model learns each row five minutes after predicting it. Where each row's target is known at a time of its own, `label_clock` gives that time row by row in place of one delay | *label delay* | yes |

## Fitting a model

A model predicts each row before it learns it. Most models keep only running sums of what they have learned.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **model** | one of the kinds of model a spec can name, such as `ewridge` or `kmeans`. Also the model a bank keeps for one spec and one [group](#the-model-bank-and-what-it-reads). That model holds one [instance](#grids) (the part of the model that fits one half-life) for each half-life the spec lists. *Model* names both, as in scikit-learn: where the kind matters, name it (*an `ewridge` model*). For the one a bank keeps, say *the model* | | yes |
| **fit** | as a noun, what a model has estimated from the rows it has learned, such as its coefficients. As a verb, estimate a model from rows. The calls that learn, such as `bank.fit`, `bank.fit_predict` and a query's `online.fit_predict`, fit a bank's models over the rows they are given. A bank's own methods go on from the [state](#saving-and-serving) it holds. A query's call goes on from the state `load_state=` loads, if it is given, and otherwise starts with nothing learned. *Learn* names what happens to one row | | yes |
| **learn** | take a row into a model's state (in most models, its running sums), so the row counts toward the predictions after it, until a [hard window](#fitting-a-model) drops it or the model resets. A model learns a row after predicting it | *update* for this step, though *update* stays in *update equation*, the docstring section that states the step's equations; *train*; *teach*, as in *teaches the model nothing* | yes |
| **out-of-sample** | a row's prediction is made before the model learns from that row, so it never reflects the row's own target. A diagnostic that reads [residuals](#fitting-a-model) reads ones made the same way. Write it hyphenated wherever it stands: *an out-of-sample residual*, *every prediction is out-of-sample* | *out of sample* unhyphenated (12); *honest* (10); *prequential* (5), a statistics term the reader is not assumed to know, except where a docstring names the prequential score and defines it | yes |
| **prediction** | a model's value for a target on a row, made before it learns the row, in the field `pred_<target>` | *forecast* (25); *score* as a noun for a prediction; *scored* for a row that is predicted, though the field `scored_clock`, the clock a row was predicted at, keeps the word | yes |
| **residual** | the target less the prediction made before learning the row, in the field `resid_<target>` | *error* where it means one row's residual | yes |
| **coefficient** | one of the numbers a regression multiplies a feature by, or its intercept, read with `bank.coef()` or from the field `coef` | *beta* (83), apart from a finance example that defines *beta* itself. `beta` stays as the name of a column of `bank.marginal()`, a slope | yes |
| **running sums** | the decayed sums and means a model keeps in place of its rows. They are all most models remember, apart from the snapshots a [hard window](#fitting-a-model) keeps and the rows a [refit](#fitting-a-model) holds. The bank, not the model, holds the rows that await their embargo, their `label_clock` time or a [formula target](#fitting-a-model)'s window | *accumulator* (51), a word of the code | yes |
| **Gram** | the sums a model with a co-moment matrix keeps, as `ewridge`, `lasso` and `ew_cov` do: the weight of the rows, the columns' means and co-moments, and each target's moments. Every least-squares fit on its columns follows from it. `bank.gram()` returns it, and `po.gram` fits from it | *sufficient statistics* (9) as a second name, kept only in a docstring's Notes as the statistics term; *running sums* where it means a least-squares model's Gram | yes |
| **solve** | computing a fit's coefficients from its Gram, on the schedule `solve_every` sets | | yes |
| **refit** | once a target has learned a set number of rows, set by a `*_refit_rows` parameter such as `huber_refit_rows`, the model fits those first rows again, all at once, and continues from that fit. Each group reserves its refit's memory when it first appears, and the group that would pass the spec's `refit_budget`, with every group after it, gives the refit up | *re-solve* for this step | yes |
| **local fit** | a fit under a finite half-life, in a model that keeps running sums: it describes the recent past and moves from row to row. With decay off (`half_life=float("inf")`) and no hard window, a fit instead weighs every row it has learned by its weight alone, however old the row is | | yes |
| **hard window** | a model's limit, set by its `window_size=`, past which a row counts for nothing. A row older than `window_size` clock units has no weight. Under the default `closed="right"`, a row exactly `window_size` old has none either, and under `closed="both"` it keeps its weight. A half-life alone never forgets a row entirely. The model keeps snapshots of its running sums, on every row or every `snapshot_every` clock units, to subtract the rows that leave | *window* alone for this, since *window* also names a window operator's span; *windowed* for a model under `window_size`: say *under a hard window* | yes |
| **window operator** | one of the exponentially weighted functions of `po.ops`: `ewm_mean`, `ewm_sum`, `ewm_rate`, `ewm_quantile`, `ewm_var` and `ewm_std`, also reached as `po.ewm_mean` and so on. Each returns a Polars expression whose value on each row is computed from that row's window, the rows near it on the clock. Under the default `closed="right"`, the window is the rows at or before the row on the clock, the rows at its own clock value included. For an operator that takes `reverse=True`, it can be the rows after the row instead. The window is weighted by an exponential decay along the clock, in the form each operator's docstring states, and cut off at `window_size` where one is given. `po.ops.increment`, the step from a column's previous value, is not one | *rolling* (37) for these. *Rolling* stays for Polars' own `rolling_*` functions | yes |
| **window run** | one call of `po.stream.with_windows`, which computes window operators over a stream chunk by chunk and can save and resume a state of its own | | yes |
| **formula target** | a target given in `targets=` as an expression that holds at least one window operator looking ahead (`reverse=True`). The spec's builder turns it into a `FormulaTarget`, and the bank holds each row until the window its operators read has closed and its embargo has passed, then learns from it | | yes |
| **diagnostic** | an optional statistic a spec writes into fields of its own as the stream runs, such as a calibration slope or a test for a break. Most are computed on the out-of-sample residuals | | yes |
| **break** | a change in the process a model fits, such as a shift in a coefficient, which `emit_breaks` tests for | | yes |
| **switch** | a parameter of a spec that turns a diagnostic or another optional output on, such as `emit_calibration=True` | | yes |

## Warm-up and readiness

A model learns from the rows of its warm-up, but its predictions are withheld until it has learned enough to give them.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **warm-up** | a model's first rows, or its first rows after a reset, on which its gates withhold its predictions until it has learned enough. The model learns from them all the same. *Warm up* is the verb | *warmup* (1), since the noun is hyphenated as *warm-up*; *burn-in* | yes |
| **gate** | a parameter that withholds a model's prediction until a condition holds: `min_settled_frac`, `min_weight`, `max_error_inflation` or `warm_rows` | | yes |
| **withheld** | a prediction held back on a row, by a gate or because the fit cannot be trusted there (`ill_conditioned`, `unstable_rate`). On that row the prediction and most fields read from the fit are null, as [OUTPUTS.md](OUTPUTS.md) lists. The fields that report readiness, such as `weight_sum`, `settled_frac` and `withheld_reason`, which says why, are filled | *gated* (5), *suppressed* (1) | yes |
| **ready** | a model is ready on a row when every gate is met there and the fit can be trusted, so its prediction is given | *settled* (52) and *settles* (45) for this; *converged* for this, since *converged* is kept for a fit approaching its limit | yes |
| **readiness warning** | the warning `ReadinessWarning`. It reports a finding: a reason to doubt what a model reports. One finding is a coefficient set more by the `ridge` penalty than by the data. Another is a gate still unmet once the model has learned long enough to judge. A call that learns issues it when its finding first holds, not on every row. The command line prints the same text. `po.gram.lasso_path` issues it too, when a penalty's descent stops before it converges. The docstring of `ReadinessWarning` describes each finding | *notice* (30), a word of the code | yes |
| **`weight_sum`** | the weight a model has accumulated before it learns a row and before the decay applied at that row. It is 0 on a model's first row, and on its first row after a reset. Under a hard window, it is the weight inside the hard window. It means the same in every model, so one `min_weight` means the same for every spec in a bank, checked against each target's own weight where the model keeps one | *n_eff* (7); *effective weight*; *effective sample size* as its name. The term stays for Kish's effective sample size, `n_kish`, a different number | as `weight_sum` |
| **`settled_frac`** | how far a model has come toward its steady state under decay: `1 - 0.5 ** (T / half_life)`. `T` is the sum of the steps the model has decayed by, as [decay](#time) defines them, since its first row or its last reset. Under `embargo`, `T` also counts the clock the held rows have covered. So it is 0.5 after one half-life and 0.75 after two, whatever the row rate, and null where nothing decays | | as `settled_frac` |
| **seeding** | placing a model's starting values from its first rows (`warm_rows`) before it predicts anything, as `kmeans` places its centers, `hmm` its hidden states and `bocpd` its prior. Those rows are part of its warm-up: the `warm_rows` gate withholds its predictions on them. `kalman` takes no `warm_rows`, but withholds a target's predictions (`below_warm_rows`) until a row has sized that target's prior | *seed* as a noun for these rows (the verb, as in *the model seeds its centers*, is fine). *Seed* stays for a random seed | yes |

## Grids

A spec given several values for one or more parameters fits every combination of them in one pass over the stream.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **grid** | the values a spec fits side by side: several half-lives in `half_life`, several penalties in `ridge` or in `lasso`'s `penalties`, and the named sets of `feature_sets`. `po.eval.grid_search` also takes a `param_grid`, scikit-learn's word, each of whose combinations builds a spec of its own | *sweep*, for a spec's grid. *Grid* alone for the times `po.stream.refresh_time` puts series on: say *the refresh grid* | yes |
| **grid point** | one combination of the values a grid lists, such as `half_life=500` with `ridge=0.1`, in a spec's grid or in `po.eval.grid_search`'s `param_grid` | *grid combination* (4), *grid slot* (3), since *slot* names a prediction; *grid point* for one of the times `po.stream.refresh_time` puts series on: say *refresh time*; *setting* for a whole combination; *candidate* (27), scikit-learn's word, for a grid point. `"candidate"` stays as a value of `po.eval.grid_search`'s `kind` column | yes |
| **instance** | the part of a model (the one a bank keeps for one spec and one group) that fits one of its half-lives, with state of its own. Python also says *instance* for an object of a class, so gloss the word at its first use on a page or in a message: *the instance for half-life 500 (the part of the model that fits that half-life)*. A spec with one half-life, or one that sets decay by `lam`, has one instance. The `instance` column of `bank.coef()`, `bank.marginal()`, `bank.closed_groups()` and `po.spec.output_index()`, and the `instance` key of each dict `bank.gram()` returns, hold its name: a suffix such as `"@h500"`, or `""` when the spec has one instance. In `po.spec.output_index()` it is null on a field that chooses across instances and on the fields `emit_clocks` writes. Grid points that differ only in `ridge`, `feature_sets` or `penalties` share one instance, so an instance is not a grid point | *decay instance* | yes, glossed |
| **slot** | one of the predictions a spec writes on each row: one target at one grid point, or a spec's pick or blend across its grid (`emit_selected`, `emit_averaged`). `po.eval.metrics` and `po.eval.grid_search` score each slot, in their `slot` column | *prediction slot*; *candidate* where it means a slot; *slot* for a coefficient's position | yes |

## What a bank writes

For each spec, a bank adds one column to the rows it reads, and `po.eval` scores what those columns hold.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **output column** | the one column a spec adds to the stream's rows, named after the spec. Its value in each row is a record of named fields | *struct column* (43), except where it names Polars' `Struct` type | yes |
| **field** | one named value in a spec's output column, such as `pred_y` or `weight_sum`. `po.spec.output_fields(spec)` lists a spec's fields, and [OUTPUTS.md](OUTPUTS.md) lists every model's | | yes |
| **score** | a number that compares predictions with their targets, such as the mean squared error, computed by `po.eval` and chosen by `scoring=` in `po.eval.grid_search`, as scikit-learn uses the word. It never means a prediction, or predicting without learning (see [serve](#saving-and-serving)) | | yes |

## Saving and serving

A bank's state can be saved to a file, and loaded to go on learning or to predict without learning.

| word | what it means | replaces (uses on 2026-11-05) | in a message |
|---|---|---|---|
| **state** | everything a bank has learned, for every spec and group, with the rows it holds until it can learn them, under an embargo, a `label_clock` or a formula target, and the closed groups' rows not yet read. A [window run](#fitting-a-model) and a run of `po.stream.refresh_time` each keep a state of their own, saved and resumed the same way | *checkpoint* (11), kept only for the checkpoints of Pathway, a Python framework for streaming pipelines; *snapshot* for a state. The word *snapshot* stays for the snapshots a hard window takes (`snapshot_every`) | yes |
| **saved state** | a state written to a file, with `save_state=` or `bank.save`, and loaded with `load_state=` or `ModelBank.load`. Where the file itself is meant, such as its path or its bytes, call it a *state file* | | yes |
| **resume** | load a saved state and go on from the row after the last one the bank, or the `po.stream` run, read before the state was saved | | yes |
| **serve** | predict rows from a state without learning from them, with a `predict` call, such as `bank.predict` or a query's `online.predict` | *score rows* (24), meaning predict them without learning | yes |
| **closed group** | a group whose rows have ended, or under `group_close="session"` one session span of a group, as the parameter `group_close` decides. The bank writes out, as rows, what the group's model has learned, its running sums where it keeps them, and frees what it held for it. `bank.closed_groups()` returns those rows | *finished group* (12) | yes |
| **command line** | the `polars-online` program, which runs a bank from a TOML file, outside Python, reading a file in and writing one out, in Parquet, Arrow IPC, CSV or NDJSON | *CLI* (31); *runner* (61), a word of the code | yes |

## Words kept out of user-facing text

*User-facing text* means what a user of the library reads: the README,
`llms.txt`, `docs/README.md`, the guides for users, the API reference's
own pages, the public docstrings and the messages.
Keep these words of the code, the tests and Polars' internals out of it. A
contributor guide may use them, since its subject is the code.

| word | what it means | say instead |
|---|---|---|
| *settled* | how far a model has come toward its steady state under decay | use the field `settled_frac`, or say *how far the model has come toward its steady state* |
| *cfg*, *ModelKind*, *ModelState*, *AnyModel* | the Rust names of a spec and a model | *the spec*, *the model* |
| `Span` | a clock parameter as the Rust code holds it | *a value in clock units* |
| *O(state)*, *O(data)* | how memory grows, in the code's shorthand | *memory grows with the model's state, and never with the number of rows that have passed* |
| *O(window)* | the same shorthand, for what a hard window or a window operator keeps | *memory grows with the snapshots a hard window keeps, or the rows a window operator holds* |
| *sink*, `sink_parquet` | a Polars method, and the word for what it does | *runs the query and writes the result to a file without holding it all in memory* |
| *collect* | a Polars method | *runs the query and returns its result* |
| `collect_batches` | a Polars method | *gives the result one chunk at a time* |
| *scan*, `scan_parquet` | a Polars function | *reads the file as a query* |
| *morsel*, *streaming engine* | the blocks of rows Polars works on, and the part of Polars that runs a query in them | *the blocks of rows Polars works on*; *Polars, when it runs a query* |
| *row group* | the unit a parquet file stores its rows in | *a block of a parquet file* |
| *prefetch* | the row groups Polars' reader decodes ahead of whatever consumes them | *Polars reading ahead in the file* |
| *pipeline*, *lane* | one copy of a parallel stage of Polars' streaming engine, one a thread | *Polars, when it runs a query* |
| *distributor*, *linearizer* | the stages of Polars' streaming engine that hand its blocks of rows out to the lanes, and gather them back in order | *Polars, when it runs a query* |
| *peak footprint* | the most physical memory a process held at once, as `/usr/bin/time -l` prints it on macOS. It is not the resident set size (RSS), which also counts the pages of a memory-mapped file | *the most memory the process held* |
| *the pool* | the bank's threads | *the bank's threads* |
| *task*, meaning the work the bank gives one thread | the bank's unit of work | *the work each thread does* |
| *stream*, meaning one spec over one group's rows | the bank's unit of parallel work, in the code's and the profiles' sense: one spec over one group, or over the whole input when the spec has no `group` | *the model's rows*; *a piece of work*, as [Use more cores](how-to/use-more-cores.md) says |
| *slot*, meaning one value an instance writes | one field an [instance](#grids) writes on every row, such as a target's prediction or one of `ew_cov`'s statistics, in the bank's output buffers | *a field* |
| *run*, meaning `ChunkOut::run_rows` | the rows a stream feeds through one set of the bank's output buffers before it starts the next | *the rows the bank writes at once* |
| *section*, meaning a part of a chunk's work | one of the four parts of a chunk's work that `POLARS_ONLINE_TIMING=1` times: `extract` (reading the columns), `group` (finding each group's rows), `process` (the models) and `assemble` (building the output columns) | the part itself: *reading the columns*, *finding each group's rows*, *running the models*, *building the output* |
| *stride*, *gather* | two ways to read one group's values from an interleaved column: a stride reads every n-th row in place, and a gather copies them into one contiguous run first | *reading a group's rows in place*, *copying a group's rows together first* |
| *solve cadence* | how often a regression solves: every `solve_every` clock units, or by default once the weight learned since the last solve reaches `ln 2 / 50` of the weight the fit holds, which is `half_life / 50` of clock in steady state | *how often the model solves* |
| *bit-identical*, *bit-exact* | the same floating-point bits, as a test compares each float column as its `u64` bits, with its validity | *the same to the bit* |
| *golden*, *oracle*, *the gate*, *mutants* | a stored reference result, an independent check of a computed value, the test run before a commit, and mutation testing | in user-facing text, what each means: *a stored reference result*, *an independent check of the value*, *the tests run before each commit*, *mutation testing*. A contributor guide may use the word itself, with *the test gate* for *the gate*, since a [gate](#warm-up-and-readiness) is a readiness parameter |
| *golden number* | a value the golden tests pin: `golden.rs` in `online-core`, and `tests/test_golden_pipeline.py` to 1e-12. A change that moves one past that tolerance is wrong by definition, and a reordering of the arithmetic that moves one within it is accepted | *a stored reference result* |

## Why the contested words were chosen

**Choose a word by three rules, in order.** First, take what the public API
already calls the thing, since the docs cannot rename the API. Second,
take Polars' or scikit-learn's word, where Polars or scikit-learn has the same concept.
Third, take the variant used most. Pass over two kinds of word whatever
the rules say. One is a word that would name two things, unless a gloss or the
context separates them. The other is a word that praises the result
without naming the mechanism, as *honest* does.

| chosen | over (the main variants) | why |
|---|---|---|
| *spec* | *specification* | the API's `po.spec` and `specs=`. *Specification* is taken by the specification test, `emit_specification` |
| *stream* | *source* | as a parameter, `source` names what a call reads from, such as a query, a frame, files or a saved state, never the rows themselves. *Stream* is the variant used most for the rows (947 uses, against 128), and the module `po.stream` uses it |
| *query* | *plan* | the second rule: Polars' user guide speaks of a lazy *query*, and of the *query plan* Polars makes from it, which belongs to Polars' internals |
| *chunk* | *batch* | `chunk_size` is the API's parameter for the concept, where `fit_predict_batches` names a method and `batches` names its whole input. Polars' own `collect_batches` takes a `chunk_size` too |
| *target* | *label*, *outcome*, *response* | the API's `targets=`, scikit-learn's word. *Label* stays only in the API's own names, such as `ew_class`'s `label=` and `label_clock`, which the docs cannot rename |
| *feature* | *signal*, *regressor*, *predictor* | the API's `features=`, scikit-learn's word |
| *half-life* | *halflife* | the parameter is `half_life`, and Polars' `ewm_mean` takes `half_life` too |
| *step back* | *goes backwards*, *out of order* | the parameter the reader acts on, `restart_after_step_back`, says it, where `bank.summary()`'s `clock_backwards` only counts the event, and it is the variant used most (74 uses, against 24 and 12). *Out of order* means something else: a query whose row order Polars does not promise |
| *reset* | *restart* | `bank.summary()` counts `resets`, and `session_gap="reset"` names the same event. `restart_after_step_back` keeps its name |
| *clock break* and *break* | *break* for both | the API's `emit_breaks` and `break_wald` give *break* to a change in the process a model fits, so the clock's event takes the longer word |
| *model*, for a kind and for the one a bank keeps | a word for each | see [Model, for a kind and for the one a bank keeps](#model-for-a-kind-and-for-the-one-a-bank-keeps) |
| *out-of-sample* | *honest*, *out of sample* | *honest* says the prediction is good without saying why, and *out-of-sample* names the reason. The hyphenated form is the one used most (69 uses against 12) |
| *prediction*, and *score* only for a metric | *forecast*, and *score* for a prediction | the API's `pred_<target>` fields and `predict`. scikit-learn's `predict` and `score` draw the same line, and `po.eval` scores predictions |
| *coefficient* | *beta* | the API's `bank.coef()` and `coef` field, and scikit-learn's `coef_` |
| *running sums* | *accumulator* | *accumulator* is the code's type, and *running sums* says what is kept |
| *Gram* | *sufficient statistics*, *running sums* | the API's `bank.gram()` and `po.gram`. *Running sums* stays for the general idea, since most models keep running sums |
| *window operator* | *rolling* | Polars' `rolling_*` functions weigh every row of a fixed count or clock span alike, or by position with `weights`, never with a decay along the clock as a window operator does, so *rolling* would name two things |
| *warm-up*, *withheld*, *ready* | *settled*, and *gated* and *suppressed* for *withheld* | see [Warm-up, withheld and ready](#warm-up-withheld-and-ready) |
| *readiness warning* | *notice* | the API's `ReadinessWarning` |
| `weight_sum` | *n_eff*, *effective sample size* | the API's `weight_sum` field and `min_weight`. *Effective sample size* is Kish's `n_kish`, a different number |
| *instance*, *slot* and *grid point*, three words | one word for all three, or scikit-learn's *candidate* | see [Instance, slot and grid point](#instance-slot-and-grid-point) |
| *output column* | *struct column* | the column is named after the spec, and the API calls its parts fields, as in `po.spec.output_fields`. *Struct* is Polars' name for the column's type, not for what it holds |
| *state* | *checkpoint* | the API's `save_state=` and `load_state=` |
| *serve* | *score rows* | *score* is the metric's word, as in `scoring=`. *Predict* alone would not do, since every call that learns predicts too |
| *closed group* | *finished group* | the API's `group_close` and `bank.closed_groups()` |
| *command line* | *CLI*, *runner* | the variant used most (80 uses, against 31 and 61) |

### Model, for a kind and for the one a bank keeps

scikit-learn says *model* for both, and the API's `po.spec` builders name kinds while `ModelBank` holds the models it keeps, so the context separates them.

### Warm-up, withheld and ready

The API's `warm_rows`, `withheld_reason` and `ReadinessWarning` use the same words: warm, withheld and readiness. `settled_frac` is a field, so the word stays in that name, but a sentence that says a model is *98% settled* asks the reader to know the field's formula.

### Instance, slot and grid point

The API names two different things: `instance` names one half-life with its own state, in `bank.coef()`, `bank.gram()`, `bank.marginal()`, `bank.closed_groups()` and `po.spec.output_index()`, and `slot` is a column of `po.eval.metrics` and `po.eval.grid_search` for one prediction. A grid point is neither: grid points that differ only in `ridge`, `feature_sets` or `penalties` share an instance's state, and each grid point has a slot per target. So each keeps its word, and a page uses each only for its own thing. *Instance* has a Python meaning, and a word with a Python meaning normally keeps it ([WRITING.md §5](WRITING.md#5-sentences)). The API's `instance` column fixes the word, so a page glosses it instead of replacing it. scikit-learn's *candidate* is the thing a search scores. Here `po.eval.grid_search` scores slots, one per target at each grid point, so *candidate* would name a slot as often as a grid point.

## Spelling

**Use American spelling on every page**, for the reason [WRITING.md §12](WRITING.md#12-one-word-for-one-thing)
gives. No function, parameter or field in the public API uses a British
spelling. `po.gram.from_row` still accepts one old British key,
`cross_centred`, and forwards it to `cross_centered` with a deprecation
warning. A quotation keeps its source's spelling. `CHANGELOG.md` and the records
keep their words but take American spelling, apart from what they quote.

On 2026-11-05 the README, the guides for users, the public docstrings and the messages
held 364 British spellings. The contributor guides and the records held
others, such as *catalogue* and *labelling* (*catalog*, *labeling*). Each
of the 364 has an American form:

| British | uses | American |
|---|---:|---|
| *centred*, *centre*, *centres*, *centring*, *uncentred* | 240 | *centered*, *center*, *centers*, *centering*, *uncentered* |
| *standardised*, *standardiser*, *standardisation*, *standardisations* | 36 | *standardized*, *standardizer*, *standardization*, *standardizations* |
| *realised* | 19 | *realized* |
| *labelled* | 14 | *labeled* |
| *neighbour*, *neighbours*, *neighbouring* | 10 | *neighbor*, *neighbors*, *neighboring* |
| *penalised*, *linearised*, *materialises*, *optimiser*, *optimising*, *minimised*, *maximising* | 9 | *penalized*, *linearized*, *materializes*, *optimizer*, *optimizing*, *minimized*, *maximizing* |
| *behaviour*, *favour*, *favours*, *honour* | 6 | *behavior*, *favor*, *favors*, *honor* |
| *parametrise*, *parametrisation*, *reparameterisation* | 5 | *parametrize*, *parametrization*, *reparameterization* |
| *symmetrised*, *synchronised*, *randomises*, *parallelises*, *pessimisation* | 5 | *symmetrized*, *synchronized*, *randomizes*, *parallelizes*, *pessimization* |
| *towards* | 4 | *toward* |
| *normalised*, *normalisation* | 4 | *normalized*, *normalization* |
| *vectorise*, *vectorised*, *vectorises* | 4 | *vectorize*, *vectorized*, *vectorizes* |
| *summarise*, *summarised*, *recognise* | 4 | *summarize*, *summarized*, *recognize* |
| *artefact*, *licence* | 3 | *artifact*, *license* |
| *cancelling* | 1 | *canceling* |
