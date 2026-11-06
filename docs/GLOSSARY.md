# Glossary

This glossary gives the one word polars-online uses for each concept, for
anyone reading or writing its docs. The README, the guides, the
docstrings and the messages all use that word. Look a word up here when a
page uses it without defining it, and choose the word here when you write
a sentence, a docstring or a message about the library. In the examples,
`po` is the package, imported as `import polars_online as po`.

*Reader-facing text* here means the README, the guides for users, the
public docstrings and the messages. A message is read with no
documentation open, so it uses only words a reader has met in the
docstrings and the guides. The *in a message*
column says how a message may use each word, in one of four ways:

| value | what it allows |
|---|---|
| yes | the word, in a sentence |
| yes, glossed | the word, with the gloss its row gives, since a message cannot assume the reader has met it |
| as `name` | only the parameter, field or column of that name, in backticks, and never the bare word in a sentence |
| as `name`, if set | as `name`, and only in a message about a [spec](#the-bank-and-what-it-reads) that set that parameter |

The number after a word in *replaces* is how often it appeared in the
README, the guides for users, the public docstrings and the messages on 2026-11-05.
The count covers the prose and the comments of code examples. A word with
no number was not counted, and is retired all the same in the sense its
row gives. These counts are what the third rule of
[Why the contested words were chosen](#why-the-contested-words-were-chosen)
reads when it picks the variant used most. Where the counts
bear on a choice, that table gives the chosen word's count too.

## The bank and what it reads

A model bank reads its rows in order, and each spec names the columns its model reads.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **spec** | the description of one model: which model it is, the columns it reads and how it treats time. `po.spec.ewridge("m", ...)` builds one, and its name, `m`, names its output column | *specification* (4), which is kept for the specification test (`emit_specification`), the diagnostic that tests for structure the model's form misses; *model spec*; *configuration* of a model | yes |
| **model bank**, then **bank** | a set of models, one for each spec and group, fitted together over the same rows, and held by the class `ModelBank`. A page says *model bank* at its first use and *bank* after it. Call a bank with no state *a bank that starts with nothing learned* | *fresh bank* (8) | yes |
| **stream** | the rows a bank reads, in the order it reads them, however many there are | *source* (128) for the rows themselves; `source` stays as the name of `po.eval.grid_search`'s parameter and in `ConsumedSourceWarning`, for the query or frame the rows are read from | yes |
| **query** | a Polars `LazyFrame`: a chain of steps that runs only when you ask for its result | *plan* (110), a word of Polars' own internals; *LazyFrame* in a sentence, where it is not naming the type | yes |
| **expression** | a Polars expression, a column computation such as `pl.col("x") * 2`, which a query's steps take and a window operator returns. A page or a message glosses it at its first use: *an expression (a column computation, such as `pl.col("x") * 2`)* | | yes, glossed |
| **frame** | a Polars `DataFrame`, whose rows are already in memory | *DataFrame* in a sentence, where it is not naming the type | yes |
| **chunk** | the rows the bank takes in at once: `chunk_size` rows of a query (100,000 by default), or each frame passed to `fit`, `fit_predict` or `fit_predict_batches` | *batch* (109), kept only in the name `fit_predict_batches` and in code that calls Polars' `collect_batches` | yes |
| **target** | a column a model predicts, named in `targets=` | *outcome* (21), *response* (17), *label* where it means a target (12, as in *labels that arrive late*); *label* is kept for the class column `ew_class` reads, `label=`. A heading that uses *label*, such as *Labels that arrive late*, keeps that wording until every link to it is updated, since a heading's text is its anchor | yes |
| **feature** | a column a model reads to predict a target, named in `features=` | *signal* (10), *regressor* (7), *predictor* (1), *covariate* | yes |
| **weight** | how much a row counts in a fit: the value in the column `weight=` names, or 1 without one. A row of weight 2 counts as two rows of weight 1, and a row of weight 0 is predicted and moves the clock, but the model learns nothing from it. Once learned, a row's weight decays along the clock (see **decay**) | | yes |
| **group**, **group key** | the rows that share one value of the column `group=` names. The bank fits a separate model for each group, and the value is the group's key, as Polars calls it | *partition* (3); *key* alone, for a group's value | yes |
| **null** | the value a Polars column holds where it has none. NaN, ±inf and any magnitude above `1e100` count as null in a feature, a target or a weight. A null target leaves the row predicted and the clock moving, but the model learns nothing about that target from the row. A model that still learns something from such a row, such as its features' means, says so in its docstring | *missing value* | yes |
| **skipped row** | a row with a null in any feature or in its weight: the model neither predicts it nor learns from it, every field of its output is null, and the clock still moves. `bank.summary()` counts these rows in `rows_skipped` | | yes |

## Time

Every parameter that measures time is measured on the clock, in clock units. A model forgets along the clock.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **clock** | the column, named by `clock=`, that says how far apart two rows are. Without one, each row is one clock unit after the row before it | *timestamp column*; *time column* (1), as names for the clock. *Timestamp column* stays as a description of a clock that holds times, as in *a clock (timestamp column `ts`)* | yes |
| **clock units** | the unit of every parameter measured on the clock: the clock column's own numbers, or a duration such as `"5m"` when the clock holds times | | yes |
| **decay** | how a model forgets old rows: at each row it multiplies the weight of every row it has learned by `0.5 ** (Δclock / half_life)`, where `Δclock` is the clock's step since the previous row of its group. So a learned row's weight halves every half-life | *forgetting factor* (1). *Forget* stays an ordinary verb in an explanation, not a second name | yes |
| **half-life** | the distance on the clock over which a learned row's weight halves, set by `half_life=`. `half_life=float("inf")` turns decay off | *halflife* (54), an old spelling of the parameter | yes |
| **`lam`** | the parameter that sets decay by its factor, in place of `half_life`: over each clock unit, a learned row's weight is multiplied by `lam`, and `lam = 0.5 ** (1 / half_life)` | *λ* outside a docstring's update equation; *lambda* for the decay (55), since *lambda* is also the lasso's penalty | as `lam`, if set |
| **session** | a column, named by `session=`, whose value changes where the stream crosses a boundary such as the start of a trading day. At the change, the clock moves by `session_gap` in place of its real jump, or, under `session_gap="reset"`, the model resets | | yes |
| **step back** | a row whose clock is earlier than the previous row's in the same group. If `restart_after_step_back` is set and the step is larger than it, the model resets, and `bank.summary()` counts the step in `clock_backwards`. If `restart_after_step_back` is not set, or the step is no larger than it, the bank refuses the chunk before learning any of it, with an error that names the row. A small step back usually means a late row, which the model cannot learn in clock order. A large one usually means the data starts over, such as a replayed day | *goes backwards* (24) in a sentence; *out of order* (12) for a clock step. *Out of order* stays for a query whose row order Polars does not promise, the case `OrderNotGuaranteedWarning` reports | yes |
| **reset** | a model starting over with nothing learned: at a session boundary under `session_gap="reset"`, or, where `restart_after_step_back` is set, at a step back larger than it. `bank.summary()` counts them in `resets` | *restart* (50), kept only in the name `restart_after_step_back` | yes |
| **embargo** | the delay on the clock between predicting a row and learning from it, for a target whose value is known only some time after the row: under `embargo="5m"` the model learns each row five minutes after predicting it | *label delay* | yes |

## Fitting a model

A model predicts each row before it learns it. Most models keep only running sums of what they have learned.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **model** | one of the kinds of model a spec can name, such as `ewridge` or `kmeans`, and also the model a bank keeps for one spec and one group, which holds one instance for each half-life the spec lists. This is the one word chosen to name two different things, as in scikit-learn: name the kind (*an `ewridge` model*) where the kind matters, and say *the model* for the one a bank keeps | | yes |
| **fit** | as a noun, what a model has estimated from its running sums, such as its coefficients. As a verb, the calls `fit` and `fit_predict` and a whole pass over the stream. *Learn* names the same step for one row | | yes |
| **learn** | take a row into a model's running sums, so it counts toward every later prediction. A model learns a row after predicting it | *update* for this step, though *update* stays in *update equation*, the docstring section that states the step's equations; *train*; *teach*, as in *teaches the model nothing* | yes |
| **out-of-sample** | a row's prediction is made before the model learns from that row, so it never reflects the row's own target. Every diagnostic reads residuals made the same way. Write it hyphenated wherever it stands: *an out-of-sample error*, *every prediction is out-of-sample* | *out of sample* unhyphenated (12); *honest* (10); *prequential* (5), except where a docstring names the prequential score and defines it | yes |
| **prediction** | a model's value for a target on a row, made before it learns the row, in the field `pred_<target>` | *forecast* (25); *score* as a noun for a prediction | yes |
| **residual** | the target less the prediction made before learning the row, in the field `resid_<target>` | *error* where it means one row's residual | yes |
| **coefficient** | one of the numbers a regression multiplies a feature by, read with `bank.coef()` or from the field `coef` | *beta* (83), apart from a finance example that defines *beta* itself | yes |
| **running sums** | the decayed sums and means a model keeps in place of its rows. They are all most models remember, apart from the snapshots a hard window keeps and the rows a refit holds. The bank, not the model, holds the rows that await their embargo or a formula target's window | *accumulator* (51), a word of the code | yes |
| **Gram** | the running sums a least-squares model keeps: the weight of the rows, the columns' means and co-moments, and each target's moments. Every least-squares fit on its columns follows from it. `bank.gram()` returns it, and `po.gram` fits from it | *sufficient statistics* (9) as a second name, kept only in a docstring's Notes as the statistics term; *running sums* where it means a least-squares model's Gram | yes |
| **solve** | computing a fit's coefficients from its Gram, on the schedule `solve_every` sets | | yes |
| **refit** | once a target has learned a set number of rows (`huber_refit_rows` in `huber`, `quantile_refit_rows` in `quantile`, `bin_refit_rows` in `marginal`'s bins), the model fits those first rows again, all at once, and continues from that fit | *re-solve* for this step | yes |
| **local fit** | a fit under a finite half-life, which describes the recent past and moves from row to row. A fit with decay off (`half_life=float("inf")`) instead weighs every row it has learned alike | | yes |
| **hard window** | a limit, set by `window_size=`, past which a row counts for nothing: a row `window_size` clock units old or older has no weight. A half-life alone never forgets a row entirely. The model keeps snapshots of its running sums, every `snapshot_every` clock units, to subtract the rows that leave | *window* alone for this, since *window* also names a window operator's span; *windowed* for a model under `window_size`: say *under a hard window* | yes |
| **window operator** | a function of `po.ops`, such as `po.ops.ewm_mean` (also reached as `po.ewm_mean`), that returns a Polars expression (a column computation such as `pl.col("x") * 2`). The expression's value on each row is computed from the rows near it on the clock, which are its window: the rows before it, or the rows after it with `reverse=True`. It weighs each row by its age on the clock, exponentially, and cuts the window off at `window_size` where one is given | *rolling* (37) for these; *rolling* stays for Polars' own `rolling_*` functions | yes |
| **formula target** | a target written with window operators that look ahead, built as a `FormulaTarget`: the bank holds each row until the window its operators read has closed and its embargo has passed, then learns from it | | yes |
| **diagnostic** | a statistic a model computes on its out-of-sample residuals as the stream runs, such as a calibration slope or a break test, written into its own fields | | yes |
| **switch** | a parameter of a spec that turns a diagnostic or another optional output on, such as `emit_calibration=True` | | yes |

## Warm-up and readiness

A model learns from the rows of its warm-up, but its predictions are held back until it has learned enough to give them.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **warm-up** | a model's first rows, or its first rows after a reset, on which its gates withhold its predictions until it has learned enough. The model learns from them all the same. *Warm up* is the verb | *warmup* (1); *burn-in* | yes |
| **gate** | a setting that withholds a model's prediction until a condition holds: `min_weight`, `min_settled_frac`, `max_error_inflation` or `warm_rows` | | yes |
| **withheld** | a prediction held back on a row, by a gate or because the fit cannot be trusted there (`ill_conditioned`, `unstable_rate`): the prediction and the fields read from the fit are null on that row, while `weight_sum`, the other readiness fields and `withheld_reason`, which says why, are filled | *gated* (5), *suppressed* (1) | yes |
| **ready** | a model is ready on a row when nothing withholds its prediction there | *settled* (52) and *settles* (45) for this; *converged* for this, since *converged* is kept for a fit approaching its limit | yes |
| **readiness warning** | the warning `ReadinessWarning`. The calls that learn (`fit`, `fit_predict`, `fit_predict_batches` and a query's `online.fit_predict`) issue it once for each spec and group, and for each session where sessions reset the model. The command line prints the same text. It names one of three causes. A coefficient may be set more by the `ridge` penalty than by the data. A gate may still be unmet once the stream has run long enough to judge. Or a diagnostic may need to know how many rows ahead its target looks, on a spec that does not give `horizon_rows` | *notice* (30), a word of the code | yes |
| **`weight_sum`** | the weight a model has accumulated before it learns a row and before the decay applied at that row. It is 0 on a model's first row, and on its first row after a reset. Under a hard window, it is the weight inside the window. It means the same in every model, so one `min_weight` means the same for every spec in a bank | *n_eff* (7); *effective weight*; *effective sample size* as its name; the term stays for Kish's effective sample size, `n_kish`, a different number | as `weight_sum` |
| **`settled_frac`** | the share of its long-run weight a model has reached. The long-run weight is the weight it approaches on an endless stream. The share is `1 - 0.5 ** (T / half_life)`, where `T` is the clock time since the model's first row, or since its last reset, so it is 0.5 after one half-life and 0.75 after two | | as `settled_frac` |
| **seeding** | placing a clustering or regime model's starting values from its first rows (`warm_rows`), before it predicts anything. Those rows are part of its warm-up: the `warm_rows` gate withholds its predictions on them | *seed* as a noun for these rows (the verb, as in *the model seeds its centers*, is fine); *seed* stays for a random seed | yes |

## Grids

A spec given a list of values in place of one value fits every combination of them in one pass over the stream.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **grid** | the lists a spec takes in place of one value, for `half_life`, `ridge`, `feature_sets` or `penalties`. The spec fits every combination in one pass over the stream | *sweep*, for a spec's grid | yes |
| **grid point** | one combination of the values a spec's grid lists, such as `half_life=500` with `ridge=0.1` | *grid combination* (4), *grid slot* (3); *setting* for a whole combination; *candidate* (27), scikit-learn's word, for a grid point | yes |
| **instance** | the part of a model (the one a bank keeps for one spec and one group) that fits one of its half-lives, with state of its own. Python also says *instance* for an object of a class, so a page or a message glosses the word at its first use: *the instance for half-life 500 (the part of the model that fits that half-life)*. A spec with one half-life has one instance. The `instance` column of `bank.coef()`, `bank.gram()`, `bank.closed_groups()` and `po.spec.output_index()` holds its name: a suffix such as `"@h500"`, or `""` when the spec has one half-life. Grid points that differ only in `ridge`, `feature_sets` or `penalties` share one instance, so an instance is not a grid point | *decay instance* | yes, glossed |
| **slot** | one of the predictions a spec writes on each row: one target at one grid point. `po.eval` and `po.eval.grid_search` score each slot, in their `slot` column, and a diagnostic writes one field per slot | *prediction slot*; *candidate* where it means a slot | yes |

## What a bank writes

For each spec, a bank adds one column to the rows it reads, and `po.eval` scores what those columns hold.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **output column** | the one column a spec adds to the stream's rows, named after the spec. Its value in each row is a record of named fields | *struct column* (43), except where it names Polars' `Struct` type | yes |
| **field** | one named value in a spec's output column, such as `pred_y` or `weight_sum`. `po.spec.output_fields(spec)` lists a spec's fields, and [OUTPUTS.md](OUTPUTS.md) lists every model's | | yes |
| **score** | a number that compares predictions with their targets, such as the mean squared error, computed by `po.eval` and chosen by `scoring=` in `po.eval.grid_search`, as scikit-learn uses the word. It never means a prediction, or predicting without learning (see **serve**) | | yes |

## Saving and serving

A bank's state can be saved to a file, and loaded to go on learning or to predict without learning.

| word | what it means | replaces (uses) | in a message |
|---|---|---|---|
| **state** | everything a bank has learned, for every spec and group | *checkpoint* (11), kept only for the checkpoints of Pathway, a Python framework for streaming pipelines; *snapshot* for a state; the word *snapshot* stays for the snapshots a hard window takes (`snapshot_every`) | yes |
| **saved state** | a state written to a file, with `save_state=` or `bank.save`, and loaded with `load_state=` or `ModelBank.load`. Call it a *state file* only where the file itself is meant, such as its path or its bytes | | yes |
| **resume** | load a saved state and go on learning from the row after the last one the bank learned before the state was saved | | yes |
| **serve** | predict rows from a state without learning from them, with `bank.predict` or a query's `online.predict` | *score rows* (24), meaning predict them without learning | yes |
| **closed group** | a group whose rows have ended, as the parameter `group_close` decides: the bank writes the group's running sums out as one row and frees what it held for the group. `bank.closed_groups()` returns those rows | *finished group* (12) | yes |
| **command line** | the `polars-online` program, which runs a bank from a TOML file with no Python, reading parquet in and writing parquet out | *CLI* (31), *runner* (61) | yes |

## Words kept out of reader-facing text

Every word in a *replaces* column is kept out of reader-facing text in
the sense its row retires, apart from the uses its row keeps. This table
adds the words of the code, the tests and Polars' internals, which have no
row of their own in this glossary. It repeats a retired word only where
its replacement needs a longer phrase. None of these words appears in
reader-facing prose, except *λ* inside a docstring's update equation. A
code example calls Polars' own methods, such as `pl.scan_parquet`,
`collect` and `collect_batches`, by name. Where the word is also the name
of a parameter or a field, as `lam` and `settled_frac` are, the name may
appear in backticks, as the *in a message* column of its row allows. A
quotation keeps its source's words, and a guide that names a retired word
as an example of what to replace may print it.

| word | what it means | say instead |
|---|---|---|
| *settled* | the share of its long-run weight a model has reached, or a model whose prediction nothing withholds | use the field `settled_frac`, or say *the share of its long-run weight the model has reached*; for the second sense, *ready* |
| *lam* as a bare word in a sentence, *λ* | the factor a learned row's weight is multiplied by over one clock unit | the half-life, or the parameter `lam` if the reader set it |
| *cfg*, *ModelKind*, *ModelState*, *AnyModel* | the Rust types of a spec and a model | *the spec*, *the model* |
| *Span* | a clock parameter as the Rust code holds it | *a value in clock units* |
| *O(state)*, *O(data)* | how memory grows, in the code's shorthand | *memory grows with the model's state, and never with the number of rows that have passed* |
| *O(window)* | the same shorthand, for what a hard window or a window operator keeps | *memory grows with the snapshots a hard window keeps, or the rows a window operator holds* |
| *sink*, `sink_parquet` | a Polars method | *runs the query and writes the result to a file without holding it all in memory* |
| *collect* | a Polars method | *runs the query and returns its result* |
| `collect_batches` | a Polars method | *gives the result one chunk at a time* |
| *scan*, `scan_parquet` | a Polars method | *reads the file as a query* |
| *morsel*, *streaming engine* | the blocks of rows Polars works on, and the part of Polars that runs a query in them | *the blocks of rows Polars works on*; *Polars, when it runs a query* |
| *row group* | the unit a parquet file stores its rows in | *a block of a parquet file* |
| *the pool* | the bank's threads | *the bank's threads* |
| *task*, meaning the work the bank gives one thread | the bank's unit of work | *the work each thread does* |
| *golden*, *oracle*, *the gate* (the test run before a commit), *mutants* | a stored reference result, an independent check of a computed value, the test run before a commit, and mutation testing | in a guide for users, what each means; in the contributor guides, the word itself, with *the test gate* for *the gate* |

## Why the contested words were chosen

A word is chosen by three rules, in order. First, what the public API
already calls the thing, since the docs cannot rename the API. Second,
Polars' or scikit-learn's word, where the concept is theirs too. Third,
the variant used most. Two kinds of word are passed over whatever the
rules say. One is a word that would name two things, unless a gloss or the
context separates them. The other is a word that praises the result
without naming the mechanism, as *honest* does.

| chosen | over (the main variants) | why |
|---|---|---|
| *model*, for a kind and for the one a bank keeps | a word for each | scikit-learn says *model* for both, and the API's `po.spec` builders name kinds while `ModelBank` holds the models it keeps, so the context separates them. A page names the kind, as *an `ewridge` model*, where it matters |
| *instance*, *slot* and *grid point*, three words | one word for all three, or scikit-learn's *candidate* | the API names two different things: `instance` is a column of `bank.coef()`, `bank.gram()`, `bank.closed_groups()` and `po.spec.output_index()` for one half-life with its own state, and `slot` is a column of `po.eval` for one prediction. A grid point is neither: several grid points share an instance's state, and each grid point has a slot per target. So each keeps its word, and a page uses each only for its own thing. *Instance* has a Python meaning, which a word normally keeps, but the API's `instance` column fixes the word, so a page glosses it instead of replacing it. scikit-learn's *candidate* is one combination of parameter values, which here would be a grid point, but `grid_search`'s docstring used it for a slot, so it would name two things |
| *warm-up*, *withheld*, *ready* | *settled*, and *gated* and *suppressed* for *withheld* | the API's `warm_rows`, `withheld_reason` and `ReadinessWarning` use the same words: warm, withheld and readiness. `settled_frac` is a field, so the word stays in that name, but a sentence that says a stream is *98% settled* asks the reader to know the field's formula |
| *chunk* | *batch* | `chunk_size` is the API's parameter, and names the concept, where `fit_predict_batches` names only a method. Polars says *batch* for the same idea, but the first rule comes before the second, so *batch* stays only in `fit_predict_batches` and Polars' `collect_batches` |
| *reset* | *restart* | `bank.summary()` counts `resets`, and `session_gap="reset"` names the same event. `restart_after_step_back` keeps its name |
| *step back* | *goes backwards*, *out of order* | the parameter `restart_after_step_back` says it, and the reader acts on that parameter, where the count `clock_backwards` in `bank.summary()` only records it. It is also the variant used most (74 uses, against 24 and 12). *Out of order* means something else: a query whose row order Polars does not promise |
| *closed group* | *finished group* | the API's `group_close` and `bank.closed_groups()` |
| *target* | *label*, *outcome* | the API's `targets=`, scikit-learn's word. *Label* is the API's own word for `ew_class`'s class column |
| *prediction*, and *score* only for a metric | *forecast*, and *score* for a prediction | the API's `pred_<target>` fields and `predict`. scikit-learn's `predict` and `score` draw the same line, and `po.eval` scores predictions |
| *window operator* | *rolling* | a window operator weighs each row by its age on the clock, exponentially, and cuts it off only at `window_size` where one is given. Polars' `rolling_*` functions weigh the rows of a fixed count or span alike, so *rolling* would name two things |
| *feature* | *signal*, *regressor*, *predictor* | the API's `features=`, scikit-learn's word |
| *running sums* | *accumulator* | *accumulator* is the code's type, and *running sums* says what is kept |
| *state* | *checkpoint* | the API's `save_state=` and `load_state=` |
| *serve* | *score rows* | *score* is the metric's word, as in `scoring=` |
| *query* | *plan* | the second rule: Polars' user guide speaks of a lazy *query*, and of the *query plan* Polars makes from it, which is its internals |
| *stream* | *source* | as `po.eval.grid_search`'s parameter, `source` names the query or frame the rows are read from, not the rows themselves. *Stream* is the variant used most for the rows (947 uses, against 128) |
| *readiness warning* | *notice* | the API's `ReadinessWarning` |
| *command line* | *CLI*, *runner* | the variant used most (80 uses, against 31 and 61) |
| *Gram* | *sufficient statistics*, *running sums* | the API's `bank.gram()` and `po.gram`. *Running sums* stays for the general idea, which most models have |
| *coefficient* | *beta* | the API's `bank.coef()` and `coef` field, and scikit-learn's `coef_` |
| *half-life* | *halflife* | the parameter is `half_life`, and Polars' `ewm_mean` takes `half_life` too |
| *out-of-sample* | *honest*, *out of sample* | *honest* says the prediction is good without saying why, and *out-of-sample* names the reason. The hyphenated form is the one used most (69 uses against 12) |
| *spec* | *specification* | the API's `po.spec` and `specs=`. *Specification* is taken by the specification test, `emit_specification` |

## Spelling

Every page uses American spelling, the contributor guides included, for
the reason [WRITING.md §12](WRITING.md#12-one-word-for-one-thing)
gives. No function, parameter or field in the public API uses a British
spelling. A quotation keeps its source's
spelling. The records are not rewritten, but their spelling is made
American like every page's, apart from what they quote.

On 2026-11-05 the README, the guides for users, the public docstrings and the messages
held 336 British spellings. The contributor guides and the records held
more, such as *catalogue* and *labelling* (*catalog*, *labeling*). Each
of the 336, with its American form:

| British | uses | American |
|---|---:|---|
| *centred*, *centre*, *centres*, *centring* | 227 | *centered*, *center*, *centers*, *centering* |
| *standardised*, *standardiser*, *standardisation* | 36 | *standardized*, *standardizer*, *standardization* |
| *realised* | 19 | *realized* |
| *labelled* | 14 | *labeled* |
| *neighbour*, *neighbours*, *neighbouring* | 10 | *neighbor*, *neighbors*, *neighboring* |
| *behaviour*, *favour*, *favours*, *honour* | 6 | *behavior*, *favor*, *favors*, *honor* |
| *penalised*, *linearised*, *materialises*, *optimiser* | 5 | *penalized*, *linearized*, *materializes*, *optimizer* |
| *towards* | 4 | *toward* |
| *normalised*, *normalisation* | 4 | *normalized*, *normalization* |
| *vectorise*, *vectorised*, *vectorises* | 4 | *vectorize*, *vectorized*, *vectorizes* |
| *summarise*, *summarised*, *recognise* | 4 | *summarize*, *summarized*, *recognize* |
| *artefact*, *licence* | 3 | *artifact*, *license* |
