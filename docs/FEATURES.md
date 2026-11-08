<!-- kind: map -->
# Every feature

This page lists every feature of polars-online, one line each, with a link
to the page that shows it, for a reader who wants the whole list or is
looking for a capability by name. A name links its entry in the [API
reference](https://hgilde.github.io/polars-online/), which gives its
parameters and their defaults, and the last column links the guide that
shows it in use. `po` is `import polars_online as po`, and `lf.online`
stands for the `.online` methods of a `LazyFrame` and of a `DataFrame`
alike. The [README's map](../README.md#where-to-go-next) groups the same
pages by the task you have in hand.

| family | its features |
|---|---|
| [Models](#models) | the twenty-two models a spec can name |
| [Running a bank](#running-a-bank) | a bank in a query, in a Python loop, and its output as Arrow |
| [Saving, loading and serving](#saving-loading-and-serving) | a bank's state, written and read back |
| [Reading a bank](#reading-a-bank) | what a bank holds, and the names of its output |
| [Preparing a stream](#preparing-a-stream) | features and targets made from windows over the stream, and grids for series that tick at their own times |
| [Scoring and comparing](#scoring-and-comparing) | `po.eval`: scores, tests and searches over a bank's output |
| [Fits from Grams](#fits-from-grams) | `po.gram`: fits over any blocks and columns, from the Grams one pass leaves |
| [Correlation matrices](#correlation-matrices) | `po.corr`: repair, shrink and score a correlation matrix |
| [Trees and simulation](#trees-and-simulation) | `po.gbdt` and `po.sim` |
| [Parameters every spec takes](#parameters-every-spec-takes) | the parameters shared by every model, by family |
| [Errors and warnings](#errors-and-warnings) | the classes a caller can catch or filter |
| [Versions and the environment](#versions-and-the-environment) | the version numbers, and the environment variables the library reads |
| [The command line](#the-command-line) | `polars-online`, its flags and its configuration file's keys |

## Models

Each model is built by a function of `po.spec`, whose first parameter is
the spec's name. [The models at a glance](MODELS.md#the-models-at-a-glance)
compares them, and [Choose a model](how-to/choose-a-model.md) says which
to start with.

| model | what it is | its pages |
|---|---|---|
| [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) | exponentially weighted ridge regression on running sums, a grid of ridges and feature sets solved from one set of sums | [entry](MODELS.md#ewridge--exponentially-weighted-ridge-regression) · [choose a ridge and a half-life](how-to/choose-a-ridge-and-half-life.md) |
| [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) | recursive least squares, which solves the ridge fit exactly on every row | [entry](MODELS.md#rls--recursive-least-squares) · [targets that go missing on different rows](how-to/fit-targets-that-go-missing-on-different-rows.md) |
| [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) | a lasso or elastic-net path, with the penalty chosen as the stream runs | [entry](MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs) · [choose columns on the lasso path](how-to/choose-columns-on-the-lasso-path.md) |
| [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) | a regression whose coefficients drift, tracked by a Kalman filter | [entry](MODELS.md#kalman--random-walk-β-dynamic-linear-model) · [choose a drift speed](how-to/choose-a-kalman-drift-speed.md) |
| [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) | Huber regression, which down-weights the rows a prediction misses by far | [entry](MODELS.md#huber--robust-regression) · [the robust scale](explanation/robust-scale.md) |
| [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) | quantile regression, the fit that a share `tau` of the targets fall below | [entry](MODELS.md#quantile--robust-regression) |
| [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) | stochastic gradient descent under squared, Huber, quantile, ε-insensitive, Poisson and logistic losses | [entry](MODELS.md#sgd--stochastic-gradient-descent) · [choose a learning rate](how-to/choose-a-learning-rate.md) · [a count target](how-to/fit-a-count-target.md) · [bound the coefficients](how-to/bound-the-coefficients.md) |
| [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) | passive-aggressive regression, the smallest step that brings each row within a margin | [entry](MODELS.md#pa--passive-aggressive-regression) · [follow a level that moves](how-to/follow-a-level-that-moves.md) |
| [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) | FTRL-proximal regression, logistic for a 0/1 target, with an L1 penalty that zeroes coefficients | [entry](MODELS.md#ftrl--online-logistic-regression) · [choose its penalties](how-to/choose-ftrl-penalties.md) |
| [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) | Holt's linear trend, a baseline that extrapolates the target's own level and slope | [entry](MODELS.md#holt--holts-linear-trend) · [choose Holt's smoothing](how-to/choose-holts-smoothing.md) |
| [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) | running means, variances, covariances, correlations, partial correlations, Mahalanobis distances and principal components | [entry](MODELS.md#ew_cov--exponentially-weighted-moments) · [flag joint outliers](how-to/flag-joint-outliers.md) · [track principal components](how-to/track-principal-components.md) · [find which series leads](how-to/find-which-series-leads.md) |
| [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) | every feature and target pair's moments, slope and t, kept in the state and read back as a table | [entry](MODELS.md#marginal--every-pairs-moments-kept-in-the-state) · [screen features against a target](how-to/screen-features-against-a-target.md) |
| [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) | one correlation for the whole matrix, or one per block and pair of blocks | [entry](MODELS.md#deco--one-correlation-for-the-whole-matrix) · [track the average correlation](how-to/track-the-average-correlation.md) |
| [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) | a block's realized covariance, robust to microstructure noise | [entry](MODELS.md#rcov--a-blocks-realized-covariance-robust-to-noise) · [a block's covariance from ticks](how-to/estimate-a-blocks-covariance-from-ticks.md) |
| [`po.spec.audit`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit) | a count of what each column holds, read by `ModelBank.check` | [entry](MODELS.md#audit--what-a-streams-columns-hold) · [check data quality](how-to/check-data-quality.md) |
| [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) | exponentially weighted k-means, each row labeled before it is learned | [entry](MODELS.md#kmeans--exponentially-weighted-k-means) · [keep centers on the data](how-to/keep-kmeans-centers-on-the-data.md) |
| [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) | density-based clustering into clusters of any shape and number | [entry](MODELS.md#micro--density-based-clustering-any-shape) · [choose `eps`](how-to/choose-micro-eps.md) |
| [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) | Gaussian classification, one set of running moments per class | [entry](MODELS.md#ew_class--gaussian-classification-on-ew_cov-moments) · [choose a covariance type](how-to/choose-a-covariance-type.md) |
| [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) | a sequential test of a sign by betting, read at any row | [entry](MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) · [read a sequential test](how-to/read-a-sequential-test.md) · [compare two models](how-to/compare-two-models.md) |
| [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) | tests of whether the correlations between the features have changed | [entry](MODELS.md#corrchange--has-the-correlation-structure-changed) · [detect a break](how-to/detect-a-break.md) |
| [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) | Bayesian online changepoint detection: how long the current regime has lasted | [entry](MODELS.md#bocpd--how-long-has-this-regime-lasted) · [choose a changepoint hazard](how-to/choose-a-changepoint-hazard.md) |
| [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) | a Gaussian hidden Markov model, filtered as the stream runs | [entry](MODELS.md#hmm--which-regime-are-we-in) · [keep its states on their regimes](how-to/keep-hmm-states-on-their-regimes.md) |

## Running a bank

A [model bank](GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set of
specs together over the same rows. It runs inside a Polars query, in your
own loop, or from the command line ([The command line](#the-command-line)).

| feature | what it does | shown in |
|---|---|---|
| [`lf.online.fit_predict`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict) | adds each spec's output column to a query's rows, learning them as the query runs | [Fit a model bank over parquet files in a query](tutorials/stream-from-parquet.md) · [Run a bank in a query](how-to/run-a-bank-in-a-query.md) |
| [`lf.online.predict`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.predict) | predicts a query's rows from a saved state or a bank, learning nothing | [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| [`lf.online.unnest`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.unnest) | turns each spec's output column into one column per field | [Run a bank in a query](how-to/run-a-bank-in-a-query.md) |
| [`po.fit_predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.fit_predict), [`po.predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict), [`po.unnest`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest) | the same three as functions that take the frame first, which a type checker can see | [Run a bank in a query](how-to/run-a-bank-in-a-query.md#call-the-functions-a-type-checker-sees) |
| [`po.ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank) | a set of specs fitted together, fed one chunk at a time from your own code | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit_predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.fit_predict) | learns a frame's rows and returns them with each spec's output | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit_predict_batches`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.fit_predict_batches) | learns a query or a frame one chunk at a time, and yields each chunk's output | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.fit`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.fit) | learns every row and keeps only the state, for a run whose product is the state | [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.predict) | predicts a frame's rows from the bank as it stands, learning nothing | [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| [`ModelBank.fit_predict_arrow`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.fit_predict_arrow), [`ModelBank.predict_arrow`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.predict_arrow), [`po.ArrowStruct`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ArrowStruct) | each spec's output as an Arrow struct array, read once by any consumer of the Arrow PyCapsule interface | [Hand the output to Arrow](how-to/hand-the-output-to-arrow.md) |
| [`po.thread_pool_size`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size), [`ModelBank.threads`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.threads) | the number of threads a bank runs on, and the `threads` it was made with | [Use more cores](how-to/use-more-cores.md) |
| `chunk_size` | the rows a query feeds a bank at a time, which never changes the numbers | [Set the chunk size](how-to/set-the-chunk-size.md) |

## Saving, loading and serving

A bank's state, everything it has learned, can be saved to one file and
loaded back, on any operating system, to go on learning or to serve.

| feature | what it does | shown in |
|---|---|---|
| [`ModelBank.save`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.save), [`ModelBank.save_bytes`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.save_bytes) | writes the state, everything the bank has learned, to a file or to bytes; a query takes `save_state=` | [Save a state](how-to/save-a-state.md) |
| [`ModelBank.load`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.load), [`ModelBank.load_bytes`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.load_bytes) | reads a state back, on any operating system; a query takes `load_state=` | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [Save a state](how-to/save-a-state.md) |
| [`ModelBank.skip_learned`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.skip_learned) | keeps only the rows of an input the state has not learned, to resume on input that overlaps it | [Resume from a saved state](how-to/resume-from-a-saved-state.md) |
| [`ModelBank.write_json`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.write_json) | writes the state as JSON, for a program without this library | [Read a state without this library](how-to/read-a-state-without-this-library.md) |

## Reading a bank

A bank describes what it holds, live or loaded from its file, and names
the fields of its output.

| feature | what it does | shown in |
|---|---|---|
| [`ModelBank.coef`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.coef) | every fit's coefficients, one row per coefficient | [Read the coefficients](how-to/read-the-coefficients.md) |
| [`ModelBank.groups`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.groups), [`ModelBank.drop_groups`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.drop_groups) | the groups the bank holds a model for, and their removal | [Fit one model per group](how-to/fit-one-model-per-group.md) · [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md) |
| [`ModelBank.summary`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.summary), [`ModelBank.describe`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.describe), [`ModelBank.last_row`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.last_row) | what each model was fed and how ready it is, each input column's statistics, and each model's last output row | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.specs`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.specs), [`ModelBank.rows_fed`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.rows_fed), [`ModelBank.output_fields`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.output_fields) | the specs a bank runs with every field filled in, the rows fed so far, and each spec's output fields | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.solve_failures`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.solve_failures) | how many of each model's solves needed help | [Inspect a bank](how-to/inspect-a-bank.md) |
| [`ModelBank.closed_groups`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.closed_groups) | one row per closed group, with its running sums | [Close groups and read each one's row](how-to/close-groups.md) |
| [`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram) | the Gram behind a spec's fit, per group, as numpy arrays | [Keep a Gram per block in one pass](how-to/fit-weekly-grams-in-one-pass.md) · [Store a compact Gram](how-to/store-a-compact-gram.md) |
| [`ModelBank.marginal`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.marginal) | the feature and target pairs a `marginal` spec keeps, one row per pair | [Screen features against a target](how-to/screen-features-against-a-target.md) |
| [`ModelBank.audit`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.audit), [`ModelBank.check`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.check) | what an `audit` spec counted, and the findings about the data each model was fed, the worst first | [Check a stream's data quality](how-to/check-data-quality.md) · [DATA-QUALITY.md](DATA-QUALITY.md) |
| [`po.spec.output_fields`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.output_fields), [`po.spec.output_index`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.output_index) | the names of the fields a spec writes, and the values each name encodes | [Find an output field by what it holds](how-to/find-an-output-field.md) · [OUTPUTS.md](OUTPUTS.md) |
| [`po.spec.coef_fields`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.coef_fields), [`po.spec.coef_index`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.coef_index) | every coefficient a spec reports, with its place in `coef` and its unnested column | [Find an output field by what it holds](how-to/find-an-output-field.md) · [Read the coefficients](how-to/read-the-coefficients.md) |

## Preparing a stream

The window operators return Polars expressions, computed over a window of
the clock and run in `lf.online.with_windows`. `ewm_mean`, `ewm_sum`,
`ewm_rate` and `ewm_quantile` look back from a row, or ahead under
`reverse=True`, and a look-ahead one can be a spec's target. `ewm_var` and
`ewm_std` look back only ([WINDOWS.md](WINDOWS.md) describes each).

| feature | what it does | shown in |
|---|---|---|
| [`po.ewm_mean`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_mean), also `po.ops.ewm_mean` | the time-weighted mean of a column over each row's window | [Make features that look back](how-to/make-window-features.md) · [Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md) |
| [`po.ewm_sum`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_sum), also `po.ops.ewm_sum` | the weighted sum over each row's window, such as the two sums of a VWAP | [Make features that look back](how-to/make-window-features.md#make-a-weighted-mean-such-as-a-vwap) |
| [`po.ewm_rate`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_rate), also `po.ops.ewm_rate` | a weighted sum per unit of clock time, such as volume per second | [WINDOWS.md](WINDOWS.md) |
| [`po.ewm_var`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_var), [`po.ewm_std`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_std), also `po.ops.ewm_var` and `po.ops.ewm_std` | the variance and standard deviation about the time-weighted mean | [WINDOWS.md](WINDOWS.md) · [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md) |
| [`po.ewm_quantile`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.ewm_quantile), also `po.ops.ewm_quantile` | a weighted quantile of the window's values, such as a median | [Weight rows by a robust scale](how-to/weight-rows-by-a-robust-scale.md) |
| [`po.increment`](https://hgilde.github.io/polars-online/ops.html#polars_online.ops.increment), also `po.ops.increment` | a column's step from its last value within the group and session | [WINDOWS.md](WINDOWS.md) |
| [`lf.online.with_windows`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.with_windows), [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | adds window columns to a stream in one pass, before the bank | [Make features that look back](how-to/make-window-features.md) · [Resume a window run](how-to/resume-a-window-run.md) |
| [`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time) | puts series that tick at their own times on one refresh grid | [Align series that tick at their own times](how-to/align-series-that-tick-at-their-own-times.md) |
| [`po.stream.embargo`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.embargo), [`po.stream.ROLE`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.ROLE) | doubles a stream so each row's target is learned a delay after its row, marking each copy in a column named by `ROLE` | [Learn targets known late](how-to/learn-targets-known-late.md) |
| [`po.target`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.target) | a target column whose output fields take a name of your own | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`po.Target`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.Target), [`po.FormulaTarget`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.FormulaTarget) | the types of a named target and of a formula target written as a table, as a spec dict and the command line's TOML hold them | [Specs in TOML](RUNNER.md#specs-in-toml), which writes a formula target |

## Scoring and comparing

`po.eval` reads a bank's output, in memory or as sums kept chunk by chunk.

| feature | what it does | shown in |
|---|---|---|
| [`po.eval.metrics`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.metrics), [`po.eval.window_metrics`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.window_metrics), [`po.eval.compare_specs`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.compare_specs) | a spec's out-of-sample scores over a frame, in windows of the clock, and several specs' in one table | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`po.eval.sums`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.sums), [`po.eval.merge_sums`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.merge_sums), [`po.eval.from_sums`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.from_sums), [`po.eval.SUM_FIELDS`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.SUM_FIELDS) | scores of an output too large to hold, from sums kept chunk by chunk, and the names of those sums | [Evaluate an output frame](how-to/evaluate-an-output-frame.md#score-an-output-too-large-to-hold) |
| [`po.eval.unpack`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.unpack), [`po.eval.RESERVED`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.RESERVED) | a spec's output in long form, one row per input row and slot, and the column names it takes for itself | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`po.eval.grid_search`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.grid_search) | scores every point of a grid in one pass, and chooses from the rows before each test span | [CALIBRATION.md](CALIBRATION.md) |
| [`po.eval.diebold_mariano`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.diebold_mariano), [`po.eval.clark_west`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.clark_west), [`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest) | tests of whether one model predicts better than another, and a sequential test of a sign over a frame | [Compare two models](how-to/compare-two-models.md) |

## Fits from Grams

A Gram is the running sums a least-squares fit is solved from. `po.gram`
fits from the Grams a bank keeps, without reading the rows again
([GRAM.md](GRAM.md)).

| feature | what it does | shown in |
|---|---|---|
| [`po.gram.Gram`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.Gram), [`po.gram.from_row`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.from_row), [`po.gram.INTERCEPT`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.INTERCEPT) | the mapping a bank's Gram is returned as, a closed group's row read as one, and the intercept column's name | [Keep a Gram per block in one pass](how-to/fit-weekly-grams-in-one-pass.md) · [Close groups](how-to/close-groups.md) |
| [`po.gram.merge`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.merge), [`po.gram.subset`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.subset) | pools the Grams of several blocks, and keeps some of a Gram's columns | [Pool Grams](how-to/pool-grams.md) |
| [`po.gram.solve`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.solve), [`po.gram.solve_subsets`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.solve_subsets), [`po.gram.coef_stats`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.coef_stats) | ridge coefficients from a Gram, for many subsets of blocks at once, and their standard errors and t | [Fit a ridge from a Gram](how-to/fit-a-ridge-from-a-gram.md) · [Select columns stably over weeks](how-to/select-columns-stably-over-weeks.md) |
| [`po.gram.lasso_path`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.lasso_path), [`po.gram.lars_path`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.lars_path), [`po.gram.lars_paths`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.lars_paths) | the elastic-net path by coordinate descent, and the lasso path by least angle regression | [Choose columns on the lasso path](how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path) |
| [`po.gram.condition`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.condition), [`po.gram.vif`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.vif), [`po.gram.correlation`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.correlation) | collinearity diagnostics, variance inflation factors and the correlation matrix of a Gram's columns | [Check a fit is determined](how-to/check-a-fit-is-determined.md) |

## Correlation matrices

`po.corr` reads a correlation matrix from an array, a Gram or a closed
group's row.

| feature | what it does | shown in |
|---|---|---|
| [`po.corr.nearest`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.nearest), [`po.corr.shrink`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.shrink) | the nearest correlation matrix, and shrinkage toward a structured target | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md) |
| [`po.corr.loss`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.loss), [`po.corr.fisher_se`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.fisher_se), [`po.corr.to_z`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.to_z), [`po.corr.from_z`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.from_z), [`po.corr.Z_CLIP`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.Z_CLIP) | a forecast matrix scored against the realized one, a correlation's standard error, and Fisher's transform and its inverse, clipped at `Z_CLIP` | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md) |
| [`po.corr.spectral`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.spectral), [`po.corr.from_spectral`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.from_spectral), [`po.corr.mp_edge`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.mp_edge), [`po.corr.mp_density`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.mp_density) | the top eigenpairs, a matrix rebuilt from them, and the Marchenko-Pastur noise edges and density | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`po.corr.absorption`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.absorption), [`po.corr.absorption_shift`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.absorption_shift) | the absorption ratio, and its standardized shift | [Repair and score a correlation matrix](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`po.corr.equicorr`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.equicorr), [`po.corr.equicorr_row`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.equicorr_row), [`po.corr.equicorr_loglik`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.equicorr_loglik), [`po.corr.block_means`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.block_means), [`po.corr.from_blocks`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.from_blocks), [`po.corr.signal_share`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.signal_share) | one correlation for a whole matrix or per block, and how much of its movement across blocks is not noise | [Track the average correlation](how-to/track-the-average-correlation.md) · [Choose another function](how-to/repair-and-score-a-correlation-matrix.md#choose-another-function) |
| [`po.corr.epps_invert`](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.epps_invert) | the correlation over several rows from lagged co-moments, undoing the Epps effect | [REGIMES.md](REGIMES.md) |

## Trees and simulation

Two modules sit beside the bank: one builds training data for
gradient-boosted trees, and one simulates streams whose truth is known.

| feature | what it does | shown in |
|---|---|---|
| [`po.gbdt.xgboost_matrix`](https://hgilde.github.io/polars-online/gbdt.html#polars_online.gbdt.xgboost_matrix), [`po.gbdt.lightgbm_dataset`](https://hgilde.github.io/polars-online/gbdt.html#polars_online.gbdt.lightgbm_dataset) | XGBoost's and LightGBM's training objects, built from files too large to hold | [GBDT.md](GBDT.md) · [How `po.gbdt` builds training data](explanation/how-po-gbdt-builds-training-data.md) |
| [`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes) | a simulated stream whose correlation changes by regime, with its truth | [Test a detector on known truth](how-to/test-a-detector-on-known-truth.md) · [REGIMES.md](REGIMES.md) |

## Parameters every spec takes

Every model takes these, apart from those its docstring says it refuses
([What every spec takes](MODELS.md#what-every-spec-takes)). A model's own
parameters are on its builder's reference page.

| family | parameters | what they set | shown in |
|---|---|---|---|
| the columns | `targets`, `features`, `fit_intercept` | what the model reads, and whether the fit has a level of its own | [What a spec names](MODELS.md#what-a-spec-names) |
| the clock and its decay | `clock`, `half_life`, `lam`, `gap_cap` | how far apart two rows are, how fast the past fades, and the longest step a gap counts as | [The clock and its units](MODELS.md#the-clock-and-its-units) · [Time and decay](explanation/time-and-decay.md) |
| the clock's policy | `session`, `session_gap`, `restart_after_step_back` | what a session boundary and a step back do | [Handle sessions, gaps and steps back](how-to/handle-sessions-gaps-and-steps-back.md) |
| groups | `group`, `group_close` | one model per value of a column, and when a group closes | [Fit one model per group](how-to/fit-one-model-per-group.md) · [Close groups](how-to/close-groups.md) |
| weights | `weight` | how much each row counts | [Weight rows](how-to/weight-rows.md) |
| warm-up | `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_error_inflation` | when a model has seen enough to report | [Warm-up gates](MODELS.md#warm-up-gates) · [Decide when a model is ready](how-to/decide-when-a-model-is-ready.md) |
| targets known late | `embargo`, `label_clock`, `max_held_rows`, `held_overflow` | when each row's target is learned, and how many rows may wait | [Learn targets known late](how-to/learn-targets-known-late.md) |
| coefficients | `coef_every`, `max_rows_between_coefs`, `emit_se_coef` | which rows carry the coefficients, and their standard errors | [Read the coefficients](how-to/read-the-coefficients.md) · [Check standard errors](how-to/check-standard-errors.md) |
| clocks in the output | `emit_clocks` | each row's clock, and the clock it was learned at | [Learn targets known late](how-to/learn-targets-known-late.md#give-every-row-the-same-delay) |
| the residuals | `emit_sigma`, `emit_zscore`, `emit_metrics`, `emit_autocorr`, `resid_autocorr_lag`, `resid_quantiles` | the residuals' spread, each one's z-score, running scores, autocorrelation and quantiles | [DIAGNOSTICS.md](DIAGNOSTICS.md) |
| intervals | `conformal`, `conformal_rate` | a conformal interval on each prediction | [Put an interval on a prediction](how-to/put-an-interval-on-a-prediction.md) |
| a grid's choice | `emit_selected`, `emit_averaged`, `average_eta` | the grid point that has predicted best so far, or the grid's predictions averaged, as the stream runs | [Choose among a grid's settings](how-to/choose-among-a-grid-online.md) |
| drift | `emit_drift`, `drift_delta`, `drift_threshold`, `drift_action` | a test of the residuals' level, and whether a flag resets the model | [Detect a break](how-to/detect-a-break.md) · [Measure heavy tails](how-to/measure-heavy-tails.md) |
| calibration | `emit_calibration`, `calibration_half_life` | whether the predictions are the right size | [Check the scale](how-to/check-the-scale.md) |
| breaks | `emit_breaks`, `breaks_half_life` | tests of whether the relationship has broken | [Detect a break](how-to/detect-a-break.md) · [Has it broken?](DIAGNOSTICS.md#has-it-broken) |
| robust standard errors | `emit_robust_se`, `robust_se_half_life`, `robust_se_lags` | standard errors that allow for uneven and correlated residuals | [Check standard errors](how-to/check-standard-errors.md) · [Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted) |
| specification | `emit_specification`, `specification_half_life`, `ljung_box_lags` | tests of what a linear model leaves out | [Find what a model misses](how-to/find-what-the-model-misses.md) · [What is it missing?](DIAGNOSTICS.md#what-is-it-missing) |
| a target's horizon | `horizon_rows` | the rows a target looks ahead, which the diagnostics allow for | [Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted) |
| tails | `emit_tails`, `tails_half_life` | how heavy the residuals' tails are | [Measure heavy tails](how-to/measure-heavy-tails.md) · [Are its tails heavy?](DIAGNOSTICS.md#are-its-tails-heavy) |
| influence | `emit_influence`, `influence_half_life` | how far each row moved the fit | [Find an influential row](how-to/find-an-influential-row.md) · [Which row moved it?](DIAGNOSTICS.md#which-row-moved-it) |
| feature health | `emit_feature_health`, `feature_health_half_life` | whether a feature has frozen or moved | [Check feature health](how-to/check-feature-health.md) |

## Errors and warnings

Each error and warning says where it happened, what is wrong and what to
do. The library's errors subclass Python's and Polars' own, and its
warnings Python's, so a handler you already have catches them.

| feature | what it is | shown in |
|---|---|---|
| [`po.ColumnNotFoundError`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ColumnNotFoundError) | a column a call reads is not in its input | [What the bank refuses](MODELS.md#what-the-bank-refuses) |
| [`po.OutputColumnNotFoundError`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.OutputColumnNotFoundError) | a spec's output column is not in the frame a scoring function of `po.eval` reads; a `ColumnNotFoundError`, and until the next major version a `KeyError` too | [its reference entry](https://hgilde.github.io/polars-online/api/polars_online.OutputColumnNotFoundError.html) · [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |
| [`po.WrongTypeError`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.WrongTypeError), [`po.RefusedValueError`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.RefusedValueError) | a value of the wrong type, a `TypeError`, and a refused value, a `ValueError`; until the next major version each also keeps the other class as a parent | [What the bank refuses](MODELS.md#what-the-bank-refuses) |
| [`po.PolarsOnlineWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.PolarsOnlineWarning) | the base class of every warning the library issues, so `warnings.filterwarnings("ignore", category=po.PolarsOnlineWarning)` silences them all | [its reference entry](https://hgilde.github.io/polars-online/api/polars_online.PolarsOnlineWarning.html) |
| [`po.ReadinessWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ReadinessWarning) | a model is not ready, or may never become ready | [Decide when a model is ready](how-to/decide-when-a-model-is-ready.md) |
| [`po.OrderNotGuaranteedWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.OrderNotGuaranteedWarning) | Polars may change the row order of the query a bank reads | [Put rows in clock order](how-to/put-rows-in-clock-order.md) |
| [`po.ConsumedSourceWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ConsumedSourceWarning) | a query whose source is written in Python may give its rows only once | [Feed a bank from a database](how-to/feed-a-bank-from-a-database.md) |
| [`po.UnstableWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.UnstableWarning), [`po.PolarsOnlineDeprecationWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.PolarsOnlineDeprecationWarning) | a part of the API not yet promised to stay the same was used, or a renamed name by its old spelling | [VERSIONS.md](VERSIONS.md) |

## Versions and the environment

The library reports its versions, and reads three environment variables.

| feature | what it is | shown in |
|---|---|---|
| `po.__version__`, [`po.native_version`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.native_version) | the package's version, and its compiled extension's, which match in a consistent install | [`native_version`'s reference entry](https://hgilde.github.io/polars-online/api/polars_online.native_version.html) · [VERSIONS.md](VERSIONS.md), what a version promises |
| [`po.schema_version`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.schema_version) | the version of the state file's layout this build writes | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [A bank's state and resuming it](explanation/state-and-resuming.md) |
| `POLARS_ONLINE_MAX_THREADS` | the number of threads a bank runs on when it is given no `threads` | [Use more cores](how-to/use-more-cores.md) |
| `POLARS_ONLINE_TIMING` | prints where a run's time went, stage by stage | [PERFORMANCE.md](PERFORMANCE.md) |
| `POLARS_ONLINE_WARN_UNSTABLE` | turns on an `UnstableWarning` for each use of an unstable part of the API | [VERSIONS.md](VERSIONS.md) |

## The command line

`polars-online` runs a bank from a file to a file, with no Python, from a
TOML configuration file. Each `[[specs]]` table takes a spec's `name` and
the keywords every spec shares, and a `[specs.model]` table under it names
the model with `type` and takes the model's own keywords ([Run a bank from the command
line](how-to/run-a-bank-from-the-command-line.md), and every key and flag
in [RUNNER.md](RUNNER.md)).

| feature | what it does | shown in |
|---|---|---|
| `--config`, and the keys `input`, `output`, `specs` | the configuration file, the files it reads and writes, and its specs | [Run a bank from the command line](how-to/run-a-bank-from-the-command-line.md) |
| `--input`, `--output`, `--no-output`, `--input-format`, `--output-format`, `--output-layout`, and the keys `input_format`, `output_format`, `output_layout`, `keep_columns` | the files read and written, their formats, flat columns or one per spec, and which input columns pass through | [Flags and the TOML keys they override](RUNNER.md#flags-and-the-toml-keys-they-override) · [Input and output files](RUNNER.md#input-and-output-files) |
| `--chunk-size`, `--threads`, and the keys `chunk_size`, `threads` | the rows read at a time, and the bank's threads | [Memory, threads and chunk size](RUNNER.md#memory-threads-and-chunk-size) |
| `--load-state`, `--save-state`, `--skip-learned`, `--predict`, and the keys `load_state`, `save_state`, `skip_learned`, `predict` | resume from a state, save one, skip the rows a state has learned, and serve without learning | [Resume from a saved state](how-to/resume-from-a-saved-state.md) · [Serve from a saved state](how-to/serve-from-a-saved-state.md) |
| `--closed-groups`, and the key `closed_groups` | writes each closed group's row to a file of its own as the run goes | [Closed groups to a sidecar file](RUNNER.md#closed-groups-to-a-sidecar-file) |
| `--dry-run`, `--quiet` (`-q`) | checks the configuration and the input without fitting, and turns the progress line off | [Run a bank from the command line](how-to/run-a-bank-from-the-command-line.md) |
| `--help` (`-h`), `--version` (`-V`) | the flags, and the program's version | [RUNNER.md](RUNNER.md) |
