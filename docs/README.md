<!-- kind: map -->
# The documents, and which to read for what

This page lists every document of polars-online by kind, for a reader
deciding which to read. The [README](../README.md) is the landing page:
what polars-online is, how to install it, a first fit, and a map from each
task to its pages.
[FEATURES.md](FEATURES.md) lists every feature, one line each, and
[HIGHLIGHTS.md](HIGHLIGHTS.md) runs four use cases end to end. The API
reference, built from the docstrings, is at
<https://hgilde.github.io/polars-online/>, and its Sphinx sources are in
[`reference/`](reference/). Every other document under `docs/` is one of
two kinds. A **guide** is what you read to use or change the library. A
**record** says how a decision was reached, and is kept because the code
or another document cites it. A record that nothing outside `docs/` cites
is kept in [`records/`](records/), and its row links it there.

## Guides

Each guide for using the library is one of four kinds, and is grouped
under its kind. A **tutorial** takes you from nothing to one working
result, and a **how-to** solves one task. A **reference** page describes
one thing completely, and an **explanation** gives the reasons and the
mechanism.

| document | read it when |
|---|---|
| **Start here** | |
| [FEATURES.md](FEATURES.md) | you want the whole list of what the library does, or a feature by its name — every model, function, method, shared parameter and command-line flag, one line each with a link to the page that shows it |
| [HIGHLIGHTS.md](HIGHLIGHTS.md) | you want to see what the library does for a task like yours — four use cases run end to end on data with a planted answer (a round of model selection, a forward-looking signal built and resumed in a fresh process, prediction intervals on late-arriving residuals, and tick data), and an index of the capabilities they show, each with the step that shows it and a guide |
| [GLOSSARY.md](GLOSSARY.md) | you meet a word a page uses without defining it, or you choose the word for a sentence, a docstring or a message — every concept's one word, what it means, the words it replaces and whether a message may use it, with the words kept out of user-facing text and the American spelling of each British one the docs held |
| **Tutorials** | |
| [tutorials/stream-from-parquet.md](tutorials/stream-from-parquet.md) | you are new to the library and want to fit a model over parquet files too large to hold, in a Polars query, and write its predictions to a file |
| [tutorials/local-fit-with-a-clock.md](tutorials/local-fit-with-a-clock.md) | you are new to the library and want a regression that follows the recent rows along a clock, through a trading day and across the next day's open |
| [tutorials/fit-save-and-serve.md](tutorials/fit-save-and-serve.md) | you are new to the library and want to fit a model, save it, load it in a new Python process and predict new rows from it |
| **How-tos: running a bank** | |
| [how-to/run-a-bank-in-a-query.md](how-to/run-a-bank-in-a-query.md) | you run a bank inside a Polars query — how to choose the chunk size, unnest a wide spec before writing, filter after the bank, catch a misnamed column, and call the functions a type checker sees |
| [how-to/run-a-bank-in-a-python-loop.md](how-to/run-a-bank-in-a-python-loop.md) | you feed a bank from your own loop — how to read it between chunks, feed it from one thread, and drop groups that have gone quiet |
| [how-to/run-a-bank-from-the-command-line.md](how-to/run-a-bank-from-the-command-line.md) | you want to run a bank as a scheduled job with the `polars-online` program and a TOML file, with no Python |
| [how-to/hand-the-output-to-arrow.md](how-to/hand-the-output-to-arrow.md) | you want a bank's output in a library that is not Polars, such as pyarrow or DuckDB — how to export it with `fit_predict_arrow` and read each export once |
| [how-to/feed-a-bank-from-a-database.md](how-to/feed-a-bank-from-a-database.md) | your rows live in DuckDB or behind an ADBC driver — how to fit a bank on a query's result, and the cursor traps to avoid |
| [how-to/run-a-bank-in-pathway.md](how-to/run-a-bank-in-pathway.md) | you run a Pathway pipeline — how to wrap a bank in an operator of your own, feed it chunks, and save and restore it as bytes for your pipeline's persistence |
| [how-to/set-the-chunk-size.md](how-to/set-the-chunk-size.md) | you want a run faster or holding less memory — how to set how many rows a bank takes at a time, and check that the predictions did not move |
| [how-to/use-more-cores.md](how-to/use-more-cores.md) | your bank runs on fewer cores than the machine has, and you want to give it more work it can run at once |
| [how-to/fit-hundreds-of-features.md](how-to/fit-hundreds-of-features.md) | your `ewridge` or `ew_cov` reads hundreds or thousands of features — how to add rows to the Gram in blocks with `gram_block_rows` and run that work on idle threads with `gram_threads` |
| [how-to/bound-memory.md](how-to/bound-memory.md) | your run holds more memory than you can give it, and you want Polars' settings and the robust models' refit budget that bound it |
| **How-tos: choosing a model and what it reads** | |
| [how-to/choose-a-model.md](how-to/choose-a-model.md) | you are about to write a spec and do not know which model to name — how to choose the family by your question and the model by your data, and which of the four models that watch for change answers which question |
| [how-to/fit-one-model-per-group.md](how-to/fit-one-model-per-group.md) | you want a separate model for each value of a column, such as one per stock, fitted in one pass |
| [how-to/close-groups.md](how-to/close-groups.md) | your stream keeps bringing new group values — how to set `group_close`, read `closed_groups()`, and write closed rows to a file as the run goes |
| [how-to/weight-rows.md](how-to/weight-rows.md) | you want some rows to count more than others — how to name a weight column, scale `min_weight` with it, and which models read a weight differently |
| [how-to/hold-a-row-back.md](how-to/hold-a-row-back.md) | you want rows predicted but not learned — how to use a null target, weight 0 or `predict`, and what a null skips |
| [how-to/fit-targets-that-go-missing-on-different-rows.md](how-to/fit-targets-that-go-missing-on-different-rows.md) | you fit several targets on the same features and they are null on different rows |
| [how-to/fit-a-count-target.md](how-to/fit-a-count-target.md) | your target is a count, and you want a Poisson fit with `sgd` |
| [how-to/bound-the-coefficients.md](how-to/bound-the-coefficients.md) | you need a regression's slopes held to bounds or to a fixed total, with `coef_min`, `coef_max` and `coef_sum` |
| [how-to/follow-a-level-that-moves.md](how-to/follow-a-level-that-moves.md) | you fit `pa` or `sgd` on a target whose level moves, and want the fit to keep up |
| **How-tos: time, order and late targets** | |
| [how-to/handle-sessions-gaps-and-steps-back.md](how-to/handle-sessions-gaps-and-steps-back.md) | you fit a model with a clock on a stream with session boundaries, quiet spells or replays — how to choose what each does to the fit |
| [how-to/put-rows-in-clock-order.md](how-to/put-rows-in-clock-order.md) | you fit a bank on a stream with a clock — how to give it each group's rows in clock order, read its warning about a query's row order, and fix a chunk it refused |
| [how-to/forget-rows-past-a-window.md](how-to/forget-rows-past-a-window.md) | you want rows older than a set span to count for nothing, with `window_size` |
| [how-to/fit-locally-along-a-feature.md](how-to/fit-locally-along-a-feature.md) | you want a fit of the rows near each value of a feature, such as a curve — how to sort by the feature and use it as the clock, with `half_life` as the bandwidth |
| [how-to/learn-targets-known-late.md](how-to/learn-targets-known-late.md) | your target is known only some time after its row, and you want each row predicted where it sits and learned once its target is known |
| **How-tos: features and targets** | |
| [how-to/make-window-features.md](how-to/make-window-features.md) | you want features that look back over the stream, made as columns before the bank |
| [how-to/make-a-target-that-looks-ahead.md](how-to/make-a-target-that-looks-ahead.md) | you want a target computed from the rows after each row, as a column or as a formula target the bank holds until its window closes |
| [how-to/scale-a-feature-by-its-spread.md](how-to/scale-a-feature-by-its-spread.md) | you want a coefficient per current standard deviation of a feature, from its z-score against its own recent spread |
| [how-to/weight-rows-by-a-robust-scale.md](how-to/weight-rows-by-a-robust-scale.md) | you want each row to count by a robust scale of its target, without reading targets not yet known |
| [how-to/align-series-that-tick-at-their-own-times.md](how-to/align-series-that-tick-at-their-own-times.md) | you have series observed at instants of their own — how to put them on one refresh grid with `po.stream.refresh_time` and fit on its changes |
| [how-to/fit-on-factor-residuals.md](how-to/fit-on-factor-residuals.md) | you fit factor ridges and want their out-of-sample residuals as features of a second ridge with a target that looks ahead |
| **How-tos: saving, resuming and serving** | |
| [how-to/save-a-state.md](how-to/save-a-state.md) | you want to keep what a bank learned in a file, from a bank object, a query or the command line — how to save it and load it back |
| [how-to/resume-from-a-saved-state.md](how-to/resume-from-a-saved-state.md) | you have a saved state and want the bank to go on learning, from Python or the command line — including on input that overlaps what it learned |
| [how-to/serve-from-a-saved-state.md](how-to/serve-from-a-saved-state.md) | you have a saved state and want predictions for new rows without the bank learning from them |
| [how-to/read-a-state-without-this-library.md](how-to/read-a-state-without-this-library.md) | you want another program, in any language, to read a bank's state — how to export it as JSON and read back NaN and inf |
| [how-to/reproduce-an-earlier-versions-fit.md](how-to/reproduce-an-earlier-versions-fit.md) | you need the numbers an earlier release gave, and want the settings that restore them |
| [how-to/resume-a-window-run.md](how-to/resume-a-window-run.md) | you feed `with_windows` a stream in parts, such as a file a day — how to save its state at the end of one part, load it at the next, and check which inputs a resumed run takes or refuses |
| **How-tos: reading a fit** | |
| [how-to/inspect-a-bank.md](how-to/inspect-a-bank.md) | you want to know what a bank holds and what it was fed, live or from a state file — how to read `repr`, `groups()`, `coef()`, `summary()`, `describe()` and `last_row()`, and stack tables from several runs |
| [how-to/find-an-output-field.md](how-to/find-an-output-field.md) | you read the output by field name — how to find a field's or a coefficient's name from the values it holds, with `po.spec.output_index` and `coef_fields` |
| [how-to/read-the-coefficients.md](how-to/read-the-coefficients.md) | you want a regression's coefficients — at the end, row by row, from a saved output, or as a matrix, and for each block of a grid |
| [how-to/fit-weekly-grams-in-one-pass.md](how-to/fit-weekly-grams-in-one-pass.md) | you want many least-squares fits over any weeks, hours or columns from one pass — how to keep a Gram per block and pool them |
| [how-to/pool-grams.md](how-to/pool-grams.md) | you hold a Gram per block, such as one per week — how to pool any set into one Gram, aged to a common row, with or without the blocks' levels, and fit or score on it |
| [how-to/fit-a-ridge-from-a-gram.md](how-to/fit-a-ridge-from-a-gram.md) | you want a ridge and its t from Grams — how to fit a grid of ridges in one call, choose one on weeks held out, and read the t |
| [how-to/check-a-fit-is-determined.md](how-to/check-a-fit-is-determined.md) | your fit's columns may be nearly dependent — how to read Belsley's indexes and the VIFs from a Gram, find the columns, and refit without one |
| [how-to/select-columns-stably-over-weeks.md](how-to/select-columns-stably-over-weeks.md) | you want columns that hold across the weeks — stability selection over halves of weekly Grams, a threshold from sign-flip null runs, and a screen per half and null run |
| [how-to/fit-per-block-coefficients.md](how-to/fit-per-block-coefficients.md) | you want to know whether a coefficient holds from week to week — how to fit each block alone and read Fama and MacBeth's t beside the pooled t |
| [how-to/fit-huber-on-a-gram.md](how-to/fit-huber-on-a-gram.md) | you want Huber's converged fit of a stream on disk while factoring its Gram once — how to run algorithm H, two Polars queries and a solve against one factor each pass |
| [how-to/read-grams-from-saved-files.md](how-to/read-grams-from-saved-files.md) | you want to fit from a pass's Grams in a later program that runs no model — how to write each finished week's Gram to a parquet file and the last to a state, and read them back with `po.gram.from_row` and `ModelBank.load`, or with numpy alone |
| [how-to/store-a-compact-gram.md](how-to/store-a-compact-gram.md) | your Grams are too many bytes — how to check that a fit can bear float32's rounding, then ask `bank.gram()` for float32, the upper triangle, or both, a quarter of the bytes |
| [how-to/repair-and-score-a-correlation-matrix.md](how-to/repair-and-score-a-correlation-matrix.md) | you have a correlation matrix, estimated or pieced together pair by pair — how to repair, shrink and score it, and choose among `po.corr`'s functions |
| **How-tos: moments, correlations, clusters and regimes** | |
| [how-to/flag-joint-outliers.md](how-to/flag-joint-outliers.md) | you watch several columns of a stream — how to flag the rows that are odd only in combination, with `ew_cov`'s `mahal` and a threshold from its own past distances |
| [how-to/track-principal-components.md](how-to/track-principal-components.md) | you want the main direction several columns move in together — how to track its loadings, share and scores with `pca`, refreshed on a schedule |
| [how-to/find-which-series-leads.md](how-to/find-which-series-leads.md) | you want to know whether one column moves before another, and by how many rows — how to read `ew_cov`'s lagged correlations in both directions |
| [how-to/screen-features-against-a-target.md](how-to/screen-features-against-a-target.md) | you have many features and a few targets — how to rank every pair in one pass with `marginal`, correct the ranking for smooth rows, and find relations a correlation misses |
| [how-to/track-the-average-correlation.md](how-to/track-the-average-correlation.md) | you watch many series — how to track one number that moves with their average correlation with `deco`, per block, and read its level from an `ew_cov` |
| [how-to/estimate-a-blocks-covariance-from-ticks.md](how-to/estimate-a-blocks-covariance-from-ticks.md) | you have tick prices of several series — how to estimate each block's covariance robust to noise and to series not observed on every row with `rcov` |
| [how-to/keep-kmeans-centers-on-the-data.md](how-to/keep-kmeans-centers-on-the-data.md) | you fit `kmeans` to clusters that come and go — how to keep `init` at its default and set `dead_frac` so a vanished cluster's center is re-placed |
| [how-to/choose-micro-eps.md](how-to/choose-micro-eps.md) | you cluster with `micro` — how to choose `eps`, and read a value too small or too coarse from its outputs |
| [how-to/choose-a-covariance-type.md](how-to/choose-a-covariance-type.md) | you class rows with `ew_class` — how to choose `covariance_type`, and give `window_size` when classes move and `embargo` when they arrive late |
| [how-to/keep-hmm-states-on-their-regimes.md](how-to/keep-hmm-states-on-their-regimes.md) | you fit `hmm` to name regimes — how to give a sticky transition prior and keep an extreme row from capturing a state |
| [how-to/read-a-sequential-test.md](how-to/read-a-sequential-test.md) | you run `seqtest` — how to reject the first time its evidence reaches ln 20, one- or two-sided, however often you look |
| **How-tos: checking the data and the model** | |
| [how-to/check-data-quality.md](how-to/check-data-quality.md) | you have fitted a bank and want to know what is wrong with the data, with `bank.check()` and an `audit` spec, and how to fix it |
| [how-to/decide-when-a-model-is-ready.md](how-to/decide-when-a-model-is-ready.md) | you serve predictions from a model's first rows — how to read why a prediction is withheld, choose a readiness gate, and act on a readiness warning |
| [how-to/check-the-scale.md](how-to/check-the-scale.md) | you want to know whether a model's predictions are the right size — how to read its calibration, rescale a prediction by it, and fix a fit that is too small or too large |
| [how-to/detect-a-break.md](how-to/detect-a-break.md) | you want to know whether a model's relationship has broken, how, and when — how to run the break tests as the stream runs, tell an intercept, noise or slope break apart, date one, and watch the features with `corrchange` and `bocpd` |
| [how-to/check-a-two-stage-fit.md](how-to/check-a-two-stage-fit.md) | you read a second-stage coefficient fitted on a first stage's residual — how to check each stage with calibration and Newey and West's standard error, against a planted answer |
| [how-to/check-standard-errors.md](how-to/check-standard-errors.md) | you read a coefficient's t on a stream — how to compare its plain, HC0 and HAC standard errors and choose the one whose assumptions the residuals meet |
| [how-to/find-what-the-model-misses.md](how-to/find-what-the-model-misses.md) | you want to know what a linear model leaves out — how to run the specification tests, read which one fires, and add the lag or curvature it asks for |
| [how-to/measure-heavy-tails.md](how-to/measure-heavy-tails.md) | you set a threshold on a model's residuals — how to measure their tails over a stable stretch, set `drift_threshold` from the kurtosis, and choose a detector that holds its false-alarm rate |
| [how-to/find-an-influential-row.md](how-to/find-an-influential-row.md) | your fit may have been moved by a few bad rows — how to rank rows by `influence`, flag them against a cutoff that scales with the fit, and decide whether to drop them |
| [how-to/check-feature-health.md](how-to/check-feature-health.md) | your feature columns can go stale — how to watch each feature's spread and level and flag a frozen or moved feature before any coefficient shows it |
| [how-to/compare-two-models.md](how-to/compare-two-models.md) | you have two models on one stream — how to test whether one predicts better, with Diebold and Mariano's test, Clark and West's for nested models, and a `seqtest` read at any row |
| [how-to/evaluate-an-output-frame.md](how-to/evaluate-an-output-frame.md) | you have a finished run's output — how to score it per spec, group or clock window with `po.eval`, and score an output too large to hold from per-chunk sums |
| [how-to/put-an-interval-on-a-prediction.md](how-to/put-an-interval-on-a-prediction.md) | you serve predictions — how to put a Gaussian or conformal interval on each one, and keep its coverage when targets arrive late |
| [how-to/test-a-detector-on-known-truth.md](how-to/test-a-detector-on-known-truth.md) | you fit a regime or break detector — how to simulate a stream with a planted change with `po.sim.regimes` and print what the detector found beside the truth |
| **How-tos: choosing a model's settings** | |
| [CALIBRATION.md](CALIBRATION.md) | you are choosing a model's settings — a half-life, a ridge, a Kalman filter's drift speed, a learning rate, Holt's smoothing or a changepoint hazard — from the stream's own out-of-sample errors in one pass with `po.eval.grid_search`: the pattern, why the best error over the rows you report is optimistic, a recipe for each setting, and the settings it cannot choose |
| [how-to/choose-a-ridge-and-half-life.md](how-to/choose-a-ridge-and-half-life.md) | you fit `ewridge` and want to choose its `ridge`, alone or with its `half_life` — how to search them with `po.eval.grid_search`, and why a ridge carries across feature scales only under `standardize` |
| [how-to/choose-when-ewridge-solves.md](how-to/choose-when-ewridge-solves.md) | you trade how current `ewridge`'s coefficients are against run time — how to set `solve_every` and `max_rows_between_solves`, and keep the solves a group's warm-up skips |
| [how-to/start-from-a-prior-fit.md](how-to/start-from-a-prior-fit.md) | you start `ewridge` from yesterday's fit — how to give it as `coef_prior`, with `ridge_scale="sum"` for a pull that fades or `"mean"` for one that lasts |
| [how-to/blend-each-session-with-the-long-run.md](how-to/blend-each-session-with-the-long-run.md) | you start each session of an `ewridge` fit from a blend with the long run — how to set `session_shrink` and `long_half_life`, and tell whether the blend helps |
| [how-to/choose-a-kalman-drift-speed.md](how-to/choose-a-kalman-drift-speed.md) | you fit `kalman` and want to choose how fast its coefficients may drift, revert or extrapolate |
| [how-to/choose-a-learning-rate.md](how-to/choose-a-learning-rate.md) | you fit `sgd`, `ftrl` or `pa` and want to choose the size of each step |
| [how-to/choose-ftrl-penalties.md](how-to/choose-ftrl-penalties.md) | you fit `ftrl` and want to choose its penalties, which weigh more on a short half-life or a sparse stream |
| [how-to/choose-holts-smoothing.md](how-to/choose-holts-smoothing.md) | you fit `holt` and want to choose how fast its level and trend forget, or a flat prediction |
| [how-to/choose-columns-on-the-lasso-path.md](how-to/choose-columns-on-the-lasso-path.md) | you choose columns with the lasso — on the stream with `lasso` (its penalties, how they are ranked, and its sweeps), or from Grams with `lars_path` and `lasso_path` |
| [how-to/choose-a-changepoint-hazard.md](how-to/choose-a-changepoint-hazard.md) | you fit `bocpd` and want to choose its `hazard`, priors and `robust_beta` — how to score each by its out-of-sample log-likelihood with `grid_search(scoring="loglik")` |
| [how-to/choose-among-a-grid-online.md](how-to/choose-among-a-grid-online.md) | you fit a spec over a grid of settings — how to let it choose or average the grid's predictions as the stream runs, with `emit_selected` and `emit_averaged` |
| [how-to/choose-settings-by-hand.md](how-to/choose-settings-by-hand.md) | you hold a bank's output for a grid and want to choose from it yourself — how to choose on one half with `po.eval.metrics` and report on the other |
| **How-tos: training trees** | |
| [GBDT.md](GBDT.md) | you train XGBoost or LightGBM on files larger than memory — how to build each library's training object with `po.gbdt` from Arrow IPC, Parquet or a query, with categorical features, binning parameters, a Huber loss or an online model's prediction as the margin |
| **Reference** | |
| [MODELS.md](MODELS.md) | you are choosing a model, or want to know how one learns — each model's entry under its family: when to reach for it, its update rule, its parameters and what it is checked against |
| [OUTPUTS.md](OUTPUTS.md) | you want to know what a model writes into its column — each model's fields, generated from the code and checked against it, with where each optional output is shown, such as the diagnostics on [DIAGNOSTICS.md](DIAGNOSTICS.md) |
| [RUNNER.md](RUNNER.md) | you run a bank as a job — the `polars-online` command line from a configuration file — with no live Python process driving it |
| [STATE-WORKFLOW.md](STATE-WORKFLOW.md) | you rely on a saved state across runs — what each step of saving, serving and resuming guarantees, what a saved bank carries, how a window run's state resumes, and the measured fact about Polars each rule rests on |
| [DIAGNOSTICS.md](DIAGNOSTICS.md) | you want to know whether a model is working — one section per question (is it scaled right, has it broken, can its t be trusted, what is it missing, are its tails heavy, which row moved it, are its features healthy, is another model better, is it ready, how wide is its error): each diagnostic's definition, threshold and false-alarm rate, cost and the theory, with a link to the how-to that runs it |
| [DATA-QUALITY.md](DATA-QUALITY.md) | `bank.check()` found a problem in your data, or you want to know what can go wrong before you fit — one section per finding: what it does to each model, the fix in Polars, the theory, and a recipe that runs |
| [REGIMES.md](REGIMES.md) | you are about to trust `hmm`, `corrchange`, `bocpd`, `deco` or `rcov`, or a remedy for series that tick at different times — what each finds on data whose truth is known, and what it misses |
| [VALIDATION.md](VALIDATION.md) | you wonder where a default came from — generated by `scripts/validate.py` on public data; never edited by hand |
| [PERFORMANCE.md](PERFORMANCE.md) | you want a measured cost — rows a second for each model and at each thread count, peak memory of each way of running a bank, what a window operator costs, how cost grows with size — and how each figure was made |
| [VERSIONS.md](VERSIONS.md) | you pin polars-online in your requirements or upgrade it — what each kind of release may change, what is stable, which defaults changed in a minor release, how state files carry across releases, and which Polars versions it runs with |
| [EXAMPLE-DATA.md](EXAMPLE-DATA.md) | you run an example from a how-to, explanation or reference page, or a docstring, that reads `df`, `today`, `trades` or their parquet files — the code that builds them, what each is for, and the answers planted in `df` |
| **Explanations** | |
| [WINDOWS.md](WINDOWS.md) | you make features or targets from windows over the stream and want to know how a window operator weighs the rows near each row — which rows a window holds under `closed`, how each operator weighs them on an uneven clock, why a formula around the operators is element-wise, what a gap, a session or a reset does to a window, and how `po.ewm_quantile` reads its buckets |
| [GRAM.md](GRAM.md) | you fit least squares from the Grams one pass leaves — how a Gram stands in for the rows, which rows it holds, what it cannot answer and what each fit from it assumes, with a table from each task to its how-to |
| [explanation/out-of-sample.md](explanation/out-of-sample.md) | you want to know why each prediction and diagnostic is made before its row is learned, what that guarantees, and how a target that looks ahead is learned |
| [explanation/time-and-decay.md](explanation/time-and-decay.md) | you set `half_life` and want to know how a model forgets along its clock, what `gap_cap` and a session do to the decay, why a huge half-life is not the same as none, and how a hard window forgets differently |
| [explanation/chunk-invariance-and-streaming.md](explanation/chunk-invariance-and-streaming.md) | you want to know why the chunking never changes the numbers, why memory stays bounded, which query steps hold the stream in memory, and why the row order is yours to set |
| [explanation/state-and-resuming.md](explanation/state-and-resuming.md) | you save a bank and load it later — what its state holds, why a resumed run matches one that never stopped, why the file loads on every operating system and in later versions, and the one gap |
| [explanation/warm-up-and-readiness.md](explanation/warm-up-and-readiness.md) | you want to know why a model withholds its first predictions, what each readiness gate measures, why `min_weight` can be out of reach, and how a model that seeds itself from its first rows differs |
| [explanation/robust-scale.md](explanation/robust-scale.md) | you fit `huber`, `quantile` or `sgd` with a Huber loss and want to know how the residuals' scale is kept, why each target's first rows are refit, and what a robust scale gives up when the level shifts |
| [explanation/grids-and-instances.md](explanation/grids-and-instances.md) | you give a spec several values of a setting and want to know how one pass fits them all, what an instance and a slot are, and why the best error over a grid is optimistic |
| [explanation/where-the-time-and-memory-go.md](explanation/where-the-time-and-memory-go.md) | you want to know what sets a bank's speed and its memory — the bank's own work against the models', why memory stays flat as the rows grow, and a state's size measured |
| [explanation/against-scikit-learn.md](explanation/against-scikit-learn.md) | you would otherwise fit a streaming regression with scikit-learn's `SGDRegressor.partial_fit` — how exact solves differ from gradient steps, and where each one wins |
| [explanation/polars-interfaces.md](explanation/polars-interfaces.md) | you upgrade Polars under polars-online and want to know the four ways the library crosses into Polars, which carry a promise, why the wheel carries its own Polars, and how a break shows |
| [explanation/how-the-library-is-tested.md](explanation/how-the-library-is-tested.md) | you are deciding whether to trust the numbers — which test holds each guarantee, the invariants every model keeps, what the suite does to a bank on purpose, and where it runs |
| [explanation/how-po-gbdt-builds-training-data.md](explanation/how-po-gbdt-builds-training-data.md) | you train trees with `po.gbdt` and want to know how it bins rows that do not fit in memory, how its bins compare with each library's own, when a margin beats a feature, and why the kind of file decides its memory |
| [explanation/how-diagnostics-work.md](explanation/how-diagnostics-work.md) | you read the diagnostics a spec writes on every row — how each keeps a memory of its own, when it starts to read, how to read a statistic read on every row, and how a fit built in stages is checked |
| [explanation/how-data-quality-checks-work.md](explanation/how-data-quality-checks-work.md) | you run `bank.check()` — how it finds problems from what a bank keeps anyway, at no cost to the run, and why each finding harms a fit |
| [explanation/asynchronous-series.md](explanation/asynchronous-series.md) | you fit a correlation across series observed at their own times — why they bias it toward zero (the Epps effect), and the two ways out |
| **Changing the library** | |
| [EXTENDING.md](EXTENDING.md) | you are adding a model — every layer it touches, with the test that catches skipping each one |
| [contributing/add-an-output-or-a-parameter.md](contributing/add-an-output-or-a-parameter.md) | you are adding an output field, a spec parameter, a kind of input column or a withheld reason, and want to know what each changes in a saved state |
| [contributing/change-the-api.md](contributing/change-the-api.md) | you are adding, renaming or removing a public name or changing a default, and need the snapshot diff and the right kind of release |
| [contributing/iterate-on-a-change.md](contributing/iterate-on-a-change.md) | you want to run the tests for what you changed while you work, and the full suite before a push |
| [contributing/merge-a-wave-of-branches.md](contributing/merge-a-wave-of-branches.md) | you want to land several finished branches on `main` behind one run of the tests |
| [contributing/use-a-worktree-slot.md](contributing/use-a-worktree-slot.md) | you want to take a warm worktree slot for a task, work in it, and give it back |
| [contributing/clear-the-caches.md](contributing/clear-the-caches.md) | you want to free disk by clearing the build caches, and to know when to |
| [contributing/update-the-oracle-ledger.md](contributing/update-the-oracle-ledger.md) | you want to rerun a grid of independent checks your change moved and record its error, which only tightens |
| [contributing/register-an-oracle-grid.md](contributing/register-an-oracle-grid.md) | you want a new grid of independent checks held to the ledger |
| [TESTING.md](TESTING.md) | you want to know what the suite proves, which tests hold each guarantee, how it runs and where it is thin |
| [contributing/cut-a-release.md](contributing/cut-a-release.md) | you are releasing a version — compare with the last release, benchmark, set the version, dispatch, approve and verify |
| [contributing/move-the-polars-range.md](contributing/move-the-polars-range.md) | a new Polars broke the library, or you need to move the Polars floor or ceiling |
| [RELEASE-READINESS.md](RELEASE-READINESS.md) | you change the release workflow, or want to know what a release is held to before it reaches PyPI |
| [contributing/repository-settings.md](contributing/repository-settings.md) | you want to know how the GitHub repository is set up (rulesets, the PyPI environment, security settings) and why |
| **Writing the documentation** | |
| [WRITING.md](WRITING.md) | you are writing or rewriting any reader-facing doc — the rules the docs are held to, each drawn from a report in [PHRASING.md](records/PHRASING.md), with the README's rewrite of 2026-09-23 as the worked example. A pass over the README starts from [README-ITERATIONS.md](README-ITERATIONS.md), and a new report goes into PHRASING.md |

## Records

A record is not rewritten when the code moves on. Most open with a dated
status line that says what happened next. Where one has no such line, or an
out-of-date one, its row here says what became of it.

| document | what it records |
|---|---|
| **The plan** | |
| [PLAN.md](PLAN.md) | the design and the task list — the source of truth. §1–§10 are the original design. §11 is the task list, ticked as each task lands, and §11a holds the decisions taken while building. §11b–§11h hold one summary per follow-on document below, §12 the questions the design left open and their answers, §13 `window_size`, an exponentially weighted sum with a hard cutoff, and §14 the nine review rounds of the window operators on 2026-10-03, each finding with the test that pins it. From §15 on, each section records one later review or decision, dated in its heading. The first three are the review of the 53 commits after v0.13.0 (§15), the whole-project review of 2026-10-05 (§16), and the parameters counted in rows where a clock would fit (§17) |
| [ENHANCEMENTS.md](ENHANCEMENTS.md) | every model and feature after the first seven (E1–E75): what was proposed, what was measured, what was built and what was declined |
| **Designs, as built** | |
| [ANSWERS-E54-E64.md](records/ANSWERS-E54-E64.md) | the formulae and constants for the correlation and regime models, read from the papers before they were built |
| [MARGINAL-LAGS-AND-BINS.md](MARGINAL-LAGS-AND-BINS.md) | two extensions to `marginal`, E66 and E67: lagged pair moments for a sample size that allows for serial correlation, and binned target moments for a single-split gain. Both shipped in 0.3.0, revised after a review before that release |
| [MARGINAL-AT-WIDTH.md](records/MARGINAL-AT-WIDTH.md) | five asks against `marginal`'s cost per row at 10⁴ features × 50 targets, E70–E74, from the same caller: the measured cost and its scaling with the target count, then cross-lag terms on request, binning once per row, feature moments shared across targets, sharding a wide spec across the bank's threads, and running a chunk pair-major. E71, E70 and E73 were built on 2026-09-25 and E72 on 2026-09-29; E74 is open |
| [WARMUP-AND-CONVERGENCE.md](WARMUP-AND-CONVERGENCE.md) | the settings that keep a model from being used before it is ready: the design as built for `ewridge` in 0.9.0, its evidence, what was tried and dropped, and §7, the questions the design left, some answered and the rest still open |
| [HUBER.md](HUBER.md) | Huber regression from first principles, the analysis behind `huber_scale` and `huber_scale_half_life` as built in 1.3.0: the loss and its weights, the three scales and their measured breakdown, one pass against the converged fit, the regime trade-off a robust scale buys into, and what the API policy allows in 1.x; the numbers measured on 2026-10-23. §8 analyses five requests from a downstream project: what was built from them (`huber`'s `feature_sets` and `ridge_scale="sum"`, `marginal`'s Huber bins, `po.ewm_quantile`, GRAM.md's algorithm H, GBDT.md's objectives), and what was refused or left to the requester. Its §2 measured Proposal 2 at statsmodels' default scale cut, 2.5, not 1.345; GRAM.md has it at 1.345. §9 measures the streams one pass leaves captured, over 40 seeds, and the refit of the first rows built against them (`huber_refit_rows`) |
| [STATE-WORKFLOW-RESEARCH.md](records/STATE-WORKFLOW-RESEARCH.md) | the research of 2026-09-03 that decided how a saved state leaves a query (`save_state=` on `lf.online.fit_predict`): how Polars runs a Python source, measured on 1.34.0, 1.38.1 and 1.44.1 (F1–F8), the candidates, the rules R1–R7 and the prototype's checks C1–C8b, moved out of STATE-WORKFLOW.md with its section numbers |
| **Surveys and prototypes** | |
| [BEYOND-O-STATE.md](records/BEYOND-O-STATE.md) | what relaxing the rule that memory grows only with the model's state, to let it grow with the snapshots a window keeps, would unlock — a survey; four of its six candidates were since built (conformal intervals, `ew_cov` lags, `embargo`, and `window_size`, from snapshots under a memory budget) |
| [ORACLE-COVERAGE.md](records/ORACLE-COVERAGE.md) | which features and parameters an independent check or a known dataset holds, begun 2026-11-01 at the user's request: one row per model, parameter and diagnostic, with the test and the library or closed form that checks it, its dataset, and a status (COVERED, PARTIAL, GAP). It ranks the gaps by risk and lists candidate checks and datasets for each, all still to be researched |
| [BOOSTED-TREES.md](BOOSTED-TREES.md) | how far gradient-boosted trees can be pushed toward the library's contract — a numpy prototype, measured, nothing in Rust |
| [GBDT-ADAPTER.md](GBDT-ADAPTER.md) | how to train LightGBM and XGBoost from Arrow files as fast as they go: five options, from Python iterators to Rust driving the libraries' C APIs with cuts it places itself, measured against each other on data that fits in memory and on data that does not, what each library's interfaces take, read at their sources, and the design recommended, built as `po.gbdt`, whose measurements are §7. The prototypes are `scripts/gbdt_adapter.py` and `scripts/gbdt_driver` |
| [TREE-FEATURES.md](records/TREE-FEATURES.md) | what the library can add to XGBoost and LightGBM without rewriting them, measured on public and planted data and ranked, in three versions with a self-review and a fresh review answered: `label_clock=` for labels that arrive at their own time, target statistics of a high-cardinality categorical, a base margin, an offset, a reader of the libraries' model files and ranking groups through `po.gbdt`, beside the recipes that need no build (a nowcast of recent labels, rolling split-conformal, a combination of retrained boosters, input drift against `po.gbdt`'s cuts). Design only; the prototypes are `scripts/tree_features/` |
| [CONFORMAL-STEP.md](records/CONFORMAL-STEP.md) | the conformal step while labels arrive late: the tracker derived as an integral controller whose feedback is late by the rows waiting, the stability bound `rate · g · n < π/2` checked by simulation, and a step divided by the mean rows waiting, measured beside today's fixed rate, a step against the current radius, AgACI and SAOCP on flights and generated streams at three coverage levels. No rule held everywhere, so the default stays; the rule of thumb is in DIAGNOSTICS.md, and an opt-in is designed for the user. The prototypes are `scripts/conformal_step/` |
| [CLUSTERING.md](CLUSTERING.md) | the survey behind `kmeans` and `micro`: every clustering family, decided against the library's contract. Seven classes passed and were prototyped, two of them were built, and it says why the rest did not pass |
| [ARROW-SOURCES.md](ARROW-SOURCES.md) | whether anything that produces Arrow chunks, DuckDB first, can feed a bank. Its status line, dated 2026-10-06, says tier 0, which works through Polars, has shipped, and the native import is parked. Its §4, corrected on 2026-09-22, finds no blocker to that import |
| **Logs of measurements, tests and releases** | |
| [PERFORMANCE-LOG.md](records/PERFORMANCE-LOG.md) | every dated measurement of speed and memory from 2026-08-30 to 2026-11-06, §1 to §47 by topic, moved out of PERFORMANCE.md with its section numbers, which code comments and records cite |
| [TESTING-LOG.md](records/TESTING-LOG.md) | what the suite found and when, the hardening rounds, the mutation runs and the crash-safety audit, and the improvement entries by ID (sections A to E, such as T-D4), moved out of TESTING.md |
| [RELEASE-HISTORY.md](records/RELEASE-HISTORY.md) | how the release workflow, the Polars range, the API and the public repository came to be as they are, moved out of RELEASE-READINESS.md: the workflow's faults, Polars 2.0 measured, the two copies of Polars, CI cost while private, and going public |
| [REGIMES-EARLIER-FIGURES.md](records/REGIMES-EARLIER-FIGURES.md) | the figures REGIMES.md first reported for its sections 1 to 5, from builds before a change that moved them (the kernel weights, `D-hat`'s gradient, the window test's critical value), with the experiments' earlier run times, moved out of REGIMES.md's dated notes |
| **Logs of the documentation** | |
| [PHRASING.md](records/PHRASING.md) | every phrasing report, kept verbatim with its reading, begun for the first clarity pass: each rule in WRITING.md names the report it came from |
| [README-ITERATIONS.md](README-ITERATIONS.md) | every pass over the README, with its prompt verbatim, its counts and its verdict, and every idea the reviews brought up with its status |
| [DOCS-REWRITE-MAP.md](records/DOCS-REWRITE-MAP.md) | the map of the documentation rewrite, written before any page of it: every section of the README and the guides with its kind and the page it moves to, the pages the rewrite produces, why the guides stay Markdown, and the first set of pieces it writes |
| **Reviews, oldest first** | |
| [SIMPLIFICATION.md](SIMPLIFICATION.md) | what complexity was removed after the performance work, and what was kept on purpose |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | the pre-release review by axis — correctness, performance, usability, extensibility, testing — with the decision on each item |
| [REVIEW-E54-E64.md](REVIEW-E54-E64.md) | the code review of the correlation and regime models, E54–E64: 29 items, all resolved |
| [REVIEW-2026-09-12.md](REVIEW-2026-09-12.md) | a review by reading the code, in nine passes, each finding with the test that would confirm it. It has no status line and is left as the reviewer wrote it; the progress file below says what was done about each finding |
| [REVIEW-2026-09-12-pass10.md](records/REVIEW-2026-09-12-pass10.md) | the tenth pass of that review (`ftrl`, `pa`, `hmm`), written as a delta to fold into it; not folded in |
| [REVIEW-2026-09-12.pass10.patch](records/REVIEW-2026-09-12.pass10.patch) | the same pass as a patch to REVIEW-2026-09-12.md; not applied |
| [REVIEW-2026-09-12-PROGRESS.md](REVIEW-2026-09-12-PROGRESS.md) | what was done about each finding of the 2026-09-12 review, taking first the fixes an independent library could test |
| [REVIEW-2026-09-15.md](records/REVIEW-2026-09-15.md) | the eleven commits that answered the 2026-09-12 review, re-derived from first principles: three findings survived, and were fixed in batch 7 |
| [REVIEW-2026-09-17.md](REVIEW-2026-09-17.md) | the changes since 0.6.0 (the removal of the expression plugin, the Arrow output and the 0.7.0 release), every finding reproduced by running the code; all were applied except A2, which is recorded |
| [REVIEW-2026-09-18.md](REVIEW-2026-09-18.md) | a review of every code file, kept as its working state. Its opening still reads as in progress, but its four batches of fixes are on `main` and were released in 0.8.0 |

## At the top of the repository

| document | what it holds |
|---|---|
| [`CHANGELOG.md`](../CHANGELOG.md) | one entry per release, with what changed and why |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | the tests run before each commit, and the commit conventions |
| [`SECURITY.md`](../SECURITY.md) | how to report a vulnerability privately, which versions receive fixes, and what is in scope |
| [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | the Contributor Covenant, version 2.1, as adapted here, and where to report unacceptable behavior privately |
| [`llms.txt`](../llms.txt) | the map a coding agent reads before it writes against this library |

## Section numbers

**Sections keep their numbers, even where the order inside a file has been
tidied.** The code and the records cite the performance log's sections by
number, as `docs/records/PERFORMANCE-LOG.md §11`, the testing log's entries
by ID, as `T-D4`, and the other records' sections and entry IDs
(`docs/PLAN.md §4.4`, `E56`, `C5`).
