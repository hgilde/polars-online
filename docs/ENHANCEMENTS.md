# Suggested enhancements

Status as of 2026-08-30: **all 17 tasks in `docs/PLAN.md` §11 and all 27
enhancements below are complete.** Ten models (`ewridge`, `rls`, `lasso`,
`kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`, `holt`) plus `ew_cov`, three
entry points with identical numerics, chunk invariance and out-of-sample-ness
enforced by tests, release CI defined, defaults validated on public data.

This document lists what *followed from* those goals: first the gaps against our
own plan, then features our models were one step away from, then a comparison
against [river](https://riverml.xyz) (the reference online-ML library) and
[Pathway](https://pathway.com) (a Rust-engined live-data framework) — both what
we adopted and what we deliberately leave out. Everything marked **done** is
implemented and tested. The forward-looking parts are §4 (the standing list of
what we will *not* build), §5 (two candidates the river audit left undecided)
and §6 (one accessor the accumulators are missing).

Priorities: **P1** = promised by PLAN.md or fixes a real sharp edge; **P2** =
cheap and clearly goal-aligned; **P3** = worthwhile, larger.

## 1. Gaps against our own plan

| # | P | Enhancement | Where it comes from |
|---|---|---|---|
| E1 | ~~P1~~ **done** | **Expose `EwCov` as `online.ew_cov()`** Done: `po.spec.ew_cov(...)` and `pl.col("x").online.ew_cov([...])`, emitting any of mean / var / std / cov / corr, pairwise stats named after the columns (`corr_x0_x1`). No targets and no coefficients; values are read from the state *before* each row, so an `ew_cov` output is safe to use as a feature for that same row. | PLAN §4.7. Also unblocked test T-R3, which now confirms exact agreement with river's Welford-based `stats.Mean/Var/Cov/PearsonCorr`. |
| E2 | ~~P2~~ **done** | **Optional Sherman–Morrison inverse on `EwCov`**, Done: `EwCov::with_inverse(k, prior)` tracks `(C + s·prior·I)⁻¹` incrementally, and `ew_cov` exposes it as the `partial_corr` statistic — correlation between two columns *controlling for every other column*. Verified against a from-scratch Gauss-Jordan inversion at every step, and on a common-driver setup where the marginal correlation is 0.99 and the partial one is 0.006. Two subtleties the implementation had to get right: the prior's scale must decay by the co-moment factor `a`, not by `lam`, or the update is not rank-1; and `a = 0` on the first row, where there is no rank-1 step to take and the inverse is just `I/prior`. | PLAN §4.7. |
| E3 | ~~P1~~ **done** | **Strict clock-monotonicity mode.** Today a backwards clock *within* a group is always routed through `on_clock_reset` — including when it is a data bug (mis-sorted chunks), which is silently absorbed. Done as a fourth `on_clock_reset` variant, `"error"`, which fits the existing parameter rather than adding a second one. The error names the column, the magnitude of the backwards jump, the row, and how to fix it. Per group, so interleaved ascending streams are fine; a repeated timestamp is a zero delta, not an error. | PLAN §5: "the bank asserts monotonicity (after reset handling) and errors loudly otherwise". |
| E4 | ~~P1~~ **done** | **Validate weights: reject negative values.** A negative weight currently corrupts state: `EwCov::update` silently no-ops when `λW + w ≤ 0`, while the per-target `r_j`/`w_j` update runs anyway with a negative denominator. Done: finite negative weights are rejected at extraction (naming column, value and row), non-finite ones skip the row like other non-finite inputs, and `w = 0` is a legal pure-decay row. | PLAN §3 defines `weight` but never constrains it. |
| E5 | ~~P2~~ **done** | **Surface `solve_failures`** Done: `Bank::solve_failures()` / `ModelBank.solve_failures()` return the count per spec and group. | PLAN §7: "never NaN silently, record `solve_failures`" — now reportable. |
| E6 | ~~P2~~ **done** | **Overnight/session shrinkage toward a long-run prior**: Done as the slow-twin variant, which is the part `session_gap` could not already express: `session_gap` only changes how *confident* the model is (it is a decay in clock units), whereas this changes what it *believes*. A second accumulator runs at `long_halflife`, and on a session boundary the two are mixed weight-respectingly. Measured on a stream whose long run is +1 and whose last session ran at −1: `session_shrink=0` carries −0.971 through the break, `0.9` reverts to +0.853, and the reversion is monotone in the parameter. | PLAN §12 open question 1, now answered. |
| E7 | ~~P3~~ **done** | **Per-target `min_periods`** Done: `min_periods` takes a scalar or one value per target. Gating happens in the stream layer — the model predicts once the smallest threshold is met and a not-yet-ready target's slots are withheld before they can reach `resid`, `sigma`, `resid_z`, drift or selection. Warmup gates **output, not learning**, which is asserted: a late-reporting target's eventual fit is identical to one that reported from the start. | PLAN §12 open question 2, now answered. |
| E8 | ~~P2~~ **done** | **Expose the streaming runner to Python**: Done: `po.run(config_or_kwargs)` takes a dict, a TOML path, or plain keywords, with keywords overriding the config so a checked-in TOML can be reused with a different input. Releases the GIL for the whole run. Tested to produce byte-identical predictions to `ModelBank` on the same data, and to resume from state exactly. | PLAN §1: "Layers 2 and 3 share `online-polars`". |
| E9 | ~~P3~~ **done** | **Multi-target expression API** Done: every model namespace takes `extra_targets`, so the calling column is the first target and the rest follow. Multi-target specs share one `X'X`, which makes this much cheaper than one expression per horizon. Verified identical to a multi-target bank. | PLAN §3 (`targets ≥ 1`) vs. §6 implementation. |
| E10 | ~~P3~~ **done** | **State migration scaffold.** The bank file now carries a `format_version` (v2) and v1 files still load, because the new `GroupKey` serializes transparently as its inner `Option<String>` — so hard rule 5 has one worked example. Done: `crates/online-polars/tests/state_v1.rs` carries a real v1 file frozen as a hex constant (a hex constant rather than a binary, because of hard rule 1). Four tests: it loads, it continues the stream identically to a bank that never left v2, saving it produces v2 which also loads, and a truncated or empty state is refused. The spec is recovered *from* the fixture rather than restated, so the test cannot pass by two hand-written copies agreeing. | CLAUDE.md hard rule 5, now exercised rather than argued. |
| E11 | ~~P3~~ **done** | PyPI publishing job Done: `release.yml` gained a `publish` job using PyPI **trusted publishing** (OIDC, no stored token), gated behind a `pypi` environment so a tag can build artifacts without also publishing. `benchmark.yml` runs the throughput script on every push and writes the table into the job summary, plus an artifact per commit. Reported, never gating — a shared runner is too noisy for a hard threshold, so this catches large regressions by putting the numbers next to the diff. **Before the first publish:** the name is free (`pypi.org/pypi/polars-online/json` 404s, as does the underscore spelling — checked 2026-08-30), so what remains is registering this repository and workflow as a trusted publisher, which needs the account. | PLAN §10. |

## 2. One step from what we already track

These need no new mathematics — the state is already maintained.

| # | P | Enhancement |
|---|---|---|
| E11b | ~~P2~~ **done** | **Centered (Welford) updates in `EwCov`.** Done: `EwCov` now keeps centered co-moments (`C' = a·C + a·b·δδᵀ`), and every solve reads them directly instead of re-deriving `E[x²] − m²`. Measured improvement in slope recovery: **1e4 offset 3.7e-06 → 2.6e-12; 1e6 offset 2.0e-03 → 6.8e-10; 1e8 and 1e10 went from the feature being dropped entirely to 2.5e-08 and 4.8e-07.** The golden tests were unchanged by the switch, confirming it is numerically equivalent at ordinary scales, and the zero-variance threshold no longer needs a fuzzy scale reference — a constant feature now gives exactly zero variance. |
| E12 | ~~P2~~ **done** | **Predictive variance outputs** Done for the residual-noise component: `emit_sigma` emits `sigma_<slot>`, the EW standard deviation of that slot's out-of-sample residuals, for **every** model. Computed in the stream layer rather than per model, so the definition is identical everywhere (the models' own internal `sigma2` fields serve different purposes — robust scaling, Kalman observation noise — and are not all present or comparable). The parameter-uncertainty term is now there too, for the model that can give it exactly: `Kalman::pred_var()` returns `zᵀP_j z + R_j`. That is the piece `sigma` cannot supply — `sigma` is the spread of realized errors, while this also knows how unsure the filter is about its own coefficients, so it is wide during warmup and narrows with evidence (asserted), and never falls below the observation noise (asserted). |
| E13 | ~~P2~~ **done** | **Online model selection across grid combos** Done: `emit_selected` emits `selected_<t>` (the winning slot's label) and `pred_<t>__selected`, choosing by lowest EW out-of-sample error across ridge values, feature sets **and** halflives. It reuses the per-slot residual variance added for E12, so it costs nothing extra. Verified to discriminate: a light ridge wins on a strong signal, shrinkage wins on pure noise. |
| E14 | ~~P3~~ **done** | **EW model averaging** Done: `emit_averaged` emits `pred_<t>__averaged`, a `softmax(−eta · EW squared error)` blend over every slot, reusing the same tracked error E13 selects on. Both limits are asserted: `eta → ∞` reproduces `emit_selected` exactly, `eta → 0` gives the equal-weight mean, and the result is always a convex combination of the slots. Averaging hedges where selection commits — it loses slightly when one slot dominates and wins when the best slot changes, which is the trade the tests pin. |
| E15 | ~~P3~~ **done** | **Warm priors for `ewridge`**: Done, and it surfaced a semantic worth knowing: because `S` is a weighted **mean**, a plain `ridge` is a fixed per-observation penalty and its pull toward `coef0` is **permanent** — the opposite of the usual "the prior washes out" intuition. The fading warm start needs `ridge_decay`, where the prior sits on the decaying sum scale. Both behaviours are documented and pinned by tests: with prior 5 and truth 1.5, a fixed ridge ends at 4.69 while `ridge_decay` starts at 3.69 and converges to 1.512. The prior is stated in original units and is scaled correctly through the standardized path too. |

## 3. Missing relative to river — recommended

river is organized around single-row `learn_one`/`predict_one` with pluggable
losses/optimizers. Everything below fits our stated goal (online *regression* on
clock-ordered financial event streams) and our architecture (models behind
`OnlineModel`, plumbing shared).

| # | P | river feature | What it would look like here |
|---|---|---|---|
| E16 | ~~P2~~ **done** | `linear_model.LinearRegression` + `optim.*` (SGD/Adam/…) with pluggable losses (`Squared`, `Huber`, `Quantile`, `EpsilonInsensitive`, `Poisson`) | Done: an `sgd` model with six losses (squared, huber, quantile, epsilon_insensitive, **poisson**, logistic) and three schedules (constant, inv_scaling, adagrad), O(k) per row and no solves. Poisson recovers a log-rate from real count data. Two behaviours worth knowing, both pinned by tests: `clip_gradient` defaults to `1e3` rather than off, because a log link makes the gradient explode (unclipped, the Poisson intercept ran to −4e10; the cap does not bind for squared loss, whose fit is bit-identical either way); and `epsilon_insensitive` has a sign-valued subgradient, so a constant rate oscillates in a band and only an annealed schedule settles. |
| E17 | ~~P2~~ **done** | `linear_model.PARegressor` (passive-aggressive) | Done: `pa` with all three variants (`pa` unbounded, `pa1` capped, `pa2` damped), no learning rate to tune. Two properties worth knowing, both tested: the row weight scales the step, and **plain `pa` satisfies each row's constraint exactly**, so after an outlier the next clean row pulls it straight back — the final coefficient looks fine while the predictions in between are wild, which is why robustness has to be measured over the stream and why `pa1` is the default. |
| E18 | ~~P2~~ **done** | `optim.FTRLProximal` used for *regression* | Done: `loss="squared"` alongside the default `"logistic"`. The two differ only in the link (the gradient is `(p − y)·z` either way), so this is sparse linear regression with no solves and L1 support that `ew_ridge` lacks. Recovers `[1.5, −0.5]` on a 3-feature problem and, with `l1=0.5`, zeroes the pure-noise feature while keeping the signal. |
| E19 | ~~P2~~ **done** | `linear_model.BayesianLinearRegression` | Done: Kalman gained `standardize` (default true). With `standardize=False`, `q=0`, `add_intercept=False`, `p0 = 1/alpha` and `obs_var = 1/beta`, our filter reproduces river's `BayesianLinearRegression` **to 3.6e-15** — verified across three prior/noise settings in T-R2, with a guard test that turning standardization back on breaks the correspondence. |
| E20 | ~~P2~~ **done** | `drift.ADWIN`, `drift.PageHinkley` | Done (Page-Hinkley; ADWIN not needed for this): `emit_drift` runs a detector on each slot's absolute out-of-sample residual **scaled by that slot's own EW residual std**, so `drift_delta` means the same thing whatever the target's units. `drift_action="reset"` restarts the stream's models down the same path a clock reset takes, and the flag is still reported so the reset is visible rather than silent. Measured: a sign flip mid-stream is caught **2 rows later**, with zero false positives over 8000 stationary rows. |
| E21 | ~~P2~~ **done** | `anomaly.GaussianScorer` | Done: `emit_resid_z` emits `resid_z_<slot> = resid / sigma`, where sigma is read from the state *before* the row, so the score is out-of-sample like the prediction it scales. |
| E22 | ~~P3~~ **done** | `metrics.*` + `utils.Rolling` (streaming metrics) | Done: `emit_metrics` emits `ic_<slot>`, `r2_<slot>` and `hit_rate_<slot>`, exponentially weighted on the model's own clock and read before each row is scored, so they are out-of-sample like everything else. Verified to land within 0.02 of `eval.py` on the same data with no decay — the two now measure the same thing, one in O(state) and one over the collected frame. |
| E23 | ~~P3~~ **done** | `stats.Quantile` (P² algorithm), `stats.AutoCorr` | Done: `resid_quantiles` emits `absresid_q<p>_<slot>` via P² (five numbers per level, no window), and `emit_autocorr` emits `autocorr_<slot>`. Both earn their place: on a stream with 1% gross outliers `sigma` reads 2.95 while the median |resid| is 0.43, so a Gaussian interval is the wrong number; and an omitted slow-moving driver shows up as residual autocorrelation above 0.3 where a well-specified fit stays under 0.1. |
| E24 | ~~P3~~ **done** | `preprocessing.AdaptiveStandardScaler`, `TargetStandardScaler` | Done for `sgd`, which is where it matters: `scale_features` standardizes against the moments from *before* each row and unscales the coefficients on the way out, so they stay in the caller's units. Demonstrated on features measured in thousands and thousandths — a single learning rate cannot suit both, and the scaled fit recovers `[0.002, 900]` where the unscaled one does not. Not added to `ftrl`/`pa`: FTRL's per-coordinate rates already adapt to scale, and PA's step is normalized by `‖z‖²`. |
| E25 | ~~P3~~ **done** | `time_series.HoltWinters` | Done as `holt`: Holt's linear trend method, the one model that takes **no features** — `pred = level + trend·Δclock`, then `level' = α·y + (1−α)·pred` and `trend' = β·(level'−level)/Δclock + (1−β)·trend`, with α and β from separate halflives. The trend is per *clock unit*, not per row, so an irregular clock extrapolates the right distance: on a series stepping 5 clock units a row it forecasts 5 units ahead and lands within 1%. `coef` is `[level, trend]`; `trend_halflife=inf` pins the trend at zero and reduces it to a plain EW level. Deliberately no seasonal term — a seasonal index is a `group_by` on the phase, which the bank already does. It is a *baseline*: the tests assert a real feature beats it, which is what makes it useful for saying how much a regression is actually adding. Holt is also why `features` may now be empty — for that model only; every other model still rejects an empty feature list. |

## 3b. Pathway — a capability check

[Pathway](https://pathway.com) (`pathway` 0.32.1) is the other reference to keep
in view: a live-data framework with a **Rust engine** (differential dataflow)
under a Python API. Since it is already Rust, the goal is not to reimplement any
of it — it is to know where the boundary is so we do not grow a second, worse
copy of what it does well.

What it actually contains (verified against the package metadata and the
`python/pathway/stdlib` tree, not from memory):

| Pathway area | Contents | Relation to us |
|---|---|---|
| `stdlib/temporal` | `asof_join`, `asof_now_join`, `interval_join`, tumbling/sliding/session `window`, `window_join`, `temporal_behavior` (lateness/cutoff policies) | **Theirs.** This is the event-time plumbing *upstream* of a model: aligning a feature to an event, bucketing, handling late arrivals across a pipeline. We take an already-aligned, already-ordered frame. |
| Connectors | Kafka, CDC, S3, filesystem, streaming outputs | **Theirs.** We are parquet-in/parquet-out plus whatever Polars reads. |
| Engine | Incremental recomputation, multi-worker, exactly-once persistence for a whole pipeline | **Theirs.** Ours is a different shape: one O(state) recursion per stream, with `save`/`load` of *model* state only. |
| `stdlib/ml` | LSH kNN index, LSH clustering, HMM, fuzzy-join helpers | Disjoint. Nothing in the online-linear-model family; kNN and clustering are on our own out-of-scope list below for the same memory reasons. |
| `stdlib/statistical`, `stdlib/stateful`, `stdlib/ordered` | interpolation, deduplication, row diffing | Disjoint and small. |

**Conclusion: essentially no model overlap, real overlap on the layer above.**
Pathway has no online regression — no EW-ridge/RLS/Kalman/lasso analogue — so
nothing in our model backlog is redundant with it. Conversely, everything in the
"deliberately out of scope" list below that is *pipeline*-shaped (windowing,
temporal joins, ingestion, distributed execution, late-arrival policy) now has a
named answer: use Pathway, or Polars expressions upstream, rather than growing
spec parameters for it.

Two consequences for this backlog:

| # | P | Item |
|---|---|---|
| E26 | ~~P3~~ **done** | **A Pathway integration example, not a dependency.** Pathway supports stateful Python UDFs, so a `ModelBank` can run as an operator inside a Pathway pipeline: Pathway does ingestion, event-time alignment and windowing; we do the model. Worth an `examples/` script and a README paragraph. **Note the licence**: Pathway is BSL / "Other/Proprietary" while this project is Apache-2.0, so it must stay an optional example — never a required dependency, and not a dev-dependency in the default group. Done: `examples/pathway_integration.py` puts a `ModelBank` behind a `BankOperator` class — `__call__(batch)` for the data path, `snapshot()`/`restore()` for the engine's persistence hooks — and shows it driven two ways: plain ordered batches (which runs everywhere, and is what CI exercises) and a Pathway `@pw.udf` sketch behind a lazy import. Two properties this project already tests are exactly what make the operator safe under a streaming engine, and the example says so: chunk invariance means the engine's batching cannot change the numbers, and exact `save_bytes`/`load_bytes` round-tripping means a pipeline checkpoint can carry the model. `pathway` appears in no dependency group; the script prints the install line under the user's own licence if it is missing. |
| E27 | ~~P3~~ **done** | **Say what we do not do, in the README.** The clock semantics (`session`, `max_dclock`, `on_clock_reset`) are deliberately a *within-stream* facility, not a late-arrival or windowing policy. Readers arriving from a streaming-framework background will otherwise expect watermarks and tumbling windows here. One short "what this is not" section removes that expectation and points at Pathway/Polars. |

## 4. In river and Pathway, deliberately out of scope

Listed so the omission is a decision, not an oversight. These conflict with the
library's shape (deterministic linear-family models, O(k²) state, exact
save/resume, three numerics-identical surfaces) rather than merely being work.

**Audited against river 0.26.1's actual module list**, not from memory — 40
top-level modules, of which this project overlaps `linear_model`, `time_series`,
`drift`, `anomaly`, `covariance`, `stats`, `metrics`, `optim` and
`preprocessing`. The audit found seven exclusions that were being made in
practice without being written down; they are now in the list below, and the
two items it found that are arguably *in* scope are in §5.

- **Trees and forests** (`tree.HoeffdingTreeRegressor`, `forest.ARFRegressor`,
  SGT): unbounded/adaptive state, nondeterministic under resampling, no clean
  clock-decay semantics. A different library.
- **Neighbors** (`neighbors.KNNRegressor`): O(window) memory contradicts
  "memory is O(state), not O(data)".
- **Clustering / naive Bayes / multiclass softmax**: not regression on ordered
  streams (PLAN §4.6 scopes classification to binary).
- **Bandit-based model selection** (`model_selection.*`): E13/E14 cover the
  need deterministically; bandits add randomness to the prediction path.
- **Pipelines / feature extraction** (`compose.*`, `feature_extraction.*`,
  hashing, n-grams): Polars expressions upstream of the model are our
  composition layer; duplicating it inside specs would fork the API.
- **Imbalanced-learning wrappers, text models, generic sketches**: no use case
  in the stated goals.
- **Momentum-family optimizers** (`optim.Adam`, `RMSProp`, `Momentum`, `Nadam`,
  `AdaDelta`, `AdaMax`, `AMSGrad`, `AdaBound`, `NesterovMomentum`, `Averager`,
  `Newton`): each carries extra per-coefficient state with no clean
  clock-decay semantics — what does a momentum buffer mean after a six-hour
  gap? `sgd` offers the three schedules that do have an answer (constant,
  `inv_scaling` on `n_eff`, AdaGrad, all decayed on the model's clock).
- **`drift.KSWIN`**: a Kolmogorov–Smirnov test over a sliding window, so
  O(window) memory. Same reason as `neighbors`. `PageHinkley` (E20) is the
  O(state) detector; `ADWIN` is also excluded, being adaptive-window.
- **Shrinkage covariance estimators** (`covariance.LedoitWolfCovariance`,
  `OASCovariance`, `ShrunkCovariance`): their shrinkage intensity is estimated
  from the whole sample, which has no streaming form that keeps our exactness
  guarantee. `covariance.EwaCovariance` / `EwaPrecision` — the two that *do* —
  are what `ew_cov` and its Sherman–Morrison inverse (E1/E2) already are.
- **Automatic feature selection** (`feature_selection.SelectKBest`,
  `VarianceThreshold`, `PoissonInclusion`): E13 selects among feature sets the
  caller *names*, which keeps the output schema fixed and declarable. Selection
  that changes the feature set per row cannot have a static schema, which the
  expression plugin requires.
- **Non-linear anomaly detectors** (`anomaly.HalfSpaceTrees`, `LODA`,
  `LocalOutlierFactor`, `OneClassSVM`): not the linear family, and the tree and
  window-based ones carry the memory problem too. `GaussianScorer` is covered by
  E21, and `StandardAbsoluteDeviation`'s robust-scale role by E23's P² quantiles
  of the absolute residual.
- **Encoders, imputers and projections** (`preprocessing.OneHotEncoder`,
  `StatImputer`, `FeatureHasher`, the random projectors, `MinMax`/`MaxAbs`/
  `Robust` scalers): Polars expressions upstream. E24 implements the two that
  have to be *inside* the model to be out-of-sample — `AdaptiveStandardScaler`
  and `TargetStandardScaler` — because they read statistics from before the row.
- **Event-time windowing, temporal/asof joins, connectors, late-arrival
  policy, distributed execution** (Pathway's `stdlib/temporal` and engine): a
  different layer, already implemented in Rust by someone else. Feed us an
  aligned, ordered frame — from Polars expressions or from Pathway — rather
  than teaching specs to window.

## 5. Open candidates from the river audit

Everything numbered E1–E27 is implemented. These two are the only things the
audit of river 0.26.1 turned up that are arguably *in* scope rather than
excluded, and neither has been built or decided on.

PLAN §4.6 scopes classification to **binary**, and `ftrl` covers the logistic
case. river has three more binary linear classifiers, and two of them would be
a loss function rather than a model:

| # | P | Candidate | Cost, and the argument against |
|---|---|---|---|
| E28 | P3 | **`linear_model.Perceptron` and `PAClassifier`** — the perceptron is `sgd` with a hinge-at-zero loss, and `PAClassifier` is `pa` with a hinge loss. Both are a `match` arm in an existing model, not new state, and both would inherit the clock decay, chunk invariance and save/resume for free. | Neither gives calibrated probabilities, which is what `ftrl`'s logistic loss is for and what a financial signal usually wants. They win only where the margin matters more than the probability. Cheap, but "cheap" is not a use case. |
| E29 | P3 | **`linear_model.ALMAClassifier` and `AdPredictor`** — an approximate large-margin classifier, and a Bayesian probit model for click-through rates. | Real new state and real new mathematics, for a use case (CTR) outside the stated one. `AdPredictor`'s per-weight Gaussian posterior overlaps what `kalman` already provides for regression. |

Neither is recommended without a use case; they are listed so that "we do not
have a perceptron" is a recorded decision rather than an oversight.

## 6. Reaching the accumulators directly

| # | P | Enhancement |
|---|---|---|
| E30 | P2 | **Export the EW Gram matrix and cross-moments as arrays.** `EwCov` maintains the centred `k × k` co-moment matrix and `EwRidge` the per-target cross-moment vector `r`; every solve reads them. Nothing exposes them to a caller. The only route today is `ew_cov`, which emits *pairwise* statistics as struct fields — the right shape at `k = 4` (6 columns) and the wrong one at `k = 400` (**79,800** columns). Proposed: `ModelBank.gram(spec, group=None) -> (G, b, scales)` returning numpy arrays, plus the Rust equivalent. |

**Related:** [`docs/BEYOND-O-STATE.md`](BEYOND-O-STATE.md) surveys what becomes
possible if the `O(state)` memory rule is relaxed to `O(window)` or a sketch bound —
six candidates that are genuinely absent from Rust and C++, of which adaptive conformal
prediction is the strongest. None is proposed for implementation; the survey exists so
the question has an answer that is not "no, because §4 says so".

**Why this is worth having.** The accumulators are the expensive part and they are
already exact, already centred (E11b), already decayed on the model's clock with
session and `max_dclock` handling, and already resumable. Anyone who wants to do
something *other than* our solve with them — a custom penalty, an information
criterion, a conditioning diagnostic (`cond(G)`, a scree plot), forward stepwise or
an orthogonal matching pursuit over `G`, or simply to check a fit by hand — currently
cannot, and has to recompute `X'X` from raw data in a second pass.

**What it is not.** Not a speed claim: for a *single* batch Gram matrix over
materialised data, BLAS `dgemm` is blocked and vectorised and beats an O(k²)-per-row
streaming update comfortably. The value is that the matrix comes from **one pass over
data that is never materialised**, at every point in the stream rather than at one, and
that it is the same matrix the deployed model solves against — so an analysis built on
it cannot silently disagree with the model it is analysing.

**Cost:** small. The data is already in the right layout; this is an accessor, a copy
into a contiguous array, and tests that it agrees with a from-scratch
`X'X` on the same rows — which is a check `ewcov.rs` already performs internally
(`tracked_inverse_matches_a_from_scratch_solve` follows the same pattern).

**Open question:** whether to return the standardised or raw form. `EwRidge` solves
standardised and unscales on the way out, so both are available; returning `scales`
alongside `G` lets the caller choose without a second call.

## What was verified against river, and what was not

Six correspondences are checked numerically in `tests/test_river.py` (T-R1–T-R6):
FTRL's z/n recursion to 1e-12, Kalman ≡ `BayesianLinearRegression` to 3.6e-15,
`EwCov` ≡ river's Welford `Mean`/`Var`/`Cov`/`PearsonCorr` exactly, the EW
mean/var convention difference stated in closed form, and the quantile and
Huber models agreeing statistically.

river's own test suite is **not** ported and should not be: it would be testing
river. Two of those six tests exist specifically to pin places where the two
libraries legitimately *disagree* — river's `LogisticRegression` predicts one
proximal step behind McMahan Algorithm 1, and river's `EWMean` is an
un-normalized EWMA seeded at its first value where ours is the bias-corrected
weighted mean. Both are asserted, so neither can be mistaken for a bug later.
