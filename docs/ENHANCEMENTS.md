# Suggested enhancements

Status as of 2026-08-30: all 17 tasks in `docs/PLAN.md` §11 are complete — seven
models, three entry points with identical numerics, chunk invariance and
out-of-sample-ness enforced by tests, release CI defined, defaults validated on
public data. This document lists what *follows from* those goals but is not built:
first the gaps against our own plan, then features our models are one step away
from, then a comparison against [river](https://riverml.xyz) (the reference
online-ML library) — both what we should adopt and what we deliberately leave out.

Priorities: **P1** = promised by PLAN.md or fixes a real sharp edge; **P2** =
cheap and clearly goal-aligned; **P3** = worthwhile, larger.

## 1. Gaps against our own plan

| # | P | Enhancement | Where it comes from |
|---|---|---|---|
| E1 | ~~P1~~ **done** | **Expose `EwCov` as `online.ew_cov()`** Done: `po.spec.ew_cov(...)` and `pl.col("x").online.ew_cov([...])`, emitting any of mean / var / std / cov / corr, pairwise stats named after the columns (`corr_x0_x1`). No targets and no coefficients; values are read from the state *before* each row, so an `ew_cov` output is safe to use as a feature for that same row. | PLAN §4.7. Also unblocked test T-R3, which now confirms exact agreement with river's Welford-based `stats.Mean/Var/Cov/PearsonCorr`. |
| E2 | P2 | **Optional Sherman–Morrison inverse on `EwCov`**, so precision matrices are available without a solve (river's `covariance.EmpiricalPrecision` is the analogue). | PLAN §4.7: "EW covariance matrix with optional Sherman–Morrison inverse". RLS maintains its own inverse inline; `EwCov` itself never got the option. |
| E3 | ~~P1~~ **done** | **Strict clock-monotonicity mode.** Today a backwards clock *within* a group is always routed through `on_clock_reset` — including when it is a data bug (mis-sorted chunks), which is silently absorbed. Done as a fourth `on_clock_reset` variant, `"error"`, which fits the existing parameter rather than adding a second one. The error names the column, the magnitude of the backwards jump, the row, and how to fix it. Per group, so interleaved ascending streams are fine; a repeated timestamp is a zero delta, not an error. | PLAN §5: "the bank asserts monotonicity (after reset handling) and errors loudly otherwise". |
| E4 | ~~P1~~ **done** | **Validate weights: reject negative values.** A negative weight currently corrupts state: `EwCov::update` silently no-ops when `λW + w ≤ 0`, while the per-target `r_j`/`w_j` update runs anyway with a negative denominator. Done: finite negative weights are rejected at extraction (naming column, value and row), non-finite ones skip the row like other non-finite inputs, and `w = 0` is a legal pure-decay row. | PLAN §3 defines `weight` but never constrains it. |
| E5 | ~~P2~~ **done** | **Surface `solve_failures`** Done: `Bank::solve_failures()` / `ModelBank.solve_failures()` return the count per spec and group. | PLAN §7: "never NaN silently, record `solve_failures`" — now reportable. |
| E6 | P2 | **Overnight/session shrinkage toward a long-run prior**: `session_shrink: float` mixing `S, r` toward either zero or a slow-halflife twin accumulator on session change, instead of the all-or-nothing `session_gap`/`reset`. | PLAN §12 open question 1. |
| E7 | P3 | **Per-target `min_periods`** (long-horizon targets need more warmup than short ones). Accept `min_periods: float | list[float]`. | PLAN §12 open question 2. |
| E8 | P2 | **Expose the streaming runner to Python**: `polars_online.run_config(toml_or_dict)` wrapping `online_polars::run_config`, so Python gets the same O(state + chunk) parquet→parquet path as the CLI without spawning a process. | PLAN §1: "Layers 2 and 3 share `online-polars`" — the runner is in the shared crate already. |
| E9 | P3 | **Multi-target expression API** — accept a struct expression or list of columns as the target so the expression surface stops being single-target-only. | PLAN §3 (`targets ≥ 1`) vs. §6 implementation. |
| E10 | P3 (**half done**) | **State migration scaffold.** The bank file now carries a `format_version` (v2) and v1 files still load, because the new `GroupKey` serializes transparently as its inner `Option<String>` — so hard rule 5 has one worked example. What is still missing is a **committed v1 fixture** that the test suite actually loads: today the backward-compatible path is argued from the encoding, not exercised. Generating a fixture is awkward precisely because no v1 writer exists any more; check in a small binary blob or a hex constant. | CLAUDE.md hard rule 5. |
| E11 | P3 | PyPI publishing job (name check + trusted publishing) and benchmark tracking in CI (store `scripts/benchmark.py` output per commit; alert on regression). | PLAN §10. |

## 2. One step from what we already track

These need no new mathematics — the state is already maintained.

| # | P | Enhancement |
|---|---|---|
| E11b | ~~P2~~ **done** | **Centered (Welford) updates in `EwCov`.** Done: `EwCov` now keeps centered co-moments (`C' = a·C + a·b·δδᵀ`), and every solve reads them directly instead of re-deriving `E[x²] − m²`. Measured improvement in slope recovery: **1e4 offset 3.7e-06 → 2.6e-12; 1e6 offset 2.0e-03 → 6.8e-10; 1e8 and 1e10 went from the feature being dropped entirely to 2.5e-08 and 4.8e-07.** The golden tests were unchanged by the switch, confirming it is numerically equivalent at ordinary scales, and the zero-variance threshold no longer needs a fuzzy scale reference — a constant feature now gives exactly zero variance. |
| E12 | ~~P2~~ **partly done** | **Predictive variance outputs** Done for the residual-noise component: `emit_sigma` emits `sigma_<slot>`, the EW standard deviation of that slot's out-of-sample residuals, for **every** model. Computed in the stream layer rather than per model, so the definition is identical everywhere (the models' own internal `sigma2` fields serve different purposes — robust scaling, Kalman observation noise — and are not all present or comparable). Still open: adding the *parameter* uncertainty term, which Kalman has exactly as `zᵀP_j z` and ridge/RLS would need `S⁻¹` for; `sigma` alone is the right interval only once `n_eff >> k`. |
| E13 | ~~P2~~ **done** | **Online model selection across grid combos** Done: `emit_selected` emits `selected_<t>` (the winning slot's label) and `pred_<t>__selected`, choosing by lowest EW out-of-sample error across ridge values, feature sets **and** halflives. It reuses the per-slot residual variance added for E12, so it costs nothing extra. Verified to discriminate: a light ridge wins on a strong signal, shrinkage wins on pure noise. |
| E14 | P3 | **EW model averaging** over combos or specs (exponentially weighted forecaster weights, river's `ensemble.EWARegressor`): softmax(−η · EW loss) weights instead of argmin. A natural extension of E13. |
| E15 | P3 | **Warm priors for `ewridge`**: `coef0` + shrink-toward-prior ridge (`(S + λD)β = r + λD β₀`), matching RLS's `coef0` and enabling E6's long-run-prior variant. |

## 3. Missing relative to river — recommended

river is organized around single-row `learn_one`/`predict_one` with pluggable
losses/optimizers. Everything below fits our stated goal (online *regression* on
clock-ordered financial event streams) and our architecture (models behind
`OnlineModel`, plumbing shared).

| # | P | river feature | What it would look like here |
|---|---|---|---|
| E16 | P2 | `linear_model.LinearRegression` + `optim.*` (SGD/Adam/…) with pluggable losses (`Squared`, `Huber`, `Quantile`, `EpsilonInsensitive`, `Poisson`) | One `sgd` model: per-row gradient step, loss enum, learning-rate schedule, decayed like everything else. Cheap (no solves), the standard baseline everyone expects, and the natural home for **Poisson regression on count targets** (trade counts, arrivals) which none of our exact solvers cover. |
| E17 | P2 | `linear_model.PARegressor` (passive-aggressive) | A ~40-line model on the existing trait; classic online-learning baseline, useful in tests as a third independent family. |
| E18 | P2 | `optim.FTRLProximal` used for *regression* | Our `ftrl` is logistic-only. Adding a squared-loss mode is a loss-function swap in `ftrl.rs` and gives river-comparable sparse linear regression. |
| E19 | ~~P2~~ **done** | `linear_model.BayesianLinearRegression` | Done: Kalman gained `standardize` (default true). With `standardize=False`, `q=0`, `add_intercept=False`, `p0 = 1/alpha` and `obs_var = 1/beta`, our filter reproduces river's `BayesianLinearRegression` **to 3.6e-15** — verified across three prior/noise settings in T-R2, with a guard test that turning standardization back on breaks the correspondence. |
| E20 | P2 | `drift.ADWIN`, `drift.PageHinkley` | A drift detector on the (already out-of-sample) residual stream, emitting a `drift_<t>` flag and optionally triggering the existing reset path. Complements halflife decay: decay forgets smoothly, drift detection catches breaks. |
| E21 | ~~P2~~ **done** | `anomaly.GaussianScorer` | Done: `emit_resid_z` emits `resid_z_<slot> = resid / sigma`, where sigma is read from the state *before* the row, so the score is out-of-sample like the prediction it scales. |
| E22 | P3 | `metrics.*` + `utils.Rolling` (streaming metrics) | Our `eval.py` is batch-Polars over collected output. A streaming metrics accumulator inside the bank (EW IC / R² / hit-rate per (spec, group, target), O(state)) matches the library's memory story and gives the CLI a `--metrics` summary for free. |
| E23 | P3 | `stats.Quantile` (P² algorithm), `stats.AutoCorr` | Streaming scalar stats that complement `ew_cov` (E1): P² quantiles of residuals give distribution-free intervals; autocorrelation of residuals is a model-health diagnostic. |
| E24 | P3 | `preprocessing.AdaptiveStandardScaler`, `TargetStandardScaler` | We standardize *inside* solves; SGD/FTRL-family models (E16/E18) would benefit from explicit running input/target scaling as a spec option. |
| E25 | P3 | `time_series.HoltWinters` | An intercept-only level(+trend) model — i.e. EW mean with trend — as a forecasting baseline. SNARIMAX-style lag features are better served by Polars expressions upstream; only the recursive level/trend state needs a model. |

## 4. In river, deliberately out of scope

Listed so the omission is a decision, not an oversight. These conflict with the
library's shape (deterministic linear-family models, O(k²) state, exact
save/resume, three numerics-identical surfaces) rather than merely being work:

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

## Suggested order

1. Sharp edges and plan debts: E4 (negative weights), E3 (strict clock), E1/E2
   (`ew_cov`), E5 (`solve_failures`).
2. Highest value per line: E21 (residual z-score), E12 (predictive variance),
   E13 (grid selection), E18 (FTRL regression).
3. New surface area: E16/E17 (SGD/PA + Poisson), E19 + Kalman `standardize`
   switch, E20 (drift), E8 (Python runner), E6 (session shrinkage).
4. The rest as demand appears.
