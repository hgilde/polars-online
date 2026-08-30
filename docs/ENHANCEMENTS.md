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
| E1 | P1 | **Expose `EwCov` as `online.ew_cov()`** (expression + bank spec): EW mean / variance / covariance / correlation matrix as a struct output. | PLAN §4.7 says it is "exposed on its own as `online.ew_cov()` (replaces pure-Polars pairwise EW correlations when k>2)". The primitive exists in core; no surface was ever built. |
| E2 | P2 | **Optional Sherman–Morrison inverse on `EwCov`**, so precision matrices are available without a solve (river's `covariance.EmpiricalPrecision` is the analogue). | PLAN §4.7: "EW covariance matrix with optional Sherman–Morrison inverse". RLS maintains its own inverse inline; `EwCov` itself never got the option. |
| E3 | P1 | **Strict clock-monotonicity mode.** Today a backwards clock *within* a group is always routed through `on_clock_reset` — including when it is a data bug (mis-sorted chunks), which is silently absorbed. Add `strict_clock: bool` (error on negative delta) or make `on_clock_reset: "error"` a variant. | PLAN §5: "the bank asserts monotonicity (after reset handling) and errors loudly otherwise" — only the null-clock check was implemented. |
| E4 | ~~P1~~ **done** | **Validate weights: reject negative values.** A negative weight currently corrupts state: `EwCov::update` silently no-ops when `λW + w ≤ 0`, while the per-target `r_j`/`w_j` update runs anyway with a negative denominator. Done: finite negative weights are rejected at extraction (naming column, value and row), non-finite ones skip the row like other non-finite inputs, and `w = 0` is a legal pure-decay row. | PLAN §3 defines `weight` but never constrains it. |
| E5 | ~~P2~~ **done** | **Surface `solve_failures`** Done: `Bank::solve_failures()` / `ModelBank.solve_failures()` return the count per spec and group. | PLAN §7: "never NaN silently, record `solve_failures`" — now reportable. |
| E6 | P2 | **Overnight/session shrinkage toward a long-run prior**: `session_shrink: float` mixing `S, r` toward either zero or a slow-halflife twin accumulator on session change, instead of the all-or-nothing `session_gap`/`reset`. | PLAN §12 open question 1. |
| E7 | P3 | **Per-target `min_periods`** (long-horizon targets need more warmup than short ones). Accept `min_periods: float | list[float]`. | PLAN §12 open question 2. |
| E8 | P2 | **Expose the streaming runner to Python**: `polars_online.run_config(toml_or_dict)` wrapping `online_polars::run_config`, so Python gets the same O(state + chunk) parquet→parquet path as the CLI without spawning a process. | PLAN §1: "Layers 2 and 3 share `online-polars`" — the runner is in the shared crate already. |
| E9 | P3 | **Multi-target expression API** — accept a struct expression or list of columns as the target so the expression surface stops being single-target-only. | PLAN §3 (`targets ≥ 1`) vs. §6 implementation. |
| E10 | P3 | **State migration scaffold**: an actual v1→v2 loader exercised by a test fixture, so hard rule 5 ("keep a loader for the previous version") has a worked path before it is needed in anger. | CLAUDE.md hard rule 5. |
| E11 | P3 | PyPI publishing job (name check + trusted publishing) and benchmark tracking in CI (store `scripts/benchmark.py` output per commit; alert on regression). | PLAN §10. |

## 2. One step from what we already track

These need no new mathematics — the state is already maintained.

| # | P | Enhancement |
|---|---|---|
| E11b | P2 | **Centered (Welford) updates in `EwCov`.** The accumulator stores `E[x]` and `E[x xᵀ]` and derives variance as `E[x²] − m²`, which loses precision on large offsets. The drop threshold was fixed (see `docs/TESTING.md` T-E9) so ordinary financial scales work, but the measured operating range still degrades around 1e6 and features are dropped beyond ~1e7. Centered updates would remove the limit; `tests/test_edge_cases.py::TestNumericalScale` is the baseline to beat. |
| E12 | P2 | **Predictive variance outputs** (`pred_var_<t>` or `pred_std_<t>`). Kalman has it exactly (`zᵀP_j z + σ²_j`); EW-ridge/RLS can emit the standard approximation from `S⁻¹` and `σ²_j`; every model can at minimum emit its EW residual variance. river's `BayesianLinearRegression.predict_one(..., with_dist=True)` is the analogue. Finance users will want intervals before they want another model. |
| E13 | P2 | **Online model selection across grid combos** — generalize the lasso's `lam_selected` (EW mean squared OOS error, argmin) to the ridge/feature-set grids: emit `selected_<t>` naming the best combo and `pred_<t>__selected`. Same machinery, already proven in `lasso.rs`. |
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
| E19 | P2 | `linear_model.BayesianLinearRegression` | Exactly our Kalman with `q = 0` and fixed `obs_var` — **once Kalman gains a `standardize: false` switch** (today it always standardizes internally, so the correspondence is approximate). Add the switch, then document the equivalence and use it as a cross-library oracle (see `docs/TESTING.md` T-R2). |
| E20 | P2 | `drift.ADWIN`, `drift.PageHinkley` | A drift detector on the (already out-of-sample) residual stream, emitting a `drift_<t>` flag and optionally triggering the existing reset path. Complements halflife decay: decay forgets smoothly, drift detection catches breaks. |
| E21 | P2 | `anomaly.GaussianScorer` | Trivial for us: `resid_z_<t> = resid / sqrt(σ²_j)` is already computable from tracked state; emit it as an output field. One of the highest value-per-line items in this table. |
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
