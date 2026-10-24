# Huber regression in a stream: the analysis and the plan (2026-10-23)

A record, not a guide: it cites the plan and the review that prompted it.
Review round 9 found (PLAN §23, C-1) that `sgd(loss="huber")` on a target
with a level and no decay is squared loss under another name, because the
scale its cut is drawn in is the plain root mean square of every residual,
and the first residuals, the level's size, set it for good. The user then
asked for an analysis of Huber regression from first principles and a plan.
Every number here was measured on this machine on 2026-10-23 (load 20-50,
the user's jobs and four workers running; the numbers are ratios within one
script, not throughput); the scripts are `scripts/huber/m1_breakdown.py`
to `m5_request3.py`, each runnable as it stands
(`uv run python scripts/huber/m1_breakdown.py`).

## 1. What Huber regression is

**Least squares is the maximum-likelihood fit for normal errors and the
worst fit for a few wild ones.** Its loss is `e²`, so one row that misses by
50 pulls as hard as 10,000 rows that miss by 0.5. Huber (1964) replaced the
loss by one that is quadratic inside a cut and linear beyond it:

```
ρ_δ(u) = u²/2            |u| ≤ δ
       = δ|u| − δ²/2     |u| > δ         u = e/σ, the residual in units of a scale σ
```

Its derivative, the influence of a row on the fit, is `ψ_δ(u) = u` inside
the cut and `±δ` beyond it: bounded, so a row's pull stops growing with its
miss. The fit is the `β` that solves `Σ ψ_δ((yᵢ − xᵢ'β)/σ) xᵢ = 0`.

**Solving it is reweighted least squares.** Writing `ψ(u) = w(u)·u` with
`w(u) = min(1, δ/|u|)`, the equation is the normal equation of least squares
with each row weighted by `w(eᵢ/σ)`. The weights depend on the residuals,
which depend on the fit, so a batch fit iterates: fit, compute residuals,
reweight, fit again, until the coefficients stop moving (IRLS). Measured
with statsmodels' `RLM` on 20,000-row streams at a level of 1,000: 7 to 11
passes to a tolerance of 1e-8 and 10 to 16 to 1e-12, little dependent on
the contamination; the first pass does most of the work (at 10% outliers one
pass equals the converged fit; at 2% it closes most of least squares' error).

**The scale σ is part of the estimator, not a detail.** The cut is `δσ`,
so the estimator is scale-equivariant only if σ scales with the data, and
robust only if σ is. Three estimates of σ are in use:

| scale | definition | breakdown point | note |
|---|---|---|---|
| plain RMS | `√mean(e²)` over every residual | 0: one row at 1e6 sets it | what `huber` and `sgd` use today, as their docs state |
| Huber's Proposal 2 (1981) | `σ²` solving `mean(ψ_δ(e/σ)²) = E[ψ_δ(Z)²]` with `Z` standard normal (0.7102 at δ = 1.345) | about `E[ψ²]/δ² ≈ 39%` for location in theory | the clipped residual: a wild row counts `δ²σ²`, never more |
| MAD | `median\|e\| / Φ⁻¹(3/4)`, `Φ⁻¹(3/4) = 0.6745` | 50% | statsmodels' default; one quantile per target |

**δ = 1.345 is the cut at which the estimator keeps 95% of least squares'
efficiency when the errors are normal:** with `Z` standard normal, the
efficiency is `(2Φ(δ) − 1)² / E[ψ_δ(Z)²]`, 0.95 at 1.345. statsmodels'
`HuberT` and scikit-learn's `HuberRegressor` take 1.345 and 1.35.

**What Huber's estimator does not protect against.** Its weights read the
residual alone. A row whose *features* are wild (a leverage point) pulls
the fit toward itself and so has a small residual, and is not down-weighted;
contamination concentrated in one region of `x` acts the same way, because
there the outliers are the majority. Estimators that resist leverage
(Mallows and Schweppe weights, S- and MM-estimators, least trimmed squares)
either need the hat matrix per row, which `ewridge` has (`z'A⁻¹z`, the error
inflation), or several passes over the data, which a stream does not have.

## 2. How the three scales break, measured

Converged batch IRLS (statsmodels `RLM`, `HuberT(1.345)`, tolerance
1e-10), `y = 1000 + 2x + 0.5ε`, 20,000 rows, outliers of +50 on a share of
the rows with `x > 0` only (so they tilt the slope; the share of all rows is
given), median slope and range over 10 seeds, truth 2.0:

| outliers | least squares | MAD | Proposal 2 | plain RMS |
|---|---|---|---|---|
| 1% | 2.400 [2.334, 2.462] | 2.008 [1.999, 2.011] | 2.008 [1.999, 2.011] | 2.055 [2.037, 2.070] |
| 5% | 4.003 [3.798, 4.125] | 2.040 [2.027, 2.042] | 2.042 [2.030, 2.044] | 2.656 [2.570, 2.709] |
| 12.4% | 6.960 [6.825, 7.159] | 2.124 [2.110, 2.130] | 2.222 [2.192, 2.249] | 4.848 [4.745, 5.019] |
| 20% | 9.975 [9.753, 10.129] | 2.317 [2.283, 2.330] | 8.556 [8.245, 8.705] | 8.492 [8.181, 8.639] |
| 30% | 13.999 | 14.826 | 14.767 | 14.781 |

Three readings. The plain RMS is never robust: at 5% it gives back most of
least squares' error, because the outliers inflate the scale that is meant
to catch them. Proposal 2 holds to about 12% and breaks between 12% and 20%
here, below its theoretical 39%, because these outliers are one-sided and
on half the rows. The MAD holds to 20% and breaks at 30%, where the
outliers are 60% of the `x > 0` rows: the majority there, which no
residual-based estimator can tell from the truth (§1's last paragraph).
The residual bias at 1-5% (2.008, 2.040) is Huber's own: a wild row still
teaches at weight `δσ/|e|`, about 0.014 here, and 200 to 1,000 such rows add
up; the median regression, which takes only the sign, lands at 2.03 in the
docs' example for the same reason.

## 3. One pass: what a stream can and cannot do

**A stream weights each row once, by its residual against the fit before
the row, and never revisits it.** That is this library's rule (out of sample
by construction, CLAUDE.md rule 2) and its memory bound: the state is the
weighted sums, not the rows. So the online estimator is one pass of IRLS
whose fit improves as it goes: the early rows get weights from a poor fit;
the late rows, weights from a good one. Under decay the early rows are
forgotten; without decay they stay, at a share that shrinks as `1/n`. The
weights at entry converge to the batch weights, so the online fit converges
to the batch fit for a stationary stream, and differs from it in finite
samples by what the early rows did.

Measured, a numpy simulation of the library's recursion (the prior residual
weighted against the fit before the row, the Gram accumulated, a solve per
row), no decay, level 1,000, outliers as in §2, median slope over 5 seeds:

| outliers | online, RMS scale (today) | online, Proposal 2 | online, MAD | MAD + 1 re-weight pass every 256 rows | MAD + 3 passes | converged batch MAD (§2) |
|---|---|---|---|---|---|---|
| 0% | 1.999 | 1.999 | 2.000 | 2.000 | 2.000 | — |
| 1% | 2.181 | 2.199 | 2.008 | 2.008 | 2.008 | 2.008 |
| 5% | 3.067 | 3.104 | 2.041 | 2.040 | 2.040 | 2.040 |
| 12.5% | 5.825 | 6.066 | 2.183 | 2.156 | 2.155 | 2.124 |

Four readings, each of which decides something in the plan.

1. **With the MAD, one pass is the converged fit** at 1% and 5%, to the
   third decimal. The gap opens only at 12.5% (2.183 against 2.124), and
   even three extra passes over a trailing buffer of 256 rows close less than
   half of it. Re-weighting passes are not worth their memory (the rows) or
   their chunk-invariance cost (a pass must run on a row or clock cadence,
   never at a chunk's end).
2. **The RMS scale fails online far worse than it does in batch** (3.07
   against 2.66 at 5%): the start-up residuals, the level's size while the
   intercept is still near 0, enter the scale at full weight and, without
   decay, never leave. That is C-1, reproduced in the simulation.
3. **Proposal 2 does not repair a bad start.** Started from the plain RMS
   over the first rows, as it must be (its own fixed point at `σ = 0` clips
   every residual to 0 and stays there), it inherits the inflated scale, and
   once σ is wide every ordinary residual is inside the cut, so nothing is
   clipped and the recursion is the plain RMS over 0.71: it stays wide. The
   scale decision D1 took on 2026-10-23 (Proposal 2) would have moved `sgd`'s
   numbers without fixing C-1 under no decay. **D1 is withdrawn** by this
   analysis.
4. **The MAD needs no start rule.** The first residuals' size sets the
   median for a while, which widens the cut for those rows and down-weights
   none, and the median leaves them once the fit has the level, which the
   plain mean never does. The existing rule stays: until a target has a scale
   above 0 (every residual so far exactly 0), no row is down-weighted.

## 4. Outliers or a new regime: the trade-off a robust scale buys into

In a time series, a run of large residuals is either a burst of outliers or
a regime change, and no per-row rule tells them apart. A robust scale treats
every post-shift row as an outlier until the scale itself moves, so it must
slow adaptation; the plain RMS, which inflates at once, does not. Measured
in the same simulation with the fit's half-life at 500 rows, a +20 level
shift at row 10,000 and no outliers (rows until the trailing 200-row mean
|error| stays under 1.0, and the mean |error| in two windows after the
shift):

| scale | rows to re-converge | mean \|error\|, rows 0-200 after | rows 200-400 after |
|---|---|---|---|
| least squares (`δ = ∞`) | 2,047 | 17.50 | 13.23 |
| Huber, RMS scale (today) | 2,037 | 18.91 | 15.02 |
| Huber, EW-MAD scale, half-life 500 (the fit's) | 2,051 | 19.90 | 19.51 |
| Huber, EW-MAD scale, half-life 100 | 2,030 | 19.13 | 14.44 |

The fit's half-life sets the time to re-converge (about 2,000 rows for
every row of the table). Inside that time, the MAD with the fit's memory
holds the fit back for 400 rows (every post-shift row at weight about
`1.345 · 0.5 / 20 ≈ 0.03`) until the median of |e| has crossed to the new
level; the MAD with a shorter memory of its own (100 rows) adapts as the
RMS does. With 5% outliers before and after the shift the ranking is the
same (the library's `huber` on the same shift, today: 2,064 rows, 18.92 and
15.16). So the scale's memory is a lever of its own: shorter than the fit's,
it follows a shift sooner at the price of letting a longer burst of outliers
widen the cut. A shift detector (`breaks`, `drift`) resetting the scale is
the other way; it is a later item.

## 5. What the library does today

- `huber` (`robust.rs`): one Gram per target; each row's weight is
  `w · min(1, δσ/|e|)` with `e` the prior residual and `σ²` the EW mean of
  `e²` over every predicted row (`sig2`, `wsig`), aged on every row. The
  docstring and MODELS.md state that σ is "the plain spread rather than a
  robust one" and that "a burst of outliers widens the cut ... until the
  exponentially weighted mean forgets them".
- `sgd(loss="huber")`: the gradient `p − y` clipped at `±δσ`, Huber's ψ,
  with the same `sig2` recursion over the rows it scored; the module doc
  says the cut "narrows as [the residuals] leave the mean", which holds only
  under decay (C-1).
- `quantile`: the check loss smoothed over a band of `quantile_eps · σ`;
  σ sets only the band's width, not a weight, so a wide σ smooths it toward
  the mean rather than switching it off. Out of this plan's scope.
- Nothing weights by leverage; `ewridge`'s error inflation computes the
  quantity (`z'A⁻¹z`) a Mallows weight would need.

## 6. What the API policy allows

RELEASE-READINESS's policy table: from 1.0, "a fix to numbers that were
wrong against the model's stated definition" is a minor; "a default or a
meaning changed" is a major. The plain RMS is the stated definition, in the
docstring and in MODELS.md, with its weakness stated. So the scale cannot
be replaced in 1.x; it can be added as an option with the present behaviour
as the default, recommended in the docs, and made the default at 2.0. The
same holds for `sgd`'s cut. What *is* a fix within the policy: `sgd.rs`'s
module doc sentence that does not hold (C-1's documentary half), and any
test that pins it.

## 7. The plan

Built as PLAN task 271 (D1 withdrawn; C-1 reassigned from task 270 (a)).

| # | item | test or measurement that holds it |
|---|---|---|
| P1 | **`huber_scale`** on `po.spec.huber` and `po.spec.sgd(loss="huber")`: `"rms"` (today's, the default) or `"mad"`: the exponentially weighted median of \|e\| over `Φ⁻¹(3/4)`, one `EwQuantile` at level 0.5 per target (`stats.rs`: within `tanh(1/128)` = 0.78% relative, so the cut is within 0.8% of the exact EW-MAD's), aged on the model's clock as `sig2` is. The scale above 0 rule unchanged. State: a new per-target field, `#[serde(default)]`; `SCHEMA_VERSION` 63 with the v62 sets kept and a loader per rule 5; the fixtures freeze both forms | the second opinion: `huber_scale="mad"` on §2's stream at level 1,000, no decay, against statsmodels `RLM(HuberT(1.345), scale_est=mad)` within the tolerance measured here (0.01 on the slope at 1% and 5%); the breakdown table at 1% and 5%; `sgd` huber's slope nearer the truth than squared loss's at level 1,000 (fails today: equal); state fixtures; chunk invariance |
| P2 | **`huber_scale_half_life`**: the scale's own memory, default the model's `half_life` (today's behaviour under `"rms"`, so no number moves), a shorter one for a stream with regime changes (§4) | §4's shift test: with the scale's half-life at a fifth of the fit's, the post-shift error in rows 200-400 within 10% of least squares'; with the fit's, documented as slower |
| P3 | **`sgd`'s documentation**: the module doc's "narrows as they leave the mean" replaced by the mechanism and its condition (decay), and the `huber_delta` entries of `sgd` and `huber` state which scale each setting draws the cut in, with §2-§4's numbers | the docstring tests that pin the text, updated |
| P4 | **MODELS.md's `huber` section rewritten from first principles**: §1-§4 of this record in the guide's voice (the loss, ψ, the weights; the three scales and their breakdown table; one pass against the converged fit with the table; the regime trade-off with the table; what Huber does not protect against), the example run under `huber_scale="mad"` beside today's, and the recommendation | `test_models_page` runs the examples; `test_no_plan_references` holds the guide to cite no record |
| P5 | **The release comparison**: a `huber` and an `sgd` huber spec with `huber_scale="mad"` in `release_probe.py`'s workload (J-2's extension lands in task 270 (e); this adds the new path) | `test_this_build_runs_the_whole_workload` |
| P6 | **2.0**: `huber_scale` defaults to `"mad"`; recorded in the policy's "at the next major" list now | the CHANGELOG's *Unreleased* names it under *Added* with the 2.0 note |
| P7 | *Later, measured first*: Mallows weights (`w · min(1, c/√h)` with `h = z'A⁻¹z`, the error inflation's own quantity) as `huber_leverage`; a shift-aware scale (reset on a `breaks` event); `sgd`'s intercept started at the target's running mean (a start that removes the level problem for every loss; moves `sgd`'s early numbers, so an option). None of the three is in task 271 | each with its own §2-§4 style table before it is built |

**Not planned, with the reason.** Re-weighting passes over a trailing
buffer (§3: three passes close under half of the 12.5% gap and nothing at
5%, for the rows' memory and a cadence rule). Proposal 2 (§3: it does not
repair a bad start, and its breakdown is lower than the MAD's at the same
cost). Changing the default in 1.x (§6).

**Decided** (the user, 2026-10-23: "Build this ... Move away from rms
now. Your reco on the rest."): (a) `huber_scale` defaults to `"mad"` in
this minor, against the policy's row for a changed default, by the user's
call; `"rms"` stays as an option so 1.2.0's numbers can be reproduced, and
the CHANGELOG declares the change under *Changed* with its measured size and
the setting that restores the old numbers. P6 is therefore done by P1.
(b) `huber_scale_half_life` built now. (c) P7's three items stay out of
task 271.

## 8. Five requests from a downstream project, analysed (2026-10-23)

A project that uses the bank, `po.gram`, `po.eval` and `marginal` asked for
Huber regression to run through each of them: a bounded influence for large
residuals and a scale learned causally from the stream. The user asked for
the requests to be analysed and built after task 270 and task 271. Each is
taken on its merits against what the library has; one is refused as asked,
with the measurement, and replaced by the construction the request itself
names.

| # | the request | what the library has | verdict | built as |
|---|---|---|---|---|
| 1 | `loss="huber"`, `huber_c` and a robust scale with its own half-life on `ewridge` and `rls`; the weight from the prediction made before the row is learned; with `feature_sets`, the largest set's residual | `po.spec.huber` is that model: `ewridge`'s sums with each row weighted by `min(1, δσ/\|e\|)` from its out-of-sample residual, per target; task 271 gives it the robust scale (`huber_scale="mad"`, the default) and the scale's own memory (`huber_scale_half_life`). Missing: `feature_sets`, `ridge_scale` (so the `rls`-equivalent Huber is `huber(ridge_scale="sum")`, as `rls` is `ewridge(ridge_scale="sum")`, a tested identity), an emitted scale, and a stated rule for a delayed label | build the gaps on `huber`; no `loss=` on `ewridge` or `rls` (one model kind per estimator: a second spelling of `huber` doubles the surface and its tests); `ewridge`'s and `rls`' docs point to `huber`. The requested scale, an EWMA of \|r\|, has breakdown 0 (one row at 1e6 sets it), the failure §2 measures for the RMS; the EW-MAD is its robust form | task 272 (a) |
| 2 | a Huber loss in `po.eval.sums` / `from_sums`: additive sums of `ρ_c(r/σ)·σ²` given a per-row scale, mergeable like `sse` | `sse`, `mse`, `r2`, `ic` and the rest, mergeable | build: `huber_sse = Σ w · 2σ²·ρ_c(r/σ)` (equal to `sse` inside the cut, linear beyond it), `huber_mse = huber_sse / w` in `from_sums`; the scale a column the caller names or the spec's emitted scale (1 above). The scale must be causal, read before the row, or a spike inflates its own σ and shrinks its own loss: stated, and the emitted scale is causal by construction | task 272 (b) |
| 3 | per-target row weights on a Gram's cross-moments `x·y_t` only, `x·x'` keeping the shared weight; or the pseudo-response `y* = ŷ_ref + σψ(r/σ)` | targets as columns on one shared Gram | **refused as asked**: `β_t = (X'X)⁻¹X'W_t y_t` is not weighted least squares; a down-weighted row's target is shrunk toward 0, not toward the fit. Measured (`scripts/huber/m5_request3.py`, level 1,000, slope 2, 5% outliers): slope **−59.4** after 30 passes, where full weighted least squares and the converged Huber fit give 2.0393. **The pseudo-response is the right construction** and needs no library change: it is an ordinary target column on the shared, unweighted Gram, and iterating it is Huber's algorithm H (1981), which converges to the Huber estimate with one factor reused every pass: slope 2.620 / 2.199 / 2.042 / 2.0393 after 1 / 2 / 5 / 10 passes, equal to statsmodels' `RLM` to four digits. The tail pass's one Gram per target is freed the same way | task 272 (b): `po.gram.pseudo_response(y, pred, scale, c)`, an expression helper, and a GRAM.md section with the passes table |
| 4 | a robust per-bin mean in `marginal`: `bin_stat="huber"` or a winsorized mean, a running per-bin location and a per-feature scale | West's weighted mean per cell (`bin_mean_y`) | build `bin_stat="huber"`: each cell's location moved by the clipped step `(w/W)·σψ((y − m)/σ)` around its own prior mean, the online M-location. The scale per *target* (an `EwQuantile` of \|y − m_cell\| per target), not per feature: per feature it is `p·T` quantile sketches, a few MB at 10⁴ features × 50 targets for a scale the cells share. Default `"mean"`, so no number moves | task 272 (c) |
| 5 | a causal robust scale shared across specs: an EWMA absolute deviation of a column, the label delay respected, usable as a weight or a derived column | the window operators (`po.ewm_mean` & co., elementwise inputs such as `abs` accepted, `closed="left"` to leave the row out); `EwQuantile` in the core, not exposed as an operator | build `po.ewm_quantile(input, quantile=, half_life=, window_size=, closed=)` on the window machinery, so the EW-MAD of a column is `po.ewm_quantile(pl.col("r").abs(), quantile=0.5, ...) / 0.6745`; for a model's residual the emitted scale of 1 is the primitive, and it respects `embargo` because it is the model's own state. A weight derived from a *target* column must leave the label out until it matures: `closed="left"` leaves the row out, and a delay longer than one row is the caller's lag; stated in WINDOWS.md | task 272 (d) |
| — | trees: XGBoost's `reg:pseudohubererror` with `huber_slope` | `po.gbdt` builds the engines' training input and leaves the objective to the caller's train call | nothing to build; GBDT.md names the objective (and LightGBM's `huber` with `alpha`) and the scale rule | task 272 (d), one paragraph |

The requester's ordering (1, then 2, then 3-5) stands, and 1 is the one
that only the library can do: the bank predicts and learns inside one call,
so the weight must be computed in its state.

**The requester's reply (2026-10-23).** They checked request 3 and
withdrew it (slope 1.92 for 2.00 with outliers independent of `x`; the
bias grows with leverage), confirmed the pseudo-response iteration (a step
of 5e-5 by pass 5, 8e-9 by pass 10, equal to a direct reweighted fit to
5e-15), and moved requests 2 and 3 into their own code. Task 272 keeps
(a) `huber`'s gaps, (c) the robust bin mean and (d) `po.ewm_quantile`, and
narrows (b) to the GRAM.md section on algorithm H. Three points from §2-§4
went back to them: request 1 is `po.spec.huber` (one online pass with the
MAD scale equals the converged fit at 1% and 5%, so their "pass 3" needs
no re-streaming); Proposal 2 is additive but broke between 12% and 20%
one-sided contamination here where the MAD held, and online it keeps an
inflated start; and a first step "at `b = 0`" clips every row of a target
with a level, so the first step should be taken at an intercept-only fit
(the target's median).
