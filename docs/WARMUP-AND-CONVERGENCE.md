# Warmup and convergence: not using a model before it is ready — a design to iterate on

**Status: built for `ewridge` in 0.9.0 (2026-09-21).** Sections 1–4 are the
design as built; §8 says what was built and for which models; §7.4 records
what the identities found when the implementation was held to them, which
changed one claim in §2.1. Every number below was measured in this
repository, not recalled; the design-time numbers (2026-09-20) are
reproducible from `bank.gram()` and the emitted `n_eff`, the build-time
ones from `crates/online-core/tests/readiness.rs` and
`tests/test_readiness.py`. Section 5 is the evidence. Section 6 is what was
tried and dropped, with the measurement that dropped it, so it is not
proposed again without new evidence. Section 7 is open.

---

## 0. The goal

> One or more settings that prevent us from using models that have not yet
> warmed up or reached an acceptable amount of convergence.

Two constraints on the shape of any setting, both the user's:

- **State the intent, never a value that needs a formula in your head.**
  Warmup is a *fraction of steady state*, not a count of rows or halflives,
  because a fraction cannot be set to an unreachable value.
- **The library may introduce a new concept if that is the most
  user-friendly way to track the goal.** Concepts are welcome; calibration
  is not.

Today the only such setting is `min_periods`: an absolute floor on `n_eff`
(the decayed effective sample), defaulting to one observation per unknown.
It is hard to use for three separate reasons (§5.1): its unit is weight, not
rows; under decay it tops out at a ceiling nobody can compute in their head,
and with a clock column nobody can compute at all in advance; and when it is
set above that ceiling the output is null for the life of the stream, in
silence. That silence is the pain this design exists to remove.

---

## 1. What "ready" decomposes into

"Ready" is not one question. The measurements found three, answered by
different quantities, and one number cannot serve two of them (§5.3).

| question | quantity | kind | user setting |
|---|---|---|---|
| **Warmed up?** Has the decay window filled toward steady state? | `settled_frac` = `1 − 2^(−T/h)`, `T` = decay time elapsed | continuous, 0 → 1 | `min_settled_frac` — the gate the goal asked for; off by default, set by the user who knows the process has regimes (§4.1.1) |
| **Identified?** How much of each coefficient did the data determine, and how much the ridge? | `support_coef_j = (G_raw·G⁻¹)_jj`, the shrinkage matrix's diagonal (§2.2) | continuous, `[0, 1]` per coefficient | none — a **diagnostic**, not a gate (§5.7); no tolerance to set |
| **Noise?** Would estimation error swamp *this* prediction? | `error_inflation` = `√(1 + h(x))`, `h = edf/n_Kish` for the stream (free), `h(x) = x'Σ̂⁻¹x / n_Kish` per row (opt-in) — the estimation variance over the noise (§2.1) | continuous, ≥ 1 | `max_error_inflation` — replaces `min_n_eff`/`min_periods` (§2.1, §5.8) |

**Convergence** in the goal's sense — the estimate has stopped moving — is
`settled_frac` for a stationary target: after `k` halflives the effective
sample is within `2^(−k)` of its steady state, rate-independently (§5.2).
For a *moving* target the coefficients never converge, and that is not a
warmup question but a drift question, which the library's drift detection
(`emit_drift`, `drift_action`) already addresses. A coefficient-stability
signal beyond this is **open** (§7.1).

---

## 2. The settings (current proposal)

| setting | unit | default | meaning |
|---|---|---|---|
| `min_settled_frac` | fraction of steady state, `[0, 1)` | `0` (off, §4.1.1) | withhold predictions until the decay window has filled this far. Bounded, so it cannot be set unreachably; `>= 1` is refused. Off by default because under a stationary process the mean-form fit is unbiased from row one and its variance is the noise gate's; **set it (`0.5` = one halflife) when the process has regimes or seasons the halflife was chosen to average across.** |
| `max_error_inflation` (replaces `min_periods`) | ratio, `> 1` | `√2 ≈ 1.41`, i.e. `h = 1`: withhold while the estimation variance of the prediction exceeds the noise it is fitting (≈ today's one-per-unknown on average, §5.8) | withhold predictions while the stream's expected `error_inflation = √(1 + edf/n_Kish)` is above this (§2.1); `O(1)` per row. Tracks the model, so adding a feature moves the gate with it; nothing to maintain. The per-row `error_inflation` field (opt-in) gives the same quantity for each prediction. `1.1` says rough is not acceptable, `2` says it is. Refused `<= 1`. |

Two settings; a user who sets anything sets `min_settled_frac`. (A third,
`full_rank_halflives`, was a tolerance on a boolean and is gone — §2.2.)

**Interaction.** Output is withheld while *any* active gate is unmet. On
clean, weakly-decayed data the gates fire in a fixed order — rank (as a
diagnostic) at one row per unknown, `max_error_inflation` a few rows later,
settled much later — so `min_settled_frac` is the one the user feels. On a
short-halflife, many-feature spec `max_error_inflation` can be unreachable
(the `n_eff` ceiling is below what the ratio needs); that is now *reported*
(§3), never silent, and it is the correct verdict: the model *is* noise.

### 2.1 The noise gate: options

The purpose of `min_periods` (user, 2026-09-21): **avoid outputting model
values that are known to be noise**, with a setting that needs no
maintenance when the number of variables changes. `min_periods` fails the
second half — it is an absolute floor in weight units, so adding a feature
silently lowers the observations per unknown.

**The standard (user, 2026-09-21): the setting rests on theory, not on a
sweep.** A sweep validates the cases swept and is more likely to miss a
case than to prove one. So the statistic is the exact one, and the sweep
(§7.4) verifies the implementation against identities the theory predicts.

**What theory gives exactly.** For every model whose fit is linear in the
past targets — `ewridge` at any ridge, `rls`, the EW mean / variance /
covariance models — the estimation variance of a prediction at `x` is
exact, with no assumption on the design, the row rate, the weights or
stationarity:

```
Var(ŷ(x)) = σ² · h(x),    h(x) = x' G⁻¹ G₂ G⁻¹ x
G  = Σ λⁱ wᵢ xᵢxᵢ' (+ ridge)     the Gram the model already keeps
G₂ = Σ λ²ⁱ wᵢ² xᵢxᵢ'             the same accumulator, decay and weight squared
```

`σ²` multiplies both the noise floor and the estimation error, so the
inflation **`error_inflation = √(1 + h(x))` needs no noise estimate**.

**What `G₂` buys, and why it is not kept** (user's question, 2026-09-21).
Everything that makes `h` useful comes from `G⁻¹`: an unsupported
direction makes it huge (the collinearity-break row), an outlying `x` reads
large through it, the ridge is in it. `G₂` contributes only the
*directional* difference between how recent and old rows shaped the
design, because `λ²ⁱ` favours recent rows more than `λⁱ`. If the design's
covariance is constant across the memory window — the assumption every
ridge-regression standard error makes — then `G₂ = (s₂/s₁)·G` exactly,
with `s₁ = Σλⁱwᵢ = n_eff` and `s₂ = Σλ²ⁱwᵢ²`, and `G₂` carries nothing.
Under that one named assumption the statistic collapses to

```
h(x) = x' Σ̂⁻¹ x / n_Kish      Σ̂ = the ridged normalised Gram the model already keeps
n_Kish = s₁² / s₂               one extra f64, decay λ², weight w²
```

`n_Kish = (1+λ)/(1−λ)` at steady state (≈ `2·n_eff`), so on average
`h → k/n_Kish` and the `k/n` formula of §5.8 is the special case. The
scalar models are the `k = 1` case, `h = 1/n_Kish`. What is given up is
exactness when the design covariance moves *within* a halflife or two —
second-order for a readiness gate, and design drift is a drift question,
not a warmup one. `G₂` would earn its `k×k` only for a per-row statistic
of design drift, which nobody has asked for.

**What the identities found when the build was held to them (§7.4,
2026-09-21).** The claim above was too strong in one place. The collapse
`G₂ = (s₂/s₁)·G` holds *in expectation* on a stationary design, and the
gate's average `edf/n_kish` tracks the observed out-of-sample error within
2% down to `n_kish ≈ 2.3 k_eff`. But the *per-row* form
`x'Σ̂⁻¹x / n_kish` overstates the exact conditional variance
`x'G⁻¹G₂G⁻¹x` at small `n`, by 10% at `n_kish/k_eff ≈ 12` and by 54% at
2.3, because `G` and `G₂` share their sampled rows: the recent rows that
dominate `G₂` are the ones `G` has already bent toward, so their leverage
through `G⁻¹` is smaller than a fresh row's. Sampling correlation, not
design drift. So `G₂` does buy something: exactness of the opt-in per-row
field at short halflives. What it buys is in the conservative direction
(the field overstates the risk, never understates it), and the gate does
not need it. The decision stands; the reason is now the measured one.

What the exact statistic protects against that no average can:

| hazard | `h(x)` |
|---|---|
| weights, capped gaps, irregular clocks | exact in the scalar: `s₂` saw the same weights and decays the fit saw |
| a row whose `x` breaks a collinearity the fit relied on | large for that row, small for rows that respect it — the §5.7 hazard (1.022 vs 0.104) caught **per prediction**, where the rank flag can only name it per stream |
| an outlying `x` on a settled model | same mechanism |
| ridge | `G⁻¹` is the ridged inverse, so shrinkage is in `h` exactly. The ridge's *bias* is not, by design: under the mean-form ridge it never fades, so it is the user's regularisation, not warmup noise |

**Where the `k²` comes from, and the two tiers** (user's question,
2026-09-21). With the Cholesky factor `Σ̂ = LLᵀ` already kept for solving,
`x'Σ̂⁻¹x = ‖L⁻¹x‖²`: one forward substitution, `k(k+1)/2` multiply-adds —
the same count as the rank-one Gram update the model already does per
row. Unavoidable for a *per-row* value: a per-prediction variance in a
ridge model is a leverage, and a leverage is a solve. But the statistic
answers two questions with different costs:

| question | statistic | per row |
|---|---|---|
| **Is the model ready?** (per stream) | the average of `h` over rows drawn from the design: `tr(Σ̂⁻¹Σ̂_raw)/n_Kish = edf/n_Kish`, `edf` free from the Cholesky pivots on the solve schedule (§5.4) | **`O(1)`** |
| **Is *this* prediction safe?** (per row: the collinearity-break row, the outlying `x`) | `h(x) = ‖L⁻¹x‖²/n_Kish` | `k²/2` |

So the **gate runs on the average**, `√(1 + edf/n_Kish) ≤
max_error_inflation`, at no per-row cost — with a tiny ridge `edf = k` and
this is the `k/n_Kish` formula of §5.8, now derived rather than assumed.
The **per-row `error_inflation` is the field that costs `k²/2`**, opt-in
like the library's other diagnostic fields; a user who wants
per-prediction protection pays for it, and can gate on it downstream.
State either way: one `f64` (`s₂`). Nothing per solve.

**Limits, stated.** Homoskedastic noise independent of `x`; under
heteroskedasticity the ratio is approximate, not wrong. A moving target
adds staleness bias that no variance sees — drift detection's job.
`lasso` is post-selection: `h` on the active-set Gram is the theory-backed
approximation (its degrees of freedom are the active count, Zou–Hastie–
Tibshirani 2007), labelled approximate. Where theory gives nothing, the
field is null and this setting does not gate: `sgd`, `pa`, `ftrl` have no
linear fit; `min_settled_frac` is their gate.

**The options, for the record:**

| option | gate | verdict |
|---|---|---|
| **A. `obs_per_unknown`** | `n_eff >= r × n_coef` | promises a ratio, not an error; the user must know the textbook rule. **Dropped** (§6): where theory gives `h` it is dominated, and where it does not, a count is a guess dressed as a gate. |
| **B. formula** `√(1 + n_coef/n_eff)` | average over a stationary Gaussian design | the special case of `h`; kept as the *explanation* of what today's `min_periods` was doing (§5.8), not as the statistic. |
| **C. exact `h(x)`** | `√(1 + h(x)) <= max_error_inflation` | **chosen.** Exact for the linear-fit family, per prediction, no noise estimate, one `O(k²)` per row. |
| **D. empirical** out/in residual variance | measured overfitting | needs its own warmup, circular; noisy. Not proposed. |

`max_error_inflation` replaces `min_periods` rather than sitting beside it:
one setting, one question. The unreachable case stays reported (§3): under
a short halflife `h` never falls below the threshold, which is the right
answer said out loud.

### 2.2 Identification: from a flag with a tolerance to an exact share

`full_rank` (boolean) with `full_rank_halflives` (its tolerance, default
20) is dropped, 2026-09-21. It was a threshold on an eigenvalue ratio —
a number calibrated by measurement (§5.6) — and under a real ridge no cut
can be made honest (§5.4). Both jobs it did are done exactly now:

- **For predictions**, the hazard it warned about — a prediction leaning
  on a direction the data never showed — is `h(x)`, per row (§2.1).
- **For coefficients**, with the mean-form ridge `G = G_raw + λI` exactly,
  so `S = G_raw·G⁻¹` is the shrinkage matrix in coefficient space and

  ```
  support_coef_j = S_jj ∈ [0, 1]
  ```

  is **the share of coefficient `j` determined by the data rather than the
  ridge**. A duplicated pair reads 0.5 each — the "2 and 2" split of §5.7
  in numbers. A feature flat for `n` halflives slides toward 0 smoothly
  instead of falling off a cliff at a tolerance. A heavy ridge reads low
  everywhere, which is the truth about that spec, not a blind spot.
  `Σ_j S_jj = edf`, the effective degrees of freedom, so this is the
  per-coefficient decomposition of the statistic §5.4 measured.

Exact, threshold-free, per coefficient; `O(k²)` on the solve schedule
given the `G⁻¹` already computed for `M`. Emitted with `coef` (same
schedule, same shape), with `min(support_coef)` and its feature name in the
summary. Same exclusions as `h`: intercept column out; `standardize` puts
it in standardised coefficient space, where the per-coefficient meaning is
unchanged. Removes a setting.

**Settled (2026-09-21): warn, and only where the fit is in use.** The
field is the instrument — exact, per coefficient, threshold-free — and the
warning is only the pointer that makes a user who never thought to look
learn the field exists. That is the whole of its job, which is why it is
once per (spec, group) and suppressible by category, and why no gate was
added: predictions are correct under rank deficiency (§5.7), so withholding
them would take away correct output. Duplicate feature *names* are already
refused at validate time, so what is left for a runtime signal is exactly
the data-dependent case nothing static can see.

The cut stays at `0.5`, the equal-parts point, with one maturity condition:
the warning is raised only on a row whose prediction the gates let through
(`reason == 0`). Without it the *first* solve of any spec fires it — one row
against `k` slopes is under-determined by construction, and the warning
cannot be retracted. Found verifying the shipped 0.9.0 wheel: an ordinary
two-feature fit warned at `n_eff = 1.00` ("0.27 data and 0.73 ridge") and
read `support_coef = 1.00` from the next row to the end of the stream. A
user who chose a heavy ridge still hears it once, which is right — their
coefficient really is mostly prior.

The condition does not silence a true positive on a mature fit.
`tests/test_window_budget.py`'s spec sets `min_periods = 0`, so it solves at
one row, where the slope has no variance to read and is entirely the ridge,
and `halflife = 1e9` leaves no cadence to refit: `pred_y` is one constant
for all 300 rows, `coef` is `[4.100824, 0.0]`, and `support_coef` reads 0.00
on every row. The warning names that correctly — the feature finding a
degenerate spec in this repository's own tests.

---

## 3. What the output carries

Per row, inside each spec's struct beside `n_eff`, on by default:

| field | type | bytes/row (§5.9) | meaning |
|---|---|---|---|
| `settled_frac` | Float64 | 8 | progress toward steady state; **null when the spec has no decay** (no steady state to settle toward) |
| `support_coef` | Float64 × k, on the `coef` schedule | ~0 per row (only on `coef` rows) | share of each coefficient determined by the data rather than the ridge (§2.2); null for models with no Gram |
| `error_inflation` (**opt-in**: costs `k²/2` per row, §2.1) | Float64 | 8 | `√(1 + h(x))`: how much this row's estimation variance inflates its expected error over the noise floor; the linear-fit family, approximate for `lasso`, null for `sgd`/`pa`/`ftrl`. The gate uses the stream average, which is free; the summary reports it |
| `withheld_reason` | Enum | 1.41 | why `pred_*` is null this row: `below_min_settled_frac`, `above_max_error_inflation`; **null once the row is real**. Never a String (16 B/row even when every value is null, §5.9). |

Per group in `summary()`: `settled_frac`, `error_inflation`,
`min_support_coef` and the feature it belongs to, `n_eff_settled` (the
effective sample once settled; null before settledness is high enough to
estimate it), and `n_coef`.

Warnings, raised from the Python layer by inspecting the returned fields
after each chunk (as `ConsumedSourceWarning` already does; there is no
Rust→Python warning channel), **once per (spec, group)**:

- **a coefficient more ridge than data** (`support_coef < 0.5`, if the
  open point in §2.2 lands on warning), naming the feature(s);
- **`max_error_inflation` unreachable**: the stream has settled and
  `error_inflation` is still above the ceiling, so output will never appear
  — with the fix in the message (raise the halflife to at least *h* rows,
  or raise `max_error_inflation` to at least the settled value).

Only report, never warn, when output appears but is degraded (a user who
raised `max_error_inflation` asked for it): a short halflife can be
deliberate.

---

## 4. Defaults

### 4.1 `min_settled_frac`: why its default kept changing, and the decision

This setting is the gate the design exists for, and across the discussion
its default was proposed on, then off, then off with a rule written to
justify it ("diagnostics on, gates off"), then reopened. That was not
indecision about whether the setting should exist. It should: it is the
warmup half of what `min_periods` is used for (§5.3 shows the other half is
a different question) restated in a unit a person can reason about, and
unlike a downstream filter it is portable to the CLI, applies to
`predict()`, and lives in the one place the model's intent lives. The oscillation was about
the default, and it came from two goods that genuinely conflict:

- **The goal.** Never use a model that has not warmed up. Taken seriously,
  that is a gate that is *on* by default, at some value.
- **The scar.** Earlier the same day, 0.8.1 shipped an on-by-default check
  whose default was derived from the wrong quantity, a halflife standing in
  for a session length, and it refused every intraday stream whose sessions
  were shorter than the model's memory. After that, "a default gate that
  withholds output" read as the thing to avoid, and the rule "diagnostics
  on, gates off" was written to avoid it.

The rule was too broad, and it is the reason the position kept moving:
applied to this design it argues against its own purpose. What went wrong
in 0.8.1 was not that a gate was on by default; `min_backwards_jump` is on
by default and is right. What went wrong was that the default was derived
from a quantity with the wrong meaning, and the refusal did not say which
setting to change. The rule that actually separates the good defaults from
the bad one is:

> **A gate may be on by default when its default is derived from a quantity
> the user already chose, the derivation is measured to hold, and every
> refusal names the setting that caused it.**

`min_backwards_jump` passes: derived from `max_dclock`, measured, named in
the message. 0.8.1's `min_session_clock` failed the second clause.
`require_full_rank` failed the first, there being no user-chosen quantity
to derive it from, and predictions were correct anyway (§5.7). And
`min_settled_frac` passes the first and third — its natural default
derives from the user's own halflife, and every withheld row carries
`withheld_reason = below_min_settled_frac` — which is what the position
below first rested on. It **fails the second**, as §4.1.1 shows.

The position first held here was `0.5`, one halflife — "before it, the
fit is dominated by less history than the user said matters" — with the
promise that it would not move again without something breaking one of
the three clauses. Something did.

#### 4.1.1 The options, analysed (2026-09-21)

**What the gate actually guards.** The Gram family is mean-form: the
comoments are normalised by the weight sum, so under a stationary process
the fit is **unbiased from the first row**; only its variance is high, and
the noise gate sees that exactly — `n_Kish` at row 9 is ≈ 9 whatever the
halflife. Waiting for `settled_frac = 0.5` on a halflife-1000 model buys
nothing over a tighter `max_error_inflation` under stationarity: at
`1.05` the noise gate opens near row 40, and a 40-row fit *is* within 5%
of the noise floor. What the settled gate guards is **bias from
unrepresentative history** — 40 rows of one regime when the halflife was
chosen to average across regimes. That is real, but it is knowledge of the
process, no statistic in the model can size it, and it is exactly the
within-window non-stationarity the rest of this design assumes away (it is
why `G₂` went, §2.1). A nonzero default therefore asserts a hazard the
design otherwise assumes absent — that is the second clause failing:
halflife → "not ready before half the weight" is not a derivation that
holds.

| default | what it says | cost | standing |
|---|---|---|---|
| **`0` (off)** | "warmed up" = the noise gate; settledness is reported, and gated by the user who knows the process is not stationary | none under stationarity; a user with regimes or seasonality sets it | consistent with the design's own assumption; `settled_frac` and the summary keep the state visible |
| `0.5` (one halflife) | half the eventual weight must be present | a stream shorter than a halflife emits nothing; a late-appearing group loses its first halflife | a convention: the "half" has no derivation, and under stationarity it withholds correct output |
| `0.75` / `0.875` | two or three halflives — burn-in folklore | the same, two to three times over | the same, stricter |
| derived from `max_error_inflation` | no separate default | — | this *is* `0`: under the assumption the noise gate already tracks readiness |

**Position: default `0`.** The setting stays, and is the gate the goal
asked for — a user whose process has regimes sets `0.5` and gets exactly
"do not use this model before it has a halflife of history". What the
default no longer does is impose that on the user whose process is
stationary, for whom it would withhold output the library's own theory
calls correct. Out of the box nothing changes: `predict()` and the CLI
output from the noise gate, `settled_frac` is on every row, and the
summary shows it. The 0.8.1 scar is not the reason for this position; the
theory is, and the position moves again only if a statistic is found that
sizes the representativeness bias — none is known.

### 4.2 The rest

| item | default | why |
|---|---|---|
| row fields, summary columns, the two warnings | on | no behaviour change; the pain was invisibility |
| `max_error_inflation` | `√2` | today's one-per-unknown floor re-expressed in the unit of its purpose (§5.8); its failure mode is now named |
| `full_rank_halflives` | dropped | a tolerance on a boolean; replaced by the exact `support_coef` (§2.2) |
| `require_full_rank` | dropped | §6 |

**Out of the box, predictions change only where today's `min_periods` and
the `√2` noise gate disagree** (they coincide as `k` grows, §5.8);
otherwise the output is unchanged except that a null now says why, every
row says how settled the stream is, and a coefficient the ridge determined
more than the data did is named instead of silent.

---

## 5. The evidence

### 5.1 Why `min_periods` is hard

- The `n_eff` ceiling under decay, regular spacing `d`, halflife `h`:
  **`1/(1 − 2^(−d/h))`**, matched to every digit (h=50: d=1 → 72.636,
  d=5 → 14.933, d=0.5 → 144.770). `1.44·h` is only the large-`h` limit (7%
  low at h=5).
- A weight column scales the ceiling by the **mean weight, exactly**.
- With a clock column the ceiling depends on the row rate, which the spec
  does not know: **it cannot be validated in advance**, only observed once
  settled. (A row-count clock has `d = 1` and can be validated.)
- `validate` bounds `min_periods` only at `>= 0` and by list length: an
  unreachable value emits null forever with no message.
- Inverse, for the message: **`h = −1/log2(1 − 1/N)`** rows makes `N`
  reachable (N=10 → 6.579, N=210 → 145.214, exact).

### 5.2 Warmup as a fraction of steady state is rate-independent

Halflife 50 clock units, three row rates:

| rows per halflife | ceiling | at 1 halflife | at 2 | at 3 |
|---|---|---|---|---|
| 10 | 14.933 | 0.5000 | 0.7500 | 0.8750 |
| 50 | 72.636 | 0.5000 | 0.7500 | 0.8750 |
| 100 | 144.770 | 0.5000 | 0.7500 | 0.8750 |

Ceilings differ tenfold; the fraction is identical, because it is
`1 − 2^(−T/h)` by construction. So `settled_frac` needs the halflife and a
running total of decay time, and nothing about rate or ceiling — computable
with any clock from the first row.

### 5.3 Warmup and identification are different questions

k=4, halflife 50, tiny ridge:

| rows | identified (edf/k) | settled (n_eff/ceiling) |
|---|---|---|
| 6 | 1.000 | 0.080 |
| 50 | 1.000 | 0.500 |
| 400 | 1.000 | 0.996 |

Identification saturates ~50× earlier. One number cannot serve both.

### 5.4 The rank diagnostic

*(Evidence kept; the verdict moved on 2026-09-21. These measurements are
what showed a boolean with a cut cannot be made honest under a ridge; the
exact per-coefficient share replaced it, §2.2.)*

- Tiny ridge: `edf/k` = 0.25, 0.50, 0.75 at rows 2–4 and **1.000 at 5 =
  k+1** (centring costs a row; no-intercept solves the raw system and is
  full at k). So it reproduces today's default for free.
- Duplicate column: pins at **0.750 from 4 rows to 400** (min eigenvalue
  ~1e-16). No count can see this.
- **The ridge here is mean-form** (`spec.rs`: `a[i*kc+i] += ridge` on the
  normalised Gram), so after full rank the statistic is a *constant* set by
  variance and ridge; it is a rank test, not a graded sample-size signal.
  The classical, fading version exists only under `ridge_decay`, which is
  refused with `standardize` and grids.
- **Compute it on the raw Gram, not the ridged pivots.** Pivots are free
  (the Cholesky already computes them; `log_det` walks the diagonal) and
  match exact edf to 3 decimals under a real ridge, but the healthy/dead gap
  collapses as the ridge approaches the feature variance (1.000/0.500 at
  1e-8, **0.533/0.347 at 1, 0.186/0.157 at 5**). The raw Gram's numerical
  rank read 3 of 4 at every ridge. Cost: one k×k factorisation per solve —
  cheap, not free; at k in the thousands use pivots when ridge ≪ trace/k
  (the standardised default) and report null otherwise.
- The already-returned `log_det` is the **wrong** free quantity: unbounded,
  it reads 1.000 on the collinear case.
- Pivot calibration differs: a duplicate reads 0.5, a constant 0 (the
  earlier copy's ridge leaks through elimination, pivot 2λ vs λ); mildly
  order-dependent (4th decimal). Moot with a high threshold.
- **Exclude the intercept column**: in the centred Gram it has zero
  variance by construction and would never read as supported.

### 5.5 Rank under decay: lost slowly, regained in one row

k=4, halflife 20, feature `x3` held constant; ratio of weakest to strongest
direction ≈ `2^(−halflives)` almost exactly:

| halflives flat | ratio | rank at tol 1e-6 | at 1e-10 |
|---|---|---|---|
| 10 | 1.1e-3 | 4 | 4 |
| 20 | 1.3e-6 | 4 | 4 |
| 30 | 9.1e-10 | 3 | 4 |
| 40 | 1.1e-12 | 3 | 3 |

One row of `x3` varying again restored rank at both tolerances. A gap
**capped** by `max_dclock` changes nothing (uniform decay, ratios
unchanged). An **uncapped** gap of 100 halflives collapses rank on the
second new row and rebuilds one direction per row (the model has genuinely
forgotten; only reachable with `max_dclock = inf`).

### 5.6 `full_rank_halflives` at startup

The same threshold reads near-collinearity before any decay:

| corr(x0, x3) | support, in halflives | N=10 | N=20 | N=30 |
|---|---|---|---|---|
| 0.99 | 7.8 | ok | ok | ok |
| 0.999 | 11.0 | dead | ok | ok |
| 0.9999 | 14.2 | dead | ok | ok |
| 0.999999 | 21.0 | dead | dead | ok |
| exact duplicate | 53 | dead | dead | dead |

Default 20 passes 0.9999 and catches 0.999999.

### 5.7 Rank deficiency does not hurt predictions (why there is no rank gate)

Truth built from the same design the model sees; noise floor 0.100:

| design | prediction RMS | coefficients on the twins |
|---|---|---|
| clean | 0.104 | 1 and 3 |
| `x3 == x0` | 0.104 | **2 and 2** (any split summing to 4 fits) |
| `x3` constant, with intercept | 0.104 | intercept absorbs it |
| `x3 == x0` in training, `x3 = x0 + 1` in scoring | **1.022** | wrong by the split |

A collinear fit predicts correctly in-sample and is wrong by an arbitrary
amount the moment the collinearity breaks. That is a coefficient problem
and a drift hazard, not a readiness question — so it is a flag and a
warning, not a gate.

### 5.8 The noise gate: what `min_periods` was really doing

k=10, halflife 2 (ceiling 3.41): rank is full at row 11 with `n_eff` 3.34,
so one-observation-per-unknown was never the identifiability condition
under decay — it is a variance heuristic. And the variance it guards is
predictable: for a least-squares fit the expected prediction error over the
irreducible noise is about **`√(1 + k/n_eff)`**.

| k | `n_eff` | predicted inflation | observed (RMS / noise floor) |
|---|---|---|---|
| 10 | 3.4 | 1.98× | **1.8×** at every noise level (0.18/0.10, 1.84/1.00, 5.52/3.00) |
| 10 | 72.6 (halflife 50) | 1.07× | 1.0× |

The ridge shrinks the observed value a little below the formula, in the
safe direction. Two points only; the sweep over `k`, halflife, ridge and
`standardize` is open (§7). Today's default `n_eff >= k + 1` corresponds to
an inflation of `√(1 + k/(k+1))` → `√2` as `k` grows (1.34 at k=4, 1.38 at
k=10), which is why `√2` — estimation error equal to the noise — is the
default that reproduces it.

**Why the formula is the explanation and not the statistic** (2026-09-21).
It is the large-`n` average over a stationary Gaussian design, and the
gate operates where that average is least trustworthy: the exact
Gaussian-design result `σ²(1 + k/(n − k − 1))` **diverges at `n = k + 1`,
today's default gate point** — pure OLS at one observation per unknown is
noise in expectation; only the ridge kept it at 1.8×. Under EW decay the
`n` that governs variance is Kish's `(Σw)²/Σw² = (1+λ)/(1−λ) ≈ 2·n_eff`,
not `n_eff`: at k=10, `n_eff`=3.4 the Kish form predicts 1.65×, the
`n_eff` form 1.98×, observed 1.8× — neither clean, and no sweep would make
one of them so. The exact `h(x)` of §2.1 has none of these approximations;
per model:

| model | statistic | status (0.9.0) |
|---|---|---|
| `ewridge` (any ridge) | gate `edf/n_Kish`; per row `h(x) = x'Σ̂⁻¹x / n_Kish` | **built**: the gate within 2% of the observed error down to `n_Kish ≈ 2.3 k_eff`; the per-row form conservative (§2.1) |
| `rls` | the same, from its own `R` factor | not built: keeps `min_periods` |
| `marginal`, `ew_cov` and the other scalar EW estimators | `h = 1/n_Kish` | not built: keep `min_periods` |
| `kalman` | `x'Px / R` — its own posterior | not built: keeps `min_periods` |
| `lasso`, `ftrl` with L1 | `h` on the active-set Gram | not built (post-selection; df = active count): keep `min_periods` |
| `sgd`, `pa`, `ftrl` without L1 | none | null; `min_settled_frac` and `min_periods` gate them |

### 5.9 Memory of the output fields (10M rows, process growth, fresh process each)

| type | bytes/row |
|---|---|
| Float64 | 8.03 |
| **String, every value null** | **16.17** |
| String, one value per 100 rows | 16.17 |
| Enum, 3 categories | 1.41 |
| UInt8 | 1.16 |
| Boolean | 0.17 |

Polars stores strings as fixed 16-byte views. Nothing accumulates (output
is chunked and released); the cost is per emitted row. `estimated_size()`
reported 0.00 for String and must not be trusted for it.

---

## 6. Tried and dropped

| idea | why it fell |
|---|---|
| **`min_session_clock`** (0.8.0) and its **halflife default** (0.8.1, tagged, never published) | measured the span between two backwards jumps against a "session length" no parameter carries; `max_dclock` caught 0 of 2,965 engine-reordering jumps, the halflife refused every intraday stream whose sessions were shorter than the model's memory. Replaced by `min_backwards_jump` (branch `min-backwards-jump`). |
| **typical-step jitter rule** (0.8.0) | fed only by deltas within the cap, so it could refuse nothing `max_dclock` does not; cost two persisted fields and a warmup. |
| **`obs_per_unknown`** | first a second number beside `min_periods`; reopened 2026-09-21 as a replacement; **dropped the same day** once the exact `h(x)` was on the table (§2.1): where theory gives `h` a count is dominated, and where it does not (`sgd`, `pa`) a count is a guess dressed as a gate. |
| **`√(1 + k/n_eff)` as the statistic** | the large-`n` stationary-Gaussian average, least trustworthy at `n ≈ k` where the gate operates; diverges exactly at today's default point (§5.8). Kept as the explanation, replaced by `h(x)`. |
| **`min_n_eff`** (rename of `min_periods`, kept as the floor) | the floor's unit is the reason it needs maintaining: absolute weight, blind to `k`. Replaced by `max_error_inflation` (§2.1). |
| **`G₂`**, the squared-decay Gram, for an exact `h(x)` | a `k×k` of state and an `O(k²)` per row that buy only the directional difference between recent and old design rows; under a design covariance constant over the window it is `(s₂/s₁)·G` and carries nothing. One `f64` (`s₂`) keeps everything a readiness gate needs (§2.1). |
| **`full_rank`** (boolean) with **`full_rank_halflives`** (its tolerance) | a threshold on an eigenvalue ratio, calibrated by sweep (§5.6), and blind under a real ridge whatever the cut (§5.4). Replaced by the exact, threshold-free `support_coef = diag(G_raw·G⁻¹)` (§2.2); the prediction-side hazard is `h(x)`. |
| **`require_full_rank`** as a gate | predictions are correct under rank deficiency (§5.7); a structural deficiency never recovers, so the "gate" is a permanent refusal; a flat feature recovers in one row. A diagnostic and a warning give both kinds of user what they need without taking predictions from everyone else. |
| **`log_det` as the free rank statistic** | unbounded; reads fully determined on a collinear design. |
| **ridged pivots as the rank statistic** | blind under a heavy ridge (§5.4). Fine as a cheap path when ridge ≪ variance. |
| **a String `withheld_reason`** | 16 B/row even when null (§5.9). |
| **validating reachability at construction with a clock** | the ceiling depends on the row rate (§5.1); runtime only. |
| **`warmup_halflives`** as the spelling | same knob as the fraction, one transform away; the fraction states the intent and cannot be set unreachably. |

---

## 7. Open — to iterate on

1. **Coefficient stability — what theory gives (2026-09-21; proposal,
   not built).** All of it rides on `M = Σ̂⁻¹ / n_Kish`, the coefficient
   covariance up to `σ²` under the same assumption as `h` (§2.1), for the
   linear-fit family:
   - **Coefficient covariance** `Cov(β̂) = σ²M`, so `se_j = σ√M_jj` and
     `t_j = β̂_j/se_j`. `h(x) = x'Mx` is this covariance seen through one
     row. Emit `se` on the `coef` schedule (`k` floats per solve).
   - **σ² unbiased under stable coefficients.** The one-step error
     `e_t = y_t − x_t'β̂_{t−1}` has variance `σ²(1 + h(x_t))` when β is
     constant (recursive residuals, Brown–Durbin–Evans 1975), so
     `w_t = e_t/√(1 + h_t)` has variance exactly `σ²`; its EW mean of
     squares is the estimator. One scalar of state. The raw residual's EW
     variance is biased upward during warmup by exactly the `h` factor.
   - **Stability test.** Under constant β (and Gaussian noise for the exact
     null) the `w_t` are iid: that is the null of the CUSUM /
     CUSUM-of-squares tests, and rejecting it *is* "coefficients unstable".
     Today's detector (Page-Hinkley, `drift.rs`) is fed `|e_t|/σ̂_raw`
     with `σ̂_raw` the slot's EW residual std (`stream.rs` ~3217). That raw
     scaling already cancels most of the warmup inflation — numerator and
     denominator both carry `√(1 + h)` — so it does *not* simply fire on
     warmup; the std merely lags, averaging past, larger `h`. Minimal step:
     feed it `|w_t|/σ̂` with `σ̂` from `w_t²`, so `drift_delta` means the
     same thing during warmup as at steady state. Principled step: the BDE
     CUSUM with its `√t` bound.
   - **"Converged" defined.** Under stationarity β̂ never stops moving; it
     fluctuates at the floor `σ²M` forever. So "stopped moving" is not a
     criterion. Converged = the floor is small enough (`se_j`) **and** the
     coefficients move no more than the floor (the test above). A
     relative-change-below-ε rule has no null distribution and would be
     tuned by sweep — ruled out by the standard in §2.1.
   Caveats as for `h`: variance only, ridge bias excluded; lasso
   post-selection; nothing for `sgd`/`pa`/`ftrl`. **Open:** whether any of
   this gates (a `min_t` on coefficients is a coefficient-reader's setting,
   not a prediction gate) or only reports.
2. **Default of `min_settled_frac`.** Settled at `0` by §4.1.1 on a
   theory argument (the mean-form fit is unbiased from row one under
   stationarity; the gate guards a representativeness bias only the user
   can size). Reopens only if a statistic for that bias is found.
3. **Is `withheld_reason` needed** given `settled_frac` and `n_eff` are
   emitted? The user wants it ("more obvious"); Enum keeps it cheap.
4. **Verifying `h(x)` — identities, not calibration (done, 2026-09-21;
   `crates/online-core/tests/readiness.rs`).** The tests hold the
   implementation to what the theory predicts, and each fails loudly if
   `s₂`, the weights or the ridge are wired wrongly. What they found:
   (i) on a stationary design at `n_Kish/k_eff ≈ 12` the mean per-row `h`
   is 10% above `edf/n_Kish`, and `edf/n_Kish` itself is within 5% of
   `(k+1)/n_Kish` with `n_Kish = (1+λ)/(1−λ)`; (ii) the observed RMS/noise
   at `n_Kish ≈ 2.3 k_eff` is 1.214 against the gate's 1.197 — within 2% —
   while the per-row mean says 1.315: the per-row form is conservative, by
   half there (§2.1 says why); (iii) scaling every weight by a constant
   leaves `h` unchanged to 1e-9, and splitting every row into two halves
   at the same clock halves it exactly; (iv) the §5.7 collinearity-break
   row reads `h` more than 100× the in-sample rows'; (v) a duplicated pair
   reads `support_coef` 0.5 each, a clean design 1, a ridge of 5 below 0.3,
   and the standardized solve with `ridge = 1` reads 1/(1+1) on an
   orthogonal design. `lasso`'s active-set `h` is not built, so not
   measured. `min_periods` was kept for every model but `ewridge` rather
   than aliased away (§8).
5. **Enum emission from Rust** — answered by building it: new plumbing.
   A `u32`-key dictionary array carrying polars' own `_PL_ENUM_VALUES2`
   field metadata (`column::{code_array, enum_metadata}`), which polars
   reads back as an `Enum` over exactly the three names. `ew_class`'s
   `class` was no precedent: it is a plain string column.
6. **`support_coef` warning** — settled 2026-09-21 (§2.2): warn at
   `< 0.5`, and only on a row whose prediction the gates let through.
7. **Standardisation.** `support_coef` and `h` live in the space the ridge
   acts in (standardised under `standardize`); the
   standardiser's own noisy first rows can make the flag flicker before it
   settles.
8. **`n_eff_settled`** — from what settledness is the estimate
   (`n_eff / settled_frac`) reported? 0.9?
9. **CLI** — one closing line counting groups not settled / low support?
10. **Names** (§9).

---

## 8. Implementation notes

- **State:** one new persisted `f64` per stream instance — accumulated
  *capped* decay time (`Σ d_clock` the models actually decayed by, so a
  weekend capped to `max_dclock` counts as `max_dclock` of warming), reset
  on `reset_state` and on a drift reset. Schema 13, with the clock change.
  No libm in the state: `2^(−T/h)` is computed for the field, never
  persisted. **Convention:** a group's first row decays by nothing
  (`d_clock = 0`), so on a row-count clock the decay time seen before row
  `i` is `i − 1`, and "one halflife" reads on the row *after* the one
  `n_eff`'s fraction of its ceiling would name: the field is the clock the
  decay has covered, not a count of rows. Under `label_delay` the clock the
  held rows have covered counts too, though the models have not yet decayed
  by it: a scored row then reads what the doubled stream (E47's oracle)
  reads for it, where every held row has already decayed the model as a
  weight-0 row -- and the mean-form fit is the same either way.
- **`settled_frac`** per decay instance (a halflife grid gives one per
  instance); null when decay is off (`halflife = inf` / `lam = 1`).
- **`support_coef`** `= diag(G_raw·G⁻¹)` on the solve schedule, from the
  same `G⁻¹` as `M`; intercept excluded, uncentred for no-intercept; null
  when the jittered solve had to add a diagonal shift (the nominal `λ` is
  then not the one in `G`).
- **Chunk invariance (hard rule 3):** `settled_frac` (accumulated time),
  `error_inflation` (the factor and `s₂` from the last solve) and `withheld_reason`
  (derived) must be
  identical under any chunking. Only `coef` may follow the chunking (it is
  emitted on each *group's* last row in each chunk).
- **Model applicability:** `support_coef` and `h` exist for the
  Gram-factorising family
  (`ewridge`, `lasso`, `ew_cov`, the robust models); `rls`/`kalman` track an
  inverse (a different, arguably better signal — out of scope); the
  gradient models (`sgd`, `pa`, `ftrl`) have no second moment. Field null,
  no warning, where it does not exist. `settled_frac` exists for every
  decayed model.
- **`error_inflation`:** `s₂ = Σλ²ⁱwᵢ²` (one `f64`, decay `λ²`, weight
  `w²`, the same capped `d_clock` as the Gram) persisted beside `n_eff`
  and reset with it; `h(x) = x'Σ̂⁻¹x / n_Kish` per row from the Cholesky
  factor of the last solve (chunk invariance: the factor is state already).
  `Σ̂` is the mean-form normalised, ridged Gram, so no extra normalisation
  is needed. Scalar models: `h = 1/n_Kish`. `Σ̂` singular before `k` rows →
  `h = inf` (withheld). Under `window` the weights are a hard cutoff, so
  `s₂` is the same accumulator with the same cutoff — to confirm when
  built.
- **Predict:** `predict()` on a not-settled or too-noisy bank returns null
  with the reason, consistently with `min_periods` today.
- **Version:** 0.9.0 (schema bump), with the clock change.

**Cost, honestly (2026-09-21).** Per spec:

| item | output | per row | state | per solve |
|---|---|---|---|---|
| `settled_frac` | 8 B | one accumulate, one `exp2` | 1 `f64` | — |
| `withheld_reason` | 1.4 B | derived | — | — |
| the noise **gate** (`edf/n_Kish`) | — | `O(1)` | +1 `f64` (`s₂`) | `edf` from the pivots, free |
| `error_inflation` per row (`‖L⁻¹x‖²/n_Kish`), **opt-in** | 8 B | `k²/2`: one triangular solve against the kept factor | the ridged system per (Gram, combo), `k_c²`, persisted so the factor survives the state file (rebuilt on load); the factor itself beside it in memory | — |
| `support_coef` | `k` floats on `coef` rows only | — | — | `O(k²)` given `G⁻¹`, or `O(k³)` if the inverse is not otherwise formed |

So out of the box the Gram family's per-row cost does not change: the
gate is `O(1)` per row. Only a user who opts into the per-row
`error_inflation` field pays a second `O(k²)` pass (invisible at `k ≤ 10`,
up to ~2× on the accumulate path at `k ≥ 100`). State grows by one `f64`
per (spec, group) — the `k×k` `G₂` was dropped as buying nothing a
readiness gate needs (§2.1). Output is +9.4 B/row per spec uncompressed
by default (`settled_frac`, `withheld_reason`), +8 with the opt-in field; parquet compresses
`settled_frac` and `withheld_reason` to little, `error_inflation` like any
float. Precedent (`emit_drift`, the residual quantiles) is that
diagnostic *fields* are opt-in while the behaviour they diagnose is not:
the gates can always run and the two floats be emitted on request — a
default to decide (§7). Exact shortcut for large `k`: `h(x) ≤
‖x‖²·λ_max(M)` is `O(k)`, and when the bound clears the gate the exact
value is needed only if the field is being emitted.

---

## 9. Names (current; open)

Three roots — `settled`, `error_inflation`, `support_coef` — and two
rules: a field is its setting minus the prefix (`min_settled_frac` →
`settled_frac`, `max_error_inflation` → `error_inflation`); an enum value
is `below_`/`above_` + the bound it broke. `support_coef` has no setting;
it names what it is, a support share, and what it is a share of, and does
**not** start with `coef_` -- so `^coef_` still selects the coefficients
and nothing else (the user's call, 2026-09-21; it was `coef_support` for a
day). Unnested it sits beside them as `support_coef_<t>_<term>`. Counts follow
the codebase's `n_` convention (`n_eff`, `n_coef`). Avoid "window" (taken by
the row window) and "warm" (metaphor, where every other key is a quantity).
