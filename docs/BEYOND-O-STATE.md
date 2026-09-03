# What relaxing O(state) would unlock

Status as of 2026-08-31: **survey, nothing proposed for implementation.** A scoping
question: if this library were willing to hold memory that grows with a *window* or a
*sketch* — `O(W·k)`, `O(1/ε)`, `O(log n)` — rather than strictly `O(k²)` state, how
many genuinely new things could it offer, excluding anything already well served by a
Rust or C++ implementation?

**Answer: about six, of which three are strong.** The list is short on purpose. Most
of what O(window) unlocks is either already in Rust, or is the *shape* of library this
project deliberately is not (`docs/ENHANCEMENTS.md` §4).

---

## Method

Two filters, applied in order.

**Filter 1 — is it already in Rust or C++?** Checked against crates.io download counts
(2026-08-31) rather than memory, because "I think nobody has done this" is how
duplicated effort starts. Download counts stand in for maturity:

| area | strongest Rust crate | downloads | verdict |
|---|---|---|---|
| t-digest / streaming quantiles | `tdigests`, `tdigest` | 1.2M, 980k | **well served — do not build** |
| change-point (BOCPD) | `changepoint`, `augurs-changepoint` | 155k, 69k | **well served** |
| exponential histograms | `exponential-decay-histogram` | 783k | **well served** |
| reservoir sampling | `reservoir-sampling` | 21k | **well served** |
| isotonic regression (PAV) | `pav_regression` | 80k | **served** (batch, but the algorithm is there) |
| RLS | `recless` | 9.6k | served, and we have our own |
| conformal prediction | `conformal-prediction` | **461** | **effectively a vacuum** |
| frequent-directions sketch | — | — | **nothing** |
| ADWIN as a usable component | only inside grab-bag crates | — | **nothing dedicated** |
| rolling/windowed regression | — | — | **nothing streaming** |
| online PCA | batch sklearn ports only | — | **nothing streaming** |
| Hoeffding trees | — | — | nothing, but see below |

Incumbents outside Rust worth naming: **Vowpal Wabbit** (C++) owns online gradient
descent, feature hashing and contextual bandits; **MOA** (Java) owns streaming trees
and ADWIN; **river** (Python) is the reference this project already benchmarks against.
Anything squarely inside VW's or MOA's territory is a bad target regardless of language.

**Filter 2 — does it fit what this library is?** A candidate must keep the two
guarantees that everything here rests on — **out-of-sample by construction** and
**chunk invariance** — plus exact save/resume and clock/session semantics. That is a
sharper filter than it sounds: it rules out anything randomised per-row, anything whose
state cannot be serialized, and anything whose answer depends on chunk boundaries.

---

## The three strong candidates

### B1 — Adaptive conformal prediction (ACI) `O(1/ε)`

Prediction *intervals* with a coverage guarantee that holds under distribution shift,
rather than the point estimate plus `sigma` we emit today. The online form (Gibbs &
Candès) is a one-line update on a running quantile of conformity scores:
`α_{t+1} = α_t + γ(target_coverage − 1{y_t ∈ interval_t})`.

- **Memory:** a quantile sketch of past `|resid|`, which we already have (`P2Quantile`,
  E23). Effectively free — this is `O(1/ε)`, not `O(window)`.
- **Why it is not duplicated:** `conformal-prediction` has 461 downloads; the serious
  implementations are Python (MAPIE, `crepes`). In a streaming Rust library: nothing.
- **Why it fits here better than anywhere:** conformal needs conformity scores from a
  model that never saw the row — which is precisely the property this library
  guarantees by construction and tests for. Most implementations have to arrange
  data-splitting to get what we get for free.
- **Output:** `pred_lo_<slot>` / `pred_hi_<slot>` and a running realised coverage.
- **Effort:** small. It is a quantile tracker plus a scalar recursion.

**This is the strongest item on the list** — smallest, most novel, most demanded, and
it composes with every model here rather than being a new model.

### B2 — Rolling-window regression `O(W·k)`

Everything here decays exponentially. A hard rectangular window — "the last N rows" or
"the last T clock units" — is the other convention people expect, and it is what
`statsmodels.RollingOLS` and every pandas user reaches for.

- **Memory:** `O(W·k)` to hold the rows leaving the window, plus the same `k²`
  accumulator.
- **Why it is not duplicated:** no Rust crate does streaming rolling regression.
- **The real contribution is numerical, not algorithmic.** The naive
  add-on-entry/subtract-on-exit update of a covariance matrix suffers catastrophic
  cancellation and drifts to indefinite over long runs — the classic failure. A
  correct implementation needs either periodic recomputation on a schedule or a
  compensated/blocked scheme. This library already has the machinery to get that right
  (centred Welford co-moments, the jittered-Cholesky fallback, `solve_failures`
  reporting) and a test culture that would catch the drift. Most rolling
  implementations in the wild are quietly wrong at long horizons.
- **Effort:** medium. The window bookkeeping is easy; the numerics deserve care and an
  oracle test against a from-scratch fit at every step.

### B3 — Frequent-directions sketch / streaming PCA `O(ℓ·k)`

A deterministic low-rank sketch of `X` with a proven error bound, giving streaming
principal components — factor structure, a conditioning diagnostic, or dimension
reduction ahead of a fit.

- **Memory:** `O(ℓ·k)` for sketch size `ℓ ≪ n`.
- **Why it is not duplicated:** nothing in Rust. Python has `river.decomposition` and
  research code; Rust has batch sklearn ports only.
- **Why it fits:** frequent directions is *deterministic* — no RNG — so it preserves
  chunk invariance exactly, which randomised sketches (JL projections, sampling) do
  not. That distinction is what makes it the right sketch for this library and the
  others wrong for it.
- **Pairs with E30** (exporting the Gram matrix): the same users want both.
- **Effort:** medium. The algorithm is ~60 lines plus an SVD; `faer` supplies the SVD.

---

## The three weaker ones

### B4 — Fixed-lag Kalman smoother `O(lag·k)`

We have the filter. A fixed-lag smoother revises the last `L` estimates using
subsequent observations — strictly better coefficient estimates at the cost of a lag.
Nothing comparable in streaming Rust. **But** it breaks out-of-sample-ness by design:
a smoothed estimate at row *t* uses rows after *t*. That is legitimate for *research*
(what were the coefficients really doing?) and forbidden for the prediction path, so it
would need to be a clearly separated output that cannot be mistaken for `pred`. Worth
doing only with that separation designed first.

### B5 — Multi-lag residual diagnostics `O(max_lag)`

`emit_autocorr` does lag 1. A Ljung–Box statistic over lags 1..L, streaming, is `O(L)`
and is the standard "is my model misspecified?" test. Small, useful, unglamorous, and
absent from Rust in streaming form.

### B6 — Delayed-label handling `O(delay)`

Production streams often learn a label hours after the features. A buffer of pending
rows, joined when the label lands, is `O(delay·k)`. river has `utils.Rolling` patterns
for this; Rust has nothing. **But** this is arguably upstream Polars' job (an asof join
against a later label table), which is the answer §4 already gives for windowing — so
it may be a documentation item rather than a feature.

---

## Explicitly still out, even with O(window) allowed

Relaxing the memory bound does *not* change the answer for these, and the reasons are
worth restating so nobody re-opens them on the memory argument alone:

- **Hoeffding trees / streaming forests.** Nothing in Rust, and that is a real gap —
  but it is MOA's territory, it is a different library's identity (`ENHANCEMENTS` §4),
  and the state is adaptive rather than bounded. Building it here would double the
  project's surface area and halve its coherence. *Reassessed 2026-09-03 in
  [`BOOSTED-TREES.md`](BOOSTED-TREES.md): a gradient-boosted design with bounded
  state, no per-row randomness and `ewridge`'s clock decay is prototyped and
  measured there; the surface-area cost stands and is the decision.*
- **kNN regression.** `O(window)` is now permitted, but it is river-and-MOA territory,
  and the value of a Rust version is speed on a workload this library is not shaped for.
- **Contextual bandits.** Vowpal Wabbit owns this and is very good at it.
- **Randomised sketches** (Johnson–Lindenstrauss projections, sampling-based
  quantiles): they break chunk invariance, which is a stated guarantee rather than a
  preference. Frequent directions is the deterministic alternative — hence B3.
- **t-digest / DDSketch / HyperLogLog / reservoir sampling.** All well served in Rust.
  A *clock-decayed* t-digest would be mildly novel, but P² already covers the need here.

---

## Suggested framing if any of this is pursued

The honest count is **three strong, three weak, and one large thing deliberately left
to others**. If the goal is "what could this library give the community that nothing
else does", the ranked answer is:

1. **B1 adaptive conformal prediction** — smallest effort, largest gap, and it exploits
   the one property this library has that most competitors have to work for.
2. **B3 frequent directions** — genuinely absent from Rust, deterministic so it keeps
   chunk invariance, and pairs with E30.
3. **B2 rolling-window regression** — the most *expected* feature, where the
   contribution is doing the numerics correctly rather than doing it at all.

A memory bound would need to become a documented, tested property rather than a habit:
something like a `max_memory_bytes` per spec, asserted in the soak test the way
`n_eff` boundedness already is. Relaxing `O(state)` without replacing it with a
*stated* bound is how a streaming library stops being one.
