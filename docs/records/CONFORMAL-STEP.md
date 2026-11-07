# The conformal step under a label delay

**Status as of 2026-11-07: PLAN task 342, measured; the default is
unchanged.** The question (TREE-FEATURES.md §14, question 2): the
conformal tracker steps its radius by `conformal_rate · sigma` on each
scored row, 0.05 by default, with no account of how many predictions are
in flight while their labels are late. Is there a rule for the step that
holds across delays, row rates, noises and coverage levels, or none? The
answer here: **no candidate held everywhere**, so the default stays. A
rule derived from the delay, `rate / (1 + rate · g(α) · n / c)` with `n`
the mean rows waiting, removes the default's failure under long delays at
no cost when nothing waits; it is now the reader's rule of thumb in
DIAGNOSTICS.md, and an opt-in that computes `n` itself is designed in §7
for the user to decide. The prototypes are `scripts/conformal_step/`
(§9 runs them).

---

## 0. The short version

- **The derivation (§2).** The tracker is an integral controller whose
  feedback is `n` releases late. Linearized, its error follows
  `e_{t+1} = e_t − k·e_{t−n}` with `k = rate · g(α)`, `g(α)` the density
  of `|resid|/sigma` at the asked quantile (0.206 on Gaussian residuals at
  90%). It is stable for `k·n` below about π/2 and free of oscillation
  below about `1/e`; above π/2 it is a limit cycle. The default at 90%
  with 300 rows waiting has `k·n = 3.1`. A simulation with an exact delay
  collapses onto `k·n` at n = 10, 100 and 1,000 (§2.4).
- **The rule (§2.5).** Divide the rate by `1 + rate·g(α)·n̄/c`: unchanged
  when nothing waits, and `k·n` held below `c` when much does. `n̄` must
  be a slow mean of the rows waiting: a step that follows the instant
  count, or the row's own lag, weights the misses unevenly and moves the
  coverage itself (flights, 8 h: 92.1% and 85.8% for a 90% target).
- **The measurement (§4, §5).** 100 streams at 90% (flights' tree
  residual at four delays; tests/data.py's stream at four noises, two row
  rates, four delays, three seeds), and 36 at 50% and 99%. Every rule is
  replayed over the bank's own served residuals, its fixed-rate form
  equal to the bank's `lo` to the bit on every stream. At 90%, today's
  default holds 73 of 100 (coverage within 0.005 and half-width within
  10% of the best fixed rate in hindsight); the delay rule at `c = 1/2`
  holds 88, at `c = 1` 91; a fixed 0.02 holds 93 and a step against the
  current radius 93. At 50% today's default holds 23 of 36 and the delay
  rule 36. At 99% today's default holds 30 and the delay rule 27.
- **What holds and what does not (§6).** Under delays of hundreds of
  rows the default swings: flights' 90% half-width 189 minutes against
  45 at 8 hours, and at 50% a coverage of 70%. The delay rule fixes
  that. It fails where a smaller step is slower: recovering from a cold
  warm start, and following a noise that moves within the delay. AgACI
  and SAOCP, which choose their step themselves, do not survive the
  delay: AgACI's fastest experts swing and pull it to 414 minutes,
  SAOCP's experts learn from stale gradients and cover 85%. The step
  against the current radius does not swing, but it gives up the
  coverage guarantee and covers flights at 86.7%.
- **For the user (§7).** Keep the default (no rule held everywhere, as
  the brief required for a change); the rule of thumb is documented. An
  opt-in `conformal_rate="auto"` needs one f64 in the tracker's state,
  in schema 67, which is unreleased.

---

## 1. The tracker

For a slot with target coverage `1 − α`, score `s = |resid|` and radius
`q` (`crates/online-core/src/conformal.rs`):

```text
err   = 1{ s > q_shown }
q    ← max(0, q + η · (w / w̄) · (err − α)),    η = rate · sigma
```

Without a delay `q_shown` is the current `q`. Under `embargo` and
`label_clock` each held row keeps the radius it was shown, and the
release scores the label against it: `update_against`. The guarantee
telescopes: `Σ η_t (err_t − α)` is `q`'s total movement, bounded, so the
`η`-weighted miss rate tends to `α` on any sequence, the clamp at zero
only adding coverage. A delay adds at most the steps in flight to the
bound; it does not void it. What a delay does is make the step too big.

---

## 2. Derivation: an integral controller with a feedback lag

### 2.1 The loop

Number the releases `t = 1, 2, …`. The row released at `t` was shown
the radius of `n` releases before, so `err_t = 1{s_t > q_{t−n}}`. With
`F` the distribution of `s` and `q*` its `1 − α` quantile,
`E[err_t | q_{t−n}] = 1 − F(q_{t−n})`. Write `e_t = q_t − q*` and
linearize: `1 − F(q* + e) − α ≈ −f(q*) e`. In expectation

```text
e_{t+1} = e_t − k · e_{t−n},      k = η · f(q*) = rate · sigma · f(q*) = rate · g
```

`g = sigma · f(q*)` is the loop gain per unit of rate, unit-free. On
Gaussian residuals `g(α) = 2 φ(Φ⁻¹(1 − α/2))`: 0.636 at 50%, 0.206 at
90%, 0.117 at 95%, 0.029 at 99%. Measured on the streams here (the
density of `|resid|/sigma` at its 90% quantile, over the scored rows):
flights 0.165–0.168, Gaussian 0.208, Student's t(3) 0.156, the
heteroskedastic stream 0.194. The tracker is the integral term of a
controller (Angelopoulos, Candès and Tibshirani 2023 call it the P step
of a PID, since it acts on the quantile; on the coverage error it
integrates), and `n` is pure dead time.

### 2.2 Stability

The characteristic equation is `λ^{n+1} − λ^n + k = 0`. For this
delay-difference equation (Levin and May 1976; Kuruklis 1994) every root
is inside the unit circle if and only if

```text
0 < k < 2 cos( n π / (2n + 1) ) = 2 sin( π / (4n + 2) ) ≈ π / (2n + 1)
```

so **`k·n < π/2` is the stability bound**, whatever `n` is. Every root is
real and positive, so the error decays without overshoot, if and only if
`k ≤ nⁿ / (n + 1)^{n+1} ≈ 1 / (e (n + 1))`. At that edge the dominant
root is `n / (n + 1)`: the error decays with a time constant of `n + 1`
releases. No step corrects the radius faster than its news arrives.

### 2.3 Noise

Without a delay the radius wanders around `q*` with variance about
`η² α(1 − α) / (2k)`, so its spread grows as `sqrt(rate)`: a smaller step
is narrower and slower. With a delay the spread grows as `k·n` nears π/2
and the loop's damping vanishes; past it the linearization fails and the
radius settles into a limit cycle bounded by the clamp at zero and by
the misses running out.

The flights stream adds a nonlinear form of the same effect: its
residuals move with the time of day and the weather, so the rows in
flight miss together. When the distribution moves within the delay,
every one of the `n` rows in flight reports a miss, and the radius
rises by about `rate · sigma · (1 − α) · n` before the first covered row
is heard from. Holding that overshoot under a fraction of `sigma` asks
the same thing: `rate · n` under a constant.

### 2.4 Checked by simulation

`scripts/conformal_step/linear.py`: 400,000 standard normal residuals,
`sigma = 1` known, each row's feedback exactly `n` releases late, the
first 100,000 rows dropped; the radius's mean and spread as a share of
`q* = 1.645`.

| n | k·n | rate | coverage | mean half-width / q* | spread of the radius / q* |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.10 | 0.04848 | 0.9000 | 1.004 | 0.066 |
| 10 | 0.37 | 0.17835 | 0.9000 | 1.029 | 0.148 |
| 10 | 0.50 | 0.24240 | 0.9000 | 1.047 | 0.191 |
| 10 | 1.00 | 0.48480 | 0.9000 | 1.179 | 0.412 |
| 10 | 1.50 | 0.72720 | 0.9000 | 1.478 | 0.777 |
| 10 | 2.00 | 0.96960 | 0.9004 | 1.954 | 1.246 |
| 10 | 3.00 | 1.45440 | 0.9025 | 3.179 | 2.187 |
| 100 | 0.10 | 0.00485 | 0.9000 | 1.000 | 0.020 |
| 100 | 0.37 | 0.01783 | 0.9000 | 1.002 | 0.046 |
| 100 | 0.50 | 0.02424 | 0.9000 | 1.003 | 0.057 |
| 100 | 1.00 | 0.04848 | 0.9000 | 1.017 | 0.117 |
| 100 | 1.50 | 0.07272 | 0.9000 | 1.069 | 0.242 |
| 100 | 2.00 | 0.09696 | 0.9000 | 1.244 | 0.501 |
| 100 | 3.00 | 0.14544 | 0.9000 | 2.170 | 1.352 |
| 1000 | 0.10 | 0.00048 | 0.9000 | 0.998 | 0.005 |
| 1000 | 0.37 | 0.00178 | 0.9000 | 0.999 | 0.013 |
| 1000 | 0.50 | 0.00242 | 0.9000 | 0.999 | 0.017 |
| 1000 | 1.00 | 0.00485 | 0.9000 | 1.000 | 0.035 |
| 1000 | 1.50 | 0.00727 | 0.9000 | 1.012 | 0.094 |
| 1000 | 2.00 | 0.00970 | 0.9005 | 1.171 | 0.381 |
| 1000 | 3.00 | 0.01454 | 0.9003 | 1.995 | 1.141 |

The coverage holds at every setting, as the guarantee says; the width is
what the delay costs. At `k·n = 2`, past π/2, the mean half-width is 17%
to 95% over `q*` at every `n`; at `k·n ≤ 1` it is within 2% (n = 100,
1,000; at n = 10 the step itself is large and its noise shows). The
bank's own trace agrees (tests/data.py's stream, 300 rows waiting, the
default rate, `k·n = 3.1`): the radius at rows 1k, 3k, 10k, 20k, 40k,
60k, 90k was 20.8, 10.5, 6.6, 7.5, 2.9, 7.3, 4.2 against `q* = 1.65`; at
0.02 (`k·n = 1.2`) 21.2, 17.1, 3.1, 1.6, 1.5, 1.8, 1.7.

### 2.5 The rule, and why `n` must be a slow mean

To keep `k·n` under a constant `c` and leave the undelayed tracker as it
is:

```text
rate_eff = rate / (1 + rate · g(α) · n̄ / c)
```

`rate_eff = rate` when `n̄ = 0`, bit for bit, and `rate_eff · g · n̄ → c`
as `n̄` grows: a step proportional to 1 / (predictions in flight). `g(α)`
is the Gaussian gain, computed from the level the user asked for; the
tracker cannot know the true density, and §2.1's measurements put real
streams up to a quarter below it, which only adds margin.

**The guarantee constrains `n̄`.** The telescoping holds for any `η_t`
fixed before the row's error is known, but what it holds to `α` is the
`η`-weighted miss rate. If `η_t` moves with something that predicts the
miss, the share of rows missed moves away from `α`:

- the released row's own lag (releases since it was served): a flight
  that lands late is a flight that is delayed, the miss, and its step is
  the smallest. Flights at 8 h covered 85.8% (§5.3, "own lag");
- the count waiting at the instant of release: flights pile up in the
  afternoon, when the delays are worst, so the misses take the small
  steps. Flights at 8 h covered 92.1% ("held now"). This is why the
  earlier trial of `rate / sqrt(rows in flight)` over-covered at 91.3%
  (TREE-FEATURES.md §2.3): it read the instant count;
- an exponentially weighted mean of the count waiting, at the model's own
  decay over the released rows (the same weights as the mean weight
  `w̄`): it moves on the scale of the half-life, not of a row. 89.86%.

`c` is the design's one constant. Between the monotone edge `1/e` and
the stability edge π/2, the trade is §2.3's: a smaller `c` damps the
burst of correlated misses (flights), a larger one recovers faster from a
radius that is wrong for another reason (the warm start, a jump in the
noise). §5 measures `c = 1/e, 1/2, 1`.

---

## 3. The candidates

| rule | what it does | source and oracle |
|---|---|---|
| fixed `r` | today's tracker at `r` ∈ {0.05, 0.02, 0.01, 0.005, 0.002, 0.001}: the default, and the best rate in hindsight per stream | the bank: the fold equals its `lo` to the bit (§4) |
| lag, mean held, `c` | §2.5 with the EW mean of the rows waiting (the released row aside), `c` ∈ {1/e, 1/2, 1} | derived here |
| lag, held now | §2.5 with the instant count | derived here, to show §2.5's point |
| lag, own lag | §2.5 with the released row's own lag in releases | the same |
| current radius | the step's error read against the radius held at the release, not the one shown; the coverage is still scored on the shown radius. Quantile SGD on stale samples: no dead time in the loop, so no swing, and no guarantee | the obvious alternative; measured to show what it gives up |
| agreeing radii | a step only where the shown and the current radius agree on the miss | an attempt to keep the shown error and remove the swing |
| AgACI | Zaffran et al. (2022): K trackers at fixed rates {0.001, …, 0.1}, each scored against the radius it showed, combined by Bernstein online aggregation (Wintenberger 2017) of the pinball loss with the gradient trick and per-expert adaptive learning rates. The paper aggregates the two bounds; a symmetric radius has one | coded from the papers (the authors' aggregation is R's `opera`; no Python implementation exists). Checked: one expert, and three equal experts, equal the fixed tracker (to the bit; 3e-14) |
| SAOCP | Bhatnagar et al. (2023), Algorithm 1: a scale-free OGD expert (`s ← s − (D/√3)·∇/√Σ∇²`) started at each release with lifetime `g·2^{v(t)}`, `g = 8` (the paper's time-series setting), combined by coin betting with prior `∝ i⁻²(1 + ⌊log₂ i⌋)⁻¹`; `D/√3` the largest residual seen so far (the paper uses a calibration split's). Its experts read the error at their current radius (they may expire before a held row returns) | coded from the paper; the authors' code (Salesforce's `online_conformal`) is not on PyPI and was not fetched |

---

## 4. How it was measured

**The bank fixes the predictions; each rule is a fold.** The model never
reads its interval, so one run of the bank under `label_clock` fixes
every served prediction, and every rule replays the same residuals in
the bank's release order (`scripts/conformal_step/fold.py`, numba): a
held row leaves before the first row whose clock reaches its label time,
in order of (label time, arrival), decayed from the label time released
before it, reading `sigma` from the same fold the bank keeps (the oracle
of `tests/test_label_clock.py`). `check_replica.py` holds the fixed rule
at 0.05 to the bank's `lo` to the bit, and the rows with no interval to
the bank's nulls, on all 20 stream-and-delay pairs it runs (flights at
four delays; the Gaussian and level-shift streams at both row rates and
four delays).

**Streams.**

| | rows | the delay | scored |
|---|---:|---|---|
| flights (nycflights13 through `tests/data.py`'s download): `arr_delay` less a LightGBM model trained on January–August, early-stopped on September (TREE-FEATURES.md §2.3), an intercept `ewridge` of half-life 7 days, `gap_cap` 10 hours | 327,346 | none (`label_clock = t`); 1 hour; 8 hours; at landing (`dep_delay + air_time + 20` minutes after `t`) | October–December |
| `tests/data.py`'s `synthetic` (three features, coefficients that walk, steady exponential row times of mean 10), its noise replaced: Gaussian; t(3) scaled to unit spread; heteroskedastic (scale `e^{±0.7}` over a 20,000-row cycle, and three times as large on rows 60–70k); a level shift (3 noise sds on rows 50–75k). Row rate steady, or bursty (rows twentyfold denser at a 2,000-row cycle's peak, the same mean). An `ewridge` of half-life 500 rows' clock, `gap_cap` 50 rows' | 100,000, seeds 0, 1, 2 | none; short (50 units, about 5 rows); long (3,000, about 300 rows); variable (exponential, mean 3,000) | rows 10,000 on |

Rows waiting, on average: flights 0, 54, 338, 152; generated 0, 6 (15
bursty), 301 (501 bursty), 300 (432 bursty).

**What counts as holding.** On a stream a rule *holds* when its coverage
is within 0.005 of 90% (0.01 at 50%, 0.002 at 99%) and its mean
half-width within 10% of the narrowest fixed rate that also covers
within that tolerance: the rate a user who knew the stream would have
chosen. Where no fixed rate covers (the bursty level shift under the
long delay, at 90%), only the coverage is judged. Each table is computed
again on the second half of the scored rows alone, to separate the
tracker's start from its steady state. The levels 50% and 99% run on
flights and seed 0 (36 streams each).

**Load.** The bank ran in a dev-profile wheel; the folds are
single-threaded. The machine's one-minute load was 3 to 70 from other
workers during the runs; no number here is a timing.

---

## 5. Results

### 5.1 How often each rule holds

| rule | 90%: holds of 100 | coverage off | too wide | worst coverage error | widest / best | 90%, second half: holds | 50%: holds of 36 | 99%: holds of 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.05 (today) | 73 | 13 | 14 | 0.0182 | 4.36 | 73 | 23 | 30 |
| fixed 0.02 | 93 | 3 | 4 | 0.0102 | 1.50 | 86 | 25 | 15 |
| fixed 0.01 | 83 | 15 | 2 | 0.0221 | 2.15 | 80 | 28 | 15 |
| fixed 0.005 | 66 | 34 | 0 | 0.0545 | 4.90 | 72 | 35 | 13 |
| fixed 0.002 | 42 | 58 | 0 | 0.1000 | 11.77 | 57 | 35 | 5 |
| fixed 0.001 | 33 | 67 | 0 | 0.1000 | 15.02 | 36 | 26 | 1 |
| lag, mean held, c = 1/e | 84 | 15 | 1 | 0.0507 | 4.60 | 85 | 35 | 27 |
| lag, mean held, c = 1/2 | 88 | 10 | 2 | 0.0372 | 3.36 | 90 | 36 | 27 |
| lag, mean held, c = 1 | 91 | 6 | 3 | 0.0162 | 1.70 | 83 | 34 | 27 |
| lag, held now, c = 1/e | 85 | 14 | 1 | 0.0505 | 4.58 | 89 | 33 | 27 |
| lag, own lag, c = 1/e | 89 | 11 | 0 | 0.0506 | 4.58 | 88 | 32 | 27 |
| agreeing radii | 49 | 51 | 0 | 0.0991 | 1.07 | 46 | 34 | 24 |
| current radius | 93 | 7 | 0 | 0.0334 | 1.08 | 81 | 34 | 27 |
| AgACI | 48 | 35 | 17 | 0.0731 | 9.57 | 47 | 13 | 16 |
| SAOCP | 61 | 15 | 24 | 0.0462 | 1.42 | 53 | 9 | 3 |

### 5.2 Flights, 90%: coverage / half-width in minutes

| rule | none | 1 h | 8 h | landing |
|---|---:|---:|---:|---:|
| fixed 0.05 (today) | 0.8996 / 44.5 | 0.8995 / 47.4 | 0.9034 / 188.9 | 0.8996 / 63.6 |
| fixed 0.02 | 0.8994 / 44.4 | 0.8993 / 45.9 | 0.9008 / 64.9 | 0.8993 / 52.1 |
| fixed 0.01 | 0.8992 / 44.1 | 0.8992 / 44.9 | 0.8992 / 50.4 | 0.8992 / 47.4 |
| fixed 0.005 | 0.8988 / 43.3 | 0.8988 / 43.8 | 0.8988 / 45.6 | 0.8986 / 44.7 |
| fixed 0.002 | 0.8974 / 42.4 | 0.8974 / 42.6 | 0.8973 / 43.3 | 0.8972 / 43.0 |
| fixed 0.001 | 0.8952 / 41.5 | 0.8951 / 41.7 | 0.8950 / 42.1 | 0.8948 / 41.9 |
| lag, mean held, c = 1/e | 0.8996 / 44.5 | 0.8993 / 45.9 | 0.8986 / 45.4 | 0.8990 / 47.2 |
| lag, mean held, c = 1/2 | 0.8996 / 44.5 | 0.8993 / 46.0 | 0.8989 / 46.5 | 0.8992 / 48.6 |
| lag, mean held, c = 1 | 0.8996 / 44.5 | 0.8993 / 46.2 | 0.8991 / 51.6 | 0.8992 / 52.0 |
| lag, held now, c = 1/e | 0.8996 / 44.5 | 0.8992 / 46.3 | 0.9209 / 55.8 | 0.9131 / 54.2 |
| lag, own lag, c = 1/e | 0.8996 / 44.5 | 0.8946 / 45.0 | 0.8579 / 39.6 | 0.8660 / 40.2 |
| agreeing radii | 0.8996 / 44.5 | 0.8827 / 43.6 | 0.8009 / 46.5 | 0.8590 / 44.4 |
| current radius | 0.8996 / 44.5 | 0.8919 / 44.6 | 0.8666 / 46.9 | 0.8810 / 45.9 |
| AgACI | 0.8952 / 41.5 | 0.8997 / 58.3 | 0.9057 / 414.4 | 0.8996 / 115.8 |
| SAOCP | 0.9014 / 52.3 | 0.8978 / 59.1 | 0.8538 / 52.0 | 0.8767 / 53.9 |
| rows waiting | 0 | 54 | 338 | 152 |

The rule with a fixed `n` per stream, `0.05 / (1 + 0.1 · g(α) · n)`
(`c = 1/2`; `formula.py`), as a reader would set it from the mean rows
waiting:

| flights | level | n | rate | coverage / half-width | default 0.05 |
|---|---:|---:|---:|---:|---:|
| 1 h | 0.5 | 54 | 0.0113 | 0.4997 / 16.5 | 0.5473 / 24.1 |
| 1 h | 0.9 | 54 | 0.0237 | 0.8993 / 46.1 | 0.8995 / 47.4 |
| 1 h | 0.99 | 54 | 0.0433 | 0.9894 / 137.4 | 0.9895 / 137.0 |
| 8 h | 0.5 | 338 | 0.0022 | 0.4995 / 15.4 | 0.6965 / 127.3 |
| 8 h | 0.9 | 338 | 0.0063 | 0.8991 / 46.6 | 0.9034 / 188.9 |
| 8 h | 0.99 | 338 | 0.0253 | 0.9896 / 150.9 | 0.9895 / 161.2 |
| landing | 0.5 | 152 | 0.0047 | 0.4996 / 16.1 | 0.5547 / 28.8 |
| landing | 0.9 | 152 | 0.0121 | 0.8993 / 48.5 | 0.8996 / 63.6 |
| landing | 0.99 | 152 | 0.0347 | 0.9895 / 149.0 | 0.9895 / 150.0 |

With no delay the rate is the default's and every number is the
default's (16.7, 44.5, 135.6 minutes at 50%, 90%, 99%).

### 5.3 Generated, 90%: coverage / half-width, mean over three seeds and four noises

| rule | steady none | steady short | steady long | steady variable | bursty none | bursty short | bursty long | bursty variable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.05 (today) | 0.8995 / 1.77 | 0.8995 / 1.77 | 0.9012 / 4.10 | 0.8993 / 1.83 | 0.8995 / 1.77 | 0.8995 / 1.77 | 0.9165 / 5.34 | 0.8993 / 1.87 |
| fixed 0.02 | 0.8995 / 1.77 | 0.8993 / 1.77 | 0.9001 / 1.90 | 0.8990 / 1.79 | 0.8995 / 1.77 | 0.8993 / 1.77 | 0.9014 / 2.83 | 0.8990 / 1.85 |
| fixed 0.01 | 0.9027 / 1.93 | 0.8999 / 1.78 | 0.9038 / 2.13 | 0.8991 / 1.79 | 0.9026 / 1.94 | 0.8994 / 1.77 | 0.9009 / 2.30 | 0.8992 / 1.86 |
| fixed 0.005 | 0.9121 / 2.58 | 0.9055 / 1.91 | 0.9144 / 2.99 | 0.9019 / 1.84 | 0.9120 / 2.58 | 0.9027 / 1.83 | 0.9036 / 2.09 | 0.9021 / 1.89 |
| fixed 0.002 | 0.9356 / 4.64 | 0.9278 / 2.61 | 0.9407 / 5.56 | 0.9129 / 2.08 | 0.9356 / 4.64 | 0.9154 / 2.12 | 0.9128 / 2.16 | 0.9127 / 2.09 |
| lag, mean held, c = 1/e | 0.8995 / 1.77 | 0.8995 / 1.77 | 0.9131 / 2.89 | 0.9014 / 1.83 | 0.8995 / 1.77 | 0.8994 / 1.77 | 0.9068 / 2.03 | 0.9038 / 1.91 |
| lag, mean held, c = 1/2 | 0.8995 / 1.77 | 0.8995 / 1.77 | 0.9083 / 2.50 | 0.8998 / 1.80 | 0.8995 / 1.77 | 0.8994 / 1.77 | 0.9041 / 2.00 | 0.9016 / 1.87 |
| lag, mean held, c = 1 | 0.8995 / 1.77 | 0.8995 / 1.77 | 0.9020 / 2.01 | 0.8989 / 1.79 | 0.8995 / 1.77 | 0.8994 / 1.77 | 0.9005 / 2.05 | 0.8991 / 1.83 |
| current radius | 0.8995 / 1.77 | 0.8994 / 1.77 | 0.8978 / 1.78 | 0.8989 / 1.78 | 0.8995 / 1.77 | 0.8994 / 1.77 | 0.8965 / 1.78 | 0.8983 / 1.78 |
| AgACI | 0.9120 / 2.10 | 0.9290 / 2.66 | 0.9110 / 12.36 | 0.9045 / 2.07 | 0.9085 / 1.97 | 0.9049 / 1.98 | 0.9214 / 7.69 | 0.9044 / 2.12 |
| SAOCP | 0.8977 / 1.88 | 0.8974 / 1.88 | 0.8964 / 1.89 | 0.8976 / 1.89 | 0.8974 / 1.88 | 0.8977 / 1.88 | 0.8950 / 1.89 | 0.8970 / 1.90 |

By noise, over the three delays (coverage / half-width): today's default
0.9025 / 2.53 (Gaussian), 0.9024 / 2.13 (t3), 0.9026 / 3.74
(heteroskedastic), 0.9026 / 2.73 (shift); the delay rule at `c = 1/2`
0.9010 / 1.74, 0.9019 / 1.60, 0.9026 / 2.52, 0.9030 / 1.94; the current
radius 0.8999 / 1.66, 0.8994 / 1.38, 0.8972 / 2.37, 0.8971 / 1.71.

---

## 6. What holds, and what does not

**Today's default fails exactly where §2 says.** Every stream it fails at
90% has a mean of 54 rows waiting or more (flights at 1 h, 8 h and
landing; every generated stream under the long delay). Its coverage
stays near 90% there, but its half-width is 1.1 to 4.4 times the best
rate's. At 50%, where `g` is three times as large, the swing pins the
radius at zero and the clamp adds coverage: 70% on flights at 8 h, 63–76%
on the generated long delay. At 99% `g` is a seventh of 90%'s, and the
default holds in 30 of 36.

**The delay rule removes that failure and costs nothing without a
delay.** At `n̄ = 0` it is the default to the bit (every "none" column).
It held every flights stream's coverage and the widths 45.4–52.0
minutes against the default's 189. At 50% it held all 36 (c = 1/2).

**It fails where a slow step is slow.** Its 90% failures are two kinds.
First, the start: under the long delay the first released residuals come
from a model that has learned a handful of rows, `sigma` is several
times its later value, and the warm start sets the radius from it (seed
0's first radius was 21 against a steady 1.65); a step a twentieth of
the default's takes tens of thousands of rows to come down. The second
half of the rows shows the steady state, where c = 1/2 holds 90 of 100.
Second, a noise that moves within the delay: the heteroskedastic stretch
and the level shift under the long delay. There every rule is late, but
the slower step is later to come back down (the bursty heteroskedastic
stream covers 91% in its second half). At 99% the step that brings the
radius down on a hit is `rate · sigma · 0.01`, so the slower rule takes
longer from a high start than the default does: 27 of 36 against 30.

**No `c` serves both kinds of stream.** Flights want `c ≤ 1/2`, because
its misses come in bursts (§2.3): at c = 1 the 8 h half-width is 51.6
against 45.4–46.5. The generated streams want `c ≥ 1`, because their
misses are independent and the slow start dominates: c = 1/e leaves the
steady long stream at 91.3% and 2.89 against 1.90 for a fixed 0.02.

**A fixed 0.02 holds 93 of 100 at 90%, and is not an answer.** It is
the best fixed rate on the generated streams (whose delay is at most 500
rows), but on flights at 8 h it is 64.9 minutes against 45.6, at 50% it
holds 25 of 36, and at 99% 15: it is a rate tuned to this grid's delays,
not a rule.

**The step against the current radius holds 93 of 100 and gives up the
guarantee.** It removes the dead time from the loop, so it never swings,
and on the generated streams, whose distribution changes slowly against
the delay, stale samples of it are as good as fresh. On flights the
residuals move with the hour within the delay, the radius is the
quantile of hours-old misses, and the coverage falls to 86.7% at 8 h.
Its coverage is not tied to the intervals served, so nothing bounds the
gap.

**The adaptive aggregators do not survive a delay.** AgACI's aggregation
is fed the same late news as its experts; its fast experts swing (as
fixed 0.05 does) and the pinball loss, linearized at the aggregate,
rewards a wide expert after every burst of misses: 414 minutes at 8 h.
Without a delay it settles on its slowest expert (on flights its 0.8952
and 41.5 are fixed 0.001's to the printed digits) and under-covers.
SAOCP's experts read their errors at radii that have moved since the row
was served, and it under-covers under a delay, by up to 4.6 points
(flights at 8 h).

**The agreement rule was an attempt to have both** (step only when the
shown and the current radius agree on the miss). It under-covers everywhere a row waits (80.1% at 8 h), so the rises it
skips outnumber the falls.

---

## 7. The decision, and what it needs from the user

**The default stays 0.05.** No rule held across all the conditions, which
the brief made the bar for changing it. The derived rule is the best
general improvement found (90%: 88–91 of 100 against 73; 50%: 36 of 36
against 23; 99%: 27 against 30), it is the default exactly when nothing
waits, and it fails only where every slow step does.

**Built in this task, documentation only.** DIAGNOSTICS.md's *Under a long
delay* paragraph gives the reader the rule with a fixed `n`, the mean
rows waiting (the row rate times the mean delay): `0.05 / (1 + 0.1 · g ·
n)`, for 90% `0.05 / (1 + n / 48)`, with the flights table of §5.2 and
the reason. The README's `label_clock` section and the spec reference's
`conformal` entry say the same in a sentence. The earlier sentence that a
rule was "being studied", and its square-root trial, are gone.

**For the user: an opt-in, `conformal_rate="auto"`.** Designed, not built,
because it needs state:

| | |
|---|---|
| what it does | the step `r / (1 + r · g(α) · n̄ / c)`, `r = 0.05`, `c = 1/2`, `n̄` the EW mean of the rows waiting at each release, the released row aside, on the tracker's own row weights (`w_rows`, the decay `w̄` uses); `"auto"` with a number, e.g. `conformal_rate={"auto": 0.05}`, could set the base |
| where `n` comes from | the stream's held rows per group (`pending.len()` under `embargo`, the label buffer under `label_clock`), passed to `Conformal::update_against` at each release; 0 everywhere else, so an undelayed stream is unchanged to the bit |
| state (rule 5) | one f64 in `Conformal` (`held_mean`, `#[serde(default)]`) and the mode (the spec keeps it; the tracker's `rate` field would hold the base). Schema 67 is unreleased and shared with tasks 340-341, so it rides that bump: a fixture with a delayed `auto` tracker past its warm start, its `held_mean` held non-empty; a 66 file loads with `held_mean = 0`, which only matters under `auto`, which no 66 file has |
| cost | one multiply-add and a divide per release, beside the tracker's own; nothing per held row |
| tests | the fold of `tests/test_label_clock.py` with the rule added, to the bit; `embargo = d` against `label_clock = clock + d` as now; the rule equal to the fixed rate at no delay, to the bit; the derivation's simulation (§2.4) as a statistical test of `k·n` |

Three ways to go, the user's: (a) keep the documentation alone; (b) build
the opt-in; (c) build it and make it the default for specs with `embargo`
or `label_clock` only, which changes no undelayed stream but does change
the delayed streams' bytes and outputs, and loses 3 of 36 at 99%.

**Not a defect, but worth knowing.** The warm start reads `sigma` from
as few as one residual. Under a long delay that residual is from a model
that had learned almost nothing, and the first radius can be ten times
the steady one; at 99%, where a hit lowers the radius by `rate · sigma ·
0.01`, even the default took more than 50,000 rows to come down on one
generated stream (seed 0, long delay: half-width 10.4 over the scored
rows, 2.9 over the second half, against 2.6). A warm start that waits
for `sigma` to have the weight of a few dozen rows, or reads `min_weight`,
would help every rate; it is a separate question.

---

## 8. Sources

- Angelopoulos, A. N., Candès, E. J. and Tibshirani, R. J. (2023).
  Conformal PID control for time series prediction. *NeurIPS 36*.
- Gibbs, I. and Candès, E. (2021). Adaptive conformal inference under
  distribution shift. *NeurIPS 34*.
- Zaffran, M., Féron, O., Goude, Y., Josse, J. and Dieuleveut, A. (2022).
  Adaptive conformal predictions for time series. *ICML*. (AgACI; read at
  arXiv 2202.07282.)
- Bhatnagar, A., Wang, H., Xiong, C. and Bai, Y. (2023). Improved online
  conformal prediction via strongly adaptive online learning. *ICML*.
  (SAOCP and SF-OGD; Algorithm 1, the lifetime of eq. 8, `g = 8` for time
  series; read at arXiv 2302.07869.)
- Wintenberger, O. (2017). Optimal learning with Bernstein online
  aggregation. *Machine Learning* 106.
- Levin, S. A. and May, R. M. (1976). A note on difference-delay
  equations. *Theoretical Population Biology* 9; Kuruklis, S. A. (1994).
  The asymptotic stability of `x_{n+1} − a x_n + b x_{n−k} = 0`. *J.
  Math. Anal. Appl.* 188. (The bound of §2.2.)

---

## 9. Reproducing

```sh
.venv/bin/python -I scripts/conformal_step/check_replica.py   # the fold is the bank, to the bit; exits 1 if not
.venv/bin/python -I scripts/conformal_step/linear.py          # §2.4
.venv/bin/python -I scripts/conformal_step/measure.py 2       # §5, two processes, a few minutes
.venv/bin/python -I scripts/conformal_step/summarize.py       # the tables of §5.1-§5.3
.venv/bin/python -I scripts/conformal_step/formula.py         # §5.2's rule with a fixed n
```

They need the package built (`maturin develop`), `lightgbm` and `numba`
(both in the dev group), and nycflights13 in `.cache/tree-features/`
(`scripts/tree_features/fetch.py`). The residuals and every result are
written to `.cache/conformal-step/`.
